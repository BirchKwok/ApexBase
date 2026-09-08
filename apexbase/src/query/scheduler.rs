//! Query Scheduler - enables parallel query execution using thread pool
//!
//! This module provides a thread-pool based query scheduler that allows
//! multiple queries to execute concurrently in Rust, bypassing Python's GIL.
//!
//! Architecture review R4: every task carries the caller's session context
//! (root/temp dirs) so worker threads observe the same query context as a
//! direct `Session` execution, a bounded task queue gives callers admission
//! control, and each task exposes a cancellation token that the batched
//! scan pipeline checks at batch boundaries.

use parking_lot::{Condvar, Mutex};
use std::cell::RefCell;
use std::collections::VecDeque;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::mpsc::{channel, Receiver, Sender};
use std::sync::Arc;
use std::thread;

use crate::query::executor::{ApexExecutor, ApexResult};

/// Maximum number of worker threads
pub const DEFAULT_THREADS: usize = 4;

/// Default bound for queued (not yet running) queries. Submissions beyond
/// the bound are rejected immediately instead of growing an unbounded queue.
pub const DEFAULT_QUEUE_CAPACITY: usize = 1024;

/// Query execution result
pub enum QueryResult {
    Data(arrow::record_batch::RecordBatch),
    Error(String),
    Done,
}

/// Session context propagated to the executing worker thread. Mirrors the
/// thread-local context installed by `crate::Session`.
#[derive(Clone, Default)]
pub struct QueryContext {
    pub root_dir: Option<PathBuf>,
    pub temp_dir: Option<PathBuf>,
}

/// A query task submitted to the scheduler
struct QueryTask {
    sql: String,
    table_path: PathBuf,
    root_dir: Option<PathBuf>,
    temp_dir: Option<PathBuf>,
    cancel: Arc<AtomicBool>,
    result_sender: Sender<QueryResult>,
}

/// Handle to a submitted query: wait for the result or request cancellation.
pub struct ScheduledQuery {
    receiver: Receiver<QueryResult>,
    cancel: Arc<AtomicBool>,
}

impl ScheduledQuery {
    /// Request cancellation. The running query observes the token at the
    /// next batch boundary; an already finished query is unaffected.
    pub fn cancel(&self) {
        self.cancel.store(true, Ordering::Relaxed);
    }

    /// Block until the query finishes and return its result.
    pub fn wait(self) -> Result<QueryResult, String> {
        match self.receiver.recv() {
            Ok(result) => Ok(result),
            Err(_) => Err("Scheduler channel closed".to_string()),
        }
    }
}

/// Thread pool based query executor
pub struct ThreadPoolExecutor {
    worker_handles: Vec<thread::JoinHandle<()>>,
    task_queue: Arc<Mutex<VecDeque<QueryTask>>>,
    condvar: Arc<Condvar>,
    shutdown_flag: Arc<AtomicBool>,
    active_count: Arc<AtomicUsize>,
    queue_capacity: usize,
}

impl ThreadPoolExecutor {
    /// Create a new thread pool with the default queue capacity
    pub fn new(num_threads: usize) -> Self {
        Self::with_queue_capacity(num_threads, DEFAULT_QUEUE_CAPACITY)
    }

    /// Create a new thread pool with an explicit bound on queued queries.
    pub fn with_queue_capacity(num_threads: usize, queue_capacity: usize) -> Self {
        let task_queue = Arc::new(Mutex::new(VecDeque::new()));
        let condvar = Arc::new(Condvar::new());
        let shutdown_flag = Arc::new(AtomicBool::new(false));
        let active_count = Arc::new(AtomicUsize::new(0));

        let mut worker_handles = Vec::with_capacity(num_threads);

        for _ in 0..num_threads {
            let queue = Arc::clone(&task_queue);
            let cvar = Arc::clone(&condvar);
            let flag = Arc::clone(&shutdown_flag);
            let active = Arc::clone(&active_count);

            let handle = thread::spawn(move || {
                loop {
                    let task = {
                        let mut queue = queue.lock();
                        // Park on condvar until a task arrives or shutdown is requested
                        while queue.is_empty() && !flag.load(Ordering::Relaxed) {
                            cvar.wait(&mut queue);
                        }
                        if flag.load(Ordering::Relaxed) {
                            break;
                        }
                        queue.pop_front()
                    };

                    if let Some(t) = task {
                        active.fetch_add(1, Ordering::Relaxed);
                        Self::execute_query(t);
                        active.fetch_sub(1, Ordering::Relaxed);
                    }
                }
            });

            worker_handles.push(handle);
        }

        Self {
            worker_handles,
            task_queue,
            condvar,
            shutdown_flag,
            active_count,
            queue_capacity,
        }
    }

    /// Execute a single query on the current (worker) thread.
    ///
    /// Installs the task's session context and cancellation token into the
    /// executor's thread-locals, then restores the previous state, so worker
    /// threads behave exactly like a `Session`-scoped execution.
    fn execute_query(task: QueryTask) {
        let previous_root_dir = crate::query::executor::get_query_root_dir();
        let previous_temp_dir = crate::query::executor::get_temp_dir();
        if let Some(root_dir) = &task.root_dir {
            crate::query::executor::set_query_root_dir(root_dir);
        }
        if let Some(temp_dir) = &task.temp_dir {
            crate::query::executor::set_temp_dir(temp_dir);
        }
        crate::query::executor::set_query_cancel_token(Some(Arc::clone(&task.cancel)));

        let result = (|| -> Result<QueryResult, String> {
            // Use the execute function that takes storage_path
            let exec_result =
                ApexExecutor::execute(&task.sql, &task.table_path).map_err(|e| e.to_string())?;

            match exec_result {
                ApexResult::Data(batch) => Ok(QueryResult::Data(batch)),
                ApexResult::Empty(_) => {
                    let schema = Arc::new(arrow::datatypes::Schema::empty());
                    let batch = arrow::record_batch::RecordBatch::new_empty(schema);
                    Ok(QueryResult::Data(batch))
                }
                ApexResult::Scalar(val) => {
                    let schema = Arc::new(arrow::datatypes::Schema::new(vec![
                        arrow::datatypes::Field::new(
                            "result",
                            arrow::datatypes::DataType::Int64,
                            false,
                        ),
                    ]));
                    let array: arrow::array::ArrayRef =
                        Arc::new(arrow::array::Int64Array::from(vec![val]));
                    let batch = arrow::record_batch::RecordBatch::try_new(schema, vec![array])
                        .map_err(|e| e.to_string())?;
                    Ok(QueryResult::Data(batch))
                }
            }
        })();

        crate::query::executor::set_query_cancel_token(None);
        match previous_temp_dir {
            Some(path) => crate::query::executor::set_temp_dir(&path),
            None => crate::query::executor::clear_temp_dir(),
        }
        match previous_root_dir {
            Some(path) => crate::query::executor::set_query_root_dir(&path),
            None => crate::query::executor::clear_query_root_dir(),
        }

        let result = match result {
            Ok(r) => r,
            Err(e) => QueryResult::Error(e),
        };

        let _ = task.result_sender.send(result);
    }

    /// Submit a query without session context.
    /// Returns `None` when the queue is full (admission rejected).
    pub fn submit(&self, sql: String, table_path: PathBuf) -> Option<ScheduledQuery> {
        self.submit_with_context(sql, table_path, &QueryContext::default())
    }

    /// Submit a query carrying the caller's session context.
    /// Returns `None` when the queue is full (admission rejected).
    pub fn submit_with_context(
        &self,
        sql: String,
        table_path: PathBuf,
        context: &QueryContext,
    ) -> Option<ScheduledQuery> {
        let (sender, receiver) = channel();
        let cancel = Arc::new(AtomicBool::new(false));
        let task = QueryTask {
            sql,
            table_path,
            root_dir: context.root_dir.clone(),
            temp_dir: context.temp_dir.clone(),
            cancel: Arc::clone(&cancel),
            result_sender: sender,
        };

        {
            let mut queue = self.task_queue.lock();
            if queue.len() >= self.queue_capacity {
                return None;
            }
            queue.push_back(task);
        }
        self.condvar.notify_one();

        Some(ScheduledQuery { receiver, cancel })
    }

    /// Get number of active queries
    pub fn active_count(&self) -> usize {
        self.active_count.load(Ordering::Relaxed)
    }

    /// Get the number of queries waiting in the queue
    pub fn queued_count(&self) -> usize {
        self.task_queue.lock().len()
    }

    /// Shutdown the thread pool
    pub fn shutdown(&mut self) {
        self.shutdown_flag.store(true, Ordering::Relaxed);
        self.condvar.notify_all();
        for handle in self.worker_handles.drain(..) {
            let _ = handle.join();
        }
    }
}

impl Drop for ThreadPoolExecutor {
    fn drop(&mut self) {
        // catch_unwind: on Windows, thread-local destructors may run after
        // parking_lot's internal TLS is torn down, causing a panic in join().
        // Absorb it so the process exits cleanly after all tests pass.
        let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            self.shutdown();
        }));
    }
}

// Thread-local scheduler storage
thread_local! {
    static SCHEDULER: RefCell<Option<Box<ThreadPoolExecutor>>> = RefCell::new(None);
}

/// Initialize the global scheduler with specified number of threads
pub fn init_scheduler(num_threads: usize) {
    SCHEDULER.with(|s| {
        *s.borrow_mut() = Some(Box::new(ThreadPoolExecutor::new(num_threads)));
    });
}

/// Initialize the global scheduler with an explicit queue capacity.
pub fn init_scheduler_with_capacity(num_threads: usize, queue_capacity: usize) {
    SCHEDULER.with(|s| {
        *s.borrow_mut() =
            Some(Box::new(ThreadPoolExecutor::with_queue_capacity(num_threads, queue_capacity)));
    });
}

/// Initialize with default threads (4)
pub fn init_scheduler_default() {
    init_scheduler(DEFAULT_THREADS);
}

/// Execute query through scheduler - returns a handle to wait on or cancel.
/// Returns `None` when the scheduler is not initialized or the queue is full.
pub fn execute_through_scheduler(
    sql: String,
    table_path: PathBuf,
    context: &QueryContext,
) -> Option<ScheduledQuery> {
    SCHEDULER.with(|s| s.borrow().as_ref().and_then(|s| s.submit_with_context(sql, table_path, context)))
}

/// Check if scheduler is initialized
pub fn is_scheduler_initialized() -> bool {
    SCHEDULER.with(|s| s.borrow().is_some())
}

/// Get active query count
pub fn get_active_count() -> Option<usize> {
    SCHEDULER.with(|s| s.borrow().as_ref().map(|s| s.active_count()))
}

/// Get queued (waiting) query count
pub fn get_queued_count() -> Option<usize> {
    SCHEDULER.with(|s| s.borrow().as_ref().map(|s| s.queued_count()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::storage::OnDemandStorage;
    use std::collections::HashMap;
    use std::path::Path;
    use tempfile::tempdir;

    fn make_table(dir: &Path, name: &str) -> PathBuf {
        let path = dir.join(format!("{name}.apex"));
        let storage = OnDemandStorage::create(&path).unwrap();
        let mut int_cols: HashMap<String, Vec<i64>> = HashMap::new();
        let mut string_cols: HashMap<String, Vec<String>> = HashMap::new();
        int_cols.insert("id".to_string(), vec![1, 2, 3]);
        string_cols.insert(
            "name".to_string(),
            vec!["a".to_string(), "b".to_string(), "c".to_string()],
        );
        storage
            .insert_typed(int_cols, HashMap::new(), string_cols, HashMap::new(), HashMap::new())
            .unwrap();
        storage.save().unwrap();
        path
    }

    #[test]
    fn scheduled_query_propagates_session_context() {
        let ws = tempdir().unwrap();
        let ws = ws.path();

        // Current table nested two levels below the root; the qualified
        // table d2.other is only reachable through an explicit root_dir.
        let current_dir = ws.join("d1").join("sub");
        std::fs::create_dir_all(&current_dir).unwrap();
        let current = make_table(&current_dir, "t");
        let other_dir = ws.join("d2");
        std::fs::create_dir_all(&other_dir).unwrap();
        make_table(&other_dir, "other");

        init_scheduler(2);
        let sql = "SELECT COUNT(*) FROM d2.other".to_string();

        // Without context the worker falls back to base_dir.parent()
        // (ws/d1) and cannot resolve ws/d2/other.apex.
        let no_ctx = execute_through_scheduler(
            sql.clone(),
            current.clone(),
            &QueryContext::default(),
        )
        .expect("scheduler initialized")
        .wait()
        .expect("worker reports a result");
        assert!(
            matches!(no_ctx, QueryResult::Error(_)),
            "qualified lookup without root_dir must fail"
        );

        // With root_dir the same query resolves ws/d2/other.apex.
        let with_ctx = execute_through_scheduler(
            sql,
            current,
            &QueryContext {
                root_dir: Some(ws.to_path_buf()),
                temp_dir: None,
            },
        )
        .expect("scheduler initialized")
        .wait()
        .expect("worker reports a result");
        assert!(
            matches!(with_ctx, QueryResult::Data(_)),
            "qualified lookup with root_dir must succeed"
        );
    }

    #[test]
    fn scheduled_queue_rejects_when_full() {
        let dir = tempdir().unwrap();
        let table = make_table(dir.path(), "t");

        // Hold the cross-process write lock so the worker blocks on INSERT.
        let lock_path = {
            let mut p = table.clone();
            p.set_file_name("t.apex.lock");
            p
        };
        let lock_file = std::fs::OpenOptions::new()
            .create(true)
            .write(true)
            .open(&lock_path)
            .unwrap();
        use fs2::FileExt;
        lock_file.lock_exclusive().unwrap();

        init_scheduler_with_capacity(1, 1);
        let no_ctx = QueryContext::default();

        // Task 1 occupies the single worker (blocked on the flock).
        let first = execute_through_scheduler(
            "INSERT INTO t (id, name) VALUES (10, 'x')".to_string(),
            table.clone(),
            &no_ctx,
        )
        .expect("within queue capacity");

        // Wait until the worker has picked Task 1 up; while it is blocked on
        // the flock the queue is empty and Task 2 below can enter it.
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
        while get_active_count().unwrap_or(0) == 0 {
            assert!(
                std::time::Instant::now() < deadline,
                "worker must pick up the blocking write (is it blocked on the lock?)"
            );
            std::thread::sleep(std::time::Duration::from_millis(1));
        }

        // Task 2 fills the queue.
        let second = execute_through_scheduler(
            "INSERT INTO t (id, name) VALUES (11, 'y')".to_string(),
            table.clone(),
            &no_ctx,
        )
        .expect("within queue capacity");

        // Task 3 exceeds the bound: rejected immediately, without blocking.
        let started = std::time::Instant::now();
        let rejected = execute_through_scheduler(
            "SELECT COUNT(*) FROM t".to_string(),
            table.clone(),
            &no_ctx,
        );
        assert!(rejected.is_none(), "queue at capacity must reject");
        assert!(
            started.elapsed() < std::time::Duration::from_secs(2),
            "rejection must not block"
        );

        // Release the lock; the queued writes complete in order.
        lock_file.unlock().unwrap();
        drop(lock_file);
        assert!(matches!(
            first.wait().unwrap(),
            QueryResult::Data(_) | QueryResult::Done
        ));
        assert!(matches!(
            second.wait().unwrap(),
            QueryResult::Data(_) | QueryResult::Done
        ));
    }
}
