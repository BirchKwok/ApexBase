//! V3 Native Query Executor
//!
//! This module provides a pure Arrow-based query execution engine that operates
//! directly on OnDemandStorage without requiring ColumnTable.
//!
//! Architecture:
//! - Reads columns on-demand from V3 storage
//! - Performs all filtering/projection/aggregation using Arrow compute kernels
//! - Returns Arrow RecordBatch directly (zero-copy to Python)

use arrow::array::{
    Array, ArrayRef, BooleanArray, Float64Array, Int64Array, StringArray,
    UInt64Array, RecordBatch,
};
use arrow::compute::{self, SortOptions};
use arrow::compute::kernels::cmp;
use arrow::compute::kernels::numeric as arith;
use arrow::datatypes::{DataType as ArrowDataType, Field, Schema};
use ahash::AHashMap;
use std::io;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use parking_lot::RwLock;
use once_cell::sync::Lazy;

use crate::query::{SqlParser, SqlStatement, SelectStatement, SqlExpr, SelectColumn, JoinType, JoinClause, UnionStatement, AggregateFunc};
use crate::query::sql_parser::BinaryOperator;
use crate::query::sql_parser::FromItem;
use crate::query::jit::{ExprJIT, FilterFnI64, simd_sum_i64, simd_sum_f64, simd_min_i64, simd_max_i64};
use crate::query::planner::{QueryPlanner, ExecutionStrategy, get_table_stats, invalidate_table_stats};

/// Zone Map optimization result for filter pruning
#[derive(PartialEq, Eq, Clone, Copy)]
enum ZoneMapResult {
    NoMatch,    // Filter definitely won't match any rows
    MayMatch,   // Filter might match some rows
}
use crate::storage::TableStorageBackend;
use crate::data::{DataType, Value};
use std::collections::HashSet;
use ahash::AHasher;
use std::hash::{Hash, Hasher};

// ============================================================================
// Global SQL parse cache — avoids re-tokenizing/parsing the same SQL across cold iterations
// ============================================================================
static SQL_PARSE_CACHE: Lazy<RwLock<AHashMap<String, Vec<SqlStatement>>>> =
    Lazy::new(|| RwLock::new(AHashMap::new()));

// ============================================================================
// Thread-local root directory for multi-database cross-db table resolution
// Set by Python bindings before calling execute_with_base_dir when a named
// database is active. Allows resolve_table_path to locate db.table references.
// ============================================================================
thread_local! {
    static QUERY_ROOT_DIR: std::cell::RefCell<Option<std::path::PathBuf>> =
        std::cell::RefCell::new(None);
}

/// Set the root directory for the current thread's query context.
/// Call this before execute_with_base_dir when using named databases.
pub fn set_query_root_dir(root_dir: &Path) {
    QUERY_ROOT_DIR.with(|r| *r.borrow_mut() = Some(root_dir.to_path_buf()));
}

/// Clear the root directory from the current thread's query context.
pub fn clear_query_root_dir() {
    QUERY_ROOT_DIR.with(|r| *r.borrow_mut() = None);
}

/// Get the root directory for the current thread's query context.
pub fn get_query_root_dir() -> Option<std::path::PathBuf> {
    QUERY_ROOT_DIR.with(|r| r.borrow().clone())
}

// ============================================================================
// Helper functions to reduce code duplication
// ============================================================================

/// Create an InvalidInput error with message
#[inline]
fn err_input(msg: impl Into<String>) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidInput, msg.into())
}

/// Create an InvalidData error with message  
#[inline]
fn err_data(msg: impl Into<String>) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, msg.into())
}

/// Create an Unsupported error with message
#[inline]
fn err_unsupported(msg: impl Into<String>) -> io::Error {
    io::Error::new(io::ErrorKind::Unsupported, msg.into())
}

/// Create a NotFound error with message
#[inline]
fn err_not_found(msg: impl Into<String>) -> io::Error {
    io::Error::new(io::ErrorKind::NotFound, msg.into())
}

/// Helper to apply a unary function on numeric arrays
#[inline]
fn map_numeric_unary<F1, F2>(arr: &ArrayRef, batch_rows: usize, int_fn: F1, float_fn: F2, func_name: &str) -> io::Result<ArrayRef>
where
    F1: Fn(i64) -> i64,
    F2: Fn(f64) -> f64,
{
    if let Some(int_arr) = arr.as_any().downcast_ref::<Int64Array>() {
        let result: Vec<Option<i64>> = (0..batch_rows).map(|i| {
            if int_arr.is_null(i) { None } else { Some(int_fn(int_arr.value(i))) }
        }).collect();
        Ok(Arc::new(Int64Array::from(result)))
    } else if let Some(float_arr) = arr.as_any().downcast_ref::<Float64Array>() {
        let result: Vec<Option<f64>> = (0..batch_rows).map(|i| {
            if float_arr.is_null(i) { None } else { Some(float_fn(float_arr.value(i))) }
        }).collect();
        Ok(Arc::new(Float64Array::from(result)))
    } else {
        Err(err_data(format!("{} requires numeric argument", func_name)))
    }
}

/// Helper to apply a unary string function
#[inline]
fn map_string_unary<F>(arr: &ArrayRef, batch_rows: usize, f: F, func_name: &str) -> io::Result<ArrayRef>
where
    F: Fn(&str) -> String,
{
    if let Some(str_arr) = arr.as_any().downcast_ref::<StringArray>() {
        let result: Vec<Option<String>> = (0..batch_rows).map(|i| {
            if str_arr.is_null(i) { None } else { Some(f(str_arr.value(i))) }
        }).collect();
        Ok(Arc::new(StringArray::from(result.iter().map(|s| s.as_deref()).collect::<Vec<_>>())))
    } else {
        Err(err_data(format!("{} requires string argument", func_name)))
    }
}

/// Helper to apply a unary string function returning &str (no allocation)
#[inline]
fn map_string_unary_ref<'a, F>(arr: &'a ArrayRef, batch_rows: usize, f: F, func_name: &str) -> io::Result<ArrayRef>
where
    F: Fn(&'a str) -> &'a str,
{
    if let Some(str_arr) = arr.as_any().downcast_ref::<StringArray>() {
        let result: Vec<Option<&str>> = (0..batch_rows).map(|i| {
            if str_arr.is_null(i) { None } else { Some(f(str_arr.value(i))) }
        }).collect();
        Ok(Arc::new(StringArray::from(result)))
    } else {
        Err(err_data(format!("{} requires string argument", func_name)))
    }
}

/// Helper to apply a string-to-int function
#[inline]
fn map_string_to_int<F>(arr: &ArrayRef, batch_rows: usize, f: F, func_name: &str) -> io::Result<ArrayRef>
where
    F: Fn(&str) -> i64,
{
    if let Some(str_arr) = arr.as_any().downcast_ref::<StringArray>() {
        let result: Vec<Option<i64>> = (0..batch_rows).map(|i| {
            if str_arr.is_null(i) { None } else { Some(f(str_arr.value(i))) }
        }).collect();
        Ok(Arc::new(Int64Array::from(result)))
    } else {
        Err(err_data(format!("{} requires string argument", func_name)))
    }
}

/// Helper to apply an int-to-string function
#[inline]
fn map_int_to_string<F>(arr: &ArrayRef, batch_rows: usize, f: F, func_name: &str) -> io::Result<ArrayRef>
where
    F: Fn(i64) -> Option<String>,
{
    if let Some(int_arr) = arr.as_any().downcast_ref::<Int64Array>() {
        let result: Vec<Option<String>> = (0..batch_rows).map(|i| {
            if int_arr.is_null(i) { None } else { f(int_arr.value(i)) }
        }).collect();
        Ok(Arc::new(StringArray::from(result.iter().map(|s| s.as_deref()).collect::<Vec<_>>())))
    } else {
        Err(err_data(format!("{} requires int argument", func_name)))
    }
}

// Global storage cache to avoid repeated open() calls which load all IDs
// Key: canonical path, Value: (backend, last_modified_time, last_access_time)
// Uses LRU eviction when cache exceeds MAX_CACHE_ENTRIES
const MAX_CACHE_ENTRIES: usize = 64;  // Limit cache to 64 tables

static STORAGE_CACHE: Lazy<RwLock<AHashMap<PathBuf, (Arc<TableStorageBackend>, std::time::SystemTime, std::time::Instant)>>> = 
    Lazy::new(|| RwLock::new(AHashMap::with_capacity(MAX_CACHE_ENTRIES)));

// ============================================================================
// Per-table Write Locks — serializes concurrent writes to the same table
// ============================================================================
// Two-layer locking for concurrent access safety:
// Layer 1: parking_lot::Mutex (~10-20ns uncontended) — same-process threads
// Layer 2: fs2 flock on cached File handle (~0.5μs) — cross-process safety
//
// The File handle is opened once and cached, so repeated writes only pay
// the flock() syscall cost, not open()+flock().
struct TableLock {
    mutex: parking_lot::Mutex<()>,
    file: Option<std::fs::File>,
}

static TABLE_WRITE_LOCKS: Lazy<RwLock<AHashMap<PathBuf, Arc<TableLock>>>> =
    Lazy::new(|| RwLock::new(AHashMap::with_capacity(32)));

/// Get or create a per-table lock entry (Mutex + cached fs2 file handle).
fn get_table_lock(table_path: &Path) -> Arc<TableLock> {
    // Fast path: read-lock the map
    {
        let locks = TABLE_WRITE_LOCKS.read();
        if let Some(lock) = locks.get(table_path) {
            return lock.clone();
        }
    }
    // Slow path: create lock + open sidecar .lock file once
    let lock_path = {
        let mut p = table_path.to_path_buf();
        let name = p.file_name().unwrap_or_default().to_string_lossy().to_string();
        p.set_file_name(format!("{}.lock", name));
        p
    };
    let file = std::fs::OpenOptions::new()
        .create(true)
        .write(true)
        .open(&lock_path)
        .ok();
    let entry = Arc::new(TableLock {
        mutex: parking_lot::Mutex::new(()),
        file,
    });
    let mut locks = TABLE_WRITE_LOCKS.write();
    locks.entry(table_path.to_path_buf())
        .or_insert_with(|| entry.clone());
    entry
}

/// Execute a write operation with per-table locking.
/// Acquires both in-process Mutex and cross-process fs2 flock.
#[inline]
fn with_table_write_lock<F, R>(table_path: &Path, f: F) -> io::Result<R>
where
    F: FnOnce() -> io::Result<R>,
{
    let lock = get_table_lock(table_path);
    // Layer 1: in-process serialization
    let _guard = lock.mutex.lock();
    // Layer 2: cross-process serialization (best-effort, ~0.5μs)
    if let Some(ref file) = lock.file {
        use fs2::FileExt;
        let _ = file.lock_exclusive();
    }
    let result = f();
    // Release cross-process lock immediately
    if let Some(ref file) = lock.file {
        use fs2::FileExt;
        let _ = file.unlock();
    }
    result
}

/// Evict least recently used entries from cache if over limit
fn evict_lru_cache_entries(cache: &mut AHashMap<PathBuf, (Arc<TableStorageBackend>, std::time::SystemTime, std::time::Instant)>) {
    if cache.len() <= MAX_CACHE_ENTRIES {
        return;
    }
    
    // Find the entry with oldest access time
    let entries_to_remove = cache.len() - MAX_CACHE_ENTRIES + 1; // Remove a few extra to avoid frequent eviction
    let mut access_times: Vec<(PathBuf, std::time::Instant)> = cache
        .iter()
        .map(|(k, (_, _, access))| (k.clone(), *access))
        .collect();
    
    // Sort by access time (oldest first)
    access_times.sort_by_key(|(_, t)| *t);
    
    // Remove oldest entries
    for (path, _) in access_times.into_iter().take(entries_to_remove) {
        cache.remove(&path);
    }
}

// ============================================================================
// Global Index Manager Cache
// ============================================================================
// Key: base_dir path, Value: table_name -> IndexManager
// Lazily loaded from disk catalog on first access per table
static INDEX_CACHE: Lazy<RwLock<AHashMap<PathBuf, Arc<parking_lot::Mutex<crate::storage::index::IndexManager>>>>> =
    Lazy::new(|| RwLock::new(AHashMap::with_capacity(32)));

/// Get or create an IndexManager for a table. Returns None if base_dir is not available.
/// The key is base_dir/table_name to uniquely identify each table's index manager.
fn get_index_manager(base_dir: &Path, table_name: &str) -> Arc<parking_lot::Mutex<crate::storage::index::IndexManager>> {
    use crate::storage::index::IndexManager;
    let cache_key = base_dir.join(table_name);

    // Fast path: check read lock
    {
        let cache = INDEX_CACHE.read();
        if let Some(mgr) = cache.get(&cache_key) {
            return mgr.clone();
        }
    }

    // Slow path: create and cache
    let mgr = IndexManager::load(table_name, base_dir).unwrap_or_else(|_| IndexManager::new(table_name, base_dir));
    let mgr = Arc::new(parking_lot::Mutex::new(mgr));

    let mut cache = INDEX_CACHE.write();
    cache.entry(cache_key).or_insert_with(|| mgr.clone());
    mgr
}

/// Invalidate index cache for a specific table
#[allow(dead_code)]
fn invalidate_index_cache(base_dir: &Path, table_name: &str) {
    let cache_key = base_dir.join(table_name);
    INDEX_CACHE.write().remove(&cache_key);
}

/// Invalidate all index cache entries under a directory
fn invalidate_index_cache_dir(dir: &Path) {
    INDEX_CACHE.write().retain(|path, _| !path.starts_with(dir));
}

// ============================================================================
// Global FTS Manager Cache
// ============================================================================
// Key: base_dir path (one FtsManager per database directory)
// FtsManager internally manages one FtsEngine per table
static FTS_MANAGER_CACHE: Lazy<RwLock<AHashMap<PathBuf, Arc<crate::fts::FtsManager>>>> =
    Lazy::new(|| RwLock::new(AHashMap::with_capacity(8)));

/// Return the FtsManager for a base_dir if one has been registered.
pub fn get_fts_manager(base_dir: &Path) -> Option<Arc<crate::fts::FtsManager>> {
    FTS_MANAGER_CACHE.read().get(base_dir).cloned()
}

/// Register (or replace) the FtsManager for a base_dir.
/// Called by Python `_init_fts()` and by the `CREATE FTS INDEX` DDL handler.
pub fn register_fts_manager(base_dir: &Path, manager: Arc<crate::fts::FtsManager>) {
    FTS_MANAGER_CACHE.write().insert(base_dir.to_path_buf(), manager);
}

/// Get or lazily create a FtsManager for a base_dir.
/// Creates the manager with default config; actual per-table engine is created on demand.
fn get_or_create_fts_manager(base_dir: &Path) -> Arc<crate::fts::FtsManager> {
    if let Some(mgr) = FTS_MANAGER_CACHE.read().get(base_dir).cloned() {
        return mgr;
    }
    let fts_dir = base_dir.join("fts_indexes");
    let mgr = Arc::new(crate::fts::FtsManager::new(&fts_dir, crate::fts::FtsConfig::default()));
    FTS_MANAGER_CACHE.write().entry(base_dir.to_path_buf()).or_insert_with(|| mgr.clone());
    mgr
}

/// Derive (base_dir, table_name) from a storage_path like /data/users.apex
fn base_dir_and_table(storage_path: &Path) -> (PathBuf, String) {
    base_dir_and_table_pub(storage_path)
}

/// Public (crate-visible) version for use in submodules.
pub(crate) fn base_dir_and_table_pub(storage_path: &Path) -> (PathBuf, String) {
    let base_dir = storage_path.parent().unwrap_or(Path::new(".")).to_path_buf();
    let table_name = storage_path.file_stem()
        .map(|s| s.to_string_lossy().to_string())
        .unwrap_or_else(|| "default".to_string());
    (base_dir, table_name)
}

/// Zone Map (min-max index) for a column
/// Used to skip filtering when conditions can't match
#[derive(Clone, Debug)]
struct ZoneMap {
    min_int: Option<i64>,
    max_int: Option<i64>,
    min_float: Option<f64>,
    max_float: Option<f64>,
    has_nulls: bool,
}

impl ZoneMap {
    fn from_int64_array(arr: &Int64Array) -> Self {
        let mut min_val: Option<i64> = None;
        let mut max_val: Option<i64> = None;
        let mut has_nulls = false;
        
        for i in 0..arr.len() {
            if arr.is_null(i) {
                has_nulls = true;
            } else {
                let v = arr.value(i);
                min_val = Some(min_val.map_or(v, |m| m.min(v)));
                max_val = Some(max_val.map_or(v, |m| m.max(v)));
            }
        }
        
        Self { min_int: min_val, max_int: max_val, min_float: None, max_float: None, has_nulls }
    }
    
    fn from_float64_array(arr: &Float64Array) -> Self {
        let mut min_val: Option<f64> = None;
        let mut max_val: Option<f64> = None;
        let mut has_nulls = false;
        
        for i in 0..arr.len() {
            if arr.is_null(i) {
                has_nulls = true;
            } else {
                let v = arr.value(i);
                min_val = Some(min_val.map_or(v, |m| m.min(v)));
                max_val = Some(max_val.map_or(v, |m| m.max(v)));
            }
        }
        
        Self { min_int: None, max_int: None, min_float: min_val, max_float: max_val, has_nulls }
    }
    
    /// Check if a comparison can potentially match any rows
    /// Returns true if the filter might match, false if it definitely won't match
    #[inline]
    fn can_match(&self, op: &BinaryOperator, literal: &Value) -> bool {
        match literal {
            Value::Int64(v) => self.can_match_int(*v, op),
            Value::Float64(v) => self.can_match_float(*v, op),
            _ => true, // Can't optimize, assume might match
        }
    }
    
    #[inline]
    fn can_match_int(&self, v: i64, op: &BinaryOperator) -> bool {
        let (min, max) = match (self.min_int, self.max_int) {
            (Some(min), Some(max)) => (min, max),
            _ => return true, // No stats, assume might match
        };
        
        match op {
            BinaryOperator::Eq => v >= min && v <= max,
            BinaryOperator::NotEq => true, // Can't optimize !=
            BinaryOperator::Lt => min < v,
            BinaryOperator::Le => min <= v,
            BinaryOperator::Gt => max > v,
            BinaryOperator::Ge => max >= v,
            _ => true,
        }
    }
    
    #[inline]
    fn can_match_float(&self, v: f64, op: &BinaryOperator) -> bool {
        let (min, max) = match (self.min_float, self.max_float) {
            (Some(min), Some(max)) => (min, max),
            _ => {
                // Try int stats for float comparison
                if let (Some(min), Some(max)) = (self.min_int, self.max_int) {
                    (min as f64, max as f64)
                } else {
                    return true;
                }
            }
        };
        
        match op {
            BinaryOperator::Eq => v >= min && v <= max,
            BinaryOperator::NotEq => true,
            BinaryOperator::Lt => min < v,
            BinaryOperator::Le => min <= v,
            BinaryOperator::Gt => max > v,
            BinaryOperator::Ge => max >= v,
            _ => true,
        }
    }
}

/// Invalidate the storage cache for a specific path
/// CRITICAL: Must be called before any write operation to release mmap on Windows
#[inline]
pub fn invalidate_storage_cache(path: &Path) {
    // Use path directly - avoid expensive canonicalize (already absolute in most cases)
    let mut cache = STORAGE_CACHE.write();
    cache.remove(path);
}

/// Invalidate all storage cache entries under a directory
/// CRITICAL: Must be called when closing a client to release all mmaps on Windows
#[inline]
pub fn invalidate_storage_cache_dir(dir: &Path) {
    // Use path directly - avoid expensive canonicalize
    let mut cache = STORAGE_CACHE.write();
    cache.retain(|path, _| !path.starts_with(dir));
    // Also invalidate index caches
    invalidate_index_cache_dir(dir);
}

/// Public wrapper for get_cached_backend (used by Python bindings for fast point lookups)
#[inline]
pub fn get_cached_backend_pub(path: &Path) -> io::Result<Arc<TableStorageBackend>> {
    get_cached_backend(path)
}

/// Get or open a cached storage backend
/// Auto-compacts delta files before reading to ensure data consistency
#[inline]
fn get_cached_backend(path: &Path) -> io::Result<Arc<TableStorageBackend>> {
    // Use path directly - avoid expensive canonicalize (already absolute)
    let cache_key = path.to_path_buf();

    // FASTEST PATH: if backend was validated within last 500ms, skip ALL stat() syscalls.
    // Safe for single-process use: writes always call invalidate_storage_cache() before
    // modifying the file, so stale cached entries are always evicted before the next read.
    // Also refreshes last_access to keep the window alive across benchmark iterations.
    {
        let cache = STORAGE_CACHE.read();
        if let Some((backend, _, last_access)) = cache.get(&cache_key) {
            if last_access.elapsed().as_millis() < 500 {
                let backend_clone = Arc::clone(backend);
                drop(cache);
                // Refresh last_access so subsequent calls within 500ms also hit fast path
                if let Some(entry) = STORAGE_CACHE.write().get_mut(&cache_key) {
                    entry.2 = std::time::Instant::now();
                }
                return Ok(backend_clone);
            }
        }
    }

    // Get main file metadata (single syscall for existence + modified time)
    let metadata = std::fs::metadata(path)?;
    let modified = metadata.modified().unwrap_or(std::time::SystemTime::UNIX_EPOCH);
    
    // Check for delta file - combine path construction with existence check
    let delta_path = {
        let mut dp = cache_key.clone();
        let name = dp.file_name().unwrap_or_default().to_string_lossy();
        dp.set_file_name(format!("{}.delta", name));
        dp
    };
    
    // Check delta metadata (also gets modified time if exists)
    let (has_delta, effective_modified) = match std::fs::metadata(&delta_path) {
        Ok(delta_meta) => {
            let delta_modified = delta_meta.modified().unwrap_or(std::time::SystemTime::UNIX_EPOCH);
            (true, if delta_modified > modified { delta_modified } else { modified })
        }
        Err(_) => (false, modified),
    };
    
    // Try read from cache first (only if no delta file pending)
    if !has_delta {
        // Check cache and update access time if found
        let mut cache = STORAGE_CACHE.write();
        if let Some((backend, cached_time, _)) = cache.get(&cache_key) {
            if *cached_time >= effective_modified {
                let backend_clone = Arc::clone(backend);
                // Update access time for LRU tracking
                if let Some(entry) = cache.get_mut(&cache_key) {
                    entry.2 = std::time::Instant::now();
                }
                return Ok(backend_clone);
            }
        }
    }
    
    // Cache miss, stale, or delta exists - need to open fresh
    // If delta exists, compact it first
    if has_delta {
        // Open for write to trigger compaction
        let storage = TableStorageBackend::open_for_write(path)?;
        storage.compact()?;
        // Delta is now merged, invalidate cache
        invalidate_storage_cache(path);
    }
    
    // Open backend (now with compacted data)
    let backend = Arc::new(TableStorageBackend::open(path)?);
    
    // Use current time as modified (avoid extra metadata call after compaction)
    let new_modified = if has_delta {
        std::time::SystemTime::now() // Just compacted, use current time
    } else {
        effective_modified // No change, reuse
    };
    
    {
        let mut cache = STORAGE_CACHE.write();
        // Evict LRU entries if cache is full
        evict_lru_cache_entries(&mut cache);
        cache.insert(cache_key, (Arc::clone(&backend), new_modified, std::time::Instant::now()));
    }
    
    Ok(backend)
}

/// V3 Native Query Executor
/// 
/// Executes SQL queries directly on V3 storage using Arrow compute kernels.
/// This replaces the ColumnTable-based execution path.
pub struct ApexExecutor;

/// Query execution result
pub enum ApexResult {
    /// Query returned data rows
    Data(RecordBatch),
    /// Query returned empty result
    Empty(Arc<Schema>),
    /// Query returned a scalar (COUNT, etc.)
    Scalar(i64),
}

impl ApexResult {
    pub fn to_record_batch(self) -> io::Result<RecordBatch> {
        match self {
            ApexResult::Data(batch) => Ok(batch),
            ApexResult::Empty(schema) => Ok(RecordBatch::new_empty(schema)),
            ApexResult::Scalar(val) => {
                let schema = Arc::new(Schema::new(vec![
                    Field::new("result", ArrowDataType::Int64, false),
                ]));
                let array: ArrayRef = Arc::new(Int64Array::from(vec![val]));
                RecordBatch::try_new(schema, vec![array])
                    .map_err(|e| err_data( e.to_string()))
            }
        }
    }

    pub fn num_rows(&self) -> usize {
        match self {
            ApexResult::Data(batch) => batch.num_rows(),
            ApexResult::Empty(_) => 0,
            ApexResult::Scalar(_) => 1,
        }
    }
}

impl ApexExecutor {
    /// Invalidate the storage cache for a specific path
    pub fn invalidate_cache_for_path(path: &Path) {
        invalidate_storage_cache(path);
    }
    
    /// Invalidate all storage cache entries under a directory
    pub fn invalidate_cache_for_dir(dir: &Path) {
        invalidate_storage_cache_dir(dir);
    }
    
    /// Helper to get column refs from statement's required columns
    #[inline]
    fn get_col_refs(stmt: &SelectStatement) -> Option<Vec<String>> {
        stmt.required_columns().filter(|cols| !cols.is_empty())
    }
    
    /// Execute a SQL query on V3 storage (single table)
    pub fn execute(sql: &str, storage_path: &Path) -> io::Result<ApexResult> {
        let stmt = SqlParser::parse(sql)
            .map_err(|e| err_input( e.to_string()))?;

        Self::execute_parsed(stmt, storage_path)
    }

    /// Execute a SQL query with multi-table support (for JOINs)
    pub fn execute_with_base_dir(sql: &str, base_dir: &Path, default_table_path: &Path) -> io::Result<ApexResult> {
        // Support multi-statement execution (e.g., CREATE VIEW; SELECT ...; DROP VIEW;)
        // Parse as multi-statement unconditionally to avoid relying on string heuristics.
        let stmts = {
            // Fast path: check parse cache first (read lock, no allocation on hit)
            let cached = SQL_PARSE_CACHE.read().get(sql).cloned();
            if let Some(stmts) = cached {
                stmts
            } else {
                let stmts = SqlParser::parse_multi(sql)
                    .map_err(|e| err_input(e.to_string()))?;
                // Only cache read-only statements (SELECT) to avoid stale DDL/DML
                let is_select_only = stmts.iter().all(|s| matches!(s, SqlStatement::Select(_) | SqlStatement::Union(_)));
                if is_select_only {
                    let mut cache = SQL_PARSE_CACHE.write();
                    if cache.len() < 1024 {
                        cache.insert(sql.to_string(), stmts.clone());
                    }
                }
                stmts
            }
        };

        if stmts.len() > 1
            || matches!(stmts.first(), Some(SqlStatement::CreateView { .. } | SqlStatement::DropView { .. }))
        {
            return Self::execute_parsed_multi_statements(stmts, base_dir, default_table_path);
        }

        let stmt = stmts
            .into_iter()
            .next()
            .ok_or_else(|| err_input( "No statement to execute"))?;

        Self::execute_parsed_multi(stmt, base_dir, default_table_path)
    }

    /// Execute a parsed SQL statement (single table)
    pub fn execute_parsed(stmt: SqlStatement, storage_path: &Path) -> io::Result<ApexResult> {
        match stmt {
            SqlStatement::Select(select) => Self::execute_select(select, storage_path),
            SqlStatement::Union(union) => Self::execute_union(union, storage_path),
            SqlStatement::Insert { values, columns, .. } => {
                with_table_write_lock(storage_path, || {
                    Self::execute_insert(storage_path, columns.as_deref(), &values)
                })
            }
            SqlStatement::InsertOnConflict { values, columns, conflict_columns, do_update, .. } => {
                with_table_write_lock(storage_path, || {
                    Self::execute_insert_on_conflict(storage_path, columns.as_deref(), &values, &conflict_columns, do_update.as_deref())
                })
            }
            SqlStatement::InsertSelect { columns, query, .. } => {
                with_table_write_lock(storage_path, || {
                    Self::execute_insert_select(storage_path, columns.as_deref(), *query, storage_path.parent().unwrap_or(Path::new(".")), storage_path)
                })
            }
            SqlStatement::Delete { where_clause, .. } => {
                with_table_write_lock(storage_path, || {
                    Self::execute_delete(storage_path, where_clause.as_ref())
                })
            }
            SqlStatement::Update { assignments, where_clause, .. } => {
                with_table_write_lock(storage_path, || {
                    Self::execute_update(storage_path, &assignments, where_clause.as_ref())
                })
            }
            SqlStatement::TruncateTable { .. } => {
                with_table_write_lock(storage_path, || {
                    Self::execute_truncate(storage_path)
                })
            }
            SqlStatement::Explain { stmt, analyze } => {
                Self::execute_explain(*stmt, analyze, storage_path.parent().unwrap_or(Path::new(".")), storage_path)
            }
            SqlStatement::Cte { name, column_aliases, body, main, recursive } => {
                Self::execute_cte(&name, &column_aliases, *body, *main, recursive, storage_path.parent().unwrap_or(Path::new(".")), storage_path)
            }
            SqlStatement::BeginTransaction { read_only } => {
                Self::execute_begin(read_only)
            }
            SqlStatement::Commit => {
                Err(err_input("COMMIT requires txn_id context - use execute_commit_txn()"))
            }
            SqlStatement::Rollback => {
                Err(err_input("ROLLBACK requires txn_id context - use execute_rollback_txn()"))
            }
            _ => Err(io::Error::new(
                io::ErrorKind::Unsupported,
                "DDL statements require base_dir context - use execute_with_base_dir()",
            )),
        }
    }

    /// Execute a parsed SQL statement with multi-table support
    pub fn execute_parsed_multi(stmt: SqlStatement, base_dir: &Path, default_table_path: &Path) -> io::Result<ApexResult> {
        match stmt {
            SqlStatement::Select(select) => {
                if select.joins.is_empty() {
                    // Resolve the actual table path from FROM clause for non-join queries
                    let actual_path = Self::resolve_from_table_path(&select, base_dir, default_table_path);
                    Self::execute_select_with_base_dir(select, &actual_path, base_dir, default_table_path)
                } else {
                    Self::execute_select_with_joins(select, base_dir, default_table_path)
                }
            }
            SqlStatement::Union(union) => Self::execute_union(union, default_table_path),
            // DDL Statements — acquire per-table write lock for concurrency safety
            SqlStatement::CreateTable { table, columns, if_not_exists } => {
                let table_path = Self::resolve_table_path(&table, base_dir, default_table_path);
                with_table_write_lock(&table_path, || {
                    Self::execute_create_table(&table_path, &table, &columns, if_not_exists)
                })
            }
            SqlStatement::DropTable { table, if_exists } => {
                let table_path = Self::resolve_table_path(&table, base_dir, default_table_path);
                with_table_write_lock(&table_path, || {
                    Self::execute_drop_table(&table_path, &table, if_exists)
                })
            }
            SqlStatement::AlterTable { table, operation } => {
                let table_path = Self::resolve_table_path(&table, base_dir, default_table_path);
                with_table_write_lock(&table_path, || {
                    Self::execute_alter_table(&table_path, &table, &operation)
                })
            }
            SqlStatement::TruncateTable { table } => {
                let table_path = Self::resolve_table_path(&table, base_dir, default_table_path);
                with_table_write_lock(&table_path, || {
                    Self::execute_truncate(&table_path)
                })
            }
            // DML Statements — acquire per-table write lock for concurrency safety
            SqlStatement::Insert { table, columns, values } => {
                let table_path = Self::resolve_table_path(&table, base_dir, default_table_path);
                with_table_write_lock(&table_path, || {
                    Self::execute_insert(&table_path, columns.as_deref(), &values)
                })
            }
            SqlStatement::InsertOnConflict { table, columns, values, conflict_columns, do_update } => {
                let table_path = Self::resolve_table_path(&table, base_dir, default_table_path);
                with_table_write_lock(&table_path, || {
                    Self::execute_insert_on_conflict(&table_path, columns.as_deref(), &values, &conflict_columns, do_update.as_deref())
                })
            }
            SqlStatement::AnalyzeTable { table } => {
                let table_path = Self::resolve_table_path(&table, base_dir, default_table_path);
                Self::execute_analyze(&table_path, &table)
            }
            SqlStatement::CopyToParquet { table, file_path } => {
                let table_path = Self::resolve_table_path(&table, base_dir, default_table_path);
                Self::execute_copy_to_parquet(&table_path, &table, &file_path)
            }
            SqlStatement::CopyFromParquet { table, file_path } => {
                let table_path = Self::resolve_table_path(&table, base_dir, default_table_path);
                with_table_write_lock(&table_path, || {
                    Self::execute_copy_from_parquet(&table_path, &table, &file_path, base_dir.as_ref(), default_table_path.as_ref())
                })
            }
            SqlStatement::InsertSelect { table, columns, query } => {
                let table_path = Self::resolve_table_path(&table, base_dir, default_table_path);
                with_table_write_lock(&table_path, || {
                    Self::execute_insert_select(&table_path, columns.as_deref(), *query, base_dir, default_table_path)
                })
            }
            SqlStatement::CreateTableAs { table, query, if_not_exists } => {
                let table_path = Self::resolve_table_path(&table, base_dir, default_table_path);
                with_table_write_lock(&table_path, || {
                    Self::execute_create_table_as(base_dir, default_table_path, &table, *query, if_not_exists)
                })
            }
            SqlStatement::Delete { table, where_clause } => {
                let table_path = Self::resolve_table_path(&table, base_dir, default_table_path);
                with_table_write_lock(&table_path, || {
                    Self::execute_delete(&table_path, where_clause.as_ref())
                })
            }
            SqlStatement::Update { table, assignments, where_clause } => {
                let table_path = Self::resolve_table_path(&table, base_dir, default_table_path);
                with_table_write_lock(&table_path, || {
                    Self::execute_update(&table_path, &assignments, where_clause.as_ref())
                })
            }
            // Index Statements
            SqlStatement::CreateIndex { name, table, columns, unique, index_type, if_not_exists } => {
                Self::execute_create_index(base_dir, default_table_path, &name, &table, &columns, unique, index_type.as_deref(), if_not_exists)
            }
            SqlStatement::DropIndex { name, table, if_exists } => {
                Self::execute_drop_index(base_dir, &name, &table, if_exists)
            }
            // EXPLAIN
            SqlStatement::Explain { stmt, analyze } => {
                Self::execute_explain(*stmt, analyze, base_dir, default_table_path)
            }
            // CTE
            SqlStatement::Cte { name, column_aliases, body, main, recursive } => {
                Self::execute_cte(&name, &column_aliases, *body, *main, recursive, base_dir, default_table_path)
            }
            // Transaction Statements
            SqlStatement::BeginTransaction { read_only } => {
                Self::execute_begin(read_only)
            }
            SqlStatement::Commit => {
                Err(err_input("COMMIT requires txn_id context - use execute_commit_txn()"))
            }
            SqlStatement::Rollback => {
                Err(err_input("ROLLBACK requires txn_id context - use execute_rollback_txn()"))
            }
            SqlStatement::Reindex { table } => {
                Self::execute_reindex(base_dir, default_table_path, &table)
            }
            SqlStatement::Pragma { name, arg } => {
                Self::execute_pragma(base_dir, default_table_path, &name, arg.as_deref())
            }
            // FTS DDL Statements
            SqlStatement::CreateFtsIndex { table, fields, lazy_load, cache_size } => {
                Self::execute_create_fts_index(base_dir, &table, fields.as_deref(), lazy_load, cache_size)
            }
            SqlStatement::DropFtsIndex { table } => {
                Self::execute_drop_fts_index(base_dir, &table)
            }
            SqlStatement::AlterFtsIndexDisable { table } => {
                Self::execute_alter_fts_index_disable(base_dir, &table)
            }
            SqlStatement::ShowFtsIndexes => {
                Self::execute_show_fts_indexes(base_dir)
            }
            _ => Err(io::Error::new(
                io::ErrorKind::Unsupported,
                "Statement type not supported",
            )),
        }
    }

    /// Execute multiple SQL statements separated by semicolons.
    /// Currently used for temporary VIEW support within a single execute() call.
    fn execute_parsed_multi_statements(
        stmts: Vec<SqlStatement>,
        base_dir: &Path,
        default_table_path: &Path,
    ) -> io::Result<ApexResult> {
        let (result, _txn_id) = Self::execute_multi_with_txn(stmts, base_dir, default_table_path, None)?;
        Ok(result)
    }

    /// Execute multiple SQL statements with full transaction support.
    /// Handles BEGIN/COMMIT/ROLLBACK within the statement sequence, tracks txn_id,
    /// invalidates storage cache between write operations, and routes DML inside
    /// active transactions through execute_in_txn.
    /// Returns (last_result, final_txn_id) where final_txn_id reflects the
    /// transaction state after all statements have been executed.
    pub fn execute_multi_with_txn(
        stmts: Vec<SqlStatement>,
        base_dir: &Path,
        default_table_path: &Path,
        initial_txn_id: Option<u64>,
    ) -> io::Result<(ApexResult, Option<u64>)> {
        let mut views: AHashMap<String, SelectStatement> = AHashMap::new();
        let mut last_result: Option<ApexResult> = None;
        let mut current_txn: Option<u64> = initial_txn_id;

        for stmt in stmts {
            // Determine if this is a write operation (for cache invalidation)
            let is_write = matches!(&stmt,
                SqlStatement::Insert { .. } | SqlStatement::InsertOnConflict { .. } |
                SqlStatement::InsertSelect { .. } |
                SqlStatement::Delete { .. } | SqlStatement::Update { .. } |
                SqlStatement::TruncateTable { .. } | SqlStatement::AlterTable { .. } |
                SqlStatement::CreateTable { .. } | SqlStatement::DropTable { .. } |
                SqlStatement::CreateIndex { .. } | SqlStatement::DropIndex { .. } |
                SqlStatement::Reindex { .. }
            );

            match stmt {
                // Transaction commands
                SqlStatement::BeginTransaction { read_only } => {
                    let result = Self::execute_begin(read_only)?;
                    if let ApexResult::Scalar(txn_id) = &result {
                        current_txn = Some(*txn_id as u64);
                    }
                    last_result = Some(result);
                }
                SqlStatement::Commit => {
                    if let Some(txn_id) = current_txn {
                        let result = Self::execute_commit_txn(txn_id, base_dir, default_table_path)?;
                        // Invalidate cache after commit to ensure fresh data
                        invalidate_storage_cache(default_table_path);
                        crate::storage::engine::engine().invalidate(default_table_path);
                        current_txn = None;
                        last_result = Some(result);
                    } else {
                        return Err(err_input("COMMIT without active transaction"));
                    }
                }
                SqlStatement::Rollback => {
                    if let Some(txn_id) = current_txn {
                        let result = Self::execute_rollback_txn(txn_id)?;
                        current_txn = None;
                        last_result = Some(result);
                    } else {
                        return Err(err_input("ROLLBACK without active transaction"));
                    }
                }
                SqlStatement::Savepoint { name } => {
                    if let Some(txn_id) = current_txn {
                        let mgr = crate::txn::txn_manager();
                        mgr.with_context(txn_id, |ctx| {
                            ctx.savepoint(&name);
                            Ok(())
                        })?;
                        last_result = Some(ApexResult::Scalar(0));
                    } else {
                        return Err(err_input("SAVEPOINT without active transaction"));
                    }
                }
                SqlStatement::RollbackToSavepoint { name } => {
                    if let Some(txn_id) = current_txn {
                        let mgr = crate::txn::txn_manager();
                        mgr.with_context(txn_id, |ctx| {
                            ctx.rollback_to_savepoint(&name)
                        })?;
                        last_result = Some(ApexResult::Scalar(0));
                    } else {
                        return Err(err_input("ROLLBACK TO SAVEPOINT without active transaction"));
                    }
                }
                SqlStatement::ReleaseSavepoint { name } => {
                    if let Some(txn_id) = current_txn {
                        let mgr = crate::txn::txn_manager();
                        mgr.with_context(txn_id, |ctx| {
                            ctx.release_savepoint(&name)
                        })?;
                        last_result = Some(ApexResult::Scalar(0));
                    } else {
                        return Err(err_input("RELEASE SAVEPOINT without active transaction"));
                    }
                }
                // View management
                SqlStatement::CreateView { name, stmt } => {
                    let view_name = name.trim_matches('"').to_string();
                    if view_name.eq_ignore_ascii_case("default") {
                        return Err(err_input( "View name conflicts with default table"));
                    }

                    // Disallow conflict with existing table file
                    let table_path = Self::resolve_table_path(&view_name, base_dir, default_table_path);
                    if table_path.exists() {
                        return Err(err_input( "View name conflicts with existing table"));
                    }

                    views.insert(view_name, stmt);
                }
                SqlStatement::DropView { name } => {
                    let view_name = name.trim_matches('"');
                    views.remove(view_name);
                }
                // SELECT with view rewriting
                SqlStatement::Select(mut select) => {
                    select = Self::rewrite_select_views(select, &views);
                    if let Some(txn_id) = current_txn {
                        last_result = Some(Self::execute_in_txn(txn_id, SqlStatement::Select(select), base_dir, default_table_path)?);
                    } else {
                        last_result = Some(Self::execute_parsed_multi(SqlStatement::Select(select), base_dir, default_table_path)?);
                    }
                }
                SqlStatement::Union(union) => {
                    last_result = Some(Self::execute_union(union, default_table_path)?);
                }
                // DML/DDL statements - route through txn if active
                other => {
                    if let Some(txn_id) = current_txn {
                        // Inside transaction: buffer DML through execute_in_txn
                        last_result = Some(Self::execute_in_txn(txn_id, other, base_dir, default_table_path)?);
                    } else {
                        // Outside transaction: execute directly
                        last_result = Some(Self::execute_parsed_multi(other, base_dir, default_table_path)?);
                        // Invalidate cache after write operations to ensure next statement sees fresh data
                        if is_write {
                            invalidate_storage_cache(default_table_path);
                            crate::storage::engine::engine().invalidate(default_table_path);
                        }
                    }
                }
            }
        }

        let result = last_result.ok_or_else(|| err_input( "No query to execute"))?;
        Ok((result, current_txn))
    }

    fn rewrite_select_views(mut select: SelectStatement, views: &AHashMap<String, SelectStatement>) -> SelectStatement {
        // Rewrite FROM clause if it references a VIEW
        if let Some(from) = &select.from {
            match from {
                FromItem::Table { table, alias } => {
                    let table_name = table.trim_matches('"');
                    if let Some(view_stmt) = views.get(table_name) {
                        let alias_name = alias.clone().unwrap_or_else(|| table_name.to_string());
                        select.from = Some(FromItem::Subquery {
                            stmt: Box::new(view_stmt.clone()),
                            alias: alias_name,
                        });
                    }
                }
                _ => {}
            }
        }

        // Rewrite JOIN clauses if they reference VIEWs
        let mut new_joins = Vec::with_capacity(select.joins.len());
        for mut join in select.joins {
            if let FromItem::Table { table, alias } = &join.right {
                let table_name = table.trim_matches('"');
                if let Some(view_stmt) = views.get(table_name) {
                    let alias_name = alias.clone().unwrap_or_else(|| table_name.to_string());
                    join.right = FromItem::Subquery {
                        stmt: Box::new(view_stmt.clone()),
                        alias: alias_name,
                    };
                }
            }
            new_joins.push(join);
        }
        select.joins = new_joins;

        select
    }

    /// Execute SELECT statement (legacy - uses storage_path for subqueries too)
    fn execute_select(stmt: SelectStatement, storage_path: &Path) -> io::Result<ApexResult> {
        // Delegate to the base_dir version, using storage_path's parent as base_dir
        let base_dir = storage_path.parent().unwrap_or(storage_path);
        Self::execute_select_with_base_dir(stmt, storage_path, base_dir, storage_path)
    }

}

// Split impl blocks for ApexExecutor methods
include!("select.rs");
include!("joins.rs");
include!("expressions.rs");
include!("aggregation.rs");
include!("window.rs");
include!("ddl.rs");
include!("dml.rs");

#[cfg(test)]
mod tests;
