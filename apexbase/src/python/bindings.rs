//! V3 PyO3 bindings - On-demand storage engine without ColumnTable
//!
//! This module provides Python bindings that use V3 storage directly,
//! enabling on-demand reading without loading entire tables into memory.

use crate::data::Value;
use crate::storage::{TableStorageBackend, StorageManager, DurabilityLevel, StorageEngine};
use crate::storage::on_demand::ColumnValue;
use crate::query::{ApexExecutor, ApexResult, SqlParser};
use crate::fts::FtsManager;
use crate::fts::FtsConfig;
use fs2::FileExt;
use pyo3::exceptions::{PyIOError, PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use std::collections::HashMap;
use std::sync::Arc;
use std::path::{Path, PathBuf};
use std::fs::{self, File, OpenOptions};
use std::io;
use parking_lot::RwLock;
use arrow::record_batch::RecordBatch;

/// Convert Python dict to HashMap<String, Value>
fn dict_to_values(dict: &Bound<'_, PyDict>) -> PyResult<HashMap<String, Value>> {
    let mut fields = HashMap::new();

    for (key, value) in dict.iter() {
        let key: String = key.extract()?;
        if key == "_id" {
            continue;
        }
        let val = py_to_value(&value)?;
        fields.insert(key, val);
    }

    Ok(fields)
}

/// Convert Python value to Value
fn py_to_value(obj: &Bound<'_, PyAny>) -> PyResult<Value> {
    use pyo3::types::PyBytes;
    
    if obj.is_none() {
        return Ok(Value::Null);
    }

    if let Ok(b) = obj.extract::<bool>() {
        return Ok(Value::Bool(b));
    }

    if let Ok(i) = obj.extract::<i64>() {
        return Ok(Value::Int64(i));
    }

    if let Ok(f) = obj.extract::<f64>() {
        return Ok(Value::Float64(f));
    }

    // Check for bytes BEFORE string (bytes can be extracted as string)
    if obj.is_instance_of::<PyBytes>() {
        if let Ok(bytes) = obj.extract::<Vec<u8>>() {
            return Ok(Value::Binary(bytes));
        }
    }

    if let Ok(s) = obj.extract::<String>() {
        return Ok(Value::String(s));
    }

    if let Ok(bytes) = obj.extract::<Vec<u8>>() {
        return Ok(Value::Binary(bytes));
    }

    Ok(Value::Null)
}

/// Convert ColumnValue to Python object
#[allow(dead_code)]
fn column_value_to_py(py: Python<'_>, val: &ColumnValue) -> PyResult<PyObject> {
    match val {
        ColumnValue::Null => Ok(py.None()),
        ColumnValue::Bool(b) => Ok(b.into_py(py)),
        ColumnValue::Int64(i) => Ok(i.into_py(py)),
        ColumnValue::Float64(f) => Ok(f.into_py(py)),
        ColumnValue::String(s) => Ok(s.into_py(py)),
        ColumnValue::Binary(b) => Ok(b.clone().into_py(py)),
    }
}

/// Convert Value to Python object
fn value_to_py(py: Python<'_>, val: &Value) -> PyResult<PyObject> {
    use pyo3::types::PyBytes;
    
    match val {
        Value::Null => Ok(py.None()),
        Value::Bool(b) => Ok(b.into_py(py)),
        Value::Int8(i) => Ok((*i as i64).into_py(py)),
        Value::Int16(i) => Ok((*i as i64).into_py(py)),
        Value::Int32(i) => Ok((*i as i64).into_py(py)),
        Value::Int64(i) => Ok(i.into_py(py)),
        Value::UInt8(i) => Ok((*i as i64).into_py(py)),
        Value::UInt16(i) => Ok((*i as i64).into_py(py)),
        Value::UInt32(i) => Ok((*i as i64).into_py(py)),
        Value::UInt64(i) => Ok((*i as i64).into_py(py)),
        Value::Float32(f) => Ok((*f as f64).into_py(py)),
        Value::Float64(f) => Ok(f.into_py(py)),
        Value::String(s) => Ok(s.into_py(py)),
        Value::Binary(b) => Ok(PyBytes::new_bound(py, b).into()),
        Value::Json(j) => Ok(j.to_string().into_py(py)),
        Value::Timestamp(t) => Ok(t.into_py(py)),
        Value::Date(d) => Ok(d.into_py(py)),
        Value::Array(arr) => {
            let list = PyList::empty_bound(py);
            for v in arr {
                list.append(value_to_py(py, v)?)?;
            }
            Ok(list.into())
        }
    }
}

/// ApexStorage - On-demand columnar storage engine (V3)
///
/// This storage engine uses V3 format (.apex) for persistence and supports:
/// - On-demand column reading (only loads requested columns)
/// - On-demand row range reading (only loads requested rows)
/// - Soft delete with deleted bitmap
/// - Full SQL query support via ApexExecutor
/// - Cross-platform file locking for concurrent access safety
///   - Read operations use shared locks (multiple readers allowed)
///   - Write operations use exclusive locks (single writer)
/// - Multi-database support: named databases stored in subdirectories
#[pyclass(name = "ApexStorage")]
pub struct ApexStorageImpl {
    /// Root directory (top-level dir; contains both default tables and named-db subdirs)
    root_dir: PathBuf,
    /// Current database name. "" or "default" means root_dir (backward-compat default).
    /// Named databases (e.g. "analytics") reside at root_dir/analytics/.
    current_database: RwLock<String>,
    /// Current base directory = root_dir (default) or root_dir/db_name (named db).
    /// Updated atomically by use_database_().
    base_dir: RwLock<PathBuf>,
    /// Table paths (table_name -> path) - lazily populated
    table_paths: RwLock<HashMap<String, PathBuf>>,
    /// Whether table_paths has been fully scanned from directory
    tables_scanned: RwLock<bool>,
    /// Cached storage backends per table (table_name -> backend)
    /// Backends are opened once and reused for all operations
    cached_backends: RwLock<HashMap<String, Arc<TableStorageBackend>>>,
    /// Current table name
    current_table: RwLock<String>,
    /// FTS Manager (optional) — Arc so it can be shared with the global SQL executor registry
    fts_manager: RwLock<Option<Arc<FtsManager>>>,
    /// FTS index field names per table
    fts_index_fields: RwLock<HashMap<String, Vec<String>>>,
    /// Durability level for ACID guarantees
    durability: DurabilityLevel,
    /// Current active transaction ID (None if not in a transaction)
    current_txn_id: RwLock<Option<u64>>,
    /// Python-level query result cache: SQL -> PyObject
    /// Caches the FINAL Python dict to avoid all conversion overhead on repeated queries
    py_query_cache: RwLock<HashMap<String, PyObject>>,
    /// Auto-flush row threshold (struct-level so it survives backend cache invalidation)
    auto_flush_rows: RwLock<u64>,
    /// Auto-flush byte threshold (struct-level so it survives backend cache invalidation)
    auto_flush_bytes: RwLock<u64>,
}

/// Internal Rust-only methods (not exposed to Python)
impl ApexStorageImpl {
    /// Get the lock file path for a table
    #[inline]
    fn get_lock_path(table_path: &Path) -> PathBuf {
        table_path.with_extension("apex.lock")
    }
    
    /// Acquire a lock on the table (shared for read, exclusive for write).
    /// Uses retry with exponential backoff (100µs → 200µs → ... → 50ms max total wait).
    /// This avoids spurious "Database is locked" errors under concurrent load.
    fn acquire_lock(table_path: &Path, exclusive: bool) -> io::Result<File> {
        let lock_path = Self::get_lock_path(table_path);
        let lock_file = OpenOptions::new()
            .read(true).write(true).create(true).truncate(false)
            .open(&lock_path)?;
        
        let max_wait = std::time::Duration::from_millis(50);
        let mut backoff = std::time::Duration::from_micros(100);
        let start = std::time::Instant::now();
        
        loop {
            let result: io::Result<()> = if exclusive {
                lock_file.try_lock_exclusive()
            } else {
                lock_file.try_lock_shared().map_err(|e| io::Error::new(io::ErrorKind::WouldBlock, e.to_string()))
            };
            
            match result {
                Ok(()) => return Ok(lock_file),
                Err(_) if start.elapsed() < max_wait => {
                    std::thread::sleep(backoff);
                    backoff = (backoff * 2).min(std::time::Duration::from_millis(5));
                }
                Err(e) => {
                    return Err(io::Error::new(
                        io::ErrorKind::WouldBlock,
                        format!("Database is locked (waited {}ms): {}", start.elapsed().as_millis(), e)
                    ));
                }
            }
        }
    }
    
    #[inline]
    fn acquire_read_lock(table_path: &Path) -> io::Result<File> {
        Self::acquire_lock(table_path, false)
    }
    
    #[inline]
    fn acquire_write_lock(table_path: &Path) -> io::Result<File> {
        Self::acquire_lock(table_path, true)
    }
    
    /// Release a lock (unlock and drop the file handle)
    #[inline]
    fn release_lock(lock_file: File) {
        let _ = lock_file.unlock();
        drop(lock_file);
    }

    /// Parse a Python dict {col_name: type_str} into Vec<(String, ColumnType)>
    fn parse_schema_dict(dict: &Bound<'_, PyDict>) -> PyResult<Vec<(String, crate::storage::on_demand::ColumnType)>> {
        use crate::storage::on_demand::ColumnType;
        let mut cols = Vec::with_capacity(dict.len());
        for (key, value) in dict.iter() {
            let col_name: String = key.extract()?;
            let type_str: String = value.extract()?;
            let ct = match type_str.to_lowercase().as_str() {
                "int8" | "i8" => ColumnType::Int8,
                "int16" | "i16" => ColumnType::Int16,
                "int32" | "i32" | "int" => ColumnType::Int32,
                "int64" | "i64" | "integer" => ColumnType::Int64,
                "uint8" | "u8" => ColumnType::UInt8,
                "uint16" | "u16" => ColumnType::UInt16,
                "uint32" | "u32" => ColumnType::UInt32,
                "uint64" | "u64" => ColumnType::UInt64,
                "float32" | "f32" | "float" => ColumnType::Float32,
                "float64" | "f64" | "double" => ColumnType::Float64,
                "bool" | "boolean" => ColumnType::Bool,
                "str" | "string" | "text" | "varchar" => ColumnType::String,
                "bytes" | "binary" => ColumnType::Binary,
                "timestamp" | "datetime" => ColumnType::Timestamp,
                "date" => ColumnType::Date,
                _ => return Err(PyValueError::new_err(format!(
                    "Unknown column type '{}' for column '{}'. Supported: int8, int16, int32, int64, \
                     uint8, uint16, uint32, uint64, float32, float64, bool, string, binary, timestamp, date",
                    type_str, col_name
                ))),
            };
            cols.push((col_name, ct));
        }
        Ok(cols)
    }
    
    /// Get the path for the current table
    #[inline]
    fn get_current_table_path(&self) -> PyResult<PathBuf> {
        let table_name = self.current_table.read().clone();
        if table_name.is_empty() {
            return Err(PyValueError::new_err(
                "No table selected. Call create_table() or use_table() first."
            ));
        }
        let paths = self.table_paths.read();
        if let Some(p) = paths.get(&table_name) {
            return Ok(p.clone());
        }
        drop(paths);
        // Lazy: check disk using current base_dir
        let base_dir = self.current_base_dir();
        let p = base_dir.join(format!("{}.apex", table_name));
        if p.exists() {
            self.table_paths.write().insert(table_name.clone(), p.clone());
            return Ok(p);
        }
        Err(PyValueError::new_err(format!("Table not found: {}", table_name)))
    }
    
    /// Get both table path and name in one lock acquisition (optimization)
    #[inline]
    fn get_current_table_info(&self) -> PyResult<(PathBuf, String)> {
        let table_name = self.current_table.read().clone();
        if table_name.is_empty() {
            return Err(PyValueError::new_err(
                "No table selected. Call create_table() or use_table() first."
            ));
        }
        let path = {
            let paths = self.table_paths.read();
            paths.get(&table_name).cloned()
        };
        if let Some(p) = path {
            return Ok((p, table_name));
        }
        // Lazy: check disk using current base_dir
        let base_dir = self.current_base_dir();
        let p = base_dir.join(format!("{}.apex", table_name));
        if p.exists() {
            self.table_paths.write().insert(table_name.clone(), p.clone());
            return Ok((p, table_name));
        }
        Err(PyValueError::new_err(format!("Table not found: {}", table_name)))
    }
    
    /// Get or create cached backend for current table
    /// Uses open_for_write to ensure existing data is loaded for write operations
    /// Get backend for INSERT operations - memory efficient!
    /// Uses open_for_insert which doesn't load existing column data.
    /// Data is written to delta file and merged on read.
    fn get_backend_for_insert(&self) -> PyResult<Arc<TableStorageBackend>> {
        let table_name = self.current_table.read().clone();
        let table_path = self.get_current_table_path()?;
        let cache_key = format!("{}_insert", table_name);
        
        // Check if backend is already cached
        {
            let backends = self.cached_backends.read();
            if let Some(backend) = backends.get(&cache_key) {
                return Ok(backend.clone());
            }
        }
        
        // Create new backend with open_for_insert (memory efficient)
        let backend = if table_path.exists() {
            TableStorageBackend::open_for_insert_with_durability(&table_path, self.durability)
                .map_err(|e| PyIOError::new_err(e.to_string()))?
        } else {
            TableStorageBackend::create_with_durability(&table_path, self.durability)
                .map_err(|e| PyIOError::new_err(e.to_string()))?
        };
        
        let backend = Arc::new(backend);
        self.cached_backends.write().insert(cache_key, backend.clone());
        
        Ok(backend)
    }
    
    /// Get backend for UPDATE/DELETE operations - loads all data into memory.
    /// This is required because save() rewrites the entire file.
    fn get_backend(&self) -> PyResult<Arc<TableStorageBackend>> {
        let table_name = self.current_table.read().clone();
        let table_path = self.get_current_table_path()?;
        
        // Check if backend is already cached
        {
            let backends = self.cached_backends.read();
            if let Some(backend) = backends.get(&table_name) {
                return Ok(backend.clone());
            }
        }
        
        // Create new backend with durability level and cache it
        // Use open_for_write to ensure existing column data is loaded
        // This is necessary because save() rewrites the entire file from in-memory columns
        let backend = if table_path.exists() {
            TableStorageBackend::open_for_write_with_durability(&table_path, self.durability)
                .map_err(|e| PyIOError::new_err(e.to_string()))?
        } else {
            TableStorageBackend::create_with_durability(&table_path, self.durability)
                .map_err(|e| PyIOError::new_err(e.to_string()))?
        };
        
        let backend = Arc::new(backend);
        self.cached_backends.write().insert(table_name, backend.clone());
        
        Ok(backend)
    }
    
    /// Invalidate cached backend for a table (used when table is dropped or modified externally)
    fn invalidate_backend(&self, table_name: &str) {
        let mut backends = self.cached_backends.write();
        backends.remove(table_name);
        backends.remove(&format!("{}_insert", table_name));
    }

    /// Return current base directory (root_dir for default db, root_dir/db for named db)
    #[inline]
    fn current_base_dir(&self) -> PathBuf {
        self.base_dir.read().clone()
    }
}

#[pymethods]
impl ApexStorageImpl {
    /// Create or open a V3 storage
    ///
    /// Parameters:
    /// - path: Path to the storage file (will use .apex extension)
    /// - drop_if_exists: If true, delete existing database
    /// - durability: Durability level ('fast', 'safe', or 'max')
    #[new]
    #[pyo3(signature = (path, drop_if_exists = false, durability = "fast"))]
    fn new(path: &str, drop_if_exists: bool, durability: &str) -> PyResult<Self> {
        // Parse durability level
        let durability_level = DurabilityLevel::from_str(durability)
            .ok_or_else(|| PyValueError::new_err(
                format!("Invalid durability level '{}'. Must be 'fast', 'safe', or 'max'", durability)
            ))?;
        // Convert to absolute path to avoid issues with relative paths
        let path_obj = PathBuf::from(path);
        let abs_path = if path_obj.is_absolute() {
            path_obj
        } else {
            std::env::current_dir()
                .unwrap_or_else(|_| PathBuf::from("."))
                .join(&path_obj)
        };
        let root_dir = abs_path.parent()
            .map(|p| p.to_path_buf())
            .unwrap_or_else(|| PathBuf::from("."));

        // Handle drop_if_exists
        if drop_if_exists {
            // Remove all .apex files in the directory
            if let Ok(entries) = fs::read_dir(&root_dir) {
                for entry in entries.flatten() {
                    let p = entry.path();
                    if p.extension().map(|e| e == "apex").unwrap_or(false) {
                        let _ = fs::remove_file(&p);
                    }
                }
            }
            
            // Also remove FTS indexes
            let fts_dir = root_dir.join("fts_indexes");
            if fts_dir.exists() {
                let _ = fs::remove_dir_all(&fts_dir);
            }
        }

        // Pre-warm rayon global thread pool so first parallel CSV/parquet read has no thread-startup delay
        rayon::spawn(|| {});

        // No default table - users must explicitly create or use a table
        // Existing .apex files in the directory are discovered lazily via use_table() or list_tables()

        Ok(Self {
            root_dir: root_dir.clone(),
            current_database: RwLock::new(String::new()),
            base_dir: RwLock::new(root_dir),
            table_paths: RwLock::new(HashMap::new()),
            tables_scanned: RwLock::new(false),
            cached_backends: RwLock::new(HashMap::new()),
            current_table: RwLock::new(String::new()),
            fts_manager: RwLock::new(None::<Arc<FtsManager>>),
            fts_index_fields: RwLock::new(HashMap::new()),
            durability: durability_level,
            current_txn_id: RwLock::new(None),
            py_query_cache: RwLock::new(HashMap::new()),
            auto_flush_rows: RwLock::new(0),
            auto_flush_bytes: RwLock::new(0),
        })
    }

    /// Store a single record using StorageEngine
    /// Automatically chooses delta or full write based on conditions
    fn store(&self, py: Python<'_>, data: &Bound<'_, PyDict>) -> PyResult<i64> {
        let fields = dict_to_values(data)?;
        let (table_path, table_name) = self.get_current_table_info()?;
        let durability = self.durability;
        
        // Skip file lock for 'fast' durability — StorageEngine handles thread safety
        // internally via parking_lot::RwLock. File locks only needed for cross-process safety.
        let lock_file = if durability != DurabilityLevel::Fast {
            Some(Self::acquire_write_lock(&table_path)
                .map_err(|e| PyIOError::new_err(e.to_string()))?)
        } else {
            None
        };
        
        // Use StorageEngine for smart write routing
        let result = py.allow_threads(|| {
            let engine = crate::storage::engine::engine();
            let ids = engine.write(&table_path, &[fields], durability)
                .map_err(|e| PyIOError::new_err(e.to_string()))?;
            Ok::<i64, PyErr>(ids.first().copied().unwrap_or(0) as i64)
        });
        
        if let Some(lf) = lock_file {
            Self::release_lock(lf);
        }
        
        // Invalidate local backend cache (StorageEngine handles its own cache)
        self.invalidate_backend(&table_name);
        
        let id = result?;

        // Index in FTS if enabled
        self.index_for_fts(id, data)?;

        Ok(id)
    }

    /// Store multiple records using StorageEngine
    /// Automatically chooses delta or full write based on conditions
    fn store_batch(&self, py: Python<'_>, data: &Bound<'_, PyList>) -> PyResult<Vec<i64>> {
        let num_rows = data.len();
        if num_rows == 0 {
            return Ok(Vec::new());
        }

        // Collect all rows
        let mut rows: Vec<HashMap<String, Value>> = Vec::with_capacity(num_rows);
        for item in data.iter() {
            let dict = item.downcast::<PyDict>()?;
            let fields = dict_to_values(dict)?;
            rows.push(fields);
        }

        let (table_path, table_name) = self.get_current_table_info()?;
        let durability = self.durability;
        
        // Skip file lock for 'fast' durability
        let lock_file = if durability != DurabilityLevel::Fast {
            Some(Self::acquire_write_lock(&table_path)
                .map_err(|e| PyIOError::new_err(e.to_string()))?)
        } else {
            None
        };
        
        // Use StorageEngine for smart write routing
        let result = py.allow_threads(|| {
            let engine = crate::storage::engine::engine();
            engine.write(&table_path, &rows, durability)
                .map_err(|e| PyIOError::new_err(e.to_string()))
        });
        
        if let Some(lf) = lock_file {
            Self::release_lock(lf);
        }
        
        // Invalidate local backend cache
        self.invalidate_backend(&table_name);
        
        let ids = result?;

        // Index in FTS if enabled (batch operation - only if FTS manager exists)
        // OPTIMIZED: Use add_documents_arrow_str (🥈 ~3.3M docs/s, zero-copy &str path)
        {
            let mgr = self.fts_manager.read();
            if mgr.is_some() {
                let table_name = self.current_table.read().clone();
                let index_fields = self.fts_index_fields.read().get(&table_name).cloned();
                
                if let Some(m) = mgr.as_ref() {
                    if let Ok(engine) = m.get_engine(&table_name) {
                        // Determine which fields to index
                        let fields_to_index: Vec<String> = match &index_fields {
                            Some(fields) => fields.clone(),
                            None => {
                                // Auto-detect string fields from first document
                                let mut auto_fields = Vec::new();
                                if let Some(first_item) = data.iter().next() {
                                    if let Ok(dict) = first_item.downcast::<PyDict>() {
                                        for (key, value) in dict.iter() {
                                            if let Ok(key_str) = key.extract::<String>() {
                                                if key_str != "_id" && value.extract::<String>().is_ok() {
                                                    auto_fields.push(key_str);
                                                }
                                            }
                                        }
                                    }
                                }
                                auto_fields
                            }
                        };
                        
                        if !fields_to_index.is_empty() {
                            let num_docs = ids.len();
                            // Build columnar String data — direct per-field lookup, no per-doc HashMap
                            let mut columns: Vec<(String, Vec<String>)> = fields_to_index
                                .iter()
                                .map(|f| (f.clone(), Vec::with_capacity(num_docs)))
                                .collect();
                            
                            for (i, item) in data.iter().enumerate() {
                                if i >= ids.len() { break; }
                                if let Ok(dict) = item.downcast::<PyDict>() {
                                    for (field_idx, field_name) in fields_to_index.iter().enumerate() {
                                        let value = dict.get_item(field_name)
                                            .ok().flatten()
                                            .and_then(|v| v.extract::<String>().ok())
                                            .unwrap_or_default();
                                        columns[field_idx].1.push(value);
                                    }
                                }
                            }
                            
                            // 🥈 add_documents_arrow_str: zero-copy &str slices, ~3.3M docs/s
                            if !columns.is_empty() && !columns[0].1.is_empty() {
                                let doc_ids_u32: Vec<u32> = ids.iter().map(|&id| id as u32).collect();
                                let columns_ref: Vec<(String, Vec<&str>)> = columns.iter()
                                    .map(|(name, vals)| (name.clone(), vals.iter().map(|s| s.as_str()).collect()))
                                    .collect();
                                let _ = py.allow_threads(|| {
                                    engine.add_documents_arrow_str(&doc_ids_u32, columns_ref)
                                });
                            }
                        }
                    }
                }
            }
        }

        Ok(ids.into_iter().map(|id| id as i64).collect())
    }

    /// Store columnar data directly - bypasses row-by-row conversion
    /// Much faster for bulk inserts with homogeneous data
    /// 
    /// Args:
    ///     columns: Dict[str, list] - column name to list of values
    ///     
    /// Returns:
    ///     List[int] - list of generated IDs
    fn store_columnar(&self, py: Python<'_>, columns: &Bound<'_, PyDict>) -> PyResult<Vec<i64>> {
        if columns.is_empty() {
            return Ok(Vec::new());
        }

        // First pass: validate all columns have the same length
        let mut col_lengths: Vec<(String, usize)> = Vec::new();
        for (key, value) in columns.iter() {
            let col_name: String = key.extract()?;
            if col_name == "_id" { continue; }
            
            let list = value.downcast::<PyList>()
                .map_err(|_| PyValueError::new_err(format!("Column '{}' must be a list", col_name)))?;
            col_lengths.push((col_name, list.len()));
        }
        
        if col_lengths.is_empty() {
            return Ok(Vec::new());
        }
        
        // Check all columns have same length
        let first_len = col_lengths[0].1;
        for (name, len) in &col_lengths {
            if *len != first_len {
                return Err(PyValueError::new_err(format!(
                    "All columns must have the same length: '{}' has {} rows, expected {}", 
                    name, len, first_len
                )));
            }
        }
        
        let num_rows = first_len;
        if num_rows == 0 {
            return Ok(Vec::new());
        }

        // Separate columns by type with NULL tracking
        let mut int_columns: HashMap<String, Vec<i64>> = HashMap::new();
        let mut float_columns: HashMap<String, Vec<f64>> = HashMap::new();
        let mut string_columns: HashMap<String, Vec<String>> = HashMap::new();
        let mut binary_columns_map: HashMap<String, Vec<Vec<u8>>> = HashMap::new();
        let mut bool_columns: HashMap<String, Vec<bool>> = HashMap::new();
        let mut null_positions: HashMap<String, Vec<bool>> = HashMap::new();

        for (key, value) in columns.iter() {
            let col_name: String = key.extract()?;
            if col_name == "_id" { continue; }
            
            let list = value.downcast::<PyList>()
                .map_err(|_| PyValueError::new_err(format!("Column '{}' must be a list", col_name)))?;
            
            let col_len = list.len();
            if col_len == 0 { continue; }
            
            // Detect type from first non-None element
            // NOTE: Check bool before int because in Python bool is a subclass of int
            // NOTE: Check bytes before string because PyBytes can also be extracted as str in some pyo3 versions
            let mut col_type: Option<&str> = None;
            for item in list.iter() {
                if !item.is_none() {
                    if item.extract::<bool>().is_ok() && item.get_type().name().map_or(false, |n| n == "bool") {
                        col_type = Some("bool");
                    } else if item.downcast::<pyo3::types::PyBytes>().is_ok() {
                        col_type = Some("bytes");
                    } else if item.extract::<i64>().is_ok() {
                        col_type = Some("int");
                    } else if item.extract::<f64>().is_ok() {
                        col_type = Some("float");
                    } else if item.extract::<String>().is_ok() {
                        col_type = Some("string");
                    }
                    break;
                }
            }
            
            match col_type {
                Some("int") => {
                    let mut vals = Vec::with_capacity(col_len);
                    let mut nulls = Vec::with_capacity(col_len);
                    for item in list.iter() {
                        let is_null = item.is_none();
                        nulls.push(is_null);
                        vals.push(if is_null { 0 } else { item.extract::<i64>().unwrap_or(0) });
                    }
                    int_columns.insert(col_name.clone(), vals);
                    null_positions.insert(col_name, nulls);
                }
                Some("float") => {
                    let mut vals = Vec::with_capacity(col_len);
                    let mut nulls = Vec::with_capacity(col_len);
                    for item in list.iter() {
                        let is_null = item.is_none();
                        nulls.push(is_null);
                        vals.push(if is_null { 0.0 } else { item.extract::<f64>().unwrap_or(0.0) });
                    }
                    float_columns.insert(col_name.clone(), vals);
                    null_positions.insert(col_name, nulls);
                }
                Some("bool") => {
                    let mut vals = Vec::with_capacity(col_len);
                    let mut nulls = Vec::with_capacity(col_len);
                    for item in list.iter() {
                        let is_null = item.is_none();
                        nulls.push(is_null);
                        vals.push(if is_null { false } else { item.extract::<bool>().unwrap_or(false) });
                    }
                    bool_columns.insert(col_name.clone(), vals);
                    null_positions.insert(col_name, nulls);
                }
                Some("bytes") => {
                    let mut vals: Vec<Vec<u8>> = Vec::with_capacity(col_len);
                    let mut nulls = Vec::with_capacity(col_len);
                    for item in list.iter() {
                        let is_null = item.is_none();
                        nulls.push(is_null);
                        if is_null {
                            vals.push(Vec::new());
                        } else if let Ok(b) = item.downcast::<pyo3::types::PyBytes>() {
                            vals.push(b.as_bytes().to_vec());
                        } else if let Ok(s) = item.extract::<Vec<u8>>() {
                            vals.push(s);
                        } else {
                            vals.push(Vec::new());
                        }
                    }
                    binary_columns_map.insert(col_name.clone(), vals);
                    null_positions.insert(col_name, nulls);
                }
                Some("string") | None => {
                    let mut vals = Vec::with_capacity(col_len);
                    let mut nulls = Vec::with_capacity(col_len);
                    for item in list.iter() {
                        let is_null = item.is_none();
                        nulls.push(is_null);
                        vals.push(if is_null { String::new() } else { item.extract::<String>().unwrap_or_default() });
                    }
                    string_columns.insert(col_name.clone(), vals);
                    null_positions.insert(col_name, nulls);
                }
                _ => {}
            }
        }

        if num_rows == 0 {
            return Ok(Vec::new());
        }

        let table_path = self.get_current_table_path()?;
        let table_name = self.current_table.read().clone();
        let durability = self.durability;
        
        // Skip file lock for 'fast' durability
        let lock_file = if durability != DurabilityLevel::Fast {
            Some(Self::acquire_write_lock(&table_path)
                .map_err(|e| PyIOError::new_err(e.to_string()))?)
        } else {
            None
        };
        
        // Save a copy of string_columns for FTS indexing (before insert_typed consumes it)
        let string_columns_for_fts = string_columns.clone();

        // Use StorageEngine for unified write
        let result = py.allow_threads(|| {
            let engine = crate::storage::engine::engine();
            engine.write_typed(
                &table_path,
                int_columns, float_columns, string_columns,
                binary_columns_map,
                bool_columns,
                null_positions,
                durability,
            ).map_err(|e| PyIOError::new_err(e.to_string()))
        });
        
        if let Some(lf) = lock_file {
            Self::release_lock(lf);
        }
        
        // Invalidate local backend cache
        self.invalidate_backend(&table_name);
        // On Windows, engine.insert_cache holds a mmap'd backend after write_typed.
        // Clearing it ensures set_len() in subsequent transaction-commit delete paths succeeds
        // (ERROR_USER_MAPPED_FILE / os error 1224 is triggered when any mmap is open).
        #[cfg(target_os = "windows")]
        crate::storage::engine::engine().invalidate(&table_path);
        
        let ids = result?;
        
        // Index in FTS if enabled - OPTIMIZED: Use add_documents_arrow_str (🥈 zero-copy &str path)
        {
            let mgr = self.fts_manager.read();
            if mgr.is_some() {
                let table_name = self.current_table.read().clone();
                let index_fields = self.fts_index_fields.read().get(&table_name).cloned();
                
                if let Some(m) = mgr.as_ref() {
                    if let Ok(engine) = m.get_engine(&table_name) {
                        // Determine which string fields to index
                        let string_field_names: Vec<String> = match &index_fields {
                            Some(fields) => fields.iter().cloned().filter(|f| string_columns_for_fts.contains_key(f)).collect(),
                            None => string_columns_for_fts.keys().cloned().collect(),
                        };
                        
                        if !string_field_names.is_empty() {
                            // Build owned String columns, then convert to &str for zero-copy call
                            let fts_columns: Vec<(String, Vec<String>)> = string_field_names.iter()
                                .filter_map(|f| string_columns_for_fts.get(f).map(|v| (f.clone(), v.clone())))
                                .collect();
                            
                            // 🥈 add_documents_arrow_str: zero-copy &str slices, ~3.3M docs/s
                            if !fts_columns.is_empty() {
                                let doc_ids_u32: Vec<u32> = ids.iter().map(|&id| id as u32).collect();
                                let columns_ref: Vec<(String, Vec<&str>)> = fts_columns.iter()
                                    .map(|(name, vals)| (name.clone(), vals.iter().map(|s| s.as_str()).collect()))
                                    .collect();
                                let _ = py.allow_threads(|| {
                                    engine.add_documents_arrow_str(&doc_ids_u32, columns_ref)
                                });
                            }
                        }
                    }
                }
            }
        }
        
        Ok(ids.into_iter().map(|id| id as i64).collect())
    }
    
    /// Helper to index a document for FTS (single document - uses slower path)
    fn index_for_fts(&self, id: i64, data: &Bound<'_, PyDict>) -> PyResult<()> {
        let table_name = self.current_table.read().clone();
        let mgr = self.fts_manager.read();
        
        if mgr.is_none() {
            return Ok(());
        }
        
        // Get index fields config
        let index_fields = self.fts_index_fields.read().get(&table_name).cloned();
        
        // Build fields map from dict
        let mut fields = HashMap::new();
        for (key, value) in data.iter() {
            let key_str: String = key.extract()?;
            if key_str == "_id" {
                continue;
            }
            
            // Check if this field should be indexed
            let should_index = match &index_fields {
                Some(idx_fields) => idx_fields.contains(&key_str),
                None => value.extract::<String>().is_ok(), // Index all string fields by default
            };
            
            if should_index {
                if let Ok(s) = value.extract::<String>() {
                    fields.insert(key_str, s);
                }
            }
        }
        
        if fields.is_empty() {
            return Ok(());
        }
        
        // 🥇 Index the document via add_documents_arrow_texts (pre-joined text, zero-copy &str)
        if let Some(m) = mgr.as_ref() {
            if let Ok(engine) = m.get_engine(&table_name) {
                // Pre-join all field values into a single text (fastest path for single doc)
                let joined = fields.values().cloned().collect::<Vec<_>>().join(" ");
                let doc_id = id as u32;
                let _ = engine.add_documents_arrow_texts(&[doc_id], &[joined.as_str()]);
            }
        }
        
        Ok(())
    }

    /// Delete a record by ID using StorageEngine
    fn delete(&self, id: i64) -> PyResult<bool> {
        let table_path = self.get_current_table_path()?;
        let table_name = self.current_table.read().clone();
        let durability = self.durability;
        
        // Skip file lock for 'fast' durability
        let lock_file = if durability != DurabilityLevel::Fast {
            Some(Self::acquire_write_lock(&table_path)
                .map_err(|e| PyIOError::new_err(e.to_string()))?)
        } else {
            None
        };
        
        // Use StorageEngine for unified delete
        let engine = crate::storage::engine::engine();
        let result = engine.delete_one(&table_path, id as u64, durability)
            .map_err(|e| PyIOError::new_err(e.to_string()))?;
        
        if let Some(lf) = lock_file {
            Self::release_lock(lf);
        }
        
        // Invalidate local backend cache
        self.invalidate_backend(&table_name);
        
        Ok(result)
    }

    /// Delete multiple records by IDs using StorageEngine
    fn delete_batch(&self, ids: Vec<i64>) -> PyResult<bool> {
        // Empty list is a successful no-op
        if ids.is_empty() {
            return Ok(true);
        }
        
        let table_path = self.get_current_table_path()?;
        let table_name = self.current_table.read().clone();
        let durability = self.durability;
        
        // Skip file lock for 'fast' durability
        let lock_file = if durability != DurabilityLevel::Fast {
            Some(Self::acquire_write_lock(&table_path)
                .map_err(|e| PyIOError::new_err(e.to_string()))?)
        } else {
            None
        };
        
        // Use StorageEngine for unified delete
        let engine = crate::storage::engine::engine();
        let ids_u64: Vec<u64> = ids.into_iter().map(|id| id as u64).collect();
        let deleted = engine.delete(&table_path, &ids_u64, durability)
            .map_err(|e| PyIOError::new_err(e.to_string()))?;
        
        if let Some(lf) = lock_file {
            Self::release_lock(lf);
        }
        
        // Invalidate local backend cache
        self.invalidate_backend(&table_name);
        
        Ok(deleted > 0)
    }
    
    /// Delete records matching a WHERE clause
    /// Returns the number of deleted rows
    fn delete_where(&self, where_clause: &str) -> PyResult<i64> {
        let table_path = self.get_current_table_path()?;
        let table_name = self.current_table.read().clone();
        
        // Build DELETE SQL statement
        let sql = format!("DELETE FROM {} WHERE {}", table_name, where_clause);
        
        // Execute using ApexExecutor
        let base_dir = self.current_base_dir();
        crate::query::executor::set_query_root_dir(&self.root_dir);
        let exec_result = ApexExecutor::execute_with_base_dir(&sql, &base_dir, &table_path);
        crate::query::executor::clear_query_root_dir();
        let result = exec_result
            .map_err(|e| PyIOError::new_err(e.to_string()))?;
        
        // Invalidate cached backend since data changed
        self.invalidate_backend(&table_name);
        // Invalidate StorageEngine cache so count_rows() sees updated state
        crate::storage::engine::engine().invalidate(&table_path);
        
        // Extract scalar result (number of deleted rows)
        match result {
            ApexResult::Scalar(count) => Ok(count),
            _ => Ok(0),
        }
    }
    
    /// Delete all records (no WHERE clause)
    /// Returns the number of deleted rows
    fn delete_all(&self) -> PyResult<i64> {
        let table_path = self.get_current_table_path()?;
        let table_name = self.current_table.read().clone();
        
        // Build DELETE SQL statement without WHERE
        let sql = format!("DELETE FROM {}", table_name);
        
        // Execute using ApexExecutor
        let base_dir = self.current_base_dir();
        crate::query::executor::set_query_root_dir(&self.root_dir);
        let exec_result = ApexExecutor::execute_with_base_dir(&sql, &base_dir, &table_path);
        crate::query::executor::clear_query_root_dir();
        let result = exec_result
            .map_err(|e| PyIOError::new_err(e.to_string()))?;
        
        // Invalidate cached backend since data changed
        self.invalidate_backend(&table_name);
        // Invalidate StorageEngine cache so count_rows() sees updated state
        crate::storage::engine::engine().invalidate(&table_path);
        
        // Extract scalar result (number of deleted rows)
        match result {
            ApexResult::Scalar(count) => Ok(count),
            _ => Ok(0),
        }
    }

    /// Execute SQL query
    fn execute(&self, py: Python<'_>, sql: &str) -> PyResult<PyObject> {
        let sql_upper = sql.trim().to_uppercase();
        let is_ddl = sql_upper.starts_with("CREATE ") || sql_upper.starts_with("DROP TABLE");
        let is_write_op = sql_upper.starts_with("DELETE") 
            || sql_upper.starts_with("TRUNCATE") 
            || sql_upper.starts_with("UPDATE")
            || sql_upper.starts_with("INSERT")
            || sql_upper.starts_with("ALTER")
            || sql_upper.starts_with("DROP");
        
        // Invalidate cached backend before write operations to avoid stale data
        let table_name = self.current_table.read().clone();
        if is_write_op && !table_name.is_empty() {
            self.invalidate_backend(&table_name);
        }
        
        // For DDL (CREATE/DROP TABLE), don't require a current table
        // For DDL, CTE, or queries with no current table (cross-db qualified refs), fall back to base_dir
        let table_path = self.get_current_table_path().unwrap_or_else(|_| self.current_base_dir());
        if !is_ddl && !sql_upper.starts_with("WITH ") && table_path == self.current_base_dir() {
            // Only enforce table selection for plain single-table ops
            let table_name = self.current_table.read().clone();
            if table_name.is_empty() && !is_ddl {
                // Allow it — executor will use qualified table names from SQL
            }
        }
        
        // PYTHON-LEVEL QUERY RESULT CACHE: return cached PyObject for identical read queries
        // Cache is invalidated on any write operation
        if !is_write_op && !is_ddl && sql_upper.starts_with("SELECT") {
            let cache = self.py_query_cache.read();
            if let Some(cached_obj) = cache.get(&sql_upper) {
                return Ok(cached_obj.clone_ref(py));
            }
        }
        
        // Invalidate py_query_cache on writes
        if is_write_op || is_ddl {
            let mut cache = self.py_query_cache.write();
            cache.clear();
        }

        // FAST PATH: SELECT COUNT(*) FROM <table> — bypass SQL parser + executor entirely.
        // Returns count directly from metadata (active_row_count atomic load).
        // NOTE: result intentionally not cached (count changes on writes).
        if !is_write_op && sql_upper.starts_with("SELECT COUNT(*) FROM ")
            && !sql_upper.contains("WHERE") && !sql_upper.contains("GROUP")
            && !sql_upper.contains("HAVING") && !sql_upper.contains("JOIN")
            && !sql_upper.contains("DISTINCT")
        {
            let after_from = sql_upper["SELECT COUNT(*) FROM ".len()..].trim();
            let tname = after_from.trim_end_matches(';').trim();
            if !tname.is_empty() && !tname.contains(' ') {
                if let Ok(backend) = crate::query::get_cached_backend_pub(&table_path) {
                    let count = backend.active_row_count() as i64;
                    let out = PyDict::new_bound(py);
                    out.set_item("columns", PyList::new_bound(py, ["COUNT(*)"]))?;
                    let row = PyList::new_bound(py, [count]);
                    out.set_item("rows", PyList::new_bound(py, [row]))?;
                    out.set_item("rows_affected", 0i64)?;
                    return Ok(out.into());
                }
            }
        }

        // FAST PATH: SELECT * FROM <table> LIMIT N — skip SQL parse, use pread RCIX directly.
        // Returns columnar dict format (handled by Python client's fast path at line 713).
        if !is_write_op && sql_upper.starts_with("SELECT *") && sql_upper.contains("LIMIT")
            && !sql_upper.contains("WHERE") && !sql_upper.contains("ORDER")
            && !sql_upper.contains("GROUP") && !sql_upper.contains("JOIN") {
            if let Some(limit_str) = sql_upper.rsplit("LIMIT").next() {
                if let Ok(limit) = limit_str.trim().trim_end_matches(';').parse::<usize>() {
                    if let Ok(backend) = crate::query::get_cached_backend_pub(&table_path) {
                        let batch_result = py.allow_threads(|| {
                            match backend.storage.get_or_load_footer() {
                                Ok(Some(footer)) => {
                                    let col_indices: Vec<usize> = (0..footer.schema.column_count()).collect();
                                    backend.storage.to_arrow_batch_pread_rcix(&col_indices, true, limit)
                                }
                                _ => Ok(None),
                            }
                        });
                        if let Ok(Some(batch)) = batch_result {
                            if batch.num_rows() > 0 {
                                let out = PyDict::new_bound(py);
                                let columns_dict = PyDict::new_bound(py);
                                let schema = batch.schema();
                                for col_idx in 0..batch.num_columns() {
                                    let col_name = schema.field(col_idx).name();
                                    let arr = batch.column(col_idx);
                                    let col_list = arrow_col_to_pylist(py, arr)?;
                                    columns_dict.set_item(col_name, col_list)?;
                                }
                                out.set_item("columns_dict", columns_dict)?;
                                out.set_item("rows_affected", 0i64)?;
                                let cached: PyObject = out.clone().into();
                                self.py_query_cache.write().insert(sql_upper.clone(), cached);
                                return Ok(out.into());
                            }
                        }
                    }
                }
            }
        }
        
        // FAST PATH: Point lookup SELECT * FROM <table> WHERE _id = X
        if !is_write_op && sql_upper.starts_with("SELECT") && sql_upper.contains("_ID") {
            // Extract _id = N pattern from WHERE clause
            if let Some(id_pos) = sql_upper.find("_ID") {
                let rest = &sql_upper[id_pos + 3..];
                let rest = rest.trim_start();
                if rest.starts_with('=') {
                    let val_str = rest[1..].trim().trim_end_matches(';');
                    if let Ok(id) = val_str.parse::<u64>() {
                        if let Ok(backend) = crate::query::get_cached_backend_pub(&table_path) {
                            // Primary: retrieve_rcix — page cache, zero Arrow allocation
                            let rcix_result = py.allow_threads(|| backend.storage.retrieve_rcix(id));
                            if let Ok(Some(vals)) = rcix_result {
                                let out = PyDict::new_bound(py);
                                let columns_dict = PyDict::new_bound(py);
                                for (col_name, val) in &vals {
                                    let col_list = PyList::empty_bound(py);
                                    col_list.append(value_to_py(py, val)?)?;
                                    columns_dict.set_item(col_name.as_str(), col_list)?;
                                }
                                out.set_item("columns_dict", columns_dict)?;
                                out.set_item("rows_affected", 0)?;
                                return Ok(out.into());
                            }
                            // Fallback: Arrow batch path (compressed RG or no RCIX index)
                            if let Ok(Some(batch)) = backend.read_row_by_id_to_arrow(id) {
                                if batch.num_rows() > 0 {
                                    let out = PyDict::new_bound(py);
                                    let columns_dict = PyDict::new_bound(py);
                                    let schema = batch.schema();
                                    for col_idx in 0..batch.num_columns() {
                                        let col_name = schema.field(col_idx).name();
                                        let arr = batch.column(col_idx);
                                        let col_list = PyList::empty_bound(py);
                                        for row_idx in 0..batch.num_rows() {
                                            let val = arrow_value_at(arr, row_idx);
                                            col_list.append(value_to_py(py, &val)?)?;
                                        }
                                        columns_dict.set_item(col_name, col_list)?;
                                    }
                                    out.set_item("columns_dict", columns_dict)?;
                                    out.set_item("rows_affected", 0)?;
                                    return Ok(out.into());
                                }
                            }
                        }
                    }
                }
            }
        }

        // FAST PATH: SELECT * WHERE col = 'val' — bypass SQL executor overhead (~1.3ms)
        // Uses mmap scan directly, same as LIMIT N fast path. Handles string equality only.
        if !is_write_op && sql_upper.starts_with("SELECT *")
            && sql_upper.contains("WHERE") && !sql_upper.contains("LIMIT")
            && !sql_upper.contains("ORDER") && !sql_upper.contains("GROUP")
            && !sql_upper.contains("JOIN") && !sql_upper.contains("BETWEEN")
            && !sql_upper.contains(" IN ") && !sql_upper.contains('>')
            && !sql_upper.contains('<') && sql.contains('\'')
        {
            if let Some(where_pos) = sql_upper.find("WHERE") {
                let after_where = sql[where_pos + 5..].trim().trim_end_matches(';');
                if let Some(eq_pos) = after_where.find('=') {
                    let col = after_where[..eq_pos].trim().trim_matches('"').to_string();
                    let rhs = after_where[eq_pos + 1..].trim();
                    if !col.contains(' ') && !col.contains('(') && rhs.starts_with('\'') {
                        if let Some(val_end) = rhs[1..].find('\'') {
                            let val = rhs[1..1 + val_end].to_string();
                            if let Ok(backend) = crate::query::get_cached_backend_pub(&table_path) {
                                let scan_result: io::Result<Option<RecordBatch>> = (|| {
                                    let indices = match backend.scan_string_filter_mmap(&col, &val, None)? {
                                        Some(v) => v,
                                        None => return Ok(None),
                                    };
                                    if indices.is_empty() {
                                        Ok(Some(backend.read_columns_to_arrow(None, 0, Some(0))?))
                                    } else {
                                        Ok(Some(backend.read_columns_by_indices_to_arrow(&indices)?))
                                    }
                                })();
                                if let Ok(Some(batch)) = scan_result {
                                    let schema = batch.schema();
                                    let col_names: Vec<String> = schema.fields().iter()
                                        .map(|f| f.name().clone()).collect();
                                    let mut rows_out: Vec<Vec<Value>> = Vec::with_capacity(batch.num_rows());
                                    for row_idx in 0..batch.num_rows() {
                                        let mut row: Vec<Value> = Vec::with_capacity(batch.num_columns());
                                        for col_idx in 0..batch.num_columns() {
                                            row.push(arrow_value_at(batch.column(col_idx), row_idx));
                                        }
                                        rows_out.push(row);
                                    }
                                    let out = PyDict::new_bound(py);
                                    out.set_item("columns", &col_names)?;
                                    let py_rows = PyList::empty_bound(py);
                                    for row in &rows_out {
                                        let py_row = PyList::empty_bound(py);
                                        for v in row { py_row.append(value_to_py(py, v)?)?; }
                                        py_rows.append(py_row)?;
                                    }
                                    out.set_item("rows", py_rows)?;
                                    out.set_item("rows_affected", 0)?;
                                    let mut cache = self.py_query_cache.write();
                                    cache.insert(sql_upper.clone(), out.clone().into());
                                    return Ok(out.into());
                                }
                            }
                        }
                    }
                }
            }
        }

        // Transaction handling: intercept BEGIN/COMMIT/ROLLBACK/SAVEPOINT
        let is_begin = sql_upper.starts_with("BEGIN");
        let is_commit = sql_upper == "COMMIT" || sql_upper == "COMMIT;";
        let is_rollback = sql_upper == "ROLLBACK" || sql_upper == "ROLLBACK;";
        let is_savepoint = sql_upper.starts_with("SAVEPOINT ");
        let is_rollback_to = sql_upper.starts_with("ROLLBACK TO");
        let is_release = sql_upper.starts_with("RELEASE");
        let current_txn = *self.current_txn_id.read();
        let is_txn_dml = current_txn.is_some() && (is_write_op || sql_upper.starts_with("INSERT"));
        let is_txn_select = current_txn.is_some() && sql_upper.starts_with("SELECT");

        let sql = sql.to_string();
        let base_dir = self.current_base_dir();
        crate::query::executor::set_query_root_dir(&self.root_dir);

        let (columns, rows) = py.allow_threads(|| -> PyResult<(Vec<String>, Vec<Vec<Value>>)> {
            // Transaction-aware execution
            if is_begin {
                let result = ApexExecutor::execute_with_base_dir(&sql, &base_dir, &table_path)
                    .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
                // extract txn_id from Scalar result
                if let ApexResult::Scalar(txn_id) = &result {
                    // Store txn_id - will be set after allow_threads
                    return Ok((vec!["txn_id".to_string()], vec![vec![Value::Int64(*txn_id)]]));
                }
                let batch = result.to_record_batch()
                    .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
                return Ok((vec![], vec![]));
            }

            if is_commit {
                if let Some(txn_id) = current_txn {
                    let result = ApexExecutor::execute_commit_txn(txn_id, &base_dir, &table_path)
                        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
                    if let ApexResult::Scalar(n) = &result {
                        return Ok((vec!["rows_applied".to_string()], vec![vec![Value::Int64(*n)]]));
                    }
                }
                return Ok((vec![], vec![]));
            }

            if is_rollback {
                if let Some(txn_id) = current_txn {
                    ApexExecutor::execute_rollback_txn(txn_id)
                        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
                }
                return Ok((vec![], vec![]));
            }

            // SAVEPOINT name — create savepoint within active transaction
            if is_savepoint {
                if let Some(txn_id) = current_txn {
                    let name = sql.trim().strip_prefix("SAVEPOINT ").or_else(|| sql.trim().strip_prefix("savepoint "))
                        .unwrap_or("").trim().trim_end_matches(';').to_string();
                    let mgr = crate::txn::txn_manager();
                    mgr.with_context(txn_id, |ctx| {
                        ctx.savepoint(&name);
                        Ok(())
                    }).map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
                }
                return Ok((vec![], vec![]));
            }

            // ROLLBACK TO [SAVEPOINT] name — partial rollback
            if is_rollback_to {
                if let Some(txn_id) = current_txn {
                    let upper_trimmed = sql.trim().to_uppercase();
                    let rest = upper_trimmed.strip_prefix("ROLLBACK TO").unwrap_or("").trim();
                    let rest = rest.strip_prefix("SAVEPOINT").unwrap_or(rest).trim().trim_end_matches(';');
                    // Use original case from SQL for the name
                    let name_start = sql.trim().to_uppercase().find(rest).unwrap_or(0);
                    let name = sql.trim()[name_start..].trim().trim_end_matches(';').to_string();
                    let mgr = crate::txn::txn_manager();
                    mgr.with_context(txn_id, |ctx| {
                        ctx.rollback_to_savepoint(&name)
                    }).map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
                }
                return Ok((vec![], vec![]));
            }

            // RELEASE [SAVEPOINT] name — release savepoint
            if is_release {
                if let Some(txn_id) = current_txn {
                    let upper_trimmed = sql.trim().to_uppercase();
                    let rest = upper_trimmed.strip_prefix("RELEASE").unwrap_or("").trim();
                    let rest = rest.strip_prefix("SAVEPOINT").unwrap_or(rest).trim().trim_end_matches(';');
                    let name_start = sql.trim().to_uppercase().find(rest).unwrap_or(0);
                    let name = sql.trim()[name_start..].trim().trim_end_matches(';').to_string();
                    let mgr = crate::txn::txn_manager();
                    mgr.with_context(txn_id, |ctx| {
                        ctx.release_savepoint(&name)
                    }).map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
                }
                return Ok((vec![], vec![]));
            }

            if is_txn_dml || is_txn_select {
                // Inside a transaction: buffer DML writes or read with overlay
                let txn_id = current_txn.unwrap();
                let parsed = SqlParser::parse(&sql)
                    .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
                let result = ApexExecutor::execute_in_txn(txn_id, parsed, &base_dir, &table_path)
                    .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
                if let ApexResult::Scalar(n) = &result {
                    return Ok((vec!["rows_buffered".to_string()], vec![vec![Value::Int64(*n)]]));
                }
                let batch = result.to_record_batch()
                    .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
                let columns: Vec<String> = batch.schema().fields().iter().map(|f| f.name().clone()).collect();
                let mut rows: Vec<Vec<Value>> = Vec::with_capacity(batch.num_rows());
                for row_idx in 0..batch.num_rows() {
                    let mut row: Vec<Value> = Vec::with_capacity(batch.num_columns());
                    for col_idx in 0..batch.num_columns() {
                        row.push(arrow_value_at(batch.column(col_idx), row_idx));
                    }
                    rows.push(row);
                }
                return Ok((columns, rows));
            }

            // Normal (non-transaction) execution
            let result = ApexExecutor::execute_with_base_dir(&sql, &base_dir, &table_path)
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
            
            let batch = result.to_record_batch()
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
            
            // Convert RecordBatch to columns and rows
            let columns: Vec<String> = batch.schema()
                .fields()
                .iter()
                .map(|f| f.name().clone())
                .collect();
            
            let mut rows: Vec<Vec<Value>> = Vec::with_capacity(batch.num_rows());
            for row_idx in 0..batch.num_rows() {
                let mut row: Vec<Value> = Vec::with_capacity(batch.num_columns());
                for col_idx in 0..batch.num_columns() {
                    let arr = batch.column(col_idx);
                    let val = arrow_value_at(arr, row_idx);
                    row.push(val);
                }
                rows.push(row);
            }
            
            Ok((columns, rows))
        })?;
        crate::query::executor::clear_query_root_dir();

        // Update transaction state after execution
        if is_begin {
            // Extract txn_id from first row
            if let Some(row) = rows.first() {
                if let Some(Value::Int64(txn_id)) = row.first() {
                    *self.current_txn_id.write() = Some(*txn_id as u64);
                }
            }
        }
        if is_commit || is_rollback {
            *self.current_txn_id.write() = None;
            // Invalidate backend after transaction completes
            if !table_name.is_empty() {
                self.invalidate_backend(&table_name);
            }
        }

        let out = PyDict::new_bound(py);
        out.set_item("columns", &columns)?;

        let py_rows = PyList::empty_bound(py);
        for row in &rows {
            let py_row = PyList::empty_bound(py);
            for v in row {
                py_row.append(value_to_py(py, v)?)?;
            }
            py_rows.append(py_row)?;
        }
        out.set_item("rows", py_rows)?;
        out.set_item("rows_affected", 0)?;
        
        // Invalidate cached backend AFTER write operations to ensure fresh data on next access
        if is_write_op && !table_name.is_empty() {
            self.invalidate_backend(&table_name);
        }
        
        // After CREATE TABLE, register the new table and set it as current
        if sql_upper.starts_with("CREATE TABLE") || sql_upper.starts_with("CREATE TABLE IF NOT EXISTS") {
            let rest = sql_upper.strip_prefix("CREATE TABLE")
                .unwrap_or("")
                .trim();
            let rest = if rest.starts_with("IF NOT EXISTS") {
                rest.strip_prefix("IF NOT EXISTS").unwrap_or(rest).trim()
            } else {
                rest
            };
            if let Some(name) = rest.split(|c: char| c.is_whitespace() || c == '(').next() {
                let tbl = name.trim_matches(|c: char| c == '"' || c == '\'' || c == '`').to_lowercase();
                if !tbl.is_empty() {
                    let tbl_path = self.current_base_dir().join(format!("{}.apex", tbl));
                    self.table_paths.write().insert(tbl.clone(), tbl_path);
                    *self.current_table.write() = tbl;
                }
            }
        }
        
        // Cache the result for future identical read queries
        let result_obj: PyObject = out.into();
        if !is_write_op && !is_ddl && sql_upper.starts_with("SELECT") {
            let mut cache = self.py_query_cache.write();
            if cache.len() > 200 { cache.clear(); }
            cache.insert(sql_upper.clone(), result_obj.clone_ref(py));
        }
        
        Ok(result_obj)
    }
    
    /// Execute SQL query and return Arrow FFI pointers for zero-copy transfer
    /// Returns (schema_ptr, array_ptr) that can be imported by PyArrow
    fn _execute_arrow_ffi(&self, py: Python<'_>, sql: &str) -> PyResult<(usize, usize)> {
        use arrow::ffi::{FFI_ArrowArray, FFI_ArrowSchema};
        use arrow::array::{StructArray, Array};
        
        let sql = sql.to_string();
        let base_dir = self.current_base_dir();
        // Fall back to base_dir when no table selected (e.g. SELECT * FROM read_csv(...)).
        // Table-function queries don't use the default_table_path at all.
        let table_path = self.get_current_table_path().unwrap_or_else(|_| base_dir.clone());
        crate::query::executor::set_query_root_dir(&self.root_dir);

        // Execute query in Rust thread pool
        let batch = py.allow_threads(|| -> PyResult<RecordBatch> {
            let result = ApexExecutor::execute_with_base_dir(&sql, &base_dir, &table_path)
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
            
            result.to_record_batch()
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))
        })?;
        crate::query::executor::clear_query_root_dir();
        
        // Empty result
        if batch.num_rows() == 0 {
            return Ok((0, 0));
        }
        
        // Convert RecordBatch to StructArray for FFI export
        let struct_array: StructArray = batch.into();
        let array_data = struct_array.to_data();
        
        // Export to FFI
        let (ffi_array, ffi_schema) = arrow::ffi::to_ffi(&array_data)
            .map_err(|e| PyRuntimeError::new_err(format!("FFI export failed: {}", e)))?;
        
        // Leak the FFI structs to get stable pointers (caller must free via _free_arrow_ffi)
        let schema_ptr = Box::into_raw(Box::new(ffi_schema)) as usize;
        let array_ptr = Box::into_raw(Box::new(ffi_array)) as usize;
        
        Ok((schema_ptr, array_ptr))
    }
    
    /// Free Arrow FFI pointers allocated by _execute_arrow_ffi or _query_arrow_ffi
    fn _free_arrow_ffi(&self, schema_ptr: usize, array_ptr: usize) -> PyResult<()> {
        use arrow::ffi::{FFI_ArrowArray, FFI_ArrowSchema};
        
        if schema_ptr != 0 {
            unsafe {
                let _ = Box::from_raw(schema_ptr as *mut FFI_ArrowSchema);
            }
        }
        if array_ptr != 0 {
            unsafe {
                let _ = Box::from_raw(array_ptr as *mut FFI_ArrowArray);
            }
        }
        Ok(())
    }
    
    /// Execute SQL and return Arrow IPC bytes for efficient transfer
    fn _execute_arrow_ipc(&self, py: Python<'_>, sql: &str) -> PyResult<PyObject> {
        use arrow::ipc::writer::StreamWriter;
        use pyo3::types::PyBytes;
        
        // Invalidate cached backend before write operations
        let sql_upper = sql.trim().to_uppercase();
        let is_ddl = sql_upper.starts_with("CREATE ") || sql_upper.starts_with("DROP TABLE");
        let is_write_op = sql_upper.starts_with("DELETE") 
            || sql_upper.starts_with("TRUNCATE") 
            || sql_upper.starts_with("UPDATE")
            || sql_upper.starts_with("INSERT")
            || sql_upper.starts_with("ALTER")
            || sql_upper.starts_with("DROP");
        let table_name = self.current_table.read().clone();
        if is_write_op && !table_name.is_empty() {
            self.invalidate_backend(&table_name);
        }
        
        // Detect multi-statement SQL: contains ';' with non-whitespace content after
        // Also treat BEGIN/COMMIT/ROLLBACK-containing SQL as multi-statement candidate
        let is_multi_stmt = {
            let trimmed = sql.trim().trim_end_matches(';').trim();
            trimmed.contains(';')
        };
        let contains_txn_cmd = sql_upper.contains("BEGIN") || sql_upper.contains("COMMIT") || sql_upper.contains("ROLLBACK");
        
        // For DDL, CTE, multi-statement, or txn-containing SQL, don't require a current table
        let is_cte = sql_upper.starts_with("WITH ");
        // Fall back to base_dir when no table is selected (e.g. cross-db qualified queries)
        let table_path = self.get_current_table_path().unwrap_or_else(|_| self.current_base_dir());
        let base_dir = self.current_base_dir();
        crate::query::executor::set_query_root_dir(&self.root_dir);

        // FAST PATH: SELECT * FROM <table> LIMIT N — build Arrow batch directly from V4
        if !is_multi_stmt && !is_write_op && sql_upper.starts_with("SELECT *") && sql_upper.contains("LIMIT")
            && !sql_upper.contains("WHERE") && !sql_upper.contains("ORDER")
            && !sql_upper.contains("GROUP") && !sql_upper.contains("JOIN") {
            if let Some(limit_str) = sql_upper.rsplit("LIMIT").next() {
                if let Ok(limit) = limit_str.trim().trim_end_matches(';').parse::<usize>() {
                    if let Ok(backend) = crate::query::get_cached_backend_pub(&table_path) {
                        if let Ok(batch) = backend.storage.to_arrow_batch_with_limit(None, false, limit) {
                            if batch.num_rows() > 0 || batch.num_columns() > 0 {
                                let mut buf = Vec::with_capacity(batch.get_array_memory_size() + 256);
                                {
                                    let mut writer = StreamWriter::try_new(&mut buf, batch.schema().as_ref())
                                        .map_err(|e| PyRuntimeError::new_err(format!("IPC writer error: {}", e)))?;
                                    writer.write(&batch)
                                        .map_err(|e| PyRuntimeError::new_err(format!("IPC write error: {}", e)))?;
                                    writer.finish()
                                        .map_err(|e| PyRuntimeError::new_err(format!("IPC finish error: {}", e)))?;
                                }
                                return Ok(PyBytes::new_bound(py, &buf).into());
                            }
                        }
                    }
                }
            }
        }

        let sql = sql.to_string();
        
        // For multi-statement SQL with transaction commands, use txn-aware path
        let current_txn = *self.current_txn_id.read();
        
        let (batch, new_txn_id) = if is_multi_stmt {
            // Multi-statement: parse all and execute with txn support
            py.allow_threads(|| -> PyResult<(RecordBatch, Option<u64>)> {
                let stmts = crate::query::sql_parser::SqlParser::parse_multi(&sql)
                    .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
                let (result, final_txn) = ApexExecutor::execute_multi_with_txn(stmts, &base_dir, &table_path, current_txn)
                    .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
                let batch = result.to_record_batch()
                    .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
                Ok((batch, final_txn))
            })?
        } else {
            // Single statement: standard path
            let batch = py.allow_threads(|| -> PyResult<RecordBatch> {
                let result = ApexExecutor::execute_with_base_dir(&sql, &base_dir, &table_path)
                    .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
                result.to_record_batch()
                    .map_err(|e| PyRuntimeError::new_err(e.to_string()))
            })?;
            (batch, current_txn)
        };
        
        // Update transaction state if changed by multi-statement execution
        if is_multi_stmt && new_txn_id != current_txn {
            *self.current_txn_id.write() = new_txn_id;
        }
        
        // Serialize to IPC format (pre-allocate buffer to avoid reallocations)
        let estimated_size = batch.get_array_memory_size() + 512;
        let mut buf = Vec::with_capacity(estimated_size);
        {
            let mut writer = StreamWriter::try_new(&mut buf, batch.schema().as_ref())
                .map_err(|e| PyRuntimeError::new_err(format!("IPC writer error: {}", e)))?;
            writer.write(&batch)
                .map_err(|e| PyRuntimeError::new_err(format!("IPC write error: {}", e)))?;
            writer.finish()
                .map_err(|e| PyRuntimeError::new_err(format!("IPC finish error: {}", e)))?;
        }
        
        // Invalidate cached backend AFTER write operations
        if (is_write_op || is_multi_stmt) && !table_name.is_empty() {
            self.invalidate_backend(&table_name);
        }
        
        // After DROP TABLE, remove the table from table_paths and invalidate backend
        if sql_upper.starts_with("DROP TABLE") {
            let rest = sql_upper.strip_prefix("DROP TABLE")
                .unwrap_or("")
                .trim();
            let rest = if rest.starts_with("IF EXISTS") {
                rest.strip_prefix("IF EXISTS").unwrap_or(rest).trim()
            } else {
                rest
            };
            if let Some(name) = rest.split(|c: char| c.is_whitespace() || c == ';').next() {
                let tbl = name.trim_matches(|c: char| c == '"' || c == '\'' || c == '`').to_lowercase();
                if !tbl.is_empty() {
                    self.table_paths.write().remove(&tbl);
                    self.invalidate_backend(&tbl);
                    // Clear current table if it was the dropped table
                    if *self.current_table.read() == tbl {
                        *self.current_table.write() = String::new();
                    }
                }
            }
        }
        
        // After CREATE TABLE, register the new table and set it as current
        if sql_upper.starts_with("CREATE TABLE") {
            let rest = sql_upper.strip_prefix("CREATE TABLE")
                .unwrap_or("")
                .trim();
            let rest = if rest.starts_with("IF NOT EXISTS") {
                rest.strip_prefix("IF NOT EXISTS").unwrap_or(rest).trim()
            } else {
                rest
            };
            if let Some(name) = rest.split(|c: char| c.is_whitespace() || c == '(').next() {
                let tbl = name.trim_matches(|c: char| c == '"' || c == '\'' || c == '`').to_lowercase();
                if !tbl.is_empty() {
                    let tbl_path = self.current_base_dir().join(format!("{}.apex", tbl));
                    self.table_paths.write().insert(tbl.clone(), tbl_path);
                    *self.current_table.write() = tbl;
                }
            }
        }
        
        crate::query::executor::clear_query_root_dir();
        // Return as Python bytes
        Ok(PyBytes::new_bound(py, &buf).into())
    }
    
    /// Query with Arrow FFI (zero-copy transfer)
    fn _query_arrow_ffi(&self, py: Python<'_>, where_clause: &str, limit: Option<usize>) -> PyResult<(usize, usize)> {
        use arrow::ffi::{FFI_ArrowArray, FFI_ArrowSchema};
        use arrow::array::{StructArray, Array};
        
        let table_path = self.get_current_table_path()?;
        let base_dir = self.current_base_dir();
        crate::query::executor::set_query_root_dir(&self.root_dir);
        let table_name = self.current_table.read().clone();
        let where_clause = where_clause.to_string();
        
        // Build SQL from where clause using current table name
        let sql = if let Some(lim) = limit {
            if where_clause == "1=1" || where_clause.is_empty() {
                format!("SELECT * FROM \"{}\" LIMIT {}", table_name, lim)
            } else {
                format!("SELECT * FROM \"{}\" WHERE {} LIMIT {}", table_name, where_clause, lim)
            }
        } else {
            if where_clause == "1=1" || where_clause.is_empty() {
                format!("SELECT * FROM \"{}\"", table_name)
            } else {
                format!("SELECT * FROM \"{}\" WHERE {}", table_name, where_clause)
            }
        };
        
        // Execute query
        let batch = py.allow_threads(|| -> PyResult<RecordBatch> {
            let result = ApexExecutor::execute_with_base_dir(&sql, &base_dir, &table_path)
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
            
            result.to_record_batch()
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))
        })?;
        
        // Empty result
        if batch.num_rows() == 0 {
            return Ok((0, 0));
        }
        
        // Convert to StructArray for FFI
        let struct_array: StructArray = batch.into();
        let array_data = struct_array.to_data();
        
        let (ffi_array, ffi_schema) = arrow::ffi::to_ffi(&array_data)
            .map_err(|e| PyRuntimeError::new_err(format!("FFI export failed: {}", e)))?;
        
        let schema_ptr = Box::into_raw(Box::new(ffi_schema)) as usize;
        let array_ptr = Box::into_raw(Box::new(ffi_array)) as usize;
        
        Ok((schema_ptr, array_ptr))
    }

    // ========== Table Management ==========

    /// Use a table
    fn use_table(&self, name: &str) -> PyResult<()> {
        // First check cache
        {
            let paths = self.table_paths.read();
            if paths.contains_key(name) {
                drop(paths);
                *self.current_table.write() = name.to_string();
                return Ok(());
            }
        }
        
        // Table not in cache - check if it exists on disk (lazy discovery)
        let table_path = self.current_base_dir().join(format!("{}.apex", name));
        if table_path.exists() {
            // Add to cache
            self.table_paths.write().insert(name.to_string(), table_path);
            *self.current_table.write() = name.to_string();
            return Ok(());
        }
        
        Err(PyValueError::new_err(format!("Table not found: {}", name)))
    }

    /// Get current table name
    fn current_table(&self) -> String {
        self.current_table.read().clone()
    }

    /// Create a new table, optionally with a pre-defined schema dict {col_name: type_str}.
    /// Pre-defining schema avoids type inference on the first insert.
    #[pyo3(signature = (name, schema=None))]
    fn create_table(&self, name: &str, schema: Option<&Bound<'_, PyDict>>) -> PyResult<()> {
        let mut paths = self.table_paths.write();
        if paths.contains_key(name) {
            // Verify the file actually exists on disk (table_paths may be stale after SQL DROP TABLE)
            let existing_path = self.current_base_dir().join(format!("{}.apex", name));
            if existing_path.exists() {
                return Err(PyValueError::new_err(format!("Table already exists: {}", name)));
            }
            // Stale entry — remove it and proceed with creation
            paths.remove(name);
        }

        let table_path = self.current_base_dir().join(format!("{}.apex", name));
        let engine = crate::storage::engine::engine();

        if let Some(schema_dict) = schema {
            let schema_cols = Self::parse_schema_dict(schema_dict)?;
            engine.create_table_with_schema(&table_path, self.durability, &schema_cols)
                .map_err(|e| PyIOError::new_err(format!("Failed to create table: {}", e)))?;
        } else {
            engine.create_table(&table_path, self.durability)
                .map_err(|e| PyIOError::new_err(format!("Failed to create table: {}", e)))?;
        }
        
        paths.insert(name.to_string(), table_path);
        drop(paths);

        *self.current_table.write() = name.to_string();
        Ok(())
    }

    /// Drop a table
    fn drop_table(&self, name: &str) -> PyResult<()> {
        // Invalidate cached backend first (releases file lock)
        self.invalidate_backend(name);
        
        let mut paths = self.table_paths.write();
        if let Some(path) = paths.remove(name) {
            fs::remove_file(&path)
                .map_err(|e| PyIOError::new_err(format!("Failed to delete table file: {}", e)))?;
        } else {
            return Err(PyValueError::new_err(format!("Table not found: {}", name)));
        }
        drop(paths);

        if *self.current_table.read() == name {
            *self.current_table.write() = String::new();
        }
        Ok(())
    }

    /// List all tables
    fn list_tables(&self) -> Vec<String> {
        // Scan directory for .apex files to ensure we catch tables created via SQL
        let mut tables = Vec::new();
        let base_dir = self.current_base_dir();
        if let Ok(entries) = fs::read_dir(&base_dir) {
            for entry in entries.flatten() {
                let p = entry.path();
                if p.extension().and_then(|e| e.to_str()).map(|s| s == "apex").unwrap_or(false) {
                    if let Some(stem) = p.file_stem().and_then(|s| s.to_str()) {
                        tables.push(stem.to_string());
                    }
                }
            }
        }
        tables.sort();
        tables.dedup();
        tables
    }

    // ========== Multi-Database Operations ==========

    /// Switch to a named database (creates its subdirectory if needed).
    /// "default" or "" means the root directory (backward-compatible default).
    #[pyo3(name = "use_database_")]
    fn use_database_(&self, db_name: &str) -> PyResult<()> {
        let new_base_dir = if db_name.is_empty() || db_name.eq_ignore_ascii_case("default") {
            self.root_dir.clone()
        } else {
            let db_dir = self.root_dir.join(db_name);
            fs::create_dir_all(&db_dir)
                .map_err(|e| PyIOError::new_err(format!("Cannot create database '{}': {}", db_name, e)))?;
            db_dir
        };

        *self.current_database.write() = db_name.to_string();
        *self.base_dir.write() = new_base_dir;

        // Clear all per-database caches
        self.cached_backends.write().clear();
        self.table_paths.write().clear();
        *self.tables_scanned.write() = false;
        *self.current_table.write() = String::new();
        self.py_query_cache.write().clear();

        Ok(())
    }

    /// Return the current database name ("" / "default" means root/default).
    #[pyo3(name = "current_database_")]
    fn current_database_(&self) -> String {
        self.current_database.read().clone()
    }

    /// List all available databases (named subdirectories of root_dir).
    /// "default" is always included to represent the root-level tables.
    #[pyo3(name = "list_databases_")]
    fn list_databases_(&self) -> Vec<String> {
        let mut dbs = vec!["default".to_string()];
        if let Ok(entries) = fs::read_dir(&self.root_dir) {
            for entry in entries.flatten() {
                let p = entry.path();
                if p.is_dir() {
                    if let Some(name) = p.file_name().and_then(|n| n.to_str()) {
                        // Skip hidden dirs and internal dirs
                        if !name.starts_with('.') && name != "fts_indexes" {
                            dbs.push(name.to_string());
                        }
                    }
                }
            }
        }
        dbs.sort();
        dbs.dedup();
        dbs
    }

    /// Get row count for current table (excluding deleted rows) using StorageEngine.
    /// LOCK-FREE: active_count is an AtomicU64 — no file lock needed for this metadata read.
    fn row_count(&self) -> PyResult<u64> {
        let table_path = self.get_current_table_path()?;
        // If file doesn't exist (e.g., after drop_if_exists), return 0
        if !table_path.exists() {
            return Ok(0);
        }
        
        // No file lock needed — active_count is atomic and always consistent
        let engine = crate::storage::engine::engine();
        let count = engine.active_row_count(&table_path)
            .map_err(|e| PyIOError::new_err(e.to_string()))?;
        
        Ok(count)
    }
    
    /// Alias for row_count (compatibility)
    fn count_rows(&self) -> PyResult<u64> {
        self.row_count()
    }

    /// Save current table
    fn save(&self) -> PyResult<()> {
        // V3 storage auto-saves on each operation
        Ok(())
    }
    
    /// Flush changes to disk with fsync
    /// 
    /// For 'safe' and 'max' durability levels, save() automatically calls fsync.
    /// For 'fast' durability, call this method explicitly when you need durability guarantees.
    fn flush(&self) -> PyResult<()> {
        let table_path = self.get_current_table_path()?;
        
        // Acquire shared read lock (sync doesn't modify data, just ensures it's on disk)
        let lock_file = Self::acquire_read_lock(&table_path)
            .map_err(|e| PyIOError::new_err(e.to_string()))?;
        
        let result = {
            let backends = self.cached_backends.read();
            let table_name = self.current_table.read().clone();
            if let Some(backend) = backends.get(&table_name) {
                backend.sync()
                    .map_err(|e| PyIOError::new_err(format!("Failed to sync: {}", e)))
            } else {
                Ok(()) // No backend means no data to sync
            }
        };
        
        Self::release_lock(lock_file);
        result
    }
    
    /// Get the current durability level
    fn get_durability(&self) -> String {
        self.durability.as_str().to_string()
    }
    
    /// Set auto-flush thresholds
    /// 
    /// When either threshold is exceeded during writes, data is automatically 
    /// written to file. Set to 0 to disable the respective threshold.
    /// 
    /// Parameters:
    /// - rows: Auto-flush when pending rows exceed this count (0 = disabled)
    /// - bytes: Auto-flush when estimated memory exceeds this size (0 = disabled)
    #[pyo3(signature = (rows = 0, bytes = 0))]
    fn set_auto_flush(&self, rows: u64, bytes: u64) -> PyResult<()> {
        // Persist at struct level so thresholds survive backend cache invalidation
        *self.auto_flush_rows.write() = rows;
        *self.auto_flush_bytes.write() = bytes;
        // Also apply to cached backend if present
        let mut backends = self.cached_backends.write();
        let table_name = self.current_table.read().clone();
        if let Some(backend) = backends.get_mut(&table_name) {
            backend.set_auto_flush(rows, bytes);
        }
        Ok(())
    }
    
    /// Get current auto-flush configuration
    /// 
    /// Returns a tuple of (rows_threshold, bytes_threshold)
    fn get_auto_flush(&self) -> PyResult<(u64, u64)> {
        Ok((*self.auto_flush_rows.read(), *self.auto_flush_bytes.read()))
    }
    
    /// Get estimated memory usage in bytes
    fn estimate_memory_bytes(&self) -> PyResult<u64> {
        let backends = self.cached_backends.read();
        let table_name = self.current_table.read().clone();
        if let Some(backend) = backends.get(&table_name) {
            let mem = backend.estimate_memory_bytes();
            if mem > 0 {
                return Ok(mem);
            }
        }
        // No in-memory data (flushed to disk): estimate from file size
        drop(backends);
        if let Ok(table_path) = self.get_current_table_path() {
            if let Ok(meta) = std::fs::metadata(&table_path) {
                return Ok(meta.len());
            }
        }
        Ok(0)
    }

    /// Set compression type for the current table.
    /// Only effective on empty tables (row_count == 0); ignored if table has data.
    /// The setting persists across restarts.
    ///
    /// Args:
    ///     compression: "none", "lz4", or "zstd"
    ///
    /// Returns:
    ///     True if applied, False if table is non-empty (no-op)
    fn set_compression(&self, compression: &str) -> PyResult<bool> {
        use crate::storage::on_demand::{CompressionType, OnDemandStorage};
        let comp = CompressionType::from_str_opt(compression)
            .ok_or_else(|| PyValueError::new_err(
                format!("Invalid compression type '{}'. Use 'none', 'lz4', or 'zstd'.", compression)
            ))?;
        let table_path = self.get_current_table_path()?;
        let storage = if table_path.exists() {
            OnDemandStorage::open_with_durability(&table_path, self.durability)
        } else {
            OnDemandStorage::create_with_durability(&table_path, self.durability)
        }.map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        storage.set_compression(comp)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))
    }

    /// Get the current compression type for the current table.
    ///
    /// Returns:
    ///     "none", "lz4", or "zstd"
    fn get_compression(&self) -> PyResult<String> {
        use crate::storage::on_demand::OnDemandStorage;
        let table_path = self.get_current_table_path()?;
        if table_path.exists() {
            let storage = OnDemandStorage::open_with_durability(&table_path, self.durability)
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
            Ok(storage.compression().as_str().to_string())
        } else {
            Ok("none".to_string())
        }
    }

    /// Close storage
    fn close(&self) -> PyResult<()> {
        // Clear per-instance cached backends (releases per-instance references)
        self.cached_backends.write().clear();
        
        // On Windows: release all mmaps so temp directories can be cleaned up.
        // On Unix: mmaps remain valid after atomic rename; keep STORAGE_CACHE alive
        // so the 50ms fast path in get_cached_backend skips stat() calls on next retrieve().
        #[cfg(target_os = "windows")]
        ApexExecutor::invalidate_cache_for_dir(&self.current_base_dir());
        Ok(())
    }
    
    // ========== Retrieve Operations ==========
    
    /// Retrieve a single record by ID
    fn retrieve(&self, py: Python<'_>, id: i64) -> PyResult<Option<PyObject>> {
        let table_path = self.get_current_table_path()?;
        
        if id < 0 {
            return Ok(None);
        }
        
        // ULTRA-FAST PATH: Direct V4 value read - no file lock, no Arrow, no GIL release
        // Skip allow_threads() for sub-0.1ms operations where GIL overhead dominates
        // Use per-instance cached_backends first: no stat() syscalls (~600µs saved vs get_cached_backend_pub).
        let table_name = self.current_table.read().clone();
        let maybe_cached = {
            let cb = self.cached_backends.read();
            cb.get(&table_name).cloned()
        };
        let backend_opt: Option<Arc<TableStorageBackend>> = if let Some(b) = maybe_cached {
            Some(b)
        } else if let Ok(b) = crate::query::get_cached_backend_pub(&table_path) {
            // Populate per-instance cache so next call is zero-syscall
            self.cached_backends.write().insert(table_name.clone(), Arc::clone(&b));
            Some(b)
        } else {
            None
        };
        if let Some(backend) = backend_opt {
            // Release GIL for all Rust computation; re-acquire only for PyDict construction.
            // retrieve_rcix: page-cached RCIX read, handles PLAIN/BITPACK/RLE/StringDict.
            let rcix_result = py.allow_threads(|| backend.storage.retrieve_rcix(id as u64));
            if let Ok(Some(vals)) = rcix_result {
                let dict = PyDict::new_bound(py);
                for (k, v) in vals {
                    dict.set_item(k, value_to_py(py, &v)?)?;
                }
                return Ok(Some(dict.into()));
            }
            // Fallback: may need to (re)create mmap after save_v4 invalidation
            let vals_result = py.allow_threads(|| backend.storage.read_row_by_id_values(id as u64));
            if let Ok(Some(vals)) = vals_result {
                let dict = PyDict::new_bound(py);
                for (k, v) in vals {
                    dict.set_item(k, value_to_py(py, &v)?)?;
                }
                return Ok(Some(dict.into()));
            }
            // Arrow batch cache path: O(1) index lookup + batch.slice(idx, 1)
            let batch_result = py.allow_threads(|| backend.read_row_by_id_to_arrow(id as u64));
            if let Ok(Some(batch)) = batch_result {
                if batch.num_rows() > 0 {
                    let dict = PyDict::new_bound(py);
                    let schema = batch.schema();
                    for col_idx in 0..batch.num_columns() {
                        let col_name = schema.field(col_idx).name();
                        let val = arrow_value_at(batch.column(col_idx), 0);
                        dict.set_item(col_name.as_str(), value_to_py(py, &val)?)?;
                    }
                    return Ok(Some(dict.into()));
                }
            }
        }
        
        // FALLBACK: File lock + Arrow path for edge cases
        let lock_file = Self::acquire_read_lock(&table_path)
            .map_err(|e| PyIOError::new_err(e.to_string()))?;
        
        let base_dir = self.current_base_dir();
        crate::query::executor::set_query_root_dir(&self.root_dir);
        let table_name = self.current_table.read().clone();
        
        let result = py.allow_threads(|| -> PyResult<Option<HashMap<String, Value>>> {
            let sql = format!("SELECT * FROM \"{}\" WHERE _id = {}", table_name, id);
            let result = ApexExecutor::execute_with_base_dir(&sql, &base_dir, &table_path)
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
            
            let batch = result.to_record_batch()
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
            
            if batch.num_rows() == 0 {
                return Ok(None);
            }
            
            let mut row_data = HashMap::new();
            for (col_idx, field) in batch.schema().fields().iter().enumerate() {
                let val = arrow_value_at(batch.column(col_idx), 0);
                row_data.insert(field.name().clone(), val);
            }
            Ok(Some(row_data))
        });
        
        Self::release_lock(lock_file);
        
        let result = result?;
        
        match result {
            None => Ok(None),
            Some(row_data) => {
                let dict = PyDict::new_bound(py);
                for (k, v) in row_data {
                    dict.set_item(k, value_to_py(py, &v)?)?;
                }
                Ok(Some(dict.into()))
            }
        }
    }
    
    /// Retrieve multiple records by IDs
    fn retrieve_many(&self, py: Python<'_>, ids: Vec<i64>) -> PyResult<Vec<PyObject>> {
        if ids.is_empty() {
            return Ok(Vec::new());
        }
        
        let table_path = self.get_current_table_path()?;
        let table_name = self.current_table.read().clone();
        let ids_str = ids.iter().map(|id| id.to_string()).collect::<Vec<_>>().join(",");
        
        let rows = py.allow_threads(|| -> PyResult<Vec<HashMap<String, Value>>> {
            let sql = format!("SELECT * FROM {} WHERE _id IN ({})", table_name, ids_str);
            let result = ApexExecutor::execute(&sql, &table_path)
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
            
            let batch = result.to_record_batch()
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
            
            let mut rows = Vec::with_capacity(batch.num_rows());
            for row_idx in 0..batch.num_rows() {
                let mut row_data = HashMap::new();
                for (col_idx, field) in batch.schema().fields().iter().enumerate() {
                    let val = arrow_value_at(batch.column(col_idx), row_idx);
                    row_data.insert(field.name().clone(), val);
                }
                rows.push(row_data);
            }
            
            Ok(rows)
        })?;
        
        let mut result = Vec::with_capacity(rows.len());
        for row_data in rows {
            let dict = PyDict::new_bound(py);
            for (k, v) in row_data {
                dict.set_item(k, value_to_py(py, &v)?)?;
            }
            result.push(dict.into());
        }
        
        Ok(result)
    }
    
    /// Retrieve all records
    fn retrieve_all(&self, py: Python<'_>) -> PyResult<Vec<PyObject>> {
        let table_path = self.get_current_table_path()?;
        let table_name = self.current_table.read().clone();
        
        let rows = py.allow_threads(|| -> PyResult<Vec<HashMap<String, Value>>> {
            let sql = format!("SELECT * FROM {}", table_name);
            let sql = sql.as_str();
            let result = ApexExecutor::execute(sql, &table_path)
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
            
            let batch = result.to_record_batch()
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
            
            let mut rows = Vec::with_capacity(batch.num_rows());
            for row_idx in 0..batch.num_rows() {
                let mut row_data = HashMap::new();
                for (col_idx, field) in batch.schema().fields().iter().enumerate() {
                    let val = arrow_value_at(batch.column(col_idx), row_idx);
                    row_data.insert(field.name().clone(), val);
                }
                rows.push(row_data);
            }
            
            Ok(rows)
        })?;
        
        let mut result = Vec::with_capacity(rows.len());
        for row_data in rows {
            let dict = PyDict::new_bound(py);
            for (k, v) in row_data {
                dict.set_item(k, value_to_py(py, &v)?)?;
            }
            result.push(dict.into());
        }
        
        Ok(result)
    }
    
    /// Query with WHERE clause
    #[pyo3(signature = (where_clause, limit=None))]
    fn query(&self, py: Python<'_>, where_clause: &str, limit: Option<usize>) -> PyResult<Vec<PyObject>> {
        let table_path = self.get_current_table_path()?;
        let table_name = self.current_table.read().clone();
        
        let rows = py.allow_threads(|| -> PyResult<Vec<HashMap<String, Value>>> {
            let sql = if let Some(lim) = limit {
                format!("SELECT * FROM {} WHERE {} LIMIT {}", table_name, where_clause, lim)
            } else {
                format!("SELECT * FROM {} WHERE {}", table_name, where_clause)
            };
            
            let result = ApexExecutor::execute(&sql, &table_path)
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
            
            let batch = result.to_record_batch()
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
            
            let mut rows = Vec::with_capacity(batch.num_rows());
            for row_idx in 0..batch.num_rows() {
                let mut row_data = HashMap::new();
                for (col_idx, field) in batch.schema().fields().iter().enumerate() {
                    let val = arrow_value_at(batch.column(col_idx), row_idx);
                    row_data.insert(field.name().clone(), val);
                }
                rows.push(row_data);
            }
            
            Ok(rows)
        })?;
        
        let mut result = Vec::with_capacity(rows.len());
        for row_data in rows {
            let dict = PyDict::new_bound(py);
            for (k, v) in row_data {
                dict.set_item(k, value_to_py(py, &v)?)?;
            }
            result.push(dict.into());
        }
        
        Ok(result)
    }
    
    /// Replace a record by ID using StorageEngine
    fn replace(&self, py: Python<'_>, id: i64, data: &Bound<'_, PyDict>) -> PyResult<bool> {
        let fields = dict_to_values(data)?;
        let table_path = self.get_current_table_path()?;
        let table_name = self.current_table.read().clone();
        let durability = self.durability;
        
        // Acquire exclusive write lock
        let lock_file = Self::acquire_write_lock(&table_path)
            .map_err(|e| PyIOError::new_err(e.to_string()))?;
        
        // Use StorageEngine for unified replace
        let result = py.allow_threads(|| {
            let engine = crate::storage::engine::engine();
            engine.replace(&table_path, id as u64, &fields, durability)
                .map_err(|e| PyIOError::new_err(e.to_string()))
        });
        
        Self::release_lock(lock_file);
        
        // Invalidate local backend cache
        self.invalidate_backend(&table_name);
        
        result
    }
    
    // ========== Schema Operations ==========
    
    /// Add a column to current table using StorageEngine
    fn add_column(&self, column_name: &str, column_type: &str) -> PyResult<()> {
        let dtype = match column_type.to_lowercase().as_str() {
            "int" | "int64" | "i64" | "integer" => crate::data::DataType::Int64,
            "float" | "float64" | "f64" | "double" => crate::data::DataType::Float64,
            "bool" | "boolean" => crate::data::DataType::Bool,
            "str" | "string" | "text" => crate::data::DataType::String,
            "bytes" | "binary" => crate::data::DataType::Binary,
            "timestamp" | "datetime" => crate::data::DataType::Timestamp,
            "date" => crate::data::DataType::Date,
            _ => crate::data::DataType::String,
        };
        
        let table_path = self.get_current_table_path()?;
        let table_name = self.current_table.read().clone();
        let durability = self.durability;
        
        // Invalidate local backend cache before operation
        self.invalidate_backend(&table_name);
        
        // Acquire exclusive write lock
        let lock_file = Self::acquire_write_lock(&table_path)
            .map_err(|e| PyIOError::new_err(e.to_string()))?;
        
        // Use StorageEngine for unified add_column
        let engine = crate::storage::engine::engine();
        let result = engine.add_column(&table_path, column_name, dtype, durability)
            .map_err(|e| PyIOError::new_err(e.to_string()));
        
        Self::release_lock(lock_file);
        
        // Invalidate local backend cache after operation
        self.invalidate_backend(&table_name);
        
        result
    }
    
    /// Drop a column from current table using StorageEngine
    fn drop_column(&self, column_name: &str) -> PyResult<()> {
        let table_path = self.get_current_table_path()?;
        let table_name = self.current_table.read().clone();
        let durability = self.durability;
        
        // Invalidate local backend cache before operation
        self.invalidate_backend(&table_name);
        
        // Acquire exclusive write lock
        let lock_file = Self::acquire_write_lock(&table_path)
            .map_err(|e| PyIOError::new_err(e.to_string()))?;
        
        // Use StorageEngine for unified drop_column
        let engine = crate::storage::engine::engine();
        let result = engine.drop_column(&table_path, column_name, durability)
            .map_err(|e| PyIOError::new_err(e.to_string()));
        
        Self::release_lock(lock_file);
        
        // Invalidate local backend cache after operation
        self.invalidate_backend(&table_name);
        
        result
    }
    
    /// Rename a column using StorageEngine
    fn rename_column(&self, old_name: &str, new_name: &str) -> PyResult<()> {
        let table_path = self.get_current_table_path()?;
        let table_name = self.current_table.read().clone();
        let durability = self.durability;
        
        // Acquire exclusive write lock
        let lock_file = Self::acquire_write_lock(&table_path)
            .map_err(|e| PyIOError::new_err(e.to_string()))?;
        
        // Use StorageEngine for unified rename_column
        let engine = crate::storage::engine::engine();
        let result = engine.rename_column(&table_path, old_name, new_name, durability)
            .map_err(|e| PyIOError::new_err(e.to_string()));
        
        Self::release_lock(lock_file);
        
        // Invalidate local backend cache
        self.invalidate_backend(&table_name);
        
        result
    }
    
    /// List fields (columns) in current table using StorageEngine
    fn list_fields(&self) -> PyResult<Vec<String>> {
        let table_path = self.get_current_table_path()?;
        
        // Acquire shared read lock
        let lock_file = Self::acquire_read_lock(&table_path)
            .map_err(|e| PyIOError::new_err(e.to_string()))?;
        
        // Use StorageEngine for unified list_columns
        let engine = crate::storage::engine::engine();
        let columns = engine.list_columns(&table_path)
            .map_err(|e| PyIOError::new_err(e.to_string()))?;
        
        Self::release_lock(lock_file);
        Ok(columns)
    }
    
    /// Get column data type using StorageEngine
    fn get_column_dtype(&self, column_name: &str) -> PyResult<Option<String>> {
        let table_path = self.get_current_table_path()?;
        
        // Acquire shared read lock
        let lock_file = Self::acquire_read_lock(&table_path)
            .map_err(|e| PyIOError::new_err(e.to_string()))?;
        
        // Use StorageEngine for unified get_column_type
        let engine = crate::storage::engine::engine();
        let dtype = engine.get_column_type(&table_path, column_name)
            .map_err(|e| PyIOError::new_err(e.to_string()))?
            .map(|dt| format!("{:?}", dt));
        
        Self::release_lock(lock_file);
        Ok(dtype)
    }
    
    // ========== FTS Operations ==========
    
    /// Initialize FTS for current table
    #[pyo3(name = "_init_fts")]
    #[pyo3(signature = (index_fields=None, lazy_load=false, cache_size=10000))]
    fn init_fts(&self, index_fields: Option<Vec<String>>, lazy_load: bool, cache_size: usize) -> PyResult<()> {
        let table_name = self.current_table.read().clone();

        // Record index field configuration
        if let Some(fields) = index_fields.clone() {
            self.fts_index_fields.write().insert(table_name.clone(), fields);
        }

        // Ensure manager exists
        if self.fts_manager.read().is_none() {
            let fts_dir = self.current_base_dir().join("fts_indexes");
            let config = FtsConfig {
                lazy_load,
                cache_size,
                ..FtsConfig::default()
            };
            let manager = Arc::new(FtsManager::new(&fts_dir, config));
            // Register with the global SQL executor registry (enables MATCH() in PG Wire / Arrow Flight)
            crate::query::executor::register_fts_manager(&self.current_base_dir(), manager.clone());
            *self.fts_manager.write() = Some(manager);
        } else {
            // Already initialized — ensure global registry is up to date
            let mgr_arc = self.fts_manager.read().clone();
            if let Some(m) = mgr_arc {
                crate::query::executor::register_fts_manager(&self.current_base_dir(), m);
            }
        }

        // Touch/create engine for current table
        let mgr = self.fts_manager.read();
        if let Some(m) = mgr.as_ref() {
            let _ = m.get_engine(&table_name);
        }

        Ok(())
    }
    
    /// Check if FTS is enabled
    #[pyo3(name = "_is_fts_enabled")]
    fn is_fts_enabled(&self) -> bool {
        self.fts_manager.read().is_some()
    }
    
    /// Get FTS index fields for current table
    #[pyo3(name = "_get_fts_config")]
    fn get_fts_config(&self) -> Option<Vec<String>> {
        let table_name = self.current_table.read().clone();
        self.fts_index_fields.read().get(&table_name).cloned()
    }
    
    /// FTS search
    #[pyo3(signature = (query, limit=None))]
    fn search_text(&self, py: Python<'_>, query: &str, limit: Option<usize>) -> PyResult<Vec<(i64, f32)>> {
        let table_name = self.current_table.read().clone();
        let mgr = self.fts_manager.read();
        
        if let Some(m) = mgr.as_ref() {
            let engine = m.get_engine(&table_name)
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
            // Release GIL during search for better concurrency
            let results = py.allow_threads(|| {
                engine.search_top_n(query, limit.unwrap_or(100))
                    .map_err(|e| PyRuntimeError::new_err(e.to_string()))
            })?;
            // Return with score=1.0 for each result (nanofts doesn't return scores directly)
            Ok(results.into_iter().map(|id| (id as i64, 1.0f32)).collect())
        } else {
            Err(PyRuntimeError::new_err("FTS not initialized"))
        }
    }

    /// Remove FTS engine for current table (and optionally delete index files)
    #[pyo3(name = "_fts_remove_engine")]
    #[pyo3(signature = (delete_files=false))]
    fn fts_remove_engine(&self, py: Python<'_>, delete_files: bool) -> PyResult<()> {
        let table_name = self.current_table.read().clone();

        // Remove any cached index field configuration for this table
        self.fts_index_fields.write().remove(&table_name);

        let mut mgr = self.fts_manager.write();
        if let Some(m) = mgr.as_ref() {
            // Release GIL during engine removal (I/O operation)
            py.allow_threads(|| {
                m.remove_engine(&table_name, delete_files)
                    .map_err(|e| PyRuntimeError::new_err(e.to_string()))
            })?;
        }

        Ok(())
    }
    
    /// FTS fuzzy search
    #[pyo3(signature = (query, limit=None, _max_distance=None))]
    fn fuzzy_search_text(&self, py: Python<'_>, query: &str, limit: Option<usize>, _max_distance: Option<u8>) -> PyResult<Vec<(i64, f32)>> {
        let table_name = self.current_table.read().clone();
        let mgr = self.fts_manager.read();
        
        if let Some(m) = mgr.as_ref() {
            let engine = m.get_engine(&table_name)
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
            // Release GIL during fuzzy search for better concurrency
            let ids: Vec<u64> = py.allow_threads(|| -> PyResult<Vec<u64>> {
                let result = engine.fuzzy_search(query, limit.unwrap_or(100))
                    .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
                // Convert result handle to Vec<u64>
                Ok(result.page(0, limit.unwrap_or(100)).into_iter().map(|id| id as u64).collect())
            })?;
            Ok(ids.into_iter().map(|id| (id as i64, 1.0f32)).collect())
        } else {
            Err(PyRuntimeError::new_err("FTS not initialized"))
        }
    }
    
    /// Search and retrieve records
    #[pyo3(signature = (query, limit=None))]
    fn search_and_retrieve(&self, py: Python<'_>, query: &str, limit: Option<usize>) -> PyResult<Vec<PyObject>> {
        let results = self.search_text(py, query, limit)?;
        let ids: Vec<i64> = results.into_iter().map(|(id, _)| id).collect();
        self.retrieve_many(py, ids)
    }
    
    /// Index a document for FTS
    #[pyo3(name = "_fts_index")]
    fn fts_index(&self, py: Python<'_>, id: i64, text: &str) -> PyResult<()> {
        let table_name = self.current_table.read().clone();
        let mgr = self.fts_manager.read();
        
        if let Some(m) = mgr.as_ref() {
            let engine = m.get_engine(&table_name)
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
            let mut fields = HashMap::new();
            fields.insert("content".to_string(), text.to_string());
            // Release GIL during indexing operation
            py.allow_threads(|| {
                engine.add_document(id as u64, fields)
                    .map_err(|e| PyRuntimeError::new_err(e.to_string()))
            })?;
        }
        
        Ok(())
    }
    
    /// Remove a document from FTS index
    #[pyo3(name = "_fts_remove")]
    fn fts_remove(&self, py: Python<'_>, id: i64) -> PyResult<()> {
        let table_name = self.current_table.read().clone();
        let mgr = self.fts_manager.read();
        
        if let Some(m) = mgr.as_ref() {
            let engine = m.get_engine(&table_name)
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
            // Release GIL during remove operation
            py.allow_threads(|| {
                engine.remove_document(id as u64)
                    .map_err(|e| PyRuntimeError::new_err(e.to_string()))
            })?;
        }
        
        Ok(())
    }
    
    /// Flush FTS index
    #[pyo3(name = "_fts_flush")]
    fn fts_flush(&self, py: Python<'_>) -> PyResult<()> {
        let table_name = self.current_table.read().clone();
        let mgr = self.fts_manager.read();
        
        if let Some(m) = mgr.as_ref() {
            let engine = m.get_engine(&table_name)
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
            // Release GIL during flush (I/O operation)
            py.allow_threads(|| {
                engine.flush()
                    .map_err(|e| PyRuntimeError::new_err(e.to_string()))
            })?;
        }
        
        Ok(())
    }
    
    /// Bulk index columnar string data into FTS (no storage write, GIL released)
    #[pyo3(name = "_fts_index_columns")]
    fn fts_index_columns(
        &self,
        py: Python<'_>,
        ids: Vec<i64>,
        columns: HashMap<String, Vec<String>>,
    ) -> PyResult<usize> {
        if ids.is_empty() || columns.is_empty() {
            return Ok(0);
        }
        let table_name = self.current_table.read().clone();
        let mgr = self.fts_manager.read();
        if let Some(m) = mgr.as_ref() {
            if let Ok(engine) = m.get_engine(&table_name) {
                let count = ids.len();
                let doc_ids_u32: Vec<u32> = ids.iter().map(|&id| id as u32).collect();
                // Build owned Vec<String> columns then borrow as &str — zero extra copy
                let owned: Vec<(String, Vec<String>)> = columns.into_iter().collect();
                let columns_ref: Vec<(String, Vec<&str>)> = owned.iter()
                    .map(|(name, vals)| (name.clone(), vals.iter().map(|s| s.as_str()).collect()))
                    .collect();
                py.allow_threads(|| {
                    let _ = engine.add_documents_arrow_str(&doc_ids_u32, columns_ref);
                });
                return Ok(count);
            }
        }
        Ok(0)
    }

    /// Heap-based TopK vector distance search — O(n log k), faster than ORDER BY + LIMIT.
    ///
    /// Builds a `SELECT * FROM topk_distance(col, [q], k, 'metric')` SQL and executes
    /// it via Arrow FFI for zero-copy result transfer.
    ///
    /// Parameters:
    /// - col: name of the binary vector column
    /// - query_bytes: raw little-endian float32 bytes of the query vector
    /// - k: number of nearest neighbours to return
    /// - metric: distance metric name ("l2", "cosine", "dot", "l1", "linf", "l2_squared")
    ///
    /// Returns (schema_ptr, array_ptr) for PyArrow import, or (0, 0) for empty result.
    #[pyo3(name = "_topk_distance_ffi")]
    fn topk_distance_ffi(
        &self,
        py: Python<'_>,
        col: &str,
        query_bytes: &[u8],
        k: usize,
        metric: &str,
    ) -> PyResult<(usize, usize)> {
        use arrow::ffi::{FFI_ArrowArray, FFI_ArrowSchema};
        use arrow::array::{StructArray, Array};
        use crate::query::vector_ops::bytes_to_query_vec_f32;

        let query = bytes_to_query_vec_f32(query_bytes).ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err(
                "_topk_distance_ffi: query_bytes must be raw little-endian float32 bytes",
            )
        })?;

        let q_str = query
            .iter()
            .map(|v| format!("{:.7}", v))
            .collect::<Vec<_>>()
            .join(",");
        let sql = format!(
            "SELECT * FROM topk_distance({}, [{}], {}, '{}')",
            col, q_str, k, metric
        );

        let base_dir = self.current_base_dir();
        let table_path = self.get_current_table_path()
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        crate::query::executor::set_query_root_dir(&self.root_dir);

        let batch = py.allow_threads(|| -> PyResult<RecordBatch> {
            let result = ApexExecutor::execute_with_base_dir(&sql, &base_dir, &table_path)
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
            result
                .to_record_batch()
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))
        })?;
        crate::query::executor::clear_query_root_dir();

        if batch.num_rows() == 0 {
            return Ok((0, 0));
        }

        let struct_array: StructArray = batch.into();
        let array_data = struct_array.to_data();
        let (ffi_array, ffi_schema) = arrow::ffi::to_ffi(&array_data)
            .map_err(|e| PyRuntimeError::new_err(format!("FFI export failed: {}", e)))?;

        let schema_ptr = Box::into_raw(Box::new(ffi_schema)) as usize;
        let array_ptr  = Box::into_raw(Box::new(ffi_array))  as usize;
        Ok((schema_ptr, array_ptr))
    }

    /// Get FTS stats
    fn get_fts_stats(&self) -> PyResult<Option<(usize, usize)>> {
        let table_name = self.current_table.read().clone();
        let mgr = self.fts_manager.read();
        
        if let Some(m) = mgr.as_ref() {
            if let Ok(engine) = m.get_engine(&table_name) {
                let stats = engine.stats();
                let doc_count = stats.get("doc_count").copied().unwrap_or(0) as usize;
                let term_count = stats.get("term_count").copied().unwrap_or(0) as usize;
                Ok(Some((doc_count, term_count)))
            } else {
                Ok(None)
            }
        } else {
            Ok(None)
        }
    }
}

/// Batch-convert an Arrow column to a Python list. Single downcast per column.
/// Much faster than calling arrow_value_at() per element (no per-row dispatch/downcast/Value alloc).
fn arrow_col_to_pylist(py: Python<'_>, arr: &arrow::array::ArrayRef) -> PyResult<PyObject> {
    use arrow::array::*;
    use arrow::datatypes::DataType as ArrowDT;
    use pyo3::types::PyList;
    let n = arr.len();
    match arr.data_type() {
        ArrowDT::Int64 => {
            let a = arr.as_any().downcast_ref::<Int64Array>().unwrap();
            let has_nulls = a.null_count() > 0;
            if !has_nulls {
                Ok(PyList::new_bound(py, a.values().iter().take(n)).into())
            } else {
                let list = PyList::empty_bound(py);
                for i in 0..n {
                    if a.is_null(i) { list.append(py.None())?; } else { list.append(a.value(i))?; }
                }
                Ok(list.into())
            }
        }
        ArrowDT::Float64 => {
            let a = arr.as_any().downcast_ref::<Float64Array>().unwrap();
            let has_nulls = a.null_count() > 0;
            if !has_nulls {
                Ok(PyList::new_bound(py, a.values().iter().take(n)).into())
            } else {
                let list = PyList::empty_bound(py);
                for i in 0..n {
                    if a.is_null(i) { list.append(py.None())?; } else { list.append(a.value(i))?; }
                }
                Ok(list.into())
            }
        }
        ArrowDT::Utf8 => {
            let a = arr.as_any().downcast_ref::<StringArray>().unwrap();
            let list = PyList::empty_bound(py);
            for i in 0..n {
                if a.is_null(i) {
                    list.append(py.None())?;
                } else {
                    let s = a.value(i);
                    if s == "\x00__NULL__\x00" { list.append(py.None())?; } else { list.append(s)?; }
                }
            }
            Ok(list.into())
        }
        ArrowDT::Boolean => {
            let a = arr.as_any().downcast_ref::<BooleanArray>().unwrap();
            let list = PyList::empty_bound(py);
            for i in 0..n {
                if a.is_null(i) { list.append(py.None())?; } else { list.append(a.value(i))?; }
            }
            Ok(list.into())
        }
        _ => {
            // Fallback: per-element generic path
            let list = PyList::empty_bound(py);
            for i in 0..n {
                list.append(value_to_py(py, &arrow_value_at(arr, i))?)?;
            }
            Ok(list.into())
        }
    }
}

/// Extract a Value from an Arrow array at a given index
fn arrow_value_at(array: &arrow::array::ArrayRef, idx: usize) -> Value {
    use arrow::array::*;
    use arrow::datatypes::DataType as ArrowDataType;

    if array.is_null(idx) {
        return Value::Null;
    }

    match array.data_type() {
        ArrowDataType::Int64 => {
            let arr = array.as_any().downcast_ref::<Int64Array>().unwrap();
            Value::Int64(arr.value(idx))
        }
        ArrowDataType::Int32 => {
            let arr = array.as_any().downcast_ref::<Int32Array>().unwrap();
            Value::Int64(arr.value(idx) as i64)
        }
        ArrowDataType::Float64 => {
            let arr = array.as_any().downcast_ref::<Float64Array>().unwrap();
            Value::Float64(arr.value(idx))
        }
        ArrowDataType::Utf8 => {
            let arr = array.as_any().downcast_ref::<StringArray>().unwrap();
            let s = arr.value(idx);
            // Check for NULL marker
            if s == "\x00__NULL__\x00" {
                Value::Null
            } else {
                Value::String(s.to_string())
            }
        }
        ArrowDataType::Boolean => {
            let arr = array.as_any().downcast_ref::<BooleanArray>().unwrap();
            Value::Bool(arr.value(idx))
        }
        ArrowDataType::UInt64 => {
            let arr = array.as_any().downcast_ref::<UInt64Array>().unwrap();
            Value::UInt64(arr.value(idx))
        }
        ArrowDataType::Binary => {
            let arr = array.as_any().downcast_ref::<BinaryArray>().unwrap();
            Value::Binary(arr.value(idx).to_vec())
        }
        ArrowDataType::Dictionary(_, _) => {
            // Handle DictionaryArray<UInt32Type> with Utf8 values
            use arrow::datatypes::UInt32Type;
            if let Some(dict_arr) = array.as_any().downcast_ref::<DictionaryArray<UInt32Type>>() {
                if dict_arr.is_null(idx) {
                    Value::Null
                } else {
                    let key = dict_arr.keys().value(idx) as usize;
                    let values = dict_arr.values();
                    if let Some(str_values) = values.as_any().downcast_ref::<StringArray>() {
                        if key < str_values.len() {
                            let s = str_values.value(key);
                            if s == "\x00__NULL__\x00" {
                                Value::Null
                            } else {
                                Value::String(s.to_string())
                            }
                        } else {
                            Value::Null
                        }
                    } else {
                        Value::Null
                    }
                }
            } else {
                Value::Null
            }
        }
        _ => Value::Null,
    }
}
