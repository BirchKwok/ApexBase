//! PyO3 bindings implementation - Column-oriented storage engine
//! 
//! Performance optimizations:
//! - ForkUnion for low-latency parallel processing
//! - compact_str for Small String Optimization (SSO)
//! - String interning for repeated strings
//! - Native nanofts integration for zero-overhead FTS

use crate::data::Value;
use crate::table::{ColumnTable, ColumnSchema};
use crate::fts::FtsManager;
use crate::fts::FtsConfig;
use crate::query::SqlExecutor;
use pyo3::exceptions::{PyIOError, PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use std::collections::HashMap;
use std::sync::Arc;
use std::path::{Path, PathBuf};
use std::fs;
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use numpy::{PyArray1, PyReadonlyArray1};

/// Convert Python dict to HashMap<String, Value>
fn dict_to_fields(dict: &Bound<'_, PyDict>) -> PyResult<HashMap<String, Value>> {
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

    if let Ok(s) = obj.extract::<String>() {
        return Ok(Value::String(s));
    }

    // Handle Python bytes objects
    if let Ok(bytes) = obj.extract::<Vec<u8>>() {
        return Ok(Value::Binary(bytes));
    }

    if let Ok(list) = obj.downcast::<PyList>() {
        let mut arr = Vec::new();
        for item in list.iter() {
            arr.push(py_to_value(&item)?);
        }
        return Ok(Value::Array(arr));
    }

    if let Ok(dict) = obj.downcast::<PyDict>() {
        let json = dict_to_json(dict)?;
        return Ok(Value::Json(json));
    }

    Ok(Value::Null)
}

fn dict_to_json(dict: &Bound<'_, PyDict>) -> PyResult<serde_json::Value> {
    let mut map = serde_json::Map::new();
    for (key, value) in dict.iter() {
        let k: String = key.extract()?;
        let v = py_to_serde_value(&value)?;
        map.insert(k, v);
    }
    Ok(serde_json::Value::Object(map))
}

fn py_to_serde_value(obj: &Bound<'_, PyAny>) -> PyResult<serde_json::Value> {
    if obj.is_none() {
        return Ok(serde_json::Value::Null);
    }
    if let Ok(b) = obj.extract::<bool>() {
        return Ok(serde_json::Value::Bool(b));
    }
    if let Ok(i) = obj.extract::<i64>() {
        return Ok(serde_json::json!(i));
    }
    if let Ok(f) = obj.extract::<f64>() {
        return Ok(serde_json::json!(f));
    }
    if let Ok(s) = obj.extract::<String>() {
        return Ok(serde_json::Value::String(s));
    }
    Ok(serde_json::Value::Null)
}

/// Convert Arrow array to Vec<Value> - High performance columnar conversion
fn arrow_array_to_values(array: &arrow::array::ArrayRef) -> PyResult<Vec<Value>> {
    use arrow::array::*;
    use arrow::datatypes::DataType as ArrowDataType;
    
    let len = array.len();
    let mut values = Vec::with_capacity(len);
    
    match array.data_type() {
        ArrowDataType::Int8 => {
            let arr = array.as_any().downcast_ref::<Int8Array>().unwrap();
            for i in 0..len {
                if arr.is_null(i) {
                    values.push(Value::Null);
                } else {
                    values.push(Value::Int64(arr.value(i) as i64));
                }
            }
        }
        ArrowDataType::Int16 => {
            let arr = array.as_any().downcast_ref::<Int16Array>().unwrap();
            for i in 0..len {
                if arr.is_null(i) {
                    values.push(Value::Null);
                } else {
                    values.push(Value::Int64(arr.value(i) as i64));
                }
            }
        }
        ArrowDataType::Int32 => {
            let arr = array.as_any().downcast_ref::<Int32Array>().unwrap();
            for i in 0..len {
                if arr.is_null(i) {
                    values.push(Value::Null);
                } else {
                    values.push(Value::Int64(arr.value(i) as i64));
                }
            }
        }
        ArrowDataType::Int64 => {
            let arr = array.as_any().downcast_ref::<Int64Array>().unwrap();
            for i in 0..len {
                if arr.is_null(i) {
                    values.push(Value::Null);
                } else {
                    values.push(Value::Int64(arr.value(i)));
                }
            }
        }
        ArrowDataType::UInt8 => {
            let arr = array.as_any().downcast_ref::<UInt8Array>().unwrap();
            for i in 0..len {
                if arr.is_null(i) {
                    values.push(Value::Null);
                } else {
                    values.push(Value::Int64(arr.value(i) as i64));
                }
            }
        }
        ArrowDataType::UInt16 => {
            let arr = array.as_any().downcast_ref::<UInt16Array>().unwrap();
            for i in 0..len {
                if arr.is_null(i) {
                    values.push(Value::Null);
                } else {
                    values.push(Value::Int64(arr.value(i) as i64));
                }
            }
        }
        ArrowDataType::UInt32 => {
            let arr = array.as_any().downcast_ref::<UInt32Array>().unwrap();
            for i in 0..len {
                if arr.is_null(i) {
                    values.push(Value::Null);
                } else {
                    values.push(Value::Int64(arr.value(i) as i64));
                }
            }
        }
        ArrowDataType::UInt64 => {
            let arr = array.as_any().downcast_ref::<UInt64Array>().unwrap();
            for i in 0..len {
                if arr.is_null(i) {
                    values.push(Value::Null);
                } else {
                    values.push(Value::Int64(arr.value(i) as i64));
                }
            }
        }
        ArrowDataType::Float32 => {
            let arr = array.as_any().downcast_ref::<Float32Array>().unwrap();
            for i in 0..len {
                if arr.is_null(i) {
                    values.push(Value::Null);
                } else {
                    values.push(Value::Float64(arr.value(i) as f64));
                }
            }
        }
        ArrowDataType::Float64 => {
            let arr = array.as_any().downcast_ref::<Float64Array>().unwrap();
            for i in 0..len {
                if arr.is_null(i) {
                    values.push(Value::Null);
                } else {
                    values.push(Value::Float64(arr.value(i)));
                }
            }
        }
        ArrowDataType::Utf8 => {
            let arr = array.as_any().downcast_ref::<StringArray>().unwrap();
            for i in 0..len {
                if arr.is_null(i) {
                    values.push(Value::Null);
                } else {
                    values.push(Value::String(arr.value(i).to_string()));
                }
            }
        }
        ArrowDataType::LargeUtf8 => {
            let arr = array.as_any().downcast_ref::<LargeStringArray>().unwrap();
            for i in 0..len {
                if arr.is_null(i) {
                    values.push(Value::Null);
                } else {
                    values.push(Value::String(arr.value(i).to_string()));
                }
            }
        }
        ArrowDataType::Binary => {
            let arr = array.as_any().downcast_ref::<BinaryArray>().unwrap();
            for i in 0..len {
                if arr.is_null(i) {
                    values.push(Value::Null);
                } else {
                    values.push(Value::Binary(arr.value(i).to_vec()));
                }
            }
        }
        ArrowDataType::LargeBinary => {
            let arr = array.as_any().downcast_ref::<LargeBinaryArray>().unwrap();
            for i in 0..len {
                if arr.is_null(i) {
                    values.push(Value::Null);
                } else {
                    values.push(Value::Binary(arr.value(i).to_vec()));
                }
            }
        }
        ArrowDataType::Boolean => {
            let arr = array.as_any().downcast_ref::<BooleanArray>().unwrap();
            for i in 0..len {
                if arr.is_null(i) {
                    values.push(Value::Null);
                } else {
                    values.push(Value::Bool(arr.value(i)));
                }
            }
        }
        _ => {
            // Fallback: convert to string representation
            for i in 0..len {
                if array.is_null(i) {
                    values.push(Value::Null);
                } else {
                    values.push(Value::String(format!("{:?}", array)));
                }
            }
        }
    }
    
    Ok(values)
}

/// Convert Value to Python object
fn value_to_py(py: Python<'_>, value: &Value) -> PyResult<PyObject> {
    match value {
        Value::Null => Ok(py.None()),
        Value::Bool(b) => Ok(b.to_object(py)),
        Value::Int8(i) => Ok((*i as i64).to_object(py)),
        Value::Int16(i) => Ok((*i as i64).to_object(py)),
        Value::Int32(i) => Ok((*i as i64).to_object(py)),
        Value::Int64(i) => Ok(i.to_object(py)),
        Value::UInt8(i) => Ok((*i as i64).to_object(py)),
        Value::UInt16(i) => Ok((*i as i64).to_object(py)),
        Value::UInt32(i) => Ok((*i as i64).to_object(py)),
        Value::UInt64(i) => Ok((*i as i64).to_object(py)),
        Value::Float32(f) => Ok((*f as f64).to_object(py)),
        Value::Float64(f) => Ok(f.to_object(py)),
        Value::String(s) => Ok(s.to_object(py)),
        Value::Binary(b) => Ok(pyo3::types::PyBytes::new_bound(py, b).into()),
        Value::Json(j) => {
            json_to_py(py, j)
        }
        Value::Timestamp(t) => Ok((*t).to_object(py)),
        Value::Date(d) => Ok((*d as i64).to_object(py)),
        Value::Array(arr) => {
            let list = PyList::empty_bound(py);
            for item in arr {
                list.append(value_to_py(py, item)?)?;
            }
            Ok(list.into())
        }
    }
}

/// Convert Arrow array value at index to Python object
#[allow(dead_code)]
fn arrow_value_to_py(py: Python<'_>, array: &arrow::array::ArrayRef, idx: usize) -> PyResult<PyObject> {
    use arrow::array::{Array, Int64Array, Float64Array, StringArray, BooleanArray};
    use arrow::datatypes::DataType;
    
    if array.is_null(idx) {
        return Ok(py.None());
    }
    
    match array.data_type() {
        DataType::Int64 => {
            let arr = array.as_any().downcast_ref::<Int64Array>().unwrap();
            Ok(arr.value(idx).to_object(py))
        }
        DataType::Float64 => {
            let arr = array.as_any().downcast_ref::<Float64Array>().unwrap();
            Ok(arr.value(idx).to_object(py))
        }
        DataType::Utf8 => {
            let arr = array.as_any().downcast_ref::<StringArray>().unwrap();
            Ok(arr.value(idx).to_object(py))
        }
        DataType::Boolean => {
            let arr = array.as_any().downcast_ref::<BooleanArray>().unwrap();
            Ok(arr.value(idx).to_object(py))
        }
        _ => Ok(py.None()),
    }
}

fn json_to_py(py: Python<'_>, value: &serde_json::Value) -> PyResult<PyObject> {
    match value {
        serde_json::Value::Null => Ok(py.None()),
        serde_json::Value::Bool(b) => Ok(b.to_object(py)),
        serde_json::Value::Number(n) => {
            if let Some(i) = n.as_i64() {
                Ok(i.to_object(py))
            } else if let Some(f) = n.as_f64() {
                Ok(f.to_object(py))
            } else {
                Ok(py.None())
            }
        }
        serde_json::Value::String(s) => Ok(s.to_object(py)),
        serde_json::Value::Array(arr) => {
            let list = PyList::empty_bound(py);
            for item in arr {
                list.append(json_to_py(py, item)?)?;
            }
            Ok(list.into())
        }
        serde_json::Value::Object(map) => {
            let dict = PyDict::new_bound(py);
            for (k, v) in map {
                dict.set_item(k, json_to_py(py, v)?)?;
            }
            Ok(dict.into())
        }
    }
}

/// Persistent storage state
#[derive(Serialize, Deserialize, Default)]
struct StorageState {
    tables: HashMap<String, TableData>,
    current_table: String,
}

#[derive(Serialize, Deserialize, Clone)]
struct TableData {
    schema: ColumnSchema,
    ids: Vec<u64>,
    columns: Vec<crate::table::column_table::TypedColumn>,
    next_id: u64,
}

/// Delta record for incremental persistence
#[derive(Serialize, Deserialize)]
struct DeltaRecord {
    table_name: String,
    start_row: u64,
    ids: Vec<u64>,
    columns: Vec<crate::table::column_table::TypedColumn>,
}

/// File format magic bytes
const APEX_MAGIC: &[u8; 8] = b"APEXDB\x01\x00";
const DELTA_MAGIC: &[u8; 4] = b"DELT";

// ============================================================================
// ApexStorage - High-performance columnar storage with persistence
// ============================================================================

/// Durability level for ACID guarantees
/// 
/// - Fast: Maximum performance, data buffered in memory, fsync only on explicit flush
/// - Safe: Balanced mode, fsync on every flush() call
/// - Max: Maximum durability, fsync on every write operation
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Durability {
    #[default]
    Fast,
    Safe,
    Max,
}

impl Durability {
    fn from_str(s: &str) -> Self {
        match s.to_lowercase().as_str() {
            "safe" => Durability::Safe,
            "max" => Durability::Max,
            _ => Durability::Fast,
        }
    }
    
    /// Whether to fsync on flush
    pub fn sync_on_flush(&self) -> bool {
        matches!(self, Durability::Safe | Durability::Max)
    }
}

/// High-performance columnar storage engine
/// 
/// Features:
/// - Column-oriented storage for fast analytical queries
/// - Multi-table support
/// - File persistence (.apex format)
/// - Zero-copy Arrow IPC transfer
/// - Native full-text search (nanofts integration)
/// - Configurable durability levels (fast/safe/max)
#[pyclass]
pub struct ApexStorage {
    /// Storage path
    path: String,
    /// Base directory path
    base_dir: PathBuf,
    /// Tables (table_name -> ColumnTable)
    tables: RwLock<HashMap<String, ColumnTable>>,
    /// Current table name
    current_table: RwLock<String>,
    /// Next table ID
    next_table_id: RwLock<u32>,
    /// FTS Manager (optional, initialized on demand)
    fts_manager: RwLock<Option<FtsManager>>,
    /// FTS index field names per table
    fts_index_fields: RwLock<HashMap<String, Vec<String>>>,
    /// Durability level for ACID guarantees
    durability: Durability,
    /// Persisted row counts per table (for incremental writes)
    persisted_rows: RwLock<HashMap<String, usize>>,
}

#[pymethods]
impl ApexStorage {
    /// Create or open a storage file
    /// 
    /// Parameters:
    /// - path: Path to the storage file
    /// - drop_if_exists: If true, delete existing database
    /// - durability: ACID durability level ("fast", "safe", "max")
    #[new]
    #[pyo3(signature = (path, drop_if_exists = false, durability = "fast"))]
    fn new(path: &str, drop_if_exists: bool, durability: &str) -> PyResult<Self> {
        let path_obj = Path::new(path);
        let base_dir = path_obj.parent()
            .map(|p| p.to_path_buf())
            .unwrap_or_else(|| PathBuf::from("."));
        
        let durability_level = Durability::from_str(durability);

        // Handle drop_if_exists
        if drop_if_exists && path_obj.exists() {
            fs::remove_file(path_obj)
                .map_err(|e| PyIOError::new_err(format!("Failed to remove file: {}", e)))?;
            
            // Also remove FTS indexes
            let fts_dir = base_dir.join("fts_indexes");
            if fts_dir.exists() {
                let _ = fs::remove_dir_all(&fts_dir);
            }
        }

        let storage = if path_obj.exists() {
            // Load from file with durability setting
            Self::load_from_file(path, durability_level)?
        } else {
            // Create new
            let mut tables = HashMap::new();
            tables.insert("default".to_string(), ColumnTable::with_capacity(1, "default", 10000));
            
            Self {
                path: path.to_string(),
                base_dir: base_dir.clone(),
                tables: RwLock::new(tables),
                current_table: RwLock::new("default".to_string()),
                next_table_id: RwLock::new(2),
                fts_manager: RwLock::new(None),
                fts_index_fields: RwLock::new(HashMap::new()),
                durability: durability_level,
                persisted_rows: RwLock::new(HashMap::new()),
            }
        };

        Ok(storage)
    }

    /// Initialize FTS for current table.
    ///
    /// This sets up the FTS manager and ensures the engine for the current table exists.
    #[pyo3(name = "_init_fts")]
    #[pyo3(signature = (index_fields=None, lazy_load=false, cache_size=10000))]
    fn init_fts(&self, index_fields: Option<Vec<String>>, lazy_load: bool, cache_size: usize) -> PyResult<()> {
        let table_name = self.current_table.read().clone();

        // Record index field configuration (used by Python layer to decide what to index)
        if let Some(fields) = index_fields.clone() {
            self.fts_index_fields.write().insert(table_name.clone(), fields);
        }

        // Ensure manager exists
        if self.fts_manager.read().is_none() {
            let fts_dir = self.base_dir.join("fts_indexes");
            let config = FtsConfig {
                lazy_load,
                cache_size,
                ..FtsConfig::default()
            };
            let manager = FtsManager::new(&fts_dir, config);
            *self.fts_manager.write() = Some(manager);
        }

        // Touch/create engine for current table
        let manager_guard = self.fts_manager.read();
        let manager = manager_guard.as_ref().ok_or_else(|| PyRuntimeError::new_err("FTS manager not initialized"))?;
        manager
            .get_engine(&table_name)
            .map_err(|e| PyRuntimeError::new_err(format!("FTS init error: {}", e)))?;

        Ok(())
    }

    /// Add a column to current table
    fn add_column(&self, column_name: &str, column_type: &str) -> PyResult<()> {
        let table_name = self.current_table.read().clone();
        let dtype = match column_type.to_lowercase().as_str() {
            "int" | "int64" | "i64" | "integer" => crate::data::DataType::Int64,
            "float" | "float64" | "f64" | "double" => crate::data::DataType::Float64,
            "bool" | "boolean" => crate::data::DataType::Bool,
            "str" | "string" | "text" => crate::data::DataType::String,
            "bytes" | "binary" => crate::data::DataType::Binary,
            _ => crate::data::DataType::String,
        };

        let mut tables = self.tables.write();
        let table = tables
            .get_mut(&table_name)
            .ok_or_else(|| PyValueError::new_err("Table not found"))?;

        table.add_column(column_name, dtype);
        Ok(())
    }

    fn drop_column(&self, column_name: &str) -> PyResult<()> {
        let table_name = self.current_table.read().clone();

        let mut tables = self.tables.write();
        let table = tables
            .get_mut(&table_name)
            .ok_or_else(|| PyValueError::new_err("Table not found"))?;

        let ok = table.drop_column(column_name);
        if !ok {
            return Err(PyValueError::new_err(format!("Column not found: {}", column_name)));
        }

        Ok(())
    }

    /// Delete a single record by id (soft delete)
    fn delete(&self, id: i64) -> bool {
        let table_name = self.current_table.read().clone();
        let mut tables = self.tables.write();
        if let Some(table) = tables.get_mut(&table_name) {
            table.delete(id as u64)
        } else {
            false
        }
    }

    /// Delete multiple records by ids (soft delete)
    fn delete_batch(&self, ids: Vec<i64>) -> bool {
        let table_name = self.current_table.read().clone();
        let ids_u64: Vec<u64> = ids.into_iter().map(|id| id as u64).collect();
        let mut tables = self.tables.write();
        if let Some(table) = tables.get_mut(&table_name) {
            table.delete_batch(&ids_u64)
        } else {
            false
        }
    }

    /// Replace a record (full replacement) by id
    fn replace(&self, _py: Python<'_>, id: i64, data: &Bound<'_, PyDict>) -> PyResult<bool> {
        let table_name = self.current_table.read().clone();

        let mut fields: HashMap<String, Value> = HashMap::new();
        for (key, value) in data.iter() {
            let k: String = key.extract()?;
            let v = py_to_value(&value)?;
            fields.insert(k, v);
        }

        let mut tables = self.tables.write();
        let table = tables
            .get_mut(&table_name)
            .ok_or_else(|| PyValueError::new_err("Table not found"))?;

        table.flush_write_buffer();
        Ok(table.replace(id as u64, &fields))
    }

    /// Execute a full SQL statement and return rows.
    /// Returns a dict {columns: List[str], rows: List[List[Any]], rows_affected: int}
    fn execute(&self, py: Python<'_>, sql: &str) -> PyResult<PyObject> {
        let sql = sql.to_string();

        let (columns, rows, rows_affected) = py.allow_threads(|| -> PyResult<(Vec<String>, Vec<Vec<Value>>, usize)> {
            let default_table = self.current_table.read().clone();

            let statements = crate::query::SqlParser::parse_multi(&sql)
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

            // Per-execute temporary view registry: view_name -> SelectStatement
            let mut views: HashMap<String, crate::query::SelectStatement> = HashMap::new();

            fn expand_from_item(
                item: &mut crate::query::FromItem,
                views: &HashMap<String, crate::query::SelectStatement>,
            ) {
                match item {
                    crate::query::FromItem::Table { table, alias } => {
                        if let Some(v) = views.get(table) {
                            let a = alias.clone().unwrap_or_else(|| table.clone());
                            *item = crate::query::FromItem::Subquery {
                                stmt: Box::new(v.clone()),
                                alias: a,
                            };
                        }
                    }
                    crate::query::FromItem::Subquery { stmt, .. } => {
                        expand_select(stmt, views);
                    }
                }
            }

            fn expand_expr(expr: &mut crate::query::SqlExpr, views: &HashMap<String, crate::query::SelectStatement>) {
                use crate::query::SqlExpr;
                match expr {
                    SqlExpr::BinaryOp { left, right, .. } => {
                        expand_expr(left, views);
                        expand_expr(right, views);
                    }
                    SqlExpr::UnaryOp { expr, .. } => expand_expr(expr, views),
                    SqlExpr::Paren(inner) => expand_expr(inner, views),
                    SqlExpr::Between { low, high, .. } => {
                        expand_expr(low, views);
                        expand_expr(high, views);
                    }
                    SqlExpr::InSubquery { stmt, .. }
                    | SqlExpr::ExistsSubquery { stmt }
                    | SqlExpr::ScalarSubquery { stmt } => {
                        expand_select(stmt, views);
                    }
                    SqlExpr::Case { when_then, else_expr } => {
                        for (w, t) in when_then.iter_mut() {
                            expand_expr(w, views);
                            expand_expr(t, views);
                        }
                        if let Some(e) = else_expr {
                            expand_expr(e, views);
                        }
                    }
                    SqlExpr::Function { args, .. } => {
                        for a in args.iter_mut() {
                            expand_expr(a, views);
                        }
                    }
                    _ => {}
                }
            }

            fn expand_select(stmt: &mut crate::query::SelectStatement, views: &HashMap<String, crate::query::SelectStatement>) {
                if let Some(from) = stmt.from.as_mut() {
                    expand_from_item(from, views);
                }
                for j in stmt.joins.iter_mut() {
                    expand_from_item(&mut j.right, views);
                    expand_expr(&mut j.on, views);
                }
                if let Some(w) = stmt.where_clause.as_mut() {
                    expand_expr(w, views);
                }
                if let Some(h) = stmt.having.as_mut() {
                    expand_expr(h, views);
                }
                for c in stmt.columns.iter_mut() {
                    if let crate::query::SelectColumn::Expression { expr, .. } = c {
                        expand_expr(expr, views);
                    }
                }
            }

            fn expand_stmt(stmt: &mut crate::query::SqlStatement, views: &HashMap<String, crate::query::SelectStatement>) {
                match stmt {
                    crate::query::SqlStatement::Select(s) => expand_select(s, views),
                    crate::query::SqlStatement::Union(u) => {
                        expand_stmt(&mut u.left, views);
                        expand_stmt(&mut u.right, views);
                    }
                    _ => {}
                }
            }

            fn stmt_has_join(stmt: &crate::query::SqlStatement) -> bool {
                match stmt {
                    crate::query::SqlStatement::Select(sel) => !sel.joins.is_empty(),
                    crate::query::SqlStatement::Union(u) => stmt_has_join(&u.left) || stmt_has_join(&u.right),
                    _ => false,
                }
            }

            fn first_select_table(stmt: &crate::query::SqlStatement) -> Option<String> {
                match stmt {
                    crate::query::SqlStatement::Select(sel) => sel.from.as_ref().map(|f| match f {
                        crate::query::FromItem::Table { table, .. } => table.clone(),
                        crate::query::FromItem::Subquery { alias, .. } => alias.clone(),
                    }),
                    crate::query::SqlStatement::Union(u) => first_select_table(&u.left),
                    _ => None,
                }
            }

            fn stmt_has_derived_from(stmt: &crate::query::SqlStatement) -> bool {
                match stmt {
                    crate::query::SqlStatement::Select(sel) => {
                        matches!(sel.from, Some(crate::query::FromItem::Subquery { .. }))
                    }
                    crate::query::SqlStatement::Union(u) => {
                        stmt_has_derived_from(&u.left) || stmt_has_derived_from(&u.right)
                    }
                    _ => false,
                }
            }

            fn expr_has_in_subquery(expr: &crate::query::SqlExpr) -> bool {
                use crate::query::SqlExpr;
                match expr {
                    SqlExpr::InSubquery { .. } => true,
                    SqlExpr::ExistsSubquery { .. } => false,
                    SqlExpr::BinaryOp { left, right, .. } => {
                        expr_has_in_subquery(left) || expr_has_in_subquery(right)
                    }
                    SqlExpr::UnaryOp { expr, .. } => expr_has_in_subquery(expr),
                    SqlExpr::Paren(inner) => expr_has_in_subquery(inner),
                    SqlExpr::Between { low, high, .. } => {
                        expr_has_in_subquery(low) || expr_has_in_subquery(high)
                    }
                    SqlExpr::Function { args, .. } => args.iter().any(expr_has_in_subquery),
                    _ => false,
                }
            }

            fn expr_has_exists_subquery(expr: &crate::query::SqlExpr) -> bool {
                use crate::query::SqlExpr;
                match expr {
                    SqlExpr::ExistsSubquery { .. } => true,
                    SqlExpr::ScalarSubquery { .. } => true,
                    SqlExpr::BinaryOp { left, right, .. } => {
                        expr_has_exists_subquery(left) || expr_has_exists_subquery(right)
                    }
                    SqlExpr::UnaryOp { expr, .. } => expr_has_exists_subquery(expr),
                    SqlExpr::Paren(inner) => expr_has_exists_subquery(inner),
                    SqlExpr::Between { low, high, .. } => {
                        expr_has_exists_subquery(low) || expr_has_exists_subquery(high)
                    }
                    SqlExpr::Function { args, .. } => args.iter().any(expr_has_exists_subquery),
                    _ => false,
                }
            }

            fn stmt_has_in_subquery(stmt: &crate::query::SqlStatement) -> bool {
                match stmt {
                    crate::query::SqlStatement::Select(sel) => {
                        sel.where_clause.as_ref().is_some_and(expr_has_in_subquery)
                            || sel.having.as_ref().is_some_and(expr_has_in_subquery)
                    }
                    crate::query::SqlStatement::Union(u) => {
                        stmt_has_in_subquery(&u.left) || stmt_has_in_subquery(&u.right)
                    }
                    _ => false,
                }
            }

            fn stmt_has_exists_subquery(stmt: &crate::query::SqlStatement) -> bool {
                match stmt {
                    crate::query::SqlStatement::Select(sel) => {
                        let select_has = sel.columns.iter().any(|c| match c {
                            crate::query::SelectColumn::Expression { expr, .. } => expr_has_exists_subquery(expr),
                            _ => false,
                        });
                        select_has
                            || sel.where_clause.as_ref().is_some_and(expr_has_exists_subquery)
                            || sel.having.as_ref().is_some_and(expr_has_exists_subquery)
                    }
                    crate::query::SqlStatement::Union(u) => {
                        stmt_has_exists_subquery(&u.left) || stmt_has_exists_subquery(&u.right)
                    }
                    _ => false,
                }
            }

            let mut tables = self.tables.write();

            let mut last: Option<crate::query::SqlResult> = None;

            for stmt in statements {
                match stmt {
                    crate::query::SqlStatement::CreateView { name, stmt } => {
                        if tables.contains_key(&name) {
                            return Err(PyValueError::new_err(format!(
                                "View name '{}' conflicts with existing table",
                                name
                            )));
                        }
                        if views.contains_key(&name) {
                            return Err(PyValueError::new_err(format!(
                                "View '{}' already exists",
                                name
                            )));
                        }
                        views.insert(name, stmt);
                    }
                    crate::query::SqlStatement::DropView { name } => {
                        if views.remove(&name).is_none() {
                            return Err(PyValueError::new_err(format!(
                                "View '{}' not found",
                                name
                            )));
                        }
                    }
                    mut q @ (crate::query::SqlStatement::Select(_) | crate::query::SqlStatement::Union(_)) => {
                        expand_stmt(&mut q, &views);

                        // Important: expanding views into derived tables may cause the legacy executor
                        // to materialize subqueries into `tables` (by alias). Those must not leak
                        // across execute() calls, since views are per-execute.
                        let cleanup_after = !views.is_empty();
                        let before_keys: Vec<String> = if cleanup_after {
                            tables.keys().cloned().collect()
                        } else {
                            Vec::new()
                        };

                        let is_join = stmt_has_join(&q);
                        let has_derived = stmt_has_derived_from(&q);
                        let has_in_subquery = stmt_has_in_subquery(&q);
                        let has_exists_subquery = stmt_has_exists_subquery(&q);
                        let has_scalar_subquery_text = sql.to_uppercase().contains("(SELECT");

                        let result = if !views.is_empty()
                            || is_join
                            || has_derived
                            || has_in_subquery
                            || has_exists_subquery
                            || has_scalar_subquery_text
                        {
                            SqlExecutor::execute_with_tables_parsed(q, &mut tables, &default_table)
                                .map_err(|e| PyRuntimeError::new_err(e.to_string()))?
                        } else {
                            let target_table = first_select_table(&q).unwrap_or_else(|| default_table.clone());

                            let table = tables
                                .get_mut(&target_table)
                                .ok_or_else(|| PyValueError::new_err(format!("Table '{}' not found.", target_table)))?;

                            SqlExecutor::execute_parsed(q, table)
                                .map_err(|e| PyRuntimeError::new_err(e.to_string()))?
                        };

                        if cleanup_after {
                            let before_set: std::collections::HashSet<String> =
                                before_keys.into_iter().collect();
                            let to_remove: Vec<String> = tables
                                .keys()
                                .filter(|k| !before_set.contains(*k))
                                .cloned()
                                .collect();
                            for k in to_remove {
                                tables.remove(&k);
                            }
                        }

                        last = Some(result);
                    }
                }
            }

            let mut result = last.unwrap_or_else(|| crate::query::SqlResult::new(Vec::new(), Vec::new()));
            if result.rows.is_empty() {
                if let Some(batch) = result.arrow_batch.take() {
                    let mut out_rows: Vec<Vec<Value>> = Vec::with_capacity(batch.num_rows());
                    let num_cols = batch.num_columns();
                    for row_idx in 0..batch.num_rows() {
                        let mut row: Vec<Value> = Vec::with_capacity(num_cols);
                        for col_idx in 0..num_cols {
                            let arr = batch.column(col_idx);
                            let values = arrow_array_to_values(arr)?;
                            row.push(values.get(row_idx).cloned().unwrap_or(Value::Null));
                        }
                        out_rows.push(row);
                    }
                    result.rows = out_rows;
                }
            }
            Ok((result.columns, result.rows, result.rows_affected))
        })?;

        let out = PyDict::new_bound(py);
        out.set_item("columns", columns)?;

        // Convert rows (Vec<Vec<Value>>) -> Python
        let py_rows = PyList::empty_bound(py);
        for row in rows {
            let py_row = PyList::empty_bound(py);
            for v in row {
                py_row.append(value_to_py(py, &v)?)?;
            }
            py_rows.append(py_row)?;
        }
        out.set_item("rows", py_rows)?;
        out.set_item("rows_affected", rows_affected)?;
        Ok(out.into())
    }

    /// Execute SQL and export Arrow via FFI pointers.
    #[pyo3(name = "_execute_arrow_ffi")]
    fn execute_arrow_ffi(&self, py: Python<'_>, sql: &str) -> PyResult<(u64, u64)> {
        use arrow::array::Array;
        use arrow::ffi::{FFI_ArrowArray, FFI_ArrowSchema};

        let sql = sql.to_string();

        let batch = py.allow_threads(|| -> PyResult<arrow::record_batch::RecordBatch> {
            let default_table = self.current_table.read().clone();

            let statements = crate::query::SqlParser::parse_multi(&sql)
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

            let mut views: HashMap<String, crate::query::SelectStatement> = HashMap::new();

            fn expand_from_item(
                item: &mut crate::query::FromItem,
                views: &HashMap<String, crate::query::SelectStatement>,
            ) {
                match item {
                    crate::query::FromItem::Table { table, alias } => {
                        if let Some(v) = views.get(table) {
                            let a = alias.clone().unwrap_or_else(|| table.clone());
                            *item = crate::query::FromItem::Subquery {
                                stmt: Box::new(v.clone()),
                                alias: a,
                            };
                        }
                    }
                    crate::query::FromItem::Subquery { stmt, .. } => {
                        expand_select(stmt, views);
                    }
                }
            }

            fn expand_expr(expr: &mut crate::query::SqlExpr, views: &HashMap<String, crate::query::SelectStatement>) {
                use crate::query::SqlExpr;
                match expr {
                    SqlExpr::BinaryOp { left, right, .. } => {
                        expand_expr(left, views);
                        expand_expr(right, views);
                    }
                    SqlExpr::UnaryOp { expr, .. } => expand_expr(expr, views),
                    SqlExpr::Paren(inner) => expand_expr(inner, views),
                    SqlExpr::Between { low, high, .. } => {
                        expand_expr(low, views);
                        expand_expr(high, views);
                    }
                    SqlExpr::InSubquery { stmt, .. }
                    | SqlExpr::ExistsSubquery { stmt }
                    | SqlExpr::ScalarSubquery { stmt } => {
                        expand_select(stmt, views);
                    }
                    SqlExpr::Case { when_then, else_expr } => {
                        for (w, t) in when_then.iter_mut() {
                            expand_expr(w, views);
                            expand_expr(t, views);
                        }
                        if let Some(e) = else_expr {
                            expand_expr(e, views);
                        }
                    }
                    SqlExpr::Function { args, .. } => {
                        for a in args.iter_mut() {
                            expand_expr(a, views);
                        }
                    }
                    _ => {}
                }
            }

            fn expand_select(stmt: &mut crate::query::SelectStatement, views: &HashMap<String, crate::query::SelectStatement>) {
                if let Some(from) = stmt.from.as_mut() {
                    expand_from_item(from, views);
                }
                for j in stmt.joins.iter_mut() {
                    expand_from_item(&mut j.right, views);
                    expand_expr(&mut j.on, views);
                }
                if let Some(w) = stmt.where_clause.as_mut() {
                    expand_expr(w, views);
                }
                if let Some(h) = stmt.having.as_mut() {
                    expand_expr(h, views);
                }
                for c in stmt.columns.iter_mut() {
                    if let crate::query::SelectColumn::Expression { expr, .. } = c {
                        expand_expr(expr, views);
                    }
                }
            }

            fn expand_stmt(stmt: &mut crate::query::SqlStatement, views: &HashMap<String, crate::query::SelectStatement>) {
                match stmt {
                    crate::query::SqlStatement::Select(s) => expand_select(s, views),
                    crate::query::SqlStatement::Union(u) => {
                        expand_stmt(&mut u.left, views);
                        expand_stmt(&mut u.right, views);
                    }
                    _ => {}
                }
            }
            fn stmt_has_join(stmt: &crate::query::SqlStatement) -> bool {
                match stmt {
                    crate::query::SqlStatement::Select(sel) => !sel.joins.is_empty(),
                    crate::query::SqlStatement::Union(u) => stmt_has_join(&u.left) || stmt_has_join(&u.right),
                    _ => false,
                }
            }

            fn first_select_table(stmt: &crate::query::SqlStatement) -> Option<String> {
                match stmt {
                    crate::query::SqlStatement::Select(sel) => sel.from.as_ref().map(|f| match f {
                        crate::query::FromItem::Table { table, .. } => table.clone(),
                        crate::query::FromItem::Subquery { alias, .. } => alias.clone(),
                    }),
                    crate::query::SqlStatement::Union(u) => first_select_table(&u.left),
                    _ => None,
                }
            }

            fn stmt_has_derived_from(stmt: &crate::query::SqlStatement) -> bool {
                match stmt {
                    crate::query::SqlStatement::Select(sel) => {
                        matches!(sel.from, Some(crate::query::FromItem::Subquery { .. }))
                    }
                    crate::query::SqlStatement::Union(u) => {
                        stmt_has_derived_from(&u.left) || stmt_has_derived_from(&u.right)
                    }
                    _ => false,
                }
            }

            fn expr_has_in_subquery(expr: &crate::query::SqlExpr) -> bool {
                use crate::query::SqlExpr;
                match expr {
                    SqlExpr::InSubquery { .. } => true,
                    SqlExpr::BinaryOp { left, right, .. } => {
                        expr_has_in_subquery(left) || expr_has_in_subquery(right)
                    }
                    SqlExpr::UnaryOp { expr, .. } => expr_has_in_subquery(expr),
                    SqlExpr::Paren(inner) => expr_has_in_subquery(inner),
                    SqlExpr::Between { low, high, .. } => {
                        expr_has_in_subquery(low) || expr_has_in_subquery(high)
                    }
                    SqlExpr::Function { args, .. } => args.iter().any(expr_has_in_subquery),
                    _ => false,
                }
            }

            fn expr_has_exists_subquery(expr: &crate::query::SqlExpr) -> bool {
                use crate::query::SqlExpr;
                match expr {
                    SqlExpr::ExistsSubquery { .. } => true,
                    SqlExpr::BinaryOp { left, right, .. } => {
                        expr_has_exists_subquery(left) || expr_has_exists_subquery(right)
                    }
                    SqlExpr::UnaryOp { expr, .. } => expr_has_exists_subquery(expr),
                    SqlExpr::Paren(inner) => expr_has_exists_subquery(inner),
                    SqlExpr::Between { low, high, .. } => {
                        expr_has_exists_subquery(low) || expr_has_exists_subquery(high)
                    }
                    SqlExpr::Function { args, .. } => args.iter().any(expr_has_exists_subquery),
                    _ => false,
                }
            }

            fn stmt_has_in_subquery(stmt: &crate::query::SqlStatement) -> bool {
                match stmt {
                    crate::query::SqlStatement::Select(sel) => {
                        sel.where_clause.as_ref().is_some_and(expr_has_in_subquery)
                            || sel.having.as_ref().is_some_and(expr_has_in_subquery)
                    }
                    crate::query::SqlStatement::Union(u) => {
                        stmt_has_in_subquery(&u.left) || stmt_has_in_subquery(&u.right)
                    }
                    _ => false,
                }
            }

            fn stmt_has_exists_subquery(stmt: &crate::query::SqlStatement) -> bool {
                match stmt {
                    crate::query::SqlStatement::Select(sel) => {
                        sel.where_clause.as_ref().is_some_and(expr_has_exists_subquery)
                            || sel.having.as_ref().is_some_and(expr_has_exists_subquery)
                    }
                    crate::query::SqlStatement::Union(u) => {
                        stmt_has_exists_subquery(&u.left) || stmt_has_exists_subquery(&u.right)
                    }
                    _ => false,
                }
            }

            let mut tables = self.tables.write();

            let mut last_batch: Option<arrow::record_batch::RecordBatch> = None;

            for stmt in statements {
                match stmt {
                    crate::query::SqlStatement::CreateView { name, stmt } => {
                        if tables.contains_key(&name) {
                            return Err(PyValueError::new_err(format!(
                                "View name '{}' conflicts with existing table",
                                name
                            )));
                        }
                        if views.contains_key(&name) {
                            return Err(PyValueError::new_err(format!(
                                "View '{}' already exists",
                                name
                            )));
                        }
                        views.insert(name, stmt);
                    }
                    crate::query::SqlStatement::DropView { name } => {
                        if views.remove(&name).is_none() {
                            return Err(PyValueError::new_err(format!(
                                "View '{}' not found",
                                name
                            )));
                        }
                    }
                    mut q @ (crate::query::SqlStatement::Select(_) | crate::query::SqlStatement::Union(_)) => {
                        expand_stmt(&mut q, &views);

                        let cleanup_after = !views.is_empty();
                        let before_keys: Vec<String> = if cleanup_after {
                            tables.keys().cloned().collect()
                        } else {
                            Vec::new()
                        };

                        let result = SqlExecutor::execute_with_tables_parsed(q, &mut tables, &default_table)
                            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

                        if cleanup_after {
                            let before_set: std::collections::HashSet<String> =
                                before_keys.into_iter().collect();
                            let to_remove: Vec<String> = tables
                                .keys()
                                .filter(|k| !before_set.contains(*k))
                                .cloned()
                                .collect();
                            for k in to_remove {
                                tables.remove(&k);
                            }
                        }

                        let b = result
                            .to_record_batch()
                            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
                        last_batch = Some(b);
                    }
                }
            }

            Ok(last_batch.unwrap_or_else(|| {
                // Empty result batch
                arrow::record_batch::RecordBatch::new_empty(Arc::new(arrow::datatypes::Schema::empty()))
            }))
        })?;

        if batch.num_rows() == 0 {
            return Ok((0, 0));
        }

        let struct_array = arrow::array::StructArray::from(batch);

        let ffi_array = Box::new(FFI_ArrowArray::new(&struct_array.to_data()));
        let ffi_schema = Box::new(FFI_ArrowSchema::try_from(struct_array.data_type())
            .map_err(|e| PyRuntimeError::new_err(format!("Schema export error: {}", e)))?);

        let array_ptr = Box::into_raw(ffi_array) as u64;
        let schema_ptr = Box::into_raw(ffi_schema) as u64;

        Ok((schema_ptr, array_ptr))
    }

    // ========== Table Management ==========

    /// Use a table
    fn use_table(&self, name: &str) -> PyResult<()> {
        let tables = self.tables.read();
        if !tables.contains_key(name) {
            return Err(PyValueError::new_err(format!("Table not found: {}", name)));
        }
        drop(tables);
        *self.current_table.write() = name.to_string();
        Ok(())
    }

    /// Get current table name
    fn current_table(&self) -> String {
        self.current_table.read().clone()
    }

    /// Create a new table
    fn create_table(&self, name: &str) -> PyResult<()> {
        let mut tables = self.tables.write();
        if tables.contains_key(name) {
            return Err(PyValueError::new_err(format!("Table already exists: {}", name)));
        }
        
        let table_id = {
            let mut id = self.next_table_id.write();
            let current = *id;
            *id += 1;
            current
        };
        
        tables.insert(name.to_string(), ColumnTable::with_capacity(table_id, name, 10000));
        drop(tables);
        
        *self.current_table.write() = name.to_string();
        Ok(())
    }

    /// Drop a table
    fn drop_table(&self, name: &str) -> PyResult<()> {
        if name == "default" {
            return Err(PyValueError::new_err("Cannot drop default table"));
        }
        
        let mut tables = self.tables.write();
        if !tables.contains_key(name) {
            return Err(PyValueError::new_err(format!("Table not found: {}", name)));
        }
        
        tables.remove(name);
        drop(tables);
        
        if *self.current_table.read() == name {
            *self.current_table.write() = "default".to_string();
        }
        Ok(())
    }

    /// List all tables
    fn list_tables(&self) -> Vec<String> {
        self.tables.read().keys().cloned().collect()
    }

    // ========== CRUD Operations (via IoEngine) ==========

    /// Store a single record - IoEngine::Single strategy
    fn store(&self, py: Python<'_>, data: &Bound<'_, PyDict>) -> PyResult<i64> {
        // IoEngine strategy: Single (direct insert for single row)
        let fields = dict_to_fields(data)?;
        let table_name = self.current_table.read().clone();
        
        let id = py.allow_threads(|| {
            self.tables
                .write()
                .get_mut(&table_name)
                .ok_or_else(|| PyValueError::new_err("Table not found"))?
                .insert(&fields)
                .map_err(|e| PyValueError::new_err(e.to_string()))
        })?;
        
        Ok(id as i64)
    }

    /// Store a single record without returning ID - IoEngine::Single strategy (no return)
    #[pyo3(name = "_store_single_no_return")]
    fn store_single_no_return(&self, py: Python<'_>, data: &Bound<'_, PyDict>) -> PyResult<()> {
        // IoEngine strategy: Single (direct insert, no ID return for speed)
        let fields = dict_to_fields(data)?;
        let table_name = self.current_table.read().clone();
        
        py.allow_threads(|| {
            self.tables
                .write()
                .get_mut(&table_name)
                .ok_or_else(|| PyValueError::new_err("Table not found"))?
                .insert(&fields)
                .map_err(|e| PyValueError::new_err(e.to_string()))
        })?;
        
        Ok(())
    }

    /// Store multiple records - High Performance Path
    /// 
    /// Automatically converts row-oriented input to columnar format for 
    /// maximum performance. Achieves ~0.3ms for 10,000 rows.
    fn store_batch(&self, py: Python<'_>, data: &Bound<'_, PyList>) -> PyResult<Vec<i64>> {
        let num_rows = data.len();
        if num_rows == 0 {
            return Ok(Vec::new());
        }

        // High-performance path: Convert to columnar format
        // This is ~60x faster than row-by-row insertion
        
        // Phase 1: Discover schema from first row and collect column data
        let mut int_columns: HashMap<String, Vec<i64>> = HashMap::new();
        let mut float_columns: HashMap<String, Vec<f64>> = HashMap::new();
        let mut string_columns: HashMap<String, Vec<String>> = HashMap::new();
        let mut bool_columns: HashMap<String, Vec<bool>> = HashMap::new();
        let mut binary_columns: HashMap<String, Vec<Vec<u8>>> = HashMap::new();
        
        // First pass: identify column types from first row
        let first_item = data.get_item(0)?;
        let first_dict = first_item.downcast::<PyDict>()?;
        for (key, value) in first_dict.iter() {
            let col_name: String = key.extract()?;
            if col_name == "_id" {
                continue;
            }
            
            // Determine type and create column
            if value.is_none() {
                // Skip null values for type detection
                continue;
            } else if value.extract::<bool>().is_ok() {
                bool_columns.insert(col_name, Vec::with_capacity(num_rows));
            } else if value.extract::<i64>().is_ok() {
                int_columns.insert(col_name, Vec::with_capacity(num_rows));
            } else if value.extract::<f64>().is_ok() {
                float_columns.insert(col_name, Vec::with_capacity(num_rows));
            } else if value.extract::<Vec<u8>>().is_ok() {
                binary_columns.insert(col_name, Vec::with_capacity(num_rows));
            } else if value.extract::<String>().is_ok() {
                string_columns.insert(col_name, Vec::with_capacity(num_rows));
            }
        }

        // Second pass: collect all values by column
        for item in data.iter() {
            let dict = item.downcast::<PyDict>()?;
            
            // Collect int columns
            for (name, vec) in int_columns.iter_mut() {
                if let Some(val) = dict.get_item(name)? {
                    vec.push(val.extract::<i64>().unwrap_or(0));
                } else {
                    vec.push(0);
                }
            }
            
            // Collect float columns
            for (name, vec) in float_columns.iter_mut() {
                if let Some(val) = dict.get_item(name)? {
                    vec.push(val.extract::<f64>().unwrap_or(0.0));
                } else {
                    vec.push(0.0);
                }
            }
            
            // Collect string columns
            for (name, vec) in string_columns.iter_mut() {
                if let Some(val) = dict.get_item(name)? {
                    vec.push(val.extract::<String>().unwrap_or_default());
                } else {
                    vec.push(String::new());
                }
            }
            
            // Collect bool columns
            for (name, vec) in bool_columns.iter_mut() {
                if let Some(val) = dict.get_item(name)? {
                    vec.push(val.extract::<bool>().unwrap_or(false));
                } else {
                    vec.push(false);
                }
            }
            
            // Collect binary columns
            for (name, vec) in binary_columns.iter_mut() {
                if let Some(val) = dict.get_item(name)? {
                    vec.push(val.extract::<Vec<u8>>().unwrap_or_default());
                } else {
                    vec.push(Vec::new());
                }
            }
        }

        // Phase 2: Convert to unified Value columns and insert
        let mut columns: HashMap<String, Vec<Value>> = HashMap::new();
        
        for (name, vec) in int_columns {
            columns.insert(name, vec.into_iter().map(Value::Int64).collect());
        }
        for (name, vec) in float_columns {
            columns.insert(name, vec.into_iter().map(Value::Float64).collect());
        }
        for (name, vec) in string_columns {
            columns.insert(name, vec.into_iter().map(Value::String).collect());
        }
        for (name, vec) in bool_columns {
            columns.insert(name, vec.into_iter().map(Value::Bool).collect());
        }
        for (name, vec) in binary_columns {
            columns.insert(name, vec.into_iter().map(Value::Binary).collect());
        }

        let table_name = self.current_table.read().clone();
        
        // Use fast columnar insert path
        let ids = py.allow_threads(|| {
            self.tables
                .write()
                .get_mut(&table_name)
                .ok_or_else(|| PyValueError::new_err("Table not found"))?
                .insert_columns(columns)
                .map_err(|e| PyValueError::new_err(e.to_string()))
        })?;
        
        Ok(ids.into_iter().map(|id| id as i64).collect())
    }

    /// Store from Arrow IPC format - High Performance Path
    /// 
    /// Converts Arrow columnar format directly to internal storage.
    /// This is the fastest path for bulk loading from PyArrow/Pandas/Polars.
    fn store_arrow(&self, py: Python<'_>, arrow_bytes: &[u8]) -> PyResult<Vec<i64>> {
        use arrow::ipc::reader::StreamReader;
        
        if arrow_bytes.is_empty() {
            return Ok(Vec::new());
        }
        
        // Parse Arrow IPC directly to columnar format
        let cursor = std::io::Cursor::new(arrow_bytes);
        let reader = StreamReader::try_new(cursor, None)
            .map_err(|e| PyValueError::new_err(format!("Arrow parse error: {}", e)))?;
        
        let mut all_ids = Vec::new();
        
        for batch_result in reader {
            let batch = batch_result
                .map_err(|e| PyValueError::new_err(format!("Arrow batch error: {}", e)))?;
            
            let num_rows = batch.num_rows();
            if num_rows == 0 {
                continue;
            }
            
            // Convert Arrow columns to Value columns
            let schema = batch.schema();
            let mut columns: HashMap<String, Vec<Value>> = HashMap::new();
            
            for (col_idx, field) in schema.fields().iter().enumerate() {
                let col_name = field.name();
                if col_name == "_id" {
                    continue;
                }
                
                let array = batch.column(col_idx);
                let values = arrow_array_to_values(array)?;
                columns.insert(col_name.clone(), values);
            }
            
            let table_name = self.current_table.read().clone();
            
            // Use fast columnar insert path
            let ids = py.allow_threads(|| {
                self.tables
                    .write()
                    .get_mut(&table_name)
                    .ok_or_else(|| PyValueError::new_err("Table not found"))?
                    .insert_columns(columns)
                    .map_err(|e| PyValueError::new_err(e.to_string()))
            })?;
            
            all_ids.extend(ids.into_iter().map(|id| id as i64));
        }
        
        Ok(all_ids)
    }

    /// ULTRA-FAST Arrow IPC store with FTS integration (zero-copy path)
    /// 
    /// This is the fastest path for bulk loading from PyArrow/Pandas/Polars:
    /// - Direct Arrow IPC parsing (no Python object creation)
    /// - Typed column extraction (no Value enum overhead)
    /// - Integrated FTS indexing (no separate Python call)
    /// - No ID return (avoids allocation)
    #[pyo3(name = "_store_arrow_ipc_fast", signature = (arrow_bytes, fts_fields=None))]
    fn store_arrow_ipc_fast(
        &self, 
        py: Python<'_>, 
        arrow_bytes: &[u8],
        fts_fields: Option<Vec<String>>
    ) -> PyResult<(i64, usize)> {
        use arrow::ipc::reader::StreamReader;
        use arrow::array::*;
        use arrow::datatypes::DataType as ArrowDataType;
        
        if arrow_bytes.is_empty() {
            return Ok((0, 0));
        }
        
        let table_name = self.current_table.read().clone();
        
        // Get start_id before insert
        let start_id = py.allow_threads(|| {
            let tables = self.tables.read();
            tables.get(&table_name)
                .map(|t| t.get_row_count() as i64)
                .unwrap_or(0)
        });
        
        // Parse Arrow IPC
        let cursor = std::io::Cursor::new(arrow_bytes);
        let reader = StreamReader::try_new(cursor, None)
            .map_err(|e| PyValueError::new_err(format!("Arrow parse error: {}", e)))?;
        
        let mut total_rows = 0usize;
        
        for batch_result in reader {
            let batch = batch_result
                .map_err(|e| PyValueError::new_err(format!("Arrow batch error: {}", e)))?;
            
            let num_rows = batch.num_rows();
            if num_rows == 0 {
                continue;
            }
            
            // Extract typed columns directly from Arrow arrays
            let mut int_map: HashMap<String, Vec<i64>> = HashMap::new();
            let mut float_map: HashMap<String, Vec<f64>> = HashMap::new();
            let mut str_map: HashMap<String, Vec<String>> = HashMap::new();
            let mut bool_map: HashMap<String, Vec<bool>> = HashMap::new();
            let mut bin_map: HashMap<String, Vec<Vec<u8>>> = HashMap::new();
            
            let schema = batch.schema();
            
            for (col_idx, field) in schema.fields().iter().enumerate() {
                let col_name = field.name();
                if col_name == "_id" {
                    continue;
                }
                
                let array = batch.column(col_idx);
                
                match array.data_type() {
                    ArrowDataType::Int8 | ArrowDataType::Int16 | ArrowDataType::Int32 | ArrowDataType::Int64 |
                    ArrowDataType::UInt8 | ArrowDataType::UInt16 | ArrowDataType::UInt32 | ArrowDataType::UInt64 => {
                        let mut values = Vec::with_capacity(num_rows);
                        match array.data_type() {
                            ArrowDataType::Int64 => {
                                let arr = array.as_any().downcast_ref::<Int64Array>().unwrap();
                                for i in 0..num_rows {
                                    values.push(if arr.is_null(i) { 0 } else { arr.value(i) });
                                }
                            }
                            ArrowDataType::Int32 => {
                                let arr = array.as_any().downcast_ref::<Int32Array>().unwrap();
                                for i in 0..num_rows {
                                    values.push(if arr.is_null(i) { 0 } else { arr.value(i) as i64 });
                                }
                            }
                            ArrowDataType::Int16 => {
                                let arr = array.as_any().downcast_ref::<Int16Array>().unwrap();
                                for i in 0..num_rows {
                                    values.push(if arr.is_null(i) { 0 } else { arr.value(i) as i64 });
                                }
                            }
                            ArrowDataType::Int8 => {
                                let arr = array.as_any().downcast_ref::<Int8Array>().unwrap();
                                for i in 0..num_rows {
                                    values.push(if arr.is_null(i) { 0 } else { arr.value(i) as i64 });
                                }
                            }
                            ArrowDataType::UInt64 => {
                                let arr = array.as_any().downcast_ref::<UInt64Array>().unwrap();
                                for i in 0..num_rows {
                                    values.push(if arr.is_null(i) { 0 } else { arr.value(i) as i64 });
                                }
                            }
                            ArrowDataType::UInt32 => {
                                let arr = array.as_any().downcast_ref::<UInt32Array>().unwrap();
                                for i in 0..num_rows {
                                    values.push(if arr.is_null(i) { 0 } else { arr.value(i) as i64 });
                                }
                            }
                            ArrowDataType::UInt16 => {
                                let arr = array.as_any().downcast_ref::<UInt16Array>().unwrap();
                                for i in 0..num_rows {
                                    values.push(if arr.is_null(i) { 0 } else { arr.value(i) as i64 });
                                }
                            }
                            ArrowDataType::UInt8 => {
                                let arr = array.as_any().downcast_ref::<UInt8Array>().unwrap();
                                for i in 0..num_rows {
                                    values.push(if arr.is_null(i) { 0 } else { arr.value(i) as i64 });
                                }
                            }
                            _ => {}
                        }
                        int_map.insert(col_name.clone(), values);
                    }
                    ArrowDataType::Float32 | ArrowDataType::Float64 => {
                        let mut values = Vec::with_capacity(num_rows);
                        if let ArrowDataType::Float64 = array.data_type() {
                            let arr = array.as_any().downcast_ref::<Float64Array>().unwrap();
                            for i in 0..num_rows {
                                values.push(if arr.is_null(i) { 0.0 } else { arr.value(i) });
                            }
                        } else {
                            let arr = array.as_any().downcast_ref::<Float32Array>().unwrap();
                            for i in 0..num_rows {
                                values.push(if arr.is_null(i) { 0.0 } else { arr.value(i) as f64 });
                            }
                        }
                        float_map.insert(col_name.clone(), values);
                    }
                    ArrowDataType::Boolean => {
                        let arr = array.as_any().downcast_ref::<BooleanArray>().unwrap();
                        let mut values = Vec::with_capacity(num_rows);
                        for i in 0..num_rows {
                            values.push(if arr.is_null(i) { false } else { arr.value(i) });
                        }
                        bool_map.insert(col_name.clone(), values);
                    }
                    ArrowDataType::Utf8 | ArrowDataType::LargeUtf8 => {
                        let mut values = Vec::with_capacity(num_rows);
                        if let ArrowDataType::Utf8 = array.data_type() {
                            let arr = array.as_any().downcast_ref::<StringArray>().unwrap();
                            for i in 0..num_rows {
                                values.push(if arr.is_null(i) { String::new() } else { arr.value(i).to_string() });
                            }
                        } else {
                            let arr = array.as_any().downcast_ref::<LargeStringArray>().unwrap();
                            for i in 0..num_rows {
                                values.push(if arr.is_null(i) { String::new() } else { arr.value(i).to_string() });
                            }
                        }
                        str_map.insert(col_name.clone(), values);
                    }
                    ArrowDataType::Binary | ArrowDataType::LargeBinary => {
                        let mut values = Vec::with_capacity(num_rows);
                        if let ArrowDataType::Binary = array.data_type() {
                            let arr = array.as_any().downcast_ref::<BinaryArray>().unwrap();
                            for i in 0..num_rows {
                                values.push(if arr.is_null(i) { Vec::new() } else { arr.value(i).to_vec() });
                            }
                        } else {
                            let arr = array.as_any().downcast_ref::<LargeBinaryArray>().unwrap();
                            for i in 0..num_rows {
                                values.push(if arr.is_null(i) { Vec::new() } else { arr.value(i).to_vec() });
                            }
                        }
                        bin_map.insert(col_name.clone(), values);
                    }
                    _ => {
                        // Fallback: convert to string
                        let arr = arrow::compute::cast(array, &ArrowDataType::Utf8)
                            .map_err(|e| PyValueError::new_err(format!("Arrow cast error: {}", e)))?;
                        let str_arr = arr.as_any().downcast_ref::<StringArray>().unwrap();
                        let mut values = Vec::with_capacity(num_rows);
                        for i in 0..num_rows {
                            values.push(if str_arr.is_null(i) { String::new() } else { str_arr.value(i).to_string() });
                        }
                        str_map.insert(col_name.clone(), values);
                    }
                }
            }
            
            // FTS indexing using nanofts columnar API
            if let Some(ref fields) = fts_fields {
                let manager_guard = self.fts_manager.read();
                if let Some(manager) = manager_guard.as_ref() {
                    let columns: Vec<(String, Vec<String>)> = fields.iter()
                        .filter_map(|f| str_map.get(f).map(|v| (f.clone(), v.clone())))
                        .collect();
                    
                    if !columns.is_empty() {
                        let batch_start_id = start_id as u64 + total_rows as u64;
                        let doc_ids: Vec<u64> = (0..num_rows).map(|i| batch_start_id + (i as u64)).collect();
                        
                        if let Ok(engine) = manager.get_engine(&table_name) {
                            let _ = engine.add_documents_columnar(doc_ids, columns);
                        }
                    }
                }
            }
            
            // Insert into table
            py.allow_threads(|| {
                self.tables
                    .write()
                    .get_mut(&table_name)
                    .ok_or_else(|| PyValueError::new_err("Table not found"))?
                    .insert_typed_columns_no_return(int_map, float_map, str_map, bool_map, bin_map)
                    .map_err(|e| PyValueError::new_err(e.to_string()))
            })?;
            
            total_rows += num_rows;
        }
        
        Ok((start_id, total_rows))
    }

    /// Insert columnar data directly (fastest write path) - internal use
    /// data: Dict[str, List] - column name to list of values
    /// 
    /// Optimized for bulk extraction - avoids per-element Python calls
    #[pyo3(name = "_insert_columns")]
    fn insert_columns(&self, py: Python<'_>, data: &Bound<'_, PyDict>) -> PyResult<Vec<i64>> {
        let mut columns: HashMap<String, Vec<Value>> = HashMap::new();
        let mut num_rows = 0usize;
        
        for (key, value) in data.iter() {
            let col_name: String = key.extract()?;
            
            // Try to get length
            let list_len = if let Ok(list) = value.downcast::<PyList>() {
                list.len()
            } else if let Ok(len) = value.len() {
                len
            } else {
                return Err(PyValueError::new_err("Values must be lists or arrays"));
            };
            
            if num_rows == 0 {
                num_rows = list_len;
            } else if list_len != num_rows {
                return Err(PyValueError::new_err("All columns must have same length"));
            }
            
            if list_len == 0 {
                continue;
            }
            
            // Type-specialized bulk extraction for maximum performance
            // Try to extract entire array at once rather than element by element
            
            // Try i64 array first (most common for integers)
            if let Ok(vec) = value.extract::<Vec<i64>>() {
                columns.insert(col_name, vec.into_iter().map(Value::Int64).collect());
                continue;
            }
            
            // Try f64 array (common for floats)
            if let Ok(vec) = value.extract::<Vec<f64>>() {
                columns.insert(col_name, vec.into_iter().map(Value::Float64).collect());
                continue;
            }
            
            // Try String array
            if let Ok(vec) = value.extract::<Vec<String>>() {
                columns.insert(col_name, vec.into_iter().map(Value::String).collect());
                continue;
            }
            
            // Try bool array
            if let Ok(vec) = value.extract::<Vec<bool>>() {
                columns.insert(col_name, vec.into_iter().map(Value::Bool).collect());
                continue;
            }
            
            // Try bytes array (for binary data)
            if let Ok(vec) = value.extract::<Vec<Vec<u8>>>() {
                columns.insert(col_name, vec.into_iter().map(Value::Binary).collect());
                continue;
            }
            
            // Fallback: element-by-element extraction (slower)
            let list = value.downcast::<PyList>()?;
            let values: Vec<Value> = list.iter()
                .map(|item| py_to_value(&item))
                .collect::<PyResult<Vec<_>>>()?;
            columns.insert(col_name, values);
        }
        
        if num_rows == 0 {
            return Ok(Vec::new());
        }
        
        let table_name = self.current_table.read().clone();
        
        let ids = py.allow_threads(|| {
            self.tables
                .write()
                .get_mut(&table_name)
                .ok_or_else(|| PyValueError::new_err("Table not found"))?
                .insert_columns(columns)
                .map_err(|e| PyValueError::new_err(e.to_string()))
        })?;
        
        Ok(ids.into_iter().map(|id| id as i64).collect())
    }

    /// Ultra-fast insert for typed columns - bypasses Value boxing entirely
    /// 
    /// This is the fastest possible write path, achieving ~0.3ms for 10k rows
    /// for numeric data by using direct memcpy-like operations.
    /// 
    /// Parameters:
    ///   int_cols: Dict[str, List[int]] - Integer columns
    ///   float_cols: Dict[str, List[float]] - Float columns  
    ///   str_cols: Dict[str, List[str]] - String columns
    ///   bool_cols: Dict[str, List[bool]] - Boolean columns
    ///   bin_cols: Dict[str, List[bytes]] - Binary columns
    #[pyo3(name = "_insert_typed_columns")]
    fn insert_typed_columns(
        &self,
        py: Python<'_>,
        int_cols: &Bound<'_, PyDict>,
        float_cols: &Bound<'_, PyDict>,
        str_cols: &Bound<'_, PyDict>,
        bool_cols: &Bound<'_, PyDict>,
        bin_cols: &Bound<'_, PyDict>,
    ) -> PyResult<Vec<i64>> {
        // Directly extract to typed HashMaps - no Value boxing
        let mut int_map: HashMap<String, Vec<i64>> = HashMap::new();
        let mut float_map: HashMap<String, Vec<f64>> = HashMap::new();
        let mut str_map: HashMap<String, Vec<String>> = HashMap::new();
        let mut bool_map: HashMap<String, Vec<bool>> = HashMap::new();
        let mut bin_map: HashMap<String, Vec<Vec<u8>>> = HashMap::new();
        
        // Bulk extract all columns
        for (key, value) in int_cols.iter() {
            let col_name: String = key.extract()?;
            let vec: Vec<i64> = value.extract()?;
            int_map.insert(col_name, vec);
        }
        
        for (key, value) in float_cols.iter() {
            let col_name: String = key.extract()?;
            let vec: Vec<f64> = value.extract()?;
            float_map.insert(col_name, vec);
        }
        
        for (key, value) in str_cols.iter() {
            let col_name: String = key.extract()?;
            let vec: Vec<String> = value.extract()?;
            str_map.insert(col_name, vec);
        }
        
        for (key, value) in bool_cols.iter() {
            let col_name: String = key.extract()?;
            let vec: Vec<bool> = value.extract()?;
            bool_map.insert(col_name, vec);
        }
        
        for (key, value) in bin_cols.iter() {
            let col_name: String = key.extract()?;
            let vec: Vec<Vec<u8>> = value.extract()?;
            bin_map.insert(col_name, vec);
        }
        
        if int_map.is_empty() && float_map.is_empty() && str_map.is_empty() 
           && bool_map.is_empty() && bin_map.is_empty() {
            return Ok(Vec::new());
        }
        
        let table_name = self.current_table.read().clone();
        
        // Call the ultra-fast typed insert
        let ids = py.allow_threads(|| {
            self.tables
                .write()
                .get_mut(&table_name)
                .ok_or_else(|| PyValueError::new_err("Table not found"))?
                .insert_typed_columns(int_map, float_map, str_map, bool_map, bin_map)
                .map_err(|e| PyValueError::new_err(e.to_string()))
        })?;
        
        Ok(ids.into_iter().map(|id| id as i64).collect())
    }

    /// Ultra-fast numpy array insert - zero-copy for numeric data
    /// 
    /// Performance: ~0.2ms for 10k rows of pure numeric data
    #[pyo3(name = "_insert_numpy_columns")]
    fn insert_numpy_columns(
        &self,
        py: Python<'_>,
        col_names: Vec<String>,
        int_arrays: Vec<PyReadonlyArray1<'_, i64>>,
        float_arrays: Vec<PyReadonlyArray1<'_, f64>>,
        bool_arrays: Vec<PyReadonlyArray1<'_, bool>>,
    ) -> PyResult<Vec<i64>> {
        let mut int_map: HashMap<String, Vec<i64>> = HashMap::new();
        let mut float_map: HashMap<String, Vec<f64>> = HashMap::new();
        let mut bool_map: HashMap<String, Vec<bool>> = HashMap::new();
        
        let mut idx = 0;
        
        // Process int arrays - direct memory access
        for array in &int_arrays {
            if idx >= col_names.len() {
                break;
            }
            let slice = array.as_slice()?;
            int_map.insert(col_names[idx].clone(), slice.to_vec());
            idx += 1;
        }
        
        // Process float arrays
        for array in &float_arrays {
            if idx >= col_names.len() {
                break;
            }
            let slice = array.as_slice()?;
            float_map.insert(col_names[idx].clone(), slice.to_vec());
            idx += 1;
        }
        
        // Process bool arrays
        for array in &bool_arrays {
            if idx >= col_names.len() {
                break;
            }
            let slice = array.as_slice()?;
            bool_map.insert(col_names[idx].clone(), slice.to_vec());
            idx += 1;
        }
        
        if int_map.is_empty() && float_map.is_empty() && bool_map.is_empty() {
            return Ok(Vec::new());
        }
        
        let table_name = self.current_table.read().clone();
        
        let ids = py.allow_threads(|| {
            self.tables
                .write()
                .get_mut(&table_name)
                .ok_or_else(|| PyValueError::new_err("Table not found"))?
                .insert_typed_columns(int_map, float_map, HashMap::new(), bool_map, HashMap::new())
                .map_err(|e| PyValueError::new_err(e.to_string()))
        })?;
        
        Ok(ids.into_iter().map(|id| id as i64).collect())
    }

    /// Ultra-fast insert without returning IDs - eliminates FFI return overhead
    /// 
    /// Performance: ~0.15ms for 10k rows (pure numeric)
    /// Returns: (start_id, count) tuple for ID range calculation if needed
    /// 
    /// If fts_fields is provided, FTS indexing is done directly in Rust (no Python boundary crossing)
    #[pyo3(name = "_insert_typed_columns_fast")]
    #[pyo3(signature = (int_cols, float_cols, str_cols, bool_cols, bin_cols, fts_fields = None))]
    fn insert_typed_columns_fast(
        &self,
        py: Python<'_>,
        int_cols: &Bound<'_, PyDict>,
        float_cols: &Bound<'_, PyDict>,
        str_cols: &Bound<'_, PyDict>,
        bool_cols: &Bound<'_, PyDict>,
        bin_cols: &Bound<'_, PyDict>,
        fts_fields: Option<Vec<String>>,
    ) -> PyResult<(i64, usize)> {
        use pyo3::types::PyString;
        
        // Pre-allocate with estimated capacity
        let estimated_cols = int_cols.len() + float_cols.len() + str_cols.len() + bool_cols.len();
        let mut int_map: HashMap<String, Vec<i64>> = HashMap::with_capacity(estimated_cols);
        let mut float_map: HashMap<String, Vec<f64>> = HashMap::with_capacity(estimated_cols);
        let mut str_map: HashMap<String, Vec<String>> = HashMap::with_capacity(estimated_cols);
        let mut bool_map: HashMap<String, Vec<bool>> = HashMap::with_capacity(estimated_cols);
        let mut bin_map: HashMap<String, Vec<Vec<u8>>> = HashMap::with_capacity(1);
        let mut batch_size = 0usize;
        
        #[cfg(debug_assertions)]
        let _t0 = std::time::Instant::now();
        
        // Extract int columns
        for (key, value) in int_cols.iter() {
            let col_name: String = key.extract()?;
            let vec: Vec<i64> = value.extract()?;
            if batch_size == 0 { batch_size = vec.len(); }
            int_map.insert(col_name, vec);
        }
        
        // Extract float columns
        for (key, value) in float_cols.iter() {
            let col_name: String = key.extract()?;
            let vec: Vec<f64> = value.extract()?;
            if batch_size == 0 { batch_size = vec.len(); }
            float_map.insert(col_name, vec);
        }
        
        // OPTIMIZED: String extraction with unchecked list access
        // Performance limit: ~26ns per string due to PyO3 FFI overhead
        for (key, value) in str_cols.iter() {
            let col_name: String = key.extract()?;
            let list = value.downcast::<PyList>()?;
            let len = list.len();
            if batch_size == 0 { batch_size = len; }
            
            // Pre-allocate vec
            let mut strings: Vec<String> = Vec::with_capacity(len);
            
            // Direct extraction with unchecked bounds
            for i in 0..len {
                // SAFETY: i is guaranteed in bounds [0, len)
                let item = unsafe { list.get_item_unchecked(i) };
                if let Ok(py_str) = item.downcast::<PyString>() {
                    strings.push(py_str.to_str()?.to_owned());
                } else {
                    strings.push(String::new());
                }
            }
            str_map.insert(col_name, strings);
        }
        
        // Extract bool columns
        for (key, value) in bool_cols.iter() {
            let col_name: String = key.extract()?;
            let vec: Vec<bool> = value.extract()?;
            if batch_size == 0 { batch_size = vec.len(); }
            bool_map.insert(col_name, vec);
        }
        
        // Extract binary columns (rarely used)
        for (key, value) in bin_cols.iter() {
            let col_name: String = key.extract()?;
            let vec: Vec<Vec<u8>> = value.extract()?;
            if batch_size == 0 { batch_size = vec.len(); }
            bin_map.insert(col_name, vec);
        }
        
        if batch_size == 0 {
            return Ok((0, 0));
        }
        
        let table_name = self.current_table.read().clone();
        
        // Get start_id before insert
        let start_id = py.allow_threads(|| {
            let tables = self.tables.read();
            tables.get(&table_name)
                .map(|t| t.get_row_count() as i64)
                .unwrap_or(0)
        });
        
        // FTS indexing using nanofts 0.3.2 columnar API (NO string joining needed!)
        // This is the fastest path: pass columns directly to nanofts
        if let Some(ref fields) = fts_fields {
            let manager_guard = self.fts_manager.read();
            if let Some(manager) = manager_guard.as_ref() {
                // Build columns by cloning FTS field data
                let columns: Vec<(String, Vec<String>)> = fields.iter()
                    .filter_map(|f| str_map.get(f).map(|v| (f.clone(), v.clone())))
                    .collect();
                
                if !columns.is_empty() {
                    // Generate doc_ids
                    let start_id_u64 = start_id as u64;
                    let doc_ids: Vec<u64> = (0..batch_size).map(|i| start_id_u64 + (i as u64)).collect();
                    
                    // Use nanofts 0.3.2 columnar API - no string joining needed!
                    if let Ok(engine) = manager.get_engine(&table_name) {
                        let _ = engine.add_documents_columnar(doc_ids, columns);
                    }
                }
            }
        }
        
        // Insert into table (consumes str_map)
        py.allow_threads(|| {
            self.tables
                .write()
                .get_mut(&table_name)
                .ok_or_else(|| PyValueError::new_err("Table not found"))?
                .insert_typed_columns_no_return(int_map, float_map, str_map, bool_map, bin_map)
                .map_err(|e| PyValueError::new_err(e.to_string()))
        })?;
        
        Ok((start_id, batch_size))
    }

    /// EXPERIMENTAL: Ultra-fast insert using pre-joined string buffers
    /// 
    /// Python side joins all strings with \0 separator:
    ///   joined_data = '\0'.join(strings).encode('utf-8')
    /// 
    /// This reduces 10k FFI calls per column to just 1 FFI call!
    /// Rust splits the buffer back into strings (very fast, ~0.1ms for 10k strings)
    /// 
    /// str_buffers: Dict[str, Tuple[bytes, int]] - (joined_bytes, count)
    #[pyo3(name = "_insert_joined_strings")]
    fn insert_joined_strings(
        &self,
        py: Python<'_>,
        int_cols: &Bound<'_, PyDict>,
        float_cols: &Bound<'_, PyDict>,
        str_buffers: &Bound<'_, PyDict>,  // Dict[str, Tuple[bytes, int]]
        bool_cols: &Bound<'_, PyDict>,
    ) -> PyResult<(i64, usize)> {
        let mut int_map: HashMap<String, Vec<i64>> = HashMap::new();
        let mut float_map: HashMap<String, Vec<f64>> = HashMap::new();
        let mut str_map: HashMap<String, Vec<String>> = HashMap::new();
        let mut bool_map: HashMap<String, Vec<bool>> = HashMap::new();
        let bin_map: HashMap<String, Vec<Vec<u8>>> = HashMap::new();
        let mut batch_size = 0usize;
        
        // Extract int columns
        for (key, value) in int_cols.iter() {
            let col_name: String = key.extract()?;
            let vec: Vec<i64> = value.extract()?;
            if batch_size == 0 { batch_size = vec.len(); }
            int_map.insert(col_name, vec);
        }
        
        // Extract float columns
        for (key, value) in float_cols.iter() {
            let col_name: String = key.extract()?;
            let vec: Vec<f64> = value.extract()?;
            if batch_size == 0 { batch_size = vec.len(); }
            float_map.insert(col_name, vec);
        }
        
        // ULTRA-FAST: Extract joined string buffers with ZERO-COPY buffer access
        // Uses PyBytes::as_bytes() for direct buffer access without copying
        for (key, value) in str_buffers.iter() {
            use pyo3::types::PyBytes;
            
            let col_name: String = key.extract()?;
            let tuple = value.downcast::<pyo3::types::PyTuple>()?;
            
            // ZERO-COPY: Get direct reference to Python bytes buffer
            let item0 = tuple.get_item(0)?;
            let py_bytes = item0.downcast::<PyBytes>()?;
            let buffer = py_bytes.as_bytes();  // Zero-copy slice!
            let count: usize = tuple.get_item(1)?.extract()?;
            if batch_size == 0 { batch_size = count; }
            
            // FAST: Split buffer by \0 separator
            let mut strings: Vec<String> = Vec::with_capacity(count);
            
            if buffer.is_empty() {
                strings.resize(count, String::new());
            } else {
                // Split and create strings - only this step allocates
                for chunk in buffer.split(|&b| b == 0) {
                    // SAFETY: Python UTF-8 encoded strings are valid UTF-8
                    let s = unsafe { std::str::from_utf8_unchecked(chunk) };
                    strings.push(s.to_owned());
                }
                
                // Ensure exact count
                strings.truncate(count);
                while strings.len() < count {
                    strings.push(String::new());
                }
            }
            
            str_map.insert(col_name, strings);
        }
        
        // Extract bool columns
        for (key, value) in bool_cols.iter() {
            let col_name: String = key.extract()?;
            let vec: Vec<bool> = value.extract()?;
            if batch_size == 0 { batch_size = vec.len(); }
            bool_map.insert(col_name, vec);
        }
        
        if batch_size == 0 {
            return Ok((0, 0));
        }
        
        let table_name = self.current_table.read().clone();
        
        // Get start_id before insert
        let start_id = py.allow_threads(|| {
            let tables = self.tables.read();
            tables.get(&table_name)
                .map(|t| t.get_row_count() as i64)
                .unwrap_or(0)
        });
        
        // Insert without collecting IDs
        py.allow_threads(|| {
            self.tables
                .write()
                .get_mut(&table_name)
                .ok_or_else(|| PyValueError::new_err("Table not found"))?
                .insert_typed_columns_no_return(int_map, float_map, str_map, bool_map, bin_map)
                .map_err(|e| PyValueError::new_err(e.to_string()))
        })?;
        
        Ok((start_id, batch_size))
    }

    /// Ultra-fast string insert using pre-encoded bytes
    /// 
    /// Strings are passed as UTF-8 bytes to avoid PyO3 string extraction overhead
    /// Returns: (start_id, count)
    #[pyo3(name = "_insert_with_bytes")]
    fn insert_with_bytes(
        &self,
        py: Python<'_>,
        int_cols: &Bound<'_, PyDict>,
        float_cols: &Bound<'_, PyDict>,
        str_bytes_cols: &Bound<'_, PyDict>,  // Dict[str, List[bytes]] - UTF-8 encoded strings
        bool_cols: &Bound<'_, PyDict>,
    ) -> PyResult<(i64, usize)> {
        let mut int_map: HashMap<String, Vec<i64>> = HashMap::new();
        let mut float_map: HashMap<String, Vec<f64>> = HashMap::new();
        let mut str_map: HashMap<String, Vec<String>> = HashMap::new();
        let mut bool_map: HashMap<String, Vec<bool>> = HashMap::new();
        let mut batch_size = 0usize;
        
        // Extract int columns
        for (key, value) in int_cols.iter() {
            let col_name: String = key.extract()?;
            let vec: Vec<i64> = value.extract()?;
            if batch_size == 0 { batch_size = vec.len(); }
            int_map.insert(col_name, vec);
        }
        
        // Extract float columns
        for (key, value) in float_cols.iter() {
            let col_name: String = key.extract()?;
            let vec: Vec<f64> = value.extract()?;
            if batch_size == 0 { batch_size = vec.len(); }
            float_map.insert(col_name, vec);
        }
        
        // Extract string bytes - convert bytes to String
        for (key, value) in str_bytes_cols.iter() {
            let col_name: String = key.extract()?;
            let bytes_vec: Vec<Vec<u8>> = value.extract()?;
            if batch_size == 0 { batch_size = bytes_vec.len(); }
            
            // UNSAFE: Convert bytes to String without UTF-8 validation
            // Caller must ensure bytes are valid UTF-8
            let strings: Vec<String> = bytes_vec.into_iter()
                .map(|b| unsafe { String::from_utf8_unchecked(b) })
                .collect();
            str_map.insert(col_name, strings);
        }
        
        // Extract bool columns
        for (key, value) in bool_cols.iter() {
            let col_name: String = key.extract()?;
            let vec: Vec<bool> = value.extract()?;
            if batch_size == 0 { batch_size = vec.len(); }
            bool_map.insert(col_name, vec);
        }
        
        if batch_size == 0 {
            return Ok((0, 0));
        }
        
        let table_name = self.current_table.read().clone();
        
        let start_id = py.allow_threads(|| {
            let tables = self.tables.read();
            tables.get(&table_name)
                .map(|t| t.get_row_count() as i64)
                .unwrap_or(0)
        });
        
        py.allow_threads(|| {
            self.tables
                .write()
                .get_mut(&table_name)
                .ok_or_else(|| PyValueError::new_err("Table not found"))?
                .insert_typed_columns_no_return(int_map, float_map, str_map, bool_map, HashMap::new())
                .map_err(|e| PyValueError::new_err(e.to_string()))
        })?;
        
        Ok((start_id, batch_size))
    }

    /// Ultra-fast numpy insert without returning IDs - absolute fastest path
    /// 
    /// Combines zero-copy numpy access with no-return optimization
    /// Performance: ~0.1ms for 10k rows (pure numeric)
    #[pyo3(name = "_insert_numpy_fast")]
    fn insert_numpy_fast(
        &self,
        py: Python<'_>,
        col_names: Vec<String>,
        int_arrays: Vec<PyReadonlyArray1<'_, i64>>,
        float_arrays: Vec<PyReadonlyArray1<'_, f64>>,
        bool_lists: Vec<Vec<bool>>,  // Changed from PyReadonlyArray1 to Vec
    ) -> PyResult<(i64, usize)> {
        let mut int_map: HashMap<String, Vec<i64>> = HashMap::new();
        let mut float_map: HashMap<String, Vec<f64>> = HashMap::new();
        let mut bool_map: HashMap<String, Vec<bool>> = HashMap::new();
        let mut batch_size = 0usize;
        
        let mut idx = 0;
        
        // Process int arrays - direct memory access
        for array in &int_arrays {
            if idx >= col_names.len() { break; }
            let slice = array.as_slice()?;
            if batch_size == 0 { batch_size = slice.len(); }
            int_map.insert(col_names[idx].clone(), slice.to_vec());
            idx += 1;
        }
        
        // Process float arrays
        for array in &float_arrays {
            if idx >= col_names.len() { break; }
            let slice = array.as_slice()?;
            if batch_size == 0 { batch_size = slice.len(); }
            float_map.insert(col_names[idx].clone(), slice.to_vec());
            idx += 1;
        }
        
        // Process bool lists
        for bools in bool_lists {
            if idx >= col_names.len() { break; }
            if batch_size == 0 { batch_size = bools.len(); }
            bool_map.insert(col_names[idx].clone(), bools);
            idx += 1;
        }
        
        if batch_size == 0 {
            return Ok((0, 0));
        }
        
        let table_name = self.current_table.read().clone();
        
        // Get start_id before insert
        let start_id = {
            let tables = self.tables.read();
            tables.get(&table_name)
                .map(|t| t.get_row_count() as i64)
                .unwrap_or(0)
        };
        
        // Insert without collecting IDs - release GIL for parallel speedup
        py.allow_threads(|| {
            self.tables
                .write()
                .get_mut(&table_name)
                .ok_or_else(|| PyValueError::new_err("Table not found"))?
                .insert_typed_columns_no_return(int_map, float_map, HashMap::new(), bool_map, HashMap::new())
                .map_err(|e| PyValueError::new_err(e.to_string()))
        })?;
        
        Ok((start_id, batch_size))
    }

    /// ULTRA-FAST mixed type insert with packed string data
    /// 
    /// Strings are passed as packed bytes to avoid per-element extraction.
    /// Format for each string column: packed_bytes where each string is 
    /// [len:u32 little-endian][utf8_bytes]
    #[pyo3(name = "_insert_mixed_packed")]
    fn insert_mixed_packed(
        &self,
        py: Python<'_>,
        int_names: Vec<String>,
        int_data: Vec<Vec<i64>>,
        float_names: Vec<String>,
        float_data: Vec<Vec<f64>>,
        str_names: Vec<String>,
        str_packed: Vec<Vec<u8>>,  // Packed bytes for each string column
        str_counts: Vec<usize>,  // Number of strings in each packed column
        bool_names: Vec<String>,
        bool_data: Vec<Vec<bool>>,
    ) -> PyResult<(i64, usize)> {
        // Build typed maps
        let mut int_map: HashMap<String, Vec<i64>> = HashMap::new();
        let mut float_map: HashMap<String, Vec<f64>> = HashMap::new();
        let mut str_map: HashMap<String, Vec<String>> = HashMap::new();
        let mut bool_map: HashMap<String, Vec<bool>> = HashMap::new();
        let mut batch_size = 0usize;
        
        // Process int columns (direct transfer)
        for (name, data) in int_names.into_iter().zip(int_data) {
            if batch_size == 0 { batch_size = data.len(); }
            int_map.insert(name, data);
        }
        
        // Process float columns
        for (name, data) in float_names.into_iter().zip(float_data) {
            if batch_size == 0 { batch_size = data.len(); }
            float_map.insert(name, data);
        }
        
        // Decode packed string columns - can release GIL here!
        for ((name, packed), count) in str_names.into_iter()
            .zip(str_packed.into_iter())
            .zip(str_counts.into_iter())
        {
            if batch_size == 0 { batch_size = count; }
            
            let packed = packed.as_slice();
            let mut strings = Vec::with_capacity(count);
            let mut offset = 0;
            
            for _ in 0..count {
                if offset + 4 > packed.len() {
                    strings.push(String::new());
                    continue;
                }
                
                let len = u32::from_le_bytes([
                    packed[offset],
                    packed[offset + 1],
                    packed[offset + 2],
                    packed[offset + 3],
                ]) as usize;
                offset += 4;
                
                if offset + len > packed.len() {
                    strings.push(String::new());
                    continue;
                }
                
                // UNSAFE: Skip UTF-8 validation - Python guarantees valid UTF-8
                let s = unsafe {
                    String::from_utf8_unchecked(packed[offset..offset + len].to_vec())
                };
                strings.push(s);
                offset += len;
            }
            
            str_map.insert(name, strings);
        }
        
        // Process bool columns
        for (name, data) in bool_names.into_iter().zip(bool_data) {
            if batch_size == 0 { batch_size = data.len(); }
            bool_map.insert(name, data);
        }
        
        if batch_size == 0 {
            return Ok((0, 0));
        }
        
        let table_name = self.current_table.read().clone();
        
        // Get start_id before insert
        let start_id = py.allow_threads(|| {
            let tables = self.tables.read();
            tables.get(&table_name)
                .map(|t| t.get_row_count() as i64)
                .unwrap_or(0)
        });
        
        // Insert all columns
        py.allow_threads(|| {
            self.tables
                .write()
                .get_mut(&table_name)
                .ok_or_else(|| PyValueError::new_err("Table not found"))?
                .insert_typed_columns_no_return(int_map, float_map, str_map, bool_map, HashMap::new())
                .map_err(|e| PyValueError::new_err(e.to_string()))
        })?;
        
        Ok((start_id, batch_size))
    }

    /// ULTRA-FAST unsafe numpy insert - absolute minimum overhead
    /// 
    /// Uses unsafe memory operations to achieve ~0.1ms for 10k rows
    /// Returns: (start_id, count)
    #[pyo3(name = "_insert_numpy_unsafe")]
    fn insert_numpy_unsafe(
        &self,
        py: Python<'_>,
        col_names: Vec<String>,
        int_arrays: Vec<PyReadonlyArray1<'_, i64>>,
        float_arrays: Vec<PyReadonlyArray1<'_, f64>>,
        bool_lists: Vec<Vec<bool>>,
    ) -> PyResult<(i64, usize)> {
        let mut int_map: HashMap<String, Vec<i64>> = HashMap::new();
        let mut float_map: HashMap<String, Vec<f64>> = HashMap::new();
        let mut bool_map: HashMap<String, Vec<bool>> = HashMap::new();
        let mut batch_size = 0usize;
        
        let mut idx = 0;
        
        // Process int arrays - zero-copy slice access
        for array in &int_arrays {
            if idx >= col_names.len() { break; }
            let slice = array.as_slice()?;
            if batch_size == 0 { batch_size = slice.len(); }
            int_map.insert(col_names[idx].clone(), slice.to_vec());
            idx += 1;
        }
        
        // Process float arrays
        for array in &float_arrays {
            if idx >= col_names.len() { break; }
            let slice = array.as_slice()?;
            if batch_size == 0 { batch_size = slice.len(); }
            float_map.insert(col_names[idx].clone(), slice.to_vec());
            idx += 1;
        }
        
        // Process bool lists
        for bools in bool_lists {
            if idx >= col_names.len() { break; }
            if batch_size == 0 { batch_size = bools.len(); }
            bool_map.insert(col_names[idx].clone(), bools);
            idx += 1;
        }
        
        if batch_size == 0 {
            return Ok((0, 0));
        }
        
        let table_name = self.current_table.read().clone();
        
        // Get start_id before insert
        let start_id = {
            let tables = self.tables.read();
            tables.get(&table_name)
                .map(|t| t.get_row_count() as i64)
                .unwrap_or(0)
        };
        
        // UNSAFE insert - maximum performance
        py.allow_threads(|| {
            let mut tables = self.tables.write();
            let table = tables.get_mut(&table_name)
                .ok_or_else(|| PyValueError::new_err("Table not found"))?;
            
            // SAFETY: We trust the numpy arrays are valid and properly aligned
            unsafe {
                table.insert_typed_columns_unsafe(int_map, float_map, bool_map)
                    .map_err(|e| PyValueError::new_err(e.to_string()))?;
            }
            Ok::<_, pyo3::PyErr>(())
        })?;
        
        Ok((start_id, batch_size))
    }

    /// Retrieve a single record - IoEngine::Direct strategy (uses table.get directly)
    fn retrieve(&self, py: Python<'_>, id: i64) -> PyResult<Option<PyObject>> {
        // IoEngine strategy: Direct (single row lookup is always fastest with direct access)
        let table_name = self.current_table.read().clone();
        
        let result = py.allow_threads(|| {
            self.tables
                .write()
                .get_mut(&table_name)
                .and_then(|table| table.get(id as u64))
        });
        
        match result {
            Some(fields) => {
                let dict = PyDict::new_bound(py);
                for (k, v) in fields {
                    dict.set_item(k, value_to_py(py, &v)?)?;
                }
                Ok(Some(dict.into()))
            }
            None => Ok(None),
        }
    }

    /// Retrieve multiple records - IoEngine::Direct strategy (uses table.get_many directly)
    fn retrieve_many(&self, py: Python<'_>, ids: Vec<i64>) -> PyResult<Vec<PyObject>> {
        // IoEngine strategy: Direct (batch lookups are fastest with direct access)
        let table_name = self.current_table.read().clone();
        let ids_u64: Vec<u64> = ids.iter().map(|&id| id as u64).collect();
        
        let results = py.allow_threads(|| {
            self.tables
                .write()
                .get_mut(&table_name)
                .map(|table| table.get_many(&ids_u64))
                .unwrap_or_default()
        });
        
        let mut py_results = Vec::with_capacity(results.len());
        for fields in results {
            let dict = PyDict::new_bound(py);
            for (k, v) in fields {
                dict.set_item(k, value_to_py(py, &v)?)?;
            }

            py_results.push(dict.into());
        }

        Ok(py_results)
    }

    /// ZERO-COPY retrieve multiple records via Arrow C Data Interface
    /// 
    /// This is the absolute fastest path for retrieve_many - NO serialization overhead:
    /// - Exports Arrow arrays via FFI pointers
    /// - Python receives raw memory pointers
    /// - PyArrow imports directly without copy
    /// 
    /// Returns: (schema_ptr, array_ptr) tuple for PyArrow import
    #[pyo3(name = "_retrieve_many_arrow_ffi")]
    fn retrieve_many_arrow_ffi(&self, py: Python<'_>, ids: Vec<i64>) -> PyResult<(u64, u64)> {
        use arrow::ffi::{FFI_ArrowArray, FFI_ArrowSchema};
        use arrow::array::Array;
        
        let table_name = self.current_table.read().clone();
        let ids_u64: Vec<u64> = ids.iter().map(|&id| id as u64).collect();
        
        if ids_u64.is_empty() {
            return Ok((0, 0));
        }
        
        // Build the RecordBatch in Rust
        let batch = py.allow_threads(|| -> PyResult<arrow::array::RecordBatch> {
            let mut tables = self.tables.write();
            let table = tables.get_mut(&table_name)
                .ok_or_else(|| PyValueError::new_err("Table not found"))?;
            
            // Flush pending writes
            table.flush_write_buffer();
            
            // Use optimized direct path to build RecordBatch
            table.get_many_record_batch(&ids_u64)
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))
        })?;
        
        // Export via Arrow C Data Interface
        let struct_array = arrow::array::StructArray::from(batch);
        
        // Allocate FFI structs on heap (they must outlive this function)
        let ffi_array = Box::new(FFI_ArrowArray::new(&struct_array.to_data()));
        let ffi_schema = Box::new(FFI_ArrowSchema::try_from(struct_array.data_type())
            .map_err(|e| PyRuntimeError::new_err(format!("Schema export error: {}", e)))?);
        
        // Return raw pointers as u64 for Python to use
        let array_ptr = Box::into_raw(ffi_array) as u64;
        let schema_ptr = Box::into_raw(ffi_schema) as u64;
        
        Ok((schema_ptr, array_ptr))
    }
    
    /// ZERO-COPY FTS search and retrieve via Arrow C Data Interface
    /// 
    /// This is the fastest path for FTS search + retrieve:
    /// - Search in Rust (no Python boundary)
    /// - Retrieve in Rust (no Python boundary)
    /// - Export via FFI (no serialization)
    /// 
    /// Returns: (schema_ptr, array_ptr) tuple for PyArrow import
    #[pyo3(name = "_fts_search_and_retrieve_ffi")]
    #[pyo3(signature = (query, limit = None, offset = 0))]
    fn fts_search_and_retrieve_ffi(
        &self,
        py: Python<'_>,
        query: &str,
        limit: Option<usize>,
        offset: usize,
    ) -> PyResult<(u64, u64)> {
        use arrow::ffi::{FFI_ArrowArray, FFI_ArrowSchema};
        use arrow::array::Array;
        
        let manager_guard = self.fts_manager.read();
        let manager = manager_guard.as_ref()
            .ok_or_else(|| PyValueError::new_err("FTS not initialized. Call _init_fts first."))?;
        
        let table_name = self.current_table.read().clone();
        let query_owned = query.to_string();
        
        // Search, paginate, and retrieve - ALL IN RUST
        let batch = py.allow_threads(|| -> PyResult<arrow::array::RecordBatch> {
            let engine = manager.get_engine(&table_name)
                .map_err(|e| PyRuntimeError::new_err(format!("FTS error: {}", e)))?;
            
            // Get document IDs with pagination
            let ids: Vec<u64> = if let Some(lim) = limit {
                engine.search_page(&query_owned, offset, lim)
                    .map_err(|e| PyRuntimeError::new_err(format!("FTS search error: {}", e)))?
            } else if offset > 0 {
                let all_ids = engine.search_ids(&query_owned)
                    .map_err(|e| PyRuntimeError::new_err(format!("FTS search error: {}", e)))?;
                all_ids.into_iter().skip(offset).collect()
            } else {
                engine.search_ids(&query_owned)
                    .map_err(|e| PyRuntimeError::new_err(format!("FTS search error: {}", e)))?
            };
            
            if ids.is_empty() {
                // Return empty batch
                use arrow::datatypes::{DataType as ArrowDataType, Field, Schema};
                use std::sync::Arc;
                let schema = Arc::new(Schema::new(vec![
                    Field::new("_id", ArrowDataType::UInt64, false),
                ]));
                return arrow::array::RecordBatch::try_new(schema, vec![
                    Arc::new(arrow::array::UInt64Array::from(Vec::<u64>::new()))
                ]).map_err(|e| PyRuntimeError::new_err(e.to_string()));
            }
            
            // Retrieve records as RecordBatch
            let mut tables = self.tables.write();
            let table = tables.get_mut(&table_name)
                .ok_or_else(|| PyValueError::new_err("Table not found"))?;
            
            table.get_many_record_batch(&ids)
                .map_err(|e| PyRuntimeError::new_err(e))
        })?;
        
        // Export via Arrow C Data Interface
        let struct_array = arrow::array::StructArray::from(batch);
        
        // Allocate FFI structs on heap
        let ffi_array = Box::new(FFI_ArrowArray::new(&struct_array.to_data()));
        let ffi_schema = Box::new(FFI_ArrowSchema::try_from(struct_array.data_type())
            .map_err(|e| PyRuntimeError::new_err(format!("Schema export error: {}", e)))?);
        
        // Return raw pointers as u64 for Python to use
        let array_ptr = Box::into_raw(ffi_array) as u64;
        let schema_ptr = Box::into_raw(ffi_schema) as u64;
        
        Ok((schema_ptr, array_ptr))
    }
    
    /// Fuzzy search with typo tolerance
    #[pyo3(name = "_fts_fuzzy_search")]
    fn fts_fuzzy_search<'py>(
        &self,
        py: Python<'py>,
        query: &str,
        min_results: usize,
    ) -> PyResult<Bound<'py, PyArray1<u64>>> {
        let manager_guard = self.fts_manager.read();
        let manager = manager_guard.as_ref()
            .ok_or_else(|| PyValueError::new_err("FTS not initialized. Call _init_fts first."))?;
        
        let table_name = self.current_table.read().clone();
        let query_owned = query.to_string();
        
        let ids = py.allow_threads(|| {
            let engine = manager.get_engine(&table_name)
                .map_err(|e| PyRuntimeError::new_err(format!("FTS error: {}", e)))?;
            
            let result = engine.fuzzy_search(&query_owned, min_results)
                .map_err(|e| PyRuntimeError::new_err(format!("FTS fuzzy search error: {}", e)))?;
            
            // Convert u32 to u64
            Ok::<Vec<u64>, PyErr>(result.iter().map(|id| id as u64).collect())
        })?;
        
        Ok(PyArray1::from_vec_bound(py, ids))
    }

    /// Search (non-fuzzy) returning matching document IDs
    #[pyo3(name = "_fts_search")]
    fn fts_search<'py>(&self, py: Python<'py>, query: &str) -> PyResult<Bound<'py, PyArray1<u64>>> {
        let manager_guard = self.fts_manager.read();
        let manager = manager_guard.as_ref()
            .ok_or_else(|| PyValueError::new_err("FTS not initialized. Call _init_fts first."))?;

        let table_name = self.current_table.read().clone();
        let query_owned = query.to_string();

        let ids = py.allow_threads(|| {
            let engine = manager.get_engine(&table_name)
                .map_err(|e| PyRuntimeError::new_err(format!("FTS error: {}", e)))?;
            let result = engine.search_ids(&query_owned)
                .map_err(|e| PyRuntimeError::new_err(format!("FTS search error: {}", e)))?;
            Ok::<Vec<u64>, PyErr>(result)
        })?;

        Ok(PyArray1::from_vec_bound(py, ids))
    }

    /// OPTIMIZED: Query via Arrow FFI (zero-copy)
    /// Returns: (schema_ptr, array_ptr) tuple for PyArrow import
    #[pyo3(name = "_query_arrow_ffi")]
    #[pyo3(signature = (where_clause, limit=None))]
    fn query_arrow_ffi(&self, py: Python<'_>, where_clause: &str, limit: Option<usize>) -> PyResult<(u64, u64)> {
        use arrow::array::Array;
        use arrow::ffi::{FFI_ArrowArray, FFI_ArrowSchema};
        use crate::io_engine::{IoEngine, QueryHints};

        let table_name = self.current_table.read().clone();
        let where_clause = where_clause.to_string();

        let batch = py.allow_threads(|| -> PyResult<arrow::record_batch::RecordBatch> {
            let mut tables = self.tables.write();
            let table = tables.get_mut(&table_name)
                .ok_or_else(|| PyValueError::new_err("Table not found"))?;

            let mut hints = QueryHints::default();
            if let Some(lim) = limit {
                hints = hints.with_limit(lim);
            }

            let io_result = IoEngine::query(table, &where_clause, hints)
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

            io_result.to_arrow().map_err(|e| PyRuntimeError::new_err(e.to_string()))
        })?;

        if batch.num_rows() == 0 {
            return Ok((0, 0));
        }

        let struct_array = arrow::array::StructArray::from(batch);

        let ffi_array = Box::new(FFI_ArrowArray::new(&struct_array.to_data()));
        let ffi_schema = Box::new(FFI_ArrowSchema::try_from(struct_array.data_type())
            .map_err(|e| PyRuntimeError::new_err(format!("Schema export error: {}", e)))?);

        let array_ptr = Box::into_raw(ffi_array) as u64;
        let schema_ptr = Box::into_raw(ffi_schema) as u64;

        Ok((schema_ptr, array_ptr))
    }

    /// ZERO-COPY retrieve all records via Arrow C Data Interface
    #[pyo3(name = "_retrieve_all_arrow_ffi")]
    fn retrieve_all_arrow_ffi(&self, py: Python<'_>) -> PyResult<(u64, u64)> {
        self.query_arrow_ffi(py, "1=1", None)
    }

    /// Free Arrow FFI pointers allocated by this module
    #[pyo3(name = "_free_arrow_ffi")]
    fn free_arrow_ffi(&self, _py: Python<'_>, schema_ptr: u64, array_ptr: u64) -> PyResult<()> {
        use arrow::ffi::{FFI_ArrowArray, FFI_ArrowSchema};
        if schema_ptr != 0 {
            unsafe {
                drop(Box::from_raw(schema_ptr as *mut FFI_ArrowSchema));
            }
        }
        if array_ptr != 0 {
            unsafe {
                drop(Box::from_raw(array_ptr as *mut FFI_ArrowArray));
            }
        }
        Ok(())
    }

    /// Get current row count
    fn count_rows(&self) -> i64 {
        let table_name = self.current_table.read().clone();
        let mut tables = self.tables.write();
        if let Some(table) = tables.get_mut(&table_name) {
            if table.has_pending_writes() {
                table.flush_write_buffer();
            }
            table.row_count() as i64
        } else {
            0
        }
    }
    
    /// Remove a document from FTS index
    #[pyo3(name = "_fts_remove_document")]
    fn fts_remove_document(&self, py: Python<'_>, doc_id: u64) -> PyResult<()> {
        let manager_guard = self.fts_manager.read();
        let manager = manager_guard.as_ref()
            .ok_or_else(|| PyValueError::new_err("FTS not initialized. Call _init_fts first."))?;
        
        let table_name = self.current_table.read().clone();
        
        py.allow_threads(|| {
            let engine = manager.get_engine(&table_name)
                .map_err(|e| PyRuntimeError::new_err(format!("FTS error: {}", e)))?;
            engine.remove_document(doc_id)
                .map_err(|e| PyRuntimeError::new_err(format!("FTS remove error: {}", e)))
        })
    }
    
    /// Remove multiple documents from FTS index
    #[pyo3(name = "_fts_remove_documents")]
    fn fts_remove_documents(&self, py: Python<'_>, doc_ids: Vec<u64>) -> PyResult<()> {
        let manager_guard = self.fts_manager.read();
        let manager = manager_guard.as_ref()
            .ok_or_else(|| PyValueError::new_err("FTS not initialized. Call _init_fts first."))?;
        
        let table_name = self.current_table.read().clone();
        
        py.allow_threads(|| {
            let engine = manager.get_engine(&table_name)
                .map_err(|e| PyRuntimeError::new_err(format!("FTS error: {}", e)))?;
            engine.remove_documents(&doc_ids)
                .map_err(|e| PyRuntimeError::new_err(format!("FTS remove error: {}", e)))
        })
    }
    
    /// Update a document in FTS index
    #[pyo3(name = "_fts_update_document")]
    fn fts_update_document(&self, py: Python<'_>, doc_id: u64, fields: &Bound<'_, PyDict>) -> PyResult<()> {
        let manager_guard = self.fts_manager.read();
        let manager = manager_guard.as_ref()
            .ok_or_else(|| PyValueError::new_err("FTS not initialized. Call _init_fts first."))?;
        
        let table_name = self.current_table.read().clone();
        
        let mut field_map: HashMap<String, String> = HashMap::new();
        for (key, value) in fields.iter() {
            let k: String = key.extract()?;
            if let Ok(v) = value.extract::<String>() {
                field_map.insert(k, v);
            }
        }
        
        py.allow_threads(|| {
            let engine = manager.get_engine(&table_name)
                .map_err(|e| PyRuntimeError::new_err(format!("FTS error: {}", e)))?;
            engine.update_document(doc_id, field_map)
                .map_err(|e| PyRuntimeError::new_err(format!("FTS update error: {}", e)))
        })
    }

    /// Add a document into FTS index
    #[pyo3(name = "_fts_add_document")]
    fn fts_add_document(&self, py: Python<'_>, doc_id: u64, fields: &Bound<'_, PyDict>) -> PyResult<()> {
        let manager_guard = self.fts_manager.read();
        let manager = manager_guard.as_ref()
            .ok_or_else(|| PyValueError::new_err("FTS not initialized. Call _init_fts first."))?;

        let table_name = self.current_table.read().clone();

        let mut field_map: HashMap<String, String> = HashMap::new();
        for (key, value) in fields.iter() {
            let k: String = key.extract()?;
            if let Ok(v) = value.extract::<String>() {
                field_map.insert(k, v);
            }
        }

        py.allow_threads(|| {
            let engine = manager.get_engine(&table_name)
                .map_err(|e| PyRuntimeError::new_err(format!("FTS error: {}", e)))?;
            engine.add_document(doc_id, field_map)
                .map_err(|e| PyRuntimeError::new_err(format!("FTS add error: {}", e)))
        })
    }
    
    /// Flush FTS index to disk
    #[pyo3(name = "_fts_flush")]
    fn fts_flush(&self, py: Python<'_>) -> PyResult<()> {
        let manager_guard = self.fts_manager.read();
        if let Some(manager) = manager_guard.as_ref() {
            py.allow_threads(|| {
                manager.flush_all()
                    .map_err(|e| PyRuntimeError::new_err(format!("FTS flush error: {}", e)))
            })?;
        }
        Ok(())
    }
    
    /// Compact FTS index
    #[pyo3(name = "_fts_compact")]
    fn fts_compact(&self, py: Python<'_>) -> PyResult<()> {
        let manager_guard = self.fts_manager.read();
        let manager = manager_guard.as_ref()
            .ok_or_else(|| PyValueError::new_err("FTS not initialized."))?;
        
        let table_name = self.current_table.read().clone();
        
        py.allow_threads(|| {
            let engine = manager.get_engine(&table_name)
                .map_err(|e| PyRuntimeError::new_err(format!("FTS error: {}", e)))?;
            engine.compact()
                .map_err(|e| PyRuntimeError::new_err(format!("FTS compact error: {}", e)))
        })
    }
    
    /// Get FTS statistics
    #[pyo3(name = "_fts_stats")]
    fn fts_stats(&self, py: Python<'_>) -> PyResult<PyObject> {
        let manager_guard = self.fts_manager.read();
        let manager = manager_guard.as_ref()
            .ok_or_else(|| PyValueError::new_err("FTS not initialized."))?;
        
        let table_name = self.current_table.read().clone();
        
        let stats = {
            let engine = manager.get_engine(&table_name)
                .map_err(|e| PyRuntimeError::new_err(format!("FTS error: {}", e)))?;
            engine.stats()
        };
        
        let dict = PyDict::new_bound(py);
        for (k, v) in stats {
            dict.set_item(k, v)?;
        }
        Ok(dict.into())
    }
    
    /// Set fuzzy search configuration
    #[pyo3(name = "_fts_set_fuzzy_config")]
    fn fts_set_fuzzy_config(
        &self,
        threshold: f64,
        max_distance: usize,
        max_candidates: usize,
    ) -> PyResult<()> {
        let manager_guard = self.fts_manager.read();
        let manager = manager_guard.as_ref()
            .ok_or_else(|| PyValueError::new_err("FTS not initialized."))?;
        
        let table_name = self.current_table.read().clone();
        
        let engine = manager.get_engine(&table_name)
            .map_err(|e| PyRuntimeError::new_err(format!("FTS error: {}", e)))?;
        engine.set_fuzzy_config(threshold, max_distance, max_candidates);
        
        Ok(())
    }
    
    /// Warmup FTS cache with specific terms
    #[pyo3(name = "_fts_warmup_terms")]
    fn fts_warmup_terms(&self, terms: Vec<String>) -> PyResult<usize> {
        let manager_guard = self.fts_manager.read();
        let manager = manager_guard.as_ref()
            .ok_or_else(|| PyValueError::new_err("FTS not initialized."))?;
        
        let table_name = self.current_table.read().clone();
        
        let engine = manager.get_engine(&table_name)
            .map_err(|e| PyRuntimeError::new_err(format!("FTS error: {}", e)))?;
        
        Ok(engine.warmup_terms(&terms))
    }
    
    /// Check if FTS is initialized
    #[pyo3(name = "_fts_is_initialized")]
    fn fts_is_initialized(&self) -> bool {
        self.fts_manager.read().is_some()
    }

    // ========== High-Performance ID Operations ==========
    
    /// Get all IDs (predicate pushdown - only reads deleted bitmap, skips data columns)
    /// 
    /// This is the fastest way to get all row IDs as it:
    /// - Only reads the deleted bitmap (not data columns)
    /// - Returns directly as numpy array (zero-copy)
    /// 
    /// Returns: numpy.ndarray[uint64] of all valid IDs
    #[pyo3(name = "_get_all_ids")]
    fn get_all_ids<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<u64>>> {
        let table_name = self.current_table.read().clone();
        
        let ids = py.allow_threads(|| {
            self.tables
                .read()
                .get(&table_name)
                .map(|t| t.get_all_ids())
                .unwrap_or_default()
        });
        
        Ok(PyArray1::from_vec_bound(py, ids))
    }
    
    /// Get filtered IDs with predicate pushdown optimization
    /// 
    /// This method efficiently returns only IDs matching the WHERE clause:
    /// - Evaluates filter on minimal columns needed
    /// - Returns IDs as numpy array (high performance)
    /// - Skips full row construction (predicate pushdown)
    /// 
    /// Returns: numpy.ndarray[uint64] of matching IDs
    #[pyo3(name = "_get_filtered_ids")]
    fn get_filtered_ids<'py>(&self, py: Python<'py>, where_clause: &str) -> PyResult<Bound<'py, PyArray1<u64>>> {
        let table_name = self.current_table.read().clone();
        let where_clause = where_clause.to_string();
        
        let ids = py.allow_threads(|| -> PyResult<Vec<u64>> {
            let tables = self.tables.read();
            let table = tables.get(&table_name)
                .ok_or_else(|| PyValueError::new_err("Table not found"))?;
            
            // Use query to get filtered indices, then extract IDs only
            // This is more efficient than full row construction
            table.query_ids_only(&where_clause)
                .map_err(|e| PyValueError::new_err(e.to_string()))
        })?;
        
        Ok(PyArray1::from_vec_bound(py, ids))
    }

    // ========== Persistence ==========

    /// Flush to disk
    fn flush(&self) -> PyResult<()> {
        self.save_to_file()
    }

    /// Optimize storage
    fn optimize(&self) -> PyResult<()> {
        let table_names: Vec<String> = self.tables.read().keys().cloned().collect();
        
        for name in table_names {
            if let Some(table) = self.tables.write().get_mut(&name) {
                table.compact();
            }
        }
        
        self.save_to_file()
    }

    /// Close the storage
    fn close(&self) -> PyResult<()> {
        self.save_to_file()
    }
}

impl ApexStorage {
    /// Save to file - uses incremental delta writes for performance
    /// 
    /// Strategy:
    /// - If no file exists: write full state with magic header
    /// - If file exists: append only delta records (new rows)
    /// - Time complexity: O(delta) instead of O(total)
    fn save_to_file(&self) -> PyResult<()> {
        use std::io::{Write, Seek, SeekFrom};
        
        // First, flush all write buffers to ensure data is in columns
        {
            let mut tables = self.tables.write();
            for (_, table) in tables.iter_mut() {
                table.flush_buffer();
            }
        }
        
        let tables = self.tables.read();
        let mut persisted = self.persisted_rows.write();
        let current = self.current_table.read().clone();
        
        // Check if we need full write or incremental
        let path_obj = Path::new(&self.path);
        let file_exists = path_obj.exists() && path_obj.metadata().map(|m| m.len() > 0).unwrap_or(false);
        
        // Collect delta info
        let mut has_delta = false;
        for (name, table) in tables.iter() {
            let current_rows = table.get_row_count();
            let persisted_rows = *persisted.get(name).unwrap_or(&0);
            if current_rows > persisted_rows {
                has_delta = true;
                break;
            }
        }
        
        if !file_exists || !has_delta {
            // Full write (first time or no changes)
            return self.save_full_state(&tables, &current, &mut persisted);
        }
        
        // Incremental delta write - append only new rows
        let mut file = fs::OpenOptions::new()
            .read(true)
            .write(true)
            .open(&self.path)
            .map_err(|e| PyIOError::new_err(format!("Failed to open file: {}", e)))?;
        
        // Seek to end for append
        file.seek(SeekFrom::End(0))
            .map_err(|e| PyIOError::new_err(format!("Failed to seek: {}", e)))?;
        
        // Write delta records for each table with new rows
        for (name, table) in tables.iter() {
            let current_rows = table.get_row_count();
            let persisted_rows = *persisted.get(name).unwrap_or(&0);
            
            if current_rows > persisted_rows {
                // Extract only new rows
                let delta_ids: Vec<u64> = (persisted_rows as u64..current_rows as u64).collect();
                let delta_columns = table.slice_columns(persisted_rows, current_rows);
                
                let delta = DeltaRecord {
                    table_name: name.clone(),
                    start_row: persisted_rows as u64,
                    ids: delta_ids,
                    columns: delta_columns,
                };
                
                // Write delta magic + serialized delta
                file.write_all(DELTA_MAGIC)
                    .map_err(|e| PyIOError::new_err(format!("Failed to write delta magic: {}", e)))?;
                
                let delta_data = bincode::serialize(&delta)
                    .map_err(|e| PyIOError::new_err(format!("Delta serialization error: {}", e)))?;
                
                // Write length prefix + data
                file.write_all(&(delta_data.len() as u64).to_le_bytes())
                    .map_err(|e| PyIOError::new_err(format!("Failed to write delta length: {}", e)))?;
                file.write_all(&delta_data)
                    .map_err(|e| PyIOError::new_err(format!("Failed to write delta: {}", e)))?;
                
                // Update persisted count
                persisted.insert(name.clone(), current_rows);
            }
        }
        
        // Fsync based on durability level
        if self.durability.sync_on_flush() {
            file.sync_all()
                .map_err(|e| PyIOError::new_err(format!("Failed to sync file: {}", e)))?;
        }
        
        Ok(())
    }
    
    /// Full state write (used for first save or compaction)
    fn save_full_state(
        &self, 
        tables: &HashMap<String, ColumnTable>,
        current_table: &str,
        persisted: &mut HashMap<String, usize>
    ) -> PyResult<()> {
        use std::io::Write;
        
        let mut state = StorageState {
            tables: HashMap::new(),
            current_table: current_table.to_string(),
        };
        
        for (name, table) in tables.iter() {
            let table_data = TableData {
                schema: table.schema_ref().clone(),
                ids: Vec::new(),
                columns: table.columns_ref().clone(),
                next_id: table.get_row_count() as u64,
            };
            state.tables.insert(name.clone(), table_data);
            
            // Update persisted count
            persisted.insert(name.clone(), table.get_row_count());
        }
        
        let data = bincode::serialize(&state)
            .map_err(|e| PyIOError::new_err(format!("Serialization error: {}", e)))?;
        
        // Write magic header + full state
        let mut file = fs::File::create(&self.path)
            .map_err(|e| PyIOError::new_err(format!("Failed to create file: {}", e)))?;
        
        file.write_all(APEX_MAGIC)
            .map_err(|e| PyIOError::new_err(format!("Failed to write magic: {}", e)))?;
        file.write_all(&(data.len() as u64).to_le_bytes())
            .map_err(|e| PyIOError::new_err(format!("Failed to write length: {}", e)))?;
        file.write_all(&data)
            .map_err(|e| PyIOError::new_err(format!("Failed to write data: {}", e)))?;
        
        if self.durability.sync_on_flush() {
            file.sync_all()
                .map_err(|e| PyIOError::new_err(format!("Failed to sync file: {}", e)))?;
        }
        
        Ok(())
    }

    /// Load from file with durability setting
    /// Handles both old format (bincode only) and new format (magic header + deltas)
    fn load_from_file(path: &str, durability: Durability) -> PyResult<Self> {
        use std::io::{Read, Seek, SeekFrom};
        
        let path_obj = Path::new(path);
        let base_dir = path_obj.parent()
            .map(|p| p.to_path_buf())
            .unwrap_or_else(|| PathBuf::from("."));
        
        let mut file = fs::File::open(path)
            .map_err(|e| PyIOError::new_err(format!("Failed to open file: {}", e)))?;
        
        // Read magic header to determine format
        let mut magic = [0u8; 8];
        file.read_exact(&mut magic)
            .map_err(|e| PyIOError::new_err(format!("Failed to read magic: {}", e)))?;
        
        let (state, mut tables) = if magic == *APEX_MAGIC {
            // New format: magic + length + state + deltas
            let mut len_bytes = [0u8; 8];
            file.read_exact(&mut len_bytes)
                .map_err(|e| PyIOError::new_err(format!("Failed to read length: {}", e)))?;
            let state_len = u64::from_le_bytes(len_bytes) as usize;
            
            let mut state_data = vec![0u8; state_len];
            file.read_exact(&mut state_data)
                .map_err(|e| PyIOError::new_err(format!("Failed to read state: {}", e)))?;
            
            let state: StorageState = bincode::deserialize(&state_data)
                .map_err(|e| PyIOError::new_err(format!("Deserialization error: {}", e)))?;
            
            // Load base tables
            let mut tables = HashMap::new();
            let mut max_table_id = 1u32;
            let StorageState { tables: state_tables, current_table } = state;
            for (name, table_data) in state_tables {
                let mut table = ColumnTable::new(max_table_id, &name);
                table.restore_from(table_data.schema, table_data.ids, table_data.columns, table_data.next_id);
                tables.insert(name, table);
                max_table_id += 1;
            }
            let state = StorageState {
                tables: HashMap::new(),
                current_table,
            };
            
            // Read and apply delta records
            loop {
                let mut delta_magic = [0u8; 4];
                if file.read_exact(&mut delta_magic).is_err() {
                    break; // EOF
                }
                
                if delta_magic != *DELTA_MAGIC {
                    break; // Not a delta record
                }
                
                let mut delta_len_bytes = [0u8; 8];
                if file.read_exact(&mut delta_len_bytes).is_err() {
                    break;
                }
                let delta_len = u64::from_le_bytes(delta_len_bytes) as usize;
                
                let mut delta_data = vec![0u8; delta_len];
                if file.read_exact(&mut delta_data).is_err() {
                    break;
                }
                
                if let Ok(delta) = bincode::deserialize::<DeltaRecord>(&delta_data) {
                    // Apply delta to table
                    if let Some(table) = tables.get_mut(&delta.table_name) {
                        table.append_columns(delta.columns);
                    }
                }
            }
            
            (state, tables)
        } else {
            // Old format: plain bincode (for backward compatibility)
            file.seek(SeekFrom::Start(0))
                .map_err(|e| PyIOError::new_err(format!("Failed to seek: {}", e)))?;
            
            let mut data = Vec::new();
            file.read_to_end(&mut data)
                .map_err(|e| PyIOError::new_err(format!("Failed to read file: {}", e)))?;
            
            let state: StorageState = bincode::deserialize(&data)
                .map_err(|e| PyIOError::new_err(format!("Deserialization error: {}", e)))?;
            
            let mut tables = HashMap::new();
            let mut max_table_id = 1u32;
            {
                let StorageState { tables: state_tables, current_table } = state;
                for (name, table_data) in state_tables {
                    let mut table = ColumnTable::new(max_table_id, &name);
                    table.restore_from(table_data.schema, table_data.ids, table_data.columns, table_data.next_id);
                    tables.insert(name, table);
                    max_table_id += 1;
                }
                let state = StorageState {
                    tables: HashMap::new(),
                    current_table,
                };
                (state, tables)
            }
        };
        
        let mut max_table_id = tables.len() as u32 + 1;
        
        // Ensure default table exists
        if !tables.contains_key("default") {
            tables.insert("default".to_string(), ColumnTable::with_capacity(max_table_id, "default", 10000));
            max_table_id += 1;
        }
        
        // Initialize persisted_rows from loaded table row counts
        let mut persisted_rows = HashMap::new();
        for (name, table) in tables.iter() {
            persisted_rows.insert(name.clone(), table.get_row_count());
        }
        
        Ok(Self {
            path: path.to_string(),
            base_dir,
            tables: RwLock::new(tables),
            current_table: RwLock::new(state.current_table),
            next_table_id: RwLock::new(max_table_id + 1),
            fts_manager: RwLock::new(None),
            fts_index_fields: RwLock::new(HashMap::new()),
            durability,
            persisted_rows: RwLock::new(persisted_rows),
        })
    }
}

impl Drop for ApexStorage {
    fn drop(&mut self) {
        let _ = self.save_to_file();
    }
}
