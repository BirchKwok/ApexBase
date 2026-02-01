//! Storage Backend Bridge
//!
//! This module bridges OnDemandStorage with ColumnTable, enabling:
//! - Lazy loading: only load data when needed
//! - Column projection: only load requested columns
//! - Memory-efficient persistence using the V3 format

use std::collections::HashMap;
use std::io;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use parking_lot::RwLock;

use arrow::record_batch::RecordBatch;

use crate::data::{DataType, Value};
use crate::storage::on_demand::{ColumnData, ColumnType, OnDemandStorage};
use crate::table::column_table::{BitVec, TypedColumn};
use crate::table::arrow_column::ArrowStringColumn;

// ============================================================================
// Type Conversions
// ============================================================================

/// Convert DataType to OnDemand ColumnType
pub fn datatype_to_column_type(dt: &DataType) -> ColumnType {
    match dt {
        DataType::Int64 | DataType::Int32 | DataType::Int16 | DataType::Int8 => ColumnType::Int64,
        DataType::Float64 | DataType::Float32 => ColumnType::Float64,
        DataType::String => ColumnType::String,
        DataType::Bool => ColumnType::Bool,
        DataType::Binary => ColumnType::Binary,
        _ => ColumnType::String, // Fallback for complex types
    }
}

/// Convert OnDemand ColumnType to DataType
pub fn column_type_to_datatype(ct: ColumnType) -> DataType {
    match ct {
        ColumnType::Int64 | ColumnType::Int32 | ColumnType::Int16 | ColumnType::Int8 |
        ColumnType::UInt64 | ColumnType::UInt32 | ColumnType::UInt16 | ColumnType::UInt8 => DataType::Int64,
        ColumnType::Float64 | ColumnType::Float32 => DataType::Float64,
        ColumnType::String | ColumnType::StringDict => DataType::String,
        ColumnType::Bool => DataType::Bool,
        ColumnType::Binary => DataType::Binary,
        ColumnType::Null => DataType::String,
    }
}

/// Convert TypedColumn to OnDemand ColumnData
pub fn typed_column_to_column_data(col: &TypedColumn) -> ColumnData {
    match col {
        TypedColumn::Int64 { data, .. } => {
            let mut cd = ColumnData::new(ColumnType::Int64);
            cd.extend_i64(data);
            cd
        }
        TypedColumn::Float64 { data, .. } => {
            let mut cd = ColumnData::new(ColumnType::Float64);
            cd.extend_f64(data);
            cd
        }
        TypedColumn::String(arrow_col) => {
            let mut cd = ColumnData::new(ColumnType::String);
            for i in 0..arrow_col.len() {
                if let Some(s) = arrow_col.get(i) {
                    cd.push_string(&s);
                } else {
                    cd.push_string("");
                }
            }
            cd
        }
        TypedColumn::Bool { data, .. } => {
            let mut cd = ColumnData::new(ColumnType::Bool);
            for i in 0..data.len() {
                cd.push_bool(data.get(i));
            }
            cd
        }
        TypedColumn::Mixed { data, .. } => {
            // Serialize mixed as JSON strings
            let mut cd = ColumnData::new(ColumnType::String);
            for v in data {
                let s = match v {
                    Value::String(s) => s.clone(),
                    Value::Int64(i) => i.to_string(),
                    Value::Float64(f) => f.to_string(),
                    Value::Bool(b) => b.to_string(),
                    _ => serde_json::to_string(v).unwrap_or_default(),
                };
                cd.push_string(&s);
            }
            cd
        }
    }
}

/// Convert OnDemand ColumnData to TypedColumn
pub fn column_data_to_typed_column(cd: &ColumnData, _dtype: DataType) -> TypedColumn {
    match cd {
        ColumnData::Int64(data) => {
            let mut nulls = BitVec::new();
            nulls.extend_false(data.len());
            TypedColumn::Int64 {
                data: data.clone(),
                nulls,
            }
        }
        ColumnData::Float64(data) => {
            let mut nulls = BitVec::new();
            nulls.extend_false(data.len());
            TypedColumn::Float64 {
                data: data.clone(),
                nulls,
            }
        }
        ColumnData::Bool { data, len } => {
            let mut bit_data = BitVec::new();
            let mut nulls = BitVec::new();
            for i in 0..*len {
                let byte_idx = i / 8;
                let bit_idx = i % 8;
                let val = if byte_idx < data.len() {
                    (data[byte_idx] >> bit_idx) & 1 == 1
                } else {
                    false
                };
                bit_data.push(val);
                nulls.push(false);
            }
            TypedColumn::Bool { data: bit_data, nulls }
        }
        ColumnData::String { offsets, data } => {
            let mut arrow_col = ArrowStringColumn::new();
            let count = offsets.len().saturating_sub(1);
            for i in 0..count {
                let start = offsets[i] as usize;
                let end = offsets[i + 1] as usize;
                if let Ok(s) = std::str::from_utf8(&data[start..end]) {
                    arrow_col.push(s);
                } else {
                    arrow_col.push_null();
                }
            }
            TypedColumn::String(arrow_col)
        }
        ColumnData::Binary { offsets, data } => {
            // Convert binary to Mixed with Binary values
            let mut values = Vec::new();
            let mut nulls = BitVec::new();
            let count = offsets.len().saturating_sub(1);
            for i in 0..count {
                let start = offsets[i] as usize;
                let end = offsets[i + 1] as usize;
                values.push(Value::Binary(data[start..end].to_vec()));
                nulls.push(false);
            }
            TypedColumn::Mixed { data: values, nulls }
        }
        ColumnData::StringDict { indices, dict_offsets, dict_data } => {
            // Convert dictionary-encoded string to regular String column
            let mut arrow_col = ArrowStringColumn::new();
            for &idx in indices {
                if idx == 0 {
                    arrow_col.push_null();
                } else {
                    let dict_idx = (idx - 1) as usize;
                    if dict_idx + 1 < dict_offsets.len() {
                        let start = dict_offsets[dict_idx] as usize;
                        let end = dict_offsets[dict_idx + 1] as usize;
                        if let Ok(s) = std::str::from_utf8(&dict_data[start..end]) {
                            arrow_col.push(s);
                        } else {
                            arrow_col.push_null();
                        }
                    } else {
                        arrow_col.push_null();
                    }
                }
            }
            TypedColumn::String(arrow_col)
        }
    }
}

// ============================================================================
// TableStorageBackend - Lazy Loading Storage Backend
// ============================================================================

/// Metadata for a lazy-loaded table
#[derive(Debug, Clone)]
pub struct TableMetadata {
    pub name: String,
    pub row_count: u64,
    pub schema: Vec<(String, DataType)>,
}

/// Storage backend with lazy loading support
/// 
/// This backend uses OnDemandStorage for persistence and supports:
/// - Lazy loading: data is only loaded when requested
/// - Column projection: only load specific columns
/// - Memory release: unload columns when not needed
/// - Configurable durability levels for ACID guarantees
pub struct TableStorageBackend {
    path: PathBuf,
    storage: OnDemandStorage,
    /// Cached column data (column_name -> TypedColumn)
    /// Only loaded columns are in cache
    cached_columns: RwLock<HashMap<String, TypedColumn>>,
    /// Schema mapping (column_name -> DataType)
    schema: RwLock<Vec<(String, DataType)>>,
    /// Cached row count
    row_count: RwLock<u64>,
    /// Whether data has been modified (needs save)
    dirty: RwLock<bool>,
}

impl TableStorageBackend {
    /// Create a new storage backend with default durability (Fast)
    pub fn create(path: &Path) -> io::Result<Self> {
        Self::create_with_durability(path, super::DurabilityLevel::Fast)
    }
    
    /// Create a new storage backend with specified durability level
    pub fn create_with_durability(path: &Path, durability: super::DurabilityLevel) -> io::Result<Self> {
        let storage = OnDemandStorage::create_with_durability(path, durability)?;
        Ok(Self {
            path: path.to_path_buf(),
            storage,
            cached_columns: RwLock::new(HashMap::new()),
            schema: RwLock::new(Vec::new()),
            row_count: RwLock::new(0),
            dirty: RwLock::new(false),
        })
    }

    /// Open existing storage with default durability (Fast)
    pub fn open(path: &Path) -> io::Result<Self> {
        Self::open_with_durability(path, super::DurabilityLevel::Fast)
    }
    
    /// Open existing storage with specified durability level
    pub fn open_with_durability(path: &Path, durability: super::DurabilityLevel) -> io::Result<Self> {
        let storage = OnDemandStorage::open_with_durability(path, durability)?;
        
        // Read schema from storage
        let storage_schema = storage.get_schema();
        let schema: Vec<(String, DataType)> = storage_schema
            .into_iter()
            .map(|(name, ct)| (name, column_type_to_datatype(ct)))
            .collect();
        
        let row_count = storage.row_count();
        
        Ok(Self {
            path: path.to_path_buf(),
            storage,
            cached_columns: RwLock::new(HashMap::new()),
            schema: RwLock::new(schema),
            row_count: RwLock::new(row_count),
            dirty: RwLock::new(false),
        })
    }

    /// Open or create storage with default durability (Fast)
    pub fn open_or_create(path: &Path) -> io::Result<Self> {
        Self::open_or_create_with_durability(path, super::DurabilityLevel::Fast)
    }
    
    /// Open or create storage with specified durability level
    pub fn open_or_create_with_durability(path: &Path, durability: super::DurabilityLevel) -> io::Result<Self> {
        if path.exists() {
            Self::open_with_durability(path, durability)
        } else {
            Self::create_with_durability(path, durability)
        }
    }

    /// Open for write with default durability (Fast)
    pub fn open_for_write(path: &Path) -> io::Result<Self> {
        Self::open_for_write_with_durability(path, super::DurabilityLevel::Fast)
    }
    
    /// Open for write with specified durability level - loads all existing data for append operations
    pub fn open_for_write_with_durability(path: &Path, durability: super::DurabilityLevel) -> io::Result<Self> {
        let storage = OnDemandStorage::open_for_write_with_durability(path, durability)?;
        
        let storage_schema = storage.get_schema();
        let schema: Vec<(String, DataType)> = storage_schema
            .into_iter()
            .map(|(name, ct)| (name, column_type_to_datatype(ct)))
            .collect();
        
        let row_count = storage.row_count();
        
        Ok(Self {
            path: path.to_path_buf(),
            storage,
            cached_columns: RwLock::new(HashMap::new()),
            schema: RwLock::new(schema),
            row_count: RwLock::new(row_count),
            dirty: RwLock::new(false),
        })
    }

    /// Get metadata without loading data
    pub fn metadata(&self) -> TableMetadata {
        TableMetadata {
            name: self.path.file_stem()
                .and_then(|s| s.to_str())
                .unwrap_or("unknown")
                .to_string(),
            row_count: *self.row_count.read(),
            schema: self.schema.read().clone(),
        }
    }

    /// Get row count
    pub fn row_count(&self) -> u64 {
        *self.row_count.read()
    }

    /// Get schema
    pub fn get_schema(&self) -> Vec<(String, DataType)> {
        self.schema.read().clone()
    }

    /// Get column names
    pub fn column_names(&self) -> Vec<String> {
        let mut names = vec!["_id".to_string()];
        names.extend(self.schema.read().iter().map(|(n, _)| n.clone()));
        names
    }

    // ========================================================================
    // Lazy Loading APIs
    // ========================================================================

    /// Load specific columns into cache (lazy load)
    /// Only loads columns that are not already cached
    pub fn load_columns(&self, column_names: &[&str]) -> io::Result<()> {
        let cached = self.cached_columns.read();
        let to_load: Vec<&str> = column_names
            .iter()
            .filter(|&name| !cached.contains_key(*name))
            .copied()
            .collect();
        drop(cached);

        if to_load.is_empty() {
            return Ok(());
        }

        // Read columns from storage
        let col_data = self.storage.read_columns(Some(&to_load), 0, None)?;
        
        // Convert and cache
        let schema = self.schema.read();
        let mut cached = self.cached_columns.write();
        
        for (name, data) in col_data {
            let dtype = schema.iter()
                .find(|(n, _)| n == &name)
                .map(|(_, dt)| dt.clone())
                .unwrap_or(DataType::String);
            
            let typed_col = column_data_to_typed_column(&data, dtype);
            cached.insert(name, typed_col);
        }

        Ok(())
    }

    /// Load all columns into cache
    pub fn load_all_columns(&self) -> io::Result<()> {
        let names: Vec<String> = self.schema.read().iter().map(|(n, _)| n.clone()).collect();
        let refs: Vec<&str> = names.iter().map(|s| s.as_str()).collect();
        self.load_columns(&refs)
    }

    /// Get a cached column (returns None if not loaded)
    pub fn get_cached_column(&self, name: &str) -> Option<TypedColumn> {
        self.cached_columns.read().get(name).cloned()
    }

    /// Get column, loading if necessary
    pub fn get_column(&self, name: &str) -> io::Result<Option<TypedColumn>> {
        // Check cache first
        if let Some(col) = self.cached_columns.read().get(name).cloned() {
            return Ok(Some(col));
        }

        // Load from storage
        self.load_columns(&[name])?;
        Ok(self.cached_columns.read().get(name).cloned())
    }

    /// Release cached columns to free memory
    pub fn release_columns(&self, column_names: &[&str]) {
        let mut cached = self.cached_columns.write();
        for name in column_names {
            cached.remove(*name);
        }
    }

    /// Release all cached columns
    pub fn release_all_columns(&self) {
        self.cached_columns.write().clear();
    }

    /// Get memory usage of cached columns (approximate)
    pub fn cached_memory_bytes(&self) -> usize {
        let cached = self.cached_columns.read();
        let mut total = 0;
        for (_, col) in cached.iter() {
            total += match col {
                TypedColumn::Int64 { data, .. } => data.len() * 8,
                TypedColumn::Float64 { data, .. } => data.len() * 8,
                TypedColumn::String(arrow_col) => arrow_col.len() * 32, // Approximate: 32 bytes per string
                TypedColumn::Bool { data, .. } => data.len() / 8 + 1,
                TypedColumn::Mixed { data, .. } => data.len() * 64, // Approximate
            };
        }
        total
    }

    // ========================================================================
    // Write APIs
    // ========================================================================

    /// Insert rows (updates cache and marks dirty)
    /// Optimized with parallel conversion for large batches
    pub fn insert_rows(&self, rows: &[HashMap<String, Value>]) -> io::Result<Vec<u64>> {
        use crate::storage::on_demand::ColumnValue;
        use rayon::prelude::*;

        if rows.is_empty() {
            return Ok(Vec::new());
        }

        // Convert to ColumnValue format - use parallel for large batches
        let converted: Vec<HashMap<String, ColumnValue>> = if rows.len() > 1000 {
            rows.par_iter()
                .map(|row| {
                    row.iter()
                        .map(|(k, v)| {
                            let cv = match v {
                                Value::Int64(i) => ColumnValue::Int64(*i),
                                Value::Int32(i) => ColumnValue::Int64(*i as i64),
                                Value::Float64(f) => ColumnValue::Float64(*f),
                                Value::Float32(f) => ColumnValue::Float64(*f as f64),
                                Value::String(s) => ColumnValue::String(s.clone()),
                                Value::Bool(b) => ColumnValue::Bool(*b),
                                Value::Binary(b) => ColumnValue::Binary(b.clone()),
                                Value::Null => ColumnValue::Null,
                                _ => ColumnValue::String(serde_json::to_string(v).unwrap_or_default()),
                            };
                            (k.clone(), cv)
                        })
                        .collect()
                })
                .collect()
        } else {
            rows.iter()
                .map(|row| {
                    row.iter()
                        .map(|(k, v)| {
                            let cv = match v {
                                Value::Int64(i) => ColumnValue::Int64(*i),
                                Value::Int32(i) => ColumnValue::Int64(*i as i64),
                                Value::Float64(f) => ColumnValue::Float64(*f),
                                Value::Float32(f) => ColumnValue::Float64(*f as f64),
                                Value::String(s) => ColumnValue::String(s.clone()),
                                Value::Bool(b) => ColumnValue::Bool(*b),
                                Value::Binary(b) => ColumnValue::Binary(b.clone()),
                                Value::Null => ColumnValue::Null,
                                _ => ColumnValue::String(serde_json::to_string(v).unwrap_or_default()),
                            };
                            (k.clone(), cv)
                        })
                        .collect()
                })
                .collect()
        };

        // Insert into storage
        let ids = self.storage.insert_rows(&converted)?;

        // Update schema if new columns (only check first row for perf)
        {
            let mut schema = self.schema.write();
            if let Some(row) = rows.first() {
                for (k, v) in row {
                    if k != "_id" && !schema.iter().any(|(n, _)| n == k) {
                        schema.push((k.clone(), v.data_type()));
                    }
                }
            }
        }

        // Update row count
        *self.row_count.write() += rows.len() as u64;

        // Invalidate cache (data changed)
        self.cached_columns.write().clear();
        *self.dirty.write() = true;

        Ok(ids)
    }

    /// Insert typed columns directly - bypasses row-by-row conversion
    /// Much faster for bulk inserts with homogeneous columnar data
    pub fn insert_typed(
        &self,
        int_columns: HashMap<String, Vec<i64>>,
        float_columns: HashMap<String, Vec<f64>>,
        string_columns: HashMap<String, Vec<String>>,
        binary_columns: HashMap<String, Vec<Vec<u8>>>,
        bool_columns: HashMap<String, Vec<bool>>,
    ) -> io::Result<Vec<u64>> {
        // Delegate to storage
        let ids = self.storage.insert_typed(
            int_columns.clone(), 
            float_columns.clone(), 
            string_columns.clone(), 
            binary_columns.clone(), 
            bool_columns.clone()
        )?;

        // Update schema if new columns
        {
            let mut schema = self.schema.write();
            for name in int_columns.keys() {
                if !schema.iter().any(|(n, _)| n == name) {
                    schema.push((name.clone(), crate::data::DataType::Int64));
                }
            }
            for name in float_columns.keys() {
                if !schema.iter().any(|(n, _)| n == name) {
                    schema.push((name.clone(), crate::data::DataType::Float64));
                }
            }
            for name in string_columns.keys() {
                if !schema.iter().any(|(n, _)| n == name) {
                    schema.push((name.clone(), crate::data::DataType::String));
                }
            }
            for name in binary_columns.keys() {
                if !schema.iter().any(|(n, _)| n == name) {
                    schema.push((name.clone(), crate::data::DataType::Binary));
                }
            }
            for name in bool_columns.keys() {
                if !schema.iter().any(|(n, _)| n == name) {
                    schema.push((name.clone(), crate::data::DataType::Bool));
                }
            }
        }

        // Update row count
        *self.row_count.write() += ids.len() as u64;

        // Invalidate cache (data changed)
        self.cached_columns.write().clear();
        *self.dirty.write() = true;

        Ok(ids)
    }

    /// Save changes to disk
    pub fn save(&self) -> io::Result<()> {
        self.storage.save()?;
        *self.dirty.write() = false;
        Ok(())
    }
    
    /// Explicitly sync data to disk (fsync)
    /// 
    /// This ensures all buffered data is written to persistent storage.
    /// For Safe/Max durability levels, save() automatically calls fsync.
    /// For Fast durability, use this method when you need explicit durability.
    pub fn sync(&self) -> io::Result<()> {
        self.storage.sync()
    }
    
    /// Get the current durability level
    pub fn durability(&self) -> super::DurabilityLevel {
        self.storage.durability()
    }
    
    /// Set auto-flush thresholds
    /// 
    /// When either threshold is exceeded during writes, data is automatically 
    /// written to file. Set to 0 to disable the respective threshold.
    pub fn set_auto_flush(&self, rows: u64, bytes: u64) {
        self.storage.set_auto_flush(rows, bytes);
    }
    
    /// Get current auto-flush configuration
    pub fn get_auto_flush(&self) -> (u64, u64) {
        self.storage.get_auto_flush()
    }
    
    /// Estimate current in-memory data size in bytes
    pub fn estimate_memory_bytes(&self) -> u64 {
        self.storage.estimate_memory_bytes()
    }

    /// Check if there are unsaved changes
    pub fn is_dirty(&self) -> bool {
        *self.dirty.read()
    }

    /// Flush and close - releases mmap and file handle
    /// IMPORTANT: On Windows, this must be called before temp directory cleanup
    pub fn close(&self) -> io::Result<()> {
        if self.is_dirty() {
            self.save()?;
        }
        // Release mmap and file handle (critical for Windows)
        self.storage.close()
    }

    // ========================================================================
    // Delete/Update APIs
    // ========================================================================

    /// Delete a row by ID (soft delete)
    pub fn delete(&self, id: u64) -> bool {
        let result = self.storage.delete(id);
        if result {
            *self.dirty.write() = true;
        }
        result
    }

    /// Delete multiple rows by IDs (soft delete)
    pub fn delete_batch(&self, ids: &[u64]) -> bool {
        let result = self.storage.delete_batch(ids);
        if result {
            *self.dirty.write() = true;
        }
        result
    }

    /// Check if a row exists and is not deleted
    pub fn exists(&self, id: u64) -> bool {
        self.storage.exists(id)
    }

    /// Get active (non-deleted) row count
    pub fn active_row_count(&self) -> u64 {
        self.storage.active_row_count()
    }

    /// Replace a row (delete + insert new)
    pub fn replace(&self, id: u64, data: &HashMap<String, Value>) -> io::Result<bool> {
        use crate::storage::on_demand::ColumnValue;
        
        // Convert Value to ColumnValue
        let cv_data: HashMap<String, ColumnValue> = data.iter()
            .map(|(k, v)| {
                let cv = match v {
                    Value::Int64(i) => ColumnValue::Int64(*i),
                    Value::Int32(i) => ColumnValue::Int64(*i as i64),
                    Value::Float64(f) => ColumnValue::Float64(*f),
                    Value::Float32(f) => ColumnValue::Float64(*f as f64),
                    Value::String(s) => ColumnValue::String(s.clone()),
                    Value::Bool(b) => ColumnValue::Bool(*b),
                    Value::Binary(b) => ColumnValue::Binary(b.clone()),
                    Value::Null => ColumnValue::Null,
                    _ => ColumnValue::String(serde_json::to_string(v).unwrap_or_default()),
                };
                (k.clone(), cv)
            })
            .collect();
        
        let result = self.storage.replace(id, &cv_data)?;
        if result {
            *self.dirty.write() = true;
            // Invalidate cache
            self.cached_columns.write().clear();
            // Update row count
            *self.row_count.write() = self.storage.row_count();
        }
        Ok(result)
    }

    // ========================================================================
    // Schema Operations
    // ========================================================================

    /// Add a column to the schema and storage with padding for existing rows
    pub fn add_column(&self, name: &str, dtype: DataType) -> io::Result<()> {
        // Check if column already exists
        {
            let schema = self.schema.read();
            if schema.iter().any(|(n, _)| n == name) {
                return Err(io::Error::new(io::ErrorKind::AlreadyExists, format!("Column '{}' already exists", name)));
            }
        }
        
        // Use the underlying storage's add_column_with_padding for proper data alignment
        self.storage.add_column_with_padding(name, dtype)?;
        
        // Update our schema cache
        let mut schema = self.schema.write();
        schema.push((name.to_string(), dtype));
        *self.dirty.write() = true;
        Ok(())
    }

    /// Drop a column from the schema
    pub fn drop_column(&self, name: &str) -> io::Result<()> {
        let mut schema = self.schema.write();
        let pos = schema.iter().position(|(n, _)| n == name);
        if let Some(idx) = pos {
            schema.remove(idx);
            *self.dirty.write() = true;
            Ok(())
        } else {
            Err(io::Error::new(io::ErrorKind::NotFound, format!("Column '{}' not found", name)))
        }
    }

    /// Rename a column
    pub fn rename_column(&self, old_name: &str, new_name: &str) -> io::Result<()> {
        let mut schema = self.schema.write();
        for (name, _) in schema.iter_mut() {
            if name == old_name {
                *name = new_name.to_string();
                *self.dirty.write() = true;
                return Ok(());
            }
        }
        Err(io::Error::new(io::ErrorKind::NotFound, format!("Column '{}' not found", old_name)))
    }

    /// List all column names
    pub fn list_columns(&self) -> Vec<String> {
        self.schema.read().iter().map(|(n, _)| n.clone()).collect()
    }

    /// Get column data type
    pub fn get_column_type(&self, name: &str) -> Option<DataType> {
        self.schema.read().iter()
            .find(|(n, _)| n == name)
            .map(|(_, dt)| dt.clone())
    }

    // ========================================================================
    // True On-Demand Column Projection APIs
    // ========================================================================

    /// Read specific columns directly to Arrow RecordBatch (TRUE on-demand read)
    /// 
    /// This method bypasses ColumnTable and reads only the requested columns
    /// from storage, converting directly to Arrow format.
    /// 
    /// Features:
    /// - Column projection: only reads requested columns from disk
    /// - Row range: supports start_row and row_count for partial reads
    /// - Caching: caches full column reads for repeated access
    /// 
    /// # Arguments
    /// * `column_names` - Columns to read (None = all columns)
    /// * `start_row` - Starting row index
    /// * `row_count` - Number of rows to read (None = all)
    pub fn read_columns_to_arrow(
        &self,
        column_names: Option<&[&str]>,
        start_row: usize,
        row_count: Option<usize>,
    ) -> io::Result<arrow::record_batch::RecordBatch> {
        use arrow::array::{ArrayRef, Int64Array, Float64Array, StringArray, BooleanArray};
        use arrow::datatypes::{Schema, Field, DataType as ArrowDataType};
        use std::sync::Arc;

        // For full column reads (start_row=0, row_count=None), check cache first
        let use_cache = start_row == 0 && row_count.is_none();
        
        // Handle SELECT _id ONLY case FIRST (before reading columns)
        if let Some(cols) = column_names {
            if cols.len() == 1 && cols[0] == "_id" {
                // Only _id requested - return batch with just _id column
                let ids = self.storage.read_ids(start_row, row_count)?;
                let fields = vec![Field::new("_id", ArrowDataType::Int64, false)];
                // OPTIMIZATION: Direct transmute from Vec<u64> to Vec<i64>
                let ids_i64: Vec<i64> = unsafe {
                    let mut ids = std::mem::ManuallyDrop::new(ids);
                    Vec::from_raw_parts(ids.as_mut_ptr() as *mut i64, ids.len(), ids.capacity())
                };
                let arrays: Vec<ArrayRef> = vec![Arc::new(Int64Array::from(ids_i64))];
                let schema = Arc::new(Schema::new(fields));
                return arrow::record_batch::RecordBatch::try_new(schema, arrays)
                    .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e.to_string()));
            }
        }
        
        // Read columns from storage (only the requested ones!)
        let mut col_data = self.storage.read_columns(column_names, start_row, row_count)?;
        
        if col_data.is_empty() {
            // Return empty batch with schema (including _id if requested)
            let schema = self.schema.read();
            let include_id = column_names.map(|cols| cols.contains(&"_id")).unwrap_or(true);
            let mut fields: Vec<Field> = Vec::new();
            
            if include_id {
                fields.push(Field::new("_id", ArrowDataType::Int64, false));
            }
            
            for (name, dt) in schema.iter() {
                if column_names.map(|cols| cols.contains(&name.as_str())).unwrap_or(true) {
                    let arrow_dt = match dt {
                        DataType::Int64 | DataType::Int32 | DataType::Int16 | DataType::Int8 => ArrowDataType::Int64,
                        DataType::Float64 | DataType::Float32 => ArrowDataType::Float64,
                        DataType::String => ArrowDataType::Utf8,
                        DataType::Bool => ArrowDataType::Boolean,
                        _ => ArrowDataType::Utf8,
                    };
                    fields.push(Field::new(name, arrow_dt, true));
                }
            }
            let schema = Arc::new(Schema::new(fields));
            return Ok(arrow::record_batch::RecordBatch::new_empty(schema));
        }

        // Build Arrow arrays from ColumnData
        let schema = self.schema.read();
        let mut fields: Vec<Field> = Vec::new();
        let mut arrays: Vec<ArrayRef> = Vec::new();

        // Always include _id as the first column (unless explicitly excluded)
        let include_id = column_names.map(|cols| cols.contains(&"_id")).unwrap_or(true);
        let expected_row_count: usize;
        if include_id {
            let ids = self.storage.read_ids(start_row, row_count)?;
            expected_row_count = ids.len();
            fields.push(Field::new("_id", ArrowDataType::Int64, false));
            // OPTIMIZATION: Direct transmute from Vec<u64> to Vec<i64> - same memory layout
            let ids_i64: Vec<i64> = unsafe {
                let mut ids = std::mem::ManuallyDrop::new(ids);
                Vec::from_raw_parts(ids.as_mut_ptr() as *mut i64, ids.len(), ids.capacity())
            };
            arrays.push(Arc::new(Int64Array::from(ids_i64)));
        } else {
            // If no _id, get row count from any column
            expected_row_count = col_data.values().next().map(|d| d.len()).unwrap_or(0);
        }

        // Determine column order from schema (or from column_names if specified)
        let col_order: Vec<String> = if let Some(names) = column_names {
            names.iter()
                .filter(|&s| *s != "_id")  // Skip _id, already handled
                .map(|s| s.to_string())
                .collect()
        } else {
            schema.iter().map(|(n, _)| n.clone()).collect()
        };

        for col_name in &col_order {
            // Use remove() to take ownership and avoid clone
            if let Some(data) = col_data.remove(col_name) {
                let (arrow_dt, array): (ArrowDataType, ArrayRef) = match data {
                    ColumnData::Int64(values) => {
                        // Zero-copy: take ownership of the Vec directly
                        if values.len() < expected_row_count {
                            let mut padded: Vec<Option<i64>> = values.into_iter().map(Some).collect();
                            padded.extend(std::iter::repeat(None).take(expected_row_count - padded.len()));
                            (ArrowDataType::Int64, Arc::new(Int64Array::from(padded)))
                        } else if values.len() > expected_row_count {
                            // Truncate to expected row count
                            let truncated: Vec<i64> = values.into_iter().take(expected_row_count).collect();
                            (ArrowDataType::Int64, Arc::new(Int64Array::from(truncated)))
                        } else {
                            // Direct conversion without clone
                            (ArrowDataType::Int64, Arc::new(Int64Array::from(values)))
                        }
                    }
                    ColumnData::Float64(values) => {
                        if values.len() < expected_row_count {
                            let mut padded: Vec<Option<f64>> = values.into_iter().map(Some).collect();
                            padded.extend(std::iter::repeat(None).take(expected_row_count - padded.len()));
                            (ArrowDataType::Float64, Arc::new(Float64Array::from(padded)))
                        } else if values.len() > expected_row_count {
                            // Truncate to expected row count
                            let truncated: Vec<f64> = values.into_iter().take(expected_row_count).collect();
                            (ArrowDataType::Float64, Arc::new(Float64Array::from(truncated)))
                        } else {
                            (ArrowDataType::Float64, Arc::new(Float64Array::from(values)))
                        }
                    }
                    ColumnData::String { offsets, data: bytes } => {
                        // OPTIMIZATION: Build StringArray directly from offsets and data buffers
                        // Avoids per-element iteration and String allocation
                        use arrow::buffer::{Buffer, OffsetBuffer};
                        use arrow::array::GenericStringArray;
                        
                        let count = offsets.len().saturating_sub(1);
                        
                        if count == expected_row_count {
                            // Fast path: direct buffer construction (zero-copy for offsets)
                            // Convert u32 offsets to i32 for Arrow
                            let arrow_offsets: Vec<i32> = offsets.iter().map(|&o| o as i32).collect();
                            let offset_buffer = OffsetBuffer::new(arrow_offsets.into());
                            let data_buffer = Buffer::from(bytes);
                            
                            // SAFETY: We trust the offsets are valid UTF-8 boundaries
                            let string_array = unsafe {
                                GenericStringArray::<i32>::new_unchecked(offset_buffer, data_buffer, None)
                            };
                            (ArrowDataType::Utf8, Arc::new(string_array) as ArrayRef)
                        } else {
                            // Fallback: standard conversion for mismatched counts
                            let strings: Vec<Option<&str>> = (0..count.min(expected_row_count))
                                .map(|i| {
                                    let start = offsets[i] as usize;
                                    let end = offsets[i + 1] as usize;
                                    std::str::from_utf8(&bytes[start..end]).ok()
                                })
                                .collect();
                            if strings.len() < expected_row_count {
                                let mut owned: Vec<Option<String>> = strings.into_iter()
                                    .map(|s| s.map(|s| s.to_string()))
                                    .collect();
                                owned.extend(std::iter::repeat(None).take(expected_row_count - owned.len()));
                                (ArrowDataType::Utf8, Arc::new(StringArray::from(owned)))
                            } else {
                                (ArrowDataType::Utf8, Arc::new(StringArray::from(strings)))
                            }
                        }
                    }
                    ColumnData::Bool { data: packed, len } => {
                        let mut bools: Vec<Option<bool>> = (0..len)
                            .map(|i| {
                                let byte_idx = i / 8;
                                let bit_idx = i % 8;
                                Some(byte_idx < packed.len() && (packed[byte_idx] >> bit_idx) & 1 == 1)
                            })
                            .collect();
                        if bools.len() < expected_row_count {
                            bools.extend(std::iter::repeat(None).take(expected_row_count - bools.len()));
                        } else if bools.len() > expected_row_count {
                            bools.truncate(expected_row_count);
                        }
                        (ArrowDataType::Boolean, Arc::new(BooleanArray::from(bools)))
                    }
                    ColumnData::Binary { offsets, data: bytes } => {
                        use arrow::array::BinaryArray;
                        let count = offsets.len().saturating_sub(1);
                        let binary_data: Vec<Option<&[u8]>> = (0..count)
                            .map(|i| {
                                let start = offsets[i] as usize;
                                let end = offsets[i + 1] as usize;
                                Some(&bytes[start..end] as &[u8])
                            })
                            .collect();
                        if binary_data.len() < expected_row_count {
                            // Need owned data for padding
                            let mut owned: Vec<Option<Vec<u8>>> = binary_data.into_iter()
                                .map(|b| b.map(|s| s.to_vec()))
                                .collect();
                            owned.extend(std::iter::repeat(None).take(expected_row_count - owned.len()));
                            let refs: Vec<Option<&[u8]>> = owned.iter()
                                .map(|o| o.as_ref().map(|v| v.as_slice()))
                                .collect();
                            (ArrowDataType::Binary, Arc::new(BinaryArray::from(refs)))
                        } else if binary_data.len() > expected_row_count {
                            // Truncate to expected row count
                            let truncated: Vec<Option<&[u8]>> = binary_data.into_iter().take(expected_row_count).collect();
                            (ArrowDataType::Binary, Arc::new(BinaryArray::from(truncated)))
                        } else {
                            (ArrowDataType::Binary, Arc::new(BinaryArray::from(binary_data)))
                        }
                    }
                    ColumnData::StringDict { indices, dict_offsets, dict_data } => {
                        // OPTIMIZATION: Use Arrow DictionaryArray to preserve dictionary encoding
                        // This avoids string decoding and allows executor to use indices directly
                        use arrow::array::{DictionaryArray, UInt32Array};
                        use arrow::datatypes::UInt32Type;
                        
                        // Build dictionary values (unique strings)
                        let dict_count = dict_offsets.len().saturating_sub(1);
                        let dict_strings: Vec<Option<&str>> = (0..dict_count)
                            .map(|i| {
                                let start = dict_offsets[i] as usize;
                                let end = dict_offsets[i + 1] as usize;
                                std::str::from_utf8(&dict_data[start..end]).ok()
                            })
                            .collect();
                        let values = StringArray::from(dict_strings);
                        
                        // Convert indices (0 = NULL, 1+ = dict index)
                        // Arrow DictionaryArray uses 0-based indices, NULL is separate
                        // Truncate or pad indices to match expected_row_count
                        let keys: Vec<Option<u32>> = if indices.len() > expected_row_count {
                            indices.iter().take(expected_row_count)
                                .map(|&idx| if idx == 0 { None } else { Some(idx - 1) })
                                .collect()
                        } else if indices.len() < expected_row_count {
                            let mut keys: Vec<Option<u32>> = indices.iter()
                                .map(|&idx| if idx == 0 { None } else { Some(idx - 1) })
                                .collect();
                            keys.extend(std::iter::repeat(None).take(expected_row_count - keys.len()));
                            keys
                        } else {
                            indices.iter()
                                .map(|&idx| if idx == 0 { None } else { Some(idx - 1) })
                                .collect()
                        };
                        let keys_array = UInt32Array::from(keys);
                        
                        // Create DictionaryArray
                        let dict_array = DictionaryArray::<UInt32Type>::try_new(keys_array, Arc::new(values))
                            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e.to_string()))?;
                        
                        (ArrowDataType::Dictionary(Box::new(ArrowDataType::UInt32), Box::new(ArrowDataType::Utf8)), 
                         Arc::new(dict_array) as ArrayRef)
                    }
                };

                fields.push(Field::new(col_name, arrow_dt, true));
                arrays.push(array);
            }
        }

        let schema = Arc::new(Schema::new(fields));
        arrow::record_batch::RecordBatch::try_new(schema, arrays)
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e.to_string()))
    }

    /// Read all columns to Arrow (convenience method)
    pub fn read_all_to_arrow(&self) -> io::Result<arrow::record_batch::RecordBatch> {
        self.read_columns_to_arrow(None, 0, None)
    }

    /// Read columns with predicate pushdown to Arrow
    /// Filters rows at storage level before converting to Arrow
    pub fn read_columns_filtered_to_arrow(
        &self,
        column_names: Option<&[&str]>,
        filter_column: &str,
        filter_op: &str,
        filter_value: f64,
    ) -> io::Result<arrow::record_batch::RecordBatch> {
        use arrow::array::{ArrayRef, Int64Array, Float64Array, StringArray, BooleanArray};
        use arrow::datatypes::{Schema, Field, DataType as ArrowDataType};
        use std::sync::Arc;

        let (col_data, matching_indices) = self.storage.read_columns_filtered(
            column_names, filter_column, filter_op, filter_value
        )?;
        
        if col_data.is_empty() || matching_indices.is_empty() {
            // Return empty batch with proper schema
            let schema = self.schema.read();
            let include_id = column_names.map(|cols| cols.contains(&"_id")).unwrap_or(true);
            let mut fields: Vec<Field> = Vec::new();
            if include_id {
                fields.push(Field::new("_id", ArrowDataType::Int64, false));
            }
            for (name, dt) in schema.iter() {
                if column_names.map(|cols| cols.contains(&name.as_str())).unwrap_or(true) {
                    let arrow_dt = match dt {
                        DataType::Int64 | DataType::Int32 | DataType::Int16 | DataType::Int8 => ArrowDataType::Int64,
                        DataType::Float64 | DataType::Float32 => ArrowDataType::Float64,
                        DataType::String => ArrowDataType::Utf8,
                        DataType::Bool => ArrowDataType::Boolean,
                        _ => ArrowDataType::Utf8,
                    };
                    fields.push(Field::new(name, arrow_dt, true));
                }
            }
            let schema = Arc::new(Schema::new(fields));
            return Ok(arrow::record_batch::RecordBatch::new_empty(schema));
        }

        // Build Arrow arrays from filtered ColumnData
        let schema = self.schema.read();
        let mut fields: Vec<Field> = Vec::new();
        let mut arrays: Vec<ArrayRef> = Vec::new();
        let expected_row_count = matching_indices.len();

        // Include _id column with filtered indices
        let include_id = column_names.map(|cols| cols.contains(&"_id")).unwrap_or(true);
        if include_id {
            // OPTIMIZED: Read only the IDs we need instead of all IDs
            let filtered_ids = self.storage.read_ids_by_indices(&matching_indices)?;
            fields.push(Field::new("_id", ArrowDataType::Int64, false));
            arrays.push(Arc::new(Int64Array::from(filtered_ids)));
        }

        // Determine column order
        let col_order: Vec<String> = if let Some(names) = column_names {
            names.iter().filter(|&s| *s != "_id").map(|s| s.to_string()).collect()
        } else {
            schema.iter().map(|(n, _)| n.clone()).collect()
        };

        for col_name in &col_order {
            if let Some(data) = col_data.get(col_name) {
                let (arrow_dt, array): (ArrowDataType, ArrayRef) = match data {
                    ColumnData::Int64(values) => {
                        (ArrowDataType::Int64, Arc::new(Int64Array::from_iter_values(values.iter().copied())))
                    }
                    ColumnData::Float64(values) => {
                        (ArrowDataType::Float64, Arc::new(Float64Array::from_iter_values(values.iter().copied())))
                    }
                    ColumnData::String { offsets, data: bytes } => {
                        let count = offsets.len().saturating_sub(1);
                        let strings: Vec<Option<&str>> = (0..count)
                            .map(|i| {
                                let start = offsets[i] as usize;
                                let end = offsets[i + 1] as usize;
                                std::str::from_utf8(&bytes[start..end]).ok()
                            })
                            .collect();
                        (ArrowDataType::Utf8, Arc::new(StringArray::from(strings)))
                    }
                    ColumnData::Bool { data: packed, len } => {
                        let bools: Vec<Option<bool>> = (0..*len)
                            .map(|i| {
                                let byte_idx = i / 8;
                                let bit_idx = i % 8;
                                Some(byte_idx < packed.len() && (packed[byte_idx] >> bit_idx) & 1 == 1)
                            })
                            .collect();
                        (ArrowDataType::Boolean, Arc::new(BooleanArray::from(bools)))
                    }
                    ColumnData::Binary { offsets, data: bytes } => {
                        use arrow::array::BinaryArray;
                        let count = offsets.len().saturating_sub(1);
                        let binary_data: Vec<Option<&[u8]>> = (0..count)
                            .map(|i| {
                                let start = offsets[i] as usize;
                                let end = offsets[i + 1] as usize;
                                Some(&bytes[start..end] as &[u8])
                            })
                            .collect();
                        (ArrowDataType::Binary, Arc::new(BinaryArray::from(binary_data)))
                    }
                    ColumnData::StringDict { indices, dict_offsets, dict_data } => {
                        let strings: Vec<Option<String>> = indices.iter()
                            .map(|&idx| {
                                if idx == 0 { None } else {
                                    let dict_idx = (idx - 1) as usize;
                                    if dict_idx + 1 < dict_offsets.len() {
                                        let start = dict_offsets[dict_idx] as usize;
                                        let end = dict_offsets[dict_idx + 1] as usize;
                                        std::str::from_utf8(&dict_data[start..end]).ok().map(|s| s.to_string())
                                    } else { None }
                                }
                            })
                            .collect();
                        (ArrowDataType::Utf8, Arc::new(StringArray::from(strings)))
                    }
                };

                fields.push(Field::new(col_name, arrow_dt, true));
                arrays.push(array);
            }
        }

        let schema = Arc::new(Schema::new(fields));
        arrow::record_batch::RecordBatch::try_new(schema, arrays)
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e.to_string()))
    }

    /// Read columns for specific row indices to Arrow (for late materialization)
    /// Only reads the specified rows from disk, reducing I/O for filtered queries
    pub fn read_columns_by_indices_to_arrow(
        &self,
        row_indices: &[usize],
    ) -> io::Result<arrow::record_batch::RecordBatch> {
        use arrow::array::{ArrayRef, Int64Array, Float64Array, StringArray, BooleanArray};
        use arrow::datatypes::{Schema, Field, DataType as ArrowDataType};
        use std::sync::Arc;

        if row_indices.is_empty() {
            return self.read_columns_to_arrow(None, 0, Some(0));
        }

        let schema = self.schema.read();
        let mut fields: Vec<Field> = Vec::new();
        let mut arrays: Vec<ArrayRef> = Vec::new();

        // Include _id column
        // OPTIMIZED: Read only the IDs we need instead of all IDs
        let filtered_ids = self.storage.read_ids_by_indices(row_indices)?;
        fields.push(Field::new("_id", ArrowDataType::Int64, false));
        arrays.push(Arc::new(Int64Array::from(filtered_ids)));

        // Read each column for the specified row indices
        for (col_name, _dt) in schema.iter() {
            let col_data = self.storage.read_column_by_indices(col_name, row_indices)?;
            
            let (arrow_dt, array): (ArrowDataType, ArrayRef) = match col_data {
                ColumnData::Int64(values) => {
                    (ArrowDataType::Int64, Arc::new(Int64Array::from(values)))
                }
                ColumnData::Float64(values) => {
                    (ArrowDataType::Float64, Arc::new(Float64Array::from(values)))
                }
                ColumnData::String { offsets, data: bytes } => {
                    let count = offsets.len().saturating_sub(1);
                    let strings: Vec<Option<&str>> = (0..count)
                        .map(|i| {
                            let start = offsets[i] as usize;
                            let end = offsets[i + 1] as usize;
                            std::str::from_utf8(&bytes[start..end]).ok()
                        })
                        .collect();
                    (ArrowDataType::Utf8, Arc::new(StringArray::from(strings)))
                }
                ColumnData::Bool { data: packed, len } => {
                    let bools: Vec<Option<bool>> = (0..len)
                        .map(|i| {
                            let byte_idx = i / 8;
                            let bit_idx = i % 8;
                            Some(byte_idx < packed.len() && (packed[byte_idx] >> bit_idx) & 1 == 1)
                        })
                        .collect();
                    (ArrowDataType::Boolean, Arc::new(BooleanArray::from(bools)))
                }
                ColumnData::Binary { offsets, data: bytes } => {
                    use arrow::array::BinaryArray;
                    let count = offsets.len().saturating_sub(1);
                    let binary_data: Vec<Option<&[u8]>> = (0..count)
                        .map(|i| {
                            let start = offsets[i] as usize;
                            let end = offsets[i + 1] as usize;
                            Some(&bytes[start..end] as &[u8])
                        })
                        .collect();
                    (ArrowDataType::Binary, Arc::new(BinaryArray::from(binary_data)))
                }
                ColumnData::StringDict { indices, dict_offsets, dict_data } => {
                    let strings: Vec<Option<String>> = indices.iter()
                        .map(|&idx| {
                            if idx == 0 { None } else {
                                let dict_idx = (idx - 1) as usize;
                                if dict_idx + 1 < dict_offsets.len() {
                                    let start = dict_offsets[dict_idx] as usize;
                                    let end = dict_offsets[dict_idx + 1] as usize;
                                    std::str::from_utf8(&dict_data[start..end]).ok().map(|s| s.to_string())
                                } else { None }
                            }
                        })
                        .collect();
                    (ArrowDataType::Utf8, Arc::new(StringArray::from(strings)))
                }
            };
            
            fields.push(Field::new(col_name, arrow_dt, true));
            arrays.push(array);
        }

        let schema = Arc::new(Schema::new(fields));
        arrow::record_batch::RecordBatch::try_new(schema, arrays)
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e.to_string()))
    }

    /// OPTIMIZED: Read a single row by ID using O(1) index lookup
    /// Much faster than WHERE _id = X which scans all data
    pub fn read_row_by_id_to_arrow(&self, id: u64) -> io::Result<Option<arrow::record_batch::RecordBatch>> {
        use arrow::array::{ArrayRef, Int64Array, Float64Array, StringArray, BooleanArray, NullArray};
        use arrow::datatypes::{Schema, Field, DataType as ArrowDataType};
        use std::sync::Arc;
        use crate::data::DataType;

        let row_data = match self.storage.read_row_by_id(id, None)? {
            Some(data) => data,
            None => return Ok(None),
        };

        let schema = self.schema.read();
        let mut fields: Vec<Field> = Vec::new();
        let mut arrays: Vec<ArrayRef> = Vec::new();

        // Add _id column first
        fields.push(Field::new("_id", ArrowDataType::Int64, false));
        arrays.push(Arc::new(Int64Array::from(vec![id as i64])));

        // Add other columns in schema order - ensure all have length 1
        for (col_name, dt) in schema.iter() {
            let (arrow_dt, array): (ArrowDataType, ArrayRef) = if let Some(col_data) = row_data.get(col_name) {
                match col_data {
                    ColumnData::Int64(values) if !values.is_empty() => {
                        (ArrowDataType::Int64, Arc::new(Int64Array::from(vec![values[0]])))
                    }
                    ColumnData::Float64(values) if !values.is_empty() => {
                        (ArrowDataType::Float64, Arc::new(Float64Array::from(vec![values[0]])))
                    }
                    ColumnData::String { offsets, data: bytes } if offsets.len() > 1 => {
                        let start = offsets[0] as usize;
                        let end = offsets[1] as usize;
                        let s = std::str::from_utf8(&bytes[start..end]).ok();
                        (ArrowDataType::Utf8, Arc::new(StringArray::from(vec![s])))
                    }
                    ColumnData::Bool { data: packed, len } if *len > 0 => {
                        let val = !packed.is_empty() && (packed[0] & 1) == 1;
                        (ArrowDataType::Boolean, Arc::new(BooleanArray::from(vec![Some(val)])))
                    }
                    ColumnData::Binary { offsets, data: bytes } if offsets.len() > 1 => {
                        use arrow::array::BinaryArray;
                        let start = offsets[0] as usize;
                        let end = offsets[1] as usize;
                        (ArrowDataType::Binary, Arc::new(BinaryArray::from(vec![Some(&bytes[start..end] as &[u8])])))
                    }
                    ColumnData::StringDict { indices, dict_offsets, dict_data } if !indices.is_empty() => {
                        let idx = indices[0];
                        let s = if idx == 0 { None } else {
                            let dict_idx = (idx - 1) as usize;
                            if dict_idx + 1 < dict_offsets.len() {
                                let start = dict_offsets[dict_idx] as usize;
                                let end = dict_offsets[dict_idx + 1] as usize;
                                std::str::from_utf8(&dict_data[start..end]).ok().map(|s| s.to_string())
                            } else { None }
                        };
                        (ArrowDataType::Utf8, Arc::new(StringArray::from(vec![s])))
                    }
                    // Empty column data - return typed null
                    _ => Self::create_typed_null_array(dt),
                }
            } else {
                // Column not in row_data - return typed null value
                Self::create_typed_null_array(dt)
            };
            fields.push(Field::new(col_name, arrow_dt, true));
            arrays.push(array);
        }

        let schema = Arc::new(Schema::new(fields));
        let batch = arrow::record_batch::RecordBatch::try_new(schema, arrays)
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e.to_string()))?;
        Ok(Some(batch))
    }

    /// Create a typed null array with a single null value
    fn create_typed_null_array(dt: &crate::data::DataType) -> (arrow::datatypes::DataType, arrow::array::ArrayRef) {
        use arrow::array::{ArrayRef, Int64Array, Float64Array, StringArray, BooleanArray};
        use arrow::datatypes::DataType as ArrowDataType;
        use std::sync::Arc;
        use crate::data::DataType;

        match dt {
            DataType::Int64 | DataType::Int32 | DataType::Int16 | DataType::Int8 |
            DataType::UInt64 | DataType::UInt32 | DataType::UInt16 | DataType::UInt8 => {
                (ArrowDataType::Int64, Arc::new(Int64Array::from(vec![None as Option<i64>])) as ArrayRef)
            }
            DataType::Float64 | DataType::Float32 => {
                (ArrowDataType::Float64, Arc::new(Float64Array::from(vec![None as Option<f64>])) as ArrayRef)
            }
            DataType::Bool => {
                (ArrowDataType::Boolean, Arc::new(BooleanArray::from(vec![None as Option<bool>])) as ArrayRef)
            }
            DataType::String | _ => {
                (ArrowDataType::Utf8, Arc::new(StringArray::from(vec![None as Option<&str>])) as ArrayRef)
            }
        }
    }

    /// Read columns with STRING predicate pushdown to Arrow
    /// Filters rows at storage level for string equality (much faster than post-filtering)
    pub fn read_columns_filtered_string_to_arrow(
        &self,
        column_names: Option<&[&str]>,
        filter_column: &str,
        filter_value: &str,
        filter_eq: bool,
    ) -> io::Result<arrow::record_batch::RecordBatch> {
        use arrow::array::{ArrayRef, Int64Array, Float64Array, StringArray, BooleanArray};
        use arrow::datatypes::{Schema, Field, DataType as ArrowDataType};
        use std::sync::Arc;

        let (col_data, matching_indices) = self.storage.read_columns_filtered_string(
            column_names, filter_column, filter_value, filter_eq
        )?;
        
        if col_data.is_empty() || matching_indices.is_empty() {
            // Return empty batch with proper schema
            return self.read_columns_to_arrow(column_names, 0, Some(0));
        }

        let schema = self.schema.read();
        let mut fields: Vec<Field> = Vec::new();
        let mut arrays: Vec<ArrayRef> = Vec::new();

        // Include _id column with filtered indices
        let include_id = column_names.map(|cols| cols.contains(&"_id")).unwrap_or(true);
        if include_id {
            // OPTIMIZED: Read only the IDs we need instead of all IDs
            let filtered_ids = self.storage.read_ids_by_indices(&matching_indices)?;
            fields.push(Field::new("_id", ArrowDataType::Int64, false));
            arrays.push(Arc::new(Int64Array::from(filtered_ids)));
        }

        // Determine column order
        let col_order: Vec<String> = if let Some(names) = column_names {
            names.iter().filter(|&s| *s != "_id").map(|s| s.to_string()).collect()
        } else {
            schema.iter().map(|(n, _)| n.clone()).collect()
        };

        for col_name in &col_order {
            if let Some(data) = col_data.get(col_name) {
                let (arrow_dt, array): (ArrowDataType, ArrayRef) = match data {
                    ColumnData::Int64(values) => {
                        (ArrowDataType::Int64, Arc::new(Int64Array::from_iter_values(values.iter().copied())))
                    }
                    ColumnData::Float64(values) => {
                        (ArrowDataType::Float64, Arc::new(Float64Array::from_iter_values(values.iter().copied())))
                    }
                    ColumnData::String { offsets, data: bytes } => {
                        let count = offsets.len().saturating_sub(1);
                        let strings: Vec<Option<&str>> = (0..count)
                            .map(|i| {
                                let start = offsets[i] as usize;
                                let end = offsets[i + 1] as usize;
                                std::str::from_utf8(&bytes[start..end]).ok()
                            })
                            .collect();
                        (ArrowDataType::Utf8, Arc::new(StringArray::from(strings)))
                    }
                    ColumnData::Bool { data: packed, len } => {
                        let bools: Vec<Option<bool>> = (0..*len)
                            .map(|i| {
                                let byte_idx = i / 8;
                                let bit_idx = i % 8;
                                Some(byte_idx < packed.len() && (packed[byte_idx] >> bit_idx) & 1 == 1)
                            })
                            .collect();
                        (ArrowDataType::Boolean, Arc::new(BooleanArray::from(bools)))
                    }
                    ColumnData::Binary { offsets, data: bytes } => {
                        use arrow::array::BinaryArray;
                        let count = offsets.len().saturating_sub(1);
                        let binary_data: Vec<Option<&[u8]>> = (0..count)
                            .map(|i| {
                                let start = offsets[i] as usize;
                                let end = offsets[i + 1] as usize;
                                Some(&bytes[start..end] as &[u8])
                            })
                            .collect();
                        (ArrowDataType::Binary, Arc::new(BinaryArray::from(binary_data)))
                    }
                    ColumnData::StringDict { indices, dict_offsets, dict_data } => {
                        let strings: Vec<Option<String>> = indices.iter()
                            .map(|&idx| {
                                if idx == 0 { None } else {
                                    let dict_idx = (idx - 1) as usize;
                                    if dict_idx + 1 < dict_offsets.len() {
                                        let start = dict_offsets[dict_idx] as usize;
                                        let end = dict_offsets[dict_idx + 1] as usize;
                                        std::str::from_utf8(&dict_data[start..end]).ok().map(|s| s.to_string())
                                    } else { None }
                                }
                            })
                            .collect();
                        (ArrowDataType::Utf8, Arc::new(StringArray::from(strings)))
                    }
                };

                fields.push(Field::new(col_name, arrow_dt, true));
                arrays.push(array);
            }
        }

        let schema = Arc::new(Schema::new(fields));
        arrow::record_batch::RecordBatch::try_new(schema, arrays)
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e.to_string()))
    }
    
    /// Read columns with STRING predicate pushdown and LIMIT early termination
    /// Much faster for queries like SELECT * WHERE col = 'value' LIMIT n
    pub fn read_columns_filtered_string_with_limit_to_arrow(
        &self,
        column_names: Option<&[&str]>,
        filter_column: &str,
        filter_value: &str,
        filter_eq: bool,
        limit: usize,
        offset: usize,
    ) -> io::Result<arrow::record_batch::RecordBatch> {
        use arrow::array::{ArrayRef, Int64Array, Float64Array, StringArray, BooleanArray};
        use arrow::datatypes::{Schema, Field, DataType as ArrowDataType};
        use std::sync::Arc;

        let (col_data, matching_indices) = self.storage.read_columns_filtered_string_with_limit(
            column_names, filter_column, filter_value, filter_eq, limit, offset
        )?;
        
        if col_data.is_empty() || matching_indices.is_empty() {
            return self.read_columns_to_arrow(column_names, 0, Some(0));
        }

        let schema = self.schema.read();
        let mut fields: Vec<Field> = Vec::new();
        let mut arrays: Vec<ArrayRef> = Vec::new();

        let include_id = column_names.map(|cols| cols.contains(&"_id")).unwrap_or(true);
        if include_id {
            // OPTIMIZED: Read only the IDs we need instead of all IDs
            let filtered_ids = self.storage.read_ids_by_indices(&matching_indices)?;
            fields.push(Field::new("_id", ArrowDataType::Int64, false));
            arrays.push(Arc::new(Int64Array::from(filtered_ids)));
        }

        let col_order: Vec<String> = if let Some(names) = column_names {
            names.iter().filter(|&s| *s != "_id").map(|s| s.to_string()).collect()
        } else {
            schema.iter().map(|(n, _)| n.clone()).collect()
        };

        for col_name in &col_order {
            if let Some(data) = col_data.get(col_name) {
                let (arrow_dt, array): (ArrowDataType, ArrayRef) = match data {
                    ColumnData::Int64(values) => {
                        (ArrowDataType::Int64, Arc::new(Int64Array::from_iter_values(values.iter().copied())))
                    }
                    ColumnData::Float64(values) => {
                        (ArrowDataType::Float64, Arc::new(Float64Array::from_iter_values(values.iter().copied())))
                    }
                    ColumnData::String { offsets, data: bytes } => {
                        let count = offsets.len().saturating_sub(1);
                        let strings: Vec<Option<&str>> = (0..count)
                            .map(|i| {
                                let start = offsets[i] as usize;
                                let end = offsets[i + 1] as usize;
                                std::str::from_utf8(&bytes[start..end]).ok()
                            })
                            .collect();
                        (ArrowDataType::Utf8, Arc::new(StringArray::from(strings)))
                    }
                    ColumnData::Bool { data: packed, len } => {
                        let bools: Vec<Option<bool>> = (0..*len)
                            .map(|i| {
                                let byte_idx = i / 8;
                                let bit_idx = i % 8;
                                Some(byte_idx < packed.len() && (packed[byte_idx] >> bit_idx) & 1 == 1)
                            })
                            .collect();
                        (ArrowDataType::Boolean, Arc::new(BooleanArray::from(bools)))
                    }
                    ColumnData::Binary { offsets, data: bytes } => {
                        use arrow::array::BinaryArray;
                        let count = offsets.len().saturating_sub(1);
                        let binary_data: Vec<Option<&[u8]>> = (0..count)
                            .map(|i| {
                                let start = offsets[i] as usize;
                                let end = offsets[i + 1] as usize;
                                Some(&bytes[start..end] as &[u8])
                            })
                            .collect();
                        (ArrowDataType::Binary, Arc::new(BinaryArray::from(binary_data)))
                    }
                    ColumnData::StringDict { indices, dict_offsets, dict_data } => {
                        // OPTIMIZED: Use Arrow DictionaryArray to avoid string allocations
                        use arrow::array::{DictionaryArray, UInt32Array};
                        use arrow::datatypes::UInt32Type;
                        
                        // Build dictionary values (unique strings) - use &str references
                        let dict_count = dict_offsets.len().saturating_sub(1);
                        let dict_strings: Vec<Option<&str>> = (0..dict_count)
                            .map(|i| {
                                let start = dict_offsets[i] as usize;
                                let end = dict_offsets[i + 1] as usize;
                                std::str::from_utf8(&dict_data[start..end]).ok()
                            })
                            .collect();
                        let values = StringArray::from(dict_strings);
                        
                        // Convert indices (0 = NULL, 1+ = dict index)
                        let keys: Vec<Option<u32>> = indices.iter()
                            .map(|&idx| if idx == 0 { None } else { Some(idx - 1) })
                            .collect();
                        let keys_array = UInt32Array::from(keys);
                        
                        let dict_array = DictionaryArray::<UInt32Type>::try_new(keys_array, Arc::new(values))
                            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e.to_string()))?;
                        
                        (ArrowDataType::Dictionary(Box::new(ArrowDataType::UInt32), Box::new(ArrowDataType::Utf8)), 
                         Arc::new(dict_array) as ArrayRef)
                    }
                };
                fields.push(Field::new(col_name, arrow_dt, true));
                arrays.push(array);
            }
        }

        let schema = Arc::new(Schema::new(fields));
        arrow::record_batch::RecordBatch::try_new(schema, arrays)
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e.to_string()))
    }

    /// Read columns with numeric RANGE predicate pushdown and LIMIT early termination
    /// Much faster for queries like SELECT * WHERE col BETWEEN low AND high LIMIT n
    pub fn read_columns_filtered_range_with_limit_to_arrow(
        &self,
        column_names: Option<&[&str]>,
        filter_column: &str,
        low: f64,
        high: f64,
        limit: usize,
        offset: usize,
    ) -> io::Result<arrow::record_batch::RecordBatch> {
        use arrow::array::{ArrayRef, Int64Array, Float64Array, StringArray, BooleanArray};
        use arrow::datatypes::{Schema, Field, DataType as ArrowDataType};
        use std::sync::Arc;

        let (col_data, matching_indices) = self.storage.read_columns_filtered_range_with_limit(
            column_names, filter_column, low, high, limit, offset
        )?;
        
        if col_data.is_empty() || matching_indices.is_empty() {
            return self.read_columns_to_arrow(column_names, 0, Some(0));
        }

        let schema = self.schema.read();
        let mut fields: Vec<Field> = Vec::new();
        let mut arrays: Vec<ArrayRef> = Vec::new();

        let include_id = column_names.map(|cols| cols.contains(&"_id")).unwrap_or(true);
        if include_id {
            let filtered_ids = self.storage.read_ids_by_indices(&matching_indices)?;
            fields.push(Field::new("_id", ArrowDataType::Int64, false));
            arrays.push(Arc::new(Int64Array::from(filtered_ids)));
        }

        let col_order: Vec<String> = if let Some(names) = column_names {
            names.iter().filter(|&s| *s != "_id").map(|s| s.to_string()).collect()
        } else {
            schema.iter().map(|(n, _)| n.clone()).collect()
        };

        for col_name in &col_order {
            if let Some(data) = col_data.get(col_name) {
                let (arrow_dt, array): (ArrowDataType, ArrayRef) = match data {
                    ColumnData::Int64(values) => {
                        (ArrowDataType::Int64, Arc::new(Int64Array::from_iter_values(values.iter().copied())))
                    }
                    ColumnData::Float64(values) => {
                        (ArrowDataType::Float64, Arc::new(Float64Array::from_iter_values(values.iter().copied())))
                    }
                    ColumnData::String { offsets, data: bytes } => {
                        let count = offsets.len().saturating_sub(1);
                        let strings: Vec<Option<&str>> = (0..count)
                            .map(|i| {
                                let start = offsets[i] as usize;
                                let end = offsets[i + 1] as usize;
                                std::str::from_utf8(&bytes[start..end]).ok()
                            })
                            .collect();
                        (ArrowDataType::Utf8, Arc::new(StringArray::from(strings)))
                    }
                    ColumnData::Bool { data: packed, len } => {
                        let bools: Vec<Option<bool>> = (0..*len)
                            .map(|i| {
                                let byte_idx = i / 8;
                                let bit_idx = i % 8;
                                Some(byte_idx < packed.len() && (packed[byte_idx] >> bit_idx) & 1 == 1)
                            })
                            .collect();
                        (ArrowDataType::Boolean, Arc::new(BooleanArray::from(bools)))
                    }
                    ColumnData::Binary { offsets, data: bytes } => {
                        use arrow::array::BinaryArray;
                        let count = offsets.len().saturating_sub(1);
                        let binary_data: Vec<Option<&[u8]>> = (0..count)
                            .map(|i| {
                                let start = offsets[i] as usize;
                                let end = offsets[i + 1] as usize;
                                Some(&bytes[start..end])
                            })
                            .collect();
                        (ArrowDataType::Binary, Arc::new(BinaryArray::from(binary_data)))
                    }
                    ColumnData::StringDict { indices, dict_offsets, dict_data } => {
                        // OPTIMIZED: Use Arrow DictionaryArray to avoid string allocations
                        use arrow::array::{DictionaryArray, UInt32Array};
                        use arrow::datatypes::UInt32Type;
                        
                        // Build dictionary values (unique strings) - use &str references
                        let dict_count = dict_offsets.len().saturating_sub(1);
                        let dict_strings: Vec<Option<&str>> = (0..dict_count)
                            .map(|i| {
                                let start = dict_offsets[i] as usize;
                                let end = dict_offsets[i + 1] as usize;
                                std::str::from_utf8(&dict_data[start..end]).ok()
                            })
                            .collect();
                        let values = StringArray::from(dict_strings);
                        
                        // Convert indices (0 = NULL, 1+ = dict index)
                        let keys: Vec<Option<u32>> = indices.iter()
                            .map(|&idx| if idx == 0 { None } else { Some(idx - 1) })
                            .collect();
                        let keys_array = UInt32Array::from(keys);
                        
                        let dict_array = DictionaryArray::<UInt32Type>::try_new(keys_array, Arc::new(values))
                            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e.to_string()))?;
                        
                        (ArrowDataType::Dictionary(Box::new(ArrowDataType::UInt32), Box::new(ArrowDataType::Utf8)), 
                         Arc::new(dict_array) as ArrayRef)
                    }
                };
                fields.push(Field::new(col_name, arrow_dt, true));
                arrays.push(array);
            }
        }

        let schema = Arc::new(Schema::new(fields));
        arrow::record_batch::RecordBatch::try_new(schema, arrays)
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e.to_string()))
    }

    /// Get underlying storage for direct access
    pub fn storage(&self) -> &OnDemandStorage {
        &self.storage
    }

    /// Read columns with combined STRING + NUMERIC filter and LIMIT early termination
    /// Optimized for SELECT * WHERE string_col = 'value' AND numeric_col > N LIMIT n
    pub fn read_columns_filtered_string_numeric_with_limit_to_arrow(
        &self,
        column_names: Option<&[&str]>,
        string_column: &str,
        string_value: &str,
        numeric_column: &str,
        numeric_op: &str,
        numeric_value: f64,
        limit: usize,
        offset: usize,
    ) -> io::Result<arrow::record_batch::RecordBatch> {
        use arrow::array::{ArrayRef, Int64Array, Float64Array, StringArray, BooleanArray};
        use arrow::datatypes::{Schema, Field, DataType as ArrowDataType};
        use std::sync::Arc;

        let (col_data, matching_indices) = self.storage.read_columns_filtered_string_numeric_with_limit(
            column_names, string_column, string_value, numeric_column, numeric_op, numeric_value, limit, offset
        )?;
        
        if col_data.is_empty() || matching_indices.is_empty() {
            return self.read_columns_to_arrow(column_names, 0, Some(0));
        }

        let schema = self.schema.read();
        let mut fields: Vec<Field> = Vec::new();
        let mut arrays: Vec<ArrayRef> = Vec::new();

        let include_id = column_names.map(|cols| cols.contains(&"_id")).unwrap_or(true);
        if include_id {
            let filtered_ids = self.storage.read_ids_by_indices(&matching_indices)?;
            fields.push(Field::new("_id", ArrowDataType::Int64, false));
            arrays.push(Arc::new(Int64Array::from(filtered_ids)));
        }

        let col_order: Vec<String> = if let Some(names) = column_names {
            names.iter().filter(|&s| *s != "_id").map(|s| s.to_string()).collect()
        } else {
            schema.iter().map(|(n, _)| n.clone()).collect()
        };

        for col_name in &col_order {
            if let Some(data) = col_data.get(col_name) {
                let (arrow_dt, array): (ArrowDataType, ArrayRef) = match data {
                    ColumnData::Int64(values) => {
                        (ArrowDataType::Int64, Arc::new(Int64Array::from_iter_values(values.iter().copied())))
                    }
                    ColumnData::Float64(values) => {
                        (ArrowDataType::Float64, Arc::new(Float64Array::from_iter_values(values.iter().copied())))
                    }
                    ColumnData::String { offsets, data: bytes } => {
                        let count = offsets.len().saturating_sub(1);
                        let strings: Vec<Option<&str>> = (0..count)
                            .map(|i| {
                                let start = offsets[i] as usize;
                                let end = offsets[i + 1] as usize;
                                std::str::from_utf8(&bytes[start..end]).ok()
                            })
                            .collect();
                        (ArrowDataType::Utf8, Arc::new(StringArray::from(strings)))
                    }
                    ColumnData::Bool { data: packed, len } => {
                        let bools: Vec<Option<bool>> = (0..*len)
                            .map(|i| {
                                let byte_idx = i / 8;
                                let bit_idx = i % 8;
                                Some(byte_idx < packed.len() && (packed[byte_idx] >> bit_idx) & 1 == 1)
                            })
                            .collect();
                        (ArrowDataType::Boolean, Arc::new(BooleanArray::from(bools)))
                    }
                    ColumnData::Binary { offsets, data: bytes } => {
                        use arrow::array::BinaryArray;
                        let count = offsets.len().saturating_sub(1);
                        let binary_data: Vec<Option<&[u8]>> = (0..count)
                            .map(|i| {
                                let start = offsets[i] as usize;
                                let end = offsets[i + 1] as usize;
                                Some(&bytes[start..end])
                            })
                            .collect();
                        (ArrowDataType::Binary, Arc::new(BinaryArray::from(binary_data)))
                    }
                    ColumnData::StringDict { indices, dict_offsets, dict_data } => {
                        // OPTIMIZED: Use Arrow DictionaryArray
                        use arrow::array::{DictionaryArray, UInt32Array};
                        use arrow::datatypes::UInt32Type;
                        
                        let dict_count = dict_offsets.len().saturating_sub(1);
                        let dict_strings: Vec<Option<&str>> = (0..dict_count)
                            .map(|i| {
                                let start = dict_offsets[i] as usize;
                                let end = dict_offsets[i + 1] as usize;
                                std::str::from_utf8(&dict_data[start..end]).ok()
                            })
                            .collect();
                        let values = StringArray::from(dict_strings);
                        
                        let keys: Vec<Option<u32>> = indices.iter()
                            .map(|&idx| if idx == 0 { None } else { Some(idx - 1) })
                            .collect();
                        let keys_array = UInt32Array::from(keys);
                        
                        let dict_array = DictionaryArray::<UInt32Type>::try_new(keys_array, Arc::new(values))
                            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e.to_string()))?;
                        
                        (ArrowDataType::Dictionary(Box::new(ArrowDataType::UInt32), Box::new(ArrowDataType::Utf8)), 
                         Arc::new(dict_array) as ArrayRef)
                    }
                };
                fields.push(Field::new(col_name, arrow_dt, true));
                arrays.push(array);
            }
        }

        let schema = Arc::new(Schema::new(fields));
        arrow::record_batch::RecordBatch::try_new(schema, arrays)
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e.to_string()))
    }

    /// Execute Complex (Filter+Group+Order) query with single-pass optimization
    /// This is the key optimization for queries like:
    /// SELECT region, SUM(value) FROM table WHERE status = 'active' GROUP BY region ORDER BY total DESC LIMIT 5
    pub fn execute_filter_group_order(
        &self,
        filter_col: &str,
        filter_val: &str,
        group_col: &str,
        agg_col: Option<&str>,
        agg_func: crate::query::AggregateFunc,
        _order_col: &str,
        descending: bool,
        limit: usize,
        offset: usize,
    ) -> io::Result<Option<RecordBatch>> {
        use crate::query::AggregateFunc;
        use arrow::array::{Int64Array, Float64Array, StringArray, DictionaryArray, UInt32Array};
        use arrow::datatypes::UInt32Type;
        use std::collections::BinaryHeap;
        use std::cmp::Ordering;

        // This optimization requires dictionary-encoded columns for maximum performance
        // Get filter column info
        let schema_guard = self.schema.read();
        let filter_idx = schema_guard.iter().position(|(name, _)| name == filter_col);
        let group_idx = schema_guard.iter().position(|(name, _)| name == group_col);
        
        if filter_idx.is_none() || group_idx.is_none() {
            return Ok(None);
        }

        // For now, delegate to the OnDemandStorage implementation
        let result = self.storage.execute_filter_group_order(
            filter_col,
            filter_val,
            group_col,
            agg_col,
            agg_func,
            descending,
            limit,
            offset,
        )?;

        Ok(result)
    }
}

// ============================================================================
// Incremental Storage Backend - Fast Writes with WAL
// ============================================================================

use crate::storage::incremental::IncrementalStorage;
use crate::storage::on_demand::ColumnValue as OnDemandColumnValue;

/// High-performance storage backend with incremental writes
/// 
/// Uses WAL (Write-Ahead Log) for fast append-only writes:
/// - Writes append to WAL file - O(1) time
/// - Reads merge main file + WAL transparently
/// - Background compaction merges WAL into main file
/// 
/// This provides significantly faster write performance compared to
/// TableStorageBackend which rewrites the entire file on each save.
pub struct IncrementalStorageBackend {
    storage: IncrementalStorage,
    /// Schema mapping (column_name -> DataType)
    schema: RwLock<Vec<(String, DataType)>>,
}

impl IncrementalStorageBackend {
    /// Create a new incremental storage
    pub fn create(path: &Path) -> io::Result<Self> {
        let storage = IncrementalStorage::create(path)?;
        Ok(Self {
            storage,
            schema: RwLock::new(Vec::new()),
        })
    }

    /// Open existing incremental storage
    pub fn open(path: &Path) -> io::Result<Self> {
        let storage = IncrementalStorage::open(path)?;
        let schema: Vec<(String, DataType)> = storage.get_schema()
            .into_iter()
            .map(|(name, ct)| (name, column_type_to_datatype(ct)))
            .collect();
        
        Ok(Self {
            storage,
            schema: RwLock::new(schema),
        })
    }

    /// Open or create
    pub fn open_or_create(path: &Path) -> io::Result<Self> {
        if path.exists() {
            Self::open(path)
        } else {
            Self::create(path)
        }
    }

    /// Get row count
    pub fn row_count(&self) -> u64 {
        self.storage.row_count()
    }

    /// Get schema
    pub fn get_schema(&self) -> Vec<(String, DataType)> {
        self.schema.read().clone()
    }

    /// Insert rows - FAST incremental write
    pub fn insert_rows(&self, rows: &[HashMap<String, Value>]) -> io::Result<Vec<u64>> {
        if rows.is_empty() {
            return Ok(Vec::new());
        }

        // Convert Value to ColumnValue
        let converted: Vec<HashMap<String, OnDemandColumnValue>> = rows
            .iter()
            .map(|row| {
                row.iter()
                    .map(|(k, v)| {
                        let cv = match v {
                            Value::Int64(i) => OnDemandColumnValue::Int64(*i),
                            Value::Int32(i) => OnDemandColumnValue::Int64(*i as i64),
                            Value::Float64(f) => OnDemandColumnValue::Float64(*f),
                            Value::Float32(f) => OnDemandColumnValue::Float64(*f as f64),
                            Value::String(s) => OnDemandColumnValue::String(s.clone()),
                            Value::Bool(b) => OnDemandColumnValue::Bool(*b),
                            Value::Binary(b) => OnDemandColumnValue::Binary(b.clone()),
                            Value::Null => OnDemandColumnValue::Null,
                            _ => OnDemandColumnValue::String(serde_json::to_string(v).unwrap_or_default()),
                        };
                        (k.clone(), cv)
                    })
                    .collect()
            })
            .collect();

        // Insert into storage (fast WAL append)
        let ids = self.storage.insert_rows(&converted)?;

        // Update schema if new columns
        {
            let mut schema = self.schema.write();
            if let Some(row) = rows.first() {
                for (k, v) in row {
                    if k != "_id" && !schema.iter().any(|(n, _)| n == k) {
                        schema.push((k.clone(), v.data_type()));
                    }
                }
            }
        }

        Ok(ids)
    }

    /// Delete row by ID
    pub fn delete(&self, id: u64) -> io::Result<bool> {
        self.storage.delete(id)
    }

    /// Save/compact (merge WAL into main file)
    pub fn save(&self) -> io::Result<()> {
        self.storage.save()
    }

    /// Flush WAL to disk (without compaction)
    pub fn flush(&self) -> io::Result<()> {
        self.storage.flush()
    }

    /// Check if compaction is needed
    pub fn needs_compaction(&self) -> bool {
        self.storage.needs_compaction()
    }

    /// Compact WAL into main file
    pub fn compact(&self) -> io::Result<()> {
        self.storage.compact()
    }

    /// Get WAL record count
    pub fn wal_record_count(&self) -> usize {
        self.storage.wal_record_count()
    }

    /// Close storage
    pub fn close(&self) -> io::Result<()> {
        self.storage.close()
    }
}

// ============================================================================
// Multi-Table Storage Manager
// ============================================================================

/// Manages multiple tables with lazy loading
pub struct StorageManager {
    base_dir: PathBuf,
    /// Table backends (table_name -> backend)
    tables: RwLock<HashMap<String, Arc<TableStorageBackend>>>,
    /// Current table name
    current_table: RwLock<String>,
}

impl StorageManager {
    /// Create or open a storage manager
    pub fn open_or_create(base_path: &Path) -> io::Result<Self> {
        let base_dir = base_path.parent()
            .map(|p| p.to_path_buf())
            .unwrap_or_else(|| PathBuf::from("."));
        
        let mut tables = HashMap::new();
        
        // Check if main file exists
        if base_path.exists() {
            // Load existing table
            let backend = TableStorageBackend::open(base_path)?;
            let name = backend.metadata().name;
            tables.insert(name.clone(), Arc::new(backend));
        }
        
        Ok(Self {
            base_dir,
            tables: RwLock::new(tables),
            current_table: RwLock::new("default".to_string()),
        })
    }

    /// Get or create a table
    pub fn get_or_create_table(&self, name: &str) -> io::Result<Arc<TableStorageBackend>> {
        // Check if already loaded
        if let Some(backend) = self.tables.read().get(name).cloned() {
            return Ok(backend);
        }

        // Create new table
        let path = self.base_dir.join(format!("{}.apex", name));
        let backend = Arc::new(TableStorageBackend::open_or_create(&path)?);
        
        self.tables.write().insert(name.to_string(), backend.clone());
        
        Ok(backend)
    }

    /// Get current table
    pub fn current_table(&self) -> io::Result<Arc<TableStorageBackend>> {
        let name = self.current_table.read().clone();
        self.get_or_create_table(&name)
    }

    /// Set current table
    pub fn set_current_table(&self, name: &str) {
        *self.current_table.write() = name.to_string();
    }

    /// List all tables
    pub fn list_tables(&self) -> Vec<String> {
        self.tables.read().keys().cloned().collect()
    }

    /// Save all tables
    pub fn save_all(&self) -> io::Result<()> {
        for (_, backend) in self.tables.read().iter() {
            backend.save()?;
        }
        Ok(())
    }

    /// Release memory from all tables
    pub fn release_all_memory(&self) {
        for (_, backend) in self.tables.read().iter() {
            backend.release_all_columns();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn test_backend_create_and_open() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test.apex");

        // Create and insert
        {
            let backend = TableStorageBackend::create(&path).unwrap();
            
            let mut row = HashMap::new();
            row.insert("name".to_string(), Value::String("Alice".to_string()));
            row.insert("age".to_string(), Value::Int64(30));
            
            let ids = backend.insert_rows(&[row]).unwrap();
            assert_eq!(ids.len(), 1);
            
            backend.save().unwrap();
        }

        // Reopen and verify (lazy load)
        {
            let backend = TableStorageBackend::open(&path).unwrap();
            
            // Metadata available without loading data
            assert_eq!(backend.row_count(), 1);
            
            // Cache should be empty
            assert!(backend.get_cached_column("name").is_none());
            
            // Load specific column
            backend.load_columns(&["name"]).unwrap();
            assert!(backend.get_cached_column("name").is_some());
        }
    }

    #[test]
    fn test_column_type_conversions() {
        // Test Int64
        let mut col = TypedColumn::Int64 {
            data: vec![1, 2, 3],
            nulls: BitVec::new(),
        };
        if let TypedColumn::Int64 { nulls, .. } = &mut col {
            nulls.extend_false(3);
        }
        
        let cd = typed_column_to_column_data(&col);
        let back = column_data_to_typed_column(&cd, DataType::Int64);
        
        if let TypedColumn::Int64 { data, .. } = back {
            assert_eq!(data, vec![1, 2, 3]);
        } else {
            panic!("Expected Int64 column");
        }
    }

    #[test]
    fn test_insert_typed_and_reload() {
        use crate::storage::OnDemandStorage;
        
        let dir = tempdir().unwrap();
        let path = dir.path().join("test_typed.apex");

        // Save using insert_typed (like save_to_v3 does)
        {
            let storage = OnDemandStorage::create(&path).unwrap();
            
            let mut int_cols: HashMap<String, Vec<i64>> = HashMap::new();
            let mut string_cols: HashMap<String, Vec<String>> = HashMap::new();
            
            int_cols.insert("age".to_string(), vec![30, 25]);
            string_cols.insert("name".to_string(), vec!["Alice".to_string(), "Bob".to_string()]);
            
            let ids = storage.insert_typed(
                int_cols,
                HashMap::new(), // float
                string_cols,
                HashMap::new(), // binary
                HashMap::new(), // bool
            ).unwrap();
            
            assert_eq!(ids.len(), 2);
            assert_eq!(storage.row_count(), 2);
            
            storage.save().unwrap();
        }

        // Reopen and verify with backend
        {
            let backend = TableStorageBackend::open(&path).unwrap();
            
            // Check metadata
            let schema = backend.get_schema();
            println!("Schema after reopen: {:?}", schema);
            assert!(!schema.is_empty(), "Schema should not be empty");
            
            let row_count = backend.row_count();
            println!("Row count after reopen: {}", row_count);
            assert_eq!(row_count, 2, "Should have 2 rows");
            
            // Load all columns
            backend.load_all_columns().unwrap();
            
            // Check cached columns
            let name_col = backend.get_cached_column("name");
            println!("Name column: {:?}", name_col.is_some());
            assert!(name_col.is_some(), "Name column should be loaded");
            
            let age_col = backend.get_cached_column("age");
            println!("Age column: {:?}", age_col.is_some());
            assert!(age_col.is_some(), "Age column should be loaded");
        }
    }

    #[test]
    fn test_read_columns_to_arrow() {
        use crate::storage::OnDemandStorage;
        
        let dir = tempdir().unwrap();
        let path = dir.path().join("test_arrow.apex");

        // Create test data
        {
            let storage = OnDemandStorage::create(&path).unwrap();
            
            let mut int_cols: HashMap<String, Vec<i64>> = HashMap::new();
            let mut float_cols: HashMap<String, Vec<f64>> = HashMap::new();
            let mut string_cols: HashMap<String, Vec<String>> = HashMap::new();
            
            int_cols.insert("age".to_string(), vec![30, 25, 35]);
            float_cols.insert("score".to_string(), vec![85.5, 90.0, 78.5]);
            string_cols.insert("name".to_string(), vec!["Alice".to_string(), "Bob".to_string(), "Charlie".to_string()]);
            
            storage.insert_typed(int_cols, float_cols, string_cols, HashMap::new(), HashMap::new()).unwrap();
            storage.save().unwrap();
        }

        // Test read_columns_to_arrow
        {
            let backend = TableStorageBackend::open(&path).unwrap();
            
            // Read all columns
            let batch = backend.read_columns_to_arrow(None, 0, None).unwrap();
            assert_eq!(batch.num_rows(), 3);
            assert_eq!(batch.num_columns(), 3);
            
            // Read specific columns (column projection)
            let batch2 = backend.read_columns_to_arrow(Some(&["name", "age"]), 0, None).unwrap();
            assert_eq!(batch2.num_rows(), 3);
            assert_eq!(batch2.num_columns(), 2);
            
            // Read with row limit
            let batch3 = backend.read_columns_to_arrow(None, 0, Some(2)).unwrap();
            assert_eq!(batch3.num_rows(), 2);
            
            // Read single column with limit
            let batch4 = backend.read_columns_to_arrow(Some(&["name"]), 0, Some(1)).unwrap();
            assert_eq!(batch4.num_rows(), 1);
            assert_eq!(batch4.num_columns(), 1);
        }
    }

    #[test]
    fn test_column_projection_correctness() {
        use crate::storage::OnDemandStorage;
        use arrow::array::{Int64Array, StringArray};
        
        let dir = tempdir().unwrap();
        let path = dir.path().join("test_proj.apex");

        // Create test data
        {
            let storage = OnDemandStorage::create(&path).unwrap();
            
            let mut int_cols: HashMap<String, Vec<i64>> = HashMap::new();
            let mut string_cols: HashMap<String, Vec<String>> = HashMap::new();
            
            int_cols.insert("id".to_string(), vec![1, 2, 3]);
            int_cols.insert("value".to_string(), vec![100, 200, 300]);
            string_cols.insert("label".to_string(), vec!["a".to_string(), "b".to_string(), "c".to_string()]);
            
            storage.insert_typed(int_cols, HashMap::new(), string_cols, HashMap::new(), HashMap::new()).unwrap();
            storage.save().unwrap();
        }

        // Verify column projection returns correct data
        {
            let backend = TableStorageBackend::open(&path).unwrap();
            
            // Read only 'id' and 'label' columns
            let batch = backend.read_columns_to_arrow(Some(&["id", "label"]), 0, None).unwrap();
            
            assert_eq!(batch.num_columns(), 2);
            
            // Verify column names
            let schema = batch.schema();
            let field_names: Vec<&str> = schema.fields().iter().map(|f| f.name().as_str()).collect();
            assert!(field_names.contains(&"id"));
            assert!(field_names.contains(&"label"));
            assert!(!field_names.contains(&"value")); // Should NOT include 'value'
        }
    }

    #[test]
    fn test_row_range_scan() {
        use crate::storage::OnDemandStorage;
        use arrow::array::Int64Array;
        
        let dir = tempdir().unwrap();
        let path = dir.path().join("test_range.apex");

        // Create test data with 10 rows
        {
            let storage = OnDemandStorage::create(&path).unwrap();
            
            let mut int_cols: HashMap<String, Vec<i64>> = HashMap::new();
            int_cols.insert("index".to_string(), (0..10).collect());
            
            storage.insert_typed(int_cols, HashMap::new(), HashMap::new(), HashMap::new(), HashMap::new()).unwrap();
            storage.save().unwrap();
        }

        // Test row range scanning
        {
            let backend = TableStorageBackend::open(&path).unwrap();
            
            // Read first 3 rows
            let batch1 = backend.read_columns_to_arrow(None, 0, Some(3)).unwrap();
            assert_eq!(batch1.num_rows(), 3);
            
            // Read middle 4 rows (rows 3-6)
            let batch2 = backend.read_columns_to_arrow(None, 3, Some(4)).unwrap();
            assert_eq!(batch2.num_rows(), 4);
            
            // Read last 2 rows
            let batch3 = backend.read_columns_to_arrow(None, 8, Some(10)).unwrap();
            assert_eq!(batch3.num_rows(), 2); // Only 2 rows left
        }
    }
}
