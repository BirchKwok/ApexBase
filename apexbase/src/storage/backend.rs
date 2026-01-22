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
        ColumnType::String => DataType::String,
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
    /// Create a new storage backend
    pub fn create(path: &Path) -> io::Result<Self> {
        let storage = OnDemandStorage::create(path)?;
        Ok(Self {
            path: path.to_path_buf(),
            storage,
            cached_columns: RwLock::new(HashMap::new()),
            schema: RwLock::new(Vec::new()),
            row_count: RwLock::new(0),
            dirty: RwLock::new(false),
        })
    }

    /// Open existing storage (lazy - only reads metadata)
    pub fn open(path: &Path) -> io::Result<Self> {
        let storage = OnDemandStorage::open(path)?;
        
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

    /// Open or create storage (for read-only or query operations)
    pub fn open_or_create(path: &Path) -> io::Result<Self> {
        if path.exists() {
            Self::open(path)
        } else {
            Self::create(path)
        }
    }

    /// Open for write - loads all existing data for append operations
    pub fn open_for_write(path: &Path) -> io::Result<Self> {
        let storage = OnDemandStorage::open_for_write(path)?;
        
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
    pub fn insert_rows(&self, rows: &[HashMap<String, Value>]) -> io::Result<Vec<u64>> {
        use crate::storage::on_demand::ColumnValue;

        if rows.is_empty() {
            return Ok(Vec::new());
        }

        // Convert to ColumnValue format
        let converted: Vec<HashMap<String, ColumnValue>> = rows
            .iter()
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
            .collect();

        // Insert into storage
        let ids = self.storage.insert_rows(&converted)?;

        // Update schema if new columns
        {
            let mut schema = self.schema.write();
            for row in rows {
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

    /// Save changes to disk
    pub fn save(&self) -> io::Result<()> {
        self.storage.save()?;
        *self.dirty.write() = false;
        Ok(())
    }

    /// Check if there are unsaved changes
    pub fn is_dirty(&self) -> bool {
        *self.dirty.read()
    }

    /// Flush and close
    pub fn close(&self) -> io::Result<()> {
        if self.is_dirty() {
            self.save()?;
        }
        Ok(())
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
                let arrays: Vec<ArrayRef> = vec![Arc::new(Int64Array::from(
                    ids.iter().map(|&id| id as i64).collect::<Vec<i64>>()
                ))];
                let schema = Arc::new(Schema::new(fields));
                return arrow::record_batch::RecordBatch::try_new(schema, arrays)
                    .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e.to_string()));
            }
        }
        
        // Read columns from storage (only the requested ones!)
        let col_data = self.storage.read_columns(column_names, start_row, row_count)?;
        
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
            arrays.push(Arc::new(Int64Array::from(ids.iter().map(|&id| id as i64).collect::<Vec<i64>>())));
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
            if let Some(data) = col_data.get(col_name) {
                let dtype = schema.iter()
                    .find(|(n, _)| n == col_name)
                    .map(|(_, dt)| dt.clone())
                    .unwrap_or(DataType::String);

                let col_len = data.len();
                let (arrow_dt, array): (ArrowDataType, ArrayRef) = match data {
                    ColumnData::Int64(values) => {
                        // Pad with NULLs if column is shorter than expected
                        if values.len() < expected_row_count {
                            let mut padded: Vec<Option<i64>> = values.iter().map(|&v| Some(v)).collect();
                            padded.extend(std::iter::repeat(None).take(expected_row_count - values.len()));
                            (ArrowDataType::Int64, Arc::new(Int64Array::from(padded)))
                        } else {
                            (ArrowDataType::Int64, Arc::new(Int64Array::from(values.clone())))
                        }
                    }
                    ColumnData::Float64(values) => {
                        if values.len() < expected_row_count {
                            let mut padded: Vec<Option<f64>> = values.iter().map(|&v| Some(v)).collect();
                            padded.extend(std::iter::repeat(None).take(expected_row_count - values.len()));
                            (ArrowDataType::Float64, Arc::new(Float64Array::from(padded)))
                        } else {
                            (ArrowDataType::Float64, Arc::new(Float64Array::from(values.clone())))
                        }
                    }
                    ColumnData::String { offsets, data: bytes } => {
                        let count = offsets.len().saturating_sub(1);
                        let mut strings: Vec<Option<String>> = (0..count)
                            .map(|i| {
                                let start = offsets[i] as usize;
                                let end = offsets[i + 1] as usize;
                                std::str::from_utf8(&bytes[start..end])
                                    .ok()
                                    .map(|s| s.to_string())
                            })
                            .collect();
                        // Pad with NULLs if column is shorter than expected
                        if strings.len() < expected_row_count {
                            strings.extend(std::iter::repeat(None).take(expected_row_count - strings.len()));
                        }
                        (ArrowDataType::Utf8, Arc::new(StringArray::from(strings)))
                    }
                    ColumnData::Bool { data: packed, len } => {
                        let mut bools: Vec<Option<bool>> = (0..*len)
                            .map(|i| {
                                let byte_idx = i / 8;
                                let bit_idx = i % 8;
                                Some(byte_idx < packed.len() && (packed[byte_idx] >> bit_idx) & 1 == 1)
                            })
                            .collect();
                        // Pad with NULLs if column is shorter than expected
                        if bools.len() < expected_row_count {
                            bools.extend(std::iter::repeat(None).take(expected_row_count - bools.len()));
                        }
                        (ArrowDataType::Boolean, Arc::new(BooleanArray::from(bools)))
                    }
                    ColumnData::Binary { offsets, data: bytes } => {
                        use arrow::array::BinaryArray;
                        // Keep as binary data in Arrow
                        let count = offsets.len().saturating_sub(1);
                        let mut binary_data: Vec<Option<&[u8]>> = (0..count)
                            .map(|i| {
                                let start = offsets[i] as usize;
                                let end = offsets[i + 1] as usize;
                                Some(&bytes[start..end] as &[u8])
                            })
                            .collect();
                        // Pad with NULLs if column is shorter than expected
                        if binary_data.len() < expected_row_count {
                            binary_data.extend(std::iter::repeat(None).take(expected_row_count - binary_data.len()));
                        }
                        (ArrowDataType::Binary, Arc::new(BinaryArray::from(binary_data)))
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

    /// Get underlying storage for direct access
    pub fn storage(&self) -> &OnDemandStorage {
        &self.storage
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
