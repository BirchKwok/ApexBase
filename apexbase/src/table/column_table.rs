//! High-performance columnar table storage
//! 
//! This module provides a column-oriented storage format that achieves:
//! - Fast writes: O(1) amortized append per column
//! - Fast queries: Zero-copy iteration, SIMD-friendly layout
//! - Fast lookups: O(1) by ID via HashMap index

use crate::data::{DataType, Value};
use crate::query::{Filter, QueryExecutor};
use crate::Result;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Bit vector for efficient boolean storage
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct BitVec {
    data: Vec<u64>,
    len: usize,
}

impl BitVec {
    pub fn new() -> Self {
        Self { data: Vec::new(), len: 0 }
    }

    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            data: Vec::with_capacity((capacity + 63) / 64),
            len: 0,
        }
    }

    #[inline]
    pub fn push(&mut self, value: bool) {
        let word_idx = self.len / 64;
        let bit_idx = self.len % 64;
        
        if word_idx >= self.data.len() {
            self.data.push(0);
        }
        
        if value {
            self.data[word_idx] |= 1u64 << bit_idx;
        }
        self.len += 1;
    }

    #[inline]
    pub fn get(&self, index: usize) -> bool {
        if index >= self.len {
            return false;
        }
        let word_idx = index / 64;
        let bit_idx = index % 64;
        (self.data[word_idx] >> bit_idx) & 1 == 1
    }

    #[inline]
    pub fn set(&mut self, index: usize, value: bool) {
        let word_idx = index / 64;
        let bit_idx = index % 64;
        
        // Extend if needed
        while word_idx >= self.data.len() {
            self.data.push(0);
        }
        if index >= self.len {
            self.len = index + 1;
        }
        
        if value {
            self.data[word_idx] |= 1u64 << bit_idx;
        } else {
            self.data[word_idx] &= !(1u64 << bit_idx);
        }
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Count set bits (for counting non-deleted rows)
    pub fn count_ones(&self) -> usize {
        self.data.iter().map(|w| w.count_ones() as usize).sum()
    }

    /// Check if all bits are false (no nulls) - O(words) not O(bits)
    #[inline]
    pub fn all_false(&self) -> bool {
        self.data.iter().all(|&w| w == 0)
    }

    /// Get raw u64 data for direct Arrow buffer conversion
    #[inline]
    pub fn raw_data(&self) -> &[u64] {
        &self.data
    }

    /// Extend with n false values - fast batch operation
    #[inline]
    pub fn extend_false(&mut self, count: usize) {
        // Calculate new length
        let new_len = self.len + count;
        let required_words = (new_len + 63) / 64;
        
        // Extend data vec if needed (new words are already 0)
        self.data.resize(required_words, 0);
        self.len = new_len;
    }

    /// Slice BitVec for delta extraction [start, end)
    pub fn slice(&self, start: usize, end: usize) -> Self {
        let mut result = Self::with_capacity(end - start);
        for i in start..end {
            result.push(self.get(i));
        }
        result
    }

    /// Extend from another BitVec
    pub fn extend(&mut self, other: &BitVec) {
        for i in 0..other.len() {
            self.push(other.get(i));
        }
    }

    /// Extend with n true values - fast batch operation
    #[inline]
    pub fn extend_true(&mut self, count: usize) {
        let start_idx = self.len;
        let new_len = self.len + count;
        let required_words = (new_len + 63) / 64;
        
        // Ensure capacity
        self.data.resize(required_words, 0);
        
        // Set bits from start_idx to new_len
        for i in start_idx..new_len {
            let word_idx = i / 64;
            let bit_idx = i % 64;
            self.data[word_idx] |= 1u64 << bit_idx;
        }
        self.len = new_len;
    }

    /// Batch extend from a boolean slice - optimized for SIMD
    #[inline]
    pub fn extend_from_bools(&mut self, values: &[bool]) {
        let count = values.len();
        if count == 0 { return; }
        
        let new_len = self.len + count;
        let required_words = (new_len + 63) / 64;
        
        // Ensure capacity
        self.data.resize(required_words, 0);
        
        // Process in chunks of 64 for better performance
        let mut idx = 0;
        let base_bit = self.len;
        
        // Fast path: process full words at once
        while idx + 64 <= count {
            let mut word = 0u64;
            for bit in 0..64 {
                if values[idx + bit] {
                    word |= 1u64 << bit;
                }
            }
            let word_idx = (base_bit + idx) / 64;
            let bit_offset = (base_bit + idx) % 64;
            
            if bit_offset == 0 {
                self.data[word_idx] = word;
            } else {
                // Handle cross-word boundary
                self.data[word_idx] |= word << bit_offset;
                if word_idx + 1 < self.data.len() {
                    self.data[word_idx + 1] |= word >> (64 - bit_offset);
                }
            }
            idx += 64;
        }
        
        // Handle remaining bits
        while idx < count {
            let bit_idx = base_bit + idx;
            let word_idx = bit_idx / 64;
            let bit_pos = bit_idx % 64;
            if values[idx] {
                self.data[word_idx] |= 1u64 << bit_pos;
            }
            idx += 1;
        }
        
        self.len = new_len;
    }
}

/// Type-specific column storage for maximum performance
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TypedColumn {
    Int64 {
        data: Vec<i64>,
        nulls: BitVec,
    },
    Float64 {
        data: Vec<f64>,
        nulls: BitVec,
    },
    String {
        data: Vec<String>,
        nulls: BitVec,
    },
    Bool {
        data: BitVec,
        nulls: BitVec,
    },
    /// For mixed/unknown types, fall back to Value
    Mixed {
        data: Vec<Value>,
        nulls: BitVec,
    },
}

impl TypedColumn {
    pub fn new(dtype: DataType) -> Self {
        match dtype {
            DataType::Int64 | DataType::Int32 | DataType::Int16 | DataType::Int8 => {
                TypedColumn::Int64 { data: Vec::new(), nulls: BitVec::new() }
            }
            DataType::Float64 | DataType::Float32 => {
                TypedColumn::Float64 { data: Vec::new(), nulls: BitVec::new() }
            }
            DataType::String => {
                TypedColumn::String { data: Vec::new(), nulls: BitVec::new() }
            }
            DataType::Bool => {
                TypedColumn::Bool { data: BitVec::new(), nulls: BitVec::new() }
            }
            _ => {
                TypedColumn::Mixed { data: Vec::new(), nulls: BitVec::new() }
            }
        }
    }

    pub fn with_capacity(dtype: DataType, capacity: usize) -> Self {
        match dtype {
            DataType::Int64 | DataType::Int32 | DataType::Int16 | DataType::Int8 => {
                TypedColumn::Int64 {
                    data: Vec::with_capacity(capacity),
                    nulls: BitVec::with_capacity(capacity),
                }
            }
            DataType::Float64 | DataType::Float32 => {
                TypedColumn::Float64 {
                    data: Vec::with_capacity(capacity),
                    nulls: BitVec::with_capacity(capacity),
                }
            }
            DataType::String => {
                TypedColumn::String {
                    data: Vec::with_capacity(capacity),
                    nulls: BitVec::with_capacity(capacity),
                }
            }
            DataType::Bool => {
                TypedColumn::Bool {
                    data: BitVec::with_capacity(capacity),
                    nulls: BitVec::with_capacity(capacity),
                }
            }
            _ => {
                TypedColumn::Mixed {
                    data: Vec::with_capacity(capacity),
                    nulls: BitVec::with_capacity(capacity),
                }
            }
        }
    }

    /// Push a value to the column - O(1) amortized
    #[inline]
    pub fn push(&mut self, value: &Value) {
        match (self, value) {
            (TypedColumn::Int64 { data, nulls }, Value::Int64(v)) => {
                data.push(*v);
                nulls.push(false);
            }
            (TypedColumn::Int64 { data, nulls }, Value::Int32(v)) => {
                data.push(*v as i64);
                nulls.push(false);
            }
            (TypedColumn::Int64 { data, nulls }, Value::Null) => {
                data.push(0);
                nulls.push(true);
            }
            (TypedColumn::Float64 { data, nulls }, Value::Float64(v)) => {
                data.push(*v);
                nulls.push(false);
            }
            (TypedColumn::Float64 { data, nulls }, Value::Float32(v)) => {
                data.push(*v as f64);
                nulls.push(false);
            }
            (TypedColumn::Float64 { data, nulls }, Value::Int64(v)) => {
                data.push(*v as f64);
                nulls.push(false);
            }
            (TypedColumn::Float64 { data, nulls }, Value::Null) => {
                data.push(0.0);
                nulls.push(true);
            }
            (TypedColumn::String { data, nulls }, Value::String(v)) => {
                data.push(v.clone());
                nulls.push(false);
            }
            (TypedColumn::String { data, nulls }, Value::Null) => {
                data.push(String::new());
                nulls.push(true);
            }
            (TypedColumn::Bool { data, nulls }, Value::Bool(v)) => {
                data.push(*v);
                nulls.push(false);
            }
            (TypedColumn::Bool { data, nulls }, Value::Null) => {
                data.push(false);
                nulls.push(true);
            }
            (TypedColumn::Mixed { data, nulls }, v) => {
                nulls.push(v.is_null());
                data.push(v.clone());
            }
            // Type mismatch - convert to mixed or store as null
            (col, value) => {
                col.push_null();
                // Log warning in debug mode
                #[cfg(debug_assertions)]
                eprintln!("Warning: Type mismatch when pushing {:?}", value);
            }
        }
    }

    /// Push a null value
    #[inline]
    pub fn push_null(&mut self) {
        match self {
            TypedColumn::Int64 { data, nulls } => {
                data.push(0);
                nulls.push(true);
            }
            TypedColumn::Float64 { data, nulls } => {
                data.push(0.0);
                nulls.push(true);
            }
            TypedColumn::String { data, nulls } => {
                data.push(String::new());
                nulls.push(true);
            }
            TypedColumn::Bool { data, nulls } => {
                data.push(false);
                nulls.push(true);
            }
            TypedColumn::Mixed { data, nulls } => {
                data.push(Value::Null);
                nulls.push(true);
            }
        }
    }

    /// Get value at index
    #[inline]
    pub fn get(&self, index: usize) -> Option<Value> {
        match self {
            TypedColumn::Int64 { data, nulls } => {
                if index >= data.len() || nulls.get(index) {
                    Some(Value::Null)
                } else {
                    Some(Value::Int64(data[index]))
                }
            }
            TypedColumn::Float64 { data, nulls } => {
                if index >= data.len() || nulls.get(index) {
                    Some(Value::Null)
                } else {
                    Some(Value::Float64(data[index]))
                }
            }
            TypedColumn::String { data, nulls } => {
                if index >= data.len() || nulls.get(index) {
                    Some(Value::Null)
                } else {
                    Some(Value::String(data[index].clone()))
                }
            }
            TypedColumn::Bool { data, nulls } => {
                if index >= data.len() || nulls.get(index) {
                    Some(Value::Null)
                } else {
                    Some(Value::Bool(data.get(index)))
                }
            }
            TypedColumn::Mixed { data, nulls } => {
                if index >= data.len() {
                    None
                } else if nulls.get(index) {
                    Some(Value::Null)
                } else {
                    Some(data[index].clone())
                }
            }
        }
    }

    /// Set value at index
    #[inline]
    pub fn set(&mut self, index: usize, value: &Value) {
        match (self, value) {
            (TypedColumn::Int64 { data, nulls }, Value::Int64(v)) => {
                if index < data.len() {
                    data[index] = *v;
                    nulls.set(index, false);
                }
            }
            (TypedColumn::Int64 { data, nulls }, Value::Null) => {
                if index < data.len() {
                    nulls.set(index, true);
                }
            }
            (TypedColumn::Float64 { data, nulls }, Value::Float64(v)) => {
                if index < data.len() {
                    data[index] = *v;
                    nulls.set(index, false);
                }
            }
            (TypedColumn::Float64 { data, nulls }, Value::Null) => {
                if index < data.len() {
                    nulls.set(index, true);
                }
            }
            (TypedColumn::String { data, nulls }, Value::String(v)) => {
                if index < data.len() {
                    data[index] = v.clone();
                    nulls.set(index, false);
                }
            }
            (TypedColumn::String { data, nulls }, Value::Null) => {
                if index < data.len() {
                    nulls.set(index, true);
                }
            }
            (TypedColumn::Bool { data, nulls }, Value::Bool(v)) => {
                if index < data.len() {
                    data.set(index, *v);
                    nulls.set(index, false);
                }
            }
            (TypedColumn::Bool { data, nulls }, Value::Null) => {
                if index < data.len() {
                    nulls.set(index, true);
                }
            }
            (TypedColumn::Mixed { data, nulls }, v) => {
                if index < data.len() {
                    data[index] = v.clone();
                    nulls.set(index, v.is_null());
                }
            }
            _ => {}
        }
    }

    pub fn len(&self) -> usize {
        match self {
            TypedColumn::Int64 { data, .. } => data.len(),
            TypedColumn::Float64 { data, .. } => data.len(),
            TypedColumn::String { data, .. } => data.len(),
            TypedColumn::Bool { data, .. } => data.len(),
            TypedColumn::Mixed { data, .. } => data.len(),
        }
    }

    pub fn is_null(&self, index: usize) -> bool {
        match self {
            TypedColumn::Int64 { nulls, .. } => nulls.get(index),
            TypedColumn::Float64 { nulls, .. } => nulls.get(index),
            TypedColumn::String { nulls, .. } => nulls.get(index),
            TypedColumn::Bool { nulls, .. } => nulls.get(index),
            TypedColumn::Mixed { nulls, .. } => nulls.get(index),
        }
    }

    pub fn data_type(&self) -> DataType {
        match self {
            TypedColumn::Int64 { .. } => DataType::Int64,
            TypedColumn::Float64 { .. } => DataType::Float64,
            TypedColumn::String { .. } => DataType::String,
            TypedColumn::Bool { .. } => DataType::Bool,
            TypedColumn::Mixed { .. } => DataType::Json, // fallback
        }
    }

    /// Slice column for delta extraction [start, end)
    pub fn slice(&self, start: usize, end: usize) -> Self {
        match self {
            TypedColumn::Int64 { data, nulls } => TypedColumn::Int64 {
                data: data[start..end].to_vec(),
                nulls: nulls.slice(start, end),
            },
            TypedColumn::Float64 { data, nulls } => TypedColumn::Float64 {
                data: data[start..end].to_vec(),
                nulls: nulls.slice(start, end),
            },
            TypedColumn::String { data, nulls } => TypedColumn::String {
                data: data[start..end].to_vec(),
                nulls: nulls.slice(start, end),
            },
            TypedColumn::Bool { data, nulls } => TypedColumn::Bool {
                data: data.slice(start, end),
                nulls: nulls.slice(start, end),
            },
            TypedColumn::Mixed { data, nulls } => TypedColumn::Mixed {
                data: data[start..end].to_vec(),
                nulls: nulls.slice(start, end),
            },
        }
    }

    /// Append another TypedColumn to this one (for delta loading)
    pub fn append(&mut self, other: Self) {
        match (self, other) {
            (TypedColumn::Int64 { data, nulls }, TypedColumn::Int64 { data: other_data, nulls: other_nulls }) => {
                data.extend(other_data);
                nulls.extend(&other_nulls);
            }
            (TypedColumn::Float64 { data, nulls }, TypedColumn::Float64 { data: other_data, nulls: other_nulls }) => {
                data.extend(other_data);
                nulls.extend(&other_nulls);
            }
            (TypedColumn::String { data, nulls }, TypedColumn::String { data: other_data, nulls: other_nulls }) => {
                data.extend(other_data);
                nulls.extend(&other_nulls);
            }
            (TypedColumn::Bool { data, nulls }, TypedColumn::Bool { data: other_data, nulls: other_nulls }) => {
                data.extend(&other_data);
                nulls.extend(&other_nulls);
            }
            (TypedColumn::Mixed { data, nulls }, TypedColumn::Mixed { data: other_data, nulls: other_nulls }) => {
                data.extend(other_data);
                nulls.extend(&other_nulls);
            }
            _ => {} // Type mismatch - ignore
        }
    }
}

/// Column schema with fast lookup
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ColumnSchema {
    /// Column definitions: (name, data_type)
    pub columns: Vec<(String, DataType)>,
    /// Name to index mapping for O(1) lookup
    pub name_to_index: HashMap<String, usize>,
}

impl ColumnSchema {
    pub fn new() -> Self {
        Self {
            columns: Vec::new(),
            name_to_index: HashMap::new(),
        }
    }

    pub fn add_column(&mut self, name: &str, dtype: DataType) -> usize {
        if let Some(&idx) = self.name_to_index.get(name) {
            return idx;
        }
        let idx = self.columns.len();
        self.columns.push((name.to_string(), dtype));
        self.name_to_index.insert(name.to_string(), idx);
        idx
    }

    pub fn get_index(&self, name: &str) -> Option<usize> {
        self.name_to_index.get(name).copied()
    }

    pub fn get_type(&self, index: usize) -> Option<DataType> {
        self.columns.get(index).map(|(_, t)| *t)
    }

    pub fn column_names(&self) -> Vec<String> {
        self.columns.iter().map(|(n, _)| n.clone()).collect()
    }

    pub fn len(&self) -> usize {
        self.columns.len()
    }
}

impl Default for ColumnSchema {
    fn default() -> Self {
        Self::new()
    }
}

/// Write buffer for fast batch inserts
#[derive(Debug, Clone, Default)]
struct WriteBuffer {
    /// Buffered rows waiting to be flushed to columns
    rows: Vec<(u64, HashMap<String, Value>)>,
    /// Buffer capacity before auto-flush
    capacity: usize,
}

impl WriteBuffer {
    fn new(capacity: usize) -> Self {
        Self {
            rows: Vec::with_capacity(capacity),
            capacity,
        }
    }

    fn push(&mut self, id: u64, fields: HashMap<String, Value>) {
        self.rows.push((id, fields));
    }

    fn is_full(&self) -> bool {
        self.rows.len() >= self.capacity
    }

    fn is_empty(&self) -> bool {
        self.rows.is_empty()
    }

    fn len(&self) -> usize {
        self.rows.len()
    }

    fn drain(&mut self) -> Vec<(u64, HashMap<String, Value>)> {
        std::mem::take(&mut self.rows)
    }
}

/// High-performance columnar table with write buffer
/// 
/// ID Design: Row ID = Row Index (0-based)
/// - No HashMap overhead for ID lookup
/// - O(1) direct array access
/// - Soft delete via BitVec
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ColumnTable {
    /// Table ID
    id: u32,
    /// Table name
    name: String,
    /// Schema
    schema: ColumnSchema,
    /// Typed column data
    columns: Vec<TypedColumn>,
    /// Deleted row markers (soft delete)
    deleted: BitVec,
    /// Total number of rows (including deleted)
    row_count: usize,
    /// Number of active (non-deleted) rows
    active_count: usize,
    /// Write buffer for fast inserts
    #[serde(skip)]
    write_buffer: WriteBuffer,
}

/// Buffer size for write operations
const WRITE_BUFFER_SIZE: usize = 1000;

impl ColumnTable {
    /// Create a new column table
    pub fn new(id: u32, name: &str) -> Self {
        Self {
            id,
            name: name.to_string(),
            schema: ColumnSchema::new(),
            columns: Vec::new(),
            deleted: BitVec::new(),
            row_count: 0,
            active_count: 0,
            write_buffer: WriteBuffer::new(WRITE_BUFFER_SIZE),
        }
    }

    /// Create with estimated capacity for better performance
    pub fn with_capacity(id: u32, name: &str, capacity: usize) -> Self {
        Self {
            id,
            name: name.to_string(),
            schema: ColumnSchema::new(),
            columns: Vec::new(),
            deleted: BitVec::with_capacity(capacity),
            row_count: 0,
            active_count: 0,
            write_buffer: WriteBuffer::new(WRITE_BUFFER_SIZE),
        }
    }

    pub fn id(&self) -> u32 {
        self.id
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    /// Number of active (non-deleted) rows
    pub fn row_count(&self) -> u64 {
        self.active_count as u64
    }

    /// Total number of rows including deleted
    pub fn total_rows(&self) -> usize {
        self.row_count
    }

    /// Insert a single row - O(columns) time
    /// Returns row ID (= row index, 0-based)
    pub fn insert(&mut self, fields: &HashMap<String, Value>) -> Result<u64> {
        let row_id = (self.row_count + self.write_buffer.len()) as u64;
        
        // Add to write buffer
        self.write_buffer.push(row_id, fields.clone());
        self.active_count += 1;
        
        // Auto-flush if buffer is full
        if self.write_buffer.is_full() {
            self.flush_write_buffer();
        }
        
        Ok(row_id)
    }
    
    /// Public method to flush write buffer (for persistence)
    pub fn flush_buffer(&mut self) {
        self.flush_write_buffer();
    }
    
    /// Check if there are pending writes in the buffer
    #[inline]
    pub fn has_pending_writes(&self) -> bool {
        !self.write_buffer.is_empty()
    }

    /// Flush write buffer to column storage
    pub fn flush_write_buffer(&mut self) {
        if self.write_buffer.is_empty() {
            return;
        }
        
        let buffered_rows = self.write_buffer.drain();
        let batch_size = buffered_rows.len();
        
        // Collect schema from buffered rows
        for (_, fields) in &buffered_rows {
            for (name, value) in fields {
                self.ensure_column(name, value.data_type());
            }
        }
        
        // Pre-allocate columns
        for col in &mut self.columns {
            match col {
                TypedColumn::Int64 { data, .. } => data.reserve(batch_size),
                TypedColumn::Float64 { data, .. } => data.reserve(batch_size),
                TypedColumn::String { data, .. } => data.reserve(batch_size),
                TypedColumn::Mixed { data, .. } => data.reserve(batch_size),
                _ => {}
            }
        }
        
        // Flush to columns - ID is just the row index
        for (_row_id, fields) in buffered_rows.into_iter() {
            self.deleted.push(false);
            
            // Push values to each column
            for (col_idx, (col_name, _)) in self.schema.columns.iter().enumerate() {
                if let Some(value) = fields.get(col_name) {
                    self.columns[col_idx].push(value);
                } else {
                    self.columns[col_idx].push_null();
                }
            }
            self.row_count += 1;
        }
    }

    /// Batch insert - optimized for bulk loading
    /// Uses column-wise collection for better cache utilization
    /// Returns row IDs (= row indices, 0-based)
    pub fn insert_batch(&mut self, rows: &[HashMap<String, Value>]) -> Result<Vec<u64>> {
        if rows.is_empty() {
            return Ok(Vec::new());
        }
        
        // Flush any pending writes first
        self.flush_write_buffer();

        let batch_size = rows.len();
        let start_id = self.row_count as u64;
        
        // Phase 1: Discover schema from all rows
        let mut all_columns: Vec<(String, DataType)> = Vec::new();
        let mut seen_columns: HashMap<String, usize> = HashMap::new();
        
        for row in rows {
            for (name, value) in row {
                if !seen_columns.contains_key(name) {
                    seen_columns.insert(name.clone(), all_columns.len());
                    all_columns.push((name.clone(), value.data_type()));
                }
            }
        }
        
        // Ensure all columns exist
        for (name, dtype) in &all_columns {
            self.ensure_column(name, *dtype);
        }
        
        // Phase 2: Pre-allocate and prepare column data collectors
        // Collect data by column instead of by row for better cache locality
        let num_cols = self.schema.columns.len();
        let mut col_data: Vec<Vec<Option<&Value>>> = vec![Vec::with_capacity(batch_size); num_cols];
        
        // Build column name to local index mapping
        let _col_name_to_idx: HashMap<&str, usize> = self.schema.columns
            .iter()
            .enumerate()
            .map(|(i, (name, _))| (name.as_str(), i))
            .collect();
        
        // Collect values column by column
        for row in rows {
            for col_idx in 0..num_cols {
                let col_name = &self.schema.columns[col_idx].0;
                col_data[col_idx].push(row.get(col_name));
            }
        }
        
        // Phase 3: Generate IDs (= row indices) and extend deleted marker
        let result_ids: Vec<u64> = (0..batch_size)
            .map(|i| {
                self.deleted.push(false);
                start_id + i as u64
            })
            .collect();
        
        // Phase 4: Extend each column with collected data
        for (col_idx, values) in col_data.into_iter().enumerate() {
            let col = &mut self.columns[col_idx];
            
            // Reserve capacity first
            match col {
                TypedColumn::Int64 { data, .. } => data.reserve(batch_size),
                TypedColumn::Float64 { data, .. } => data.reserve(batch_size),
                TypedColumn::String { data, .. } => data.reserve(batch_size),
                TypedColumn::Mixed { data, .. } => data.reserve(batch_size),
                _ => {}
            }
            
            // Batch push values
            for opt_val in values {
                match opt_val {
                    Some(value) => col.push(value),
                    None => col.push_null(),
                }
            }
        }
        
        self.row_count += batch_size;
        self.active_count += batch_size;
        Ok(result_ids)
    }

    /// Insert columnar data directly - fastest path for bulk loading
    /// Input: HashMap<column_name, Vec<Value>>
    /// All column Vecs must have the same length
    pub fn insert_columns(&mut self, columns: HashMap<String, Vec<Value>>) -> Result<Vec<u64>> {
        if columns.is_empty() {
            return Ok(Vec::new());
        }
        
        // Flush any pending writes
        self.flush_write_buffer();
        
        // Determine batch size from first column
        let batch_size = columns.values().next().map(|v| v.len()).unwrap_or(0);
        if batch_size == 0 {
            return Ok(Vec::new());
        }
        
        // Validate all columns have same length
        for (name, values) in &columns {
            if values.len() != batch_size {
                return Err(crate::ApexError::InvalidDataType(
                    format!("Column {} has {} rows, expected {}", name, values.len(), batch_size)
                ));
            }
        }
        
        let start_id = self.row_count as u64;
        
        // Ensure all columns exist
        for (name, values) in &columns {
            if !values.is_empty() {
                self.ensure_column(name, values[0].data_type());
            }
        }
        
        // Generate IDs (= row indices) - no HashMap needed
        let result_ids: Vec<u64> = (0..batch_size)
            .map(|i| {
                self.deleted.push(false);
                start_id + i as u64
            })
            .collect();
        
        // Direct column extension - fastest possible path
        for (name, values) in columns {
            if let Some(col_idx) = self.schema.get_index(&name) {
                let col = &mut self.columns[col_idx];
                
                // Reserve and extend
                match col {
                    TypedColumn::Int64 { data, nulls } => {
                        data.reserve(batch_size);
                        for val in &values {
                            match val {
                                Value::Int64(v) => { data.push(*v); nulls.push(false); }
                                Value::Null => { data.push(0); nulls.push(true); }
                                _ => { data.push(0); nulls.push(true); }
                            }
                        }
                    }
                    TypedColumn::Float64 { data, nulls } => {
                        data.reserve(batch_size);
                        for val in &values {
                            match val {
                                Value::Float64(v) => { data.push(*v); nulls.push(false); }
                                Value::Int64(v) => { data.push(*v as f64); nulls.push(false); }
                                Value::Null => { data.push(0.0); nulls.push(true); }
                                _ => { data.push(0.0); nulls.push(true); }
                            }
                        }
                    }
                    TypedColumn::String { data, nulls } => {
                        data.reserve(batch_size);
                        for val in &values {
                            match val {
                                Value::String(v) => { data.push(v.clone()); nulls.push(false); }
                                Value::Null => { data.push(String::new()); nulls.push(true); }
                                _ => { data.push(String::new()); nulls.push(true); }
                            }
                        }
                    }
                    TypedColumn::Bool { data, nulls } => {
                        for val in &values {
                            match val {
                                Value::Bool(v) => { data.push(*v); nulls.push(false); }
                                Value::Null => { data.push(false); nulls.push(true); }
                                _ => { data.push(false); nulls.push(true); }
                            }
                        }
                    }
                    TypedColumn::Mixed { data, nulls } => {
                        data.reserve(batch_size);
                        for val in values {
                            nulls.push(val.is_null());
                            data.push(val);
                        }
                    }
                }
            }
        }
        
        // Fill any columns not provided with nulls
        for col in &mut self.columns {
            let target_len = self.row_count + batch_size;
            while col.len() < target_len {
                col.push_null();
            }
        }
        
        self.row_count += batch_size;
        self.active_count += batch_size;
        Ok(result_ids)
    }

    /// Ultra-fast typed column insert - bypasses Value boxing entirely
    /// 
    /// This is the fastest possible write path, achieving ~0.3ms for 10k rows
    /// by avoiding all intermediate allocations and type conversions.
    pub fn insert_typed_columns(
        &mut self,
        int_cols: HashMap<String, Vec<i64>>,
        float_cols: HashMap<String, Vec<f64>>,
        string_cols: HashMap<String, Vec<String>>,
        bool_cols: HashMap<String, Vec<bool>>,
        binary_cols: HashMap<String, Vec<Vec<u8>>>,
    ) -> Result<Vec<u64>> {
        // Determine batch size from first non-empty column
        let batch_size = int_cols.values().next().map(|v| v.len())
            .or_else(|| float_cols.values().next().map(|v| v.len()))
            .or_else(|| string_cols.values().next().map(|v| v.len()))
            .or_else(|| bool_cols.values().next().map(|v| v.len()))
            .or_else(|| binary_cols.values().next().map(|v| v.len()))
            .unwrap_or(0);
        
        if batch_size == 0 {
            return Ok(Vec::new());
        }
        
        // Flush any pending writes
        self.flush_write_buffer();
        
        let start_id = self.row_count as u64;
        
        // Generate IDs (= row indices) - ultra fast, no HashMap
        self.deleted.extend_false(batch_size);
        let result_ids: Vec<u64> = (start_id..start_id + batch_size as u64).collect();
        
        // Direct typed insert - int64 columns (fastest - memcpy-like)
        for (name, mut values) in int_cols {
            if values.len() != batch_size {
                continue;
            }
            let col_idx = self.schema.add_column(&name, DataType::Int64);
            while col_idx >= self.columns.len() {
                self.columns.push(TypedColumn::Int64 {
                    data: Vec::new(),
                    nulls: BitVec::new(),
                });
            }
            if let TypedColumn::Int64 { data, nulls } = &mut self.columns[col_idx] {
                // Direct append - fastest possible
                data.reserve(batch_size);
                data.append(&mut values);
                // Batch extend nulls with false
                nulls.extend_false(batch_size);
            }
        }
        
        // Float64 columns
        for (name, mut values) in float_cols {
            if values.len() != batch_size {
                continue;
            }
            let col_idx = self.schema.add_column(&name, DataType::Float64);
            while col_idx >= self.columns.len() {
                self.columns.push(TypedColumn::Float64 {
                    data: Vec::new(),
                    nulls: BitVec::new(),
                });
            }
            if let TypedColumn::Float64 { data, nulls } = &mut self.columns[col_idx] {
                data.reserve(batch_size);
                data.append(&mut values);
                nulls.extend_false(batch_size);
            }
        }
        
        // String columns
        for (name, mut values) in string_cols {
            if values.len() != batch_size {
                continue;
            }
            let col_idx = self.schema.add_column(&name, DataType::String);
            while col_idx >= self.columns.len() {
                self.columns.push(TypedColumn::String {
                    data: Vec::new(),
                    nulls: BitVec::new(),
                });
            }
            if let TypedColumn::String { data, nulls } = &mut self.columns[col_idx] {
                data.reserve(batch_size);
                data.append(&mut values);
                nulls.extend_false(batch_size);
            }
        }
        
        // Bool columns - batch insert
        for (name, values) in bool_cols {
            if values.len() != batch_size {
                continue;
            }
            let col_idx = self.schema.add_column(&name, DataType::Bool);
            while col_idx >= self.columns.len() {
                self.columns.push(TypedColumn::Bool {
                    data: BitVec::new(),
                    nulls: BitVec::new(),
                });
            }
            if let TypedColumn::Bool { data, nulls } = &mut self.columns[col_idx] {
                // Batch insert bools
                for v in values {
                    data.push(v);
                }
                nulls.extend_false(batch_size);
            }
        }
        
        // Binary columns (stored as Mixed for now)
        for (name, values) in binary_cols {
            if values.len() != batch_size {
                continue;
            }
            let col_idx = self.schema.add_column(&name, DataType::Binary);
            while col_idx >= self.columns.len() {
                self.columns.push(TypedColumn::Mixed {
                    data: Vec::new(),
                    nulls: BitVec::new(),
                });
            }
            if let TypedColumn::Mixed { data, nulls } = &mut self.columns[col_idx] {
                data.reserve(batch_size);
                for v in values {
                    data.push(Value::Binary(v));
                }
                nulls.extend_false(batch_size);
            }
        }
        
        // Fill any columns not provided with nulls
        for col in &mut self.columns {
            let target_len = self.row_count + batch_size;
            while col.len() < target_len {
                col.push_null();
            }
        }
        
        self.row_count += batch_size;
        self.active_count += batch_size;
        Ok(result_ids)
    }

    /// Ultra-fast typed column insert - no ID return overhead
    /// 
    /// This is the absolute fastest write path, achieving ~0.1ms for 10k rows
    /// by eliminating ID collection and return.
    pub fn insert_typed_columns_no_return(
        &mut self,
        int_cols: HashMap<String, Vec<i64>>,
        float_cols: HashMap<String, Vec<f64>>,
        string_cols: HashMap<String, Vec<String>>,
        bool_cols: HashMap<String, Vec<bool>>,
        binary_cols: HashMap<String, Vec<Vec<u8>>>,
    ) -> Result<()> {
        // Determine batch size from first non-empty column
        let batch_size = int_cols.values().next().map(|v| v.len())
            .or_else(|| float_cols.values().next().map(|v| v.len()))
            .or_else(|| string_cols.values().next().map(|v| v.len()))
            .or_else(|| bool_cols.values().next().map(|v| v.len()))
            .or_else(|| binary_cols.values().next().map(|v| v.len()))
            .unwrap_or(0);
        
        if batch_size == 0 {
            return Ok(());
        }
        
        // Flush any pending writes
        self.flush_write_buffer();
        
        // Extend deleted marker - no ID generation needed
        self.deleted.extend_false(batch_size);
        
        // Direct typed insert - int64 columns
        for (name, mut values) in int_cols {
            if values.len() != batch_size { continue; }
            let col_idx = self.schema.add_column(&name, DataType::Int64);
            while col_idx >= self.columns.len() {
                self.columns.push(TypedColumn::Int64 {
                    data: Vec::new(),
                    nulls: BitVec::new(),
                });
            }
            if let TypedColumn::Int64 { data, nulls } = &mut self.columns[col_idx] {
                data.reserve(batch_size);
                data.append(&mut values);
                nulls.extend_false(batch_size);
            }
        }
        
        // Float64 columns
        for (name, mut values) in float_cols {
            if values.len() != batch_size { continue; }
            let col_idx = self.schema.add_column(&name, DataType::Float64);
            while col_idx >= self.columns.len() {
                self.columns.push(TypedColumn::Float64 {
                    data: Vec::new(),
                    nulls: BitVec::new(),
                });
            }
            if let TypedColumn::Float64 { data, nulls } = &mut self.columns[col_idx] {
                data.reserve(batch_size);
                data.append(&mut values);
                nulls.extend_false(batch_size);
            }
        }
        
        // String columns
        for (name, mut values) in string_cols {
            if values.len() != batch_size { continue; }
            let col_idx = self.schema.add_column(&name, DataType::String);
            while col_idx >= self.columns.len() {
                self.columns.push(TypedColumn::String {
                    data: Vec::new(),
                    nulls: BitVec::new(),
                });
            }
            if let TypedColumn::String { data, nulls } = &mut self.columns[col_idx] {
                data.reserve(batch_size);
                data.append(&mut values);
                nulls.extend_false(batch_size);
            }
        }
        
        // Bool columns
        for (name, values) in bool_cols {
            if values.len() != batch_size { continue; }
            let col_idx = self.schema.add_column(&name, DataType::Bool);
            while col_idx >= self.columns.len() {
                self.columns.push(TypedColumn::Bool {
                    data: BitVec::new(),
                    nulls: BitVec::new(),
                });
            }
            if let TypedColumn::Bool { data, nulls } = &mut self.columns[col_idx] {
                for v in values {
                    data.push(v);
                }
                nulls.extend_false(batch_size);
            }
        }
        
        // Binary columns
        for (name, values) in binary_cols {
            if values.len() != batch_size { continue; }
            let col_idx = self.schema.add_column(&name, DataType::Binary);
            while col_idx >= self.columns.len() {
                self.columns.push(TypedColumn::Mixed {
                    data: Vec::new(),
                    nulls: BitVec::new(),
                });
            }
            if let TypedColumn::Mixed { data, nulls } = &mut self.columns[col_idx] {
                data.reserve(batch_size);
                for v in values {
                    data.push(Value::Binary(v));
                }
                nulls.extend_false(batch_size);
            }
        }
        
        // Fill missing columns
        let target_len = self.row_count + batch_size;
        for col in &mut self.columns {
            while col.len() < target_len {
                col.push_null();
            }
        }
        
        self.row_count += batch_size;
        self.active_count += batch_size;
        Ok(())
    }

    /// ULTRA-FAST unsafe batch insert - absolute minimum overhead
    /// 
    /// Uses unsafe operations to eliminate bounds checking and achieve
    /// near-memcpy speeds. Safety: caller must ensure data validity.
    /// 
    /// Performance: ~0.1ms for 10k rows (pure numeric)
    #[allow(clippy::uninit_vec)]
    pub unsafe fn insert_typed_columns_unsafe(
        &mut self,
        int_cols: HashMap<String, Vec<i64>>,
        float_cols: HashMap<String, Vec<f64>>,
        bool_cols: HashMap<String, Vec<bool>>,
    ) -> Result<usize> {
        // Determine batch size
        let batch_size = int_cols.values().next().map(|v| v.len())
            .or_else(|| float_cols.values().next().map(|v| v.len()))
            .or_else(|| bool_cols.values().next().map(|v| v.len()))
            .unwrap_or(0);
        
        if batch_size == 0 {
            return Ok(0);
        }
        
        // Flush pending writes
        self.flush_write_buffer();
        
        // Pre-extend deleted marker using unsafe set_len
        {
            let words_needed = (self.deleted.len + batch_size + 63) / 64;
            if words_needed > self.deleted.data.len() {
                let new_capacity = words_needed.max(self.deleted.data.capacity() * 2);
                self.deleted.data.reserve(new_capacity - self.deleted.data.len());
                // Zero-fill new words (false = 0)
                self.deleted.data.resize(words_needed, 0);
            }
            self.deleted.len += batch_size;
        }
        
        // Process int64 columns - direct memory copy
        for (name, values) in int_cols {
            if values.len() != batch_size { continue; }
            
            let col_idx = self.schema.add_column(&name, DataType::Int64);
            while col_idx >= self.columns.len() {
                self.columns.push(TypedColumn::Int64 {
                    data: Vec::new(),
                    nulls: BitVec::new(),
                });
            }
            
            if let TypedColumn::Int64 { data, nulls } = &mut self.columns[col_idx] {
                let old_len = data.len();
                let new_len = old_len + batch_size;
                
                // Reserve and copy in one operation
                data.reserve(batch_size);
                
                // UNSAFE: Direct memory copy without bounds checks
                let src_ptr = values.as_ptr();
                let dst_ptr = data.as_mut_ptr().add(old_len);
                std::ptr::copy_nonoverlapping(src_ptr, dst_ptr, batch_size);
                data.set_len(new_len);
                
                // Extend nulls bitmap
                nulls.extend_false(batch_size);
            }
        }
        
        // Process float64 columns
        for (name, values) in float_cols {
            if values.len() != batch_size { continue; }
            
            let col_idx = self.schema.add_column(&name, DataType::Float64);
            while col_idx >= self.columns.len() {
                self.columns.push(TypedColumn::Float64 {
                    data: Vec::new(),
                    nulls: BitVec::new(),
                });
            }
            
            if let TypedColumn::Float64 { data, nulls } = &mut self.columns[col_idx] {
                let old_len = data.len();
                let new_len = old_len + batch_size;
                
                data.reserve(batch_size);
                
                // UNSAFE: Direct memory copy
                let src_ptr = values.as_ptr();
                let dst_ptr = data.as_mut_ptr().add(old_len);
                std::ptr::copy_nonoverlapping(src_ptr, dst_ptr, batch_size);
                data.set_len(new_len);
                
                nulls.extend_false(batch_size);
            }
        }
        
        // Process bool columns - need to convert to BitVec
        for (name, values) in bool_cols {
            if values.len() != batch_size { continue; }
            
            let col_idx = self.schema.add_column(&name, DataType::Bool);
            while col_idx >= self.columns.len() {
                self.columns.push(TypedColumn::Bool {
                    data: BitVec::new(),
                    nulls: BitVec::new(),
                });
            }
            
            if let TypedColumn::Bool { data, nulls } = &mut self.columns[col_idx] {
                // Batch push booleans - less optimized but still fast
                data.extend_from_bools(&values);
                nulls.extend_false(batch_size);
            }
        }
        
        // Fill missing columns
        let target_len = self.row_count + batch_size;
        for col in &mut self.columns {
            while col.len() < target_len {
                col.push_null();
            }
        }
        
        self.row_count += batch_size;
        self.active_count += batch_size;
        Ok(batch_size)
    }

    /// Get a row by ID - O(1) direct array access
    /// ID = row index (0-based)
    pub fn get(&mut self, id: u64) -> Option<HashMap<String, Value>> {
        let row_idx = id as usize;
        
        // Check write buffer first (for pending writes)
        let buffer_start = self.row_count;
        if row_idx >= buffer_start {
            let buffer_idx = row_idx - buffer_start;
            if buffer_idx < self.write_buffer.rows.len() {
                let (_, fields) = &self.write_buffer.rows[buffer_idx];
                let mut result = fields.clone();
                result.insert("_id".to_string(), Value::Int64(id as i64));
                return Some(result);
            }
            return None;
        }
        
        // Direct array access - no HashMap lookup!
        if row_idx >= self.row_count || self.deleted.get(row_idx) {
            return None;
        }
        
        let mut result = HashMap::new();
        result.insert("_id".to_string(), Value::Int64(id as i64));
        
        for (col_idx, (name, _)) in self.schema.columns.iter().enumerate() {
            if let Some(value) = self.columns[col_idx].get(row_idx) {
                if !value.is_null() {
                    result.insert(name.clone(), value);
                }
            }
        }
        
        Some(result)
    }

    /// Get multiple rows by IDs
    pub fn get_many(&mut self, ids: &[u64]) -> Vec<HashMap<String, Value>> {
        ids.iter()
            .filter_map(|id| self.get(*id))
            .collect()
    }

    /// ULTRA-FAST retrieve all records as Arrow IPC - bypasses query parsing and cloning
    /// 
    /// Performance optimizations:
    /// - No query parsing overhead
    /// - No column cloning - direct reference access
    /// - Contiguous range detection for zero-copy paths
    /// - Parallel column conversion for large datasets
    /// 
    /// Target: 10M rows in <500ms
    pub fn retrieve_all_arrow_direct(&mut self) -> std::result::Result<Vec<u8>, String> {
        // Flush pending writes to ensure data consistency
        self.flush_write_buffer();
        
        if self.row_count == 0 {
            return Ok(Vec::new());
        }
        
        // Build contiguous row indices (excluding deleted rows)
        let row_indices: Vec<usize> = if self.deleted.count_ones() == 0 {
            // Fast path: no deletions, use contiguous range
            (0..self.row_count).collect()
        } else {
            // Some deletions: filter out deleted rows
            (0..self.row_count)
                .filter(|&idx| !self.deleted.get(idx))
                .collect()
        };
        
        if row_indices.is_empty() {
            return Ok(Vec::new());
        }
        
        // Get column names
        let column_names: Vec<String> = self.schema.columns.iter()
            .map(|(name, _)| name.clone())
            .collect();
        
        // Generate ID array (row index = ID)
        let id_array: Vec<u64> = (0..self.row_count as u64).collect();
        
        // Call the optimized Arrow conversion with direct column references
        crate::data::typed_columns_to_arrow_ipc(
            &id_array,
            &self.columns,
            &column_names,
            &row_indices,
        )
    }

    /// Get multiple rows by IDs and return as Arrow IPC bytes - OPTIMIZED PATH
    /// This method bypasses HashMap and Row creation entirely for maximum performance
    /// Returns Arrow IPC format bytes directly from columnar storage
    pub fn get_many_arrow(&mut self, ids: &[u64]) -> std::result::Result<Vec<u8>, String> {
        if ids.is_empty() {
            return Ok(Vec::new());
        }

        // Flush pending writes to ensure data consistency
        self.flush_write_buffer();

        // Filter valid row indices (not deleted, within bounds)
        let row_indices: Vec<usize> = ids.iter()
            .map(|&id| id as usize)
            .filter(|&idx| idx < self.row_count && !self.deleted.get(idx))
            .collect();

        if row_indices.is_empty() {
            return Ok(Vec::new());
        }

        // Get column names
        let column_names: Vec<String> = self.schema.columns.iter()
            .map(|(name, _)| name.clone())
            .collect();

        // Generate ID array (row index = ID)
        let id_array: Vec<u64> = (0..self.row_count as u64).collect();

        // Call the optimized Arrow conversion
        crate::data::typed_columns_to_arrow_ipc(
            &id_array,
            &self.columns,
            &column_names,
            &row_indices,
        )
    }

    /// Get multiple rows by IDs and return as Arrow RecordBatch - ZERO-COPY FFI PATH
    /// This method returns a RecordBatch directly for Arrow C Data Interface export
    /// Used by _retrieve_many_arrow_ffi for maximum performance
    pub fn get_many_record_batch(&mut self, ids: &[u64]) -> std::result::Result<arrow::array::RecordBatch, String> {
        use arrow::array::RecordBatch;
        use arrow::datatypes::{DataType as ArrowDataType, Field, Schema};
        use std::sync::Arc;

        if ids.is_empty() {
            // Return empty batch
            let schema = Arc::new(Schema::new(vec![
                Field::new("_id", ArrowDataType::UInt64, false),
            ]));
            return RecordBatch::try_new(schema, vec![
                Arc::new(arrow::array::UInt64Array::from(Vec::<u64>::new()))
            ]).map_err(|e| e.to_string());
        }

        // Flush pending writes to ensure data consistency
        self.flush_write_buffer();

        // Filter valid row indices (not deleted, within bounds)
        let row_indices: Vec<usize> = ids.iter()
            .map(|&id| id as usize)
            .filter(|&idx| idx < self.row_count && !self.deleted.get(idx))
            .collect();

        if row_indices.is_empty() {
            // Return empty batch
            let schema = Arc::new(Schema::new(vec![
                Field::new("_id", ArrowDataType::UInt64, false),
            ]));
            return RecordBatch::try_new(schema, vec![
                Arc::new(arrow::array::UInt64Array::from(Vec::<u64>::new()))
            ]).map_err(|e| e.to_string());
        }

        // Get column names
        let column_names: Vec<String> = self.schema.columns.iter()
            .map(|(name, _)| name.clone())
            .collect();

        // Generate ID array (row index = ID)
        let id_array: Vec<u64> = (0..self.row_count as u64).collect();

        // Call the optimized RecordBatch builder
        crate::data::build_record_batch_direct(
            &id_array,
            &self.columns,
            &column_names,
            &row_indices,
        )
    }

    /// Delete a row - O(1) soft delete
    /// ID = row index (0-based)
    pub fn delete(&mut self, id: u64) -> bool {
        let row_idx = id as usize;
        if row_idx < self.row_count && !self.deleted.get(row_idx) {
            self.deleted.set(row_idx, true);
            self.active_count -= 1;
            return true;
        }
        false
    }

    /// Delete multiple rows
    pub fn delete_batch(&mut self, ids: &[u64]) -> bool {
        let mut all_deleted = true;
        for &id in ids {
            if !self.delete(id) {
                all_deleted = false;
            }
        }
        all_deleted
    }

    /// Update a row - O(1) direct access + O(fields) update
    /// ID = row index (0-based)
    pub fn update(&mut self, id: u64, fields: &HashMap<String, Value>) -> bool {
        let row_idx = id as usize;
        if row_idx >= self.row_count || self.deleted.get(row_idx) {
            return false;
        }
        
        for (name, value) in fields {
            let col_idx = self.ensure_column(name, value.data_type());
            // Extend column if needed
            while self.columns[col_idx].len() <= row_idx {
                self.columns[col_idx].push_null();
            }
            self.columns[col_idx].set(row_idx, value);
        }
        
        true
    }

    /// Query with filter - returns row data directly (no cloning overhead)
    pub fn query(&mut self, where_clause: &str) -> Result<Vec<HashMap<String, Value>>> {
        self.query_with_limit(where_clause, None)
    }
    
    /// Query with filter and optional limit - uses streaming early termination
    pub fn query_with_limit(&mut self, where_clause: &str, limit: Option<usize>) -> Result<Vec<HashMap<String, Value>>> {
        // Flush write buffer before querying
        self.flush_write_buffer();
        
        let executor = QueryExecutor::new();
        let filter = executor.parse(where_clause)?;
        
        // If limit is specified, use streaming early termination
        if let Some(max_rows) = limit {
            return self.query_streaming(&filter, max_rows);
        }
        
        // Use column-based filtering for better performance
        let matching_indices = filter.filter_columns(
            &self.schema,
            &self.columns,
            self.row_count,
            &self.deleted,
        );
        
        let mut results = Vec::with_capacity(matching_indices.len());
        for row_idx in matching_indices {
            // ID = row index
            results.push(self.build_row(row_idx, row_idx as u64));
        }
        
        Ok(results)
    }
    
    /// Streaming query with early termination - stops after finding `limit` matches
    fn query_streaming(&self, filter: &Filter, limit: usize) -> Result<Vec<HashMap<String, Value>>> {
        let mut results = Vec::with_capacity(limit);
        let mut found = 0;
        
        for row_idx in 0..self.row_count {
            if self.deleted.get(row_idx) {
                continue;
            }
            
            if self.filter_matches_row(filter, row_idx) {
                results.push(self.build_row(row_idx, row_idx as u64));
                found += 1;
                if found >= limit {
                    break;
                }
            }
        }
        
        Ok(results)
    }
    
    /// Check if a single row matches the filter
    #[inline]
    fn filter_matches_row(&self, filter: &Filter, row_idx: usize) -> bool {
        use crate::query::LikeMatcher;
        
        match filter {
            Filter::True => true,
            Filter::False => false,
            Filter::Compare { field, op, value } => {
                if let Some(col_idx) = self.schema.get_index(field) {
                    if let Some(row_val) = self.columns[col_idx].get(row_idx) {
                        return Self::compare_values(&row_val, op, value);
                    }
                }
                false
            }
            Filter::Like { field, pattern } => {
                if let Some(col_idx) = self.schema.get_index(field) {
                    if let TypedColumn::String { data, nulls } = &self.columns[col_idx] {
                        if row_idx < data.len() && !nulls.get(row_idx) {
                            let matcher = LikeMatcher::new(pattern);
                            return matcher.matches(&data[row_idx]);
                        }
                    }
                }
                false
            }
            Filter::Range { field, low, high, low_inclusive, high_inclusive } => {
                if let Some(col_idx) = self.schema.get_index(field) {
                    if let Some(row_val) = self.columns[col_idx].get(row_idx) {
                        let low_ok = if *low_inclusive {
                            row_val >= *low
                        } else {
                            row_val > *low
                        };
                        let high_ok = if *high_inclusive {
                            row_val <= *high
                        } else {
                            row_val < *high
                        };
                        return low_ok && high_ok;
                    }
                }
                false
            }
            Filter::And(filters) => filters.iter().all(|f| self.filter_matches_row(f, row_idx)),
            Filter::Or(filters) => filters.iter().any(|f| self.filter_matches_row(f, row_idx)),
            Filter::Not(inner) => !self.filter_matches_row(inner, row_idx),
            Filter::In { field, values } => {
                if let Some(col_idx) = self.schema.get_index(field) {
                    if let Some(row_val) = self.columns[col_idx].get(row_idx) {
                        return values.contains(&row_val);
                    }
                }
                false
            }
        }
    }
    
    /// Compare two values with the given operator
    #[inline]
    fn compare_values(row_val: &Value, op: &crate::query::CompareOp, target: &Value) -> bool {
        use crate::query::CompareOp;
        match op {
            CompareOp::Equal => row_val == target,
            CompareOp::NotEqual => row_val != target,
            CompareOp::LessThan => row_val < target,
            CompareOp::LessEqual => row_val <= target,
            CompareOp::GreaterThan => row_val > target,
            CompareOp::GreaterEqual => row_val >= target,
            _ => false,
        }
    }

    /// Query returning only IDs (predicate pushdown optimization)
    /// 
    /// This is the most efficient way to get matching row IDs:
    /// - Only evaluates filter conditions
    /// - Skips full row construction
    /// - Returns IDs directly without data column reads
    /// 
    /// Used by get_ids() method for high-performance ID retrieval
    pub fn query_ids_only(&self, where_clause: &str) -> Result<Vec<u64>> {
        let executor = QueryExecutor::new();
        let filter = executor.parse(where_clause)?;
        
        // Use column-based filtering (only reads columns needed for filter)
        let matching_indices = filter.filter_columns(
            &self.schema,
            &self.columns,
            self.row_count,
            &self.deleted,
        );
        
        // IDs = row indices (direct conversion, no data column reads)
        Ok(matching_indices.iter().map(|&i| i as u64).collect())
    }

    /// Query returning column data directly (fastest for Python)
    /// Returns: (column_names, column_data) where column_data is typed arrays
    pub fn query_columnar(&mut self, where_clause: &str) -> Result<QueryColumnarResult> {
        // Flush write buffer before querying
        self.flush_write_buffer();
        
        let executor = QueryExecutor::new();
        let filter = executor.parse(where_clause)?;
        
        // Use column-based filtering for all cases
        let matching_indices = filter.filter_columns(
            &self.schema,
            &self.columns,
            self.row_count,
            &self.deleted,
        );
        
        // IDs = row indices
        let result_ids: Vec<u64> = matching_indices.iter().map(|&i| i as u64).collect();
        
        Ok(QueryColumnarResult {
            ids: result_ids,
            columns: self.columns.clone(), // TODO: could return references
            schema: self.schema.clone(),
            row_indices: matching_indices,
        })
    }

    /// ULTRA-OPTIMIZED: Query and build Arrow RecordBatch with zero-copy fast paths
    /// 
    /// Performance optimizations:
    /// 1. Contiguous range detection: O(1) check, enables slice-based construction
    /// 2. Full table scan: Direct column conversion without gather
    /// 3. Parallel string building with chunked processing
    /// 4. Pre-allocated buffers to avoid reallocation
    pub fn query_to_record_batch(&mut self, where_clause: &str) -> Result<arrow::record_batch::RecordBatch> {
        use arrow::array::{ArrayRef, BooleanArray, Float64Array, Int64Array, StringBuilder, UInt64Array};
        use arrow::datatypes::{DataType as ArrowDataType, Field, Schema};
        use arrow::buffer::{NullBuffer, ScalarBuffer};
        use std::sync::Arc;
        use rayon::prelude::*;

        // Flush write buffer before querying
        self.flush_write_buffer();
        
        let executor = QueryExecutor::new();
        let filter = executor.parse(where_clause)?;
        
        // Get matching row indices (parallel filtering)
        let matching_indices = filter.filter_columns(
            &self.schema,
            &self.columns,
            self.row_count,
            &self.deleted,
        );
        
        let num_rows = matching_indices.len();
        if num_rows == 0 {
            // Return empty batch
            let schema = Arc::new(Schema::new(vec![
                Field::new("_id", ArrowDataType::UInt64, false),
            ]));
            return arrow::record_batch::RecordBatch::try_new(schema, vec![
                Arc::new(UInt64Array::from(Vec::<u64>::new()))
            ]).map_err(|e| crate::ApexError::SerializationError(e.to_string()));
        }
        
        // OPTIMIZATION: Check for contiguous range [0, N) - enables fast paths
        let is_contiguous_from_zero = num_rows > 0 && 
            matching_indices[0] == 0 && 
            matching_indices[num_rows - 1] == num_rows - 1 &&
            (num_rows < 3 || matching_indices[num_rows / 2] == num_rows / 2);
        
        // Build schema
        let mut fields = vec![Field::new("_id", ArrowDataType::UInt64, false)];
        for (name, dtype) in &self.schema.columns {
            let arrow_type = match dtype {
                DataType::Int64 | DataType::Int32 | DataType::Int16 | DataType::Int8 => ArrowDataType::Int64,
                DataType::Float64 | DataType::Float32 => ArrowDataType::Float64,
                DataType::Bool => ArrowDataType::Boolean,
                DataType::String => ArrowDataType::Utf8,
                _ => ArrowDataType::Utf8,
            };
            fields.push(Field::new(name, arrow_type, true));
        }
        let schema = Arc::new(Schema::new(fields));
        
        // Build _id array (IDs = row indices)
        let id_array: ArrayRef = if is_contiguous_from_zero {
            // Fast path: sequential IDs from 0 to num_rows-1
            Arc::new(UInt64Array::from_iter_values(0..num_rows as u64))
        } else {
            Arc::new(UInt64Array::from_iter_values(
                matching_indices.iter().map(|&i| i as u64)
            ))
        };
        
        // Build data column arrays - choose strategy based on result size
        let data_arrays: Vec<ArrayRef> = if is_contiguous_from_zero {
            // FAST PATH: Contiguous range - use direct slice conversion
            self.columns.par_iter()
                .map(|col| -> ArrayRef {
                    match col {
                        TypedColumn::Int64 { data, nulls } => {
                            let slice = &data[..num_rows];
                            if nulls.all_false() {
                                Arc::new(Int64Array::new(ScalarBuffer::from(slice.to_vec()), None))
                            } else {
                                let null_bits: Vec<bool> = (0..num_rows).map(|i| !nulls.get(i)).collect();
                                Arc::new(Int64Array::new(ScalarBuffer::from(slice.to_vec()), Some(NullBuffer::from(null_bits))))
                            }
                        }
                        TypedColumn::Float64 { data, nulls } => {
                            let slice = &data[..num_rows];
                            if nulls.all_false() {
                                Arc::new(Float64Array::new(ScalarBuffer::from(slice.to_vec()), None))
                            } else {
                                let null_bits: Vec<bool> = (0..num_rows).map(|i| !nulls.get(i)).collect();
                                Arc::new(Float64Array::new(ScalarBuffer::from(slice.to_vec()), Some(NullBuffer::from(null_bits))))
                            }
                        }
                        TypedColumn::String { data, nulls } => {
                            // OPTIMIZED: Build string array with pre-computed offsets
                            Self::build_string_array_contiguous(&data[..num_rows], nulls, num_rows)
                        }
                        TypedColumn::Bool { data, nulls } => {
                            let values: Vec<Option<bool>> = (0..num_rows)
                                .map(|i| {
                                    if !nulls.get(i) { Some(data.get(i)) } else { None }
                                })
                                .collect();
                            Arc::new(BooleanArray::from(values))
                        }
                        TypedColumn::Mixed { data, nulls } => {
                            let mut builder = StringBuilder::with_capacity(num_rows, num_rows * 32);
                            for i in 0..num_rows {
                                if nulls.get(i) {
                                    builder.append_null();
                                } else {
                                    builder.append_value(data[i].to_string_value());
                                }
                            }
                            Arc::new(builder.finish())
                        }
                    }
                })
                .collect()
        } else {
            // GATHER PATH: Non-contiguous indices
            self.columns.par_iter()
                .map(|col| -> ArrayRef {
                    match col {
                        TypedColumn::Int64 { data, nulls } => {
                            let no_nulls = nulls.all_false();
                            let values: Vec<i64> = matching_indices.iter()
                                .map(|&idx| if idx < data.len() { data[idx] } else { 0 })
                                .collect();
                            if no_nulls {
                                Arc::new(Int64Array::new(ScalarBuffer::from(values), None))
                            } else {
                                let null_bits: Vec<bool> = matching_indices.iter()
                                    .map(|&idx| !nulls.get(idx))
                                    .collect();
                                Arc::new(Int64Array::new(ScalarBuffer::from(values), Some(NullBuffer::from(null_bits))))
                            }
                        }
                        TypedColumn::Float64 { data, nulls } => {
                            let no_nulls = nulls.all_false();
                            let values: Vec<f64> = matching_indices.iter()
                                .map(|&idx| if idx < data.len() { data[idx] } else { 0.0 })
                                .collect();
                            if no_nulls {
                                Arc::new(Float64Array::new(ScalarBuffer::from(values), None))
                            } else {
                                let null_bits: Vec<bool> = matching_indices.iter()
                                    .map(|&idx| !nulls.get(idx))
                                    .collect();
                                Arc::new(Float64Array::new(ScalarBuffer::from(values), Some(NullBuffer::from(null_bits))))
                            }
                        }
                        TypedColumn::String { data, nulls } => {
                            Self::build_string_array_gather(data, nulls, &matching_indices)
                        }
                        TypedColumn::Bool { data, nulls } => {
                            let values: Vec<Option<bool>> = matching_indices.iter()
                                .map(|&idx| {
                                    if idx < data.len() && !nulls.get(idx) {
                                        Some(data.get(idx))
                                    } else {
                                        None
                                    }
                                })
                                .collect();
                            Arc::new(BooleanArray::from(values))
                        }
                        TypedColumn::Mixed { data, nulls } => {
                            let mut builder = StringBuilder::with_capacity(num_rows, num_rows * 32);
                            for &idx in &matching_indices {
                                if idx >= data.len() || nulls.get(idx) {
                                    builder.append_null();
                                } else {
                                    builder.append_value(data[idx].to_string_value());
                                }
                            }
                            Arc::new(builder.finish())
                        }
                    }
                })
                .collect()
        };
        
        // Combine arrays
        let mut arrays = Vec::with_capacity(self.columns.len() + 1);
        arrays.push(id_array);
        arrays.extend(data_arrays);
        
        arrow::record_batch::RecordBatch::try_new(schema, arrays)
            .map_err(|e| crate::ApexError::SerializationError(e.to_string()))
    }
    
    /// Build Arrow RecordBatch from pre-computed row indices
    /// This is optimized for cases where filtering has already been done
    pub fn build_record_batch_from_indices(&self, indices: &[usize]) -> Result<arrow::record_batch::RecordBatch> {
        use arrow::array::{ArrayRef, BooleanArray, Float64Array, Int64Array, UInt64Array};
        use arrow::datatypes::{DataType as ArrowDataType, Field, Schema};
        use arrow::buffer::{NullBuffer, ScalarBuffer};
        use std::sync::Arc;
        use rayon::prelude::*;

        let num_rows = indices.len();
        if num_rows == 0 {
            let schema = Arc::new(Schema::new(vec![
                Field::new("_id", ArrowDataType::UInt64, false),
            ]));
            return arrow::record_batch::RecordBatch::try_new(schema, vec![
                Arc::new(UInt64Array::from(Vec::<u64>::new()))
            ]).map_err(|e| crate::ApexError::SerializationError(e.to_string()));
        }

        // Build schema
        let mut fields = vec![Field::new("_id", ArrowDataType::UInt64, false)];
        for (name, dtype) in &self.schema.columns {
            let arrow_type = match dtype {
                DataType::Int64 | DataType::Int32 | DataType::Int16 | DataType::Int8 => ArrowDataType::Int64,
                DataType::Float64 | DataType::Float32 => ArrowDataType::Float64,
                DataType::Bool => ArrowDataType::Boolean,
                DataType::String => ArrowDataType::Utf8,
                _ => ArrowDataType::Utf8,
            };
            fields.push(Field::new(name, arrow_type, true));
        }
        let schema = Arc::new(Schema::new(fields));

        // Build _id array
        let id_array: ArrayRef = Arc::new(UInt64Array::from_iter_values(
            indices.iter().map(|&i| i as u64)
        ));

        // Build data column arrays in parallel
        let data_arrays: Vec<ArrayRef> = self.columns.par_iter()
            .map(|col| -> ArrayRef {
                match col {
                    TypedColumn::Int64 { data, nulls } => {
                        let no_nulls = nulls.all_false();
                        let values: Vec<i64> = indices.iter()
                            .map(|&idx| if idx < data.len() { data[idx] } else { 0 })
                            .collect();
                        if no_nulls {
                            Arc::new(Int64Array::new(ScalarBuffer::from(values), None))
                        } else {
                            let null_bits: Vec<bool> = indices.iter()
                                .map(|&idx| !nulls.get(idx))
                                .collect();
                            Arc::new(Int64Array::new(ScalarBuffer::from(values), Some(NullBuffer::from(null_bits))))
                        }
                    }
                    TypedColumn::Float64 { data, nulls } => {
                        let no_nulls = nulls.all_false();
                        let values: Vec<f64> = indices.iter()
                            .map(|&idx| if idx < data.len() { data[idx] } else { 0.0 })
                            .collect();
                        if no_nulls {
                            Arc::new(Float64Array::new(ScalarBuffer::from(values), None))
                        } else {
                            let null_bits: Vec<bool> = indices.iter()
                                .map(|&idx| !nulls.get(idx))
                                .collect();
                            Arc::new(Float64Array::new(ScalarBuffer::from(values), Some(NullBuffer::from(null_bits))))
                        }
                    }
                    TypedColumn::String { data, nulls } => {
                        Self::build_string_array_gather(data, nulls, indices)
                    }
                    TypedColumn::Bool { data, nulls } => {
                        let values: Vec<Option<bool>> = indices.iter()
                            .map(|&i| {
                                if !nulls.get(i) { Some(data.get(i)) } else { None }
                            })
                            .collect();
                        Arc::new(BooleanArray::from(values))
                    }
                    TypedColumn::Mixed { data, nulls } => {
                        use arrow::array::StringBuilder;
                        let mut builder = StringBuilder::with_capacity(num_rows, num_rows * 32);
                        for &i in indices {
                            if nulls.get(i) {
                                builder.append_null();
                            } else {
                                builder.append_value(data[i].to_string_value());
                            }
                        }
                        Arc::new(builder.finish())
                    }
                }
            })
            .collect();

        // Combine arrays
        let mut arrays = Vec::with_capacity(self.columns.len() + 1);
        arrays.push(id_array);
        arrays.extend(data_arrays);

        arrow::record_batch::RecordBatch::try_new(schema, arrays)
            .map_err(|e| crate::ApexError::SerializationError(e.to_string()))
    }

    /// Build Arrow StringArray from contiguous string slice - optimized for sequential access
    /// 
    /// Uses direct buffer construction instead of StringBuilder for better performance.
    /// Pre-calculates total bytes to avoid reallocation.
    #[inline]
    fn build_string_array_contiguous(data: &[String], nulls: &BitVec, num_rows: usize) -> std::sync::Arc<dyn arrow::array::Array> {
        use arrow::array::StringArray;
        use arrow::buffer::{OffsetBuffer, Buffer};
        use std::sync::Arc;
        use rayon::prelude::*;
        
        let no_nulls = nulls.all_false();
        
        // Parallel size calculation for large datasets
        let total_bytes: usize = if num_rows >= 100_000 {
            const CHUNK_SIZE: usize = 50_000;
            let num_chunks = (num_rows + CHUNK_SIZE - 1) / CHUNK_SIZE;
            (0..num_chunks)
                .into_par_iter()
                .map(|chunk_idx| {
                    let start = chunk_idx * CHUNK_SIZE;
                    let end = (start + CHUNK_SIZE).min(num_rows);
                    data[start..end].iter().map(|s| s.len()).sum::<usize>()
                })
                .sum()
        } else {
            data.iter().map(|s| s.len()).sum()
        };
        
        if no_nulls {
            // FAST PATH: No nulls - build directly from data buffer
            let mut value_buffer = Vec::with_capacity(total_bytes);
            let mut offsets: Vec<i32> = Vec::with_capacity(num_rows + 1);
            offsets.push(0);
            
            for s in data {
                value_buffer.extend_from_slice(s.as_bytes());
                offsets.push(value_buffer.len() as i32);
            }
            
            let offset_buffer = OffsetBuffer::new(offsets.into());
            let value_buf = Buffer::from_vec(value_buffer);
            
            Arc::new(unsafe {
                StringArray::new_unchecked(offset_buffer, value_buf, None)
            })
        } else {
            // Has nulls - use builder
            use arrow::array::StringBuilder;
            let mut builder = StringBuilder::with_capacity(num_rows, total_bytes);
            for (i, s) in data.iter().enumerate() {
                if nulls.get(i) {
                    builder.append_null();
                } else {
                    builder.append_value(s);
                }
            }
            Arc::new(builder.finish())
        }
    }
    
    /// Build Arrow StringArray with gather from non-contiguous indices
    #[inline]
    fn build_string_array_gather(data: &[String], nulls: &BitVec, indices: &[usize]) -> std::sync::Arc<dyn arrow::array::Array> {
        use arrow::array::{StringBuilder, StringArray};
        use arrow::buffer::{OffsetBuffer, Buffer};
        use std::sync::Arc;
        
        let num_rows = indices.len();
        let no_nulls = nulls.all_false();
        
        // Pre-calculate total bytes
        let total_bytes: usize = indices.iter()
            .filter(|&&idx| idx < data.len())
            .map(|&idx| data[idx].len())
            .sum();
        
        if no_nulls {
            // FAST PATH: No nulls - build directly from gathered data
            let mut value_buffer = Vec::with_capacity(total_bytes);
            let mut offsets: Vec<i32> = Vec::with_capacity(num_rows + 1);
            offsets.push(0);
            
            for &idx in indices {
                if idx < data.len() {
                    value_buffer.extend_from_slice(data[idx].as_bytes());
                }
                offsets.push(value_buffer.len() as i32);
            }
            
            let offset_buffer = OffsetBuffer::new(offsets.into());
            let value_buf = Buffer::from_vec(value_buffer);
            
            Arc::new(unsafe {
                StringArray::new_unchecked(offset_buffer, value_buf, None)
            })
        } else {
            // Has nulls - use builder
            let mut builder = StringBuilder::with_capacity(num_rows, total_bytes);
            for &idx in indices {
                if idx >= data.len() || nulls.get(idx) {
                    builder.append_null();
                } else {
                    builder.append_value(&data[idx]);
                }
            }
            Arc::new(builder.finish())
        }
    }

    /// Ensure a column exists, creating it if necessary
    fn ensure_column(&mut self, name: &str, dtype: DataType) -> usize {
        if let Some(idx) = self.schema.get_index(name) {
            return idx;
        }
        
        let idx = self.schema.add_column(name, dtype);
        let mut col = TypedColumn::with_capacity(dtype, self.row_count);
        
        // Fill with nulls for existing rows
        for _ in 0..self.row_count {
            col.push_null();
        }
        
        self.columns.push(col);
        idx
    }

    /// Build a row HashMap from column data
    #[inline]
    fn build_row(&self, row_idx: usize, id: u64) -> HashMap<String, Value> {
        let mut result = HashMap::with_capacity(self.schema.len() + 1);
        result.insert("_id".to_string(), Value::Int64(id as i64));
        
        for (col_idx, (name, _)) in self.schema.columns.iter().enumerate() {
            if let Some(value) = self.columns[col_idx].get(row_idx) {
                if !value.is_null() {
                    result.insert(name.clone(), value);
                }
            }
        }
        
        result
    }

    /// Get column names
    pub fn column_names(&self) -> Vec<String> {
        let mut names = vec!["_id".to_string()];
        names.extend(self.schema.column_names());
        names
    }

    /// Compact the table (remove deleted rows)
    /// Note: After compacting, IDs will be reassigned (0 to N-1)
    pub fn compact(&mut self) {
        if self.deleted.count_ones() == 0 {
            return; // Nothing to compact
        }
        
        let mut new_columns: Vec<TypedColumn> = self.columns.iter()
            .map(|c| TypedColumn::with_capacity(c.data_type(), self.active_count))
            .collect();
        
        for old_idx in 0..self.row_count {
            if !self.deleted.get(old_idx) {
                for (col_idx, col) in self.columns.iter().enumerate() {
                    if let Some(value) = col.get(old_idx) {
                        new_columns[col_idx].push(&value);
                    } else {
                        new_columns[col_idx].push_null();
                    }
                }
            }
        }
        
        self.columns = new_columns;
        self.row_count = self.active_count;
        self.deleted = BitVec::new();
        self.deleted.extend_false(self.active_count);
    }

    /// Check if table contains a row (and it's not deleted)
    pub fn contains(&self, id: u64) -> bool {
        let row_idx = id as usize;
        row_idx < self.row_count && !self.deleted.get(row_idx)
    }

    // ========== Persistence support ==========

    /// Get reference to schema
    pub fn schema_ref(&self) -> &ColumnSchema {
        &self.schema
    }

    /// Get all row IDs (0 to row_count-1, excluding deleted)
    pub fn get_all_ids(&self) -> Vec<u64> {
        (0..self.row_count as u64)
            .filter(|&id| !self.deleted.get(id as usize))
            .collect()
    }

    /// Get reference to columns
    pub fn columns_ref(&self) -> &Vec<TypedColumn> {
        &self.columns
    }
    
    /// Get reference to deleted bitmap
    pub fn deleted_ref(&self) -> &BitVec {
        &self.deleted
    }

    /// Get total row count
    pub fn get_row_count(&self) -> usize {
        self.row_count
    }

    /// Slice columns for delta extraction (incremental persistence)
    /// Returns only rows in range [start, end)
    pub fn slice_columns(&self, start: usize, end: usize) -> Vec<TypedColumn> {
        self.columns.iter().map(|col| col.slice(start, end)).collect()
    }

    /// Append columns from delta record (for loading incremental data)
    pub fn append_columns(&mut self, delta_columns: Vec<TypedColumn>) {
        if delta_columns.is_empty() {
            return;
        }
        
        let delta_len = delta_columns.first().map(|c| c.len()).unwrap_or(0);
        
        // Append each delta column to existing columns
        for (col_idx, delta_col) in delta_columns.into_iter().enumerate() {
            if col_idx < self.columns.len() {
                self.columns[col_idx].append(delta_col);
            }
        }
        
        // Update row counts
        self.row_count += delta_len;
        self.active_count += delta_len;
        self.deleted.extend_false(delta_len);
    }

    /// Restore table from serialized data
    /// Note: In the new design, IDs are row indices
    pub fn restore_from(
        &mut self,
        schema: ColumnSchema,
        _ids: Vec<u64>,  // Ignored - IDs are row indices
        columns: Vec<TypedColumn>,
        _next_id: u64,   // Ignored - derived from row_count
    ) {
        // Determine row count from first column
        let row_count = columns.first().map(|c| c.len()).unwrap_or(0);
        
        self.schema = schema;
        self.columns = columns;
        self.row_count = row_count;
        self.active_count = row_count;
        self.deleted = BitVec::new();
        self.deleted.extend_false(row_count);
    }
}

/// Result of a columnar query
#[derive(Debug, Clone)]
pub struct QueryColumnarResult {
    pub ids: Vec<u64>,
    pub columns: Vec<TypedColumn>,
    pub schema: ColumnSchema,
    pub row_indices: Vec<usize>,
}

impl QueryColumnarResult {
    /// Convert to Arrow IPC format bytes (zero-copy transfer to Python/PyArrow)
    /// Uses parallel processing for large datasets
    pub fn to_arrow_ipc(&self) -> std::result::Result<Vec<u8>, String> {
        use arrow::array::{ArrayRef, BooleanArray, Float64Array, Int64Array, StringArray, UInt64Array};
        use arrow::datatypes::{DataType as ArrowDataType, Field, Schema};
        use arrow::ipc::writer::StreamWriter;
        use arrow::record_batch::RecordBatch;
        use rayon::prelude::*;
        use std::sync::Arc;
        
        if self.ids.is_empty() {
            return Ok(Vec::new());
        }
        
        let num_rows = self.row_indices.len();
        
        // Build schema: _id + data columns
        let mut fields = vec![Field::new("_id", ArrowDataType::UInt64, false)];
        for (name, dtype) in &self.schema.columns {
            let arrow_type = match dtype {
                DataType::Int64 | DataType::Int32 | DataType::Int16 | DataType::Int8 => ArrowDataType::Int64,
                DataType::Float64 | DataType::Float32 => ArrowDataType::Float64,
                DataType::Bool => ArrowDataType::Boolean,
                DataType::String => ArrowDataType::Utf8,
                _ => ArrowDataType::Utf8,
            };
            fields.push(Field::new(name, arrow_type, true));
        }
        let schema = Arc::new(Schema::new(fields));
        
        // Build _id array (IDs = row indices)
        let id_array: ArrayRef = Arc::new(UInt64Array::from(self.ids.clone()));  // ids are already row indices
        
        // Build data column arrays in parallel
        let data_arrays: Vec<ArrayRef> = self.columns.par_iter()
            .map(|col| -> ArrayRef {
                match col {
                    TypedColumn::Int64 { data, nulls } => {
                        let values: Vec<Option<i64>> = self.row_indices.iter()
                            .map(|&idx| {
                                if idx < data.len() && !nulls.get(idx) {
                                    Some(data[idx])
                                } else {
                                    None
                                }
                            })
                            .collect();
                        Arc::new(Int64Array::from(values))
                    }
                    TypedColumn::Float64 { data, nulls } => {
                        let values: Vec<Option<f64>> = self.row_indices.iter()
                            .map(|&idx| {
                                if idx < data.len() && !nulls.get(idx) {
                                    Some(data[idx])
                                } else {
                                    None
                                }
                            })
                            .collect();
                        Arc::new(Float64Array::from(values))
                    }
                    TypedColumn::String { data, nulls } => {
                        let values: Vec<Option<&str>> = self.row_indices.iter()
                            .map(|&idx| {
                                if idx < data.len() && !nulls.get(idx) {
                                    Some(data[idx].as_str())
                                } else {
                                    None
                                }
                            })
                            .collect();
                        Arc::new(StringArray::from(values))
                    }
                    TypedColumn::Bool { data, nulls } => {
                        let values: Vec<Option<bool>> = self.row_indices.iter()
                            .map(|&idx| {
                                if idx < data.len() && !nulls.get(idx) {
                                    Some(data.get(idx))
                                } else {
                                    None
                                }
                            })
                            .collect();
                        Arc::new(BooleanArray::from(values))
                    }
                    TypedColumn::Mixed { data, nulls } => {
                        let values: Vec<Option<String>> = self.row_indices.iter()
                            .map(|&idx| {
                                if idx < data.len() && !nulls.get(idx) {
                                    Some(data[idx].to_string_value())
                                } else {
                                    None
                                }
                            })
                            .collect();
                        Arc::new(StringArray::from(values.iter().map(|s| s.as_deref()).collect::<Vec<_>>()))
                    }
                }
            })
            .collect();
        
        // Combine arrays
        let mut arrays = vec![id_array];
        arrays.extend(data_arrays);
        
        // Create RecordBatch
        let batch = RecordBatch::try_new(schema.clone(), arrays)
            .map_err(|e| format!("Failed to create RecordBatch: {}", e))?;
        
        // Serialize to IPC format
        let mut buffer = Vec::with_capacity(num_rows * 100); // Pre-allocate
        {
            let mut writer = StreamWriter::try_new(&mut buffer, &schema)
                .map_err(|e| format!("Failed to create StreamWriter: {}", e))?;
            writer.write(&batch)
                .map_err(|e| format!("Failed to write batch: {}", e))?;
            writer.finish()
                .map_err(|e| format!("Failed to finish writer: {}", e))?;
        }
        
        Ok(buffer)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_column_table_insert() {
        let mut table = ColumnTable::new(1, "test");
        
        let mut row1 = HashMap::new();
        row1.insert("name".to_string(), Value::String("Alice".to_string()));
        row1.insert("age".to_string(), Value::Int64(30));
        
        let id1 = table.insert(&row1).unwrap();
        assert_eq!(id1, 1);
        assert_eq!(table.row_count(), 1);
        
        let retrieved = table.get(id1).unwrap();
        assert_eq!(retrieved.get("name"), Some(&Value::String("Alice".to_string())));
        assert_eq!(retrieved.get("age"), Some(&Value::Int64(30)));
    }

    #[test]
    fn test_column_table_batch_insert() {
        let mut table = ColumnTable::new(1, "test");
        
        let rows: Vec<HashMap<String, Value>> = (0..1000)
            .map(|i| {
                let mut row = HashMap::new();
                row.insert("value".to_string(), Value::Int64(i));
                row
            })
            .collect();
        
        let ids = table.insert_batch(&rows).unwrap();
        assert_eq!(ids.len(), 1000);
        assert_eq!(table.row_count(), 1000);
    }

    #[test]
    fn test_column_table_delete() {
        let mut table = ColumnTable::new(1, "test");
        
        let mut row = HashMap::new();
        row.insert("value".to_string(), Value::Int64(42));
        
        let id = table.insert(&row).unwrap();
        assert!(table.contains(id));
        
        table.delete(id);
        assert!(!table.contains(id));
        assert_eq!(table.row_count(), 0);
    }

    #[test]
    fn test_column_table_query() {
        let mut table = ColumnTable::new(1, "test");
        
        for i in 0..100 {
            let mut row = HashMap::new();
            row.insert("value".to_string(), Value::Int64(i));
            table.insert(&row).unwrap();
        }
        
        let results = table.query("value > 90").unwrap();
        assert_eq!(results.len(), 9); // 91-99
    }

    #[test]
    fn test_bitvec() {
        let mut bv = BitVec::new();
        
        for i in 0..100 {
            bv.push(i % 3 == 0);
        }
        
        for i in 0..100 {
            assert_eq!(bv.get(i), i % 3 == 0);
        }
    }
}

