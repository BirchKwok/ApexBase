//! ApexBase V3 On-Demand Columnar Format
//!
//! A custom binary file format supporting:
//! - Column projection: read only required columns
//! - Row range scan: read only required row ranges  
//! - Zero-copy reads via pread/mmap
//! - No external serialization dependencies (bincode-free)
//!
//! File Format:
//! ```text
//! ┌─────────────────────────────────────────────────────────────┐
//! │ Header (256 bytes)                                          │
//! │   - Magic: "APEXV3\0\0" (8 bytes)                          │
//! │   - Version: u32                                            │
//! │   - Flags: u32                                              │
//! │   - Row count: u64                                          │
//! │   - Column count: u32                                       │
//! │   - Row group size: u32 (rows per group, default 65536)    │
//! │   - Schema offset: u64                                      │
//! │   - Column index offset: u64                                │
//! │   - ID column offset: u64                                   │
//! │   - Timestamps, checksum, reserved                          │
//! ├─────────────────────────────────────────────────────────────┤
//! │ Schema Block                                                │
//! │   - For each column: [name_len:u16][name:bytes][type:u8]   │
//! ├─────────────────────────────────────────────────────────────┤
//! │ Column Index (32 bytes per column)                          │
//! │   - data_offset: u64                                        │
//! │   - data_length: u64                                        │
//! │   - null_offset: u64                                        │
//! │   - null_length: u64                                        │
//! ├─────────────────────────────────────────────────────────────┤
//! │ ID Column (contiguous u64 array)                            │
//! ├─────────────────────────────────────────────────────────────┤
//! │ Column Data Blocks                                          │
//! │   Per column: [null_bitmap][column_data]                    │
//! ├─────────────────────────────────────────────────────────────┤
//! │ Footer (24 bytes)                                           │
//! │   - Magic: "APEXEND\0"                                     │
//! │   - Checksum: u32                                           │
//! │   - File size: u64                                          │
//! └─────────────────────────────────────────────────────────────┘
//! ```

use std::collections::HashMap;
use std::fs::{File, OpenOptions};
use std::io::{self, BufWriter, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::cell::RefCell;

use memmap2::Mmap;
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use arrow::record_batch::RecordBatch;
use arrow::array::ArrayRef;

// Thread-local buffer for scattered reads to avoid repeated allocations
thread_local! {
    static SCATTERED_READ_BUF: RefCell<Vec<u8>> = RefCell::new(Vec::with_capacity(8192));
}

// ============================================================================
// Cross-platform file reading (mmap with pread fallback)
// ============================================================================

/// Memory-mapped file cache for fast repeated reads
/// Uses OS page cache for automatic caching
struct MmapCache {
    mmap: Option<Mmap>,
    file_size: u64,
}

impl MmapCache {
    fn new() -> Self {
        Self { mmap: None, file_size: 0 }
    }
    
    /// Get or create mmap for the file
    fn get_or_create(&mut self, file: &File) -> io::Result<&Mmap> {
        let metadata = file.metadata()?;
        let current_size = metadata.len();
        
        // Invalidate cache if file size changed
        if self.mmap.is_none() || self.file_size != current_size {
            if current_size == 0 {
                return Err(io::Error::new(io::ErrorKind::InvalidData, "Empty file"));
            }
            // SAFETY: File must remain open while mmap is in use
            // We ensure this by keeping mmap in the same struct as file
            let mmap = unsafe { Mmap::map(file)? };
            self.mmap = Some(mmap);
            self.file_size = current_size;
        }
        
        Ok(self.mmap.as_ref().unwrap())
    }
    
    /// Read bytes at offset using mmap (zero-copy when possible)
    fn read_at(&mut self, file: &File, buf: &mut [u8], offset: u64) -> io::Result<()> {
        let mmap = self.get_or_create(file)?;
        let start = offset as usize;
        let end = start + buf.len();
        
        if end > mmap.len() {
            return Err(io::Error::new(
                io::ErrorKind::UnexpectedEof,
                format!("Read past EOF: offset={}, len={}, file_size={}", offset, buf.len(), mmap.len())
            ));
        }
        
        buf.copy_from_slice(&mmap[start..end]);
        Ok(())
    }
    
    /// Get a slice directly from mmap (true zero-copy)
    fn slice(&mut self, file: &File, offset: u64, len: usize) -> io::Result<&[u8]> {
        let mmap = self.get_or_create(file)?;
        let start = offset as usize;
        let end = start + len;
        
        if end > mmap.len() {
            return Err(io::Error::new(
                io::ErrorKind::UnexpectedEof,
                format!("Slice past EOF: offset={}, len={}, file_size={}", offset, len, mmap.len())
            ));
        }
        
        Ok(&mmap[start..end])
    }
    
    /// Invalidate cache (call after writes)
    fn invalidate(&mut self) {
        self.mmap = None;
        self.file_size = 0;
    }
}

/// Cross-platform positioned read (fallback for when mmap is not available)
#[cfg(unix)]
fn pread_fallback(file: &File, buf: &mut [u8], offset: u64) -> io::Result<()> {
    use std::os::unix::fs::FileExt;
    file.read_exact_at(buf, offset)
}

#[cfg(windows)]
fn pread_fallback(file: &File, buf: &mut [u8], offset: u64) -> io::Result<()> {
    use std::os::windows::fs::FileExt;
    let mut total_read = 0;
    while total_read < buf.len() {
        let n = file.seek_read(&mut buf[total_read..], offset + total_read as u64)?;
        if n == 0 {
            return Err(io::Error::new(io::ErrorKind::UnexpectedEof, "Unexpected EOF"));
        }
        total_read += n;
    }
    Ok(())
}

#[cfg(not(any(unix, windows)))]
fn pread_fallback(file: &mut File, buf: &mut [u8], offset: u64) -> io::Result<()> {
    // Generic fallback using seek + read (not thread-safe)
    file.seek(SeekFrom::Start(offset))?;
    file.read_exact(buf)
}

// ============================================================================
// Constants
// ============================================================================

const MAGIC_V3: &[u8; 8] = b"APEXV3\0\0";
const MAGIC_FOOTER_V3: &[u8; 8] = b"APEXEND\0";
const FORMAT_VERSION_V3: u32 = 3;
const HEADER_SIZE_V3: usize = 256;
const FOOTER_SIZE_V3: usize = 24;
const COLUMN_INDEX_ENTRY_SIZE: usize = 32;
const DEFAULT_ROW_GROUP_SIZE: u32 = 65536;

// Column type identifiers
const TYPE_NULL: u8 = 0;
const TYPE_BOOL: u8 = 1;
const TYPE_INT8: u8 = 2;
const TYPE_INT16: u8 = 3;
const TYPE_INT32: u8 = 4;
const TYPE_INT64: u8 = 5;
const TYPE_UINT8: u8 = 6;
const TYPE_UINT16: u8 = 7;
const TYPE_UINT32: u8 = 8;
const TYPE_UINT64: u8 = 9;
const TYPE_FLOAT32: u8 = 10;
const TYPE_FLOAT64: u8 = 11;
const TYPE_STRING: u8 = 12;
const TYPE_BINARY: u8 = 13;
const TYPE_STRING_DICT: u8 = 14;  // Dictionary-encoded string (DuckDB-style)

// ============================================================================
// Data Types
// ============================================================================

/// Column data type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[repr(u8)]
pub enum ColumnType {
    Null = TYPE_NULL,
    Bool = TYPE_BOOL,
    Int8 = TYPE_INT8,
    Int16 = TYPE_INT16,
    Int32 = TYPE_INT32,
    Int64 = TYPE_INT64,
    UInt8 = TYPE_UINT8,
    UInt16 = TYPE_UINT16,
    UInt32 = TYPE_UINT32,
    UInt64 = TYPE_UINT64,
    Float32 = TYPE_FLOAT32,
    Float64 = TYPE_FLOAT64,
    String = TYPE_STRING,
    Binary = TYPE_BINARY,
    StringDict = TYPE_STRING_DICT,  // Dictionary-encoded string for low-cardinality columns
}

impl ColumnType {
    pub fn from_u8(v: u8) -> Option<Self> {
        match v {
            TYPE_NULL => Some(ColumnType::Null),
            TYPE_BOOL => Some(ColumnType::Bool),
            TYPE_INT8 => Some(ColumnType::Int8),
            TYPE_INT16 => Some(ColumnType::Int16),
            TYPE_INT32 => Some(ColumnType::Int32),
            TYPE_INT64 => Some(ColumnType::Int64),
            TYPE_UINT8 => Some(ColumnType::UInt8),
            TYPE_UINT16 => Some(ColumnType::UInt16),
            TYPE_UINT32 => Some(ColumnType::UInt32),
            TYPE_UINT64 => Some(ColumnType::UInt64),
            TYPE_FLOAT32 => Some(ColumnType::Float32),
            TYPE_FLOAT64 => Some(ColumnType::Float64),
            TYPE_STRING => Some(ColumnType::String),
            TYPE_BINARY => Some(ColumnType::Binary),
            TYPE_STRING_DICT => Some(ColumnType::StringDict),
            _ => None,
        }
    }

    /// Fixed size in bytes (0 for variable-length types)
    pub fn fixed_size(&self) -> usize {
        match self {
            ColumnType::Null => 0,
            ColumnType::Bool => 1,
            ColumnType::Int8 | ColumnType::UInt8 => 1,
            ColumnType::Int16 | ColumnType::UInt16 => 2,
            ColumnType::Int32 | ColumnType::UInt32 | ColumnType::Float32 => 4,
            ColumnType::Int64 | ColumnType::UInt64 | ColumnType::Float64 => 8,
            ColumnType::String | ColumnType::Binary | ColumnType::StringDict => 0,
        }
    }

    pub fn is_variable_length(&self) -> bool {
        matches!(self, ColumnType::String | ColumnType::Binary | ColumnType::StringDict)
    }
}

/// Generic column value for API
#[derive(Debug, Clone)]
pub enum ColumnValue {
    Null,
    Bool(bool),
    Int64(i64),
    Float64(f64),
    String(String),
    Binary(Vec<u8>),
}

/// Column definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ColumnDef {
    pub name: String,
    pub dtype: ColumnType,
}

/// Schema definition (for API compatibility)
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct FileSchema {
    pub columns: Vec<ColumnDef>,
    name_to_idx: HashMap<String, usize>,
}

impl FileSchema {
    pub fn new() -> Self {
        Self {
            columns: Vec::new(),
            name_to_idx: HashMap::new(),
        }
    }

    pub fn add_column(&mut self, name: &str, dtype: ColumnType) -> usize {
        if let Some(&idx) = self.name_to_idx.get(name) {
            return idx;
        }
        let idx = self.columns.len();
        self.columns.push(ColumnDef {
            name: name.to_string(),
            dtype,
        });
        self.name_to_idx.insert(name.to_string(), idx);
        idx
    }

    pub fn get_index(&self, name: &str) -> Option<usize> {
        self.name_to_idx.get(name).copied()
    }

    pub fn column_count(&self) -> usize {
        self.columns.len()
    }
}

// ============================================================================
// Column Data Storage
// ============================================================================

/// Efficient column data storage
#[derive(Debug, Clone)]
pub enum ColumnData {
    Bool {
        data: Vec<u8>,  // Packed bits
        len: usize,
    },
    Int64(Vec<i64>),
    Float64(Vec<f64>),
    String {
        offsets: Vec<u32>,  // Offset into data
        data: Vec<u8>,      // UTF-8 bytes
    },
    Binary {
        offsets: Vec<u32>,  // Offset into data
        data: Vec<u8>,      // Raw bytes
    },
    /// Dictionary-encoded string column (DuckDB-style optimization)
    /// - indices: u32 index per row pointing into dictionary
    /// - dict_offsets: offset into dict_data for each unique string
    /// - dict_data: concatenated unique string bytes
    /// 
    /// Benefits:
    /// - GROUP BY/DISTINCT work on integer indices instead of string hashing
    /// - Much smaller storage for low-cardinality columns
    /// - Faster comparisons (integer vs string)
    StringDict {
        indices: Vec<u32>,      // Per-row dictionary index (0 = NULL)
        dict_offsets: Vec<u32>, // Offsets into dict_data
        dict_data: Vec<u8>,     // Unique string bytes
    },
}

impl ColumnData {
    pub fn new(dtype: ColumnType) -> Self {
        match dtype {
            ColumnType::Bool => ColumnData::Bool { data: Vec::new(), len: 0 },
            ColumnType::Int8 | ColumnType::Int16 | ColumnType::Int32 | ColumnType::Int64 |
            ColumnType::UInt8 | ColumnType::UInt16 | ColumnType::UInt32 | ColumnType::UInt64 => {
                ColumnData::Int64(Vec::new())
            }
            ColumnType::Float32 | ColumnType::Float64 => ColumnData::Float64(Vec::new()),
            ColumnType::String => ColumnData::String {
                offsets: vec![0],
                data: Vec::new(),
            },
            ColumnType::Binary => ColumnData::Binary {
                offsets: vec![0],
                data: Vec::new(),
            },
            ColumnType::StringDict => ColumnData::StringDict {
                indices: Vec::new(),
                dict_offsets: vec![0],
                dict_data: Vec::new(),
            },
            ColumnType::Null => ColumnData::Int64(Vec::new()),
        }
    }

    #[inline]
    pub fn len(&self) -> usize {
        match self {
            ColumnData::Bool { len, .. } => *len,
            ColumnData::Int64(v) => v.len(),
            ColumnData::Float64(v) => v.len(),
            ColumnData::String { offsets, .. } => offsets.len().saturating_sub(1),
            ColumnData::Binary { offsets, .. } => offsets.len().saturating_sub(1),
            ColumnData::StringDict { indices, .. } => indices.len(),
        }
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    #[inline]
    pub fn push_i64(&mut self, value: i64) {
        if let ColumnData::Int64(v) = self {
            v.push(value);
        }
    }

    #[inline]
    pub fn push_f64(&mut self, value: f64) {
        if let ColumnData::Float64(v) = self {
            v.push(value);
        }
    }

    #[inline]
    pub fn push_bool(&mut self, value: bool) {
        if let ColumnData::Bool { data, len } = self {
            let byte_idx = *len / 8;
            let bit_idx = *len % 8;
            if byte_idx >= data.len() {
                data.push(0);
            }
            if value {
                data[byte_idx] |= 1 << bit_idx;
            }
            *len += 1;
        }
    }

    #[inline]
    pub fn push_string(&mut self, value: &str) {
        if let ColumnData::String { offsets, data } = self {
            data.extend_from_slice(value.as_bytes());
            offsets.push(data.len() as u32);
        }
    }

    #[inline]
    pub fn push_bytes(&mut self, value: &[u8]) {
        if let ColumnData::Binary { offsets, data } = self {
            data.extend_from_slice(value);
            offsets.push(data.len() as u32);
        }
    }

    pub fn extend_i64(&mut self, values: &[i64]) {
        if let ColumnData::Int64(v) = self {
            v.extend_from_slice(values);
        }
    }

    pub fn extend_f64(&mut self, values: &[f64]) {
        if let ColumnData::Float64(v) = self {
            v.extend_from_slice(values);
        }
    }

    /// Batch extend strings - much faster than individual push_string calls
    #[inline]
    pub fn extend_strings(&mut self, values: &[String]) {
        if let ColumnData::String { offsets, data } = self {
            // Pre-calculate total size needed
            let total_len: usize = values.iter().map(|s| s.len()).sum();
            data.reserve(total_len);
            offsets.reserve(values.len());
            
            for s in values {
                data.extend_from_slice(s.as_bytes());
                offsets.push(data.len() as u32);
            }
        }
    }

    /// Batch extend binary data
    #[inline]
    pub fn extend_bytes(&mut self, values: &[Vec<u8>]) {
        if let ColumnData::Binary { offsets, data } = self {
            let total_len: usize = values.iter().map(|b| b.len()).sum();
            data.reserve(total_len);
            offsets.reserve(values.len());
            
            for b in values {
                data.extend_from_slice(b);
                offsets.push(data.len() as u32);
            }
        }
    }

    /// Batch extend bools
    #[inline]
    pub fn extend_bools(&mut self, values: &[bool]) {
        if let ColumnData::Bool { data, len } = self {
            for &value in values {
                let byte_idx = *len / 8;
                let bit_idx = *len % 8;
                if byte_idx >= data.len() {
                    data.push(0);
                }
                if value {
                    data[byte_idx] |= 1 << bit_idx;
                }
                *len += 1;
            }
        }
    }

    /// Serialize to bytes
    /// OPTIMIZED: Uses bulk memcpy for numeric columns instead of per-element loops
    pub fn to_bytes(&self) -> Vec<u8> {
        match self {
            ColumnData::Bool { data, len } => {
                let mut buf = Vec::with_capacity(8 + data.len());
                buf.extend_from_slice(&(*len as u64).to_le_bytes());
                buf.extend_from_slice(data);
                buf
            }
            ColumnData::Int64(v) => {
                // OPTIMIZATION: Bulk memcpy instead of per-element loop
                // ~10x faster for large arrays
                let mut buf = Vec::with_capacity(8 + v.len() * 8);
                buf.extend_from_slice(&(v.len() as u64).to_le_bytes());
                // SAFETY: i64 slice can be safely viewed as bytes on all platforms
                let bytes = unsafe {
                    std::slice::from_raw_parts(v.as_ptr() as *const u8, v.len() * 8)
                };
                buf.extend_from_slice(bytes);
                buf
            }
            ColumnData::Float64(v) => {
                // OPTIMIZATION: Bulk memcpy instead of per-element loop
                let mut buf = Vec::with_capacity(8 + v.len() * 8);
                buf.extend_from_slice(&(v.len() as u64).to_le_bytes());
                // SAFETY: f64 slice can be safely viewed as bytes on all platforms
                let bytes = unsafe {
                    std::slice::from_raw_parts(v.as_ptr() as *const u8, v.len() * 8)
                };
                buf.extend_from_slice(bytes);
                buf
            }
            ColumnData::String { offsets, data } | ColumnData::Binary { offsets, data } => {
                // OPTIMIZATION: Pre-allocate and use bulk memcpy for offsets
                let count = offsets.len().saturating_sub(1);
                let mut buf = Vec::with_capacity(8 + offsets.len() * 4 + 8 + data.len());
                buf.extend_from_slice(&(count as u64).to_le_bytes());
                // Bulk copy offsets (u32 array)
                let offset_bytes = unsafe {
                    std::slice::from_raw_parts(offsets.as_ptr() as *const u8, offsets.len() * 4)
                };
                buf.extend_from_slice(offset_bytes);
                buf.extend_from_slice(&(data.len() as u64).to_le_bytes());
                buf.extend_from_slice(data);
                buf
            }
            ColumnData::StringDict { indices, dict_offsets, dict_data } => {
                // Format: [row_count][dict_size][indices...][dict_offsets...][dict_data_len][dict_data]
                // OPTIMIZATION: Pre-allocate and use bulk memcpy
                let mut buf = Vec::with_capacity(
                    16 + indices.len() * 4 + dict_offsets.len() * 4 + 8 + dict_data.len()
                );
                buf.extend_from_slice(&(indices.len() as u64).to_le_bytes());
                buf.extend_from_slice(&(dict_offsets.len() as u64).to_le_bytes());
                // Bulk copy indices (u32 array)
                let indices_bytes = unsafe {
                    std::slice::from_raw_parts(indices.as_ptr() as *const u8, indices.len() * 4)
                };
                buf.extend_from_slice(indices_bytes);
                // Bulk copy dict_offsets (u32 array)
                let offsets_bytes = unsafe {
                    std::slice::from_raw_parts(dict_offsets.as_ptr() as *const u8, dict_offsets.len() * 4)
                };
                buf.extend_from_slice(offsets_bytes);
                buf.extend_from_slice(&(dict_data.len() as u64).to_le_bytes());
                buf.extend_from_slice(dict_data);
                buf
            }
        }
    }

    /// Create an empty column with the same type
    pub fn clone_empty(&self) -> Self {
        match self {
            ColumnData::Bool { .. } => ColumnData::Bool { data: Vec::new(), len: 0 },
            ColumnData::Int64(_) => ColumnData::Int64(Vec::new()),
            ColumnData::Float64(_) => ColumnData::Float64(Vec::new()),
            ColumnData::String { .. } => ColumnData::String { offsets: vec![0], data: Vec::new() },
            ColumnData::Binary { .. } => ColumnData::Binary { offsets: vec![0], data: Vec::new() },
            ColumnData::StringDict { .. } => ColumnData::StringDict { 
                indices: Vec::new(), 
                dict_offsets: vec![0], 
                dict_data: Vec::new() 
            },
        }
    }
    
    /// Append another column's data to this column
    pub fn append(&mut self, other: &Self) {
        match (self, other) {
            (ColumnData::Bool { data, len }, ColumnData::Bool { data: other_data, len: other_len }) => {
                // Append bits from other
                for i in 0..*other_len {
                    let byte_idx = i / 8;
                    let bit_idx = i % 8;
                    let val = byte_idx < other_data.len() && (other_data[byte_idx] >> bit_idx) & 1 == 1;
                    
                    let new_byte = *len / 8;
                    let new_bit = *len % 8;
                    if new_byte >= data.len() {
                        data.push(0);
                    }
                    if val {
                        data[new_byte] |= 1 << new_bit;
                    }
                    *len += 1;
                }
            }
            (ColumnData::Int64(v), ColumnData::Int64(other_v)) => {
                v.extend_from_slice(other_v);
            }
            (ColumnData::Float64(v), ColumnData::Float64(other_v)) => {
                v.extend_from_slice(other_v);
            }
            (ColumnData::String { offsets, data }, ColumnData::String { offsets: other_offsets, data: other_data }) => {
                let base_offset = *offsets.last().unwrap_or(&0);
                for i in 1..other_offsets.len() {
                    offsets.push(base_offset + other_offsets[i]);
                }
                data.extend_from_slice(other_data);
            }
            (ColumnData::Binary { offsets, data }, ColumnData::Binary { offsets: other_offsets, data: other_data }) => {
                let base_offset = *offsets.last().unwrap_or(&0);
                for i in 1..other_offsets.len() {
                    offsets.push(base_offset + other_offsets[i]);
                }
                data.extend_from_slice(other_data);
            }
            _ => {} // Type mismatch - ignore
        }
    }

    /// Filter column data to only include rows at specified indices
    pub fn filter_by_indices(&self, indices: &[usize]) -> Self {
        match self {
            ColumnData::Bool { data, len } => {
                let mut new_data = Vec::new();
                let mut new_len = 0usize;
                for &idx in indices {
                    if idx < *len {
                        let old_byte = idx / 8;
                        let old_bit = idx % 8;
                        let val = old_byte < data.len() && (data[old_byte] >> old_bit) & 1 == 1;
                        let new_byte = new_len / 8;
                        let new_bit = new_len % 8;
                        if new_byte >= new_data.len() {
                            new_data.push(0);
                        }
                        if val {
                            new_data[new_byte] |= 1 << new_bit;
                        }
                        new_len += 1;
                    }
                }
                ColumnData::Bool { data: new_data, len: new_len }
            }
            ColumnData::Int64(v) => {
                ColumnData::Int64(indices.iter().filter_map(|&i| v.get(i).copied()).collect())
            }
            ColumnData::Float64(v) => {
                ColumnData::Float64(indices.iter().filter_map(|&i| v.get(i).copied()).collect())
            }
            ColumnData::String { offsets, data } => {
                let mut new_offsets = vec![0u32];
                let mut new_data = Vec::new();
                for &idx in indices {
                    if idx + 1 < offsets.len() {
                        let start = offsets[idx] as usize;
                        let end = offsets[idx + 1] as usize;
                        new_data.extend_from_slice(&data[start..end]);
                        new_offsets.push(new_data.len() as u32);
                    }
                }
                ColumnData::String { offsets: new_offsets, data: new_data }
            }
            ColumnData::Binary { offsets, data } => {
                let mut new_offsets = vec![0u32];
                let mut new_data = Vec::new();
                for &idx in indices {
                    if idx + 1 < offsets.len() {
                        let start = offsets[idx] as usize;
                        let end = offsets[idx + 1] as usize;
                        new_data.extend_from_slice(&data[start..end]);
                        new_offsets.push(new_data.len() as u32);
                    }
                }
                ColumnData::Binary { offsets: new_offsets, data: new_data }
            }
            ColumnData::StringDict { indices: row_indices, dict_offsets, dict_data } => {
                // Just filter the indices array, dictionary stays the same
                let new_indices: Vec<u32> = indices.iter()
                    .filter_map(|&i| row_indices.get(i).copied())
                    .collect();
                ColumnData::StringDict { 
                    indices: new_indices, 
                    dict_offsets: dict_offsets.clone(), 
                    dict_data: dict_data.clone() 
                }
            }
        }
    }
    
    /// Convert regular String column to dictionary-encoded StringDict
    /// This is beneficial for low-cardinality columns (e.g., category, status)
    pub fn to_dict_encoded(&self) -> Option<Self> {
        if let ColumnData::String { offsets, data } = self {
            use ahash::AHashMap;
            
            let row_count = offsets.len().saturating_sub(1);
            if row_count == 0 {
                return Some(ColumnData::StringDict {
                    indices: Vec::new(),
                    dict_offsets: vec![0],
                    dict_data: Vec::new(),
                });
            }
            
            // Build dictionary: string -> dict_index
            let mut dict_map: AHashMap<&[u8], u32> = AHashMap::with_capacity(1000);
            let mut dict_offsets_new = vec![0u32];
            let mut dict_data_new = Vec::new();
            let mut row_indices = Vec::with_capacity(row_count);
            let mut next_dict_idx = 1u32; // 0 reserved for NULL
            
            for i in 0..row_count {
                let start = offsets[i] as usize;
                let end = offsets[i + 1] as usize;
                let str_bytes = &data[start..end];
                
                let dict_idx = *dict_map.entry(str_bytes).or_insert_with(|| {
                    let idx = next_dict_idx;
                    next_dict_idx += 1;
                    dict_data_new.extend_from_slice(str_bytes);
                    dict_offsets_new.push(dict_data_new.len() as u32);
                    idx
                });
                row_indices.push(dict_idx);
            }
            
            Some(ColumnData::StringDict {
                indices: row_indices,
                dict_offsets: dict_offsets_new,
                dict_data: dict_data_new,
            })
        } else {
            None
        }
    }
    
    /// Get dictionary index for a row (for StringDict columns)
    #[inline]
    pub fn get_dict_index(&self, row: usize) -> Option<u32> {
        if let ColumnData::StringDict { indices, .. } = self {
            indices.get(row).copied()
        } else {
            None
        }
    }
    
    /// Estimate memory usage in bytes
    pub fn estimate_memory_bytes(&self) -> usize {
        match self {
            ColumnData::Bool { data, .. } => data.len(),
            ColumnData::Int64(v) => v.len() * 8,
            ColumnData::Float64(v) => v.len() * 8,
            ColumnData::String { offsets, data } => offsets.len() * 4 + data.len(),
            ColumnData::Binary { offsets, data } => offsets.len() * 4 + data.len(),
            ColumnData::StringDict { indices, dict_offsets, dict_data } => {
                indices.len() * 4 + dict_offsets.len() * 4 + dict_data.len()
            }
        }
    }
}

// ============================================================================
// File Header (256 bytes)
// ============================================================================

#[derive(Debug, Clone)]
pub struct OnDemandHeader {
    pub version: u32,
    pub flags: u32,
    pub row_count: u64,
    pub column_count: u32,
    pub row_group_size: u32,
    pub schema_offset: u64,
    pub column_index_offset: u64,
    pub id_column_offset: u64,
    pub created_at: i64,
    pub modified_at: i64,
    pub checksum: u32,
}

impl OnDemandHeader {
    pub fn new() -> Self {
        let now = chrono::Utc::now().timestamp();
        Self {
            version: FORMAT_VERSION_V3,
            flags: 0,
            row_count: 0,
            column_count: 0,
            row_group_size: DEFAULT_ROW_GROUP_SIZE,
            schema_offset: HEADER_SIZE_V3 as u64,
            column_index_offset: 0,
            id_column_offset: 0,
            created_at: now,
            modified_at: now,
            checksum: 0,
        }
    }

    pub fn to_bytes(&self) -> [u8; HEADER_SIZE_V3] {
        let mut buf = [0u8; HEADER_SIZE_V3];
        let mut pos = 0;

        // Magic (8 bytes)
        buf[pos..pos + 8].copy_from_slice(MAGIC_V3);
        pos += 8;

        // Version (4 bytes)
        buf[pos..pos + 4].copy_from_slice(&self.version.to_le_bytes());
        pos += 4;

        // Flags (4 bytes)
        buf[pos..pos + 4].copy_from_slice(&self.flags.to_le_bytes());
        pos += 4;

        // Row count (8 bytes)
        buf[pos..pos + 8].copy_from_slice(&self.row_count.to_le_bytes());
        pos += 8;

        // Column count (4 bytes)
        buf[pos..pos + 4].copy_from_slice(&self.column_count.to_le_bytes());
        pos += 4;

        // Row group size (4 bytes)
        buf[pos..pos + 4].copy_from_slice(&self.row_group_size.to_le_bytes());
        pos += 4;

        // Schema offset (8 bytes)
        buf[pos..pos + 8].copy_from_slice(&self.schema_offset.to_le_bytes());
        pos += 8;

        // Column index offset (8 bytes)
        buf[pos..pos + 8].copy_from_slice(&self.column_index_offset.to_le_bytes());
        pos += 8;

        // ID column offset (8 bytes)
        buf[pos..pos + 8].copy_from_slice(&self.id_column_offset.to_le_bytes());
        pos += 8;

        // Created timestamp (8 bytes)
        buf[pos..pos + 8].copy_from_slice(&self.created_at.to_le_bytes());
        pos += 8;

        // Modified timestamp (8 bytes)
        buf[pos..pos + 8].copy_from_slice(&self.modified_at.to_le_bytes());
        pos += 8;

        // Checksum (4 bytes) - computed over previous bytes
        let checksum = crc32fast::hash(&buf[0..pos]);
        buf[pos..pos + 4].copy_from_slice(&checksum.to_le_bytes());

        buf
    }

    pub fn from_bytes(bytes: &[u8; HEADER_SIZE_V3]) -> io::Result<Self> {
        let mut pos = 0;

        // Verify magic
        if &bytes[pos..pos + 8] != MAGIC_V3 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "Invalid V3 file magic",
            ));
        }
        pos += 8;

        let version = u32::from_le_bytes(bytes[pos..pos + 4].try_into().unwrap());
        pos += 4;
        let flags = u32::from_le_bytes(bytes[pos..pos + 4].try_into().unwrap());
        pos += 4;
        let row_count = u64::from_le_bytes(bytes[pos..pos + 8].try_into().unwrap());
        pos += 8;
        let column_count = u32::from_le_bytes(bytes[pos..pos + 4].try_into().unwrap());
        pos += 4;
        let row_group_size = u32::from_le_bytes(bytes[pos..pos + 4].try_into().unwrap());
        pos += 4;
        let schema_offset = u64::from_le_bytes(bytes[pos..pos + 8].try_into().unwrap());
        pos += 8;
        let column_index_offset = u64::from_le_bytes(bytes[pos..pos + 8].try_into().unwrap());
        pos += 8;
        let id_column_offset = u64::from_le_bytes(bytes[pos..pos + 8].try_into().unwrap());
        pos += 8;
        let created_at = i64::from_le_bytes(bytes[pos..pos + 8].try_into().unwrap());
        pos += 8;
        let modified_at = i64::from_le_bytes(bytes[pos..pos + 8].try_into().unwrap());
        pos += 8;

        let checksum = u32::from_le_bytes(bytes[pos..pos + 4].try_into().unwrap());

        // Verify checksum
        let computed = crc32fast::hash(&bytes[0..pos]);
        if computed != checksum {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "Header checksum mismatch",
            ));
        }

        Ok(Self {
            version,
            flags,
            row_count,
            column_count,
            row_group_size,
            schema_offset,
            column_index_offset,
            id_column_offset,
            created_at,
            modified_at,
            checksum,
        })
    }
}

// ============================================================================
// Column Index Entry (32 bytes per column)
// ============================================================================

#[derive(Debug, Clone, Copy, Default)]
pub struct ColumnIndexEntry {
    pub data_offset: u64,
    pub data_length: u64,
    pub null_offset: u64,
    pub null_length: u64,
}

impl ColumnIndexEntry {
    pub fn to_bytes(&self) -> [u8; COLUMN_INDEX_ENTRY_SIZE] {
        let mut buf = [0u8; COLUMN_INDEX_ENTRY_SIZE];
        buf[0..8].copy_from_slice(&self.data_offset.to_le_bytes());
        buf[8..16].copy_from_slice(&self.data_length.to_le_bytes());
        buf[16..24].copy_from_slice(&self.null_offset.to_le_bytes());
        buf[24..32].copy_from_slice(&self.null_length.to_le_bytes());
        buf
    }

    pub fn from_bytes(bytes: &[u8]) -> Self {
        Self {
            data_offset: u64::from_le_bytes(bytes[0..8].try_into().unwrap()),
            data_length: u64::from_le_bytes(bytes[8..16].try_into().unwrap()),
            null_offset: u64::from_le_bytes(bytes[16..24].try_into().unwrap()),
            null_length: u64::from_le_bytes(bytes[24..32].try_into().unwrap()),
        }
    }
}

// ============================================================================
// Schema (bincode-free serialization)
// ============================================================================

#[derive(Debug, Clone, Default)]
pub struct OnDemandSchema {
    pub columns: Vec<(String, ColumnType)>,
    name_to_idx: HashMap<String, usize>,
}

impl OnDemandSchema {
    pub fn new() -> Self {
        Self {
            columns: Vec::new(),
            name_to_idx: HashMap::new(),
        }
    }

    pub fn add_column(&mut self, name: &str, dtype: ColumnType) -> usize {
        if let Some(&idx) = self.name_to_idx.get(name) {
            return idx;
        }
        let idx = self.columns.len();
        self.columns.push((name.to_string(), dtype));
        self.name_to_idx.insert(name.to_string(), idx);
        idx
    }

    pub fn get_index(&self, name: &str) -> Option<usize> {
        self.name_to_idx.get(name).copied()
    }

    pub fn column_count(&self) -> usize {
        self.columns.len()
    }

    /// Serialize schema to bytes (no bincode)
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut buf = Vec::new();
        
        // Column count
        buf.extend_from_slice(&(self.columns.len() as u32).to_le_bytes());
        
        // Each column: [name_len:u16][name:bytes][type:u8]
        for (name, dtype) in &self.columns {
            let name_bytes = name.as_bytes();
            buf.extend_from_slice(&(name_bytes.len() as u16).to_le_bytes());
            buf.extend_from_slice(name_bytes);
            buf.push(*dtype as u8);
        }
        
        buf
    }

    /// Deserialize schema from bytes (no bincode)
    pub fn from_bytes(bytes: &[u8]) -> io::Result<Self> {
        let mut pos = 0;
        
        if bytes.len() < 4 {
            return Err(io::Error::new(io::ErrorKind::InvalidData, "Schema too short"));
        }
        
        let column_count = u32::from_le_bytes(bytes[pos..pos + 4].try_into().unwrap()) as usize;
        pos += 4;
        
        let mut schema = Self::new();
        
        for _ in 0..column_count {
            if pos + 2 > bytes.len() {
                return Err(io::Error::new(io::ErrorKind::InvalidData, "Truncated schema"));
            }
            
            let name_len = u16::from_le_bytes(bytes[pos..pos + 2].try_into().unwrap()) as usize;
            pos += 2;
            
            if pos + name_len + 1 > bytes.len() {
                return Err(io::Error::new(io::ErrorKind::InvalidData, "Truncated column name"));
            }
            
            let name = std::str::from_utf8(&bytes[pos..pos + name_len])
                .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?
                .to_string();
            pos += name_len;
            
            let dtype = ColumnType::from_u8(bytes[pos])
                .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "Invalid column type"))?;
            pos += 1;
            
            schema.add_column(&name, dtype);
        }
        
        Ok(schema)
    }
}

// ============================================================================
// On-Demand Storage Engine
// ============================================================================

/// High-performance on-demand columnar storage
/// 
/// Key features:
/// - Read only required columns (column projection)
/// - Read only required row ranges  
/// - Uses mmap for zero-copy reads with OS page cache (cross-platform)
/// - Soft delete with deleted bitmap
/// - Update via delete + insert
pub struct OnDemandStorage {
    path: PathBuf,
    file: RwLock<Option<File>>,
    /// Memory-mapped file cache for fast repeated reads
    mmap_cache: RwLock<MmapCache>,
    header: RwLock<OnDemandHeader>,
    schema: RwLock<OnDemandSchema>,
    column_index: RwLock<Vec<ColumnIndexEntry>>,
    /// In-memory column data (for writes)
    columns: RwLock<Vec<ColumnData>>,
    /// Row IDs
    ids: RwLock<Vec<u64>>,
    /// Next row ID
    next_id: AtomicU64,
    /// Null bitmaps per column
    nulls: RwLock<Vec<Vec<u8>>>,
    /// Deleted row bitmap (packed bits, 1 = deleted)
    deleted: RwLock<Vec<u8>>,
    /// ID to row index mapping for fast lookups (lazy-loaded)
    /// Only built when needed for delete/exists operations
    id_to_idx: RwLock<Option<HashMap<u64, usize>>>,
    /// Cached count of active (non-deleted) rows for O(1) COUNT(*)
    active_count: AtomicU64,
    /// Durability level for controlling fsync behavior
    durability: super::DurabilityLevel,
    /// WAL writer for safe/max durability modes (None for fast mode)
    wal_writer: RwLock<Option<super::incremental::WalWriter>>,
    /// WAL buffer for pending writes (used for recovery)
    wal_buffer: RwLock<Vec<super::incremental::WalRecord>>,
    /// Auto-flush threshold: number of pending rows (0 = disabled)
    auto_flush_rows: AtomicU64,
    /// Auto-flush threshold: estimated memory bytes (0 = disabled)
    auto_flush_bytes: AtomicU64,
    /// Count of rows inserted since last save (for auto-flush)
    pending_rows: AtomicU64,
}

impl OnDemandStorage {
    /// Create a new V3 storage file with default durability (Fast)
    pub fn create(path: &Path) -> io::Result<Self> {
        Self::create_with_durability(path, super::DurabilityLevel::Fast)
    }
    
    /// Create a new V3 storage file with specified durability level
    pub fn create_with_durability(path: &Path, durability: super::DurabilityLevel) -> io::Result<Self> {
        let header = OnDemandHeader::new();
        let schema = OnDemandSchema::new();

        // Initialize WAL for safe/max durability modes
        let wal_writer = if durability != super::DurabilityLevel::Fast {
            let wal_path = Self::wal_path(path);
            Some(super::incremental::WalWriter::create(&wal_path, 0)?)
        } else {
            None
        };

        let storage = Self {
            path: path.to_path_buf(),
            file: RwLock::new(None),
            mmap_cache: RwLock::new(MmapCache::new()),
            header: RwLock::new(header),
            schema: RwLock::new(schema),
            column_index: RwLock::new(Vec::new()),
            columns: RwLock::new(Vec::new()),
            ids: RwLock::new(Vec::new()),
            next_id: AtomicU64::new(0),
            nulls: RwLock::new(Vec::new()),
            deleted: RwLock::new(Vec::new()),
            id_to_idx: RwLock::new(Some(HashMap::new())),
            active_count: AtomicU64::new(0),
            durability,
            wal_writer: RwLock::new(wal_writer),
            wal_buffer: RwLock::new(Vec::new()),
            auto_flush_rows: AtomicU64::new(100000),
            auto_flush_bytes: AtomicU64::new(500 * 1024 * 1024),
            pending_rows: AtomicU64::new(0),
        };

        // Write initial file
        storage.save()?;

        Ok(storage)
    }
    
    /// Get WAL file path for a given data file path
    fn wal_path(main_path: &Path) -> PathBuf {
        let mut wal_path = main_path.to_path_buf();
        let ext = wal_path.extension()
            .map(|e| format!("{}.wal", e.to_string_lossy()))
            .unwrap_or_else(|| "wal".to_string());
        wal_path.set_extension(ext);
        wal_path
    }

    /// Open existing V3 storage with default durability (Fast)
    pub fn open(path: &Path) -> io::Result<Self> {
        Self::open_with_durability(path, super::DurabilityLevel::Fast)
    }
    
    /// Open existing V3 storage with specified durability level
    /// Uses mmap for fast zero-copy reads with OS page cache
    pub fn open_with_durability(path: &Path, durability: super::DurabilityLevel) -> io::Result<Self> {
        let file = File::open(path)?;
        
        // Create mmap cache and use it for initial reads
        let mut mmap_cache = MmapCache::new();
        
        // Read header using mmap (zero-copy)
        let mut header_bytes = [0u8; HEADER_SIZE_V3];
        mmap_cache.read_at(&file, &mut header_bytes, 0)?;
        let header = OnDemandHeader::from_bytes(&header_bytes)?;

        // Read schema using mmap
        let schema_size = header.column_index_offset - header.schema_offset;
        let mut schema_bytes = vec![0u8; schema_size as usize];
        mmap_cache.read_at(&file, &mut schema_bytes, header.schema_offset)?;
        let schema = OnDemandSchema::from_bytes(&schema_bytes)?;

        // Read column index using mmap
        let index_size = header.column_count as usize * COLUMN_INDEX_ENTRY_SIZE;
        let mut index_bytes = vec![0u8; index_size];
        mmap_cache.read_at(&file, &mut index_bytes, header.column_index_offset)?;
        
        let mut column_index = Vec::with_capacity(header.column_count as usize);
        for i in 0..header.column_count as usize {
            let start = i * COLUMN_INDEX_ENTRY_SIZE;
            let entry = ColumnIndexEntry::from_bytes(&index_bytes[start..start + COLUMN_INDEX_ENTRY_SIZE]);
            column_index.push(entry);
        }

        // Read IDs into memory using mmap (needed for read_ids and row count)
        let id_count = header.row_count as usize;
        let mut id_bytes = vec![0u8; id_count * 8];
        if id_count > 0 {
            mmap_cache.read_at(&file, &mut id_bytes, header.id_column_offset)?;
        }
        let mut ids = Vec::with_capacity(id_count);
        for i in 0..id_count {
            let id = u64::from_le_bytes(id_bytes[i * 8..(i + 1) * 8].try_into().unwrap());
            ids.push(id);
        }
        // NOTE: id_to_idx HashMap is NOT built here - lazy loaded when needed
        // This saves ~200MB+ memory for 10M rows (only needed for delete/exists ops)
        
        // If no rows exist, start from 0; otherwise start after max existing ID
        let next_id = if ids.is_empty() {
            0
        } else {
            ids.iter().max().copied().unwrap_or(0) + 1
        };

        // NOTE: Column data is NOT loaded - will be read on-demand via mmap
        let columns = vec![ColumnData::new(ColumnType::Int64); header.column_count as usize];
        let nulls = vec![Vec::new(); header.column_count as usize];
        
        // Initialize deleted bitmap (all zeros = no deleted rows)
        let deleted_len = (id_count + 7) / 8;
        let deleted = vec![0u8; deleted_len];

        // Handle WAL recovery and initialization for safe/max durability
        let wal_path = Self::wal_path(path);
        let (wal_writer, wal_buffer, recovered_next_id) = if durability != super::DurabilityLevel::Fast {
            if wal_path.exists() {
                // Replay WAL for crash recovery
                let mut reader = super::incremental::WalReader::open(&wal_path)?;
                let records = reader.read_all()?;
                
                // Find max ID from WAL records (handles both Insert and BatchInsert)
                let max_wal_id = records.iter().filter_map(|r| {
                    match r {
                        super::incremental::WalRecord::Insert { id, .. } => Some(*id),
                        super::incremental::WalRecord::BatchInsert { start_id, rows } => {
                            Some(*start_id + rows.len() as u64 - 1)
                        }
                        _ => None,
                    }
                }).max();
                
                let recovered_id = max_wal_id.map(|id| id + 1).unwrap_or(next_id);
                
                // Open for append
                let writer = super::incremental::WalWriter::open(&wal_path)?;
                (Some(writer), records, recovered_id)
            } else {
                // Create new WAL
                let writer = super::incremental::WalWriter::create(&wal_path, next_id)?;
                (Some(writer), Vec::new(), next_id)
            }
        } else {
            (None, Vec::new(), next_id)
        };
        
        let final_next_id = recovered_next_id.max(next_id);

        Ok(Self {
            path: path.to_path_buf(),
            file: RwLock::new(Some(file)),
            mmap_cache: RwLock::new(mmap_cache),
            header: RwLock::new(header),
            schema: RwLock::new(schema),
            column_index: RwLock::new(column_index),
            columns: RwLock::new(columns),
            ids: RwLock::new(ids),
            next_id: AtomicU64::new(final_next_id),
            nulls: RwLock::new(nulls),
            deleted: RwLock::new(deleted),
            id_to_idx: RwLock::new(None),  // Lazy loaded when needed
            active_count: AtomicU64::new(id_count as u64),  // All rows active on fresh open
            durability,
            wal_writer: RwLock::new(wal_writer),
            wal_buffer: RwLock::new(wal_buffer),
            auto_flush_rows: AtomicU64::new(10000),
            auto_flush_bytes: AtomicU64::new(500 * 1024 * 1024),
            pending_rows: AtomicU64::new(0),
        })
    }
    
    /// Set auto-flush thresholds
    /// 
    /// When either threshold is exceeded, data is automatically written to file.
    /// Set to 0 to disable the respective threshold.
    /// 
    /// # Arguments
    /// * `rows` - Auto-flush when pending rows exceed this count (0 = disabled)
    /// * `bytes` - Auto-flush when estimated memory exceeds this size (0 = disabled)
    pub fn set_auto_flush(&self, rows: u64, bytes: u64) {
        self.auto_flush_rows.store(rows, Ordering::SeqCst);
        self.auto_flush_bytes.store(bytes, Ordering::SeqCst);
    }
    
    /// Get current auto-flush configuration
    pub fn get_auto_flush(&self) -> (u64, u64) {
        (self.auto_flush_rows.load(Ordering::SeqCst), self.auto_flush_bytes.load(Ordering::SeqCst))
    }
    
    /// Estimate current in-memory data size in bytes
    pub fn estimate_memory_bytes(&self) -> u64 {
        let columns = self.columns.read();
        let mut total: u64 = 0;
        
        for col in columns.iter() {
            total += col.estimate_memory_bytes() as u64;
        }
        
        // Add overhead for IDs (8 bytes each)
        total += self.ids.read().len() as u64 * 8;
        
        // Add overhead for null bitmaps
        for null_bitmap in self.nulls.read().iter() {
            total += null_bitmap.len() as u64;
        }
        
        // Add deleted bitmap
        total += self.deleted.read().len() as u64;
        
        total
    }
    
    /// Check if auto-flush is needed and perform it if so
    /// Returns true if auto-flush was performed
    fn maybe_auto_flush(&self) -> io::Result<bool> {
        let rows_threshold = self.auto_flush_rows.load(Ordering::SeqCst);
        let bytes_threshold = self.auto_flush_bytes.load(Ordering::SeqCst);
        
        // Check row threshold
        if rows_threshold > 0 {
            let pending = self.pending_rows.load(Ordering::SeqCst);
            if pending >= rows_threshold {
                self.save()?;
                self.pending_rows.store(0, Ordering::SeqCst);
                return Ok(true);
            }
        }
        
        // Check memory threshold
        if bytes_threshold > 0 {
            let mem_bytes = self.estimate_memory_bytes();
            if mem_bytes >= bytes_threshold {
                self.save()?;
                self.pending_rows.store(0, Ordering::SeqCst);
                return Ok(true);
            }
        }
        
        Ok(false)
    }

    /// Create or open storage with default durability (Fast)
    pub fn open_or_create(path: &Path) -> io::Result<Self> {
        Self::open_or_create_with_durability(path, super::DurabilityLevel::Fast)
    }
    
    /// Create or open storage with specified durability level
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
    
    /// Open for write with specified durability level
    /// Uses on-demand loading - does NOT load all existing data into memory
    /// Data is loaded lazily when needed for specific operations
    pub fn open_for_write_with_durability(path: &Path, durability: super::DurabilityLevel) -> io::Result<Self> {
        if !path.exists() {
            return Self::create_with_durability(path, durability);
        }
        
        // Simply open the storage - no pre-loading of column data
        // The on-demand read APIs will fetch data from disk when needed
        Self::open_with_durability(path, durability)
    }

    // ========================================================================
    // On-Demand Read APIs (the key feature)
    // ========================================================================

    /// Read specific columns for a row range
    /// 
    /// This is the core on-demand read function:
    /// - Only reads the requested columns from disk
    /// - Only reads the requested row range
    /// - Uses pread for efficient random access
    ///
    /// # Arguments
    /// * `column_names` - Columns to read (None = all columns)
    /// * `start_row` - Starting row index (0-based)
    /// * `row_count` - Number of rows to read (None = to end)
    pub fn read_columns(
        &self,
        column_names: Option<&[&str]>,
        start_row: usize,
        row_count: Option<usize>,
    ) -> io::Result<HashMap<String, ColumnData>> {
        let header = self.header.read();
        let schema = self.schema.read();
        let column_index = self.column_index.read();
        
        let total_rows = header.row_count as usize;
        let actual_start = start_row.min(total_rows);
        let actual_count = row_count
            .map(|c| c.min(total_rows - actual_start))
            .unwrap_or(total_rows - actual_start);
        
        if actual_count == 0 {
            return Ok(HashMap::new());
        }

        // Determine which columns to read
        let col_indices: Vec<usize> = match column_names {
            Some(names) => names
                .iter()
                .filter_map(|name| schema.get_index(name))
                .collect(),
            None => (0..schema.column_count()).collect(),
        };

        let file_guard = self.file.read();
        let file = file_guard.as_ref().ok_or_else(|| {
            io::Error::new(io::ErrorKind::NotConnected, "File not open")
        })?;
        
        // Use mmap for efficient reads with OS page cache
        let mut mmap_cache = self.mmap_cache.write();

        // Sequential read using mmap (mmap is shared across all reads)
        let mut result = HashMap::new();
        for &col_idx in &col_indices {
            let (col_name, col_type) = &schema.columns[col_idx];
            let index_entry = &column_index[col_idx];
            
            let col_data = self.read_column_range_mmap(
                &mut mmap_cache,
                file,
                index_entry,
                *col_type,
                actual_start,
                actual_count,
                total_rows,
            )?;
            
            result.insert(col_name.clone(), col_data);
        }
        Ok(result)
    }

    /// Read columns with predicate pushdown - filter rows at storage level
    /// This avoids loading rows that don't match the filter condition
    /// 
    /// # Arguments
    /// * `column_names` - Columns to read (None = all columns)
    /// * `filter_column` - Column name to filter on
    /// * `filter_op` - Comparison operator: ">=", ">", "<=", "<", "=", "!="
    /// * `filter_value` - Value to compare against (i64 or f64)
    pub fn read_columns_filtered(
        &self,
        column_names: Option<&[&str]>,
        filter_column: &str,
        filter_op: &str,
        filter_value: f64,
    ) -> io::Result<(HashMap<String, ColumnData>, Vec<usize>)> {
        let header = self.header.read();
        let schema = self.schema.read();
        let column_index = self.column_index.read();
        
        let total_rows = header.row_count as usize;
        if total_rows == 0 {
            return Ok((HashMap::new(), Vec::new()));
        }

        // First, read the filter column to determine matching rows
        let filter_col_idx = schema.get_index(filter_column).ok_or_else(|| {
            io::Error::new(io::ErrorKind::NotFound, format!("Filter column not found: {}", filter_column))
        })?;
        
        let (_, filter_col_type) = &schema.columns[filter_col_idx];
        let filter_index = &column_index[filter_col_idx];
        
        let file_guard = self.file.read();
        let file = file_guard.as_ref().ok_or_else(|| {
            io::Error::new(io::ErrorKind::NotConnected, "File not open")
        })?;
        
        let mut mmap_cache = self.mmap_cache.write();

        // Read filter column data using mmap
        let filter_data = self.read_column_range_mmap(&mut mmap_cache, file, filter_index, *filter_col_type, 0, total_rows, total_rows)?;
        
        // Apply filter and collect matching row indices
        let matching_indices: Vec<usize> = match &filter_data {
            ColumnData::Int64(values) => {
                let filter_val = filter_value as i64;
                values.iter().enumerate()
                    .filter(|(_, &v)| match filter_op {
                        ">=" => v >= filter_val,
                        ">" => v > filter_val,
                        "<=" => v <= filter_val,
                        "<" => v < filter_val,
                        "=" | "==" => v == filter_val,
                        "!=" | "<>" => v != filter_val,
                        _ => true,
                    })
                    .map(|(i, _)| i)
                    .collect()
            }
            ColumnData::Float64(values) => {
                values.iter().enumerate()
                    .filter(|(_, &v)| match filter_op {
                        ">=" => v >= filter_value,
                        ">" => v > filter_value,
                        "<=" => v <= filter_value,
                        "<" => v < filter_value,
                        "=" | "==" => (v - filter_value).abs() < f64::EPSILON,
                        "!=" | "<>" => (v - filter_value).abs() >= f64::EPSILON,
                        _ => true,
                    })
                    .map(|(i, _)| i)
                    .collect()
            }
            _ => (0..total_rows).collect(), // Non-numeric columns: return all for numeric filter
        };

        if matching_indices.is_empty() {
            return Ok((HashMap::new(), Vec::new()));
        }

        // Determine which columns to read
        let col_indices: Vec<usize> = match column_names {
            Some(names) => names
                .iter()
                .filter_map(|name| schema.get_index(name))
                .collect(),
            None => (0..schema.column_count()).collect(),
        };

        let mut result = HashMap::new();

        // Read only matching rows for each column using mmap
        for &col_idx in &col_indices {
            let (col_name, col_type) = &schema.columns[col_idx];
            let index_entry = &column_index[col_idx];
            
            // Use scattered read for non-contiguous rows
            let col_data = self.read_column_scattered_mmap(&mut mmap_cache, file, index_entry, *col_type, &matching_indices, total_rows)?;
            result.insert(col_name.clone(), col_data);
        }

        Ok((result, matching_indices))
    }

    /// Read columns with STRING predicate pushdown - filter rows at storage level
    /// This is optimized for string equality filters (column = 'value')
    pub fn read_columns_filtered_string(
        &self,
        column_names: Option<&[&str]>,
        filter_column: &str,
        filter_value: &str,
        filter_eq: bool,  // true = equals, false = not equals
    ) -> io::Result<(HashMap<String, ColumnData>, Vec<usize>)> {
        let header = self.header.read();
        let schema = self.schema.read();
        let column_index = self.column_index.read();
        
        let total_rows = header.row_count as usize;
        if total_rows == 0 {
            return Ok((HashMap::new(), Vec::new()));
        }

        // Find filter column
        let filter_col_idx = schema.get_index(filter_column).ok_or_else(|| {
            io::Error::new(io::ErrorKind::NotFound, format!("Filter column not found: {}", filter_column))
        })?;
        
        let (_, filter_col_type) = &schema.columns[filter_col_idx];
        let filter_index = &column_index[filter_col_idx];
        
        // Only works for string columns (including dictionary-encoded)
        if !matches!(filter_col_type, ColumnType::String | ColumnType::StringDict) {
            return Err(io::Error::new(io::ErrorKind::InvalidInput, "String filter requires string column"));
        }
        
        let file_guard = self.file.read();
        let file = file_guard.as_ref().ok_or_else(|| {
            io::Error::new(io::ErrorKind::NotConnected, "File not open")
        })?;
        
        let mut mmap_cache = self.mmap_cache.write();

        // Read filter column data using mmap
        let filter_data = self.read_column_range_mmap(&mut mmap_cache, file, filter_index, *filter_col_type, 0, total_rows, total_rows)?;
        
        // Apply string filter
        let matching_indices: Vec<usize> = match filter_data {
            ColumnData::String { offsets, data } => {
                let count = offsets.len().saturating_sub(1);
                let filter_bytes = filter_value.as_bytes();
                (0..count)
                    .filter(|&i| {
                        let start = offsets[i] as usize;
                        let end = offsets[i + 1] as usize;
                        let matches = &data[start..end] == filter_bytes;
                        if filter_eq { matches } else { !matches }
                    })
                    .collect()
            }
            ColumnData::StringDict { indices, dict_offsets, dict_data } => {
                // OPTIMIZATION: Find matching dictionary index first, then scan indices
                // This is O(dict_size + row_count) vs O(row_count * string_len)
                let filter_bytes = filter_value.as_bytes();
                let mut matching_dict_idx: Option<u32> = None;
                
                // Find which dictionary entry matches the filter value
                for i in 0..dict_offsets.len().saturating_sub(1) {
                    let start = dict_offsets[i] as usize;
                    let end = dict_offsets[i + 1] as usize;
                    if &dict_data[start..end] == filter_bytes {
                        matching_dict_idx = Some((i + 1) as u32); // +1 because 0 = NULL
                        break;
                    }
                }
                
                // Now scan indices array (fast integer comparison)
                match (matching_dict_idx, filter_eq) {
                    (Some(target_idx), true) => {
                        // Equality: find rows where index == target
                        indices.iter().enumerate()
                            .filter(|(_, &idx)| idx == target_idx)
                            .map(|(i, _)| i)
                            .collect()
                    }
                    (Some(target_idx), false) => {
                        // Not equal: find rows where index != target and index != 0 (not NULL)
                        indices.iter().enumerate()
                            .filter(|(_, &idx)| idx != target_idx && idx != 0)
                            .map(|(i, _)| i)
                            .collect()
                    }
                    (None, true) => {
                        // Value not in dictionary, no matches for equality
                        Vec::new()
                    }
                    (None, false) => {
                        // Value not in dictionary, all non-NULL rows match for not-equal
                        indices.iter().enumerate()
                            .filter(|(_, &idx)| idx != 0)
                            .map(|(i, _)| i)
                            .collect()
                    }
                }
            }
            _ => return Err(io::Error::new(io::ErrorKind::InvalidInput, "Expected string column")),
        };

        if matching_indices.is_empty() {
            return Ok((HashMap::new(), Vec::new()));
        }

        // Read only matching rows for each requested column using mmap
        let col_indices: Vec<usize> = match column_names {
            Some(names) => names
                .iter()
                .filter_map(|name| schema.get_index(name))
                .collect(),
            None => (0..schema.column_count()).collect(),
        };

        let mut result = HashMap::new();
        for &col_idx in &col_indices {
            let (col_name, col_type) = &schema.columns[col_idx];
            let index_entry = &column_index[col_idx];
            let col_data = self.read_column_scattered_mmap(&mut mmap_cache, file, index_entry, *col_type, &matching_indices, total_rows)?;
            result.insert(col_name.clone(), col_data);
        }

        Ok((result, matching_indices))
    }
    
    /// Read columns with STRING predicate pushdown and early termination for LIMIT
    /// Stops scanning once we have enough matching rows - much faster for LIMIT queries
    pub fn read_columns_filtered_string_with_limit(
        &self,
        column_names: Option<&[&str]>,
        filter_column: &str,
        filter_value: &str,
        filter_eq: bool,
        limit: usize,
        offset: usize,
    ) -> io::Result<(HashMap<String, ColumnData>, Vec<usize>)> {
        let header = self.header.read();
        let schema = self.schema.read();
        let column_index = self.column_index.read();
        
        let total_rows = header.row_count as usize;
        if total_rows == 0 {
            return Ok((HashMap::new(), Vec::new()));
        }

        let filter_col_idx = schema.get_index(filter_column).ok_or_else(|| {
            io::Error::new(io::ErrorKind::NotFound, format!("Filter column not found: {}", filter_column))
        })?;
        
        let (_, filter_col_type) = &schema.columns[filter_col_idx];
        let filter_index = &column_index[filter_col_idx];
        
        if !matches!(filter_col_type, ColumnType::String | ColumnType::StringDict) {
            return Err(io::Error::new(io::ErrorKind::InvalidInput, "String filter requires string column"));
        }
        
        let file_guard = self.file.read();
        let file = file_guard.as_ref().ok_or_else(|| {
            io::Error::new(io::ErrorKind::NotConnected, "File not open")
        })?;
        
        let mut mmap_cache = self.mmap_cache.write();
        let needed = offset + limit;

        // For dictionary-encoded strings, use fast integer key scan with early termination
        // Format: [row_count:u64][dict_size:u64][indices:u32*row_count][dict_offsets:u32*dict_size][dict_data_len:u64][dict_data]
        if *filter_col_type == ColumnType::StringDict {
            let base_offset = filter_index.data_offset;
            
            // Read header: [row_count:u64][dict_size:u64]
            let mut header = [0u8; 16];
            mmap_cache.read_at(file, &mut header, base_offset)?;
            let stored_rows = u64::from_le_bytes(header[0..8].try_into().unwrap()) as usize;
            let dict_size = u64::from_le_bytes(header[8..16].try_into().unwrap()) as usize;
            
            if stored_rows == 0 || dict_size == 0 {
                return Ok((HashMap::new(), Vec::new()));
            }
            
            // Read dict_offsets
            let dict_offsets_offset = base_offset + 16 + (stored_rows * 4) as u64;
            let mut dict_offsets_buf = vec![0u8; dict_size * 4];
            mmap_cache.read_at(file, &mut dict_offsets_buf, dict_offsets_offset)?;
            
            let mut dict_offsets = Vec::with_capacity(dict_size);
            for i in 0..dict_size {
                dict_offsets.push(u32::from_le_bytes(dict_offsets_buf[i * 4..(i + 1) * 4].try_into().unwrap()));
            }
            
            // Read dict_data_len and dict_data
            let dict_data_len_offset = dict_offsets_offset + (dict_size * 4) as u64;
            let mut data_len_buf = [0u8; 8];
            mmap_cache.read_at(file, &mut data_len_buf, dict_data_len_offset)?;
            let dict_data_len = u64::from_le_bytes(data_len_buf) as usize;
            
            let dict_data_offset = dict_data_len_offset + 8;
            let mut dict_data = vec![0u8; dict_data_len];
            if dict_data_len > 0 {
                mmap_cache.read_at(file, &mut dict_data, dict_data_offset)?;
            }
            
            // Find target key in dictionary
            // dict_offsets[i] gives start of string i, dict_offsets[i+1] or dict_data_len gives end
            let filter_bytes = filter_value.as_bytes();
            let mut target_key: Option<u32> = None;
            let dict_count = dict_size.saturating_sub(1);
            
            // Linear search for dictionary lookup (small dictionaries are common)
            for i in 0..dict_count {
                let start = dict_offsets[i] as usize;
                let end = if i + 1 < dict_size { dict_offsets[i + 1] as usize } else { dict_data_len };
                if end <= dict_data.len() && start <= end && &dict_data[start..end] == filter_bytes {
                    target_key = Some((i + 1) as u32);
                    break;
                }
            }
            
            let target_key = match (target_key, filter_eq) {
                (Some(k), true) => k,
                (None, true) => return Ok((HashMap::new(), Vec::new())),
                _ => return self.read_columns_filtered_string(column_names, filter_column, filter_value, filter_eq),
            };
            
            // Stream through indices with early termination - OPTIMIZED with pointer arithmetic
            let indices_offset = base_offset + 16;
            let mut matching_indices = Vec::with_capacity(needed.min(1000));
            
            // Read indices in larger chunks for better throughput
            const CHUNK_SIZE: usize = 8192;
            let mut chunk_buf = vec![0u32; CHUNK_SIZE];
            let mut row = 0usize;
            
            while row < stored_rows && matching_indices.len() < needed {
                let chunk_rows = CHUNK_SIZE.min(stored_rows - row);
                let chunk_bytes = unsafe {
                    std::slice::from_raw_parts_mut(chunk_buf.as_mut_ptr() as *mut u8, chunk_rows * 4)
                };
                mmap_cache.read_at(file, chunk_bytes, indices_offset + (row * 4) as u64)?;
                
                // OPTIMIZED: Use pointer arithmetic for faster scanning
                let buf_ptr = chunk_buf.as_ptr();
                for i in 0..chunk_rows {
                    // unsafe pointer dereference avoids bounds check
                    if unsafe { *buf_ptr.add(i) } == target_key {
                        matching_indices.push(row + i);
                        if matching_indices.len() >= needed {
                            break;
                        }
                    }
                }
                row += chunk_rows;
            }
            
            // Apply offset
            let final_indices: Vec<usize> = matching_indices.into_iter().skip(offset).take(limit).collect();
            
            if final_indices.is_empty() {
                return Ok((HashMap::new(), Vec::new()));
            }
            
            // Read columns for matching rows - SIMPLIFIED approach without sorting
            let col_indices: Vec<usize> = match column_names {
                Some(names) => names.iter().filter_map(|name| schema.get_index(name)).collect(),
                None => (0..schema.column_count()).collect(),
            };
            
            // OPTIMIZATION: Read columns directly without sorting
            // The overhead of sorting may not be worth it for small result sets
            let mut result: HashMap<String, ColumnData> = HashMap::with_capacity(col_indices.len());
            for &col_idx in &col_indices {
                let (col_name, col_type) = &schema.columns[col_idx];
                let index_entry = &column_index[col_idx];
                let col_data = self.read_column_scattered_mmap(&mut mmap_cache, file, index_entry, *col_type, &final_indices, total_rows)?;
                result.insert(col_name.clone(), col_data);
            }
            
            return Ok((result, final_indices));
        }
        
        // Fallback to regular method for non-dictionary strings
        self.read_columns_filtered_string(column_names, filter_column, filter_value, filter_eq)
    }

    /// Read columns with numeric range filter and early termination for LIMIT
    /// Optimized for SELECT * WHERE col BETWEEN low AND high LIMIT n
    pub fn read_columns_filtered_range_with_limit(
        &self,
        column_names: Option<&[&str]>,
        filter_column: &str,
        low: f64,
        high: f64,
        limit: usize,
        offset: usize,
    ) -> io::Result<(HashMap<String, ColumnData>, Vec<usize>)> {
        let schema = self.schema.read();
        let column_index = self.column_index.read();
        let header = self.header.read();
        let deleted = self.deleted.read();
        
        let filter_col_idx = schema.get_index(filter_column).ok_or_else(|| {
            io::Error::new(io::ErrorKind::NotFound, format!("Column not found: {}", filter_column))
        })?;
        
        let (_, filter_col_type) = &schema.columns[filter_col_idx];
        let filter_index = &column_index[filter_col_idx];
        let total_rows = header.row_count as usize;
        
        let file_guard = self.file.read();
        let file = file_guard.as_ref().ok_or_else(|| {
            io::Error::new(io::ErrorKind::NotConnected, "File not open")
        })?;
        
        let mut mmap_cache = self.mmap_cache.write();
        let needed = offset + limit;
        
        // Only works for numeric columns
        if !matches!(filter_col_type, ColumnType::Int64 | ColumnType::Float64 | 
                     ColumnType::Int32 | ColumnType::Int16 | ColumnType::Int8 |
                     ColumnType::UInt64 | ColumnType::UInt32 | ColumnType::UInt16 | ColumnType::UInt8 |
                     ColumnType::Float32) {
            return Err(io::Error::new(io::ErrorKind::InvalidInput, "Range filter only works on numeric columns"));
        }
        
        // Stream through the filter column in chunks with early termination
        const CHUNK_SIZE: usize = 8192;
        let mut matching_indices = Vec::with_capacity(needed);
        let mut row_start = 0;
        
        while row_start < total_rows && matching_indices.len() < needed {
            let chunk_rows = CHUNK_SIZE.min(total_rows - row_start);
            
            // Read chunk of filter column
            let chunk_data = self.read_column_range_mmap(
                &mut mmap_cache, file, filter_index, *filter_col_type, 
                row_start, chunk_rows, total_rows
            )?;
            
            // Evaluate range predicate on chunk
            match &chunk_data {
                ColumnData::Int64(values) => {
                    let low_i = low as i64;
                    let high_i = high as i64;
                    for (i, &v) in values.iter().enumerate() {
                        let row_idx = row_start + i;
                        // Check deleted bitmap
                        let byte_idx = row_idx / 8;
                        let bit_idx = row_idx % 8;
                        let is_deleted = byte_idx < deleted.len() && (deleted[byte_idx] >> bit_idx) & 1 == 1;
                        
                        if !is_deleted && v >= low_i && v <= high_i {
                            matching_indices.push(row_idx);
                            if matching_indices.len() >= needed {
                                break;
                            }
                        }
                    }
                }
                ColumnData::Float64(values) => {
                    for (i, &v) in values.iter().enumerate() {
                        let row_idx = row_start + i;
                        let byte_idx = row_idx / 8;
                        let bit_idx = row_idx % 8;
                        let is_deleted = byte_idx < deleted.len() && (deleted[byte_idx] >> bit_idx) & 1 == 1;
                        
                        if !is_deleted && v >= low && v <= high {
                            matching_indices.push(row_idx);
                            if matching_indices.len() >= needed {
                                break;
                            }
                        }
                    }
                }
                _ => {}
            }
            
            row_start += chunk_rows;
        }
        
        // Apply offset
        let final_indices: Vec<usize> = matching_indices.into_iter().skip(offset).take(limit).collect();
        
        if final_indices.is_empty() {
            return Ok((HashMap::new(), Vec::new()));
        }
        
        // Read columns for matching rows
        let col_indices: Vec<usize> = match column_names {
            Some(names) => names.iter().filter_map(|name| schema.get_index(name)).collect(),
            None => (0..schema.column_count()).collect(),
        };
        
        let mut result = HashMap::new();
        for &col_idx in &col_indices {
            let (col_name, col_type) = &schema.columns[col_idx];
            let index_entry = &column_index[col_idx];
            let col_data = self.read_column_scattered_mmap(&mut mmap_cache, file, index_entry, *col_type, &final_indices, total_rows)?;
            result.insert(col_name.clone(), col_data);
        }
        
        Ok((result, final_indices))
    }

    /// Read columns with combined STRING + NUMERIC filter and early termination
    /// Optimized for SELECT * WHERE string_col = 'value' AND numeric_col > N LIMIT n
    /// Two-stage filter: first string equality (fast dict scan), then numeric comparison
    pub fn read_columns_filtered_string_numeric_with_limit(
        &self,
        column_names: Option<&[&str]>,
        string_column: &str,
        string_value: &str,
        numeric_column: &str,
        numeric_op: &str,  // ">" | ">=" | "<" | "<=" | "="
        numeric_value: f64,
        limit: usize,
        offset: usize,
    ) -> io::Result<(HashMap<String, ColumnData>, Vec<usize>)> {
        let header = self.header.read();
        let schema = self.schema.read();
        let column_index = self.column_index.read();
        let deleted = self.deleted.read();
        
        let total_rows = header.row_count as usize;
        if total_rows == 0 {
            return Ok((HashMap::new(), Vec::new()));
        }

        // Get string column info
        let str_col_idx = schema.get_index(string_column).ok_or_else(|| {
            io::Error::new(io::ErrorKind::NotFound, format!("String column not found: {}", string_column))
        })?;
        let (_, str_col_type) = &schema.columns[str_col_idx];
        let str_index = &column_index[str_col_idx];
        
        // Get numeric column info
        let num_col_idx = schema.get_index(numeric_column).ok_or_else(|| {
            io::Error::new(io::ErrorKind::NotFound, format!("Numeric column not found: {}", numeric_column))
        })?;
        let (_, num_col_type) = &schema.columns[num_col_idx];
        let num_index = &column_index[num_col_idx];
        
        // Validate column types
        if !matches!(str_col_type, ColumnType::String | ColumnType::StringDict) {
            return Err(io::Error::new(io::ErrorKind::InvalidInput, "String filter requires string column"));
        }
        if !matches!(num_col_type, ColumnType::Int64 | ColumnType::Float64 | 
                     ColumnType::Int32 | ColumnType::Int16 | ColumnType::Int8 |
                     ColumnType::UInt64 | ColumnType::UInt32 | ColumnType::UInt16 | ColumnType::UInt8 |
                     ColumnType::Float32) {
            return Err(io::Error::new(io::ErrorKind::InvalidInput, "Numeric filter requires numeric column"));
        }
        
        let file_guard = self.file.read();
        let file = file_guard.as_ref().ok_or_else(|| {
            io::Error::new(io::ErrorKind::NotConnected, "File not open")
        })?;
        
        let mut mmap_cache = self.mmap_cache.write();
        let needed = offset + limit;

        // For StringDict, use fast dictionary-based filter
        if *str_col_type == ColumnType::StringDict {
            let base_offset = str_index.data_offset;
            
            // Read dictionary header and find target key
            let mut str_header = [0u8; 16];
            mmap_cache.read_at(file, &mut str_header, base_offset)?;
            let stored_rows = u64::from_le_bytes(str_header[0..8].try_into().unwrap()) as usize;
            let dict_size = u64::from_le_bytes(str_header[8..16].try_into().unwrap()) as usize;
            
            if stored_rows == 0 || dict_size == 0 {
                return Ok((HashMap::new(), Vec::new()));
            }
            
            // Read dictionary
            let dict_offsets_offset = base_offset + 16 + (stored_rows * 4) as u64;
            let mut dict_offsets_buf = vec![0u8; dict_size * 4];
            mmap_cache.read_at(file, &mut dict_offsets_buf, dict_offsets_offset)?;
            
            let mut dict_offsets = Vec::with_capacity(dict_size);
            for i in 0..dict_size {
                dict_offsets.push(u32::from_le_bytes(dict_offsets_buf[i * 4..(i + 1) * 4].try_into().unwrap()));
            }
            
            let dict_data_len_offset = dict_offsets_offset + (dict_size * 4) as u64;
            let mut data_len_buf = [0u8; 8];
            mmap_cache.read_at(file, &mut data_len_buf, dict_data_len_offset)?;
            let dict_data_len = u64::from_le_bytes(data_len_buf) as usize;
            
            let dict_data_offset = dict_data_len_offset + 8;
            let mut dict_data = vec![0u8; dict_data_len];
            if dict_data_len > 0 {
                mmap_cache.read_at(file, &mut dict_data, dict_data_offset)?;
            }
            
            // Find target key
            let filter_bytes = string_value.as_bytes();
            let mut target_key: Option<u32> = None;
            let dict_count = dict_size.saturating_sub(1);
            
            for i in 0..dict_count {
                let start = dict_offsets[i] as usize;
                let end = if i + 1 < dict_size { dict_offsets[i + 1] as usize } else { dict_data_len };
                if end <= dict_data.len() && start <= end && &dict_data[start..end] == filter_bytes {
                    target_key = Some((i + 1) as u32);
                    break;
                }
            }
            
            let target_key = match target_key {
                Some(k) => k,
                None => return Ok((HashMap::new(), Vec::new())),
            };
            
            // Two-stage streaming filter with early termination
            let str_indices_offset = base_offset + 16;
            let mut matching_indices = Vec::with_capacity(needed.min(1000));
            
            const CHUNK_SIZE: usize = 8192;
            let mut row = 0usize;
            
            while row < stored_rows && matching_indices.len() < needed {
                let chunk_rows = CHUNK_SIZE.min(stored_rows - row);
                
                // Read string indices chunk
                let mut str_chunk = vec![0u32; chunk_rows];
                let chunk_bytes = unsafe {
                    std::slice::from_raw_parts_mut(str_chunk.as_mut_ptr() as *mut u8, chunk_rows * 4)
                };
                mmap_cache.read_at(file, chunk_bytes, str_indices_offset + (row * 4) as u64)?;
                
                // Read numeric column chunk
                let num_chunk = self.read_column_range_mmap(
                    &mut mmap_cache, file, num_index, *num_col_type, row, chunk_rows, total_rows
                )?;
                
                // Combined filter
                for i in 0..chunk_rows {
                    let row_idx = row + i;
                    
                    // Check deleted
                    let byte_idx = row_idx / 8;
                    let bit_idx = row_idx % 8;
                    let is_deleted = byte_idx < deleted.len() && (deleted[byte_idx] >> bit_idx) & 1 == 1;
                    if is_deleted {
                        continue;
                    }
                    
                    // Check string match
                    if str_chunk[i] != target_key {
                        continue;
                    }
                    
                    // Check numeric condition
                    let num_match = match &num_chunk {
                        ColumnData::Int64(values) => {
                            let v = values[i] as f64;
                            match numeric_op {
                                ">" => v > numeric_value,
                                ">=" => v >= numeric_value,
                                "<" => v < numeric_value,
                                "<=" => v <= numeric_value,
                                "=" => (v - numeric_value).abs() < f64::EPSILON,
                                _ => false,
                            }
                        }
                        ColumnData::Float64(values) => {
                            let v = values[i];
                            match numeric_op {
                                ">" => v > numeric_value,
                                ">=" => v >= numeric_value,
                                "<" => v < numeric_value,
                                "<=" => v <= numeric_value,
                                "=" => (v - numeric_value).abs() < f64::EPSILON,
                                _ => false,
                            }
                        }
                        _ => false,
                    };
                    
                    if num_match {
                        matching_indices.push(row_idx);
                        if matching_indices.len() >= needed {
                            break;
                        }
                    }
                }
                
                row += chunk_rows;
            }
            
            // Apply offset
            let final_indices: Vec<usize> = matching_indices.into_iter().skip(offset).take(limit).collect();
            
            if final_indices.is_empty() {
                return Ok((HashMap::new(), Vec::new()));
            }
            
            // Read columns for matching rows
            let col_indices: Vec<usize> = match column_names {
                Some(names) => names.iter().filter_map(|name| schema.get_index(name)).collect(),
                None => (0..schema.column_count()).collect(),
            };
            
            let mut result = HashMap::with_capacity(col_indices.len());
            for &col_idx in &col_indices {
                let (col_name, col_type) = &schema.columns[col_idx];
                let index_entry = &column_index[col_idx];
                let col_data = self.read_column_scattered_mmap(&mut mmap_cache, file, index_entry, *col_type, &final_indices, total_rows)?;
                result.insert(col_name.clone(), col_data);
            }
            
            return Ok((result, final_indices));
        }
        
        // Fallback for non-dictionary strings
        Err(io::Error::new(io::ErrorKind::InvalidInput, "Combined filter requires dictionary-encoded string column"))
    }

    /// Read a single column for specific row indices
    pub fn read_column_by_indices(
        &self,
        column_name: &str,
        row_indices: &[usize],
    ) -> io::Result<ColumnData> {
        let schema = self.schema.read();
        let column_index = self.column_index.read();
        let header = self.header.read();
        
        let col_idx = schema.get_index(column_name).ok_or_else(|| {
            io::Error::new(io::ErrorKind::NotFound, format!("Column not found: {}", column_name))
        })?;
        
        let (_, col_type) = &schema.columns[col_idx];
        let index_entry = &column_index[col_idx];
        let total_rows = header.row_count as usize;

        let file_guard = self.file.read();
        let file = file_guard.as_ref().ok_or_else(|| {
            io::Error::new(io::ErrorKind::NotConnected, "File not open")
        })?;
        
        let mut mmap_cache = self.mmap_cache.write();
        self.read_column_scattered_mmap(&mut mmap_cache, file, index_entry, *col_type, row_indices, total_rows)
    }

    /// Read IDs for a row range
    pub fn read_ids(&self, start_row: usize, row_count: Option<usize>) -> io::Result<Vec<u64>> {
        let ids = self.ids.read();
        let total = ids.len();
        let start = start_row.min(total);
        let count = row_count.map(|c| c.min(total - start)).unwrap_or(total - start);
        Ok(ids[start..start + count].to_vec())
    }

    /// Read IDs for specific row indices (optimized for scattered access)
    pub fn read_ids_by_indices(&self, row_indices: &[usize]) -> io::Result<Vec<i64>> {
        let ids = self.ids.read();
        let total = ids.len();
        Ok(row_indices.iter()
            .map(|&i| if i < total { ids[i] as i64 } else { 0 })
            .collect())
    }

    /// Execute Complex (Filter+Group+Order) query with single-pass optimization
    /// Key optimization for: SELECT group_col, AGG(agg_col) FROM table WHERE filter_col = 'value'
    /// GROUP BY group_col ORDER BY total DESC LIMIT n
    pub fn execute_filter_group_order(
        &self,
        filter_col: &str,
        filter_val: &str,
        group_col: &str,
        agg_col: Option<&str>,
        agg_func: crate::query::AggregateFunc,
        descending: bool,
        limit: usize,
        offset: usize,
    ) -> io::Result<Option<RecordBatch>> {
        use crate::query::AggregateFunc;
        use arrow::array::{Int64Array, Float64Array, StringArray, UInt32Array};
        use arrow::datatypes::{Field, Schema, DataType as ArrowDataType};
        use std::collections::BinaryHeap;
        use std::cmp::Ordering;
        use std::sync::Arc;

        let schema = self.schema.read();
        let column_index = self.column_index.read();
        let header = self.header.read();

        let total_rows = header.row_count as usize;
        if total_rows == 0 {
            return Ok(Some(RecordBatch::new_empty(Arc::new(Schema::empty()))));
        }

        // Get column indices
        let filter_idx = match schema.get_index(filter_col) {
            Some(idx) => idx,
            None => return Ok(None),
        };
        let group_idx = match schema.get_index(group_col) {
            Some(idx) => idx,
            None => return Ok(None),
        };

        let file_guard = self.file.read();
        let file = file_guard.as_ref().ok_or_else(|| {
            io::Error::new(io::ErrorKind::NotConnected, "File not open")
        })?;
        
        let mut mmap_cache = self.mmap_cache.write();

        // Read filter column to find matching rows
        let filter_index = &column_index[filter_idx];
        let (_, filter_col_type) = &schema.columns[filter_idx];
        
        if *filter_col_type != ColumnType::StringDict {
            return Ok(None); // Only optimized for dictionary-encoded strings
        }

        // Read filter column header and dictionary
        let filter_base = filter_index.data_offset;
        let mut header_buf = [0u8; 16];
        mmap_cache.read_at(file, &mut header_buf, filter_base)?;
        let stored_rows = u64::from_le_bytes(header_buf[0..8].try_into().unwrap()) as usize;
        let dict_size = u64::from_le_bytes(header_buf[8..16].try_into().unwrap()) as usize;
        
        // Read dictionary
        let dict_offsets_offset = filter_base + 16 + (stored_rows * 4) as u64;
        let mut dict_offsets_buf = vec![0u8; dict_size * 4];
        mmap_cache.read_at(file, &mut dict_offsets_buf, dict_offsets_offset)?;
        let dict_offsets: Vec<u32> = (0..dict_size)
            .map(|i| u32::from_le_bytes(dict_offsets_buf[i*4..(i+1)*4].try_into().unwrap()))
            .collect();
        
        let data_len_offset = dict_offsets_offset + (dict_size * 4) as u64;
        let mut data_len_buf = [0u8; 8];
        mmap_cache.read_at(file, &mut data_len_buf, data_len_offset)?;
        let dict_data_len = u64::from_le_bytes(data_len_buf) as usize;
        
        let dict_data_offset = data_len_offset + 8;
        let mut dict_data = vec![0u8; dict_data_len];
        if dict_data_len > 0 {
            mmap_cache.read_at(file, &mut dict_data, dict_data_offset)?;
        }
        
        // Find filter value in dictionary
        let filter_bytes = filter_val.as_bytes();
        let mut target_dict_idx: Option<u32> = None;
        for i in 0..dict_size.saturating_sub(1) {
            let start = dict_offsets[i] as usize;
            let end = if i + 1 < dict_size { dict_offsets[i + 1] as usize } else { dict_data_len };
            if &dict_data[start..end] == filter_bytes {
                target_dict_idx = Some((i + 1) as u32);
                break;
            }
        }
        
        let target_idx = match target_dict_idx {
            Some(idx) => idx,
            None => return Ok(Some(RecordBatch::new_empty(Arc::new(Schema::empty())))),
        };
        
        // Read group column dictionary
        let group_index = &column_index[group_idx];
        let (_, group_col_type) = &schema.columns[group_idx];
        
        if *group_col_type != ColumnType::StringDict {
            return Ok(None);
        }
        
        let group_base = group_index.data_offset;
        let mut group_header = [0u8; 16];
        mmap_cache.read_at(file, &mut group_header, group_base)?;
        let group_rows = u64::from_le_bytes(group_header[0..8].try_into().unwrap()) as usize;
        let group_dict_size = u64::from_le_bytes(group_header[8..16].try_into().unwrap()) as usize;
        
        let group_dict_offsets_offset = group_base + 16 + (group_rows * 4) as u64;
        let mut group_dict_offsets_buf = vec![0u8; group_dict_size * 4];
        mmap_cache.read_at(file, &mut group_dict_offsets_buf, group_dict_offsets_offset)?;
        let group_dict_offsets: Vec<u32> = (0..group_dict_size)
            .map(|i| u32::from_le_bytes(group_dict_offsets_buf[i*4..(i+1)*4].try_into().unwrap()))
            .collect();
        
        let group_data_len_offset = group_dict_offsets_offset + (group_dict_size * 4) as u64;
        let mut group_data_len_buf = [0u8; 8];
        mmap_cache.read_at(file, &mut group_data_len_buf, group_data_len_offset)?;
        let group_dict_data_len = u64::from_le_bytes(group_data_len_buf) as usize;
        
        let group_dict_data_offset = group_data_len_offset + 8;
        let mut group_dict_data = vec![0u8; group_dict_data_len];
        if group_dict_data_len > 0 {
            mmap_cache.read_at(file, &mut group_dict_data, group_dict_data_offset)?;
        }
        
        // Aggregate: group_idx -> (count, sum)
        let mut group_counts: Vec<i64> = vec![0; group_dict_size];
        let mut group_sums: Vec<f64> = vec![0.0; group_dict_size];
        
        // Read filter indices and aggregate
        let filter_indices_offset = filter_base + 16;
        let group_indices_offset = group_base + 16;
        
        // Read agg column if specified
        let agg_idx = agg_col.and_then(|name| schema.get_index(name));
        let agg_values: Option<Vec<f64>> = if let Some(idx) = agg_idx {
            let agg_index = &column_index[idx];
            let (_, agg_col_type) = &schema.columns[idx];
            if *agg_col_type == ColumnType::Float64 || *agg_col_type == ColumnType::Int64 {
                let agg_base = agg_index.data_offset + 8; // Skip count header
                let mut agg_buf = vec![0u8; stored_rows * 8];
                mmap_cache.read_at(file, &mut agg_buf, agg_base)?;
                let values: Vec<f64> = (0..stored_rows)
                    .map(|i| {
                        let bytes = &agg_buf[i*8..(i+1)*8];
                        f64::from_le_bytes(bytes.try_into().unwrap())
                    })
                    .collect();
                Some(values)
            } else {
                None
            }
        } else {
            None
        };
        
        // Single-pass aggregation
        const CHUNK_SIZE: usize = 8192;
        let mut filter_chunk = vec![0u32; CHUNK_SIZE];
        let mut group_chunk = vec![0u32; CHUNK_SIZE];
        
        let mut row = 0;
        while row < stored_rows {
            let chunk_rows = CHUNK_SIZE.min(stored_rows - row);
            
            // Read filter indices chunk
            let filter_bytes = unsafe {
                std::slice::from_raw_parts_mut(filter_chunk.as_mut_ptr() as *mut u8, chunk_rows * 4)
            };
            mmap_cache.read_at(file, filter_bytes, filter_indices_offset + (row * 4) as u64)?;
            
            // Read group indices chunk
            let group_bytes = unsafe {
                std::slice::from_raw_parts_mut(group_chunk.as_mut_ptr() as *mut u8, chunk_rows * 4)
            };
            mmap_cache.read_at(file, group_bytes, group_indices_offset + (row * 4) as u64)?;
            
            // Aggregate
            for i in 0..chunk_rows {
                if filter_chunk[i] == target_idx {
                    let g_idx = group_chunk[i] as usize;
                    if g_idx > 0 && g_idx < group_dict_size {
                        group_counts[g_idx] += 1;
                        if let Some(ref vals) = agg_values {
                            group_sums[g_idx] += vals[row + i];
                        }
                    }
                }
            }
            row += chunk_rows;
        }
        
        // Build result with top-k
        #[derive(Clone, Copy)]
        struct HeapItem { idx: usize, count: i64, sum: f64 }
        
        impl Ord for HeapItem {
            fn cmp(&self, other: &Self) -> Ordering {
                let self_val = self.count;
                let other_val = other.count;
                self_val.cmp(&other_val)
            }
        }
        
        impl PartialOrd for HeapItem {
            fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
                Some(self.cmp(other))
            }
        }
        
        impl Eq for HeapItem {}
        impl PartialEq for HeapItem {
            fn eq(&self, other: &Self) -> bool {
                self.count == other.count
            }
        }
        
        let mut heap = BinaryHeap::new();
        for i in 1..group_dict_size {
            if group_counts[i] > 0 {
                if heap.len() < limit + offset {
                    heap.push(HeapItem { idx: i, count: group_counts[i], sum: group_sums[i] });
                } else if let Some(min) = heap.peek() {
                    if group_counts[i] > min.count {
                        heap.pop();
                        heap.push(HeapItem { idx: i, count: group_counts[i], sum: group_sums[i] });
                    }
                }
            }
        }
        
        // Extract results
        let mut results: Vec<HeapItem> = heap.into_vec();
        results.sort_by(|a, b| {
            if descending {
                b.count.cmp(&a.count)
            } else {
                a.count.cmp(&b.count)
            }
        });
        
        // Apply offset and limit
        let final_results: Vec<HeapItem> = results.into_iter().skip(offset).take(limit).collect();
        
        if final_results.is_empty() {
            return Ok(Some(RecordBatch::new_empty(Arc::new(Schema::empty()))));
        }
        
        // Build Arrow arrays
        let group_strings: Vec<&str> = final_results.iter()
            .map(|item| {
                let dict_idx = item.idx - 1;
                let start = group_dict_offsets[dict_idx] as usize;
                let end = if dict_idx + 1 < group_dict_size { group_dict_offsets[dict_idx + 1] as usize } else { group_dict_data_len };
                std::str::from_utf8(&group_dict_data[start..end]).unwrap_or("")
            })
            .collect();
        
        let counts: Vec<i64> = final_results.iter().map(|item| item.count).collect();
        
        let result_schema = Arc::new(Schema::new(vec![
            Field::new(group_col, ArrowDataType::Utf8, false),
            Field::new("total", ArrowDataType::Int64, false),
        ]));
        
        let result_batch = RecordBatch::try_new(
            result_schema,
            vec![
                Arc::new(StringArray::from(group_strings)) as ArrayRef,
                Arc::new(Int64Array::from(counts)) as ArrayRef,
            ],
        ).map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e.to_string()))?;
        
        Ok(Some(result_batch))
    }

    // ========================================================================
    // Internal helpers
    // ========================================================================

    /// Ensure id_to_idx HashMap is built (lazy load)
    /// Called automatically by delete/exists/get_row_idx operations
    fn ensure_id_index(&self) {
        let mut id_to_idx = self.id_to_idx.write();
        if id_to_idx.is_none() {
            let ids = self.ids.read();
            let mut map = HashMap::with_capacity(ids.len());
            for (idx, &id) in ids.iter().enumerate() {
                map.insert(id, idx);
            }
            *id_to_idx = Some(map);
        }
    }

    // ========================================================================
    // Internal read helpers (mmap-based for cross-platform zero-copy reads)
    // ========================================================================

    fn read_column_range_mmap(
        &self,
        mmap_cache: &mut MmapCache,
        file: &File,
        index: &ColumnIndexEntry,
        dtype: ColumnType,
        start_row: usize,
        row_count: usize,
        _total_rows: usize,
    ) -> io::Result<ColumnData> {
        // ColumnData format has an 8-byte count header for all types
        // Format: [count:u64][data...]
        const HEADER_SIZE: u64 = 8;
        
        match dtype {
            ColumnType::Int64 | ColumnType::Int8 | ColumnType::Int16 | ColumnType::Int32 |
            ColumnType::UInt8 | ColumnType::UInt16 | ColumnType::UInt32 | ColumnType::UInt64 => {
                // Format: [count:u64][values:i64*]
                // Zero-copy optimization: read directly into i64 buffer
                let byte_offset = HEADER_SIZE + (start_row * 8) as u64;
                
                let mut values: Vec<i64> = vec![0i64; row_count];
                // SAFETY: i64 has the same memory layout as [u8; 8] on little-endian systems
                // We read directly into the Vec's backing memory to avoid byte-by-byte parsing
                let bytes_slice = unsafe {
                    std::slice::from_raw_parts_mut(
                        values.as_mut_ptr() as *mut u8,
                        row_count * 8
                    )
                };
                mmap_cache.read_at(file, bytes_slice, index.data_offset + byte_offset)?;
                
                // Handle endianness: convert from LE if on BE system
                #[cfg(target_endian = "big")]
                for v in &mut values {
                    *v = i64::from_le(*v);
                }
                
                Ok(ColumnData::Int64(values))
            }
            ColumnType::Float64 | ColumnType::Float32 => {
                // Format: [count:u64][values:f64*]
                // Zero-copy optimization: read directly into f64 buffer
                let byte_offset = HEADER_SIZE + (start_row * 8) as u64;
                
                let mut values: Vec<f64> = vec![0f64; row_count];
                // SAFETY: f64 has the same memory layout as [u8; 8] on little-endian systems
                let bytes_slice = unsafe {
                    std::slice::from_raw_parts_mut(
                        values.as_mut_ptr() as *mut u8,
                        row_count * 8
                    )
                };
                mmap_cache.read_at(file, bytes_slice, index.data_offset + byte_offset)?;
                
                // Handle endianness: convert from LE if on BE system
                #[cfg(target_endian = "big")]
                for v in &mut values {
                    *v = f64::from_le_bytes(v.to_ne_bytes());
                }
                
                Ok(ColumnData::Float64(values))
            }
            ColumnType::Bool => {
                // Format: [len:u64][packed_bits...]
                let start_byte = start_row / 8;
                let end_byte = (start_row + row_count + 7) / 8;
                let byte_count = end_byte - start_byte;
                
                let mut packed = vec![0u8; byte_count];
                mmap_cache.read_at(file, &mut packed, index.data_offset + HEADER_SIZE + start_byte as u64)?;
                
                Ok(ColumnData::Bool { data: packed, len: row_count })
            }
            ColumnType::String | ColumnType::Binary => {
                // Variable-length type: need to read offsets first
                self.read_variable_column_range_mmap(mmap_cache, file, index, dtype, start_row, row_count)
            }
            ColumnType::StringDict => {
                // Native dictionary-encoded string reading
                self.read_string_dict_column_range_mmap(mmap_cache, file, index, start_row, row_count)
            }
            ColumnType::Null => {
                Ok(ColumnData::Int64(vec![0; row_count]))
            }
        }
    }

    fn read_variable_column_range_mmap(
        &self,
        mmap_cache: &mut MmapCache,
        file: &File,
        index: &ColumnIndexEntry,
        dtype: ColumnType,
        start_row: usize,
        row_count: usize,
    ) -> io::Result<ColumnData> {
        // Variable-length format: [count:u64][offsets:u32*][data_len:u64][data:bytes]
        // Read header to get total count
        let mut header_buf = [0u8; 8];
        mmap_cache.read_at(file, &mut header_buf, index.data_offset)?;
        let total_count = u64::from_le_bytes(header_buf) as usize;
        
        if start_row >= total_count {
            return Ok(ColumnData::String { offsets: vec![0], data: Vec::new() });
        }
        
        let actual_count = row_count.min(total_count - start_row);
        
        // OPTIMIZATION: Read offsets directly into u32 Vec using bulk read
        let offset_start = 8 + start_row * 4; // skip count header
        let offset_count = actual_count + 1;
        let mut offsets: Vec<u32> = vec![0u32; offset_count];
        // SAFETY: u32 slice can be safely viewed as bytes for reading
        let offset_bytes = unsafe {
            std::slice::from_raw_parts_mut(offsets.as_mut_ptr() as *mut u8, offset_count * 4)
        };
        mmap_cache.read_at(file, offset_bytes, index.data_offset + offset_start as u64)?;
        
        // Handle endianness on big-endian systems
        #[cfg(target_endian = "big")]
        for off in &mut offsets {
            *off = u32::from_le(*off);
        }
        
        // Calculate data range
        let data_start = offsets[0];
        let data_end = offsets[actual_count];
        let data_len = (data_end - data_start) as usize;
        
        // Read data portion
        // Data starts after: 8 (count) + (total_count+1)*4 (offsets) + 8 (data_len)
        let data_offset_in_file = index.data_offset + 8 + (total_count + 1) as u64 * 4 + 8 + data_start as u64;
        let mut data = vec![0u8; data_len];
        if data_len > 0 {
            mmap_cache.read_at(file, &mut data, data_offset_in_file)?;
        }
        
        // Normalize offsets to start at 0 using SIMD-friendly subtraction
        let base = offsets[0];
        if base != 0 {
            for off in &mut offsets {
                *off -= base;
            }
        }
        
        match dtype {
            ColumnType::String => Ok(ColumnData::String { offsets, data }),
            ColumnType::Binary => Ok(ColumnData::Binary { offsets, data }),
            _ => Err(io::Error::new(io::ErrorKind::InvalidData, "Not a variable type")),
        }
    }

    /// Read StringDict column with native format
    /// Format: [row_count:u64][dict_size:u64][indices:u32*row_count][dict_offsets:u32*dict_size][dict_data_len:u64][dict_data]
    /// OPTIMIZED: Uses bulk read for u32 arrays instead of per-element parsing
    fn read_string_dict_column_range_mmap(
        &self,
        mmap_cache: &mut MmapCache,
        file: &File,
        index: &ColumnIndexEntry,
        start_row: usize,
        row_count: usize,
    ) -> io::Result<ColumnData> {
        let base_offset = index.data_offset;
        
        // Read header: [row_count:u64][dict_size:u64]
        let mut header = [0u8; 16];
        mmap_cache.read_at(file, &mut header, base_offset)?;
        let total_rows = u64::from_le_bytes(header[0..8].try_into().unwrap()) as usize;
        let dict_size = u64::from_le_bytes(header[8..16].try_into().unwrap()) as usize;
        
        if start_row >= total_rows {
            return Ok(ColumnData::StringDict {
                indices: Vec::new(),
                dict_offsets: vec![0],
                dict_data: Vec::new(),
            });
        }
        
        let actual_count = row_count.min(total_rows - start_row);
        
        // OPTIMIZATION: Read indices directly into Vec<u32>
        let indices_offset = base_offset + 16 + (start_row * 4) as u64;
        let mut indices: Vec<u32> = vec![0u32; actual_count];
        let indices_bytes = unsafe {
            std::slice::from_raw_parts_mut(indices.as_mut_ptr() as *mut u8, actual_count * 4)
        };
        mmap_cache.read_at(file, indices_bytes, indices_offset)?;
        
        #[cfg(target_endian = "big")]
        for idx in &mut indices {
            *idx = u32::from_le(*idx);
        }
        
        // OPTIMIZATION: Read dict_offsets directly into Vec<u32>
        let dict_offsets_offset = base_offset + 16 + (total_rows * 4) as u64;
        let mut dict_offsets: Vec<u32> = vec![0u32; dict_size];
        let dict_offsets_bytes = unsafe {
            std::slice::from_raw_parts_mut(dict_offsets.as_mut_ptr() as *mut u8, dict_size * 4)
        };
        mmap_cache.read_at(file, dict_offsets_bytes, dict_offsets_offset)?;
        
        #[cfg(target_endian = "big")]
        for off in &mut dict_offsets {
            *off = u32::from_le(*off);
        }
        
        // Read dict_data_len and dict_data
        let dict_data_len_offset = dict_offsets_offset + (dict_size * 4) as u64;
        let mut data_len_buf = [0u8; 8];
        mmap_cache.read_at(file, &mut data_len_buf, dict_data_len_offset)?;
        let dict_data_len = u64::from_le_bytes(data_len_buf) as usize;
        
        let dict_data_offset = dict_data_len_offset + 8;
        let mut dict_data = vec![0u8; dict_data_len];
        if dict_data_len > 0 {
            mmap_cache.read_at(file, &mut dict_data, dict_data_offset)?;
        }
        
        Ok(ColumnData::StringDict {
            indices,
            dict_offsets,
            dict_data,
        })
    }

    /// Read StringDict column with scattered row indices
    /// OPTIMIZED: Only reads the specific indices needed, not all indices
    fn read_string_dict_column_scattered_mmap(
        &self,
        mmap_cache: &mut MmapCache,
        file: &File,
        index: &ColumnIndexEntry,
        row_indices: &[usize],
    ) -> io::Result<ColumnData> {
        if row_indices.is_empty() {
            return Ok(ColumnData::StringDict {
                indices: Vec::new(),
                dict_offsets: vec![0],
                dict_data: Vec::new(),
            });
        }
        
        let base_offset = index.data_offset;
        
        // Read header: [row_count:u64][dict_size:u64]
        let mut header = [0u8; 16];
        mmap_cache.read_at(file, &mut header, base_offset)?;
        let total_rows = u64::from_le_bytes(header[0..8].try_into().unwrap()) as usize;
        let dict_size = u64::from_le_bytes(header[8..16].try_into().unwrap()) as usize;
        
        let all_indices_offset = base_offset + 16;
        let n = row_indices.len();
        
        // OPTIMIZED: Read only the specific indices we need instead of all indices
        // For small scattered reads, read individually; for dense reads, read a range
        let mut indices = Vec::with_capacity(n);
        
        if n <= 256 {
            // Small number of indices - read each one individually using thread-local buffer
            SCATTERED_READ_BUF.with(|buf| {
                let mut buf = buf.borrow_mut();
                buf.resize(4, 0);
                for &row_idx in row_indices {
                    if row_idx < total_rows {
                        mmap_cache.read_at(file, &mut buf[..4], all_indices_offset + (row_idx * 4) as u64)?;
                        indices.push(u32::from_le_bytes([buf[0], buf[1], buf[2], buf[3]]));
                    } else {
                        indices.push(0);
                    }
                }
                Ok::<(), io::Error>(())
            })?;
        } else {
            // For larger reads, find min/max and read that range
            let min_idx = *row_indices.iter().min().unwrap_or(&0);
            let max_idx = *row_indices.iter().max().unwrap_or(&0);
            let range_size = max_idx - min_idx + 1;
            
            // OPTIMIZATION: If range is reasonably dense, read whole range as Vec<u32>
            if range_size <= n * 4 && range_size <= total_rows {
                let mut range_values: Vec<u32> = vec![0u32; range_size];
                let range_bytes = unsafe {
                    std::slice::from_raw_parts_mut(range_values.as_mut_ptr() as *mut u8, range_size * 4)
                };
                mmap_cache.read_at(file, range_bytes, all_indices_offset + (min_idx * 4) as u64)?;
                
                #[cfg(target_endian = "big")]
                for v in &mut range_values {
                    *v = u32::from_le(*v);
                }
                
                for &row_idx in row_indices {
                    if row_idx < total_rows {
                        let local_idx = row_idx - min_idx;
                        indices.push(range_values[local_idx]);
                    } else {
                        indices.push(0);
                    }
                }
            } else {
                // Sparse - read individually using thread-local buffer
                SCATTERED_READ_BUF.with(|buf| {
                    let mut buf = buf.borrow_mut();
                    buf.resize(4, 0);
                    for &row_idx in row_indices {
                        if row_idx < total_rows {
                            mmap_cache.read_at(file, &mut buf[..4], all_indices_offset + (row_idx * 4) as u64)?;
                            indices.push(u32::from_le_bytes([buf[0], buf[1], buf[2], buf[3]]));
                        } else {
                            indices.push(0);
                        }
                    }
                    Ok::<(), io::Error>(())
                })?;
            }
        }
        
        // OPTIMIZATION: Read dict_offsets directly into Vec<u32>
        let dict_offsets_offset = base_offset + 16 + (total_rows * 4) as u64;
        let mut dict_offsets: Vec<u32> = vec![0u32; dict_size];
        let dict_offsets_bytes = unsafe {
            std::slice::from_raw_parts_mut(dict_offsets.as_mut_ptr() as *mut u8, dict_size * 4)
        };
        mmap_cache.read_at(file, dict_offsets_bytes, dict_offsets_offset)?;
        
        #[cfg(target_endian = "big")]
        for off in &mut dict_offsets {
            *off = u32::from_le(*off);
        }
        
        // Read dict_data_len and dict_data
        let dict_data_len_offset = dict_offsets_offset + (dict_size * 4) as u64;
        let mut data_len_buf = [0u8; 8];
        mmap_cache.read_at(file, &mut data_len_buf, dict_data_len_offset)?;
        let dict_data_len = u64::from_le_bytes(data_len_buf) as usize;
        
        let dict_data_offset = dict_data_len_offset + 8;
        let mut dict_data = vec![0u8; dict_data_len];
        if dict_data_len > 0 {
            mmap_cache.read_at(file, &mut dict_data, dict_data_offset)?;
        }
        
        Ok(ColumnData::StringDict {
            indices,
            dict_offsets,
            dict_data,
        })
    }

    /// Optimized scattered read for variable-length columns (String/Binary) using mmap
    fn read_variable_column_scattered_mmap(
        &self,
        mmap_cache: &mut MmapCache,
        file: &File,
        index: &ColumnIndexEntry,
        dtype: ColumnType,
        row_indices: &[usize],
    ) -> io::Result<ColumnData> {
        if row_indices.is_empty() {
            return Ok(match dtype {
                ColumnType::String => ColumnData::String { offsets: vec![0], data: Vec::new() },
                _ => ColumnData::Binary { offsets: vec![0], data: Vec::new() },
            });
        }

        // Variable-length format: [count:u64][offsets:u32*(count+1)][data_len:u64][data:bytes]
        // Read header to get total count
        let mut header_buf = [0u8; 8];
        mmap_cache.read_at(file, &mut header_buf, index.data_offset)?;
        let total_count = u64::from_le_bytes(header_buf) as usize;

        // Read only the offsets we need (need idx and idx+1 for each row)
        // Collect unique offset indices needed
        let mut offset_indices: Vec<usize> = Vec::with_capacity(row_indices.len() * 2);
        for &idx in row_indices {
            if idx < total_count {
                offset_indices.push(idx);
                offset_indices.push(idx + 1);
            }
        }
        offset_indices.sort_unstable();
        offset_indices.dedup();

        if offset_indices.is_empty() {
            return Ok(match dtype {
                ColumnType::String => ColumnData::String { offsets: vec![0], data: Vec::new() },
                _ => ColumnData::Binary { offsets: vec![0], data: Vec::new() },
            });
        }

        // Read required offsets in batches (optimize for contiguous ranges)
        let mut offset_map: HashMap<usize, u32> = HashMap::with_capacity(offset_indices.len());
        let offset_base = index.data_offset + 8; // skip count header
        
        // For small number of indices, read individually
        // For larger sets, read a range that covers all needed offsets
        let min_idx = *offset_indices.first().unwrap();
        let max_idx = *offset_indices.last().unwrap();
        
        if max_idx - min_idx < offset_indices.len() * 4 {
            // Indices are sparse enough - read range
            let range_count = max_idx - min_idx + 1;
            let mut offset_buf = vec![0u8; range_count * 4];
            mmap_cache.read_at(file, &mut offset_buf, offset_base + (min_idx * 4) as u64)?;
            
            for &idx in &offset_indices {
                let local_idx = idx - min_idx;
                let off = u32::from_le_bytes(offset_buf[local_idx * 4..(local_idx + 1) * 4].try_into().unwrap());
                offset_map.insert(idx, off);
            }
        } else {
            // Very sparse - read individually
            let mut buf = [0u8; 4];
            for &idx in &offset_indices {
                mmap_cache.read_at(file, &mut buf, offset_base + (idx * 4) as u64)?;
                offset_map.insert(idx, u32::from_le_bytes(buf));
            }
        }

        // Calculate data offset base: skip count(8) + offsets((total_count+1)*4) + data_len(8)
        let data_base = index.data_offset + 8 + (total_count + 1) as u64 * 4 + 8;

        // Read data for each requested row and build result
        let mut result_offsets = vec![0u32];
        let mut result_data = Vec::new();

        for &idx in row_indices {
            if idx < total_count {
                let start = *offset_map.get(&idx).unwrap_or(&0);
                let end = *offset_map.get(&(idx + 1)).unwrap_or(&start);
                let len = (end - start) as usize;
                
                if len > 0 {
                    let mut chunk = vec![0u8; len];
                    mmap_cache.read_at(file, &mut chunk, data_base + start as u64)?;
                    result_data.extend_from_slice(&chunk);
                }
                result_offsets.push(result_data.len() as u32);
            } else {
                // Out of bounds - push empty
                result_offsets.push(result_data.len() as u32);
            }
        }

        match dtype {
            ColumnType::String => Ok(ColumnData::String { offsets: result_offsets, data: result_data }),
            ColumnType::Binary => Ok(ColumnData::Binary { offsets: result_offsets, data: result_data }),
            _ => Err(io::Error::new(io::ErrorKind::InvalidData, "Not a variable type")),
        }
    }
    
    /// Optimized scattered read for numeric types using row-group based I/O
    /// Reads data in larger chunks (row-groups) to reduce number of I/O operations
    fn read_numeric_scattered_optimized<T: Copy + Default + 'static>(
        mmap_cache: &mut MmapCache,
        file: &File,
        index: &ColumnIndexEntry,
        row_indices: &[usize],
        header_size: u64,
    ) -> io::Result<Vec<T>> {
        if row_indices.is_empty() {
            return Ok(Vec::new());
        }
        
        let n = row_indices.len();
        let elem_size = std::mem::size_of::<T>();
        
        // For small numbers, simple sequential read without sorting is faster
        // Typical LIMIT queries (100-500 rows) benefit from avoiding sort overhead
        if n <= 256 {
            let mut values = Vec::with_capacity(n);
            SCATTERED_READ_BUF.with(|buf| {
                let mut buf = buf.borrow_mut();
                buf.resize(8, 0);
                for &idx in row_indices {
                    mmap_cache.read_at(file, &mut buf[..elem_size], index.data_offset + header_size + (idx * elem_size) as u64)?;
                    let val: T = unsafe { std::ptr::read(buf.as_ptr() as *const T) };
                    values.push(val);
                }
                Ok::<(), io::Error>(())
            })?;
            return Ok(values);
        }
        
        // ROW-GROUP BASED READING for larger scattered reads
        const ROW_GROUP_SIZE: usize = 8192;
        
        // Sort indices and track original positions
        let mut indexed: Vec<(usize, usize)> = row_indices.iter().enumerate().map(|(i, &idx)| (idx, i)).collect();
        indexed.sort_unstable_by_key(|&(idx, _)| idx);
        
        let mut result: Vec<T> = vec![T::default(); n];
        let mut i = 0;
        
        // Process by row-groups
        while i < indexed.len() {
            let first_idx = indexed[i].0;
            let group_start = (first_idx / ROW_GROUP_SIZE) * ROW_GROUP_SIZE;
            let group_end = group_start + ROW_GROUP_SIZE;
            
            // Find all indices within this row-group
            let mut group_indices = Vec::new();
            while i < indexed.len() && indexed[i].0 < group_end {
                group_indices.push(indexed[i]);
                i += 1;
            }
            
            // Decide read strategy based on density within group
            let indices_in_group = group_indices.len();
            let span = group_indices.last().unwrap().0 - group_indices.first().unwrap().0 + 1;
            
            // If indices are dense enough, read the span; otherwise read full group
            if indices_in_group * 4 >= span || span <= 256 {
                // Dense or small span: read just the span
                let read_start = group_indices.first().unwrap().0;
                let read_len = span;
                let mut buf: Vec<u8> = vec![0u8; read_len * elem_size];
                mmap_cache.read_at(file, &mut buf, index.data_offset + header_size + (read_start * elem_size) as u64)?;
                
                for (idx, orig_pos) in group_indices {
                    let offset = idx - read_start;
                    let val: T = unsafe { std::ptr::read(buf.as_ptr().add(offset * elem_size) as *const T) };
                    result[orig_pos] = val;
                }
            } else {
                // Sparse: read individual values (but they're sorted so still sequential-ish)
                let mut buf = [0u8; 8];
                for (idx, orig_pos) in group_indices {
                    mmap_cache.read_at(file, &mut buf[..elem_size], index.data_offset + header_size + (idx * elem_size) as u64)?;
                    let val: T = unsafe { std::ptr::read(buf.as_ptr() as *const T) };
                    result[orig_pos] = val;
                }
            }
        }
        
        Ok(result)
    }

    fn read_column_scattered_mmap(
        &self,
        mmap_cache: &mut MmapCache,
        file: &File,
        index: &ColumnIndexEntry,
        dtype: ColumnType,
        row_indices: &[usize],
        _total_rows: usize,
    ) -> io::Result<ColumnData> {
        // ColumnData format has an 8-byte count header
        const HEADER_SIZE: u64 = 8;
        
        match dtype {
            ColumnType::Int64 | ColumnType::Int8 | ColumnType::Int16 | ColumnType::Int32 |
            ColumnType::UInt8 | ColumnType::UInt16 | ColumnType::UInt32 | ColumnType::UInt64 => {
                Self::read_numeric_scattered_optimized::<i64>(mmap_cache, file, index, row_indices, HEADER_SIZE)
                    .map(ColumnData::Int64)
            }
            ColumnType::Float64 | ColumnType::Float32 => {
                Self::read_numeric_scattered_optimized::<f64>(mmap_cache, file, index, row_indices, HEADER_SIZE)
                    .map(ColumnData::Float64)
            }
            ColumnType::String | ColumnType::Binary => {
                // Optimized scattered read for variable-length types
                self.read_variable_column_scattered_mmap(mmap_cache, file, index, dtype, row_indices)
            }
            ColumnType::Bool => {
                // Bool is stored as packed bits: [count:u64][packed_bits...]
                // Read the packed bits and extract specific indices
                let packed_len = (index.data_length as usize - 8 + 7) / 8;
                let mut packed = vec![0u8; packed_len.max(1)];
                if packed_len > 0 {
                    mmap_cache.read_at(file, &mut packed, index.data_offset + HEADER_SIZE)?;
                }
                
                // Extract the specific bits for requested indices
                let mut result_packed = vec![0u8; (row_indices.len() + 7) / 8];
                for (result_idx, &src_idx) in row_indices.iter().enumerate() {
                    let src_byte = src_idx / 8;
                    let src_bit = src_idx % 8;
                    let bit_value = if src_byte < packed.len() {
                        (packed[src_byte] >> src_bit) & 1
                    } else {
                        0
                    };
                    
                    let dst_byte = result_idx / 8;
                    let dst_bit = result_idx % 8;
                    if bit_value == 1 {
                        result_packed[dst_byte] |= 1 << dst_bit;
                    }
                }
                
                Ok(ColumnData::Bool { data: result_packed, len: row_indices.len() })
            }
            ColumnType::StringDict => {
                // Native dictionary-encoded string scattered read
                self.read_string_dict_column_scattered_mmap(mmap_cache, file, index, row_indices)
            }
            ColumnType::Null => {
                // Null column - return empty Int64 as placeholder
                Ok(ColumnData::Int64(vec![0i64; row_indices.len()]))
            }
        }
    }

    // ========================================================================
    // Write APIs
    // ========================================================================

    /// Insert typed columns directly
    pub fn insert_typed(
        &self,
        int_columns: HashMap<String, Vec<i64>>,
        float_columns: HashMap<String, Vec<f64>>,
        string_columns: HashMap<String, Vec<String>>,
        binary_columns: HashMap<String, Vec<Vec<u8>>>,
        bool_columns: HashMap<String, Vec<bool>>,
    ) -> io::Result<Vec<u64>> {
        // Determine row count as maximum across all columns (for heterogeneous schemas)
        let row_count = int_columns.values().map(|v| v.len()).max().unwrap_or(0)
            .max(float_columns.values().map(|v| v.len()).max().unwrap_or(0))
            .max(string_columns.values().map(|v| v.len()).max().unwrap_or(0))
            .max(binary_columns.values().map(|v| v.len()).max().unwrap_or(0))
            .max(bool_columns.values().map(|v| v.len()).max().unwrap_or(0));

        if row_count == 0 {
            return Ok(Vec::new());
        }

        // Allocate IDs atomically
        let start_id = self.next_id.fetch_add(row_count as u64, Ordering::SeqCst);
        let ids: Vec<u64> = (start_id..start_id + row_count as u64).collect();

        // Ensure schema has all columns, padding existing rows with defaults for new columns
        {
            let mut schema = self.schema.write();
            let mut columns = self.columns.write();
            let mut nulls = self.nulls.write();
            let existing_row_count = self.ids.read().len();

            for name in int_columns.keys() {
                let is_new = schema.get_index(name).is_none();
                let idx = schema.add_column(name, ColumnType::Int64);
                while columns.len() <= idx {
                    let mut col = ColumnData::new(ColumnType::Int64);
                    // Pad with defaults for existing rows if this is a new column
                    if is_new && existing_row_count > 0 {
                        if let ColumnData::Int64(v) = &mut col {
                            v.resize(existing_row_count, 0);
                        }
                    }
                    columns.push(col);
                    nulls.push(Vec::new());
                }
            }
            for name in float_columns.keys() {
                let is_new = schema.get_index(name).is_none();
                let idx = schema.add_column(name, ColumnType::Float64);
                while columns.len() <= idx {
                    let mut col = ColumnData::new(ColumnType::Float64);
                    if is_new && existing_row_count > 0 {
                        if let ColumnData::Float64(v) = &mut col {
                            v.resize(existing_row_count, 0.0);
                        }
                    }
                    columns.push(col);
                    nulls.push(Vec::new());
                }
            }
            for name in string_columns.keys() {
                let is_new = schema.get_index(name).is_none();
                let idx = schema.add_column(name, ColumnType::String);
                while columns.len() <= idx {
                    let mut col = ColumnData::new(ColumnType::String);
                    if is_new && existing_row_count > 0 {
                        if let ColumnData::String { offsets, .. } = &mut col {
                            for _ in 0..existing_row_count {
                                offsets.push(0); // Empty string offset
                            }
                        }
                    }
                    columns.push(col);
                    nulls.push(Vec::new());
                }
            }
            for name in binary_columns.keys() {
                let is_new = schema.get_index(name).is_none();
                let idx = schema.add_column(name, ColumnType::Binary);
                while columns.len() <= idx {
                    let mut col = ColumnData::new(ColumnType::Binary);
                    if is_new && existing_row_count > 0 {
                        if let ColumnData::Binary { offsets, .. } = &mut col {
                            for _ in 0..existing_row_count {
                                offsets.push(0);
                            }
                        }
                    }
                    columns.push(col);
                    nulls.push(Vec::new());
                }
            }
            for name in bool_columns.keys() {
                let is_new = schema.get_index(name).is_none();
                let idx = schema.add_column(name, ColumnType::Bool);
                while columns.len() <= idx {
                    let mut col = ColumnData::new(ColumnType::Bool);
                    if is_new && existing_row_count > 0 {
                        if let ColumnData::Bool { len, .. } = &mut col {
                            *len = existing_row_count;
                        }
                    }
                    columns.push(col);
                    nulls.push(Vec::new());
                }
            }
        }

        // Append IDs
        self.ids.write().extend_from_slice(&ids);

        // Append column data
        {
            let schema = self.schema.read();
            let mut columns = self.columns.write();

            for (name, values) in int_columns {
                if let Some(idx) = schema.get_index(&name) {
                    columns[idx].extend_i64(&values);
                }
            }
            for (name, values) in float_columns {
                if let Some(idx) = schema.get_index(&name) {
                    columns[idx].extend_f64(&values);
                }
            }
            for (name, values) in string_columns {
                if let Some(idx) = schema.get_index(&name) {
                    columns[idx].extend_strings(&values);
                }
            }
            for (name, values) in binary_columns {
                if let Some(idx) = schema.get_index(&name) {
                    columns[idx].extend_bytes(&values);
                }
            }
            for (name, values) in bool_columns {
                if let Some(idx) = schema.get_index(&name) {
                    columns[idx].extend_bools(&values);
                }
            }
        }

        // Update header
        {
            let mut header = self.header.write();
            header.row_count += row_count as u64;
            header.column_count = self.schema.read().column_count() as u32;
            header.modified_at = chrono::Utc::now().timestamp();
        }
        
        // Update id_to_idx mapping only if it's already built
        {
            let ids_guard = self.ids.read();
            let mut id_to_idx = self.id_to_idx.write();
            if let Some(map) = id_to_idx.as_mut() {
                let start_idx = ids_guard.len() - ids.len();
                for (i, &id) in ids.iter().enumerate() {
                    map.insert(id, start_idx + i);
                }
            }
        }
        
        // Extend deleted bitmap with zeros for new rows
        {
            let mut deleted = self.deleted.write();
            let new_len = (self.ids.read().len() + 7) / 8;
            deleted.resize(new_len, 0);
        }
        
        // Update active count (new rows are not deleted)
        self.active_count.fetch_add(row_count as u64, Ordering::Relaxed);

        Ok(ids)
    }

    /// Insert typed columns with explicit NULL tracking for heterogeneous schemas
    pub fn insert_typed_with_nulls(
        &self,
        int_columns: HashMap<String, Vec<i64>>,
        float_columns: HashMap<String, Vec<f64>>,
        string_columns: HashMap<String, Vec<String>>,
        binary_columns: HashMap<String, Vec<Vec<u8>>>,
        bool_columns: HashMap<String, Vec<bool>>,
        null_positions: HashMap<String, Vec<bool>>,
    ) -> io::Result<Vec<u64>> {
        // Determine row count as maximum across all columns
        let row_count = int_columns.values().map(|v| v.len()).max().unwrap_or(0)
            .max(float_columns.values().map(|v| v.len()).max().unwrap_or(0))
            .max(string_columns.values().map(|v| v.len()).max().unwrap_or(0))
            .max(binary_columns.values().map(|v| v.len()).max().unwrap_or(0))
            .max(bool_columns.values().map(|v| v.len()).max().unwrap_or(0));

        if row_count == 0 {
            return Ok(Vec::new());
        }

        // Allocate IDs atomically
        let start_id = self.next_id.fetch_add(row_count as u64, Ordering::SeqCst);
        let ids: Vec<u64> = (start_id..start_id + row_count as u64).collect();

        // Ensure schema has all columns and track column indices
        let mut col_name_to_idx: HashMap<String, usize> = HashMap::new();
        {
            let mut schema = self.schema.write();
            let mut columns = self.columns.write();
            let mut nulls = self.nulls.write();

            for name in int_columns.keys() {
                let idx = schema.add_column(name, ColumnType::Int64);
                col_name_to_idx.insert(name.clone(), idx);
                while columns.len() <= idx {
                    columns.push(ColumnData::new(ColumnType::Int64));
                    nulls.push(Vec::new());
                }
            }
            for name in float_columns.keys() {
                let idx = schema.add_column(name, ColumnType::Float64);
                col_name_to_idx.insert(name.clone(), idx);
                while columns.len() <= idx {
                    columns.push(ColumnData::new(ColumnType::Float64));
                    nulls.push(Vec::new());
                }
            }
            for name in string_columns.keys() {
                let idx = schema.add_column(name, ColumnType::String);
                col_name_to_idx.insert(name.clone(), idx);
                while columns.len() <= idx {
                    columns.push(ColumnData::new(ColumnType::String));
                    nulls.push(Vec::new());
                }
            }
            for name in binary_columns.keys() {
                let idx = schema.add_column(name, ColumnType::Binary);
                col_name_to_idx.insert(name.clone(), idx);
                while columns.len() <= idx {
                    columns.push(ColumnData::new(ColumnType::Binary));
                    nulls.push(Vec::new());
                }
            }
            for name in bool_columns.keys() {
                let idx = schema.add_column(name, ColumnType::Bool);
                col_name_to_idx.insert(name.clone(), idx);
                while columns.len() <= idx {
                    columns.push(ColumnData::new(ColumnType::Bool));
                    nulls.push(Vec::new());
                }
            }
        }

        // Append IDs
        self.ids.write().extend_from_slice(&ids);

        // Append column data
        {
            let schema = self.schema.read();
            let mut columns = self.columns.write();

            for (name, values) in int_columns {
                if let Some(idx) = schema.get_index(&name) {
                    columns[idx].extend_i64(&values);
                }
            }
            for (name, values) in float_columns {
                if let Some(idx) = schema.get_index(&name) {
                    columns[idx].extend_f64(&values);
                }
            }
            for (name, values) in string_columns {
                if let Some(idx) = schema.get_index(&name) {
                    for v in &values {
                        columns[idx].push_string(v);
                    }
                }
            }
            for (name, values) in binary_columns {
                if let Some(idx) = schema.get_index(&name) {
                    for v in &values {
                        columns[idx].push_bytes(v);
                    }
                }
            }
            for (name, values) in bool_columns {
                if let Some(idx) = schema.get_index(&name) {
                    for v in values {
                        columns[idx].push_bool(v);
                    }
                }
            }
        }

        // Update null bitmaps for each column
        {
            let mut nulls = self.nulls.write();
            let base_row = self.ids.read().len() - row_count;
            
            for (col_name, is_null_vec) in null_positions {
                if let Some(&col_idx) = col_name_to_idx.get(&col_name) {
                    if col_idx < nulls.len() {
                        // Extend null bitmap for this column
                        let null_bitmap = &mut nulls[col_idx];
                        for (i, &is_null) in is_null_vec.iter().enumerate() {
                            if is_null {
                                let row_idx = base_row + i;
                                let byte_idx = row_idx / 8;
                                let bit_idx = row_idx % 8;
                                while null_bitmap.len() <= byte_idx {
                                    null_bitmap.push(0);
                                }
                                null_bitmap[byte_idx] |= 1 << bit_idx;
                            }
                        }
                    }
                }
            }
        }

        // Update header
        {
            let mut header = self.header.write();
            header.row_count += row_count as u64;
            header.column_count = self.schema.read().column_count() as u32;
            header.modified_at = chrono::Utc::now().timestamp();
        }
        
        // Update id_to_idx mapping only if it's already built
        {
            let ids_guard = self.ids.read();
            let mut id_to_idx = self.id_to_idx.write();
            if let Some(map) = id_to_idx.as_mut() {
                let start_idx = ids_guard.len() - ids.len();
                for (i, &id) in ids.iter().enumerate() {
                    map.insert(id, start_idx + i);
                }
            }
        }
        
        // Extend deleted bitmap with zeros for new rows
        {
            let mut deleted = self.deleted.write();
            let new_len = (self.ids.read().len() + 7) / 8;
            deleted.resize(new_len, 0);
        }
        
        // Update active count (new rows are not deleted)
        self.active_count.fetch_add(row_count as u64, Ordering::Relaxed);

        Ok(ids)
    }

    // ========================================================================
    // Delete/Update APIs
    // ========================================================================

    /// Delete a row by ID (soft delete)
    /// Returns true if the row was found and deleted
    pub fn delete(&self, id: u64) -> bool {
        self.ensure_id_index();
        let id_to_idx = self.id_to_idx.read();
        let map = id_to_idx.as_ref().unwrap();
        if let Some(&row_idx) = map.get(&id) {
            drop(id_to_idx);  // Release read lock before write
            let mut deleted = self.deleted.write();
            let byte_idx = row_idx / 8;
            let bit_idx = row_idx % 8;
            
            // Ensure bitmap is large enough
            if byte_idx >= deleted.len() {
                deleted.resize(byte_idx + 1, 0);
            }
            
            // Only decrement if not already deleted
            let was_deleted = (deleted[byte_idx] >> bit_idx) & 1 == 1;
            if !was_deleted {
                self.active_count.fetch_sub(1, Ordering::Relaxed);
            }
            
            // Set the deleted bit
            deleted[byte_idx] |= 1 << bit_idx;
            true
        } else {
            false
        }
    }

    /// Delete multiple rows by IDs (soft delete)
    /// Returns true if all rows were found and deleted
    pub fn delete_batch(&self, ids: &[u64]) -> bool {
        self.ensure_id_index();
        let id_to_idx = self.id_to_idx.read();
        let map = id_to_idx.as_ref().unwrap();
        let mut deleted = self.deleted.write();
        let mut all_found = true;
        let mut deleted_count = 0u64;
        
        for &id in ids {
            if let Some(&row_idx) = map.get(&id) {
                let byte_idx = row_idx / 8;
                let bit_idx = row_idx % 8;
                
                if byte_idx >= deleted.len() {
                    deleted.resize(byte_idx + 1, 0);
                }
                
                // Only count if not already deleted
                let was_deleted = (deleted[byte_idx] >> bit_idx) & 1 == 1;
                if !was_deleted {
                    deleted_count += 1;
                }
                
                deleted[byte_idx] |= 1 << bit_idx;
            } else {
                all_found = false;
            }
        }
        
        // Update active count
        if deleted_count > 0 {
            self.active_count.fetch_sub(deleted_count, Ordering::Relaxed);
        }
        
        all_found
    }

    /// Check if a row is deleted
    pub fn is_deleted(&self, row_idx: usize) -> bool {
        let deleted = self.deleted.read();
        let byte_idx = row_idx / 8;
        let bit_idx = row_idx % 8;
        
        if byte_idx < deleted.len() {
            (deleted[byte_idx] >> bit_idx) & 1 == 1
        } else {
            false
        }
    }

    /// Check if an ID exists and is not deleted
    pub fn exists(&self, id: u64) -> bool {
        self.ensure_id_index();
        let id_to_idx = self.id_to_idx.read();
        let map = id_to_idx.as_ref().unwrap();
        if let Some(&row_idx) = map.get(&id) {
            !self.is_deleted(row_idx)
        } else {
            false
        }
    }

    /// Get row index for an ID (None if not found or deleted)
    pub fn get_row_idx(&self, id: u64) -> Option<usize> {
        self.ensure_id_index();
        let id_to_idx = self.id_to_idx.read();
        let map = id_to_idx.as_ref().unwrap();
        if let Some(&row_idx) = map.get(&id) {
            if !self.is_deleted(row_idx) {
                Some(row_idx)
            } else {
                None
            }
        } else {
            None
        }
    }

    /// OPTIMIZED: Read a single row by ID using O(1) index lookup
    /// Returns HashMap of column_name -> ColumnData (single element)
    /// Much faster than WHERE _id = X which scans all data
    pub fn read_row_by_id(&self, id: u64, column_names: Option<&[&str]>) -> io::Result<Option<HashMap<String, ColumnData>>> {
        // O(1) lookup using id_to_idx index
        let row_idx = match self.get_row_idx(id) {
            Some(idx) => idx,
            None => return Ok(None),
        };
        
        // Read only the single row using scattered read
        let indices = vec![row_idx];
        let schema = self.schema.read();
        let column_index = self.column_index.read();
        
        // Get columns to read
        let cols_to_read: Vec<(usize, &str, ColumnType)> = if let Some(names) = column_names {
            names.iter()
                .filter_map(|&name| {
                    if name == "_id" {
                        None // Handle _id separately
                    } else {
                        schema.get_index(name).map(|idx| {
                            (idx, name, schema.columns[idx].1)
                        })
                    }
                })
                .collect()
        } else {
            schema.columns.iter().enumerate()
                .map(|(idx, (name, dtype))| (idx, name.as_str(), *dtype))
                .collect()
        };
        
        let mut result = HashMap::new();
        
        // Add _id if requested or no column filter
        let include_id = column_names.map(|cols| cols.contains(&"_id")).unwrap_or(true);
        if include_id {
            result.insert("_id".to_string(), ColumnData::Int64(vec![id as i64]));
        }
        
        // Read each requested column for the single row
        let file_guard = self.file.read();
        let file = file_guard.as_ref().ok_or_else(|| {
            io::Error::new(io::ErrorKind::NotFound, "File not open")
        })?;
        let mut mmap_cache = self.mmap_cache.write();
        
        let total_rows = self.header.read().row_count as usize;
        for (col_idx, col_name, col_type) in cols_to_read {
            if col_idx >= column_index.len() {
                continue;
            }
            let index = &column_index[col_idx];
            let col_data = self.read_column_scattered_mmap(&mut mmap_cache, file, index, col_type, &indices, total_rows)?;
            result.insert(col_name.to_string(), col_data);
        }
        
        Ok(Some(result))
    }

    /// OPTIMIZED: Read multiple rows by IDs using O(1) index lookups
    /// Returns Vec of (id, row_data) for found rows
    pub fn read_rows_by_ids(&self, ids: &[u64], column_names: Option<&[&str]>) -> io::Result<Vec<(u64, HashMap<String, ColumnData>)>> {
        if ids.is_empty() {
            return Ok(Vec::new());
        }
        
        // Build id_to_idx if needed
        self.ensure_id_index();
        let id_to_idx = self.id_to_idx.read();
        let map = id_to_idx.as_ref().unwrap();
        
        // Collect valid row indices
        let mut valid_ids_indices: Vec<(u64, usize)> = Vec::with_capacity(ids.len());
        for &id in ids {
            if let Some(&row_idx) = map.get(&id) {
                if !self.is_deleted(row_idx) {
                    valid_ids_indices.push((id, row_idx));
                }
            }
        }
        
        if valid_ids_indices.is_empty() {
            return Ok(Vec::new());
        }
        
        let indices: Vec<usize> = valid_ids_indices.iter().map(|(_, idx)| *idx).collect();
        drop(id_to_idx);
        
        // Read columns using scattered read
        let schema = self.schema.read();
        let column_index = self.column_index.read();
        
        let cols_to_read: Vec<(usize, String, ColumnType)> = if let Some(names) = column_names {
            names.iter()
                .filter_map(|&name| {
                    if name == "_id" {
                        None
                    } else {
                        schema.get_index(name).map(|idx| {
                            (idx, name.to_string(), schema.columns[idx].1)
                        })
                    }
                })
                .collect()
        } else {
            schema.columns.iter().enumerate()
                .map(|(idx, (name, dtype))| (idx, name.clone(), *dtype))
                .collect()
        };
        
        let file_guard = self.file.read();
        let file = file_guard.as_ref().ok_or_else(|| {
            io::Error::new(io::ErrorKind::NotFound, "File not open")
        })?;
        let mut mmap_cache = self.mmap_cache.write();
        
        // Read all columns for all indices
        let mut column_data: HashMap<String, ColumnData> = HashMap::new();
        let include_id = column_names.map(|cols| cols.contains(&"_id")).unwrap_or(true);
        
        let total_rows = self.header.read().row_count as usize;
        for (col_idx, col_name, col_type) in cols_to_read {
            if col_idx >= column_index.len() {
                continue;
            }
            let index = &column_index[col_idx];
            let col_data = self.read_column_scattered_mmap(&mut mmap_cache, file, index, col_type, &indices, total_rows)?;
            column_data.insert(col_name, col_data);
        }
        
        // Split into per-row results
        let mut results = Vec::with_capacity(valid_ids_indices.len());
        for (i, (id, _)) in valid_ids_indices.iter().enumerate() {
            let mut row_data = HashMap::new();
            if include_id {
                row_data.insert("_id".to_string(), ColumnData::Int64(vec![*id as i64]));
            }
            for (col_name, col_data) in &column_data {
                let single_val = col_data.filter_by_indices(&[i]);
                row_data.insert(col_name.clone(), single_val);
            }
            results.push((*id, row_data));
        }
        
        Ok(results)
    }

    /// Get the count of non-deleted rows - O(1) from cached value
    pub fn active_row_count(&self) -> u64 {
        self.active_count.load(std::sync::atomic::Ordering::Relaxed)
    }

    /// Add a new column to schema and storage with padding for existing rows
    pub fn add_column_with_padding(&self, name: &str, dtype: crate::data::DataType) -> io::Result<()> {
        use crate::data::DataType;
        
        let col_type = match dtype {
            DataType::Int64 | DataType::Int32 | DataType::Int16 | DataType::Int8 => ColumnType::Int64,
            DataType::Float64 | DataType::Float32 => ColumnType::Float64,
            DataType::String => ColumnType::String,
            DataType::Bool => ColumnType::Bool,
            DataType::Binary => ColumnType::Binary,
            _ => ColumnType::String,
        };
        
        let mut schema = self.schema.write();
        let mut columns = self.columns.write();
        let mut nulls = self.nulls.write();
        let ids = self.ids.read();
        let existing_row_count = ids.len();
        drop(ids);
        
        // Add to schema
        let idx = schema.add_column(name, col_type);
        
        // Ensure columns vector is large enough
        while columns.len() <= idx {
            let mut col = ColumnData::new(col_type);
            // Pad with defaults for existing rows
            match &mut col {
                ColumnData::Int64(v) => v.resize(existing_row_count, 0),
                ColumnData::Float64(v) => v.resize(existing_row_count, 0.0),
                ColumnData::String { offsets, .. } => {
                    for _ in 0..existing_row_count {
                        offsets.push(0);
                    }
                }
                ColumnData::Binary { offsets, .. } => {
                    for _ in 0..existing_row_count {
                        offsets.push(0);
                    }
                }
                ColumnData::Bool { len, .. } => {
                    *len = existing_row_count;
                }
                ColumnData::StringDict { indices, .. } => {
                    indices.resize(existing_row_count, 0);
                }
            }
            columns.push(col);
            nulls.push(Vec::new());
        }
        
        // Update header
        {
            let mut header = self.header.write();
            header.column_count = schema.column_count() as u32;
        }
        
        Ok(())
    }

    /// Replace a row by ID (delete old row, insert new with SAME ID)
    /// Returns true if successful
    pub fn replace(&self, id: u64, data: &HashMap<String, ColumnValue>) -> io::Result<bool> {
        // Check if ID exists
        if !self.exists(id) {
            return Ok(false);
        }
        
        // Delete the old row (soft delete)
        self.delete(id);
        
        // Convert data to typed columns for insert_typed
        let mut int_columns: HashMap<String, Vec<i64>> = HashMap::new();
        let mut float_columns: HashMap<String, Vec<f64>> = HashMap::new();
        let mut string_columns: HashMap<String, Vec<String>> = HashMap::new();
        let mut binary_columns: HashMap<String, Vec<Vec<u8>>> = HashMap::new();
        let mut bool_columns: HashMap<String, Vec<bool>> = HashMap::new();
        
        for (name, val) in data {
            match val {
                ColumnValue::Int64(v) => { int_columns.insert(name.clone(), vec![*v]); }
                ColumnValue::Float64(v) => { float_columns.insert(name.clone(), vec![*v]); }
                ColumnValue::String(v) => { string_columns.insert(name.clone(), vec![v.clone()]); }
                ColumnValue::Binary(v) => { binary_columns.insert(name.clone(), vec![v.clone()]); }
                ColumnValue::Bool(v) => { bool_columns.insert(name.clone(), vec![*v]); }
                ColumnValue::Null => {}
            }
        }
        
        // Use insert_typed but override the ID
        // First, determine row count (should be 1)
        let row_count = 1;
        
        // Instead of using next_id, we'll use the original ID
        let ids = vec![id];
        
        // Ensure schema has all columns and pad new columns with defaults
        {
            let mut schema = self.schema.write();
            let mut columns = self.columns.write();
            let mut nulls = self.nulls.write();
            let ids = self.ids.read();
            let existing_row_count = ids.len();
            drop(ids);
            
            for name in int_columns.keys() {
                let idx = schema.add_column(name, ColumnType::Int64);
                while columns.len() <= idx {
                    // New column - pad with defaults for existing rows
                    let mut col = ColumnData::new(ColumnType::Int64);
                    if let ColumnData::Int64(v) = &mut col {
                        v.resize(existing_row_count, 0);
                    }
                    columns.push(col);
                    nulls.push(Vec::new());
                }
            }
            for name in float_columns.keys() {
                let idx = schema.add_column(name, ColumnType::Float64);
                while columns.len() <= idx {
                    let mut col = ColumnData::new(ColumnType::Float64);
                    if let ColumnData::Float64(v) = &mut col {
                        v.resize(existing_row_count, 0.0);
                    }
                    columns.push(col);
                    nulls.push(Vec::new());
                }
            }
            for name in string_columns.keys() {
                let idx = schema.add_column(name, ColumnType::String);
                while columns.len() <= idx {
                    let mut col = ColumnData::new(ColumnType::String);
                    if let ColumnData::String { offsets, .. } = &mut col {
                        // For strings, push empty string offsets for existing rows
                        for _ in 0..existing_row_count {
                            offsets.push(0);
                        }
                    }
                    columns.push(col);
                    nulls.push(Vec::new());
                }
            }
            for name in binary_columns.keys() {
                let idx = schema.add_column(name, ColumnType::Binary);
                while columns.len() <= idx {
                    let mut col = ColumnData::new(ColumnType::Binary);
                    if let ColumnData::Binary { offsets, .. } = &mut col {
                        for _ in 0..existing_row_count {
                            offsets.push(0);
                        }
                    }
                    columns.push(col);
                    nulls.push(Vec::new());
                }
            }
            for name in bool_columns.keys() {
                let idx = schema.add_column(name, ColumnType::Bool);
                while columns.len() <= idx {
                    let mut col = ColumnData::new(ColumnType::Bool);
                    if let ColumnData::Bool { len, .. } = &mut col {
                        *len = existing_row_count;
                    }
                    columns.push(col);
                    nulls.push(Vec::new());
                }
            }
        }
        
        // Append ID
        self.ids.write().extend_from_slice(&ids);
        
        // Append column data
        {
            let schema = self.schema.read();
            let mut columns = self.columns.write();
            
            for (name, values) in int_columns {
                if let Some(idx) = schema.get_index(&name) {
                    columns[idx].extend_i64(&values);
                }
            }
            for (name, values) in float_columns {
                if let Some(idx) = schema.get_index(&name) {
                    columns[idx].extend_f64(&values);
                }
            }
            for (name, values) in string_columns {
                if let Some(idx) = schema.get_index(&name) {
                    for v in &values {
                        columns[idx].push_string(v);
                    }
                }
            }
            for (name, values) in binary_columns {
                if let Some(idx) = schema.get_index(&name) {
                    for v in &values {
                        columns[idx].push_bytes(v);
                    }
                }
            }
            for (name, values) in bool_columns {
                if let Some(idx) = schema.get_index(&name) {
                    for v in values {
                        columns[idx].push_bool(v);
                    }
                }
            }
        }
        
        // Update header
        {
            let mut header = self.header.write();
            header.row_count = self.ids.read().len() as u64;
            header.column_count = self.schema.read().column_count() as u32;
        }
        
        // Update id_to_idx mapping only if it's already built
        {
            let ids_guard = self.ids.read();
            let mut id_to_idx = self.id_to_idx.write();
            if let Some(map) = id_to_idx.as_mut() {
                let row_idx = ids_guard.len() - 1;
                map.insert(id, row_idx);
            }
        }
        
        // Extend deleted bitmap
        {
            let mut deleted = self.deleted.write();
            let new_len = (self.ids.read().len() + 7) / 8;
            deleted.resize(new_len, 0);
        }
        
        Ok(true)
    }

    // ========================================================================
    // Persistence
    // ========================================================================

    /// Check if a string column should use dictionary encoding
    /// Returns true if unique values < 20% of row count and row count > 1000
    fn should_dict_encode(col: &ColumnData) -> bool {
        if let ColumnData::String { offsets, data } = col {
            let row_count = offsets.len().saturating_sub(1);
            if row_count < 1000 {
                return false;
            }
            // Estimate unique values by sampling
            use ahash::AHashSet;
            let sample_size = (row_count / 10).min(1000);
            let mut unique: AHashSet<&[u8]> = AHashSet::with_capacity(sample_size);
            for i in 0..sample_size {
                let idx = i * 10; // Sample every 10th row
                if idx < row_count {
                    let start = offsets[idx] as usize;
                    let end = offsets[idx + 1] as usize;
                    unique.insert(&data[start..end]);
                }
            }
            // Use dictionary if cardinality < 20% of sampled rows
            unique.len() < sample_size / 5
        } else {
            false
        }
    }

    /// Save to file (full rewrite with V3 format)
    /// OPTIMIZATION: Automatically converts low-cardinality string columns to dictionary encoding
    pub fn save(&self) -> io::Result<()> {
        // CRITICAL: On Windows, mmap must be released BEFORE opening file for writing
        // Otherwise we get "os error 1224: user-mapped section open"
        self.mmap_cache.write().invalidate();
        *self.file.write() = None;  // Close file handle too
        
        // Also invalidate the global executor storage cache for this path
        // This releases any mmap held by cached backends used for queries
        crate::query::ApexExecutor::invalidate_cache_for_path(&self.path);
        
        let file = OpenOptions::new()
            .write(true)
            .create(true)
            .truncate(true)
            .open(&self.path)?;

        let mut writer = BufWriter::with_capacity(256 * 1024, file);

        let schema = self.schema.read();
        let ids = self.ids.read();
        let columns = self.columns.read();
        let nulls = self.nulls.read();
        let deleted = self.deleted.read();

        // Check if there are any deleted rows (optimization: skip filtering if none)
        let has_deleted = deleted.iter().any(|&b| b != 0);
        
        // Filter out deleted rows - get indices of non-deleted rows
        let (active_indices, active_ids): (Option<Vec<usize>>, Vec<u64>) = if has_deleted {
            let indices: Vec<usize> = (0..ids.len())
                .filter(|&i| {
                    let byte_idx = i / 8;
                    let bit_idx = i % 8;
                    byte_idx >= deleted.len() || (deleted[byte_idx] >> bit_idx) & 1 == 0
                })
                .collect();
            let filtered_ids: Vec<u64> = indices.iter().map(|&i| ids[i]).collect();
            (Some(indices), filtered_ids)
        } else {
            // No deleted rows - use all IDs directly (avoid copying)
            (None, ids.clone())
        };

        // Pre-compute filtered columns and detect which ones to dictionary-encode
        let filtered_columns: Vec<ColumnData> = if let Some(ref indices) = active_indices {
            columns.iter().enumerate().map(|(col_idx, col)| {
                if col_idx < columns.len() {
                    let filtered = col.filter_by_indices(indices);
                    if Self::should_dict_encode(&filtered) {
                        filtered.to_dict_encoded().unwrap_or(filtered)
                    } else {
                        filtered
                    }
                } else {
                    ColumnData::new(ColumnType::Int64)
                }
            }).collect()
        } else {
            columns.iter().map(|col| {
                if Self::should_dict_encode(col) {
                    col.to_dict_encoded().unwrap_or_else(|| col.clone())
                } else {
                    col.clone()
                }
            }).collect()
        };
        
        // Build modified schema with updated types for dictionary-encoded columns
        let mut modified_schema = OnDemandSchema::new();
        for (col_idx, (col_name, col_type)) in schema.columns.iter().enumerate() {
            let actual_type = if col_idx < filtered_columns.len() {
                match &filtered_columns[col_idx] {
                    ColumnData::StringDict { .. } => ColumnType::StringDict,
                    _ => *col_type,
                }
            } else {
                *col_type
            };
            modified_schema.add_column(col_name, actual_type);
        }

        // Serialize modified schema (with updated types)
        let schema_bytes = modified_schema.to_bytes();

        // Calculate offsets
        let schema_offset = HEADER_SIZE_V3 as u64;
        let column_index_offset = schema_offset + schema_bytes.len() as u64;
        let id_column_offset = column_index_offset + (schema.column_count() * COLUMN_INDEX_ENTRY_SIZE) as u64;

        // Build column index while calculating data offsets (using active row count)
        let mut current_offset = id_column_offset + (active_ids.len() * 8) as u64;
        let mut column_index_entries = Vec::with_capacity(modified_schema.column_count());

        for (col_idx, _col_def) in modified_schema.columns.iter().enumerate() {
            let expected_null_len = (active_ids.len() + 7) / 8;

            let col_data_bytes = filtered_columns[col_idx].to_bytes();

            let entry = ColumnIndexEntry {
                data_offset: current_offset + expected_null_len as u64,
                data_length: col_data_bytes.len() as u64,
                null_offset: current_offset,
                null_length: expected_null_len as u64,
            };

            column_index_entries.push(entry);
            current_offset += expected_null_len as u64 + col_data_bytes.len() as u64;
        }

        // Update header
        {
            let mut header = self.header.write();
            header.schema_offset = schema_offset;
            header.column_index_offset = column_index_offset;
            header.id_column_offset = id_column_offset;
            header.column_count = modified_schema.column_count() as u32;
            header.row_count = active_ids.len() as u64;
        }

        // Write header
        let header = self.header.read();
        writer.write_all(&header.to_bytes())?;

        // Write schema
        writer.write_all(&schema_bytes)?;

        // Write column index
        for entry in &column_index_entries {
            writer.write_all(&entry.to_bytes())?;
        }

        // Write active IDs only (excluding deleted)
        for &id in active_ids.iter() {
            writer.write_all(&id.to_le_bytes())?;
        }

        // Write column data (filtered by active rows)
        for (col_idx, _col_def) in modified_schema.columns.iter().enumerate() {
            // Build filtered null bitmap for active rows only
            let original_nulls = nulls.get(col_idx).map(|v| v.as_slice()).unwrap_or(&[]);
            let expected_len = (active_ids.len() + 7) / 8;
            let mut filtered_nulls = vec![0u8; expected_len];
            
            // Get indices to iterate over (either filtered or all)
            let indices_iter: Box<dyn Iterator<Item = (usize, usize)>> = match &active_indices {
                Some(indices) => Box::new(indices.iter().enumerate().map(|(new_idx, &old_idx)| (new_idx, old_idx))),
                None => Box::new((0..ids.len()).map(|i| (i, i))),
            };
            
            for (new_idx, old_idx) in indices_iter {
                let old_byte = old_idx / 8;
                let old_bit = old_idx % 8;
                let is_null = old_byte < original_nulls.len() && (original_nulls[old_byte] >> old_bit) & 1 == 1;
                if is_null {
                    let new_byte = new_idx / 8;
                    let new_bit = new_idx % 8;
                    filtered_nulls[new_byte] |= 1 << new_bit;
                }
            }
            writer.write_all(&filtered_nulls)?;

            // Write filtered column data for active rows only
            if col_idx < filtered_columns.len() {
                writer.write_all(&filtered_columns[col_idx].to_bytes())?;
            }
        }

        // Write footer
        writer.write_all(MAGIC_FOOTER_V3)?;
        let checksum = 0u32; // TODO: compute actual checksum
        writer.write_all(&checksum.to_le_bytes())?;
        let file_size = writer.stream_position()?;
        writer.write_all(&file_size.to_le_bytes())?;

        writer.flush()?;
        
        // fsync based on durability level
        // Max: sync on every save (strongest guarantee)
        // Safe: sync only when explicitly called via sync() or flush()
        // Fast: no sync
        if self.durability == super::DurabilityLevel::Max {
            let inner_file = writer.get_ref();
            inner_file.sync_all()?;
        }

        // Update column index in memory
        *self.column_index.write() = column_index_entries;

        // Invalidate mmap cache since file has changed
        self.mmap_cache.write().invalidate();

        // Reopen file for reading
        drop(writer);
        let file = File::open(&self.path)?;
        *self.file.write() = Some(file);

        Ok(())
    }
    
    /// Explicitly sync data to disk (fsync)
    /// 
    /// This ensures all buffered data is written to persistent storage.
    /// For safe/max durability modes, also syncs the WAL file.
    /// Called automatically for Safe/Max durability levels on save().
    /// For Fast durability, call this manually when you need durability guarantees.
    pub fn sync(&self) -> io::Result<()> {
        // Sync WAL first (for safe/max modes)
        if self.durability != super::DurabilityLevel::Fast {
            let mut wal_writer = self.wal_writer.write();
            if let Some(writer) = wal_writer.as_mut() {
                writer.sync()?;
            }
        }
        
        // Sync main data file
        // On Windows, sync_all() requires write access. Since save() already flushes
        // data via BufWriter and does fsync for Max durability, we need to open
        // the file with write access specifically for syncing.
        if self.path.exists() {
            // Open with write access for fsync (append mode to avoid truncation)
            let file = OpenOptions::new()
                .write(true)
                .append(true)
                .open(&self.path)?;
            file.sync_all()?;
        }
        Ok(())
    }
    
    /// Get the current durability level
    pub fn durability(&self) -> super::DurabilityLevel {
        self.durability
    }
    
    /// Set the durability level
    /// 
    /// Note: This only affects future operations. Existing buffered data
    /// is not automatically synced when changing to a higher durability level.
    pub fn set_durability(&mut self, level: super::DurabilityLevel) {
        self.durability = level;
    }
    
    /// Checkpoint: merge WAL records into main file and clear WAL
    /// 
    /// This is called automatically on save() for safe/max modes.
    /// After checkpoint, all data is in the main file and WAL is cleared.
    /// This improves read performance by eliminating WAL merge overhead.
    pub fn checkpoint(&self) -> io::Result<()> {
        if self.durability == super::DurabilityLevel::Fast {
            return Ok(()); // No WAL in fast mode
        }
        
        let wal_buffer = self.wal_buffer.read();
        if wal_buffer.is_empty() {
            return Ok(()); // Nothing to checkpoint
        }
        drop(wal_buffer);
        
        // Save main file (this persists all in-memory data including WAL records)
        self.save()?;
        
        // Clear WAL after successful save
        {
            let mut wal_buffer = self.wal_buffer.write();
            let mut wal_writer = self.wal_writer.write();
            
            wal_buffer.clear();
            
            // Create fresh WAL file
            if let Some(_) = wal_writer.take() {
                let wal_path = Self::wal_path(&self.path);
                *wal_writer = Some(super::incremental::WalWriter::create(
                    &wal_path, 
                    self.next_id.load(Ordering::SeqCst)
                )?);
            }
        }
        
        Ok(())
    }
    
    /// Get number of pending WAL records
    pub fn wal_record_count(&self) -> usize {
        self.wal_buffer.read().len()
    }
    
    /// Check if WAL needs checkpoint (has pending records)
    pub fn needs_checkpoint(&self) -> bool {
        !self.wal_buffer.read().is_empty()
    }

    // ========================================================================
    // Query APIs
    // ========================================================================

    /// Get row count
    pub fn row_count(&self) -> u64 {
        self.header.read().row_count
    }

    /// Get column names
    pub fn column_names(&self) -> Vec<String> {
        let mut names = vec!["_id".to_string()];
        names.extend(self.schema.read().columns.iter().map(|(name, _)| name.clone()));
        names
    }

    /// Get schema
    pub fn get_schema(&self) -> Vec<(String, ColumnType)> {
        self.schema.read().columns.clone()
    }

    // ========================================================================
    // Compatibility APIs (matching ColumnarStorage interface)
    // ========================================================================

    /// Insert rows using generic value type (compatibility with ColumnarStorage)
    /// Optimized with single-pass column collection
    /// 
    /// For safe/max durability modes, rows are written to WAL first for crash recovery.
    /// - Safe mode: WAL is flushed but fsync is deferred to flush() call
    /// - Max mode: WAL is fsync'd immediately after each insert for strongest guarantee
    pub fn insert_rows(&self, rows: &[HashMap<String, ColumnValue>]) -> io::Result<Vec<u64>> {
        if rows.is_empty() {
            return Ok(Vec::new());
        }
        
        // For safe/max durability with batch writes: use WAL for efficiency
        // Single-row writes skip WAL (original fsync-on-save behavior is faster)
        // WAL benefit: single I/O for many rows; WAL overhead: extra I/O for single rows
        let start_id = self.next_id.load(Ordering::SeqCst);
        let use_wal = self.durability != super::DurabilityLevel::Fast && rows.len() > 1;
        
        if use_wal {
            // Batch writes: use WAL for efficiency (single I/O for all rows)
            let mut wal_writer = self.wal_writer.write();
            
            if let Some(writer) = wal_writer.as_mut() {
                let record = super::incremental::WalRecord::BatchInsert { 
                    start_id, 
                    rows: rows.to_vec()
                };
                writer.append(&record)?;
                writer.flush()?;
                
                // For max durability: fsync WAL immediately
                if self.durability == super::DurabilityLevel::Max {
                    writer.sync()?;
                }
            }
        }
        // Note: For single-row writes, fsync happens in save() based on durability level
        
        // Handle case where all rows are empty dicts - still create rows with just _id
        let all_empty = rows.iter().all(|r| r.is_empty());
        if all_empty {
            let row_count = rows.len();
            let start_id = self.next_id.fetch_add(row_count as u64, Ordering::SeqCst);
            let ids: Vec<u64> = (start_id..start_id + row_count as u64).collect();
            
            // Add IDs
            self.ids.write().extend_from_slice(&ids);
            
            // Update header
            {
                let mut header = self.header.write();
                header.row_count = self.ids.read().len() as u64;
            }
            
            // Update id_to_idx mapping only if it's already built
            {
                let ids_guard = self.ids.read();
                let mut id_to_idx = self.id_to_idx.write();
                if let Some(map) = id_to_idx.as_mut() {
                    let start_idx = ids_guard.len() - ids.len();
                    for (i, &id) in ids.iter().enumerate() {
                        map.insert(id, start_idx + i);
                    }
                }
            }
            
            // Extend deleted bitmap
            {
                let mut deleted = self.deleted.write();
                let new_len = (self.ids.read().len() + 7) / 8;
                deleted.resize(new_len, 0);
            }
            
            // Update active count
            self.active_count.fetch_add(row_count as u64, Ordering::Relaxed);
            
            // Update pending rows counter and check auto-flush
            self.pending_rows.fetch_add(row_count as u64, Ordering::Relaxed);
            self.maybe_auto_flush()?;
            
            return Ok(ids);
        }

        // Single-pass optimized: determine column types from first non-empty row
        // and pre-allocate all vectors
        let num_rows = rows.len();
        let mut int_columns: HashMap<String, Vec<i64>> = HashMap::new();
        let mut float_columns: HashMap<String, Vec<f64>> = HashMap::new();
        let mut string_columns: HashMap<String, Vec<String>> = HashMap::new();
        let mut binary_columns: HashMap<String, Vec<Vec<u8>>> = HashMap::new();
        let mut bool_columns: HashMap<String, Vec<bool>> = HashMap::new();

        // Determine schema from first few rows (most data is homogeneous)
        let sample_size = std::cmp::min(10, num_rows);
        for row in rows.iter().take(sample_size) {
            for (key, val) in row {
                if int_columns.contains_key(key) || float_columns.contains_key(key) 
                    || string_columns.contains_key(key) || binary_columns.contains_key(key)
                    || bool_columns.contains_key(key) {
                    continue;
                }
                match val {
                    ColumnValue::Int64(_) => { int_columns.insert(key.clone(), Vec::with_capacity(num_rows)); }
                    ColumnValue::Float64(_) => { float_columns.insert(key.clone(), Vec::with_capacity(num_rows)); }
                    ColumnValue::String(_) => { string_columns.insert(key.clone(), Vec::with_capacity(num_rows)); }
                    ColumnValue::Binary(_) => { binary_columns.insert(key.clone(), Vec::with_capacity(num_rows)); }
                    ColumnValue::Bool(_) => { bool_columns.insert(key.clone(), Vec::with_capacity(num_rows)); }
                    ColumnValue::Null => { string_columns.insert(key.clone(), Vec::with_capacity(num_rows)); }
                }
            }
        }

        // Pre-allocate NULL string to avoid repeated allocation
        static NULL_MARKER: &str = "\x00__NULL__\x00";
        
        // Single pass: collect all values
        // Note: For homogeneous data (common case), new columns won't be discovered mid-stream
        for row in rows {
            // Handle new columns discovered mid-stream (rare case for heterogeneous data)
            for (key, val) in row {
                if !int_columns.contains_key(key) && !float_columns.contains_key(key) 
                    && !string_columns.contains_key(key) && !binary_columns.contains_key(key)
                    && !bool_columns.contains_key(key) {
                    match val {
                        ColumnValue::Int64(_) => { int_columns.insert(key.clone(), Vec::with_capacity(num_rows)); }
                        ColumnValue::Float64(_) => { float_columns.insert(key.clone(), Vec::with_capacity(num_rows)); }
                        ColumnValue::String(_) => { string_columns.insert(key.clone(), Vec::with_capacity(num_rows)); }
                        ColumnValue::Binary(_) => { binary_columns.insert(key.clone(), Vec::with_capacity(num_rows)); }
                        ColumnValue::Bool(_) => { bool_columns.insert(key.clone(), Vec::with_capacity(num_rows)); }
                        ColumnValue::Null => { string_columns.insert(key.clone(), Vec::with_capacity(num_rows)); }
                    }
                }
            }
            
            // Collect values for all columns
            for (key, col) in int_columns.iter_mut() {
                col.push(match row.get(key) {
                    Some(ColumnValue::Int64(v)) => *v,
                    _ => 0,
                });
            }
            for (key, col) in float_columns.iter_mut() {
                col.push(match row.get(key) {
                    Some(ColumnValue::Float64(v)) => *v,
                    _ => 0.0,
                });
            }
            for (key, col) in string_columns.iter_mut() {
                col.push(match row.get(key) {
                    Some(ColumnValue::String(v)) => v.clone(),
                    Some(ColumnValue::Null) => NULL_MARKER.to_string(),
                    _ => String::new(),
                });
            }
            for (key, col) in binary_columns.iter_mut() {
                col.push(match row.get(key) {
                    Some(ColumnValue::Binary(v)) => v.clone(),
                    _ => Vec::new(),
                });
            }
            for (key, col) in bool_columns.iter_mut() {
                col.push(match row.get(key) {
                    Some(ColumnValue::Bool(v)) => *v,
                    _ => false,
                });
            }
        }

        let result = self.insert_typed(int_columns, float_columns, string_columns, binary_columns, bool_columns)?;
        
        // Update pending rows counter and check auto-flush
        self.pending_rows.fetch_add(result.len() as u64, Ordering::Relaxed);
        self.maybe_auto_flush()?;
        
        Ok(result)
    }

    /// Insert typed columns and immediately persist to disk
    /// 
    /// This is the preferred method for V3 direct writes - data is immediately
    /// visible to V3Executor after this call returns.
    pub fn insert_typed_and_persist(
        &self,
        int_columns: HashMap<String, Vec<i64>>,
        float_columns: HashMap<String, Vec<f64>>,
        string_columns: HashMap<String, Vec<String>>,
        bool_columns: HashMap<String, Vec<bool>>,
    ) -> io::Result<Vec<u64>> {
        let ids = self.insert_typed(int_columns, float_columns, string_columns, HashMap::new(), bool_columns)?;
        if !ids.is_empty() {
            self.save()?;
        }
        Ok(ids)
    }

    /// Append delta (for compatibility - just calls insert_rows + save)
    pub fn append_delta(&self, rows: &[HashMap<String, ColumnValue>]) -> io::Result<Vec<u64>> {
        let ids = self.insert_rows(rows)?;
        self.save()?;
        Ok(ids)
    }

    /// Fast delta append (same as append_delta for this format)
    pub fn append_delta_fast(&self, rows: &[HashMap<String, ColumnValue>]) -> io::Result<Vec<u64>> {
        self.append_delta(rows)
    }

    /// Compact storage (no-op for V3 format - already compact)
    pub fn compact(&self) -> io::Result<()> {
        self.save()
    }

    /// Check if compaction is needed (always false for V3)
    pub fn needs_compaction(&self) -> bool {
        false
    }

    /// Flush changes to disk
    pub fn flush(&self) -> io::Result<()> {
        self.save()
    }

    /// Close storage and release all resources
    /// IMPORTANT: On Windows, mmap must be released before temp directory cleanup
    pub fn close(&self) -> io::Result<()> {
        // Save any pending changes first
        self.save()?;
        
        // Release mmap cache BEFORE closing file (critical for Windows)
        self.mmap_cache.write().invalidate();
        
        // Close file handle
        *self.file.write() = None;
        
        Ok(())
    }
    
    /// Release mmap without saving (for cleanup scenarios)
    pub fn release_mmap(&self) {
        self.mmap_cache.write().invalidate();
    }
}

/// Drop implementation to ensure mmap is released before file handle
/// This is critical for Windows where mmap must be unmapped before file deletion
impl Drop for OnDemandStorage {
    fn drop(&mut self) {
        // Release mmap first (critical for Windows)
        // parking_lot's try_write returns Option, not Result
        if let Some(mut cache) = self.mmap_cache.try_write() {
            cache.invalidate();
        }
        // File handle will be dropped automatically after mmap is released
    }
}


// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn test_v3_create_and_open() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test.apex");

        // Create and insert
        {
            let storage = OnDemandStorage::create(&path).unwrap();

            let mut int_cols = HashMap::new();
            int_cols.insert("age".to_string(), vec![25, 30, 35, 40, 45]);

            let mut string_cols = HashMap::new();
            string_cols.insert("name".to_string(), vec![
                "Alice".to_string(),
                "Bob".to_string(), 
                "Charlie".to_string(),
                "David".to_string(),
                "Eve".to_string(),
            ]);

            let ids = storage.insert_typed(
                int_cols,
                HashMap::new(),
                string_cols,
                HashMap::new(),
                HashMap::new(),
            ).unwrap();

            assert_eq!(ids.len(), 5);
            storage.save().unwrap();
        }

        // Reopen and verify
        {
            let storage = OnDemandStorage::open(&path).unwrap();
            assert_eq!(storage.row_count(), 5);
            assert_eq!(storage.column_names().len(), 3); // _id, age, name
        }
    }

    #[test]
    fn test_v3_column_projection() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test_proj.apex");

        // Create with multiple columns
        let storage = OnDemandStorage::create(&path).unwrap();

        let mut int_cols = HashMap::new();
        int_cols.insert("a".to_string(), vec![1, 2, 3, 4, 5]);
        int_cols.insert("b".to_string(), vec![10, 20, 30, 40, 50]);
        int_cols.insert("c".to_string(), vec![100, 200, 300, 400, 500]);

        storage.insert_typed(int_cols, HashMap::new(), HashMap::new(), HashMap::new(), HashMap::new()).unwrap();
        storage.save().unwrap();

        // Reopen
        let storage = OnDemandStorage::open(&path).unwrap();

        // Read only column "b"
        let result = storage.read_columns(Some(&["b"]), 0, None).unwrap();
        assert_eq!(result.len(), 1);
        assert!(result.contains_key("b"));

        if let ColumnData::Int64(vals) = &result["b"] {
            assert_eq!(vals, &[10, 20, 30, 40, 50]);
        } else {
            panic!("Expected Int64 column");
        }
    }

    #[test]
    fn test_v3_row_range() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test_range.apex");

        let storage = OnDemandStorage::create(&path).unwrap();

        let mut int_cols = HashMap::new();
        int_cols.insert("val".to_string(), (0..100).collect());

        storage.insert_typed(int_cols, HashMap::new(), HashMap::new(), HashMap::new(), HashMap::new()).unwrap();
        storage.save().unwrap();

        let storage = OnDemandStorage::open(&path).unwrap();

        // Read rows 10-19 (10 rows starting at row 10)
        let result = storage.read_columns(Some(&["val"]), 10, Some(10)).unwrap();

        if let ColumnData::Int64(vals) = &result["val"] {
            assert_eq!(vals.len(), 10);
            assert_eq!(vals[0], 10);
            assert_eq!(vals[9], 19);
        } else {
            panic!("Expected Int64 column");
        }
    }

    #[test]
    fn test_v3_string_column() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test_string.apex");

        let storage = OnDemandStorage::create(&path).unwrap();

        let mut string_cols = HashMap::new();
        string_cols.insert("text".to_string(), vec![
            "hello".to_string(),
            "world".to_string(),
            "foo".to_string(),
            "bar".to_string(),
        ]);

        storage.insert_typed(HashMap::new(), HashMap::new(), string_cols, HashMap::new(), HashMap::new()).unwrap();
        storage.save().unwrap();

        let storage = OnDemandStorage::open(&path).unwrap();

        // Read middle 2 rows
        let result = storage.read_columns(Some(&["text"]), 1, Some(2)).unwrap();

        if let ColumnData::String { offsets, data } = &result["text"] {
            assert_eq!(offsets.len(), 3); // 2 strings + 1 trailing offset
            let s0 = std::str::from_utf8(&data[offsets[0] as usize..offsets[1] as usize]).unwrap();
            let s1 = std::str::from_utf8(&data[offsets[1] as usize..offsets[2] as usize]).unwrap();
            assert_eq!(s0, "world");
            assert_eq!(s1, "foo");
        } else {
            panic!("Expected String column");
        }
    }

    #[test]
    fn benchmark_on_demand_vs_full_load() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("bench_demand.apex");

        // Create large dataset
        let n = 100_000;
        let storage = OnDemandStorage::create(&path).unwrap();

        let mut int_cols = HashMap::new();
        int_cols.insert("col_a".to_string(), (0..n as i64).collect());
        int_cols.insert("col_b".to_string(), (0..n as i64).map(|x| x * 2).collect());
        int_cols.insert("col_c".to_string(), (0..n as i64).map(|x| x * 3).collect());

        storage.insert_typed(int_cols, HashMap::new(), HashMap::new(), HashMap::new(), HashMap::new()).unwrap();
        storage.save().unwrap();

        // Benchmark: read single column, small range
        let storage = OnDemandStorage::open(&path).unwrap();

        let start = std::time::Instant::now();
        for _ in 0..100 {
            let _ = storage.read_columns(Some(&["col_b"]), 50000, Some(100)).unwrap();
        }
        let elapsed = start.elapsed();

        println!("\n=== On-Demand Read Benchmark ===");
        println!("Dataset: {} rows x 3 columns", n);
        println!("Query: 100 rows from middle of col_b");
        println!("100 iterations: {:?}", elapsed);
        println!("Per query: {:?}", elapsed / 100);
        println!("=================================\n");

        // Should be very fast since we only read 100 rows of 1 column
        assert!(elapsed.as_millis() < 100, "On-demand reads should be fast");
    }

    #[test]
    fn test_insert_rows_compatibility() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test_compat.apex");

        let storage = OnDemandStorage::create(&path).unwrap();

        // Use insert_rows API (ColumnarStorage compatible)
        let mut rows = Vec::new();
        for i in 0..10 {
            let mut row = HashMap::new();
            row.insert("id".to_string(), ColumnValue::Int64(i));
            row.insert("name".to_string(), ColumnValue::String(format!("user_{}", i)));
            row.insert("score".to_string(), ColumnValue::Float64(i as f64 * 1.5));
            rows.push(row);
        }

        let ids = storage.insert_rows(&rows).unwrap();
        assert_eq!(ids.len(), 10);

        storage.save().unwrap();

        // Reopen and verify
        let storage = OnDemandStorage::open(&path).unwrap();
        assert_eq!(storage.row_count(), 10);

        let result = storage.read_columns(Some(&["id", "score"]), 0, None).unwrap();
        assert_eq!(result.len(), 2);

        if let ColumnData::Int64(vals) = &result["id"] {
            assert_eq!(vals, &[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]);
        }
    }

    #[test]
    fn test_append_delta_compatibility() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test_delta.apex");

        let storage = OnDemandStorage::create(&path).unwrap();

        // First batch
        let mut rows = Vec::new();
        for i in 0..5 {
            let mut row = HashMap::new();
            row.insert("val".to_string(), ColumnValue::Int64(i));
            rows.push(row);
        }
        storage.append_delta(&rows).unwrap();

        // Second batch
        let mut rows2 = Vec::new();
        for i in 5..10 {
            let mut row = HashMap::new();
            row.insert("val".to_string(), ColumnValue::Int64(i));
            rows2.push(row);
        }
        storage.append_delta(&rows2).unwrap();

        // Verify
        let storage = OnDemandStorage::open(&path).unwrap();
        assert_eq!(storage.row_count(), 10);

        let result = storage.read_columns(Some(&["val"]), 0, None).unwrap();
        if let ColumnData::Int64(vals) = &result["val"] {
            assert_eq!(vals, &[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]);
        }
    }
}
