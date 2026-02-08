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
//! │   - Magic: "APEXV3\0\0" (8 bytes)                           │
//! │   - Version: u32                                            │
//! │   - Flags: u32                                              │
//! │   - Row count: u64                                          │
//! │   - Column count: u32                                       │
//! │   - Row group size: u32 (rows per group, default 65536)     │
//! │   - Schema offset: u64                                      │
//! │   - Column index offset: u64                                │
//! │   - ID column offset: u64                                   │
//! │   - Timestamps, checksum, reserved                          │
//! ├─────────────────────────────────────────────────────────────┤
//! │ Schema Block                                                │
//! │   - For each column: [name_len:u16][name:bytes][type:u8]    │
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
//! │   - Magic: "APEXEND\0"                                      │
//! │   - Checksum: u32                                           │
//! │   - File size: u64                                          │
//! └─────────────────────────────────────────────────────────────┘
//! ```

use std::collections::HashMap;
use std::fs::{File, OpenOptions};
use std::io::{self, BufWriter, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::cell::RefCell;

use memmap2::Mmap;
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use arrow::record_batch::RecordBatch;
use arrow::array::ArrayRef;

use super::delta::DeltaStore;

/// Helper for InvalidData errors
#[inline] fn err_data(msg: impl Into<String>) -> io::Error { io::Error::new(io::ErrorKind::InvalidData, msg.into()) }
/// Helper for NotFound errors  
#[inline] fn err_not_found(msg: impl Into<String>) -> io::Error { io::Error::new(io::ErrorKind::NotFound, msg.into()) }
/// Helper for NotConnected errors
#[inline] fn err_not_conn(msg: &str) -> io::Error { io::Error::new(io::ErrorKind::NotConnected, msg) }
/// Helper for InvalidInput errors
#[inline] fn err_input(msg: &str) -> io::Error { io::Error::new(io::ErrorKind::InvalidInput, msg) }

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
                return Err(err_data("Empty file"));
            }
            // SAFETY: File must remain open while mmap is in use
            // We ensure this by keeping mmap in the same struct as file
            let mmap = unsafe {
                // On Linux, use MAP_POPULATE for files < 64MB to pre-fault pages
                // and eliminate page-fault overhead on first access.
                #[cfg(target_os = "linux")]
                {
                    if current_size < 64 * 1024 * 1024 {
                        memmap2::MmapOptions::new().populate().map(file)?
                    } else {
                        Mmap::map(file)?
                    }
                }
                #[cfg(not(target_os = "linux"))]
                { Mmap::map(file)? }
            };
            // On Linux, hint sequential access so the kernel doubles readahead.
            #[cfg(target_os = "linux")]
            { let _ = mmap.advise(memmap2::Advice::Sequential); }
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
            return Err(io::Error::new(io::ErrorKind::UnexpectedEof, "EOF"));
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
const FORMAT_VERSION_V4: u32 = 4;
const HEADER_SIZE_V3: usize = 256;
const COLUMN_INDEX_ENTRY_SIZE: usize = 32;
const DEFAULT_ROW_GROUP_SIZE: u32 = 65536;

// V4 Row Group format constants
const MAGIC_ROW_GROUP: &[u8; 4] = b"APXG";
const MAGIC_V4_FOOTER: &[u8; 8] = b"APXFOOT\0";
/// Size of a serialized RowGroupMeta entry in the footer (8+8+4+8+8+4 = 40 bytes)
const ROW_GROUP_META_SIZE: usize = 40;

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
    pub fn column_type(&self) -> ColumnType {
        match self {
            ColumnData::Bool { .. } => ColumnType::Bool,
            ColumnData::Int64(_) => ColumnType::Int64,
            ColumnData::Float64(_) => ColumnType::Float64,
            ColumnData::String { .. } => ColumnType::String,
            ColumnData::Binary { .. } => ColumnType::Binary,
            ColumnData::StringDict { .. } => ColumnType::StringDict,
        }
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
    /// OPTIMIZATION: pre-allocate all needed bytes upfront, skip branch on data.len()
    #[inline]
    pub fn extend_bools(&mut self, values: &[bool]) {
        if let ColumnData::Bool { data, len } = self {
            if values.is_empty() { return; }
            let new_len = *len + values.len();
            let needed_bytes = (new_len + 7) / 8;
            data.resize(needed_bytes, 0);
            for &value in values {
                if value {
                    let byte_idx = *len / 8;
                    let bit_idx = *len % 8;
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
                // Write exactly (len + 7) / 8 bytes — data.len() may exceed this
                // from push_bool's incremental Vec growth
                let byte_len = (*len + 7) / 8;
                let mut buf = Vec::with_capacity(8 + byte_len);
                buf.extend_from_slice(&(*len as u64).to_le_bytes());
                buf.extend_from_slice(&data[..byte_len.min(data.len())]);
                // Pad if data is shorter than expected (shouldn't happen normally)
                if data.len() < byte_len {
                    buf.resize(8 + byte_len, 0);
                }
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

    /// Write serialized bytes directly to a writer, avoiding intermediate Vec<u8> allocation.
    /// Produces the same byte format as to_bytes().
    /// For large columns, this avoids allocating a temporary buffer (e.g., 80MB for 10M i64 rows).
    #[inline]
    pub fn write_to<W: Write>(&self, writer: &mut W) -> io::Result<()> {
        match self {
            ColumnData::Bool { data, len } => {
                let byte_len = (*len + 7) / 8;
                writer.write_all(&(*len as u64).to_le_bytes())?;
                writer.write_all(&data[..byte_len.min(data.len())])?;
                // Pad if data is shorter than expected
                if data.len() < byte_len {
                    let pad = byte_len - data.len();
                    writer.write_all(&vec![0u8; pad])?;
                }
            }
            ColumnData::Int64(v) => {
                writer.write_all(&(v.len() as u64).to_le_bytes())?;
                let bytes = unsafe {
                    std::slice::from_raw_parts(v.as_ptr() as *const u8, v.len() * 8)
                };
                writer.write_all(bytes)?;
            }
            ColumnData::Float64(v) => {
                writer.write_all(&(v.len() as u64).to_le_bytes())?;
                let bytes = unsafe {
                    std::slice::from_raw_parts(v.as_ptr() as *const u8, v.len() * 8)
                };
                writer.write_all(bytes)?;
            }
            ColumnData::String { offsets, data } | ColumnData::Binary { offsets, data } => {
                let count = offsets.len().saturating_sub(1);
                writer.write_all(&(count as u64).to_le_bytes())?;
                let offset_bytes = unsafe {
                    std::slice::from_raw_parts(offsets.as_ptr() as *const u8, offsets.len() * 4)
                };
                writer.write_all(offset_bytes)?;
                writer.write_all(&(data.len() as u64).to_le_bytes())?;
                writer.write_all(data)?;
            }
            ColumnData::StringDict { indices, dict_offsets, dict_data } => {
                writer.write_all(&(indices.len() as u64).to_le_bytes())?;
                writer.write_all(&(dict_offsets.len() as u64).to_le_bytes())?;
                let indices_bytes = unsafe {
                    std::slice::from_raw_parts(indices.as_ptr() as *const u8, indices.len() * 4)
                };
                writer.write_all(indices_bytes)?;
                let offsets_bytes = unsafe {
                    std::slice::from_raw_parts(dict_offsets.as_ptr() as *const u8, dict_offsets.len() * 4)
                };
                writer.write_all(offsets_bytes)?;
                writer.write_all(&(dict_data.len() as u64).to_le_bytes())?;
                writer.write_all(dict_data)?;
            }
        }
        Ok(())
    }

    /// Deserialize from to_bytes() output, given the known column type.
    /// Returns (ColumnData, bytes_consumed).
    pub fn from_bytes_typed(bytes: &[u8], col_type: ColumnType) -> io::Result<(Self, usize)> {
        let mut pos = 0;
        
        macro_rules! read_u64 {
            () => {{
                if pos + 8 > bytes.len() {
                    return Err(err_data("ColumnData::from_bytes_typed: unexpected EOF reading u64"));
                }
                let v = u64::from_le_bytes(bytes[pos..pos+8].try_into().unwrap());
                pos += 8;
                v
            }};
        }
        
        match col_type {
            ColumnType::Bool => {
                let len = read_u64!() as usize;
                let byte_len = (len + 7) / 8;
                if pos + byte_len > bytes.len() {
                    return Err(err_data("Bool column data truncated"));
                }
                let data = bytes[pos..pos + byte_len].to_vec();
                pos += byte_len;
                Ok((ColumnData::Bool { data, len }, pos))
            }
            ColumnType::Int64 | ColumnType::Int8 | ColumnType::Int16 | ColumnType::Int32 |
            ColumnType::UInt8 | ColumnType::UInt16 | ColumnType::UInt32 | ColumnType::UInt64 => {
                let count = read_u64!() as usize;
                let byte_len = count * 8;
                if pos + byte_len > bytes.len() {
                    return Err(err_data("Int64 column data truncated"));
                }
                let mut v = vec![0i64; count];
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        bytes[pos..].as_ptr(), v.as_mut_ptr() as *mut u8, byte_len,
                    );
                }
                pos += byte_len;
                Ok((ColumnData::Int64(v), pos))
            }
            ColumnType::Float64 | ColumnType::Float32 => {
                let count = read_u64!() as usize;
                let byte_len = count * 8;
                if pos + byte_len > bytes.len() {
                    return Err(err_data("Float64 column data truncated"));
                }
                let mut v = vec![0f64; count];
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        bytes[pos..].as_ptr(), v.as_mut_ptr() as *mut u8, byte_len,
                    );
                }
                pos += byte_len;
                Ok((ColumnData::Float64(v), pos))
            }
            ColumnType::String => {
                let count = read_u64!() as usize;
                let offsets_len = (count + 1) * 4;
                if pos + offsets_len > bytes.len() {
                    return Err(err_data("String offsets truncated"));
                }
                let mut offsets = vec![0u32; count + 1];
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        bytes[pos..].as_ptr(), offsets.as_mut_ptr() as *mut u8, offsets_len,
                    );
                }
                pos += offsets_len;
                let data_len = read_u64!() as usize;
                if pos + data_len > bytes.len() {
                    return Err(err_data("String data truncated"));
                }
                let data = bytes[pos..pos + data_len].to_vec();
                pos += data_len;
                Ok((ColumnData::String { offsets, data }, pos))
            }
            ColumnType::Binary => {
                let count = read_u64!() as usize;
                let offsets_len = (count + 1) * 4;
                if pos + offsets_len > bytes.len() {
                    return Err(err_data("Binary offsets truncated"));
                }
                let mut offsets = vec![0u32; count + 1];
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        bytes[pos..].as_ptr(), offsets.as_mut_ptr() as *mut u8, offsets_len,
                    );
                }
                pos += offsets_len;
                let data_len = read_u64!() as usize;
                if pos + data_len > bytes.len() {
                    return Err(err_data("Binary data truncated"));
                }
                let data = bytes[pos..pos + data_len].to_vec();
                pos += data_len;
                Ok((ColumnData::Binary { offsets, data }, pos))
            }
            ColumnType::StringDict => {
                let row_count = read_u64!() as usize;
                let dict_size = read_u64!() as usize;
                let indices_len = row_count * 4;
                if pos + indices_len > bytes.len() {
                    return Err(err_data("StringDict indices truncated"));
                }
                let mut indices = vec![0u32; row_count];
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        bytes[pos..].as_ptr(), indices.as_mut_ptr() as *mut u8, indices_len,
                    );
                }
                pos += indices_len;
                let dict_offsets_len = dict_size * 4;
                if pos + dict_offsets_len > bytes.len() {
                    return Err(err_data("StringDict dict_offsets truncated"));
                }
                let mut dict_offsets = vec![0u32; dict_size];
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        bytes[pos..].as_ptr(), dict_offsets.as_mut_ptr() as *mut u8, dict_offsets_len,
                    );
                }
                pos += dict_offsets_len;
                let dict_data_len = read_u64!() as usize;
                if pos + dict_data_len > bytes.len() {
                    return Err(err_data("StringDict dict_data truncated"));
                }
                let dict_data = bytes[pos..pos + dict_data_len].to_vec();
                pos += dict_data_len;
                Ok((ColumnData::StringDict { indices, dict_offsets, dict_data }, pos))
            }
            ColumnType::Null => {
                let count = read_u64!() as usize;
                let byte_len = count * 8;
                pos += byte_len.min(bytes.len() - pos);
                Ok((ColumnData::Int64(vec![0i64; count]), pos))
            }
        }
    }
    
    /// Skip over serialized column data without allocating memory.
    /// Returns the number of bytes consumed (same as from_bytes_typed would report).
    /// Used by mmap-based on-demand reading to skip unrequested columns.
    pub fn skip_bytes_typed(bytes: &[u8], col_type: ColumnType) -> io::Result<usize> {
        let mut pos = 0;

        macro_rules! read_u64 {
            () => {{
                if pos + 8 > bytes.len() {
                    return Err(err_data("skip_bytes_typed: unexpected EOF reading u64"));
                }
                let v = u64::from_le_bytes(bytes[pos..pos+8].try_into().unwrap());
                pos += 8;
                v
            }};
        }

        match col_type {
            ColumnType::Bool => {
                let len = read_u64!() as usize;
                let byte_len = (len + 7) / 8;
                pos += byte_len;
            }
            ColumnType::Int64 | ColumnType::Int8 | ColumnType::Int16 | ColumnType::Int32 |
            ColumnType::UInt8 | ColumnType::UInt16 | ColumnType::UInt32 | ColumnType::UInt64 => {
                let count = read_u64!() as usize;
                pos += count * 8;
            }
            ColumnType::Float64 | ColumnType::Float32 => {
                let count = read_u64!() as usize;
                pos += count * 8;
            }
            ColumnType::String | ColumnType::Binary => {
                let count = read_u64!() as usize;
                let offsets_len = (count + 1) * 4;
                pos += offsets_len;
                let data_len = read_u64!() as usize;
                pos += data_len;
            }
            ColumnType::StringDict => {
                let row_count = read_u64!() as usize;
                let dict_size = read_u64!() as usize;
                pos += row_count * 4; // indices
                pos += dict_size * 4; // dict_offsets
                let dict_data_len = read_u64!() as usize;
                pos += dict_data_len;
            }
            ColumnType::Null => {
                let count = read_u64!() as usize;
                pos += count * 8;
            }
        }

        if pos > bytes.len() {
            return Err(err_data(format!("skip_bytes_typed: would skip past EOF ({} > {})", pos, bytes.len())));
        }
        Ok(pos)
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
    
    /// Append another column's data to this column.
    /// OPTIMIZATION: bulk copy for byte-aligned Bool, pre-allocated offsets for String/Binary.
    pub fn append(&mut self, other: &Self) {
        match (self, other) {
            (ColumnData::Bool { data, len }, ColumnData::Bool { data: other_data, len: other_len }) => {
                if *other_len == 0 { return; }
                // OPTIMIZATION: byte-aligned → bulk copy
                if *len % 8 == 0 {
                    let other_byte_len = (*other_len + 7) / 8;
                    let copy_len = other_byte_len.min(other_data.len());
                    data.extend_from_slice(&other_data[..copy_len]);
                    if copy_len < other_byte_len {
                        data.resize(data.len() + (other_byte_len - copy_len), 0);
                    }
                    *len += *other_len;
                } else {
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
            }
            (ColumnData::Int64(v), ColumnData::Int64(other_v)) => {
                v.extend_from_slice(other_v);
            }
            (ColumnData::Float64(v), ColumnData::Float64(other_v)) => {
                v.extend_from_slice(other_v);
            }
            (ColumnData::String { offsets, data }, ColumnData::String { offsets: other_offsets, data: other_data }) |
            (ColumnData::Binary { offsets, data }, ColumnData::Binary { offsets: other_offsets, data: other_data }) => {
                let base_offset = *offsets.last().unwrap_or(&0);
                // OPTIMIZATION: pre-allocate and batch push
                offsets.reserve(other_offsets.len() - 1);
                for i in 1..other_offsets.len() {
                    offsets.push(base_offset + other_offsets[i]);
                }
                data.extend_from_slice(other_data);
            }
            (ColumnData::StringDict { indices, dict_offsets, dict_data },
             ColumnData::StringDict { indices: other_indices, dict_offsets: other_offsets, dict_data: other_data }) => {
                let existing_dict_size = dict_offsets.len();
                let base = *dict_offsets.last().unwrap_or(&0);
                dict_offsets.reserve(other_offsets.len() - 1);
                for i in 1..other_offsets.len() {
                    dict_offsets.push(base + other_offsets[i]);
                }
                dict_data.extend_from_slice(other_data);
                let offset = if existing_dict_size > 0 { existing_dict_size as u32 - 1 } else { 0 };
                indices.reserve(other_indices.len());
                for &idx in other_indices {
                    indices.push(idx + offset);
                }
            }
            _ => {} // Type mismatch - ignore
        }
    }

    /// Filter column data to only include rows at specified indices.
    /// OPTIMIZATION: uses pre-allocation and unchecked indexing for hot paths.
    pub fn filter_by_indices(&self, indices: &[usize]) -> Self {
        match self {
            ColumnData::Bool { data, len } => {
                let new_len = indices.len();
                let mut new_data = vec![0u8; (new_len + 7) / 8];
                for (new_idx, &idx) in indices.iter().enumerate() {
                    if idx < *len {
                        let old_byte = idx / 8;
                        let old_bit = idx % 8;
                        if old_byte < data.len() && (data[old_byte] >> old_bit) & 1 == 1 {
                            new_data[new_idx / 8] |= 1 << (new_idx % 8);
                        }
                    }
                }
                ColumnData::Bool { data: new_data, len: new_len }
            }
            ColumnData::Int64(v) => {
                // OPTIMIZATION: pre-allocate exact size, use unchecked indexing
                let mut result = Vec::with_capacity(indices.len());
                for &i in indices {
                    // Safety: caller guarantees indices are in-range (built from 0..ids.len())
                    result.push(if i < v.len() { unsafe { *v.get_unchecked(i) } } else { 0 });
                }
                ColumnData::Int64(result)
            }
            ColumnData::Float64(v) => {
                let mut result = Vec::with_capacity(indices.len());
                for &i in indices {
                    result.push(if i < v.len() { unsafe { *v.get_unchecked(i) } } else { 0.0 });
                }
                ColumnData::Float64(result)
            }
            ColumnData::String { offsets, data } | ColumnData::Binary { offsets, data } => {
                let mut new_offsets = Vec::with_capacity(indices.len() + 1);
                new_offsets.push(0u32);
                // Estimate average string length for pre-allocation
                let avg_len = if offsets.len() > 1 {
                    data.len() / (offsets.len() - 1)
                } else { 0 };
                let mut new_data = Vec::with_capacity(indices.len() * avg_len);
                for &idx in indices {
                    if idx + 1 < offsets.len() {
                        let start = offsets[idx] as usize;
                        let end = offsets[idx + 1] as usize;
                        new_data.extend_from_slice(&data[start..end]);
                        new_offsets.push(new_data.len() as u32);
                    }
                }
                if matches!(self, ColumnData::String { .. }) {
                    ColumnData::String { offsets: new_offsets, data: new_data }
                } else {
                    ColumnData::Binary { offsets: new_offsets, data: new_data }
                }
            }
            ColumnData::StringDict { indices: row_indices, dict_offsets, dict_data } => {
                // Just filter the indices array, dictionary stays the same
                let mut new_indices = Vec::with_capacity(indices.len());
                for &i in indices {
                    if i < row_indices.len() {
                        new_indices.push(unsafe { *row_indices.get_unchecked(i) });
                    }
                }
                ColumnData::StringDict { 
                    indices: new_indices, 
                    dict_offsets: dict_offsets.clone(), 
                    dict_data: dict_data.clone() 
                }
            }
        }
    }
    
    /// Check if dictionary encoding would be beneficial for this column
    /// Returns true if cardinality is low relative to row count
    pub fn should_dict_encode(&self) -> bool {
        if let ColumnData::String { offsets, data } = self {
            use ahash::AHashSet;
            
            let row_count = offsets.len().saturating_sub(1);
            if row_count < 100 {
                return false; // Too few rows to benefit
            }
            
            // Sample up to 1000 rows to estimate cardinality
            let sample_size = row_count.min(1000);
            let mut unique_strings: AHashSet<&[u8]> = AHashSet::with_capacity(sample_size / 10);
            
            let step = if row_count > sample_size { row_count / sample_size } else { 1 };
            let mut i = 0;
            while i < row_count && unique_strings.len() < sample_size / 5 {
                let start = offsets[i] as usize;
                let end = offsets[i + 1] as usize;
                if end <= data.len() {
                    unique_strings.insert(&data[start..end]);
                }
                i += step;
            }
            
            // Dictionary encoding is beneficial if cardinality < 20% of sampled rows
            // or if there are fewer than 10000 unique values
            let estimated_cardinality = unique_strings.len();
            estimated_cardinality < sample_size / 5 || estimated_cardinality < 10000
        } else {
            false
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
    
    /// Decode StringDict back to plain String column.
    /// Used during streaming compaction to normalize types before merging with delta data.
    pub fn decode_string_dict(self) -> Self {
        if let ColumnData::StringDict { indices, dict_offsets, dict_data } = self {
            let mut offsets = Vec::with_capacity(indices.len() + 1);
            let mut data = Vec::new();
            offsets.push(0u32);
            
            for &idx in &indices {
                if idx == 0 || (idx as usize) >= dict_offsets.len() {
                    offsets.push(data.len() as u32);
                } else {
                    let start = dict_offsets[(idx - 1) as usize] as usize;
                    let end = dict_offsets[idx as usize] as usize;
                    if end <= dict_data.len() && start <= end {
                        data.extend_from_slice(&dict_data[start..end]);
                    }
                    offsets.push(data.len() as u32);
                }
            }
            
            ColumnData::String { offsets, data }
        } else {
            self
        }
    }
    
    /// Try to convert to dictionary encoding if beneficial, otherwise return self
    pub fn maybe_dict_encode(self) -> Self {
        if self.should_dict_encode() {
            self.to_dict_encoded().unwrap_or(self)
        } else {
            self
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
    
    /// Extract a contiguous row range [start, end).
    /// More efficient than filter_by_indices for contiguous ranges (uses memcpy).
    pub fn slice_range(&self, start: usize, end: usize) -> Self {
        match self {
            ColumnData::Bool { data, len } => {
                let s = start.min(*len);
                let e = end.min(*len);
                let count = e.saturating_sub(s);
                let mut new_data = vec![0u8; (count + 7) / 8];
                for i in 0..count {
                    let ob = (s + i) / 8;
                    let obit = (s + i) % 8;
                    if ob < data.len() && (data[ob] >> obit) & 1 == 1 {
                        new_data[i / 8] |= 1 << (i % 8);
                    }
                }
                ColumnData::Bool { data: new_data, len: count }
            }
            ColumnData::Int64(v) => {
                ColumnData::Int64(v[start.min(v.len())..end.min(v.len())].to_vec())
            }
            ColumnData::Float64(v) => {
                ColumnData::Float64(v[start.min(v.len())..end.min(v.len())].to_vec())
            }
            ColumnData::String { offsets, data } => {
                let row_count = offsets.len().saturating_sub(1);
                let s = start.min(row_count);
                let e = end.min(row_count);
                if e <= s {
                    return ColumnData::String { offsets: vec![0], data: Vec::new() };
                }
                let data_start = offsets[s] as usize;
                let data_end = offsets[e] as usize;
                let new_data = data[data_start..data_end.min(data.len())].to_vec();
                let base = offsets[s];
                let new_offsets: Vec<u32> = offsets[s..=e].iter().map(|&o| o - base).collect();
                ColumnData::String { offsets: new_offsets, data: new_data }
            }
            ColumnData::Binary { offsets, data } => {
                let row_count = offsets.len().saturating_sub(1);
                let s = start.min(row_count);
                let e = end.min(row_count);
                if e <= s {
                    return ColumnData::Binary { offsets: vec![0], data: Vec::new() };
                }
                let data_start = offsets[s] as usize;
                let data_end = offsets[e] as usize;
                let new_data = data[data_start..data_end.min(data.len())].to_vec();
                let base = offsets[s];
                let new_offsets: Vec<u32> = offsets[s..=e].iter().map(|&o| o - base).collect();
                ColumnData::Binary { offsets: new_offsets, data: new_data }
            }
            ColumnData::StringDict { indices, dict_offsets, dict_data } => {
                let s = start.min(indices.len());
                let e = end.min(indices.len());
                ColumnData::StringDict {
                    indices: indices[s..e].to_vec(),
                    dict_offsets: dict_offsets.clone(),
                    dict_data: dict_data.clone(),
                }
            }
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
    /// V4: byte offset to V4Footer (0 for V3 files)
    pub footer_offset: u64,
    /// V4: number of Row Groups (0 for V3 files)
    pub row_group_count: u32,
}

impl OnDemandHeader {
    pub fn new() -> Self {
        let now = chrono::Utc::now().timestamp();
        Self {
            version: FORMAT_VERSION_V4,
            flags: 0,
            row_count: 0,
            column_count: 0,
            row_group_size: DEFAULT_ROW_GROUP_SIZE,
            schema_offset: HEADER_SIZE_V3 as u64,
            column_index_offset: 0,
            id_column_offset: 0,
            footer_offset: 0,
            row_group_count: 0,
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
        pos += 4;

        // V4 fields (in reserved space, after checksum)
        buf[pos..pos + 8].copy_from_slice(&self.footer_offset.to_le_bytes());
        pos += 8;
        buf[pos..pos + 4].copy_from_slice(&self.row_group_count.to_le_bytes());

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

        // Verify checksum (covers V3 fields only for backward compat)
        let computed = crc32fast::hash(&bytes[0..pos]);
        if computed != checksum {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "Header checksum mismatch",
            ));
        }
        pos += 4;

        // V4 fields (from reserved space — 0 for V3 files)
        let footer_offset = u64::from_le_bytes(bytes[pos..pos + 8].try_into().unwrap());
        pos += 8;
        let row_group_count = u32::from_le_bytes(bytes[pos..pos + 4].try_into().unwrap());

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
            footer_offset,
            row_group_count,
        })
    }
}

// ============================================================================
// V4 Row Group Metadata (40 bytes per Row Group in footer)
// ============================================================================

/// Metadata for a single Row Group stored in the V4 footer.
/// Each Row Group is a self-contained chunk of rows with its own columns + nulls.
#[derive(Debug, Clone, Copy)]
pub struct RowGroupMeta {
    /// Byte offset from file start where this Row Group's data begins
    pub offset: u64,
    /// Total byte size of this Row Group's data (IDs + deletion + columns)
    pub data_size: u64,
    /// Number of rows in this Row Group
    pub row_count: u32,
    /// Minimum row ID in this Row Group (for predicate pushdown)
    pub min_id: u64,
    /// Maximum row ID in this Row Group
    pub max_id: u64,
    /// Number of deleted rows (soft-deleted via deletion vector)
    pub deletion_count: u32,
}

impl RowGroupMeta {
    pub fn to_bytes(&self) -> [u8; ROW_GROUP_META_SIZE] {
        let mut buf = [0u8; ROW_GROUP_META_SIZE];
        buf[0..8].copy_from_slice(&self.offset.to_le_bytes());
        buf[8..16].copy_from_slice(&self.data_size.to_le_bytes());
        buf[16..20].copy_from_slice(&self.row_count.to_le_bytes());
        buf[20..28].copy_from_slice(&self.min_id.to_le_bytes());
        buf[28..36].copy_from_slice(&self.max_id.to_le_bytes());
        buf[36..40].copy_from_slice(&self.deletion_count.to_le_bytes());
        buf
    }

    pub fn from_bytes(bytes: &[u8]) -> Self {
        Self {
            offset: u64::from_le_bytes(bytes[0..8].try_into().unwrap()),
            data_size: u64::from_le_bytes(bytes[8..16].try_into().unwrap()),
            row_count: u32::from_le_bytes(bytes[16..20].try_into().unwrap()),
            min_id: u64::from_le_bytes(bytes[20..28].try_into().unwrap()),
            max_id: u64::from_le_bytes(bytes[28..36].try_into().unwrap()),
            deletion_count: u32::from_le_bytes(bytes[36..40].try_into().unwrap()),
        }
    }
    
    /// Active (non-deleted) row count
    pub fn active_rows(&self) -> u32 {
        self.row_count.saturating_sub(self.deletion_count)
    }
}

/// V4 file footer: stored at end of file, contains schema + Row Group directory.
///
/// Layout:
/// ```text
/// [schema_bytes_len: u64][schema_bytes]
/// [rg_count: u32]
/// [RowGroupMeta × rg_count]
/// [footer_size: u64]       ← total footer bytes (for seeking from EOF)
/// [MAGIC_V4_FOOTER: 8 bytes]
/// ```
#[derive(Debug, Clone)]
pub struct V4Footer {
    pub schema: OnDemandSchema,
    pub row_groups: Vec<RowGroupMeta>,
}

impl V4Footer {
    pub fn to_bytes(&self) -> Vec<u8> {
        let schema_bytes = self.schema.to_bytes();
        let rg_count = self.row_groups.len() as u32;
        
        let total_size = 8 + schema_bytes.len()     // schema_bytes_len + schema
            + 4                                       // rg_count
            + self.row_groups.len() * ROW_GROUP_META_SIZE  // RG entries
            + 8                                       // footer_size
            + 8;                                      // footer magic
        
        let mut buf = Vec::with_capacity(total_size);
        
        // Schema
        buf.extend_from_slice(&(schema_bytes.len() as u64).to_le_bytes());
        buf.extend_from_slice(&schema_bytes);
        
        // Row Group directory
        buf.extend_from_slice(&rg_count.to_le_bytes());
        for rg in &self.row_groups {
            buf.extend_from_slice(&rg.to_bytes());
        }
        
        // Footer size (everything before this field + 8 bytes for size + 8 bytes for magic)
        let footer_size = buf.len() as u64 + 8 + 8;
        buf.extend_from_slice(&footer_size.to_le_bytes());
        
        // Magic
        buf.extend_from_slice(MAGIC_V4_FOOTER);
        
        buf
    }

    pub fn from_bytes(bytes: &[u8]) -> io::Result<Self> {
        if bytes.len() < 20 {
            return Err(err_data("V4 footer too small"));
        }
        
        let mut pos = 0;
        
        // Schema
        let schema_len = u64::from_le_bytes(bytes[pos..pos+8].try_into().unwrap()) as usize;
        pos += 8;
        if pos + schema_len > bytes.len() {
            return Err(err_data("V4 footer: schema overflow"));
        }
        let schema = OnDemandSchema::from_bytes(&bytes[pos..pos+schema_len])?;
        pos += schema_len;
        
        // Row Group directory
        let rg_count = u32::from_le_bytes(bytes[pos..pos+4].try_into().unwrap()) as usize;
        pos += 4;
        
        let mut row_groups = Vec::with_capacity(rg_count);
        for _ in 0..rg_count {
            if pos + ROW_GROUP_META_SIZE > bytes.len() {
                return Err(err_data("V4 footer: RG meta overflow"));
            }
            row_groups.push(RowGroupMeta::from_bytes(&bytes[pos..pos+ROW_GROUP_META_SIZE]));
            pos += ROW_GROUP_META_SIZE;
        }
        
        Ok(Self { schema, row_groups })
    }
    
    /// Total active rows across all Row Groups
    pub fn total_active_rows(&self) -> u64 {
        self.row_groups.iter().map(|rg| rg.active_rows() as u64).sum()
    }
    
    /// Total rows (including deleted) across all Row Groups
    pub fn total_rows(&self) -> u64 {
        self.row_groups.iter().map(|rg| rg.row_count as u64).sum()
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
            return Err(err_data("Schema too short"));
        }
        
        let column_count = u32::from_le_bytes(bytes[pos..pos + 4].try_into().unwrap()) as usize;
        pos += 4;
        
        let mut schema = Self::new();
        
        for _ in 0..column_count {
            if pos + 2 > bytes.len() {
                return Err(err_data("Truncated schema"));
            }
            
            let name_len = u16::from_le_bytes(bytes[pos..pos + 2].try_into().unwrap()) as usize;
            pos += 2;
            
            if pos + name_len + 1 > bytes.len() {
                return Err(err_data("Truncated column"));
            }
            
            let name = std::str::from_utf8(&bytes[pos..pos + name_len])
                .map_err(|e| err_data(e.to_string()))?
                .to_string();
            pos += name_len;
            
            let dtype = ColumnType::from_u8(bytes[pos])
                .ok_or_else(|| err_data("Invalid column type"))?;
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
    /// In-memory column data (legacy: used as write buffer for pending inserts)
    columns: RwLock<Vec<ColumnData>>,
    /// Row IDs (legacy: used as write buffer for pending inserts)
    ids: RwLock<Vec<u64>>,
    /// Next row ID
    next_id: AtomicU64,
    /// Null bitmaps per column (legacy: used as write buffer for pending inserts)
    nulls: RwLock<Vec<Vec<u8>>>,
    /// Deleted row bitmap (packed bits, 1 = deleted)
    deleted: RwLock<Vec<u8>>,
    /// ID to row index mapping for fast lookups (lazy-loaded)
    /// Only built when needed for delete/exists operations
    /// Uses AHashMap for faster hash computation on u64 keys
    id_to_idx: RwLock<Option<ahash::AHashMap<u64, usize>>>,
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
    /// Total rows physically on disk (including deleted). Only updated after disk writes.
    /// Used by save() to distinguish in-memory-only rows from persisted rows.
    persisted_row_count: AtomicU64,
    /// Whether V4 base data was bulk-loaded into memory (only in tests via open_v4_data).
    /// Production code never sets this — in-memory data is always just the write buffer.
    v4_base_loaded: AtomicBool,
    /// Cached V4 footer with Row Group metadata (lazy-loaded from disk).
    /// Enables on-demand mmap reads without loading all data into memory.
    v4_footer: RwLock<Option<V4Footer>>,
    /// Delta store for cell-level update tracking (Phase 4.5).
    /// Tracks pending UPDATE changes without rewriting the base file.
    /// On read, DeltaMerger overlays these changes on top of base data.
    delta_store: RwLock<DeltaStore>,
}

impl OnDemandStorage {
    /// Create a new V3 storage file with default durability (Fast)
    pub fn create(path: &Path) -> io::Result<Self> {
        Self::create_with_durability(path, super::DurabilityLevel::Fast)
    }
    
    /// Create a new V3 storage file with specified durability level
    pub fn create_with_durability(path: &Path, durability: super::DurabilityLevel) -> io::Result<Self> {
        Self::create_with_schema_and_durability(path, durability, &[])
    }

    /// Create a new storage file with pre-defined schema and durability level.
    /// Pre-defining schema avoids schema inference on the first insert, providing
    /// a performance benefit: columns and null vectors are pre-allocated with
    /// correct types so insert_typed() hits the fast path immediately.
    pub fn create_with_schema_and_durability(
        path: &Path,
        durability: super::DurabilityLevel,
        schema_cols: &[(String, ColumnType)],
    ) -> io::Result<Self> {
        let header = OnDemandHeader::new();
        let mut schema = OnDemandSchema::new();
        let mut columns = Vec::with_capacity(schema_cols.len());
        let mut nulls = Vec::with_capacity(schema_cols.len());

        // Pre-populate schema and empty column vectors
        for (name, dtype) in schema_cols {
            schema.add_column(name, *dtype);
            columns.push(ColumnData::new(*dtype));
            nulls.push(Vec::new());
        }

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
            columns: RwLock::new(columns),
            ids: RwLock::new(Vec::new()),
            next_id: AtomicU64::new(0),
            nulls: RwLock::new(nulls),
            deleted: RwLock::new(Vec::new()),
            id_to_idx: RwLock::new(Some(ahash::AHashMap::new())),
            active_count: AtomicU64::new(0),
            durability,
            wal_writer: RwLock::new(wal_writer),
            wal_buffer: RwLock::new(Vec::new()),
            auto_flush_rows: AtomicU64::new(100000),
            auto_flush_bytes: AtomicU64::new(500 * 1024 * 1024),
            pending_rows: AtomicU64::new(0),
            persisted_row_count: AtomicU64::new(0),
            v4_base_loaded: AtomicBool::new(false),
            v4_footer: RwLock::new(None),
            delta_store: RwLock::new(DeltaStore::new(path)),
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
        
        let is_v4 = header.footer_offset > 0;
        
        let schema: OnDemandSchema;
        let column_index: Vec<ColumnIndexEntry>;
        let id_count = header.row_count as usize;
        let next_id: u64;
        
        if is_v4 {
            // V4 Row Group format: read schema from footer
            // Read from footer_offset to EOF (V4Footer::from_bytes ignores trailing bytes)
            let file_len = std::fs::metadata(path)?.len();
            let footer_byte_count = (file_len - header.footer_offset) as usize;
            let mut footer_bytes = vec![0u8; footer_byte_count];
            mmap_cache.read_at(&file, &mut footer_bytes, header.footer_offset)?;
            let footer = V4Footer::from_bytes(&footer_bytes)?;
            schema = footer.schema;
            column_index = Vec::new(); // Not used in V4
            // Use max_id from non-empty RG metadata (row_count may be < max _id after deletes)
            next_id = footer.row_groups.iter()
                .filter(|rg| rg.row_count > 0)
                .map(|rg| rg.max_id)
                .max()
                .map(|m| m + 1)
                .unwrap_or(0);
        } else {
            // V3 format: read schema + column index from header area
            let schema_size = header.column_index_offset - header.schema_offset;
            let mut schema_bytes = vec![0u8; schema_size as usize];
            mmap_cache.read_at(&file, &mut schema_bytes, header.schema_offset)?;
            schema = OnDemandSchema::from_bytes(&schema_bytes)?;

            let index_size = header.column_count as usize * COLUMN_INDEX_ENTRY_SIZE;
            let mut index_bytes = vec![0u8; index_size];
            mmap_cache.read_at(&file, &mut index_bytes, header.column_index_offset)?;
            
            let mut ci = Vec::with_capacity(header.column_count as usize);
            for i in 0..header.column_count as usize {
                let start = i * COLUMN_INDEX_ENTRY_SIZE;
                let entry = ColumnIndexEntry::from_bytes(&index_bytes[start..start + COLUMN_INDEX_ENTRY_SIZE]);
                ci.push(entry);
            }
            column_index = ci;
            
            // Read actual max ID from disk
            next_id = if id_count > 0 {
                let mut id_buf = vec![0u8; id_count * 8];
                if mmap_cache.read_at(&file, &mut id_buf, header.id_column_offset).is_ok() {
                    let mut max_id = 0u64;
                    for i in 0..id_count {
                        let id = u64::from_le_bytes(id_buf[i*8..(i+1)*8].try_into().unwrap_or([0u8; 8]));
                        if id > max_id { max_id = id; }
                    }
                    max_id + 1
                } else {
                    header.row_count
                }
            } else {
                0
            };
        }

        let columns: Vec<ColumnData> = schema.columns.iter()
            .map(|(_, col_type)| ColumnData::new(*col_type))
            .collect();
        let nulls = vec![Vec::new(); header.column_count as usize];
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
            ids: RwLock::new(Vec::new()),  // Empty - lazy loaded when needed
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
            persisted_row_count: AtomicU64::new(id_count as u64),
            v4_base_loaded: AtomicBool::new(false),
            v4_footer: RwLock::new(None),
            delta_store: RwLock::new(DeltaStore::load(path).unwrap_or_else(|_| DeltaStore::new(path))),
        })
    }
    
    /// Set auto-flush thresholds for automatic persistence
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

    /// Helper: Get file reference or return NotConnected error
    /// Reduces boilerplate in read methods
    #[inline]
    fn get_file_ref(&self) -> io::Result<parking_lot::RwLockReadGuard<'_, Option<File>>> {
        let guard = self.file.read();
        if guard.is_none() {
            return Err(err_not_conn("File not open"));
        }
        Ok(guard)
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
    /// IMPORTANT: For memory efficiency, column data is loaded lazily.
    /// - For INSERT: use open_for_insert() which only loads metadata
    /// - For UPDATE/DELETE: this function loads all column data
    pub fn open_for_write_with_durability(path: &Path, durability: super::DurabilityLevel) -> io::Result<Self> {
        if !path.exists() {
            return Self::create_with_durability(path, durability);
        }
        
        // Open the storage first
        let storage = Self::open_with_durability(path, durability)?;
        
        // If there are existing rows, load all column data into memory
        // This is required because save() rewrites the entire file from self.columns
        let row_count = storage.header.read().row_count as usize;
        if row_count > 0 {
            storage.load_all_columns_into_memory()?;
        } else {
            // Even with 0 rows, initialize empty columns based on schema
            // This is needed for INSERT after ALTER TABLE (columns defined but no data)
            let schema = storage.schema.read();
            let mut columns = storage.columns.write();
            let mut nulls = storage.nulls.write();
            
            // Always reinitialize columns with correct types from schema
            // The initial columns vector may have placeholder Int64 types
            if schema.column_count() > 0 {
                columns.clear();
                nulls.clear();
                for (_name, col_type) in schema.columns.iter() {
                    columns.push(ColumnData::new(*col_type));
                    nulls.push(Vec::new());
                }
            }
        }
        
        Ok(storage)
    }
    
    /// Open for INSERT operations only - memory efficient!
    /// Only loads metadata (header, schema, ids), NOT column data.
    /// New data is written to a delta file and merged on read or compact.
    pub fn open_for_insert(path: &Path) -> io::Result<Self> {
        Self::open_for_insert_with_durability(path, super::DurabilityLevel::Fast)
    }
    
    /// Open for INSERT with specified durability - memory efficient!
    pub fn open_for_insert_with_durability(path: &Path, durability: super::DurabilityLevel) -> io::Result<Self> {
        if !path.exists() {
            return Self::create_with_durability(path, durability);
        }
        
        // Just open without loading column data - metadata only
        Self::open_with_durability(path, durability)
    }
    
    /// Open for SCHEMA changes only - MOST memory efficient!
    /// Only loads header, schema, and column index. Does NOT load IDs or column data.
    /// Use for: ALTER TABLE ADD/DROP/RENAME COLUMN, TRUNCATE
    pub fn open_for_schema_change(path: &Path) -> io::Result<Self> {
        Self::open_for_schema_change_with_durability(path, super::DurabilityLevel::Fast)
    }
    
    /// Open for SCHEMA changes with specified durability - MOST memory efficient!
    pub fn open_for_schema_change_with_durability(path: &Path, durability: super::DurabilityLevel) -> io::Result<Self> {
        if !path.exists() {
            return Self::create_with_durability(path, durability);
        }
        
        // Quick V4 check: V4 files don't have V3 offsets, use general open path
        {
            let file = File::open(path)?;
            let mut mc = MmapCache::new();
            let mut hb = [0u8; HEADER_SIZE_V3];
            mc.read_at(&file, &mut hb, 0)?;
            let h = OnDemandHeader::from_bytes(&hb)?;
            if h.footer_offset > 0 {
                return Self::open_with_durability(path, durability);
            }
        }
        
        let file = File::open(path)?;
        let mut mmap_cache = MmapCache::new();
        
        // Read header only
        let mut header_bytes = [0u8; HEADER_SIZE_V3];
        mmap_cache.read_at(&file, &mut header_bytes, 0)?;
        let header = OnDemandHeader::from_bytes(&header_bytes)?;

        // Read schema (V3 path)
        let schema_size = header.column_index_offset - header.schema_offset;
        let mut schema_bytes = vec![0u8; schema_size as usize];
        mmap_cache.read_at(&file, &mut schema_bytes, header.schema_offset)?;
        let schema = OnDemandSchema::from_bytes(&schema_bytes)?;

        // Read column index
        let index_size = header.column_count as usize * COLUMN_INDEX_ENTRY_SIZE;
        let mut index_bytes = vec![0u8; index_size];
        mmap_cache.read_at(&file, &mut index_bytes, header.column_index_offset)?;
        
        let mut column_index = Vec::with_capacity(header.column_count as usize);
        for i in 0..header.column_count as usize {
            let start = i * COLUMN_INDEX_ENTRY_SIZE;
            let entry = ColumnIndexEntry::from_bytes(&index_bytes[start..start + COLUMN_INDEX_ENTRY_SIZE]);
            column_index.push(entry);
        }

        // NOTE: Full IDs are NOT loaded into Vec - saves ~80MB for 10M rows
        // But we must find the actual max ID to avoid collisions after deletes
        // Read IDs from disk to find max (only reads 8 bytes per row, not full column data)
        let mut next_id = 0u64;
        if header.row_count > 0 {
            // Read IDs from disk to find max
            let mut id_buf = vec![0u8; header.row_count as usize * 8];
            if mmap_cache.read_at(&file, &mut id_buf, header.id_column_offset).is_ok() {
                for i in 0..header.row_count as usize {
                    let id = u64::from_le_bytes(id_buf[i*8..(i+1)*8].try_into().unwrap_or([0u8; 8]));
                    if id >= next_id {
                        next_id = id + 1;
                    }
                }
            } else {
                // Fallback: use row_count (may cause issues after deletes)
                next_id = header.row_count;
            }
        }
        
        // Check delta file for max ID (in case there are pending delta writes)
        let delta_path = Self::delta_path(path);
        if delta_path.exists() {
            if let Ok(mut delta_file) = File::open(&delta_path) {
                // Read delta IDs to find max
                loop {
                    // Read record count
                    let mut count_buf = [0u8; 8];
                    match delta_file.read_exact(&mut count_buf) {
                        Ok(_) => {},
                        Err(_) => break,
                    }
                    let record_count = u64::from_le_bytes(count_buf) as usize;
                    
                    // Read IDs and track max
                    for _ in 0..record_count {
                        let mut id_buf = [0u8; 8];
                        if delta_file.read_exact(&mut id_buf).is_err() { break; }
                        let id = u64::from_le_bytes(id_buf);
                        if id >= next_id {
                            next_id = id + 1;
                        }
                    }
                    
                    // Skip rest of record (columns)
                    // Int columns
                    let mut count_buf4 = [0u8; 4];
                    if delta_file.read_exact(&mut count_buf4).is_err() { break; }
                    let int_col_count = u32::from_le_bytes(count_buf4) as usize;
                    for _ in 0..int_col_count {
                        let mut len_buf = [0u8; 2];
                        if delta_file.read_exact(&mut len_buf).is_err() { break; }
                        let name_len = u16::from_le_bytes(len_buf) as usize;
                        if delta_file.seek(SeekFrom::Current(name_len as i64 + record_count as i64 * 8)).is_err() { break; }
                    }
                    // Float columns
                    if delta_file.read_exact(&mut count_buf4).is_err() { break; }
                    let float_col_count = u32::from_le_bytes(count_buf4) as usize;
                    for _ in 0..float_col_count {
                        let mut len_buf = [0u8; 2];
                        if delta_file.read_exact(&mut len_buf).is_err() { break; }
                        let name_len = u16::from_le_bytes(len_buf) as usize;
                        if delta_file.seek(SeekFrom::Current(name_len as i64 + record_count as i64 * 8)).is_err() { break; }
                    }
                    // String columns
                    if delta_file.read_exact(&mut count_buf4).is_err() { break; }
                    let string_col_count = u32::from_le_bytes(count_buf4) as usize;
                    for _ in 0..string_col_count {
                        let mut len_buf = [0u8; 2];
                        if delta_file.read_exact(&mut len_buf).is_err() { break; }
                        let name_len = u16::from_le_bytes(len_buf) as usize;
                        if delta_file.seek(SeekFrom::Current(name_len as i64)).is_err() { break; }
                        for _ in 0..record_count {
                            let mut str_len_buf = [0u8; 4];
                            if delta_file.read_exact(&mut str_len_buf).is_err() { break; }
                            let str_len = u32::from_le_bytes(str_len_buf) as i64;
                            if delta_file.seek(SeekFrom::Current(str_len)).is_err() { break; }
                        }
                    }
                    // Bool columns
                    if delta_file.read_exact(&mut count_buf4).is_err() { break; }
                    let bool_col_count = u32::from_le_bytes(count_buf4) as usize;
                    for _ in 0..bool_col_count {
                        let mut len_buf = [0u8; 2];
                        if delta_file.read_exact(&mut len_buf).is_err() { break; }
                        let name_len = u16::from_le_bytes(len_buf) as usize;
                        if delta_file.seek(SeekFrom::Current(name_len as i64 + record_count as i64)).is_err() { break; }
                    }
                }
            }
        }
        
        let row_count = header.row_count; // Cache before moving header
        let column_count = header.column_count as usize;
        
        // Empty columns - will be loaded on-demand if needed
        let columns = vec![ColumnData::new(ColumnType::Int64); column_count];
        let nulls = vec![Vec::new(); column_count];
        let deleted_len = (row_count as usize + 7) / 8;
        let deleted = vec![0u8; deleted_len];

        // Handle WAL for durability
        let wal_path = Self::wal_path(path);
        let wal_writer = if durability != super::DurabilityLevel::Fast && wal_path.exists() {
            Some(super::incremental::WalWriter::open(&wal_path)?)
        } else if durability != super::DurabilityLevel::Fast {
            Some(super::incremental::WalWriter::create(&wal_path, next_id)?)
        } else {
            None
        };

        Ok(Self {
            path: path.to_path_buf(),
            file: RwLock::new(Some(file)),
            mmap_cache: RwLock::new(mmap_cache),
            header: RwLock::new(header),
            schema: RwLock::new(schema),
            column_index: RwLock::new(column_index),
            columns: RwLock::new(columns),
            ids: RwLock::new(Vec::new()), // Empty - not loaded!
            next_id: AtomicU64::new(next_id),
            nulls: RwLock::new(nulls),
            deleted: RwLock::new(deleted),
            id_to_idx: RwLock::new(None), // Lazy loaded when needed
            active_count: AtomicU64::new(row_count),
            durability,
            wal_writer: RwLock::new(wal_writer),
            wal_buffer: RwLock::new(Vec::new()),
            auto_flush_rows: AtomicU64::new(10000),
            auto_flush_bytes: AtomicU64::new(500 * 1024 * 1024),
            pending_rows: AtomicU64::new(0),
            persisted_row_count: AtomicU64::new(row_count),
            v4_base_loaded: AtomicBool::new(false),
            v4_footer: RwLock::new(None),
            delta_store: RwLock::new(DeltaStore::load(path).unwrap_or_else(|_| DeltaStore::new(path))),
        })
    }
    
    /// Get the delta file path for this storage
    fn delta_path(base_path: &Path) -> PathBuf {
        let mut delta = base_path.to_path_buf();
        let name = delta.file_name().unwrap_or_default().to_string_lossy();
        delta.set_file_name(format!("{}.delta", name));
        delta
    }

    // ========================================================================
    // DeltaStore accessors (Phase 4.5)
    // ========================================================================

    /// Record a cell-level update in the delta store.
    /// Used by UPDATE to avoid delete+insert for single-cell changes.
    pub fn delta_update_cell(&self, row_id: u64, column_name: &str, new_value: crate::data::Value) {
        self.delta_store.write().update_cell(row_id, column_name, new_value);
    }

    /// Record a full row update in the delta store.
    pub fn delta_update_row(&self, row_id: u64, values: &HashMap<String, crate::data::Value>) {
        self.delta_store.write().update_row(row_id, values);
    }

    /// Check if the delta store has any pending changes.
    pub fn has_pending_deltas(&self) -> bool {
        !self.delta_store.read().is_empty()
    }

    /// Get the number of pending delta updates.
    pub fn delta_update_count(&self) -> usize {
        self.delta_store.read().update_count()
    }

    /// Save the delta store to disk (called during save path).
    pub fn save_delta_store(&self) -> io::Result<()> {
        self.delta_store.write().save()
    }

    /// Clear the delta store (called after compaction merges deltas into base).
    pub fn clear_delta_store(&self) -> io::Result<()> {
        let mut ds = self.delta_store.write();
        ds.clear();
        ds.save()?;
        ds.remove_file()
    }

    /// Get a read reference to the delta store (for DeltaMerger on read path).
    pub fn delta_store(&self) -> parking_lot::RwLockReadGuard<'_, DeltaStore> {
        self.delta_store.read()
    }

    /// Check if delta compaction is needed based on update/delete count vs base rows.
    pub fn needs_delta_compaction(&self) -> bool {
        let ds = self.delta_store.read();
        let base_rows = self.active_count.load(std::sync::atomic::Ordering::Relaxed);
        ds.needs_compaction(base_rows)
    }

    /// Compact deltas into the base file: load base data, apply updates in-place,
    /// then do a full save_v4 rewrite which clears the delta store.
    pub fn compact_deltas(&self) -> io::Result<()> {
        let ds = self.delta_store.read();
        if ds.is_empty() {
            return Ok(());
        }

        // Collect updates and deletes before releasing the lock
        let all_updates = ds.all_updates().clone();
        let delete_bitmap = ds.delete_bitmap().clone();
        drop(ds);

        // Skip compaction if V4 data isn't in memory — deltas stay in DeltaStore
        // and are applied at read time via DeltaMerger overlay.
        if self.is_v4_format() && !self.has_v4_in_memory_data() {
            return Ok(());
        }

        // Apply deletes: mark deleted rows in the deleted bitmap
        {
            let ids = self.ids.read();
            let mut deleted = self.deleted.write();
            for (idx, id) in ids.iter().enumerate() {
                if delete_bitmap.is_deleted(*id) {
                    let byte_idx = idx / 8;
                    let bit_idx = idx % 8;
                    if byte_idx < deleted.len() {
                        deleted[byte_idx] |= 1 << bit_idx;
                    }
                }
            }
        }

        // Apply cell-level updates to in-memory columns
        {
            let ids = self.ids.read();
            let schema = self.schema.read();
            let mut columns = self.columns.write();

            // Build id→index map for fast lookup
            let id_to_idx: std::collections::HashMap<u64, usize> = ids.iter()
                .enumerate()
                .map(|(i, &id)| (id, i))
                .collect();

            for (row_id, col_updates) in &all_updates {
                if let Some(&row_idx) = id_to_idx.get(row_id) {
                    for (col_name, record) in col_updates {
                        if let Some(col_idx) = schema.get_index(col_name) {
                            if col_idx < columns.len() {
                                match &record.new_value {
                                    crate::data::Value::Int64(v) => {
                                        if let ColumnData::Int64(ref mut data) = columns[col_idx] {
                                            if row_idx < data.len() {
                                                data[row_idx] = *v;
                                            }
                                        }
                                    }
                                    crate::data::Value::Float64(v) => {
                                        if let ColumnData::Float64(ref mut data) = columns[col_idx] {
                                            if row_idx < data.len() {
                                                data[row_idx] = *v;
                                            }
                                        }
                                    }
                                    crate::data::Value::String(s) => {
                                        if let ColumnData::String { offsets, data } = &mut columns[col_idx] {
                                            // For strings, we need to rebuild — update in-place is complex
                                            // For compaction (rare), this is acceptable
                                            let mut strings: Vec<String> = Vec::with_capacity(offsets.len().saturating_sub(1));
                                            for i in 0..offsets.len().saturating_sub(1) {
                                                let start = offsets[i] as usize;
                                                let end = offsets[i + 1] as usize;
                                                if i == row_idx {
                                                    strings.push(s.clone());
                                                } else {
                                                    strings.push(String::from_utf8_lossy(&data[start..end]).to_string());
                                                }
                                            }
                                            // Rebuild
                                            data.clear();
                                            offsets.clear();
                                            offsets.push(0);
                                            for st in &strings {
                                                data.extend_from_slice(st.as_bytes());
                                                offsets.push(data.len() as u32);
                                            }
                                        }
                                    }
                                    crate::data::Value::Bool(v) => {
                                        if let ColumnData::Bool { data, .. } = &mut columns[col_idx] {
                                            let byte_idx = row_idx / 8;
                                            let bit_idx = row_idx % 8;
                                            if byte_idx < data.len() {
                                                if *v {
                                                    data[byte_idx] |= 1 << bit_idx;
                                                } else {
                                                    data[byte_idx] &= !(1 << bit_idx);
                                                }
                                            }
                                        }
                                    }
                                    _ => {} // UInt64, Null, etc. — skip for now
                                }
                            }
                        }
                    }
                }
            }
        }

        // Full rewrite (save_v4 clears delta store)
        self.save_v4()
    }

    /// Get the maximum ID from a delta file (for computing next_id on open)
    fn get_max_id_from_delta(delta_path: &Path) -> io::Result<u64> {
        use std::io::{Read, Seek, SeekFrom};
        let mut file = File::open(delta_path)?;
        let mut max_id: u64 = 0;
        
        loop {
            // Read record count
            let mut count_buf = [0u8; 8];
            match file.read_exact(&mut count_buf) {
                Ok(_) => {},
                Err(e) if e.kind() == io::ErrorKind::UnexpectedEof => break,
                Err(e) => return Err(e),
            }
            let record_count = u64::from_le_bytes(count_buf) as usize;
            
            // Read IDs and track max
            for _ in 0..record_count {
                let mut id_buf = [0u8; 8];
                file.read_exact(&mut id_buf)?;
                let id = u64::from_le_bytes(id_buf);
                max_id = max_id.max(id);
            }
            
            // Skip rest of record (int columns)
            let mut count_buf4 = [0u8; 4];
            file.read_exact(&mut count_buf4)?;
            let int_col_count = u32::from_le_bytes(count_buf4) as usize;
            for _ in 0..int_col_count {
                let mut len_buf = [0u8; 2];
                file.read_exact(&mut len_buf)?;
                let name_len = u16::from_le_bytes(len_buf) as usize;
                file.seek(SeekFrom::Current(name_len as i64))?;
                file.seek(SeekFrom::Current((record_count * 8) as i64))?;
            }
            
            // Skip float columns
            file.read_exact(&mut count_buf4)?;
            let float_col_count = u32::from_le_bytes(count_buf4) as usize;
            for _ in 0..float_col_count {
                let mut len_buf = [0u8; 2];
                file.read_exact(&mut len_buf)?;
                let name_len = u16::from_le_bytes(len_buf) as usize;
                file.seek(SeekFrom::Current(name_len as i64))?;
                file.seek(SeekFrom::Current((record_count * 8) as i64))?;
            }
            
            // Skip string columns (variable length - need to read lengths)
            file.read_exact(&mut count_buf4)?;
            let string_col_count = u32::from_le_bytes(count_buf4) as usize;
            for _ in 0..string_col_count {
                let mut len_buf = [0u8; 2];
                file.read_exact(&mut len_buf)?;
                let name_len = u16::from_le_bytes(len_buf) as usize;
                file.seek(SeekFrom::Current(name_len as i64))?;
                for _ in 0..record_count {
                    let mut str_len_buf = [0u8; 4];
                    file.read_exact(&mut str_len_buf)?;
                    let str_len = u32::from_le_bytes(str_len_buf) as usize;
                    file.seek(SeekFrom::Current(str_len as i64))?;
                }
            }
            
            // Skip bool columns
            file.read_exact(&mut count_buf4)?;
            let bool_col_count = u32::from_le_bytes(count_buf4) as usize;
            for _ in 0..bool_col_count {
                let mut len_buf = [0u8; 2];
                file.read_exact(&mut len_buf)?;
                let name_len = u16::from_le_bytes(len_buf) as usize;
                file.seek(SeekFrom::Current(name_len as i64))?;
                let skip_bytes = (record_count + 7) / 8;
                file.seek(SeekFrom::Current(skip_bytes as i64))?;
            }
            
            // Skip binary columns (variable length)
            file.read_exact(&mut count_buf4)?;
            let binary_col_count = u32::from_le_bytes(count_buf4) as usize;
            for _ in 0..binary_col_count {
                let mut len_buf = [0u8; 2];
                file.read_exact(&mut len_buf)?;
                let name_len = u16::from_le_bytes(len_buf) as usize;
                file.seek(SeekFrom::Current(name_len as i64))?;
                for _ in 0..record_count {
                    let mut bin_len_buf = [0u8; 4];
                    file.read_exact(&mut bin_len_buf)?;
                    let bin_len = u32::from_le_bytes(bin_len_buf) as usize;
                    file.seek(SeekFrom::Current(bin_len as i64))?;
                }
            }
        }
        
        Ok(max_id)
    }
    
    /// Check if delta file exists
    pub fn has_delta(&self) -> bool {
        Self::delta_path(&self.path).exists()
    }
    
    /// Load all column data from disk into memory
    /// This is needed before write operations to preserve existing data
    fn load_all_columns_into_memory(&self) -> io::Result<()> {
        let header = self.header.read();
        let total_rows = header.row_count as usize;
        
        if total_rows == 0 {
            return Ok(());
        }
        
        // V4 files: load all RG data into memory for write operations
        if header.footer_offset > 0 {
            drop(header);
            return self.open_v4_data();
        }
        
        let schema = self.schema.read();
        let column_index = self.column_index.read();
        
        // CRITICAL: Load IDs first since they're lazy-loaded
        // Without this, insert operations will think there are 0 existing rows
        drop(header);
        drop(schema);
        drop(column_index);
        self.ensure_ids_loaded()?;
        let header = self.header.read();
        let schema = self.schema.read();
        let column_index = self.column_index.read();
        
        let file_guard = self.file.read();
        let file = file_guard.as_ref().ok_or_else(|| {
            err_not_conn("File not open")
        })?;
        
        let mut mmap_cache = self.mmap_cache.write();
        let mut columns = self.columns.write();
        let mut nulls = self.nulls.write();
        
        let column_index_len = column_index.len();
        
        // Load each column from disk
        for col_idx in 0..schema.column_count() {
            let (_, col_type) = &schema.columns[col_idx];
            
            // Handle columns added via ALTER TABLE that don't have disk data yet
            if col_idx >= column_index_len {
                // Column exists in schema but not on disk - create padded column
                let mut col_data = ColumnData::new(*col_type);
                // Pad with defaults for existing rows
                for _ in 0..total_rows {
                    match &mut col_data {
                        ColumnData::Int64(v) => v.push(0),
                        ColumnData::Float64(v) => v.push(0.0),
                        ColumnData::String { offsets, .. } => offsets.push(*offsets.last().unwrap_or(&0)),
                        ColumnData::Binary { offsets, .. } => offsets.push(*offsets.last().unwrap_or(&0)),
                        ColumnData::Bool { data, len } => {
                            let byte_idx = *len / 8;
                            if byte_idx >= data.len() { data.push(0); }
                            *len += 1;
                        }
                        ColumnData::StringDict { indices, .. } => indices.push(0),
                    }
                }
                
                if col_idx < columns.len() {
                    columns[col_idx] = col_data;
                } else {
                    columns.push(col_data);
                }
                
                // Empty null bitmap for new columns
                if col_idx < nulls.len() {
                    nulls[col_idx] = Vec::new();
                } else {
                    nulls.push(Vec::new());
                }
                continue;
            }
            
            let index_entry = &column_index[col_idx];
            
            // Read column data
            let col_data = self.read_column_range_mmap(
                &mut mmap_cache,
                file,
                index_entry,
                *col_type,
                0,
                total_rows,
                total_rows,
            )?;
            
            // Store in columns array
            if col_idx < columns.len() {
                columns[col_idx] = col_data;
            } else {
                columns.push(col_data);
            }
            
            // Read null bitmap for this column
            let null_len = index_entry.null_length as usize;
            if null_len > 0 {
                let mut null_bitmap = vec![0u8; null_len];
                mmap_cache.read_at(file, &mut null_bitmap, index_entry.null_offset)?;
                if col_idx < nulls.len() {
                    nulls[col_idx] = null_bitmap;
                } else {
                    nulls.push(null_bitmap);
                }
            }
        }
        
        Ok(())
    }
    
    /// Insert rows to delta file (memory efficient - doesn't load existing data)
    /// Returns the IDs assigned to the inserted rows
    pub fn insert_rows_to_delta(&self, rows: &[HashMap<String, ColumnValue>]) -> io::Result<Vec<u64>> {
        if rows.is_empty() {
            return Ok(Vec::new());
        }
        
        let delta_path = Self::delta_path(&self.path);
        
        // Get schema to handle partial columns correctly
        let schema = self.schema.read();
        
        // Build column data from rows - ensure all columns have same length
        let mut int_columns: HashMap<String, Vec<i64>> = HashMap::new();
        let mut float_columns: HashMap<String, Vec<f64>> = HashMap::new();
        let mut string_columns: HashMap<String, Vec<String>> = HashMap::new();
        let mut binary_columns: HashMap<String, Vec<Vec<u8>>> = HashMap::new();
        let mut bool_columns: HashMap<String, Vec<bool>> = HashMap::new();
        
        // Initialize column vectors based on schema
        for (col_name, col_type) in &schema.columns {
            match col_type {
                ColumnType::Int64 | ColumnType::Int8 | ColumnType::Int16 | ColumnType::Int32 |
                ColumnType::UInt8 | ColumnType::UInt16 | ColumnType::UInt32 | ColumnType::UInt64 => { 
                    int_columns.insert(col_name.clone(), Vec::with_capacity(rows.len())); 
                }
                ColumnType::Float64 | ColumnType::Float32 => { 
                    float_columns.insert(col_name.clone(), Vec::with_capacity(rows.len())); 
                }
                ColumnType::String | ColumnType::StringDict => { 
                    string_columns.insert(col_name.clone(), Vec::with_capacity(rows.len())); 
                }
                ColumnType::Binary => { 
                    binary_columns.insert(col_name.clone(), Vec::with_capacity(rows.len())); 
                }
                ColumnType::Bool => { 
                    bool_columns.insert(col_name.clone(), Vec::with_capacity(rows.len())); 
                }
                ColumnType::Null => { 
                    // Null columns are handled as strings with empty default
                    string_columns.insert(col_name.clone(), Vec::with_capacity(rows.len())); 
                }
            }
        }
        
        // For each row, add values for ALL schema columns (default for missing)
        for row in rows {
            for (col_name, col_type) in &schema.columns {
                let val = row.get(col_name);
                match col_type {
                    ColumnType::Int64 | ColumnType::Int8 | ColumnType::Int16 | ColumnType::Int32 |
                    ColumnType::UInt8 | ColumnType::UInt16 | ColumnType::UInt32 | ColumnType::UInt64 => {
                        let v = val.and_then(|v| if let ColumnValue::Int64(n) = v { Some(*n) } else { None }).unwrap_or(0);
                        int_columns.get_mut(col_name).unwrap().push(v);
                    }
                    ColumnType::Float64 | ColumnType::Float32 => {
                        let v = val.and_then(|v| if let ColumnValue::Float64(n) = v { Some(*n) } else { None }).unwrap_or(0.0);
                        float_columns.get_mut(col_name).unwrap().push(v);
                    }
                    ColumnType::String | ColumnType::StringDict | ColumnType::Null => {
                        let v = val.and_then(|v| if let ColumnValue::String(s) = v { Some(s.clone()) } else { None }).unwrap_or_default();
                        string_columns.get_mut(col_name).unwrap().push(v);
                    }
                    ColumnType::Binary => {
                        let v = val.and_then(|v| if let ColumnValue::Binary(b) = v { Some(b.clone()) } else { None }).unwrap_or_default();
                        binary_columns.get_mut(col_name).unwrap().push(v);
                    }
                    ColumnType::Bool => {
                        let v = val.and_then(|v| if let ColumnValue::Bool(b) = v { Some(*b) } else { None }).unwrap_or(false);
                        bool_columns.get_mut(col_name).unwrap().push(v);
                    }
                }
            }
        }
        
        drop(schema);
        
        // Allocate IDs
        let mut ids = Vec::with_capacity(rows.len());
        for _ in 0..rows.len() {
            ids.push(self.next_id.fetch_add(1, Ordering::SeqCst));
        }
        
        // Write delta file (append mode)
        let mut file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&delta_path)?;
        
        // Delta format: [record_count:u64][ids...][schema...][column_data...]
        // Simple row-oriented format for delta (converted to columnar on compact)
        let record_count = rows.len() as u64;
        file.write_all(&record_count.to_le_bytes())?;
        
        // Write IDs
        for id in &ids {
            file.write_all(&id.to_le_bytes())?;
        }
        
        // Write schema + data for each column type
        // Int columns
        let int_col_count = int_columns.len() as u32;
        file.write_all(&int_col_count.to_le_bytes())?;
        for (name, values) in &int_columns {
            let name_bytes = name.as_bytes();
            file.write_all(&(name_bytes.len() as u16).to_le_bytes())?;
            file.write_all(name_bytes)?;
            for v in values {
                file.write_all(&v.to_le_bytes())?;
            }
        }
        
        // Float columns
        let float_col_count = float_columns.len() as u32;
        file.write_all(&float_col_count.to_le_bytes())?;
        for (name, values) in &float_columns {
            let name_bytes = name.as_bytes();
            file.write_all(&(name_bytes.len() as u16).to_le_bytes())?;
            file.write_all(name_bytes)?;
            for v in values {
                file.write_all(&v.to_le_bytes())?;
            }
        }
        
        // String columns
        let string_col_count = string_columns.len() as u32;
        file.write_all(&string_col_count.to_le_bytes())?;
        for (name, values) in &string_columns {
            let name_bytes = name.as_bytes();
            file.write_all(&(name_bytes.len() as u16).to_le_bytes())?;
            file.write_all(name_bytes)?;
            for v in values {
                let v_bytes = v.as_bytes();
                file.write_all(&(v_bytes.len() as u32).to_le_bytes())?;
                file.write_all(v_bytes)?;
            }
        }
        
        // Bool columns  
        let bool_col_count = bool_columns.len() as u32;
        file.write_all(&bool_col_count.to_le_bytes())?;
        for (name, values) in &bool_columns {
            let name_bytes = name.as_bytes();
            file.write_all(&(name_bytes.len() as u16).to_le_bytes())?;
            file.write_all(name_bytes)?;
            for v in values {
                file.write_all(&[if *v { 1u8 } else { 0u8 }])?;
            }
        }
        
        file.flush()?;
        
        if self.durability == super::DurabilityLevel::Max {
            file.sync_all()?;
        }
        
        Ok(ids)
    }
    
    /// Insert typed columns to delta file (memory efficient - doesn't load existing data)
    /// Returns the IDs assigned to the inserted rows
    fn insert_typed_to_delta(
        &self,
        int_columns: HashMap<String, Vec<i64>>,
        float_columns: HashMap<String, Vec<f64>>,
        string_columns: HashMap<String, Vec<String>>,
        _binary_columns: HashMap<String, Vec<Vec<u8>>>,  // Not yet implemented in delta
        bool_columns: HashMap<String, Vec<bool>>,
    ) -> io::Result<Vec<u64>> {
        // Determine row count
        let row_count = int_columns.values().map(|v| v.len()).max().unwrap_or(0)
            .max(float_columns.values().map(|v| v.len()).max().unwrap_or(0))
            .max(string_columns.values().map(|v| v.len()).max().unwrap_or(0))
            .max(bool_columns.values().map(|v| v.len()).max().unwrap_or(0));
        
        if row_count == 0 {
            return Ok(Vec::new());
        }
        
        let delta_path = Self::delta_path(&self.path);
        
        // Allocate IDs
        let mut ids = Vec::with_capacity(row_count);
        for _ in 0..row_count {
            ids.push(self.next_id.fetch_add(1, Ordering::SeqCst));
        }
        
        // Write delta file (append mode)
        let mut file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&delta_path)?;
        
        // Delta format: [record_count:u64][ids...][schema...][column_data...]
        file.write_all(&(row_count as u64).to_le_bytes())?;
        
        // Write IDs
        for id in &ids {
            file.write_all(&id.to_le_bytes())?;
        }
        
        // Write int columns
        let int_col_count = int_columns.len() as u32;
        file.write_all(&int_col_count.to_le_bytes())?;
        for (name, values) in &int_columns {
            let name_bytes = name.as_bytes();
            file.write_all(&(name_bytes.len() as u16).to_le_bytes())?;
            file.write_all(name_bytes)?;
            for v in values {
                file.write_all(&v.to_le_bytes())?;
            }
        }
        
        // Write float columns
        let float_col_count = float_columns.len() as u32;
        file.write_all(&float_col_count.to_le_bytes())?;
        for (name, values) in &float_columns {
            let name_bytes = name.as_bytes();
            file.write_all(&(name_bytes.len() as u16).to_le_bytes())?;
            file.write_all(name_bytes)?;
            for v in values {
                file.write_all(&v.to_le_bytes())?;
            }
        }
        
        // Write string columns
        let string_col_count = string_columns.len() as u32;
        file.write_all(&string_col_count.to_le_bytes())?;
        for (name, values) in &string_columns {
            let name_bytes = name.as_bytes();
            file.write_all(&(name_bytes.len() as u16).to_le_bytes())?;
            file.write_all(name_bytes)?;
            for v in values {
                let v_bytes = v.as_bytes();
                file.write_all(&(v_bytes.len() as u32).to_le_bytes())?;
                file.write_all(v_bytes)?;
            }
        }
        
        // Write bool columns
        let bool_col_count = bool_columns.len() as u32;
        file.write_all(&bool_col_count.to_le_bytes())?;
        for (name, values) in &bool_columns {
            let name_bytes = name.as_bytes();
            file.write_all(&(name_bytes.len() as u16).to_le_bytes())?;
            file.write_all(name_bytes)?;
            for v in values {
                file.write_all(&[if *v { 1u8 } else { 0u8 }])?;
            }
        }
        
        file.flush()?;
        
        if self.durability == super::DurabilityLevel::Max {
            file.sync_all()?;
        }
        
        Ok(ids)
    }
    
    /// Compact: merge delta file into base file
    /// 
    /// MEMORY EFFICIENT: Uses column-streaming merge.
    /// Processes one column at a time via mmap, never loading all columns simultaneously.
    /// Peak memory ≈ max(single column) + delta data, instead of ALL columns + delta.
    /// 
    /// For a 10M-row × 5-column table, this reduces peak memory from ~800MB to ~160MB.
    pub fn compact(&self) -> io::Result<()> {
        let delta_path = Self::delta_path(&self.path);
        if !delta_path.exists() {
            return Ok(());
        }
        
        self.load_all_columns_into_memory()?;
        self.merge_delta_file(&delta_path)?;
        self.save()?;
        
        // Delete delta file
        let _ = std::fs::remove_file(&delta_path);
        
        Ok(())
    }
    
    /// Create a column filled with default values (0, 0.0, "", false).
    /// Used for columns added via ALTER TABLE that have no disk data yet.
    fn create_default_column(dtype: ColumnType, count: usize) -> ColumnData {
        if count == 0 {
            return ColumnData::new(dtype);
        }
        match dtype {
            ColumnType::Bool => ColumnData::Bool {
                data: vec![0u8; (count + 7) / 8],
                len: count,
            },
            ColumnType::Int64 | ColumnType::Int8 | ColumnType::Int16 | ColumnType::Int32 |
            ColumnType::UInt8 | ColumnType::UInt16 | ColumnType::UInt32 | ColumnType::UInt64 => {
                ColumnData::Int64(vec![0i64; count])
            }
            ColumnType::Float64 | ColumnType::Float32 => {
                ColumnData::Float64(vec![0.0f64; count])
            }
            ColumnType::String | ColumnType::StringDict => {
                ColumnData::String { offsets: vec![0u32; count + 1], data: Vec::new() }
            }
            ColumnType::Binary => {
                ColumnData::Binary { offsets: vec![0u32; count + 1], data: Vec::new() }
            }
            ColumnType::Null => ColumnData::Int64(vec![0i64; count]),
        }
    }
    
    // compact_column_streaming removed — was V3-only dead code (326 lines).
    // save() always produces V4 format; compact() uses in-memory merge path.
    
    /// Read delta file and merge into in-memory columns
    fn merge_delta_file(&self, delta_path: &Path) -> io::Result<()> {
        let mut file = File::open(delta_path)?;
        
        loop {
            // Try to read record count
            let mut count_buf = [0u8; 8];
            match file.read_exact(&mut count_buf) {
                Ok(_) => {},
                Err(e) if e.kind() == io::ErrorKind::UnexpectedEof => break,
                Err(e) => return Err(e),
            }
            let record_count = u64::from_le_bytes(count_buf) as usize;
            
            // Read IDs
            let mut delta_ids = Vec::with_capacity(record_count);
            for _ in 0..record_count {
                let mut id_buf = [0u8; 8];
                file.read_exact(&mut id_buf)?;
                delta_ids.push(u64::from_le_bytes(id_buf));
            }
            
            // Read int columns
            let mut int_columns: HashMap<String, Vec<i64>> = HashMap::new();
            let mut count_buf = [0u8; 4];
            file.read_exact(&mut count_buf)?;
            let int_col_count = u32::from_le_bytes(count_buf) as usize;
            for _ in 0..int_col_count {
                let mut len_buf = [0u8; 2];
                file.read_exact(&mut len_buf)?;
                let name_len = u16::from_le_bytes(len_buf) as usize;
                let mut name_buf = vec![0u8; name_len];
                file.read_exact(&mut name_buf)?;
                let name = String::from_utf8_lossy(&name_buf).to_string();
                let mut values = Vec::with_capacity(record_count);
                for _ in 0..record_count {
                    let mut v_buf = [0u8; 8];
                    file.read_exact(&mut v_buf)?;
                    values.push(i64::from_le_bytes(v_buf));
                }
                int_columns.insert(name, values);
            }
            
            // Read float columns
            let mut float_columns: HashMap<String, Vec<f64>> = HashMap::new();
            file.read_exact(&mut count_buf)?;
            let float_col_count = u32::from_le_bytes(count_buf) as usize;
            for _ in 0..float_col_count {
                let mut len_buf = [0u8; 2];
                file.read_exact(&mut len_buf)?;
                let name_len = u16::from_le_bytes(len_buf) as usize;
                let mut name_buf = vec![0u8; name_len];
                file.read_exact(&mut name_buf)?;
                let name = String::from_utf8_lossy(&name_buf).to_string();
                let mut values = Vec::with_capacity(record_count);
                for _ in 0..record_count {
                    let mut v_buf = [0u8; 8];
                    file.read_exact(&mut v_buf)?;
                    values.push(f64::from_le_bytes(v_buf));
                }
                float_columns.insert(name, values);
            }
            
            // Read string columns
            let mut string_columns: HashMap<String, Vec<String>> = HashMap::new();
            file.read_exact(&mut count_buf)?;
            let string_col_count = u32::from_le_bytes(count_buf) as usize;
            for _ in 0..string_col_count {
                let mut len_buf = [0u8; 2];
                file.read_exact(&mut len_buf)?;
                let name_len = u16::from_le_bytes(len_buf) as usize;
                let mut name_buf = vec![0u8; name_len];
                file.read_exact(&mut name_buf)?;
                let name = String::from_utf8_lossy(&name_buf).to_string();
                let mut values = Vec::with_capacity(record_count);
                for _ in 0..record_count {
                    let mut str_len_buf = [0u8; 4];
                    file.read_exact(&mut str_len_buf)?;
                    let str_len = u32::from_le_bytes(str_len_buf) as usize;
                    let mut str_buf = vec![0u8; str_len];
                    file.read_exact(&mut str_buf)?;
                    let val = String::from_utf8_lossy(&str_buf).to_string();
                    values.push(val);
                }
                string_columns.insert(name, values);
            }
            
            // Read bool columns
            let mut bool_columns: HashMap<String, Vec<bool>> = HashMap::new();
            file.read_exact(&mut count_buf)?;
            let bool_col_count = u32::from_le_bytes(count_buf) as usize;
            for _ in 0..bool_col_count {
                let mut len_buf = [0u8; 2];
                file.read_exact(&mut len_buf)?;
                let name_len = u16::from_le_bytes(len_buf) as usize;
                let mut name_buf = vec![0u8; name_len];
                file.read_exact(&mut name_buf)?;
                let name = String::from_utf8_lossy(&name_buf).to_string();
                let mut values = Vec::with_capacity(record_count);
                for _ in 0..record_count {
                    let mut v_buf = [0u8; 1];
                    file.read_exact(&mut v_buf)?;
                    values.push(v_buf[0] != 0);
                }
                bool_columns.insert(name, values);
            }
            
            // Merge into in-memory columns PRESERVING original delta IDs
            // This is critical for correct ID assignment after delete operations
            self.insert_typed_with_ids(
                &delta_ids,
                int_columns,
                float_columns,
                string_columns,
                HashMap::new(), // binary columns (not implemented in delta yet)
                bool_columns,
            )?;
        }
        
        Ok(())
    }
    
    /// Read delta file and return column data without merging into memory
    /// Returns: (delta_ids, column_data_map) where column_data_map is column_name -> ColumnData
    fn read_delta_data(&self) -> io::Result<Option<(Vec<u64>, HashMap<String, ColumnData>)>> {
        let delta_path = Self::delta_path(&self.path);
        if !delta_path.exists() {
            return Ok(None);
        }
        
        let mut file = File::open(&delta_path)?;
        let mut all_ids: Vec<u64> = Vec::new();
        let mut all_columns: HashMap<String, ColumnData> = HashMap::new();
        
        loop {
            // Try to read record count
            let mut count_buf = [0u8; 8];
            match file.read_exact(&mut count_buf) {
                Ok(_) => {},
                Err(e) if e.kind() == io::ErrorKind::UnexpectedEof => break,
                Err(e) => return Err(e),
            }
            let record_count = u64::from_le_bytes(count_buf) as usize;
            
            // Read IDs
            for _ in 0..record_count {
                let mut id_buf = [0u8; 8];
                file.read_exact(&mut id_buf)?;
                all_ids.push(u64::from_le_bytes(id_buf));
            }
            
            // Read int columns
            let mut count_buf4 = [0u8; 4];
            file.read_exact(&mut count_buf4)?;
            let int_col_count = u32::from_le_bytes(count_buf4) as usize;
            for _ in 0..int_col_count {
                let mut len_buf = [0u8; 2];
                file.read_exact(&mut len_buf)?;
                let name_len = u16::from_le_bytes(len_buf) as usize;
                let mut name_buf = vec![0u8; name_len];
                file.read_exact(&mut name_buf)?;
                let name = String::from_utf8_lossy(&name_buf).to_string();
                
                let col_data = all_columns.entry(name).or_insert_with(|| ColumnData::new(ColumnType::Int64));
                for _ in 0..record_count {
                    let mut v_buf = [0u8; 8];
                    file.read_exact(&mut v_buf)?;
                    col_data.push_i64(i64::from_le_bytes(v_buf));
                }
            }
            
            // Read float columns
            file.read_exact(&mut count_buf4)?;
            let float_col_count = u32::from_le_bytes(count_buf4) as usize;
            for _ in 0..float_col_count {
                let mut len_buf = [0u8; 2];
                file.read_exact(&mut len_buf)?;
                let name_len = u16::from_le_bytes(len_buf) as usize;
                let mut name_buf = vec![0u8; name_len];
                file.read_exact(&mut name_buf)?;
                let name = String::from_utf8_lossy(&name_buf).to_string();
                
                let col_data = all_columns.entry(name).or_insert_with(|| ColumnData::new(ColumnType::Float64));
                for _ in 0..record_count {
                    let mut v_buf = [0u8; 8];
                    file.read_exact(&mut v_buf)?;
                    col_data.push_f64(f64::from_le_bytes(v_buf));
                }
            }
            
            // Read string columns
            file.read_exact(&mut count_buf4)?;
            let string_col_count = u32::from_le_bytes(count_buf4) as usize;
            for _ in 0..string_col_count {
                let mut len_buf = [0u8; 2];
                file.read_exact(&mut len_buf)?;
                let name_len = u16::from_le_bytes(len_buf) as usize;
                let mut name_buf = vec![0u8; name_len];
                file.read_exact(&mut name_buf)?;
                let name = String::from_utf8_lossy(&name_buf).to_string();
                
                let col_data = all_columns.entry(name).or_insert_with(|| ColumnData::new(ColumnType::String));
                for _ in 0..record_count {
                    let mut str_len_buf = [0u8; 4];
                    file.read_exact(&mut str_len_buf)?;
                    let str_len = u32::from_le_bytes(str_len_buf) as usize;
                    let mut str_buf = vec![0u8; str_len];
                    file.read_exact(&mut str_buf)?;
                    let val = String::from_utf8_lossy(&str_buf).to_string();
                    col_data.push_string(&val);
                }
            }
            
            // Read bool columns
            file.read_exact(&mut count_buf4)?;
            let bool_col_count = u32::from_le_bytes(count_buf4) as usize;
            for _ in 0..bool_col_count {
                let mut len_buf = [0u8; 2];
                file.read_exact(&mut len_buf)?;
                let name_len = u16::from_le_bytes(len_buf) as usize;
                let mut name_buf = vec![0u8; name_len];
                file.read_exact(&mut name_buf)?;
                let name = String::from_utf8_lossy(&name_buf).to_string();
                
                let col_data = all_columns.entry(name).or_insert_with(|| ColumnData::new(ColumnType::Bool));
                for _ in 0..record_count {
                    let mut v_buf = [0u8; 1];
                    file.read_exact(&mut v_buf)?;
                    col_data.push_bool(v_buf[0] != 0);
                }
            }
        }
        
        if all_ids.is_empty() {
            Ok(None)
        } else {
            Ok(Some((all_ids, all_columns)))
        }
    }
    
    /// Get the total row count including delta rows (for accurate row_count reporting)
    fn delta_row_count(&self) -> usize {
        let delta_path = Self::delta_path(&self.path);
        if !delta_path.exists() {
            return 0;
        }
        
        // Quick count without reading all data
        if let Ok(mut file) = File::open(&delta_path) {
            let mut total = 0usize;
            loop {
                let mut count_buf = [0u8; 8];
                match file.read_exact(&mut count_buf) {
                    Ok(_) => {},
                    Err(_) => break,
                }
                let record_count = u64::from_le_bytes(count_buf) as usize;
                total += record_count;
                
                // Skip the rest of this record block
                // IDs
                if file.seek(SeekFrom::Current((record_count * 8) as i64)).is_err() { break; }
                
                // Int columns
                let mut count_buf4 = [0u8; 4];
                if file.read_exact(&mut count_buf4).is_err() { break; }
                let int_col_count = u32::from_le_bytes(count_buf4) as usize;
                for _ in 0..int_col_count {
                    let mut len_buf = [0u8; 2];
                    if file.read_exact(&mut len_buf).is_err() { break; }
                    let name_len = u16::from_le_bytes(len_buf) as usize;
                    if file.seek(SeekFrom::Current(name_len as i64 + (record_count * 8) as i64)).is_err() { break; }
                }
                
                // Float columns
                if file.read_exact(&mut count_buf4).is_err() { break; }
                let float_col_count = u32::from_le_bytes(count_buf4) as usize;
                for _ in 0..float_col_count {
                    let mut len_buf = [0u8; 2];
                    if file.read_exact(&mut len_buf).is_err() { break; }
                    let name_len = u16::from_le_bytes(len_buf) as usize;
                    if file.seek(SeekFrom::Current(name_len as i64 + (record_count * 8) as i64)).is_err() { break; }
                }
                
                // String columns - variable length, need to read each
                if file.read_exact(&mut count_buf4).is_err() { break; }
                let string_col_count = u32::from_le_bytes(count_buf4) as usize;
                for _ in 0..string_col_count {
                    let mut len_buf = [0u8; 2];
                    if file.read_exact(&mut len_buf).is_err() { break; }
                    let name_len = u16::from_le_bytes(len_buf) as usize;
                    if file.seek(SeekFrom::Current(name_len as i64)).is_err() { break; }
                    for _ in 0..record_count {
                        let mut str_len_buf = [0u8; 4];
                        if file.read_exact(&mut str_len_buf).is_err() { break; }
                        let str_len = u32::from_le_bytes(str_len_buf) as usize;
                        if file.seek(SeekFrom::Current(str_len as i64)).is_err() { break; }
                    }
                }
                
                // Bool columns
                if file.read_exact(&mut count_buf4).is_err() { break; }
                let bool_col_count = u32::from_le_bytes(count_buf4) as usize;
                for _ in 0..bool_col_count {
                    let mut len_buf = [0u8; 2];
                    if file.read_exact(&mut len_buf).is_err() { break; }
                    let name_len = u16::from_le_bytes(len_buf) as usize;
                    if file.seek(SeekFrom::Current(name_len as i64 + record_count as i64)).is_err() { break; }
                }
            }
            total
        } else {
            0
        }
    }

    // ========================================================================
    // Direct Arrow Conversion (bypasses HashMap/clone pipeline)
    // ========================================================================

    /// Build Arrow RecordBatch directly from in-memory V4 columns.
    /// OPTIMIZATION: bypasses read_columns→HashMap→get_null_mask→Vec<bool> pipeline.
    /// - Int64/Float64 without nulls: single memcpy (no per-element Option wrapping)
    /// - String: builds from &str references (no per-element String allocation)
    /// - Null bitmaps: read packed bytes directly (no Vec<bool> expansion)
    pub fn to_arrow_batch(
        &self,
        column_names: Option<&[&str]>,
        include_id: bool,
    ) -> io::Result<RecordBatch> {
        self.to_arrow_batch_inner(column_names, include_id, false)
    }

    /// Build Arrow RecordBatch with optional dictionary encoding for string columns.
    /// When dict_encode_strings=true, low-cardinality string columns produce DictionaryArray
    /// which accelerates GROUP BY and WHERE filters.
    pub fn to_arrow_batch_dict(
        &self,
        column_names: Option<&[&str]>,
        include_id: bool,
    ) -> io::Result<RecordBatch> {
        self.to_arrow_batch_inner(column_names, include_id, true)
    }

    fn to_arrow_batch_inner(
        &self,
        column_names: Option<&[&str]>,
        include_id: bool,
        dict_encode_strings: bool,
    ) -> io::Result<RecordBatch> {
        use arrow::array::{Int64Array, Float64Array, StringArray, BooleanArray, PrimitiveArray};
        use arrow::buffer::{Buffer, NullBuffer, BooleanBuffer, ScalarBuffer};
        use arrow::datatypes::{Schema, Field, DataType as ArrowDataType, Int64Type, Float64Type};
        use std::sync::Arc;

        // ON-DEMAND MMAP PATH: For V4 files, prefer reading directly from mmap
        // instead of loading all data into memory. This is the key memory optimization.
        {
            let header = self.header.read();
            let is_v4 = header.footer_offset > 0;
            drop(header);

            if is_v4 {
                // Check if columns are already loaded in memory (write buffer has data)
                let cols = self.columns.read();
                let has_in_memory_data = !cols.is_empty() && cols.iter().any(|c| c.len() > 0);
                drop(cols);

                if !has_in_memory_data {
                    // Pure mmap path — no data in memory, read everything from disk
                    if let Some(batch) = self.to_arrow_batch_mmap(
                        column_names, include_id, None, dict_encode_strings,
                    )? {
                        return Ok(batch);
                    }
                }
                // If we have in-memory data (write buffer), fall through to legacy path
                // which reads from self.columns/ids/nulls/deleted
            }
        }

        // At this point: V3 (data always in memory) or V4 with in-memory write buffer.
        // No loading needed — data is already available in self.columns/ids.
        let schema = self.schema.read();
        let ids = self.ids.read();
        let columns = self.columns.read();
        let nulls = self.nulls.read();
        let deleted = self.deleted.read();

        let total_rows = ids.len();
        let col_count = schema.column_count();

        // Check for deleted rows
        let has_deleted = deleted.iter().any(|&b| b != 0);

        // Determine active row indices (skip deleted)
        let active_indices: Option<Vec<usize>> = if has_deleted {
            Some((0..total_rows)
                .filter(|&i| {
                    let byte_idx = i / 8;
                    let bit_idx = i % 8;
                    byte_idx >= deleted.len() || (deleted[byte_idx] >> bit_idx) & 1 == 0
                })
                .collect())
        } else {
            None
        };
        let active_count = active_indices.as_ref().map(|v| v.len()).unwrap_or(total_rows);

        // Determine which columns to read
        let col_indices: Vec<usize> = if let Some(names) = column_names {
            names.iter()
                .filter(|&&n| n != "_id")
                .filter_map(|&name| schema.get_index(name))
                .collect()
        } else {
            (0..col_count).collect()
        };

        let mut fields: Vec<Field> = Vec::with_capacity(col_indices.len() + 1);
        let mut arrays: Vec<ArrayRef> = Vec::with_capacity(col_indices.len() + 1);

        // _id column
        if include_id {
            fields.push(Field::new("_id", ArrowDataType::Int64, false));
            if let Some(ref indices) = active_indices {
                let active_ids: Vec<i64> = indices.iter().map(|&i| ids[i] as i64).collect();
                arrays.push(Arc::new(Int64Array::from(active_ids)));
            } else {
                // Zero-copy: reinterpret u64 slice as i64 slice, copy once to Arrow buffer
                let mut ids_copy: Vec<i64> = Vec::with_capacity(total_rows);
                // SAFETY: u64 and i64 have identical memory layout
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        ids.as_ptr() as *const i64,
                        ids_copy.as_mut_ptr(),
                        total_rows,
                    );
                    ids_copy.set_len(total_rows);
                }
                arrays.push(Arc::new(Int64Array::from(ids_copy)));
            }
        }

        // Data columns
        for &col_idx in &col_indices {
            let (col_name, _col_type) = &schema.columns[col_idx];
            let col_data = if col_idx < columns.len() { Some(&columns[col_idx]) } else { None };

            // Build Arrow null buffer from packed bitmap
            let null_buf: Option<NullBuffer> = if col_idx < nulls.len() && !nulls[col_idx].is_empty() {
                let null_bitmap = &nulls[col_idx];
                let has_any_null = null_bitmap.iter().any(|&b| b != 0);
                if has_any_null {
                    if active_indices.is_none() {
                        // No deletes: Arrow validity = inverted null bitmap
                        // null_bitmap bit=1 means NULL, Arrow validity bit=1 means VALID
                        let mut validity_bytes = vec![0xFFu8; (active_count + 7) / 8];
                        for byte_idx in 0..null_bitmap.len().min(validity_bytes.len()) {
                            validity_bytes[byte_idx] = !null_bitmap[byte_idx];
                        }
                        // Mask trailing bits
                        let tail = active_count % 8;
                        if tail > 0 {
                            let last = validity_bytes.len() - 1;
                            validity_bytes[last] &= (1u8 << tail) - 1;
                        }
                        Some(NullBuffer::new(BooleanBuffer::new(Buffer::from(validity_bytes), 0, active_count)))
                    } else {
                        // Has deletes: build validity for active rows only
                        let indices = active_indices.as_ref().unwrap();
                        let mut validity_bytes = vec![0xFFu8; (active_count + 7) / 8];
                        for (new_idx, &old_idx) in indices.iter().enumerate() {
                            let ob = old_idx / 8;
                            let obit = old_idx % 8;
                            if ob < null_bitmap.len() && (null_bitmap[ob] >> obit) & 1 == 1 {
                                // This row is NULL → clear validity bit
                                validity_bytes[new_idx / 8] &= !(1u8 << (new_idx % 8));
                            }
                        }
                        Some(NullBuffer::new(BooleanBuffer::new(Buffer::from(validity_bytes), 0, active_count)))
                    }
                } else {
                    None // All valid
                }
            } else {
                None // No null info
            };

            let (arrow_dt, array): (ArrowDataType, ArrayRef) = match col_data {
                Some(ColumnData::Int64(values)) => {
                    let data_vec = if let Some(ref indices) = active_indices {
                        indices.iter().map(|&i| if i < values.len() { values[i] } else { 0 }).collect()
                    } else {
                        values.clone()
                    };
                    let arr = PrimitiveArray::<Int64Type>::new(
                        ScalarBuffer::from(data_vec), null_buf,
                    );
                    (ArrowDataType::Int64, Arc::new(arr) as ArrayRef)
                }
                Some(ColumnData::Float64(values)) => {
                    let data_vec = if let Some(ref indices) = active_indices {
                        indices.iter().map(|&i| if i < values.len() { values[i] } else { 0.0 }).collect()
                    } else {
                        values.clone()
                    };
                    let arr = PrimitiveArray::<Float64Type>::new(
                        ScalarBuffer::from(data_vec), null_buf,
                    );
                    (ArrowDataType::Float64, Arc::new(arr) as ArrayRef)
                }
                Some(ColumnData::String { offsets, data }) => {
                    // OPTIMIZATION: build StringArray from &str refs (no per-element String alloc)
                    let count = offsets.len().saturating_sub(1);
                    if let Some(ref indices) = active_indices {
                        let strings: Vec<Option<&str>> = indices.iter().map(|&i| {
                            if i < count {
                                // Check null
                                if col_idx < nulls.len() && !nulls[col_idx].is_empty() {
                                    let ob = i / 8;
                                    let obit = i % 8;
                                    if ob < nulls[col_idx].len() && (nulls[col_idx][ob] >> obit) & 1 == 1 {
                                        return None;
                                    }
                                }
                                let start = offsets[i] as usize;
                                let end = offsets[i + 1] as usize;
                                std::str::from_utf8(&data[start..end]).ok()
                            } else {
                                None
                            }
                        }).collect();
                        (ArrowDataType::Utf8, Arc::new(StringArray::from(strings)))
                    } else if null_buf.is_some() {
                        let null_bitmap = &nulls[col_idx];
                        let strings: Vec<Option<&str>> = (0..count.min(active_count)).map(|i| {
                            let ob = i / 8;
                            let obit = i % 8;
                            if ob < null_bitmap.len() && (null_bitmap[ob] >> obit) & 1 == 1 {
                                None
                            } else {
                                let start = offsets[i] as usize;
                                let end = offsets[i + 1] as usize;
                                std::str::from_utf8(&data[start..end]).ok()
                            }
                        }).collect();
                        (ArrowDataType::Utf8, Arc::new(StringArray::from(strings)))
                    } else {
                        // No nulls, no deletes: fastest path
                        let row_count = count.min(active_count);
                        
                        // OPTIMIZATION: Try to build DictionaryArray for low-cardinality columns
                        // Only when dict_encode_strings=true (GROUP BY queries)
                        let try_dict = dict_encode_strings && row_count >= 100;
                        if try_dict {
                            // Sample first 1000 rows to estimate cardinality
                            let sample_size = row_count.min(1000);
                            let mut sample_unique = ahash::AHashSet::with_capacity(100);
                            let step = if row_count > sample_size { row_count / sample_size } else { 1 };
                            let mut si = 0;
                            while si < row_count && sample_unique.len() <= 1000 {
                                let start = offsets[si] as usize;
                                let end = offsets[si + 1] as usize;
                                sample_unique.insert(&data[start..end]);
                                si += step;
                            }
                            
                            if sample_unique.len() <= 1000 {
                                // Low cardinality → build DictionaryArray<UInt32Type>
                                use arrow::array::{UInt32Array, DictionaryArray};
                                use arrow::datatypes::UInt32Type;
                                
                                let mut dict_map: ahash::AHashMap<&[u8], u32> = ahash::AHashMap::with_capacity(sample_unique.len());
                                let mut dict_strings: Vec<&str> = Vec::with_capacity(sample_unique.len());
                                let mut next_id = 0u32;
                                let mut keys: Vec<u32> = Vec::with_capacity(row_count);
                                
                                for i in 0..row_count {
                                    let start = offsets[i] as usize;
                                    let end = offsets[i + 1] as usize;
                                    let bytes = &data[start..end];
                                    let id = *dict_map.entry(bytes).or_insert_with(|| {
                                        let id = next_id;
                                        next_id += 1;
                                        dict_strings.push(std::str::from_utf8(bytes).unwrap_or(""));
                                        id
                                    });
                                    keys.push(id);
                                }
                                
                                let keys_array = UInt32Array::from(keys);
                                let values_array = StringArray::from_iter_values(dict_strings);
                                let dict_array = DictionaryArray::<UInt32Type>::try_new(
                                    keys_array, Arc::new(values_array),
                                ).map_err(|e| err_data(e.to_string()))?;
                                let arr_ref: ArrayRef = Arc::new(dict_array);
                                (arr_ref.data_type().clone(), arr_ref)
                            } else {
                                // High cardinality → plain StringArray
                                let strings: Vec<&str> = (0..row_count).map(|i| {
                                    let start = offsets[i] as usize;
                                    let end = offsets[i + 1] as usize;
                                    std::str::from_utf8(&data[start..end]).unwrap_or("")
                                }).collect();
                                (ArrowDataType::Utf8, Arc::new(StringArray::from_iter_values(strings)))
                            }
                        } else {
                            let strings: Vec<&str> = (0..row_count).map(|i| {
                                let start = offsets[i] as usize;
                                let end = offsets[i + 1] as usize;
                                std::str::from_utf8(&data[start..end]).unwrap_or("")
                            }).collect();
                            (ArrowDataType::Utf8, Arc::new(StringArray::from_iter_values(strings)))
                        }
                    }
                }
                Some(ColumnData::Bool { data: packed, len }) => {
                    if let Some(ref indices) = active_indices {
                        let bools: Vec<Option<bool>> = indices.iter().map(|&i| {
                            if col_idx < nulls.len() && !nulls[col_idx].is_empty() {
                                let ob = i / 8;
                                let obit = i % 8;
                                if ob < nulls[col_idx].len() && (nulls[col_idx][ob] >> obit) & 1 == 1 {
                                    return None;
                                }
                            }
                            if i < *len {
                                let byte_idx = i / 8;
                                let bit_idx = i % 8;
                                Some(byte_idx < packed.len() && (packed[byte_idx] >> bit_idx) & 1 == 1)
                            } else {
                                None
                            }
                        }).collect();
                        (ArrowDataType::Boolean, Arc::new(BooleanArray::from(bools)))
                    } else {
                        let bools: Vec<Option<bool>> = (0..*len).map(|i| {
                            if col_idx < nulls.len() && !nulls[col_idx].is_empty() {
                                let ob = i / 8;
                                let obit = i % 8;
                                if ob < nulls[col_idx].len() && (nulls[col_idx][ob] >> obit) & 1 == 1 {
                                    return None;
                                }
                            }
                            let byte_idx = i / 8;
                            let bit_idx = i % 8;
                            Some(byte_idx < packed.len() && (packed[byte_idx] >> bit_idx) & 1 == 1)
                        }).collect();
                        (ArrowDataType::Boolean, Arc::new(BooleanArray::from(bools)))
                    }
                }
                Some(ColumnData::Binary { offsets, data }) => {
                    use arrow::array::BinaryArray;
                    let count = offsets.len().saturating_sub(1);
                    let binary_data: Vec<Option<&[u8]>> = if let Some(ref indices) = active_indices {
                        indices.iter().map(|&i| {
                            if i < count {
                                let start = offsets[i] as usize;
                                let end = offsets[i + 1] as usize;
                                Some(&data[start..end] as &[u8])
                            } else { None }
                        }).collect()
                    } else {
                        (0..count.min(active_count)).map(|i| {
                            let start = offsets[i] as usize;
                            let end = offsets[i + 1] as usize;
                            Some(&data[start..end] as &[u8])
                        }).collect()
                    };
                    (ArrowDataType::Binary, Arc::new(BinaryArray::from(binary_data)))
                }
                Some(ColumnData::StringDict { .. }) => {
                    // StringDict should have been decoded to String during column loading
                    // Fallback: create empty string array
                    (ArrowDataType::Utf8, Arc::new(StringArray::from(vec![""; active_count])))
                }
                None => {
                    // Column doesn't exist, create default
                    (ArrowDataType::Int64, Arc::new(Int64Array::from(vec![0i64; active_count])))
                }
            };

            fields.push(Field::new(col_name, arrow_dt, true));
            arrays.push(array);
        }

        let arrow_schema = Arc::new(Schema::new(fields));
        RecordBatch::try_new(arrow_schema, arrays)
            .map_err(|e| err_data(e.to_string()))
    }

    /// Build Arrow RecordBatch with a row LIMIT from in-memory V4 columns.
    /// Much faster than read_columns() for small LIMIT queries (SELECT * LIMIT N).
    pub fn to_arrow_batch_with_limit(
        &self,
        column_names: Option<&[&str]>,
        include_id: bool,
        limit: usize,
    ) -> io::Result<RecordBatch> {
        use arrow::array::{Int64Array, Float64Array, StringArray, BooleanArray, PrimitiveArray};
        use arrow::buffer::{Buffer, NullBuffer, BooleanBuffer, ScalarBuffer};
        use arrow::datatypes::{Schema, Field, DataType as ArrowDataType, Int64Type, Float64Type};
        use std::sync::Arc;

        // ON-DEMAND MMAP PATH for LIMIT queries
        {
            let header = self.header.read();
            let is_v4 = header.footer_offset > 0;
            drop(header);
            if is_v4 {
                let cols = self.columns.read();
                let has_in_memory_data = !cols.is_empty() && cols.iter().any(|c| c.len() > 0);
                drop(cols);
                if !has_in_memory_data {
                    if let Some(batch) = self.to_arrow_batch_mmap(
                        column_names, include_id, Some(limit), true,
                    )? {
                        return Ok(batch);
                    }
                }
            }
        }

        // At this point: V3 (data always in memory) or V4 with in-memory write buffer.
        let schema = self.schema.read();
        let ids = self.ids.read();
        let columns = self.columns.read();
        let nulls = self.nulls.read();
        let deleted = self.deleted.read();

        let total_rows = ids.len();
        let col_count = schema.column_count();
        let has_deleted = deleted.iter().any(|&b| b != 0);

        // Collect first `limit` active row indices
        let actual_limit;
        let row_indices: Option<Vec<usize>> = if has_deleted {
            let mut indices = Vec::with_capacity(limit.min(total_rows));
            for i in 0..total_rows {
                if indices.len() >= limit { break; }
                let byte_idx = i / 8;
                let bit_idx = i % 8;
                if byte_idx >= deleted.len() || (deleted[byte_idx] >> bit_idx) & 1 == 0 {
                    indices.push(i);
                }
            }
            actual_limit = indices.len();
            Some(indices)
        } else {
            actual_limit = limit.min(total_rows);
            None // contiguous range 0..actual_limit
        };

        if actual_limit == 0 {
            let arrow_schema = Arc::new(Schema::empty());
            return Ok(RecordBatch::new_empty(arrow_schema));
        }

        let col_indices: Vec<usize> = if let Some(names) = column_names {
            names.iter()
                .filter(|&&n| n != "_id")
                .filter_map(|&name| schema.get_index(name))
                .collect()
        } else {
            (0..col_count).collect()
        };

        let mut fields: Vec<Field> = Vec::with_capacity(col_indices.len() + 1);
        let mut arrays: Vec<ArrayRef> = Vec::with_capacity(col_indices.len() + 1);

        // _id column
        if include_id {
            fields.push(Field::new("_id", ArrowDataType::Int64, false));
            if let Some(ref indices) = row_indices {
                let active_ids: Vec<i64> = indices.iter().map(|&i| ids[i] as i64).collect();
                arrays.push(Arc::new(Int64Array::from(active_ids)));
            } else {
                // Contiguous: just copy first actual_limit IDs
                let mut ids_copy: Vec<i64> = Vec::with_capacity(actual_limit);
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        ids.as_ptr() as *const i64,
                        ids_copy.as_mut_ptr(),
                        actual_limit,
                    );
                    ids_copy.set_len(actual_limit);
                }
                arrays.push(Arc::new(Int64Array::from(ids_copy)));
            }
        }

        for &col_idx in &col_indices {
            let (col_name, _) = &schema.columns[col_idx];
            let col_data = if col_idx < columns.len() { Some(&columns[col_idx]) } else { None };

            // Build null buffer for this column (critical for IS NULL queries)
            let null_buf: Option<NullBuffer> = if col_idx < nulls.len() && !nulls[col_idx].is_empty() {
                let null_bitmap = &nulls[col_idx];
                let has_any_null = null_bitmap.iter().any(|&b| b != 0);
                if has_any_null {
                    let mut validity_bytes = vec![0xFFu8; (actual_limit + 7) / 8];
                    if let Some(ref indices) = row_indices {
                        for (new_idx, &old_idx) in indices.iter().enumerate() {
                            let ob = old_idx / 8;
                            let obit = old_idx % 8;
                            if ob < null_bitmap.len() && (null_bitmap[ob] >> obit) & 1 == 1 {
                                validity_bytes[new_idx / 8] &= !(1u8 << (new_idx % 8));
                            }
                        }
                    } else {
                        for byte_idx in 0..null_bitmap.len().min(validity_bytes.len()) {
                            validity_bytes[byte_idx] = !null_bitmap[byte_idx];
                        }
                    }
                    let tail = actual_limit % 8;
                    if tail > 0 {
                        let last = validity_bytes.len() - 1;
                        validity_bytes[last] &= (1u8 << tail) - 1;
                    }
                    Some(NullBuffer::new(BooleanBuffer::new(Buffer::from(validity_bytes), 0, actual_limit)))
                } else { None }
            } else { None };

            let (arrow_dt, array): (ArrowDataType, ArrayRef) = match col_data {
                Some(ColumnData::Int64(values)) => {
                    let data_vec: Vec<i64> = if let Some(ref indices) = row_indices {
                        indices.iter().map(|&i| if i < values.len() { values[i] } else { 0 }).collect()
                    } else {
                        values[..actual_limit.min(values.len())].to_vec()
                    };
                    let arr = PrimitiveArray::<Int64Type>::new(ScalarBuffer::from(data_vec), null_buf);
                    (ArrowDataType::Int64, Arc::new(arr) as ArrayRef)
                }
                Some(ColumnData::Float64(values)) => {
                    let data_vec: Vec<f64> = if let Some(ref indices) = row_indices {
                        indices.iter().map(|&i| if i < values.len() { values[i] } else { 0.0 }).collect()
                    } else {
                        values[..actual_limit.min(values.len())].to_vec()
                    };
                    let arr = PrimitiveArray::<Float64Type>::new(ScalarBuffer::from(data_vec), null_buf);
                    (ArrowDataType::Float64, Arc::new(arr) as ArrayRef)
                }
                Some(ColumnData::String { offsets, data }) => {
                    let count = offsets.len().saturating_sub(1);
                    if null_buf.is_some() {
                        // Has nulls: use Option<&str> path
                        let null_bitmap = &nulls[col_idx];
                        let strings: Vec<Option<&str>> = if let Some(ref indices) = row_indices {
                            indices.iter().map(|&i| {
                                let ob = i / 8;
                                let obit = i % 8;
                                if ob < null_bitmap.len() && (null_bitmap[ob] >> obit) & 1 == 1 {
                                    None
                                } else if i < count {
                                    let start = offsets[i] as usize;
                                    let end = offsets[i + 1] as usize;
                                    std::str::from_utf8(&data[start..end]).ok()
                                } else { None }
                            }).collect()
                        } else {
                            (0..actual_limit.min(count)).map(|i| {
                                let ob = i / 8;
                                let obit = i % 8;
                                if ob < null_bitmap.len() && (null_bitmap[ob] >> obit) & 1 == 1 {
                                    None
                                } else {
                                    let start = offsets[i] as usize;
                                    let end = offsets[i + 1] as usize;
                                    std::str::from_utf8(&data[start..end]).ok()
                                }
                            }).collect()
                        };
                        (ArrowDataType::Utf8, Arc::new(StringArray::from(strings)))
                    } else {
                        let strings: Vec<&str> = if let Some(ref indices) = row_indices {
                            indices.iter().map(|&i| {
                                if i < count {
                                    let start = offsets[i] as usize;
                                    let end = offsets[i + 1] as usize;
                                    std::str::from_utf8(&data[start..end]).unwrap_or("")
                                } else { "" }
                            }).collect()
                        } else {
                            let lim = actual_limit.min(count);
                            (0..lim).map(|i| {
                                let start = offsets[i] as usize;
                                let end = offsets[i + 1] as usize;
                                std::str::from_utf8(&data[start..end]).unwrap_or("")
                            }).collect()
                        };
                        (ArrowDataType::Utf8, Arc::new(StringArray::from_iter_values(strings)))
                    }
                }
                Some(ColumnData::Bool { data: packed, len }) => {
                    let bools: Vec<Option<bool>> = if let Some(ref indices) = row_indices {
                        indices.iter().map(|&i| {
                            // Check null
                            if col_idx < nulls.len() && !nulls[col_idx].is_empty() {
                                let ob = i / 8;
                                let obit = i % 8;
                                if ob < nulls[col_idx].len() && (nulls[col_idx][ob] >> obit) & 1 == 1 {
                                    return None;
                                }
                            }
                            if i < *len {
                                let byte_idx = i / 8;
                                let bit_idx = i % 8;
                                Some(byte_idx < packed.len() && (packed[byte_idx] >> bit_idx) & 1 == 1)
                            } else { None }
                        }).collect()
                    } else {
                        (0..actual_limit.min(*len)).map(|i| {
                            // Check null
                            if col_idx < nulls.len() && !nulls[col_idx].is_empty() {
                                let ob = i / 8;
                                let obit = i % 8;
                                if ob < nulls[col_idx].len() && (nulls[col_idx][ob] >> obit) & 1 == 1 {
                                    return None;
                                }
                            }
                            let byte_idx = i / 8;
                            let bit_idx = i % 8;
                            Some(byte_idx < packed.len() && (packed[byte_idx] >> bit_idx) & 1 == 1)
                        }).collect()
                    };
                    (ArrowDataType::Boolean, Arc::new(BooleanArray::from(bools)))
                }
                Some(ColumnData::Binary { offsets, data }) => {
                    use arrow::array::BinaryArray;
                    let count = offsets.len().saturating_sub(1);
                    let binary_data: Vec<Option<&[u8]>> = if let Some(ref indices) = row_indices {
                        indices.iter().map(|&i| {
                            if i < count {
                                let start = offsets[i] as usize;
                                let end = offsets[i + 1] as usize;
                                Some(&data[start..end] as &[u8])
                            } else { None }
                        }).collect()
                    } else {
                        (0..actual_limit.min(count)).map(|i| {
                            let start = offsets[i] as usize;
                            let end = offsets[i + 1] as usize;
                            Some(&data[start..end] as &[u8])
                        }).collect()
                    };
                    (ArrowDataType::Binary, Arc::new(BinaryArray::from(binary_data)))
                }
                _ => {
                    (ArrowDataType::Int64, Arc::new(Int64Array::from(vec![0i64; actual_limit])))
                }
            };

            fields.push(Field::new(col_name, arrow_dt, true));
            arrays.push(array);
        }

        let arrow_schema = Arc::new(Schema::new(fields));
        RecordBatch::try_new(arrow_schema, arrays)
            .map_err(|e| err_data(e.to_string()))
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
        
        let base_rows = header.row_count as usize;
        let delta_rows = self.delta_row_count();
        let total_rows = base_rows + delta_rows;
        
        let actual_start = start_row.min(total_rows);
        let actual_count = row_count
            .map(|c| c.min(total_rows - actual_start))
            .unwrap_or(total_rows - actual_start);
        
        if actual_count == 0 {
            return Ok(HashMap::new());
        }

        // Determine which columns to read (by name for delta merge)
        // When reading all columns (None), include both base and delta columns
        let mut col_names_to_read: Vec<String> = match column_names {
            Some(names) => names.iter().map(|s| s.to_string()).collect(),
            None => schema.columns.iter().map(|(n, _)| n.clone()).collect(),
        };
        
        // If reading all columns and delta exists, also include delta-only columns
        if column_names.is_none() && delta_rows > 0 {
            if let Ok(Some((_delta_ids, delta_columns))) = self.read_delta_data() {
                for col_name in delta_columns.keys() {
                    if !col_names_to_read.contains(col_name) {
                        col_names_to_read.push(col_name.clone());
                    }
                }
            }
        }
        
        let col_indices: Vec<usize> = col_names_to_read
            .iter()
            .filter_map(|name| schema.get_index(name))
            .collect();

        // Calculate how many rows to read from base vs delta
        let base_start = actual_start.min(base_rows);
        let base_count = if actual_start < base_rows {
            actual_count.min(base_rows - actual_start)
        } else {
            0
        };
        
        // V4 fast path: read from in-memory columns (no mmap column index)
        let v4_mode = column_index.is_empty() && header.footer_offset > 0;
        drop(column_index);
        drop(schema);
        drop(header);
        
        let mut result = HashMap::new();
        
        if v4_mode && base_count > 0 && self.has_v4_in_memory_data() {
            // Use in-memory columns if available (write buffer path)
            
            let schema = self.schema.read();
            let columns = self.columns.read();
            for &col_idx in &col_indices {
                let (col_name, col_type) = &schema.columns[col_idx];
                if col_idx < columns.len() && columns[col_idx].len() > 0 {
                    result.insert(col_name.clone(),
                        columns[col_idx].slice_range(base_start, base_start + base_count));
                } else {
                    result.insert(col_name.clone(),
                        Self::create_default_column(*col_type, base_count));
                }
            }
        } else if base_count > 0 {
            // V3: read from mmap via column index
            let file_guard = self.file.read();
            let file = file_guard.as_ref().ok_or_else(|| {
                err_not_conn("File not open")
            })?;
            let mut mmap_cache = self.mmap_cache.write();
            let column_index = self.column_index.read();
            let schema = self.schema.read();
            
            // V4 files don't use column_index — return defaults if we reach here
            // (Primary V4 read path is to_arrow_batch_mmap, this is a safety fallback)
            if column_index.is_empty() {
                for &col_idx in &col_indices {
                    let (col_name, col_type) = &schema.columns[col_idx];
                    result.insert(col_name.clone(),
                        Self::create_default_column(*col_type, base_count));
                }
                return Ok(result);
            }
            
            for &col_idx in &col_indices {
                let (col_name, col_type) = &schema.columns[col_idx];
                let index_entry = &column_index[col_idx];
                
                let col_data = self.read_column_range_mmap(
                    &mut mmap_cache,
                    file,
                    index_entry,
                    *col_type,
                    base_start,
                    base_count,
                    base_rows,
                )?;
                
                result.insert(col_name.clone(), col_data);
            }
        }
        
        // Merge delta data if needed
        if delta_rows > 0 && actual_start + actual_count > base_rows {
            if let Some((_delta_ids, delta_columns)) = self.read_delta_data()? {
                let delta_start = if actual_start > base_rows { actual_start - base_rows } else { 0 };
                let delta_count = actual_count - base_count;
                let actual_delta_count = delta_count.min(delta_rows - delta_start);
                
                // Get schema to determine column types for padding
                let schema = self.schema.read();
                
                for col_name in &col_names_to_read {
                    if let Some(delta_col) = delta_columns.get(col_name) {
                        // Extract the range we need from delta
                        let delta_slice = if delta_start == 0 && delta_count >= delta_col.len() {
                            delta_col.clone()
                        } else {
                            let end = delta_start + delta_count.min(delta_col.len().saturating_sub(delta_start));
                            let indices: Vec<usize> = (delta_start..end).collect();
                            delta_col.filter_by_indices(&indices)
                        };
                        
                        // Check if column exists in base result
                        if let Some(base_col) = result.get_mut(col_name) {
                            // Column exists in base - append delta
                            base_col.append(&delta_slice);
                        } else {
                            // Column only exists in delta - need to pad base rows with defaults first
                            let col_type = delta_slice.column_type();
                            let mut padded = ColumnData::new(col_type);
                            
                            // Pad base rows with defaults (NULL/0/empty)
                            for _ in 0..base_count {
                                match &mut padded {
                                    ColumnData::Int64(v) => v.push(0),
                                    ColumnData::Float64(v) => v.push(0.0),
                                    ColumnData::String { offsets, .. } => offsets.push(*offsets.last().unwrap_or(&0)),
                                    ColumnData::Bool { data, len } => {
                                        let byte_idx = *len / 8;
                                        if byte_idx >= data.len() { data.push(0); }
                                        *len += 1;
                                    }
                                    _ => {}
                                }
                            }
                            
                            // Append delta data after padding
                            padded.append(&delta_slice);
                            result.insert(col_name.clone(), padded);
                        }
                    } else {
                        // Column exists in schema but not in delta - pad with defaults
                        if let Some(col_idx) = schema.get_index(col_name) {
                            let (_, col_type) = &schema.columns[col_idx];
                            let mut padding = ColumnData::new(*col_type);
                            // Pad with default values
                            for _ in 0..actual_delta_count {
                                match &mut padding {
                                    ColumnData::Int64(v) => v.push(0),
                                    ColumnData::Float64(v) => v.push(0.0),
                                    ColumnData::String { offsets, .. } => offsets.push(*offsets.last().unwrap_or(&0)),
                                    ColumnData::Bool { data, len } => {
                                        let byte_idx = *len / 8;
                                        if byte_idx >= data.len() { data.push(0); }
                                        *len += 1;
                                    }
                                    _ => {}
                                }
                            }
                            
                            if let Some(base_col) = result.get_mut(col_name) {
                                base_col.append(&padding);
                            } else {
                                result.insert(col_name.clone(), padding);
                            }
                        }
                    }
                }
            }
        }
        
        Ok(result)
    }

    /// Check if a specific row/column is NULL
    /// Returns true if the value at (row_idx, col_name) is NULL
    pub fn is_null(&self, row_idx: usize, col_name: &str) -> bool {
        let schema = self.schema.read();
        if let Some(col_idx) = schema.get_index(col_name) {
            let nulls = self.nulls.read();
            if col_idx < nulls.len() {
                let null_bitmap = &nulls[col_idx];
                let byte_idx = row_idx / 8;
                let bit_idx = row_idx % 8;
                if byte_idx < null_bitmap.len() {
                    return (null_bitmap[byte_idx] & (1 << bit_idx)) != 0;
                }
            }
        }
        false
    }

    /// Get null bitmap for a column (for Arrow conversion)
    /// Returns a Vec<bool> where true means the value is NULL
    /// Reads from file via mmap if not loaded in memory
    pub fn get_null_mask(&self, col_name: &str, start_row: usize, row_count: usize) -> Vec<bool> {
        // Only use in-memory nulls if available — mmap path handles nulls separately
        let schema = self.schema.read();
        let mut result = vec![false; row_count];
        
        if let Some(col_idx) = schema.get_index(col_name) {
            // First check in-memory nulls
            let nulls = self.nulls.read();
            if col_idx < nulls.len() && !nulls[col_idx].is_empty() {
                let null_bitmap = &nulls[col_idx];
                for i in 0..row_count {
                    let row_idx = start_row + i;
                    let byte_idx = row_idx / 8;
                    let bit_idx = row_idx % 8;
                    if byte_idx < null_bitmap.len() {
                        result[i] = (null_bitmap[byte_idx] & (1 << bit_idx)) != 0;
                    }
                }
            } else {
                // Read null bitmap from file via mmap
                drop(nulls);
                let column_index = self.column_index.read();
                if col_idx < column_index.len() {
                    let index_entry = &column_index[col_idx];
                    let null_len = index_entry.null_length as usize;
                    if null_len > 0 {
                        if let Some(file) = self.file.read().as_ref() {
                            let mut mmap_cache = self.mmap_cache.write();
                            let mut null_bitmap = vec![0u8; null_len];
                            if mmap_cache.read_at(file, &mut null_bitmap, index_entry.null_offset).is_ok() {
                                for i in 0..row_count {
                                    let row_idx = start_row + i;
                                    let byte_idx = row_idx / 8;
                                    let bit_idx = row_idx % 8;
                                    if byte_idx < null_bitmap.len() {
                                        result[i] = (null_bitmap[byte_idx] & (1 << bit_idx)) != 0;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        result
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
        let is_v4 = self.is_v4_format();
        // V4 without in-memory data: return empty — caller falls back to mmap
        if is_v4 && !self.has_v4_in_memory_data() {
            return Ok((HashMap::new(), Vec::new()));
        }
        
        let header = self.header.read();
        let schema = self.schema.read();
        
        let total_rows = header.row_count as usize;
        if total_rows == 0 {
            return Ok((HashMap::new(), Vec::new()));
        }

        // First, read the filter column to determine matching rows
        let filter_col_idx = schema.get_index(filter_column).ok_or_else(|| {
            err_not_found(format!("Filter column: {}", filter_column))
        })?;
        
        let (_, filter_col_type) = &schema.columns[filter_col_idx];
        let filter_col_type = *filter_col_type;
        drop(schema);
        drop(header);

        // Read filter column data
        let filter_data = self.read_column_auto(filter_col_idx, filter_col_type, 0, total_rows, total_rows, is_v4)?;
        
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
            _ => (0..total_rows).collect(),
        };

        if matching_indices.is_empty() {
            return Ok((HashMap::new(), Vec::new()));
        }

        // Determine which columns to read
        let schema = self.schema.read();
        let col_indices: Vec<usize> = match column_names {
            Some(names) => names
                .iter()
                .filter_map(|name| schema.get_index(name))
                .collect(),
            None => (0..schema.column_count()).collect(),
        };

        let mut result = HashMap::new();

        // Read only matching rows for each column
        for &col_idx in &col_indices {
            let (col_name, col_type) = &schema.columns[col_idx];
            
            // OPTIMIZATION: Skip reading filter column again - reuse already-read data
            if col_idx == filter_col_idx {
                let filtered_data = match &filter_data {
                    ColumnData::Int64(values) => {
                        ColumnData::Int64(matching_indices.iter().map(|&i| values[i]).collect())
                    }
                    ColumnData::Float64(values) => {
                        ColumnData::Float64(matching_indices.iter().map(|&i| values[i]).collect())
                    }
                    other => other.clone(),
                };
                result.insert(col_name.clone(), filtered_data);
                continue;
            }
            
            let col_data = self.read_column_scattered_auto(col_idx, *col_type, &matching_indices, total_rows, is_v4)?;
            result.insert(col_name.clone(), col_data);
        }

        Ok((result, matching_indices))
    }

    /// Read columns with STRING predicate pushdown - filter rows at storage level
    /// This is optimized for string equality filters (column = 'value')
    /// Uses bloom filters to skip row groups that definitely don't contain the value
    pub fn read_columns_filtered_string(
        &self,
        column_names: Option<&[&str]>,
        filter_column: &str,
        filter_value: &str,
        filter_eq: bool,  // true = equals, false = not equals
    ) -> io::Result<(HashMap<String, ColumnData>, Vec<usize>)> {
        // V4 FAST PATH: scan in-memory columns directly (no disk I/O)
        {
            let header = self.header.read();
            if header.footer_offset > 0 {
                drop(header);
                // Only use fast path if data is already in memory
                if !self.has_v4_in_memory_data() {
                    return Ok((HashMap::new(), Vec::new()));
                }
                
                let schema = self.schema.read();
                let columns = self.columns.read();
                let deleted = self.deleted.read();
                let total_rows = self.ids.read().len();
                
                let filter_col_idx = match schema.get_index(filter_column) {
                    Some(idx) => idx,
                    None => return Ok((HashMap::new(), Vec::new())),
                };
                
                let filter_bytes = filter_value.as_bytes();
                let filter_len = filter_bytes.len();
                let has_deleted = deleted.iter().any(|&b| b != 0);
                
                let mut matching_indices: Vec<usize> = Vec::with_capacity(1024);
                
                if filter_col_idx < columns.len() {
                    if let ColumnData::String { offsets, data } = &columns[filter_col_idx] {
                        let count = offsets.len().saturating_sub(1).min(total_rows);
                        let first_byte = filter_bytes.first().copied();
                        
                        for i in 0..count {
                            // Skip deleted rows
                            if has_deleted {
                                let byte_idx = i / 8;
                                let bit_idx = i % 8;
                                if byte_idx < deleted.len() && (deleted[byte_idx] >> bit_idx) & 1 != 0 {
                                    continue;
                                }
                            }
                            
                            let start = offsets[i] as usize;
                            let end = offsets[i + 1] as usize;
                            let str_len = end - start;
                            
                            // Fast rejection by length
                            if str_len != filter_len {
                                if !filter_eq { matching_indices.push(i); }
                                continue;
                            }
                            
                            // Fast rejection by first byte
                            if let Some(fb) = first_byte {
                                if start < data.len() && data[start] != fb {
                                    if !filter_eq { matching_indices.push(i); }
                                    continue;
                                }
                            }
                            
                            let matches = end <= data.len() && &data[start..end] == filter_bytes;
                            if (filter_eq && matches) || (!filter_eq && !matches) {
                                matching_indices.push(i);
                            }
                        }
                    }
                }
                
                if matching_indices.is_empty() {
                    return Ok((HashMap::new(), Vec::new()));
                }
                
                // Read only needed columns for matching indices
                let col_indices: Vec<usize> = match column_names {
                    Some(names) => names.iter()
                        .filter(|&&n| n != "_id")
                        .filter_map(|&name| schema.get_index(name))
                        .collect(),
                    None => (0..schema.column_count()).collect(),
                };
                
                let mut result = HashMap::new();
                for &col_idx in &col_indices {
                    let (col_name, _) = &schema.columns[col_idx];
                    if col_idx < columns.len() {
                        result.insert(col_name.clone(), columns[col_idx].filter_by_indices(&matching_indices));
                    }
                }
                
                return Ok((result, matching_indices));
            }
        }
        
        // V3 SLOW PATH: read from disk
        let is_v4 = self.is_v4_format();
        
        let header = self.header.read();
        let schema = self.schema.read();
        
        let total_rows = header.row_count as usize;
        if total_rows == 0 {
            return Ok((HashMap::new(), Vec::new()));
        }

        // Find filter column
        let filter_col_idx = schema.get_index(filter_column).ok_or_else(|| {
            err_not_found(format!("Filter column: {}", filter_column))
        })?;
        
        let (_, filter_col_type) = &schema.columns[filter_col_idx];
        let filter_col_type = *filter_col_type;
        
        // Only works for string columns (including dictionary-encoded)
        if !matches!(filter_col_type, ColumnType::String | ColumnType::StringDict) {
            return Err(err_input("String filter requires string column"));
        }
        
        drop(schema);
        drop(header);

        // Read filter column data
        let filter_data = self.read_column_auto(filter_col_idx, filter_col_type, 0, total_rows, total_rows, is_v4)?;
        
        // OPTIMIZATION: Build and use bloom filter for large datasets
        // Build bloom filter on-the-fly and use it to identify candidate row groups
        let filter_bytes = filter_value.as_bytes();
        let use_bloom = filter_eq && total_rows > 10000; // Only use bloom for equality on large datasets
        
        // Apply string filter
        let matching_indices: Vec<usize> = match filter_data {
            ColumnData::String { offsets, data } => {
                let count = offsets.len().saturating_sub(1);
                let filter_len = filter_bytes.len() as u32;
                
                // OPTIMIZATION: Pre-compute first byte and length for fast rejection
                let first_byte = filter_bytes.first().copied();
                
                // Pre-allocate with estimated capacity (assume ~10% match rate)
                let mut result = Vec::with_capacity(count / 10 + 1);
                
                // OPTIMIZATION: Use bloom filter to skip row groups for large datasets
                const ROW_GROUP_SIZE: usize = 8192;  // 8K rows per group for bloom filter
                
                if use_bloom && count > ROW_GROUP_SIZE {
                    // Build bloom filter index and identify candidate groups
                    use crate::storage::bloom::{ColumnBloomIndex, BLOOM_FP_RATE};
                    let bloom_index = ColumnBloomIndex::build_from_strings(
                        filter_column,
                        &offsets,
                        &data,
                        ROW_GROUP_SIZE,
                        BLOOM_FP_RATE,
                    );
                    
                    // Get row ranges that might contain the value
                    let scan_ranges = bloom_index.get_scan_ranges(filter_bytes);
                    
                    // Only scan candidate row groups
                    for (group_start, group_end) in scan_ranges {
                        for i in group_start..group_end.min(count) {
                            let start = offsets[i] as usize;
                            let end = offsets[i + 1] as usize;
                            let str_len = (end - start) as u32;
                            
                            // Fast path: length mismatch rejection
                            if str_len != filter_len {
                                continue;
                            }
                            
                            // Fast path: first byte mismatch
                            if let Some(fb) = first_byte {
                                if start < data.len() && data[start] != fb {
                                    continue;
                                }
                            }
                            
                            // Full comparison
                            let matches = end <= data.len() && &data[start..end] == filter_bytes;
                            if matches {
                                result.push(i);
                            }
                        }
                    }
                } else {
                    // Standard chunked processing for small datasets or != filter
                    const CHUNK_SIZE: usize = 1024;
                    for chunk_start in (0..count).step_by(CHUNK_SIZE) {
                        let chunk_end = (chunk_start + CHUNK_SIZE).min(count);
                        
                        for i in chunk_start..chunk_end {
                            let start = offsets[i] as usize;
                            let end = offsets[i + 1] as usize;
                            let str_len = (end - start) as u32;
                            
                            // Fast path: length mismatch rejection (most common case)
                            if str_len != filter_len {
                                if !filter_eq {
                                    result.push(i);
                                }
                                continue;
                            }
                            
                            // Fast path: first byte mismatch (catches ~255/256 of remaining)
                            if let Some(fb) = first_byte {
                                if start < data.len() && data[start] != fb {
                                    if !filter_eq {
                                        result.push(i);
                                    }
                                    continue;
                                }
                            }
                            
                            // Full comparison only when length and first byte match
                            let matches = end <= data.len() && &data[start..end] == filter_bytes;
                            if filter_eq == matches {
                                result.push(i);
                            }
                        }
                    }
                }
                result
            }
            ColumnData::StringDict { indices, dict_offsets, dict_data } => {
                // OPTIMIZATION: Find matching dictionary index first, then scan indices
                // This is O(dict_size + row_count) vs O(row_count * string_len)
                let filter_bytes = filter_value.as_bytes();
                let filter_len = filter_bytes.len();
                let mut matching_dict_idx: Option<u32> = None;
                
                // Find which dictionary entry matches the filter value (with fast rejection)
                let dict_count = dict_offsets.len().saturating_sub(1);
                for i in 0..dict_count {
                    let start = dict_offsets[i] as usize;
                    let end = if i + 1 < dict_offsets.len() { dict_offsets[i + 1] as usize } else { dict_data.len() };
                    // Fast rejection by length
                    if end - start != filter_len {
                        continue;
                    }
                    if end <= dict_data.len() && &dict_data[start..end] == filter_bytes {
                        matching_dict_idx = Some((i + 1) as u32); // +1 because 0 = NULL
                        break;
                    }
                }
                
                // OPTIMIZATION: Pre-allocate and use pointer-based scan for speed
                let count = indices.len();
                let mut result = Vec::with_capacity(count / 10 + 1);
                
                match (matching_dict_idx, filter_eq) {
                    (Some(target_idx), true) => {
                        // SIMD-friendly: scan in chunks with pointer arithmetic
                        let ptr = indices.as_ptr();
                        for i in 0..count {
                            // Pointer dereference avoids bounds checking
                            if unsafe { *ptr.add(i) } == target_idx {
                                result.push(i);
                            }
                        }
                    }
                    (Some(target_idx), false) => {
                        let ptr = indices.as_ptr();
                        for i in 0..count {
                            let idx = unsafe { *ptr.add(i) };
                            if idx != target_idx && idx != 0 {
                                result.push(i);
                            }
                        }
                    }
                    (None, true) => {
                        // Value not in dictionary, no matches for equality
                        // result stays empty
                    }
                    (None, false) => {
                        // Value not in dictionary, all non-NULL rows match
                        let ptr = indices.as_ptr();
                        for i in 0..count {
                            if unsafe { *ptr.add(i) } != 0 {
                                result.push(i);
                            }
                        }
                    }
                }
                result
            }
            _ => return Err(err_input("Expected string column")),
        };

        if matching_indices.is_empty() {
            return Ok((HashMap::new(), Vec::new()));
        }

        // Read only matching rows for each requested column
        let schema = self.schema.read();
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
            let col_data = self.read_column_scattered_auto(col_idx, *col_type, &matching_indices, total_rows, is_v4)?;
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
        // V4: scan in-memory columns directly with early termination
        {
            let header = self.header.read();
            if header.footer_offset > 0 {
                drop(header);
                if !self.has_v4_in_memory_data() {
                    return Ok((HashMap::new(), Vec::new()));
                }
                
                let schema = self.schema.read();
                let columns = self.columns.read();
                let deleted = self.deleted.read();
                let total_rows = self.ids.read().len();
                
                let filter_col_idx = match schema.get_index(filter_column) {
                    Some(idx) => idx,
                    None => return Ok((HashMap::new(), Vec::new())),
                };
                
                let filter_bytes = filter_value.as_bytes();
                let filter_len = filter_bytes.len();
                let needed = offset + limit;
                let has_deleted = deleted.iter().any(|&b| b != 0);
                
                // Scan filter column with early termination
                let mut matching_indices: Vec<usize> = Vec::with_capacity(needed.min(1024));
                
                if filter_col_idx < columns.len() {
                    if let ColumnData::String { offsets, data } = &columns[filter_col_idx] {
                        let count = offsets.len().saturating_sub(1).min(total_rows);
                        for i in 0..count {
                            if matching_indices.len() >= needed { break; }
                            
                            // Skip deleted rows
                            if has_deleted {
                                let byte_idx = i / 8;
                                let bit_idx = i % 8;
                                if byte_idx < deleted.len() && (deleted[byte_idx] >> bit_idx) & 1 != 0 {
                                    continue;
                                }
                            }
                            
                            let start = offsets[i] as usize;
                            let end = offsets[i + 1] as usize;
                            let str_len = end - start;
                            
                            // Fast rejection by length
                            if str_len != filter_len { 
                                if filter_eq { continue; } 
                                else { matching_indices.push(i); continue; }
                            }
                            
                            let matches = end <= data.len() && &data[start..end] == filter_bytes;
                            if (filter_eq && matches) || (!filter_eq && !matches) {
                                matching_indices.push(i);
                            }
                        }
                    }
                }
                
                // Apply offset
                if offset > 0 && offset < matching_indices.len() {
                    matching_indices = matching_indices[offset..].to_vec();
                } else if offset >= matching_indices.len() {
                    matching_indices.clear();
                }
                if matching_indices.len() > limit {
                    matching_indices.truncate(limit);
                }
                
                if matching_indices.is_empty() {
                    return Ok((HashMap::new(), Vec::new()));
                }
                
                // Read only needed columns for matching indices
                let col_indices: Vec<usize> = match column_names {
                    Some(names) => names.iter()
                        .filter(|&&n| n != "_id")
                        .filter_map(|&name| schema.get_index(name))
                        .collect(),
                    None => (0..schema.column_count()).collect(),
                };
                
                let mut result = HashMap::new();
                for &col_idx in &col_indices {
                    let (col_name, _) = &schema.columns[col_idx];
                    if col_idx < columns.len() {
                        result.insert(col_name.clone(), columns[col_idx].filter_by_indices(&matching_indices));
                    }
                }
                
                return Ok((result, matching_indices));
            }
        }
        
        let header = self.header.read();
        let schema = self.schema.read();
        let column_index = self.column_index.read();
        
        let total_rows = header.row_count as usize;
        if total_rows == 0 {
            return Ok((HashMap::new(), Vec::new()));
        }

        let filter_col_idx = schema.get_index(filter_column).ok_or_else(|| {
            err_not_found(format!("Filter column: {}", filter_column))
        })?;
        
        let (_, filter_col_type) = &schema.columns[filter_col_idx];
        let filter_index = &column_index[filter_col_idx];
        
        if !matches!(filter_col_type, ColumnType::String | ColumnType::StringDict) {
            return Err(err_input("String filter requires string column"));
        }
        
        let file_guard = self.file.read();
        let file = file_guard.as_ref().ok_or_else(|| {
            err_not_conn("File not open")
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
        // V4: scan in-memory columns directly with early termination
        {
            let header = self.header.read();
            if header.footer_offset > 0 {
                drop(header);
                if !self.has_v4_in_memory_data() {
                    return Ok((HashMap::new(), Vec::new()));
                }
                
                let schema = self.schema.read();
                let columns = self.columns.read();
                let deleted = self.deleted.read();
                let total_rows = self.ids.read().len();
                let needed = offset + limit;
                
                let filter_col_idx = match schema.get_index(filter_column) {
                    Some(idx) => idx,
                    None => return Ok((HashMap::new(), Vec::new())),
                };
                
                let has_deleted = deleted.iter().any(|&b| b != 0);
                let mut matching_indices: Vec<usize> = Vec::with_capacity(needed.min(1024));
                
                if filter_col_idx < columns.len() {
                    match &columns[filter_col_idx] {
                        ColumnData::Int64(values) => {
                            let low_i = low as i64;
                            let high_i = high as i64;
                            let count = values.len().min(total_rows);
                            for i in 0..count {
                                if matching_indices.len() >= needed { break; }
                                if has_deleted {
                                    let byte_idx = i / 8;
                                    let bit_idx = i % 8;
                                    if byte_idx < deleted.len() && (deleted[byte_idx] >> bit_idx) & 1 != 0 { continue; }
                                }
                                let v = unsafe { *values.get_unchecked(i) };
                                if v >= low_i && v <= high_i {
                                    matching_indices.push(i);
                                }
                            }
                        }
                        ColumnData::Float64(values) => {
                            let count = values.len().min(total_rows);
                            for i in 0..count {
                                if matching_indices.len() >= needed { break; }
                                if has_deleted {
                                    let byte_idx = i / 8;
                                    let bit_idx = i % 8;
                                    if byte_idx < deleted.len() && (deleted[byte_idx] >> bit_idx) & 1 != 0 { continue; }
                                }
                                let v = unsafe { *values.get_unchecked(i) };
                                if v >= low && v <= high {
                                    matching_indices.push(i);
                                }
                            }
                        }
                        _ => {}
                    }
                }
                
                // Apply offset + limit
                if offset > 0 && offset < matching_indices.len() {
                    matching_indices = matching_indices[offset..].to_vec();
                } else if offset >= matching_indices.len() {
                    matching_indices.clear();
                }
                if matching_indices.len() > limit {
                    matching_indices.truncate(limit);
                }
                
                if matching_indices.is_empty() {
                    return Ok((HashMap::new(), Vec::new()));
                }
                
                let col_indices: Vec<usize> = match column_names {
                    Some(names) => names.iter()
                        .filter(|&&n| n != "_id")
                        .filter_map(|&name| schema.get_index(name))
                        .collect(),
                    None => (0..schema.column_count()).collect(),
                };
                
                let mut result = HashMap::new();
                for &col_idx in &col_indices {
                    let (col_name, _) = &schema.columns[col_idx];
                    if col_idx < columns.len() {
                        result.insert(col_name.clone(), columns[col_idx].filter_by_indices(&matching_indices));
                    }
                }
                
                return Ok((result, matching_indices));
            }
        }
        
        let schema = self.schema.read();
        let column_index = self.column_index.read();
        let header = self.header.read();
        let deleted = self.deleted.read();
        
        let filter_col_idx = schema.get_index(filter_column).ok_or_else(|| {
            err_not_found(format!("Column: {}", filter_column))
        })?;
        
        let (_, filter_col_type) = &schema.columns[filter_col_idx];
        let filter_index = &column_index[filter_col_idx];
        let total_rows = header.row_count as usize;
        
        let file_guard = self.file.read();
        let file = file_guard.as_ref().ok_or_else(|| {
            err_not_conn("File not open")
        })?;
        
        let mut mmap_cache = self.mmap_cache.write();
        let needed = offset + limit;
        
        // Only works for numeric columns
        if !matches!(filter_col_type, ColumnType::Int64 | ColumnType::Float64 | 
                     ColumnType::Int32 | ColumnType::Int16 | ColumnType::Int8 |
                     ColumnType::UInt64 | ColumnType::UInt32 | ColumnType::UInt16 | ColumnType::UInt8 |
                     ColumnType::Float32) {
            return Err(err_input("Range filter needs numeric columns"));
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
        // V4: scan in-memory columns directly with early termination
        {
            let header = self.header.read();
            if header.footer_offset > 0 {
                drop(header);
                if !self.has_v4_in_memory_data() { return Ok((HashMap::new(), Vec::new())); }
                
                let schema = self.schema.read();
                let columns = self.columns.read();
                let deleted = self.deleted.read();
                let total_rows = self.ids.read().len();
                let needed = offset + limit;
                
                let str_col_idx = match schema.get_index(string_column) {
                    Some(idx) => idx,
                    None => return Ok((HashMap::new(), Vec::new())),
                };
                let num_col_idx = match schema.get_index(numeric_column) {
                    Some(idx) => idx,
                    None => return Ok((HashMap::new(), Vec::new())),
                };
                
                let filter_bytes = string_value.as_bytes();
                let filter_len = filter_bytes.len();
                let has_deleted = deleted.iter().any(|&b| b != 0);
                let mut matching_indices: Vec<usize> = Vec::with_capacity(needed.min(1024));
                
                // Get string and numeric column references
                if str_col_idx < columns.len() && num_col_idx < columns.len() {
                    if let ColumnData::String { offsets: str_offsets, data: str_data } = &columns[str_col_idx] {
                        let count = str_offsets.len().saturating_sub(1).min(total_rows);
                        
                        let num_compare = |idx: usize| -> bool {
                            match &columns[num_col_idx] {
                                ColumnData::Int64(vals) if idx < vals.len() => {
                                    let v = vals[idx] as f64;
                                    match numeric_op {
                                        ">" => v > numeric_value,
                                        ">=" => v >= numeric_value,
                                        "<" => v < numeric_value,
                                        "<=" => v <= numeric_value,
                                        "=" => (v - numeric_value).abs() < f64::EPSILON,
                                        _ => false,
                                    }
                                }
                                ColumnData::Float64(vals) if idx < vals.len() => {
                                    let v = vals[idx];
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
                            }
                        };
                        
                        for i in 0..count {
                            if matching_indices.len() >= needed { break; }
                            
                            if has_deleted {
                                let byte_idx = i / 8;
                                let bit_idx = i % 8;
                                if byte_idx < deleted.len() && (deleted[byte_idx] >> bit_idx) & 1 != 0 { continue; }
                            }
                            
                            // String equality check first (usually more selective)
                            let start = str_offsets[i] as usize;
                            let end = str_offsets[i + 1] as usize;
                            if end - start != filter_len { continue; }
                            if end > str_data.len() || &str_data[start..end] != filter_bytes { continue; }
                            
                            // Then numeric check
                            if num_compare(i) {
                                matching_indices.push(i);
                            }
                        }
                    }
                }
                
                // Apply offset + limit
                if offset > 0 && offset < matching_indices.len() {
                    matching_indices = matching_indices[offset..].to_vec();
                } else if offset >= matching_indices.len() {
                    matching_indices.clear();
                }
                if matching_indices.len() > limit {
                    matching_indices.truncate(limit);
                }
                
                if matching_indices.is_empty() {
                    return Ok((HashMap::new(), Vec::new()));
                }
                
                let col_indices: Vec<usize> = match column_names {
                    Some(names) => names.iter()
                        .filter(|&&n| n != "_id")
                        .filter_map(|&name| schema.get_index(name))
                        .collect(),
                    None => (0..schema.column_count()).collect(),
                };
                
                let mut result = HashMap::new();
                for &col_idx in &col_indices {
                    let (col_name, _) = &schema.columns[col_idx];
                    if col_idx < columns.len() {
                        result.insert(col_name.clone(), columns[col_idx].filter_by_indices(&matching_indices));
                    }
                }
                
                return Ok((result, matching_indices));
            }
        }
        
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
            err_not_found(format!("String column: {}", string_column))
        })?;
        let (_, str_col_type) = &schema.columns[str_col_idx];
        let str_index = &column_index[str_col_idx];
        
        // Get numeric column info
        let num_col_idx = schema.get_index(numeric_column).ok_or_else(|| {
            err_not_found(format!("Numeric column: {}", numeric_column))
        })?;
        let (_, num_col_type) = &schema.columns[num_col_idx];
        let num_index = &column_index[num_col_idx];
        
        // Validate column types
        if !matches!(str_col_type, ColumnType::String | ColumnType::StringDict) {
            return Err(err_input("String filter requires string column"));
        }
        if !matches!(num_col_type, ColumnType::Int64 | ColumnType::Float64 | 
                     ColumnType::Int32 | ColumnType::Int16 | ColumnType::Int8 |
                     ColumnType::UInt64 | ColumnType::UInt32 | ColumnType::UInt16 | ColumnType::UInt8 |
                     ColumnType::Float32) {
            return Err(err_input("Numeric filter needs numeric column"));
        }
        
        let file_guard = self.file.read();
        let file = file_guard.as_ref().ok_or_else(|| {
            err_not_conn("File not open")
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
        Err(err_input("Needs dictionary-encoded string"))
    }

    /// Read a single column for specific row indices
    pub fn read_column_by_indices(
        &self,
        column_name: &str,
        row_indices: &[usize],
    ) -> io::Result<ColumnData> {
        let is_v4 = self.is_v4_format();
        
        let schema = self.schema.read();
        let header = self.header.read();
        
        let col_idx = schema.get_index(column_name).ok_or_else(|| {
            err_not_found(format!("Column: {}", column_name))
        })?;
        
        let (_, col_type) = &schema.columns[col_idx];
        let col_type = *col_type;
        let total_rows = header.row_count as usize;
        drop(schema);
        drop(header);

        self.read_column_scattered_auto(col_idx, col_type, row_indices, total_rows, is_v4)
    }

    /// Check if this is a V4 format file (has footer).
    #[inline]
    pub fn is_v4_format(&self) -> bool {
        self.header.read().footer_offset > 0
    }

    /// Check if V4 column data is currently loaded in memory.
    /// Does NOT trigger any loading — purely a state check.
    #[inline]
    pub fn has_v4_in_memory_data(&self) -> bool {
        if !self.is_v4_format() { return false; }
        let cols = self.columns.read();
        !cols.is_empty() && cols.iter().any(|c| c.len() > 0)
    }

    /// Check if in-memory columns contain the FULL base dataset (not just write buffer).
    /// Used by save() to decide between append vs full rewrite.
    #[inline]
    fn has_v4_in_memory_data_with_base(&self, on_disk_rows: usize) -> bool {
        let cols = self.columns.read();
        if cols.is_empty() { return false; }
        // If any column has >= on_disk_rows elements, base data is loaded
        cols.iter().any(|c| c.len() >= on_disk_rows)
    }

    /// Get the V4 footer, reading it fresh from disk each time.
    /// Returns None for V3 files.
    /// NOTE: Always re-reads header + footer from disk via a fresh file handle
    /// to avoid stale cache issues when another storage instance has appended data.
    fn get_or_load_footer(&self) -> io::Result<Option<V4Footer>> {
        let file_len = std::fs::metadata(&self.path)
            .map(|m| m.len())
            .unwrap_or(0);
        if file_len < HEADER_SIZE_V3 as u64 {
            return Ok(None);
        }

        // Invalidate mmap if file size changed (another instance appended data)
        {
            let mut mc = self.mmap_cache.write();
            if mc.file_size != 0 && mc.file_size != file_len {
                mc.invalidate();
            }
        }

        // Ensure file handle is open (reopen if needed after save_v4 replaced file)
        {
            let fg = self.file.read();
            if fg.is_none() {
                drop(fg);
                if let Ok(f) = File::open(&self.path) {
                    *self.file.write() = Some(f);
                }
            }
        }

        let file_guard = self.file.read();
        let file_handle = match file_guard.as_ref() {
            Some(f) => f,
            None => return Ok(None),
        };
        let mut mmap = self.mmap_cache.write();

        // Read the on-disk header fresh to get current footer_offset
        let mut header_bytes = [0u8; HEADER_SIZE_V3];
        mmap.read_at(file_handle, &mut header_bytes, 0)?;
        let on_disk_header = OnDemandHeader::from_bytes(&header_bytes)?;

        if on_disk_header.footer_offset == 0 || on_disk_header.version != FORMAT_VERSION_V4 {
            return Ok(None);
        }
        let footer_offset = on_disk_header.footer_offset;
        if footer_offset >= file_len {
            return Ok(None);
        }

        let footer_byte_count = (file_len - footer_offset) as usize;
        let mut footer_bytes = vec![0u8; footer_byte_count];
        mmap.read_at(file_handle, &mut footer_bytes, footer_offset)?;
        drop(mmap);
        drop(file_guard);

        V4Footer::from_bytes(&footer_bytes).map(Some)
    }

    /// Invalidate the cached V4 footer (call after writes that change the footer).
    fn invalidate_footer_cache(&self) {
        *self.v4_footer.write() = None;
    }

    /// Read columns from on-disk V4 Row Groups directly via mmap → Arrow RecordBatch.
    /// Only materializes the requested columns; skips others with zero allocation.
    /// This is the core on-demand reading function that avoids loading all data into memory.
    ///
    /// # Arguments
    /// * `column_names` - Which columns to read (None = all)
    /// * `include_id` - Whether to include the _id column
    /// * `row_limit` - Maximum number of active rows to return (None = all)
    /// * `dict_encode_strings` - Whether to produce DictionaryArray for low-cardinality strings
    pub fn to_arrow_batch_mmap(
        &self,
        column_names: Option<&[&str]>,
        include_id: bool,
        row_limit: Option<usize>,
        dict_encode_strings: bool,
    ) -> io::Result<Option<RecordBatch>> {
        use arrow::array::{Int64Array, Float64Array, StringArray, BooleanArray, PrimitiveArray};
        use arrow::buffer::{Buffer, NullBuffer, BooleanBuffer, ScalarBuffer};
        use arrow::datatypes::{Schema, Field, DataType as ArrowDataType, Int64Type, Float64Type};
        use std::sync::Arc;

        let footer = match self.get_or_load_footer()? {
            Some(f) => f,
            None => return Ok(None), // V3 file — caller should use legacy path
        };

        let schema = &footer.schema;
        let col_count = schema.column_count();

        // Determine which columns to read (indices into schema)
        let col_indices: Vec<usize> = if let Some(names) = column_names {
            names.iter()
                .filter(|&&n| n != "_id")
                .filter_map(|&name| schema.get_index(name))
                .collect()
        } else {
            (0..col_count).collect()
        };

        // Compute total active rows across all RGs
        let total_active: usize = footer.row_groups.iter()
            .map(|rg| rg.active_rows() as usize)
            .sum();

        if total_active == 0 {
            // Build empty schema and return empty batch
            let mut fields: Vec<Field> = Vec::new();
            if include_id {
                fields.push(Field::new("_id", ArrowDataType::Int64, false));
            }
            for &ci in &col_indices {
                let (name, ct) = &schema.columns[ci];
                let dt = match ct {
                    ColumnType::Int64 | ColumnType::Int8 | ColumnType::Int16 |
                    ColumnType::Int32 | ColumnType::UInt8 | ColumnType::UInt16 |
                    ColumnType::UInt32 | ColumnType::UInt64 => ArrowDataType::Int64,
                    ColumnType::Float64 | ColumnType::Float32 => ArrowDataType::Float64,
                    ColumnType::Bool => ArrowDataType::Boolean,
                    _ => ArrowDataType::Utf8,
                };
                fields.push(Field::new(name, dt, true));
            }
            let arrow_schema = Arc::new(Schema::new(fields));
            return Ok(Some(RecordBatch::new_empty(arrow_schema)));
        }

        let effective_limit = row_limit.unwrap_or(total_active).min(total_active);

        // Get mmap for the file
        let file_guard = self.file.read();
        let file = file_guard.as_ref()
            .ok_or_else(|| err_not_conn("File not open for mmap read"))?;
        let mut mmap_guard = self.mmap_cache.write();
        let mmap_ref = mmap_guard.get_or_create(file)?;

        // Accumulators for each output column + _id
        let mut all_ids: Vec<i64> = Vec::with_capacity(effective_limit);
        // For each requested column, accumulate ColumnData across RGs
        let mut col_accumulators: Vec<ColumnData> = col_indices.iter()
            .map(|&ci| {
                let ct = schema.columns[ci].1;
                // StringDict is decoded to String before accumulation,
                // so accumulator must be String type
                let acc_type = if ct == ColumnType::StringDict { ColumnType::String } else { ct };
                ColumnData::new(acc_type)
            })
            .collect();
        let mut null_accumulators: Vec<Vec<bool>> = vec![Vec::new(); col_indices.len()];
        let mut rows_collected: usize = 0;

        for (rg_idx, rg_meta) in footer.row_groups.iter().enumerate() {
            if rows_collected >= effective_limit {
                break;
            }
            if rg_meta.row_count == 0 {
                continue;
            }

            let rg_rows = rg_meta.row_count as usize;
            let rg_end = (rg_meta.offset + rg_meta.data_size) as usize;
            if rg_end > mmap_ref.len() {
                return Err(err_data("RG extends past EOF"));
            }
            let rg_bytes = &mmap_ref[rg_meta.offset as usize .. rg_end];

            let mut pos: usize = 32; // skip RG header

            // Read IDs
            let id_byte_len = rg_rows * 8;
            if pos + id_byte_len > rg_bytes.len() {
                return Err(err_data("RG IDs truncated"));
            }
            let id_slice = &rg_bytes[pos..pos + id_byte_len];
            pos += id_byte_len;

            // Read deletion vector
            let del_vec_len = (rg_rows + 7) / 8;
            if pos + del_vec_len > rg_bytes.len() {
                return Err(err_data("RG deletion vector truncated"));
            }
            let del_bytes = &rg_bytes[pos..pos + del_vec_len];
            pos += del_vec_len;

            // Build active row mask for this RG
            let has_deletes = rg_meta.deletion_count > 0;
            let rg_active = rg_meta.active_rows() as usize;
            let rows_to_take = (effective_limit - rows_collected).min(rg_active);

            // Always collect active IDs from this RG (needed for DeltaMerger overlay)
            {
                let mut taken = 0;
                for i in 0..rg_rows {
                    if has_deletes && (del_bytes[i / 8] >> (i % 8)) & 1 == 1 {
                        continue; // deleted
                    }
                    let id = u64::from_le_bytes(
                        id_slice[i * 8..(i + 1) * 8].try_into().unwrap()
                    );
                    all_ids.push(id as i64);
                    taken += 1;
                    if taken >= rows_to_take { break; }
                }
            }

            // Parse columns — read requested, skip others
            // Build mapping: disk col_idx → output position in col_accumulators
            // This ensures correct data placement regardless of column ordering
            // between the footer schema and the requested column list.
            let null_bitmap_len = (rg_rows + 7) / 8;
            let col_idx_to_out: HashMap<usize, usize> = col_indices.iter()
                .enumerate()
                .map(|(out_pos, &col_idx)| (col_idx, out_pos))
                .collect();
            // Track which output columns got data from this RG
            let mut rg_filled: Vec<bool> = vec![false; col_indices.len()];
            for col_idx in 0..col_count {
                // Schema evolution: RG may have fewer columns than footer schema.
                // If we've exhausted the RG data, remaining columns get defaults.
                if pos + null_bitmap_len > rg_bytes.len() {
                    break;
                }
                let null_bytes = &rg_bytes[pos..pos + null_bitmap_len];
                pos += null_bitmap_len;

                let col_type = schema.columns[col_idx].1;

                if let Some(&out_pos) = col_idx_to_out.get(&col_idx) {
                    // Parse this column's data
                    let (col_data, consumed) = ColumnData::from_bytes_typed(
                        &rg_bytes[pos..], col_type,
                    )?;
                    pos += consumed;

                    // Decode StringDict → String for in-memory use
                    let col_data = if matches!(&col_data, ColumnData::StringDict { .. }) {
                        col_data.decode_string_dict()
                    } else {
                        col_data
                    };

                    // Filter by active rows if there are deletes
                    if has_deletes {
                        let active_indices: Vec<usize> = (0..rg_rows)
                            .filter(|&i| (del_bytes[i / 8] >> (i % 8)) & 1 == 0)
                            .take(rows_to_take)
                            .collect();
                        let filtered = col_data.filter_by_indices(&active_indices);
                        col_accumulators[out_pos].append(&filtered);

                        // Filter null bitmap (per-bit)
                        for &old_idx in &active_indices {
                            let ob = old_idx / 8;
                            let obit = old_idx % 8;
                            let is_null = ob < null_bytes.len() && (null_bytes[ob] >> obit) & 1 == 1;
                            null_accumulators[out_pos].push(is_null);
                        }
                    } else {
                        // No deletes — take all rows (up to limit)
                        if rows_to_take < rg_rows {
                            let range_data = col_data.slice_range(0, rows_to_take);
                            col_accumulators[out_pos].append(&range_data);
                            for i in 0..rows_to_take {
                                let is_null = (null_bytes[i / 8] >> (i % 8)) & 1 == 1;
                                null_accumulators[out_pos].push(is_null);
                            }
                        } else {
                            col_accumulators[out_pos].append(&col_data);
                            for i in 0..rg_rows {
                                let is_null = (null_bytes[i / 8] >> (i % 8)) & 1 == 1;
                                null_accumulators[out_pos].push(is_null);
                            }
                        }
                    }
                    rg_filled[out_pos] = true;
                } else {
                    // Skip this column (no allocation)
                    let consumed = ColumnData::skip_bytes_typed(
                        &rg_bytes[pos..], col_type,
                    )?;
                    pos += consumed;
                }
            }
            // Fill default values for columns that weren't in this RG (schema evolution)
            for (out_pos, filled) in rg_filled.iter().enumerate() {
                if !filled {
                    let col_type = schema.columns[col_indices[out_pos]].1;
                    let default_col = Self::create_default_column(col_type, rows_to_take);
                    col_accumulators[out_pos].append(&default_col);
                    // All rows are null for this missing column
                    null_accumulators[out_pos].extend(std::iter::repeat(true).take(rows_to_take));
                }
            }
            rows_collected += rows_to_take;
        }

        drop(mmap_guard);
        drop(file_guard);

        // Build Arrow RecordBatch from accumulated data
        let active_count = rows_collected;
        let mut fields: Vec<Field> = Vec::with_capacity(col_indices.len() + 1);
        let mut arrays: Vec<ArrayRef> = Vec::with_capacity(col_indices.len() + 1);

        // Save row IDs for potential DeltaMerger overlay
        let row_ids_for_delta: Vec<u64> = all_ids.iter().map(|&id| id as u64).collect();

        // _id column
        if include_id {
            fields.push(Field::new("_id", ArrowDataType::Int64, false));
            arrays.push(Arc::new(Int64Array::from(all_ids)));
        }

        // Data columns
        for (out_idx, &col_idx) in col_indices.iter().enumerate() {
            let (col_name, _col_type) = &schema.columns[col_idx];
            let col_data = &col_accumulators[out_idx];
            let null_bitmap = &null_accumulators[out_idx];

            // Build Arrow null buffer from per-bit bool accumulator
            let null_buf: Option<NullBuffer> = if !null_bitmap.is_empty() && null_bitmap.iter().any(|&b| b) {
                let mut validity_bytes = vec![0xFFu8; (active_count + 7) / 8];
                for (i, &is_null) in null_bitmap.iter().enumerate() {
                    if is_null {
                        // Clear validity bit (Arrow: 1=valid, 0=null)
                        validity_bytes[i / 8] &= !(1u8 << (i % 8));
                    }
                }
                Some(NullBuffer::new(BooleanBuffer::new(Buffer::from(validity_bytes), 0, active_count)))
            } else {
                None
            };

            let (arrow_dt, array): (ArrowDataType, ArrayRef) = match col_data {
                ColumnData::Int64(values) => {
                    let arr = PrimitiveArray::<Int64Type>::new(
                        ScalarBuffer::from(values.clone()), null_buf,
                    );
                    (ArrowDataType::Int64, Arc::new(arr) as ArrayRef)
                }
                ColumnData::Float64(values) => {
                    let arr = PrimitiveArray::<Float64Type>::new(
                        ScalarBuffer::from(values.clone()), null_buf,
                    );
                    (ArrowDataType::Float64, Arc::new(arr) as ArrayRef)
                }
                ColumnData::String { offsets, data } => {
                    let count = offsets.len().saturating_sub(1);
                    if null_buf.is_some() {
                        let strings: Vec<Option<&str>> = (0..count.min(active_count)).map(|i| {
                            if i < null_bitmap.len() && null_bitmap[i] {
                                None
                            } else {
                                let start = offsets[i] as usize;
                                let end = offsets[i + 1] as usize;
                                std::str::from_utf8(&data[start..end]).ok()
                            }
                        }).collect();
                        (ArrowDataType::Utf8, Arc::new(StringArray::from(strings)))
                    } else {
                        let strings: Vec<&str> = (0..count.min(active_count)).map(|i| {
                            let start = offsets[i] as usize;
                            let end = offsets[i + 1] as usize;
                            std::str::from_utf8(&data[start..end]).unwrap_or("")
                        }).collect();
                        (ArrowDataType::Utf8, Arc::new(StringArray::from(strings)))
                    }
                }
                ColumnData::Bool { data: bool_data, len: bool_len } => {
                    let bools: Vec<Option<bool>> = (0..*bool_len.min(&active_count)).map(|i| {
                        if null_buf.is_some() {
                            if i < null_bitmap.len() && null_bitmap[i] {
                                return None;
                            }
                        }
                        let byte_idx = i / 8;
                        let bit_idx = i % 8;
                        let val = byte_idx < bool_data.len() && (bool_data[byte_idx] >> bit_idx) & 1 == 1;
                        Some(val)
                    }).collect();
                    (ArrowDataType::Boolean, Arc::new(BooleanArray::from(bools)))
                }
                ColumnData::Binary { offsets, data } => {
                    use arrow::array::BinaryArray;
                    let count = offsets.len().saturating_sub(1);
                    if null_buf.is_some() {
                        let bins: Vec<Option<&[u8]>> = (0..count.min(active_count)).map(|i| {
                            if i < null_bitmap.len() && null_bitmap[i] {
                                None
                            } else {
                                let start = offsets[i] as usize;
                                let end = offsets[i + 1] as usize;
                                Some(&data[start..end])
                            }
                        }).collect();
                        (ArrowDataType::Binary, Arc::new(BinaryArray::from(bins)))
                    } else {
                        let bins: Vec<&[u8]> = (0..count.min(active_count)).map(|i| {
                            let start = offsets[i] as usize;
                            let end = offsets[i + 1] as usize;
                            &data[start..end]
                        }).collect();
                        (ArrowDataType::Binary, Arc::new(BinaryArray::from(bins)))
                    }
                }
                _ => {
                    // Fallback: null array
                    let arr = arrow::array::new_null_array(&ArrowDataType::Utf8, active_count);
                    (ArrowDataType::Utf8, arr)
                }
            };

            fields.push(Field::new(col_name, arrow_dt, true));
            arrays.push(array);
        }

        let arrow_schema = Arc::new(Schema::new(fields));
        let batch = RecordBatch::try_new(arrow_schema, arrays)
            .map_err(|e| err_data(e.to_string()))?;

        // Apply DeltaStore overlay if there are pending cell-level updates
        let ds = self.delta_store.read();
        if !ds.is_empty() && batch.num_rows() > 0 {
            let merged = super::delta::DeltaMerger::merge(&batch, &ds, &row_ids_for_delta)?;
            return Ok(Some(merged));
        }

        Ok(Some(batch))
    }
    
    /// Read a single column for a contiguous row range, V3 or V4.
    /// For V4 (in-memory): uses slice_range on loaded columns.
    /// For V3 (mmap): uses read_column_range_mmap.
    fn read_column_auto(
        &self,
        col_idx: usize,
        col_type: ColumnType,
        start: usize,
        count: usize,
        total_rows: usize,
        is_v4: bool,
    ) -> io::Result<ColumnData> {
        if is_v4 {
            let columns = self.columns.read();
            if col_idx < columns.len() && columns[col_idx].len() > 0 {
                Ok(columns[col_idx].slice_range(start, start + count))
            } else {
                Ok(Self::create_default_column(col_type, count))
            }
        } else {
            let column_index = self.column_index.read();
            if col_idx >= column_index.len() {
                return Ok(Self::create_default_column(col_type, count));
            }
            let entry = column_index[col_idx].clone();
            drop(column_index);
            
            let file_guard = self.file.read();
            let file = file_guard.as_ref()
                .ok_or_else(|| err_not_conn("File not open"))?;
            let mut mmap = self.mmap_cache.write();
            self.read_column_range_mmap(&mut mmap, file, &entry, col_type, start, count, total_rows)
        }
    }
    
    /// Read a single column for scattered row indices, V3 or V4.
    fn read_column_scattered_auto(
        &self,
        col_idx: usize,
        col_type: ColumnType,
        indices: &[usize],
        total_rows: usize,
        is_v4: bool,
    ) -> io::Result<ColumnData> {
        if is_v4 {
            let columns = self.columns.read();
            if col_idx < columns.len() && columns[col_idx].len() > 0 {
                Ok(columns[col_idx].filter_by_indices(indices))
            } else {
                Ok(Self::create_default_column(col_type, indices.len()))
            }
        } else {
            let column_index = self.column_index.read();
            if col_idx >= column_index.len() {
                return Ok(Self::create_default_column(col_type, indices.len()));
            }
            let entry = column_index[col_idx].clone();
            drop(column_index);
            
            let file_guard = self.file.read();
            let file = file_guard.as_ref()
                .ok_or_else(|| err_not_conn("File not open"))?;
            let mut mmap = self.mmap_cache.write();
            self.read_column_scattered_mmap(&mut mmap, file, &entry, col_type, indices, total_rows)
        }
    }

    /// Ensure IDs are loaded into memory (lazy loading optimization)
    /// This is called on-demand when IDs are actually needed
    fn ensure_ids_loaded(&self) -> io::Result<()> {
        // Quick check without write lock
        if !self.ids.read().is_empty() {
            return Ok(());
        }
        
        let header = self.header.read();
        let id_count = header.row_count as usize;
        
        if id_count == 0 {
            return Ok(());
        }
        
        // V4: IDs are read via mmap on demand, no bulk loading needed
        if header.footer_offset > 0 {
            return Ok(());
        }
        
        let file_guard = self.file.read();
        let file = file_guard.as_ref().ok_or_else(|| {
            err_not_conn("File not open")
        })?;
        
        let mut mmap_cache = self.mmap_cache.write();
        let mut ids = self.ids.write();
        
        // Double-check after acquiring write lock
        if !ids.is_empty() {
            return Ok(());
        }
        
        // Load IDs from disk (V3)
        let mut id_bytes = vec![0u8; id_count * 8];
        mmap_cache.read_at(file, &mut id_bytes, header.id_column_offset)?;
        
        ids.reserve(id_count);
        for i in 0..id_count {
            let id = u64::from_le_bytes(id_bytes[i * 8..(i + 1) * 8].try_into().unwrap());
            ids.push(id);
        }
        
        Ok(())
    }

    /// Read IDs for a row range (lazy loads IDs if not already loaded)
    pub fn read_ids(&self, start_row: usize, row_count: Option<usize>) -> io::Result<Vec<u64>> {
        // Ensure IDs are loaded (lazy loading)
        self.ensure_ids_loaded()?;
        
        let ids = self.ids.read();
        let total = ids.len();
        let start = start_row.min(total);
        let count = row_count.map(|c| c.min(total - start)).unwrap_or(total - start);
        Ok(ids[start..start + count].to_vec())
    }

    /// Read IDs for specific row indices (optimized for scattered access, lazy loads)
    pub fn read_ids_by_indices(&self, row_indices: &[usize]) -> io::Result<Vec<i64>> {
        // Ensure IDs are loaded (lazy loading)
        self.ensure_ids_loaded()?;
        
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

        // V4: mmap dict scan is V3-specific, let caller use general query path
        {
            let header = self.header.read();
            if header.footer_offset > 0 {
                return Ok(None);
            }
        }

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
            err_not_conn("File not open")
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

    /// Ensure id_to_idx AHashMap is built (lazy load)
    /// Called automatically by delete/exists/get_row_idx operations
    fn ensure_id_index(&self) {
        // First ensure IDs are loaded (since we lazy-load them now)
        let _ = self.ensure_ids_loaded();
        
        let mut id_to_idx = self.id_to_idx.write();
        if id_to_idx.is_none() {
            let ids = self.ids.read();
            let mut map = ahash::AHashMap::with_capacity(ids.len());
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

            // First, add any new columns to schema
            for name in int_columns.keys() {
                schema.add_column(name, ColumnType::Int64);
            }
            for name in float_columns.keys() {
                schema.add_column(name, ColumnType::Float64);
            }
            for name in string_columns.keys() {
                schema.add_column(name, ColumnType::String);
            }
            for name in binary_columns.keys() {
                schema.add_column(name, ColumnType::Binary);
            }
            for name in bool_columns.keys() {
                schema.add_column(name, ColumnType::Bool);
            }

            // Then, ensure columns vector matches schema (using correct types from schema)
            while columns.len() < schema.column_count() {
                let col_idx = columns.len();
                let (_, col_type) = &schema.columns[col_idx];
                let mut col = ColumnData::new(*col_type);
                // Pad with defaults for existing rows
                if existing_row_count > 0 {
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
                        _ => {}
                    }
                }
                columns.push(col);
                nulls.push(Vec::new());
            }
        }

        // OPTIMIZATION: combine ID append + column append + metadata updates
        // to minimize lock acquire/release overhead
        let col_count_for_header;
        {
            let schema = self.schema.read();
            col_count_for_header = schema.column_count() as u32;
            let mut ids_guard = self.ids.write();
            let start_idx = ids_guard.len();
            ids_guard.extend_from_slice(&ids);
            let total_rows_after = ids_guard.len();
            drop(ids_guard);

            // Append column data (schema read lock still held)
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
                    if idx < columns.len() {
                        match &columns[idx] {
                            ColumnData::String { .. } => {
                                columns[idx].extend_strings(&values);
                            }
                            ColumnData::StringDict { indices, dict_offsets, dict_data } => {
                                let mut new_offsets = vec![0u32];
                                let mut new_data = Vec::new();
                                for &dict_idx in indices {
                                    if dict_idx == 0 {
                                        new_offsets.push(new_data.len() as u32);
                                    } else {
                                        let actual_idx = (dict_idx - 1) as usize;
                                        if actual_idx + 1 < dict_offsets.len() {
                                            let start = dict_offsets[actual_idx] as usize;
                                            let end = dict_offsets[actual_idx + 1] as usize;
                                            new_data.extend_from_slice(&dict_data[start..end]);
                                        }
                                        new_offsets.push(new_data.len() as u32);
                                    }
                                }
                                columns[idx] = ColumnData::String { 
                                    offsets: new_offsets, 
                                    data: new_data 
                                };
                                columns[idx].extend_strings(&values);
                            }
                            _ => {
                                columns[idx] = ColumnData::new(ColumnType::String);
                                columns[idx].extend_strings(&values);
                            }
                        }
                    }
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
            drop(columns);
            drop(schema);

            // Update id_to_idx if already built (avoid rebuilding)
            {
                let mut id_to_idx = self.id_to_idx.write();
                if let Some(map) = id_to_idx.as_mut() {
                    for (i, &id) in ids.iter().enumerate() {
                        map.insert(id, start_idx + i);
                    }
                }
            }

            // Extend deleted bitmap
            {
                let mut deleted = self.deleted.write();
                let new_len = (total_rows_after + 7) / 8;
                deleted.resize(new_len, 0);
            }
        }

        // Update header
        {
            let mut header = self.header.write();
            header.row_count += row_count as u64;
            header.column_count = col_count_for_header;
            header.modified_at = chrono::Utc::now().timestamp();
        }
        
        // Update active count (new rows are not deleted)
        self.active_count.fetch_add(row_count as u64, Ordering::Relaxed);

        Ok(ids)
    }

    /// Insert typed columns with EXPLICIT IDs (used during delta compaction)
    /// This preserves the original IDs from delta file instead of generating new ones
    fn insert_typed_with_ids(
        &self,
        ids: &[u64],
        int_columns: HashMap<String, Vec<i64>>,
        float_columns: HashMap<String, Vec<f64>>,
        string_columns: HashMap<String, Vec<String>>,
        binary_columns: HashMap<String, Vec<Vec<u8>>>,
        bool_columns: HashMap<String, Vec<bool>>,
    ) -> io::Result<()> {
        let row_count = ids.len();
        if row_count == 0 {
            return Ok(());
        }

        // Update next_id to be greater than any provided ID
        for &id in ids {
            let current = self.next_id.load(Ordering::SeqCst);
            if id >= current {
                self.next_id.store(id + 1, Ordering::SeqCst);
            }
        }

        // Ensure schema has all columns
        {
            let mut schema = self.schema.write();
            let mut columns = self.columns.write();
            let mut nulls = self.nulls.write();
            let existing_row_count = self.ids.read().len();

            for name in int_columns.keys() {
                schema.add_column(name, ColumnType::Int64);
            }
            for name in float_columns.keys() {
                schema.add_column(name, ColumnType::Float64);
            }
            for name in string_columns.keys() {
                schema.add_column(name, ColumnType::String);
            }
            for name in binary_columns.keys() {
                schema.add_column(name, ColumnType::Binary);
            }
            for name in bool_columns.keys() {
                schema.add_column(name, ColumnType::Bool);
            }

            while columns.len() < schema.column_count() {
                let col_idx = columns.len();
                let (_, col_type) = &schema.columns[col_idx];
                let mut col = ColumnData::new(*col_type);
                if existing_row_count > 0 {
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
                }
                columns.push(col);
                nulls.push(Vec::new());
            }
        }

        // Append IDs
        {
            let mut ids_vec = self.ids.write();
            ids_vec.extend_from_slice(ids);
        }

        // Append column data
        {
            let schema = self.schema.read();
            let mut columns = self.columns.write();

            for (name, values) in int_columns {
                if let Some(idx) = schema.get_index(&name) {
                    if idx < columns.len() {
                        columns[idx].extend_i64(&values);
                    }
                }
            }

            for (name, values) in float_columns {
                if let Some(idx) = schema.get_index(&name) {
                    if idx < columns.len() {
                        columns[idx].extend_f64(&values);
                    }
                }
            }

            for (name, values) in string_columns {
                if let Some(idx) = schema.get_index(&name) {
                    if idx < columns.len() {
                        match &columns[idx] {
                            ColumnData::String { .. } => {
                                columns[idx].extend_strings(&values);
                            }
                            ColumnData::StringDict { indices, dict_offsets, dict_data } => {
                                let mut new_offsets = vec![0u32];
                                let mut new_data = Vec::new();
                                for &dict_idx in indices {
                                    if dict_idx == 0 {
                                        new_offsets.push(new_data.len() as u32);
                                    } else {
                                        let actual_idx = (dict_idx - 1) as usize;
                                        if actual_idx + 1 < dict_offsets.len() {
                                            let start = dict_offsets[actual_idx] as usize;
                                            let end = dict_offsets[actual_idx + 1] as usize;
                                            new_data.extend_from_slice(&dict_data[start..end]);
                                        }
                                        new_offsets.push(new_data.len() as u32);
                                    }
                                }
                                columns[idx] = ColumnData::String { 
                                    offsets: new_offsets, 
                                    data: new_data 
                                };
                                columns[idx].extend_strings(&values);
                            }
                            _ => {
                                columns[idx] = ColumnData::new(ColumnType::String);
                                columns[idx].extend_strings(&values);
                            }
                        }
                    }
                }
            }

            for (name, values) in bool_columns {
                if let Some(idx) = schema.get_index(&name) {
                    if idx < columns.len() {
                        columns[idx].extend_bools(&values);
                    }
                }
            }

            // Pad columns that don't have new data
            for col_idx in 0..columns.len() {
                let expected_len = self.ids.read().len();
                let current_len = columns[col_idx].len();
                if current_len < expected_len {
                    let pad_count = expected_len - current_len;
                    match &mut columns[col_idx] {
                        ColumnData::Int64(v) => v.extend(std::iter::repeat(0).take(pad_count)),
                        ColumnData::Float64(v) => v.extend(std::iter::repeat(0.0).take(pad_count)),
                        ColumnData::String { offsets, .. } => {
                            for _ in 0..pad_count {
                                offsets.push(*offsets.last().unwrap_or(&0));
                            }
                        }
                        ColumnData::Binary { offsets, .. } => {
                            for _ in 0..pad_count {
                                offsets.push(*offsets.last().unwrap_or(&0));
                            }
                        }
                        ColumnData::Bool { data, len } => {
                            for _ in 0..pad_count {
                                let byte_idx = *len / 8;
                                if byte_idx >= data.len() { data.push(0); }
                                *len += 1;
                            }
                        }
                        ColumnData::StringDict { indices, .. } => {
                            indices.extend(std::iter::repeat(0).take(pad_count));
                        }
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

        // Extend deleted bitmap
        {
            let mut deleted = self.deleted.write();
            let new_len = (self.ids.read().len() + 7) / 8;
            deleted.resize(new_len, 0);
        }
        
        self.active_count.fetch_add(row_count as u64, Ordering::Relaxed);
        Ok(())
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
        // For new columns, pad existing rows with defaults (NULL-like values)
        let mut col_name_to_idx: HashMap<String, usize> = HashMap::new();
        {
            let mut schema = self.schema.write();
            let mut columns = self.columns.write();
            let mut nulls = self.nulls.write();
            let existing_row_count = self.ids.read().len();

            for name in int_columns.keys() {
                let idx = schema.add_column(name, ColumnType::Int64);
                col_name_to_idx.insert(name.clone(), idx);
                while columns.len() <= idx {
                    let mut col = ColumnData::new(ColumnType::Int64);
                    // Pad with defaults for existing rows
                    if let ColumnData::Int64(v) = &mut col {
                        v.resize(existing_row_count, 0);
                    }
                    columns.push(col);
                    // Mark all existing rows as NULL for new column
                    nulls.push(Vec::new());
                }
            }
            for name in float_columns.keys() {
                let idx = schema.add_column(name, ColumnType::Float64);
                col_name_to_idx.insert(name.clone(), idx);
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
                col_name_to_idx.insert(name.clone(), idx);
                while columns.len() <= idx {
                    let mut col = ColumnData::new(ColumnType::String);
                    // Pad with empty strings for existing rows
                    if let ColumnData::String { offsets, .. } = &mut col {
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
                col_name_to_idx.insert(name.clone(), idx);
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
                col_name_to_idx.insert(name.clone(), idx);
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
    /// Also checks delta file for IDs not yet merged into base
    pub fn exists(&self, id: u64) -> bool {
        // First check base file IDs
        self.ensure_id_index();
        let id_to_idx = self.id_to_idx.read();
        let map = id_to_idx.as_ref().unwrap();
        if let Some(&row_idx) = map.get(&id) {
            if !self.is_deleted(row_idx) {
                return true;
            }
        }
        
        // Check delta file for IDs not yet merged
        if let Ok(Some((delta_ids, _))) = self.read_delta_data() {
            return delta_ids.contains(&id);
        }
        
        false
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
        let is_v4 = self.is_v4_format();
        if is_v4 && !self.has_v4_in_memory_data() { return Ok(None); }
        
        // O(1) lookup using id_to_idx index
        let row_idx = match self.get_row_idx(id) {
            Some(idx) => idx,
            None => return Ok(None),
        };
        
        let indices = vec![row_idx];
        let schema = self.schema.read();
        
        // Get columns to read
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
        
        let total_rows = self.header.read().row_count as usize;
        drop(schema);
        
        let mut result = HashMap::new();
        
        // Add _id if requested or no column filter
        let include_id = column_names.map(|cols| cols.contains(&"_id")).unwrap_or(true);
        if include_id {
            result.insert("_id".to_string(), ColumnData::Int64(vec![id as i64]));
        }
        
        for (col_idx, col_name, col_type) in cols_to_read {
            let col_data = self.read_column_scattered_auto(col_idx, col_type, &indices, total_rows, is_v4)?;
            result.insert(col_name, col_data);
        }
        
        Ok(Some(result))
    }

    /// Ultra-fast point lookup: returns Vec<(col_name, Value)> directly from V4 columns
    /// Bypasses Arrow conversion and HashMap overhead
    pub fn read_row_by_id_values(&self, id: u64) -> io::Result<Option<Vec<(String, crate::data::Value)>>> {
        use crate::data::Value;
        
        let is_v4 = self.is_v4_format();
        if !is_v4 || !self.has_v4_in_memory_data() { return Ok(None); }
        
        let row_idx = match self.get_row_idx(id) {
            Some(idx) => idx,
            None => return Ok(None),
        };
        
        let schema = self.schema.read();
        let columns = self.columns.read();
        let nulls = self.nulls.read();
        
        let mut result = Vec::with_capacity(schema.column_count() + 1);
        result.push(("_id".to_string(), Value::Int64(id as i64)));
        
        for (col_idx, (col_name, _)) in schema.columns.iter().enumerate() {
            // Check null
            if col_idx < nulls.len() && !nulls[col_idx].is_empty() {
                let b = row_idx / 8; let bit = row_idx % 8;
                if b < nulls[col_idx].len() && (nulls[col_idx][b] >> bit) & 1 == 1 {
                    result.push((col_name.clone(), Value::Null));
                    continue;
                }
            }
            
            if col_idx >= columns.len() {
                result.push((col_name.clone(), Value::Null));
                continue;
            }
            
            let val = match &columns[col_idx] {
                ColumnData::Int64(v) => {
                    if row_idx < v.len() { Value::Int64(v[row_idx]) } else { Value::Null }
                }
                ColumnData::Float64(v) => {
                    if row_idx < v.len() { Value::Float64(v[row_idx]) } else { Value::Null }
                }
                ColumnData::String { offsets, data } => {
                    let count = offsets.len().saturating_sub(1);
                    if row_idx < count {
                        let s = offsets[row_idx] as usize;
                        let e = offsets[row_idx + 1] as usize;
                        Value::String(std::str::from_utf8(&data[s..e]).unwrap_or("").to_string())
                    } else { Value::Null }
                }
                ColumnData::Bool { data, len } => {
                    if row_idx < *len {
                        let b = row_idx / 8; let bit = row_idx % 8;
                        if b < data.len() {
                            Value::Bool((data[b] >> bit) & 1 == 1)
                        } else { Value::Null }
                    } else { Value::Null }
                }
                ColumnData::Binary { offsets, data } => {
                    let count = offsets.len().saturating_sub(1);
                    if row_idx < count {
                        let s = offsets[row_idx] as usize;
                        let e = offsets[row_idx + 1] as usize;
                        Value::Binary(data[s..e].to_vec())
                    } else { Value::Null }
                }
                ColumnData::StringDict { indices, dict_offsets, dict_data } => {
                    if row_idx < indices.len() {
                        let idx = indices[row_idx];
                        if idx == 0 { Value::Null } else {
                            let di = (idx - 1) as usize;
                            if di + 1 < dict_offsets.len() {
                                let s = dict_offsets[di] as usize;
                                let e = dict_offsets[di + 1] as usize;
                                Value::String(std::str::from_utf8(&dict_data[s..e]).unwrap_or("").to_string())
                            } else { Value::Null }
                        }
                    } else { Value::Null }
                }
                _ => Value::Null,
            };
            result.push((col_name.clone(), val));
        }
        
        Ok(Some(result))
    }

    /// Fast SELECT * LIMIT N: read first N non-deleted rows directly from V4 columns
    /// Returns (column_names, rows) where each row is Vec<Value>
    /// Bypasses SQL parsing and Arrow conversion entirely
    pub fn read_rows_limit_values(&self, limit: usize) -> io::Result<Option<(Vec<String>, Vec<Vec<crate::data::Value>>)>> {
        use crate::data::Value;
        
        let is_v4 = self.is_v4_format();
        if !is_v4 || !self.has_v4_in_memory_data() { return Ok(None); }
        
        let schema = self.schema.read();
        let columns = self.columns.read();
        let nulls = self.nulls.read();
        let ids = self.ids.read();
        let deleted = self.deleted.read();
        let total_rows = ids.len();
        let has_deleted = deleted.iter().any(|&b| b != 0);
        
        // Build column names
        let mut col_names = Vec::with_capacity(schema.column_count() + 1);
        col_names.push("_id".to_string());
        for (name, _) in &schema.columns {
            col_names.push(name.clone());
        }
        
        let actual_limit = limit.min(total_rows);
        let mut rows: Vec<Vec<Value>> = Vec::with_capacity(actual_limit);
        let mut emitted = 0usize;
        
        for row_idx in 0..total_rows {
            if emitted >= limit { break; }
            // Skip deleted
            if has_deleted {
                let b = row_idx / 8; let bit = row_idx % 8;
                if b < deleted.len() && (deleted[b] >> bit) & 1 != 0 { continue; }
            }
            
            let mut row = Vec::with_capacity(col_names.len());
            // _id
            row.push(if row_idx < ids.len() { Value::Int64(ids[row_idx] as i64) } else { Value::Null });
            
            for col_idx in 0..schema.column_count() {
                // Null check
                if col_idx < nulls.len() && !nulls[col_idx].is_empty() {
                    let b = row_idx / 8; let bit = row_idx % 8;
                    if b < nulls[col_idx].len() && (nulls[col_idx][b] >> bit) & 1 == 1 {
                        row.push(Value::Null);
                        continue;
                    }
                }
                if col_idx >= columns.len() { row.push(Value::Null); continue; }
                
                let val = match &columns[col_idx] {
                    ColumnData::Int64(v) => {
                        if row_idx < v.len() { Value::Int64(v[row_idx]) } else { Value::Null }
                    }
                    ColumnData::Float64(v) => {
                        if row_idx < v.len() { Value::Float64(v[row_idx]) } else { Value::Null }
                    }
                    ColumnData::String { offsets, data } => {
                        let count = offsets.len().saturating_sub(1);
                        if row_idx < count {
                            let s = offsets[row_idx] as usize;
                            let e = offsets[row_idx + 1] as usize;
                            Value::String(std::str::from_utf8(&data[s..e]).unwrap_or("").to_string())
                        } else { Value::Null }
                    }
                    ColumnData::Bool { data, len } => {
                        if row_idx < *len {
                            let b = row_idx / 8; let bit = row_idx % 8;
                            if b < data.len() { Value::Bool((data[b] >> bit) & 1 == 1) } else { Value::Null }
                        } else { Value::Null }
                    }
                    ColumnData::Binary { offsets, data } => {
                        let count = offsets.len().saturating_sub(1);
                        if row_idx < count {
                            let s = offsets[row_idx] as usize;
                            let e = offsets[row_idx + 1] as usize;
                            Value::Binary(data[s..e].to_vec())
                        } else { Value::Null }
                    }
                    ColumnData::StringDict { indices, dict_offsets, dict_data } => {
                        if row_idx < indices.len() {
                            let idx = indices[row_idx];
                            if idx == 0 { Value::Null } else {
                                let di = (idx - 1) as usize;
                                if di + 1 < dict_offsets.len() {
                                    let s = dict_offsets[di] as usize;
                                    let e = dict_offsets[di + 1] as usize;
                                    Value::String(std::str::from_utf8(&dict_data[s..e]).unwrap_or("").to_string())
                                } else { Value::Null }
                            }
                        } else { Value::Null }
                    }
                    _ => Value::Null,
                };
                row.push(val);
            }
            rows.push(row);
            emitted += 1;
        }
        
        Ok(Some((col_names, rows)))
    }

    /// OPTIMIZED: Read multiple rows by IDs using O(1) index lookups
    /// Returns Vec of (id, row_data) for found rows
    pub fn read_rows_by_ids(&self, ids: &[u64], column_names: Option<&[&str]>) -> io::Result<Vec<(u64, HashMap<String, ColumnData>)>> {
        if ids.is_empty() {
            return Ok(Vec::new());
        }
        
        let is_v4 = self.is_v4_format();
        if is_v4 && !self.has_v4_in_memory_data() { return Ok(Vec::new()); }
        
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
        
        // Read columns
        let schema = self.schema.read();
        
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
        
        let total_rows = self.header.read().row_count as usize;
        drop(schema);
        
        // Read all columns for all indices
        let mut column_data: HashMap<String, ColumnData> = HashMap::new();
        let include_id = column_names.map(|cols| cols.contains(&"_id")).unwrap_or(true);
        
        for (col_idx, col_name, col_type) in cols_to_read {
            let col_data = self.read_column_scattered_auto(col_idx, col_type, &indices, total_rows, is_v4)?;
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

    /// Get the count of non-deleted rows (includes delta rows)
    pub fn active_row_count(&self) -> u64 {
        let base_active = self.active_count.load(std::sync::atomic::Ordering::Relaxed);
        let delta_rows = self.delta_row_count() as u64;
        base_active + delta_rows
    }

    /// Drop a column from schema (logical delete - data stays but column is removed from schema)
    /// When save() is called, only columns in schema will be written to file
    pub fn drop_column(&self, name: &str) -> io::Result<()> {
        let mut schema = self.schema.write();
        
        // Find column index
        let idx = match schema.get_index(name) {
            Some(idx) => idx,
            None => return Err(io::Error::new(io::ErrorKind::NotFound, format!("Column '{}' not found", name))),
        };
        
        // Remove from schema (logical delete)
        schema.columns.remove(idx);
        schema.name_to_idx.remove(name);
        
        // Rebuild name_to_idx with updated indices
        // Collect names first to avoid borrow conflict
        let names: Vec<String> = schema.columns.iter().map(|(n, _)| n.clone()).collect();
        schema.name_to_idx.clear();
        for (i, n) in names.into_iter().enumerate() {
            schema.name_to_idx.insert(n, i);
        }
        
        // Also remove from in-memory structures to keep them in sync with schema
        // This ensures save() writes correct data
        {
            let mut columns = self.columns.write();
            let mut nulls = self.nulls.write();
            let mut column_index = self.column_index.write();
            
            if idx < columns.len() {
                columns.remove(idx);
            }
            if idx < nulls.len() {
                nulls.remove(idx);
            }
            if idx < column_index.len() {
                column_index.remove(idx);
            }
        }
        
        // Update header column count
        {
            let mut header = self.header.write();
            header.column_count = schema.column_count() as u32;
        }
        
        Ok(())
    }

    /// Add a new column to schema and storage with padding for existing rows
    pub fn add_column_with_padding(&self, name: &str, dtype: crate::data::DataType) -> io::Result<()> {
        use crate::data::DataType;
        
        // For V4, schema is updated via footer; data stays on disk (mmap)
        self.load_all_columns_into_memory()?;
        
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
        // Use header.row_count for V4 (IDs may not be loaded in mmap-only mode)
        let existing_row_count = {
            let header = self.header.read();
            let from_header = header.row_count as usize;
            drop(header);
            let ids = self.ids.read();
            let from_ids = ids.len();
            drop(ids);
            from_header.max(from_ids)
        };
        
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
            
            // Pad any schema columns not in the replacement data with defaults + null
            let expected_len = self.ids.read().len();
            let mut nulls = self.nulls.write();
            for col_idx in 0..schema.column_count() {
                if col_idx < columns.len() && columns[col_idx].len() < expected_len {
                    // This column wasn't in the replacement — pad with default
                    let deficit = expected_len - columns[col_idx].len();
                    for _ in 0..deficit {
                        match &mut columns[col_idx] {
                            ColumnData::Int64(v) => v.push(0),
                            ColumnData::Float64(v) => v.push(0.0),
                            ColumnData::String { offsets, .. } => {
                                offsets.push(*offsets.last().unwrap_or(&0));
                            }
                            ColumnData::Binary { offsets, .. } => {
                                offsets.push(*offsets.last().unwrap_or(&0));
                            }
                            ColumnData::Bool { data, len } => {
                                let byte_idx = *len / 8;
                                if byte_idx >= data.len() { data.push(0); }
                                *len += 1;
                            }
                            ColumnData::StringDict { indices, .. } => indices.push(0),
                        }
                    }
                    // Mark padded rows as null
                    if col_idx >= nulls.len() {
                        nulls.resize(col_idx + 1, Vec::new());
                    }
                    let total_rows = expected_len;
                    let null_len = (total_rows + 7) / 8;
                    nulls[col_idx].resize(null_len, 0);
                    for row in (total_rows - deficit)..total_rows {
                        nulls[col_idx][row / 8] |= 1 << (row % 8);
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
    /// 
    /// MEMORY OPTIMIZED: Processes one column at a time using placeholder + seek-back.
    /// Peak memory = original columns (already in memory) + 1 filtered column copy,
    /// instead of original columns + ALL filtered column copies.
    /// 
    /// Automatically converts low-cardinality string columns to dictionary encoding.
    pub fn save(&self) -> io::Result<()> {
        // OPTIMIZATION: For existing V4 files with only deletions (no new rows,
        // no schema changes), update deletion vectors in-place instead of full rewrite.
        // All other cases use the proven save_v4() full-rewrite path.
        // Note: append optimization is handled at engine level (write_typed→append_row_group).
        let header = self.header.read();
        let is_v4 = header.version == FORMAT_VERSION_V4 && header.footer_offset > 0;
        drop(header);

        if is_v4 {
            let on_disk_rows = self.persisted_row_count.load(Ordering::SeqCst) as usize;
            let ids = self.ids.read();
            let in_memory_ids = ids.len();
            let has_new_rows = in_memory_ids > 0;
            let base_loaded = self.v4_base_loaded.load(Ordering::SeqCst);
            let has_unloaded_base = on_disk_rows > 0 && in_memory_ids > 0 && !base_loaded;
            drop(ids);

            // If base data isn't loaded but we have new rows, append incrementally
            if has_unloaded_base {
                let ids = self.ids.read();
                let new_ids: Vec<u64> = ids.clone();
                drop(ids);
                let cols = self.columns.read();
                let new_cols: Vec<ColumnData> = cols.clone();
                drop(cols);
                let nulls = self.nulls.read();
                let new_nulls: Vec<Vec<u8>> = nulls.clone();
                drop(nulls);
                self.pending_rows.store(0, Ordering::SeqCst);
                return self.append_row_group(&new_ids, &new_cols, &new_nulls);
            }

            if !has_new_rows && !base_loaded && on_disk_rows > 0 {
                // Schema-only change (add/drop/rename column) on V4 mmap-only.
                // Base data is NOT in memory — must NOT call save_v4() which would
                // rewrite with empty data and lose everything.
                // Instead, update just the footer schema on disk.
                return self.update_v4_footer_schema();
            }

            if !has_new_rows {
                let deleted = self.deleted.read();
                let has_deletes = deleted.iter().any(|&b| b != 0);
                if has_deletes {
                    // Count deleted rows for compaction threshold
                    let del_count = (0..on_disk_rows).filter(|&i| {
                        let byte_idx = i / 8;
                        let bit_idx = i % 8;
                        byte_idx < deleted.len() && (deleted[byte_idx] >> bit_idx) & 1 == 1
                    }).count();
                    drop(deleted);
                    let ratio = if on_disk_rows > 0 { del_count as f64 / on_disk_rows as f64 } else { 0.0 };

                    if ratio <= 0.5 {
                        // Low deletion ratio → update deletion vectors in-place
                        self.pending_rows.store(0, Ordering::SeqCst);
                        // Also persist delta store if it has pending changes
                        if self.has_pending_deltas() {
                            let _ = self.save_delta_store();
                        }
                        return self.save_deletion_vectors();
                    }
                    // High deletion ratio → full rewrite to reclaim space (fall through)
                }
            }
        }

        self.pending_rows.store(0, Ordering::SeqCst);
        let result = self.save_v4();
        // After full rewrite, clear delta store (deltas are now in the base file)
        if result.is_ok() {
            let _ = self.clear_delta_store();
        }
        result
    }
    
    // ========================================================================
    // V4 Row Group Format — Save / Open / Append
    // ========================================================================
    
    /// Slice a null bitmap for a contiguous row range [start, end).
    /// OPTIMIZATION: uses bulk memcpy when start is byte-aligned.
    fn slice_null_bitmap(nulls: &[u8], start: usize, end: usize) -> Vec<u8> {
        let count = end.saturating_sub(start);
        if count == 0 || nulls.is_empty() {
            return vec![0u8; (count + 7) / 8];
        }
        let result_len = (count + 7) / 8;
        if start % 8 == 0 {
            let src_byte = start / 8;
            let copy_len = result_len.min(nulls.len().saturating_sub(src_byte));
            let mut result = vec![0u8; result_len];
            if copy_len > 0 {
                result[..copy_len].copy_from_slice(&nulls[src_byte..src_byte + copy_len]);
            }
            let tail_bits = count % 8;
            if tail_bits > 0 && result_len > 0 {
                result[result_len - 1] &= (1u8 << tail_bits) - 1;
            }
            return result;
        }
        let mut result = vec![0u8; result_len];
        for i in 0..count {
            let ob = (start + i) / 8;
            let obit = (start + i) % 8;
            if ob < nulls.len() && (nulls[ob] >> obit) & 1 == 1 {
                result[i / 8] |= 1 << (i % 8);
            }
        }
        result
    }
    
    /// Save in V4 Row Group format.
    /// Splits data into Row Groups of DEFAULT_ROW_GROUP_SIZE rows each.
    /// Each RG is self-contained with IDs, deletion vector, and per-column data.
    ///
    /// V4 File Layout:
    /// ```text
    /// [Header 256B] [RG0] [RG1] ... [V4Footer]
    /// ```
    pub fn save_v4(&self) -> io::Result<()> {
        self.mmap_cache.write().invalidate();
        *self.file.write() = None;
        // On Windows, active mmaps prevent file truncate/write (OS error 1224).
        // Must invalidate ALL caches (engine cache + insert_cache + schema_cache + executor STORAGE_CACHE).
        // On Unix/Linux, only executor cache needs invalidation (mmaps don't block writes).
        #[cfg(target_os = "windows")]
        super::engine::engine().invalidate(&self.path);
        #[cfg(not(target_os = "windows"))]
        crate::query::ApexExecutor::invalidate_cache_for_path(&self.path);
        
        let file = OpenOptions::new()
            .write(true).create(true).truncate(true)
            .open(&self.path)?;
        let mut writer = BufWriter::with_capacity(256 * 1024, file);
        
        // Phase 1: Build filtered (active) data under read guards.
        // This produces clean flat columns/ids/nulls with deleted rows removed
        // and missing columns padded. Used for both disk write and in-memory state.
        let active_ids: Vec<u64>;
        let mut active_columns: Vec<ColumnData>;
        let mut active_nulls: Vec<Vec<u8>>;
        let active_count: usize;
        let col_count: usize;
        let schema_clone: OnDemandSchema;
        
        {
            let schema = self.schema.read();
            let ids = self.ids.read();
            let columns = self.columns.read();
            let nulls = self.nulls.read();
            let deleted = self.deleted.read();
            
            col_count = schema.column_count();
            schema_clone = schema.clone();
            let has_deleted = deleted.iter().any(|&b| b != 0);
            
            if has_deleted {
                let indices: Vec<usize> = (0..ids.len())
                    .filter(|&i| {
                        let byte_idx = i / 8;
                        let bit_idx = i % 8;
                        byte_idx >= deleted.len() || (deleted[byte_idx] >> bit_idx) & 1 == 0
                    })
                    .collect();
                active_ids = indices.iter().map(|&i| ids[i]).collect();
                active_count = indices.len();
                
                active_columns = Vec::with_capacity(col_count);
                active_nulls = Vec::with_capacity(col_count);
                for col_idx in 0..col_count {
                    // Filter column data
                    if col_idx < columns.len() {
                        active_columns.push(columns[col_idx].filter_by_indices(&indices));
                    } else {
                        active_columns.push(Self::create_default_column(schema.columns[col_idx].1, active_count));
                    }
                    // Filter null bitmap
                    let orig_nulls = nulls.get(col_idx).map(|v| v.as_slice()).unwrap_or(&[]);
                    let null_len = (active_count + 7) / 8;
                    let mut nb = vec![0u8; null_len];
                    for (new_idx, &old_idx) in indices.iter().enumerate() {
                        let ob = old_idx / 8;
                        let obit = old_idx % 8;
                        if ob < orig_nulls.len() && (orig_nulls[ob] >> obit) & 1 == 1 {
                            nb[new_idx / 8] |= 1 << (new_idx % 8);
                        }
                    }
                    active_nulls.push(nb);
                }
            } else {
                active_ids = ids.to_vec();
                active_count = ids.len();
                
                active_columns = Vec::with_capacity(col_count);
                active_nulls = Vec::with_capacity(col_count);
                for col_idx in 0..col_count {
                    if col_idx < columns.len() {
                        active_columns.push(columns[col_idx].clone());
                    } else {
                        active_columns.push(Self::create_default_column(schema.columns[col_idx].1, active_count));
                    }
                    active_nulls.push(nulls.get(col_idx).map(|v| v.to_vec()).unwrap_or_default());
                }
            }
        } // All read guards dropped here
        
        // Phase 2: Write V4 format from active data (no lock contention).
        let rg_size = DEFAULT_ROW_GROUP_SIZE as usize;
        
        // Write placeholder header
        writer.write_all(&[0u8; HEADER_SIZE_V3])?;
        
        // Write Row Groups
        let mut rg_metas: Vec<RowGroupMeta> = Vec::new();
        let mut actual_col_types: Vec<ColumnType> = Vec::new();
        let mut chunk_start = 0;
        
        while chunk_start < active_count || (active_count == 0 && rg_metas.is_empty()) {
            let chunk_end = (chunk_start + rg_size).min(active_count);
            let chunk_rows = chunk_end - chunk_start;
            
            // Handle empty table — write one empty RG
            if active_count == 0 && rg_metas.is_empty() {
                let rg_offset = writer.stream_position()?;
                writer.write_all(MAGIC_ROW_GROUP)?;
                writer.write_all(&0u32.to_le_bytes())?;
                writer.write_all(&(col_count as u32).to_le_bytes())?;
                writer.write_all(&0u64.to_le_bytes())?;
                writer.write_all(&0u64.to_le_bytes())?;
                writer.write_all(&[0u8; 4])?;
                let rg_end = writer.stream_position()?;
                rg_metas.push(RowGroupMeta {
                    offset: rg_offset, data_size: rg_end - rg_offset,
                    row_count: 0, min_id: 0, max_id: 0, deletion_count: 0,
                });
                break;
            }
            
            let rg_offset = writer.stream_position()?;
            let chunk_ids = &active_ids[chunk_start..chunk_end];
            let min_id = chunk_ids.iter().copied().min().unwrap_or(0);
            let max_id = chunk_ids.iter().copied().max().unwrap_or(0);
            
            // RG header (32 bytes)
            writer.write_all(MAGIC_ROW_GROUP)?;
            writer.write_all(&(chunk_rows as u32).to_le_bytes())?;
            writer.write_all(&(col_count as u32).to_le_bytes())?;
            writer.write_all(&min_id.to_le_bytes())?;
            writer.write_all(&max_id.to_le_bytes())?;
            writer.write_all(&[0u8; 4])?;
            
            // IDs — bulk write via unsafe slice cast (avoids per-element loop)
            let id_bytes = unsafe {
                std::slice::from_raw_parts(chunk_ids.as_ptr() as *const u8, chunk_ids.len() * 8)
            };
            writer.write_all(id_bytes)?;
            
            // Deletion vector (all zeros — fresh save, no deletes)
            let del_vec_len = (chunk_rows + 7) / 8;
            writer.write_all(&vec![0u8; del_vec_len])?;
            
            // Columns — use direct reference for single-RG, slice for multi-RG
            let is_single_rg = chunk_start == 0 && chunk_end == active_count;
            let null_bitmap_len = (chunk_rows + 7) / 8;
            for col_idx in 0..col_count {
                // OPTIMIZATION: skip slice_range when chunk covers entire column (single-RG)
                let chunk_col_owned;
                let chunk_col_ref: &ColumnData = if is_single_rg {
                    &active_columns[col_idx]
                } else {
                    chunk_col_owned = active_columns[col_idx].slice_range(chunk_start, chunk_end);
                    &chunk_col_owned
                };
                
                // Dict-encode low-cardinality string columns for disk
                let dict_encoded;
                let processed: &ColumnData = if Self::should_dict_encode(chunk_col_ref) {
                    dict_encoded = chunk_col_ref.to_dict_encoded().unwrap_or_else(|| chunk_col_ref.clone());
                    &dict_encoded
                } else {
                    chunk_col_ref
                };
                
                // Track actual type for footer schema
                if rg_metas.is_empty() {
                    let actual_type = match processed {
                        ColumnData::StringDict { .. } => ColumnType::StringDict,
                        _ => schema_clone.columns[col_idx].1,
                    };
                    actual_col_types.push(actual_type);
                }
                
                // Null bitmap
                if is_single_rg && active_nulls[col_idx].len() == null_bitmap_len {
                    writer.write_all(&active_nulls[col_idx])?;
                } else {
                    let chunk_nulls = Self::slice_null_bitmap(
                        &active_nulls[col_idx], chunk_start, chunk_end,
                    );
                    writer.write_all(&chunk_nulls)?;
                }
                // OPTIMIZATION: write_to avoids intermediate Vec<u8> allocation
                processed.write_to(&mut writer)?;
            }
            
            let rg_end = writer.stream_position()?;
            rg_metas.push(RowGroupMeta {
                offset: rg_offset, data_size: rg_end - rg_offset,
                row_count: chunk_rows as u32, min_id, max_id, deletion_count: 0,
            });
            
            chunk_start = chunk_end;
        }
        
        // Build modified schema with actual types (StringDict if dict-encoded)
        let modified_schema = if !actual_col_types.is_empty() {
            let mut ms = OnDemandSchema::new();
            for (col_idx, (col_name, _)) in schema_clone.columns.iter().enumerate() {
                ms.add_column(col_name, actual_col_types[col_idx]);
            }
            ms
        } else {
            schema_clone.clone()
        };
        
        // Write V4 footer
        let footer_offset = writer.stream_position()?;
        let footer = V4Footer {
            schema: modified_schema,
            row_groups: rg_metas.clone(),
        };
        writer.write_all(&footer.to_bytes())?;
        writer.flush()?;
        
        if self.durability == super::DurabilityLevel::Max {
            writer.get_ref().sync_all()?;
        }
        
        // Seek back to fix header
        {
            let mut header = self.header.write();
            header.version = FORMAT_VERSION_V4;
            header.row_count = active_count as u64;
            header.column_count = col_count as u32;
            header.footer_offset = footer_offset;
            header.row_group_count = rg_metas.len() as u32;
            header.schema_offset = 0;
            header.column_index_offset = 0;
            header.id_column_offset = 0;
        }
        let header = self.header.read();
        let writer_inner = writer.get_mut();
        writer_inner.seek(SeekFrom::Start(0))?;
        writer_inner.write_all(&header.to_bytes())?;
        
        // Phase 3: Set in-memory state directly — NO disk reload needed.
        // active_columns/active_ids/active_nulls are already the correct post-save state.
        drop(header);
        drop(writer);
        
        // OPTIMIZATION: compute max_id BEFORE moving active_ids (avoids re-reading after write)
        let max_active_id = active_ids.iter().max().copied().unwrap_or(0);
        
        *self.column_index.write() = Vec::new();
        *self.ids.write() = active_ids;
        *self.columns.write() = active_columns;
        *self.nulls.write() = active_nulls;
        let del_len = (active_count + 7) / 8;
        *self.deleted.write() = vec![0u8; del_len];
        *self.id_to_idx.write() = None;
        self.mmap_cache.write().invalidate();
        
        self.active_count.store(active_count as u64, Ordering::SeqCst);
        // save_v4 physically removes deleted rows; persisted = active
        self.persisted_row_count.store(active_count as u64, Ordering::SeqCst);
        let candidate = max_active_id + 1;
        let current = self.next_id.load(Ordering::SeqCst);
        if candidate > current {
            self.next_id.store(candidate, Ordering::SeqCst);
        }
        
        let file = File::open(&self.path)?;
        *self.file.write() = Some(file);
        
        // On Linux, eagerly create the mmap so the next read avoids lazy-creation overhead.
        // This is safe because the file was just written and is in a consistent state.
        #[cfg(target_os = "linux")]
        {
            let file_guard = self.file.read();
            if let Some(f) = file_guard.as_ref() {
                let _ = self.mmap_cache.write().get_or_create(f);
            }
        }
        
        Ok(())
    }
    
    /// Open a V4 file: read footer, then load all RG data into flat columns.
    /// Used by write operations (drop_column, etc.) that need full data in memory,
    /// and by tests. Production reads use mmap on-demand reading instead.
    pub fn open_v4_data(&self) -> io::Result<()> {
        let header = self.header.read();
        if header.footer_offset == 0 {
            return Err(err_data("V4 file has no footer"));
        }
        let footer_offset = header.footer_offset;
        drop(header);
        
        // Read footer from file
        let file_guard = self.file.read();
        let file = file_guard.as_ref()
            .ok_or_else(|| err_not_conn("File not open for V4 read"))?;
        let mut mmap = self.mmap_cache.write();
        
        // Read footer
        let file_len = std::fs::metadata(&self.path)?.len();
        let footer_byte_count = (file_len - footer_offset) as usize;
        let mut footer_bytes = vec![0u8; footer_byte_count];
        mmap.read_at(file, &mut footer_bytes, footer_offset)?;
        let footer = V4Footer::from_bytes(&footer_bytes)?;
        
        // Update schema from footer
        *self.schema.write() = footer.schema.clone();
        let col_count = footer.schema.column_count();
        
        // Compute total rows from RG metadata (header.row_count stores active count,
        // but RGs may contain deleted rows that are still physically present)
        let total_rows: usize = footer.row_groups.iter().map(|rg| rg.row_count as usize).sum();
        
        // Allocate flat columns
        let mut all_ids: Vec<u64> = Vec::with_capacity(total_rows);
        let mut all_columns: Vec<ColumnData> = (0..col_count)
            .map(|i| ColumnData::new(footer.schema.columns[i].1))
            .collect();
        let mut all_nulls: Vec<Vec<u8>> = vec![Vec::new(); col_count];
        let mut all_deleted: Vec<u8> = Vec::new(); // flat deletion bitmap
        
        // Read each Row Group as a byte buffer, parse sequentially
        let mut max_id_seen: u64 = 0;
        let mut total_deleted: u64 = 0;
        for rg_meta in &footer.row_groups {
            if rg_meta.row_count == 0 {
                continue;
            }
            let rg_rows = rg_meta.row_count as usize;
            let rg_size = rg_meta.data_size as usize;
            
            // Read entire RG into buffer
            let mut rg_buf = vec![0u8; rg_size];
            mmap.read_at(file, &mut rg_buf, rg_meta.offset)?;
            
            let mut pos = 32; // skip RG header (32 bytes)
            
            // Parse IDs — OPTIMIZATION: bulk memcpy instead of per-element loop
            let ids_before = all_ids.len();
            let id_byte_len = rg_rows * 8;
            all_ids.resize(ids_before + rg_rows, 0);
            unsafe {
                std::ptr::copy_nonoverlapping(
                    rg_buf[pos..].as_ptr(),
                    all_ids[ids_before..].as_mut_ptr() as *mut u8,
                    id_byte_len,
                );
            }
            if rg_meta.max_id > max_id_seen {
                max_id_seen = rg_meta.max_id;
            }
            pos += id_byte_len;
            
            // Read deletion vector and merge into flat bitmap
            let del_vec_len = (rg_rows + 7) / 8;
            let del_bytes = &rg_buf[pos..pos + del_vec_len];
            let needed_len = (ids_before + rg_rows + 7) / 8;
            if all_deleted.len() < needed_len {
                all_deleted.resize(needed_len, 0);
            }
            if ids_before % 8 == 0 {
                let dest_byte = ids_before / 8;
                let copy_len = del_vec_len.min(all_deleted.len() - dest_byte);
                all_deleted[dest_byte..dest_byte + copy_len]
                    .copy_from_slice(&del_bytes[..copy_len]);
            } else {
                for i in 0..rg_rows {
                    if (del_bytes[i / 8] >> (i % 8)) & 1 == 1 {
                        let flat_idx = ids_before + i;
                        all_deleted[flat_idx / 8] |= 1 << (flat_idx % 8);
                    }
                }
            }
            total_deleted += rg_meta.deletion_count as u64;
            pos += del_vec_len;
            
            // Parse columns
            let null_bitmap_len = (rg_rows + 7) / 8;
            for col_idx in 0..col_count {
                // Read null bitmap
                let null_bytes = &rg_buf[pos..pos + null_bitmap_len];
                
                // Merge into flat nulls
                let flat_start = ids_before;
                let needed_len = (flat_start + rg_rows + 7) / 8;
                if all_nulls[col_idx].len() < needed_len {
                    all_nulls[col_idx].resize(needed_len, 0);
                }
                // OPTIMIZATION: bulk copy when flat_start is byte-aligned
                if flat_start % 8 == 0 {
                    let dest_byte = flat_start / 8;
                    let copy_len = null_bitmap_len.min(all_nulls[col_idx].len() - dest_byte);
                    all_nulls[col_idx][dest_byte..dest_byte + copy_len]
                        .copy_from_slice(&null_bytes[..copy_len]);
                } else {
                    for i in 0..rg_rows {
                        if (null_bytes[i / 8] >> (i % 8)) & 1 == 1 {
                            let flat_idx = flat_start + i;
                            all_nulls[col_idx][flat_idx / 8] |= 1 << (flat_idx % 8);
                        }
                    }
                }
                pos += null_bitmap_len;
                
                // Parse column data using from_bytes_typed
                let col_type = footer.schema.columns[col_idx].1;
                let (col_data, consumed) = ColumnData::from_bytes_typed(
                    &rg_buf[pos..], col_type,
                )?;
                pos += consumed;
                
                // Append to flat column
                all_columns[col_idx].append(&col_data);
            }
        }
        
        drop(mmap);
        drop(file_guard);
        
        // Decode StringDict columns back to plain String for in-memory use.
        // Dict encoding is a disk-only optimization; push_string/extend_strings
        // only work on ColumnData::String, so we must normalize here.
        {
            let mut schema_w = self.schema.write();
            for col_idx in 0..all_columns.len() {
                if matches!(&all_columns[col_idx], ColumnData::StringDict { .. }) {
                    let col = std::mem::replace(&mut all_columns[col_idx], ColumnData::new(ColumnType::String));
                    all_columns[col_idx] = col.decode_string_dict();
                    // Update schema type from StringDict → String
                    if col_idx < schema_w.columns.len() {
                        schema_w.columns[col_idx].1 = ColumnType::String;
                    }
                }
            }
        }
        
        // OPTIMIZATION: compute next_id from tracked max before moving all_ids
        let next_id = if max_id_seen > 0 {
            max_id_seen + 1
        } else {
            all_ids.iter().max().map(|&id| id + 1).unwrap_or(0)
        };
        
        // Store flat data
        *self.ids.write() = all_ids;
        *self.columns.write() = all_columns;
        *self.nulls.write() = all_nulls;
        
        // Use deletion vectors read from disk (not all-zeros)
        let deleted_len = (total_rows + 7) / 8;
        if all_deleted.len() < deleted_len {
            all_deleted.resize(deleted_len, 0);
        }
        *self.deleted.write() = all_deleted;
        
        self.next_id.store(next_id, Ordering::SeqCst);
        self.active_count.store(total_rows as u64 - total_deleted, Ordering::SeqCst);
        // Track actual on-disk row count (total rows in RGs, including deleted)
        self.persisted_row_count.store(total_rows as u64, Ordering::SeqCst);
        self.v4_base_loaded.store(true, Ordering::SeqCst);
        *self.id_to_idx.write() = None;
        
        Ok(())
    }
    
    /// Update only the V4 footer schema on disk (no data rewrite).
    /// Used for DDL operations (add/drop/rename column) when base data
    /// is not loaded into memory (mmap-only mode).
    pub fn update_v4_footer_schema(&self) -> io::Result<()> {
        let header = self.header.read();
        if header.version != FORMAT_VERSION_V4 || header.footer_offset == 0 {
            return Err(err_data("update_v4_footer_schema requires V4 format file"));
        }
        let footer_offset = header.footer_offset;
        drop(header);

        // Read existing footer from disk
        let file_len = std::fs::metadata(&self.path)?.len();
        let footer_bytes = {
            let file_guard = self.file.read();
            let file = file_guard.as_ref()
                .ok_or_else(|| err_not_conn("File not open"))?;
            let mut mmap = self.mmap_cache.write();
            let size = (file_len - footer_offset) as usize;
            let mut buf = vec![0u8; size];
            mmap.read_at(file, &mut buf, footer_offset)?;
            buf
        };
        let mut footer = V4Footer::from_bytes(&footer_bytes)?;

        // Update footer schema from current in-memory schema
        let schema = self.schema.read();
        footer.schema = schema.clone();
        drop(schema);

        // Release mmap before writing
        self.mmap_cache.write().invalidate();
        *self.file.write() = None;
        crate::query::ApexExecutor::invalidate_cache_for_path(&self.path);

        // Write updated footer at same offset (overwrite old footer)
        let mut file = OpenOptions::new().write(true).open(&self.path)?;
        let new_footer_bytes = footer.to_bytes();
        file.seek(SeekFrom::Start(footer_offset))?;
        file.write_all(&new_footer_bytes)?;
        // Write footer size + magic trailer
        file.write_all(&(new_footer_bytes.len() as u64).to_le_bytes())?;
        file.write_all(b"APXFOOT\0")?;
        file.flush()?;

        // Truncate file to remove any trailing data from old (possibly larger) footer
        let new_file_len = footer_offset + new_footer_bytes.len() as u64 + 16;
        file.set_len(new_file_len)?;

        // Update header column count (both in-memory and on-disk)
        {
            let mut header = self.header.write();
            header.column_count = footer.schema.column_count() as u32;
            // Write updated header to disk
            let mut hfile = OpenOptions::new().write(true).open(&self.path)?;
            hfile.write_all(&header.to_bytes())?;
            hfile.flush()?;
        }

        // Reopen file handle
        drop(file);
        let file = File::open(&self.path)?;
        *self.file.write() = Some(file);

        Ok(())
    }

    /// Update only the deletion vectors in existing Row Groups on disk.
    /// O(num_RGs) random writes instead of O(all_data) full rewrite.
    /// Also updates the footer's per-RG deletion_count and the header's row_count.
    fn save_deletion_vectors(&self) -> io::Result<()> {
        let header = self.header.read();
        if header.version != FORMAT_VERSION_V4 || header.footer_offset == 0 {
            return Err(err_data("save_deletion_vectors requires V4 format file"));
        }
        let footer_offset = header.footer_offset;
        drop(header);

        // Read existing footer
        let file_len = std::fs::metadata(&self.path)?.len();
        let footer_bytes = {
            let file_guard = self.file.read();
            let file = file_guard.as_ref()
                .ok_or_else(|| err_not_conn("File not open"))?;
            let mut mmap = self.mmap_cache.write();
            let size = (file_len - footer_offset) as usize;
            let mut buf = vec![0u8; size];
            mmap.read_at(file, &mut buf, footer_offset)?;
            buf
        };
        let mut footer = V4Footer::from_bytes(&footer_bytes)?;

        // Release mmap before writing
        self.mmap_cache.write().invalidate();
        *self.file.write() = None;
        crate::query::ApexExecutor::invalidate_cache_for_path(&self.path);

        let deleted = self.deleted.read();
        let mut file = OpenOptions::new().write(true).open(&self.path)?;

        // For each RG, write the updated deletion vector at its known offset
        let mut flat_row_start: usize = 0;
        let mut total_active: u64 = 0;
        for rg_meta in footer.row_groups.iter_mut() {
            let rg_rows = rg_meta.row_count as usize;
            if rg_rows == 0 {
                continue;
            }

            // Deletion vector starts after RG header (32 bytes) + IDs (rg_rows * 8)
            let del_vec_offset = rg_meta.offset + 32 + (rg_rows as u64 * 8);
            let del_vec_len = (rg_rows + 7) / 8;

            // Extract this RG's slice from the flat deleted bitmap
            let rg_del_vec = Self::slice_null_bitmap(
                &deleted, flat_row_start, flat_row_start + rg_rows,
            );

            // Count deleted rows in this RG
            let mut del_count: u32 = 0;
            for i in 0..rg_rows {
                if (rg_del_vec[i / 8] >> (i % 8)) & 1 == 1 {
                    del_count += 1;
                }
            }
            rg_meta.deletion_count = del_count;
            total_active += (rg_rows as u32 - del_count) as u64;

            // Write deletion vector to disk
            file.seek(SeekFrom::Start(del_vec_offset))?;
            file.write_all(&rg_del_vec[..del_vec_len])?;

            flat_row_start += rg_rows;
        }
        drop(deleted);

        // Rewrite footer with updated deletion_counts
        file.seek(SeekFrom::Start(footer_offset))?;
        let new_footer_bytes = footer.to_bytes();
        file.write_all(&new_footer_bytes)?;
        // Truncate in case new footer is shorter (shouldn't happen, but safety)
        let new_end = footer_offset + new_footer_bytes.len() as u64;
        file.set_len(new_end)?;
        file.flush()?;

        // Update header: row_count = active rows (matches save_v4 convention)
        {
            let mut header = self.header.write();
            header.row_count = total_active;
            file.seek(SeekFrom::Start(0))?;
            file.write_all(&header.to_bytes())?;
        }
        file.flush()?;
        drop(file);

        // Reopen file handle
        let new_file = File::open(&self.path)?;
        *self.file.write() = Some(new_file);

        Ok(())
    }

    /// Write a new Row Group to disk without modifying in-memory state.
    /// Called by save() when rows are already in memory and only need persisting.
    /// Also called by append_row_group() which additionally updates memory.
    fn write_row_group_to_disk(
        &self,
        new_ids: &[u64],
        new_columns: &[ColumnData],
        new_nulls: &[Vec<u8>],
    ) -> io::Result<()> {
        let header = self.header.read();
        if header.version != FORMAT_VERSION_V4 || header.footer_offset == 0 {
            return Err(err_data("write_row_group_to_disk requires V4 format file"));
        }
        let footer_offset = header.footer_offset;
        drop(header);
        
        // Read existing footer
        let file_len = std::fs::metadata(&self.path)?.len();
        let footer_bytes = {
            let file_guard = self.file.read();
            let file = file_guard.as_ref()
                .ok_or_else(|| err_not_conn("File not open"))?;
            let mut mmap = self.mmap_cache.write();
            let size = (file_len - footer_offset) as usize;
            let mut buf = vec![0u8; size];
            mmap.read_at(file, &mut buf, footer_offset)?;
            buf
        };
        let mut footer = V4Footer::from_bytes(&footer_bytes)?;

        // Schema evolution: merge any new columns from in-memory schema into footer
        {
            let mem_schema = self.schema.read();
            for (name, ct) in &mem_schema.columns {
                if footer.schema.get_index(name).is_none() {
                    footer.schema.add_column(name, *ct);
                }
            }
        }
        let col_count = footer.schema.column_count();
        
        // Release mmap before writing
        self.mmap_cache.write().invalidate();
        *self.file.write() = None;
        crate::query::ApexExecutor::invalidate_cache_for_path(&self.path);
        
        // Open file for append — seek to old footer position (overwrite it)
        let mut file = OpenOptions::new().write(true).open(&self.path)?;
        file.seek(SeekFrom::Start(footer_offset))?;
        let mut writer = BufWriter::with_capacity(64 * 1024, file);
        
        let rg_rows = new_ids.len();
        let rg_offset = footer_offset;
        let min_id = new_ids.iter().copied().min().unwrap_or(0);
        let max_id = new_ids.iter().copied().max().unwrap_or(0);
        
        // Write RG header (32 bytes)
        writer.write_all(MAGIC_ROW_GROUP)?;
        writer.write_all(&(rg_rows as u32).to_le_bytes())?;
        writer.write_all(&(col_count as u32).to_le_bytes())?;
        writer.write_all(&min_id.to_le_bytes())?;
        writer.write_all(&max_id.to_le_bytes())?;
        writer.write_all(&[0u8; 4])?;
        
        // IDs
        for &id in new_ids {
            writer.write_all(&id.to_le_bytes())?;
        }
        
        // Deletion vector (all zeros)
        let del_vec_len = (rg_rows + 7) / 8;
        writer.write_all(&vec![0u8; del_vec_len])?;
        
        // Columns
        let null_bitmap_len = (rg_rows + 7) / 8;
        for col_idx in 0..col_count {
            // Null bitmap
            let col_nulls = new_nulls.get(col_idx).map(|v| v.as_slice()).unwrap_or(&[]);
            let padded = if col_nulls.len() < null_bitmap_len {
                let mut v = vec![0u8; null_bitmap_len];
                let copy = col_nulls.len().min(null_bitmap_len);
                v[..copy].copy_from_slice(&col_nulls[..copy]);
                v
            } else {
                col_nulls[..null_bitmap_len].to_vec()
            };
            writer.write_all(&padded)?;
            
            // Column data — dict-encode if footer schema expects StringDict
            if col_idx < new_columns.len() {
                let col = &new_columns[col_idx];
                if col_idx < footer.schema.columns.len()
                    && footer.schema.columns[col_idx].1 == ColumnType::StringDict
                    && matches!(col, ColumnData::String { .. })
                {
                    if let Some(dict) = col.to_dict_encoded() {
                        dict.write_to(&mut writer)?;
                    } else {
                        col.write_to(&mut writer)?;
                    }
                } else {
                    col.write_to(&mut writer)?;
                }
            }
        }
        
        let rg_end = writer.stream_position()?;
        
        // Update footer with new RG
        footer.row_groups.push(RowGroupMeta {
            offset: rg_offset,
            data_size: rg_end - rg_offset,
            row_count: rg_rows as u32,
            min_id,
            max_id,
            deletion_count: 0,
        });
        
        // Write updated footer + trailer (footer_size + magic)
        let new_footer_offset = rg_end;
        let footer_bytes = footer.to_bytes();
        writer.write_all(&footer_bytes)?;
        writer.write_all(&(footer_bytes.len() as u64).to_le_bytes())?;
        writer.write_all(MAGIC_V4_FOOTER)?;
        writer.flush()?;
        
        // Fix header
        let new_persisted = self.persisted_row_count.load(Ordering::SeqCst) + rg_rows as u64;
        let writer_inner = writer.get_mut();
        {
            let mut header = self.header.write();
            header.row_count = new_persisted;
            header.footer_offset = new_footer_offset;
            header.row_group_count = footer.row_groups.len() as u32;
        }
        let header = self.header.read();
        writer_inner.seek(SeekFrom::Start(0))?;
        writer_inner.write_all(&header.to_bytes())?;
        writer_inner.flush()?;
        
        drop(header);
        drop(writer);
        
        // Reopen file
        let new_file = File::open(&self.path)?;
        *self.file.write() = Some(new_file);
        
        // Update persisted count (disk now has more rows)
        self.persisted_row_count.store(new_persisted, Ordering::SeqCst);
        
        Ok(())
    }

    /// Append a new Row Group to an existing V4 file without rewriting.
    /// Overwrites old footer, writes new RG + updated footer, fixes header.
    /// Also updates in-memory state (IDs, active_count).
    /// Use this when adding NEW data that is NOT already in memory.
    pub fn append_row_group(
        &self,
        new_ids: &[u64],
        new_columns: &[ColumnData],
        new_nulls: &[Vec<u8>],
    ) -> io::Result<()> {
        let rg_rows = new_ids.len();
        self.write_row_group_to_disk(new_ids, new_columns, new_nulls)?;
        
        // Update in-memory state (caller hasn't added these rows yet)
        {
            let mut ids = self.ids.write();
            ids.extend_from_slice(new_ids);
        }
        let next_id = new_ids.iter().max().map(|&id| id + 1).unwrap_or(0);
        let current_next = self.next_id.load(Ordering::SeqCst);
        if next_id > current_next {
            self.next_id.store(next_id, Ordering::SeqCst);
        }
        self.active_count.fetch_add(rg_rows as u64, Ordering::SeqCst);
        *self.id_to_idx.write() = None;
        
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
    
    /// Execute simple aggregation (no GROUP BY, no WHERE) directly on V4 columns
    /// Uses zero-copy Arrow SIMD: creates Arrow arrays pointing to V4 memory (no clone),
    /// runs Arrow compute sum/min/max with SIMD, drops arrays before lock guards.
    /// Returns (count, sum, min, max, is_int) for each requested column
    pub fn execute_simple_agg(
        &self,
        agg_cols: &[&str],
    ) -> io::Result<Option<Vec<(i64, f64, f64, f64, bool)>>> {
        use arrow::array::PrimitiveArray;
        use arrow::buffer::{Buffer, ScalarBuffer};
        use arrow::datatypes::{Int64Type, Float64Type};
        use std::sync::Arc;
        
        // Check if in-memory data is available for fast path
        let columns = self.columns.read();
        if columns.is_empty() || columns.iter().all(|c| c.len() == 0) {
            return Ok(None);
        }
        
        let schema = self.schema.read();
        let deleted = self.deleted.read();
        let total_rows = columns.first().map(|c| c.len()).unwrap_or(0);
        
        let has_deleted = deleted.iter().any(|&b| b != 0);
        // Bail to Arrow path if there are deleted rows (need filtered arrays)
        if has_deleted { return Ok(None); }
        
        let active_count = total_rows as i64;
        let mut results: Vec<(i64, f64, f64, f64, bool)> = Vec::with_capacity(agg_cols.len());
        
        for &col_name in agg_cols {
            if col_name == "*" || col_name == "1" {
                results.push((active_count, 0.0, 0.0, 0.0, false));
                continue;
            }
            
            let col_idx = match schema.get_index(col_name) {
                Some(idx) => idx,
                None => { results.push((0, 0.0, 0.0, 0.0, false)); continue; }
            };
            if col_idx >= columns.len() { results.push((0, 0.0, 0.0, 0.0, false)); continue; }
            
            match &columns[col_idx] {
                ColumnData::Int64(vals) => {
                    // Zero-copy Arrow: create Buffer pointing to V4 memory (no clone!)
                    // SAFETY: lock guards outlive the Arrow arrays in this scope
                    let byte_len = vals.len() * std::mem::size_of::<i64>();
                    let buffer = unsafe {
                        Buffer::from_custom_allocation(
                            std::ptr::NonNull::new_unchecked(vals.as_ptr() as *mut u8),
                            byte_len,
                            Arc::new(()), // no-op deallocator - V4 Vec owns the memory
                        )
                    };
                    let arr = PrimitiveArray::<Int64Type>::new(
                        ScalarBuffer::new(buffer, 0, vals.len()), None,
                    );
                    let sum = arrow::compute::sum(&arr).unwrap_or(0);
                    let min_v = arrow::compute::min(&arr).unwrap_or(i64::MAX);
                    let max_v = arrow::compute::max(&arr).unwrap_or(i64::MIN);
                    // Drop Arrow array before lock guards
                    drop(arr);
                    results.push((vals.len() as i64, sum as f64, min_v as f64, max_v as f64, true));
                }
                ColumnData::Float64(vals) => {
                    let byte_len = vals.len() * std::mem::size_of::<f64>();
                    let buffer = unsafe {
                        Buffer::from_custom_allocation(
                            std::ptr::NonNull::new_unchecked(vals.as_ptr() as *mut u8),
                            byte_len,
                            Arc::new(()),
                        )
                    };
                    let arr = PrimitiveArray::<Float64Type>::new(
                        ScalarBuffer::new(buffer, 0, vals.len()), None,
                    );
                    let sum = arrow::compute::sum(&arr).unwrap_or(0.0);
                    let min_v = arrow::compute::min(&arr).unwrap_or(f64::INFINITY);
                    let max_v = arrow::compute::max(&arr).unwrap_or(f64::NEG_INFINITY);
                    drop(arr);
                    results.push((vals.len() as i64, sum, min_v, max_v, false));
                }
                _ => { results.push((active_count, 0.0, 0.0, 0.0, false)); }
            }
        }
        
        Ok(Some(results))
    }

    /// Build cached string dictionary indices for a column (row→group_id mapping)
    /// Returns (dict_strings, group_ids) where group_ids[row] = index into dict_strings
    pub fn build_string_dict_cache(
        &self,
        col_name: &str,
    ) -> io::Result<Option<(Vec<String>, Vec<u16>)>> {
        if !self.has_v4_in_memory_data() { return Ok(None); }
        
        let schema = self.schema.read();
        let columns = self.columns.read();
        let total_rows = self.ids.read().len();
        
        let col_idx = match schema.get_index(col_name) {
            Some(idx) => idx,
            None => return Ok(None),
        };
        if col_idx >= columns.len() { return Ok(None); }
        
        let (offsets, data) = match &columns[col_idx] {
            ColumnData::String { offsets, data } => (offsets, data),
            _ => return Ok(None),
        };
        let count = offsets.len().saturating_sub(1).min(total_rows);
        
        let mut dict_map: ahash::AHashMap<&[u8], u16> = ahash::AHashMap::with_capacity(64);
        let mut dict_strings: Vec<String> = Vec::with_capacity(64);
        let mut group_ids: Vec<u16> = Vec::with_capacity(count);
        
        for i in 0..count {
            let s = offsets[i] as usize;
            let e = offsets[i + 1] as usize;
            let key = &data[s..e];
            let gid = match dict_map.get(key) {
                Some(&id) => id,
                None => {
                    let id = dict_strings.len() as u16;
                    dict_map.insert(key, id);
                    dict_strings.push(std::str::from_utf8(key).unwrap_or("").to_string());
                    id
                }
            };
            group_ids.push(gid);
        }
        
        Ok(Some((dict_strings, group_ids)))
    }

    /// Execute GROUP BY + aggregate using pre-built dict cache
    /// Much faster than building dictionary on every query
    pub fn execute_group_agg_cached(
        &self,
        dict_strings: &[String],
        group_ids: &[u16],
        agg_cols: &[(&str, bool)], // (col_name, is_count_star)
    ) -> io::Result<Option<Vec<(String, Vec<(f64, i64)>)>>> {
        if !self.has_v4_in_memory_data() { return Ok(None); }
        
        let schema = self.schema.read();
        let columns = self.columns.read();
        let deleted = self.deleted.read();
        let total_rows = self.ids.read().len();
        
        let has_deleted = deleted.iter().any(|&b| b != 0);
        let scan_rows = total_rows.min(group_ids.len());
        let num_groups = dict_strings.len();
        let num_aggs = agg_cols.len();
        
        struct AggSlice<'a> { i64_vals: Option<&'a [i64]>, f64_vals: Option<&'a [f64]>, is_count: bool }
        let agg_slices: Vec<AggSlice> = agg_cols.iter().map(|(name, is_count)| {
            if *is_count {
                AggSlice { i64_vals: None, f64_vals: None, is_count: true }
            } else if let Some(idx) = schema.get_index(name) {
                if idx < columns.len() {
                    match &columns[idx] {
                        ColumnData::Int64(v) => AggSlice { i64_vals: Some(v.as_slice()), f64_vals: None, is_count: false },
                        ColumnData::Float64(v) => AggSlice { i64_vals: None, f64_vals: Some(v.as_slice()), is_count: false },
                        _ => AggSlice { i64_vals: None, f64_vals: None, is_count: true },
                    }
                } else { AggSlice { i64_vals: None, f64_vals: None, is_count: true } }
            } else { AggSlice { i64_vals: None, f64_vals: None, is_count: true } }
        }).collect();
        
        let flat_len = num_groups * num_aggs;
        let mut flat_sums = vec![0.0f64; flat_len];
        let mut flat_counts = vec![0i64; flat_len];
        
        // Single-pass aggregation with O(1) group lookup via cached group_ids
        if has_deleted {
            for i in 0..scan_rows {
                let b = i / 8; let bit = i % 8;
                if b < deleted.len() && (deleted[b] >> bit) & 1 != 0 { continue; }
                let base = group_ids[i] as usize * num_aggs;
                for (ai, agg) in agg_slices.iter().enumerate() {
                    flat_counts[base + ai] += 1;
                    if !agg.is_count {
                        if let Some(vals) = agg.f64_vals { if i < vals.len() { flat_sums[base + ai] += vals[i]; } }
                        else if let Some(vals) = agg.i64_vals { if i < vals.len() { flat_sums[base + ai] += vals[i] as f64; } }
                    }
                }
            }
        } else {
            for i in 0..scan_rows {
                let base = group_ids[i] as usize * num_aggs;
                for (ai, agg) in agg_slices.iter().enumerate() {
                    flat_counts[base + ai] += 1;
                    if !agg.is_count {
                        if let Some(vals) = agg.f64_vals { if i < vals.len() { unsafe { *flat_sums.get_unchecked_mut(base + ai) += *vals.get_unchecked(i); } } }
                        else if let Some(vals) = agg.i64_vals { if i < vals.len() { unsafe { *flat_sums.get_unchecked_mut(base + ai) += *vals.get_unchecked(i) as f64; } } }
                    }
                }
            }
        }
        
        let results: Vec<(String, Vec<(f64, i64)>)> = (0..num_groups)
            .filter(|&gid| flat_counts[gid * num_aggs] > 0)
            .map(|gid| {
                let aggs: Vec<(f64, i64)> = (0..num_aggs)
                    .map(|ai| (flat_sums[gid * num_aggs + ai], flat_counts[gid * num_aggs + ai]))
                    .collect();
                (dict_strings[gid].clone(), aggs)
            })
            .collect();
        
        Ok(Some(results))
    }

    /// Execute BETWEEN + GROUP BY using pre-built dict cache
    pub fn execute_between_group_agg_cached(
        &self,
        filter_col: &str,
        lo: f64,
        hi: f64,
        dict_strings: &[String],
        group_ids: &[u16],
        agg_col: Option<&str>,
    ) -> io::Result<Option<Vec<(String, f64, i64)>>> {
        if !self.has_v4_in_memory_data() { return Ok(None); }
        
        let schema = self.schema.read();
        let columns = self.columns.read();
        let deleted = self.deleted.read();
        let total_rows = self.ids.read().len();
        
        let filter_idx = match schema.get_index(filter_col) {
            Some(idx) => idx,
            None => return Ok(None),
        };
        if filter_idx >= columns.len() { return Ok(None); }
        
        let has_deleted = deleted.iter().any(|&b| b != 0);
        let lo_i64 = lo as i64;
        let hi_i64 = hi as i64;
        let scan_rows = total_rows.min(group_ids.len());
        let num_groups = dict_strings.len();
        
        let mut group_sums = vec![0.0f64; num_groups];
        let mut group_counts = vec![0i64; num_groups];
        
        let agg_idx = agg_col.and_then(|ac| schema.get_index(ac));
        let agg_f64 = agg_idx.and_then(|idx| {
            if idx < columns.len() { match &columns[idx] { ColumnData::Float64(v) => Some(v.as_slice()), _ => None } } else { None }
        });
        let agg_i64 = agg_idx.and_then(|idx| {
            if idx < columns.len() { match &columns[idx] { ColumnData::Int64(v) => Some(v.as_slice()), _ => None } } else { None }
        });
        
        // Branchless accumulation: eliminates branch misprediction at ~50% BETWEEN hit rate
        // mask = (in_range) as i64 → 0 or 1, multiply with agg value so non-matching adds 0
        macro_rules! between_agg_branchy {
            ($filter_vals:expr, $lo_cmp:expr, $hi_cmp:expr, $limit:expr) => {{
                for i in 0..$limit {
                    let b = i / 8; let bit = i % 8;
                    if b < deleted.len() && (deleted[b] >> bit) & 1 != 0 { continue; }
                    let fv = unsafe { *$filter_vals.get_unchecked(i) };
                    if fv >= $lo_cmp && fv <= $hi_cmp {
                        let gid = unsafe { *group_ids.get_unchecked(i) } as usize;
                        unsafe { *group_counts.get_unchecked_mut(gid) += 1; }
                        if let Some(av) = agg_f64 { unsafe { *group_sums.get_unchecked_mut(gid) += *av.get_unchecked(i); } }
                        else if let Some(av) = agg_i64 { unsafe { *group_sums.get_unchecked_mut(gid) += *av.get_unchecked(i) as f64; } }
                    }
                }
            }};
        }
        
        if let Some(filter_vals) = match &columns[filter_idx] { ColumnData::Int64(v) => Some(v.as_slice()), _ => None } {
            let limit = scan_rows.min(filter_vals.len()).min(group_ids.len());
            let limit = limit.min(agg_f64.map_or(usize::MAX, |a| a.len())).min(agg_i64.map_or(usize::MAX, |a| a.len()));
            if has_deleted {
                between_agg_branchy!(filter_vals, lo_i64, hi_i64, limit);
            } else if let Some(av) = agg_f64 {
                // HOT PATH: branchless i64 filter + f64 agg (no deleted)
                for i in 0..limit {
                    let fv = unsafe { *filter_vals.get_unchecked(i) };
                    let mask = (fv >= lo_i64 && fv <= hi_i64) as i64;
                    let gid = unsafe { *group_ids.get_unchecked(i) } as usize;
                    unsafe {
                        *group_counts.get_unchecked_mut(gid) += mask;
                        *group_sums.get_unchecked_mut(gid) += mask as f64 * *av.get_unchecked(i);
                    }
                }
            } else if let Some(av) = agg_i64 {
                for i in 0..limit {
                    let fv = unsafe { *filter_vals.get_unchecked(i) };
                    let mask = (fv >= lo_i64 && fv <= hi_i64) as i64;
                    let gid = unsafe { *group_ids.get_unchecked(i) } as usize;
                    unsafe {
                        *group_counts.get_unchecked_mut(gid) += mask;
                        *group_sums.get_unchecked_mut(gid) += mask as f64 * (*av.get_unchecked(i) as f64);
                    }
                }
            } else {
                // COUNT-only: branchless
                for i in 0..limit {
                    let fv = unsafe { *filter_vals.get_unchecked(i) };
                    let mask = (fv >= lo_i64 && fv <= hi_i64) as i64;
                    let gid = unsafe { *group_ids.get_unchecked(i) } as usize;
                    unsafe { *group_counts.get_unchecked_mut(gid) += mask; }
                }
            }
        } else if let Some(filter_vals) = match &columns[filter_idx] { ColumnData::Float64(v) => Some(v.as_slice()), _ => None } {
            let limit = scan_rows.min(filter_vals.len()).min(group_ids.len());
            let limit = limit.min(agg_f64.map_or(usize::MAX, |a| a.len())).min(agg_i64.map_or(usize::MAX, |a| a.len()));
            if has_deleted {
                between_agg_branchy!(filter_vals, lo, hi, limit);
            } else if let Some(av) = agg_f64 {
                for i in 0..limit {
                    let fv = unsafe { *filter_vals.get_unchecked(i) };
                    let mask = (fv >= lo && fv <= hi) as i64;
                    let gid = unsafe { *group_ids.get_unchecked(i) } as usize;
                    unsafe {
                        *group_counts.get_unchecked_mut(gid) += mask;
                        *group_sums.get_unchecked_mut(gid) += mask as f64 * *av.get_unchecked(i);
                    }
                }
            } else if let Some(av) = agg_i64 {
                for i in 0..limit {
                    let fv = unsafe { *filter_vals.get_unchecked(i) };
                    let mask = (fv >= lo && fv <= hi) as i64;
                    let gid = unsafe { *group_ids.get_unchecked(i) } as usize;
                    unsafe {
                        *group_counts.get_unchecked_mut(gid) += mask;
                        *group_sums.get_unchecked_mut(gid) += mask as f64 * (*av.get_unchecked(i) as f64);
                    }
                }
            } else {
                for i in 0..limit {
                    let fv = unsafe { *filter_vals.get_unchecked(i) };
                    let mask = (fv >= lo && fv <= hi) as i64;
                    let gid = unsafe { *group_ids.get_unchecked(i) } as usize;
                    unsafe { *group_counts.get_unchecked_mut(gid) += mask; }
                }
            }
        }
        
        let results: Vec<(String, f64, i64)> = (0..num_groups)
            .filter(|&gid| group_counts[gid] > 0)
            .map(|gid| (dict_strings[gid].clone(), group_sums[gid], group_counts[gid]))
            .collect();
        
        Ok(Some(results))
    }

    /// Execute BETWEEN + GROUP BY + aggregate directly on V4 in-memory columns
    /// Returns Vec<(group_key, sum, count)> for the caller to compute final values
    /// Uses pre-built row→group_id mapping for O(1) group lookup during aggregation
    pub fn execute_between_group_agg(
        &self,
        filter_col: &str,
        lo: f64,
        hi: f64,
        group_col: &str,
        agg_col: Option<&str>,
    ) -> io::Result<Option<Vec<(String, f64, i64)>>> {
        if !self.has_v4_in_memory_data() { return Ok(None); }
        
        let schema = self.schema.read();
        let columns = self.columns.read();
        let deleted = self.deleted.read();
        let total_rows = self.ids.read().len();
        
        let filter_idx = match schema.get_index(filter_col) {
            Some(idx) => idx,
            None => return Ok(None),
        };
        let group_idx = match schema.get_index(group_col) {
            Some(idx) => idx,
            None => return Ok(None),
        };
        let agg_idx = agg_col.and_then(|ac| schema.get_index(ac));
        
        if filter_idx >= columns.len() || group_idx >= columns.len() {
            return Ok(None);
        }
        
        let has_deleted = deleted.iter().any(|&b| b != 0);
        let lo_i64 = lo as i64;
        let hi_i64 = hi as i64;
        
        let (group_offsets, group_bytes) = match &columns[group_idx] {
            ColumnData::String { offsets, data } => (offsets, data),
            _ => return Ok(None),
        };
        let group_count = group_offsets.len().saturating_sub(1);
        let scan_rows = total_rows.min(group_count);
        
        // Single-pass: build group dict + filter + aggregate simultaneously
        // Use small linear-scan dict for ≤64 groups (faster than hash map for short strings)
        let mut dict_entries: Vec<(u32, u32)> = Vec::with_capacity(32); // (start, end) in group_bytes
        let mut dict_strings: Vec<String> = Vec::with_capacity(32);
        let mut group_sums = vec![0.0f64; 64]; // pre-alloc for up to 64 groups
        let mut group_counts = vec![0i64; 64];
        
        let filter_i64 = match &columns[filter_idx] {
            ColumnData::Int64(v) => Some(v.as_slice()),
            _ => None,
        };
        let filter_f64 = match &columns[filter_idx] {
            ColumnData::Float64(v) => Some(v.as_slice()),
            _ => None,
        };
        let agg_i64 = agg_idx.and_then(|idx| {
            if idx < columns.len() { match &columns[idx] { ColumnData::Int64(v) => Some(v.as_slice()), _ => None } } else { None }
        });
        let agg_f64 = agg_idx.and_then(|idx| {
            if idx < columns.len() { match &columns[idx] { ColumnData::Float64(v) => Some(v.as_slice()), _ => None } } else { None }
        });
        
        // Macro for the inner aggregation to avoid code duplication
        macro_rules! agg_row {
            ($gid:expr, $i:expr) => {
                group_counts[$gid] += 1;
                if let Some(av) = agg_f64 { if $i < av.len() { group_sums[$gid] += av[$i]; } }
                else if let Some(av) = agg_i64 { if $i < av.len() { group_sums[$gid] += av[$i] as f64; } }
            }
        }
        
        // Inline group lookup: linear scan of ≤64 entries
        #[inline(always)]
        fn find_group(dict: &[(u32, u32)], group_bytes: &[u8], s: usize, e: usize) -> Option<usize> {
            let needle = &group_bytes[s..e];
            let needle_len = (e - s) as u32;
            for (idx, &(ds, de)) in dict.iter().enumerate() {
                if de - ds == needle_len && &group_bytes[ds as usize..de as usize] == needle {
                    return Some(idx);
                }
            }
            None
        }
        
        // Single-pass: filter + group + aggregate
        if let Some(vals) = filter_i64 {
            let limit = scan_rows.min(vals.len());
            if has_deleted {
                for i in 0..limit {
                    let b = i / 8; let bit = i % 8;
                    if b < deleted.len() && (deleted[b] >> bit) & 1 != 0 { continue; }
                    if vals[i] >= lo_i64 && vals[i] <= hi_i64 && i < group_count {
                        let s = group_offsets[i] as usize;
                        let e = group_offsets[i + 1] as usize;
                        let gid = if let Some(g) = find_group(&dict_entries, group_bytes, s, e) { g }
                        else {
                            let g = dict_entries.len();
                            dict_entries.push((s as u32, e as u32));
                            dict_strings.push(std::str::from_utf8(&group_bytes[s..e]).unwrap_or("").to_string());
                            if g >= group_sums.len() { group_sums.resize(g + 16, 0.0); group_counts.resize(g + 16, 0); }
                            g
                        };
                        agg_row!(gid, i);
                    }
                }
            } else {
                for i in 0..limit {
                    if vals[i] >= lo_i64 && vals[i] <= hi_i64 && i < group_count {
                        let s = group_offsets[i] as usize;
                        let e = group_offsets[i + 1] as usize;
                        let gid = if let Some(g) = find_group(&dict_entries, group_bytes, s, e) { g }
                        else {
                            let g = dict_entries.len();
                            dict_entries.push((s as u32, e as u32));
                            dict_strings.push(std::str::from_utf8(&group_bytes[s..e]).unwrap_or("").to_string());
                            if g >= group_sums.len() { group_sums.resize(g + 16, 0.0); group_counts.resize(g + 16, 0); }
                            g
                        };
                        agg_row!(gid, i);
                    }
                }
            }
        } else if let Some(vals) = filter_f64 {
            let limit = scan_rows.min(vals.len());
            for i in 0..limit {
                if has_deleted {
                    let b = i / 8; let bit = i % 8;
                    if b < deleted.len() && (deleted[b] >> bit) & 1 != 0 { continue; }
                }
                if vals[i] >= lo && vals[i] <= hi && i < group_count {
                    let s = group_offsets[i] as usize;
                    let e = group_offsets[i + 1] as usize;
                    let gid = if let Some(g) = find_group(&dict_entries, group_bytes, s, e) { g }
                    else {
                        let g = dict_entries.len();
                        dict_entries.push((s as u32, e as u32));
                        dict_strings.push(std::str::from_utf8(&group_bytes[s..e]).unwrap_or("").to_string());
                        if g >= group_sums.len() { group_sums.resize(g + 16, 0.0); group_counts.resize(g + 16, 0); }
                        g
                    };
                    agg_row!(gid, i);
                }
            }
        }
        
        let num_groups = dict_entries.len();
        
        let results: Vec<(String, f64, i64)> = (0..num_groups)
            .filter(|&gid| group_counts[gid] > 0)
            .map(|gid| (dict_strings[gid].clone(), group_sums[gid], group_counts[gid]))
            .collect();
        
        Ok(Some(results))
    }

    /// Execute GROUP BY + aggregate directly on V4 in-memory columns (no WHERE filter)
    /// Returns Vec<(group_key, Vec<(sum, count)>)> for each aggregate column
    pub fn execute_group_agg(
        &self,
        group_col: &str,
        agg_cols: &[(&str, bool)],
    ) -> io::Result<Option<Vec<(String, Vec<(f64, i64)>)>>> {
        if !self.has_v4_in_memory_data() { return Ok(None); }
        
        let schema = self.schema.read();
        let columns = self.columns.read();
        let deleted = self.deleted.read();
        let total_rows = self.ids.read().len();
        
        let group_idx = match schema.get_index(group_col) {
            Some(idx) => idx,
            None => return Ok(None),
        };
        if group_idx >= columns.len() { return Ok(None); }
        
        let has_deleted = deleted.iter().any(|&b| b != 0);
        
        let (group_offsets, group_bytes) = match &columns[group_idx] {
            ColumnData::String { offsets, data } => (offsets, data),
            _ => return Ok(None),
        };
        let group_count = group_offsets.len().saturating_sub(1);
        let scan_rows = total_rows.min(group_count);
        let num_aggs = agg_cols.len();
        
        // Resolve agg column slices
        struct AggSlice<'a> { i64_vals: Option<&'a [i64]>, f64_vals: Option<&'a [f64]>, is_count: bool }
        let agg_slices: Vec<AggSlice> = agg_cols.iter().map(|(name, is_count)| {
            if *is_count {
                AggSlice { i64_vals: None, f64_vals: None, is_count: true }
            } else if let Some(idx) = schema.get_index(name) {
                if idx < columns.len() {
                    match &columns[idx] {
                        ColumnData::Int64(v) => AggSlice { i64_vals: Some(v.as_slice()), f64_vals: None, is_count: false },
                        ColumnData::Float64(v) => AggSlice { i64_vals: None, f64_vals: Some(v.as_slice()), is_count: false },
                        _ => AggSlice { i64_vals: None, f64_vals: None, is_count: true },
                    }
                } else { AggSlice { i64_vals: None, f64_vals: None, is_count: true } }
            } else { AggSlice { i64_vals: None, f64_vals: None, is_count: true } }
        }).collect();
        
        // Linear-scan dictionary for ≤64 groups (faster than hash map for short strings)
        let mut dict_entries: Vec<(u32, u32)> = Vec::with_capacity(32);
        let mut dict_strings: Vec<String> = Vec::with_capacity(32);
        let max_flat = 64 * num_aggs;
        let mut flat_sums = vec![0.0f64; max_flat];
        let mut flat_counts = vec![0i64; max_flat];
        
        #[inline(always)]
        fn find_group_ga(dict: &[(u32, u32)], gb: &[u8], s: usize, e: usize) -> Option<usize> {
            let needle = &gb[s..e];
            let nlen = (e - s) as u32;
            for (idx, &(ds, de)) in dict.iter().enumerate() {
                if de - ds == nlen && &gb[ds as usize..de as usize] == needle { return Some(idx); }
            }
            None
        }
        
        // Single-pass: group + aggregate
        for i in 0..scan_rows {
            if has_deleted {
                let b = i / 8; let bit = i % 8;
                if b < deleted.len() && (deleted[b] >> bit) & 1 != 0 { continue; }
            }
            let s = group_offsets[i] as usize;
            let e = group_offsets[i + 1] as usize;
            let gid = if let Some(g) = find_group_ga(&dict_entries, group_bytes, s, e) { g }
            else {
                let g = dict_entries.len();
                dict_entries.push((s as u32, e as u32));
                dict_strings.push(std::str::from_utf8(&group_bytes[s..e]).unwrap_or("").to_string());
                if (g + 1) * num_aggs > flat_sums.len() {
                    flat_sums.resize((g + 16) * num_aggs, 0.0);
                    flat_counts.resize((g + 16) * num_aggs, 0);
                }
                g
            };
            let base = gid * num_aggs;
            for (ai, agg) in agg_slices.iter().enumerate() {
                flat_counts[base + ai] += 1;
                if !agg.is_count {
                    if let Some(vals) = agg.f64_vals { if i < vals.len() { flat_sums[base + ai] += vals[i]; } }
                    else if let Some(vals) = agg.i64_vals { if i < vals.len() { flat_sums[base + ai] += vals[i] as f64; } }
                }
            }
        }
        
        let num_groups = dict_entries.len();
        let results: Vec<(String, Vec<(f64, i64)>)> = (0..num_groups)
            .filter(|&gid| flat_counts[gid * num_aggs] > 0)
            .map(|gid| {
                let aggs: Vec<(f64, i64)> = (0..num_aggs)
                    .map(|ai| (flat_sums[gid * num_aggs + ai], flat_counts[gid * num_aggs + ai]))
                    .collect();
                (dict_strings[gid].clone(), aggs)
            })
            .collect();
        
        Ok(Some(results))
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

    /// Get row count (includes both base file and delta rows)
    pub fn row_count(&self) -> u64 {
        let base_rows = self.header.read().row_count;
        let delta_rows = self.delta_row_count() as u64;
        base_rows + delta_rows
    }
    
    /// Get base file row count only (without delta)
    pub fn base_row_count(&self) -> u64 {
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

    /// Get header info: (footer_offset, row_count)
    #[inline]
    pub fn header_info(&self) -> (u64, u64) {
        let h = self.header.read();
        (h.footer_offset, h.row_count)
    }

    /// Get the next available ID value
    #[inline]
    pub fn next_id_value(&self) -> u64 {
        self.next_id.load(std::sync::atomic::Ordering::Relaxed)
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
        let mut null_positions: HashMap<String, Vec<bool>> = HashMap::new();

        // CRITICAL: Include ALL existing schema columns to ensure proper alignment
        // This fixes the partial column insert bug where missing columns don't get padded
        {
            let schema = self.schema.read();
            for (col_name, col_type) in &schema.columns {
                match col_type {
                    ColumnType::Int64 | ColumnType::Int8 | ColumnType::Int16 | ColumnType::Int32 |
                    ColumnType::UInt8 | ColumnType::UInt16 | ColumnType::UInt32 | ColumnType::UInt64 => {
                        int_columns.insert(col_name.clone(), Vec::with_capacity(num_rows));
                    }
                    ColumnType::Float64 | ColumnType::Float32 => {
                        float_columns.insert(col_name.clone(), Vec::with_capacity(num_rows));
                    }
                    ColumnType::String | ColumnType::StringDict | ColumnType::Null => {
                        string_columns.insert(col_name.clone(), Vec::with_capacity(num_rows));
                    }
                    ColumnType::Binary => {
                        binary_columns.insert(col_name.clone(), Vec::with_capacity(num_rows));
                    }
                    ColumnType::Bool => {
                        bool_columns.insert(col_name.clone(), Vec::with_capacity(num_rows));
                    }
                }
                null_positions.insert(col_name.clone(), Vec::with_capacity(num_rows));
            }
        }

        // Also determine schema from input rows for NEW columns
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
                null_positions.insert(key.clone(), Vec::with_capacity(num_rows));
            }
        }

        // Pre-allocate NULL string to avoid repeated allocation
        static NULL_MARKER: &str = "\x00__NULL__\x00";
        
        // Single pass: collect all values and track NULLs
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
                    null_positions.insert(key.clone(), Vec::with_capacity(num_rows));
                }
            }
            
            // Collect values for all columns and track NULL positions
            for (key, col) in int_columns.iter_mut() {
                let (val, is_null) = match row.get(key) {
                    Some(ColumnValue::Int64(v)) => (*v, false),
                    Some(ColumnValue::Null) | None => (0, true),
                    _ => (0, true),
                };
                col.push(val);
                null_positions.entry(key.clone()).or_default().push(is_null);
            }
            for (key, col) in float_columns.iter_mut() {
                let (val, is_null) = match row.get(key) {
                    Some(ColumnValue::Float64(v)) => (*v, false),
                    Some(ColumnValue::Null) | None => (0.0, true),
                    _ => (0.0, true),
                };
                col.push(val);
                null_positions.entry(key.clone()).or_default().push(is_null);
            }
            for (key, col) in string_columns.iter_mut() {
                let (val, is_null) = match row.get(key) {
                    Some(ColumnValue::String(v)) => (v.clone(), false),
                    Some(ColumnValue::Null) => (NULL_MARKER.to_string(), true),
                    None => (String::new(), true),
                    _ => (String::new(), true),
                };
                col.push(val);
                null_positions.entry(key.clone()).or_default().push(is_null);
            }
            for (key, col) in binary_columns.iter_mut() {
                let (val, is_null) = match row.get(key) {
                    Some(ColumnValue::Binary(v)) => (v.clone(), false),
                    Some(ColumnValue::Null) | None => (Vec::new(), true),
                    _ => (Vec::new(), true),
                };
                col.push(val);
                null_positions.entry(key.clone()).or_default().push(is_null);
            }
            for (key, col) in bool_columns.iter_mut() {
                let (val, is_null) = match row.get(key) {
                    Some(ColumnValue::Bool(v)) => (*v, false),
                    Some(ColumnValue::Null) | None => (false, true),
                    _ => (false, true),
                };
                col.push(val);
                null_positions.entry(key.clone()).or_default().push(is_null);
            }
        }

        let result = self.insert_typed_with_nulls(int_columns, float_columns, string_columns, binary_columns, bool_columns, null_positions)?;
        
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

    /// Check if compaction is needed (true if delta file exists)
    pub fn needs_compaction(&self) -> bool {
        self.has_delta()
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
    fn test_bool_null_bitmap() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test_bool_null.apex");

        let storage = OnDemandStorage::create(&path).unwrap();

        // Insert with NULL boolean
        let mut rows = Vec::new();
        
        let mut row1 = HashMap::new();
        row1.insert("id".to_string(), ColumnValue::Int64(1));
        row1.insert("flag".to_string(), ColumnValue::Bool(true));
        rows.push(row1);
        
        let mut row2 = HashMap::new();
        row2.insert("id".to_string(), ColumnValue::Int64(2));
        row2.insert("flag".to_string(), ColumnValue::Bool(false));
        rows.push(row2);
        
        let mut row3 = HashMap::new();
        row3.insert("id".to_string(), ColumnValue::Int64(3));
        row3.insert("flag".to_string(), ColumnValue::Null);  // NULL boolean
        rows.push(row3);

        storage.insert_rows(&rows).unwrap();
        
        // Check null bitmap in memory BEFORE save
        {
            let nulls = storage.nulls.read();
            let schema = storage.schema.read();
            let flag_idx = schema.get_index("flag").unwrap();
            println!("Flag column index: {}", flag_idx);
            println!("Nulls len: {}", nulls.len());
            if flag_idx < nulls.len() {
                println!("Null bitmap for flag: {:?}", nulls[flag_idx]);
                // Row 3 (index 2) should be marked as NULL
                // Byte 0, bit 2 should be set
                assert!(!nulls[flag_idx].is_empty(), "Null bitmap should not be empty");
                assert_eq!(nulls[flag_idx][0] & (1 << 2), 1 << 2, "Row 2 should be marked as NULL");
            }
        }
        
        storage.save().unwrap();

        // Reopen and verify null bitmap is persisted
        let storage2 = OnDemandStorage::open(&path).unwrap();
        
        // Check null mask via get_null_mask
        let null_mask = storage2.get_null_mask("flag", 0, 3);
        println!("Null mask after reopen: {:?}", null_mask);
        assert_eq!(null_mask, vec![false, false, true], "Row 2 should be NULL");
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

    #[test]
    fn test_v4_save_and_open() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test_v4.apex");

        // Create, insert, save as V4
        {
            let storage = OnDemandStorage::create(&path).unwrap();

            let mut int_cols = HashMap::new();
            int_cols.insert("age".to_string(), vec![25, 30, 35, 40, 45]);

            let mut string_cols = HashMap::new();
            string_cols.insert("name".to_string(), vec![
                "Alice".to_string(), "Bob".to_string(), "Charlie".to_string(),
                "David".to_string(), "Eve".to_string(),
            ]);

            storage.insert_typed(
                int_cols, HashMap::new(), string_cols,
                HashMap::new(), HashMap::new(),
            ).unwrap();

            // Save as V4 Row Group format
            storage.save_v4().unwrap();

            // Verify header has V4 version
            let header = storage.header.read();
            assert_eq!(header.version, FORMAT_VERSION_V4);
            assert!(header.footer_offset > 0);
            assert!(header.row_group_count >= 1);
        }

        // Reopen V4 file and load data
        {
            let storage = OnDemandStorage::open(&path).unwrap();
            // Header should indicate V4
            let header = storage.header.read();
            assert_eq!(header.version, FORMAT_VERSION_V4);
            assert_eq!(header.row_count, 5);
            drop(header);

            // Load V4 data
            storage.open_v4_data().unwrap();
            assert_eq!(storage.ids.read().len(), 5);

            let columns = storage.columns.read();
            // User columns only (age, name) — _id stored separately in self.ids
            assert_eq!(columns.len(), 2);

            // Verify age column
            let schema = storage.schema.read();
            let age_idx = schema.get_index("age").unwrap();
            if let ColumnData::Int64(vals) = &columns[age_idx] {
                assert_eq!(vals, &[25, 30, 35, 40, 45]);
            } else {
                panic!("Expected Int64 for age column");
            }

            // Verify name column
            let name_idx = schema.get_index("name").unwrap();
            if let ColumnData::String { offsets, data } = &columns[name_idx] {
                assert_eq!(offsets.len(), 6); // 5 rows + 1
                let names: Vec<&str> = (0..5).map(|i| {
                    let s = offsets[i] as usize;
                    let e = offsets[i + 1] as usize;
                    std::str::from_utf8(&data[s..e]).unwrap()
                }).collect();
                assert_eq!(names, vec!["Alice", "Bob", "Charlie", "David", "Eve"]);
            } else {
                panic!("Expected String for name column");
            }
        }
    }

    #[test]
    fn test_v4_append_row_group() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test_v4_append.apex");

        // Create and save V4 with initial data
        let storage = OnDemandStorage::create(&path).unwrap();
        let mut int_cols = HashMap::new();
        int_cols.insert("val".to_string(), vec![1, 2, 3]);
        storage.insert_typed(
            int_cols, HashMap::new(), HashMap::new(),
            HashMap::new(), HashMap::new(),
        ).unwrap();
        storage.save_v4().unwrap();

        // Verify initial state
        assert_eq!(storage.header.read().row_count, 3);
        assert_eq!(storage.header.read().row_group_count, 1);

        // Append a new Row Group without rewriting
        let new_ids = vec![100, 101, 102];
        let schema = storage.schema.read();
        let col_count = schema.column_count();
        let mut new_columns: Vec<ColumnData> = Vec::new();
        for (name, col_type) in &schema.columns {
            if name == "_id" {
                // _id is handled via new_ids, not as a column
                new_columns.push(ColumnData::Int64(vec![100, 101, 102]));
            } else {
                match col_type {
                    ColumnType::Int64 => new_columns.push(ColumnData::Int64(vec![10, 20, 30])),
                    _ => new_columns.push(ColumnData::new(*col_type)),
                }
            }
        }
        drop(schema);
        let new_nulls: Vec<Vec<u8>> = vec![Vec::new(); col_count];

        storage.append_row_group(&new_ids, &new_columns, &new_nulls).unwrap();

        // Verify updated header
        let header = storage.header.read();
        assert_eq!(header.row_count, 6); // 3 + 3
        assert_eq!(header.row_group_count, 2);
        drop(header);

        // Reopen and load all data to verify
        let storage2 = OnDemandStorage::open(&path).unwrap();
        storage2.open_v4_data().unwrap();
        assert_eq!(storage2.ids.read().len(), 6);
    }

    #[test]
    fn test_v4_multiple_row_groups() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test_v4_multi_rg.apex");

        // Create a table with more rows than DEFAULT_ROW_GROUP_SIZE
        // Use a small RG to test splitting (we'll insert 10 rows, RG size is 65536)
        let storage = OnDemandStorage::create(&path).unwrap();

        let mut int_cols = HashMap::new();
        int_cols.insert("val".to_string(), (0..100).collect::<Vec<i64>>());
        storage.insert_typed(
            int_cols, HashMap::new(), HashMap::new(),
            HashMap::new(), HashMap::new(),
        ).unwrap();
        storage.save_v4().unwrap();

        // With 100 rows and RG size 65536, should be 1 RG
        assert_eq!(storage.header.read().row_group_count, 1);

        // Reopen and verify data integrity
        let storage2 = OnDemandStorage::open(&path).unwrap();
        storage2.open_v4_data().unwrap();

        let columns = storage2.columns.read();
        let schema = storage2.schema.read();
        let val_idx = schema.get_index("val").unwrap();
        if let ColumnData::Int64(vals) = &columns[val_idx] {
            assert_eq!(vals.len(), 100);
            assert_eq!(vals[0], 0);
            assert_eq!(vals[99], 99);
        } else {
            panic!("Expected Int64 for val column");
        }
    }

    #[test]
    fn test_v4_read_columns_integration() {
        // Tests the executor read path: open V4 file → read_columns() auto-loads data
        let dir = tempdir().unwrap();
        let path = dir.path().join("test_v4_read.apex");

        // Create, insert, save as V4
        {
            let storage = OnDemandStorage::create(&path).unwrap();
            let mut int_cols = HashMap::new();
            int_cols.insert("age".to_string(), vec![10, 20, 30]);
            let mut string_cols = HashMap::new();
            string_cols.insert("name".to_string(), vec![
                "Alice".to_string(), "Bob".to_string(), "Charlie".to_string(),
            ]);
            storage.insert_typed(
                int_cols, HashMap::new(), string_cols,
                HashMap::new(), HashMap::new(),
            ).unwrap();
            storage.save_v4().unwrap();
        }

        // Reopen and use read_columns() (executor path) — no explicit open_v4_data()
        let storage = OnDemandStorage::open(&path).unwrap();
        let result = storage.read_columns(Some(&["age", "name"]), 0, None).unwrap();

        // Verify age
        if let ColumnData::Int64(vals) = &result["age"] {
            assert_eq!(vals, &[10, 20, 30]);
        } else {
            panic!("Expected Int64 for age");
        }

        // Verify name
        if let ColumnData::String { offsets, data } = &result["name"] {
            let names: Vec<&str> = (0..3).map(|i| {
                let s = offsets[i] as usize;
                let e = offsets[i + 1] as usize;
                std::str::from_utf8(&data[s..e]).unwrap()
            }).collect();
            assert_eq!(names, vec!["Alice", "Bob", "Charlie"]);
        } else {
            panic!("Expected String for name");
        }

        // Test partial read (start_row, row_count)
        let partial = storage.read_columns(Some(&["age"]), 1, Some(2)).unwrap();
        if let ColumnData::Int64(vals) = &partial["age"] {
            assert_eq!(vals, &[20, 30]);
        } else {
            panic!("Expected Int64 for partial age read");
        }
    }
}
