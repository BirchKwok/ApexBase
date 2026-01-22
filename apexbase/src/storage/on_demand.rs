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
use std::io::{self, BufWriter, Seek, Write};
use std::os::unix::fs::FileExt;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use parking_lot::RwLock;
use serde::{Deserialize, Serialize};

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
            ColumnType::String | ColumnType::Binary => 0,
        }
    }

    pub fn is_variable_length(&self) -> bool {
        matches!(self, ColumnType::String | ColumnType::Binary)
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

    /// Serialize to bytes
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut buf = Vec::new();
        match self {
            ColumnData::Bool { data, len } => {
                buf.extend_from_slice(&(*len as u64).to_le_bytes());
                buf.extend_from_slice(data);
            }
            ColumnData::Int64(v) => {
                buf.extend_from_slice(&(v.len() as u64).to_le_bytes());
                for &val in v {
                    buf.extend_from_slice(&val.to_le_bytes());
                }
            }
            ColumnData::Float64(v) => {
                buf.extend_from_slice(&(v.len() as u64).to_le_bytes());
                for &val in v {
                    buf.extend_from_slice(&val.to_le_bytes());
                }
            }
            ColumnData::String { offsets, data } | ColumnData::Binary { offsets, data } => {
                let count = offsets.len().saturating_sub(1);
                buf.extend_from_slice(&(count as u64).to_le_bytes());
                for &off in offsets {
                    buf.extend_from_slice(&off.to_le_bytes());
                }
                buf.extend_from_slice(&(data.len() as u64).to_le_bytes());
                buf.extend_from_slice(data);
            }
        }
        buf
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
/// - Uses pread for random access without seeking
/// - Soft delete with deleted bitmap
/// - Update via delete + insert
pub struct OnDemandStorage {
    path: PathBuf,
    file: RwLock<Option<File>>,
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
    /// ID to row index mapping for fast lookups
    id_to_idx: RwLock<HashMap<u64, usize>>,
}

impl OnDemandStorage {
    /// Create a new V3 storage file
    pub fn create(path: &Path) -> io::Result<Self> {
        let header = OnDemandHeader::new();
        let schema = OnDemandSchema::new();

        let storage = Self {
            path: path.to_path_buf(),
            file: RwLock::new(None),
            header: RwLock::new(header),
            schema: RwLock::new(schema),
            column_index: RwLock::new(Vec::new()),
            columns: RwLock::new(Vec::new()),
            ids: RwLock::new(Vec::new()),
            next_id: AtomicU64::new(0),
            nulls: RwLock::new(Vec::new()),
            deleted: RwLock::new(Vec::new()),
            id_to_idx: RwLock::new(HashMap::new()),
        };

        // Write initial file
        storage.save()?;

        Ok(storage)
    }

    /// Open existing V3 storage (lazy - only reads header and index)
    pub fn open(path: &Path) -> io::Result<Self> {
        let file = File::open(path)?;
        
        // Read header using pread (no seek needed)
        let mut header_bytes = [0u8; HEADER_SIZE_V3];
        file.read_exact_at(&mut header_bytes, 0)?;
        let header = OnDemandHeader::from_bytes(&header_bytes)?;

        // Read schema
        let schema_size = header.column_index_offset - header.schema_offset;
        let mut schema_bytes = vec![0u8; schema_size as usize];
        file.read_exact_at(&mut schema_bytes, header.schema_offset)?;
        let schema = OnDemandSchema::from_bytes(&schema_bytes)?;

        // Read column index
        let index_size = header.column_count as usize * COLUMN_INDEX_ENTRY_SIZE;
        let mut index_bytes = vec![0u8; index_size];
        file.read_exact_at(&mut index_bytes, header.column_index_offset)?;
        
        let mut column_index = Vec::with_capacity(header.column_count as usize);
        for i in 0..header.column_count as usize {
            let start = i * COLUMN_INDEX_ENTRY_SIZE;
            let entry = ColumnIndexEntry::from_bytes(&index_bytes[start..start + COLUMN_INDEX_ENTRY_SIZE]);
            column_index.push(entry);
        }

        // Read IDs into memory (always needed for lookups)
        let id_count = header.row_count as usize;
        let mut id_bytes = vec![0u8; id_count * 8];
        if id_count > 0 {
            file.read_exact_at(&mut id_bytes, header.id_column_offset)?;
        }
        let mut ids = Vec::with_capacity(id_count);
        let mut id_to_idx = HashMap::with_capacity(id_count);
        for i in 0..id_count {
            let id = u64::from_le_bytes(id_bytes[i * 8..(i + 1) * 8].try_into().unwrap());
            ids.push(id);
            id_to_idx.insert(id, i);
        }
        // If no rows exist, start from 0; otherwise start after max existing ID
        let next_id = if ids.is_empty() {
            0
        } else {
            ids.iter().max().copied().unwrap_or(0) + 1
        };

        // NOTE: Column data is NOT loaded - will be read on-demand
        let columns = vec![ColumnData::new(ColumnType::Int64); header.column_count as usize];
        let nulls = vec![Vec::new(); header.column_count as usize];
        
        // Initialize deleted bitmap (all zeros = no deleted rows)
        let deleted_len = (id_count + 7) / 8;
        let deleted = vec![0u8; deleted_len];

        Ok(Self {
            path: path.to_path_buf(),
            file: RwLock::new(Some(file)),
            header: RwLock::new(header),
            schema: RwLock::new(schema),
            column_index: RwLock::new(column_index),
            columns: RwLock::new(columns),
            ids: RwLock::new(ids),
            next_id: AtomicU64::new(next_id),
            nulls: RwLock::new(nulls),
            deleted: RwLock::new(deleted),
            id_to_idx: RwLock::new(id_to_idx),
        })
    }

    /// Create or open storage
    pub fn open_or_create(path: &Path) -> io::Result<Self> {
        if path.exists() {
            Self::open(path)
        } else {
            Self::create(path)
        }
    }

    /// Open for write - loads all existing data into memory for append operations
    pub fn open_for_write(path: &Path) -> io::Result<Self> {
        if !path.exists() {
            return Self::create(path);
        }
        
        let storage = Self::open(path)?;
        
        // Load all existing columns into memory for append operations
        if storage.header.read().row_count > 0 {
            let column_data = storage.read_columns(None, 0, None)?;
            let schema = storage.schema.read();
            let mut columns = storage.columns.write();
            
            for (col_idx, (col_name, col_type)) in schema.columns.iter().enumerate() {
                if let Some(data) = column_data.get(col_name) {
                    if col_idx < columns.len() {
                        columns[col_idx] = data.clone();
                    } else {
                        columns.push(data.clone());
                    }
                } else {
                    // Initialize empty column of correct type
                    while columns.len() <= col_idx {
                        columns.push(ColumnData::new(*col_type));
                    }
                }
            }
        }
        
        Ok(storage)
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

        let mut result = HashMap::new();

        for &col_idx in &col_indices {
            let (col_name, col_type) = &schema.columns[col_idx];
            let index_entry = &column_index[col_idx];
            
            // Read column data from file
            let col_data = self.read_column_range(
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

        self.read_column_scattered(file, index_entry, *col_type, row_indices, total_rows)
    }

    /// Read IDs for a row range
    pub fn read_ids(&self, start_row: usize, row_count: Option<usize>) -> io::Result<Vec<u64>> {
        let ids = self.ids.read();
        let total = ids.len();
        let start = start_row.min(total);
        let count = row_count.map(|c| c.min(total - start)).unwrap_or(total - start);
        Ok(ids[start..start + count].to_vec())
    }

    // ========================================================================
    // Internal read helpers
    // ========================================================================

    fn read_column_range(
        &self,
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
                let byte_offset = HEADER_SIZE + (start_row * 8) as u64;
                let byte_count = row_count * 8;
                
                let mut data = vec![0u8; byte_count];
                file.read_exact_at(&mut data, index.data_offset + byte_offset)?;
                
                let mut values = Vec::with_capacity(row_count);
                for i in 0..row_count {
                    let val = i64::from_le_bytes(data[i * 8..(i + 1) * 8].try_into().unwrap());
                    values.push(val);
                }
                Ok(ColumnData::Int64(values))
            }
            ColumnType::Float64 | ColumnType::Float32 => {
                // Format: [count:u64][values:f64*]
                let byte_offset = HEADER_SIZE + (start_row * 8) as u64;
                let byte_count = row_count * 8;
                
                let mut data = vec![0u8; byte_count];
                file.read_exact_at(&mut data, index.data_offset + byte_offset)?;
                
                let mut values = Vec::with_capacity(row_count);
                for i in 0..row_count {
                    let val = f64::from_le_bytes(data[i * 8..(i + 1) * 8].try_into().unwrap());
                    values.push(val);
                }
                Ok(ColumnData::Float64(values))
            }
            ColumnType::Bool => {
                // Format: [len:u64][packed_bits...]
                let start_byte = start_row / 8;
                let end_byte = (start_row + row_count + 7) / 8;
                let byte_count = end_byte - start_byte;
                
                let mut packed = vec![0u8; byte_count];
                file.read_exact_at(&mut packed, index.data_offset + HEADER_SIZE + start_byte as u64)?;
                
                Ok(ColumnData::Bool { data: packed, len: row_count })
            }
            ColumnType::String | ColumnType::Binary => {
                // Variable-length type: need to read offsets first
                self.read_variable_column_range(file, index, dtype, start_row, row_count)
            }
            ColumnType::Null => {
                Ok(ColumnData::Int64(vec![0; row_count]))
            }
        }
    }

    fn read_variable_column_range(
        &self,
        file: &File,
        index: &ColumnIndexEntry,
        dtype: ColumnType,
        start_row: usize,
        row_count: usize,
    ) -> io::Result<ColumnData> {
        // Variable-length format: [count:u64][offsets:u32*][data_len:u64][data:bytes]
        // Read header to get total count
        let mut header_buf = [0u8; 8];
        file.read_exact_at(&mut header_buf, index.data_offset)?;
        let total_count = u64::from_le_bytes(header_buf) as usize;
        
        if start_row >= total_count {
            return Ok(ColumnData::String { offsets: vec![0], data: Vec::new() });
        }
        
        let actual_count = row_count.min(total_count - start_row);
        
        // Read relevant offsets (need start_row to start_row + actual_count + 1)
        let offset_start = 8 + start_row * 4; // skip count header
        let offset_count = actual_count + 1;
        let mut offset_buf = vec![0u8; offset_count * 4];
        file.read_exact_at(&mut offset_buf, index.data_offset + offset_start as u64)?;
        
        let mut offsets = Vec::with_capacity(offset_count);
        for i in 0..offset_count {
            let off = u32::from_le_bytes(offset_buf[i * 4..(i + 1) * 4].try_into().unwrap());
            offsets.push(off);
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
            file.read_exact_at(&mut data, data_offset_in_file)?;
        }
        
        // Normalize offsets to start at 0
        let base = offsets[0];
        for off in &mut offsets {
            *off -= base;
        }
        
        match dtype {
            ColumnType::String => Ok(ColumnData::String { offsets, data }),
            ColumnType::Binary => Ok(ColumnData::Binary { offsets, data }),
            _ => Err(io::Error::new(io::ErrorKind::InvalidData, "Not a variable type")),
        }
    }

    fn read_column_scattered(
        &self,
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
                let mut values = Vec::with_capacity(row_indices.len());
                let mut buf = [0u8; 8];
                for &idx in row_indices {
                    file.read_exact_at(&mut buf, index.data_offset + HEADER_SIZE + (idx * 8) as u64)?;
                    values.push(i64::from_le_bytes(buf));
                }
                Ok(ColumnData::Int64(values))
            }
            ColumnType::Float64 | ColumnType::Float32 => {
                let mut values = Vec::with_capacity(row_indices.len());
                let mut buf = [0u8; 8];
                for &idx in row_indices {
                    file.read_exact_at(&mut buf, index.data_offset + HEADER_SIZE + (idx * 8) as u64)?;
                    values.push(f64::from_le_bytes(buf));
                }
                Ok(ColumnData::Float64(values))
            }
            ColumnType::String | ColumnType::Binary => {
                // Variable-length: read full column for now
                // TODO: Optimize with offset index for truly scattered reads
                self.read_variable_column_range(file, index, dtype, 0, _total_rows)
            }
            _ => Err(io::Error::new(io::ErrorKind::InvalidData, "Unsupported type for scattered read")),
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
            header.row_count += row_count as u64;
            header.column_count = self.schema.read().column_count() as u32;
            header.modified_at = chrono::Utc::now().timestamp();
        }
        
        // Update id_to_idx mapping
        {
            let ids_guard = self.ids.read();
            let mut id_to_idx = self.id_to_idx.write();
            let start_idx = ids_guard.len() - ids.len();
            for (i, &id) in ids.iter().enumerate() {
                id_to_idx.insert(id, start_idx + i);
            }
        }
        
        // Extend deleted bitmap with zeros for new rows
        {
            let mut deleted = self.deleted.write();
            let new_len = (self.ids.read().len() + 7) / 8;
            deleted.resize(new_len, 0);
        }

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
        
        // Update id_to_idx mapping
        {
            let ids_guard = self.ids.read();
            let mut id_to_idx = self.id_to_idx.write();
            let start_idx = ids_guard.len() - ids.len();
            for (i, &id) in ids.iter().enumerate() {
                id_to_idx.insert(id, start_idx + i);
            }
        }
        
        // Extend deleted bitmap with zeros for new rows
        {
            let mut deleted = self.deleted.write();
            let new_len = (self.ids.read().len() + 7) / 8;
            deleted.resize(new_len, 0);
        }

        Ok(ids)
    }

    // ========================================================================
    // Delete/Update APIs
    // ========================================================================

    /// Delete a row by ID (soft delete)
    /// Returns true if the row was found and deleted
    pub fn delete(&self, id: u64) -> bool {
        let id_to_idx = self.id_to_idx.read();
        if let Some(&row_idx) = id_to_idx.get(&id) {
            let mut deleted = self.deleted.write();
            let byte_idx = row_idx / 8;
            let bit_idx = row_idx % 8;
            
            // Ensure bitmap is large enough
            if byte_idx >= deleted.len() {
                deleted.resize(byte_idx + 1, 0);
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
        let id_to_idx = self.id_to_idx.read();
        let mut deleted = self.deleted.write();
        let mut all_found = true;
        
        for &id in ids {
            if let Some(&row_idx) = id_to_idx.get(&id) {
                let byte_idx = row_idx / 8;
                let bit_idx = row_idx % 8;
                
                if byte_idx >= deleted.len() {
                    deleted.resize(byte_idx + 1, 0);
                }
                
                deleted[byte_idx] |= 1 << bit_idx;
            } else {
                all_found = false;
            }
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
        let id_to_idx = self.id_to_idx.read();
        if let Some(&row_idx) = id_to_idx.get(&id) {
            !self.is_deleted(row_idx)
        } else {
            false
        }
    }

    /// Get row index for an ID (None if not found or deleted)
    pub fn get_row_idx(&self, id: u64) -> Option<usize> {
        let id_to_idx = self.id_to_idx.read();
        if let Some(&row_idx) = id_to_idx.get(&id) {
            if !self.is_deleted(row_idx) {
                Some(row_idx)
            } else {
                None
            }
        } else {
            None
        }
    }

    /// Get the count of non-deleted rows
    pub fn active_row_count(&self) -> u64 {
        let ids = self.ids.read();
        let deleted = self.deleted.read();
        let mut count = 0u64;
        
        for i in 0..ids.len() {
            let byte_idx = i / 8;
            let bit_idx = i % 8;
            let is_deleted = byte_idx < deleted.len() && (deleted[byte_idx] >> bit_idx) & 1 == 1;
            if !is_deleted {
                count += 1;
            }
        }
        
        count
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
        
        // Update id_to_idx mapping
        {
            let ids_guard = self.ids.read();
            let mut id_to_idx = self.id_to_idx.write();
            let row_idx = ids_guard.len() - 1;
            id_to_idx.insert(id, row_idx);
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

    /// Save to file (full rewrite with V3 format)
    pub fn save(&self) -> io::Result<()> {
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

        // Filter out deleted rows - get indices of non-deleted rows
        let active_indices: Vec<usize> = (0..ids.len())
            .filter(|&i| {
                let byte_idx = i / 8;
                let bit_idx = i % 8;
                byte_idx >= deleted.len() || (deleted[byte_idx] >> bit_idx) & 1 == 0
            })
            .collect();
        
        let active_ids: Vec<u64> = active_indices.iter().map(|&i| ids[i]).collect();

        // Serialize schema
        let schema_bytes = schema.to_bytes();

        // Calculate offsets
        let schema_offset = HEADER_SIZE_V3 as u64;
        let column_index_offset = schema_offset + schema_bytes.len() as u64;
        let id_column_offset = column_index_offset + (schema.column_count() * COLUMN_INDEX_ENTRY_SIZE) as u64;

        // Build column index while calculating data offsets (using active row count)
        let mut current_offset = id_column_offset + (active_ids.len() * 8) as u64;
        let mut column_index_entries = Vec::with_capacity(schema.column_count());

        // Pre-compute filtered column data for accurate size calculation
        let mut filtered_columns: Vec<ColumnData> = Vec::with_capacity(schema.column_count());
        for col_idx in 0..schema.column_count() {
            if col_idx < columns.len() {
                filtered_columns.push(columns[col_idx].filter_by_indices(&active_indices));
            } else {
                filtered_columns.push(ColumnData::new(ColumnType::Int64));
            }
        }

        for (col_idx, _col_def) in schema.columns.iter().enumerate() {
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
            header.column_count = schema.column_count() as u32;
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
        for (col_idx, col_def) in schema.columns.iter().enumerate() {
            // Build filtered null bitmap for active rows only
            let original_nulls = nulls.get(col_idx).map(|v| v.as_slice()).unwrap_or(&[]);
            let expected_len = (active_ids.len() + 7) / 8;
            let mut filtered_nulls = vec![0u8; expected_len];
            for (new_idx, &old_idx) in active_indices.iter().enumerate() {
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

        // Update column index in memory
        *self.column_index.write() = column_index_entries;

        // Reopen file for reading
        drop(writer);
        let file = File::open(&self.path)?;
        *self.file.write() = Some(file);

        Ok(())
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
    pub fn insert_rows(&self, rows: &[HashMap<String, ColumnValue>]) -> io::Result<Vec<u64>> {
        if rows.is_empty() {
            return Ok(Vec::new());
        }
        
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
            
            // Update id_to_idx mapping
            {
                let ids_guard = self.ids.read();
                let mut id_to_idx = self.id_to_idx.write();
                let start_idx = ids_guard.len() - ids.len();
                for (i, &id) in ids.iter().enumerate() {
                    id_to_idx.insert(id, start_idx + i);
                }
            }
            
            // Extend deleted bitmap
            {
                let mut deleted = self.deleted.write();
                let new_len = (self.ids.read().len() + 7) / 8;
                deleted.resize(new_len, 0);
            }
            
            return Ok(ids);
        }

        // Collect column data by type
        let mut int_columns: HashMap<String, Vec<i64>> = HashMap::new();
        let mut float_columns: HashMap<String, Vec<f64>> = HashMap::new();
        let mut string_columns: HashMap<String, Vec<String>> = HashMap::new();
        let mut binary_columns: HashMap<String, Vec<Vec<u8>>> = HashMap::new();
        let mut bool_columns: HashMap<String, Vec<bool>> = HashMap::new();

        // First pass: determine column types from ALL rows (not just first)
        // This handles heterogeneous schemas where different rows have different columns
        for row in rows {
            for (key, val) in row {
                // Skip if column already registered
                if int_columns.contains_key(key) || float_columns.contains_key(key) 
                    || string_columns.contains_key(key) || binary_columns.contains_key(key)
                    || bool_columns.contains_key(key) {
                    continue;
                }
                match val {
                    ColumnValue::Int64(_) => { int_columns.insert(key.clone(), Vec::with_capacity(rows.len())); }
                    ColumnValue::Float64(_) => { float_columns.insert(key.clone(), Vec::with_capacity(rows.len())); }
                    ColumnValue::String(_) => { string_columns.insert(key.clone(), Vec::with_capacity(rows.len())); }
                    ColumnValue::Binary(_) => { binary_columns.insert(key.clone(), Vec::with_capacity(rows.len())); }
                    ColumnValue::Bool(_) => { bool_columns.insert(key.clone(), Vec::with_capacity(rows.len())); }
                    ColumnValue::Null => {
                        // Store NULL as a string column with special marker
                        string_columns.insert(key.clone(), Vec::with_capacity(rows.len()));
                    }
                }
            }
        }

        // Second pass: collect values, ensuring all columns have same length
        // For each row, add value if present or default if missing
        for row in rows {
            // Process int columns
            for (key, col) in int_columns.iter_mut() {
                if let Some(ColumnValue::Int64(v)) = row.get(key) {
                    col.push(*v);
                } else {
                    col.push(0); // Default for missing int
                }
            }
            // Process float columns
            for (key, col) in float_columns.iter_mut() {
                if let Some(ColumnValue::Float64(v)) = row.get(key) {
                    col.push(*v);
                } else {
                    col.push(0.0); // Default for missing float
                }
            }
            // Process string columns
            for (key, col) in string_columns.iter_mut() {
                match row.get(key) {
                    Some(ColumnValue::String(v)) => col.push(v.clone()),
                    Some(ColumnValue::Null) => col.push("\x00__NULL__\x00".to_string()), // Special NULL marker
                    _ => col.push(String::new()), // Default for missing string
                }
            }
            // Process binary columns
            for (key, col) in binary_columns.iter_mut() {
                if let Some(ColumnValue::Binary(v)) = row.get(key) {
                    col.push(v.clone());
                } else {
                    col.push(Vec::new()); // Default for missing binary
                }
            }
            // Process bool columns
            for (key, col) in bool_columns.iter_mut() {
                if let Some(ColumnValue::Bool(v)) = row.get(key) {
                    col.push(*v);
                } else {
                    col.push(false); // Default for missing bool
                }
            }
        }

        self.insert_typed(int_columns, float_columns, string_columns, binary_columns, bool_columns)
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

    /// Close storage
    pub fn close(&self) -> io::Result<()> {
        self.save()
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
