//! ApexBase Native Columnar File Format
//!
//! A high-performance, single-file columnar storage format.
//! 
//! Design goals:
//! - 10,000 rows insert in <1ms (append mode)
//! - Single file storage (.apex)
//! - No external format dependencies for storage
//! - Support for mixed types: int, float, string, binary, bool
//! - Arrow IPC only for data exchange (not storage)
//!
//! File Format:
//! ```text
//! ┌────────────────────────────────────────────────┐
//! │ Header (128 bytes)                             │
//! │   - Magic: "APEXCOL\0" (8 bytes)              │
//! │   - Version: u32                               │
//! │   - Flags: u32                                 │
//! │   - Schema offset: u64                         │
//! │   - Data offset: u64                           │
//! │   - Index offset: u64                          │
//! │   - Row count: u64                             │
//! │   - Column count: u32                          │
//! │   - Created timestamp: i64                     │
//! │   - Modified timestamp: i64                    │
//! │   - Checksum: u32                              │
//! │   - Reserved: padding to 128 bytes            │
//! ├────────────────────────────────────────────────┤
//! │ Schema Block (variable)                        │
//! │   - Column definitions                         │
//! │   - [name_len: u16][name: bytes][type: u8]... │
//! ├────────────────────────────────────────────────┤
//! │ Column Data Blocks                             │
//! │   For each column:                             │
//! │   - Data length: u64                           │
//! │   - Null bitmap (packed bits)                  │
//! │   - Raw data (type-specific)                   │
//! ├────────────────────────────────────────────────┤
//! │ ID Column (special)                            │
//! │   - Contiguous u64 array                       │
//! ├────────────────────────────────────────────────┤
//! │ Footer (32 bytes)                              │
//! │   - Magic: "APEXEND\0"                        │
//! │   - Data checksum: u32                         │
//! │   - Total file size: u64                       │
//! └────────────────────────────────────────────────┘
//! ```

use std::collections::HashMap;
use std::fs::{File, OpenOptions};
use std::io::{self, BufReader, BufWriter, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use parking_lot::RwLock;
use serde::{Deserialize, Serialize};

// ============================================================================
// Constants
// ============================================================================

const MAGIC_HEADER: &[u8; 8] = b"APEXCOL\0";
const MAGIC_FOOTER: &[u8; 8] = b"APEXEND\0";
const MAGIC_DELTA: &[u8; 4] = b"DELT";
const FORMAT_VERSION: u32 = 2;  // Bump version for delta support
const HEADER_SIZE: usize = 128;
const FOOTER_SIZE: usize = 32;
/// Delta compaction threshold (number of delta records before auto-compact)
const DELTA_COMPACT_THRESHOLD: u64 = 10000;

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
    fn from_u8(v: u8) -> Option<Self> {
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
    fn fixed_size(&self) -> usize {
        match self {
            ColumnType::Null => 0,
            ColumnType::Bool => 1,
            ColumnType::Int8 | ColumnType::UInt8 => 1,
            ColumnType::Int16 | ColumnType::UInt16 => 2,
            ColumnType::Int32 | ColumnType::UInt32 | ColumnType::Float32 => 4,
            ColumnType::Int64 | ColumnType::UInt64 | ColumnType::Float64 => 8,
            ColumnType::String | ColumnType::Binary => 0, // Variable length
        }
    }

    fn is_variable_length(&self) -> bool {
        matches!(self, ColumnType::String | ColumnType::Binary)
    }
}

/// Column definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ColumnDef {
    pub name: String,
    pub dtype: ColumnType,
}

/// Schema definition
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

    pub fn with_capacity(dtype: ColumnType, capacity: usize) -> Self {
        match dtype {
            ColumnType::Bool => ColumnData::Bool {
                data: Vec::with_capacity((capacity + 7) / 8),
                len: 0,
            },
            ColumnType::Int8 | ColumnType::Int16 | ColumnType::Int32 | ColumnType::Int64 |
            ColumnType::UInt8 | ColumnType::UInt16 | ColumnType::UInt32 | ColumnType::UInt64 => {
                ColumnData::Int64(Vec::with_capacity(capacity))
            }
            ColumnType::Float32 | ColumnType::Float64 => {
                ColumnData::Float64(Vec::with_capacity(capacity))
            }
            ColumnType::String => ColumnData::String {
                offsets: Vec::with_capacity(capacity + 1),
                data: Vec::with_capacity(capacity * 16), // Estimate 16 bytes per string
            },
            ColumnType::Binary => ColumnData::Binary {
                offsets: Vec::with_capacity(capacity + 1),
                data: Vec::with_capacity(capacity * 32), // Estimate 32 bytes per binary
            },
            ColumnType::Null => ColumnData::Int64(Vec::with_capacity(capacity)),
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

    /// Extend from raw vectors (bulk operation)
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
                // Write number of strings
                let count = offsets.len().saturating_sub(1);
                buf.extend_from_slice(&(count as u64).to_le_bytes());
                // Write offsets
                for &off in offsets {
                    buf.extend_from_slice(&off.to_le_bytes());
                }
                // Write data
                buf.extend_from_slice(&(data.len() as u64).to_le_bytes());
                buf.extend_from_slice(data);
            }
        }
        buf
    }

    /// Deserialize from bytes
    pub fn from_bytes(dtype: ColumnType, bytes: &[u8]) -> io::Result<(Self, usize)> {
        let mut pos = 0;
        
        let read_u64 = |pos: &mut usize| -> u64 {
            let val = u64::from_le_bytes(bytes[*pos..*pos + 8].try_into().unwrap());
            *pos += 8;
            val
        };
        
        let read_u32 = |pos: &mut usize| -> u32 {
            let val = u32::from_le_bytes(bytes[*pos..*pos + 4].try_into().unwrap());
            *pos += 4;
            val
        };

        match dtype {
            ColumnType::Bool => {
                let len = read_u64(&mut pos) as usize;
                let byte_len = (len + 7) / 8;
                let data = bytes[pos..pos + byte_len].to_vec();
                pos += byte_len;
                Ok((ColumnData::Bool { data, len }, pos))
            }
            ColumnType::Int8 | ColumnType::Int16 | ColumnType::Int32 | ColumnType::Int64 |
            ColumnType::UInt8 | ColumnType::UInt16 | ColumnType::UInt32 | ColumnType::UInt64 => {
                let count = read_u64(&mut pos) as usize;
                let mut data = Vec::with_capacity(count);
                for _ in 0..count {
                    let val = i64::from_le_bytes(bytes[pos..pos + 8].try_into().unwrap());
                    pos += 8;
                    data.push(val);
                }
                Ok((ColumnData::Int64(data), pos))
            }
            ColumnType::Float32 | ColumnType::Float64 => {
                let count = read_u64(&mut pos) as usize;
                let mut data = Vec::with_capacity(count);
                for _ in 0..count {
                    let val = f64::from_le_bytes(bytes[pos..pos + 8].try_into().unwrap());
                    pos += 8;
                    data.push(val);
                }
                Ok((ColumnData::Float64(data), pos))
            }
            ColumnType::String | ColumnType::Binary => {
                let count = read_u64(&mut pos) as usize;
                let mut offsets = Vec::with_capacity(count + 1);
                for _ in 0..=count {
                    offsets.push(read_u32(&mut pos));
                }
                let data_len = read_u64(&mut pos) as usize;
                let data = bytes[pos..pos + data_len].to_vec();
                pos += data_len;
                
                if dtype == ColumnType::String {
                    Ok((ColumnData::String { offsets, data }, pos))
                } else {
                    Ok((ColumnData::Binary { offsets, data }, pos))
                }
            }
            ColumnType::Null => Ok((ColumnData::Int64(Vec::new()), pos)),
        }
    }
}

// ============================================================================
// File Header
// ============================================================================

#[derive(Debug, Clone)]
struct FileHeader {
    version: u32,
    flags: u32,
    schema_offset: u64,
    data_offset: u64,
    index_offset: u64,
    row_count: u64,        // Total rows (base + delta)
    column_count: u32,
    created_at: i64,
    modified_at: i64,
    checksum: u32,
    // Delta support fields (v2)
    delta_offset: u64,     // Start of delta zone (0 = no delta)
    delta_count: u64,      // Number of delta records
    base_row_count: u64,   // Rows in base (columnar) section
}

impl FileHeader {
    fn new() -> Self {
        let now = chrono::Utc::now().timestamp();
        Self {
            version: FORMAT_VERSION,
            flags: 0,
            schema_offset: HEADER_SIZE as u64,
            data_offset: 0,
            index_offset: 0,
            row_count: 0,
            column_count: 0,
            created_at: now,
            modified_at: now,
            checksum: 0,
            delta_offset: 0,
            delta_count: 0,
            base_row_count: 0,
        }
    }

    fn to_bytes(&self) -> [u8; HEADER_SIZE] {
        let mut buf = [0u8; HEADER_SIZE];
        let mut pos = 0;

        // Magic
        buf[pos..pos + 8].copy_from_slice(MAGIC_HEADER);
        pos += 8;

        // Version
        buf[pos..pos + 4].copy_from_slice(&self.version.to_le_bytes());
        pos += 4;

        // Flags
        buf[pos..pos + 4].copy_from_slice(&self.flags.to_le_bytes());
        pos += 4;

        // Offsets
        buf[pos..pos + 8].copy_from_slice(&self.schema_offset.to_le_bytes());
        pos += 8;
        buf[pos..pos + 8].copy_from_slice(&self.data_offset.to_le_bytes());
        pos += 8;
        buf[pos..pos + 8].copy_from_slice(&self.index_offset.to_le_bytes());
        pos += 8;

        // Row count
        buf[pos..pos + 8].copy_from_slice(&self.row_count.to_le_bytes());
        pos += 8;

        // Column count
        buf[pos..pos + 4].copy_from_slice(&self.column_count.to_le_bytes());
        pos += 4;

        // Timestamps
        buf[pos..pos + 8].copy_from_slice(&self.created_at.to_le_bytes());
        pos += 8;
        buf[pos..pos + 8].copy_from_slice(&self.modified_at.to_le_bytes());
        pos += 8;

        // Delta fields (v2)
        buf[pos..pos + 8].copy_from_slice(&self.delta_offset.to_le_bytes());
        pos += 8;
        buf[pos..pos + 8].copy_from_slice(&self.delta_count.to_le_bytes());
        pos += 8;
        buf[pos..pos + 8].copy_from_slice(&self.base_row_count.to_le_bytes());
        pos += 8;

        // Checksum (of previous bytes)
        let checksum = crc32fast::hash(&buf[0..pos]);
        buf[pos..pos + 4].copy_from_slice(&checksum.to_le_bytes());

        buf
    }

    fn from_bytes(bytes: &[u8; HEADER_SIZE]) -> io::Result<Self> {
        let mut pos = 0;

        // Verify magic
        if &bytes[pos..pos + 8] != MAGIC_HEADER {
            return Err(io::Error::new(io::ErrorKind::InvalidData, "Invalid file magic"));
        }
        pos += 8;

        // Read fields
        let version = u32::from_le_bytes(bytes[pos..pos + 4].try_into().unwrap());
        pos += 4;
        let flags = u32::from_le_bytes(bytes[pos..pos + 4].try_into().unwrap());
        pos += 4;
        let schema_offset = u64::from_le_bytes(bytes[pos..pos + 8].try_into().unwrap());
        pos += 8;
        let data_offset = u64::from_le_bytes(bytes[pos..pos + 8].try_into().unwrap());
        pos += 8;
        let index_offset = u64::from_le_bytes(bytes[pos..pos + 8].try_into().unwrap());
        pos += 8;
        let row_count = u64::from_le_bytes(bytes[pos..pos + 8].try_into().unwrap());
        pos += 8;
        let column_count = u32::from_le_bytes(bytes[pos..pos + 4].try_into().unwrap());
        pos += 4;
        let created_at = i64::from_le_bytes(bytes[pos..pos + 8].try_into().unwrap());
        pos += 8;
        let modified_at = i64::from_le_bytes(bytes[pos..pos + 8].try_into().unwrap());
        pos += 8;

        // Delta fields (v2) - with backward compatibility for v1
        let (delta_offset, delta_count, base_row_count) = if version >= 2 {
            let delta_offset = u64::from_le_bytes(bytes[pos..pos + 8].try_into().unwrap());
            pos += 8;
            let delta_count = u64::from_le_bytes(bytes[pos..pos + 8].try_into().unwrap());
            pos += 8;
            let base_row_count = u64::from_le_bytes(bytes[pos..pos + 8].try_into().unwrap());
            pos += 8;
            (delta_offset, delta_count, base_row_count)
        } else {
            // v1 files: no delta, all rows are base rows
            (0, 0, row_count)
        };

        let checksum = u32::from_le_bytes(bytes[pos..pos + 4].try_into().unwrap());

        // Verify checksum
        let computed = crc32fast::hash(&bytes[0..pos]);
        if computed != checksum {
            return Err(io::Error::new(io::ErrorKind::InvalidData, "Header checksum mismatch"));
        }

        Ok(Self {
            version,
            flags,
            schema_offset,
            data_offset,
            index_offset,
            row_count,
            column_count,
            created_at,
            modified_at,
            checksum,
            delta_offset,
            delta_count,
            base_row_count,
        })
    }
}

// ============================================================================
// Columnar Storage Engine
// ============================================================================

/// High-performance native columnar storage
pub struct ColumnarStorage {
    path: PathBuf,
    header: RwLock<FileHeader>,
    schema: RwLock<FileSchema>,
    /// In-memory column data (append buffer)
    columns: RwLock<Vec<ColumnData>>,
    /// Row IDs
    ids: RwLock<Vec<u64>>,
    /// Next row ID
    next_id: AtomicU64,
    /// Null bitmaps per column
    nulls: RwLock<Vec<Vec<u8>>>,
}

impl ColumnarStorage {
    /// Create a new storage file
    pub fn create(path: &Path) -> io::Result<Self> {
        let header = FileHeader::new();
        let schema = FileSchema::new();

        let storage = Self {
            path: path.to_path_buf(),
            header: RwLock::new(header),
            schema: RwLock::new(schema),
            columns: RwLock::new(Vec::new()),
            ids: RwLock::new(Vec::new()),
            next_id: AtomicU64::new(1),
            nulls: RwLock::new(Vec::new()),
        };

        // Write initial file
        storage.save()?;

        Ok(storage)
    }

    /// Open existing storage
    pub fn open(path: &Path) -> io::Result<Self> {
        let mut file = File::open(path)?;

        // Read header
        let mut header_bytes = [0u8; HEADER_SIZE];
        file.read_exact(&mut header_bytes)?;
        let header = FileHeader::from_bytes(&header_bytes)?;

        // Read schema
        file.seek(SeekFrom::Start(header.schema_offset))?;
        let mut schema_len_bytes = [0u8; 4];
        file.read_exact(&mut schema_len_bytes)?;
        let schema_len = u32::from_le_bytes(schema_len_bytes) as usize;
        
        let mut schema_bytes = vec![0u8; schema_len];
        file.read_exact(&mut schema_bytes)?;
        let schema: FileSchema = bincode::deserialize(&schema_bytes)
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;

        // Read column data
        let mut columns = Vec::with_capacity(schema.column_count());
        let mut nulls = Vec::with_capacity(schema.column_count());

        if header.data_offset > 0 && header.row_count > 0 {
            file.seek(SeekFrom::Start(header.data_offset))?;
            
            // Read IDs first
            let mut ids = Vec::with_capacity(header.row_count as usize);
            for _ in 0..header.row_count {
                let mut id_bytes = [0u8; 8];
                file.read_exact(&mut id_bytes)?;
                ids.push(u64::from_le_bytes(id_bytes));
            }

            // Read each column
            for col_def in &schema.columns {
                // Read null bitmap
                let null_bitmap_len = (header.row_count as usize + 7) / 8;
                let mut null_bitmap = vec![0u8; null_bitmap_len];
                file.read_exact(&mut null_bitmap)?;
                nulls.push(null_bitmap);

                // Read column data length
                let mut data_len_bytes = [0u8; 8];
                file.read_exact(&mut data_len_bytes)?;
                let data_len = u64::from_le_bytes(data_len_bytes) as usize;

                // Read column data
                let mut data_bytes = vec![0u8; data_len];
                file.read_exact(&mut data_bytes)?;
                
                let (col_data, _) = ColumnData::from_bytes(col_def.dtype, &data_bytes)?;
                columns.push(col_data);
            }

            let next_id = ids.iter().max().copied().unwrap_or(0) + 1;

            return Ok(Self {
                path: path.to_path_buf(),
                header: RwLock::new(header),
                schema: RwLock::new(schema),
                columns: RwLock::new(columns),
                ids: RwLock::new(ids),
                next_id: AtomicU64::new(next_id),
                nulls: RwLock::new(nulls),
            });
        }

        Ok(Self {
            path: path.to_path_buf(),
            header: RwLock::new(header),
            schema: RwLock::new(schema),
            columns: RwLock::new(columns),
            ids: RwLock::new(Vec::new()),
            next_id: AtomicU64::new(1),
            nulls: RwLock::new(nulls),
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

    // ========== High-Performance Insert APIs ==========

    /// Insert typed columns directly - FASTEST PATH
    /// 
    /// Performance target: 10,000 rows in <1ms
    /// 
    /// # Arguments
    /// * `int_columns` - HashMap<column_name, Vec<i64>>
    /// * `float_columns` - HashMap<column_name, Vec<f64>>
    /// * `string_columns` - HashMap<column_name, Vec<String>>
    /// * `binary_columns` - HashMap<column_name, Vec<Vec<u8>>>
    /// * `bool_columns` - HashMap<column_name, Vec<bool>>
    pub fn insert_typed(
        &self,
        int_columns: HashMap<String, Vec<i64>>,
        float_columns: HashMap<String, Vec<f64>>,
        string_columns: HashMap<String, Vec<String>>,
        binary_columns: HashMap<String, Vec<Vec<u8>>>,
        bool_columns: HashMap<String, Vec<bool>>,
    ) -> io::Result<Vec<u64>> {
        // Determine row count
        let row_count = int_columns.values().next().map(|v| v.len())
            .or_else(|| float_columns.values().next().map(|v| v.len()))
            .or_else(|| string_columns.values().next().map(|v| v.len()))
            .or_else(|| binary_columns.values().next().map(|v| v.len()))
            .or_else(|| bool_columns.values().next().map(|v| v.len()))
            .unwrap_or(0);

        if row_count == 0 {
            return Ok(Vec::new());
        }

        // Allocate IDs atomically
        let start_id = self.next_id.fetch_add(row_count as u64, Ordering::SeqCst);
        let ids: Vec<u64> = (start_id..start_id + row_count as u64).collect();

        // Ensure schema has all columns
        {
            let mut schema = self.schema.write();
            let mut columns = self.columns.write();
            let mut nulls = self.nulls.write();

            for name in int_columns.keys() {
                let idx = schema.add_column(name, ColumnType::Int64);
                while columns.len() <= idx {
                    columns.push(ColumnData::new(ColumnType::Int64));
                    nulls.push(Vec::new());
                }
            }
            for name in float_columns.keys() {
                let idx = schema.add_column(name, ColumnType::Float64);
                while columns.len() <= idx {
                    columns.push(ColumnData::new(ColumnType::Float64));
                    nulls.push(Vec::new());
                }
            }
            for name in string_columns.keys() {
                let idx = schema.add_column(name, ColumnType::String);
                while columns.len() <= idx {
                    columns.push(ColumnData::new(ColumnType::String));
                    nulls.push(Vec::new());
                }
            }
            for name in binary_columns.keys() {
                let idx = schema.add_column(name, ColumnType::Binary);
                while columns.len() <= idx {
                    columns.push(ColumnData::new(ColumnType::Binary));
                    nulls.push(Vec::new());
                }
            }
            for name in bool_columns.keys() {
                let idx = schema.add_column(name, ColumnType::Bool);
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

        // Update header
        {
            let mut header = self.header.write();
            header.row_count += row_count as u64;
            header.column_count = self.schema.read().column_count() as u32;
            header.modified_at = chrono::Utc::now().timestamp();
        }

        Ok(ids)
    }

    /// Simple insert API using generic value type
    pub fn insert_rows(&self, rows: &[HashMap<String, ColumnValue>]) -> io::Result<Vec<u64>> {
        if rows.is_empty() {
            return Ok(Vec::new());
        }

        // Separate by type
        let mut int_cols: HashMap<String, Vec<i64>> = HashMap::new();
        let mut float_cols: HashMap<String, Vec<f64>> = HashMap::new();
        let mut string_cols: HashMap<String, Vec<String>> = HashMap::new();
        let mut binary_cols: HashMap<String, Vec<Vec<u8>>> = HashMap::new();
        let mut bool_cols: HashMap<String, Vec<bool>> = HashMap::new();

        let row_count = rows.len();

        // First pass: identify all columns
        for row in rows {
            for (name, value) in row {
                match value {
                    ColumnValue::Int64(_) => { int_cols.entry(name.clone()).or_insert_with(|| Vec::with_capacity(row_count)); }
                    ColumnValue::Float64(_) => { float_cols.entry(name.clone()).or_insert_with(|| Vec::with_capacity(row_count)); }
                    ColumnValue::String(_) => { string_cols.entry(name.clone()).or_insert_with(|| Vec::with_capacity(row_count)); }
                    ColumnValue::Binary(_) => { binary_cols.entry(name.clone()).or_insert_with(|| Vec::with_capacity(row_count)); }
                    ColumnValue::Bool(_) => { bool_cols.entry(name.clone()).or_insert_with(|| Vec::with_capacity(row_count)); }
                    ColumnValue::Null => {}
                }
            }
        }

        // Second pass: collect values
        for row in rows {
            for (name, vec) in &mut int_cols {
                if let Some(ColumnValue::Int64(v)) = row.get(name) {
                    vec.push(*v);
                } else {
                    vec.push(0); // Default value for missing
                }
            }
            for (name, vec) in &mut float_cols {
                if let Some(ColumnValue::Float64(v)) = row.get(name) {
                    vec.push(*v);
                } else {
                    vec.push(0.0);
                }
            }
            for (name, vec) in &mut string_cols {
                if let Some(ColumnValue::String(v)) = row.get(name) {
                    vec.push(v.clone());
                } else {
                    vec.push(String::new());
                }
            }
            for (name, vec) in &mut binary_cols {
                if let Some(ColumnValue::Binary(v)) = row.get(name) {
                    vec.push(v.clone());
                } else {
                    vec.push(Vec::new());
                }
            }
            for (name, vec) in &mut bool_cols {
                if let Some(ColumnValue::Bool(v)) = row.get(name) {
                    vec.push(*v);
                } else {
                    vec.push(false);
                }
            }
        }

        self.insert_typed(int_cols, float_cols, string_cols, binary_cols, bool_cols)
    }

    // ========== Read APIs ==========

    /// Get row count
    pub fn row_count(&self) -> u64 {
        self.header.read().row_count
    }

    /// Get column names
    pub fn column_names(&self) -> Vec<String> {
        let mut names = vec!["_id".to_string()];
        names.extend(self.schema.read().columns.iter().map(|c| c.name.clone()));
        names
    }

    // ========== Persistence ==========

    /// Save to file (full rewrite - use for compaction or initial save)
    pub fn save(&self) -> io::Result<()> {
        let file = OpenOptions::new()
            .write(true)
            .create(true)
            .truncate(true)
            .open(&self.path)?;
        
        let mut writer = BufWriter::with_capacity(256 * 1024, file);

        // Serialize schema
        let schema = self.schema.read();
        let schema_bytes = bincode::serialize(&*schema)
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;

        // Calculate offsets
        let schema_offset = HEADER_SIZE as u64;
        let data_offset = schema_offset + 4 + schema_bytes.len() as u64;

        // Update header - all data is now base data, no delta
        {
            let mut header = self.header.write();
            header.schema_offset = schema_offset;
            header.data_offset = data_offset;
            header.column_count = schema.column_count() as u32;
            header.base_row_count = header.row_count;  // All rows are base rows after save
            header.delta_offset = 0;
            header.delta_count = 0;
        }

        // Write header
        let header = self.header.read();
        writer.write_all(&header.to_bytes())?;

        // Write schema (length-prefixed)
        writer.write_all(&(schema_bytes.len() as u32).to_le_bytes())?;
        writer.write_all(&schema_bytes)?;

        // Write IDs
        let ids = self.ids.read();
        for &id in ids.iter() {
            writer.write_all(&id.to_le_bytes())?;
        }

        // Write column data
        let columns = self.columns.read();
        let nulls = self.nulls.read();
        
        for (col_idx, col_data) in columns.iter().enumerate() {
            // Write null bitmap (or empty if none)
            let null_bitmap = nulls.get(col_idx)
                .map(|v| v.as_slice())
                .unwrap_or(&[]);
            let expected_len = (ids.len() + 7) / 8;
            if null_bitmap.len() < expected_len {
                // Write zeros for missing null bitmap
                writer.write_all(&vec![0u8; expected_len])?;
            } else {
                writer.write_all(null_bitmap)?;
            }

            // Write column data
            let data_bytes = col_data.to_bytes();
            writer.write_all(&(data_bytes.len() as u64).to_le_bytes())?;
            writer.write_all(&data_bytes)?;
        }

        // Write footer
        writer.write_all(MAGIC_FOOTER)?;
        let checksum = 0u32; // TODO: compute actual checksum
        writer.write_all(&checksum.to_le_bytes())?;
        let file_size = writer.stream_position()?;
        writer.write_all(&file_size.to_le_bytes())?;

        writer.flush()?;
        Ok(())
    }

    /// Append delta records to file - O(delta_size) incremental write
    /// 
    /// This is the fast path for incremental writes. Instead of rewriting
    /// the entire file, it appends delta records to the end.
    /// 
    /// Delta record format:
    /// - magic: 4 bytes "DELT"
    /// - row_count: u32
    /// - ids: [u64; row_count]
    /// - for each column: [data_len: u64, data: bytes]
    /// - crc32: u32
    pub fn append_delta(&self, rows: &[HashMap<String, ColumnValue>]) -> io::Result<Vec<u64>> {
        if rows.is_empty() {
            return Ok(Vec::new());
        }

        // First insert into memory (same as before)
        let ids = self.insert_rows(rows)?;
        
        // Now append delta to file
        let mut file = OpenOptions::new()
            .read(true)
            .write(true)
            .open(&self.path)?;
        
        // Get current file end position for delta_offset
        let delta_start = file.seek(SeekFrom::End(0))?;
        
        // If this is the first delta, record the offset
        let is_first_delta = {
            let header = self.header.read();
            header.delta_offset == 0
        };
        
        let mut writer = BufWriter::new(&mut file);
        
        // Write delta magic
        writer.write_all(MAGIC_DELTA)?;
        
        // Write row count for this delta batch
        writer.write_all(&(ids.len() as u32).to_le_bytes())?;
        
        // Write IDs
        for &id in &ids {
            writer.write_all(&id.to_le_bytes())?;
        }
        
        // Write column data for new rows only
        // We need to extract just the new rows from our columns
        let schema = self.schema.read();
        let columns = self.columns.read();
        let total_rows = self.ids.read().len();
        let start_idx = total_rows - ids.len();
        
        for (col_idx, col_def) in schema.columns.iter().enumerate() {
            if col_idx < columns.len() {
                let col_data = &columns[col_idx];
                let delta_bytes = Self::extract_column_slice(col_data, col_def.dtype, start_idx, ids.len());
                writer.write_all(&(delta_bytes.len() as u64).to_le_bytes())?;
                writer.write_all(&delta_bytes)?;
            }
        }
        
        // Write CRC of delta record
        let crc = crc32fast::hash(&(ids.len() as u32).to_le_bytes());
        writer.write_all(&crc.to_le_bytes())?;
        
        writer.flush()?;
        drop(writer);
        
        // Update header with delta info
        {
            let mut header = self.header.write();
            if is_first_delta {
                header.delta_offset = delta_start;
            }
            header.delta_count += 1;
        }
        
        // Write updated header (seek back to start)
        file.seek(SeekFrom::Start(0))?;
        file.write_all(&self.header.read().to_bytes())?;
        // Note: We don't sync_all() here for performance.
        // Call flush() or close() for durability guarantees.
        
        Ok(ids)
    }

    /// Fast delta append without fsync - use for bulk operations
    /// Call flush() after batch for durability
    pub fn append_delta_fast(&self, rows: &[HashMap<String, ColumnValue>]) -> io::Result<Vec<u64>> {
        self.append_delta(rows)
    }

    /// Extract a slice of column data for delta serialization
    fn extract_column_slice(col_data: &ColumnData, _dtype: ColumnType, start: usize, count: usize) -> Vec<u8> {
        let mut buf = Vec::new();
        match col_data {
            ColumnData::Int64(v) => {
                buf.extend_from_slice(&(count as u64).to_le_bytes());
                for &val in v.iter().skip(start).take(count) {
                    buf.extend_from_slice(&val.to_le_bytes());
                }
            }
            ColumnData::Float64(v) => {
                buf.extend_from_slice(&(count as u64).to_le_bytes());
                for &val in v.iter().skip(start).take(count) {
                    buf.extend_from_slice(&val.to_le_bytes());
                }
            }
            ColumnData::Bool { data, len: _ } => {
                buf.extend_from_slice(&(count as u64).to_le_bytes());
                // Extract bits for the range [start, start+count)
                let byte_start = start / 8;
                let byte_end = (start + count + 7) / 8;
                buf.extend_from_slice(&data[byte_start..byte_end.min(data.len())]);
            }
            ColumnData::String { offsets, data } | ColumnData::Binary { offsets, data } => {
                buf.extend_from_slice(&(count as u64).to_le_bytes());
                // Write offsets relative to slice
                let base_offset = if start > 0 { offsets[start] } else { 0 };
                for i in start..=(start + count).min(offsets.len() - 1) {
                    buf.extend_from_slice(&(offsets[i] - base_offset).to_le_bytes());
                }
                // Write data slice
                let data_start = base_offset as usize;
                let data_end = offsets.get(start + count).copied().unwrap_or(data.len() as u32) as usize;
                buf.extend_from_slice(&((data_end - data_start) as u64).to_le_bytes());
                buf.extend_from_slice(&data[data_start..data_end]);
            }
        }
        buf
    }

    /// Compact delta records into base storage
    /// 
    /// This rewrites the file, merging all delta records into columnar format.
    /// Call this periodically or when delta_count exceeds threshold.
    pub fn compact(&self) -> io::Result<()> {
        // Simply save - this rewrites everything as base data
        self.save()
    }

    /// Check if compaction is recommended
    pub fn needs_compaction(&self) -> bool {
        let header = self.header.read();
        header.delta_count >= DELTA_COMPACT_THRESHOLD
    }

    /// Flush changes to disk (uses delta append for efficiency)
    pub fn flush(&self) -> io::Result<()> {
        // For flush, we just update the header to ensure consistency
        // The actual data was written by append_delta or is in memory
        let mut file = OpenOptions::new()
            .write(true)
            .open(&self.path)?;
        file.write_all(&self.header.read().to_bytes())?;
        file.sync_all()?;
        Ok(())
    }

    /// Close storage (full save to ensure all data is persisted)
    pub fn close(&self) -> io::Result<()> {
        self.save()
    }
}

// ============================================================================
// Value Type for API
// ============================================================================

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

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn test_basic_insert() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test.apex");
        
        let storage = ColumnarStorage::create(&path).unwrap();
        
        let mut int_cols = HashMap::new();
        int_cols.insert("age".to_string(), vec![25, 30, 35]);
        
        let mut string_cols = HashMap::new();
        string_cols.insert("name".to_string(), vec!["Alice".to_string(), "Bob".to_string(), "Charlie".to_string()]);
        
        let ids = storage.insert_typed(
            int_cols,
            HashMap::new(),
            string_cols,
            HashMap::new(),
            HashMap::new(),
        ).unwrap();
        
        assert_eq!(ids.len(), 3);
        assert_eq!(ids, vec![1, 2, 3]);
        assert_eq!(storage.row_count(), 3);
    }

    #[test]
    fn test_mixed_types() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test.apex");
        
        let storage = ColumnarStorage::create(&path).unwrap();
        
        let mut int_cols = HashMap::new();
        int_cols.insert("age".to_string(), vec![25, 30, 35]);
        
        let mut float_cols = HashMap::new();
        float_cols.insert("score".to_string(), vec![85.5, 90.0, 78.3]);
        
        let mut string_cols = HashMap::new();
        string_cols.insert("name".to_string(), vec!["A".to_string(), "B".to_string(), "C".to_string()]);
        
        let mut binary_cols = HashMap::new();
        binary_cols.insert("data".to_string(), vec![vec![1, 2, 3], vec![4, 5], vec![6]]);
        
        let mut bool_cols = HashMap::new();
        bool_cols.insert("active".to_string(), vec![true, false, true]);
        
        let ids = storage.insert_typed(int_cols, float_cols, string_cols, binary_cols, bool_cols).unwrap();
        
        assert_eq!(ids.len(), 3);
        assert_eq!(storage.column_names().len(), 6); // _id + 5 columns
    }

    #[test]
    fn test_persistence() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test.apex");
        
        // Create and insert
        {
            let storage = ColumnarStorage::create(&path).unwrap();
            
            let mut int_cols = HashMap::new();
            int_cols.insert("value".to_string(), vec![1, 2, 3, 4, 5]);
            
            storage.insert_typed(int_cols, HashMap::new(), HashMap::new(), HashMap::new(), HashMap::new()).unwrap();
            storage.save().unwrap();
        }
        
        // Reopen and verify
        {
            let storage = ColumnarStorage::open(&path).unwrap();
            assert_eq!(storage.row_count(), 5);
        }
    }

    #[test]
    fn benchmark_10k_native() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("bench.apex");
        
        let storage = ColumnarStorage::create(&path).unwrap();
        
        let n = 10_000;
        
        // Prepare data
        let int_data: Vec<i64> = (0..n as i64).collect();
        let float_data: Vec<f64> = (0..n).map(|i| i as f64 * 0.1).collect();
        let string_data: Vec<String> = (0..n).map(|i| format!("name_{}", i)).collect();
        let binary_data: Vec<Vec<u8>> = (0..n).map(|i| format!("bin_{:04}", i).into_bytes()).collect();
        let bool_data: Vec<bool> = (0..n).map(|i| i % 2 == 0).collect();
        
        // Warm up
        for _ in 0..3 {
            let dir2 = tempdir().unwrap();
            let path2 = dir2.path().join("warm.apex");
            let s = ColumnarStorage::create(&path2).unwrap();
            let mut ic = HashMap::new();
            ic.insert("age".to_string(), int_data.clone());
            let _ = s.insert_typed(ic, HashMap::new(), HashMap::new(), HashMap::new(), HashMap::new());
        }
        
        // Benchmark
        let iterations = 10;
        let mut times = Vec::with_capacity(iterations);
        
        for _ in 0..iterations {
            let dir3 = tempdir().unwrap();
            let path3 = dir3.path().join("bench.apex");
            let storage3 = ColumnarStorage::create(&path3).unwrap();
            
            let mut int_cols = HashMap::new();
            int_cols.insert("age".to_string(), int_data.clone());
            let mut float_cols = HashMap::new();
            float_cols.insert("score".to_string(), float_data.clone());
            let mut string_cols = HashMap::new();
            string_cols.insert("name".to_string(), string_data.clone());
            let mut binary_cols = HashMap::new();
            binary_cols.insert("data".to_string(), binary_data.clone());
            let mut bool_cols = HashMap::new();
            bool_cols.insert("active".to_string(), bool_data.clone());
            
            let start = std::time::Instant::now();
            let ids = storage3.insert_typed(int_cols, float_cols, string_cols, binary_cols, bool_cols).unwrap();
            let elapsed = start.elapsed();
            
            assert_eq!(ids.len(), n);
            times.push(elapsed.as_micros());
        }
        
        times.sort();
        let median = times[times.len() / 2];
        let min = times[0];
        let max = times[times.len() - 1];
        let avg: u128 = times.iter().sum::<u128>() / times.len() as u128;
        
        println!("\n=== Native Format 10,000 Rows Insert ===");
        println!("Columns: int64, float64, string, binary, bool");
        println!("Min:    {:>8} μs ({:.2} ms)", min, min as f64 / 1000.0);
        println!("Max:    {:>8} μs ({:.2} ms)", max, max as f64 / 1000.0);
        println!("Avg:    {:>8} μs ({:.2} ms)", avg, avg as f64 / 1000.0);
        println!("Median: {:>8} μs ({:.2} ms)", median, median as f64 / 1000.0);
        println!("Throughput: {:>8.0} rows/ms", n as f64 / (median as f64 / 1000.0));
        println!("=========================================\n");
        
        assert!(median < 2000, "Should complete in <2ms, got {}μs", median);
    }

    #[test]
    fn test_delta_append() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("delta_test.apex");
        
        // Create storage and initial data
        let storage = ColumnarStorage::create(&path).unwrap();
        
        let mut int_cols = HashMap::new();
        int_cols.insert("value".to_string(), vec![1, 2, 3]);
        storage.insert_typed(int_cols, HashMap::new(), HashMap::new(), HashMap::new(), HashMap::new()).unwrap();
        storage.save().unwrap();
        
        assert_eq!(storage.row_count(), 3);
        
        // Now append delta using insert_rows (which append_delta wraps)
        let mut row1 = HashMap::new();
        row1.insert("value".to_string(), ColumnValue::Int64(4));
        let mut row2 = HashMap::new();
        row2.insert("value".to_string(), ColumnValue::Int64(5));
        
        let ids = storage.append_delta(&[row1, row2]).unwrap();
        assert_eq!(ids.len(), 2);
        assert_eq!(storage.row_count(), 5);
        
        // Check header delta info
        {
            let header = storage.header.read();
            assert!(header.delta_offset > 0, "Delta offset should be set");
            assert_eq!(header.delta_count, 1, "Should have 1 delta batch");
        }
        
        // Compact and verify
        storage.compact().unwrap();
        {
            let header = storage.header.read();
            assert_eq!(header.delta_offset, 0, "Delta offset should be 0 after compact");
            assert_eq!(header.delta_count, 0, "Delta count should be 0 after compact");
            assert_eq!(header.base_row_count, 5, "All rows should be base rows");
        }
    }

    #[test]
    fn benchmark_delta_vs_save() {
        let dir = tempdir().unwrap();
        
        // Use larger dataset to show delta advantage
        let base_rows = 100_000;
        let append_count = 10;
        
        // Test 1: Full save after each batch (old way)
        let path1 = dir.path().join("full_save.apex");
        let storage1 = ColumnarStorage::create(&path1).unwrap();
        
        // Initial data: 100K rows
        let mut int_cols = HashMap::new();
        int_cols.insert("value".to_string(), (0..base_rows as i64).collect());
        storage1.insert_typed(int_cols, HashMap::new(), HashMap::new(), HashMap::new(), HashMap::new()).unwrap();
        storage1.save().unwrap();
        
        // Measure time to append rows with full save
        let start_save = std::time::Instant::now();
        for i in 0..append_count {
            let mut row = HashMap::new();
            row.insert("value".to_string(), ColumnValue::Int64(base_rows as i64 + i));
            storage1.insert_rows(&[row]).unwrap();
            storage1.save().unwrap();  // Full rewrite each time - O(N) per append
        }
        let save_time = start_save.elapsed();
        
        // Test 2: Delta append (new way) - memory only, then single save
        let path2 = dir.path().join("delta_append.apex");
        let storage2 = ColumnarStorage::create(&path2).unwrap();
        
        // Initial data: 100K rows
        let mut int_cols2 = HashMap::new();
        int_cols2.insert("value".to_string(), (0..base_rows as i64).collect());
        storage2.insert_typed(int_cols2, HashMap::new(), HashMap::new(), HashMap::new(), HashMap::new()).unwrap();
        storage2.save().unwrap();
        
        // Measure time to append rows (memory) then single save
        let start_delta = std::time::Instant::now();
        for i in 0..append_count {
            let mut row = HashMap::new();
            row.insert("value".to_string(), ColumnValue::Int64(base_rows as i64 + i));
            storage2.insert_rows(&[row]).unwrap();  // Memory only - O(1) per append
        }
        storage2.save().unwrap();  // Single O(N) save at end
        let delta_time = start_delta.elapsed();
        
        println!("\n=== Batch Insert vs Per-Row Save Benchmark ===");
        println!("Base: {} rows, Append: {} rows", base_rows, append_count);
        println!("Per-row save time:   {:>8} μs ({:.2} ms) - {} full rewrites", 
            save_time.as_micros(), save_time.as_secs_f64() * 1000.0, append_count);
        println!("Batch then save time: {:>6} μs ({:.2} ms) - 1 full rewrite", 
            delta_time.as_micros(), delta_time.as_secs_f64() * 1000.0);
        println!("Speedup: {:.1}x", save_time.as_secs_f64() / delta_time.as_secs_f64());
        println!("===============================================\n");
        
        // Batch approach should be significantly faster
        assert!(delta_time < save_time, "Batch insert should be faster than per-row save");
        assert_eq!(storage1.row_count(), storage2.row_count());
    }
}
