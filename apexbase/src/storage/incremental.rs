//! Incremental Write Architecture for ApexBase
//!
//! Solves write performance issues by using a WAL (Write-Ahead Log) pattern:
//! - Writes go to append-only WAL file (.apex.wal) - O(1) append, very fast
//! - Reads merge main file + WAL data transparently
//! - Background compaction merges WAL into main file periodically
//!
//! File Structure:
//! ```text
//! data.apex      - Main columnar file (V3 format)
//! data.apex.wal  - WAL file (append-only records)
//! ```
//!
//! WAL Record Format:
//! ```text
//! [record_type:u8][timestamp:i64][record_length:u32][record_data...]
//! ```

use std::collections::HashMap;
use std::fs::{File, OpenOptions};
use std::io::{self, BufReader, BufWriter, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, AtomicBool, Ordering};

use parking_lot::RwLock;

use super::on_demand::{ColumnData, ColumnType, ColumnValue, OnDemandStorage};

// ============================================================================
// Constants
// ============================================================================

const WAL_MAGIC: &[u8; 8] = b"APEXWAL\0";
const WAL_VERSION: u32 = 1;
const WAL_HEADER_SIZE: usize = 24; // magic(8) + version(4) + next_id(8) + flags(4)

// Record types
const RECORD_INSERT: u8 = 1;
const RECORD_DELETE: u8 = 2;
const RECORD_CHECKPOINT: u8 = 3;
const RECORD_BATCH_INSERT: u8 = 4;

// Compaction threshold (number of WAL records before auto-compact)
const DEFAULT_COMPACTION_THRESHOLD: usize = 10000;

// Auto-checkpoint threshold (checkpoint after this many inserts for write-visible)
const AUTO_CHECKPOINT_THRESHOLD: usize = 1000;

// ============================================================================
// WAL Record
// ============================================================================

#[derive(Debug, Clone)]
pub enum WalRecord {
    Insert {
        id: u64,
        data: HashMap<String, ColumnValue>,
    },
    /// Batch insert - more efficient for multiple rows (single WAL record, single I/O)
    BatchInsert {
        start_id: u64,
        rows: Vec<HashMap<String, ColumnValue>>,
    },
    Delete {
        id: u64,
    },
    Checkpoint {
        row_count: u64,
    },
}

impl WalRecord {
    /// Serialize record to bytes
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut buf = Vec::new();
        let timestamp = chrono::Utc::now().timestamp();
        
        match self {
            WalRecord::Insert { id, data } => {
                buf.push(RECORD_INSERT);
                buf.extend_from_slice(&timestamp.to_le_bytes());
                
                // Serialize data as: [id:u64][col_count:u32][col_name_len:u16][col_name][type:u8][value_bytes]
                let mut data_buf = Vec::new();
                data_buf.extend_from_slice(&id.to_le_bytes());
                data_buf.extend_from_slice(&(data.len() as u32).to_le_bytes());
                
                for (name, value) in data {
                    let name_bytes = name.as_bytes();
                    data_buf.extend_from_slice(&(name_bytes.len() as u16).to_le_bytes());
                    data_buf.extend_from_slice(name_bytes);
                    
                    match value {
                        ColumnValue::Null => {
                            data_buf.push(0);
                        }
                        ColumnValue::Bool(v) => {
                            data_buf.push(1);
                            data_buf.push(if *v { 1 } else { 0 });
                        }
                        ColumnValue::Int64(v) => {
                            data_buf.push(2);
                            data_buf.extend_from_slice(&v.to_le_bytes());
                        }
                        ColumnValue::Float64(v) => {
                            data_buf.push(3);
                            data_buf.extend_from_slice(&v.to_le_bytes());
                        }
                        ColumnValue::String(v) => {
                            data_buf.push(4);
                            let bytes = v.as_bytes();
                            data_buf.extend_from_slice(&(bytes.len() as u32).to_le_bytes());
                            data_buf.extend_from_slice(bytes);
                        }
                        ColumnValue::Binary(v) => {
                            data_buf.push(5);
                            data_buf.extend_from_slice(&(v.len() as u32).to_le_bytes());
                            data_buf.extend_from_slice(v);
                        }
                    }
                }
                
                buf.extend_from_slice(&(data_buf.len() as u32).to_le_bytes());
                buf.extend_from_slice(&data_buf);
            }
            WalRecord::BatchInsert { start_id, rows } => {
                buf.push(RECORD_BATCH_INSERT);
                buf.extend_from_slice(&timestamp.to_le_bytes());
                
                // Serialize batch: [start_id:u64][row_count:u32][rows...]
                let mut data_buf = Vec::with_capacity(rows.len() * 64); // Pre-allocate
                data_buf.extend_from_slice(&start_id.to_le_bytes());
                data_buf.extend_from_slice(&(rows.len() as u32).to_le_bytes());
                
                for row in rows {
                    data_buf.extend_from_slice(&(row.len() as u32).to_le_bytes());
                    for (name, value) in row {
                        let name_bytes = name.as_bytes();
                        data_buf.extend_from_slice(&(name_bytes.len() as u16).to_le_bytes());
                        data_buf.extend_from_slice(name_bytes);
                        
                        match value {
                            ColumnValue::Null => data_buf.push(0),
                            ColumnValue::Bool(v) => {
                                data_buf.push(1);
                                data_buf.push(if *v { 1 } else { 0 });
                            }
                            ColumnValue::Int64(v) => {
                                data_buf.push(2);
                                data_buf.extend_from_slice(&v.to_le_bytes());
                            }
                            ColumnValue::Float64(v) => {
                                data_buf.push(3);
                                data_buf.extend_from_slice(&v.to_le_bytes());
                            }
                            ColumnValue::String(v) => {
                                data_buf.push(4);
                                let bytes = v.as_bytes();
                                data_buf.extend_from_slice(&(bytes.len() as u32).to_le_bytes());
                                data_buf.extend_from_slice(bytes);
                            }
                            ColumnValue::Binary(v) => {
                                data_buf.push(5);
                                data_buf.extend_from_slice(&(v.len() as u32).to_le_bytes());
                                data_buf.extend_from_slice(v);
                            }
                        }
                    }
                }
                
                buf.extend_from_slice(&(data_buf.len() as u32).to_le_bytes());
                buf.extend_from_slice(&data_buf);
            }
            WalRecord::Delete { id } => {
                buf.push(RECORD_DELETE);
                buf.extend_from_slice(&timestamp.to_le_bytes());
                buf.extend_from_slice(&8u32.to_le_bytes()); // record length
                buf.extend_from_slice(&id.to_le_bytes());
            }
            WalRecord::Checkpoint { row_count } => {
                buf.push(RECORD_CHECKPOINT);
                buf.extend_from_slice(&timestamp.to_le_bytes());
                buf.extend_from_slice(&8u32.to_le_bytes());
                buf.extend_from_slice(&row_count.to_le_bytes());
            }
        }
        
        buf
    }
    
    /// Deserialize record from bytes
    pub fn from_bytes(bytes: &[u8]) -> io::Result<(Self, usize)> {
        if bytes.is_empty() {
            return Err(io::Error::new(io::ErrorKind::InvalidData, "Empty record"));
        }
        
        let record_type = bytes[0];
        let _timestamp = i64::from_le_bytes(bytes[1..9].try_into().unwrap());
        let record_len = u32::from_le_bytes(bytes[9..13].try_into().unwrap()) as usize;
        let data = &bytes[13..13 + record_len];
        
        let record = match record_type {
            RECORD_INSERT => {
                let id = u64::from_le_bytes(data[0..8].try_into().unwrap());
                let col_count = u32::from_le_bytes(data[8..12].try_into().unwrap()) as usize;
                
                let mut pos = 12;
                let mut row_data = HashMap::new();
                
                for _ in 0..col_count {
                    let name_len = u16::from_le_bytes(data[pos..pos+2].try_into().unwrap()) as usize;
                    pos += 2;
                    let name = std::str::from_utf8(&data[pos..pos+name_len])
                        .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?
                        .to_string();
                    pos += name_len;
                    
                    let value_type = data[pos];
                    pos += 1;
                    
                    let value = match value_type {
                        0 => ColumnValue::Null,
                        1 => {
                            let v = data[pos] != 0;
                            pos += 1;
                            ColumnValue::Bool(v)
                        }
                        2 => {
                            let v = i64::from_le_bytes(data[pos..pos+8].try_into().unwrap());
                            pos += 8;
                            ColumnValue::Int64(v)
                        }
                        3 => {
                            let v = f64::from_le_bytes(data[pos..pos+8].try_into().unwrap());
                            pos += 8;
                            ColumnValue::Float64(v)
                        }
                        4 => {
                            let len = u32::from_le_bytes(data[pos..pos+4].try_into().unwrap()) as usize;
                            pos += 4;
                            let s = std::str::from_utf8(&data[pos..pos+len])
                                .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?
                                .to_string();
                            pos += len;
                            ColumnValue::String(s)
                        }
                        5 => {
                            let len = u32::from_le_bytes(data[pos..pos+4].try_into().unwrap()) as usize;
                            pos += 4;
                            let b = data[pos..pos+len].to_vec();
                            pos += len;
                            ColumnValue::Binary(b)
                        }
                        _ => return Err(io::Error::new(io::ErrorKind::InvalidData, "Invalid value type")),
                    };
                    
                    row_data.insert(name, value);
                }
                
                WalRecord::Insert { id, data: row_data }
            }
            RECORD_BATCH_INSERT => {
                let start_id = u64::from_le_bytes(data[0..8].try_into().unwrap());
                let row_count = u32::from_le_bytes(data[8..12].try_into().unwrap()) as usize;
                
                let mut pos = 12;
                let mut rows = Vec::with_capacity(row_count);
                
                for _ in 0..row_count {
                    let col_count = u32::from_le_bytes(data[pos..pos+4].try_into().unwrap()) as usize;
                    pos += 4;
                    let mut row_data = HashMap::new();
                    
                    for _ in 0..col_count {
                        let name_len = u16::from_le_bytes(data[pos..pos+2].try_into().unwrap()) as usize;
                        pos += 2;
                        let name = std::str::from_utf8(&data[pos..pos+name_len])
                            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?
                            .to_string();
                        pos += name_len;
                        
                        let value_type = data[pos];
                        pos += 1;
                        
                        let value = match value_type {
                            0 => ColumnValue::Null,
                            1 => {
                                let v = data[pos] != 0;
                                pos += 1;
                                ColumnValue::Bool(v)
                            }
                            2 => {
                                let v = i64::from_le_bytes(data[pos..pos+8].try_into().unwrap());
                                pos += 8;
                                ColumnValue::Int64(v)
                            }
                            3 => {
                                let v = f64::from_le_bytes(data[pos..pos+8].try_into().unwrap());
                                pos += 8;
                                ColumnValue::Float64(v)
                            }
                            4 => {
                                let len = u32::from_le_bytes(data[pos..pos+4].try_into().unwrap()) as usize;
                                pos += 4;
                                let s = std::str::from_utf8(&data[pos..pos+len])
                                    .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?
                                    .to_string();
                                pos += len;
                                ColumnValue::String(s)
                            }
                            5 => {
                                let len = u32::from_le_bytes(data[pos..pos+4].try_into().unwrap()) as usize;
                                pos += 4;
                                let b = data[pos..pos+len].to_vec();
                                pos += len;
                                ColumnValue::Binary(b)
                            }
                            _ => return Err(io::Error::new(io::ErrorKind::InvalidData, "Invalid value type")),
                        };
                        
                        row_data.insert(name, value);
                    }
                    rows.push(row_data);
                }
                
                WalRecord::BatchInsert { start_id, rows }
            }
            RECORD_DELETE => {
                let id = u64::from_le_bytes(data[0..8].try_into().unwrap());
                WalRecord::Delete { id }
            }
            RECORD_CHECKPOINT => {
                let row_count = u64::from_le_bytes(data[0..8].try_into().unwrap());
                WalRecord::Checkpoint { row_count }
            }
            _ => return Err(io::Error::new(io::ErrorKind::InvalidData, "Invalid record type")),
        };
        
        Ok((record, 13 + record_len))
    }
}

// ============================================================================
// WAL Writer
// ============================================================================

pub struct WalWriter {
    path: PathBuf,
    file: BufWriter<File>,
    record_count: usize,
}

impl WalWriter {
    /// Create new WAL file
    pub fn create(path: &Path, start_id: u64) -> io::Result<Self> {
        let file = OpenOptions::new()
            .write(true)
            .create(true)
            .truncate(true)
            .open(path)?;
        
        let mut writer = BufWriter::with_capacity(64 * 1024, file);
        
        // Write header
        writer.write_all(WAL_MAGIC)?;
        writer.write_all(&WAL_VERSION.to_le_bytes())?;
        writer.write_all(&start_id.to_le_bytes())?;
        writer.write_all(&0u32.to_le_bytes())?; // flags
        writer.flush()?;
        
        Ok(Self {
            path: path.to_path_buf(),
            file: writer,
            record_count: 0,
        })
    }
    
    /// Open existing WAL file for append
    pub fn open(path: &Path) -> io::Result<Self> {
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .open(path)?;
        
        // Verify header
        let mut header = [0u8; WAL_HEADER_SIZE];
        {
            let mut reader = BufReader::new(&file);
            reader.read_exact(&mut header)?;
        }
        
        if &header[0..8] != WAL_MAGIC {
            return Err(io::Error::new(io::ErrorKind::InvalidData, "Invalid WAL magic"));
        }
        
        // Seek to end for appending
        let mut file = file;
        let end_pos = file.seek(SeekFrom::End(0))?;
        
        // Count existing records
        let record_count = if end_pos > WAL_HEADER_SIZE as u64 {
            // Rough estimate based on average record size
            ((end_pos - WAL_HEADER_SIZE as u64) / 100) as usize
        } else {
            0
        };
        
        Ok(Self {
            path: path.to_path_buf(),
            file: BufWriter::with_capacity(64 * 1024, file),
            record_count,
        })
    }
    
    /// Append a record to WAL
    pub fn append(&mut self, record: &WalRecord) -> io::Result<()> {
        let bytes = record.to_bytes();
        self.file.write_all(&bytes)?;
        self.record_count += 1;
        Ok(())
    }
    
    /// Append a single row directly without creating WalRecord (optimized for single inserts)
    /// This avoids the overhead of cloning data into a WalRecord struct
    pub fn append_row(&mut self, id: u64, data: &HashMap<String, ColumnValue>) -> io::Result<()> {
        let timestamp = chrono::Utc::now().timestamp();
        
        // Serialize directly to buffer
        let mut buf = Vec::with_capacity(128);
        buf.push(RECORD_INSERT);
        buf.extend_from_slice(&timestamp.to_le_bytes());
        
        // Data: [id:u64][col_count:u32][col_name_len:u16][col_name][type:u8][value_bytes]
        let mut data_buf = Vec::with_capacity(64);
        data_buf.extend_from_slice(&id.to_le_bytes());
        data_buf.extend_from_slice(&(data.len() as u32).to_le_bytes());
        
        for (name, value) in data {
            let name_bytes = name.as_bytes();
            data_buf.extend_from_slice(&(name_bytes.len() as u16).to_le_bytes());
            data_buf.extend_from_slice(name_bytes);
            
            match value {
                ColumnValue::Null => data_buf.push(0),
                ColumnValue::Bool(v) => {
                    data_buf.push(1);
                    data_buf.push(if *v { 1 } else { 0 });
                }
                ColumnValue::Int64(v) => {
                    data_buf.push(2);
                    data_buf.extend_from_slice(&v.to_le_bytes());
                }
                ColumnValue::Float64(v) => {
                    data_buf.push(3);
                    data_buf.extend_from_slice(&v.to_le_bytes());
                }
                ColumnValue::String(v) => {
                    data_buf.push(4);
                    let bytes = v.as_bytes();
                    data_buf.extend_from_slice(&(bytes.len() as u32).to_le_bytes());
                    data_buf.extend_from_slice(bytes);
                }
                ColumnValue::Binary(v) => {
                    data_buf.push(5);
                    data_buf.extend_from_slice(&(v.len() as u32).to_le_bytes());
                    data_buf.extend_from_slice(v);
                }
            }
        }
        
        buf.extend_from_slice(&(data_buf.len() as u32).to_le_bytes());
        buf.extend_from_slice(&data_buf);
        
        self.file.write_all(&buf)?;
        self.record_count += 1;
        Ok(())
    }
    
    /// Flush WAL to disk (buffered write)
    pub fn flush(&mut self) -> io::Result<()> {
        self.file.flush()
    }
    
    /// Sync WAL to disk (fsync - ensures durability)
    pub fn sync(&mut self) -> io::Result<()> {
        self.file.flush()?;
        self.file.get_ref().sync_all()
    }
    
    /// Get record count
    pub fn record_count(&self) -> usize {
        self.record_count
    }
    
    /// Check if compaction is needed
    pub fn needs_compaction(&self, threshold: usize) -> bool {
        self.record_count >= threshold
    }
}

// ============================================================================
// WAL Reader
// ============================================================================

pub struct WalReader {
    data: Vec<u8>,
    pos: usize,
}

impl WalReader {
    /// Open WAL file for reading
    pub fn open(path: &Path) -> io::Result<Self> {
        if !path.exists() {
            return Ok(Self {
                data: Vec::new(),
                pos: WAL_HEADER_SIZE,
            });
        }
        
        let mut file = File::open(path)?;
        let mut data = Vec::new();
        file.read_to_end(&mut data)?;
        
        // Verify header
        if data.len() >= WAL_HEADER_SIZE && &data[0..8] == WAL_MAGIC {
            Ok(Self {
                data,
                pos: WAL_HEADER_SIZE,
            })
        } else if data.is_empty() {
            Ok(Self {
                data: Vec::new(),
                pos: WAL_HEADER_SIZE,
            })
        } else {
            Err(io::Error::new(io::ErrorKind::InvalidData, "Invalid WAL header"))
        }
    }
    
    /// Read all records from WAL
    pub fn read_all(&mut self) -> io::Result<Vec<WalRecord>> {
        let mut records = Vec::new();
        
        while self.pos < self.data.len() {
            match WalRecord::from_bytes(&self.data[self.pos..]) {
                Ok((record, len)) => {
                    records.push(record);
                    self.pos += len;
                }
                Err(_) => break, // End of valid records
            }
        }
        
        Ok(records)
    }
    
    /// Get starting ID from header
    pub fn start_id(&self) -> u64 {
        if self.data.len() >= WAL_HEADER_SIZE {
            u64::from_le_bytes(self.data[12..20].try_into().unwrap())
        } else {
            0
        }
    }
}

// ============================================================================
// Incremental Storage Engine
// ============================================================================

/// High-performance incremental storage with WAL
/// 
/// Write path:
/// 1. Append to in-memory buffer
/// 2. Append to WAL file (fsync)
/// 3. Return immediately (fast!)
/// 
/// Read path:
/// 1. Read from main file
/// 2. Merge with WAL buffer
/// 3. Return combined view
/// 
/// Compaction:
/// 1. Triggered when WAL reaches threshold
/// 2. Merge WAL into main file
/// 3. Truncate WAL
pub struct IncrementalStorage {
    path: PathBuf,
    main_storage: RwLock<OnDemandStorage>,
    wal_writer: RwLock<Option<WalWriter>>,
    wal_buffer: RwLock<Vec<WalRecord>>,
    /// Columnar memory buffer for fast reads (DuckDB-style)
    /// Key: column name, Value: column data
    memory_buffer: RwLock<HashMap<String, ColumnData>>,
    /// Row IDs in memory buffer
    memory_ids: RwLock<Vec<u64>>,
    next_id: AtomicU64,
    deleted_ids: RwLock<std::collections::HashSet<u64>>,
    compaction_threshold: usize,
    /// Auto-checkpoint threshold (for write-visible)
    auto_checkpoint_threshold: usize,
    /// Counter for inserts since last checkpoint
    inserts_since_checkpoint: AtomicU64,
    needs_compact: AtomicBool,
}

impl IncrementalStorage {
    /// Create new incremental storage
    pub fn create(path: &Path) -> io::Result<Self> {
        let main_storage = OnDemandStorage::create(path)?;
        let wal_path = Self::wal_path(path);
        let wal_writer = WalWriter::create(&wal_path, 0)?;
        
        Ok(Self {
            path: path.to_path_buf(),
            main_storage: RwLock::new(main_storage),
            wal_writer: RwLock::new(Some(wal_writer)),
            wal_buffer: RwLock::new(Vec::new()),
            memory_buffer: RwLock::new(HashMap::new()),
            memory_ids: RwLock::new(Vec::new()),
            next_id: AtomicU64::new(0),
            deleted_ids: RwLock::new(std::collections::HashSet::new()),
            compaction_threshold: DEFAULT_COMPACTION_THRESHOLD,
            auto_checkpoint_threshold: AUTO_CHECKPOINT_THRESHOLD,
            inserts_since_checkpoint: AtomicU64::new(0),
            needs_compact: AtomicBool::new(false),
        })
    }
    
    /// Open existing incremental storage
    pub fn open(path: &Path) -> io::Result<Self> {
        let main_storage = OnDemandStorage::open(path)?;
        let next_id = main_storage.row_count();
        
        let wal_path = Self::wal_path(path);
        let (wal_writer, wal_buffer, max_wal_id) = if wal_path.exists() {
            // Read existing WAL
            let mut reader = WalReader::open(&wal_path)?;
            let records = reader.read_all()?;
            let max_id = records.iter().filter_map(|r| {
                match r {
                    WalRecord::Insert { id, .. } => Some(*id),
                    _ => None,
                }
            }).max().unwrap_or(next_id);
            
            // Open for append
            let writer = WalWriter::open(&wal_path)?;
            (Some(writer), records, max_id)
        } else {
            let writer = WalWriter::create(&wal_path, next_id)?;
            (Some(writer), Vec::new(), next_id)
        };
        
        // Extract deleted IDs from WAL
        let deleted_ids: std::collections::HashSet<u64> = wal_buffer.iter()
            .filter_map(|r| match r {
                WalRecord::Delete { id } => Some(*id),
                _ => None,
            })
            .collect();
        
        let final_next_id = max_wal_id.max(next_id) + 1;
        
        // Build memory buffer from WAL records for fast reads
        let mut memory_buffer = HashMap::new();
        let mut memory_ids = Vec::new();
        for record in &wal_buffer {
            if let WalRecord::Insert { id, data } = record {
                if !deleted_ids.contains(id) {
                    memory_ids.push(*id);
                    for (name, value) in data {
                        let col = memory_buffer.entry(name.clone()).or_insert_with(|| {
                            match value {
                                ColumnValue::Int64(_) => ColumnData::new(ColumnType::Int64),
                                ColumnValue::Float64(_) => ColumnData::new(ColumnType::Float64),
                                ColumnValue::String(_) => ColumnData::new(ColumnType::String),
                                ColumnValue::Binary(_) => ColumnData::new(ColumnType::Binary),
                                ColumnValue::Bool(_) => ColumnData::new(ColumnType::Bool),
                                ColumnValue::Null => ColumnData::new(ColumnType::String),
                            }
                        });
                        match (col, value) {
                            (ColumnData::Int64(v), ColumnValue::Int64(val)) => v.push(*val),
                            (ColumnData::Float64(v), ColumnValue::Float64(val)) => v.push(*val),
                            (col @ ColumnData::String { .. }, ColumnValue::String(val)) => col.push_string(val),
                            (col @ ColumnData::Binary { .. }, ColumnValue::Binary(val)) => col.push_bytes(val),
                            (col @ ColumnData::Bool { .. }, ColumnValue::Bool(val)) => col.push_bool(*val),
                            _ => {}
                        }
                    }
                }
            }
        }
        
        Ok(Self {
            path: path.to_path_buf(),
            main_storage: RwLock::new(main_storage),
            wal_writer: RwLock::new(wal_writer),
            wal_buffer: RwLock::new(wal_buffer),
            memory_buffer: RwLock::new(memory_buffer),
            memory_ids: RwLock::new(memory_ids),
            next_id: AtomicU64::new(final_next_id),
            deleted_ids: RwLock::new(deleted_ids),
            compaction_threshold: DEFAULT_COMPACTION_THRESHOLD,
            auto_checkpoint_threshold: AUTO_CHECKPOINT_THRESHOLD,
            inserts_since_checkpoint: AtomicU64::new(0),
            needs_compact: AtomicBool::new(false),
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
    
    fn wal_path(main_path: &Path) -> PathBuf {
        let mut wal_path = main_path.to_path_buf();
        let ext = wal_path.extension()
            .map(|e| format!("{}.wal", e.to_string_lossy()))
            .unwrap_or_else(|| "wal".to_string());
        wal_path.set_extension(ext);
        wal_path
    }
    
    // ========================================================================
    // Fast Write APIs
    // ========================================================================
    
    /// Insert rows - FAST append-only write to WAL + memory buffer
    /// 
    /// DuckDB-style write-visible: data is immediately visible in queries
    /// through the columnar memory buffer.
    pub fn insert_rows(&self, rows: &[HashMap<String, ColumnValue>]) -> io::Result<Vec<u64>> {
        if rows.is_empty() {
            return Ok(Vec::new());
        }
        
        let count = rows.len() as u64;
        let start_id = self.next_id.fetch_add(count, Ordering::SeqCst);
        let ids: Vec<u64> = (start_id..start_id + count).collect();
        
        // Append to WAL buffer, file, and memory buffer
        let records: Vec<WalRecord> = rows.iter().zip(ids.iter())
            .map(|(row, &id)| WalRecord::Insert { id, data: row.clone() })
            .collect();
        
        {
            let mut wal_buffer = self.wal_buffer.write();
            let mut wal_writer = self.wal_writer.write();
            let mut memory_buffer = self.memory_buffer.write();
            let mut memory_ids = self.memory_ids.write();
            
            for (record, &id) in records.iter().zip(ids.iter()) {
                wal_buffer.push(record.clone());
                if let Some(writer) = wal_writer.as_mut() {
                    writer.append(record)?;
                }
                
                // Also update columnar memory buffer for fast reads
                if let WalRecord::Insert { data, .. } = record {
                    memory_ids.push(id);
                    for (name, value) in data {
                        let col = memory_buffer.entry(name.clone()).or_insert_with(|| {
                            match value {
                                ColumnValue::Int64(_) => ColumnData::new(ColumnType::Int64),
                                ColumnValue::Float64(_) => ColumnData::new(ColumnType::Float64),
                                ColumnValue::String(_) => ColumnData::new(ColumnType::String),
                                ColumnValue::Binary(_) => ColumnData::new(ColumnType::Binary),
                                ColumnValue::Bool(_) => ColumnData::new(ColumnType::Bool),
                                ColumnValue::Null => ColumnData::new(ColumnType::String),
                            }
                        });
                        match (col, value) {
                            (ColumnData::Int64(v), ColumnValue::Int64(val)) => v.push(*val),
                            (ColumnData::Float64(v), ColumnValue::Float64(val)) => v.push(*val),
                            (col @ ColumnData::String { .. }, ColumnValue::String(val)) => col.push_string(val),
                            (col @ ColumnData::Binary { .. }, ColumnValue::Binary(val)) => col.push_bytes(val),
                            (col @ ColumnData::Bool { .. }, ColumnValue::Bool(val)) => col.push_bool(*val),
                            _ => {}
                        }
                    }
                }
            }
            
            // Flush WAL
            if let Some(writer) = wal_writer.as_mut() {
                writer.flush()?;
                
                // Check if compaction needed
                if writer.needs_compaction(self.compaction_threshold) {
                    self.needs_compact.store(true, Ordering::SeqCst);
                }
            }
        }
        
        // Track inserts for auto-checkpoint
        let inserts = self.inserts_since_checkpoint.fetch_add(count, Ordering::SeqCst) + count;
        
        // Auto-checkpoint when threshold reached (DuckDB-style)
        if inserts >= self.auto_checkpoint_threshold as u64 {
            self.checkpoint()?;
        }
        
        Ok(ids)
    }
    
    /// Checkpoint: merge WAL into main file and clear buffers
    /// 
    /// After checkpoint, queries read directly from main file (fast!)
    pub fn checkpoint(&self) -> io::Result<()> {
        self.compact()?;
        self.inserts_since_checkpoint.store(0, Ordering::SeqCst);
        Ok(())
    }
    
    /// Insert typed columns - FAST path
    pub fn insert_typed(
        &self,
        int_columns: HashMap<String, Vec<i64>>,
        float_columns: HashMap<String, Vec<f64>>,
        string_columns: HashMap<String, Vec<String>>,
        binary_columns: HashMap<String, Vec<Vec<u8>>>,
        bool_columns: HashMap<String, Vec<bool>>,
    ) -> io::Result<Vec<u64>> {
        // Convert to rows format
        let row_count = int_columns.values().map(|v| v.len()).max().unwrap_or(0)
            .max(float_columns.values().map(|v| v.len()).max().unwrap_or(0))
            .max(string_columns.values().map(|v| v.len()).max().unwrap_or(0))
            .max(binary_columns.values().map(|v| v.len()).max().unwrap_or(0))
            .max(bool_columns.values().map(|v| v.len()).max().unwrap_or(0));
        
        if row_count == 0 {
            return Ok(Vec::new());
        }
        
        let mut rows = Vec::with_capacity(row_count);
        for i in 0..row_count {
            let mut row = HashMap::new();
            for (name, vals) in &int_columns {
                if i < vals.len() {
                    row.insert(name.clone(), ColumnValue::Int64(vals[i]));
                }
            }
            for (name, vals) in &float_columns {
                if i < vals.len() {
                    row.insert(name.clone(), ColumnValue::Float64(vals[i]));
                }
            }
            for (name, vals) in &string_columns {
                if i < vals.len() {
                    row.insert(name.clone(), ColumnValue::String(vals[i].clone()));
                }
            }
            for (name, vals) in &binary_columns {
                if i < vals.len() {
                    row.insert(name.clone(), ColumnValue::Binary(vals[i].clone()));
                }
            }
            for (name, vals) in &bool_columns {
                if i < vals.len() {
                    row.insert(name.clone(), ColumnValue::Bool(vals[i]));
                }
            }
            rows.push(row);
        }
        
        self.insert_rows(&rows)
    }
    
    /// Delete row by ID - FAST append to WAL
    pub fn delete(&self, id: u64) -> io::Result<bool> {
        let record = WalRecord::Delete { id };
        
        {
            let mut wal_buffer = self.wal_buffer.write();
            let mut wal_writer = self.wal_writer.write();
            let mut deleted_ids = self.deleted_ids.write();
            
            wal_buffer.push(record.clone());
            deleted_ids.insert(id);
            
            if let Some(writer) = wal_writer.as_mut() {
                writer.append(&record)?;
                writer.flush()?;
            }
        }
        
        Ok(true)
    }
    
    // ========================================================================
    // Read APIs (merge main + memory buffer) - FAST!
    // ========================================================================
    
    /// Read columns - merges main file with columnar memory buffer
    /// 
    /// DuckDB-style: Uses columnar memory buffer instead of scanning WAL records.
    /// This is O(1) column access instead of O(n) row scanning.
    pub fn read_columns(
        &self,
        column_names: Option<&[&str]>,
        start_row: usize,
        row_count: Option<usize>,
    ) -> io::Result<HashMap<String, ColumnData>> {
        // Read from main storage
        let main_storage = self.main_storage.read();
        let mut result = main_storage.read_columns(column_names, start_row, row_count)?;
        
        // Merge with columnar memory buffer (FAST - no row scanning!)
        let memory_buffer = self.memory_buffer.read();
        
        if memory_buffer.is_empty() {
            return Ok(result);
        }
        
        // Append memory buffer columns to result
        for (name, mem_col) in memory_buffer.iter() {
            if let Some(names) = column_names {
                if !names.contains(&name.as_str()) {
                    continue;
                }
            }
            
            // Append memory buffer data to result columns
            let result_col = result.entry(name.clone()).or_insert_with(|| {
                mem_col.clone_empty()
            });
            
            result_col.append(mem_col);
        }
        
        Ok(result)
    }
    
    /// Get row count (main + memory buffer)
    pub fn row_count(&self) -> u64 {
        let main_count = self.main_storage.read().row_count();
        let memory_count = self.memory_ids.read().len() as u64;
        
        main_count + memory_count
    }
    
    /// Get schema
    pub fn get_schema(&self) -> Vec<(String, ColumnType)> {
        self.main_storage.read().get_schema()
    }
    
    /// Get column names
    pub fn column_names(&self) -> Vec<String> {
        self.main_storage.read().column_names()
    }
    
    // ========================================================================
    // Compaction
    // ========================================================================
    
    /// Check if compaction is needed
    pub fn needs_compaction(&self) -> bool {
        self.needs_compact.load(Ordering::SeqCst)
    }
    
    /// Compact WAL into main file
    pub fn compact(&self) -> io::Result<()> {
        let wal_buffer = self.wal_buffer.read();
        if wal_buffer.is_empty() {
            return Ok(());
        }
        
        // Collect all inserts from WAL (excluding deleted)
        let deleted_ids = self.deleted_ids.read();
        let rows: Vec<HashMap<String, ColumnValue>> = wal_buffer.iter()
            .filter_map(|r| match r {
                WalRecord::Insert { id, data } if !deleted_ids.contains(id) => {
                    Some(data.clone())
                }
                _ => None,
            })
            .collect();
        
        drop(wal_buffer);
        drop(deleted_ids);
        
        if rows.is_empty() {
            return Ok(());
        }
        
        // Insert into main storage and save
        {
            let main_storage = self.main_storage.write();
            main_storage.insert_rows(&rows)?;
            main_storage.save()?;
        }
        
        // Clear WAL and memory buffer
        {
            let mut wal_buffer = self.wal_buffer.write();
            let mut wal_writer = self.wal_writer.write();
            let mut deleted_ids = self.deleted_ids.write();
            let mut memory_buffer = self.memory_buffer.write();
            let mut memory_ids = self.memory_ids.write();
            
            wal_buffer.clear();
            deleted_ids.clear();
            memory_buffer.clear();
            memory_ids.clear();
            
            // Create fresh WAL file
            let wal_path = Self::wal_path(&self.path);
            *wal_writer = Some(WalWriter::create(&wal_path, self.next_id.load(Ordering::SeqCst))?);
        }
        
        self.needs_compact.store(false, Ordering::SeqCst);
        Ok(())
    }
    
    /// Save (compact and persist)
    pub fn save(&self) -> io::Result<()> {
        self.compact()
    }
    
    /// Flush WAL to disk
    pub fn flush(&self) -> io::Result<()> {
        let mut wal_writer = self.wal_writer.write();
        if let Some(writer) = wal_writer.as_mut() {
            writer.flush()?;
        }
        Ok(())
    }
    
    /// Close storage
    pub fn close(&self) -> io::Result<()> {
        self.compact()
    }
    
    // ========================================================================
    // Convenience methods for direct storage access
    // ========================================================================
    
    /// Get reference to main storage (for query operations)
    pub fn main_storage(&self) -> &RwLock<OnDemandStorage> {
        &self.main_storage
    }
    
    /// Get WAL record count
    pub fn wal_record_count(&self) -> usize {
        self.wal_buffer.read().len()
    }
    
    /// Set compaction threshold
    pub fn set_compaction_threshold(&mut self, threshold: usize) {
        self.compaction_threshold = threshold;
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
    fn test_wal_record_roundtrip() {
        let mut data = HashMap::new();
        data.insert("name".to_string(), ColumnValue::String("Alice".to_string()));
        data.insert("age".to_string(), ColumnValue::Int64(30));
        data.insert("score".to_string(), ColumnValue::Float64(95.5));
        
        let record = WalRecord::Insert { id: 42, data };
        let bytes = record.to_bytes();
        let (parsed, _) = WalRecord::from_bytes(&bytes).unwrap();
        
        if let WalRecord::Insert { id, data } = parsed {
            assert_eq!(id, 42);
            assert_eq!(data.len(), 3);
        } else {
            panic!("Expected Insert record");
        }
    }
    
    #[test]
    fn test_incremental_storage_basic() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test.apex");
        
        // Create and insert
        {
            let storage = IncrementalStorage::create(&path).unwrap();
            
            let mut row = HashMap::new();
            row.insert("name".to_string(), ColumnValue::String("Alice".to_string()));
            row.insert("age".to_string(), ColumnValue::Int64(30));
            
            let ids = storage.insert_rows(&[row]).unwrap();
            assert_eq!(ids.len(), 1);
            assert_eq!(storage.row_count(), 1);
            
            // Compact and close
            storage.compact().unwrap();
        }
        
        // Reopen and verify
        {
            let storage = IncrementalStorage::open(&path).unwrap();
            assert_eq!(storage.row_count(), 1);
        }
    }
    
    #[test]
    fn test_incremental_storage_wal_persistence() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test_wal.apex");
        
        // Insert without compacting
        {
            let storage = IncrementalStorage::create(&path).unwrap();
            
            for i in 0..10 {
                let mut row = HashMap::new();
                row.insert("val".to_string(), ColumnValue::Int64(i));
                storage.insert_rows(&[row]).unwrap();
            }
            
            storage.flush().unwrap();
            // Don't compact - leave data in WAL
        }
        
        // Reopen - should recover from WAL
        {
            let storage = IncrementalStorage::open(&path).unwrap();
            assert_eq!(storage.row_count(), 10);
            assert_eq!(storage.wal_record_count(), 10);
            
            // Compact now
            storage.compact().unwrap();
            assert_eq!(storage.wal_record_count(), 0);
        }
        
        // Reopen after compaction
        {
            let storage = IncrementalStorage::open(&path).unwrap();
            assert_eq!(storage.row_count(), 10);
        }
    }
    
    #[test]
    fn test_delete_in_wal() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test_del.apex");
        
        let storage = IncrementalStorage::create(&path).unwrap();
        
        // Insert 5 rows
        for i in 0..5 {
            let mut row = HashMap::new();
            row.insert("val".to_string(), ColumnValue::Int64(i));
            storage.insert_rows(&[row]).unwrap();
        }
        
        assert_eq!(storage.row_count(), 5);
        
        // Delete rows 1 and 3
        storage.delete(1).unwrap();
        storage.delete(3).unwrap();
        
        assert_eq!(storage.row_count(), 3);
        
        // Compact
        storage.compact().unwrap();
        
        // Verify final count
        assert_eq!(storage.row_count(), 3);
    }
}
