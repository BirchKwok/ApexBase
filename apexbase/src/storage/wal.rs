//! Write-Ahead Log (WAL) implementation for fast durable writes
//!
//! The WAL provides:
//! - Fast sequential writes (append-only)
//! - Crash recovery support
//! - Background checkpoint to main data file
//!
//! File format:
//! ```text
//! +----------------+----------------+----------------+
//! | Header (32B)   | Record 1       | Record 2 ...   |
//! +----------------+----------------+----------------+
//!
//! Header:
//! - magic: 4 bytes "AWAL"
//! - version: 2 bytes
//! - segment_id: 8 bytes
//! - record_count: 8 bytes
//! - checksum: 4 bytes
//! - reserved: 6 bytes
//!
//! Record:
//! - op: 1 byte (INSERT=1, UPDATE=2, DELETE=3, BATCH_INSERT=4)
//! - table_name_len: 1 byte
//! - table_name: variable
//! - payload_len: 4 bytes
//! - payload: variable (bincode serialized)
//! - crc32: 4 bytes
//! ```

use std::fs::{File, OpenOptions};
use std::io::{self, BufWriter, Read, Write, Seek, SeekFrom};
use std::path::{Path, PathBuf};
use std::collections::HashMap;
use crate::data::Value;
use serde::{Deserialize, Serialize};

/// WAL magic bytes
const WAL_MAGIC: &[u8; 4] = b"AWAL";
/// WAL version
const WAL_VERSION: u16 = 1;
/// WAL header size
const WAL_HEADER_SIZE: usize = 32;
/// Maximum WAL segment size (16 MB)
const MAX_WAL_SEGMENT_SIZE: u64 = 16 * 1024 * 1024;
/// Buffer size for writes
const WRITE_BUFFER_SIZE: usize = 64 * 1024; // 64 KB

/// WAL operation types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[repr(u8)]
pub enum WalOp {
    Insert = 1,
    Update = 2,
    Delete = 3,
    BatchInsert = 4,
}

impl TryFrom<u8> for WalOp {
    type Error = io::Error;
    
    fn try_from(value: u8) -> Result<Self, Self::Error> {
        match value {
            1 => Ok(WalOp::Insert),
            2 => Ok(WalOp::Update),
            3 => Ok(WalOp::Delete),
            4 => Ok(WalOp::BatchInsert),
            _ => Err(io::Error::new(io::ErrorKind::InvalidData, "Invalid WAL op")),
        }
    }
}

/// WAL record for a single operation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WalRecord {
    pub op: WalOp,
    pub table_name: String,
    pub id: Option<u64>,
    pub fields: Option<HashMap<String, Value>>,
    pub batch_fields: Option<Vec<HashMap<String, Value>>>,
}

impl WalRecord {
    pub fn insert(table_name: &str, id: u64, fields: HashMap<String, Value>) -> Self {
        Self {
            op: WalOp::Insert,
            table_name: table_name.to_string(),
            id: Some(id),
            fields: Some(fields),
            batch_fields: None,
        }
    }

    pub fn update(table_name: &str, id: u64, fields: HashMap<String, Value>) -> Self {
        Self {
            op: WalOp::Update,
            table_name: table_name.to_string(),
            id: Some(id),
            fields: Some(fields),
            batch_fields: None,
        }
    }

    pub fn delete(table_name: &str, id: u64) -> Self {
        Self {
            op: WalOp::Delete,
            table_name: table_name.to_string(),
            id: Some(id),
            fields: None,
            batch_fields: None,
        }
    }

    pub fn batch_insert(table_name: &str, rows: Vec<HashMap<String, Value>>) -> Self {
        Self {
            op: WalOp::BatchInsert,
            table_name: table_name.to_string(),
            id: None,
            fields: None,
            batch_fields: Some(rows),
        }
    }
}

/// WAL segment header
#[derive(Debug, Clone)]
struct WalHeader {
    magic: [u8; 4],
    version: u16,
    segment_id: u64,
    record_count: u64,
    checksum: u32,
}

impl WalHeader {
    fn new(segment_id: u64) -> Self {
        Self {
            magic: *WAL_MAGIC,
            version: WAL_VERSION,
            segment_id,
            record_count: 0,
            checksum: 0,
        }
    }

    fn to_bytes(&self) -> [u8; WAL_HEADER_SIZE] {
        let mut buf = [0u8; WAL_HEADER_SIZE];
        buf[0..4].copy_from_slice(&self.magic);
        buf[4..6].copy_from_slice(&self.version.to_le_bytes());
        buf[6..14].copy_from_slice(&self.segment_id.to_le_bytes());
        buf[14..22].copy_from_slice(&self.record_count.to_le_bytes());
        
        // Calculate checksum over first 22 bytes
        let checksum = crc32fast::hash(&buf[0..22]);
        buf[22..26].copy_from_slice(&checksum.to_le_bytes());
        
        buf
    }

    fn from_bytes(bytes: &[u8]) -> io::Result<Self> {
        if bytes.len() < WAL_HEADER_SIZE {
            return Err(io::Error::new(io::ErrorKind::InvalidData, "Header too short"));
        }

        let magic: [u8; 4] = bytes[0..4].try_into().unwrap();
        if &magic != WAL_MAGIC {
            return Err(io::Error::new(io::ErrorKind::InvalidData, "Invalid WAL magic"));
        }

        let version = u16::from_le_bytes(bytes[4..6].try_into().unwrap());
        let segment_id = u64::from_le_bytes(bytes[6..14].try_into().unwrap());
        let record_count = u64::from_le_bytes(bytes[14..22].try_into().unwrap());
        let checksum = u32::from_le_bytes(bytes[22..26].try_into().unwrap());

        // Verify checksum
        let computed = crc32fast::hash(&bytes[0..22]);
        if computed != checksum {
            return Err(io::Error::new(io::ErrorKind::InvalidData, "Header checksum mismatch"));
        }

        Ok(Self {
            magic,
            version,
            segment_id,
            record_count,
            checksum,
        })
    }
}

/// A single WAL segment file
struct WalSegment {
    path: PathBuf,
    file: BufWriter<File>,
    header: WalHeader,
    current_size: u64,
}

impl WalSegment {
    /// Create a new WAL segment
    fn create(dir: &Path, segment_id: u64) -> io::Result<Self> {
        let path = dir.join(format!("{:06}.wal", segment_id));
        let file = OpenOptions::new()
            .create(true)
            .write(true)
            .truncate(true)
            .open(&path)?;
        
        let mut writer = BufWriter::with_capacity(WRITE_BUFFER_SIZE, file);
        let header = WalHeader::new(segment_id);
        
        // Write header
        writer.write_all(&header.to_bytes())?;
        writer.flush()?;
        
        Ok(Self {
            path,
            file: writer,
            header,
            current_size: WAL_HEADER_SIZE as u64,
        })
    }

    /// Open an existing WAL segment for appending
    fn open_append(path: &Path) -> io::Result<Self> {
        let mut file = OpenOptions::new()
            .read(true)
            .write(true)
            .open(path)?;
        
        // Read header
        let mut header_buf = [0u8; WAL_HEADER_SIZE];
        file.read_exact(&mut header_buf)?;
        let header = WalHeader::from_bytes(&header_buf)?;
        
        // Seek to end
        let current_size = file.seek(SeekFrom::End(0))?;
        
        Ok(Self {
            path: path.to_path_buf(),
            file: BufWriter::with_capacity(WRITE_BUFFER_SIZE, file),
            header,
            current_size,
        })
    }

    /// Append a record to the segment
    fn append(&mut self, record: &WalRecord) -> io::Result<()> {
        // Serialize the record
        let payload = bincode::serialize(record)
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
        
        // Calculate CRC
        let crc = crc32fast::hash(&payload);
        
        // Write: length (4 bytes) + payload + crc (4 bytes)
        let len = payload.len() as u32;
        self.file.write_all(&len.to_le_bytes())?;
        self.file.write_all(&payload)?;
        self.file.write_all(&crc.to_le_bytes())?;
        
        self.current_size += 4 + payload.len() as u64 + 4;
        self.header.record_count += 1;
        
        Ok(())
    }

    /// Sync to disk
    fn sync(&mut self) -> io::Result<()> {
        self.file.flush()?;
        
        // Update header with record count
        let inner = self.file.get_mut();
        inner.seek(SeekFrom::Start(0))?;
        inner.write_all(&self.header.to_bytes())?;
        inner.flush()?;
        inner.sync_all()?;
        inner.seek(SeekFrom::End(0))?;
        
        Ok(())
    }

    /// Check if segment is full
    fn is_full(&self) -> bool {
        self.current_size >= MAX_WAL_SEGMENT_SIZE
    }

    /// Get segment ID
    fn segment_id(&self) -> u64 {
        self.header.segment_id
    }
}

/// WAL manager handles multiple segments
pub struct WalManager {
    wal_dir: PathBuf,
    current_segment: Option<WalSegment>,
    next_segment_id: u64,
    /// Records buffered in memory before writing
    write_buffer: Vec<WalRecord>,
    /// Buffer capacity before auto-flush
    buffer_capacity: usize,
    /// Whether WAL is enabled
    enabled: bool,
}

impl WalManager {
    /// Create a new WAL manager
    pub fn new(base_dir: &Path) -> io::Result<Self> {
        let wal_dir = base_dir.join("wal");
        std::fs::create_dir_all(&wal_dir)?;
        
        // Find existing segments and determine next ID
        let next_segment_id = Self::find_next_segment_id(&wal_dir)?;
        
        Ok(Self {
            wal_dir,
            current_segment: None,
            next_segment_id,
            write_buffer: Vec::with_capacity(1000),
            buffer_capacity: 1000, // Buffer up to 1000 records before flush
            enabled: true,
        })
    }

    /// Create a disabled WAL manager (for testing or when WAL is not needed)
    pub fn disabled() -> Self {
        Self {
            wal_dir: PathBuf::new(),
            current_segment: None,
            next_segment_id: 0,
            write_buffer: Vec::new(),
            buffer_capacity: 0,
            enabled: false,
        }
    }

    /// Check if WAL is enabled
    pub fn is_enabled(&self) -> bool {
        self.enabled
    }

    /// Set buffer capacity
    pub fn set_buffer_capacity(&mut self, capacity: usize) {
        self.buffer_capacity = capacity;
        self.write_buffer.reserve(capacity);
    }

    /// Find the next segment ID by scanning existing files
    fn find_next_segment_id(wal_dir: &Path) -> io::Result<u64> {
        let mut max_id = 0u64;
        
        if let Ok(entries) = std::fs::read_dir(wal_dir) {
            for entry in entries.flatten() {
                let name = entry.file_name();
                let name_str = name.to_string_lossy();
                if name_str.ends_with(".wal") {
                    if let Ok(id) = name_str.trim_end_matches(".wal").parse::<u64>() {
                        max_id = max_id.max(id);
                    }
                }
            }
        }
        
        Ok(max_id + 1)
    }

    /// Ensure we have an active segment
    fn ensure_segment(&mut self) -> io::Result<()> {
        if self.current_segment.is_none() || self.current_segment.as_ref().unwrap().is_full() {
            // Create new segment
            let segment = WalSegment::create(&self.wal_dir, self.next_segment_id)?;
            self.next_segment_id += 1;
            self.current_segment = Some(segment);
        }
        Ok(())
    }

    /// Append a record (buffered)
    pub fn append(&mut self, record: WalRecord) -> io::Result<()> {
        if !self.enabled {
            return Ok(());
        }

        self.write_buffer.push(record);
        
        // Auto-flush if buffer is full
        if self.write_buffer.len() >= self.buffer_capacity {
            self.flush()?;
        }
        
        Ok(())
    }

    /// Append a batch of records (buffered)
    pub fn append_batch(&mut self, records: Vec<WalRecord>) -> io::Result<()> {
        if !self.enabled {
            return Ok(());
        }

        self.write_buffer.extend(records);
        
        // Auto-flush if buffer is full
        if self.write_buffer.len() >= self.buffer_capacity {
            self.flush()?;
        }
        
        Ok(())
    }

    /// Flush buffered records to disk
    pub fn flush(&mut self) -> io::Result<()> {
        if !self.enabled || self.write_buffer.is_empty() {
            return Ok(());
        }

        self.ensure_segment()?;
        
        let segment = self.current_segment.as_mut().unwrap();
        
        // Write all buffered records
        for record in self.write_buffer.drain(..) {
            segment.append(&record)?;
            
            // Check if segment is full and rotate
            if segment.is_full() {
                segment.sync()?;
                let new_segment = WalSegment::create(&self.wal_dir, self.next_segment_id)?;
                self.next_segment_id += 1;
                *segment = new_segment;
            }
        }
        
        segment.sync()?;
        Ok(())
    }

    /// Sync current segment to disk (without clearing buffer)
    pub fn sync(&mut self) -> io::Result<()> {
        if let Some(segment) = &mut self.current_segment {
            segment.sync()?;
        }
        Ok(())
    }

    /// Read all records from all WAL segments (for recovery)
    pub fn read_all_records(&self) -> io::Result<Vec<WalRecord>> {
        if !self.enabled {
            return Ok(Vec::new());
        }

        let mut all_records = Vec::new();
        let mut segments: Vec<_> = std::fs::read_dir(&self.wal_dir)?
            .filter_map(|e| e.ok())
            .filter(|e| e.path().extension().map(|x| x == "wal").unwrap_or(false))
            .collect();
        
        // Sort by name (which includes segment ID)
        segments.sort_by_key(|e| e.path());
        
        for entry in segments {
            let path = entry.path();
            let records = self.read_segment(&path)?;
            all_records.extend(records);
        }
        
        Ok(all_records)
    }

    /// Read records from a single segment
    fn read_segment(&self, path: &Path) -> io::Result<Vec<WalRecord>> {
        let mut file = File::open(path)?;
        let mut records = Vec::new();
        
        // Read header
        let mut header_buf = [0u8; WAL_HEADER_SIZE];
        if file.read_exact(&mut header_buf).is_err() {
            return Ok(records); // Empty or corrupt segment
        }
        
        let header = WalHeader::from_bytes(&header_buf)?;
        
        // Read records
        for _ in 0..header.record_count {
            // Read length
            let mut len_buf = [0u8; 4];
            if file.read_exact(&mut len_buf).is_err() {
                break; // End of file or corrupt
            }
            let len = u32::from_le_bytes(len_buf) as usize;
            
            // Read payload
            let mut payload = vec![0u8; len];
            if file.read_exact(&mut payload).is_err() {
                break;
            }
            
            // Read and verify CRC
            let mut crc_buf = [0u8; 4];
            if file.read_exact(&mut crc_buf).is_err() {
                break;
            }
            let stored_crc = u32::from_le_bytes(crc_buf);
            let computed_crc = crc32fast::hash(&payload);
            
            if stored_crc != computed_crc {
                // CRC mismatch, skip this record but continue
                continue;
            }
            
            // Deserialize record
            if let Ok(record) = bincode::deserialize::<WalRecord>(&payload) {
                records.push(record);
            }
        }
        
        Ok(records)
    }

    /// Clear all WAL segments (after checkpoint)
    pub fn clear(&mut self) -> io::Result<()> {
        if !self.enabled {
            return Ok(());
        }

        // Close current segment
        self.current_segment = None;
        
        // Remove all .wal files
        if let Ok(entries) = std::fs::read_dir(&self.wal_dir) {
            for entry in entries.flatten() {
                let path = entry.path();
                if path.extension().map(|x| x == "wal").unwrap_or(false) {
                    let _ = std::fs::remove_file(path);
                }
            }
        }
        
        // Reset segment ID
        self.next_segment_id = 1;
        
        Ok(())
    }

    /// Get total size of WAL files
    pub fn total_size(&self) -> u64 {
        if !self.enabled {
            return 0;
        }

        let mut total = 0u64;
        if let Ok(entries) = std::fs::read_dir(&self.wal_dir) {
            for entry in entries.flatten() {
                if let Ok(metadata) = entry.metadata() {
                    total += metadata.len();
                }
            }
        }
        total
    }

    /// Get number of WAL segments
    pub fn segment_count(&self) -> usize {
        if !self.enabled {
            return 0;
        }

        std::fs::read_dir(&self.wal_dir)
            .map(|entries| {
                entries
                    .filter_map(|e| e.ok())
                    .filter(|e| e.path().extension().map(|x| x == "wal").unwrap_or(false))
                    .count()
            })
            .unwrap_or(0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn test_wal_basic() {
        let dir = tempdir().unwrap();
        let mut wal = WalManager::new(dir.path()).unwrap();
        wal.set_buffer_capacity(10); // Small buffer for testing
        
        // Append some records
        let mut fields = HashMap::new();
        fields.insert("name".to_string(), Value::String("Alice".to_string()));
        fields.insert("age".to_string(), Value::Int64(30));
        
        wal.append(WalRecord::insert("default", 1, fields.clone())).unwrap();
        wal.append(WalRecord::insert("default", 2, fields.clone())).unwrap();
        wal.flush().unwrap();
        
        // Read back
        let records = wal.read_all_records().unwrap();
        assert_eq!(records.len(), 2);
        assert_eq!(records[0].op, WalOp::Insert);
        assert_eq!(records[0].id, Some(1));
    }

    #[test]
    fn test_wal_batch() {
        let dir = tempdir().unwrap();
        let mut wal = WalManager::new(dir.path()).unwrap();
        
        let rows: Vec<HashMap<String, Value>> = (0..100)
            .map(|i| {
                let mut fields = HashMap::new();
                fields.insert("value".to_string(), Value::Int64(i));
                fields
            })
            .collect();
        
        wal.append(WalRecord::batch_insert("default", rows)).unwrap();
        wal.flush().unwrap();
        
        let records = wal.read_all_records().unwrap();
        assert_eq!(records.len(), 1);
        assert_eq!(records[0].op, WalOp::BatchInsert);
        assert_eq!(records[0].batch_fields.as_ref().unwrap().len(), 100);
    }

    #[test]
    fn test_wal_clear() {
        let dir = tempdir().unwrap();
        let mut wal = WalManager::new(dir.path()).unwrap();
        
        let mut fields = HashMap::new();
        fields.insert("test".to_string(), Value::Int64(1));
        
        wal.append(WalRecord::insert("default", 1, fields)).unwrap();
        wal.flush().unwrap();
        
        assert!(wal.segment_count() > 0);
        
        wal.clear().unwrap();
        assert_eq!(wal.segment_count(), 0);
    }
}

