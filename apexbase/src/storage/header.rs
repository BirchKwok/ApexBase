//! File header definition

use super::{MAGIC, VERSION_MAJOR, VERSION_MINOR, DEFAULT_PAGE_SIZE, HEADER_SIZE};
use crate::{ApexError, Result};
use byteorder::{LittleEndian, ReadBytesExt};
use std::io::{Cursor, Read};

/// File header structure (4KB)
#[derive(Debug, Clone)]
pub struct FileHeader {
    /// Magic bytes "APEXBASE"
    pub magic: [u8; 8],
    /// Major version
    pub version_major: u16,
    /// Minor version
    pub version_minor: u16,
    /// Page size in bytes
    pub page_size: u32,
    /// Total number of pages
    pub total_pages: u64,
    /// Offset to table catalog
    pub table_catalog_offset: u64,
    /// Offset to schema region
    pub schema_region_offset: u64,
    /// Offset to index region
    pub index_region_offset: u64,
    /// Offset to data region
    pub data_region_offset: u64,
    /// Offset to free space map
    pub free_map_offset: u64,
    /// Offset to footer
    pub footer_offset: u64,
    /// Creation timestamp (Unix timestamp)
    pub created_at: i64,
    /// Last modified timestamp
    pub modified_at: i64,
    /// Flags
    pub flags: u32,
    /// Header checksum
    pub checksum: u32,
}

impl FileHeader {
    /// Create a new header with default values
    pub fn new() -> Self {
        let now = chrono::Utc::now().timestamp();
        Self {
            magic: *MAGIC,
            version_major: VERSION_MAJOR,
            version_minor: VERSION_MINOR,
            page_size: DEFAULT_PAGE_SIZE,
            total_pages: 1,
            table_catalog_offset: HEADER_SIZE as u64,
            schema_region_offset: 0,
            index_region_offset: 0,
            data_region_offset: 0,
            free_map_offset: 0,
            footer_offset: 0,
            created_at: now,
            modified_at: now,
            flags: 0,
            checksum: 0,
        }
    }

    /// Serialize to bytes
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut buf = Vec::with_capacity(HEADER_SIZE);

        // Write fields
        buf.extend_from_slice(&self.magic);
        buf.extend_from_slice(&self.version_major.to_le_bytes());
        buf.extend_from_slice(&self.version_minor.to_le_bytes());
        buf.extend_from_slice(&self.page_size.to_le_bytes());
        buf.extend_from_slice(&self.total_pages.to_le_bytes());
        buf.extend_from_slice(&self.table_catalog_offset.to_le_bytes());
        buf.extend_from_slice(&self.schema_region_offset.to_le_bytes());
        buf.extend_from_slice(&self.index_region_offset.to_le_bytes());
        buf.extend_from_slice(&self.data_region_offset.to_le_bytes());
        buf.extend_from_slice(&self.free_map_offset.to_le_bytes());
        buf.extend_from_slice(&self.footer_offset.to_le_bytes());
        buf.extend_from_slice(&self.created_at.to_le_bytes());
        buf.extend_from_slice(&self.modified_at.to_le_bytes());
        buf.extend_from_slice(&self.flags.to_le_bytes());
        
        // Calculate checksum (CRC32 of all previous bytes)
        let checksum = crc32fast::hash(&buf);
        buf.extend_from_slice(&checksum.to_le_bytes());

        // Pad to HEADER_SIZE
        buf.resize(HEADER_SIZE, 0);
        buf
    }

    /// Deserialize from bytes
    pub fn from_bytes(bytes: &[u8]) -> Result<Self> {
        if bytes.len() < HEADER_SIZE {
            return Err(ApexError::InvalidFileFormat);
        }

        let mut cursor = Cursor::new(bytes);

        // Read magic
        let mut magic = [0u8; 8];
        cursor.read_exact(&mut magic)?;
        if &magic != MAGIC {
            return Err(ApexError::InvalidFileFormat);
        }

        let version_major = cursor.read_u16::<LittleEndian>()?;
        let version_minor = cursor.read_u16::<LittleEndian>()?;
        let page_size = cursor.read_u32::<LittleEndian>()?;
        let total_pages = cursor.read_u64::<LittleEndian>()?;
        let table_catalog_offset = cursor.read_u64::<LittleEndian>()?;
        let schema_region_offset = cursor.read_u64::<LittleEndian>()?;
        let index_region_offset = cursor.read_u64::<LittleEndian>()?;
        let data_region_offset = cursor.read_u64::<LittleEndian>()?;
        let free_map_offset = cursor.read_u64::<LittleEndian>()?;
        let footer_offset = cursor.read_u64::<LittleEndian>()?;
        let created_at = cursor.read_i64::<LittleEndian>()?;
        let modified_at = cursor.read_i64::<LittleEndian>()?;
        let flags = cursor.read_u32::<LittleEndian>()?;
        let checksum = cursor.read_u32::<LittleEndian>()?;

        // Verify checksum
        let header_data = &bytes[..cursor.position() as usize - 4];
        let computed_checksum = crc32fast::hash(header_data);
        if computed_checksum != checksum {
            return Err(ApexError::ChecksumMismatch);
        }

        Ok(Self {
            magic,
            version_major,
            version_minor,
            page_size,
            total_pages,
            table_catalog_offset,
            schema_region_offset,
            index_region_offset,
            data_region_offset,
            free_map_offset,
            footer_offset,
            created_at,
            modified_at,
            flags,
            checksum,
        })
    }

    /// Update modified timestamp
    pub fn touch(&mut self) {
        self.modified_at = chrono::Utc::now().timestamp();
    }

    /// Validate the header
    pub fn validate(&self) -> Result<()> {
        if &self.magic != MAGIC {
            return Err(ApexError::InvalidFileFormat);
        }
        if self.version_major > VERSION_MAJOR {
            return Err(ApexError::VersionMismatch {
                expected: VERSION_MAJOR as u32,
                actual: self.version_major as u32,
            });
        }
        if self.page_size < 512 || self.page_size > 65536 {
            return Err(ApexError::InvalidFileFormat);
        }
        Ok(())
    }
}

impl Default for FileHeader {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_header_serialization() {
        let header = FileHeader::new();
        let bytes = header.to_bytes();
        
        assert_eq!(bytes.len(), HEADER_SIZE);
        
        let restored = FileHeader::from_bytes(&bytes).unwrap();
        assert_eq!(restored.magic, header.magic);
        assert_eq!(restored.version_major, header.version_major);
        assert_eq!(restored.page_size, header.page_size);
    }

    #[test]
    fn test_header_validation() {
        let header = FileHeader::new();
        assert!(header.validate().is_ok());
    }

    #[test]
    fn test_invalid_magic() {
        let mut bytes = FileHeader::new().to_bytes();
        bytes[0] = b'X'; // Corrupt magic
        
        assert!(FileHeader::from_bytes(&bytes).is_err());
    }
}

