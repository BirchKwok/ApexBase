//! Page management for storage

use serde::{Deserialize, Serialize};

/// Page identifier
pub type PageId = u64;

/// Page type identifier
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[repr(u8)]
pub enum PageType {
    Free = 0,
    TableCatalog = 1,
    Schema = 2,
    BTreeInternal = 3,
    BTreeLeaf = 4,
    DataBlock = 5,
    Overflow = 6,
}

/// Page header (16 bytes)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PageHeader {
    /// Page type
    pub page_type: PageType,
    /// Flags
    pub flags: u8,
    /// Number of items in this page
    pub item_count: u16,
    /// Free space start offset
    pub free_start: u16,
    /// Free space end offset
    pub free_end: u16,
    /// Next page ID (for overflow or linked lists)
    pub next_page: PageId,
}

impl PageHeader {
    pub fn new(page_type: PageType, page_size: u16) -> Self {
        Self {
            page_type,
            flags: 0,
            item_count: 0,
            free_start: 16, // After header
            free_end: page_size,
            next_page: 0,
        }
    }
}

/// A page in the storage file
#[derive(Debug, Clone)]
pub struct Page {
    /// Page ID
    pub id: PageId,
    /// Page header
    pub header: PageHeader,
    /// Page data
    pub data: Vec<u8>,
    /// Whether the page is dirty (modified)
    pub dirty: bool,
}

impl Page {
    /// Create a new page
    pub fn new(id: PageId, page_type: PageType, page_size: usize) -> Self {
        let header = PageHeader::new(page_type, page_size as u16);
        Self {
            id,
            header,
            data: vec![0; page_size],
            dirty: true,
        }
    }

    /// Get available free space
    pub fn free_space(&self) -> usize {
        if self.header.free_end > self.header.free_start {
            (self.header.free_end - self.header.free_start) as usize
        } else {
            0
        }
    }

    /// Get a slice of the data
    pub fn data_slice(&self, start: usize, len: usize) -> &[u8] {
        &self.data[start..start + len]
    }

    /// Get a mutable slice of the data
    pub fn data_slice_mut(&mut self, start: usize, len: usize) -> &mut [u8] {
        self.dirty = true;
        &mut self.data[start..start + len]
    }

    /// Write data at offset
    pub fn write_at(&mut self, offset: usize, data: &[u8]) {
        self.data[offset..offset + data.len()].copy_from_slice(data);
        self.dirty = true;
    }

    /// Read data from offset
    pub fn read_at(&self, offset: usize, len: usize) -> &[u8] {
        &self.data[offset..offset + len]
    }

    /// Allocate space from the front
    pub fn allocate_front(&mut self, size: usize) -> Option<usize> {
        if self.free_space() < size {
            return None;
        }
        let offset = self.header.free_start as usize;
        self.header.free_start += size as u16;
        self.header.item_count += 1;
        self.dirty = true;
        Some(offset)
    }

    /// Allocate space from the back (for variable-length data)
    pub fn allocate_back(&mut self, size: usize) -> Option<usize> {
        if self.free_space() < size {
            return None;
        }
        self.header.free_end -= size as u16;
        self.dirty = true;
        Some(self.header.free_end as usize)
    }

    /// Serialize the page to bytes
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut buf = Vec::with_capacity(self.data.len());
        
        // Write header
        buf.push(self.header.page_type as u8);
        buf.push(self.header.flags);
        buf.extend_from_slice(&self.header.item_count.to_le_bytes());
        buf.extend_from_slice(&self.header.free_start.to_le_bytes());
        buf.extend_from_slice(&self.header.free_end.to_le_bytes());
        buf.extend_from_slice(&self.header.next_page.to_le_bytes());
        
        // Write data (after header)
        buf.extend_from_slice(&self.data[16..]);
        
        buf
    }

    /// Deserialize from bytes
    pub fn from_bytes(id: PageId, bytes: &[u8]) -> Self {
        let page_type = match bytes[0] {
            0 => PageType::Free,
            1 => PageType::TableCatalog,
            2 => PageType::Schema,
            3 => PageType::BTreeInternal,
            4 => PageType::BTreeLeaf,
            5 => PageType::DataBlock,
            6 => PageType::Overflow,
            _ => PageType::Free,
        };
        
        let header = PageHeader {
            page_type,
            flags: bytes[1],
            item_count: u16::from_le_bytes([bytes[2], bytes[3]]),
            free_start: u16::from_le_bytes([bytes[4], bytes[5]]),
            free_end: u16::from_le_bytes([bytes[6], bytes[7]]),
            next_page: u64::from_le_bytes([
                bytes[8], bytes[9], bytes[10], bytes[11],
                bytes[12], bytes[13], bytes[14], bytes[15],
            ]),
        };
        
        Self {
            id,
            header,
            data: bytes.to_vec(),
            dirty: false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_page_allocation() {
        let mut page = Page::new(0, PageType::DataBlock, 4096);
        
        // Allocate from front
        let offset1 = page.allocate_front(100).unwrap();
        assert_eq!(offset1, 16);
        
        let offset2 = page.allocate_front(200).unwrap();
        assert_eq!(offset2, 116);
        
        // Allocate from back
        let offset3 = page.allocate_back(50).unwrap();
        assert_eq!(offset3, 4046);
    }

    #[test]
    fn test_page_serialization() {
        let mut page = Page::new(1, PageType::DataBlock, 4096);
        page.write_at(20, b"Hello, World!");
        
        let bytes = page.to_bytes();
        let restored = Page::from_bytes(1, &bytes);
        
        assert_eq!(restored.header.page_type, PageType::DataBlock);
        assert_eq!(&restored.data[20..33], b"Hello, World!");
    }
}

