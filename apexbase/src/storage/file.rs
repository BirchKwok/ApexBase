//! File I/O with memory mapping

use super::{FileHeader, Page, PageId, HEADER_SIZE, DEFAULT_PAGE_SIZE};
use crate::{ApexError, Result};
use memmap2::{MmapMut, MmapOptions};
use std::fs::{File, OpenOptions};
use std::path::Path;

/// ApexBase file handle
pub struct ApexFile {
    /// Underlying file
    file: File,
    /// Memory-mapped region
    mmap: Option<MmapMut>,
    /// File header
    header: FileHeader,
    /// Page size
    page_size: usize,
    /// File path
    path: std::path::PathBuf,
}

impl ApexFile {
    /// Create a new file
    pub fn create(path: &Path) -> Result<Self> {
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(true)
            .open(path)?;

        // Initialize with header + one page for table catalog
        let initial_size = HEADER_SIZE + DEFAULT_PAGE_SIZE as usize;
        file.set_len(initial_size as u64)?;

        let mut apex_file = Self {
            file,
            mmap: None,
            header: FileHeader::new(),
            page_size: DEFAULT_PAGE_SIZE as usize,
            path: path.to_path_buf(),
        };

        // Memory map the file
        apex_file.remap()?;

        // Write header
        apex_file.write_header()?;

        Ok(apex_file)
    }

    /// Open an existing file
    pub fn open(path: &Path) -> Result<Self> {
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .open(path)?;

        let mmap = unsafe { MmapOptions::new().map_mut(&file)? };

        // Read header
        let header = FileHeader::from_bytes(&mmap[..HEADER_SIZE])?;
        header.validate()?;

        let page_size = header.page_size as usize;

        Ok(Self {
            file,
            mmap: Some(mmap),
            header,
            page_size,
            path: path.to_path_buf(),
        })
    }

    /// Re-map the file (after resize)
    fn remap(&mut self) -> Result<()> {
        // Drop existing mapping
        self.mmap = None;

        // Create new mapping
        let mmap = unsafe { MmapOptions::new().map_mut(&self.file)? };
        self.mmap = Some(mmap);

        Ok(())
    }

    /// Write the file header
    fn write_header(&mut self) -> Result<()> {
        let header_bytes = self.header.to_bytes();
        if let Some(mmap) = &mut self.mmap {
            mmap[..HEADER_SIZE].copy_from_slice(&header_bytes);
        }
        Ok(())
    }

    /// Get the header
    pub fn header(&self) -> &FileHeader {
        &self.header
    }

    /// Get mutable header
    pub fn header_mut(&mut self) -> &mut FileHeader {
        &mut self.header
    }

    /// Get page size
    pub fn page_size(&self) -> usize {
        self.page_size
    }

    /// Get total number of pages
    pub fn total_pages(&self) -> u64 {
        self.header.total_pages
    }

    /// Calculate page offset
    fn page_offset(&self, page_id: PageId) -> usize {
        HEADER_SIZE + (page_id as usize) * self.page_size
    }

    /// Read a page
    pub fn read_page(&self, page_id: PageId) -> Result<Page> {
        let offset = self.page_offset(page_id);
        let end = offset + self.page_size;

        if let Some(mmap) = &self.mmap {
            if end > mmap.len() {
                return Err(ApexError::Io(std::io::Error::new(
                    std::io::ErrorKind::UnexpectedEof,
                    "Page out of bounds",
                )));
            }
            Ok(Page::from_bytes(page_id, &mmap[offset..end]))
        } else {
            Err(ApexError::Io(std::io::Error::new(
                std::io::ErrorKind::Other,
                "File not mapped",
            )))
        }
    }

    /// Write a page
    pub fn write_page(&mut self, page: &Page) -> Result<()> {
        let offset = self.page_offset(page.id);
        let end = offset + self.page_size;

        // Extend file if necessary
        if end > self.file.metadata()?.len() as usize {
            self.extend_to_page(page.id)?;
        }

        if let Some(mmap) = &mut self.mmap {
            let page_bytes = page.to_bytes();
            mmap[offset..offset + page_bytes.len()].copy_from_slice(&page_bytes);
        }

        Ok(())
    }

    /// Read raw bytes at offset
    pub fn read_bytes(&self, offset: u64, len: usize) -> Result<Vec<u8>> {
        if let Some(mmap) = &self.mmap {
            let start = offset as usize;
            let end = start + len;
            if end > mmap.len() {
                return Err(ApexError::Io(std::io::Error::new(
                    std::io::ErrorKind::UnexpectedEof,
                    "Read out of bounds",
                )));
            }
            Ok(mmap[start..end].to_vec())
        } else {
            Err(ApexError::Io(std::io::Error::new(
                std::io::ErrorKind::Other,
                "File not mapped",
            )))
        }
    }

    /// Write raw bytes at offset
    pub fn write_bytes(&mut self, offset: u64, data: &[u8]) -> Result<()> {
        let start = offset as usize;
        let end = start + data.len();

        // Extend file if necessary
        if end > self.file.metadata()?.len() as usize {
            let new_size = end.max(self.file.metadata()?.len() as usize * 2);
            self.resize(new_size as u64)?;
        }

        if let Some(mmap) = &mut self.mmap {
            mmap[start..end].copy_from_slice(data);
        }

        Ok(())
    }

    /// Extend file to accommodate a page
    fn extend_to_page(&mut self, page_id: PageId) -> Result<()> {
        let required_size = self.page_offset(page_id + 1);
        let current_size = self.file.metadata()?.len() as usize;

        if required_size > current_size {
            // Double the file size or use required size, whichever is larger
            let new_size = (current_size * 2).max(required_size);
            self.resize(new_size as u64)?;
        }

        // Update total pages
        if page_id >= self.header.total_pages {
            self.header.total_pages = page_id + 1;
        }

        Ok(())
    }

    /// Resize the file
    pub fn resize(&mut self, new_size: u64) -> Result<()> {
        // Flush and unmap
        if let Some(mmap) = &self.mmap {
            mmap.flush()?;
        }
        self.mmap = None;

        // Resize file
        self.file.set_len(new_size)?;

        // Remap
        self.remap()?;

        Ok(())
    }

    /// Allocate a new page
    pub fn allocate_page(&mut self) -> Result<PageId> {
        let page_id = self.header.total_pages;
        self.extend_to_page(page_id)?;
        self.header.total_pages = page_id + 1;
        self.header.touch();
        Ok(page_id)
    }

    /// Flush changes to disk
    pub fn flush(&mut self) -> Result<()> {
        self.write_header()?;
        if let Some(mmap) = &self.mmap {
            mmap.flush()?;
        }
        Ok(())
    }

    /// Sync file to disk
    pub fn sync(&self) -> Result<()> {
        self.file.sync_all()?;
        Ok(())
    }

    /// Get file size
    pub fn file_size(&self) -> Result<u64> {
        Ok(self.file.metadata()?.len())
    }

    /// Get the file path
    pub fn path(&self) -> &Path {
        &self.path
    }
}

impl Drop for ApexFile {
    fn drop(&mut self) {
        // Try to flush on drop
        let _ = self.flush();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn test_create_and_open() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test.apex");

        // Create file
        {
            let file = ApexFile::create(&path).unwrap();
            assert_eq!(file.header().version_major, 1);
        }

        // Open file
        {
            let file = ApexFile::open(&path).unwrap();
            assert_eq!(file.header().version_major, 1);
        }
    }

    #[test]
    fn test_page_operations() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test.apex");

        let mut file = ApexFile::create(&path).unwrap();

        // Allocate and write a page
        let page_id = file.allocate_page().unwrap();
        let mut page = Page::new(page_id, super::super::page::PageType::DataBlock, file.page_size());
        page.write_at(20, b"Test data");
        file.write_page(&page).unwrap();
        file.flush().unwrap();

        // Read it back
        let read_page = file.read_page(page_id).unwrap();
        assert_eq!(&read_page.data[20..29], b"Test data");
    }

    #[test]
    fn test_raw_bytes() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test.apex");

        let mut file = ApexFile::create(&path).unwrap();

        // Write raw bytes
        file.write_bytes(HEADER_SIZE as u64 + 100, b"Hello").unwrap();
        file.flush().unwrap();

        // Read them back
        let bytes = file.read_bytes(HEADER_SIZE as u64 + 100, 5).unwrap();
        assert_eq!(&bytes, b"Hello");
    }
}

