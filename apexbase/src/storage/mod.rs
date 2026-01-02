//! Storage module - File I/O and page management

mod file;
mod page;
mod header;
mod wal;
mod segment;
mod columnar_file;

// Legacy row-based storage engine (deprecated, kept for reference)
#[allow(dead_code)]
mod engine;

pub use file::ApexFile;
pub use page::{Page, PageId};
pub use header::FileHeader;
pub use wal::{WalManager, WalRecord, WalOp};
pub use segment::{SegmentStorage, SegmentConfig, FastBatchBuilder};
pub use columnar_file::{ColumnarStorage, ColumnType, ColumnValue, FileSchema};

/// Constants for file format
pub const MAGIC: &[u8; 8] = b"APEXBASE";
pub const VERSION_MAJOR: u16 = 1;
pub const VERSION_MINOR: u16 = 0;
pub const DEFAULT_PAGE_SIZE: u32 = 4096;
pub const HEADER_SIZE: usize = 4096;

