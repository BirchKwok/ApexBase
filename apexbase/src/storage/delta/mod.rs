//! Delta Store - In-place update and delete without full file rewrite
//!
//! Architecture:
//! ```text
//! ┌─────────────────────────────────────────────┐
//! │              DeltaStore                      │
//! │  - Tracks row-level updates & deletes       │
//! │  - Merges with base columnar data on read   │
//! │  - Background compaction to base file       │
//! ├─────────────────────────────────────────────┤
//! │  UpdateLog        │  DeleteBitmap            │
//! │  - (row, col,     │  - Bitset of deleted    │
//! │    new_value)     │    row IDs              │
//! │  - Append-only    │  - O(1) lookup          │
//! └──────────────────┴──────────────────────────┘
//! ```

pub mod update_log;
pub mod merge;

pub use update_log::{DeltaStore, DeltaRecord, DeleteBitmap};
pub use merge::DeltaMerger;
