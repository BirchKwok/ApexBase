//! Index module - B+Tree and other index implementations

mod btree;

pub use btree::BTreeIndex;

/// Row ID type
pub type RowId = u64;

