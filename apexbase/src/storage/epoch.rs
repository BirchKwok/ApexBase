//! Per-table generations used to validate cross-layer caches.
//!
//! Storage owns the generation because it is the only layer that can commit a
//! logical table mutation. Readers keep the generation they observed and
//! discard cached state when it no longer matches.

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use dashmap::DashMap;
use once_cell::sync::Lazy;

static TABLE_EPOCHS: Lazy<DashMap<PathBuf, AtomicU64>> = Lazy::new(DashMap::new);

#[inline]
pub fn current(table_path: &Path) -> u64 {
    TABLE_EPOCHS
        .get(table_path)
        .map(|entry| entry.load(Ordering::Acquire))
        .unwrap_or(0)
}

/// Advance the epoch once after a logical write has committed.
#[inline]
pub fn bump(table_path: &Path) -> u64 {
    let entry = TABLE_EPOCHS
        .entry(table_path.to_path_buf())
        .or_insert_with(|| AtomicU64::new(0));
    entry.fetch_add(1, Ordering::AcqRel) + 1
}

/// Drop obsolete bookkeeping for tables removed as part of DDL.
#[inline]
pub fn remove(table_path: &Path) {
    TABLE_EPOCHS.remove(table_path);
}

#[inline]
pub fn remove_dir(dir: &Path) {
    TABLE_EPOCHS.retain(|path, _| !path.starts_with(dir));
}
