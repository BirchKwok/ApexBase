//! Python bindings via PyO3
//!
//! V3Storage is the only storage implementation (on-demand reading without ColumnTable)

mod v3_bindings;

pub use v3_bindings::V3StorageImpl as V3Storage;
