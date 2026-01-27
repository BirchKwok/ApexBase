//! Storage module - ApexV3 On-Demand Columnar Storage
//!
//! This module provides the core columnar storage format for ApexBase.
//! The V3 format supports on-demand column/row reading without loading
//! the entire dataset into memory.
//!
//! Incremental writes use WAL (Write-Ahead Log) for fast append-only writes.

pub mod on_demand;
pub mod backend;
pub mod incremental;

// ============================================================================
// Durability Level - Controls fsync behavior for ACID guarantees
// ============================================================================

/// Durability level for write operations
/// 
/// Controls how aggressively data is synced to disk:
/// - `Fast`: No fsync - fastest but data may be lost on crash
/// - `Safe`: fsync on flush() - balanced performance and durability  
/// - `Max`: fsync on every write - strongest ACID guarantee but slower
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum DurabilityLevel {
    /// Highest performance, no fsync. Data written to OS buffer only.
    /// Suitable for batch import, reconstructible data, performance-critical scenarios.
    /// Risk: Data loss possible on system crash before OS flushes buffers.
    #[default]
    Fast,
    
    /// Balanced mode. fsync called on explicit flush() calls.
    /// Suitable for most production environments.
    /// Risk: Data loss possible only for writes between last flush and crash.
    Safe,
    
    /// Strongest ACID guarantee. fsync on every write operation.
    /// Suitable for financial, orders, and critical data scenarios.
    /// Performance: ~10-50x slower than Fast mode due to disk latency.
    Max,
}

impl DurabilityLevel {
    /// Parse from string (case-insensitive)
    pub fn from_str(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "fast" => Some(DurabilityLevel::Fast),
            "safe" => Some(DurabilityLevel::Safe),
            "max" => Some(DurabilityLevel::Max),
            _ => None,
        }
    }
    
    /// Convert to string
    pub fn as_str(&self) -> &'static str {
        match self {
            DurabilityLevel::Fast => "fast",
            DurabilityLevel::Safe => "safe",
            DurabilityLevel::Max => "max",
        }
    }
}

// Re-export all public types from on_demand
pub use on_demand::{
    // Storage engine
    OnDemandStorage,
    OnDemandHeader,
    OnDemandSchema,
    ColumnIndexEntry,
    // Data types
    ColumnType,
    ColumnValue,
    ColumnData,
    ColumnDef,
    FileSchema,
};

// Re-export backend types
pub use backend::{
    TableStorageBackend,
    TableMetadata,
    StorageManager,
    IncrementalStorageBackend,
    typed_column_to_column_data,
    column_data_to_typed_column,
    datatype_to_column_type,
    column_type_to_datatype,
};

// Re-export incremental storage types
pub use incremental::{
    IncrementalStorage,
    WalRecord,
    WalWriter,
    WalReader,
};

// Type alias for backward compatibility
pub type ColumnarStorage = OnDemandStorage;

