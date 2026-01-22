//! Storage module - ApexV3 On-Demand Columnar Storage
//!
//! This module provides the core columnar storage format for ApexBase.
//! The V3 format supports on-demand column/row reading without loading
//! the entire dataset into memory.

pub mod on_demand;
pub mod backend;

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
    typed_column_to_column_data,
    column_data_to_typed_column,
    datatype_to_column_type,
    column_type_to_datatype,
};

// Type alias for backward compatibility
pub type ColumnarStorage = OnDemandStorage;

