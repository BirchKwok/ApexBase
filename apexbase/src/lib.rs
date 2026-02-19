//! ApexBase Core Storage Engine
//! 
//! A high-performance embedded database storage engine implemented in Rust.
//! Provides Python bindings via PyO3 for seamless integration.

pub mod storage;
pub mod table;
pub mod query;
pub mod data;
#[cfg(feature = "python")]
pub mod python;
pub mod fts;
pub mod txn;
pub mod scaling;
#[cfg(feature = "server")]
pub mod server;
#[cfg(feature = "flight")]
pub mod flight;

// Re-export main types
pub use storage::{ColumnarStorage, ColumnType, ColumnValue, FileSchema};
pub use table::TableCatalog;
pub use data::{DataType, Value, Row};
pub use query::{ApexExecutor, ApexResult};

#[cfg(feature = "python")]
use pyo3::prelude::*;

/// Python module entry point
#[cfg(feature = "python")]
#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<python::ApexStorage>()?;
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    #[cfg(feature = "server")]
    m.add_function(pyo3::wrap_pyfunction!(python::start_pg_server, m)?)?;
    #[cfg(feature = "flight")]
    m.add_function(pyo3::wrap_pyfunction!(python::start_flight_server, m)?)?;
    Ok(())
}

/// Storage engine error type
#[derive(Debug, thiserror::Error)]
pub enum ApexError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    
    #[error("Table not found: {0}")]
    TableNotFound(String),
    
    #[error("Table already exists: {0}")]
    TableExists(String),
    
    #[error("Column not found: {0}")]
    ColumnNotFound(String),
    
    #[error("Column already exists: {0}")]
    ColumnExists(String),
    
    #[error("Row not found: {0}")]
    RowNotFound(u64),
    
    #[error("Invalid data type: {0}")]
    InvalidDataType(String),
    
    #[error("Query parse error: {0}")]
    QueryParseError(String),
    
    #[error("Serialization error: {0}")]
    SerializationError(String),
    
    #[error("Checksum mismatch")]
    ChecksumMismatch,
    
    #[error("Invalid file format")]
    InvalidFileFormat,
    
    #[error("Version mismatch: expected {expected}, got {actual}")]
    VersionMismatch { expected: u32, actual: u32 },
    
    #[error("Cannot drop default table")]
    CannotDropDefaultTable,
    
    #[error("Cannot modify _id column")]
    CannotModifyIdColumn,
}

pub type Result<T> = std::result::Result<T, ApexError>;

