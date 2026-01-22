//! Table management module
//!
//! Provides table catalog and Arrow-based column storage.

mod catalog;
mod schema;
pub mod column_table;
pub mod arrow_column;

pub use catalog::{TableCatalog, TableEntry};
pub use schema::Schema;
pub use column_table::{BitVec, TypedColumn, ColumnSchema};
pub use arrow_column::{ArrowTypedColumn, ArrowStringColumn};

