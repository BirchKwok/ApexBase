//! Table management module
//!
//! Provides columnar table storage with high-performance operations.

mod catalog;
mod schema;
pub mod column_table;

pub use catalog::{TableCatalog, TableEntry};
pub use schema::Schema;
pub use column_table::{ColumnTable, ColumnSchema, TypedColumn, QueryColumnarResult};

