//! Table management module

mod catalog;
mod schema;

// Legacy row-based table (deprecated, kept for reference)
#[allow(dead_code)]
pub(crate) mod table;

// Current columnar table implementation
pub mod column_table;

pub use catalog::{TableCatalog, TableEntry};
pub use schema::Schema;
pub use column_table::{ColumnTable, ColumnSchema, TypedColumn, QueryColumnarResult};

