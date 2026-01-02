//! Query parsing and execution
//!
//! This module provides SQL:2023 compliant query parsing and execution.

mod executor;
mod filter;
mod sql_parser;
mod sql_executor;

pub use executor::QueryExecutor;
pub use filter::{Filter, CompareOp, LikeMatcher};
pub use sql_parser::{SqlParser, SqlStatement, SelectStatement, SelectColumn, SqlExpr, OrderByClause, AggregateFunc};
pub use sql_executor::{SqlExecutor, SqlResult};

