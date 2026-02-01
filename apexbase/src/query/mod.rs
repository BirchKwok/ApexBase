//! Query parsing and execution
//!
//! This module provides SQL:2023 compliant query parsing and execution.

mod expr_compiler;
mod filter;
mod sql_parser;
mod executor;
pub mod jit;
pub mod vectorized;
pub mod multi_column;
pub mod simd_take;

pub use expr_compiler::sql_expr_to_filter;
pub use filter::{Filter, CompareOp, LikeMatcher, RegexpMatcher};
pub use sql_parser::{
    SqlParser, SqlStatement, SelectStatement, SelectColumn, SqlExpr, OrderByClause, AggregateFunc,
    FromItem, JoinClause, JoinType, UnionStatement,
    // DDL types
    ColumnDef, AlterTableOp,
};
pub use executor::{ApexExecutor, ApexResult};
