//! Query parsing and execution
//!
//! This module provides SQL:2023 compliant query parsing and execution.

mod expr_compiler;
mod filter;
pub(crate) mod sql_parser;
pub(crate) mod executor;
pub mod jit;
pub mod vectorized;
pub mod multi_column;
pub mod simd_take;
pub mod planner;

pub use expr_compiler::sql_expr_to_filter;
pub use filter::{Filter, CompareOp, LikeMatcher, RegexpMatcher};
pub use sql_parser::{
    SqlParser, SqlStatement, SelectStatement, SelectColumn, SqlExpr, OrderByClause, AggregateFunc,
    FromItem, JoinClause, JoinType, UnionStatement, SetOpType,
    // DDL types
    ColumnDef, AlterTableOp,
};
pub use executor::{ApexExecutor, ApexResult, get_cached_backend_pub};
