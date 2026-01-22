//! Query parsing and execution
//!
//! This module provides SQL:2023 compliant query parsing and execution.
//! V3Executor is the only execution engine (on-demand reading without ColumnTable)

mod expr_compiler;
mod filter;
mod sql_parser;
mod v3_executor;

pub use expr_compiler::sql_expr_to_filter;
pub use filter::{Filter, CompareOp, LikeMatcher, RegexpMatcher};
pub use sql_parser::{
    SqlParser, SqlStatement, SelectStatement, SelectColumn, SqlExpr, OrderByClause, AggregateFunc,
    FromItem, JoinClause, JoinType, UnionStatement,
};
pub use v3_executor::{V3Executor, V3Result};
