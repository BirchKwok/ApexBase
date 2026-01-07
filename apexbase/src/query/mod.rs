//! Query parsing and execution
//!
//! This module provides SQL:2023 compliant query parsing and execution.

mod executor;
mod engine;
mod expr_compiler;
mod filter;
mod sql_parser;
mod sql_executor;

pub use executor::QueryExecutor;
pub use engine::plan_query_indices;
pub use expr_compiler::sql_expr_to_filter;
pub use filter::{Filter, CompareOp, LikeMatcher, RegexpMatcher};
pub use sql_parser::{
    SqlParser, SqlStatement, SelectStatement, SelectColumn, SqlExpr, OrderByClause, AggregateFunc,
    FromItem, JoinClause, JoinType, UnionStatement,
};
pub use sql_executor::{SqlExecutor, SqlResult};
