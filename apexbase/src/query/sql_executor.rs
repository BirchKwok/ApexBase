//! SQL Executor - Converts parsed SQL AST to query operations
//!
//! Executes SQL statements against ColumnTable storage.
//! Uses IoEngine for all data read operations.

use crate::ApexError;
use crate::data::Value;
use crate::query::Filter;
use crate::query::sql_expr_to_filter;
use crate::query::sql_parser::{
    SqlStatement, SelectStatement, SelectColumn, SqlExpr, 
    BinaryOperator, UnaryOperator, OrderByClause, AggregateFunc, FromItem
};
use crate::table::column_table::{ColumnTable, TypedColumn};
use crate::io_engine::{IoEngine, StreamingFilterEvaluator};
use std::collections::HashMap;
use std::cmp::Ordering;
use std::hash::{Hash, Hasher};
use crate::query::{LikeMatcher, RegexpMatcher};
use arrow::array::{Array, BooleanArray, Float64Array, Int64Array, StringArray, UInt64Array};
use arrow::datatypes::DataType as ArrowDataType;
 

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
enum JoinKey {
    Bool(bool),
    I64(i64),
    U64(u64),
    F32(u32),
    F64(u64),
    String(String),
    Binary(Vec<u8>),
    Bytes(Vec<u8>),
}

#[inline]
fn join_key(v: &crate::data::Value) -> JoinKey {
    use crate::data::Value;
    match v {
        Value::Null => JoinKey::Bytes(Vec::new()),
        Value::Bool(b) => JoinKey::Bool(*b),
        Value::Int8(x) => JoinKey::I64(*x as i64),
        Value::Int16(x) => JoinKey::I64(*x as i64),
        Value::Int32(x) => JoinKey::I64(*x as i64),
        Value::Int64(x) => JoinKey::I64(*x),
        Value::UInt8(x) => JoinKey::U64(*x as u64),
        Value::UInt16(x) => JoinKey::U64(*x as u64),
        Value::UInt32(x) => JoinKey::U64(*x as u64),
        Value::UInt64(x) => JoinKey::U64(*x),
        Value::Float32(x) => JoinKey::F32(x.to_bits()),
        Value::Float64(x) => JoinKey::F64(x.to_bits()),
        Value::String(s) => JoinKey::String(s.clone()),
        Value::Binary(b) => JoinKey::Binary(b.clone()),
        Value::Json(j) => JoinKey::Bytes(j.to_string().into_bytes()),
        Value::Timestamp(t) => JoinKey::I64(*t),
        Value::Date(d) => JoinKey::I64(*d as i64),
        Value::Array(arr) => JoinKey::Bytes(crate::data::Value::Array(arr.clone()).to_bytes()),
    }
}

// StreamingFilterEvaluator is now provided by IoEngine

#[path = "sql_executor/result.rs"]
mod result;
#[path = "sql_executor/simple_pipeline.rs"]
mod simple_pipeline;
#[path = "sql_executor/union.rs"]
mod union_mod;
#[path = "sql_executor/with_tables.rs"]
mod with_tables;
#[path = "sql_executor/select_single.rs"]
mod select_single;

pub use result::SqlResult;

/// SQL Executor
pub struct SqlExecutor;

impl SqlExecutor {
    /// Execute a SQL statement against a table
    pub fn execute(sql: &str, table: &mut ColumnTable) -> Result<SqlResult, ApexError> {
        use crate::query::sql_parser::SqlParser;
        
        let stmt = SqlParser::parse(sql)?;

        Self::execute_parsed(stmt, table)
    }

    pub fn execute_parsed(stmt: SqlStatement, table: &mut ColumnTable) -> Result<SqlResult, ApexError> {
        match stmt {
            SqlStatement::Select(select) => Self::execute_select(select, table),
            SqlStatement::Union(union) => Self::execute_union(union, table),
            SqlStatement::CreateView { .. } | SqlStatement::DropView { .. } => Err(ApexError::QueryParseError(
                "CREATE/DROP VIEW are only supported in multi-statement execution".to_string(),
            )),
        }
    }

    pub fn execute_with_tables(
        sql: &str,
        tables: &mut HashMap<String, ColumnTable>,
        default_table: &str,
    ) -> Result<SqlResult, ApexError> {
        use crate::query::sql_parser::SqlParser;

        let stmt = SqlParser::parse(sql)?;

        Self::execute_with_tables_parsed(stmt, tables, default_table)
    }

    pub fn execute_with_tables_parsed(
        stmt: SqlStatement,
        tables: &mut HashMap<String, ColumnTable>,
        default_table: &str,
    ) -> Result<SqlResult, ApexError> {
        match stmt {
            SqlStatement::Select(select) => {
                if !select.joins.is_empty() {
                    let plan = crate::query::engine::PlanBuilder::build_join_tables_plan(&select)?;
                    crate::query::engine::PlanExecutorTables::execute_tables_plan(tables, default_table, &plan)
                } else {
                    Self::execute_select_with_tables(select, tables, default_table)
                }
            }
            SqlStatement::Union(union) => Self::execute_union_with_tables(union, tables, default_table),
            SqlStatement::CreateView { .. } | SqlStatement::DropView { .. } => Err(ApexError::QueryParseError(
                "CREATE/DROP VIEW are only supported in multi-statement execution".to_string(),
            )),
        }
    }

}
