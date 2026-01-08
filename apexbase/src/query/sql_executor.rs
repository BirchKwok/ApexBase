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

/// SQL Execution Result
#[derive(Debug, Clone)]
pub struct SqlResult {
    /// Column names in result
    pub columns: Vec<String>,
    /// Row data (each row is a vector of values)
    pub rows: Vec<Vec<Value>>,
    /// Number of rows affected (for non-SELECT)
    pub rows_affected: usize,
    /// Pre-built Arrow batch for fast path (bypasses row conversion)
    pub arrow_batch: Option<arrow::record_batch::RecordBatch>,
}

impl SqlResult {
    pub fn new(columns: Vec<String>, rows: Vec<Vec<Value>>) -> Self {
        let rows_affected = rows.len();
        Self { columns, rows, rows_affected, arrow_batch: None }
    }

    pub fn empty() -> Self {
        Self { columns: Vec::new(), rows: Vec::new(), rows_affected: 0, arrow_batch: None }
    }
    
    /// Create SqlResult with pre-built Arrow batch (fast path)
    pub fn with_arrow_batch(columns: Vec<String>, batch: arrow::record_batch::RecordBatch) -> Self {
        let rows_affected = batch.num_rows();
        Self { columns, rows: Vec::new(), rows_affected, arrow_batch: Some(batch) }
    }
    
    /// Convert SqlResult to Arrow RecordBatch for fast Python transfer
    pub fn to_record_batch(&self) -> Result<arrow::record_batch::RecordBatch, ApexError> {
        use arrow::array::{ArrayRef, Float64Array, Int64Array, StringBuilder, BooleanArray};
        use arrow::datatypes::{DataType as ArrowDataType, Field, Schema};
        use std::sync::Arc;
        
        // Fast path: return pre-built Arrow batch if available
        if let Some(ref batch) = self.arrow_batch {
            return Ok(batch.clone());
        }
        
        if self.rows.is_empty() {
            use arrow::array::new_null_array;

            let fields: Vec<Field> = self.columns.iter()
                .map(|name| Field::new(name, ArrowDataType::Utf8, true))
                .collect();
            let schema = Arc::new(Schema::new(fields.clone()));
            let arrays: Vec<ArrayRef> = fields
                .iter()
                .map(|f| new_null_array(f.data_type(), 0))
                .collect();

            return arrow::record_batch::RecordBatch::try_new(schema, arrays)
                .map_err(|e| ApexError::SerializationError(e.to_string()));
        }
        
        // Infer types from first row
        let first_row = &self.rows[0];
        let mut fields = Vec::with_capacity(self.columns.len());
        let mut arrays: Vec<ArrayRef> = Vec::with_capacity(self.columns.len());
        
        for (col_idx, col_name) in self.columns.iter().enumerate() {
            let sample_val = first_row.get(col_idx);
            let (arrow_type, array) = match sample_val {
                Some(Value::Int64(_)) | Some(Value::Int32(_)) | Some(Value::Int16(_)) | Some(Value::Int8(_)) => {
                    let values: Vec<Option<i64>> = self.rows.iter()
                        .map(|row| row.get(col_idx).and_then(|v| v.as_i64()))
                        .collect();
                    (ArrowDataType::Int64, Arc::new(Int64Array::from(values)) as ArrayRef)
                }
                Some(Value::Float64(_)) | Some(Value::Float32(_)) => {
                    let values: Vec<Option<f64>> = self.rows.iter()
                        .map(|row| row.get(col_idx).and_then(|v| v.as_f64()))
                        .collect();
                    (ArrowDataType::Float64, Arc::new(Float64Array::from(values)) as ArrayRef)
                }
                Some(Value::Bool(_)) => {
                    let values: Vec<Option<bool>> = self.rows.iter()
                        .map(|row| row.get(col_idx).and_then(|v| v.as_bool()))
                        .collect();
                    (ArrowDataType::Boolean, Arc::new(BooleanArray::from(values)) as ArrayRef)
                }
                _ => {
                    // String or mixed - convert to string
                    let mut builder = StringBuilder::with_capacity(self.rows.len(), self.rows.len() * 32);
                    for row in &self.rows {
                        match row.get(col_idx) {
                            Some(v) if !v.is_null() => builder.append_value(v.to_string_value()),
                            _ => builder.append_null(),
                        }
                    }
                    (ArrowDataType::Utf8, Arc::new(builder.finish()) as ArrayRef)
                }
            };
            fields.push(Field::new(col_name, arrow_type, true));
            arrays.push(array);
        }
        
        let schema = Arc::new(Schema::new(fields));
        arrow::record_batch::RecordBatch::try_new(schema, arrays)
            .map_err(|e| ApexError::SerializationError(e.to_string()))
    }
}

/// SQL Executor
pub struct SqlExecutor;

// ============ Unified Single-Table Pipeline ============
//
// This is a minimal, low-risk step toward a unified execution layer.
// We only enable it for simple SELECT queries (no JOIN/GROUP BY/aggregates/window/expressions).
// All other queries keep using the existing highly-optimized legacy paths.
mod simple_pipeline {
    use super::*;
    use crate::query::engine::{PlanBuilder, PlanExecutor};

    pub(super) fn is_eligible(stmt: &SelectStatement) -> bool {
        crate::query::engine::is_simple_select_eligible(stmt)
    }

    pub(super) fn execute(stmt: &SelectStatement, table: &ColumnTable) -> Result<SqlResult, ApexError> {
        let plan = PlanBuilder::build_simple_select_plan(stmt, table)?;
        PlanExecutor::execute_sql_result(table, &plan)
    }
}

#[allow(dead_code)]
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

    fn execute_union(union: crate::query::UnionStatement, table: &mut ColumnTable) -> Result<SqlResult, ApexError> {
        if table.has_pending_writes() {
            table.flush_write_buffer();
        }

        fn exec_select_legacy(stmt: SelectStatement, table: &mut ColumnTable) -> Result<SqlResult, ApexError> {
            use crate::data::Value;
            if table.has_pending_writes() {
                table.flush_write_buffer();
            }

            if !stmt.joins.is_empty() {
                return Err(ApexError::QueryParseError(
                    "JOIN requires multi-table execution".to_string(),
                ));
            }
            if !stmt.group_by.is_empty() {
                return Err(ApexError::QueryParseError(
                    "GROUP BY in UNION branch is not supported yet".to_string(),
                ));
            }
            if stmt.having.is_some() {
                return Err(ApexError::QueryParseError(
                    "HAVING in UNION branch is not supported yet".to_string(),
                ));
            }
            if stmt.distinct {
                return Err(ApexError::QueryParseError(
                    "DISTINCT in UNION branch is not supported yet".to_string(),
                ));
            }
            if !stmt.order_by.is_empty() || stmt.limit.is_some() || stmt.offset.is_some() {
                return Err(ApexError::QueryParseError(
                    "ORDER BY/LIMIT/OFFSET must be applied to UNION result".to_string(),
                ));
            }

            let matching_indices: Vec<usize> = if let Some(ref where_expr) = stmt.where_clause {
                SqlExecutor::evaluate_where(where_expr, table)?
            } else {
                let deleted = table.deleted_ref();
                let row_count = table.get_row_count();
                (0..row_count).filter(|&i| !deleted.get(i)).collect()
            };

            let (result_columns, column_indices) = SqlExecutor::resolve_columns(&stmt.columns, table)?;
            let mut rows: Vec<Vec<Value>> = Vec::with_capacity(matching_indices.len().min(10000));
            for row_idx in matching_indices.iter() {
                let mut row_values: Vec<Value> = Vec::with_capacity(result_columns.len());
                for (col_name, col_idx) in column_indices.iter() {
                    if col_name == "_id" {
                        row_values.push(Value::Int64(*row_idx as i64));
                    } else if let Some(idx) = col_idx {
                        row_values.push(table.columns_ref()[*idx].get(*row_idx).unwrap_or(Value::Null));
                    } else {
                        row_values.push(Value::Null);
                    }
                }
                rows.push(row_values);
            }
            Ok(SqlResult::new(result_columns, rows))
        }

        fn exec_one(stmt: crate::query::SqlStatement, table: &mut ColumnTable) -> Result<SqlResult, ApexError> {
            match stmt {
                crate::query::SqlStatement::Select(sel) => exec_select_legacy(sel, table),
                crate::query::SqlStatement::Union(u) => SqlExecutor::execute_union(u, table),
                crate::query::SqlStatement::CreateView { .. }
                | crate::query::SqlStatement::DropView { .. } => Err(ApexError::QueryParseError(
                    "CREATE/DROP VIEW are only supported in multi-statement execution".to_string(),
                )),
            }
        }

        let left = exec_one(*union.left, table)?;
        let right = exec_one(*union.right, table)?;
        SqlExecutor::merge_union_results(left, right, union.all, &union.order_by, union.limit, union.offset)
    }

    fn execute_union_with_tables(
        union: crate::query::UnionStatement,
        tables: &mut HashMap<String, ColumnTable>,
        default_table: &str,
    ) -> Result<SqlResult, ApexError> {
        fn exec_select_legacy(stmt: SelectStatement, table: &mut ColumnTable) -> Result<SqlResult, ApexError> {
            use crate::data::Value;
            if table.has_pending_writes() {
                table.flush_write_buffer();
            }

            if !stmt.joins.is_empty() {
                return Err(ApexError::QueryParseError(
                    "JOIN requires multi-table execution".to_string(),
                ));
            }
            if !stmt.group_by.is_empty() {
                return Err(ApexError::QueryParseError(
                    "GROUP BY in UNION branch is not supported yet".to_string(),
                ));
            }
            if stmt.having.is_some() {
                return Err(ApexError::QueryParseError(
                    "HAVING in UNION branch is not supported yet".to_string(),
                ));
            }
            if stmt.distinct {
                return Err(ApexError::QueryParseError(
                    "DISTINCT in UNION branch is not supported yet".to_string(),
                ));
            }
            if !stmt.order_by.is_empty() || stmt.limit.is_some() || stmt.offset.is_some() {
                return Err(ApexError::QueryParseError(
                    "ORDER BY/LIMIT/OFFSET must be applied to UNION result".to_string(),
                ));
            }

            let matching_indices: Vec<usize> = if let Some(ref where_expr) = stmt.where_clause {
                SqlExecutor::evaluate_where(where_expr, table)?
            } else {
                let deleted = table.deleted_ref();
                let row_count = table.get_row_count();
                (0..row_count).filter(|&i| !deleted.get(i)).collect()
            };

            let (result_columns, column_indices) = SqlExecutor::resolve_columns(&stmt.columns, table)?;
            let mut rows: Vec<Vec<Value>> = Vec::with_capacity(matching_indices.len().min(10000));
            for row_idx in matching_indices.iter() {
                let mut row_values: Vec<Value> = Vec::with_capacity(result_columns.len());
                for (col_name, col_idx) in column_indices.iter() {
                    if col_name == "_id" {
                        row_values.push(Value::Int64(*row_idx as i64));
                    } else if let Some(idx) = col_idx {
                        row_values.push(table.columns_ref()[*idx].get(*row_idx).unwrap_or(Value::Null));
                    } else {
                        row_values.push(Value::Null);
                    }
                }
                rows.push(row_values);
            }
            Ok(SqlResult::new(result_columns, rows))
        }

        fn exec_one(
            stmt: crate::query::SqlStatement,
            tables: &mut HashMap<String, ColumnTable>,
            default_table: &str,
        ) -> Result<SqlResult, ApexError> {
            match stmt {
                crate::query::SqlStatement::Select(sel) => {
                    if !sel.joins.is_empty() {
                        return Err(ApexError::QueryParseError(
                            "UNION over JOIN queries is not supported yet".to_string(),
                        ));
                    }
                    let target_table = sel
                        .from
                        .as_ref()
                        .map(|f| match f {
                            FromItem::Table { table, .. } => table.clone(),
                            FromItem::Subquery { alias, .. } => alias.clone(),
                        })
                        .unwrap_or_else(|| default_table.to_string());
                    let table = tables.get_mut(&target_table).ok_or_else(|| {
                        ApexError::QueryParseError(format!("Table '{}' not found.", target_table))
                    })?;
                    exec_select_legacy(sel, table)
                }
                crate::query::SqlStatement::Union(u) => SqlExecutor::execute_union_with_tables(u, tables, default_table),
                crate::query::SqlStatement::CreateView { .. }
                | crate::query::SqlStatement::DropView { .. } => Err(ApexError::QueryParseError(
                    "CREATE/DROP VIEW are only supported in multi-statement execution".to_string(),
                )),
            }
        }

        let left = exec_one(*union.left, tables, default_table)?;
        let right = exec_one(*union.right, tables, default_table)?;
        Self::merge_union_results(left, right, union.all, &union.order_by, union.limit, union.offset)
    }

    pub(crate) fn merge_union_results(
        left: SqlResult,
        right: SqlResult,
        all: bool,
        order_by: &[OrderByClause],
        limit: Option<usize>,
        offset: Option<usize>,
    ) -> Result<SqlResult, ApexError> {
        fn materialize_arrow_rows(batch: &arrow::record_batch::RecordBatch) -> Vec<Vec<Value>> {
            use arrow::array::Array;
            use arrow::datatypes::DataType as ArrowDataType;

            let num_rows = batch.num_rows();
            let num_cols = batch.num_columns();
            let mut rows: Vec<Vec<Value>> = (0..num_rows).map(|_| Vec::with_capacity(num_cols)).collect();

            for ci in 0..num_cols {
                let col = batch.column(ci);
                match col.data_type() {
                    ArrowDataType::Int64 => {
                        let a = col.as_any().downcast_ref::<Int64Array>().unwrap();
                        for ri in 0..num_rows {
                            rows[ri].push(if a.is_null(ri) { Value::Null } else { Value::Int64(a.value(ri)) });
                        }
                    }
                    ArrowDataType::UInt64 => {
                        let a = col.as_any().downcast_ref::<UInt64Array>().unwrap();
                        for ri in 0..num_rows {
                            rows[ri].push(if a.is_null(ri) { Value::Null } else { Value::UInt64(a.value(ri)) });
                        }
                    }
                    ArrowDataType::Float64 => {
                        let a = col.as_any().downcast_ref::<Float64Array>().unwrap();
                        for ri in 0..num_rows {
                            rows[ri].push(if a.is_null(ri) { Value::Null } else { Value::Float64(a.value(ri)) });
                        }
                    }
                    ArrowDataType::Boolean => {
                        let a = col.as_any().downcast_ref::<BooleanArray>().unwrap();
                        for ri in 0..num_rows {
                            rows[ri].push(if a.is_null(ri) { Value::Null } else { Value::Bool(a.value(ri)) });
                        }
                    }
                    ArrowDataType::Utf8 => {
                        let a = col.as_any().downcast_ref::<StringArray>().unwrap();
                        for ri in 0..num_rows {
                            rows[ri].push(if a.is_null(ri) { Value::Null } else { Value::String(a.value(ri).to_string()) });
                        }
                    }
                    ArrowDataType::LargeUtf8 => {
                        let a = col.as_any().downcast_ref::<arrow::array::LargeStringArray>().unwrap();
                        for ri in 0..num_rows {
                            rows[ri].push(if a.is_null(ri) { Value::Null } else { Value::String(a.value(ri).to_string()) });
                        }
                    }
                    _ => {
                        for ri in 0..num_rows {
                            rows[ri].push(if col.is_null(ri) { Value::Null } else { Value::String(format!("{:?}", col)) });
                        }
                    }
                }
            }

            rows
        }

        fn ensure_rows(mut r: SqlResult) -> SqlResult {
            if r.rows.is_empty() {
                if let Some(batch) = r.arrow_batch.as_ref() {
                    r.rows = materialize_arrow_rows(batch);
                }
            }
            r
        }

        let left = ensure_rows(left);
        let right = ensure_rows(right);

        if left.columns != right.columns {
            return Err(ApexError::QueryParseError(
                "UNION requires both sides to have the same columns".to_string(),
            ));
        }

        let mut rows: Vec<Vec<Value>> = Vec::with_capacity(left.rows.len() + right.rows.len());
        rows.extend(left.rows);
        rows.extend(right.rows);

        if !all {
            let mut seen: std::collections::HashSet<String> = std::collections::HashSet::new();
            rows.retain(|r| {
                let k = r.iter().map(|v| v.to_string_value()).collect::<Vec<_>>().join("\u{1f}");
                seen.insert(k)
            });
        }

        if !order_by.is_empty() {
            // Map ORDER BY column name -> result column index
            let mut key_idx: Vec<(usize, bool, Option<bool>)> = Vec::new();
            for ob in order_by {
                let idx = left
                    .columns
                    .iter()
                    .position(|c| c == &ob.column)
                    .ok_or_else(|| {
                        ApexError::QueryParseError(format!(
                            "ORDER BY column '{}' must appear in UNION select list",
                            ob.column
                        ))
                    })?;
                key_idx.push((idx, ob.descending, ob.nulls_first));
            }

            rows.sort_by(|a, b| {
                for (idx, desc, nulls_first) in &key_idx {
                    let av = a.get(*idx);
                    let bv = b.get(*idx);
                    let cmp = SqlExecutor::compare_values(av, bv, *nulls_first);
                    let cmp = if *desc { cmp.reverse() } else { cmp };
                    if cmp != Ordering::Equal {
                        return cmp;
                    }
                }
                Ordering::Equal
            });
        }

        let off = offset.unwrap_or(0);
        let lim = limit.unwrap_or(usize::MAX);
        let rows = rows.into_iter().skip(off).take(lim).collect::<Vec<_>>();

        Ok(SqlResult::new(left.columns, rows))
    }

    pub(crate) fn execute_select_with_tables(
        stmt: SelectStatement,
        tables: &mut HashMap<String, ColumnTable>,
        default_table: &str,
    ) -> Result<SqlResult, ApexError> {
        fn strip_prefix(col: &str, alias: &str) -> String {
            if let Some((a, c)) = col.split_once('.') {
                if a == alias {
                    return c.to_string();
                }
            }
            col.to_string()
        }

        fn rewrite_expr_strip_alias(expr: &SqlExpr, alias: &str) -> SqlExpr {
            match expr {
                SqlExpr::Paren(inner) => SqlExpr::Paren(Box::new(rewrite_expr_strip_alias(inner, alias))),
                SqlExpr::Column(c) => SqlExpr::Column(strip_prefix(c, alias)),
                SqlExpr::BinaryOp { left, op, right } => SqlExpr::BinaryOp {
                    left: Box::new(rewrite_expr_strip_alias(left, alias)),
                    op: op.clone(),
                    right: Box::new(rewrite_expr_strip_alias(right, alias)),
                },
                SqlExpr::UnaryOp { op, expr } => SqlExpr::UnaryOp {
                    op: op.clone(),
                    expr: Box::new(rewrite_expr_strip_alias(expr, alias)),
                },
                SqlExpr::Like { column, pattern, negated } => SqlExpr::Like {
                    column: strip_prefix(column, alias),
                    pattern: pattern.clone(),
                    negated: *negated,
                },
                SqlExpr::Regexp { column, pattern, negated } => SqlExpr::Regexp {
                    column: strip_prefix(column, alias),
                    pattern: pattern.clone(),
                    negated: *negated,
                },
                SqlExpr::In { column, values, negated } => SqlExpr::In {
                    column: strip_prefix(column, alias),
                    values: values.clone(),
                    negated: *negated,
                },
                SqlExpr::Between { column, low, high, negated } => SqlExpr::Between {
                    column: strip_prefix(column, alias),
                    low: Box::new(rewrite_expr_strip_alias(low, alias)),
                    high: Box::new(rewrite_expr_strip_alias(high, alias)),
                    negated: *negated,
                },
                SqlExpr::IsNull { column, negated } => SqlExpr::IsNull {
                    column: strip_prefix(column, alias),
                    negated: *negated,
                },
                SqlExpr::Function { name, args } => SqlExpr::Function {
                    name: name.clone(),
                    args: args.iter().map(|a| rewrite_expr_strip_alias(a, alias)).collect(),
                },
                _ => expr.clone(),
            }
        }

        fn materialize_sql_result_to_table(alias: &str, res: SqlResult) -> Result<ColumnTable, ApexError> {
            use std::collections::HashMap;
            let mut t = ColumnTable::new(0, alias);

            if res.columns.is_empty() {
                return Ok(t);
            }

            // If the result was produced via Arrow fast path, rows may be empty.
            // Materialize from the Arrow batch when present.
            if res.rows.is_empty() {
                if let Some(batch) = res.arrow_batch {
                    use arrow::array::Array;
                    use arrow::datatypes::DataType;

                    let mut colmap: HashMap<String, Vec<Value>> = HashMap::new();
                    for (i, name) in res.columns.iter().enumerate() {
                        let arr = batch.column(i);
                        let mut values: Vec<Value> = Vec::with_capacity(arr.len());
                        match arr.data_type() {
                            DataType::Int64 => {
                                let a = arr.as_any().downcast_ref::<arrow::array::Int64Array>().unwrap();
                                for r in 0..a.len() {
                                    if a.is_null(r) {
                                        values.push(Value::Null);
                                    } else {
                                        values.push(Value::Int64(a.value(r)));
                                    }
                                }
                            }
                            DataType::Float64 => {
                                let a = arr.as_any().downcast_ref::<arrow::array::Float64Array>().unwrap();
                                for r in 0..a.len() {
                                    if a.is_null(r) {
                                        values.push(Value::Null);
                                    } else {
                                        values.push(Value::Float64(a.value(r)));
                                    }
                                }
                            }
                            DataType::Utf8 => {
                                let a = arr.as_any().downcast_ref::<arrow::array::StringArray>().unwrap();
                                for r in 0..a.len() {
                                    if a.is_null(r) {
                                        values.push(Value::Null);
                                    } else {
                                        values.push(Value::String(a.value(r).to_string()));
                                    }
                                }
                            }
                            DataType::Boolean => {
                                let a = arr.as_any().downcast_ref::<arrow::array::BooleanArray>().unwrap();
                                for r in 0..a.len() {
                                    if a.is_null(r) {
                                        values.push(Value::Null);
                                    } else {
                                        values.push(Value::Bool(a.value(r)));
                                    }
                                }
                            }
                            _ => {
                                // Fallback: stringify
                                for r in 0..arr.len() {
                                    if arr.is_null(r) {
                                        values.push(Value::Null);
                                    } else {
                                        values.push(Value::String(format!("{:?}", arr)));
                                    }
                                }
                            }
                        }
                        colmap.insert(name.clone(), values);
                    }

                    t.insert_columns(colmap)
                        .map_err(|e| ApexError::QueryParseError(format!("Failed to materialize derived table: {:?}", e)))?;
                    t.flush_write_buffer();
                    return Ok(t);
                }
            }

            let mut colmap: HashMap<String, Vec<Value>> = HashMap::new();
            for c in &res.columns {
                colmap.insert(c.clone(), Vec::with_capacity(res.rows.len()));
            }
            for row in res.rows {
                for (i, c) in res.columns.iter().enumerate() {
                    let v = row.get(i).cloned().unwrap_or(Value::Null);
                    colmap.get_mut(c).unwrap().push(v);
                }
            }
            t.insert_columns(colmap)
                .map_err(|e| ApexError::QueryParseError(format!("Failed to materialize derived table: {:?}", e)))?;
            t.flush_write_buffer();
            Ok(t)
        }

        fn collect_single_column_values(res: SqlResult) -> Result<(Vec<Value>, bool), ApexError> {
            if res.columns.len() != 1 {
                return Err(ApexError::QueryParseError(
                    "IN (subquery) requires subquery to return exactly 1 column".to_string(),
                ));
            }

            // rows path
            if !res.rows.is_empty() {
                let mut out: Vec<Value> = Vec::with_capacity(res.rows.len());
                let mut has_null = false;
                for r in res.rows {
                    let v = r.get(0).cloned().unwrap_or(Value::Null);
                    has_null |= v.is_null();
                    out.push(v);
                }
                return Ok((out, has_null));
            }

            // arrow fast path
            if let Some(batch) = res.arrow_batch {
                use arrow::array::Array;
                use arrow::datatypes::DataType;

                let arr = batch.column(0);
                let mut out: Vec<Value> = Vec::with_capacity(arr.len());
                let mut has_null = false;
                match arr.data_type() {
                    DataType::Int64 => {
                        let a = arr.as_any().downcast_ref::<arrow::array::Int64Array>().unwrap();
                        for i in 0..a.len() {
                            if a.is_null(i) {
                                has_null = true;
                                out.push(Value::Null);
                            } else {
                                out.push(Value::Int64(a.value(i)));
                            }
                        }
                    }
                    DataType::Float64 => {
                        let a = arr.as_any().downcast_ref::<arrow::array::Float64Array>().unwrap();
                        for i in 0..a.len() {
                            if a.is_null(i) {
                                has_null = true;
                                out.push(Value::Null);
                            } else {
                                out.push(Value::Float64(a.value(i)));
                            }
                        }
                    }
                    DataType::Utf8 => {
                        let a = arr.as_any().downcast_ref::<arrow::array::StringArray>().unwrap();
                        for i in 0..a.len() {
                            if a.is_null(i) {
                                has_null = true;
                                out.push(Value::Null);
                            } else {
                                out.push(Value::String(a.value(i).to_string()));
                            }
                        }
                    }
                    DataType::Boolean => {
                        let a = arr.as_any().downcast_ref::<arrow::array::BooleanArray>().unwrap();
                        for i in 0..a.len() {
                            if a.is_null(i) {
                                has_null = true;
                                out.push(Value::Null);
                            } else {
                                out.push(Value::Bool(a.value(i)));
                            }
                        }
                    }
                    _ => {
                        for i in 0..arr.len() {
                            if arr.is_null(i) {
                                has_null = true;
                                out.push(Value::Null);
                            } else {
                                out.push(Value::String(format!("{:?}", arr)));
                            }
                        }
                    }
                }
                return Ok((out, has_null));
            }

            Ok((Vec::new(), false))
        }

        fn rewrite_in_subquery_in_expr(
            expr: &SqlExpr,
            tables: &mut HashMap<String, ColumnTable>,
            default_table: &str,
        ) -> Result<SqlExpr, ApexError> {
            fn subquery_has_outer_ref(sub: &SelectStatement) -> bool {
                // Determine subquery's own qualifier (alias or table name).
                let (inner_table, inner_alias) = match sub.from.as_ref() {
                    Some(FromItem::Table { table, alias }) => {
                        (table.as_str(), alias.as_deref().unwrap_or(table.as_str()))
                    }
                    _ => ("", ""),
                };

                fn expr_has_outer_ref(expr: &SqlExpr, inner_table: &str, inner_alias: &str) -> bool {
                    match expr {
                        SqlExpr::Column(c) => {
                            if let Some((a, _)) = c.split_once('.') {
                                // Any qualifier not matching the subquery itself is treated as outer ref.
                                !(a == inner_alias || a == inner_table)
                            } else {
                                false
                            }
                        }
                        SqlExpr::BinaryOp { left, right, .. } => {
                            expr_has_outer_ref(left, inner_table, inner_alias)
                                || expr_has_outer_ref(right, inner_table, inner_alias)
                        }
                        SqlExpr::UnaryOp { expr, .. } => expr_has_outer_ref(expr, inner_table, inner_alias),
                        SqlExpr::Paren(inner) => expr_has_outer_ref(inner, inner_table, inner_alias),
                        SqlExpr::Between { low, high, .. } => {
                            expr_has_outer_ref(low, inner_table, inner_alias)
                                || expr_has_outer_ref(high, inner_table, inner_alias)
                        }
                        SqlExpr::Function { args, .. } => {
                            args.iter().any(|a| expr_has_outer_ref(a, inner_table, inner_alias))
                        }
                        SqlExpr::Case { when_then, else_expr } => {
                            when_then.iter().any(|(c, v)| {
                                expr_has_outer_ref(c, inner_table, inner_alias)
                                    || expr_has_outer_ref(v, inner_table, inner_alias)
                            }) || else_expr
                                .as_ref()
                                .is_some_and(|e| expr_has_outer_ref(e, inner_table, inner_alias))
                        }
                        SqlExpr::InSubquery { .. }
                        | SqlExpr::ExistsSubquery { .. }
                        | SqlExpr::ScalarSubquery { .. } => false,
                        SqlExpr::Like { .. }
                        | SqlExpr::Regexp { .. }
                        | SqlExpr::In { .. }
                        | SqlExpr::IsNull { .. }
                        | SqlExpr::Literal(_) => false,
                    }
                }

                if let Some(w) = sub.where_clause.as_ref() {
                    return expr_has_outer_ref(w, inner_table, inner_alias);
                }
                false
            }

            match expr {
                SqlExpr::BinaryOp { left, op, right } => Ok(SqlExpr::BinaryOp {
                    left: Box::new(rewrite_in_subquery_in_expr(left, tables, default_table)?),
                    op: op.clone(),
                    right: Box::new(rewrite_in_subquery_in_expr(right, tables, default_table)?),
                }),
                SqlExpr::UnaryOp { op, expr } => Ok(SqlExpr::UnaryOp {
                    op: op.clone(),
                    expr: Box::new(rewrite_in_subquery_in_expr(expr, tables, default_table)?),
                }),
                SqlExpr::Paren(inner) => Ok(SqlExpr::Paren(Box::new(rewrite_in_subquery_in_expr(
                    inner,
                    tables,
                    default_table,
                )?))),
                SqlExpr::Between { column, low, high, negated } => Ok(SqlExpr::Between {
                    column: column.clone(),
                    low: Box::new(rewrite_in_subquery_in_expr(low, tables, default_table)?),
                    high: Box::new(rewrite_in_subquery_in_expr(high, tables, default_table)?),
                    negated: *negated,
                }),
                SqlExpr::Function { name, args } => Ok(SqlExpr::Function {
                    name: name.clone(),
                    args: args
                        .iter()
                        .map(|a| rewrite_in_subquery_in_expr(a, tables, default_table))
                        .collect::<Result<Vec<_>, _>>()?,
                }),
                SqlExpr::InSubquery { column, stmt, negated } => {
                    // Correlated subquery: do NOT pre-execute here.
                    // Keep it for row-wise evaluation in the correlated execution path.
                    if subquery_has_outer_ref(stmt) {
                        return Ok(expr.clone());
                    }

                    if !stmt.joins.is_empty() {
                        return Err(ApexError::QueryParseError(
                            "IN (subquery) with JOIN is not supported yet".to_string(),
                        ));
                    }
                    let sub_res = SqlExecutor::execute_select_with_tables((**stmt).clone(), tables, default_table)?;
                    let (values, has_null) = collect_single_column_values(sub_res)?;

                    // Conservative NULL semantics for NOT IN: if subquery yields any NULL,
                    // the result is UNKNOWN for all non-matching rows -> filter out.
                    if *negated && has_null {
                        return Ok(SqlExpr::Literal(Value::Bool(false)));
                    }

                    Ok(SqlExpr::In {
                        column: column.clone(),
                        values,
                        negated: *negated,
                    })
                }
                _ => Ok(expr.clone()),
            }
        }

        // We will rewrite/normalize the statement in-place.
        let mut stmt = stmt;

        // Rewrite IN (subquery) in WHERE into IN (list) so we can reuse Filter::In
        if let Some(ref w) = stmt.where_clause {
            let rewritten = rewrite_in_subquery_in_expr(w, tables, default_table)?;
            stmt.where_clause = Some(rewritten);
        }

        // Derived table in FROM: execute subquery first, materialize, then execute outer query
        if let Some(FromItem::Subquery { stmt: sub, alias }) = stmt.from.clone() {
            if stmt.joins.is_empty()
                && stmt.group_by.is_empty()
                && stmt.order_by.is_empty()
                && stmt.limit.is_none()
                && stmt.offset.is_none()
                && stmt.columns.len() == 1
            {
                use crate::query::sql_parser::{SelectColumn, SqlExpr, BinaryOperator, AggregateFunc};

                let is_count_star = matches!(
                    stmt.columns[0],
                    SelectColumn::Aggregate {
                        func: AggregateFunc::Count,
                        column: None,
                        distinct: false,
                        ..
                    }
                );

                // Outer WHERE: t.<alias_of_sum> >= <literal>
                fn parse_sum_ge_threshold(where_expr: &SqlExpr, alias: &str) -> Option<(String, f64)> {
                    match where_expr {
                        SqlExpr::BinaryOp { left, op: BinaryOperator::Ge, right } => {
                            let col = match left.as_ref() {
                                SqlExpr::Column(c) => c,
                                _ => return None,
                            };
                            let lit = match right.as_ref() {
                                SqlExpr::Literal(v) => v.as_f64()?,
                                _ => return None,
                            };
                            let expected_prefix = format!("{}.", alias);
                            if let Some(rest) = col.strip_prefix(&expected_prefix) {
                                Some((rest.to_string(), lit))
                            } else {
                                None
                            }
                        }
                        _ => None,
                    }
                }

                // Inner HAVING: SUM(amount) >= K (allow optional extra parens)
                fn parse_having_sum_ge(h: &SqlExpr) -> Option<(String, f64)> {
                    match h {
                        SqlExpr::Paren(inner) => parse_having_sum_ge(inner),
                        SqlExpr::BinaryOp { left, op: BinaryOperator::Ge, right } => {
                            let (col_name, _) = match left.as_ref() {
                                SqlExpr::Function { name, args } if name.eq_ignore_ascii_case("sum") => {
                                    if args.len() != 1 {
                                        return None;
                                    }
                                    match &args[0] {
                                        SqlExpr::Column(c) => (c.clone(), ()),
                                        _ => return None,
                                    }
                                }
                                _ => return None,
                            };
                            let lit = match right.as_ref() {
                                SqlExpr::Literal(v) => v.as_f64()?,
                                _ => return None,
                            };
                            Some((col_name, lit))
                        }
                        _ => None,
                    }
                }

                // Inner SELECT must contain SUM(amount) AS <s>
                fn find_sum_alias(inner_cols: &[SelectColumn]) -> Option<(String, String)> {
                    for c in inner_cols {
                        if let SelectColumn::Aggregate { func: AggregateFunc::Sum, column: Some(col), alias: Some(a), distinct: false } = c {
                            return Some((col.clone(), a.clone()));
                        }
                    }
                    None
                }

                if is_count_star {
                    if let Some(ref outer_where) = stmt.where_clause {
                        if let Some((outer_sum_alias, outer_k)) = parse_sum_ge_threshold(outer_where, &alias) {
                            // inner statement checks
                            if sub.joins.is_empty()
                                && sub.where_clause.is_none()
                                && sub.order_by.is_empty()
                                && sub.limit.is_none()
                                && sub.offset.is_none()
                                && !sub.group_by.is_empty()
                                && sub.having.is_some()
                            {
                                if let Some((sum_col, sum_alias)) = find_sum_alias(&sub.columns) {
                                    if sum_alias == outer_sum_alias {
                                        if let Some((having_sum_col, having_k)) = parse_having_sum_ge(sub.having.as_ref().unwrap()) {
                                            if having_sum_col == sum_col && (having_k - outer_k).abs() < f64::EPSILON {
                                                // Execute inner GROUP BY once, but only return COUNT of groups passing threshold.
                                                let inner_table_name = sub
                                                    .from
                                                    .as_ref()
                                                    .map(|f| match f {
                                                        FromItem::Table { table, .. } => table.clone(),
                                                        _ => default_table.to_string(),
                                                    })
                                                    .unwrap_or_else(|| default_table.to_string());

                                                // Flush pending writes for the scanned table
                                                {
                                                    let t = tables.get_mut(&inner_table_name).ok_or_else(|| {
                                                        ApexError::QueryParseError(format!("Table '{}' not found.", inner_table_name))
                                                    })?;
                                                    if t.has_pending_writes() {
                                                        t.flush_write_buffer();
                                                    }
                                                }

                                                let table = tables.get(&inner_table_name).ok_or_else(|| {
                                                    ApexError::QueryParseError(format!("Table '{}' not found.", inner_table_name))
                                                })?;

                                                let schema = table.schema_ref();
                                                let cols = table.columns_ref();

                                                let sum_idx = schema.get_index(&sum_col).ok_or_else(|| {
                                                    ApexError::QueryParseError(format!("Unknown column: {}", sum_col))
                                                })?;

                                                let mut group_col_indices = Vec::with_capacity(sub.group_by.len());
                                                for g in &sub.group_by {
                                                    let gi = schema.get_index(g).ok_or_else(|| {
                                                        ApexError::QueryParseError(format!("Unknown GROUP BY column: {}", g))
                                                    })?;
                                                    group_col_indices.push(gi);
                                                }

                                                #[derive(Clone, Eq, PartialEq, Hash)]
                                                enum KeyPart {
                                                    Null,
                                                    Int(i64),
                                                    Float(u64),
                                                    Bool(bool),
                                                    StrId(u32),
                                                }

                                                #[derive(Clone, Eq, PartialEq, Hash)]
                                                struct GroupKey {
                                                    parts: Vec<KeyPart>,
                                                }

                                                let mut groups: HashMap<GroupKey, f64> = HashMap::new();
                                                let mut str_intern: HashMap<String, u32> = HashMap::new();
                                                let mut next_str_id: u32 = 0;
                                                let row_count = table.get_row_count();
                                                let deleted = table.deleted_ref();

                                                let mut sum_col_f64: Option<&[f64]> = None;
                                                let mut sum_col_i64: Option<&[i64]> = None;
                                                let mut sum_col_nulls: Option<&crate::table::column_table::BitVec> = None;
                                                match &cols[sum_idx] {
                                                    TypedColumn::Float64 { data, nulls } => {
                                                        sum_col_f64 = Some(data);
                                                        sum_col_nulls = Some(nulls);
                                                    }
                                                    TypedColumn::Int64 { data, nulls } => {
                                                        sum_col_i64 = Some(data);
                                                        sum_col_nulls = Some(nulls);
                                                    }
                                                    _ => {}
                                                }

                                                for row_idx in 0..row_count {
                                                    if deleted.get(row_idx) {
                                                        continue;
                                                    }
                                                    let mut parts: Vec<KeyPart> = Vec::with_capacity(group_col_indices.len());
                                                    for &gi in &group_col_indices {
                                                        match &cols[gi] {
                                                            TypedColumn::Int64 { data, nulls } => {
                                                                if row_idx >= data.len() || nulls.get(row_idx) {
                                                                    parts.push(KeyPart::Null);
                                                                } else {
                                                                    parts.push(KeyPart::Int(data[row_idx]));
                                                                }
                                                            }
                                                            TypedColumn::Float64 { data, nulls } => {
                                                                if row_idx >= data.len() || nulls.get(row_idx) {
                                                                    parts.push(KeyPart::Null);
                                                                } else {
                                                                    parts.push(KeyPart::Float(data[row_idx].to_bits()));
                                                                }
                                                            }
                                                            TypedColumn::Bool { data, nulls } => {
                                                                if row_idx >= data.len() || nulls.get(row_idx) {
                                                                    parts.push(KeyPart::Null);
                                                                } else {
                                                                    parts.push(KeyPart::Bool(data.get(row_idx)));
                                                                }
                                                            }
                                                            TypedColumn::String(sc) => {
                                                                if sc.is_null(row_idx) {
                                                                    parts.push(KeyPart::Null);
                                                                } else if let Some(s) = sc.get(row_idx) {
                                                                    if let Some(id) = str_intern.get(s) {
                                                                        parts.push(KeyPart::StrId(*id));
                                                                    } else {
                                                                        let id = next_str_id;
                                                                        next_str_id = next_str_id.wrapping_add(1);
                                                                        str_intern.insert(s.to_string(), id);
                                                                        parts.push(KeyPart::StrId(id));
                                                                    }
                                                                } else {
                                                                    parts.push(KeyPart::Null);
                                                                }
                                                            }
                                                            TypedColumn::Mixed { data, nulls } => {
                                                                if row_idx >= data.len() || nulls.get(row_idx) {
                                                                    parts.push(KeyPart::Null);
                                                                } else {
                                                                    let sv = data[row_idx].to_string_value();
                                                                    if let Some(id) = str_intern.get(&sv) {
                                                                        parts.push(KeyPart::StrId(*id));
                                                                    } else {
                                                                        let id = next_str_id;
                                                                        next_str_id = next_str_id.wrapping_add(1);
                                                                        str_intern.insert(sv, id);
                                                                        parts.push(KeyPart::StrId(id));
                                                                    }
                                                                }
                                                            }
                                                        }
                                                    }
                                                    let key = GroupKey { parts };

                                                    let add = if let (Some(data), Some(nulls)) = (sum_col_f64, sum_col_nulls) {
                                                        if row_idx >= data.len() || nulls.get(row_idx) {
                                                            0.0
                                                        } else {
                                                            data[row_idx]
                                                        }
                                                    } else if let (Some(data), Some(nulls)) = (sum_col_i64, sum_col_nulls) {
                                                        if row_idx >= data.len() || nulls.get(row_idx) {
                                                            0.0
                                                        } else {
                                                            data[row_idx] as f64
                                                        }
                                                    } else {
                                                        match cols[sum_idx].get(row_idx).and_then(|v| v.as_f64()) {
                                                            Some(v) => v,
                                                            None => 0.0,
                                                        }
                                                    };

                                                    *groups.entry(key).or_insert(0.0) += add;
                                                }

                                                let mut cnt: i64 = 0;
                                                for (_k, s) in groups {
                                                    if s >= outer_k {
                                                        cnt += 1;
                                                    }
                                                }

                                                return Ok(SqlResult::new(
                                                    vec!["big_groups".to_string()],
                                                    vec![vec![Value::Int64(cnt)]],
                                                ));
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }

            if !stmt.joins.is_empty() {
                return Err(ApexError::QueryParseError(
                    "JOIN with derived table in FROM is not supported yet".to_string(),
                ));
            }

            let sub_res = SqlExecutor::execute_select_with_tables(*sub, tables, default_table)?;
            let tmp = materialize_sql_result_to_table(&alias, sub_res)?;
            tables.insert(alias.clone(), tmp);

            // Rewrite outer query to refer to the materialized alias table and strip qualifiers
            stmt.from = Some(FromItem::Table { table: alias.clone(), alias: Some(alias.clone()) });
            for c in &mut stmt.columns {
                match c {
                    SelectColumn::Column(name) => *name = strip_prefix(name, &alias),
                    SelectColumn::ColumnAlias { column, .. } => *column = strip_prefix(column, &alias),
                    SelectColumn::Aggregate { column, .. } => {
                        if let Some(cc) = column.as_mut() {
                            *cc = strip_prefix(cc, &alias);
                        }
                    }
                    SelectColumn::Expression { expr, .. } => {
                        *expr = rewrite_expr_strip_alias(expr, &alias);
                    }
                    _ => {}
                }
            }
            if let Some(w) = stmt.where_clause.as_mut() {
                *w = rewrite_expr_strip_alias(w, &alias);
            }
            stmt.group_by = stmt.group_by.iter().map(|g| strip_prefix(g, &alias)).collect();
            if let Some(h) = stmt.having.as_mut() {
                *h = rewrite_expr_strip_alias(h, &alias);
            }
            for ob in &mut stmt.order_by {
                ob.column = strip_prefix(&ob.column, &alias);
            }
        }

        // If this is a single-table query with an explicit table alias in FROM (e.g. FROM users u),
        // normalize outer projection/order-by to unqualified column names (u.name -> name).
        // This keeps the single-table execution logic and schema lookup consistent.
        if stmt.joins.is_empty() {
            if let Some(FromItem::Table { table, alias: Some(a) }) = stmt.from.as_ref() {
                fn strip_outer_prefix(name: &str, table: &str, alias: &str) -> String {
                    if let Some((p, c)) = name.split_once('.') {
                        if p == alias || p == table {
                            return c.to_string();
                        }
                    }
                    name.to_string()
                }

                for c in &mut stmt.columns {
                    match c {
                        SelectColumn::Column(name) => {
                            *name = strip_outer_prefix(name, table, a);
                        }
                        SelectColumn::ColumnAlias { column, .. } => {
                            *column = strip_outer_prefix(column, table, a);
                        }
                        SelectColumn::Aggregate { column, .. } => {
                            if let Some(cc) = column.as_mut() {
                                *cc = strip_outer_prefix(cc, table, a);
                            }
                        }
                        SelectColumn::Expression { .. } => {
                            // Keep expressions as-is; they may legitimately reference outer alias.
                        }
                        _ => {}
                    }
                }
                for ob in &mut stmt.order_by {
                    ob.column = strip_outer_prefix(&ob.column, table, a);
                }
            }
        }

        if stmt.joins.is_empty() {
            let target_table = stmt
                .from
                .as_ref()
                .map(|f| match f {
                    FromItem::Table { table, .. } => table.clone(),
                    FromItem::Subquery { alias, .. } => alias.clone(),
                })
                .unwrap_or_else(|| default_table.to_string());

            fn expr_has_exists_subquery(expr: &SqlExpr) -> bool {
                match expr {
                    SqlExpr::ExistsSubquery { .. } => true,
                    SqlExpr::ScalarSubquery { .. } => true,
                    SqlExpr::BinaryOp { left, right, .. } => {
                        expr_has_exists_subquery(left) || expr_has_exists_subquery(right)
                    }
                    SqlExpr::UnaryOp { expr, .. } => expr_has_exists_subquery(expr),
                    SqlExpr::Paren(inner) => expr_has_exists_subquery(inner),
                    SqlExpr::Between { low, high, .. } => {
                        expr_has_exists_subquery(low) || expr_has_exists_subquery(high)
                    }
                    SqlExpr::Function { args, .. } => args.iter().any(expr_has_exists_subquery),
                    _ => false,
                }
            }

            fn expr_has_in_subquery(expr: &SqlExpr) -> bool {
                match expr {
                    SqlExpr::InSubquery { .. } => true,
                    SqlExpr::BinaryOp { left, right, .. } => {
                        expr_has_in_subquery(left) || expr_has_in_subquery(right)
                    }
                    SqlExpr::UnaryOp { expr, .. } => expr_has_in_subquery(expr),
                    SqlExpr::Paren(inner) => expr_has_in_subquery(inner),
                    SqlExpr::Between { low, high, .. } => {
                        expr_has_in_subquery(low) || expr_has_in_subquery(high)
                    }
                    SqlExpr::Function { args, .. } => args.iter().any(expr_has_in_subquery),
                    SqlExpr::Case { when_then, else_expr } => {
                        when_then.iter().any(|(c, v)| expr_has_in_subquery(c) || expr_has_in_subquery(v))
                            || else_expr.as_ref().is_some_and(|e| expr_has_in_subquery(e))
                    }
                    _ => false,
                }
            }

            // If this is a single-table query but references qualified columns (e.g. u.name)
            // or has EXISTS subquery, we can't delegate to the single-table executor because it
            // doesn't understand qualifiers and Filter conversion can't represent EXISTS.
            let needs_qualified_or_exists = {
                let has_exists = stmt
                    .where_clause
                    .as_ref()
                    .is_some_and(expr_has_exists_subquery);

                let has_in_subquery = stmt
                    .where_clause
                    .as_ref()
                    .is_some_and(expr_has_in_subquery);

                let has_scalar_in_select = stmt.columns.iter().any(|c| match c {
                    SelectColumn::Expression { expr, .. } => expr_has_exists_subquery(expr),
                    _ => false,
                });

                let outer_alias = stmt.from.as_ref().and_then(|f| match f {
                    FromItem::Table { alias, .. } => alias.clone(),
                    _ => None,
                });

                let has_qualified_select = stmt.columns.iter().any(|c| match c {
                    SelectColumn::Column(name) => name.contains('.'),
                    SelectColumn::ColumnAlias { column, .. } => column.contains('.'),
                    SelectColumn::Aggregate { column, .. } => column.as_ref().is_some_and(|x| x.contains('.')),
                    SelectColumn::Expression { expr, .. } => matches!(expr, SqlExpr::Column(c) if c.contains('.')),
                    _ => false,
                });
                let has_qualified_order = stmt.order_by.iter().any(|o| o.column.contains('.'));

                has_exists
                    || has_scalar_in_select
                    || has_in_subquery
                    || outer_alias.is_some() && (has_qualified_select || has_qualified_order)
            };

            // We may need a mutable borrow to flush pending writes, but the EXISTS path also
            // needs an immutable borrow of the full tables map. Keep the mutable borrow scope
            // minimal to satisfy Rust's borrow checker.
            {
                let t = tables
                    .get_mut(&target_table)
                    .ok_or_else(|| ApexError::QueryParseError(format!("Table '{}' not found.", target_table)))?;
                if t.has_pending_writes() {
                    t.flush_write_buffer();
                }
            }

            if !needs_qualified_or_exists {
                let table = tables
                    .get_mut(&target_table)
                    .ok_or_else(|| ApexError::QueryParseError(format!("Table '{}' not found.", target_table)))?;
                return Self::execute_select(stmt, table);
            }

            // ============ Single-table (no JOIN) path with qualifiers / EXISTS support ============
            let table = tables
                .get(&target_table)
                .ok_or_else(|| ApexError::QueryParseError(format!("Table '{}' not found.", target_table)))?;

            let outer_table_name = target_table.clone();
            let outer_alias = stmt
                .from
                .as_ref()
                .and_then(|f| match f {
                    FromItem::Table { alias, .. } => alias.clone(),
                    _ => None,
                })
                .unwrap_or_else(|| outer_table_name.clone());

            fn strip_outer(col: &str, outer_alias: &str, outer_table: &str) -> String {
                if let Some((a, c)) = col.split_once('.') {
                    // In this execution path we are guaranteed to be running a single-table query.
                    // Be permissive and strip the qualifier even if it doesn't match exactly,
                    // so selecting `u.name` works reliably.
                    if a == outer_alias || a == outer_table {
                        return c.to_string();
                    }
                    return c.to_string();
                }
                col.to_string()
            }

            fn get_col_value_qualified(
                col_ref: &str,
                outer_table: &ColumnTable,
                outer_row: usize,
                outer_alias: &str,
                outer_table_name: &str,
                inner_table: &ColumnTable,
                inner_row: usize,
                inner_alias: &str,
                inner_table_name: &str,
            ) -> Value {
                // Handle special _id
                if col_ref == "_id" {
                    return Value::Int64(inner_row as i64);
                }
                let (a, c) = if let Some((a, c)) = col_ref.split_once('.') {
                    (a, c)
                } else {
                    ("", col_ref)
                };

                // Qualified
                if !a.is_empty() {
                    if a == inner_alias || a == inner_table_name {
                        if c == "_id" {
                            return Value::Int64(inner_row as i64);
                        }
                        if let Some(ci) = inner_table.schema_ref().get_index(c) {
                            return inner_table.columns_ref()[ci].get(inner_row).unwrap_or(Value::Null);
                        }
                        return Value::Null;
                    }
                    if a == outer_alias || a == outer_table_name {
                        if c == "_id" {
                            return Value::Int64(outer_row as i64);
                        }
                        if let Some(ci) = outer_table.schema_ref().get_index(c) {
                            return outer_table.columns_ref()[ci].get(outer_row).unwrap_or(Value::Null);
                        }
                        return Value::Null;
                    }
                    return Value::Null;
                }

                // Unqualified: prefer inner table if column exists there
                if c == "_id" {
                    return Value::Int64(inner_row as i64);
                }
                if inner_table.schema_ref().get_index(c).is_some() {
                    let ci = inner_table.schema_ref().get_index(c).unwrap();
                    return inner_table.columns_ref()[ci].get(inner_row).unwrap_or(Value::Null);
                }
                if outer_table.schema_ref().get_index(c).is_some() {
                    let ci = outer_table.schema_ref().get_index(c).unwrap();
                    return outer_table.columns_ref()[ci].get(outer_row).unwrap_or(Value::Null);
                }
                Value::Null
            }

            fn eval_correlated_predicate(
                expr: &SqlExpr,
                outer_table: &ColumnTable,
                outer_row: usize,
                outer_alias: &str,
                outer_table_name: &str,
                inner_table: &ColumnTable,
                inner_row: usize,
                inner_alias: &str,
                inner_table_name: &str,
            ) -> Result<bool, ApexError> {
                fn eval_scalar(
                    expr: &SqlExpr,
                    outer_table: &ColumnTable,
                    outer_row: usize,
                    outer_alias: &str,
                    outer_table_name: &str,
                    inner_table: &ColumnTable,
                    inner_row: usize,
                    inner_alias: &str,
                    inner_table_name: &str,
                ) -> Result<Value, ApexError> {
                    match expr {
                        SqlExpr::Paren(inner) => eval_scalar(
                            inner,
                            outer_table,
                            outer_row,
                            outer_alias,
                            outer_table_name,
                            inner_table,
                            inner_row,
                            inner_alias,
                            inner_table_name,
                        ),
                        SqlExpr::Literal(v) => Ok(v.clone()),
                        SqlExpr::Column(c) => Ok(get_col_value_qualified(
                            c,
                            outer_table,
                            outer_row,
                            outer_alias,
                            outer_table_name,
                            inner_table,
                            inner_row,
                            inner_alias,
                            inner_table_name,
                        )),
                        SqlExpr::Function { name, args } => {
                            if name.eq_ignore_ascii_case("rand") {
                                if !args.is_empty() {
                                    return Err(ApexError::QueryParseError(
                                        "RAND() does not accept arguments".to_string(),
                                    ));
                                }
                                return Ok(Value::Float64(rand::random::<f64>()));
                            }
                            if name.eq_ignore_ascii_case("len") {
                                if args.len() != 1 {
                                    return Err(ApexError::QueryParseError(
                                        "LEN() expects 1 argument".to_string(),
                                    ));
                                }
                                let v = eval_scalar(
                                    &args[0],
                                    outer_table,
                                    outer_row,
                                    outer_alias,
                                    outer_table_name,
                                    inner_table,
                                    inner_row,
                                    inner_alias,
                                    inner_table_name,
                                )?;
                                if v.is_null() {
                                    return Ok(Value::Null);
                                }
                                let s = v.to_string_value();
                                return Ok(Value::Int64(s.chars().count() as i64));
                            }
                            if name.eq_ignore_ascii_case("trim") {
                                if args.len() != 1 {
                                    return Err(ApexError::QueryParseError(
                                        "TRIM() expects 1 argument".to_string(),
                                    ));
                                }
                                let v = eval_scalar(
                                    &args[0],
                                    outer_table,
                                    outer_row,
                                    outer_alias,
                                    outer_table_name,
                                    inner_table,
                                    inner_row,
                                    inner_alias,
                                    inner_table_name,
                                )?;
                                if v.is_null() {
                                    return Ok(Value::Null);
                                }
                                return Ok(Value::String(v.to_string_value().trim().to_string()));
                            }
                            if name.eq_ignore_ascii_case("upper") {
                                if args.len() != 1 {
                                    return Err(ApexError::QueryParseError(
                                        "UPPER() expects 1 argument".to_string(),
                                    ));
                                }
                                let v = eval_scalar(
                                    &args[0],
                                    outer_table,
                                    outer_row,
                                    outer_alias,
                                    outer_table_name,
                                    inner_table,
                                    inner_row,
                                    inner_alias,
                                    inner_table_name,
                                )?;
                                if v.is_null() {
                                    return Ok(Value::Null);
                                }
                                return Ok(Value::String(v.to_string_value().to_uppercase()));
                            }
                            if name.eq_ignore_ascii_case("lower") {
                                if args.len() != 1 {
                                    return Err(ApexError::QueryParseError(
                                        "LOWER() expects 1 argument".to_string(),
                                    ));
                                }
                                let v = eval_scalar(
                                    &args[0],
                                    outer_table,
                                    outer_row,
                                    outer_alias,
                                    outer_table_name,
                                    inner_table,
                                    inner_row,
                                    inner_alias,
                                    inner_table_name,
                                )?;
                                if v.is_null() {
                                    return Ok(Value::Null);
                                }
                                return Ok(Value::String(v.to_string_value().to_lowercase()));
                            }
                            if name.eq_ignore_ascii_case("replace") {
                                if args.len() != 3 {
                                    return Err(ApexError::QueryParseError(
                                        "REPLACE() expects 3 arguments".to_string(),
                                    ));
                                }
                                let s0 = eval_scalar(
                                    &args[0],
                                    outer_table,
                                    outer_row,
                                    outer_alias,
                                    outer_table_name,
                                    inner_table,
                                    inner_row,
                                    inner_alias,
                                    inner_table_name,
                                )?;
                                let from0 = eval_scalar(
                                    &args[1],
                                    outer_table,
                                    outer_row,
                                    outer_alias,
                                    outer_table_name,
                                    inner_table,
                                    inner_row,
                                    inner_alias,
                                    inner_table_name,
                                )?;
                                let to0 = eval_scalar(
                                    &args[2],
                                    outer_table,
                                    outer_row,
                                    outer_alias,
                                    outer_table_name,
                                    inner_table,
                                    inner_row,
                                    inner_alias,
                                    inner_table_name,
                                )?;
                                if s0.is_null() || from0.is_null() || to0.is_null() {
                                    return Ok(Value::Null);
                                }
                                let s = s0.to_string_value();
                                let from = from0.to_string_value();
                                let to = to0.to_string_value();
                                return Ok(Value::String(s.replace(&from, &to)));
                            }
                            if name.eq_ignore_ascii_case("mid") {
                                if args.len() != 2 && args.len() != 3 {
                                    return Err(ApexError::QueryParseError(
                                        "MID() expects 2 or 3 arguments".to_string(),
                                    ));
                                }
                                let s0 = eval_scalar(
                                    &args[0],
                                    outer_table,
                                    outer_row,
                                    outer_alias,
                                    outer_table_name,
                                    inner_table,
                                    inner_row,
                                    inner_alias,
                                    inner_table_name,
                                )?;
                                let start0 = eval_scalar(
                                    &args[1],
                                    outer_table,
                                    outer_row,
                                    outer_alias,
                                    outer_table_name,
                                    inner_table,
                                    inner_row,
                                    inner_alias,
                                    inner_table_name,
                                )?;
                                if s0.is_null() || start0.is_null() {
                                    return Ok(Value::Null);
                                }
                                let s = s0.to_string_value();
                                let mut start = start0.as_i64().unwrap_or(1);
                                if start < 1 {
                                    start = 1;
                                }
                                let start_idx = (start - 1) as usize;
                                let chars: Vec<char> = s.chars().collect();
                                if start_idx >= chars.len() {
                                    return Ok(Value::String(String::new()));
                                }
                                let end_idx = if args.len() == 3 {
                                    let len0 = eval_scalar(
                                        &args[2],
                                        outer_table,
                                        outer_row,
                                        outer_alias,
                                        outer_table_name,
                                        inner_table,
                                        inner_row,
                                        inner_alias,
                                        inner_table_name,
                                    )?;
                                    if len0.is_null() {
                                        return Ok(Value::Null);
                                    }
                                    let mut l = len0.as_i64().unwrap_or(0);
                                    if l < 0 {
                                        l = 0;
                                    }
                                    (start_idx + l as usize).min(chars.len())
                                } else {
                                    chars.len()
                                };
                                let out: String = chars[start_idx..end_idx].iter().collect();
                                return Ok(Value::String(out));
                            }
                            Err(ApexError::QueryParseError(
                                format!("Unsupported function: {}", name),
                            ))
                        }
                        SqlExpr::UnaryOp { op: UnaryOperator::Minus, expr } => {
                            let v = eval_scalar(
                                expr,
                                outer_table,
                                outer_row,
                                outer_alias,
                                outer_table_name,
                                inner_table,
                                inner_row,
                                inner_alias,
                                inner_table_name,
                            )?;
                            if let Some(i) = v.as_i64() {
                                Ok(Value::Int64(-i))
                            } else if let Some(f) = v.as_f64() {
                                Ok(Value::Float64(-f))
                            } else {
                                Ok(Value::Null)
                            }
                        }
                        _ => Ok(Value::Null),
                    }
                }

                match expr {
                    SqlExpr::Paren(inner) => eval_correlated_predicate(
                        inner,
                        outer_table,
                        outer_row,
                        outer_alias,
                        outer_table_name,
                        inner_table,
                        inner_row,
                        inner_alias,
                        inner_table_name,
                    ),
                    SqlExpr::Literal(v) => Ok(v.as_bool().unwrap_or(false)),
                    SqlExpr::UnaryOp { op: UnaryOperator::Not, expr } => Ok(!eval_correlated_predicate(
                        expr,
                        outer_table,
                        outer_row,
                        outer_alias,
                        outer_table_name,
                        inner_table,
                        inner_row,
                        inner_alias,
                        inner_table_name,
                    )?),
                    SqlExpr::BinaryOp { left, op, right } => match op {
                        BinaryOperator::And => Ok(
                            eval_correlated_predicate(
                                left,
                                outer_table,
                                outer_row,
                                outer_alias,
                                outer_table_name,
                                inner_table,
                                inner_row,
                                inner_alias,
                                inner_table_name,
                            )? && eval_correlated_predicate(
                                right,
                                outer_table,
                                outer_row,
                                outer_alias,
                                outer_table_name,
                                inner_table,
                                inner_row,
                                inner_alias,
                                inner_table_name,
                            )?,
                        ),
                        BinaryOperator::Or => Ok(
                            eval_correlated_predicate(
                                left,
                                outer_table,
                                outer_row,
                                outer_alias,
                                outer_table_name,
                                inner_table,
                                inner_row,
                                inner_alias,
                                inner_table_name,
                            )? || eval_correlated_predicate(
                                right,
                                outer_table,
                                outer_row,
                                outer_alias,
                                outer_table_name,
                                inner_table,
                                inner_row,
                                inner_alias,
                                inner_table_name,
                            )?,
                        ),
                        BinaryOperator::Eq
                        | BinaryOperator::NotEq
                        | BinaryOperator::Lt
                        | BinaryOperator::Le
                        | BinaryOperator::Gt
                        | BinaryOperator::Ge => {
                            let lv = eval_scalar(
                                left,
                                outer_table,
                                outer_row,
                                outer_alias,
                                outer_table_name,
                                inner_table,
                                inner_row,
                                inner_alias,
                                inner_table_name,
                            )?;
                            let rv = eval_scalar(
                                right,
                                outer_table,
                                outer_row,
                                outer_alias,
                                outer_table_name,
                                inner_table,
                                inner_row,
                                inner_alias,
                                inner_table_name,
                            )?;
                            if lv.is_null() || rv.is_null() {
                                return Ok(false);
                            }
                            let ord = lv.partial_cmp(&rv).unwrap_or(Ordering::Equal);
                            Ok(match op {
                                BinaryOperator::Eq => ord == Ordering::Equal,
                                BinaryOperator::NotEq => ord != Ordering::Equal,
                                BinaryOperator::Lt => ord == Ordering::Less,
                                BinaryOperator::Le => ord != Ordering::Greater,
                                BinaryOperator::Gt => ord == Ordering::Greater,
                                BinaryOperator::Ge => ord != Ordering::Less,
                                _ => false,
                            })
                        }
                        _ => Err(ApexError::QueryParseError(
                            "Unsupported operator in correlated predicate".to_string(),
                        )),
                    },
                    _ => Err(ApexError::QueryParseError(
                        "Unsupported expression in correlated predicate".to_string(),
                    )),
                }
            }

            fn exists_for_outer_row(
                sub: &SelectStatement,
                tables: &HashMap<String, ColumnTable>,
                default_table: &str,
                outer_table: &ColumnTable,
                outer_row: usize,
                outer_alias: &str,
                outer_table_name: &str,
            ) -> Result<bool, ApexError> {
                if !sub.joins.is_empty() {
                    return Err(ApexError::QueryParseError(
                        "EXISTS subquery with JOIN is not supported yet".to_string(),
                    ));
                }
                if !sub.group_by.is_empty() || sub.having.is_some() {
                    return Err(ApexError::QueryParseError(
                        "EXISTS subquery with GROUP BY/HAVING is not supported yet".to_string(),
                    ));
                }

                let (inner_table_name, inner_alias) = match sub.from.as_ref() {
                    Some(FromItem::Table { table, alias }) => {
                        (table.clone(), alias.clone().unwrap_or_else(|| table.clone()))
                    }
                    Some(FromItem::Subquery { .. }) => {
                        return Err(ApexError::QueryParseError(
                            "EXISTS subquery FROM (subquery) is not supported yet".to_string(),
                        ))
                    }
                    None => (default_table.to_string(), default_table.to_string()),
                };

                let inner_table = tables
                    .get(&inner_table_name)
                    .ok_or_else(|| ApexError::QueryParseError(format!("Table '{}' not found.", inner_table_name)))?;

                let row_count = inner_table.get_row_count();
                let deleted = inner_table.deleted_ref();
                for inner_row in 0..row_count {
                    if deleted.get(inner_row) {
                        continue;
                    }

                    let passes = if let Some(ref w) = sub.where_clause {
                        eval_correlated_predicate(
                            w,
                            outer_table,
                            outer_row,
                            outer_alias,
                            outer_table_name,
                            inner_table,
                            inner_row,
                            &inner_alias,
                            &inner_table_name,
                        )?
                    } else {
                        true
                    };

                    if passes {
                        return Ok(true);
                    }
                }
                Ok(false)
            }

            fn eval_outer_where(
                expr: &SqlExpr,
                tables: &HashMap<String, ColumnTable>,
                default_table: &str,
                outer_table: &ColumnTable,
                outer_row: usize,
                outer_alias: &str,
                outer_table_name: &str,
            ) -> Result<bool, ApexError> {
                fn get_outer_value(
                    col_ref: &str,
                    outer_table: &ColumnTable,
                    outer_row: usize,
                    outer_alias: &str,
                    outer_table_name: &str,
                ) -> Value {
                    if col_ref == "_id" {
                        return Value::Int64(outer_row as i64);
                    }
                    let (a, c) = if let Some((a, c)) = col_ref.split_once('.') {
                        (a, c)
                    } else {
                        ("", col_ref)
                    };
                    let name = if !a.is_empty() {
                        if a == outer_alias || a == outer_table_name {
                            c
                        } else {
                            // Unknown qualifier
                            return Value::Null;
                        }
                    } else {
                        c
                    };
                    if name == "_id" {
                        return Value::Int64(outer_row as i64);
                    }
                    if let Some(ci) = outer_table.schema_ref().get_index(name) {
                        outer_table.columns_ref()[ci].get(outer_row).unwrap_or(Value::Null)
                    } else {
                        Value::Null
                    }
                }

                fn eval_outer_scalar(
                    expr: &SqlExpr,
                    outer_table: &ColumnTable,
                    outer_row: usize,
                    outer_alias: &str,
                    outer_table_name: &str,
                ) -> Value {
                    match expr {
                        SqlExpr::Paren(inner) => {
                            eval_outer_scalar(inner, outer_table, outer_row, outer_alias, outer_table_name)
                        }
                        SqlExpr::Literal(v) => v.clone(),
                        SqlExpr::Column(c) => get_outer_value(c, outer_table, outer_row, outer_alias, outer_table_name),
                        SqlExpr::Function { name, args } => {
                            if name.eq_ignore_ascii_case("rand") {
                                if !args.is_empty() {
                                    return Value::Null;
                                }
                                Value::Float64(rand::random::<f64>())
                            } else if name.eq_ignore_ascii_case("len") {
                                if args.len() != 1 {
                                    return Value::Null;
                                }
                                let v = eval_outer_scalar(&args[0], outer_table, outer_row, outer_alias, outer_table_name);
                                if v.is_null() {
                                    Value::Null
                                } else {
                                    let s = v.to_string_value();
                                    Value::Int64(s.chars().count() as i64)
                                }
                            } else if name.eq_ignore_ascii_case("trim") {
                                if args.len() != 1 {
                                    return Value::Null;
                                }
                                let v = eval_outer_scalar(&args[0], outer_table, outer_row, outer_alias, outer_table_name);
                                if v.is_null() {
                                    Value::Null
                                } else {
                                    Value::String(v.to_string_value().trim().to_string())
                                }
                            } else if name.eq_ignore_ascii_case("upper") {
                                if args.len() != 1 {
                                    return Value::Null;
                                }
                                let v = eval_outer_scalar(&args[0], outer_table, outer_row, outer_alias, outer_table_name);
                                if v.is_null() {
                                    Value::Null
                                } else {
                                    Value::String(v.to_string_value().to_uppercase())
                                }
                            } else if name.eq_ignore_ascii_case("lower") {
                                if args.len() != 1 {
                                    return Value::Null;
                                }
                                let v = eval_outer_scalar(&args[0], outer_table, outer_row, outer_alias, outer_table_name);
                                if v.is_null() {
                                    Value::Null
                                } else {
                                    Value::String(v.to_string_value().to_lowercase())
                                }
                            } else if name.eq_ignore_ascii_case("replace") {
                                if args.len() != 3 {
                                    return Value::Null;
                                }
                                let s0 = eval_outer_scalar(&args[0], outer_table, outer_row, outer_alias, outer_table_name);
                                let from0 = eval_outer_scalar(&args[1], outer_table, outer_row, outer_alias, outer_table_name);
                                let to0 = eval_outer_scalar(&args[2], outer_table, outer_row, outer_alias, outer_table_name);
                                if s0.is_null() || from0.is_null() || to0.is_null() {
                                    Value::Null
                                } else {
                                    Value::String(s0.to_string_value().replace(&from0.to_string_value(), &to0.to_string_value()))
                                }
                            } else if name.eq_ignore_ascii_case("mid") {
                                if args.len() != 2 && args.len() != 3 {
                                    return Value::Null;
                                }
                                let s0 = eval_outer_scalar(&args[0], outer_table, outer_row, outer_alias, outer_table_name);
                                let start0 = eval_outer_scalar(&args[1], outer_table, outer_row, outer_alias, outer_table_name);
                                if s0.is_null() || start0.is_null() {
                                    return Value::Null;
                                }
                                let s = s0.to_string_value();
                                let mut start = start0.as_i64().unwrap_or(1);
                                if start < 1 {
                                    start = 1;
                                }
                                let start_idx = (start - 1) as usize;
                                let chars: Vec<char> = s.chars().collect();
                                if start_idx >= chars.len() {
                                    return Value::String(String::new());
                                }
                                let end_idx = if args.len() == 3 {
                                    let len0 = eval_outer_scalar(&args[2], outer_table, outer_row, outer_alias, outer_table_name);
                                    if len0.is_null() {
                                        return Value::Null;
                                    }
                                    let mut l = len0.as_i64().unwrap_or(0);
                                    if l < 0 {
                                        l = 0;
                                    }
                                    (start_idx + l as usize).min(chars.len())
                                } else {
                                    chars.len()
                                };
                                Value::String(chars[start_idx..end_idx].iter().collect())
                            } else {
                                Value::Null
                            }
                        }
                        SqlExpr::UnaryOp { op: UnaryOperator::Minus, expr } => {
                            let v = eval_outer_scalar(expr, outer_table, outer_row, outer_alias, outer_table_name);
                            if let Some(i) = v.as_i64() {
                                Value::Int64(-i)
                            } else if let Some(f) = v.as_f64() {
                                Value::Float64(-f)
                            } else {
                                Value::Null
                            }
                        }
                        _ => Value::Null,
                    }
                }

                fn in_subquery_for_outer_row(
                    column: &str,
                    sub: &SelectStatement,
                    negated: bool,
                    tables: &HashMap<String, ColumnTable>,
                    default_table: &str,
                    outer_table: &ColumnTable,
                    outer_row: usize,
                    outer_alias: &str,
                    outer_table_name: &str,
                ) -> Result<bool, ApexError> {
                    if !sub.joins.is_empty() {
                        return Err(ApexError::QueryParseError(
                            "IN (subquery) with JOIN is not supported yet".to_string(),
                        ));
                    }

                    // Determine inner table
                    let (inner_table_name, inner_alias) = match sub.from.as_ref() {
                        Some(FromItem::Table { table, alias }) => {
                            (table.clone(), alias.clone().unwrap_or_else(|| table.clone()))
                        }
                        Some(FromItem::Subquery { .. }) => {
                            return Err(ApexError::QueryParseError(
                                "IN (subquery) FROM (subquery) is not supported yet".to_string(),
                            ))
                        }
                        None => (default_table.to_string(), default_table.to_string()),
                    };

                    let inner_table = tables
                        .get(&inner_table_name)
                        .ok_or_else(|| {
                            ApexError::QueryParseError(format!("Table '{}' not found.", inner_table_name))
                        })?;

                    // Outer value
                    let outer_v = get_outer_value(column, outer_table, outer_row, outer_alias, outer_table_name);
                    if outer_v.is_null() {
                        return Ok(false);
                    }

                    // Subquery must project a single column for IN
                    if sub.columns.len() != 1 {
                        return Err(ApexError::QueryParseError(
                            "IN (subquery) requires single-column subquery".to_string(),
                        ));
                    }

                    let selected_ref: Option<String> = match &sub.columns[0] {
                        SelectColumn::Column(c) => Some(c.clone()),
                        SelectColumn::ColumnAlias { column, .. } => Some(column.clone()),
                        SelectColumn::Expression { .. } => None,
                        SelectColumn::All => None,
                        SelectColumn::Aggregate { .. } => None,
                        SelectColumn::WindowFunction { .. } => None,
                    };
                    if selected_ref.is_none() {
                        return Err(ApexError::QueryParseError(
                            "IN (subquery) projection must be a column".to_string(),
                        ));
                    }
                    let selected_ref = selected_ref.unwrap();

                    let mut has_null = false;
                    let mut any_match = false;

                    let row_count = inner_table.get_row_count();
                    let deleted = inner_table.deleted_ref();
                    for inner_row in 0..row_count {
                        if deleted.get(inner_row) {
                            continue;
                        }

                        let passes = if let Some(ref w) = sub.where_clause {
                            eval_correlated_predicate(
                                w,
                                outer_table,
                                outer_row,
                                outer_alias,
                                outer_table_name,
                                inner_table,
                                inner_row,
                                &inner_alias,
                                &inner_table_name,
                            )?
                        } else {
                            true
                        };
                        if !passes {
                            continue;
                        }

                        let v = get_col_value_qualified(
                            &selected_ref,
                            outer_table,
                            outer_row,
                            outer_alias,
                            outer_table_name,
                            inner_table,
                            inner_row,
                            &inner_alias,
                            &inner_table_name,
                        );

                        if v.is_null() {
                            has_null = true;
                            continue;
                        }
                        if v == outer_v {
                            any_match = true;
                            break;
                        }
                    }

                    if negated {
                        // NOT IN: if subquery contains NULL -> UNKNOWN -> filter out (FALSE)
                        if has_null {
                            return Ok(false);
                        }
                        Ok(!any_match)
                    } else {
                        // IN: if no match but contains NULL -> UNKNOWN -> filter out (FALSE)
                        if !any_match && has_null {
                            return Ok(false);
                        }
                        Ok(any_match)
                    }
                }

                match expr {
                    SqlExpr::Paren(inner) => eval_outer_where(
                        inner,
                        tables,
                        default_table,
                        outer_table,
                        outer_row,
                        outer_alias,
                        outer_table_name,
                    ),
                    SqlExpr::Literal(v) => Ok(v.as_bool().unwrap_or(false)),
                    SqlExpr::UnaryOp { op: UnaryOperator::Not, expr } => Ok(!eval_outer_where(
                        expr,
                        tables,
                        default_table,
                        outer_table,
                        outer_row,
                        outer_alias,
                        outer_table_name,
                    )?),
                    SqlExpr::BinaryOp { left, op, right } => match op {
                        BinaryOperator::And => Ok(
                            eval_outer_where(
                                left,
                                tables,
                                default_table,
                                outer_table,
                                outer_row,
                                outer_alias,
                                outer_table_name,
                            )? && eval_outer_where(
                                right,
                                tables,
                                default_table,
                                outer_table,
                                outer_row,
                                outer_alias,
                                outer_table_name,
                            )?,
                        ),
                        BinaryOperator::Or => Ok(
                            eval_outer_where(
                                left,
                                tables,
                                default_table,
                                outer_table,
                                outer_row,
                                outer_alias,
                                outer_table_name,
                            )? || eval_outer_where(
                                right,
                                tables,
                                default_table,
                                outer_table,
                                outer_row,
                                outer_alias,
                                outer_table_name,
                            )?,
                        ),
                        BinaryOperator::Eq
                        | BinaryOperator::NotEq
                        | BinaryOperator::Lt
                        | BinaryOperator::Le
                        | BinaryOperator::Gt
                        | BinaryOperator::Ge => {
                            let lv = eval_outer_scalar(left, outer_table, outer_row, outer_alias, outer_table_name);
                            let rv = eval_outer_scalar(right, outer_table, outer_row, outer_alias, outer_table_name);
                            if lv.is_null() || rv.is_null() {
                                return Ok(false);
                            }
                            let ord = lv.partial_cmp(&rv).unwrap_or(Ordering::Equal);
                            Ok(match op {
                                BinaryOperator::Eq => ord == Ordering::Equal,
                                BinaryOperator::NotEq => ord != Ordering::Equal,
                                BinaryOperator::Lt => ord == Ordering::Less,
                                BinaryOperator::Le => ord != Ordering::Greater,
                                BinaryOperator::Gt => ord == Ordering::Greater,
                                BinaryOperator::Ge => ord != Ordering::Less,
                                _ => false,
                            })
                        }
                        _ => Err(ApexError::QueryParseError(
                            "Unsupported operator in outer WHERE".to_string(),
                        )),
                    },
                    SqlExpr::ExistsSubquery { stmt: sub } => exists_for_outer_row(
                        sub,
                        tables,
                        default_table,
                        outer_table,
                        outer_row,
                        outer_alias,
                        outer_table_name,
                    ),
                    SqlExpr::InSubquery { column, stmt: sub, negated } => in_subquery_for_outer_row(
                        column,
                        sub,
                        *negated,
                        tables,
                        default_table,
                        outer_table,
                        outer_row,
                        outer_alias,
                        outer_table_name,
                    ),
                    SqlExpr::ScalarSubquery { .. } => Err(ApexError::QueryParseError(
                        "Scalar subquery is not supported in WHERE yet".to_string(),
                    )),
                    _ => Err(ApexError::QueryParseError(
                        "Unsupported expression in outer WHERE".to_string(),
                    )),
                }
            }

            fn eval_scalar_subquery_for_outer_row(
                sub: &SelectStatement,
                tables: &HashMap<String, ColumnTable>,
                default_table: &str,
                outer_table: &ColumnTable,
                outer_row: usize,
                outer_alias: &str,
                outer_table_name: &str,
            ) -> Result<Value, ApexError> {
                if !sub.joins.is_empty() {
                    return Err(ApexError::QueryParseError(
                        "Scalar subquery with JOIN is not supported yet".to_string(),
                    ));
                }
                if !sub.group_by.is_empty() || sub.having.is_some() {
                    return Err(ApexError::QueryParseError(
                        "Scalar subquery with GROUP BY/HAVING is not supported yet".to_string(),
                    ));
                }
                if sub.columns.len() != 1 {
                    return Err(ApexError::QueryParseError(
                        "Scalar subquery must return exactly one column".to_string(),
                    ));
                }

                let (inner_table_name, inner_alias) = match sub.from.as_ref() {
                    Some(FromItem::Table { table, alias }) => {
                        (table.clone(), alias.clone().unwrap_or_else(|| table.clone()))
                    }
                    Some(FromItem::Subquery { .. }) => {
                        return Err(ApexError::QueryParseError(
                            "Scalar subquery FROM (subquery) is not supported yet".to_string(),
                        ))
                    }
                    None => (default_table.to_string(), default_table.to_string()),
                };

                let inner_table = tables
                    .get(&inner_table_name)
                    .ok_or_else(|| ApexError::QueryParseError(format!("Table '{}' not found.", inner_table_name)))?;

                // Support only simple aggregate scalar: MAX(col), MIN(col), SUM(col), AVG(col), COUNT(*), COUNT(col)
                let (func, col) = match &sub.columns[0] {
                    SelectColumn::Aggregate { func, column, distinct, .. } => {
                        if *distinct {
                            return Err(ApexError::QueryParseError(
                                "Scalar subquery DISTINCT aggregate is not supported yet".to_string(),
                            ));
                        }
                        (func.clone(), column.clone())
                    }
                    _ => {
                        return Err(ApexError::QueryParseError(
                            "Scalar subquery only supports aggregate projection".to_string(),
                        ))
                    }
                };

                // Reuse correlated predicate evaluator for subquery WHERE
                let row_count = inner_table.get_row_count();
                let deleted = inner_table.deleted_ref();

                let mut count: i64 = 0;
                let mut sum: f64 = 0.0;
                let mut sum_count: i64 = 0;
                let mut min_v: Option<Value> = None;
                let mut max_v: Option<Value> = None;

                for inner_row in 0..row_count {
                    if deleted.get(inner_row) {
                        continue;
                    }
                    let passes = if let Some(ref w) = sub.where_clause {
                        eval_correlated_predicate(
                            w,
                            outer_table,
                            outer_row,
                            outer_alias,
                            outer_table_name,
                            inner_table,
                            inner_row,
                            &inner_alias,
                            &inner_table_name,
                        )?
                    } else {
                        true
                    };
                    if !passes {
                        continue;
                    }

                    match func {
                        AggregateFunc::Count => {
                            if col.is_none() {
                                count += 1;
                            } else if let Some(ref c) = col {
                                let v = get_col_value_qualified(
                                    c,
                                    outer_table,
                                    outer_row,
                                    outer_alias,
                                    outer_table_name,
                                    inner_table,
                                    inner_row,
                                    &inner_alias,
                                    &inner_table_name,
                                );
                                if !v.is_null() {
                                    count += 1;
                                }
                            }
                        }
                        AggregateFunc::Sum | AggregateFunc::Avg => {
                            if let Some(ref c) = col {
                                let v = get_col_value_qualified(
                                    c,
                                    outer_table,
                                    outer_row,
                                    outer_alias,
                                    outer_table_name,
                                    inner_table,
                                    inner_row,
                                    &inner_alias,
                                    &inner_table_name,
                                );
                                if let Some(n) = v.as_f64() {
                                    sum += n;
                                    sum_count += 1;
                                }
                            }
                        }
                        AggregateFunc::Min => {
                            if let Some(ref c) = col {
                                let v = get_col_value_qualified(
                                    c,
                                    outer_table,
                                    outer_row,
                                    outer_alias,
                                    outer_table_name,
                                    inner_table,
                                    inner_row,
                                    &inner_alias,
                                    &inner_table_name,
                                );
                                if v.is_null() {
                                    continue;
                                }
                                min_v = Some(match &min_v {
                                    None => v,
                                    Some(curr) => {
                                        if SqlExecutor::compare_non_null(curr, &v) == Ordering::Greater {
                                            v
                                        } else {
                                            curr.clone()
                                        }
                                    }
                                });
                            }
                        }
                        AggregateFunc::Max => {
                            if let Some(ref c) = col {
                                let v = get_col_value_qualified(
                                    c,
                                    outer_table,
                                    outer_row,
                                    outer_alias,
                                    outer_table_name,
                                    inner_table,
                                    inner_row,
                                    &inner_alias,
                                    &inner_table_name,
                                );
                                if v.is_null() {
                                    continue;
                                }
                                max_v = Some(match &max_v {
                                    None => v,
                                    Some(curr) => {
                                        if SqlExecutor::compare_non_null(curr, &v) == Ordering::Less {
                                            v
                                        } else {
                                            curr.clone()
                                        }
                                    }
                                });
                            }
                        }
                    }
                }

                Ok(match func {
                    AggregateFunc::Count => Value::Int64(count),
                    AggregateFunc::Sum => Value::Float64(sum),
                    AggregateFunc::Avg => {
                        if sum_count > 0 {
                            Value::Float64(sum / sum_count as f64)
                        } else {
                            Value::Null
                        }
                    }
                    AggregateFunc::Min => min_v.unwrap_or(Value::Null),
                    AggregateFunc::Max => max_v.unwrap_or(Value::Null),
                })
            }

            fn eval_outer_expr(
                expr: &SqlExpr,
                tables: &HashMap<String, ColumnTable>,
                default_table: &str,
                outer_table: &ColumnTable,
                outer_row: usize,
                outer_alias: &str,
                outer_table_name: &str,
            ) -> Result<Value, ApexError> {
                fn get_outer_value_local(
                    col_ref: &str,
                    outer_table: &ColumnTable,
                    outer_row: usize,
                    outer_alias: &str,
                    outer_table_name: &str,
                ) -> Value {
                    if col_ref == "_id" {
                        return Value::Int64(outer_row as i64);
                    }
                    let (a, c) = if let Some((a, c)) = col_ref.split_once('.') {
                        (a, c)
                    } else {
                        ("", col_ref)
                    };
                    let name = if !a.is_empty() {
                        if a == outer_alias || a == outer_table_name {
                            c
                        } else {
                            return Value::Null;
                        }
                    } else {
                        c
                    };
                    if name == "_id" {
                        return Value::Int64(outer_row as i64);
                    }
                    if let Some(ci) = outer_table.schema_ref().get_index(name) {
                        outer_table.columns_ref()[ci].get(outer_row).unwrap_or(Value::Null)
                    } else {
                        Value::Null
                    }
                }

                match expr {
                    SqlExpr::Paren(inner) => {
                        eval_outer_expr(inner, tables, default_table, outer_table, outer_row, outer_alias, outer_table_name)
                    }
                    SqlExpr::Literal(v) => Ok(v.clone()),
                    SqlExpr::Column(c) => Ok(get_outer_value_local(c, outer_table, outer_row, outer_alias, outer_table_name)),
                    SqlExpr::Function { name, args } => {
                        if name.eq_ignore_ascii_case("rand") {
                            if !args.is_empty() {
                                return Err(ApexError::QueryParseError(
                                    "RAND() does not accept arguments".to_string(),
                                ));
                            }
                            Ok(Value::Float64(rand::random::<f64>()))
                        } else if name.eq_ignore_ascii_case("len") {
                            if args.len() != 1 {
                                return Err(ApexError::QueryParseError(
                                    "LEN() expects 1 argument".to_string(),
                                ));
                            }
                            let v = eval_outer_expr(
                                &args[0],
                                tables,
                                default_table,
                                outer_table,
                                outer_row,
                                outer_alias,
                                outer_table_name,
                            )?;
                            if v.is_null() {
                                Ok(Value::Null)
                            } else {
                                Ok(Value::Int64(v.to_string_value().chars().count() as i64))
                            }
                        } else if name.eq_ignore_ascii_case("trim") {
                            if args.len() != 1 {
                                return Err(ApexError::QueryParseError(
                                    "TRIM() expects 1 argument".to_string(),
                                ));
                            }
                            let v = eval_outer_expr(
                                &args[0],
                                tables,
                                default_table,
                                outer_table,
                                outer_row,
                                outer_alias,
                                outer_table_name,
                            )?;
                            if v.is_null() {
                                Ok(Value::Null)
                            } else {
                                Ok(Value::String(v.to_string_value().trim().to_string()))
                            }
                        } else if name.eq_ignore_ascii_case("upper") {
                            if args.len() != 1 {
                                return Err(ApexError::QueryParseError(
                                    "UPPER() expects 1 argument".to_string(),
                                ));
                            }
                            let v = eval_outer_expr(
                                &args[0],
                                tables,
                                default_table,
                                outer_table,
                                outer_row,
                                outer_alias,
                                outer_table_name,
                            )?;
                            if v.is_null() {
                                Ok(Value::Null)
                            } else {
                                Ok(Value::String(v.to_string_value().to_uppercase()))
                            }
                        } else if name.eq_ignore_ascii_case("lower") {
                            if args.len() != 1 {
                                return Err(ApexError::QueryParseError(
                                    "LOWER() expects 1 argument".to_string(),
                                ));
                            }
                            let v = eval_outer_expr(
                                &args[0],
                                tables,
                                default_table,
                                outer_table,
                                outer_row,
                                outer_alias,
                                outer_table_name,
                            )?;
                            if v.is_null() {
                                Ok(Value::Null)
                            } else {
                                Ok(Value::String(v.to_string_value().to_lowercase()))
                            }
                        } else if name.eq_ignore_ascii_case("replace") {
                            if args.len() != 3 {
                                return Err(ApexError::QueryParseError(
                                    "REPLACE() expects 3 arguments".to_string(),
                                ));
                            }
                            let s0 = eval_outer_expr(
                                &args[0],
                                tables,
                                default_table,
                                outer_table,
                                outer_row,
                                outer_alias,
                                outer_table_name,
                            )?;
                            let from0 = eval_outer_expr(
                                &args[1],
                                tables,
                                default_table,
                                outer_table,
                                outer_row,
                                outer_alias,
                                outer_table_name,
                            )?;
                            let to0 = eval_outer_expr(
                                &args[2],
                                tables,
                                default_table,
                                outer_table,
                                outer_row,
                                outer_alias,
                                outer_table_name,
                            )?;
                            if s0.is_null() || from0.is_null() || to0.is_null() {
                                Ok(Value::Null)
                            } else {
                                Ok(Value::String(
                                    s0.to_string_value().replace(&from0.to_string_value(), &to0.to_string_value()),
                                ))
                            }
                        } else if name.eq_ignore_ascii_case("mid") {
                            if args.len() != 2 && args.len() != 3 {
                                return Err(ApexError::QueryParseError(
                                    "MID() expects 2 or 3 arguments".to_string(),
                                ));
                            }
                            let s0 = eval_outer_expr(
                                &args[0],
                                tables,
                                default_table,
                                outer_table,
                                outer_row,
                                outer_alias,
                                outer_table_name,
                            )?;
                            let start0 = eval_outer_expr(
                                &args[1],
                                tables,
                                default_table,
                                outer_table,
                                outer_row,
                                outer_alias,
                                outer_table_name,
                            )?;
                            if s0.is_null() || start0.is_null() {
                                return Ok(Value::Null);
                            }
                            let s = s0.to_string_value();
                            let mut start = start0.as_i64().unwrap_or(1);
                            if start < 1 {
                                start = 1;
                            }
                            let start_idx = (start - 1) as usize;
                            let chars: Vec<char> = s.chars().collect();
                            if start_idx >= chars.len() {
                                return Ok(Value::String(String::new()));
                            }
                            let end_idx = if args.len() == 3 {
                                let len0 = eval_outer_expr(
                                    &args[2],
                                    tables,
                                    default_table,
                                    outer_table,
                                    outer_row,
                                    outer_alias,
                                    outer_table_name,
                                )?;
                                if len0.is_null() {
                                    return Ok(Value::Null);
                                }
                                let mut l = len0.as_i64().unwrap_or(0);
                                if l < 0 {
                                    l = 0;
                                }
                                (start_idx + l as usize).min(chars.len())
                            } else {
                                chars.len()
                            };
                            Ok(Value::String(chars[start_idx..end_idx].iter().collect()))
                        } else {
                            Err(ApexError::QueryParseError(
                                format!("Unsupported function: {}", name),
                            ))
                        }
                    }
                    SqlExpr::UnaryOp { op: UnaryOperator::Minus, expr } => {
                        let v = eval_outer_expr(expr, tables, default_table, outer_table, outer_row, outer_alias, outer_table_name)?;
                        if let Some(i) = v.as_i64() {
                            Ok(Value::Int64(-i))
                        } else if let Some(f) = v.as_f64() {
                            Ok(Value::Float64(-f))
                        } else {
                            Ok(Value::Null)
                        }
                    }
                    SqlExpr::ScalarSubquery { stmt: sub } => eval_scalar_subquery_for_outer_row(
                        sub,
                        tables,
                        default_table,
                        outer_table,
                        outer_row,
                        outer_alias,
                        outer_table_name,
                    ),
                    _ => Err(ApexError::QueryParseError(
                        "Unsupported expression in SELECT list".to_string(),
                    )),
                }
            }

            // Evaluate WHERE to matching indices (row-by-row)
            let row_count = table.get_row_count();
            let deleted = table.deleted_ref();
            let mut matching_indices: Vec<usize> = Vec::new();
            for row_idx in 0..row_count {
                if deleted.get(row_idx) {
                    continue;
                }
                let passes = if let Some(ref w) = stmt.where_clause {
                    eval_outer_where(
                        w,
                        tables,
                        default_table,
                        table,
                        row_idx,
                        &outer_alias,
                        &outer_table_name,
                    )?
                } else {
                    true
                };
                if passes {
                    matching_indices.push(row_idx);
                }
            }

            // Project selected columns (strip outer qualifier)
            let mut result_columns: Vec<String> = Vec::new();
            let mut projected_cols: Vec<String> = Vec::new();
            let mut projected_exprs: Vec<Option<SqlExpr>> = Vec::new();
            for sc in &stmt.columns {
                match sc {
                    SelectColumn::Column(c) => {
                        let name = strip_outer(c, &outer_alias, &outer_table_name);
                        result_columns.push(name.clone());
                        projected_cols.push(name);
                        projected_exprs.push(None);
                    }
                    SelectColumn::ColumnAlias { column, alias } => {
                        let name = strip_outer(column, &outer_alias, &outer_table_name);
                        result_columns.push(alias.clone());
                        projected_cols.push(name);
                        projected_exprs.push(None);
                    }
                    SelectColumn::Expression { expr, alias } => {
                        let out_name = alias.clone().unwrap_or_else(|| "expr".to_string());
                        result_columns.push(out_name);
                        projected_cols.push(String::new());
                        projected_exprs.push(Some(expr.clone()));
                    }
                    _ => {
                        return Err(ApexError::QueryParseError(
                            "EXISTS single-table path only supports simple column projection and scalar subquery expressions".to_string(),
                        ))
                    }
                }
            }

            let schema = table.schema_ref();
            let columns_ref = table.columns_ref();
            let mut rows: Vec<Vec<Value>> = Vec::with_capacity(matching_indices.len());
            for &row_idx in &matching_indices {
                let mut out_row: Vec<Value> = Vec::with_capacity(projected_cols.len());
                for (i, col) in projected_cols.iter().enumerate() {
                    if let Some(expr) = projected_exprs.get(i).and_then(|e| e.clone()) {
                        out_row.push(eval_outer_expr(
                            &expr,
                            tables,
                            default_table,
                            table,
                            row_idx,
                            &outer_alias,
                            &outer_table_name,
                        )?);
                        continue;
                    }

                    if col == "_id" {
                        out_row.push(Value::Int64(row_idx as i64));
                    } else if let Some(ci) = schema.get_index(col) {
                        out_row.push(columns_ref[ci].get(row_idx).unwrap_or(Value::Null));
                    } else {
                        out_row.push(Value::Null);
                    }
                }
                rows.push(out_row);
            }

            // Apply ORDER BY (strip qualifier)
            let mut order_by = stmt.order_by.clone();
            for ob in &mut order_by {
                ob.column = strip_outer(&ob.column, &outer_alias, &outer_table_name);
            }
            if !order_by.is_empty() {
                rows = Self::apply_order_by(rows, &result_columns, &order_by)?;
            }

            let off = stmt.offset.unwrap_or(0);
            let lim = stmt.limit.unwrap_or(usize::MAX);
            let rows = rows.into_iter().skip(off).take(lim).collect::<Vec<_>>();
            return Ok(SqlResult::new(result_columns, rows));
        }

        let from_item = stmt
            .from
            .as_ref()
            .ok_or_else(|| ApexError::QueryParseError("JOIN requires FROM table".to_string()))?;

        let (left_table_name, left_alias) = match from_item {
            FromItem::Table { table, alias } => {
                let t = table.clone();
                let a = alias.clone().unwrap_or_else(|| t.clone());
                (t, a)
            }
            FromItem::Subquery { .. } => {
                return Err(ApexError::QueryParseError(
                    "JOIN with derived table is not supported yet".to_string(),
                ))
            }
        };

        {
            let left_table = tables
                .get_mut(&left_table_name)
                .ok_or_else(|| ApexError::QueryParseError(format!("Table '{}' not found.", left_table_name)))?;
            if left_table.has_pending_writes() {
                left_table.flush_write_buffer();
            }
        }

        // Attempt to push down WHERE to the right side when it only references the right table.
        fn strip_right_only_where(expr: &SqlExpr, left_alias: &str, right_alias: &str) -> Option<SqlExpr> {
            match expr {
                SqlExpr::Paren(inner) => strip_right_only_where(inner, left_alias, right_alias)
                    .map(|e| SqlExpr::Paren(Box::new(e))),
                SqlExpr::BinaryOp { left, op, right } => {
                    // Support AND pushdown
                    if *op == BinaryOperator::And {
                        let l = strip_right_only_where(left, left_alias, right_alias)?;
                        let r = strip_right_only_where(right, left_alias, right_alias)?;
                        return Some(SqlExpr::BinaryOp { left: Box::new(l), op: op.clone(), right: Box::new(r) });
                    }

                    // Comparison: right_alias.col OP literal
                    let col = match left.as_ref() {
                        SqlExpr::Column(c) => c,
                        _ => return None,
                    };
                    let (a, c) = if let Some((aa, cc)) = col.split_once('.') { (aa, cc) } else { ("", col.as_str()) };
                    if a.is_empty() {
                        // Unqualified: ambiguous, don't push down
                        return None;
                    }
                    if a == left_alias {
                        return None;
                    }
                    if a != right_alias {
                        return None;
                    }

                    let rv = match right.as_ref() {
                        SqlExpr::Literal(v) => v.clone(),
                        _ => return None,
                    };

                    Some(SqlExpr::BinaryOp {
                        left: Box::new(SqlExpr::Column(c.to_string())),
                        op: op.clone(),
                        right: Box::new(SqlExpr::Literal(rv)),
                    })
                }
                _ => None,
            }
        }

        if stmt.joins.len() != 1 {
            return Err(ApexError::QueryParseError(
                "Only single JOIN is supported yet".to_string(),
            ));
        }
        let join = &stmt.joins[0];

        let (right_table_name, right_alias) = match &join.right {
            FromItem::Table { table, alias } => {
                let t = table.clone();
                let a = alias.clone().unwrap_or_else(|| t.clone());
                (t, a)
            }
            FromItem::Subquery { .. } => {
                return Err(ApexError::QueryParseError(
                    "JOIN with derived table is not supported yet".to_string(),
                ))
            }
        };

        {
            let right_table = tables
                .get_mut(&right_table_name)
                .ok_or_else(|| ApexError::QueryParseError(format!("Table '{}' not found.", right_table_name)))?;
            if right_table.has_pending_writes() {
                right_table.flush_write_buffer();
            }
        }

        let left_table = tables
            .get(&left_table_name)
            .ok_or_else(|| ApexError::QueryParseError(format!("Table '{}' not found.", left_table_name)))?;
        let right_table = tables
            .get(&right_table_name)
            .ok_or_else(|| ApexError::QueryParseError(format!("Table '{}' not found.", right_table_name)))?;

        #[derive(Clone, Copy)]
        struct JoinRow {
            left: usize,
            right: Option<usize>,
        }

        fn split_qual(col: &str) -> (&str, &str) {
            if let Some((a, b)) = col.split_once('.') {
                (a, b)
            } else {
                ("", col)
            }
        }

        fn get_col_value(table: &ColumnTable, col: &str, row_idx: usize) -> Value {
            if col == "_id" {
                return Value::Int64(row_idx as i64);
            }
            let schema = table.schema_ref();
            if let Some(ci) = schema.get_index(col) {
                table.columns_ref()[ci].get(row_idx).unwrap_or(Value::Null)
            } else {
                Value::Null
            }
        }

        fn value_for_ref(
            col_ref: &str,
            left_table: &ColumnTable,
            right_table: &ColumnTable,
            left_alias: &str,
            right_alias: &str,
            jr: JoinRow,
        ) -> Value {
            let (a, c) = split_qual(col_ref);
            if a.is_empty() {
                // Unqualified column: resolve by schema presence.
                // If only one side contains the column, use that side.
                // If both contain it, default to left for backward-compat.
                let l_has = left_table.schema_ref().get_index(c).is_some() || c == "_id";
                let r_has = right_table.schema_ref().get_index(c).is_some() || c == "_id";
                if r_has && !l_has {
                    return jr
                        .right
                        .map(|ri| get_col_value(right_table, c, ri))
                        .unwrap_or(Value::Null);
                }
                return get_col_value(left_table, c, jr.left);
            }
            if a == left_alias {
                get_col_value(left_table, c, jr.left)
            } else if a == right_alias {
                jr.right
                    .map(|ri| get_col_value(right_table, c, ri))
                    .unwrap_or(Value::Null)
            } else {
                Value::Null
            }
        }

        fn eval_predicate(
            expr: &SqlExpr,
            left_table: &ColumnTable,
            right_table: &ColumnTable,
            left_alias: &str,
            right_alias: &str,
            jr: JoinRow,
        ) -> Result<bool, ApexError> {
            fn eval_scalar(
                expr: &SqlExpr,
                left_table: &ColumnTable,
                right_table: &ColumnTable,
                left_alias: &str,
                right_alias: &str,
                jr: JoinRow,
            ) -> Result<Value, ApexError> {
                match expr {
                    SqlExpr::Paren(inner) => eval_scalar(inner, left_table, right_table, left_alias, right_alias, jr),
                    SqlExpr::Literal(v) => Ok(v.clone()),
                    SqlExpr::Column(c) => Ok(value_for_ref(c, left_table, right_table, left_alias, right_alias, jr)),
                    SqlExpr::Function { name, args } => {
                        if name.eq_ignore_ascii_case("rand") {
                            if !args.is_empty() {
                                return Err(ApexError::QueryParseError(
                                    "RAND() does not accept arguments".to_string(),
                                ));
                            }
                            return Ok(Value::Float64(rand::random::<f64>()));
                        }
                        if name.eq_ignore_ascii_case("len") {
                            if args.len() != 1 {
                                return Err(ApexError::QueryParseError(
                                    "LEN() expects 1 argument".to_string(),
                                ));
                            }
                            let v = eval_scalar(&args[0], left_table, right_table, left_alias, right_alias, jr)?;
                            if v.is_null() {
                                return Ok(Value::Null);
                            }
                            let s = v.to_string_value();
                            return Ok(Value::Int64(s.chars().count() as i64));
                        }
                        if name.eq_ignore_ascii_case("trim") {
                            if args.len() != 1 {
                                return Err(ApexError::QueryParseError(
                                    "TRIM() expects 1 argument".to_string(),
                                ));
                            }
                            let v = eval_scalar(&args[0], left_table, right_table, left_alias, right_alias, jr)?;
                            if v.is_null() {
                                return Ok(Value::Null);
                            }
                            return Ok(Value::String(v.to_string_value().trim().to_string()));
                        }
                        if name.eq_ignore_ascii_case("upper") {
                            if args.len() != 1 {
                                return Err(ApexError::QueryParseError(
                                    "UPPER() expects 1 argument".to_string(),
                                ));
                            }
                            let v = eval_scalar(&args[0], left_table, right_table, left_alias, right_alias, jr)?;
                            if v.is_null() {
                                return Ok(Value::Null);
                            }
                            return Ok(Value::String(v.to_string_value().to_uppercase()));
                        }
                        if name.eq_ignore_ascii_case("lower") {
                            if args.len() != 1 {
                                return Err(ApexError::QueryParseError(
                                    "LOWER() expects 1 argument".to_string(),
                                ));
                            }
                            let v = eval_scalar(&args[0], left_table, right_table, left_alias, right_alias, jr)?;
                            if v.is_null() {
                                return Ok(Value::Null);
                            }
                            return Ok(Value::String(v.to_string_value().to_lowercase()));
                        }
                        if name.eq_ignore_ascii_case("replace") {
                            if args.len() != 3 {
                                return Err(ApexError::QueryParseError(
                                    "REPLACE() expects 3 arguments".to_string(),
                                ));
                            }
                            let s0 = eval_scalar(&args[0], left_table, right_table, left_alias, right_alias, jr)?;
                            let from0 = eval_scalar(&args[1], left_table, right_table, left_alias, right_alias, jr)?;
                            let to0 = eval_scalar(&args[2], left_table, right_table, left_alias, right_alias, jr)?;
                            if s0.is_null() || from0.is_null() || to0.is_null() {
                                return Ok(Value::Null);
                            }
                            return Ok(Value::String(
                                s0.to_string_value().replace(&from0.to_string_value(), &to0.to_string_value()),
                            ));
                        }
                        if name.eq_ignore_ascii_case("mid") {
                            if args.len() != 2 && args.len() != 3 {
                                return Err(ApexError::QueryParseError(
                                    "MID() expects 2 or 3 arguments".to_string(),
                                ));
                            }
                            let s0 = eval_scalar(&args[0], left_table, right_table, left_alias, right_alias, jr)?;
                            let start0 = eval_scalar(&args[1], left_table, right_table, left_alias, right_alias, jr)?;
                            if s0.is_null() || start0.is_null() {
                                return Ok(Value::Null);
                            }
                            let s = s0.to_string_value();
                            let mut start = start0.as_i64().unwrap_or(1);
                            if start < 1 {
                                start = 1;
                            }
                            let start_idx = (start - 1) as usize;
                            let chars: Vec<char> = s.chars().collect();
                            if start_idx >= chars.len() {
                                return Ok(Value::String(String::new()));
                            }
                            let end_idx = if args.len() == 3 {
                                let len0 = eval_scalar(&args[2], left_table, right_table, left_alias, right_alias, jr)?;
                                if len0.is_null() {
                                    return Ok(Value::Null);
                                }
                                let mut l = len0.as_i64().unwrap_or(0);
                                if l < 0 {
                                    l = 0;
                                }
                                (start_idx + l as usize).min(chars.len())
                            } else {
                                chars.len()
                            };
                            return Ok(Value::String(chars[start_idx..end_idx].iter().collect()));
                        }
                        Err(ApexError::QueryParseError(
                            format!("Unsupported function: {}", name),
                        ))
                    }
                    SqlExpr::UnaryOp { op, expr } => match op {
                        UnaryOperator::Not => {
                            let v = eval_predicate(expr, left_table, right_table, left_alias, right_alias, jr)?;
                            Ok(Value::Bool(!v))
                        }
                        UnaryOperator::Minus => {
                            let v = eval_scalar(expr, left_table, right_table, left_alias, right_alias, jr)?;
                            if let Some(i) = v.as_i64() {
                                Ok(Value::Int64(-i))
                            } else if let Some(f) = v.as_f64() {
                                Ok(Value::Float64(-f))
                            } else {
                                Ok(Value::Null)
                            }
                        }
                    },
                    SqlExpr::BinaryOp { left, op, right } => match op {
                        BinaryOperator::Add | BinaryOperator::Sub | BinaryOperator::Mul | BinaryOperator::Div | BinaryOperator::Mod => {
                            let lv = eval_scalar(left, left_table, right_table, left_alias, right_alias, jr)?;
                            let rv = eval_scalar(right, left_table, right_table, left_alias, right_alias, jr)?;
                            if let (Some(a), Some(b)) = (lv.as_f64(), rv.as_f64()) {
                                let out = match op {
                                    BinaryOperator::Add => a + b,
                                    BinaryOperator::Sub => a - b,
                                    BinaryOperator::Mul => a * b,
                                    BinaryOperator::Div => a / b,
                                    BinaryOperator::Mod => a % b,
                                    _ => a,
                                };
                                Ok(Value::Float64(out))
                            } else {
                                Ok(Value::Null)
                            }
                        }
                        _ => Ok(Value::Null),
                    },
                    _ => Ok(Value::Null),
                }
            }

            match expr {
                SqlExpr::Paren(inner) => eval_predicate(inner, left_table, right_table, left_alias, right_alias, jr),
                SqlExpr::Literal(v) => Ok(v.as_bool().unwrap_or(false)),
                SqlExpr::UnaryOp { op: UnaryOperator::Not, expr } => Ok(!eval_predicate(expr, left_table, right_table, left_alias, right_alias, jr)?),
                SqlExpr::BinaryOp { left, op, right } => match op {
                    BinaryOperator::And => Ok(
                        eval_predicate(left, left_table, right_table, left_alias, right_alias, jr)?
                            && eval_predicate(right, left_table, right_table, left_alias, right_alias, jr)?,
                    ),
                    BinaryOperator::Or => Ok(
                        eval_predicate(left, left_table, right_table, left_alias, right_alias, jr)?
                            || eval_predicate(right, left_table, right_table, left_alias, right_alias, jr)?,
                    ),
                    BinaryOperator::Eq
                    | BinaryOperator::NotEq
                    | BinaryOperator::Lt
                    | BinaryOperator::Le
                    | BinaryOperator::Gt
                    | BinaryOperator::Ge => {
                        let lv = eval_scalar(left, left_table, right_table, left_alias, right_alias, jr)?;
                        let rv = eval_scalar(right, left_table, right_table, left_alias, right_alias, jr)?;

                        if lv.is_null() || rv.is_null() {
                            return Ok(false);
                        }

                        let ord = lv.partial_cmp(&rv).unwrap_or(Ordering::Equal);
                        Ok(match op {
                            BinaryOperator::Eq => ord == Ordering::Equal,
                            BinaryOperator::NotEq => ord != Ordering::Equal,
                            BinaryOperator::Lt => ord == Ordering::Less,
                            BinaryOperator::Le => ord != Ordering::Greater,
                            BinaryOperator::Gt => ord == Ordering::Greater,
                            BinaryOperator::Ge => ord != Ordering::Less,
                            _ => false,
                        })
                    }
                    _ => Err(ApexError::QueryParseError(
                        "Unsupported operator in JOIN predicate".to_string(),
                    )),
                },
                _ => Err(ApexError::QueryParseError(
                    "Unsupported expression in JOIN predicate".to_string(),
                )),
            }
        }

        let (left_key_ref, right_key_ref) = match &join.on {
            SqlExpr::BinaryOp { left, op: BinaryOperator::Eq, right } => {
                let l = match left.as_ref() {
                    SqlExpr::Column(c) => c.clone(),
                    _ => {
                        return Err(ApexError::QueryParseError(
                            "JOIN ON must be column = column".to_string(),
                        ))
                    }
                };
                let r = match right.as_ref() {
                    SqlExpr::Column(c) => c.clone(),
                    _ => {
                        return Err(ApexError::QueryParseError(
                            "JOIN ON must be column = column".to_string(),
                        ))
                    }
                };
                (l, r)
            }
            _ => {
                return Err(ApexError::QueryParseError(
                    "Only equi-join ON a=b is supported yet".to_string(),
                ))
            }
        };

        let (lk_alias0, lk_col) = split_qual(&left_key_ref);
        let (rk_alias0, rk_col) = split_qual(&right_key_ref);

        let lk_alias = if lk_alias0.is_empty() { left_alias.as_str() } else { lk_alias0 };
        let rk_alias = if rk_alias0.is_empty() { right_alias.as_str() } else { rk_alias0 };

        let left_matches = lk_alias == left_alias.as_str() || lk_alias == left_table_name.as_str();
        let right_matches = rk_alias == right_alias.as_str() || rk_alias == right_table_name.as_str();
        let left_swapped = lk_alias == right_alias.as_str() || lk_alias == right_table_name.as_str();
        let right_swapped = rk_alias == left_alias.as_str() || rk_alias == left_table_name.as_str();

        let (left_key_col, right_key_col) = if left_matches && right_matches {
            (lk_col.to_string(), rk_col.to_string())
        } else if left_swapped && right_swapped {
            (rk_col.to_string(), lk_col.to_string())
        } else {
            return Err(ApexError::QueryParseError(
                "JOIN ON must reference left and right table aliases".to_string(),
            ));
        };

        let right_row_count = right_table.get_row_count();
        let right_deleted = right_table.deleted_ref();

        let mut right_allowed: Option<Vec<bool>> = None;
        let mut where_pushed_down = false;
        if let Some(ref where_expr) = stmt.where_clause {
            if let Some(stripped) = strip_right_only_where(where_expr, &left_alias, &right_alias) {
                let row_count = right_table.get_row_count();
                let idxs = Self::evaluate_where(&stripped, right_table)?;
                let mut allowed = vec![false; row_count];
                for i in idxs {
                    if i < allowed.len() {
                        allowed[i] = true;
                    }
                }
                right_allowed = Some(allowed);
                where_pushed_down = true;
            }
        }
        let mut hash: HashMap<JoinKey, Vec<usize>> = HashMap::new();
        for ri in 0..right_row_count {
            if right_deleted.get(ri) {
                continue;
            }
            if let Some(ref allowed) = right_allowed {
                if ri >= allowed.len() || !allowed[ri] {
                    continue;
                }
            }
            let v = get_col_value(right_table, &right_key_col, ri);
            if v.is_null() {
                continue;
            }
            hash.entry(join_key(&v)).or_default().push(ri);
        }

        let left_row_count = left_table.get_row_count();
        let left_deleted = left_table.deleted_ref();
        let mut joined: Vec<JoinRow> = Vec::new();
        for li in 0..left_row_count {
            if left_deleted.get(li) {
                continue;
            }
            let lv = get_col_value(left_table, &left_key_col, li);
            let matches = if lv.is_null() { None } else { hash.get(&join_key(&lv)) };

            match join.join_type {
                crate::query::JoinType::Inner => {
                    if let Some(rs) = matches {
                        for &r in rs {
                            joined.push(JoinRow { left: li, right: Some(r) });
                        }
                    }
                }
                crate::query::JoinType::Left => {
                    if let Some(rs) = matches {
                        for &r in rs {
                            joined.push(JoinRow { left: li, right: Some(r) });
                        }
                    } else {
                        joined.push(JoinRow { left: li, right: None });
                    }
                }
            }
        }

        if !where_pushed_down {
            if let Some(ref where_expr) = stmt.where_clause {
                let mut filtered: Vec<JoinRow> = Vec::with_capacity(joined.len());
                for jr in joined {
                    if eval_predicate(where_expr, left_table, right_table, &left_alias, &right_alias, jr)? {
                        filtered.push(jr);
                    }
                }
                joined = filtered;
            }
        }

        let has_aggs = stmt.columns.iter().any(|c| matches!(c, SelectColumn::Aggregate { .. }));
        if !stmt.group_by.is_empty() || has_aggs {
            use std::collections::HashSet;

            struct JoinAgg {
                func: AggregateFunc,
                distinct: bool,
                seen: Option<HashSet<Vec<u8>>>,
                count: i64,
                sum: f64,
            }

            impl JoinAgg {
                fn new(func: AggregateFunc, distinct: bool) -> Self {
                    let seen = if matches!(func, AggregateFunc::Count) && distinct {
                        Some(HashSet::new())
                    } else {
                        None
                    };
                    Self { func, distinct, seen, count: 0, sum: 0.0 }
                }
            }

            #[derive(Default)]
            struct GroupState {
                first: Option<JoinRow>,
                aggs: Vec<JoinAgg>,
            }

            // Build aggregate spec list in SELECT order
            let mut agg_specs: Vec<(AggregateFunc, Option<String>, bool, Option<String>)> = Vec::new();
            for c in &stmt.columns {
                if let SelectColumn::Aggregate { func, column, distinct, alias } = c {
                    agg_specs.push((func.clone(), column.clone(), *distinct, alias.clone()));
                }
            }

            // Group by key is serialized bytes of each key part.
            let mut groups: HashMap<String, GroupState> = HashMap::new();
            for jr in &joined {
                let mut key_parts: Vec<String> = Vec::with_capacity(stmt.group_by.len());
                for gb in &stmt.group_by {
                    let v = value_for_ref(gb, left_table, right_table, &left_alias, &right_alias, *jr);
                    key_parts.push(v.to_string_value());
                }
                let gk = key_parts.join("\u{1f}");
                let entry = groups.entry(gk).or_insert_with(|| {
                    let mut gs = GroupState::default();
                    gs.first = Some(*jr);
                    gs.aggs = agg_specs.iter().map(|(f, _, d, _)| JoinAgg::new(f.clone(), *d)).collect();
                    gs
                });

                for (i, (func, col, distinct, _)) in agg_specs.iter().enumerate() {
                    let agg = &mut entry.aggs[i];
                    match func {
                        AggregateFunc::Count => {
                            if *distinct {
                                if let Some(cn) = col.as_ref() {
                                    let v = value_for_ref(cn, left_table, right_table, &left_alias, &right_alias, *jr);
                                    if !v.is_null() {
                                        if let Some(set) = agg.seen.as_mut() {
                                            set.insert(v.to_bytes());
                                        }
                                    }
                                }
                            } else if col.is_none() {
                                agg.count += 1;
                            } else if let Some(cn) = col.as_ref() {
                                let v = value_for_ref(cn, left_table, right_table, &left_alias, &right_alias, *jr);
                                if !v.is_null() {
                                    agg.count += 1;
                                }
                            }
                        }
                        AggregateFunc::Sum | AggregateFunc::Avg => {
                            if let Some(cn) = col.as_ref() {
                                let v = value_for_ref(cn, left_table, right_table, &left_alias, &right_alias, *jr);
                                if let Some(n) = v.as_f64() {
                                    agg.sum += n;
                                    agg.count += 1;
                                }
                            }
                        }
                        _ => {}
                    }
                }
            }

            // HAVING evaluation on group state
            fn eval_having_scalar_join(
                expr: &SqlExpr,
                gs: &GroupState,
                agg_specs: &[(AggregateFunc, Option<String>, bool, Option<String>)],
                left_table: &ColumnTable,
                right_table: &ColumnTable,
                left_alias: &str,
                right_alias: &str,
            ) -> Result<Value, ApexError> {
                match expr {
                    SqlExpr::Paren(inner) => eval_having_scalar_join(inner, gs, agg_specs, left_table, right_table, left_alias, right_alias),
                    SqlExpr::Literal(v) => Ok(v.clone()),
                    SqlExpr::Column(c) => {
                        if let Some(jr) = gs.first {
                            Ok(value_for_ref(c, left_table, right_table, left_alias, right_alias, jr))
                        } else {
                            Ok(Value::Null)
                        }
                    }
                    SqlExpr::Function { name, args } => {
                        if name.eq_ignore_ascii_case("rand") {
                            if !args.is_empty() {
                                return Err(ApexError::QueryParseError(
                                    "RAND() does not accept arguments".to_string(),
                                ));
                            }
                            return Ok(Value::Float64(rand::random::<f64>()));
                        }
                        let func = match name.to_uppercase().as_str() {
                            "COUNT" => AggregateFunc::Count,
                            "SUM" => AggregateFunc::Sum,
                            "AVG" => AggregateFunc::Avg,
                            _ => {
                                return Err(ApexError::QueryParseError(
                                    format!("Unsupported function in HAVING: {}", name),
                                ))
                            }
                        };

                        let col = if args.is_empty() {
                            None
                        } else {
                            match &args[0] {
                                SqlExpr::Column(c) => Some(c.clone()),
                                _ => None,
                            }
                        };

                        let col_d = col.as_deref();
                        let col_base = col_d.map(|s| split_qual(s).1);
                        for (i, (sf, sc, sd, _)) in agg_specs.iter().enumerate() {
                            let sc_d = sc.as_deref();
                            let sc_base = sc_d.map(|s| split_qual(s).1);
                            let col_match = sc_d == col_d || (sc_base.is_some() && col_base.is_some() && sc_base == col_base);

                            if *sf == func && col_match && *sd == false {
                                // Non-distinct aggregate match
                                let a = &gs.aggs[i];
                                return Ok(match sf {
                                    AggregateFunc::Count => Value::Int64(a.count),
                                    AggregateFunc::Sum => Value::Float64(a.sum),
                                    AggregateFunc::Avg => {
                                        if a.count > 0 {
                                            Value::Float64(a.sum / a.count as f64)
                                        } else {
                                            Value::Null
                                        }
                                    }
                                    _ => Value::Null,
                                });
                            }
                            if *sf == func && col_match && *sd {
                                // Distinct COUNT match
                                let a = &gs.aggs[i];
                                return Ok(Value::Int64(a.seen.as_ref().map(|s| s.len()).unwrap_or(0) as i64));
                            }
                        }
                        Ok(Value::Null)
                    }
                    SqlExpr::UnaryOp { op, expr } => {
                        match op {
                            UnaryOperator::Not => {
                                let v = eval_having_scalar_join(expr, gs, agg_specs, left_table, right_table, left_alias, right_alias)?;
                                Ok(Value::Bool(!v.as_bool().unwrap_or(false)))
                            }
                            UnaryOperator::Minus => {
                                let v = eval_having_scalar_join(expr, gs, agg_specs, left_table, right_table, left_alias, right_alias)?;
                                if let Some(i) = v.as_i64() {
                                    Ok(Value::Int64(-i))
                                } else if let Some(f) = v.as_f64() {
                                    Ok(Value::Float64(-f))
                                } else {
                                    Ok(Value::Null)
                                }
                            }
                        }
                    }
                    SqlExpr::BinaryOp { left, op, right } => {
                        let lv = eval_having_scalar_join(left, gs, agg_specs, left_table, right_table, left_alias, right_alias)?;
                        let rv = eval_having_scalar_join(right, gs, agg_specs, left_table, right_table, left_alias, right_alias)?;
                        match op {
                            BinaryOperator::And => Ok(Value::Bool(lv.as_bool().unwrap_or(false) && rv.as_bool().unwrap_or(false))),
                            BinaryOperator::Or => Ok(Value::Bool(lv.as_bool().unwrap_or(false) || rv.as_bool().unwrap_or(false))),
                            BinaryOperator::Eq => Ok(Value::Bool(lv == rv)),
                            BinaryOperator::NotEq => Ok(Value::Bool(lv != rv)),
                            BinaryOperator::Lt | BinaryOperator::Le | BinaryOperator::Gt | BinaryOperator::Ge => {
                                if lv.is_null() || rv.is_null() {
                                    return Ok(Value::Bool(false));
                                }
                                let ord = lv.partial_cmp(&rv).unwrap_or(Ordering::Equal);
                                let b = match op {
                                    BinaryOperator::Lt => ord == Ordering::Less,
                                    BinaryOperator::Le => ord != Ordering::Greater,
                                    BinaryOperator::Gt => ord == Ordering::Greater,
                                    BinaryOperator::Ge => ord != Ordering::Less,
                                    _ => false,
                                };
                                Ok(Value::Bool(b))
                            }
                            _ => Ok(Value::Null),
                        }
                    }
                    _ => Ok(Value::Null),
                }
            }

            fn eval_having_predicate_join(
                expr: &SqlExpr,
                gs: &GroupState,
                agg_specs: &[(AggregateFunc, Option<String>, bool, Option<String>)],
                left_table: &ColumnTable,
                right_table: &ColumnTable,
                left_alias: &str,
                right_alias: &str,
            ) -> Result<bool, ApexError> {
                match expr {
                    SqlExpr::Paren(inner) => eval_having_predicate_join(inner, gs, agg_specs, left_table, right_table, left_alias, right_alias),
                    SqlExpr::Literal(v) => Ok(v.as_bool().unwrap_or(false)),
                    SqlExpr::UnaryOp { op: UnaryOperator::Not, expr } => Ok(!eval_having_predicate_join(expr, gs, agg_specs, left_table, right_table, left_alias, right_alias)?),
                    SqlExpr::BinaryOp { left, op, right } => match op {
                        BinaryOperator::And => Ok(
                            eval_having_predicate_join(left, gs, agg_specs, left_table, right_table, left_alias, right_alias)?
                                && eval_having_predicate_join(right, gs, agg_specs, left_table, right_table, left_alias, right_alias)?,
                        ),
                        BinaryOperator::Or => Ok(
                            eval_having_predicate_join(left, gs, agg_specs, left_table, right_table, left_alias, right_alias)?
                                || eval_having_predicate_join(right, gs, agg_specs, left_table, right_table, left_alias, right_alias)?,
                        ),
                        BinaryOperator::Eq
                        | BinaryOperator::NotEq
                        | BinaryOperator::Lt
                        | BinaryOperator::Le
                        | BinaryOperator::Gt
                        | BinaryOperator::Ge => {
                            let lv = eval_having_scalar_join(left, gs, agg_specs, left_table, right_table, left_alias, right_alias)?;
                            let rv = eval_having_scalar_join(right, gs, agg_specs, left_table, right_table, left_alias, right_alias)?;
                            if lv.is_null() || rv.is_null() {
                                return Ok(false);
                            }
                            let ord = lv.partial_cmp(&rv).unwrap_or(Ordering::Equal);
                            Ok(match op {
                                BinaryOperator::Eq => ord == Ordering::Equal,
                                BinaryOperator::NotEq => ord != Ordering::Equal,
                                BinaryOperator::Lt => ord == Ordering::Less,
                                BinaryOperator::Le => ord != Ordering::Greater,
                                BinaryOperator::Gt => ord == Ordering::Greater,
                                BinaryOperator::Ge => ord != Ordering::Less,
                                _ => false,
                            })
                        }
                        _ => {
                            let v = eval_having_scalar_join(expr, gs, agg_specs, left_table, right_table, left_alias, right_alias)?;
                            Ok(v.as_bool().unwrap_or(false))
                        }
                    },
                    _ => {
                        let v = eval_having_scalar_join(expr, gs, agg_specs, left_table, right_table, left_alias, right_alias)?;
                        Ok(v.as_bool().unwrap_or(false))
                    }
                }
            }

            let mut out_rows: Vec<Vec<Value>> = Vec::new();
            for (_k, gs) in groups {
                if let Some(ref having_expr) = stmt.having {
                    if !eval_having_predicate_join(having_expr, &gs, &agg_specs, left_table, right_table, &left_alias, &right_alias)? {
                        continue;
                    }
                }

                let mut row: Vec<Value> = Vec::with_capacity(stmt.columns.len());
                for c in &stmt.columns {
                    match c {
                        SelectColumn::Column(name) => {
                            if let Some(jr) = gs.first {
                                row.push(value_for_ref(name, left_table, right_table, &left_alias, &right_alias, jr));
                            } else {
                                row.push(Value::Null);
                            }
                        }
                        SelectColumn::ColumnAlias { column, .. } => {
                            if let Some(jr) = gs.first {
                                row.push(value_for_ref(column, left_table, right_table, &left_alias, &right_alias, jr));
                            } else {
                                row.push(Value::Null);
                            }
                        }
                        SelectColumn::Aggregate { func, column, distinct, .. } => {
                            let idx = agg_specs.iter().position(|(f, c, d, _)| f == func && c == column && d == distinct);
                            if let Some(i) = idx {
                                let a = &gs.aggs[i];
                                match func {
                                    AggregateFunc::Count => {
                                        if *distinct {
                                            row.push(Value::Int64(a.seen.as_ref().map(|s| s.len()).unwrap_or(0) as i64));
                                        } else {
                                            row.push(Value::Int64(a.count));
                                        }
                                    }
                                    AggregateFunc::Sum => row.push(Value::Float64(a.sum)),
                                    AggregateFunc::Avg => {
                                        if a.count > 0 {
                                            row.push(Value::Float64(a.sum / a.count as f64));
                                        } else {
                                            row.push(Value::Null);
                                        }
                                    }
                                    _ => row.push(Value::Null),
                                }
                            } else {
                                row.push(Value::Null);
                            }
                        }
                        _ => {
                            return Err(ApexError::QueryParseError(
                                "Unsupported SELECT item in JOIN GROUP BY".to_string(),
                            ))
                        }
                    }
                }
                out_rows.push(row);
            }

            let mut out_columns: Vec<String> = Vec::new();
            for c in &stmt.columns {
                match c {
                    SelectColumn::Column(name) => out_columns.push(split_qual(name).1.to_string()),
                    SelectColumn::ColumnAlias { alias, .. } => out_columns.push(alias.clone()),
                    SelectColumn::Aggregate { func, column, distinct, alias } => {
                        let nm = alias.clone().unwrap_or_else(|| {
                            let func_name = match func {
                                AggregateFunc::Count => "COUNT",
                                AggregateFunc::Sum => "SUM",
                                AggregateFunc::Avg => "AVG",
                                AggregateFunc::Min => "MIN",
                                AggregateFunc::Max => "MAX",
                            };
                            if let Some(cn) = column {
                                if *distinct {
                                    format!("{}(DISTINCT {})", func_name, cn)
                                } else {
                                    format!("{}({})", func_name, cn)
                                }
                            } else {
                                format!("{}(*)", func_name)
                            }
                        });
                        out_columns.push(nm);
                    }
                    _ => out_columns.push("expr".to_string()),
                }
            }

            if !stmt.order_by.is_empty() {
                out_rows.sort_by(|a, b| {
                    for ob in &stmt.order_by {
                        let idx = out_columns.iter().position(|c| c == &ob.column).unwrap_or(0);
                        let av = a.get(idx);
                        let bv = b.get(idx);
                        let cmp = SqlExecutor::compare_values(av, bv, ob.nulls_first);
                        let cmp = if ob.descending { cmp.reverse() } else { cmp };
                        if cmp != Ordering::Equal {
                            return cmp;
                        }
                    }
                    Ordering::Equal
                });
            }

            let offset = stmt.offset.unwrap_or(0);
            let limit = stmt.limit.unwrap_or(usize::MAX);
            let out_rows = out_rows.into_iter().skip(offset).take(limit).collect::<Vec<_>>();

            return Ok(SqlResult::new(out_columns, out_rows));
        }

        if !stmt.order_by.is_empty() {
            joined.sort_by(|a, b| {
                for ob in &stmt.order_by {
                    let col_ref = &ob.column;
                    let av = value_for_ref(col_ref, left_table, right_table, &left_alias, &right_alias, *a);
                    let bv = value_for_ref(col_ref, left_table, right_table, &left_alias, &right_alias, *b);
                    let cmp = SqlExecutor::compare_values(Some(&av), Some(&bv), ob.nulls_first);
                    let cmp = if ob.descending { cmp.reverse() } else { cmp };
                    if cmp != Ordering::Equal {
                        return cmp;
                    }
                }
                Ordering::Equal
            });
        }

        let offset = stmt.offset.unwrap_or(0);
        let limit = stmt.limit.unwrap_or(usize::MAX);
        let joined = joined.into_iter().skip(offset).take(limit).collect::<Vec<_>>();

        let mut out_columns: Vec<String> = Vec::new();
        for c in &stmt.columns {
            match c {
                SelectColumn::Column(name) => {
                    out_columns.push(split_qual(name).1.to_string());
                }
                SelectColumn::ColumnAlias { alias, .. } => out_columns.push(alias.clone()),
                _ => {
                    return Err(ApexError::QueryParseError(
                        "Only plain column projection is supported for JOIN yet".to_string(),
                    ))
                }
            }
        }

        let mut rows: Vec<Vec<Value>> = Vec::with_capacity(joined.len());
        for jr in joined {
            let mut row: Vec<Value> = Vec::with_capacity(stmt.columns.len());
            for c in &stmt.columns {
                match c {
                    SelectColumn::Column(name) => {
                        row.push(value_for_ref(name, left_table, right_table, &left_alias, &right_alias, jr));
                    }
                    SelectColumn::ColumnAlias { column, .. } => {
                        row.push(value_for_ref(column, left_table, right_table, &left_alias, &right_alias, jr));
                    }
                    _ => {}
                }
            }
            rows.push(row);
        }

        if stmt.distinct {
            let mut seen: std::collections::HashSet<String> = std::collections::HashSet::new();
            rows.retain(|r| {
                let k = r.iter().map(|v| v.to_string_value()).collect::<Vec<_>>().join("\u{1f}");
                seen.insert(k)
            });
        }

        Ok(SqlResult::new(out_columns, rows))
    }
    
    /// Execute SELECT statement - ULTRA-OPTIMIZED
    fn execute_select(stmt: SelectStatement, table: &mut ColumnTable) -> Result<SqlResult, ApexError> {
        if !stmt.joins.is_empty() {
            return Err(ApexError::QueryParseError(
                "JOIN requires multi-table execution".to_string(),
            ));
        }
        // Table name validation is handled at the binding layer
        // which can access all tables and route to the correct one
        
        // Only flush if there might be pending writes (lazy flush)
        if table.has_pending_writes() {
            table.flush_write_buffer();
        }
        
        // Pre-compute flags
        let has_aggregates = stmt.columns.iter().any(|c| matches!(c, SelectColumn::Aggregate { .. }));
        let has_window = stmt.columns.iter().any(|c| matches!(c, SelectColumn::WindowFunction { .. }));
        let no_group_by = stmt.group_by.is_empty();

        // ============ UNIFIED SIMPLE PIPELINE (GUARDED) ============
        // Only handle a very small subset here to avoid regressions.
        if !has_aggregates
            && !has_window
            && no_group_by
            && stmt.having.is_none()
            && simple_pipeline::is_eligible(&stmt)
        {
            // Pipeline expects an immutable table reference (no mutation)
            match simple_pipeline::execute(&stmt, table) {
                Ok(r) => return Ok(r),
                Err(ApexError::QueryParseError(msg))
                    if msg == "not eligible for simple select plan" =>
                {
                    // Fall back to legacy paths (supports complex WHERE expressions).
                }
                Err(e) => return Err(e),
            }
        }

        match crate::query::engine::PlanOptimizer::try_build_plan(&stmt, table) {
            Ok(Some(plan)) => return crate::query::engine::PlanExecutor::execute_sql_result(table, &plan),
            Ok(None) => {}
            Err(ApexError::QueryParseError(msg)) if msg == "not eligible for select plan" => {
                // Fall back to legacy paths (supports complex WHERE expressions).
            }
            Err(e) => return Err(e),
        }

        // ============ PATHS REQUIRING INDEX COLLECTION ============
        
        // Step 1: Get matching row indices based on WHERE clause
        let matching_indices: Vec<usize> = if let Some(ref where_expr) = stmt.where_clause {
            Self::evaluate_where(where_expr, table)?
        } else {
            // All non-deleted rows (only reached for complex queries)
            let deleted = table.deleted_ref();
            let row_count = table.get_row_count();
            (0..row_count).filter(|&i| !deleted.get(i)).collect()
        };

        // Window functions (minimal support): row_number() over(partition by ... order by ...)
        if has_window {
            if has_aggregates {
                return Err(ApexError::QueryParseError("Window functions with aggregates are not supported".to_string()));
            }
            if !no_group_by {
                return Err(ApexError::QueryParseError("Window functions with GROUP BY are not supported".to_string()));
            }
            if stmt.distinct {
                return Err(ApexError::QueryParseError("Window functions with DISTINCT are not supported".to_string()));
            }

            return Self::execute_window_row_number(&stmt, &matching_indices, table);
        }
        
        // Step 2: Aggregates / GROUP BY remain on legacy paths for now.
        // (Non-aggregate single-table SELECTs are handled by PlanOptimizer above.)
        let (result_columns, column_indices) = Self::resolve_columns(&stmt.columns, table)?;

        if has_aggregates && no_group_by {
            return Self::execute_aggregate(&stmt, &matching_indices, table);
        }

        if !no_group_by {
            let plan = crate::query::engine::PlanBuilder::build_group_by_plan(&stmt)?;
            return crate::query::engine::PlanExecutor::execute_sql_result(table, &plan);
        }

        // Fallback: row materialization for legacy-only features.
        // (Kept minimal; should be eliminated as remaining features are plan-ified.)
        let mut final_indices: Vec<usize> = matching_indices;

        // Preserve ORDER BY + LIMIT/OFFSET semantics even on legacy fallback.
        if !stmt.order_by.is_empty() {
            let offset = stmt.offset.unwrap_or(0);
            let limit = stmt.limit.unwrap_or(usize::MAX);
            let k = offset.saturating_add(limit);
            final_indices = Self::sort_indices_by_columns_topk(&final_indices, &stmt.order_by, table, k)?;
            final_indices = final_indices.into_iter().skip(offset).take(limit).collect();
        } else if stmt.limit.is_some() || stmt.offset.is_some() {
            let offset = stmt.offset.unwrap_or(0);
            let limit = stmt.limit.unwrap_or(usize::MAX);
            final_indices = final_indices.into_iter().skip(offset).take(limit).collect();
        }

        let mut rows: Vec<Vec<Value>> = Vec::with_capacity(final_indices.len().min(10000));
        for row_idx in final_indices.iter() {
            let mut row_values = Vec::with_capacity(result_columns.len());
            for (col_name, col_idx) in column_indices.iter() {
                if col_name == "_id" {
                    row_values.push(Value::Int64(*row_idx as i64));
                } else if let Some(idx) = col_idx {
                    row_values.push(table.columns_ref()[*idx].get(*row_idx).unwrap_or(Value::Null));
                } else {
                    row_values.push(Value::Null);
                }
            }
            rows.push(row_values);
        }
        Ok(SqlResult::new(result_columns, rows))
    }
    
    /// Partial sort indices by ORDER BY columns - only get top K elements
    /// Uses BinaryHeap for O(n log k) time complexity
    pub(crate) fn sort_indices_by_columns_topk(
        indices: &[usize],
        order_by: &[OrderByClause],
        table: &ColumnTable,
        k: usize,
    ) -> Result<Vec<usize>, ApexError> {
        crate::query::engine::ops::sort_indices_by_columns_topk(indices, order_by, table, k)
    }
    
    /// Resolve column names and indices from SELECT clause
    pub(crate) fn resolve_columns(
        columns: &[SelectColumn],
        table: &ColumnTable,
    ) -> Result<(Vec<String>, Vec<(String, Option<usize>)>), ApexError> {
        let (cols, idxs, _exprs) = crate::query::engine::ops::resolve_columns(columns, table)?;
        Ok((cols, idxs))
    }
    
    /// Evaluate WHERE clause and return matching row indices
    fn evaluate_where(expr: &SqlExpr, table: &ColumnTable) -> Result<Vec<usize>, ApexError> {
        // Fast path: compile to optimized Filter.
        if let Ok(filter) = sql_expr_to_filter(expr) {
            let schema = table.schema_ref();
            let columns = table.columns_ref();
            let row_count = table.get_row_count();
            let deleted = table.deleted_ref();
            return Ok(filter.filter_columns(schema, columns, row_count, deleted));
        }

        // Fallback: row-wise predicate evaluation for complex expressions
        // (e.g. WHERE a + b > 10).
        let ctx = crate::query::engine::ops::new_eval_context();
        let deleted = table.deleted_ref();
        let row_count = table.get_row_count();
        let no_deletes = deleted.all_false();

        let mut out = Vec::new();
        out.reserve(row_count.min(1024));
        for row_idx in 0..row_count {
            if !no_deletes && deleted.get(row_idx) {
                continue;
            }
            if Self::eval_where_predicate(expr, table, row_idx, &ctx)? {
                out.push(row_idx);
            }
        }
        Ok(out)
    }

    fn eval_where_predicate(
        expr: &SqlExpr,
        table: &ColumnTable,
        row_idx: usize,
        ctx: &crate::query::engine::ops::EvalContext,
    ) -> Result<bool, ApexError> {
        use crate::data::Value;

        #[inline]
        fn to_bool(v: Value) -> bool {
            v.as_bool().unwrap_or(false)
        }

        match expr {
            SqlExpr::Paren(inner) => Self::eval_where_predicate(inner, table, row_idx, ctx),
            SqlExpr::Literal(v) => Ok(v.as_bool().unwrap_or(false)),
            SqlExpr::UnaryOp { op: UnaryOperator::Not, expr } => {
                Ok(!Self::eval_where_predicate(expr, table, row_idx, ctx)?)
            }
            SqlExpr::BinaryOp { left, op, right } => match op {
                BinaryOperator::And => Ok(
                    Self::eval_where_predicate(left, table, row_idx, ctx)?
                        && Self::eval_where_predicate(right, table, row_idx, ctx)?,
                ),
                BinaryOperator::Or => Ok(
                    Self::eval_where_predicate(left, table, row_idx, ctx)?
                        || Self::eval_where_predicate(right, table, row_idx, ctx)?,
                ),
                BinaryOperator::Eq
                | BinaryOperator::NotEq
                | BinaryOperator::Lt
                | BinaryOperator::Le
                | BinaryOperator::Gt
                | BinaryOperator::Ge => {
                    let lv = crate::query::engine::ops::eval_scalar_expr(left, table, row_idx, ctx)?;
                    let rv = crate::query::engine::ops::eval_scalar_expr(right, table, row_idx, ctx)?;
                    if lv.is_null() || rv.is_null() {
                        return Ok(false);
                    }
                    let ord = lv.partial_cmp(&rv).unwrap_or(Ordering::Equal);
                    Ok(match op {
                        BinaryOperator::Eq => ord == Ordering::Equal,
                        BinaryOperator::NotEq => ord != Ordering::Equal,
                        BinaryOperator::Lt => ord == Ordering::Less,
                        BinaryOperator::Le => ord != Ordering::Greater,
                        BinaryOperator::Gt => ord == Ordering::Greater,
                        BinaryOperator::Ge => ord != Ordering::Less,
                        _ => false,
                    })
                }
                // For arithmetic (or other) operators used as predicate, evaluate as scalar and coerce to bool.
                _ => {
                    let v = crate::query::engine::ops::eval_scalar_expr(expr, table, row_idx, ctx)?;
                    Ok(to_bool(v))
                }
            },
            SqlExpr::Like {
                column,
                pattern,
                negated,
            } => {
                let v = crate::query::engine::ops::eval_scalar_expr(
                    &SqlExpr::Column(column.clone()),
                    table,
                    row_idx,
                    ctx,
                )?;
                let s = match v {
                    Value::String(s) => s,
                    _ => return Ok(false),
                };
                let m = LikeMatcher::new(pattern);
                let ok = m.matches(s.as_str());
                Ok(if *negated { !ok } else { ok })
            }
            SqlExpr::Regexp {
                column,
                pattern,
                negated,
            } => {
                let v = crate::query::engine::ops::eval_scalar_expr(
                    &SqlExpr::Column(column.clone()),
                    table,
                    row_idx,
                    ctx,
                )?;
                let s = match v {
                    Value::String(s) => s,
                    _ => return Ok(false),
                };
                let m = RegexpMatcher::new(pattern);
                let ok = m.matches(s.as_str());
                Ok(if *negated { !ok } else { ok })
            }
            SqlExpr::In {
                column,
                values,
                negated,
            } => {
                let v = crate::query::engine::ops::eval_scalar_expr(
                    &SqlExpr::Column(column.clone()),
                    table,
                    row_idx,
                    ctx,
                )?;
                let ok = values.iter().any(|x| &v == x);
                Ok(if *negated { !ok } else { ok })
            }
            SqlExpr::Between {
                column,
                low,
                high,
                negated,
            } => {
                let v = crate::query::engine::ops::eval_scalar_expr(
                    &SqlExpr::Column(column.clone()),
                    table,
                    row_idx,
                    ctx,
                )?;
                if v.is_null() {
                    return Ok(false);
                }
                let lv = crate::query::engine::ops::eval_scalar_expr(low, table, row_idx, ctx)?;
                let hv = crate::query::engine::ops::eval_scalar_expr(high, table, row_idx, ctx)?;
                if lv.is_null() || hv.is_null() {
                    return Ok(false);
                }
                let ok_low = v.partial_cmp(&lv).map(|o| o != Ordering::Less).unwrap_or(false);
                let ok_high = v.partial_cmp(&hv).map(|o| o != Ordering::Greater).unwrap_or(false);
                let ok = ok_low && ok_high;
                Ok(if *negated { !ok } else { ok })
            }
            SqlExpr::IsNull { column, negated } => {
                let v = crate::query::engine::ops::eval_scalar_expr(
                    &SqlExpr::Column(column.clone()),
                    table,
                    row_idx,
                    ctx,
                )?;
                let ok = v.is_null();
                Ok(if *negated { !ok } else { ok })
            }
            // Fallback: treat scalar expression as boolean
            _ => {
                let v = crate::query::engine::ops::eval_scalar_expr(expr, table, row_idx, ctx)?;
                Ok(to_bool(v))
            }
        }
    }

    /// Minimal window function execution: supports only row_number() OVER (PARTITION BY <col> ORDER BY <col>)
    /// Applies WHERE first (matching_indices), then computes row_number per partition.
    pub(crate) fn execute_window_row_number(
        stmt: &SelectStatement,
        matching_indices: &[usize],
        table: &ColumnTable,
    ) -> Result<SqlResult, ApexError> {
        #[derive(Clone, Eq)]
        enum PartKey {
            Null,
            Int(i64),
            Str(String),
            Other(String),
        }

        impl PartialEq for PartKey {
            fn eq(&self, other: &Self) -> bool {
                match (self, other) {
                    (PartKey::Null, PartKey::Null) => true,
                    (PartKey::Int(a), PartKey::Int(b)) => a == b,
                    (PartKey::Str(a), PartKey::Str(b)) => a == b,
                    (PartKey::Other(a), PartKey::Other(b)) => a == b,
                    _ => false,
                }
            }
        }

        impl Hash for PartKey {
            fn hash<H: Hasher>(&self, state: &mut H) {
                match self {
                    PartKey::Null => 0u8.hash(state),
                    PartKey::Int(v) => {
                        1u8.hash(state);
                        v.hash(state);
                    }
                    PartKey::Str(s) => {
                        2u8.hash(state);
                        s.hash(state);
                    }
                    PartKey::Other(s) => {
                        3u8.hash(state);
                        s.hash(state);
                    }
                }
            }
        }

        // Collect window specs (only row_number supported)
        let mut window_specs = Vec::new();
        for col in &stmt.columns {
            if let SelectColumn::WindowFunction { name, partition_by, order_by, alias } = col {
                if name.to_uppercase() != "ROW_NUMBER" {
                    return Err(ApexError::QueryParseError(format!("Unsupported window function: {}", name)));
                }
                if partition_by.len() != 1 {
                    return Err(ApexError::QueryParseError("row_number() requires exactly 1 PARTITION BY column".to_string()));
                }
                if order_by.len() != 1 {
                    return Err(ApexError::QueryParseError("row_number() requires exactly 1 ORDER BY column in OVER()".to_string()));
                }
                window_specs.push((partition_by[0].clone(), order_by[0].clone(), alias.clone().unwrap_or_else(|| "row_number".to_string())));
            }
        }
        if window_specs.len() != 1 {
            return Err(ApexError::QueryParseError("Only a single row_number() window function is supported".to_string()));
        }
        let (partition_col, order_clause, window_alias) = window_specs.remove(0);

        let schema = table.schema_ref();
        let columns = table.columns_ref();

        // Expand SELECT list (including `*`) into a concrete output schema.
        // For window execution we only support a single window column, which will
        // be represented as a synthetic column name with no underlying index.
        let (result_columns, column_indices, _exprs) =
            crate::query::engine::ops::resolve_columns(&stmt.columns, table)?;

        let part_idx = schema.get_index(&partition_col)
            .ok_or_else(|| ApexError::QueryParseError(format!("Unknown PARTITION BY column: {}", partition_col)))?;
        let order_idx = if order_clause.column == "_id" {
            None
        } else {
            schema.get_index(&order_clause.column)
        };

        // Group matching indices by partition key
        let mut groups: HashMap<PartKey, Vec<usize>> = HashMap::new();
        for &row_idx in matching_indices {
            let key = match columns[part_idx].get(row_idx).unwrap_or(Value::Null) {
                Value::Null => PartKey::Null,
                Value::Int64(v) => PartKey::Int(v),
                Value::String(s) => PartKey::Str(s),
                other => PartKey::Other(other.to_string_value()),
            };
            groups.entry(key).or_insert_with(Vec::new).push(row_idx);
        }

        // Build output rows
        let mut out_rows: Vec<Vec<Value>> = Vec::with_capacity(matching_indices.len().min(10000));

        for (_, mut idxs) in groups {
            // Sort within partition by ORDER BY
            if let Some(oidx) = order_idx {
                let desc = order_clause.descending;
                idxs.sort_by(|&a, &b| {
                    let av = columns[oidx].get(a);
                    let bv = columns[oidx].get(b);
                    let cmp = Self::compare_values(av.as_ref(), bv.as_ref(), None);
                    if desc { cmp.reverse() } else { cmp }
                });
            } else {
                // ORDER BY _id
                if order_clause.descending {
                    idxs.sort_by(|a, b| b.cmp(a));
                } else {
                    idxs.sort_unstable();
                }
            }

            for (pos, row_idx) in idxs.into_iter().enumerate() {
                let rn = (pos + 1) as i64;
                let mut row = Vec::with_capacity(result_columns.len());
                for (col_name, col_idx) in column_indices.iter() {
                    if col_name == &window_alias {
                        row.push(Value::Int64(rn));
                    } else if col_name == "_id" {
                        row.push(Value::Int64(row_idx as i64));
                    } else if let Some(ci) = col_idx {
                        row.push(columns[*ci].get(row_idx).unwrap_or(Value::Null));
                    } else {
                        // Synthetic / unsupported expression columns (not expected in window path)
                        row.push(Value::Null);
                    }
                }
                out_rows.push(row);
            }
        }

        Ok(SqlResult::new(result_columns, out_rows))
    }
    
    /// Apply DISTINCT to result rows
    pub(crate) fn apply_distinct(rows: Vec<Vec<Value>>) -> Vec<Vec<Value>> {
        crate::query::engine::ops::apply_distinct(rows)
    }
    
    /// Apply ORDER BY to result rows
    fn apply_order_by(
        mut rows: Vec<Vec<Value>>,
        columns: &[String],
        order_by: &[OrderByClause],
    ) -> Result<Vec<Vec<Value>>, ApexError> {
        // Find column indices for ORDER BY columns
        let order_indices: Vec<(usize, bool, Option<bool>)> = order_by.iter()
            .filter_map(|o| {
                columns.iter()
                    .position(|c| c == &o.column)
                    .map(|idx| (idx, o.descending, o.nulls_first))
            })
            .collect();
        
        if order_indices.is_empty() {
            return Ok(rows);
        }
        
        rows.sort_by(|a, b| {
            for &(idx, desc, nulls_first) in &order_indices {
                let av = a.get(idx);
                let bv = b.get(idx);
                
                let cmp = Self::compare_values(av, bv, nulls_first);
                
                let cmp = if desc { cmp.reverse() } else { cmp };
                
                if cmp != Ordering::Equal {
                    return cmp;
                }
            }
            Ordering::Equal
        });
        
        Ok(rows)
    }
    
    /// Compare two values for ordering
    pub(crate) fn compare_values(a: Option<&Value>, b: Option<&Value>, nulls_first: Option<bool>) -> Ordering {
        crate::query::engine::ops::compare_values(a, b, nulls_first)
    }
    
    pub(crate) fn compare_non_null(a: &Value, b: &Value) -> Ordering {
        crate::query::engine::ops::compare_non_null(a, b)
    }
    
    // ========================================================================
    // ULTRA-FAST PATH IMPLEMENTATIONS
    // ========================================================================
    
    /// O(1) COUNT(*) without WHERE clause
    pub(crate) fn try_fast_count_star(stmt: &SelectStatement, table: &ColumnTable) -> Option<SqlResult> {
        // Check if this is a simple COUNT(*) / COUNT(constant) query
        if stmt.columns.len() == 1 {
            if let SelectColumn::Aggregate { func: AggregateFunc::Count, column, distinct, alias } = &stmt.columns[0] {
                if !*distinct && Self::is_count_star_like(column) {
                    let col_name = alias.clone().unwrap_or_else(|| {
                        column
                            .as_ref()
                            .map(|c| format!("COUNT({})", c))
                            .unwrap_or_else(|| "COUNT(*)".to_string())
                    });
                    let count = table.row_count() as i64;
                    return Some(SqlResult::new(vec![col_name], vec![vec![Value::Int64(count)]]));
                }
            }
        }
        None
    }

    #[inline]
    pub(crate) fn is_count_star_like(column: &Option<String>) -> bool {
        match column {
            None => true,
            Some(c) => Self::is_count_constant_arg(c),
        }
    }

    #[inline]
    fn is_count_constant_arg(arg: &str) -> bool {
        // Parser stores COUNT(constant) as textual representation (e.g. "1", "3.14", "'x'", "true").
        let s = arg.trim();
        if s.is_empty() {
            return false;
        }

        let sl = s.to_ascii_lowercase();
        if sl == "true" || sl == "false" || sl == "null" {
            return true;
        }
        if s.starts_with('\'') && s.ends_with('\'') && s.len() >= 2 {
            return true;
        }

        // Numeric literal (int/float). Be permissive: accept leading +/- and a single dot.
        let mut saw_digit = false;
        let mut saw_dot = false;
        for (i, ch) in s.chars().enumerate() {
            if (ch == '+' || ch == '-') && i == 0 {
                continue;
            }
            if ch.is_ascii_digit() {
                saw_digit = true;
                continue;
            }
            if ch == '.' && !saw_dot {
                saw_dot = true;
                continue;
            }
            return false;
        }
        saw_digit
    }
    
    /// Direct aggregate computation without index collection - ultra-fast for full table scans
    pub(crate) fn execute_aggregate_direct(stmt: &SelectStatement, table: &ColumnTable) -> Result<SqlResult, ApexError> {
        let schema = table.schema_ref();
        let columns = table.columns_ref();
        let deleted = table.deleted_ref();
        let row_count = table.get_row_count();
        
        // Collect aggregate specs
        let mut agg_specs: Vec<(AggregateFunc, Option<String>, bool, String)> = Vec::new();
        for col in &stmt.columns {
            if let SelectColumn::Aggregate { func, column, distinct, alias } = col {
                let col_name = alias.clone().unwrap_or_else(|| {
                    let func_name = match func {
                        AggregateFunc::Count => "COUNT",
                        AggregateFunc::Sum => "SUM",
                        AggregateFunc::Avg => "AVG",
                        AggregateFunc::Min => "MIN",
                        AggregateFunc::Max => "MAX",
                    };
                    if let Some(c) = column {
                        if *distinct {
                            format!("{}(DISTINCT {})", func_name, c)
                        } else {
                            format!("{}({})", func_name, c)
                        }
                    } else {
                        format!("{}(*)", func_name)
                    }
                });
                agg_specs.push((func.clone(), column.clone(), *distinct, col_name));
            }
        }

        // Fast path: mixed aggregates on internal _id plus COUNT(*)/COUNT(constant)
        // Example: SELECT MIN(_id), MAX(_id), COUNT(1) FROM t
        if agg_specs.iter().all(|(_, _, d, _)| !*d) {
            if Self::agg_specs_id_and_count_star_like_only(&agg_specs.iter().map(|(f,c,_,n)| (f.clone(), c.clone(), n.clone())).collect::<Vec<_>>()) {
                return Self::compute_aggregates_id_mixed_direct(&agg_specs.iter().map(|(f,c,_,n)| (f.clone(), c.clone(), n.clone())).collect::<Vec<_>>(), deleted, row_count);
            }
        }
        
        // Check if all aggregates use same numeric column
        let same_column: Option<&str> = {
            let cols: Vec<_> = agg_specs.iter().filter_map(|(_, c, d, _)| if *d { None } else { c.as_deref() }).collect();
            if cols.is_empty() || cols.windows(2).all(|w| w[0] == w[1]) { cols.first().copied() }
            else { None }
        };

        // Ultra-fast path: aggregates on internal _id (row index)
        if let Some("_id") = same_column {
            return Self::compute_aggregates_id_direct(
                &agg_specs
                    .iter()
                    .map(|(f, c, _, n)| (f.clone(), c.clone(), n.clone()))
                    .collect::<Vec<_>>(),
                deleted,
                row_count,
            );
        }
        
        // Ultra-fast path: direct Int64 column scan
        if let Some(col_name) = same_column {
            if let Some(col_idx) = schema.get_index(col_name) {
                if let crate::table::column_table::TypedColumn::Int64 { data, nulls } = &columns[col_idx] {
                    return Self::compute_aggregates_int64_direct(
                        &agg_specs
                            .iter()
                            .map(|(f, c, _, n)| (f.clone(), c.clone(), n.clone()))
                            .collect::<Vec<_>>(),
                        data,
                        nulls,
                        deleted,
                        row_count,
                    );
                }
            }
        }
        
        // Fast path: direct scan without index collection
        // Generic direct path (supports COUNT(DISTINCT))
        Self::compute_aggregates_direct_with_distinct(&agg_specs, schema, columns, deleted, row_count)
    }

    fn compute_aggregates_direct_with_distinct(
        agg_specs: &[(AggregateFunc, Option<String>, bool, String)],
        schema: &crate::table::column_table::ColumnSchema,
        columns: &[crate::table::column_table::TypedColumn],
        deleted: &crate::table::column_table::BitVec,
        row_count: usize,
    ) -> Result<SqlResult, ApexError> {
        use std::collections::HashSet;

        let mut result_columns = Vec::with_capacity(agg_specs.len());
        let mut result_values = Vec::with_capacity(agg_specs.len());

        // Pre-resolve column indices
        let col_indices: Vec<Option<usize>> = agg_specs
            .iter()
            .map(|(_, c, _, _)| c.as_ref().and_then(|cc| schema.get_index(cc)))
            .collect();

        // For COUNT(DISTINCT), prepare sets
        let mut distinct_sets: Vec<Option<HashSet<Vec<u8>>>> = agg_specs
            .iter()
            .map(|(f, _, d, _)| {
                if matches!(f, AggregateFunc::Count) && *d {
                    Some(HashSet::new())
                } else {
                    None
                }
            })
            .collect();

        // Accumulators for non-distinct aggregates
        let mut count_star: Vec<i64> = vec![0; agg_specs.len()];
        let mut count_col: Vec<i64> = vec![0; agg_specs.len()];
        let mut sum: Vec<f64> = vec![0.0; agg_specs.len()];
        let mut sum_count: Vec<i64> = vec![0; agg_specs.len()];
        let mut minv: Vec<Option<Value>> = vec![None; agg_specs.len()];
        let mut maxv: Vec<Option<Value>> = vec![None; agg_specs.len()];

        for row_idx in 0..row_count {
            if deleted.get(row_idx) {
                continue;
            }
            for (i, (func, _, distinct, _)) in agg_specs.iter().enumerate() {
                let ci = col_indices[i];
                match func {
                    AggregateFunc::Count => {
                        if *distinct {
                            if let Some(col_idx) = ci {
                                if let Some(v) = columns[col_idx].get(row_idx) {
                                    if !v.is_null() {
                                        if let Some(set) = distinct_sets[i].as_mut() {
                                            set.insert(v.to_bytes());
                                        }
                                    }
                                }
                            }
                        } else if ci.is_none() {
                            count_star[i] += 1;
                        } else if let Some(col_idx) = ci {
                            if let Some(v) = columns[col_idx].get(row_idx) {
                                if !v.is_null() {
                                    count_col[i] += 1;
                                }
                            }
                        }
                    }
                    AggregateFunc::Sum | AggregateFunc::Avg => {
                        if *distinct {
                            // Not supported by parser
                        } else if let Some(col_idx) = ci {
                            if let Some(v) = columns[col_idx].get(row_idx) {
                                if let Some(n) = v.as_f64() {
                                    sum[i] += n;
                                    sum_count[i] += 1;
                                }
                            }
                        }
                    }
                    AggregateFunc::Min => {
                        if *distinct {
                            // Not supported by parser
                        } else if let Some(col_idx) = ci {
                            if let Some(v) = columns[col_idx].get(row_idx) {
                                if v.is_null() {
                                    continue;
                                }
                                let cur = minv[i].take();
                                minv[i] = Some(match cur {
                                    None => v,
                                    Some(cv) => {
                                        if SqlExecutor::compare_non_null(&cv, &v) == Ordering::Greater {
                                            v
                                        } else {
                                            cv
                                        }
                                    }
                                });
                            }
                        }
                    }
                    AggregateFunc::Max => {
                        if *distinct {
                            // Not supported by parser
                        } else if let Some(col_idx) = ci {
                            if let Some(v) = columns[col_idx].get(row_idx) {
                                if v.is_null() {
                                    continue;
                                }
                                let cur = maxv[i].take();
                                maxv[i] = Some(match cur {
                                    None => v,
                                    Some(cv) => {
                                        if SqlExecutor::compare_non_null(&cv, &v) == Ordering::Less {
                                            v
                                        } else {
                                            cv
                                        }
                                    }
                                });
                            }
                        }
                    }
                }
            }
        }

        for (i, (func, _, distinct, name)) in agg_specs.iter().enumerate() {
            result_columns.push(name.clone());
            let v = match func {
                AggregateFunc::Count => {
                    if *distinct {
                        let n = distinct_sets[i].as_ref().map(|s| s.len()).unwrap_or(0);
                        Value::Int64(n as i64)
                    } else if col_indices[i].is_none() {
                        Value::Int64(count_star[i])
                    } else {
                        Value::Int64(count_col[i])
                    }
                }
                AggregateFunc::Sum => Value::Float64(sum[i]),
                AggregateFunc::Avg => {
                    if sum_count[i] > 0 {
                        Value::Float64(sum[i] / sum_count[i] as f64)
                    } else {
                        Value::Null
                    }
                }
                AggregateFunc::Min => minv[i].clone().unwrap_or(Value::Null),
                AggregateFunc::Max => maxv[i].clone().unwrap_or(Value::Null),
            };
            result_values.push(v);
        }

        Ok(SqlResult::new(result_columns, vec![result_values]))
    }

    /// Ultra-fast aggregates for internal _id without index collection.
    /// Treats _id as the row index (Int64), skipping deleted rows.
    fn compute_aggregates_id_direct(
        agg_specs: &[(AggregateFunc, Option<String>, String)],
        deleted: &crate::table::column_table::BitVec,
        row_count: usize,
    ) -> Result<SqlResult, ApexError> {
        let no_deletes = deleted.all_false();

        // O(1) when there are no deleted rows
        let (count_star, sum, min_val, max_val) = if no_deletes {
            if row_count == 0 {
                (0i64, 0i64, i64::MAX, i64::MIN)
            } else {
                let n = row_count as i64;
                let minv = 0i64;
                let maxv = n - 1;
                let sumv = (n - 1) * n / 2;
                (n, sumv, minv, maxv)
            }
        } else {
            let mut count_star = 0i64;
            let mut sum = 0i64;
            let mut min_val = i64::MAX;
            let mut max_val = i64::MIN;

            for row_idx in 0..row_count {
                if deleted.get(row_idx) {
                    continue;
                }
                let v = row_idx as i64;
                count_star += 1;
                sum += v;
                if v < min_val {
                    min_val = v;
                }
                if v > max_val {
                    max_val = v;
                }
            }
            (count_star, sum, min_val, max_val)
        };

        let mut result_columns = Vec::with_capacity(agg_specs.len());
        let mut result_values = Vec::with_capacity(agg_specs.len());

        for (func, column, name) in agg_specs {
            result_columns.push(name.clone());
            let value = match func {
                AggregateFunc::Count => {
                    if column.is_none() {
                        Value::Int64(count_star)
                    } else {
                        // _id is never NULL for non-deleted rows
                        Value::Int64(count_star)
                    }
                }
                AggregateFunc::Sum => Value::Float64(sum as f64),
                AggregateFunc::Avg => {
                    if count_star > 0 {
                        Value::Float64(sum as f64 / count_star as f64)
                    } else {
                        Value::Null
                    }
                }
                AggregateFunc::Min => {
                    if count_star > 0 {
                        Value::Int64(min_val)
                    } else {
                        Value::Null
                    }
                }
                AggregateFunc::Max => {
                    if count_star > 0 {
                        Value::Int64(max_val)
                    } else {
                        Value::Null
                    }
                }
            };
            result_values.push(value);
        }

        Ok(SqlResult::new(result_columns, vec![result_values]))
    }

    #[inline]
    fn agg_specs_id_and_count_star_like_only(agg_specs: &[(AggregateFunc, Option<String>, String)]) -> bool {
        agg_specs.iter().all(|(func, col, _)| {
            match func {
                // Any aggregate on _id is allowed
                AggregateFunc::Min | AggregateFunc::Max | AggregateFunc::Sum | AggregateFunc::Avg => {
                    matches!(col.as_deref(), Some("_id"))
                }
                AggregateFunc::Count => {
                    // COUNT(*) / COUNT(constant) / COUNT(_id)
                    matches!(col.as_deref(), Some("_id")) || Self::is_count_star_like(col)
                }
            }
        })
    }

    /// Mixed aggregate computation for `_id` plus COUNT(*)/COUNT(constant).
    /// Keeps user-visible column names (e.g. COUNT(1)).
    fn compute_aggregates_id_mixed_direct(
        agg_specs: &[(AggregateFunc, Option<String>, String)],
        deleted: &crate::table::column_table::BitVec,
        row_count: usize,
    ) -> Result<SqlResult, ApexError> {
        // We can reuse the id-direct computation because it already produces all functions,
        // but we need COUNT(constant) to behave like COUNT(*) in value semantics.
        let no_deletes = deleted.all_false();

        // O(1) when there are no deleted rows
        let (count_star, sum, min_val, max_val) = if no_deletes {
            if row_count == 0 {
                (0i64, 0i64, i64::MAX, i64::MIN)
            } else {
                let n = row_count as i64;
                let minv = 0i64;
                let maxv = n - 1;
                let sumv = (n - 1) * n / 2;
                (n, sumv, minv, maxv)
            }
        } else {
            let mut count_star = 0i64;
            let mut sum = 0i64;
            let mut min_val = i64::MAX;
            let mut max_val = i64::MIN;
            for row_idx in 0..row_count {
                if deleted.get(row_idx) {
                    continue;
                }
                let v = row_idx as i64;
                count_star += 1;
                sum += v;
                if v < min_val {
                    min_val = v;
                }
                if v > max_val {
                    max_val = v;
                }
            }
            (count_star, sum, min_val, max_val)
        };

        let mut result_columns = Vec::with_capacity(agg_specs.len());
        let mut result_values = Vec::with_capacity(agg_specs.len());
        for (func, column, name) in agg_specs {
            result_columns.push(name.clone());
            let value = match func {
                AggregateFunc::Count => {
                    // COUNT(_id) / COUNT(*) / COUNT(constant) all equal number of non-deleted rows
                    Value::Int64(count_star)
                }
                AggregateFunc::Sum => Value::Float64(sum as f64),
                AggregateFunc::Avg => {
                    if count_star > 0 {
                        Value::Float64(sum as f64 / count_star as f64)
                    } else {
                        Value::Null
                    }
                }
                AggregateFunc::Min => {
                    if count_star > 0 {
                        Value::Int64(min_val)
                    } else {
                        Value::Null
                    }
                }
                AggregateFunc::Max => {
                    if count_star > 0 {
                        Value::Int64(max_val)
                    } else {
                        Value::Null
                    }
                }
            };
            // Keep column name as-is (COUNT(1) etc.)
            let _ = column;
            result_values.push(value);
        }

        Ok(SqlResult::new(result_columns, vec![result_values]))
    }
    
    /// Ultra-fast Int64 aggregates with direct column scan (no index collection)
    fn compute_aggregates_int64_direct(
        agg_specs: &[(AggregateFunc, Option<String>, String)],
        data: &[i64],
        nulls: &crate::table::column_table::BitVec,
        deleted: &crate::table::column_table::BitVec,
        row_count: usize,
    ) -> Result<SqlResult, ApexError> {
        use rayon::prelude::*;
        
        let no_nulls = nulls.all_false();
        let no_deletes = deleted.all_false();
        let data_len = data.len().min(row_count);
        
        // Parallel reduction for large datasets
        let (count_star, count_col, sum, min_val, max_val) = if data_len >= 100_000 && no_deletes {
            // Parallel scan using chunks
            let chunk_size = 100_000;
            let num_chunks = (data_len + chunk_size - 1) / chunk_size;
            
            let results: Vec<_> = (0..num_chunks)
                .into_par_iter()
                .map(|chunk_idx| {
                    let start = chunk_idx * chunk_size;
                    let end = (start + chunk_size).min(data_len);
                    let mut cs = 0i64;
                    let mut cc = 0i64;
                    let mut s = 0i64;
                    let mut mn = i64::MAX;
                    let mut mx = i64::MIN;
                    for i in start..end {
                        cs += 1;
                        if no_nulls || !nulls.get(i) {
                            let val = data[i];
                            cc += 1;
                            s += val;
                            if val < mn { mn = val; }
                            if val > mx { mx = val; }
                        }
                    }
                    (cs, cc, s, mn, mx)
                })
                .collect();
            
            results.into_iter().fold(
                (0i64, 0i64, 0i64, i64::MAX, i64::MIN),
                |(cs, cc, s, mn, mx), (a, b, c, d, e)| {
                    (cs + a, cc + b, s + c, mn.min(d), mx.max(e))
                }
            )
        } else {
            // Sequential scan (also handles deleted rows)
            let mut count_star = 0i64;
            let mut count_col = 0i64;
            let mut sum = 0i64;
            let mut min_val = i64::MAX;
            let mut max_val = i64::MIN;
            
            for i in 0..data_len {
                if !no_deletes && deleted.get(i) { continue; }
                count_star += 1;
                if no_nulls || !nulls.get(i) {
                    let val = data[i];
                    count_col += 1;
                    sum += val;
                    if val < min_val { min_val = val; }
                    if val > max_val { max_val = val; }
                }
            }
            (count_star, count_col, sum, min_val, max_val)
        };
        
        // Build results
        let mut result_columns = Vec::with_capacity(agg_specs.len());
        let mut result_values = Vec::with_capacity(agg_specs.len());
        
        for (func, column, name) in agg_specs {
            result_columns.push(name.clone());
            let value = match func {
                AggregateFunc::Count => {
                    if column.is_none() { Value::Int64(count_star) }
                    else { Value::Int64(count_col) }
                }
                AggregateFunc::Sum => Value::Float64(sum as f64),
                AggregateFunc::Avg => {
                    if count_col > 0 { Value::Float64(sum as f64 / count_col as f64) }
                    else { Value::Null }
                }
                AggregateFunc::Min => {
                    if count_col > 0 { Value::Int64(min_val) } else { Value::Null }
                }
                AggregateFunc::Max => {
                    if count_col > 0 { Value::Int64(max_val) } else { Value::Null }
                }
            };
            result_values.push(value);
        }
        
        Ok(SqlResult::new(result_columns, vec![result_values]))
    }
    
    /// Direct aggregate scan without index collection (generic)
    fn compute_aggregates_direct(
        agg_specs: &[(AggregateFunc, Option<String>, String)],
        schema: &crate::table::column_table::ColumnSchema,
        columns: &[crate::table::column_table::TypedColumn],
        deleted: &crate::table::column_table::BitVec,
        row_count: usize,
    ) -> Result<SqlResult, ApexError> {
        struct Accumulator {
            col_idx: Option<usize>,
            is_id: bool,
            count: i64,
            sum: f64,
            min: Option<Value>,
            max: Option<Value>,
        }
        
        let no_deletes = deleted.all_false();
        let mut accumulators: Vec<Accumulator> = agg_specs.iter()
            .map(|(_, col, _)| {
                let is_id = matches!(col.as_deref(), Some("_id"));
                let col_idx = if is_id { None } else { col.as_ref().and_then(|c| schema.get_index(c)) };
                Accumulator { col_idx, is_id, count: 0, sum: 0.0, min: None, max: None }
            })
            .collect();
        
        for row_idx in 0..row_count {
            if !no_deletes && deleted.get(row_idx) { continue; }
            
            for (i, (func, _, _)) in agg_specs.iter().enumerate() {
                let acc = &mut accumulators[i];
                if acc.is_id {
                    let val = Value::Int64(row_idx as i64);
                    acc.count += 1;
                    if let Value::Int64(n) = &val {
                        acc.sum += *n as f64;
                    }
                    if matches!(func, AggregateFunc::Min | AggregateFunc::Max) {
                        match &acc.min {
                            None => {
                                acc.min = Some(val.clone());
                                acc.max = Some(val);
                            }
                            Some(curr) => {
                                if Self::compare_non_null(curr, &val) == Ordering::Greater {
                                    acc.min = Some(val.clone());
                                }
                                if let Some(mx) = &acc.max {
                                    if Self::compare_non_null(mx, &val) == Ordering::Less {
                                        acc.max = Some(val);
                                    }
                                }
                            }
                        }
                    }
                } else if let Some(col_idx) = acc.col_idx {
                    if let Some(val) = columns[col_idx].get(row_idx) {
                        if !val.is_null() {
                            acc.count += 1;
                            if let Value::Int64(n) = &val { acc.sum += *n as f64; }
                            else if let Value::Float64(f) = &val { acc.sum += *f; }
                            if matches!(func, AggregateFunc::Min | AggregateFunc::Max) {
                                match &acc.min {
                                    None => { acc.min = Some(val.clone()); acc.max = Some(val); }
                                    Some(curr) => {
                                        if Self::compare_non_null(curr, &val) == Ordering::Greater {
                                            acc.min = Some(val.clone());
                                        }
                                        if let Some(mx) = &acc.max {
                                            if Self::compare_non_null(mx, &val) == Ordering::Less {
                                                acc.max = Some(val);
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                } else {
                    // COUNT(*)
                    acc.count += 1;
                }
            }
        }
        
        let mut result_columns = Vec::with_capacity(agg_specs.len());
        let mut result_values = Vec::with_capacity(agg_specs.len());
        
        for (i, (func, _, name)) in agg_specs.iter().enumerate() {
            result_columns.push(name.clone());
            let acc = &accumulators[i];
            let value = match func {
                AggregateFunc::Count => Value::Int64(acc.count),
                AggregateFunc::Sum => Value::Float64(acc.sum),
                AggregateFunc::Avg => if acc.count > 0 { Value::Float64(acc.sum / acc.count as f64) } else { Value::Null },
                AggregateFunc::Min => acc.min.clone().unwrap_or(Value::Null),
                AggregateFunc::Max => acc.max.clone().unwrap_or(Value::Null),
            };
            result_values.push(value);
        }
        
        Ok(SqlResult::new(result_columns, vec![result_values]))
    }
    
    /// Streaming LIMIT without ORDER BY - early termination
    /// Uses IoEngine for data reading
    pub(crate) fn execute_streaming_limit(
        stmt: &SelectStatement,
        result_columns: &[String],
        column_indices: &[(String, Option<usize>)],
        table: &ColumnTable,
    ) -> Result<SqlResult, ApexError> {
        let offset = stmt.offset.unwrap_or(0);
        let limit = stmt.limit.unwrap_or(usize::MAX);
        
        // Use IoEngine for streaming data reading with early termination
        let indices = IoEngine::read_all_indices(table, Some(limit), offset);
        
        // Build result rows using IoEngine
        let rows: Vec<Vec<Value>> = indices.iter()
            .map(|&row_idx| IoEngine::build_row_values(table, row_idx, column_indices))
            .collect();
        
        Ok(SqlResult::new(result_columns.to_vec(), rows))
    }
    
    /// Streaming Top-K for ORDER BY + LIMIT - ULTRA-OPTIMIZED
    pub(crate) fn execute_streaming_topk(
        stmt: &SelectStatement,
        result_columns: &[String],
        column_indices: &[(String, Option<usize>)],
        table: &ColumnTable,
    ) -> Result<SqlResult, ApexError> {
        let deleted = table.deleted_ref();
        let row_count = table.get_row_count();
        let no_deletes = deleted.all_false();
        let k = stmt.offset.unwrap_or(0) + stmt.limit.unwrap_or(usize::MAX);

        // Collect candidate indices, then reuse unified engine sort semantics.
        let mut candidates: Vec<usize> = Vec::with_capacity(row_count.min(k.saturating_mul(4)).max(1024));
        for row_idx in 0..row_count {
            if !no_deletes && deleted.get(row_idx) {
                continue;
            }
            candidates.push(row_idx);
        }

        let top_indices = crate::query::engine::ops::sort_indices_by_columns_topk(
            &candidates,
            &stmt.order_by,
            table,
            k,
        )?;
        let columns = table.columns_ref();
        
        // Apply offset and build rows
        let offset = stmt.offset.unwrap_or(0);
        let limit = stmt.limit.unwrap_or(usize::MAX);
        let mut rows = Vec::with_capacity(limit.min(top_indices.len()));
        
        for row_idx in top_indices.into_iter().skip(offset).take(limit) {
            let mut row_values = Vec::with_capacity(column_indices.len());
            for (col_name, col_idx) in column_indices {
                if col_name == "_id" {
                    row_values.push(Value::Int64(row_idx as i64));
                } else if let Some(idx) = col_idx {
                    row_values.push(columns[*idx].get(row_idx).unwrap_or(Value::Null));
                } else {
                    row_values.push(Value::Null);
                }
            }
            rows.push(row_values);
        }
        
        Ok(SqlResult::new(result_columns.to_vec(), rows))
    }
    
    /// Generic top-K by row index
    fn topk_generic(row_count: usize, k: usize, deleted: &crate::table::column_table::BitVec, no_deletes: bool, desc: bool) -> Vec<usize> {
        if desc {
            (0..row_count).rev().filter(|&i| no_deletes || !deleted.get(i)).take(k).collect()
        } else {
            (0..row_count).filter(|&i| no_deletes || !deleted.get(i)).take(k).collect()
        }
    }
    
    /// Extract simple LIKE expression from WHERE clause
    /// Returns (field, pattern, negated) if it's a simple LIKE, None otherwise
    fn extract_simple_like(expr: &SqlExpr) -> Option<(String, String, bool)> {
        match expr {
            SqlExpr::Like { column, pattern, negated } => {
                Some((column.clone(), pattern.clone(), *negated))
            }
            SqlExpr::Paren(inner) => Self::extract_simple_like(inner),
            _ => None,
        }
    }
    
    /// Streaming LIKE + LIMIT - ULTRA-FAST early termination
    /// Scans column and stops as soon as we have enough matches
    fn execute_streaming_like_limit(
        stmt: &SelectStatement,
        result_columns: &[String],
        column_indices: &[(String, Option<usize>)],
        table: &ColumnTable,
        like_field: &str,
        like_pattern: &str,
        negated: bool,
    ) -> Result<SqlResult, ApexError> {
        use crate::query::filter::LikeMatcher;
        
        let deleted = table.deleted_ref();
        let columns = table.columns_ref();
        let schema = table.schema_ref();
        let row_count = table.get_row_count();
        let no_deletes = deleted.all_false();
        let offset = stmt.offset.unwrap_or(0);
        let limit = stmt.limit.unwrap_or(usize::MAX);
        let need = offset + limit;
        let num_cols = column_indices.len();
        
        // Get the LIKE column
        let like_col_idx = match schema.get_index(like_field) {
            Some(idx) => idx,
            None => return Ok(SqlResult::new(result_columns.to_vec(), Vec::new())),
        };
        
        // Pre-compile the pattern matcher
        let matcher = LikeMatcher::new(like_pattern);
        
        // Get string column data
        let string_col = match &columns[like_col_idx] {
            crate::table::column_table::TypedColumn::String(col) => Some(col),
            _ => None,
        };
        
        let col = match string_col {
            Some(c) => c,
            None => return Ok(SqlResult::new(result_columns.to_vec(), Vec::new())),
        };
        
        let data_len = col.len().min(row_count);
        
        let mut rows = Vec::with_capacity(limit.min(100));
        let mut found = 0usize;
        
        // Streaming scan with early termination
        for row_idx in 0..data_len {
            // Skip deleted rows
            if !no_deletes && deleted.get(row_idx) { continue; }
            // Skip null values
            if col.is_null(row_idx) { continue; }
            
            // Check LIKE match
            let matches = col.get(row_idx).map(|s| matcher.matches(s)).unwrap_or(false);
            let passes = if negated { !matches } else { matches };
            
            if passes {
                found += 1;
                
                // Skip offset rows
                if found <= offset { continue; }
                
                // Build row
                let mut row_values = Vec::with_capacity(num_cols);
                for (col_name, col_idx) in column_indices {
                    if col_name == "_id" {
                        row_values.push(Value::Int64(row_idx as i64));
                    } else if let Some(idx) = col_idx {
                        row_values.push(columns[*idx].get(row_idx).unwrap_or(Value::Null));
                    } else {
                        row_values.push(Value::Null);
                    }
                }
                rows.push(row_values);
                
                // Early termination!
                if found >= need { break; }
            }
        }
        
        Ok(SqlResult::new(result_columns.to_vec(), rows))
    }
    
    /// Streaming WHERE + LIMIT - ULTRA-FAST early termination for any filter
    /// Handles compound conditions (AND, OR, LIKE, BETWEEN, etc.) with streaming
    /// Uses IoEngine for data reading
    pub(crate) fn execute_streaming_where_limit(
        stmt: &SelectStatement,
        result_columns: &[String],
        column_indices: &[(String, Option<usize>)],
        table: &ColumnTable,
        where_expr: &SqlExpr,
    ) -> Result<SqlResult, ApexError> {
        let offset = stmt.offset.unwrap_or(0);
        let limit = stmt.limit.unwrap_or(usize::MAX);
        
        // Convert WHERE expression to optimized filter
        if let Ok(filter) = sql_expr_to_filter(where_expr) {
            // Use IoEngine for filtered data reading with streaming early termination
            let matching_indices =
                IoEngine::read_filtered_indices(table, &filter, Some(limit), offset);

            // Build result rows using IoEngine
            let rows: Vec<Vec<Value>> = matching_indices
                .iter()
                .map(|&row_idx| IoEngine::build_row_values(table, row_idx, column_indices))
                .collect();

            return Ok(SqlResult::new(result_columns.to_vec(), rows));
        }

        // Fallback: row-wise predicate evaluation with early termination
        let ctx = crate::query::engine::ops::new_eval_context();
        let deleted = table.deleted_ref();
        let row_count = table.get_row_count();
        let no_deletes = deleted.all_false();

        let mut rows: Vec<Vec<Value>> = Vec::with_capacity(limit.min(1024));
        let mut seen_matches = 0usize;
        for row_idx in 0..row_count {
            if !no_deletes && deleted.get(row_idx) {
                continue;
            }
            if !Self::eval_where_predicate(where_expr, table, row_idx, &ctx)? {
                continue;
            }
            seen_matches += 1;
            if seen_matches <= offset {
                continue;
            }
            rows.push(IoEngine::build_row_values(table, row_idx, column_indices));
            if rows.len() >= limit {
                break;
            }
        }

        Ok(SqlResult::new(result_columns.to_vec(), rows))
    }
    
    /// Streaming WHERE + ORDER BY + LIMIT - streaming top-K with filter
    /// Uses heap-based top-K selection while streaming through filtered rows
    fn execute_streaming_where_topk(
        stmt: &SelectStatement,
        result_columns: &[String],
        column_indices: &[(String, Option<usize>)],
        table: &ColumnTable,
        where_expr: &SqlExpr,
    ) -> Result<SqlResult, ApexError> {
        let deleted = table.deleted_ref();
        let columns = table.columns_ref();
        let schema = table.schema_ref();
        let row_count = table.get_row_count();
        let no_deletes = deleted.all_false();
        let k = stmt.offset.unwrap_or(0) + stmt.limit.unwrap_or(usize::MAX);

        let mut candidates: Vec<usize> = Vec::new();
        if let Ok(filter) = sql_expr_to_filter(where_expr) {
            let evaluator = StreamingFilterEvaluator::new(&filter, schema, columns);
            for row_idx in 0..row_count {
                if !no_deletes && deleted.get(row_idx) {
                    continue;
                }
                if !evaluator.matches(row_idx) {
                    continue;
                }
                candidates.push(row_idx);
            }
        } else {
            let ctx = crate::query::engine::ops::new_eval_context();
            for row_idx in 0..row_count {
                if !no_deletes && deleted.get(row_idx) {
                    continue;
                }
                if !Self::eval_where_predicate(where_expr, table, row_idx, &ctx)? {
                    continue;
                }
                candidates.push(row_idx);
            }
        }

        let top_indices = crate::query::engine::ops::sort_indices_by_columns_topk(
            &candidates,
            &stmt.order_by,
            table,
            k,
        )?;

        let offset = stmt.offset.unwrap_or(0);
        let limit = stmt.limit.unwrap_or(usize::MAX);
        let mut rows = Vec::with_capacity(limit.min(top_indices.len()));

        for row_idx in top_indices.into_iter().skip(offset).take(limit) {
            let mut row_values = Vec::with_capacity(column_indices.len());
            for (col_name, col_idx) in column_indices {
                if col_name == "_id" {
                    row_values.push(Value::Int64(row_idx as i64));
                } else if let Some(idx) = col_idx {
                    row_values.push(columns[*idx].get(row_idx).unwrap_or(Value::Null));
                } else {
                    row_values.push(Value::Null);
                }
            }
            rows.push(row_values);
        }

        Ok(SqlResult::new(result_columns.to_vec(), rows))
    }
    
    /// Streaming DISTINCT + LIMIT - stops early when we have enough unique values
    /// Memory-optimized: uses u64 hash instead of String for deduplication
    pub(crate) fn execute_streaming_distinct(
        stmt: &SelectStatement,
        result_columns: &[String],
        column_indices: &[(String, Option<usize>)],
        table: &ColumnTable,
    ) -> Result<SqlResult, ApexError> {
        use std::hash::{Hash, Hasher};
        use std::collections::hash_map::DefaultHasher;
        
        let deleted = table.deleted_ref();
        let columns = table.columns_ref();
        let row_count = table.get_row_count();
        let offset = stmt.offset.unwrap_or(0);
        let limit = stmt.limit.unwrap_or(usize::MAX);
        let need = offset + limit;
        
        // MEMORY OPTIMIZATION: Use u64 hash instead of String (8 bytes vs ~64+ bytes)
        let mut seen: std::collections::HashSet<u64> = std::collections::HashSet::with_capacity(need.min(10001));
        let mut rows: Vec<Vec<Value>> = Vec::with_capacity(limit.min(100));
        
        // Single-column Int64 fast path - most common case
        if column_indices.len() == 1 {
            let (col_name, col_idx) = &column_indices[0];
            if col_name != "_id" {
                if let Some(idx) = col_idx {
                    if let crate::table::column_table::TypedColumn::Int64 { data, nulls } = &columns[*idx] {
                        let no_nulls = nulls.all_false();
                        let data_len = data.len().min(row_count);
                        
                        for row_idx in 0..data_len {
                            if deleted.get(row_idx) { continue; }
                            if !no_nulls && nulls.get(row_idx) { continue; }
                            
                            let val = data[row_idx];
                            if seen.insert(val as u64) {
                                if seen.len() > offset {
                                    rows.push(vec![Value::Int64(val)]);
                                }
                                if seen.len() >= need { break; }
                            }
                        }
                        return Ok(SqlResult::new(result_columns.to_vec(), rows));
                    }
                }
            }
        }
        
        // General case with hash-based deduplication
        for row_idx in 0..row_count {
            if deleted.get(row_idx) { continue; }
            
            // Compute hash for deduplication
            let mut hasher = DefaultHasher::new();
            let mut row_values = Vec::with_capacity(column_indices.len());
            
            for (col_name, col_idx) in column_indices {
                let val = if col_name == "_id" {
                    Value::Int64(row_idx as i64)
                } else if let Some(idx) = col_idx {
                    columns[*idx].get(row_idx).unwrap_or(Value::Null)
                } else {
                    Value::Null
                };
                
                // Hash the value directly (no string allocation)
                match &val {
                    Value::Int64(n) => n.hash(&mut hasher),
                    Value::Float64(f) => f.to_bits().hash(&mut hasher),
                    Value::String(s) => s.hash(&mut hasher),
                    Value::Bool(b) => b.hash(&mut hasher),
                    Value::Null => 0u8.hash(&mut hasher),
                    _ => 1u8.hash(&mut hasher),
                }
                row_values.push(val);
            }
            
            let hash = hasher.finish();
            if seen.insert(hash) {
                if seen.len() > offset {
                    rows.push(row_values);
                }
                if seen.len() >= need { break; }
            }
        }
        
        Ok(SqlResult::new(result_columns.to_vec(), rows))
    }
    
    /// Count matching rows directly without collecting indices - optimized for COUNT(*) with WHERE
    pub(crate) fn count_matching_rows(where_expr: &SqlExpr, table: &ColumnTable) -> Result<usize, ApexError> {
        use rayon::prelude::*;
        
        let filter = sql_expr_to_filter(where_expr)?;
        let schema = table.schema_ref();
        let columns = table.columns_ref();
        let deleted = table.deleted_ref();
        let row_count = table.get_row_count();
        let no_deletes = deleted.all_false();
        
        // For simple Compare filters on Int64, use parallel counting
        if let Filter::Compare { field, op, value } = &filter {
            if let Some(col_idx) = schema.get_index(field) {
                if let (crate::table::column_table::TypedColumn::Int64 { data, nulls }, Value::Int64(target)) = (&columns[col_idx], value) {
                    let no_nulls = nulls.all_false();
                    let target = *target;
                    let data_len = data.len().min(row_count);
                    
                    // Parallel count
                    let chunk_size = 100_000;
                    let num_chunks = (data_len + chunk_size - 1) / chunk_size;
                    
                    let count: usize = (0..num_chunks)
                        .into_par_iter()
                        .map(|chunk_idx| {
                            let start = chunk_idx * chunk_size;
                            let end = (start + chunk_size).min(data_len);
                            let mut cnt = 0usize;
                            for i in start..end {
                                if !no_deletes && deleted.get(i) { continue; }
                                if !no_nulls && nulls.get(i) { continue; }
                                let val = data[i];
                                let matches = match op {
                                    crate::query::filter::CompareOp::Equal => val == target,
                                    crate::query::filter::CompareOp::NotEqual => val != target,
                                    crate::query::filter::CompareOp::LessThan => val < target,
                                    crate::query::filter::CompareOp::LessEqual => val <= target,
                                    crate::query::filter::CompareOp::GreaterThan => val > target,
                                    crate::query::filter::CompareOp::GreaterEqual => val >= target,
                                    _ => false,
                                };
                                if matches { cnt += 1; }
                            }
                            cnt
                        })
                        .sum();
                    
                    return Ok(count);
                }
            }
        }
        
        // Fallback: use filter_columns and count
        let indices = filter.filter_columns(schema, columns, row_count, deleted);
        Ok(indices.len())
    }
    
    /// Top-K selection with pre-filtered indices (for WHERE + ORDER BY + LIMIT)
    fn execute_topk_with_indices(
        stmt: &SelectStatement,
        result_columns: &[String],
        column_indices: &[(String, Option<usize>)],
        matching_indices: &[usize],
        table: &ColumnTable,
    ) -> Result<SqlResult, ApexError> {
        let k = stmt.offset.unwrap_or(0) + stmt.limit.unwrap_or(usize::MAX);
        let top_indices = crate::query::engine::ops::sort_indices_by_columns_topk(
            matching_indices,
            &stmt.order_by,
            table,
            k,
        )?;

        let columns = table.columns_ref();
        let offset = stmt.offset.unwrap_or(0);
        let limit = stmt.limit.unwrap_or(usize::MAX);
        let mut rows = Vec::with_capacity(limit.min(top_indices.len()));

        for row_idx in top_indices.into_iter().skip(offset).take(limit) {
            let mut row_values = Vec::with_capacity(column_indices.len());
            for (col_name, col_idx) in column_indices {
                if col_name == "_id" {
                    row_values.push(Value::Int64(row_idx as i64));
                } else if let Some(idx) = col_idx {
                    row_values.push(columns[*idx].get(row_idx).unwrap_or(Value::Null));
                } else {
                    row_values.push(Value::Null);
                }
            }
            rows.push(row_values);
        }

        Ok(SqlResult::new(result_columns.to_vec(), rows))
    }
    
    /// Execute aggregate query without GROUP BY - FUSED single-pass implementation
    /// Computes all aggregates in one scan over the data
    fn execute_aggregate(
        stmt: &SelectStatement,
        matching_indices: &[usize],
        table: &ColumnTable,
    ) -> Result<SqlResult, ApexError> {
        let schema = table.schema_ref();
        let columns = table.columns_ref();
        
        // Collect all aggregates to compute
        let mut agg_specs: Vec<(AggregateFunc, Option<String>, bool, String)> = Vec::new();
        for col in &stmt.columns {
            if let SelectColumn::Aggregate { func, column, distinct, alias } = col {
                let col_name = alias.clone().unwrap_or_else(|| {
                    let func_name = match func {
                        AggregateFunc::Count => "COUNT",
                        AggregateFunc::Sum => "SUM",
                        AggregateFunc::Avg => "AVG",
                        AggregateFunc::Min => "MIN",
                        AggregateFunc::Max => "MAX",
                    };
                    if let Some(c) = column {
                        if *distinct {
                            format!("{}(DISTINCT {})", func_name, c)
                        } else {
                            format!("{}({})", func_name, c)
                        }
                    } else {
                        format!("{}(*)", func_name)
                    }
                });
                agg_specs.push((func.clone(), column.clone(), *distinct, col_name));
            }
        }

        let has_distinct = agg_specs.iter().any(|(f, c, d, _)| matches!(f, AggregateFunc::Count) && *d && c.is_some());
        
        // Check if all aggregates use the same numeric column - enable ultra-fast path
        let same_column: Option<&str> = {
            let cols: Vec<_> = agg_specs.iter()
                .filter_map(|(_, c, d, _)| if *d { None } else { c.as_deref() })
                .collect();
            if cols.is_empty() || cols.windows(2).all(|w| w[0] == w[1]) {
                cols.first().copied()
            } else {
                None
            }
        };

        // Ultra-fast path: all aggregates on internal _id (row index)
        if !has_distinct {
            if let Some("_id") = same_column {
                return Self::compute_aggregates_id_fused(&agg_specs.iter().map(|(f,c,_,n)| (f.clone(), c.clone(), n.clone())).collect::<Vec<_>>(), matching_indices);
            }
        }
        
        // Ultra-fast path: all aggregates on same Int64 column
        if !has_distinct {
            if let Some(col_name) = same_column {
                if let Some(col_idx) = schema.get_index(col_name) {
                    if let crate::table::column_table::TypedColumn::Int64 { data, nulls } = &columns[col_idx] {
                        return Self::compute_aggregates_int64_fused(&agg_specs.iter().map(|(f,c,_,n)| (f.clone(), c.clone(), n.clone())).collect::<Vec<_>>(), data, nulls, matching_indices);
                    }
                }
            }
        }
        
        // Fast path: compute all aggregates in single pass with accumulators
        Self::compute_aggregates_generic_fused_with_distinct(&agg_specs, schema, columns, matching_indices)
    }

    fn compute_aggregates_generic_fused_with_distinct(
        agg_specs: &[(AggregateFunc, Option<String>, bool, String)],
        schema: &crate::table::column_table::ColumnSchema,
        columns: &[crate::table::column_table::TypedColumn],
        indices: &[usize],
    ) -> Result<SqlResult, ApexError> {
        use std::collections::HashSet;

        struct Accumulator {
            col_idx: Option<usize>,
            distinct: bool,
            seen: Option<HashSet<Vec<u8>>>,
            count: i64,
            sum: f64,
            min: Option<Value>,
            max: Option<Value>,
            sum_count: i64,
        }

        let mut accumulators: Vec<Accumulator> = agg_specs
            .iter()
            .map(|(func, col, distinct, _)| {
                let col_idx = col.as_ref().and_then(|c| schema.get_index(c));
                let seen = if matches!(func, AggregateFunc::Count) && *distinct {
                    Some(HashSet::new())
                } else {
                    None
                };
                Accumulator {
                    col_idx,
                    distinct: *distinct,
                    seen,
                    count: 0,
                    sum: 0.0,
                    min: None,
                    max: None,
                    sum_count: 0,
                }
            })
            .collect();

        for &row_idx in indices {
            for (i, (func, _, _, _)) in agg_specs.iter().enumerate() {
                let acc = &mut accumulators[i];
                match func {
                    AggregateFunc::Count => {
                        if acc.distinct {
                            if let Some(ci) = acc.col_idx {
                                if let Some(v) = columns[ci].get(row_idx) {
                                    if !v.is_null() {
                                        if let Some(set) = acc.seen.as_mut() {
                                            set.insert(v.to_bytes());
                                        }
                                    }
                                }
                            }
                        } else if let Some(ci) = acc.col_idx {
                            if let Some(v) = columns[ci].get(row_idx) {
                                if !v.is_null() {
                                    acc.count += 1;
                                }
                            }
                        } else {
                            acc.count += 1;
                        }
                    }
                    AggregateFunc::Sum | AggregateFunc::Avg => {
                        if let Some(ci) = acc.col_idx {
                            if let Some(v) = columns[ci].get(row_idx) {
                                if let Some(n) = v.as_f64() {
                                    acc.sum += n;
                                    acc.sum_count += 1;
                                }
                            }
                        }
                    }
                    AggregateFunc::Min => {
                        if let Some(ci) = acc.col_idx {
                            if let Some(v) = columns[ci].get(row_idx) {
                                if v.is_null() {
                                    continue;
                                }
                                match &acc.min {
                                    None => {
                                        acc.min = Some(v.clone());
                                        acc.max = Some(v);
                                    }
                                    Some(curr_min) => {
                                        if SqlExecutor::compare_non_null(curr_min, &v) == Ordering::Greater {
                                            acc.min = Some(v.clone());
                                        }
                                        if let Some(curr_max) = &acc.max {
                                            if SqlExecutor::compare_non_null(curr_max, &v) == Ordering::Less {
                                                acc.max = Some(v);
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                    AggregateFunc::Max => {
                        // handled in Min block via shared logic above
                        if let Some(ci) = acc.col_idx {
                            if let Some(v) = columns[ci].get(row_idx) {
                                if v.is_null() {
                                    continue;
                                }
                                match &acc.max {
                                    None => acc.max = Some(v),
                                    Some(curr_max) => {
                                        if SqlExecutor::compare_non_null(curr_max, &v) == Ordering::Less {
                                            acc.max = Some(v);
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        let mut result_columns = Vec::with_capacity(agg_specs.len());
        let mut result_values = Vec::with_capacity(agg_specs.len());

        for (i, (func, _, _, name)) in agg_specs.iter().enumerate() {
            result_columns.push(name.clone());
            let acc = &accumulators[i];
            let value = match func {
                AggregateFunc::Count => {
                    if acc.distinct {
                        Value::Int64(acc.seen.as_ref().map(|s| s.len()).unwrap_or(0) as i64)
                    } else {
                        Value::Int64(acc.count)
                    }
                }
                AggregateFunc::Sum => Value::Float64(acc.sum),
                AggregateFunc::Avg => {
                    if acc.sum_count > 0 {
                        Value::Float64(acc.sum / acc.sum_count as f64)
                    } else {
                        Value::Null
                    }
                }
                AggregateFunc::Min => acc.min.clone().unwrap_or(Value::Null),
                AggregateFunc::Max => acc.max.clone().unwrap_or(Value::Null),
            };
            result_values.push(value);
        }

        Ok(SqlResult::new(result_columns, vec![result_values]))
    }

    /// Ultra-fast fused aggregates for internal _id using matching row indices.
    /// Treats _id as the row index (Int64). Note that `_id` is never NULL.
    fn compute_aggregates_id_fused(
        agg_specs: &[(AggregateFunc, Option<String>, String)],
        indices: &[usize],
    ) -> Result<SqlResult, ApexError> {
        let mut count_star = 0i64;
        let mut sum = 0i64;
        let mut min_val = i64::MAX;
        let mut max_val = i64::MIN;

        for &row_idx in indices {
            let v = row_idx as i64;
            count_star += 1;
            sum += v;
            if v < min_val {
                min_val = v;
            }
            if v > max_val {
                max_val = v;
            }
        }

        let mut result_columns = Vec::with_capacity(agg_specs.len());
        let mut result_values = Vec::with_capacity(agg_specs.len());
        for (func, column, name) in agg_specs {
            result_columns.push(name.clone());
            let value = match func {
                AggregateFunc::Count => {
                    if column.is_none() {
                        Value::Int64(count_star)
                    } else {
                        Value::Int64(count_star)
                    }
                }
                AggregateFunc::Sum => Value::Float64(sum as f64),
                AggregateFunc::Avg => {
                    if count_star > 0 {
                        Value::Float64(sum as f64 / count_star as f64)
                    } else {
                        Value::Null
                    }
                }
                AggregateFunc::Min => {
                    if count_star > 0 {
                        Value::Int64(min_val)
                    } else {
                        Value::Null
                    }
                }
                AggregateFunc::Max => {
                    if count_star > 0 {
                        Value::Int64(max_val)
                    } else {
                        Value::Null
                    }
                }
            };
            result_values.push(value);
        }

        Ok(SqlResult::new(result_columns, vec![result_values]))
    }
    
    /// Ultra-fast fused aggregates on Int64 column - single pass
    fn compute_aggregates_int64_fused(
        agg_specs: &[(AggregateFunc, Option<String>, String)],
        data: &[i64],
        nulls: &crate::table::column_table::BitVec,
        indices: &[usize],
    ) -> Result<SqlResult, ApexError> {
        let no_nulls = nulls.all_false();
        let data_len = data.len();
        
        // Accumulators
        let mut count_star = 0i64;
        let mut count_col = 0i64;
        let mut sum = 0i64;
        let mut min_val = i64::MAX;
        let mut max_val = i64::MIN;
        
        // Single pass over all matching indices
        for &i in indices {
            count_star += 1;
            if i < data_len && (no_nulls || !nulls.get(i)) {
                let val = data[i];
                count_col += 1;
                sum += val;
                if val < min_val { min_val = val; }
                if val > max_val { max_val = val; }
            }
        }
        
        // Build results
        let mut result_columns = Vec::with_capacity(agg_specs.len());
        let mut result_values = Vec::with_capacity(agg_specs.len());
        
        for (func, column, name) in agg_specs {
            result_columns.push(name.clone());
            let value = match func {
                AggregateFunc::Count => {
                    if column.is_none() {
                        Value::Int64(count_star)
                    } else {
                        Value::Int64(count_col)
                    }
                }
                AggregateFunc::Sum => Value::Float64(sum as f64),
                AggregateFunc::Avg => {
                    if count_col > 0 {
                        Value::Float64(sum as f64 / count_col as f64)
                    } else {
                        Value::Null
                    }
                }
                AggregateFunc::Min => {
                    if count_col > 0 { Value::Int64(min_val) } else { Value::Null }
                }
                AggregateFunc::Max => {
                    if count_col > 0 { Value::Int64(max_val) } else { Value::Null }
                }
            };
            result_values.push(value);
        }
        
        Ok(SqlResult::new(result_columns, vec![result_values]))
    }
    
    /// Generic fused aggregates - single pass with multiple accumulators
    fn compute_aggregates_generic_fused(
        agg_specs: &[(AggregateFunc, Option<String>, String)],
        schema: &crate::table::column_table::ColumnSchema,
        columns: &[crate::table::column_table::TypedColumn],
        indices: &[usize],
    ) -> Result<SqlResult, ApexError> {
        // For each aggregate, track: (col_idx, count, sum, min, max)
        struct Accumulator {
            col_idx: Option<usize>,
            count: i64,
            sum: f64,
            min: Option<Value>,
            max: Option<Value>,
        }
        
        let mut accumulators: Vec<Accumulator> = agg_specs.iter()
            .map(|(_, col, _)| {
                let col_idx = col.as_ref().and_then(|c| schema.get_index(c));
                Accumulator { col_idx, count: 0, sum: 0.0, min: None, max: None }
            })
            .collect();
        
        // Single pass
        for &row_idx in indices {
            for (i, (func, _, _)) in agg_specs.iter().enumerate() {
                let acc = &mut accumulators[i];
                
                if let Some(col_idx) = acc.col_idx {
                    if let Some(val) = columns[col_idx].get(row_idx) {
                        if !val.is_null() {
                            acc.count += 1;
                            // Extract numeric value
                            let num = match &val {
                                Value::Int64(n) => Some(*n as f64),
                                Value::Float64(f) => Some(*f),
                                _ => None,
                            };
                            if let Some(n) = num {
                                acc.sum += n;
                            }
                            // Update min/max
                            if matches!(func, AggregateFunc::Min | AggregateFunc::Max) {
                                match &acc.min {
                                    None => { acc.min = Some(val.clone()); acc.max = Some(val); }
                                    Some(curr_min) => {
                                        if Self::compare_non_null(curr_min, &val) == Ordering::Greater {
                                            acc.min = Some(val.clone());
                                        }
                                        if let Some(curr_max) = &acc.max {
                                            if Self::compare_non_null(curr_max, &val) == Ordering::Less {
                                                acc.max = Some(val);
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                } else {
                    // COUNT(*)
                    acc.count += 1;
                }
            }
        }
        
        // Build results
        let mut result_columns = Vec::with_capacity(agg_specs.len());
        let mut result_values = Vec::with_capacity(agg_specs.len());
        
        for (i, (func, _, name)) in agg_specs.iter().enumerate() {
            result_columns.push(name.clone());
            let acc = &accumulators[i];
            let value = match func {
                AggregateFunc::Count => Value::Int64(acc.count),
                AggregateFunc::Sum => Value::Float64(acc.sum),
                AggregateFunc::Avg => {
                    if acc.count > 0 {
                        Value::Float64(acc.sum / acc.count as f64)
                    } else {
                        Value::Null
                    }
                }
                AggregateFunc::Min => acc.min.clone().unwrap_or(Value::Null),
                AggregateFunc::Max => acc.max.clone().unwrap_or(Value::Null),
            };
            result_values.push(value);
        }
        
        Ok(SqlResult::new(result_columns, vec![result_values]))
    }
    
    /// Compute aggregate function value
    fn compute_aggregate(
        func: &AggregateFunc,
        column: Option<&str>,
        indices: &[usize],
        table: &ColumnTable,
    ) -> Result<Value, ApexError> {
        match func {
            AggregateFunc::Count => {
                if column.is_none() {
                    // COUNT(*)
                    Ok(Value::Int64(indices.len() as i64))
                } else {
                    // COUNT(column) - count non-null values
                    let col_name = column.unwrap();
                    let schema = table.schema_ref();
                    if let Some(col_idx) = schema.get_index(col_name) {
                        let typed_col = &table.columns_ref()[col_idx];
                        let count = indices.iter()
                            .filter(|&&i| {
                                typed_col.get(i).map(|v| !v.is_null()).unwrap_or(false)
                            })
                            .count();
                        Ok(Value::Int64(count as i64))
                    } else {
                        Ok(Value::Int64(0))
                    }
                }
            }
            AggregateFunc::Sum | AggregateFunc::Avg => {
                let col_name = column.ok_or_else(|| 
                    ApexError::QueryParseError("SUM/AVG requires column name".to_string())
                )?;
                let schema = table.schema_ref();
                if let Some(col_idx) = schema.get_index(col_name) {
                    let typed_col = &table.columns_ref()[col_idx];
                    let mut sum = 0.0f64;
                    let mut count = 0usize;
                    
                    for &i in indices {
                        if let Some(v) = typed_col.get(i) {
                            match v {
                                Value::Int64(n) => { sum += n as f64; count += 1; }
                                Value::Float64(f) => { sum += f; count += 1; }
                                _ => {}
                            }
                        }
                    }
                    
                    if *func == AggregateFunc::Sum {
                        Ok(Value::Float64(sum))
                    } else {
                        // AVG
                        if count > 0 {
                            Ok(Value::Float64(sum / count as f64))
                        } else {
                            Ok(Value::Null)
                        }
                    }
                } else {
                    Ok(Value::Null)
                }
            }
            AggregateFunc::Min | AggregateFunc::Max => {
                let col_name = column.ok_or_else(|| 
                    ApexError::QueryParseError("MIN/MAX requires column name".to_string())
                )?;
                let schema = table.schema_ref();
                if let Some(col_idx) = schema.get_index(col_name) {
                    let typed_col = &table.columns_ref()[col_idx];
                    let mut result: Option<Value> = None;
                    
                    for &i in indices {
                        if let Some(v) = typed_col.get(i) {
                            if v.is_null() { continue; }
                            
                            result = Some(match result {
                                None => v,
                                Some(curr) => {
                                    let cmp = Self::compare_non_null(&curr, &v);
                                    if (*func == AggregateFunc::Min && cmp == Ordering::Greater) ||
                                       (*func == AggregateFunc::Max && cmp == Ordering::Less) {
                                        v
                                    } else {
                                        curr
                                    }
                                }
                            });
                        }
                    }
                    
                    Ok(result.unwrap_or(Value::Null))
                } else {
                    Ok(Value::Null)
                }
            }
        }
    }
    
    /// Execute GROUP BY aggregation
    pub(crate) fn execute_group_by(
        stmt: &SelectStatement,
        matching_indices: &[usize],
        table: &ColumnTable,
    ) -> Result<SqlResult, ApexError> {
        let schema = table.schema_ref();
        let columns = table.columns_ref();

        #[derive(Clone, Eq)]
        enum KeyPart {
            Null,
            Int(i64),
            Float(u64),
            Bool(bool),
            StrId(u32),
            Other(String),
        }

        impl PartialEq for KeyPart {
            fn eq(&self, other: &Self) -> bool {
                match (self, other) {
                    (KeyPart::Null, KeyPart::Null) => true,
                    (KeyPart::Int(a), KeyPart::Int(b)) => a == b,
                    (KeyPart::Float(a), KeyPart::Float(b)) => a == b,
                    (KeyPart::Bool(a), KeyPart::Bool(b)) => a == b,
                    (KeyPart::StrId(a), KeyPart::StrId(b)) => a == b,
                    (KeyPart::Other(a), KeyPart::Other(b)) => a == b,
                    _ => false,
                }
            }
        }

        impl Hash for KeyPart {
            fn hash<H: Hasher>(&self, state: &mut H) {
                match self {
                    KeyPart::Null => 0u8.hash(state),
                    KeyPart::Int(v) => {
                        1u8.hash(state);
                        v.hash(state);
                    }
                    KeyPart::Float(bits) => {
                        2u8.hash(state);
                        bits.hash(state);
                    }
                    KeyPart::Bool(b) => {
                        3u8.hash(state);
                        b.hash(state);
                    }
                    KeyPart::StrId(id) => {
                        4u8.hash(state);
                        id.hash(state);
                    }
                    KeyPart::Other(s) => {
                        5u8.hash(state);
                        s.hash(state);
                    }
                }
            }
        }

        #[derive(Clone, Eq)]
        struct GroupKey {
            parts: Vec<KeyPart>,
        }

        impl PartialEq for GroupKey {
            fn eq(&self, other: &Self) -> bool {
                self.parts == other.parts
            }
        }

        impl Hash for GroupKey {
            fn hash<H: Hasher>(&self, state: &mut H) {
                self.parts.hash(state)
            }
        }

        // Local string interner for GROUP BY keys: each distinct string allocated once.
        let mut string_intern: HashMap<String, u32> = HashMap::new();
        let mut next_str_id: u32 = 0;

        #[derive(Clone)]
        struct AggState {
            func: AggregateFunc,
            col_idx: Option<usize>,
            distinct: bool,
            seen: Option<std::collections::HashSet<Vec<u8>>>,
            count_star: i64,
            count_non_null: i64,
            sum: f64,
            sum_count: i64,
            min: Option<Value>,
            max: Option<Value>,
        }

        impl AggState {
            fn new(func: AggregateFunc, col_idx: Option<usize>, distinct: bool) -> Self {
                let seen = if matches!(func, AggregateFunc::Count) && distinct {
                    Some(std::collections::HashSet::new())
                } else {
                    None
                };
                Self {
                    func,
                    col_idx,
                    distinct,
                    seen,
                    count_star: 0,
                    count_non_null: 0,
                    sum: 0.0,
                    sum_count: 0,
                    min: None,
                    max: None,
                }
            }

            fn update(&mut self, columns: &[TypedColumn], row_idx: usize) {
                match self.func {
                    AggregateFunc::Count => {
                        if self.distinct {
                            if let Some(ci) = self.col_idx {
                                if let Some(v) = columns[ci].get(row_idx) {
                                    if !v.is_null() {
                                        if let Some(set) = self.seen.as_mut() {
                                            set.insert(v.to_bytes());
                                        }
                                    }
                                }
                            }
                        } else if self.col_idx.is_none() {
                            self.count_star += 1;
                        } else if let Some(ci) = self.col_idx {
                            if let Some(v) = columns[ci].get(row_idx) {
                                if !v.is_null() {
                                    self.count_non_null += 1;
                                }
                            }
                        }
                    }
                    AggregateFunc::Sum | AggregateFunc::Avg => {
                        if let Some(ci) = self.col_idx {
                            if let Some(v) = columns[ci].get(row_idx) {
                                if v.is_null() {
                                    return;
                                }
                                if let Some(n) = v.as_f64() {
                                    self.sum += n;
                                    self.sum_count += 1;
                                }
                            }
                        }
                    }
                    AggregateFunc::Min => {
                        if let Some(ci) = self.col_idx {
                            if let Some(v) = columns[ci].get(row_idx) {
                                if v.is_null() {
                                    return;
                                }
                                self.min = Some(match &self.min {
                                    None => v,
                                    Some(curr) => {
                                        if SqlExecutor::compare_non_null(curr, &v) == Ordering::Greater {
                                            v
                                        } else {
                                            curr.clone()
                                        }
                                    }
                                });
                            }
                        }
                    }
                    AggregateFunc::Max => {
                        if let Some(ci) = self.col_idx {
                            if let Some(v) = columns[ci].get(row_idx) {
                                if v.is_null() {
                                    return;
                                }
                                self.max = Some(match &self.max {
                                    None => v,
                                    Some(curr) => {
                                        if SqlExecutor::compare_non_null(curr, &v) == Ordering::Less {
                                            v
                                        } else {
                                            curr.clone()
                                        }
                                    }
                                });
                            }
                        }
                    }
                }
            }

            fn finalize(&self) -> Value {
                match self.func {
                    AggregateFunc::Count => {
                        if self.distinct {
                            Value::Int64(self.seen.as_ref().map(|s| s.len()).unwrap_or(0) as i64)
                        } else if self.col_idx.is_none() {
                            Value::Int64(self.count_star)
                        } else {
                            Value::Int64(self.count_non_null)
                        }
                    }
                    AggregateFunc::Sum => Value::Float64(self.sum),
                    AggregateFunc::Avg => {
                        if self.sum_count > 0 {
                            Value::Float64(self.sum / self.sum_count as f64)
                        } else {
                            Value::Null
                        }
                    }
                    AggregateFunc::Min => self.min.clone().unwrap_or(Value::Null),
                    AggregateFunc::Max => self.max.clone().unwrap_or(Value::Null),
                }
            }
        }

        #[derive(Clone)]
        struct GroupState {
            first_row_idx: usize,
            aggs: Vec<AggState>,
        }

        // Resolve GROUP BY column indices once
        let mut group_col_indices = Vec::with_capacity(stmt.group_by.len());
        for col in &stmt.group_by {
            let idx = schema
                .get_index(col)
                .ok_or_else(|| ApexError::QueryParseError(format!("Unknown GROUP BY column: {}", col)))?;
            group_col_indices.push(idx);
        }

        // Collect aggregate specs in SELECT order
        let mut agg_specs: Vec<(AggregateFunc, Option<usize>, bool)> = Vec::new();

        fn add_agg_spec(
            out: &mut Vec<(AggregateFunc, Option<usize>, bool)>,
            func: AggregateFunc,
            col_idx: Option<usize>,
            distinct: bool,
        ) {
            if !out.iter().any(|(f, c, d)| *f == func && *c == col_idx && *d == distinct) {
                out.push((func, col_idx, distinct));
            }
        }

        // 1) Explicit aggregates in SELECT list
        for col in &stmt.columns {
            if let SelectColumn::Aggregate { func, column, distinct, .. } = col {
                let col_idx = column.as_ref().and_then(|c| schema.get_index(c));
                add_agg_spec(&mut agg_specs, func.clone(), col_idx, *distinct);
            }
        }

        // 2) Aggregates inside expressions (e.g. CASE WHEN SUM(amount) ...)
        fn collect_aggs_in_expr(
            expr: &SqlExpr,
            schema: &crate::table::column_table::ColumnSchema,
            out: &mut Vec<(AggregateFunc, Option<usize>, bool)>,
        ) {
            match expr {
                SqlExpr::Paren(inner) => collect_aggs_in_expr(inner, schema, out),
                SqlExpr::BinaryOp { left, right, .. } => {
                    collect_aggs_in_expr(left, schema, out);
                    collect_aggs_in_expr(right, schema, out);
                }
                SqlExpr::UnaryOp { expr, .. } => collect_aggs_in_expr(expr, schema, out),
                SqlExpr::Between { low, high, .. } => {
                    collect_aggs_in_expr(low, schema, out);
                    collect_aggs_in_expr(high, schema, out);
                }
                SqlExpr::Function { name, args } => {
                    let func = match name.to_uppercase().as_str() {
                        "COUNT" => Some(AggregateFunc::Count),
                        "SUM" => Some(AggregateFunc::Sum),
                        "AVG" => Some(AggregateFunc::Avg),
                        "MIN" => Some(AggregateFunc::Min),
                        "MAX" => Some(AggregateFunc::Max),
                        _ => None,
                    };
                    if let Some(func) = func {
                        let col_idx = if args.is_empty() {
                            None
                        } else {
                            match &args[0] {
                                SqlExpr::Column(c) => schema.get_index(c),
                                _ => None,
                            }
                        };
                        // DISTINCT inside expressions is not parsed/represented currently; default false.
                        add_agg_spec(out, func, col_idx, false);
                    }
                    for a in args {
                        collect_aggs_in_expr(a, schema, out);
                    }
                }
                SqlExpr::Case { when_then, else_expr } => {
                    for (c, v) in when_then {
                        collect_aggs_in_expr(c, schema, out);
                        collect_aggs_in_expr(v, schema, out);
                    }
                    if let Some(e) = else_expr {
                        collect_aggs_in_expr(e, schema, out);
                    }
                }
                SqlExpr::ScalarSubquery { .. } => {
                    // Scalar subqueries inside GROUP BY expressions are not supported yet.
                }
                _ => {}
            }
        }

        for col in &stmt.columns {
            if let SelectColumn::Expression { expr, .. } = col {
                collect_aggs_in_expr(expr, schema, &mut agg_specs);
            }
        }
        if let Some(h) = stmt.having.as_ref() {
            collect_aggs_in_expr(h, schema, &mut agg_specs);
        }

        // Map SELECT aggregate aliases -> agg_specs index, so HAVING can reference them
        // e.g. SELECT COUNT(*) AS c ... HAVING c > 1
        let mut agg_alias_to_spec_idx: HashMap<String, usize> = HashMap::new();
        for col in &stmt.columns {
            if let SelectColumn::Aggregate { func, column, distinct, alias } = col {
                if let Some(alias) = alias.as_ref() {
                    let col_idx = column.as_ref().and_then(|c| schema.get_index(c));
                    if let Some(pos) = agg_specs
                        .iter()
                        .position(|(f, c, d)| *f == *func && *c == col_idx && *d == *distinct)
                    {
                        agg_alias_to_spec_idx.insert(alias.clone(), pos);
                    }
                }
            }
        }

        fn eval_having_scalar(
            expr: &SqlExpr,
            schema: &crate::table::column_table::ColumnSchema,
            columns: &[TypedColumn],
            group_state: &GroupState,
            agg_specs: &[(AggregateFunc, Option<usize>, bool)],
            agg_alias_to_spec_idx: &HashMap<String, usize>,
        ) -> Result<Value, ApexError> {
            match expr {
                SqlExpr::Paren(inner) => {
                    eval_having_scalar(inner, schema, columns, group_state, agg_specs, agg_alias_to_spec_idx)
                }
                SqlExpr::Literal(v) => Ok(v.clone()),
                SqlExpr::ScalarSubquery { .. } => Err(ApexError::QueryParseError(
                    "Scalar subquery is not supported in GROUP BY/HAVING expressions".to_string(),
                )),
                SqlExpr::Case { when_then, else_expr } => {
                    for (cond, val) in when_then {
                        let cv = eval_having_scalar(
                            cond,
                            schema,
                            columns,
                            group_state,
                            agg_specs,
                            agg_alias_to_spec_idx,
                        )?;
                        if cv.as_bool().unwrap_or(false) {
                            return eval_having_scalar(
                                val,
                                schema,
                                columns,
                                group_state,
                                agg_specs,
                                agg_alias_to_spec_idx,
                            );
                        }
                    }
                    if let Some(e) = else_expr {
                        eval_having_scalar(e, schema, columns, group_state, agg_specs, agg_alias_to_spec_idx)
                    } else {
                        Ok(Value::Null)
                    }
                }
                SqlExpr::Column(name) => {
                    let row_idx = group_state.first_row_idx;
                    if name == "_id" {
                        Ok(Value::Int64(row_idx as i64))
                    } else if let Some(ci) = schema.get_index(name) {
                        Ok(columns[ci].get(row_idx).unwrap_or(Value::Null))
                    } else {
                        // Not a base column: try resolve as SELECT aggregate alias
                        if let Some(&spec_idx) = agg_alias_to_spec_idx.get(name) {
                            Ok(group_state.aggs[spec_idx].finalize())
                        } else {
                            Ok(Value::Null)
                        }
                    }
                }
                SqlExpr::Function { name, args } => {
                    if name.eq_ignore_ascii_case("rand") {
                        if !args.is_empty() {
                            return Err(ApexError::QueryParseError(
                                "RAND() does not accept arguments".to_string(),
                            ));
                        }
                        return Ok(Value::Float64(rand::random::<f64>()));
                    }
                    let func = match name.to_uppercase().as_str() {
                        "COUNT" => AggregateFunc::Count,
                        "SUM" => AggregateFunc::Sum,
                        "AVG" => AggregateFunc::Avg,
                        "MIN" => AggregateFunc::Min,
                        "MAX" => AggregateFunc::Max,
                        _ => {
                            return Err(ApexError::QueryParseError(
                                format!("Unsupported function in HAVING: {}", name),
                            ))
                        }
                    };

                    let col_idx = if args.is_empty() {
                        None
                    } else {
                        match &args[0] {
                            SqlExpr::Column(c) => schema.get_index(c),
                            _ => None,
                        }
                    };

                    for (i, (sf, sc, _sd)) in agg_specs.iter().enumerate() {
                        if *sf == func && *sc == col_idx {
                            return Ok(group_state.aggs[i].finalize());
                        }
                    }
                    Ok(Value::Null)
                }
                SqlExpr::UnaryOp { op, expr } => {
                    match op {
                        crate::query::sql_parser::UnaryOperator::Not => {
                            let v = eval_having_scalar(
                                expr,
                                schema,
                                columns,
                                group_state,
                                agg_specs,
                                agg_alias_to_spec_idx,
                            )?;
                            Ok(Value::Bool(!v.as_bool().unwrap_or(false)))
                        }
                        crate::query::sql_parser::UnaryOperator::Minus => {
                            let v = eval_having_scalar(
                                expr,
                                schema,
                                columns,
                                group_state,
                                agg_specs,
                                agg_alias_to_spec_idx,
                            )?;
                            if let Some(i) = v.as_i64() {
                                Ok(Value::Int64(-i))
                            } else if let Some(f) = v.as_f64() {
                                Ok(Value::Float64(-f))
                            } else {
                                Ok(Value::Null)
                            }
                        }
                    }
                }
                SqlExpr::BinaryOp { left, op, right } => {
                    let lv = eval_having_scalar(
                        left,
                        schema,
                        columns,
                        group_state,
                        agg_specs,
                        agg_alias_to_spec_idx,
                    )?;
                    let rv = eval_having_scalar(
                        right,
                        schema,
                        columns,
                        group_state,
                        agg_specs,
                        agg_alias_to_spec_idx,
                    )?;
                    match op {
                        BinaryOperator::And => Ok(Value::Bool(lv.as_bool().unwrap_or(false) && rv.as_bool().unwrap_or(false))),
                        BinaryOperator::Or => Ok(Value::Bool(lv.as_bool().unwrap_or(false) || rv.as_bool().unwrap_or(false))),
                        BinaryOperator::Eq => Ok(Value::Bool(lv == rv)),
                        BinaryOperator::NotEq => Ok(Value::Bool(lv != rv)),
                        BinaryOperator::Lt | BinaryOperator::Le | BinaryOperator::Gt | BinaryOperator::Ge => {
                            if lv.is_null() || rv.is_null() {
                                return Ok(Value::Bool(false));
                            }
                            let ord = match (lv.as_f64(), rv.as_f64()) {
                                (Some(a), Some(b)) => a.partial_cmp(&b).unwrap_or(Ordering::Equal),
                                _ => SqlExecutor::compare_non_null(&lv, &rv),
                            };
                            let b = match op {
                                BinaryOperator::Lt => ord == Ordering::Less,
                                BinaryOperator::Le => ord != Ordering::Greater,
                                BinaryOperator::Gt => ord == Ordering::Greater,
                                BinaryOperator::Ge => ord != Ordering::Less,
                                _ => false,
                            };
                            Ok(Value::Bool(b))
                        }
                        BinaryOperator::Add | BinaryOperator::Sub | BinaryOperator::Mul | BinaryOperator::Div | BinaryOperator::Mod => {
                            let lf = lv.as_f64();
                            let rf = rv.as_f64();
                            if let (Some(a), Some(b)) = (lf, rf) {
                                let out = match op {
                                    BinaryOperator::Add => a + b,
                                    BinaryOperator::Sub => a - b,
                                    BinaryOperator::Mul => a * b,
                                    BinaryOperator::Div => a / b,
                                    BinaryOperator::Mod => a % b,
                                    _ => a,
                                };
                                Ok(Value::Float64(out))
                            } else {
                                Ok(Value::Null)
                            }
                        }
                    }
                }
                SqlExpr::Like { .. }
                | SqlExpr::Regexp { .. }
                | SqlExpr::In { .. }
                | SqlExpr::InSubquery { .. }
                | SqlExpr::ExistsSubquery { .. }
                | SqlExpr::Between { .. }
                | SqlExpr::IsNull { .. } => Err(ApexError::QueryParseError(
                    "Unsupported predicate in HAVING".to_string(),
                )),
            }
        }

        fn eval_having_predicate(
            expr: &SqlExpr,
            schema: &crate::table::column_table::ColumnSchema,
            columns: &[TypedColumn],
            group_state: &GroupState,
            agg_specs: &[(AggregateFunc, Option<usize>, bool)],
            agg_alias_to_spec_idx: &HashMap<String, usize>,
        ) -> Result<bool, ApexError> {
            let v = eval_having_scalar(
                expr,
                schema,
                columns,
                group_state,
                agg_specs,
                agg_alias_to_spec_idx,
            )?;
            Ok(v.as_bool().unwrap_or(false))
        }

        // Group rows and compute aggregates in a single pass
        let mut groups: HashMap<GroupKey, GroupState> = HashMap::new();
        for &row_idx in matching_indices {
            let mut parts = Vec::with_capacity(group_col_indices.len());
            for &col_idx in &group_col_indices {
                let kp = match &columns[col_idx] {
                    TypedColumn::String(col) => {
                        match col.get(row_idx) {
                            None => KeyPart::Null,
                            Some(s) => {
                                if let Some(&id) = string_intern.get(s) {
                                    KeyPart::StrId(id)
                                } else {
                                    let id = next_str_id;
                                    next_str_id = next_str_id.wrapping_add(1);
                                    string_intern.insert(s.to_string(), id);
                                    KeyPart::StrId(id)
                                }
                            }
                        }
                    }
                    other => {
                        // Fallback via Value for non-string types
                        let v = other.get(row_idx).unwrap_or(Value::Null);
                        match v {
                            Value::Null => KeyPart::Null,
                            Value::Bool(b) => KeyPart::Bool(b),
                            Value::Int64(i) => KeyPart::Int(i),
                            Value::Int32(i) => KeyPart::Int(i as i64),
                            Value::Int16(i) => KeyPart::Int(i as i64),
                            Value::Int8(i) => KeyPart::Int(i as i64),
                            Value::UInt64(u) => KeyPart::Int(u as i64),
                            Value::UInt32(u) => KeyPart::Int(u as i64),
                            Value::UInt16(u) => KeyPart::Int(u as i64),
                            Value::UInt8(u) => KeyPart::Int(u as i64),
                            Value::Float64(f) => KeyPart::Float(f.to_bits()),
                            Value::Float32(f) => KeyPart::Float((f as f64).to_bits()),
                            Value::String(s) => {
                                // Should not happen (handled above), but keep safe
                                if let Some(&id) = string_intern.get(&s) {
                                    KeyPart::StrId(id)
                                } else {
                                    let id = next_str_id;
                                    next_str_id = next_str_id.wrapping_add(1);
                                    string_intern.insert(s, id);
                                    KeyPart::StrId(id)
                                }
                            }
                            other => KeyPart::Other(other.to_string_value()),
                        }
                    }
                };
                parts.push(kp);
            }

            let key = GroupKey { parts };
            let entry = groups.entry(key).or_insert_with(|| GroupState {
                first_row_idx: row_idx,
                aggs: agg_specs
                    .iter()
                    .map(|(func, col_idx, distinct)| AggState::new(func.clone(), *col_idx, *distinct))
                    .collect(),
            });

            for agg in &mut entry.aggs {
                agg.update(columns, row_idx);
            }
        }
        
        // Build result columns
        let mut result_columns = Vec::new();
        for col in &stmt.columns {
            match col {
                SelectColumn::Column(name) => result_columns.push(name.clone()),
                SelectColumn::ColumnAlias { alias, .. } => result_columns.push(alias.clone()),
                SelectColumn::Aggregate { func, column, distinct, alias } => {
                    let name = alias.clone().unwrap_or_else(|| {
                        let func_name = match func {
                            AggregateFunc::Count => "COUNT",
                            AggregateFunc::Sum => "SUM",
                            AggregateFunc::Avg => "AVG",
                            AggregateFunc::Min => "MIN",
                            AggregateFunc::Max => "MAX",
                        };
                        if let Some(c) = column {
                            if *distinct {
                                format!("{}(DISTINCT {})", func_name, c)
                            } else {
                                format!("{}({})", func_name, c)
                            }
                        } else {
                            format!("{}(*)", func_name)
                        }
                    });
                    result_columns.push(name);
                }
                SelectColumn::Expression { alias, .. } => {
                    result_columns.push(alias.clone().unwrap_or_else(|| "expr".to_string()));
                }
                _ => {}
            }
        }
        
        // Build result rows
        use arrow::array::{ArrayRef, BooleanBuilder, Float64Builder, Int64Builder, StringBuilder};
        use arrow::datatypes::{DataType as ArrowDataType, Field};
        use std::sync::Arc;

        enum ColBuilder {
            Int64(Int64Builder),
            Float64(Float64Builder),
            Bool(BooleanBuilder),
            Utf8(StringBuilder),
        }

        impl ColBuilder {
            fn append_value(&mut self, v: &Value) {
                match self {
                    ColBuilder::Int64(b) => {
                        if let Some(i) = v.as_i64() { b.append_value(i); } else { b.append_null(); }
                    }
                    ColBuilder::Float64(b) => {
                        if let Some(f) = v.as_f64() { b.append_value(f); } else { b.append_null(); }
                    }
                    ColBuilder::Bool(b) => {
                        if let Some(x) = v.as_bool() { b.append_value(x); } else { b.append_null(); }
                    }
                    ColBuilder::Utf8(b) => {
                        if v.is_null() { b.append_null(); } else { b.append_value(v.to_string_value()); }
                    }
                }
            }

            fn finish(self) -> ArrayRef {
                match self {
                    ColBuilder::Int64(mut b) => Arc::new(b.finish()),
                    ColBuilder::Float64(mut b) => Arc::new(b.finish()),
                    ColBuilder::Bool(mut b) => Arc::new(b.finish()),
                    ColBuilder::Utf8(mut b) => Arc::new(b.finish()),
                }
            }
        }

        // Pre-build Arrow schema + builders from SELECT list
        let mut fields: Vec<Field> = Vec::with_capacity(result_columns.len());
        let mut builders: Vec<ColBuilder> = Vec::with_capacity(result_columns.len());

        for col in &stmt.columns {
            match col {
                SelectColumn::Column(name) => {
                    let out_name = name.clone();
                    if name == "_id" {
                        fields.push(Field::new(&out_name, ArrowDataType::Int64, true));
                        builders.push(ColBuilder::Int64(Int64Builder::new()));
                    } else if let Some(ci) = schema.get_index(name) {
                        match &columns[ci] {
                            TypedColumn::Int64 { .. } => {
                                fields.push(Field::new(&out_name, ArrowDataType::Int64, true));
                                builders.push(ColBuilder::Int64(Int64Builder::new()));
                            }
                            TypedColumn::Float64 { .. } => {
                                fields.push(Field::new(&out_name, ArrowDataType::Float64, true));
                                builders.push(ColBuilder::Float64(Float64Builder::new()));
                            }
                            TypedColumn::Bool { .. } => {
                                fields.push(Field::new(&out_name, ArrowDataType::Boolean, true));
                                builders.push(ColBuilder::Bool(BooleanBuilder::new()));
                            }
                            TypedColumn::String(_) => {
                                fields.push(Field::new(&out_name, ArrowDataType::Utf8, true));
                                builders.push(ColBuilder::Utf8(StringBuilder::new()));
                            }
                            TypedColumn::Mixed { .. } => {
                                fields.push(Field::new(&out_name, ArrowDataType::Utf8, true));
                                builders.push(ColBuilder::Utf8(StringBuilder::new()));
                            }
                        }
                    } else {
                        fields.push(Field::new(&out_name, ArrowDataType::Utf8, true));
                        builders.push(ColBuilder::Utf8(StringBuilder::new()));
                    }
                }
                SelectColumn::ColumnAlias { column, alias } => {
                    let out_name = alias.clone();
                    if column == "_id" {
                        fields.push(Field::new(&out_name, ArrowDataType::Int64, true));
                        builders.push(ColBuilder::Int64(Int64Builder::new()));
                    } else if let Some(ci) = schema.get_index(column) {
                        match &columns[ci] {
                            TypedColumn::Int64 { .. } => {
                                fields.push(Field::new(&out_name, ArrowDataType::Int64, true));
                                builders.push(ColBuilder::Int64(Int64Builder::new()));
                            }
                            TypedColumn::Float64 { .. } => {
                                fields.push(Field::new(&out_name, ArrowDataType::Float64, true));
                                builders.push(ColBuilder::Float64(Float64Builder::new()));
                            }
                            TypedColumn::Bool { .. } => {
                                fields.push(Field::new(&out_name, ArrowDataType::Boolean, true));
                                builders.push(ColBuilder::Bool(BooleanBuilder::new()));
                            }
                            TypedColumn::String(_) => {
                                fields.push(Field::new(&out_name, ArrowDataType::Utf8, true));
                                builders.push(ColBuilder::Utf8(StringBuilder::new()));
                            }
                            TypedColumn::Mixed { .. } => {
                                fields.push(Field::new(&out_name, ArrowDataType::Utf8, true));
                                builders.push(ColBuilder::Utf8(StringBuilder::new()));
                            }
                        }
                    } else {
                        fields.push(Field::new(&out_name, ArrowDataType::Utf8, true));
                        builders.push(ColBuilder::Utf8(StringBuilder::new()));
                    }
                }
                SelectColumn::Aggregate { func, column, distinct, alias } => {
                    let out_name = alias.clone().unwrap_or_else(|| {
                        let func_name = match func {
                            AggregateFunc::Count => "COUNT",
                            AggregateFunc::Sum => "SUM",
                            AggregateFunc::Avg => "AVG",
                            AggregateFunc::Min => "MIN",
                            AggregateFunc::Max => "MAX",
                        };
                        if let Some(c) = column {
                            if *distinct {
                                format!("{}(DISTINCT {})", func_name, c)
                            } else {
                                format!("{}({})", func_name, c)
                            }
                        } else {
                            format!("{}(*)", func_name)
                        }
                    });

                    match func {
                        AggregateFunc::Count => {
                            fields.push(Field::new(&out_name, ArrowDataType::Int64, true));
                            builders.push(ColBuilder::Int64(Int64Builder::new()));
                        }
                        AggregateFunc::Sum | AggregateFunc::Avg => {
                            fields.push(Field::new(&out_name, ArrowDataType::Float64, true));
                            builders.push(ColBuilder::Float64(Float64Builder::new()));
                        }
                        AggregateFunc::Min | AggregateFunc::Max => {
                            // Use source column type when possible, else Utf8 fallback
                            if let Some(c) = column.as_ref() {
                                if let Some(ci) = schema.get_index(c) {
                                    match &columns[ci] {
                                        TypedColumn::Int64 { .. } => {
                                            fields.push(Field::new(&out_name, ArrowDataType::Int64, true));
                                            builders.push(ColBuilder::Int64(Int64Builder::new()));
                                        }
                                        TypedColumn::Float64 { .. } => {
                                            fields.push(Field::new(&out_name, ArrowDataType::Float64, true));
                                            builders.push(ColBuilder::Float64(Float64Builder::new()));
                                        }
                                        TypedColumn::Bool { .. } => {
                                            fields.push(Field::new(&out_name, ArrowDataType::Boolean, true));
                                            builders.push(ColBuilder::Bool(BooleanBuilder::new()));
                                        }
                                        TypedColumn::String(_) => {
                                            fields.push(Field::new(&out_name, ArrowDataType::Utf8, true));
                                            builders.push(ColBuilder::Utf8(StringBuilder::new()));
                                        }
                                        TypedColumn::Mixed { .. } => {
                                            fields.push(Field::new(&out_name, ArrowDataType::Utf8, true));
                                            builders.push(ColBuilder::Utf8(StringBuilder::new()));
                                        }
                                    }
                                } else {
                                    fields.push(Field::new(&out_name, ArrowDataType::Utf8, true));
                                    builders.push(ColBuilder::Utf8(StringBuilder::new()));
                                }
                            } else {
                                fields.push(Field::new(&out_name, ArrowDataType::Utf8, true));
                                builders.push(ColBuilder::Utf8(StringBuilder::new()));
                            }
                        }
                    }
                }
                SelectColumn::Expression { alias, .. } => {
                    let out_name = alias.clone().unwrap_or_else(|| "expr".to_string());
                    fields.push(Field::new(&out_name, ArrowDataType::Utf8, true));
                    builders.push(ColBuilder::Utf8(StringBuilder::new()));
                }
                _ => {}
            }
        }

        let having_expr = stmt.having.as_ref();

        // Pre-map explicit Aggregate columns to agg_specs indices (cannot rely on sequential order
        // once expressions also contribute aggregate specs).
        let mut select_agg_to_spec: Vec<Option<usize>> = Vec::with_capacity(stmt.columns.len());
        for col in &stmt.columns {
            if let SelectColumn::Aggregate { func, column, distinct, .. } = col {
                let col_idx = column.as_ref().and_then(|c| schema.get_index(c));
                let idx = agg_specs
                    .iter()
                    .position(|(f, c, d)| f == func && *c == col_idx && *d == *distinct);
                select_agg_to_spec.push(idx);
            } else {
                select_agg_to_spec.push(None);
            }
        }

        // Build rows (apply HAVING), then ORDER BY/LIMIT/OFFSET, then append into Arrow builders.
        // Note: groups is a HashMap so iteration order is nondeterministic.
        let mut out_rows: Vec<Vec<Value>> = Vec::with_capacity(groups.len());
        for (_key, group_state) in groups {
            if let Some(h) = having_expr {
                if !eval_having_predicate(h, schema, columns, &group_state, &agg_specs, &agg_alias_to_spec_idx)? {
                    continue;
                }
            }

            let mut row_values: Vec<Value> = Vec::with_capacity(result_columns.len());
            for (pos, col) in stmt.columns.iter().enumerate() {
                match col {
                    SelectColumn::Column(name) => {
                        let first_idx = group_state.first_row_idx;
                        if name == "_id" {
                            row_values.push(Value::Int64(first_idx as i64));
                        } else if let Some(ci) = schema.get_index(name) {
                            row_values.push(columns[ci].get(first_idx).unwrap_or(Value::Null));
                        } else {
                            row_values.push(Value::Null);
                        }
                    }
                    SelectColumn::ColumnAlias { column, .. } => {
                        let first_idx = group_state.first_row_idx;
                        if column == "_id" {
                            row_values.push(Value::Int64(first_idx as i64));
                        } else if let Some(ci) = schema.get_index(column) {
                            row_values.push(columns[ci].get(first_idx).unwrap_or(Value::Null));
                        } else {
                            row_values.push(Value::Null);
                        }
                    }
                    SelectColumn::Aggregate { .. } => {
                        let idx = select_agg_to_spec.get(pos).and_then(|x| *x);
                        if let Some(i) = idx {
                            row_values.push(group_state.aggs[i].finalize());
                        } else {
                            row_values.push(Value::Null);
                        }
                    }
                    SelectColumn::Expression { expr, .. } => {
                        row_values.push(eval_having_scalar(
                            expr,
                            schema,
                            columns,
                            &group_state,
                            &agg_specs,
                            &agg_alias_to_spec_idx,
                        )?);
                    }
                    SelectColumn::All | SelectColumn::WindowFunction { .. } => {
                        return Err(ApexError::QueryParseError(
                            "Unsupported SELECT item in GROUP BY".to_string(),
                        ));
                    }
                }
            }
            out_rows.push(row_values);
        }

        if !stmt.order_by.is_empty() {
            out_rows = Self::apply_order_by(out_rows, &result_columns, &stmt.order_by)?;
        }

        let off = stmt.offset.unwrap_or(0);
        let lim = stmt.limit.unwrap_or(usize::MAX);
        let out_rows = out_rows.into_iter().skip(off).take(lim).collect::<Vec<_>>();

        Ok(SqlResult::new(result_columns, out_rows))
    }
    
    /// Build Arrow arrays directly from column data using matching indices
    /// This is much faster than row-by-row construction for large result sets
    pub(crate) fn build_arrow_direct(
        result_columns: &[String],
        column_indices: &[(String, Option<usize>)],
        matching_indices: &[usize],
        table: &ColumnTable,
    ) -> Result<SqlResult, ApexError> {
        let projected_exprs = vec![None; column_indices.len()];
        let ctx = crate::query::engine::ops::new_eval_context();
        crate::query::engine::ops::build_arrow_direct(
            result_columns,
            column_indices,
            &projected_exprs,
            matching_indices,
            table,
            &ctx,
        )
    }
}
