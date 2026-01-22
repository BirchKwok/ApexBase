//! V3 Native Query Executor
//!
//! This module provides a pure Arrow-based query execution engine that operates
//! directly on OnDemandStorage without requiring ColumnTable.
//!
//! Architecture:
//! - Reads columns on-demand from V3 storage
//! - Performs all filtering/projection/aggregation using Arrow compute kernels
//! - Returns Arrow RecordBatch directly (zero-copy to Python)

use arrow::array::{
    Array, ArrayRef, BooleanArray, Float64Array, Int64Array, StringArray,
    UInt64Array, RecordBatch,
};
use arrow::compute::{self, SortOptions};
use arrow::compute::kernels::cmp;
use arrow::compute::kernels::numeric as arith;
use arrow::datatypes::{DataType as ArrowDataType, Field, Schema};
use std::collections::HashMap;
use std::io;
use std::path::Path;
use std::sync::Arc;

use crate::query::{SqlParser, SqlStatement, SelectStatement, SqlExpr, SelectColumn, JoinType, JoinClause, UnionStatement, AggregateFunc};
use crate::query::sql_parser::BinaryOperator;
use crate::query::sql_parser::FromItem;
use crate::storage::TableStorageBackend;
use crate::data::{DataType, Value};
use std::collections::HashSet;

/// V3 Native Query Executor
/// 
/// Executes SQL queries directly on V3 storage using Arrow compute kernels.
/// This replaces the ColumnTable-based execution path.
pub struct V3Executor;

/// Query execution result
pub enum V3Result {
    /// Query returned data rows
    Data(RecordBatch),
    /// Query returned empty result
    Empty(Arc<Schema>),
    /// Query returned a scalar (COUNT, etc.)
    Scalar(i64),
}

impl V3Result {
    pub fn to_record_batch(self) -> io::Result<RecordBatch> {
        match self {
            V3Result::Data(batch) => Ok(batch),
            V3Result::Empty(schema) => Ok(RecordBatch::new_empty(schema)),
            V3Result::Scalar(val) => {
                let schema = Arc::new(Schema::new(vec![
                    Field::new("result", ArrowDataType::Int64, false),
                ]));
                let array: ArrayRef = Arc::new(Int64Array::from(vec![val]));
                RecordBatch::try_new(schema, vec![array])
                    .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e.to_string()))
            }
        }
    }

    pub fn num_rows(&self) -> usize {
        match self {
            V3Result::Data(batch) => batch.num_rows(),
            V3Result::Empty(_) => 0,
            V3Result::Scalar(_) => 1,
        }
    }
}

impl V3Executor {
    /// Execute a SQL query on V3 storage (single table)
    pub fn execute(sql: &str, storage_path: &Path) -> io::Result<V3Result> {
        let stmt = SqlParser::parse(sql)
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidInput, e.to_string()))?;

        Self::execute_parsed(stmt, storage_path)
    }

    /// Execute a SQL query with multi-table support (for JOINs)
    pub fn execute_with_base_dir(sql: &str, base_dir: &Path, default_table_path: &Path) -> io::Result<V3Result> {
        // Support multi-statement execution (e.g., CREATE VIEW; SELECT ...; DROP VIEW;)
        // Parse as multi-statement unconditionally to avoid relying on string heuristics.
        let stmts = SqlParser::parse_multi(sql)
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidInput, e.to_string()))?;

        if stmts.len() > 1
            || matches!(stmts.first(), Some(SqlStatement::CreateView { .. } | SqlStatement::DropView { .. }))
        {
            return Self::execute_parsed_multi_statements(stmts, base_dir, default_table_path);
        }

        let stmt = stmts
            .into_iter()
            .next()
            .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "No statement to execute"))?;

        Self::execute_parsed_multi(stmt, base_dir, default_table_path)
    }

    /// Execute a parsed SQL statement (single table)
    pub fn execute_parsed(stmt: SqlStatement, storage_path: &Path) -> io::Result<V3Result> {
        match stmt {
            SqlStatement::Select(select) => Self::execute_select(select, storage_path),
            SqlStatement::Union(union) => Self::execute_union(union, storage_path),
            _ => Err(io::Error::new(
                io::ErrorKind::Unsupported,
                "Only SELECT/UNION statements supported",
            )),
        }
    }

    /// Execute a parsed SQL statement with multi-table support
    pub fn execute_parsed_multi(stmt: SqlStatement, base_dir: &Path, default_table_path: &Path) -> io::Result<V3Result> {
        match stmt {
            SqlStatement::Select(select) => {
                if select.joins.is_empty() {
                    // Resolve the actual table path from FROM clause for non-join queries
                    let actual_path = Self::resolve_from_table_path(&select, base_dir, default_table_path);
                    Self::execute_select(select, &actual_path)
                } else {
                    Self::execute_select_with_joins(select, base_dir, default_table_path)
                }
            }
            SqlStatement::Union(union) => Self::execute_union(union, default_table_path),
            _ => Err(io::Error::new(
                io::ErrorKind::Unsupported,
                "Only SELECT/UNION statements supported",
            )),
        }
    }

    /// Execute multiple SQL statements separated by semicolons.
    /// Currently used for temporary VIEW support within a single execute() call.
    fn execute_parsed_multi_statements(
        stmts: Vec<SqlStatement>,
        base_dir: &Path,
        default_table_path: &Path,
    ) -> io::Result<V3Result> {
        use std::collections::HashMap;

        let mut views: HashMap<String, SelectStatement> = HashMap::new();
        let mut last_result: Option<V3Result> = None;

        for stmt in stmts {
            match stmt {
                SqlStatement::CreateView { name, stmt } => {
                    let view_name = name.trim_matches('"').to_string();
                    if view_name.eq_ignore_ascii_case("default") {
                        return Err(io::Error::new(io::ErrorKind::InvalidInput, "View name conflicts with default table"));
                    }

                    // Disallow conflict with existing table file
                    let table_path = Self::resolve_table_path(&view_name, base_dir, default_table_path);
                    if table_path.exists() {
                        return Err(io::Error::new(io::ErrorKind::InvalidInput, "View name conflicts with existing table"));
                    }

                    views.insert(view_name, stmt);
                }
                SqlStatement::DropView { name } => {
                    let view_name = name.trim_matches('"');
                    views.remove(view_name);
                }
                SqlStatement::Select(mut select) => {
                    select = Self::rewrite_select_views(select, &views);
                    last_result = Some(Self::execute_parsed_multi(SqlStatement::Select(select), base_dir, default_table_path)?);
                }
                SqlStatement::Union(union) => {
                    last_result = Some(Self::execute_union(union, default_table_path)?);
                }
            }
        }

        last_result.ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "No query to execute"))
    }

    fn rewrite_select_views(mut select: SelectStatement, views: &std::collections::HashMap<String, SelectStatement>) -> SelectStatement {
        if let Some(from) = &select.from {
            match from {
                FromItem::Table { table, alias } => {
                    let table_name = table.trim_matches('"');
                    if let Some(view_stmt) = views.get(table_name) {
                        let alias_name = alias.clone().unwrap_or_else(|| table_name.to_string());
                        select.from = Some(FromItem::Subquery {
                            stmt: Box::new(view_stmt.clone()),
                            alias: alias_name,
                        });
                    }
                }
                _ => {}
            }
        }

        select
    }

    /// Execute SELECT statement
    fn execute_select(stmt: SelectStatement, storage_path: &Path) -> io::Result<V3Result> {
        // Check for derived table (FROM subquery)
        let batch = match &stmt.from {
            Some(FromItem::Subquery { stmt: sub_stmt, .. }) => {
                // Execute subquery first to get source data
                let sub_result = Self::execute_select(*sub_stmt.clone(), storage_path)?;
                sub_result.to_record_batch()?
            }
            _ => {
                // Normal table - read from storage
                // If file doesn't exist (e.g., after drop_if_exists), return empty batch
                if !storage_path.exists() {
                    let schema = Arc::new(Schema::empty());
                    RecordBatch::new_empty(schema)
                } else {
                    let backend = TableStorageBackend::open(storage_path)?;
                    
                    // Check if any SELECT column contains a scalar subquery
                    let has_scalar_subquery = stmt.columns.iter().any(|col| {
                        if let SelectColumn::Expression { expr, .. } = col {
                            Self::expr_contains_scalar_subquery(expr)
                        } else {
                            false
                        }
                    });
                    
                    // Read all columns when there's a WHERE clause or scalar subquery
                    // This ensures all referenced columns are available
                    if stmt.where_clause.is_some() || has_scalar_subquery {
                        backend.read_columns_to_arrow(None, 0, None)?
                    } else {
                        // No WHERE clause or scalar subquery - use optimized column reading
                        let required_cols = stmt.required_columns();
                        let col_refs: Option<Vec<&str>> = required_cols
                            .as_ref()
                            .filter(|cols| !cols.is_empty())
                            .map(|cols| cols.iter().map(|s| s.as_str()).collect());
                        backend.read_columns_to_arrow(col_refs.as_deref(), 0, None)?
                    }
                }
            }
        };

        // Determine row limit for early termination
        let row_limit = stmt.limit;

        // Check for aggregation BEFORE checking empty batch
        // Aggregations like COUNT(*) should return 0 for empty tables
        // Also check for aggregates inside expressions (e.g., CASE WHEN SUM(x) > 100 ...)
        let has_aggregation = stmt.columns.iter().any(|col| {
            match col {
                SelectColumn::Aggregate { .. } => true,
                SelectColumn::Expression { expr, .. } => Self::expr_contains_aggregate(expr),
                _ => false,
            }
        });

        if batch.num_rows() == 0 {
            // For aggregations on empty tables, still execute aggregation (COUNT(*) returns 0)
            if has_aggregation && stmt.group_by.is_empty() {
                return Self::execute_aggregation(&batch, &stmt);
            }
            return Ok(V3Result::Empty(batch.schema()));
        }

        // Apply WHERE filter (with storage path for subquery support)
        let filtered = if let Some(ref where_clause) = stmt.where_clause {
            Self::apply_filter_with_storage(&batch, where_clause, storage_path)?
        } else {
            batch
        };

        if filtered.num_rows() == 0 {
            // For aggregations on filtered empty result, still execute aggregation
            if has_aggregation && stmt.group_by.is_empty() {
                return Self::execute_aggregation(&filtered, &stmt);
            }
            return Ok(V3Result::Empty(filtered.schema()));
        }

        // Check for window functions
        let has_window = stmt.columns.iter().any(|col| matches!(col, SelectColumn::WindowFunction { .. }));
        if has_window {
            return Self::execute_window_function(&filtered, &stmt);
        }

        if has_aggregation && stmt.group_by.is_empty() {
            // Simple aggregation without GROUP BY
            return Self::execute_aggregation(&filtered, &stmt);
        }

        // Handle GROUP BY
        if has_aggregation && !stmt.group_by.is_empty() {
            return Self::execute_group_by(&filtered, &stmt);
        }

        // Apply ORDER BY
        let sorted = if !stmt.order_by.is_empty() {
            Self::apply_order_by(&filtered, &stmt.order_by)?
        } else {
            filtered
        };

        // Apply LIMIT/OFFSET
        let limited = Self::apply_limit_offset(&sorted, stmt.limit, stmt.offset)?;

        // Apply projection (SELECT columns) - pass storage_path for scalar subqueries
        let projected = Self::apply_projection_with_storage(&limited, &stmt.columns, Some(storage_path))?;

        // Apply DISTINCT if specified
        let result = if stmt.distinct {
            Self::deduplicate_batch(&projected)?
        } else {
            projected
        };

        Ok(V3Result::Data(result))
    }

    /// Execute SELECT statement with JOINs
    fn execute_select_with_joins(stmt: SelectStatement, base_dir: &Path, default_table_path: &Path) -> io::Result<V3Result> {
        // Get the left (base) table
        let left_table_name = match &stmt.from {
            Some(FromItem::Table { table, .. }) => table.clone(),
            Some(FromItem::Subquery { .. }) => {
                return Err(io::Error::new(
                    io::ErrorKind::Unsupported,
                    "Subquery in FROM clause not yet supported",
                ));
            }
            None => "default".to_string(),
        };

        // Load left table
        let left_path = Self::resolve_table_path(&left_table_name, base_dir, default_table_path);
        let left_backend = TableStorageBackend::open(&left_path)?;
        let mut result_batch = left_backend.read_columns_to_arrow(None, 0, None)?;

        // Process each JOIN clause
        for join_clause in &stmt.joins {
            let right_table_name = match &join_clause.right {
                FromItem::Table { table, .. } => table.clone(),
                FromItem::Subquery { .. } => {
                    return Err(io::Error::new(
                        io::ErrorKind::Unsupported,
                        "Subquery in JOIN not yet supported",
                    ));
                }
            };

            // Load right table
            let right_path = Self::resolve_table_path(&right_table_name, base_dir, default_table_path);
            let right_backend = TableStorageBackend::open(&right_path)?;
            let right_batch = right_backend.read_columns_to_arrow(None, 0, None)?;

            // Extract join keys from ON condition
            let (left_key, right_key) = Self::extract_join_keys(&join_clause.on)?;

            // Perform the join
            result_batch = Self::hash_join(
                &result_batch,
                &right_batch,
                &left_key,
                &right_key,
                &join_clause.join_type,
            )?;
        }

        if result_batch.num_rows() == 0 {
            return Ok(V3Result::Empty(result_batch.schema()));
        }

        // Apply WHERE filter (with storage path for subquery support)
        let filtered = if let Some(ref where_clause) = stmt.where_clause {
            Self::apply_filter_with_storage(&result_batch, where_clause, default_table_path)?
        } else {
            result_batch
        };

        if filtered.num_rows() == 0 {
            return Ok(V3Result::Empty(filtered.schema()));
        }

        // Check for aggregation
        let has_aggregation = stmt.columns.iter().any(|col| matches!(col, SelectColumn::Aggregate { .. }));

        if has_aggregation && stmt.group_by.is_empty() {
            return Self::execute_aggregation(&filtered, &stmt);
        }

        // Handle GROUP BY with aggregation
        if has_aggregation && !stmt.group_by.is_empty() {
            return Self::execute_group_by(&filtered, &stmt);
        }

        // Apply ORDER BY
        let sorted = if !stmt.order_by.is_empty() {
            Self::apply_order_by(&filtered, &stmt.order_by)?
        } else {
            filtered
        };

        // Apply LIMIT/OFFSET
        let limited = Self::apply_limit_offset(&sorted, stmt.limit, stmt.offset)?;

        // Apply projection - pass default_table_path for scalar subqueries
        let projected = Self::apply_projection_with_storage(&limited, &stmt.columns, Some(default_table_path))?;

        // Apply DISTINCT if specified
        let result = if stmt.distinct {
            Self::deduplicate_batch(&projected)?
        } else {
            projected
        };

        Ok(V3Result::Data(result))
    }

    /// Resolve table path from FROM clause
    fn resolve_from_table_path(stmt: &SelectStatement, base_dir: &Path, default_table_path: &Path) -> std::path::PathBuf {
        if let Some(FromItem::Table { table, .. }) = &stmt.from {
            let table_name = table.trim_matches('"');
            // For "default" table, use default_table_path
            if table_name == "default" {
                return default_table_path.to_path_buf();
            }
            // Check if table matches the default table's file stem
            if let Some(stem) = default_table_path.file_stem() {
                if stem.to_string_lossy() == table_name {
                    return default_table_path.to_path_buf();
                }
            }
            // For other tables, resolve from base directory
            base_dir.join(format!("{}.apex", table_name))
        }
        // No FROM clause - use default_table_path
        default_table_path.to_path_buf()
    }

    /// Resolve table path from table name
    fn resolve_table_path(table_name: &str, base_dir: &Path, default_table_path: &Path) -> std::path::PathBuf {
        if table_name == "default" {
            default_table_path.to_path_buf()
        } else {
            let safe_name: String = table_name.chars()
                .map(|c| if c.is_alphanumeric() || c == '_' || c == '-' { c } else { '_' })
                .collect();
            let truncated_name = if safe_name.len() > 200 { &safe_name[..200] } else { &safe_name };
            base_dir.join(format!("{}.apex", truncated_name))
        }
    }

    /// Extract join keys from ON condition (expects simple equality: left.col = right.col)
    fn extract_join_keys(on_expr: &SqlExpr) -> io::Result<(String, String)> {
        use crate::query::sql_parser::BinaryOperator;
        
        match on_expr {
            SqlExpr::BinaryOp { left, op: BinaryOperator::Eq, right } => {
                let left_col = Self::extract_column_name(left)?;
                let right_col = Self::extract_column_name(right)?;
                Ok((left_col, right_col))
            }
            _ => Err(io::Error::new(
                io::ErrorKind::Unsupported,
                "JOIN ON clause must be a simple equality (e.g., a.id = b.id)",
            )),
        }
    }

    /// Extract column name from expression (handles table.column and column)
    fn extract_column_name(expr: &SqlExpr) -> io::Result<String> {
        match expr {
            SqlExpr::Column(name) => {
                // Handle table.column format - take the column part
                if let Some(dot_pos) = name.rfind('.') {
                    Ok(name[dot_pos + 1..].to_string())
                } else {
                    Ok(name.clone())
                }
            }
            _ => Err(io::Error::new(
                io::ErrorKind::Unsupported,
                "JOIN key must be a column reference",
            )),
        }
    }

    /// Perform hash join between two RecordBatches
    fn hash_join(
        left: &RecordBatch,
        right: &RecordBatch,
        left_key: &str,
        right_key: &str,
        join_type: &JoinType,
    ) -> io::Result<RecordBatch> {
        // Get key columns
        let left_key_col = left.column_by_name(left_key)
            .ok_or_else(|| io::Error::new(
                io::ErrorKind::NotFound,
                format!("Left join key '{}' not found", left_key),
            ))?;
        let right_key_col = right.column_by_name(right_key)
            .ok_or_else(|| io::Error::new(
                io::ErrorKind::NotFound,
                format!("Right join key '{}' not found", right_key),
            ))?;

        // Build hash table from right side (smaller table ideally)
        let mut hash_table: HashMap<u64, Vec<usize>> = HashMap::new();
        for i in 0..right.num_rows() {
            let hash = Self::hash_array_value(right_key_col, i);
            hash_table.entry(hash).or_default().push(i);
        }

        // Probe with left side
        let mut left_indices: Vec<u32> = Vec::new();
        let mut right_indices: Vec<Option<u32>> = Vec::new();

        for left_idx in 0..left.num_rows() {
            let left_hash = Self::hash_array_value(left_key_col, left_idx);
            
            if let Some(right_matches) = hash_table.get(&left_hash) {
                // Verify actual equality (hash collision check)
                for &right_idx in right_matches {
                    if Self::arrays_equal_at(left_key_col, left_idx, right_key_col, right_idx) {
                        left_indices.push(left_idx as u32);
                        right_indices.push(Some(right_idx as u32));
                    }
                }
            }
            
            // For LEFT JOIN, include unmatched left rows
            if matches!(join_type, JoinType::Left) {
                let matched = right_indices.iter().rev()
                    .take_while(|r| r.is_some())
                    .any(|r| {
                        if let Some(ri) = r {
                            left_indices.iter().rev()
                                .zip(std::iter::repeat(*ri))
                                .take(1)
                                .any(|(li, _)| *li == left_idx as u32)
                        } else {
                            false
                        }
                    });
                
                if !matched && !hash_table.contains_key(&left_hash) {
                    left_indices.push(left_idx as u32);
                    right_indices.push(None);
                }
            }
        }

        // Build result schema (combine left and right, avoiding duplicate key column)
        let mut fields: Vec<Field> = left.schema().fields().iter().map(|f| f.as_ref().clone()).collect();
        for field in right.schema().fields() {
            if field.name() != right_key {
                let new_name = if fields.iter().any(|f| f.name() == field.name()) {
                    format!("{}_{}", field.name(), "right")
                } else {
                    field.name().clone()
                };
                fields.push(Field::new(&new_name, field.data_type().clone(), true));
            }
        }
        let result_schema = Arc::new(Schema::new(fields));

        // Build result columns
        let mut columns: Vec<ArrayRef> = Vec::new();

        // Take from left
        let left_indices_array = arrow::array::UInt32Array::from(left_indices.clone());
        for col in left.columns() {
            let taken = compute::take(col.as_ref(), &left_indices_array, None)
                .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e.to_string()))?;
            columns.push(taken);
        }

        // Take from right (excluding join key to avoid duplication)
        for (col_idx, field) in right.schema().fields().iter().enumerate() {
            if field.name() != right_key {
                let right_col = right.column(col_idx);
                let taken = Self::take_with_nulls(right_col, &right_indices)?;
                columns.push(taken);
            }
        }

        RecordBatch::try_new(result_schema, columns)
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e.to_string()))
    }

    /// Hash a value at given index in an array
    fn hash_array_value(array: &ArrayRef, idx: usize) -> u64 {
        use std::hash::{Hash, Hasher};
        use std::collections::hash_map::DefaultHasher;
        
        let mut hasher = DefaultHasher::new();
        
        if array.is_null(idx) {
            0u64.hash(&mut hasher);
        } else if let Some(arr) = array.as_any().downcast_ref::<Int64Array>() {
            arr.value(idx).hash(&mut hasher);
        } else if let Some(arr) = array.as_any().downcast_ref::<StringArray>() {
            arr.value(idx).hash(&mut hasher);
        } else if let Some(arr) = array.as_any().downcast_ref::<Float64Array>() {
            arr.value(idx).to_bits().hash(&mut hasher);
        } else if let Some(arr) = array.as_any().downcast_ref::<BooleanArray>() {
            arr.value(idx).hash(&mut hasher);
        } else {
            idx.hash(&mut hasher);
        }
        
        hasher.finish()
    }

    /// Check if two array values are equal at given indices
    fn arrays_equal_at(left: &ArrayRef, left_idx: usize, right: &ArrayRef, right_idx: usize) -> bool {
        if left.is_null(left_idx) && right.is_null(right_idx) {
            return true;
        }
        if left.is_null(left_idx) || right.is_null(right_idx) {
            return false;
        }

        if let (Some(l), Some(r)) = (
            left.as_any().downcast_ref::<Int64Array>(),
            right.as_any().downcast_ref::<Int64Array>(),
        ) {
            return l.value(left_idx) == r.value(right_idx);
        }
        if let (Some(l), Some(r)) = (
            left.as_any().downcast_ref::<StringArray>(),
            right.as_any().downcast_ref::<StringArray>(),
        ) {
            return l.value(left_idx) == r.value(right_idx);
        }
        if let (Some(l), Some(r)) = (
            left.as_any().downcast_ref::<Float64Array>(),
            right.as_any().downcast_ref::<Float64Array>(),
        ) {
            return (l.value(left_idx) - r.value(right_idx)).abs() < f64::EPSILON;
        }
        
        false
    }

    /// Take values from array with optional null indices (for LEFT JOIN)
    fn take_with_nulls(array: &ArrayRef, indices: &[Option<u32>]) -> io::Result<ArrayRef> {
        use arrow::array::*;
        
        let len = indices.len();
        
        if let Some(arr) = array.as_any().downcast_ref::<Int64Array>() {
            let values: Vec<Option<i64>> = indices.iter().map(|opt_idx| {
                opt_idx.map(|idx| arr.value(idx as usize))
            }).collect();
            return Ok(Arc::new(Int64Array::from(values)));
        }
        
        if let Some(arr) = array.as_any().downcast_ref::<Float64Array>() {
            let values: Vec<Option<f64>> = indices.iter().map(|opt_idx| {
                opt_idx.map(|idx| arr.value(idx as usize))
            }).collect();
            return Ok(Arc::new(Float64Array::from(values)));
        }
        
        if let Some(arr) = array.as_any().downcast_ref::<StringArray>() {
            let values: Vec<Option<&str>> = indices.iter().map(|opt_idx| {
                opt_idx.map(|idx| arr.value(idx as usize))
            }).collect();
            return Ok(Arc::new(StringArray::from(values)));
        }
        
        if let Some(arr) = array.as_any().downcast_ref::<BooleanArray>() {
            let values: Vec<Option<bool>> = indices.iter().map(|opt_idx| {
                opt_idx.map(|idx| arr.value(idx as usize))
            }).collect();
            return Ok(Arc::new(BooleanArray::from(values)));
        }

        // Fallback: create null array
        Ok(arrow::array::new_null_array(array.data_type(), len))
    }

    /// Apply WHERE clause filter using Arrow compute
    fn apply_filter(batch: &RecordBatch, expr: &SqlExpr) -> io::Result<RecordBatch> {
        let mask = Self::evaluate_predicate(batch, expr)?;
        compute::filter_record_batch(batch, &mask)
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e.to_string()))
    }

    /// Apply WHERE clause filter with storage path (for subquery support)
    fn apply_filter_with_storage(batch: &RecordBatch, expr: &SqlExpr, storage_path: &Path) -> io::Result<RecordBatch> {
        let mask = Self::evaluate_predicate_with_storage(batch, expr, storage_path)?;
        compute::filter_record_batch(batch, &mask)
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e.to_string()))
    }

    /// Evaluate a predicate expression to a boolean mask
    fn evaluate_predicate(batch: &RecordBatch, expr: &SqlExpr) -> io::Result<BooleanArray> {
        use crate::query::sql_parser::{BinaryOperator, UnaryOperator};
        
        match expr {
            SqlExpr::BinaryOp { left, op, right } => {
                // Handle logical operators (AND, OR)
                match op {
                    BinaryOperator::And => {
                        let left_mask = Self::evaluate_predicate(batch, left)?;
                        let right_mask = Self::evaluate_predicate(batch, right)?;
                        compute::and(&left_mask, &right_mask)
                            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e.to_string()))
                    }
                    BinaryOperator::Or => {
                        let left_mask = Self::evaluate_predicate(batch, left)?;
                        let right_mask = Self::evaluate_predicate(batch, right)?;
                        compute::or(&left_mask, &right_mask)
                            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e.to_string()))
                    }
                    // Comparison operators
                    _ => Self::evaluate_comparison(batch, left, op, right)
                }
            }
            SqlExpr::UnaryOp { op, expr } => {
                match op {
                    UnaryOperator::Not => {
                        let inner_mask = Self::evaluate_predicate(batch, expr)?;
                        compute::not(&inner_mask)
                            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e.to_string()))
                    }
                    _ => Err(io::Error::new(
                        io::ErrorKind::Unsupported,
                        "Unsupported unary operator in predicate",
                    ))
                }
            }
            SqlExpr::IsNull { column, negated } => {
                let col_name = column.trim_matches('"');
                let array = batch.column_by_name(col_name)
                    .ok_or_else(|| io::Error::new(io::ErrorKind::NotFound, format!("Column '{}' not found", col_name)))?;
                let null_mask = compute::is_null(array)
                    .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e.to_string()))?;
                if *negated {
                    compute::not(&null_mask)
                        .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e.to_string()))
                } else {
                    Ok(null_mask)
                }
            }
            SqlExpr::Between { column, low, high, negated } => {
                let col_name = column.trim_matches('"');
                let val = batch.column_by_name(col_name)
                    .ok_or_else(|| io::Error::new(io::ErrorKind::NotFound, format!("Column '{}' not found", col_name)))?;
                let low_val = Self::evaluate_expr_to_array(batch, low)?;
                let high_val = Self::evaluate_expr_to_array(batch, high)?;

                let (val_for_cmp, low_for_cmp) = Self::coerce_numeric_for_comparison(val.clone(), low_val)?;
                let (val_for_cmp2, high_for_cmp) = Self::coerce_numeric_for_comparison(val.clone(), high_val)?;
                
                let ge_low = cmp::gt_eq(&val_for_cmp, &low_for_cmp)
                    .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e.to_string()))?;
                let le_high = cmp::lt_eq(&val_for_cmp2, &high_for_cmp)
                    .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e.to_string()))?;
                
                let result = compute::and(&ge_low, &le_high)
                    .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e.to_string()))?;
                
                if *negated {
                    compute::not(&result)
                        .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e.to_string()))
                } else {
                    Ok(result)
                }
            }
            SqlExpr::In { column, values, negated } => {
                Self::evaluate_in_values(batch, column, values, *negated)
            }
            SqlExpr::Like { column, pattern, negated } => {
                Self::evaluate_like(batch, column, pattern, *negated)
            }
            SqlExpr::Regexp { column, pattern, negated } => {
                Self::evaluate_regexp(batch, column, pattern, *negated)
            }
            SqlExpr::Paren(inner) => {
                Self::evaluate_predicate(batch, inner)
            }
            // Subqueries require storage path - return error for now, will be handled by evaluate_predicate_with_storage
            SqlExpr::InSubquery { .. } | SqlExpr::ExistsSubquery { .. } | SqlExpr::ScalarSubquery { .. } => {
                Err(io::Error::new(
                    io::ErrorKind::Unsupported,
                    "Subqueries require storage path - use evaluate_predicate_with_storage",
                ))
            }
            _ => {
                Err(io::Error::new(
                    io::ErrorKind::Unsupported,
                    format!("Unsupported expression type in predicate: {:?}", expr),
                ))
            }
        }
    }

    /// Evaluate predicate with storage path (for subqueries)
    fn evaluate_predicate_with_storage(batch: &RecordBatch, expr: &SqlExpr, storage_path: &Path) -> io::Result<BooleanArray> {
        use crate::query::sql_parser::{BinaryOperator, UnaryOperator};
        
        match expr {
            SqlExpr::BinaryOp { left, op, right } => {
                match op {
                    BinaryOperator::And => {
                        let left_mask = Self::evaluate_predicate_with_storage(batch, left, storage_path)?;
                        let right_mask = Self::evaluate_predicate_with_storage(batch, right, storage_path)?;
                        compute::and(&left_mask, &right_mask)
                            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e.to_string()))
                    }
                    BinaryOperator::Or => {
                        let left_mask = Self::evaluate_predicate_with_storage(batch, left, storage_path)?;
                        let right_mask = Self::evaluate_predicate_with_storage(batch, right, storage_path)?;
                        compute::or(&left_mask, &right_mask)
                            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e.to_string()))
                    }
                    _ => Self::evaluate_comparison_with_storage(batch, left, op, right, storage_path)
                }
            }
            SqlExpr::UnaryOp { op, expr } => {
                match op {
                    UnaryOperator::Not => {
                        let inner_mask = Self::evaluate_predicate_with_storage(batch, expr, storage_path)?;
                        compute::not(&inner_mask)
                            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e.to_string()))
                    }
                    _ => Err(io::Error::new(io::ErrorKind::Unsupported, "Unsupported unary operator"))
                }
            }
            SqlExpr::InSubquery { column, stmt, negated } => {
                Self::evaluate_in_subquery(batch, column, stmt, *negated, storage_path)
            }
            SqlExpr::ExistsSubquery { stmt } => {
                Self::evaluate_exists_subquery(batch, stmt, storage_path)
            }
            SqlExpr::ScalarSubquery { .. } => {
                // Scalar subquery alone in predicate - treat as boolean (non-null = true)
                let val = Self::evaluate_expr_to_array_with_storage(batch, expr, storage_path)?;
                let mut results = Vec::with_capacity(batch.num_rows());
                for i in 0..batch.num_rows() {
                    results.push(!val.is_null(i));
                }
                Ok(BooleanArray::from(results))
            }
            SqlExpr::Paren(inner) => {
                Self::evaluate_predicate_with_storage(batch, inner, storage_path)
            }
            // Delegate non-subquery expressions to regular evaluate_predicate
            _ => Self::evaluate_predicate(batch, expr)
        }
    }

    /// Execute IN subquery (supports correlated subqueries)
    fn evaluate_in_subquery(
        batch: &RecordBatch, 
        column: &str, 
        stmt: &SelectStatement, 
        negated: bool,
        storage_path: &Path
    ) -> io::Result<BooleanArray> {
        // Resolve the subquery's table path from its FROM clause
        let subquery_path = Self::resolve_subquery_table_path(stmt, storage_path)?;
        
        // Check if this is a correlated subquery
        let outer_cols = Self::find_outer_column_refs(stmt, batch);
        
        let col_name = column.trim_matches('"');
        // Strip table alias (e.g., "u.user_id" -> "user_id")
        let lookup_name = if let Some(dot_pos) = col_name.find('.') {
            &col_name[dot_pos + 1..]
        } else {
            col_name
        };
        let main_col = batch.column_by_name(lookup_name)
            .ok_or_else(|| io::Error::new(io::ErrorKind::NotFound, format!("Column '{}' not found", col_name)))?;
        
        if outer_cols.is_empty() {
            // Non-correlated: execute once
            let sub_result = Self::execute_select(stmt.clone(), &subquery_path)?;
            let sub_batch = sub_result.to_record_batch()?;
            
            if sub_batch.num_rows() == 0 {
                return Ok(BooleanArray::from(vec![negated; batch.num_rows()]));
            }
            
            if sub_batch.num_columns() == 0 {
                return Err(io::Error::new(io::ErrorKind::InvalidData, "Subquery must return at least one column"));
            }
            let sub_col = sub_batch.column(0);
            
            // Build hash set of subquery values
            let mut value_set: HashSet<u64> = HashSet::with_capacity(sub_batch.num_rows());
            for i in 0..sub_batch.num_rows() {
                if !sub_col.is_null(i) {
                    value_set.insert(Self::hash_array_value(sub_col, i));
                }
            }
            
            let mut results = Vec::with_capacity(batch.num_rows());
            for i in 0..batch.num_rows() {
                let hash = Self::hash_array_value(main_col, i);
                let found = value_set.contains(&hash);
                results.push(if negated { !found } else { found });
            }
            
            Ok(BooleanArray::from(results))
        } else {
            // Correlated: evaluate for each row
            let mut results = Vec::with_capacity(batch.num_rows());
            
            for row_idx in 0..batch.num_rows() {
                let modified_stmt = Self::substitute_outer_refs(stmt, batch, row_idx, &outer_cols);
                
                let sub_result = Self::execute_select(modified_stmt, &subquery_path)?;
                let sub_batch = sub_result.to_record_batch()?;
                
                let found = if sub_batch.num_rows() == 0 || sub_batch.num_columns() == 0 {
                    false
                } else {
                    let sub_col = sub_batch.column(0);
                    let main_hash = Self::hash_array_value(main_col, row_idx);
                    (0..sub_batch.num_rows()).any(|i| !sub_col.is_null(i) && Self::hash_array_value(sub_col, i) == main_hash)
                };
                
                results.push(if negated { !found } else { found });
            }
            
            Ok(BooleanArray::from(results))
        }
    }

    /// Execute EXISTS subquery (supports correlated subqueries)
    fn evaluate_exists_subquery(
        batch: &RecordBatch, 
        stmt: &SelectStatement, 
        storage_path: &Path
    ) -> io::Result<BooleanArray> {
        // Resolve the subquery's table path from its FROM clause
        let subquery_path = Self::resolve_subquery_table_path(stmt, storage_path)?;
        
        // Check if this is a correlated subquery by looking for outer column references
        let outer_cols = Self::find_outer_column_refs(stmt, batch);
        
        if outer_cols.is_empty() {
            // Non-correlated: execute once
            let sub_result = Self::execute_select(stmt.clone(), &subquery_path)?;
            let sub_batch = sub_result.to_record_batch()?;
            let exists = sub_batch.num_rows() > 0;
            Ok(BooleanArray::from(vec![exists; batch.num_rows()]))
        } else {
            // Correlated: evaluate for each row
            let mut results = Vec::with_capacity(batch.num_rows());
            
            for row_idx in 0..batch.num_rows() {
                // Build modified subquery with outer values substituted
                let modified_stmt = Self::substitute_outer_refs(stmt, batch, row_idx, &outer_cols);
                
                let sub_result = Self::execute_select(modified_stmt, &subquery_path)?;
                let sub_batch = sub_result.to_record_batch()?;
                results.push(sub_batch.num_rows() > 0);
            }
            
            Ok(BooleanArray::from(results))
        }
    }
    
    /// Resolve subquery's table path from its FROM clause
    fn resolve_subquery_table_path(stmt: &SelectStatement, main_storage_path: &Path) -> io::Result<std::path::PathBuf> {
        // Get table name from subquery's FROM clause
        if let Some(FromItem::Table { table, .. }) = &stmt.from {
            let base_dir = main_storage_path.parent()
                .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "Cannot determine base directory"))?;
            Ok(Self::resolve_table_path(table, base_dir, main_storage_path))
        } else {
            // No FROM or derived table - use main storage path
            Ok(main_storage_path.to_path_buf())
        }
    }
    
    /// Find column references in subquery that refer to outer query columns
    fn find_outer_column_refs(stmt: &SelectStatement, outer_batch: &RecordBatch) -> Vec<String> {
        let mut outer_refs = Vec::new();
        let outer_cols: Vec<String> = outer_batch.schema().fields().iter()
            .map(|f| f.name().clone())
            .collect();
        
        // Get subquery's table alias to exclude from outer refs
        let subquery_alias = match &stmt.from {
            Some(FromItem::Table { alias, table, .. }) => {
                alias.clone().unwrap_or_else(|| table.clone())
            }
            _ => String::new(),
        };
        
        // Check WHERE clause for outer column references
        if let Some(where_clause) = &stmt.where_clause {
            Self::collect_outer_refs_from_expr(where_clause, &outer_cols, &subquery_alias, &mut outer_refs);
        }
        
        outer_refs
    }
    
    /// Recursively collect outer column references from expression
    fn collect_outer_refs_from_expr(expr: &SqlExpr, outer_cols: &[String], subquery_alias: &str, refs: &mut Vec<String>) {
        match expr {
            SqlExpr::Column(name) => {
                let clean_name = name.trim_matches('"');
                // Check if column has table qualifier like "u.id" or "outer.col"
                if let Some(dot_pos) = clean_name.find('.') {
                    let table_part = &clean_name[..dot_pos];
                    let col_part = &clean_name[dot_pos + 1..];
                    
                    // Skip if the table prefix matches the subquery's table alias
                    // (e.g., "o.user_id" when subquery is "FROM orders o")
                    if !subquery_alias.is_empty() && table_part == subquery_alias {
                        return;
                    }
                    
                    // Check if the column part exists in outer batch columns
                    if outer_cols.iter().any(|c| c == col_part || c.trim_matches('"') == col_part) {
                        if !refs.contains(&clean_name.to_string()) {
                            refs.push(clean_name.to_string());
                        }
                    }
                }
            }
            SqlExpr::BinaryOp { left, right, .. } => {
                Self::collect_outer_refs_from_expr(left, outer_cols, subquery_alias, refs);
                Self::collect_outer_refs_from_expr(right, outer_cols, subquery_alias, refs);
            }
            SqlExpr::UnaryOp { expr: inner, .. } | SqlExpr::Paren(inner) => {
                Self::collect_outer_refs_from_expr(inner, outer_cols, subquery_alias, refs);
            }
            _ => {}
        }
    }
    
    /// Substitute outer column references with actual values for a specific row
    fn substitute_outer_refs(stmt: &SelectStatement, outer_batch: &RecordBatch, row_idx: usize, outer_refs: &[String]) -> SelectStatement {
        let mut new_stmt = stmt.clone();
        
        if let Some(where_clause) = &mut new_stmt.where_clause {
            *where_clause = Self::substitute_expr(where_clause, outer_batch, row_idx, outer_refs);
        }
        
        new_stmt
    }
    
    /// Substitute outer column references in expression with literal values
    fn substitute_expr(expr: &SqlExpr, outer_batch: &RecordBatch, row_idx: usize, outer_refs: &[String]) -> SqlExpr {
        match expr {
            SqlExpr::Column(name) => {
                let clean_name = name.trim_matches('"');
                if outer_refs.iter().any(|r| r == clean_name) {
                    // This is an outer reference - substitute with value
                    if let Some(dot_pos) = clean_name.find('.') {
                        let col_part = &clean_name[dot_pos + 1..];
                        if let Some(col) = outer_batch.column_by_name(col_part) {
                            return Self::array_value_to_literal(col, row_idx);
                        }
                    }
                }
                expr.clone()
            }
            SqlExpr::BinaryOp { left, op, right } => {
                SqlExpr::BinaryOp {
                    left: Box::new(Self::substitute_expr(left, outer_batch, row_idx, outer_refs)),
                    op: op.clone(),
                    right: Box::new(Self::substitute_expr(right, outer_batch, row_idx, outer_refs)),
                }
            }
            SqlExpr::UnaryOp { op, expr: inner } => {
                SqlExpr::UnaryOp {
                    op: op.clone(),
                    expr: Box::new(Self::substitute_expr(inner, outer_batch, row_idx, outer_refs)),
                }
            }
            SqlExpr::Paren(inner) => {
                SqlExpr::Paren(Box::new(Self::substitute_expr(inner, outer_batch, row_idx, outer_refs)))
            }
            _ => expr.clone(),
        }
    }
    
    /// Convert array value at index to literal expression
    fn array_value_to_literal(array: &ArrayRef, idx: usize) -> SqlExpr {
        use crate::data::Value;
        
        if array.is_null(idx) {
            return SqlExpr::Literal(Value::Null);
        }
        
        if let Some(arr) = array.as_any().downcast_ref::<Int64Array>() {
            SqlExpr::Literal(Value::Int64(arr.value(idx)))
        } else if let Some(arr) = array.as_any().downcast_ref::<Float64Array>() {
            SqlExpr::Literal(Value::Float64(arr.value(idx)))
        } else if let Some(arr) = array.as_any().downcast_ref::<StringArray>() {
            SqlExpr::Literal(Value::String(arr.value(idx).to_string()))
        } else if let Some(arr) = array.as_any().downcast_ref::<BooleanArray>() {
            SqlExpr::Literal(Value::Bool(arr.value(idx)))
        } else {
            SqlExpr::Literal(Value::Null)
        }
    }

    /// Evaluate comparison operator
    fn evaluate_comparison(
        batch: &RecordBatch,
        left: &SqlExpr,
        op: &crate::query::sql_parser::BinaryOperator,
        right: &SqlExpr,
    ) -> io::Result<BooleanArray> {
        use crate::query::sql_parser::BinaryOperator;
        
        let left_array = Self::evaluate_expr_to_array(batch, left)?;
        let right_array = Self::evaluate_expr_to_array(batch, right)?;

        let (left_array, right_array) = Self::coerce_numeric_for_comparison(left_array, right_array)?;

        let result = match op {
            BinaryOperator::Eq => cmp::eq(&left_array, &right_array),
            BinaryOperator::NotEq => cmp::neq(&left_array, &right_array),
            BinaryOperator::Lt => cmp::lt(&left_array, &right_array),
            BinaryOperator::Le => cmp::lt_eq(&left_array, &right_array),
            BinaryOperator::Gt => cmp::gt(&left_array, &right_array),
            BinaryOperator::Ge => cmp::gt_eq(&left_array, &right_array),
            _ => return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!("Unsupported comparison operator: {:?}", op),
            )),
        };

        result.map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e.to_string()))
    }

    /// Evaluate comparison with storage path (for scalar subqueries)
    fn evaluate_comparison_with_storage(
        batch: &RecordBatch,
        left: &SqlExpr,
        op: &crate::query::sql_parser::BinaryOperator,
        right: &SqlExpr,
        storage_path: &Path,
    ) -> io::Result<BooleanArray> {
        use crate::query::sql_parser::BinaryOperator;
        
        let left_array = Self::evaluate_expr_to_array_with_storage(batch, left, storage_path)?;
        let right_array = Self::evaluate_expr_to_array_with_storage(batch, right, storage_path)?;

        let (left_array, right_array) = Self::coerce_numeric_for_comparison(left_array, right_array)?;

        let result = match op {
            BinaryOperator::Eq => cmp::eq(&left_array, &right_array),
            BinaryOperator::NotEq => cmp::neq(&left_array, &right_array),
            BinaryOperator::Lt => cmp::lt(&left_array, &right_array),
            BinaryOperator::Le => cmp::lt_eq(&left_array, &right_array),
            BinaryOperator::Gt => cmp::gt(&left_array, &right_array),
            BinaryOperator::Ge => cmp::gt_eq(&left_array, &right_array),
            _ => {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidInput,
                    format!("Unsupported comparison operator: {:?}", op),
                ))
            }
        };

        result.map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e.to_string()))
    }

    /// Evaluate expression to Arrow array
    fn evaluate_expr_to_array(batch: &RecordBatch, expr: &SqlExpr) -> io::Result<ArrayRef> {
        match expr {
            SqlExpr::Column(name) => {
                let col_name = name.trim_matches('"');
                // Strip table alias prefix if present (e.g., "o.user_id" -> "user_id")
                let actual_col = if let Some(dot_pos) = col_name.rfind('.') {
                    &col_name[dot_pos + 1..]
                } else {
                    col_name
                };
                batch
                    .column_by_name(actual_col)
                    .cloned()
                    .or_else(|| batch.column_by_name(col_name).cloned())
                    .ok_or_else(|| io::Error::new(
                        io::ErrorKind::NotFound,
                        format!("Column '{}' not found", col_name),
                    ))
            }
            SqlExpr::Literal(val) => {
                Self::value_to_array(val, batch.num_rows())
            }
            SqlExpr::BinaryOp { left, op, right } => {
                Self::evaluate_arithmetic_op(batch, left, op, right)
            }
            SqlExpr::Case { when_then, else_expr } => {
                Self::evaluate_case_expr(batch, when_then, else_expr.as_deref())
            }
            SqlExpr::Function { name, args } => {
                Self::evaluate_function_expr(batch, name, args)
            }
            SqlExpr::Cast { expr: inner, data_type } => {
                Self::evaluate_cast_expr(batch, inner, data_type)
            }
            SqlExpr::Paren(inner) => {
                Self::evaluate_expr_to_array(batch, inner)
            }
            _ => Err(io::Error::new(
                io::ErrorKind::Unsupported,
                format!("Unsupported expression type: {:?}", expr),
            )),
        }
    }

    /// Evaluate expression to Arrow array with storage path (for scalar subqueries)
    fn evaluate_expr_to_array_with_storage(batch: &RecordBatch, expr: &SqlExpr, storage_path: &Path) -> io::Result<ArrayRef> {
        match expr {
            SqlExpr::ScalarSubquery { stmt } => {
                Self::evaluate_scalar_subquery(batch, stmt, storage_path)
            }
            SqlExpr::BinaryOp { left, op, right } => {
                // Check if operands contain scalar subqueries
                let left_array = Self::evaluate_expr_to_array_with_storage(batch, left, storage_path)?;
                let right_array = Self::evaluate_expr_to_array_with_storage(batch, right, storage_path)?;
                Self::evaluate_arithmetic_op_arrays(&left_array, &right_array, op)
            }
            // Delegate non-subquery expressions
            _ => Self::evaluate_expr_to_array(batch, expr)
        }
    }

    /// Execute scalar subquery and broadcast result to array
    fn evaluate_scalar_subquery(batch: &RecordBatch, stmt: &SelectStatement, storage_path: &Path) -> io::Result<ArrayRef> {
        // Resolve the subquery's table path from its FROM clause
        let subquery_path = Self::resolve_subquery_table_path(stmt, storage_path)?;
        
        // Check if this is a correlated subquery
        let outer_cols = Self::find_outer_column_refs(stmt, batch);
        
        if outer_cols.is_empty() {
            // Non-correlated: execute once and broadcast to all rows
            let sub_result = Self::execute_select(stmt.clone(), &subquery_path)?;
            let sub_batch = sub_result.to_record_batch()?;
            
            if sub_batch.num_rows() == 0 || sub_batch.num_columns() == 0 {
                return Ok(Arc::new(Int64Array::from(vec![None::<i64>; batch.num_rows()])));
            }
            
            if sub_batch.num_rows() > 1 {
                return Err(io::Error::new(io::ErrorKind::InvalidData, "Scalar subquery returned more than one row"));
            }
            
            let sub_col = sub_batch.column(0);
            return Self::broadcast_scalar_array(sub_col, 0, batch.num_rows());
        }
        
        // Correlated: execute for each row
        let mut results: Vec<Option<i64>> = Vec::with_capacity(batch.num_rows());
        
        for row_idx in 0..batch.num_rows() {
            let modified_stmt = Self::substitute_outer_refs(stmt, batch, row_idx, &outer_cols);
            let sub_result = Self::execute_select(modified_stmt, &subquery_path)?;
            let sub_batch = sub_result.to_record_batch()?;
            
            if sub_batch.num_rows() == 0 || sub_batch.num_columns() == 0 {
                results.push(None);
            } else if sub_batch.num_rows() > 1 {
                return Err(io::Error::new(io::ErrorKind::InvalidData, "Scalar subquery returned more than one row"));
            } else {
                let sub_col = sub_batch.column(0);
                if sub_col.is_null(0) {
                    results.push(None);
                } else if let Some(arr) = sub_col.as_any().downcast_ref::<Int64Array>() {
                    results.push(Some(arr.value(0)));
                } else if let Some(arr) = sub_col.as_any().downcast_ref::<Float64Array>() {
                    results.push(Some(arr.value(0) as i64));
                } else {
                    results.push(None);
                }
            }
        }
        
        return Ok(Arc::new(Int64Array::from(results)));
    }
    
    /// Execute non-correlated scalar subquery (original implementation)
    fn evaluate_scalar_subquery_simple(batch: &RecordBatch, stmt: &SelectStatement, storage_path: &Path) -> io::Result<ArrayRef> {
        let sub_result = Self::execute_select(stmt.clone(), storage_path)?;
        let sub_batch = sub_result.to_record_batch()?;
        
        if sub_batch.num_rows() == 0 || sub_batch.num_columns() == 0 {
            // Return null array
            return Ok(Arc::new(Int64Array::from(vec![None::<i64>; batch.num_rows()])));
        }
        
        if sub_batch.num_rows() > 1 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData, 
                "Scalar subquery returned more than one row"
            ));
        }
        
        // Get single value and broadcast
        let sub_col = sub_batch.column(0);
        Self::broadcast_scalar_array(sub_col, 0, batch.num_rows())
    }

    /// Broadcast a single value from array to num_rows
    fn broadcast_scalar_array(array: &ArrayRef, idx: usize, num_rows: usize) -> io::Result<ArrayRef> {
        if array.is_null(idx) {
            return Ok(Arc::new(Int64Array::from(vec![None::<i64>; num_rows])));
        }
        
        use arrow::datatypes::DataType;
        Ok(match array.data_type() {
            DataType::Int64 => {
                let arr = array.as_any().downcast_ref::<Int64Array>().unwrap();
                Arc::new(Int64Array::from(vec![arr.value(idx); num_rows]))
            }
            DataType::Float64 => {
                let arr = array.as_any().downcast_ref::<Float64Array>().unwrap();
                Arc::new(Float64Array::from(vec![arr.value(idx); num_rows]))
            }
            DataType::Utf8 => {
                let arr = array.as_any().downcast_ref::<StringArray>().unwrap();
                Arc::new(StringArray::from(vec![arr.value(idx); num_rows]))
            }
            DataType::Boolean => {
                let arr = array.as_any().downcast_ref::<BooleanArray>().unwrap();
                Arc::new(BooleanArray::from(vec![arr.value(idx); num_rows]))
            }
            _ => {
                // Fallback - try Int64
                Arc::new(Int64Array::from(vec![None::<i64>; num_rows]))
            }
        })
    }

    /// Evaluate arithmetic operation on pre-computed arrays
    fn evaluate_arithmetic_op_arrays(left: &ArrayRef, right: &ArrayRef, op: &crate::query::sql_parser::BinaryOperator) -> io::Result<ArrayRef> {
        use crate::query::sql_parser::BinaryOperator;
        use arrow::compute::kernels::numeric;
        
        let result: ArrayRef = match op {
            BinaryOperator::Add => Arc::new(numeric::add(
                left.as_any().downcast_ref::<Int64Array>().ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "Expected Int64"))?,
                right.as_any().downcast_ref::<Int64Array>().ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "Expected Int64"))?
            ).map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e.to_string()))?),
            BinaryOperator::Sub => Arc::new(numeric::sub(
                left.as_any().downcast_ref::<Int64Array>().ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "Expected Int64"))?,
                right.as_any().downcast_ref::<Int64Array>().ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "Expected Int64"))?
            ).map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e.to_string()))?),
            BinaryOperator::Mul => Arc::new(numeric::mul(
                left.as_any().downcast_ref::<Int64Array>().ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "Expected Int64"))?,
                right.as_any().downcast_ref::<Int64Array>().ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "Expected Int64"))?
            ).map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e.to_string()))?),
            BinaryOperator::Div => Arc::new(numeric::div(
                left.as_any().downcast_ref::<Int64Array>().ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "Expected Int64"))?,
                right.as_any().downcast_ref::<Int64Array>().ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "Expected Int64"))?
            ).map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e.to_string()))?),
            _ => return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!("Unsupported arithmetic operator: {:?}", op),
            )),
        };
        Ok(result)
    }

    /// Convert Value to Arrow array (broadcast to num_rows)
    fn value_to_array(val: &Value, num_rows: usize) -> io::Result<ArrayRef> {
        Ok(match val {
            Value::Int64(i) => Arc::new(Int64Array::from(vec![*i; num_rows])),
            Value::Int32(i) => Arc::new(Int64Array::from(vec![*i as i64; num_rows])),
            Value::Int16(i) => Arc::new(Int64Array::from(vec![*i as i64; num_rows])),
            Value::Int8(i) => Arc::new(Int64Array::from(vec![*i as i64; num_rows])),
            Value::UInt64(i) => Arc::new(Int64Array::from(vec![*i as i64; num_rows])),
            Value::UInt32(i) => Arc::new(Int64Array::from(vec![*i as i64; num_rows])),
            Value::UInt16(i) => Arc::new(Int64Array::from(vec![*i as i64; num_rows])),
            Value::UInt8(i) => Arc::new(Int64Array::from(vec![*i as i64; num_rows])),
            Value::Float64(f) => Arc::new(Float64Array::from(vec![*f; num_rows])),
            Value::Float32(f) => Arc::new(Float64Array::from(vec![*f as f64; num_rows])),
            Value::String(s) => Arc::new(StringArray::from(vec![s.as_str(); num_rows])),
            Value::Bool(b) => Arc::new(BooleanArray::from(vec![*b; num_rows])),
            Value::Null => Arc::new(Int64Array::from(vec![None::<i64>; num_rows])),
            Value::Binary(b) => {
                use arrow::array::BinaryArray;
                Arc::new(BinaryArray::from(vec![Some(b.as_slice()); num_rows]))
            }
            Value::Json(j) => {
                let s = j.to_string();
                Arc::new(StringArray::from(vec![s.as_str(); num_rows]))
            }
            Value::Timestamp(ts) => Arc::new(Int64Array::from(vec![*ts; num_rows])),
            Value::Date(d) => Arc::new(Int64Array::from(vec![*d as i64; num_rows])),
            Value::Array(_) => {
                return Err(io::Error::new(
                    io::ErrorKind::Unsupported,
                    "Array values not supported in expressions",
                ));
            }
        })
    }

    /// Evaluate arithmetic expression
    fn evaluate_arithmetic_op(
        batch: &RecordBatch,
        left: &SqlExpr,
        op: &crate::query::sql_parser::BinaryOperator,
        right: &SqlExpr,
    ) -> io::Result<ArrayRef> {
        use crate::query::sql_parser::BinaryOperator;
        
        let left_array = Self::evaluate_expr_to_array(batch, left)?;
        let right_array = Self::evaluate_expr_to_array(batch, right)?;

        // Try to cast to common numeric type
        let result: ArrayRef = match op {
            BinaryOperator::Add => {
                if let (Some(l), Some(r)) = (
                    left_array.as_any().downcast_ref::<Int64Array>(),
                    right_array.as_any().downcast_ref::<Int64Array>(),
                ) {
                    Arc::new(arith::add(l, r)
                        .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e.to_string()))?)
                } else if let (Some(l), Some(r)) = (
                    left_array.as_any().downcast_ref::<Float64Array>(),
                    right_array.as_any().downcast_ref::<Float64Array>(),
                ) {
                    Arc::new(arith::add(l, r)
                        .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e.to_string()))?)
                } else {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidData,
                        "Cannot add non-numeric types",
                    ));
                }
            }
            BinaryOperator::Sub => {
                if let (Some(l), Some(r)) = (
                    left_array.as_any().downcast_ref::<Int64Array>(),
                    right_array.as_any().downcast_ref::<Int64Array>(),
                ) {
                    Arc::new(arith::sub(l, r)
                        .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e.to_string()))?)
                } else if let (Some(l), Some(r)) = (
                    left_array.as_any().downcast_ref::<Float64Array>(),
                    right_array.as_any().downcast_ref::<Float64Array>(),
                ) {
                    Arc::new(arith::sub(l, r)
                        .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e.to_string()))?)
                } else {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidData,
                        "Cannot subtract non-numeric types",
                    ));
                }
            }
            BinaryOperator::Mul => {
                if let (Some(l), Some(r)) = (
                    left_array.as_any().downcast_ref::<Int64Array>(),
                    right_array.as_any().downcast_ref::<Int64Array>(),
                ) {
                    Arc::new(arith::mul(l, r)
                        .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e.to_string()))?)
                } else if let (Some(l), Some(r)) = (
                    left_array.as_any().downcast_ref::<Float64Array>(),
                    right_array.as_any().downcast_ref::<Float64Array>(),
                ) {
                    Arc::new(arith::mul(l, r)
                        .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e.to_string()))?)
                } else {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidData,
                        "Cannot multiply non-numeric types",
                    ));
                }
            }
            BinaryOperator::Div => {
                if let (Some(l), Some(r)) = (
                    left_array.as_any().downcast_ref::<Int64Array>(),
                    right_array.as_any().downcast_ref::<Int64Array>(),
                ) {
                    Arc::new(arith::div(l, r)
                        .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e.to_string()))?)
                } else if let (Some(l), Some(r)) = (
                    left_array.as_any().downcast_ref::<Float64Array>(),
                    right_array.as_any().downcast_ref::<Float64Array>(),
                ) {
                    Arc::new(arith::div(l, r)
                        .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e.to_string()))?)
                } else {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidData,
                        "Cannot divide non-numeric types",
                    ));
                }
            }
            _ => {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidInput,
                    format!("Unsupported arithmetic operator: {:?}", op),
                ));
            }
        };

        Ok(result)
    }

    /// Evaluate CAST expression
    fn evaluate_cast_expr(
        batch: &RecordBatch,
        expr: &SqlExpr,
        target_type: &DataType,
    ) -> io::Result<ArrayRef> {
        let arr = Self::evaluate_expr_to_array(batch, expr)?;
        let num_rows = batch.num_rows();
        
        match target_type {
            DataType::Int64 => {
                if let Some(int_arr) = arr.as_any().downcast_ref::<Int64Array>() {
                    Ok(Arc::new(int_arr.clone()))
                } else if let Some(float_arr) = arr.as_any().downcast_ref::<Float64Array>() {
                    let result: Vec<Option<i64>> = (0..num_rows).map(|i| {
                        if float_arr.is_null(i) { None } else { Some(float_arr.value(i) as i64) }
                    }).collect();
                    Ok(Arc::new(Int64Array::from(result)))
                } else if let Some(str_arr) = arr.as_any().downcast_ref::<StringArray>() {
                    let mut result: Vec<Option<i64>> = Vec::with_capacity(num_rows);
                    for i in 0..num_rows {
                        if str_arr.is_null(i) {
                            result.push(None);
                            continue;
                        }
                        let s = str_arr.value(i);
                        match s.parse::<i64>() {
                            Ok(v) => result.push(Some(v)),
                            Err(_) => {
                                return Err(io::Error::new(
                                    io::ErrorKind::InvalidData,
                                    format!("Invalid cast to INT64 for value '{}'", s),
                                ));
                            }
                        }
                    }
                    Ok(Arc::new(Int64Array::from(result)))
                } else {
                    Err(io::Error::new(io::ErrorKind::InvalidData, "Cannot cast to INT64"))
                }
            }
            DataType::Float64 => {
                if let Some(float_arr) = arr.as_any().downcast_ref::<Float64Array>() {
                    Ok(Arc::new(float_arr.clone()))
                } else if let Some(int_arr) = arr.as_any().downcast_ref::<Int64Array>() {
                    let result: Vec<Option<f64>> = (0..num_rows).map(|i| {
                        if int_arr.is_null(i) { None } else { Some(int_arr.value(i) as f64) }
                    }).collect();
                    Ok(Arc::new(Float64Array::from(result)))
                } else if let Some(str_arr) = arr.as_any().downcast_ref::<StringArray>() {
                    let mut result: Vec<Option<f64>> = Vec::with_capacity(num_rows);
                    for i in 0..num_rows {
                        if str_arr.is_null(i) {
                            result.push(None);
                            continue;
                        }
                        let s = str_arr.value(i);
                        match s.parse::<f64>() {
                            Ok(v) => result.push(Some(v)),
                            Err(_) => {
                                return Err(io::Error::new(
                                    io::ErrorKind::InvalidData,
                                    format!("Invalid cast to FLOAT64 for value '{}'", s),
                                ));
                            }
                        }
                    }
                    Ok(Arc::new(Float64Array::from(result)))
                } else {
                    Err(io::Error::new(io::ErrorKind::InvalidData, "Cannot cast to FLOAT64"))
                }
            }
            DataType::String => {
                if let Some(str_arr) = arr.as_any().downcast_ref::<StringArray>() {
                    Ok(Arc::new(str_arr.clone()))
                } else if let Some(int_arr) = arr.as_any().downcast_ref::<Int64Array>() {
                    let result: Vec<Option<String>> = (0..num_rows).map(|i| {
                        if int_arr.is_null(i) { None } else { Some(int_arr.value(i).to_string()) }
                    }).collect();
                    Ok(Arc::new(StringArray::from(result.iter().map(|s| s.as_deref()).collect::<Vec<_>>())))
                } else if let Some(float_arr) = arr.as_any().downcast_ref::<Float64Array>() {
                    let result: Vec<Option<String>> = (0..num_rows).map(|i| {
                        if float_arr.is_null(i) { None } else { Some(float_arr.value(i).to_string()) }
                    }).collect();
                    Ok(Arc::new(StringArray::from(result.iter().map(|s| s.as_deref()).collect::<Vec<_>>())))
                } else {
                    Err(io::Error::new(io::ErrorKind::InvalidData, "Cannot cast to STRING"))
                }
            }
            DataType::Bool => {
                if let Some(bool_arr) = arr.as_any().downcast_ref::<BooleanArray>() {
                    Ok(Arc::new(bool_arr.clone()))
                } else if let Some(int_arr) = arr.as_any().downcast_ref::<Int64Array>() {
                    let result: Vec<Option<bool>> = (0..num_rows).map(|i| {
                        if int_arr.is_null(i) { None } else { Some(int_arr.value(i) != 0) }
                    }).collect();
                    Ok(Arc::new(BooleanArray::from(result)))
                } else if let Some(str_arr) = arr.as_any().downcast_ref::<StringArray>() {
                    let result: Vec<Option<bool>> = (0..num_rows).map(|i| {
                        if str_arr.is_null(i) { None } 
                        else { 
                            let s = str_arr.value(i).to_lowercase();
                            Some(s == "true" || s == "1" || s == "yes" || s == "t")
                        }
                    }).collect();
                    Ok(Arc::new(BooleanArray::from(result)))
                } else {
                    Err(io::Error::new(io::ErrorKind::InvalidData, "Cannot cast to BOOL"))
                }
            }
            DataType::Int32 => {
                // Treat Int32 same as Int64 for simplicity, return Int64 array
                if let Some(int_arr) = arr.as_any().downcast_ref::<Int64Array>() {
                    Ok(Arc::new(int_arr.clone()))
                } else if let Some(float_arr) = arr.as_any().downcast_ref::<Float64Array>() {
                    let result: Vec<Option<i64>> = (0..num_rows).map(|i| {
                        if float_arr.is_null(i) { None } else { Some(float_arr.value(i) as i64) }
                    }).collect();
                    Ok(Arc::new(Int64Array::from(result)))
                } else if let Some(str_arr) = arr.as_any().downcast_ref::<StringArray>() {
                    let mut result: Vec<Option<i64>> = Vec::with_capacity(num_rows);
                    for i in 0..num_rows {
                        if str_arr.is_null(i) {
                            result.push(None);
                            continue;
                        }
                        let s = str_arr.value(i);
                        match s.parse::<i64>() {
                            Ok(v) => result.push(Some(v)),
                            Err(_) => {
                                return Err(io::Error::new(
                                    io::ErrorKind::InvalidData,
                                    format!("Invalid cast to INT32 for value '{}'", s),
                                ));
                            }
                        }
                    }
                    Ok(Arc::new(Int64Array::from(result)))
                } else {
                    Err(io::Error::new(io::ErrorKind::InvalidData, "Cannot cast to INT32"))
                }
            }
            _ => Err(io::Error::new(io::ErrorKind::Unsupported, format!("CAST to {:?} not supported", target_type))),
        }
    }

    /// Evaluate CASE WHEN expression
    fn evaluate_case_expr(
        batch: &RecordBatch,
        when_then: &[(SqlExpr, SqlExpr)],
        else_expr: Option<&SqlExpr>,
    ) -> io::Result<ArrayRef> {
        let num_rows = batch.num_rows();
        
        // Determine result type from first THEN expression
        let first_then = Self::evaluate_expr_to_array(batch, &when_then[0].1)?;
        let is_string = first_then.as_any().downcast_ref::<StringArray>().is_some();
        
        if is_string {
            // Handle string CASE results
            let mut result: Vec<Option<String>> = if let Some(else_e) = else_expr {
                let else_array = Self::evaluate_expr_to_array(batch, else_e)?;
                if let Some(arr) = else_array.as_any().downcast_ref::<StringArray>() {
                    (0..num_rows).map(|i| if arr.is_null(i) { None } else { Some(arr.value(i).to_string()) }).collect()
                } else {
                    vec![None; num_rows]
                }
            } else {
                vec![None; num_rows]
            };
            
            let mut assigned = vec![false; num_rows];
            
            for (cond_expr, then_expr) in when_then {
                let cond = Self::evaluate_predicate(batch, cond_expr)?;
                let then_array = Self::evaluate_expr_to_array(batch, then_expr)?;
                
                if let Some(then_str) = then_array.as_any().downcast_ref::<StringArray>() {
                    for i in 0..num_rows {
                        if !assigned[i] && cond.value(i) {
                            result[i] = if then_str.is_null(i) { None } else { Some(then_str.value(i).to_string()) };
                            assigned[i] = true;
                        }
                    }
                }
            }
            
            Ok(Arc::new(StringArray::from(result)))
        } else {
            // Handle numeric CASE results (Int64)
            let mut result: Vec<Option<i64>> = if let Some(else_e) = else_expr {
                let else_array = Self::evaluate_expr_to_array(batch, else_e)?;
                if let Some(arr) = else_array.as_any().downcast_ref::<Int64Array>() {
                    (0..num_rows).map(|i| if arr.is_null(i) { None } else { Some(arr.value(i)) }).collect()
                } else {
                    vec![None; num_rows]
                }
            } else {
                vec![None; num_rows]
            };
            
            let mut assigned = vec![false; num_rows];
            
            for (cond_expr, then_expr) in when_then {
                let cond = Self::evaluate_predicate(batch, cond_expr)?;
                let then_array = Self::evaluate_expr_to_array(batch, then_expr)?;
                
                if let Some(then_int) = then_array.as_any().downcast_ref::<Int64Array>() {
                    for i in 0..num_rows {
                        if !assigned[i] && cond.value(i) {
                            result[i] = if then_int.is_null(i) { None } else { Some(then_int.value(i)) };
                            assigned[i] = true;
                        }
                    }
                }
            }
            
            Ok(Arc::new(Int64Array::from(result)))
        }
    }

    /// Evaluate function expression (COALESCE, etc.)
    fn evaluate_function_expr(
        batch: &RecordBatch,
        name: &str,
        args: &[SqlExpr],
    ) -> io::Result<ArrayRef> {
        let upper = name.to_uppercase();
        
        // Handle aggregate function references (for HAVING clause)
        // These should map to already-computed columns in the result batch
        match upper.as_str() {
            "COUNT" | "SUM" | "AVG" | "MIN" | "MAX" => {
                // Build possible column names as they might appear in the result batch
                let col_name = if args.is_empty() {
                    format!("{}(*)", upper)
                } else if let Some(SqlExpr::Literal(Value::String(s))) = args.first() {
                    if s == "*" {
                        format!("{}(*)", upper)
                    } else {
                        format!("{}({})", upper, s)
                    }
                } else if let Some(SqlExpr::Column(col)) = args.first() {
                    format!("{}({})", upper, col)
                } else if let Some(SqlExpr::Literal(Value::Int64(n))) = args.first() {
                    format!("{}({})", upper, n)
                } else {
                    format!("{}(*)", upper)
                };
                
                // Try to find the column in the batch by exact name
                if let Some(array) = batch.column_by_name(&col_name) {
                    return Ok(array.clone());
                }
                // Also try lowercase version
                let lower_col = col_name.to_lowercase();
                if let Some(array) = batch.column_by_name(&lower_col) {
                    return Ok(array.clone());
                }
                // Try with just the function name pattern (handles aliased columns)
                for field in batch.schema().fields() {
                    let field_upper = field.name().to_uppercase();
                    if field_upper.starts_with(&format!("{}(", upper)) {
                        return batch.column_by_name(field.name()).cloned()
                            .ok_or_else(|| io::Error::new(io::ErrorKind::NotFound, format!("Column not found: {}", col_name)));
                    }
                }
                // If column not found by name pattern, it might be aliased
                // Count how many aggregate-like columns we have (numeric, not group-by columns)
                let mut agg_columns: Vec<(usize, String)> = Vec::new();
                let mut string_columns = 0;
                for (idx, field) in batch.schema().fields().iter().enumerate() {
                    match field.data_type() {
                        arrow::datatypes::DataType::Utf8 => string_columns += 1,
                        arrow::datatypes::DataType::Int64 | arrow::datatypes::DataType::Float64 => {
                            // Numeric column after string columns are likely aggregates
                            if idx >= string_columns && !field.name().contains('(') {
                                agg_columns.push((idx, field.name().clone()));
                            }
                        }
                        _ => {}
                    }
                }
                
                // Try to match based on aggregate type position
                // SUM is typically after COUNT if both are present
                if !agg_columns.is_empty() {
                    let target_idx = match upper.as_str() {
                        "COUNT" => 0,  // COUNT is usually first
                        "SUM" => if agg_columns.len() > 1 { 1 } else { 0 },
                        "AVG" => if agg_columns.len() > 2 { 2 } else { agg_columns.len().saturating_sub(1) },
                        "MIN" | "MAX" => agg_columns.len().saturating_sub(1),
                        _ => 0,
                    };
                    let idx = target_idx.min(agg_columns.len().saturating_sub(1));
                    return Ok(batch.column(agg_columns[idx].0).clone());
                }
                
                return Err(io::Error::new(io::ErrorKind::NotFound, format!("Aggregate column '{}' not found in result", col_name)));
            }
            _ => {}
        }
        
        match upper.as_str() {
            "COALESCE" => {
                if args.is_empty() {
                    return Err(io::Error::new(io::ErrorKind::InvalidInput, "COALESCE requires at least one argument"));
                }
                
                let num_rows = batch.num_rows();
                
                // Determine result type from first non-null argument (skip arrays that are all null)
                let mut result_type: Option<&str> = None;
                let mut arrays: Vec<ArrayRef> = Vec::new();
                for arg in args {
                    let arr = Self::evaluate_expr_to_array(batch, arg)?;
                    // Only set type from arrays that have at least one non-null value
                    let has_non_null = (0..arr.len()).any(|i| !arr.is_null(i));
                    if result_type.is_none() && has_non_null {
                        if arr.as_any().downcast_ref::<StringArray>().is_some() {
                            result_type = Some("string");
                        } else if arr.as_any().downcast_ref::<Int64Array>().is_some() {
                            result_type = Some("int");
                        } else if arr.as_any().downcast_ref::<Float64Array>().is_some() {
                            result_type = Some("float");
                        }
                    }
                    arrays.push(arr);
                }
                
                match result_type.unwrap_or("int") {
                    "string" => {
                        let mut result: Vec<Option<String>> = vec![None; num_rows];
                        let mut assigned = vec![false; num_rows];
                        for arr in &arrays {
                            if let Some(str_arr) = arr.as_any().downcast_ref::<StringArray>() {
                                for i in 0..num_rows {
                                    if !assigned[i] && !str_arr.is_null(i) {
                                        result[i] = Some(str_arr.value(i).to_string());
                                        assigned[i] = true;
                                    }
                                }
                            }
                        }
                        Ok(Arc::new(StringArray::from(result.iter().map(|s| s.as_deref()).collect::<Vec<_>>())))
                    }
                    "float" => {
                        let mut result: Vec<Option<f64>> = vec![None; num_rows];
                        let mut assigned = vec![false; num_rows];
                        for arr in &arrays {
                            if let Some(f_arr) = arr.as_any().downcast_ref::<Float64Array>() {
                                for i in 0..num_rows {
                                    if !assigned[i] && !f_arr.is_null(i) {
                                        result[i] = Some(f_arr.value(i));
                                        assigned[i] = true;
                                    }
                                }
                            }
                        }
                        Ok(Arc::new(Float64Array::from(result)))
                    }
                    _ => {
                        let mut result: Vec<Option<i64>> = vec![None; num_rows];
                        let mut assigned = vec![false; num_rows];
                        for arr in &arrays {
                            if let Some(int_arr) = arr.as_any().downcast_ref::<Int64Array>() {
                                for i in 0..num_rows {
                                    if !assigned[i] && !int_arr.is_null(i) {
                                        result[i] = Some(int_arr.value(i));
                                        assigned[i] = true;
                                    }
                                }
                            }
                        }
                        Ok(Arc::new(Int64Array::from(result)))
                    }
                }
            }
            "IFNULL" | "NVL" | "ISNULL" => {
                if args.len() != 2 {
                    return Err(io::Error::new(io::ErrorKind::InvalidInput, format!("{} requires exactly 2 arguments", upper)));
                }
                
                let arr1 = Self::evaluate_expr_to_array(batch, &args[0])?;
                let arr2 = Self::evaluate_expr_to_array(batch, &args[1])?;
                let num_rows = batch.num_rows();
                
                // Check if arr1 is all nulls - use arr2's type
                let arr1_all_null = (0..num_rows).all(|i| arr1.is_null(i));
                
                // Try integer types
                if let Some(int2) = arr2.as_any().downcast_ref::<Int64Array>() {
                    if arr1_all_null || arr1.as_any().downcast_ref::<Int64Array>().is_some() {
                        let int1 = arr1.as_any().downcast_ref::<Int64Array>();
                        let result: Vec<Option<i64>> = (0..num_rows).map(|i| {
                            if let Some(i1) = int1 { if !i1.is_null(i) { return Some(i1.value(i)); } }
                            if !int2.is_null(i) { Some(int2.value(i)) } else { None }
                        }).collect();
                        return Ok(Arc::new(Int64Array::from(result)));
                    }
                }
                // Try string types
                if let Some(str2) = arr2.as_any().downcast_ref::<StringArray>() {
                    if arr1_all_null || arr1.as_any().downcast_ref::<StringArray>().is_some() {
                        let str1 = arr1.as_any().downcast_ref::<StringArray>();
                        let result: Vec<Option<&str>> = (0..num_rows).map(|i| {
                            if let Some(s1) = str1 { if !s1.is_null(i) { return Some(s1.value(i)); } }
                            if !str2.is_null(i) { Some(str2.value(i)) } else { None }
                        }).collect();
                        return Ok(Arc::new(StringArray::from(result)));
                    }
                }
                // Try float types
                if let Some(f2) = arr2.as_any().downcast_ref::<Float64Array>() {
                    if arr1_all_null || arr1.as_any().downcast_ref::<Float64Array>().is_some() {
                        let f1 = arr1.as_any().downcast_ref::<Float64Array>();
                        let result: Vec<Option<f64>> = (0..num_rows).map(|i| {
                            if let Some(ff1) = f1 { if !ff1.is_null(i) { return Some(ff1.value(i)); } }
                            if !f2.is_null(i) { Some(f2.value(i)) } else { None }
                        }).collect();
                        return Ok(Arc::new(Float64Array::from(result)));
                    }
                }
                // Default: return arr2 if arr1 is all null
                if arr1_all_null {
                    return Ok(arr2);
                }
                Err(io::Error::new(io::ErrorKind::InvalidData, "IFNULL/NVL argument types must match"))
            }
            "ABS" => {
                if args.len() != 1 {
                    return Err(io::Error::new(io::ErrorKind::InvalidInput, "ABS requires exactly 1 argument"));
                }
                let arr = Self::evaluate_expr_to_array(batch, &args[0])?;
                if let Some(int_arr) = arr.as_any().downcast_ref::<Int64Array>() {
                    let result: Vec<Option<i64>> = (0..batch.num_rows()).map(|i| {
                        if int_arr.is_null(i) { None } else { Some(int_arr.value(i).abs()) }
                    }).collect();
                    Ok(Arc::new(Int64Array::from(result)))
                } else if let Some(float_arr) = arr.as_any().downcast_ref::<Float64Array>() {
                    let result: Vec<Option<f64>> = (0..batch.num_rows()).map(|i| {
                        if float_arr.is_null(i) { None } else { Some(float_arr.value(i).abs()) }
                    }).collect();
                    Ok(Arc::new(Float64Array::from(result)))
                } else {
                    Err(io::Error::new(io::ErrorKind::InvalidData, "ABS requires numeric argument"))
                }
            }
            "NULLIF" => {
                if args.len() != 2 {
                    return Err(io::Error::new(io::ErrorKind::InvalidInput, "NULLIF requires exactly 2 arguments"));
                }
                let arr1 = Self::evaluate_expr_to_array(batch, &args[0])?;
                let arr2 = Self::evaluate_expr_to_array(batch, &args[1])?;
                let num_rows = batch.num_rows();
                
                if let (Some(int1), Some(int2)) = (
                    arr1.as_any().downcast_ref::<Int64Array>(),
                    arr2.as_any().downcast_ref::<Int64Array>(),
                ) {
                    let result: Vec<Option<i64>> = (0..num_rows).map(|i| {
                        if int1.is_null(i) { None }
                        else if !int2.is_null(i) && int1.value(i) == int2.value(i) { None }
                        else { Some(int1.value(i)) }
                    }).collect();
                    Ok(Arc::new(Int64Array::from(result)))
                } else if let (Some(str1), Some(str2)) = (
                    arr1.as_any().downcast_ref::<StringArray>(),
                    arr2.as_any().downcast_ref::<StringArray>(),
                ) {
                    let result: Vec<Option<&str>> = (0..num_rows).map(|i| {
                        if str1.is_null(i) { None }
                        else if !str2.is_null(i) && str1.value(i) == str2.value(i) { None }
                        else { Some(str1.value(i)) }
                    }).collect();
                    Ok(Arc::new(StringArray::from(result)))
                } else {
                    Err(io::Error::new(io::ErrorKind::InvalidData, "NULLIF type mismatch"))
                }
            }
            "UPPER" | "UCASE" => {
                if args.len() != 1 {
                    return Err(io::Error::new(io::ErrorKind::InvalidInput, "UPPER requires exactly 1 argument"));
                }
                let arr = Self::evaluate_expr_to_array(batch, &args[0])?;
                // Handle all-null input or NULL literal
                if (0..arr.len()).all(|i| arr.is_null(i)) {
                    let result: Vec<Option<&str>> = vec![None; batch.num_rows()];
                    return Ok(Arc::new(StringArray::from(result)));
                }
                if let Some(str_arr) = arr.as_any().downcast_ref::<StringArray>() {
                    let result: Vec<Option<String>> = (0..batch.num_rows()).map(|i| {
                        if str_arr.is_null(i) { None } else { Some(str_arr.value(i).to_uppercase()) }
                    }).collect();
                    Ok(Arc::new(StringArray::from(result.iter().map(|s| s.as_deref()).collect::<Vec<_>>())))
                } else {
                    Err(io::Error::new(io::ErrorKind::InvalidData, "UPPER requires string argument"))
                }
            }
            "LOWER" | "LCASE" => {
                if args.len() != 1 {
                    return Err(io::Error::new(io::ErrorKind::InvalidInput, "LOWER requires exactly 1 argument"));
                }
                let arr = Self::evaluate_expr_to_array(batch, &args[0])?;
                // Handle all-null input or NULL literal
                if (0..arr.len()).all(|i| arr.is_null(i)) {
                    let result: Vec<Option<&str>> = vec![None; batch.num_rows()];
                    return Ok(Arc::new(StringArray::from(result)));
                }
                if let Some(str_arr) = arr.as_any().downcast_ref::<StringArray>() {
                    let result: Vec<Option<String>> = (0..batch.num_rows()).map(|i| {
                        if str_arr.is_null(i) { None } else { Some(str_arr.value(i).to_lowercase()) }
                    }).collect();
                    Ok(Arc::new(StringArray::from(result.iter().map(|s| s.as_deref()).collect::<Vec<_>>())))
                } else {
                    Err(io::Error::new(io::ErrorKind::InvalidData, "LOWER requires string argument"))
                }
            }
            "LENGTH" | "LEN" | "CHAR_LENGTH" | "CHARACTER_LENGTH" => {
                if args.len() != 1 {
                    return Err(io::Error::new(io::ErrorKind::InvalidInput, "LENGTH requires exactly 1 argument"));
                }
                let arr = Self::evaluate_expr_to_array(batch, &args[0])?;
                if let Some(str_arr) = arr.as_any().downcast_ref::<StringArray>() {
                    let result: Vec<Option<i64>> = (0..batch.num_rows()).map(|i| {
                        if str_arr.is_null(i) { None } else { Some(str_arr.value(i).chars().count() as i64) }
                    }).collect();
                    Ok(Arc::new(Int64Array::from(result)))
                } else {
                    Err(io::Error::new(io::ErrorKind::InvalidData, "LENGTH requires string argument"))
                }
            }
            "TRIM" => {
                if args.len() != 1 {
                    return Err(io::Error::new(io::ErrorKind::InvalidInput, "TRIM requires exactly 1 argument"));
                }
                let arr = Self::evaluate_expr_to_array(batch, &args[0])?;
                if let Some(str_arr) = arr.as_any().downcast_ref::<StringArray>() {
                    let result: Vec<Option<String>> = (0..batch.num_rows()).map(|i| {
                        if str_arr.is_null(i) { None } else { Some(str_arr.value(i).trim().to_string()) }
                    }).collect();
                    Ok(Arc::new(StringArray::from(result.iter().map(|s| s.as_deref()).collect::<Vec<_>>())))
                } else {
                    Err(io::Error::new(io::ErrorKind::InvalidData, "TRIM requires string argument"))
                }
            }
            "CONCAT" => {
                if args.is_empty() {
                    return Ok(Arc::new(StringArray::from(vec![""; batch.num_rows()])));
                }
                let num_rows = batch.num_rows();
                let mut result: Vec<String> = vec![String::new(); num_rows];
                for arg in args {
                    let arr = Self::evaluate_expr_to_array(batch, arg)?;
                    if let Some(str_arr) = arr.as_any().downcast_ref::<StringArray>() {
                        for i in 0..num_rows {
                            if !str_arr.is_null(i) {
                                result[i].push_str(str_arr.value(i));
                            }
                        }
                    } else if let Some(int_arr) = arr.as_any().downcast_ref::<Int64Array>() {
                        for i in 0..num_rows {
                            if !int_arr.is_null(i) {
                                result[i].push_str(&int_arr.value(i).to_string());
                            }
                        }
                    }
                }
                Ok(Arc::new(StringArray::from(result.iter().map(|s| s.as_str()).collect::<Vec<_>>())))
            }
            "SUBSTR" | "SUBSTRING" => {
                if args.len() < 2 || args.len() > 3 {
                    return Err(io::Error::new(io::ErrorKind::InvalidInput, "SUBSTR requires 2 or 3 arguments"));
                }
                let str_arr = Self::evaluate_expr_to_array(batch, &args[0])?;
                let start_arr = Self::evaluate_expr_to_array(batch, &args[1])?;
                let len_arr = if args.len() == 3 { Some(Self::evaluate_expr_to_array(batch, &args[2])?) } else { None };
                
                if let (Some(strs), Some(starts)) = (
                    str_arr.as_any().downcast_ref::<StringArray>(),
                    start_arr.as_any().downcast_ref::<Int64Array>(),
                ) {
                    let result: Vec<Option<String>> = (0..batch.num_rows()).map(|i| {
                        if strs.is_null(i) || starts.is_null(i) { return None; }
                        let s = strs.value(i);
                        let start = (starts.value(i).max(1) - 1) as usize;
                        if start >= s.len() { return Some(String::new()); }
                        let len = if let Some(ref larr) = len_arr {
                            if let Some(la) = larr.as_any().downcast_ref::<Int64Array>() {
                                if la.is_null(i) { s.len() } else { la.value(i).max(0) as usize }
                            } else { s.len() }
                        } else { s.len() };
                        Some(s.chars().skip(start).take(len).collect())
                    }).collect();
                    Ok(Arc::new(StringArray::from(result.iter().map(|s| s.as_deref()).collect::<Vec<_>>())))
                } else {
                    Err(io::Error::new(io::ErrorKind::InvalidData, "SUBSTR type mismatch"))
                }
            }
            "REPLACE" => {
                if args.len() != 3 {
                    return Err(io::Error::new(io::ErrorKind::InvalidInput, "REPLACE requires exactly 3 arguments"));
                }
                let str_arr = Self::evaluate_expr_to_array(batch, &args[0])?;
                let from_arr = Self::evaluate_expr_to_array(batch, &args[1])?;
                let to_arr = Self::evaluate_expr_to_array(batch, &args[2])?;
                
                if let (Some(strs), Some(froms), Some(tos)) = (
                    str_arr.as_any().downcast_ref::<StringArray>(),
                    from_arr.as_any().downcast_ref::<StringArray>(),
                    to_arr.as_any().downcast_ref::<StringArray>(),
                ) {
                    let result: Vec<Option<String>> = (0..batch.num_rows()).map(|i| {
                        if strs.is_null(i) { None }
                        else {
                            let from = if froms.is_null(i) { "" } else { froms.value(i) };
                            let to = if tos.is_null(i) { "" } else { tos.value(i) };
                            Some(strs.value(i).replace(from, to))
                        }
                    }).collect();
                    Ok(Arc::new(StringArray::from(result.iter().map(|s| s.as_deref()).collect::<Vec<_>>())))
                } else {
                    Err(io::Error::new(io::ErrorKind::InvalidData, "REPLACE requires string arguments"))
                }
            }
            "ROUND" => {
                if args.is_empty() || args.len() > 2 {
                    return Err(io::Error::new(io::ErrorKind::InvalidInput, "ROUND requires 1 or 2 arguments"));
                }
                let arr = Self::evaluate_expr_to_array(batch, &args[0])?;
                let decimals = if args.len() == 2 {
                    let d = Self::evaluate_expr_to_array(batch, &args[1])?;
                    if let Some(da) = d.as_any().downcast_ref::<Int64Array>() {
                        if da.len() > 0 && !da.is_null(0) { da.value(0) as i32 } else { 0 }
                    } else { 0 }
                } else { 0 };
                
                if let Some(float_arr) = arr.as_any().downcast_ref::<Float64Array>() {
                    let factor = 10f64.powi(decimals);
                    let result: Vec<Option<f64>> = (0..batch.num_rows()).map(|i| {
                        if float_arr.is_null(i) { None } else { Some((float_arr.value(i) * factor).round() / factor) }
                    }).collect();
                    Ok(Arc::new(Float64Array::from(result)))
                } else if let Some(int_arr) = arr.as_any().downcast_ref::<Int64Array>() {
                    Ok(Arc::new(int_arr.clone()))
                } else {
                    Err(io::Error::new(io::ErrorKind::InvalidData, "ROUND requires numeric argument"))
                }
            }
            "FLOOR" => {
                if args.len() != 1 {
                    return Err(io::Error::new(io::ErrorKind::InvalidInput, "FLOOR requires exactly 1 argument"));
                }
                let arr = Self::evaluate_expr_to_array(batch, &args[0])?;
                if let Some(float_arr) = arr.as_any().downcast_ref::<Float64Array>() {
                    let result: Vec<Option<f64>> = (0..batch.num_rows()).map(|i| {
                        if float_arr.is_null(i) { None } else { Some(float_arr.value(i).floor()) }
                    }).collect();
                    Ok(Arc::new(Float64Array::from(result)))
                } else if let Some(int_arr) = arr.as_any().downcast_ref::<Int64Array>() {
                    Ok(Arc::new(int_arr.clone()))
                } else {
                    Err(io::Error::new(io::ErrorKind::InvalidData, "FLOOR requires numeric argument"))
                }
            }
            "CEIL" | "CEILING" => {
                if args.len() != 1 {
                    return Err(io::Error::new(io::ErrorKind::InvalidInput, "CEIL requires exactly 1 argument"));
                }
                let arr = Self::evaluate_expr_to_array(batch, &args[0])?;
                if let Some(float_arr) = arr.as_any().downcast_ref::<Float64Array>() {
                    let result: Vec<Option<f64>> = (0..batch.num_rows()).map(|i| {
                        if float_arr.is_null(i) { None } else { Some(float_arr.value(i).ceil()) }
                    }).collect();
                    Ok(Arc::new(Float64Array::from(result)))
                } else if let Some(int_arr) = arr.as_any().downcast_ref::<Int64Array>() {
                    Ok(Arc::new(int_arr.clone()))
                } else {
                    Err(io::Error::new(io::ErrorKind::InvalidData, "CEIL requires numeric argument"))
                }
            }
            "MOD" => {
                if args.len() != 2 {
                    return Err(io::Error::new(io::ErrorKind::InvalidInput, "MOD requires exactly 2 arguments"));
                }
                let arr1 = Self::evaluate_expr_to_array(batch, &args[0])?;
                let arr2 = Self::evaluate_expr_to_array(batch, &args[1])?;
                if let (Some(int1), Some(int2)) = (
                    arr1.as_any().downcast_ref::<Int64Array>(),
                    arr2.as_any().downcast_ref::<Int64Array>(),
                ) {
                    let result: Vec<Option<i64>> = (0..batch.num_rows()).map(|i| {
                        if int1.is_null(i) || int2.is_null(i) || int2.value(i) == 0 { None }
                        else { Some(int1.value(i) % int2.value(i)) }
                    }).collect();
                    Ok(Arc::new(Int64Array::from(result)))
                } else {
                    Err(io::Error::new(io::ErrorKind::InvalidData, "MOD requires integer arguments"))
                }
            }
            "SQRT" => {
                if args.len() != 1 {
                    return Err(io::Error::new(io::ErrorKind::InvalidInput, "SQRT requires exactly 1 argument"));
                }
                let arr = Self::evaluate_expr_to_array(batch, &args[0])?;
                if let Some(float_arr) = arr.as_any().downcast_ref::<Float64Array>() {
                    let result: Vec<Option<f64>> = (0..batch.num_rows()).map(|i| {
                        if float_arr.is_null(i) || float_arr.value(i) < 0.0 { None } 
                        else { Some(float_arr.value(i).sqrt()) }
                    }).collect();
                    Ok(Arc::new(Float64Array::from(result)))
                } else if let Some(int_arr) = arr.as_any().downcast_ref::<Int64Array>() {
                    let result: Vec<Option<f64>> = (0..batch.num_rows()).map(|i| {
                        if int_arr.is_null(i) || int_arr.value(i) < 0 { None }
                        else { Some((int_arr.value(i) as f64).sqrt()) }
                    }).collect();
                    Ok(Arc::new(Float64Array::from(result)))
                } else {
                    Err(io::Error::new(io::ErrorKind::InvalidData, "SQRT requires numeric argument"))
                }
            }
            "MID" | "SUBSTR" | "SUBSTRING" => {
                if args.len() < 2 || args.len() > 3 {
                    return Err(io::Error::new(io::ErrorKind::InvalidInput, "MID/SUBSTR requires 2 or 3 arguments"));
                }
                let str_arr = Self::evaluate_expr_to_array(batch, &args[0])?;
                let start_arr = Self::evaluate_expr_to_array(batch, &args[1])?;
                let len_arr = if args.len() == 3 {
                    Some(Self::evaluate_expr_to_array(batch, &args[2])?)
                } else { None };
                
                if let (Some(strs), Some(starts)) = (
                    str_arr.as_any().downcast_ref::<StringArray>(),
                    start_arr.as_any().downcast_ref::<Int64Array>(),
                ) {
                    let result: Vec<Option<String>> = (0..batch.num_rows()).map(|i| {
                        if strs.is_null(i) || starts.is_null(i) { return None; }
                        let s = strs.value(i);
                        let start = (starts.value(i).max(1) - 1) as usize; // 1-indexed
                        let len = if let Some(ref la) = len_arr {
                            if let Some(ia) = la.as_any().downcast_ref::<Int64Array>() {
                                if ia.is_null(i) { s.len() } else { ia.value(i).max(0) as usize }
                            } else { s.len() }
                        } else { s.len() };
                        let chars: Vec<char> = s.chars().collect();
                        if start >= chars.len() { Some(String::new()) }
                        else { Some(chars[start..].iter().take(len).collect()) }
                    }).collect();
                    Ok(Arc::new(StringArray::from(result.iter().map(|s| s.as_deref()).collect::<Vec<_>>())))
                } else {
                    Err(io::Error::new(io::ErrorKind::InvalidData, "MID requires string and integer arguments"))
                }
            }
            "NOW" | "CURRENT_TIMESTAMP" => {
                use std::time::{SystemTime, UNIX_EPOCH};
                let secs = SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0);
                // Format as ISO 8601 string
                let dt = chrono::DateTime::from_timestamp(secs as i64, 0).unwrap_or_default();
                let now_str = dt.format("%Y-%m-%d %H:%M:%S").to_string();
                let result = vec![Some(now_str.as_str()); batch.num_rows()];
                Ok(Arc::new(StringArray::from(result)))
            }
            "RAND" | "RANDOM" => {
                use std::time::{SystemTime, UNIX_EPOCH};
                let seed = SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_nanos()).unwrap_or(0) as u64;
                let result: Vec<f64> = (0..batch.num_rows()).map(|i| {
                    // Simple LCG random number generator with better distribution
                    let mut state = seed.wrapping_add((i as u64).wrapping_mul(2685821657736338717));
                    state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                    state ^= state >> 33;
                    state = state.wrapping_mul(0xff51afd7ed558ccd);
                    state ^= state >> 33;
                    (state as f64) / (u64::MAX as f64)
                }).collect();
                Ok(Arc::new(Float64Array::from(result)))
            }
            _ => Err(io::Error::new(
                io::ErrorKind::Unsupported,
                format!("Unsupported function: {}", name),
            )),
        }
    }

    /// Evaluate IN expression with Value list
    fn evaluate_in_values(
        batch: &RecordBatch,
        column: &str,
        values: &[Value],
        negated: bool,
    ) -> io::Result<BooleanArray> {
        let col_name = column.trim_matches('"');
        let target = batch.column_by_name(col_name)
            .ok_or_else(|| io::Error::new(io::ErrorKind::NotFound, format!("Column '{}' not found", col_name)))?;
        let num_rows = batch.num_rows();
        
        // Start with all false
        let mut result = BooleanArray::from(vec![false; num_rows]);
        
        for val in values {
            let val_array = Self::value_to_array(val, num_rows)?;
            let eq_mask = cmp::eq(target, &val_array)
                .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e.to_string()))?;
            result = compute::or(&result, &eq_mask)
                .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e.to_string()))?;
        }
        
        if negated {
            compute::not(&result)
                .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e.to_string()))
        } else {
            Ok(result)
        }
    }

    /// Evaluate LIKE expression
    fn evaluate_like(
        batch: &RecordBatch,
        column: &str,
        pattern: &str,
        negated: bool,
    ) -> io::Result<BooleanArray> {
        let col_name = column.trim_matches('"');
        let array = batch.column_by_name(col_name)
            .ok_or_else(|| io::Error::new(io::ErrorKind::NotFound, format!("Column '{}' not found", col_name)))?;
        
        let string_array = array
            .as_any()
            .downcast_ref::<StringArray>()
            .ok_or_else(|| io::Error::new(
                io::ErrorKind::InvalidData,
                "LIKE requires string column",
            ))?;
        
        // Convert SQL LIKE pattern to regex
        let regex_pattern = Self::like_to_regex(pattern);
        let regex = regex::Regex::new(&regex_pattern)
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidInput, e.to_string()))?;
        
        let result: BooleanArray = string_array
            .iter()
            .map(|opt| opt.map(|s| regex.is_match(s)).unwrap_or(false))
            .collect();
        
        if negated {
            compute::not(&result)
                .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e.to_string()))
        } else {
            Ok(result)
        }
    }

    /// Evaluate REGEXP expression
    fn evaluate_regexp(
        batch: &RecordBatch,
        column: &str,
        pattern: &str,
        negated: bool,
    ) -> io::Result<BooleanArray> {
        let col_name = column.trim_matches('"');
        let array = batch.column_by_name(col_name)
            .ok_or_else(|| io::Error::new(io::ErrorKind::NotFound, format!("Column '{}' not found", col_name)))?;
        
        let string_array = array
            .as_any()
            .downcast_ref::<StringArray>()
            .ok_or_else(|| io::Error::new(
                io::ErrorKind::InvalidData,
                "REGEXP requires string column",
            ))?;
        
        // Use pattern directly as regex (convert glob-style * to regex .*)
        let regex_pattern = pattern.replace("*", ".*");
        let regex = regex::Regex::new(&regex_pattern)
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidInput, e.to_string()))?;
        
        let result: BooleanArray = string_array
            .iter()
            .map(|opt| opt.map(|s| regex.is_match(s)).unwrap_or(false))
            .collect();
        
        if negated {
            compute::not(&result)
                .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e.to_string()))
        } else {
            Ok(result)
        }
    }

    /// Convert SQL LIKE pattern to regex
    fn like_to_regex(pattern: &str) -> String {
        let mut regex = String::from("^");
        let mut chars = pattern.chars().peekable();
        
        while let Some(c) = chars.next() {
            match c {
                '%' => regex.push_str(".*"),
                '_' => regex.push('.'),
                '\\' => {
                    if let Some(&next) = chars.peek() {
                        if next == '%' || next == '_' {
                            regex.push(chars.next().unwrap());
                            continue;
                        }
                    }
                    regex.push_str("\\\\");
                }
                c if "[](){}|^$.*+?\\".contains(c) => {
                    regex.push('\\');
                    regex.push(c);
                }
                c => regex.push(c),
            }
        }
        
        regex.push('$');
        regex
    }

    /// Apply ORDER BY clause
    fn apply_order_by(
        batch: &RecordBatch,
        order_by: &[crate::query::OrderByClause],
    ) -> io::Result<RecordBatch> {
        use arrow::compute::SortColumn;

        let sort_columns: Vec<SortColumn> = order_by
            .iter()
            .filter_map(|clause| {
                let col_name = clause.column.trim_matches('"');
                // Strip table prefix if present (e.g., "u.tier" -> "tier")
                let actual_col = if let Some(dot_pos) = col_name.rfind('.') {
                    &col_name[dot_pos + 1..]
                } else {
                    col_name
                };
                batch.column_by_name(actual_col).map(|col| SortColumn {
                    values: col.clone(),
                    options: Some(SortOptions {
                        descending: clause.descending,
                        nulls_first: clause.nulls_first.unwrap_or(clause.descending),
                    }),
                })
            })
            .collect();

        if sort_columns.is_empty() {
            return Ok(batch.clone());
        }

        let indices = compute::lexsort_to_indices(&sort_columns, None)
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e.to_string()))?;

        let columns: Vec<ArrayRef> = batch
            .columns()
            .iter()
            .map(|col| compute::take(col, &indices, None))
            .collect::<Result<Vec<_>, _>>()
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e.to_string()))?;

        RecordBatch::try_new(batch.schema(), columns)
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e.to_string()))
    }

    /// Apply LIMIT and OFFSET
    fn apply_limit_offset(
        batch: &RecordBatch,
        limit: Option<usize>,
        offset: Option<usize>,
    ) -> io::Result<RecordBatch> {
        let start = offset.unwrap_or(0);
        let end = limit
            .map(|l| (start + l).min(batch.num_rows()))
            .unwrap_or(batch.num_rows());

        if start >= batch.num_rows() {
            return Ok(RecordBatch::new_empty(batch.schema()));
        }

        let length = end - start;
        Ok(batch.slice(start, length))
    }

    /// Project columns according to SELECT list
    fn apply_projection(
        batch: &RecordBatch,
        columns: &[SelectColumn],
    ) -> io::Result<RecordBatch> {
        Self::apply_projection_with_storage(batch, columns, None)
    }
    
    /// Project columns according to SELECT list (with optional storage path for scalar subqueries)
    fn apply_projection_with_storage(
        batch: &RecordBatch,
        columns: &[SelectColumn],
        storage_path: Option<&Path>,
    ) -> io::Result<RecordBatch> {
        // Handle simple SELECT * (only * with no other columns)
        if columns.len() == 1 && matches!(columns[0], SelectColumn::All) {
            return Ok(batch.clone());
        }

        let mut fields: Vec<Field> = Vec::new();
        let mut arrays: Vec<ArrayRef> = Vec::new();
        let mut added_columns: std::collections::HashSet<String> = std::collections::HashSet::new();

        // Collect all explicitly named columns to exclude from * expansion
        let mut explicit_cols: std::collections::HashSet<String> = std::collections::HashSet::new();
        for col in columns {
            match col {
                SelectColumn::Column(name) => {
                    let col_name = name.trim_matches('"');
                    let actual_col = if let Some(dot_pos) = col_name.rfind('.') {
                        &col_name[dot_pos + 1..]
                    } else {
                        col_name
                    };
                    explicit_cols.insert(actual_col.to_string());
                }
                SelectColumn::ColumnAlias { column, .. } => {
                    let col_name = column.trim_matches('"');
                    explicit_cols.insert(col_name.to_string());
                }
                _ => {}
            }
        }

        // Process columns in order - preserving user-specified order
        for col in columns {
            match col {
                SelectColumn::Column(name) => {
                    let col_name = name.trim_matches('"');
                    let actual_col = if let Some(dot_pos) = col_name.rfind('.') {
                        &col_name[dot_pos + 1..]
                    } else {
                        col_name
                    };
                    if !added_columns.contains(actual_col) {
                        if let Some(array) = batch.column_by_name(actual_col) {
                            fields.push(Field::new(actual_col, array.data_type().clone(), true));
                            arrays.push(array.clone());
                            added_columns.insert(actual_col.to_string());
                        }
                    }
                }
                SelectColumn::ColumnAlias { column, alias } => {
                    let col_name = column.trim_matches('"');
                    if let Some(array) = batch.column_by_name(col_name) {
                        fields.push(Field::new(alias, array.data_type().clone(), true));
                        arrays.push(array.clone());
                        added_columns.insert(alias.clone());
                    }
                }
                SelectColumn::Expression { expr, alias } => {
                    // Use storage-aware evaluation for expressions that may contain scalar subqueries
                    let array = if let Some(path) = storage_path {
                        Self::evaluate_expr_to_array_with_storage(batch, expr, path)?
                    } else {
                        Self::evaluate_expr_to_array(batch, expr)?
                    };
                    let output_name = alias.clone().unwrap_or_else(|| "expr".to_string());
                    fields.push(Field::new(&output_name, array.data_type().clone(), true));
                    arrays.push(array);
                }
                SelectColumn::All => {
                    // Add all columns from batch EXCEPT:
                    // 1. _id (always skip here - it will be added at its explicit position if requested)
                    // 2. Columns already added (from explicit references before *)
                    for (i, field) in batch.schema().fields().iter().enumerate() {
                        let col_name = field.name();
                        // Always skip _id in * expansion - it should only appear at its explicit position
                        if col_name == "_id" {
                            continue;
                        }
                        if !added_columns.contains(col_name) {
                            fields.push(field.as_ref().clone());
                            arrays.push(batch.column(i).clone());
                            added_columns.insert(col_name.clone());
                        }
                    }
                }
                SelectColumn::Aggregate { .. } => {
                    // Aggregates are handled separately
                }
                SelectColumn::WindowFunction { .. } => {
                    // Window functions not yet supported
                }
            }
        }

        let schema = Arc::new(Schema::new(fields));
        RecordBatch::try_new(schema, arrays)
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e.to_string()))
    }

    /// Execute aggregation query
    fn execute_aggregation(
        batch: &RecordBatch,
        stmt: &SelectStatement,
    ) -> io::Result<V3Result> {
        let mut fields: Vec<Field> = Vec::new();
        let mut arrays: Vec<ArrayRef> = Vec::new();

        for col in &stmt.columns {
            if let SelectColumn::Aggregate { func, column, distinct, alias } = col {
                let (field, array) = Self::compute_aggregate(batch, func, column, *distinct, alias)?;
                fields.push(field);
                arrays.push(array);
            }
        }

        if fields.is_empty() {
            return Ok(V3Result::Scalar(batch.num_rows() as i64));
        }

        let schema = Arc::new(Schema::new(fields));
        let result = RecordBatch::try_new(schema, arrays)
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e.to_string()))?;

        Ok(V3Result::Data(result))
    }

    /// Compute a single aggregate function
    fn compute_aggregate(
        batch: &RecordBatch,
        func: &crate::query::AggregateFunc,
        column: &Option<String>,
        distinct: bool,
        alias: &Option<String>,
    ) -> io::Result<(Field, ArrayRef)> {
        use crate::query::AggregateFunc;
        use std::collections::HashSet;
        
        let func_name = match func {
            AggregateFunc::Count => "COUNT",
            AggregateFunc::Sum => "SUM",
            AggregateFunc::Avg => "AVG",
            AggregateFunc::Min => "MIN",
            AggregateFunc::Max => "MAX",
        };
        
        let output_name = alias.clone().unwrap_or_else(|| {
            if let Some(col) = column {
                format!("{}({})", func_name, col)
            } else {
                format!("{}(*)", func_name)
            }
        });

        match func {
            AggregateFunc::Count => {
                let count = if let Some(col_name) = column {
                    // Treat "*" and numeric constants (like "1") as COUNT(*) - count all rows
                    if col_name == "*" || col_name.chars().next().map(|c| c.is_ascii_digit()).unwrap_or(false) {
                        batch.num_rows() as i64
                    } else if let Some(array) = batch.column_by_name(col_name) {
                        if distinct {
                            // COUNT(DISTINCT column) - count unique non-null values
                            if let Some(int_arr) = array.as_any().downcast_ref::<Int64Array>() {
                                let unique: HashSet<i64> = int_arr.iter().filter_map(|v| v).collect();
                                unique.len() as i64
                            } else if let Some(str_arr) = array.as_any().downcast_ref::<StringArray>() {
                                let unique: HashSet<&str> = (0..str_arr.len())
                                    .filter(|&i| !str_arr.is_null(i))
                                    .map(|i| str_arr.value(i))
                                    .collect();
                                unique.len() as i64
                            } else if let Some(float_arr) = array.as_any().downcast_ref::<Float64Array>() {
                                let unique: HashSet<u64> = float_arr.iter()
                                    .filter_map(|v| v.map(|f| f.to_bits()))
                                    .collect();
                                unique.len() as i64
                            } else {
                                (array.len() - array.null_count()) as i64
                            }
                        } else {
                            (array.len() - array.null_count()) as i64
                        }
                    } else {
                        0
                    }
                } else {
                    batch.num_rows() as i64
                };
                Ok((
                    Field::new(&output_name, ArrowDataType::Int64, false),
                    Arc::new(Int64Array::from(vec![count])),
                ))
            }
            AggregateFunc::Sum => {
                let col_name = column.as_ref()
                    .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "SUM requires column"))?;
                let array = batch.column_by_name(col_name)
                    .ok_or_else(|| io::Error::new(io::ErrorKind::NotFound, format!("Column '{}' not found", col_name)))?;

                if let Some(int_array) = array.as_any().downcast_ref::<Int64Array>() {
                    let sum: i64 = int_array.iter().filter_map(|v| v).sum();
                    Ok((
                        Field::new(&output_name, ArrowDataType::Int64, false),
                        Arc::new(Int64Array::from(vec![sum])),
                    ))
                } else if let Some(uint_array) = array.as_any().downcast_ref::<UInt64Array>() {
                    let sum: i64 = uint_array.iter().filter_map(|v| v).map(|v| v as i64).sum();
                    Ok((
                        Field::new(&output_name, ArrowDataType::Int64, false),
                        Arc::new(Int64Array::from(vec![sum])),
                    ))
                } else if let Some(float_array) = array.as_any().downcast_ref::<Float64Array>() {
                    let sum: f64 = float_array.iter().filter_map(|v| v).sum();
                    Ok((
                        Field::new(&output_name, ArrowDataType::Float64, false),
                        Arc::new(Float64Array::from(vec![sum])),
                    ))
                } else {
                    Err(io::Error::new(io::ErrorKind::InvalidData, "SUM requires numeric column"))
                }
            }
            AggregateFunc::Avg => {
                let col_name = column.as_ref()
                    .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "AVG requires column"))?;
                let array = batch.column_by_name(col_name)
                    .ok_or_else(|| io::Error::new(io::ErrorKind::NotFound, format!("Column '{}' not found", col_name)))?;

                if let Some(int_array) = array.as_any().downcast_ref::<Int64Array>() {
                    let values: Vec<i64> = int_array.iter().filter_map(|v| v).collect();
                    let avg = if values.is_empty() { 0.0 } else { values.iter().sum::<i64>() as f64 / values.len() as f64 };
                    Ok((
                        Field::new(&output_name, ArrowDataType::Float64, false),
                        Arc::new(Float64Array::from(vec![avg])),
                    ))
                } else if let Some(uint_array) = array.as_any().downcast_ref::<UInt64Array>() {
                    let values: Vec<u64> = uint_array.iter().filter_map(|v| v).collect();
                    let avg = if values.is_empty() { 0.0 } else { values.iter().sum::<u64>() as f64 / values.len() as f64 };
                    Ok((
                        Field::new(&output_name, ArrowDataType::Float64, false),
                        Arc::new(Float64Array::from(vec![avg])),
                    ))
                } else if let Some(float_array) = array.as_any().downcast_ref::<Float64Array>() {
                    let values: Vec<f64> = float_array.iter().filter_map(|v| v).collect();
                    let avg = if values.is_empty() { 0.0 } else { values.iter().sum::<f64>() / values.len() as f64 };
                    Ok((
                        Field::new(&output_name, ArrowDataType::Float64, false),
                        Arc::new(Float64Array::from(vec![avg])),
                    ))
                } else {
                    Err(io::Error::new(io::ErrorKind::InvalidData, "AVG requires numeric column"))
                }
            }
            AggregateFunc::Min => {
                let col_name = column.as_ref()
                    .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "MIN requires column"))?;
                let array = batch.column_by_name(col_name)
                    .ok_or_else(|| io::Error::new(io::ErrorKind::NotFound, format!("Column '{}' not found", col_name)))?;

                if let Some(int_array) = array.as_any().downcast_ref::<Int64Array>() {
                    let min = compute::min(int_array);
                    Ok((
                        Field::new(&output_name, ArrowDataType::Int64, true),
                        Arc::new(Int64Array::from(vec![min])),
                    ))
                } else if let Some(uint_array) = array.as_any().downcast_ref::<UInt64Array>() {
                    let min = compute::min(uint_array).map(|v| v as i64);
                    Ok((
                        Field::new(&output_name, ArrowDataType::Int64, true),
                        Arc::new(Int64Array::from(vec![min])),
                    ))
                } else if let Some(float_array) = array.as_any().downcast_ref::<Float64Array>() {
                    let min = compute::min(float_array);
                    Ok((
                        Field::new(&output_name, ArrowDataType::Float64, true),
                        Arc::new(Float64Array::from(vec![min])),
                    ))
                } else {
                    Err(io::Error::new(io::ErrorKind::InvalidData, "MIN requires numeric column"))
                }
            }
            AggregateFunc::Max => {
                let col_name = column.as_ref()
                    .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "MAX requires column"))?;
                let array = batch.column_by_name(col_name)
                    .ok_or_else(|| io::Error::new(io::ErrorKind::NotFound, format!("Column '{}' not found", col_name)))?;

                if let Some(int_array) = array.as_any().downcast_ref::<Int64Array>() {
                    let max = compute::max(int_array);
                    Ok((
                        Field::new(&output_name, ArrowDataType::Int64, true),
                        Arc::new(Int64Array::from(vec![max])),
                    ))
                } else if let Some(uint_array) = array.as_any().downcast_ref::<UInt64Array>() {
                    let max = compute::max(uint_array).map(|v| v as i64);
                    Ok((
                        Field::new(&output_name, ArrowDataType::Int64, true),
                        Arc::new(Int64Array::from(vec![max])),
                    ))
                } else if let Some(float_array) = array.as_any().downcast_ref::<Float64Array>() {
                    let max = compute::max(float_array);
                    Ok((
                        Field::new(&output_name, ArrowDataType::Float64, true),
                        Arc::new(Float64Array::from(vec![max])),
                    ))
                } else {
                    Err(io::Error::new(io::ErrorKind::InvalidData, "MAX requires numeric column"))
                }
            }
        }
    }

    /// Execute GROUP BY aggregation query
    fn execute_group_by(batch: &RecordBatch, stmt: &SelectStatement) -> io::Result<V3Result> {
        use std::collections::HashMap;
        
        if stmt.group_by.is_empty() {
            return Err(io::Error::new(io::ErrorKind::InvalidInput, "GROUP BY requires at least one column"));
        }

        // Build group keys - strip table prefix if present (e.g., "u.tier" -> "tier")
        let group_cols: Vec<String> = stmt.group_by.iter().map(|s| {
            let trimmed = s.trim_matches('"');
            if let Some(dot_pos) = trimmed.rfind('.') {
                trimmed[dot_pos + 1..].to_string()
            } else {
                trimmed.to_string()
            }
        }).collect();
        
        // Create groups: key -> row indices
        let mut groups: HashMap<u64, Vec<usize>> = HashMap::new();
        
        for row_idx in 0..batch.num_rows() {
            let mut hasher = std::collections::hash_map::DefaultHasher::new();
            for col_name in &group_cols {
                if let Some(col) = batch.column_by_name(col_name) {
                    use std::hash::Hasher;
                    hasher.write_u64(Self::hash_array_value(col, row_idx));
                }
            }
            use std::hash::Hasher;
            let key = hasher.finish();
            groups.entry(key).or_insert_with(Vec::new).push(row_idx);
        }

        // Build result arrays
        let mut result_fields: Vec<Field> = Vec::new();
        let mut result_arrays: Vec<ArrayRef> = Vec::new();
        
        let num_groups = groups.len();
        let group_indices: Vec<Vec<usize>> = groups.into_values().collect();

        for col in &stmt.columns {
            match col {
                SelectColumn::Column(name) => {
                    let col_name = name.trim_matches('"');
                    // Strip table prefix if present (e.g., "u.tier" -> "tier")
                    let actual_col = if let Some(dot_pos) = col_name.rfind('.') {
                        &col_name[dot_pos + 1..]
                    } else {
                        col_name
                    };
                    if let Some(src_col) = batch.column_by_name(actual_col) {
                        let (field, array) = Self::take_first_from_groups(src_col, &group_indices, actual_col)?;
                        result_fields.push(field);
                        result_arrays.push(array);
                    }
                }
                SelectColumn::ColumnAlias { column, alias } => {
                    let col_name = column.trim_matches('"');
                    // Strip table prefix if present
                    let actual_col = if let Some(dot_pos) = col_name.rfind('.') {
                        &col_name[dot_pos + 1..]
                    } else {
                        col_name
                    };
                    if let Some(src_col) = batch.column_by_name(actual_col) {
                        let (field, array) = Self::take_first_from_groups(src_col, &group_indices, alias)?;
                        result_fields.push(field);
                        result_arrays.push(array);
                    }
                }
                SelectColumn::Aggregate { func, column, distinct, alias } => {
                    let (field, array) = Self::compute_aggregate_for_groups(batch, func, column, alias, &group_indices, *distinct)?;
                    result_fields.push(field);
                    result_arrays.push(array);
                }
                SelectColumn::Expression { expr, alias } => {
                    // For expressions containing aggregates (like CASE WHEN SUM(x) > 100 THEN ...),
                    // we need to evaluate the expression for each group
                    let (field, array) = Self::evaluate_expr_for_groups(batch, expr, alias.as_deref(), &group_indices)?;
                    result_fields.push(field);
                    result_arrays.push(array);
                }
                _ => {}
            }
        }

        if result_fields.is_empty() {
            return Ok(V3Result::Scalar(num_groups as i64));
        }

        let schema = Arc::new(Schema::new(result_fields));
        let mut result = RecordBatch::try_new(schema, result_arrays)
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e.to_string()))?;

        // Apply HAVING clause if present
        if let Some(having_expr) = &stmt.having {
            let mask = Self::evaluate_predicate(&result, having_expr)?;
            result = compute::filter_record_batch(&result, &mask)
                .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e.to_string()))?;
        }

        // Apply ORDER BY if present
        if !stmt.order_by.is_empty() {
            result = Self::apply_order_by(&result, &stmt.order_by)?;
        }

        Ok(V3Result::Data(result))
    }

    /// Evaluate expression for groups (handles CASE with aggregates)
    fn evaluate_expr_for_groups(
        batch: &RecordBatch,
        expr: &SqlExpr,
        alias: Option<&str>,
        group_indices: &[Vec<usize>],
    ) -> io::Result<(Field, ArrayRef)> {
        let output_name = alias.unwrap_or("expr");
        let num_groups = group_indices.len();
        
        // For CASE expressions with aggregates, we need to:
        // 1. For each group, compute the aggregate values
        // 2. Evaluate the CASE condition using those aggregates
        // 3. Return the CASE result for each group
        
        match expr {
            SqlExpr::Case { when_then, else_expr } => {
                // Evaluate CASE for each group
                let mut string_results: Vec<Option<String>> = Vec::with_capacity(num_groups);
                let mut int_results: Vec<Option<i64>> = Vec::with_capacity(num_groups);
                let mut is_string = false;
                
                // Determine result type from first THEN expression
                if let Some((_, then_expr)) = when_then.first() {
                    if matches!(then_expr, SqlExpr::Literal(Value::String(_))) {
                        is_string = true;
                    }
                }
                
                for indices in group_indices {
                    // Create a sub-batch for this group
                    let group_batch = Self::create_group_batch(batch, indices)?;
                    
                    // For each WHEN clause, check if condition is true for this group
                    let mut matched = false;
                    for (cond_expr, then_expr) in when_then {
                        // Evaluate condition - if it contains aggregates, compute them
                        let cond_result = Self::evaluate_aggregate_condition(&group_batch, cond_expr)?;
                        
                        if cond_result {
                            // Evaluate THEN expression
                            if is_string {
                                if let SqlExpr::Literal(Value::String(s)) = then_expr {
                                    string_results.push(Some(s.clone()));
                                } else {
                                    string_results.push(None);
                                }
                            } else {
                                let then_arr = Self::evaluate_expr_to_array(&group_batch, then_expr)?;
                                if let Some(int_arr) = then_arr.as_any().downcast_ref::<Int64Array>() {
                                    int_results.push(if int_arr.len() > 0 && !int_arr.is_null(0) { Some(int_arr.value(0)) } else { None });
                                } else {
                                    int_results.push(None);
                                }
                            }
                            matched = true;
                            break;
                        }
                    }
                    
                    if !matched {
                        // Use ELSE value
                        if let Some(else_e) = else_expr {
                            if is_string {
                                if let SqlExpr::Literal(Value::String(s)) = else_e.as_ref() {
                                    string_results.push(Some(s.clone()));
                                } else {
                                    string_results.push(None);
                                }
                            } else {
                                let else_arr = Self::evaluate_expr_to_array(&group_batch, else_e)?;
                                if let Some(int_arr) = else_arr.as_any().downcast_ref::<Int64Array>() {
                                    int_results.push(if int_arr.len() > 0 && !int_arr.is_null(0) { Some(int_arr.value(0)) } else { None });
                                } else {
                                    int_results.push(None);
                                }
                            }
                        } else {
                            if is_string {
                                string_results.push(None);
                            } else {
                                int_results.push(None);
                            }
                        }
                    }
                }
                
                if is_string {
                    let array = Arc::new(StringArray::from(string_results)) as ArrayRef;
                    Ok((Field::new(output_name, arrow::datatypes::DataType::Utf8, true), array))
                } else {
                    let array = Arc::new(Int64Array::from(int_results)) as ArrayRef;
                    Ok((Field::new(output_name, arrow::datatypes::DataType::Int64, true), array))
                }
            }
            _ => {
                // For non-CASE expressions, evaluate normally on first row of each group
                let first_indices: Vec<usize> = group_indices.iter().map(|g| g[0]).collect();
                let array = Self::evaluate_expr_to_array(batch, expr)?;
                
                // Take values at first indices
                if let Some(str_arr) = array.as_any().downcast_ref::<StringArray>() {
                    let values: Vec<Option<String>> = first_indices.iter().map(|&i| {
                        if str_arr.is_null(i) { None } else { Some(str_arr.value(i).to_string()) }
                    }).collect();
                    let result = Arc::new(StringArray::from(values)) as ArrayRef;
                    Ok((Field::new(output_name, arrow::datatypes::DataType::Utf8, true), result))
                } else if let Some(int_arr) = array.as_any().downcast_ref::<Int64Array>() {
                    let values: Vec<Option<i64>> = first_indices.iter().map(|&i| {
                        if int_arr.is_null(i) { None } else { Some(int_arr.value(i)) }
                    }).collect();
                    let result = Arc::new(Int64Array::from(values)) as ArrayRef;
                    Ok((Field::new(output_name, arrow::datatypes::DataType::Int64, true), result))
                } else {
                    Err(io::Error::new(io::ErrorKind::InvalidData, "Unsupported expression type"))
                }
            }
        }
    }

    /// Create a sub-batch containing only the specified row indices
    fn create_group_batch(batch: &RecordBatch, indices: &[usize]) -> io::Result<RecordBatch> {
        let indices_array = arrow::array::UInt64Array::from(indices.iter().map(|&i| i as u64).collect::<Vec<_>>());
        compute::take_record_batch(batch, &indices_array)
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e.to_string()))
    }

    /// Evaluate condition that may contain aggregate functions
    fn evaluate_aggregate_condition(batch: &RecordBatch, expr: &SqlExpr) -> io::Result<bool> {
        match expr {
            SqlExpr::BinaryOp { left, op, right } => {
                // Check if this is a comparison operation
                match op {
                    BinaryOperator::Ge | BinaryOperator::Gt | BinaryOperator::Le | 
                    BinaryOperator::Lt | BinaryOperator::Eq | BinaryOperator::NotEq => {
                        // Evaluate left and right, handling aggregates
                        let left_val = Self::evaluate_aggregate_expr_scalar(batch, left)?;
                        let right_val = Self::evaluate_aggregate_expr_scalar(batch, right)?;
                        
                        match op {
                            BinaryOperator::Ge => Ok(left_val >= right_val),
                            BinaryOperator::Gt => Ok(left_val > right_val),
                            BinaryOperator::Le => Ok(left_val <= right_val),
                            BinaryOperator::Lt => Ok(left_val < right_val),
                            BinaryOperator::Eq => Ok((left_val - right_val).abs() < f64::EPSILON),
                            BinaryOperator::NotEq => Ok((left_val - right_val).abs() >= f64::EPSILON),
                            _ => unreachable!(),
                        }
                    }
                    _ => {
                        // For logical operators, evaluate as predicate
                        let result = Self::evaluate_predicate(batch, expr)?;
                        Ok(result.len() > 0 && result.value(0))
                    }
                }
            }
            _ => {
                // For other expressions, try to evaluate as predicate
                let result = Self::evaluate_predicate(batch, expr)?;
                Ok(result.len() > 0 && result.value(0))
            }
        }
    }

    /// Evaluate expression that may be an aggregate, returning scalar value
    fn evaluate_aggregate_expr_scalar(batch: &RecordBatch, expr: &SqlExpr) -> io::Result<f64> {
        match expr {
            SqlExpr::Function { name, args } => {
                // Check if this is an aggregate function
                let func_upper = name.to_uppercase();
                match func_upper.as_str() {
                    "SUM" | "COUNT" | "AVG" | "MIN" | "MAX" => {
                        let func = match func_upper.as_str() {
                            "SUM" => AggregateFunc::Sum,
                            "COUNT" => AggregateFunc::Count,
                            "AVG" => AggregateFunc::Avg,
                            "MIN" => AggregateFunc::Min,
                            "MAX" => AggregateFunc::Max,
                            _ => unreachable!(),
                        };
                        let col_name = if args.is_empty() {
                            "*"
                        } else if let SqlExpr::Column(c) = &args[0] {
                            c.as_str()
                        } else {
                            "*"
                        };
                        // Create group indices covering all rows in the batch
                        let all_indices: Vec<usize> = (0..batch.num_rows()).collect();
                        let (_, result_arr) = Self::compute_aggregate_for_groups(batch, &func, &Some(col_name.to_string()), &None, &[all_indices], false)?;
                        if let Some(int_arr) = result_arr.as_any().downcast_ref::<Int64Array>() {
                            Ok(if int_arr.len() > 0 && !int_arr.is_null(0) { int_arr.value(0) as f64 } else { 0.0 })
                        } else if let Some(float_arr) = result_arr.as_any().downcast_ref::<Float64Array>() {
                            Ok(if float_arr.len() > 0 && !float_arr.is_null(0) { float_arr.value(0) } else { 0.0 })
                        } else {
                            Ok(0.0)
                        }
                    }
                    _ => {
                        let arr = Self::evaluate_expr_to_array(batch, expr)?;
                        Self::extract_scalar_from_array(&arr)
                    }
                }
            }
            SqlExpr::Literal(Value::Int64(i)) => Ok(*i as f64),
            SqlExpr::Literal(Value::Float64(f)) => Ok(*f),
            _ => {
                let arr = Self::evaluate_expr_to_array(batch, expr)?;
                Self::extract_scalar_from_array(&arr)
            }
        }
    }

    /// Extract scalar value from array
    fn extract_scalar_from_array(arr: &ArrayRef) -> io::Result<f64> {
        if let Some(int_arr) = arr.as_any().downcast_ref::<Int64Array>() {
            Ok(if int_arr.len() > 0 && !int_arr.is_null(0) { int_arr.value(0) as f64 } else { 0.0 })
        } else if let Some(float_arr) = arr.as_any().downcast_ref::<Float64Array>() {
            Ok(if float_arr.len() > 0 && !float_arr.is_null(0) { float_arr.value(0) } else { 0.0 })
        } else {
            Ok(0.0)
        }
    }

    /// Check if an expression contains a correlated subquery
    fn has_correlated_subquery(expr: &SqlExpr) -> bool {
        match expr {
            SqlExpr::ExistsSubquery { .. } | SqlExpr::InSubquery { .. } | SqlExpr::ScalarSubquery { .. } => true,
            SqlExpr::BinaryOp { left, right, .. } => {
                Self::has_correlated_subquery(left) || Self::has_correlated_subquery(right)
            }
            SqlExpr::UnaryOp { expr, .. } => Self::has_correlated_subquery(expr),
            SqlExpr::Paren(inner) => Self::has_correlated_subquery(inner),
            _ => false,
        }
    }

    fn coerce_numeric_for_comparison(left: ArrayRef, right: ArrayRef) -> io::Result<(ArrayRef, ArrayRef)> {
        use arrow::compute::cast;
        use arrow::datatypes::DataType;

        let l_is_f = left.as_any().downcast_ref::<Float64Array>().is_some();
        let r_is_f = right.as_any().downcast_ref::<Float64Array>().is_some();
        let l_is_i = left.as_any().downcast_ref::<Int64Array>().is_some();
        let r_is_i = right.as_any().downcast_ref::<Int64Array>().is_some();

        if (l_is_f && r_is_i) {
            let r2 = cast(&right, &DataType::Float64)
                .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e.to_string()))?;
            return Ok((left, r2));
        }
        if (l_is_i && r_is_f) {
            let l2 = cast(&left, &DataType::Float64)
                .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e.to_string()))?;
            return Ok((l2, right));
        }

        Ok((left, right))
    }

    /// Collect column names from an expression (for determining required columns)
    fn collect_columns_from_expr(expr: &SqlExpr, columns: &mut Vec<String>) {
        match expr {
            SqlExpr::Column(name) => {
                let col_name = name.trim_matches('"');
                // Strip table alias prefix if present
                let actual_col = if let Some(dot_pos) = col_name.rfind('.') {
                    &col_name[dot_pos + 1..]
                } else {
                    col_name
                };
                if !columns.contains(&actual_col.to_string()) {
                    columns.push(actual_col.to_string());
                }
            }
            SqlExpr::BinaryOp { left, right, .. } => {
                Self::collect_columns_from_expr(left, columns);
                Self::collect_columns_from_expr(right, columns);
            }
            SqlExpr::UnaryOp { expr, .. } => {
                Self::collect_columns_from_expr(expr, columns);
            }
            SqlExpr::Paren(inner) => {
                Self::collect_columns_from_expr(inner, columns);
            }
            SqlExpr::Function { args, .. } => {
                for arg in args {
                    Self::collect_columns_from_expr(arg, columns);
                }
            }
            SqlExpr::Case { when_then, else_expr } => {
                for (cond, then_expr) in when_then {
                    Self::collect_columns_from_expr(cond, columns);
                    Self::collect_columns_from_expr(then_expr, columns);
                }
                if let Some(else_e) = else_expr {
                    Self::collect_columns_from_expr(else_e, columns);
                }
            }
            SqlExpr::Cast { expr, .. } => {
                Self::collect_columns_from_expr(expr, columns);
            }
            SqlExpr::ExistsSubquery { stmt } | SqlExpr::ScalarSubquery { stmt } => {
                // For correlated subqueries, we need all outer columns that might be referenced
                // The subquery might reference outer columns like u.user_id
                if let Some(ref where_clause) = stmt.where_clause {
                    Self::collect_columns_from_expr(where_clause, columns);
                }
            }
            SqlExpr::InSubquery { column, stmt, .. } => {
                let col_name = column.trim_matches('"');
                let actual_col = if let Some(dot_pos) = col_name.rfind('.') {
                    &col_name[dot_pos + 1..]
                } else {
                    col_name
                };
                if !columns.contains(&actual_col.to_string()) {
                    columns.push(actual_col.to_string());
                }
                if let Some(ref where_clause) = stmt.where_clause {
                    Self::collect_columns_from_expr(where_clause, columns);
                }
            }
            _ => {}
        }
    }

    /// Check if an expression contains an aggregate function (SUM, COUNT, AVG, MIN, MAX)
    fn expr_contains_aggregate(expr: &SqlExpr) -> bool {
        match expr {
            SqlExpr::Function { name, args } => {
                let func_upper = name.to_uppercase();
                if matches!(func_upper.as_str(), "SUM" | "COUNT" | "AVG" | "MIN" | "MAX") {
                    return true;
                }
                // Check arguments recursively
                args.iter().any(Self::expr_contains_aggregate)
            }
            SqlExpr::Case { when_then, else_expr } => {
                // Check all WHEN conditions and THEN expressions
                for (cond, then_expr) in when_then {
                    if Self::expr_contains_aggregate(cond) || Self::expr_contains_aggregate(then_expr) {
                        return true;
                    }
                }
                // Check ELSE expression
                if let Some(else_e) = else_expr {
                    if Self::expr_contains_aggregate(else_e) {
                        return true;
                    }
                }
                false
            }
            SqlExpr::BinaryOp { left, right, .. } => {
                Self::expr_contains_aggregate(left) || Self::expr_contains_aggregate(right)
            }
            SqlExpr::UnaryOp { expr, .. } => Self::expr_contains_aggregate(expr),
            SqlExpr::Paren(inner) => Self::expr_contains_aggregate(inner),
            SqlExpr::Cast { expr, .. } => Self::expr_contains_aggregate(expr),
            _ => false,
        }
    }
    
    /// Check if an expression contains a scalar subquery
    fn expr_contains_scalar_subquery(expr: &SqlExpr) -> bool {
        match expr {
            SqlExpr::ScalarSubquery { .. } => true,
            SqlExpr::BinaryOp { left, right, .. } => {
                Self::expr_contains_scalar_subquery(left) || Self::expr_contains_scalar_subquery(right)
            }
            SqlExpr::UnaryOp { expr, .. } => Self::expr_contains_scalar_subquery(expr),
            SqlExpr::Paren(inner) => Self::expr_contains_scalar_subquery(inner),
            SqlExpr::Cast { expr, .. } => Self::expr_contains_scalar_subquery(expr),
            SqlExpr::Case { when_then, else_expr } => {
                for (cond, then_expr) in when_then {
                    if Self::expr_contains_scalar_subquery(cond) || Self::expr_contains_scalar_subquery(then_expr) {
                        return true;
                    }
                }
                if let Some(else_e) = else_expr {
                    if Self::expr_contains_scalar_subquery(else_e) {
                        return true;
                    }
                }
                false
            }
            _ => false,
        }
    }

    /// Take first value from each group
    fn take_first_from_groups(array: &ArrayRef, group_indices: &[Vec<usize>], output_name: &str) -> io::Result<(Field, ArrayRef)> {
        use arrow::datatypes::DataType;
        
        let first_indices: Vec<usize> = group_indices.iter().map(|g| g[0]).collect();
        
        match array.data_type() {
            DataType::Int64 => {
                let src = array.as_any().downcast_ref::<Int64Array>().unwrap();
                let values: Vec<Option<i64>> = first_indices.iter().map(|&i| {
                    if src.is_null(i) { None } else { Some(src.value(i)) }
                }).collect();
                Ok((Field::new(output_name, DataType::Int64, true), Arc::new(Int64Array::from(values))))
            }
            DataType::Float64 => {
                let src = array.as_any().downcast_ref::<Float64Array>().unwrap();
                let values: Vec<Option<f64>> = first_indices.iter().map(|&i| {
                    if src.is_null(i) { None } else { Some(src.value(i)) }
                }).collect();
                Ok((Field::new(output_name, DataType::Float64, true), Arc::new(Float64Array::from(values))))
            }
            DataType::Utf8 => {
                let src = array.as_any().downcast_ref::<StringArray>().unwrap();
                let values: Vec<Option<&str>> = first_indices.iter().map(|&i| {
                    if src.is_null(i) { None } else { Some(src.value(i)) }
                }).collect();
                Ok((Field::new(output_name, DataType::Utf8, true), Arc::new(StringArray::from(values))))
            }
            DataType::Boolean => {
                let src = array.as_any().downcast_ref::<BooleanArray>().unwrap();
                let values: Vec<Option<bool>> = first_indices.iter().map(|&i| {
                    if src.is_null(i) { None } else { Some(src.value(i)) }
                }).collect();
                Ok((Field::new(output_name, DataType::Boolean, true), Arc::new(BooleanArray::from(values))))
            }
            _ => Ok((Field::new(output_name, DataType::Int64, true), Arc::new(Int64Array::from(vec![None::<i64>; group_indices.len()]))))
        }
    }

    /// Compute aggregate for each group
    fn compute_aggregate_for_groups(
        batch: &RecordBatch,
        func: &crate::query::AggregateFunc,
        column: &Option<String>,
        alias: &Option<String>,
        group_indices: &[Vec<usize>],
        distinct: bool,
    ) -> io::Result<(Field, ArrayRef)> {
        use crate::query::AggregateFunc;
        use std::collections::HashSet;
        
        let func_name = match func {
            AggregateFunc::Count => "COUNT",
            AggregateFunc::Sum => "SUM",
            AggregateFunc::Avg => "AVG",
            AggregateFunc::Min => "MIN",
            AggregateFunc::Max => "MAX",
        };
        
        let output_name = alias.clone().unwrap_or_else(|| {
            if let Some(col) = column { format!("{}({})", func_name, col) } else { format!("{}(*)", func_name) }
        });

        // Strip table prefix from column name if present (e.g., "o.amount" -> "amount")
        let actual_column: Option<String> = column.as_ref().map(|c| {
            let trimmed = c.trim_matches('"');
            if let Some(dot_pos) = trimmed.rfind('.') {
                trimmed[dot_pos + 1..].to_string()
            } else {
                trimmed.to_string()
            }
        });

        match func {
            AggregateFunc::Count => {
                let counts: Vec<i64> = if let Some(col_name) = &actual_column {
                    if col_name == "*" || col_name.chars().next().map(|c| c.is_ascii_digit()).unwrap_or(false) {
                        group_indices.iter().map(|g| g.len() as i64).collect()
                    } else if let Some(array) = batch.column_by_name(col_name) {
                        if distinct {
                            // COUNT(DISTINCT column) - count unique values per group
                            if let Some(int_arr) = array.as_any().downcast_ref::<Int64Array>() {
                                group_indices.iter().map(|g| {
                                    let unique: HashSet<i64> = g.iter()
                                        .filter(|&&i| !int_arr.is_null(i))
                                        .map(|&i| int_arr.value(i))
                                        .collect();
                                    unique.len() as i64
                                }).collect()
                            } else if let Some(str_arr) = array.as_any().downcast_ref::<StringArray>() {
                                group_indices.iter().map(|g| {
                                    let unique: HashSet<&str> = g.iter()
                                        .filter(|&&i| !str_arr.is_null(i))
                                        .map(|&i| str_arr.value(i))
                                        .collect();
                                    unique.len() as i64
                                }).collect()
                            } else {
                                group_indices.iter().map(|g| g.iter().filter(|&&i| !array.is_null(i)).count() as i64).collect()
                            }
                        } else {
                            group_indices.iter().map(|g| g.iter().filter(|&&i| !array.is_null(i)).count() as i64).collect()
                        }
                    } else {
                        vec![0; group_indices.len()]
                    }
                } else {
                    group_indices.iter().map(|g| g.len() as i64).collect()
                };
                Ok((Field::new(&output_name, ArrowDataType::Int64, false), Arc::new(Int64Array::from(counts))))
            }
            AggregateFunc::Sum => {
                let col_name = actual_column.as_ref().ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "SUM requires column"))?;
                let array = batch.column_by_name(col_name).ok_or_else(|| io::Error::new(io::ErrorKind::NotFound, format!("Column '{}' not found", col_name)))?;
                
                if let Some(int_array) = array.as_any().downcast_ref::<Int64Array>() {
                    let sums: Vec<i64> = group_indices.iter().map(|g| {
                        g.iter().filter_map(|&i| if int_array.is_null(i) { None } else { Some(int_array.value(i)) }).sum()
                    }).collect();
                    Ok((Field::new(&output_name, ArrowDataType::Int64, false), Arc::new(Int64Array::from(sums))))
                } else if let Some(float_array) = array.as_any().downcast_ref::<Float64Array>() {
                    let sums: Vec<f64> = group_indices.iter().map(|g| {
                        g.iter().filter_map(|&i| if float_array.is_null(i) { None } else { Some(float_array.value(i)) }).sum()
                    }).collect();
                    Ok((Field::new(&output_name, ArrowDataType::Float64, false), Arc::new(Float64Array::from(sums))))
                } else {
                    Err(io::Error::new(io::ErrorKind::InvalidData, "SUM requires numeric column"))
                }
            }
            AggregateFunc::Avg => {
                let col_name = actual_column.as_ref().ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "AVG requires column"))?;
                let array = batch.column_by_name(col_name).ok_or_else(|| io::Error::new(io::ErrorKind::NotFound, format!("Column '{}' not found", col_name)))?;
                
                if let Some(int_array) = array.as_any().downcast_ref::<Int64Array>() {
                    let avgs: Vec<f64> = group_indices.iter().map(|g| {
                        let vals: Vec<i64> = g.iter().filter_map(|&i| if int_array.is_null(i) { None } else { Some(int_array.value(i)) }).collect();
                        if vals.is_empty() { 0.0 } else { vals.iter().sum::<i64>() as f64 / vals.len() as f64 }
                    }).collect();
                    Ok((Field::new(&output_name, ArrowDataType::Float64, false), Arc::new(Float64Array::from(avgs))))
                } else if let Some(float_array) = array.as_any().downcast_ref::<Float64Array>() {
                    let avgs: Vec<f64> = group_indices.iter().map(|g| {
                        let vals: Vec<f64> = g.iter().filter_map(|&i| if float_array.is_null(i) { None } else { Some(float_array.value(i)) }).collect();
                        if vals.is_empty() { 0.0 } else { vals.iter().sum::<f64>() / vals.len() as f64 }
                    }).collect();
                    Ok((Field::new(&output_name, ArrowDataType::Float64, false), Arc::new(Float64Array::from(avgs))))
                } else {
                    Err(io::Error::new(io::ErrorKind::InvalidData, "AVG requires numeric column"))
                }
            }
            AggregateFunc::Min => {
                let col_name = actual_column.as_ref().ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "MIN requires column"))?;
                let array = batch.column_by_name(col_name).ok_or_else(|| io::Error::new(io::ErrorKind::NotFound, format!("Column '{}' not found", col_name)))?;
                
                if let Some(int_array) = array.as_any().downcast_ref::<Int64Array>() {
                    let mins: Vec<Option<i64>> = group_indices.iter().map(|g| {
                        g.iter().filter_map(|&i| if int_array.is_null(i) { None } else { Some(int_array.value(i)) }).min()
                    }).collect();
                    Ok((Field::new(&output_name, ArrowDataType::Int64, true), Arc::new(Int64Array::from(mins))))
                } else if let Some(float_array) = array.as_any().downcast_ref::<Float64Array>() {
                    let mins: Vec<Option<f64>> = group_indices.iter().map(|g| {
                        g.iter().filter_map(|&i| if float_array.is_null(i) { None } else { Some(float_array.value(i)) }).reduce(f64::min)
                    }).collect();
                    Ok((Field::new(&output_name, ArrowDataType::Float64, true), Arc::new(Float64Array::from(mins))))
                } else {
                    Err(io::Error::new(io::ErrorKind::InvalidData, "MIN requires numeric column"))
                }
            }
            AggregateFunc::Max => {
                let col_name = actual_column.as_ref().ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "MAX requires column"))?;
                let array = batch.column_by_name(col_name).ok_or_else(|| io::Error::new(io::ErrorKind::NotFound, format!("Column '{}' not found", col_name)))?;
                
                if let Some(int_array) = array.as_any().downcast_ref::<Int64Array>() {
                    let maxs: Vec<Option<i64>> = group_indices.iter().map(|g| {
                        g.iter().filter_map(|&i| if int_array.is_null(i) { None } else { Some(int_array.value(i)) }).max()
                    }).collect();
                    Ok((Field::new(&output_name, ArrowDataType::Int64, true), Arc::new(Int64Array::from(maxs))))
                } else if let Some(float_array) = array.as_any().downcast_ref::<Float64Array>() {
                    let maxs: Vec<Option<f64>> = group_indices.iter().map(|g| {
                        g.iter().filter_map(|&i| if float_array.is_null(i) { None } else { Some(float_array.value(i)) }).reduce(f64::max)
                    }).collect();
                    Ok((Field::new(&output_name, ArrowDataType::Float64, true), Arc::new(Float64Array::from(maxs))))
                } else {
                    Err(io::Error::new(io::ErrorKind::InvalidData, "MAX requires numeric column"))
                }
            }
        }
    }

    /// Execute UNION statement
    fn execute_union(union: UnionStatement, storage_path: &Path) -> io::Result<V3Result> {
        // Execute left side
        let left_result = Self::execute_parsed(*union.left, storage_path)?;
        let left_batch = left_result.to_record_batch()?;

        // Execute right side
        let right_result = Self::execute_parsed(*union.right, storage_path)?;
        let right_batch = right_result.to_record_batch()?;

        // Ensure schemas are compatible
        if left_batch.num_columns() != right_batch.num_columns() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "UNION requires same number of columns",
            ));
        }

        // Concatenate the batches
        let combined = Self::concat_batches(&left_batch, &right_batch)?;

        let mut result = if union.all {
            // UNION ALL - keep all rows
            combined
        } else {
            // UNION - remove duplicates
            Self::deduplicate_batch(&combined)?
        };

        // Apply ORDER BY if present
        if !union.order_by.is_empty() {
            result = Self::apply_order_by(&result, &union.order_by)?;
        }

        // Apply LIMIT/OFFSET if present
        if union.limit.is_some() || union.offset.is_some() {
            result = Self::apply_limit_offset(&result, union.limit, union.offset)?;
        }

        Ok(V3Result::Data(result))
    }

    /// Concatenate two record batches
    fn concat_batches(left: &RecordBatch, right: &RecordBatch) -> io::Result<RecordBatch> {
        if left.num_rows() == 0 {
            return Ok(right.clone());
        }
        if right.num_rows() == 0 {
            return Ok(left.clone());
        }

        let mut columns: Vec<ArrayRef> = Vec::with_capacity(left.num_columns());
        
        for i in 0..left.num_columns() {
            let left_col = left.column(i);
            let right_col = right.column(i);
            
            let concatenated = compute::concat(&[left_col.as_ref(), right_col.as_ref()])
                .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e.to_string()))?;
            columns.push(concatenated);
        }

        RecordBatch::try_new(left.schema(), columns)
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e.to_string()))
    }

    /// Deduplicate rows in a record batch (for UNION without ALL)
    fn deduplicate_batch(batch: &RecordBatch) -> io::Result<RecordBatch> {
        if batch.num_rows() <= 1 {
            return Ok(batch.clone());
        }

        // Build a hash set of row signatures to detect duplicates
        let mut seen: HashSet<Vec<u8>> = HashSet::with_capacity(batch.num_rows());
        let mut keep_indices: Vec<u32> = Vec::with_capacity(batch.num_rows());

        for row_idx in 0..batch.num_rows() {
            let mut row_sig = Vec::new();
            
            for col_idx in 0..batch.num_columns() {
                let col = batch.column(col_idx);
                // Create a simple signature for the row
                Self::append_value_signature(&mut row_sig, col, row_idx);
            }

            if seen.insert(row_sig) {
                keep_indices.push(row_idx as u32);
            }
        }

        if keep_indices.len() == batch.num_rows() {
            return Ok(batch.clone());
        }

        // Create filtered batch
        let indices = arrow::array::UInt32Array::from(keep_indices);
        let mut columns: Vec<ArrayRef> = Vec::with_capacity(batch.num_columns());
        
        for col in batch.columns() {
            let filtered = compute::take(col.as_ref(), &indices, None)
                .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e.to_string()))?;
            columns.push(filtered);
        }

        RecordBatch::try_new(batch.schema(), columns)
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e.to_string()))
    }

    /// Append value signature for deduplication
    fn append_value_signature(sig: &mut Vec<u8>, array: &ArrayRef, idx: usize) {
        if array.is_null(idx) {
            sig.push(0);
            return;
        }
        sig.push(1);

        if let Some(arr) = array.as_any().downcast_ref::<Int64Array>() {
            sig.extend_from_slice(&arr.value(idx).to_le_bytes());
        } else if let Some(arr) = array.as_any().downcast_ref::<UInt64Array>() {
            sig.extend_from_slice(&arr.value(idx).to_le_bytes());
        } else if let Some(arr) = array.as_any().downcast_ref::<Float64Array>() {
            sig.extend_from_slice(&arr.value(idx).to_bits().to_le_bytes());
        } else if let Some(arr) = array.as_any().downcast_ref::<StringArray>() {
            let s = arr.value(idx);
            sig.extend_from_slice(&(s.len() as u32).to_le_bytes());
            sig.extend_from_slice(s.as_bytes());
        } else if let Some(arr) = array.as_any().downcast_ref::<BooleanArray>() {
            sig.push(if arr.value(idx) { 1 } else { 0 });
        }
    }

    /// Execute window function (ROW_NUMBER, RANK, DENSE_RANK, NTILE, PERCENT_RANK, CUME_DIST, LAG, LEAD)
    fn execute_window_function(batch: &RecordBatch, stmt: &SelectStatement) -> io::Result<V3Result> {
        // Collect window specs: (func_name, args, partition_by, order_by, output_name)
        let mut window_specs: Vec<(String, Vec<String>, Vec<String>, Vec<crate::query::OrderByClause>, String)> = Vec::new();
        
        let supported = ["ROW_NUMBER", "RANK", "DENSE_RANK", "NTILE", "PERCENT_RANK", "CUME_DIST", "LAG", "LEAD", "FIRST_VALUE", "LAST_VALUE", "SUM", "AVG", "COUNT"];
        
        for col in &stmt.columns {
            if let SelectColumn::WindowFunction { name, args, partition_by, order_by, alias } = col {
                let upper = name.to_uppercase();
                if !supported.contains(&upper.as_str()) {
                    return Err(io::Error::new(io::ErrorKind::InvalidInput, 
                        format!("Unsupported window function: {}", name)));
                }
                let out_name = alias.clone().unwrap_or_else(|| name.to_lowercase());
                window_specs.push((upper, args.clone(), partition_by.clone(), order_by.clone(), out_name));
            }
        }

        if window_specs.is_empty() {
            return Err(io::Error::new(io::ErrorKind::InvalidInput, "No window function found"));
        }

        let (func_name, func_args, partition_by, order_by, _) = &window_specs[0];

        // Group rows by partition key
        let mut groups: HashMap<u64, Vec<usize>> = HashMap::new();
        
        for row_idx in 0..batch.num_rows() {
            let mut hasher = std::collections::hash_map::DefaultHasher::new();
            for col_name in partition_by {
                if let Some(col) = batch.column_by_name(col_name.trim_matches('"')) {
                    use std::hash::Hasher;
                    hasher.write_u64(Self::hash_array_value(col, row_idx));
                }
            }
            use std::hash::Hasher;
            let key = hasher.finish();
            groups.entry(key).or_insert_with(Vec::new).push(row_idx);
        }

        // Build window function result array
        let mut window_values: Vec<i64> = vec![0; batch.num_rows()];
        
        for (_, mut indices) in groups {
            // Sort within partition by ORDER BY
            let order_col = if !order_by.is_empty() {
                let order_col_name = order_by[0].column.trim_matches('"');
                let desc = order_by[0].descending;
                
                if let Some(col) = batch.column_by_name(order_col_name) {
                    indices.sort_by(|&a, &b| {
                        let cmp = Self::compare_array_values(col, a, b);
                        if desc { cmp.reverse() } else { cmp }
                    });
                    Some(col.clone())
                } else {
                    None
                }
            } else {
                None
            };
            
            // Compute window values based on function type
            match func_name.as_str() {
                "ROW_NUMBER" => {
                    for (pos, &row_idx) in indices.iter().enumerate() {
                        window_values[row_idx] = (pos + 1) as i64;
                    }
                }
                "RANK" => {
                    let mut rank = 1i64;
                    let mut prev_idx: Option<usize> = None;
                    for (pos, &row_idx) in indices.iter().enumerate() {
                        if let Some(prev) = prev_idx {
                            if let Some(ref col) = order_col {
                                if Self::compare_array_values(col, prev, row_idx) != std::cmp::Ordering::Equal {
                                    rank = (pos + 1) as i64;
                                }
                            }
                        }
                        window_values[row_idx] = rank;
                        prev_idx = Some(row_idx);
                    }
                }
                "DENSE_RANK" => {
                    let mut rank = 1i64;
                    let mut prev_idx: Option<usize> = None;
                    for &row_idx in &indices {
                        if let Some(prev) = prev_idx {
                            if let Some(ref col) = order_col {
                                if Self::compare_array_values(col, prev, row_idx) != std::cmp::Ordering::Equal {
                                    rank += 1;
                                }
                            }
                        }
                        window_values[row_idx] = rank;
                        prev_idx = Some(row_idx);
                    }
                }
                "NTILE" => {
                    let n = 4i64; // Default to 4 buckets
                    let count = indices.len() as i64;
                    for (pos, &row_idx) in indices.iter().enumerate() {
                        let bucket = (pos as i64 * n / count) + 1;
                        window_values[row_idx] = bucket.min(n);
                    }
                }
                "PERCENT_RANK" => {
                    let count = indices.len();
                    if count <= 1 {
                        for &row_idx in &indices {
                            window_values[row_idx] = 0;
                        }
                    } else {
                        let mut rank = 1i64;
                        let mut prev_idx: Option<usize> = None;
                        for (pos, &row_idx) in indices.iter().enumerate() {
                            if let Some(prev) = prev_idx {
                                if let Some(ref col) = order_col {
                                    if Self::compare_array_values(col, prev, row_idx) != std::cmp::Ordering::Equal {
                                        rank = (pos + 1) as i64;
                                    }
                                }
                            }
                            // Store rank * 1000 to preserve precision as i64
                            let pct = ((rank - 1) as f64 / (count - 1) as f64 * 1000.0) as i64;
                            window_values[row_idx] = pct;
                            prev_idx = Some(row_idx);
                        }
                    }
                }
                "CUME_DIST" => {
                    let count = indices.len();
                    let mut rank = 0i64;
                    let mut prev_idx: Option<usize> = None;
                    let mut same_count = 1;
                    
                    for (pos, &row_idx) in indices.iter().enumerate() {
                        if let Some(prev) = prev_idx {
                            if let Some(ref col) = order_col {
                                if Self::compare_array_values(col, prev, row_idx) == std::cmp::Ordering::Equal {
                                    same_count += 1;
                                } else {
                                    rank = pos as i64;
                                    same_count = 1;
                                }
                            }
                        }
                        // Store as percentage * 1000
                        let cume = (((rank + same_count) as f64 / count as f64) * 1000.0) as i64;
                        window_values[row_idx] = cume;
                        prev_idx = Some(row_idx);
                    }
                }
                "LAG" => {
                    // LAG(column, offset, default) - get value from previous row
                    let offset = if func_args.len() > 1 {
                        func_args[1].trim_start_matches("Int64(").trim_end_matches(')').parse().unwrap_or(1)
                    } else { 1usize };
                    let col_name = func_args.get(0).map(|s| s.trim_matches('"')).unwrap_or("");
                    
                    if let Some(src_col) = batch.column_by_name(col_name) {
                        if let Some(int_arr) = src_col.as_any().downcast_ref::<Int64Array>() {
                            for (pos, &row_idx) in indices.iter().enumerate() {
                                if pos >= offset {
                                    let prev_row = indices[pos - offset];
                                    window_values[row_idx] = if int_arr.is_null(prev_row) { 0 } else { int_arr.value(prev_row) };
                                } else {
                                    window_values[row_idx] = 0; // default
                                }
                            }
                        }
                    }
                }
                "LEAD" => {
                    // LEAD(column, offset, default) - get value from next row
                    let offset = if func_args.len() > 1 {
                        func_args[1].trim_start_matches("Int64(").trim_end_matches(')').parse().unwrap_or(1)
                    } else { 1usize };
                    let col_name = func_args.get(0).map(|s| s.trim_matches('"')).unwrap_or("");
                    
                    if let Some(src_col) = batch.column_by_name(col_name) {
                        if let Some(int_arr) = src_col.as_any().downcast_ref::<Int64Array>() {
                            let len = indices.len();
                            for (pos, &row_idx) in indices.iter().enumerate() {
                                if pos + offset < len {
                                    let next_row = indices[pos + offset];
                                    window_values[row_idx] = if int_arr.is_null(next_row) { 0 } else { int_arr.value(next_row) };
                                } else {
                                    window_values[row_idx] = 0; // default
                                }
                            }
                        }
                    }
                }
                "FIRST_VALUE" => {
                    // FIRST_VALUE(column) - get first value in partition
                    let col_name = func_args.get(0).map(|s| s.trim_matches('"')).unwrap_or("");
                    if let Some(src_col) = batch.column_by_name(col_name) {
                        if let Some(int_arr) = src_col.as_any().downcast_ref::<Int64Array>() {
                            let first_row = indices[0];
                            let first_val = if int_arr.is_null(first_row) { 0 } else { int_arr.value(first_row) };
                            for &row_idx in &indices {
                                window_values[row_idx] = first_val;
                            }
                        }
                    }
                }
                "LAST_VALUE" => {
                    // LAST_VALUE(column) - get last value in partition
                    let col_name = func_args.get(0).map(|s| s.trim_matches('"')).unwrap_or("");
                    if let Some(src_col) = batch.column_by_name(col_name) {
                        if let Some(int_arr) = src_col.as_any().downcast_ref::<Int64Array>() {
                            let last_row = indices[indices.len() - 1];
                            let last_val = if int_arr.is_null(last_row) { 0 } else { int_arr.value(last_row) };
                            for &row_idx in &indices {
                                window_values[row_idx] = last_val;
                            }
                        }
                    }
                }
                "SUM" => {
                    // SUM(column) OVER - running sum in partition
                    let col_name = func_args.get(0).map(|s| s.trim_matches('"')).unwrap_or("");
                    if let Some(src_col) = batch.column_by_name(col_name) {
                        if let Some(int_arr) = src_col.as_any().downcast_ref::<Int64Array>() {
                            let total: i64 = indices.iter()
                                .filter_map(|&i| if int_arr.is_null(i) { None } else { Some(int_arr.value(i)) })
                                .sum();
                            for &row_idx in &indices {
                                window_values[row_idx] = total;
                            }
                        }
                    }
                }
                "AVG" => {
                    // AVG(column) OVER - average in partition
                    let col_name = func_args.get(0).map(|s| s.trim_matches('"')).unwrap_or("");
                    if let Some(src_col) = batch.column_by_name(col_name) {
                        if let Some(int_arr) = src_col.as_any().downcast_ref::<Int64Array>() {
                            let vals: Vec<i64> = indices.iter()
                                .filter_map(|&i| if int_arr.is_null(i) { None } else { Some(int_arr.value(i)) })
                                .collect();
                            let avg = if vals.is_empty() { 0 } else { vals.iter().sum::<i64>() / vals.len() as i64 };
                            for &row_idx in &indices {
                                window_values[row_idx] = avg;
                            }
                        }
                    }
                }
                "COUNT" => {
                    // COUNT(*) OVER - count rows in partition
                    let count = indices.len() as i64;
                    for &row_idx in &indices {
                        window_values[row_idx] = count;
                    }
                }
                _ => {}
            }
        }

        // Build result with original columns plus window function result
        let mut result_fields: Vec<Field> = Vec::new();
        let mut result_arrays: Vec<ArrayRef> = Vec::new();

        for col in &stmt.columns {
            match col {
                SelectColumn::Column(name) => {
                    let col_name = name.trim_matches('"');
                    if let Some(arr) = batch.column_by_name(col_name) {
                        result_fields.push(Field::new(col_name, arr.data_type().clone(), true));
                        result_arrays.push(arr.clone());
                    }
                }
                SelectColumn::ColumnAlias { column, alias } => {
                    let col_name = column.trim_matches('"');
                    if let Some(arr) = batch.column_by_name(col_name) {
                        result_fields.push(Field::new(alias, arr.data_type().clone(), true));
                        result_arrays.push(arr.clone());
                    }
                }
                SelectColumn::All => {
                    for (i, field) in batch.schema().fields().iter().enumerate() {
                        result_fields.push(field.as_ref().clone());
                        result_arrays.push(batch.column(i).clone());
                    }
                }
                SelectColumn::WindowFunction { name, alias, .. } => {
                    let out_name = alias.clone().unwrap_or_else(|| name.to_lowercase());
                    result_fields.push(Field::new(&out_name, ArrowDataType::Int64, false));
                    result_arrays.push(Arc::new(Int64Array::from(window_values.clone())));
                }
                _ => {}
            }
        }

        let schema = Arc::new(Schema::new(result_fields));
        let result = RecordBatch::try_new(schema, result_arrays)
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e.to_string()))?;

        Ok(V3Result::Data(result))
    }

    /// Compare two array values for sorting
    fn compare_array_values(array: &ArrayRef, a: usize, b: usize) -> std::cmp::Ordering {
        use std::cmp::Ordering;
        
        if array.is_null(a) && array.is_null(b) {
            return Ordering::Equal;
        }
        if array.is_null(a) {
            return Ordering::Greater;
        }
        if array.is_null(b) {
            return Ordering::Less;
        }

        if let Some(arr) = array.as_any().downcast_ref::<Int64Array>() {
            arr.value(a).cmp(&arr.value(b))
        } else if let Some(arr) = array.as_any().downcast_ref::<Float64Array>() {
            arr.value(a).partial_cmp(&arr.value(b)).unwrap_or(Ordering::Equal)
        } else if let Some(arr) = array.as_any().downcast_ref::<StringArray>() {
            arr.value(a).cmp(arr.value(b))
        } else {
            Ordering::Equal
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;
    use crate::storage::OnDemandStorage;
    use std::collections::HashMap;

    fn create_test_storage(path: &Path) {
        let storage = OnDemandStorage::create(path).unwrap();
        
        let mut int_cols: HashMap<String, Vec<i64>> = HashMap::new();
        let mut float_cols: HashMap<String, Vec<f64>> = HashMap::new();
        let mut string_cols: HashMap<String, Vec<String>> = HashMap::new();
        
        int_cols.insert("id".to_string(), vec![1, 2, 3, 4, 5]);
        int_cols.insert("age".to_string(), vec![25, 30, 35, 40, 45]);
        float_cols.insert("score".to_string(), vec![85.0, 90.0, 75.0, 88.0, 92.0]);
        string_cols.insert("name".to_string(), vec![
            "Alice".to_string(), "Bob".to_string(), "Charlie".to_string(),
            "Diana".to_string(), "Eve".to_string()
        ]);
        
        storage.insert_typed(int_cols, float_cols, string_cols, HashMap::new(), HashMap::new()).unwrap();
        storage.save().unwrap();
    }

    #[test]
    fn test_simple_select() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test.apex");
        create_test_storage(&path);

        let result = V3Executor::execute("SELECT * FROM default", &path).unwrap();
        let batch = result.to_record_batch().unwrap();
        assert_eq!(batch.num_rows(), 5);
    }

    #[test]
    fn test_select_with_where() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test.apex");
        create_test_storage(&path);

        let result = V3Executor::execute("SELECT * FROM default WHERE age > 30", &path).unwrap();
        let batch = result.to_record_batch().unwrap();
        assert_eq!(batch.num_rows(), 3); // age 35, 40, 45
    }

    #[test]
    fn test_select_with_limit() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test.apex");
        create_test_storage(&path);

        let result = V3Executor::execute("SELECT * FROM default LIMIT 2", &path).unwrap();
        let batch = result.to_record_batch().unwrap();
        assert_eq!(batch.num_rows(), 2);
    }

    #[test]
    fn test_count_aggregate() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test.apex");
        create_test_storage(&path);

        let result = V3Executor::execute("SELECT COUNT(*) FROM default", &path).unwrap();
        let batch = result.to_record_batch().unwrap();
        assert_eq!(batch.num_rows(), 1);
        
        let count_array = batch.column(0).as_any().downcast_ref::<Int64Array>().unwrap();
        assert_eq!(count_array.value(0), 5);
    }

    #[test]
    fn test_sum_aggregate() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test.apex");
        create_test_storage(&path);

        let result = V3Executor::execute("SELECT SUM(age) FROM default", &path).unwrap();
        let batch = result.to_record_batch().unwrap();
        
        let sum_array = batch.column(0).as_any().downcast_ref::<Int64Array>().unwrap();
        assert_eq!(sum_array.value(0), 175); // 25+30+35+40+45
    }

    #[test]
    fn test_order_by() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test.apex");
        create_test_storage(&path);

        let result = V3Executor::execute("SELECT * FROM default ORDER BY age DESC LIMIT 2", &path).unwrap();
        let batch = result.to_record_batch().unwrap();
        assert_eq!(batch.num_rows(), 2);
        
        let age_array = batch.column_by_name("age").unwrap()
            .as_any().downcast_ref::<Int64Array>().unwrap();
        assert_eq!(age_array.value(0), 45);
        assert_eq!(age_array.value(1), 40);
    }
}
