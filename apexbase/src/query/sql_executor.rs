//! SQL Executor - Converts parsed SQL AST to query operations
//!
//! Executes SQL statements against ColumnTable storage.
//! Uses IoEngine for all data read operations.

use crate::ApexError;
use crate::data::Value;
use crate::query::Filter;
use crate::query::filter::{LikeMatcher, CompareOp};
use crate::query::sql_parser::{
    SqlStatement, SelectStatement, SelectColumn, SqlExpr, 
    BinaryOperator, UnaryOperator, OrderByClause, AggregateFunc
};
use crate::table::column_table::{ColumnTable, ColumnSchema, TypedColumn};
use crate::io_engine::{IoEngine, StreamingFilterEvaluator};
use std::collections::HashMap;
use std::cmp::Ordering;

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
            let fields: Vec<Field> = self.columns.iter()
                .map(|name| Field::new(name, ArrowDataType::Utf8, true))
                .collect();
            let schema = Arc::new(Schema::new(fields));
            return arrow::record_batch::RecordBatch::try_new(schema, vec![])
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

impl SqlExecutor {
    /// Execute a SQL statement against a table
    pub fn execute(sql: &str, table: &mut ColumnTable) -> Result<SqlResult, ApexError> {
        use crate::query::sql_parser::SqlParser;
        
        let stmt = SqlParser::parse(sql)?;
        
        match stmt {
            SqlStatement::Select(select) => Self::execute_select(select, table),
        }
    }
    
    /// Execute SELECT statement - ULTRA-OPTIMIZED
    fn execute_select(stmt: SelectStatement, table: &mut ColumnTable) -> Result<SqlResult, ApexError> {
        // Table name validation is handled at the binding layer
        // which can access all tables and route to the correct one
        
        // Only flush if there might be pending writes (lazy flush)
        if table.has_pending_writes() {
            table.flush_write_buffer();
        }
        
        // Pre-compute flags
        let has_aggregates = stmt.columns.iter().any(|c| matches!(c, SelectColumn::Aggregate { .. }));
        let no_where = stmt.where_clause.is_none();
        let no_group_by = stmt.group_by.is_empty();
        let has_limit = stmt.limit.is_some();
        let no_order = stmt.order_by.is_empty();
        
        // ============ FAST PATHS (NO INDEX COLLECTION) ============
        
        // FAST PATH: COUNT(*) without WHERE - O(1)
        if no_where && no_group_by {
            if let Some(result) = Self::try_fast_count_star(&stmt, table) {
                return Ok(result);
            }
        }
        
        // FAST PATH: Aggregates without WHERE - direct column scan
        if has_aggregates && no_group_by && no_where {
            return Self::execute_aggregate_direct(&stmt, table);
        }
        
        // FAST PATH: Simple LIMIT without WHERE/ORDER BY - streaming (NO index collection!)
        if no_where && no_order && has_limit && !stmt.distinct && !has_aggregates {
            let (result_columns, column_indices) = Self::resolve_columns(&stmt.columns, table)?;
            return Self::execute_streaming_limit(&stmt, &result_columns, &column_indices, table);
        }
        
        // FAST PATH: DISTINCT + LIMIT without WHERE/ORDER BY - streaming with early termination
        if no_where && no_order && has_limit && stmt.distinct && !has_aggregates {
            let (result_columns, column_indices) = Self::resolve_columns(&stmt.columns, table)?;
            return Self::execute_streaming_distinct(&stmt, &result_columns, &column_indices, table);
        }
        
        // FAST PATH: ORDER BY + LIMIT without WHERE - streaming top-K
        if no_where && !no_order && has_limit && !has_aggregates {
            let (result_columns, column_indices) = Self::resolve_columns(&stmt.columns, table)?;
            return Self::execute_streaming_topk(&stmt, &result_columns, &column_indices, table);
        }
        
        // FAST PATH: WHERE + LIMIT without ORDER BY - streaming with early termination
        // Handles: simple LIKE, compound AND conditions, any filter with LIMIT
        if let Some(ref where_expr) = stmt.where_clause {
            if no_order && has_limit && !has_aggregates && !stmt.distinct {
                let (result_columns, column_indices) = Self::resolve_columns(&stmt.columns, table)?;
                return Self::execute_streaming_where_limit(&stmt, &result_columns, &column_indices, table, where_expr);
            }
        }
        
        // ============ PATHS REQUIRING INDEX COLLECTION ============
        
        // FAST PATH: Simple COUNT(*) with WHERE - count directly without collecting indices
        if has_aggregates && no_group_by && stmt.columns.len() == 1 {
            if let SelectColumn::Aggregate { func: AggregateFunc::Count, column: None, alias } = &stmt.columns[0] {
                if let Some(ref where_expr) = stmt.where_clause {
                    let count = Self::count_matching_rows(where_expr, table)?;
                    let col_name = alias.clone().unwrap_or_else(|| "COUNT(*)".to_string());
                    return Ok(SqlResult::new(vec![col_name], vec![vec![Value::Int64(count as i64)]]));
                }
            }
        }
        
        // Step 1: Get matching row indices based on WHERE clause
        let matching_indices: Vec<usize> = if let Some(ref where_expr) = stmt.where_clause {
            Self::evaluate_where(where_expr, table)?
        } else {
            // All non-deleted rows (only reached for complex queries)
            let deleted = table.deleted_ref();
            let row_count = table.get_row_count();
            (0..row_count).filter(|&i| !deleted.get(i)).collect()
        };
        
        // Step 2: Determine which columns to select
        let (result_columns, column_indices) = Self::resolve_columns(&stmt.columns, table)?;
        
        // Step 3: Check for aggregates with WHERE
        if has_aggregates && no_group_by {
            return Self::execute_aggregate(&stmt, &matching_indices, table);
        }
        
        if !no_group_by {
            return Self::execute_group_by(&stmt, &matching_indices, table);
        }
        
        // With WHERE clause: use matching_indices for ORDER BY + LIMIT
        if !no_order && has_limit {
            return Self::execute_topk_with_indices(&stmt, &result_columns, &column_indices, &matching_indices, table);
        }
        
        // FAST PATH: For SELECT * with large results, use optimized batch building
        let is_select_all = matches!(stmt.columns.as_slice(), [SelectColumn::All]);
        if is_select_all && !stmt.distinct && stmt.order_by.is_empty() 
           && stmt.limit.is_none() && matching_indices.len() > 10_000 {
            let batch = table.build_record_batch_from_indices(&matching_indices)
                .map_err(|e| ApexError::SerializationError(e.to_string()))?;
            let columns: Vec<String> = batch.schema().fields().iter()
                .map(|f| f.name().clone())
                .collect();
            return Ok(SqlResult::with_arrow_batch(columns, batch));
        }
        
        // OPTIMIZATION: For DISTINCT + LIMIT without ORDER BY, we can stop early
        // once we have enough unique values
        let has_distinct_limit = stmt.distinct && stmt.limit.is_some() && stmt.order_by.is_empty();
        
        // Calculate how many rows we actually need to process
        let max_rows_needed = if !stmt.order_by.is_empty() && stmt.limit.is_some() {
            // Already truncated above
            matching_indices.len()
        } else if stmt.order_by.is_empty() && !stmt.distinct {
            // Simple case: just take offset + limit
            match (stmt.offset, stmt.limit) {
                (Some(off), Some(lim)) => (off + lim).min(matching_indices.len()),
                (None, Some(lim)) => lim.min(matching_indices.len()),
                _ => matching_indices.len(),
            }
        } else {
            matching_indices.len()
        };
        
        // Step 4: Build result rows
        let mut rows: Vec<Vec<Value>> = Vec::with_capacity(max_rows_needed.min(10000));
        let mut seen_for_distinct: Option<std::collections::HashSet<String>> = 
            if has_distinct_limit { Some(std::collections::HashSet::new()) } else { None };
        let distinct_limit = if has_distinct_limit { 
            stmt.offset.unwrap_or(0) + stmt.limit.unwrap() 
        } else { 
            usize::MAX 
        };
        
        for (processed, row_idx) in matching_indices.iter().take(max_rows_needed).enumerate() {
            let mut row_values = Vec::with_capacity(result_columns.len());
            
            for (col_name, col_idx) in column_indices.iter() {
                if col_name == "_id" {
                    row_values.push(Value::Int64(*row_idx as i64));
                } else if let Some(idx) = col_idx {
                    let value = table.columns_ref()[*idx].get(*row_idx).unwrap_or(Value::Null);
                    row_values.push(value);
                } else {
                    row_values.push(Value::Null);
                }
            }
            
            // For DISTINCT + LIMIT: check uniqueness and stop early
            if let Some(ref mut seen) = seen_for_distinct {
                let key = format!("{:?}", row_values);
                if !seen.insert(key) {
                    continue; // Skip duplicate
                }
                if seen.len() >= distinct_limit {
                    rows.push(row_values);
                    break; // Have enough unique rows
                }
            }
            
            rows.push(row_values);
            
            // Progress check: every 100k rows for very large result sets without optimization
            if processed > 0 && processed % 100000 == 0 && max_rows_needed > 100000 {
                // Still processing, continue
            }
        }
        
        // Step 5: Apply DISTINCT if needed (for cases without early optimization)
        if stmt.distinct && seen_for_distinct.is_none() {
            rows = Self::apply_distinct(rows);
        }
        
        // Step 6: Apply ORDER BY (only if not already sorted via index optimization)
        if !stmt.order_by.is_empty() && stmt.limit.is_none() {
            rows = Self::apply_order_by(rows, &result_columns, &stmt.order_by)?;
        }
        
        // Step 7: Apply LIMIT and OFFSET
        if stmt.offset.is_some() || stmt.limit.is_some() {
            let offset = stmt.offset.unwrap_or(0);
            let limit = stmt.limit.unwrap_or(rows.len());
            if offset > 0 || limit < rows.len() {
                rows = rows.into_iter().skip(offset).take(limit).collect();
            }
        }
        
        Ok(SqlResult::new(result_columns, rows))
    }
    
    /// Partial sort indices by ORDER BY columns - only get top K elements
    /// Uses BinaryHeap for O(n log k) time complexity
    fn sort_indices_by_columns_topk(
        indices: &[usize],
        order_by: &[OrderByClause],
        table: &ColumnTable,
        k: usize,
    ) -> Result<Vec<usize>, ApexError> {
        use std::collections::BinaryHeap;
        
        if indices.is_empty() || k == 0 {
            return Ok(Vec::new());
        }
        
        let k = k.min(indices.len());
        
        // For small result sets, just sort directly
        if indices.len() <= k * 2 || indices.len() <= 100 {
            let schema = table.schema_ref();
            let columns = table.columns_ref();
            let order_col_indices: Vec<(Option<usize>, bool)> = order_by.iter()
                .map(|o| {
                    let col_idx = if o.column == "_id" { None } else { schema.get_index(&o.column) };
                    (col_idx, o.descending)
                })
                .collect();
            
            let mut result: Vec<usize> = indices.to_vec();
            result.sort_by(|&a, &b| {
                for &(col_idx, desc) in &order_col_indices {
                    let av = col_idx.map(|i| columns[i].get(a)).flatten();
                    let bv = col_idx.map(|i| columns[i].get(b)).flatten();
                    let cmp = Self::compare_values(av.as_ref(), bv.as_ref(), None);
                    let cmp = if desc { cmp.reverse() } else { cmp };
                    if cmp != Ordering::Equal { return cmp; }
                }
                Ordering::Equal
            });
            result.truncate(k);
            return Ok(result);
        }
        
        let schema = table.schema_ref();
        let columns = table.columns_ref();
        
        // Get first ORDER BY column info
        let first_order = &order_by[0];
        let col_idx = if first_order.column == "_id" { None } else { schema.get_index(&first_order.column) };
        let desc = first_order.descending;
        
        // Fast path: direct Int64 access without cloning
        #[derive(Clone, Copy)]
        struct HeapItem {
            idx: usize,
            key: i64,
        }
        
        impl Eq for HeapItem {}
        impl PartialEq for HeapItem {
            fn eq(&self, other: &Self) -> bool { self.key == other.key }
        }
        impl Ord for HeapItem {
            fn cmp(&self, other: &Self) -> Ordering {
                // Reverse for min-heap: largest key at top (to be kicked out)
                other.key.cmp(&self.key)
            }
        }
        impl PartialOrd for HeapItem {
            fn partial_cmp(&self, other: &Self) -> Option<Ordering> { Some(self.cmp(other)) }
        }
        
        let mut heap: BinaryHeap<HeapItem> = BinaryHeap::with_capacity(k + 1);
        
        // Directly iterate and extract keys without closure/cloning
        if let Some(idx) = col_idx {
            match &columns[idx] {
                crate::table::column_table::TypedColumn::Int64 { data, .. } => {
                    for &row_idx in indices {
                        let key = if desc { -data[row_idx] } else { data[row_idx] };
                        if heap.len() < k {
                            heap.push(HeapItem { idx: row_idx, key });
                        } else if let Some(top) = heap.peek() {
                            if key < top.key {
                                heap.pop();
                                heap.push(HeapItem { idx: row_idx, key });
                            }
                        }
                    }
                }
                crate::table::column_table::TypedColumn::Float64 { data, .. } => {
                    for &row_idx in indices {
                        let key = if desc { -(data[row_idx] as i64) } else { data[row_idx] as i64 };
                        if heap.len() < k {
                            heap.push(HeapItem { idx: row_idx, key });
                        } else if let Some(top) = heap.peek() {
                            if key < top.key {
                                heap.pop();
                                heap.push(HeapItem { idx: row_idx, key });
                            }
                        }
                    }
                }
                _ => {
                    // Non-numeric: use row index
                    for &row_idx in indices {
                        let key = if desc { -(row_idx as i64) } else { row_idx as i64 };
                        if heap.len() < k {
                            heap.push(HeapItem { idx: row_idx, key });
                        } else if let Some(top) = heap.peek() {
                            if key < top.key {
                                heap.pop();
                                heap.push(HeapItem { idx: row_idx, key });
                            }
                        }
                    }
                }
            }
        } else {
            // ORDER BY _id
            for &row_idx in indices {
                let key = if desc { -(row_idx as i64) } else { row_idx as i64 };
                if heap.len() < k {
                    heap.push(HeapItem { idx: row_idx, key });
                } else if let Some(top) = heap.peek() {
                    if key < top.key {
                        heap.pop();
                        heap.push(HeapItem { idx: row_idx, key });
                    }
                }
            }
        }
        
        // Extract and sort by key
        let mut items: Vec<_> = heap.into_vec();
        items.sort_by_key(|h| h.key);
        Ok(items.into_iter().map(|h| h.idx).collect())
    }
    
    /// Resolve column names and indices from SELECT clause
    fn resolve_columns(
        columns: &[SelectColumn],
        table: &ColumnTable,
    ) -> Result<(Vec<String>, Vec<(String, Option<usize>)>), ApexError> {
        let mut result_names = Vec::new();
        let mut column_indices = Vec::new();
        let schema = table.schema_ref();
        
        for col in columns {
            match col {
                SelectColumn::All => {
                    // Add _id first
                    result_names.push("_id".to_string());
                    column_indices.push(("_id".to_string(), None));
                    
                    // Then all schema columns
                    for (name, _) in &schema.columns {
                        result_names.push(name.clone());
                        let idx = schema.get_index(name);
                        column_indices.push((name.clone(), idx));
                    }
                }
                SelectColumn::Column(name) => {
                    result_names.push(name.clone());
                    if name == "_id" {
                        column_indices.push((name.clone(), None));
                    } else {
                        let idx = schema.get_index(name);
                        column_indices.push((name.clone(), idx));
                    }
                }
                SelectColumn::ColumnAlias { column, alias } => {
                    result_names.push(alias.clone());
                    if column == "_id" {
                        column_indices.push((column.clone(), None));
                    } else {
                        let idx = schema.get_index(column);
                        column_indices.push((column.clone(), idx));
                    }
                }
                SelectColumn::Aggregate { func, column, alias } => {
                    let name = alias.clone().unwrap_or_else(|| {
                        let func_name = match func {
                            AggregateFunc::Count => "COUNT",
                            AggregateFunc::Sum => "SUM",
                            AggregateFunc::Avg => "AVG",
                            AggregateFunc::Min => "MIN",
                            AggregateFunc::Max => "MAX",
                        };
                        if let Some(col) = column {
                            format!("{}({})", func_name, col)
                        } else {
                            format!("{}(*)", func_name)
                        }
                    });
                    result_names.push(name.clone());
                    column_indices.push((name, None));
                }
                SelectColumn::Expression { alias, .. } => {
                    let name = alias.clone().unwrap_or_else(|| "expr".to_string());
                    result_names.push(name.clone());
                    column_indices.push((name, None));
                }
            }
        }
        
        Ok((result_names, column_indices))
    }
    
    /// Evaluate WHERE clause and return matching row indices
    fn evaluate_where(expr: &SqlExpr, table: &ColumnTable) -> Result<Vec<usize>, ApexError> {
        let filter = Self::expr_to_filter(expr)?;
        let schema = table.schema_ref();
        let columns = table.columns_ref();
        let row_count = table.get_row_count();
        let deleted = table.deleted_ref();
        
        Ok(filter.filter_columns(schema, columns, row_count, deleted))
    }
    
    /// Convert SQL expression to Filter
    fn expr_to_filter(expr: &SqlExpr) -> Result<Filter, ApexError> {
        match expr {
            SqlExpr::BinaryOp { left, op, right } => {
                match op {
                    BinaryOperator::And => {
                        let left_filter = Self::expr_to_filter(left)?;
                        let right_filter = Self::expr_to_filter(right)?;
                        // Flatten nested ANDs for better optimization
                        let mut filters = Vec::new();
                        match left_filter {
                            Filter::And(inner) => filters.extend(inner),
                            other => filters.push(other),
                        }
                        match right_filter {
                            Filter::And(inner) => filters.extend(inner),
                            other => filters.push(other),
                        }
                        Ok(Filter::And(filters))
                    }
                    BinaryOperator::Or => {
                        let left_filter = Self::expr_to_filter(left)?;
                        let right_filter = Self::expr_to_filter(right)?;
                        Ok(Filter::Or(vec![left_filter, right_filter]))
                    }
                    BinaryOperator::Eq | BinaryOperator::NotEq | 
                    BinaryOperator::Lt | BinaryOperator::Le |
                    BinaryOperator::Gt | BinaryOperator::Ge => {
                        let field = match left.as_ref() {
                            SqlExpr::Column(name) => name.clone(),
                            _ => return Err(ApexError::QueryParseError(
                                "Left side of comparison must be column".to_string()
                            )),
                        };
                        let value = match right.as_ref() {
                            SqlExpr::Literal(v) => v.clone(),
                            _ => return Err(ApexError::QueryParseError(
                                "Right side of comparison must be literal".to_string()
                            )),
                        };
                        let compare_op = match op {
                            BinaryOperator::Eq => crate::query::filter::CompareOp::Equal,
                            BinaryOperator::NotEq => crate::query::filter::CompareOp::NotEqual,
                            BinaryOperator::Lt => crate::query::filter::CompareOp::LessThan,
                            BinaryOperator::Le => crate::query::filter::CompareOp::LessEqual,
                            BinaryOperator::Gt => crate::query::filter::CompareOp::GreaterThan,
                            BinaryOperator::Ge => crate::query::filter::CompareOp::GreaterEqual,
                            _ => unreachable!(),
                        };
                        Ok(Filter::Compare { field, op: compare_op, value })
                    }
                    _ => Err(ApexError::QueryParseError(
                        format!("Unsupported binary operator: {:?}", op)
                    )),
                }
            }
            SqlExpr::UnaryOp { op, expr } => {
                match op {
                    UnaryOperator::Not => {
                        let inner = Self::expr_to_filter(expr)?;
                        Ok(Filter::Not(Box::new(inner)))
                    }
                    _ => Err(ApexError::QueryParseError(
                        format!("Unsupported unary operator: {:?}", op)
                    )),
                }
            }
            SqlExpr::Like { column, pattern, negated } => {
                let filter = Filter::Like { field: column.clone(), pattern: pattern.clone() };
                if *negated {
                    Ok(Filter::Not(Box::new(filter)))
                } else {
                    Ok(filter)
                }
            }
            SqlExpr::In { column, values, negated } => {
                let filter = Filter::In { field: column.clone(), values: values.clone() };
                if *negated {
                    Ok(Filter::Not(Box::new(filter)))
                } else {
                    Ok(filter)
                }
            }
            SqlExpr::Between { column, low, high, negated } => {
                let low_val = match low.as_ref() {
                    SqlExpr::Literal(v) => v.clone(),
                    _ => return Err(ApexError::QueryParseError(
                        "BETWEEN bounds must be literals".to_string()
                    )),
                };
                let high_val = match high.as_ref() {
                    SqlExpr::Literal(v) => v.clone(),
                    _ => return Err(ApexError::QueryParseError(
                        "BETWEEN bounds must be literals".to_string()
                    )),
                };
                // Use native Range filter for single-pass BETWEEN evaluation
                let filter = Filter::Range {
                    field: column.clone(),
                    low: low_val,
                    high: high_val,
                    low_inclusive: true,
                    high_inclusive: true,
                };
                if *negated {
                    Ok(Filter::Not(Box::new(filter)))
                } else {
                    Ok(filter)
                }
            }
            SqlExpr::IsNull { column, negated } => {
                // IS NULL is tricky - we need to check null bitmap
                // For now, approximate with comparison to Null
                let filter = Filter::Compare { 
                    field: column.clone(), 
                    op: crate::query::filter::CompareOp::Equal, 
                    value: Value::Null 
                };
                if *negated {
                    Ok(Filter::Not(Box::new(filter)))
                } else {
                    Ok(filter)
                }
            }
            SqlExpr::Paren(inner) => Self::expr_to_filter(inner),
            SqlExpr::Literal(Value::Bool(true)) => Ok(Filter::True),
            SqlExpr::Literal(Value::Bool(false)) => Ok(Filter::False),
            _ => Err(ApexError::QueryParseError(
                format!("Cannot convert expression to filter: {:?}", expr)
            )),
        }
    }
    
    /// Apply DISTINCT to result rows
    fn apply_distinct(rows: Vec<Vec<Value>>) -> Vec<Vec<Value>> {
        let mut seen = std::collections::HashSet::new();
        let mut result = Vec::new();
        
        for row in rows {
            let key = format!("{:?}", row);
            if seen.insert(key) {
                result.push(row);
            }
        }
        
        result
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
    fn compare_values(a: Option<&Value>, b: Option<&Value>, nulls_first: Option<bool>) -> Ordering {
        match (a, b) {
            (None, None) => Ordering::Equal,
            (None, Some(_)) => if nulls_first.unwrap_or(false) { Ordering::Less } else { Ordering::Greater },
            (Some(_), None) => if nulls_first.unwrap_or(false) { Ordering::Greater } else { Ordering::Less },
            (Some(Value::Null), Some(Value::Null)) => Ordering::Equal,
            (Some(Value::Null), Some(_)) => if nulls_first.unwrap_or(false) { Ordering::Less } else { Ordering::Greater },
            (Some(_), Some(Value::Null)) => if nulls_first.unwrap_or(false) { Ordering::Greater } else { Ordering::Less },
            (Some(av), Some(bv)) => Self::compare_non_null(av, bv),
        }
    }
    
    fn compare_non_null(a: &Value, b: &Value) -> Ordering {
        match (a, b) {
            (Value::Int64(x), Value::Int64(y)) => x.cmp(y),
            (Value::Float64(x), Value::Float64(y)) => x.partial_cmp(y).unwrap_or(Ordering::Equal),
            (Value::String(x), Value::String(y)) => x.cmp(y),
            (Value::Bool(x), Value::Bool(y)) => x.cmp(y),
            _ => Ordering::Equal,
        }
    }
    
    // ========================================================================
    // ULTRA-FAST PATH IMPLEMENTATIONS
    // ========================================================================
    
    /// O(1) COUNT(*) without WHERE clause
    fn try_fast_count_star(stmt: &SelectStatement, table: &ColumnTable) -> Option<SqlResult> {
        // Check if this is a simple COUNT(*) query
        if stmt.columns.len() == 1 {
            if let SelectColumn::Aggregate { func: AggregateFunc::Count, column: None, alias } = &stmt.columns[0] {
                let col_name = alias.clone().unwrap_or_else(|| "COUNT(*)".to_string());
                let count = table.row_count() as i64;
                return Some(SqlResult::new(vec![col_name], vec![vec![Value::Int64(count)]]));
            }
        }
        None
    }
    
    /// Direct aggregate computation without index collection - ultra-fast for full table scans
    fn execute_aggregate_direct(stmt: &SelectStatement, table: &ColumnTable) -> Result<SqlResult, ApexError> {
        let schema = table.schema_ref();
        let columns = table.columns_ref();
        let deleted = table.deleted_ref();
        let row_count = table.get_row_count();
        
        // Collect aggregate specs
        let mut agg_specs: Vec<(AggregateFunc, Option<String>, String)> = Vec::new();
        for col in &stmt.columns {
            if let SelectColumn::Aggregate { func, column, alias } = col {
                let col_name = alias.clone().unwrap_or_else(|| {
                    let func_name = match func {
                        AggregateFunc::Count => "COUNT",
                        AggregateFunc::Sum => "SUM",
                        AggregateFunc::Avg => "AVG",
                        AggregateFunc::Min => "MIN",
                        AggregateFunc::Max => "MAX",
                    };
                    if let Some(c) = column { format!("{}({})", func_name, c) }
                    else { format!("{}(*)", func_name) }
                });
                agg_specs.push((func.clone(), column.clone(), col_name));
            }
        }
        
        // Check if all aggregates use same numeric column
        let same_column: Option<&str> = {
            let cols: Vec<_> = agg_specs.iter().filter_map(|(_, c, _)| c.as_deref()).collect();
            if cols.is_empty() || cols.windows(2).all(|w| w[0] == w[1]) { cols.first().copied() }
            else { None }
        };
        
        // Ultra-fast path: direct Int64 column scan
        if let Some(col_name) = same_column {
            if let Some(col_idx) = schema.get_index(col_name) {
                if let crate::table::column_table::TypedColumn::Int64 { data, nulls } = &columns[col_idx] {
                    return Self::compute_aggregates_int64_direct(&agg_specs, data, nulls, deleted, row_count);
                }
            }
        }
        
        // Fast path: direct scan without index collection
        Self::compute_aggregates_direct(&agg_specs, schema, columns, deleted, row_count)
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
            count: i64,
            sum: f64,
            min: Option<Value>,
            max: Option<Value>,
        }
        
        let no_deletes = deleted.all_false();
        let mut accumulators: Vec<Accumulator> = agg_specs.iter()
            .map(|(_, col, _)| Accumulator {
                col_idx: col.as_ref().and_then(|c| schema.get_index(c)),
                count: 0, sum: 0.0, min: None, max: None,
            })
            .collect();
        
        for row_idx in 0..row_count {
            if !no_deletes && deleted.get(row_idx) { continue; }
            
            for (i, (func, _, _)) in agg_specs.iter().enumerate() {
                let acc = &mut accumulators[i];
                if let Some(col_idx) = acc.col_idx {
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
    fn execute_streaming_limit(
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
    fn execute_streaming_topk(
        stmt: &SelectStatement,
        result_columns: &[String],
        column_indices: &[(String, Option<usize>)],
        table: &ColumnTable,
    ) -> Result<SqlResult, ApexError> {
        let deleted = table.deleted_ref();
        let columns = table.columns_ref();
        let schema = table.schema_ref();
        let row_count = table.get_row_count();
        let no_deletes = deleted.all_false();
        let k = stmt.offset.unwrap_or(0) + stmt.limit.unwrap_or(usize::MAX);
        
        let first_order = &stmt.order_by[0];
        let order_col_idx = if first_order.column == "_id" { None } else { schema.get_index(&first_order.column) };
        let desc = first_order.descending;
        
        // For Int64 ORDER BY DESC/ASC, use optimized direct scan
        let top_indices: Vec<usize> = if let Some(idx) = order_col_idx {
            if let crate::table::column_table::TypedColumn::Int64 { data, .. } = &columns[idx] {
                // Direct top-K selection with minimal overhead
                let mut top_k: Vec<(i64, usize)> = Vec::with_capacity(k + 1);
                let data_len = data.len().min(row_count);
                
                if no_deletes {
                    // Ultra-fast path: no deleted rows
                    for row_idx in 0..data_len {
                        let val = data[row_idx];
                        let key = if desc { -val } else { val };
                        
                        if top_k.len() < k {
                            top_k.push((key, row_idx));
                            if top_k.len() == k {
                                // Build heap once when full
                                top_k.sort_unstable_by_key(|x| std::cmp::Reverse(x.0));
                            }
                        } else if key < top_k[0].0 {
                            // Replace worst element and re-heapify
                            top_k[0] = (key, row_idx);
                            // Bubble down
                            let mut i = 0;
                            loop {
                                let left = 2 * i + 1;
                                let right = 2 * i + 2;
                                let mut largest = i;
                                if left < k && top_k[left].0 > top_k[largest].0 { largest = left; }
                                if right < k && top_k[right].0 > top_k[largest].0 { largest = right; }
                                if largest == i { break; }
                                top_k.swap(i, largest);
                                i = largest;
                            }
                        }
                    }
                } else {
                    for row_idx in 0..data_len {
                        if deleted.get(row_idx) { continue; }
                        let val = data[row_idx];
                        let key = if desc { -val } else { val };
                        
                        if top_k.len() < k {
                            top_k.push((key, row_idx));
                            if top_k.len() == k {
                                top_k.sort_unstable_by_key(|x| std::cmp::Reverse(x.0));
                            }
                        } else if key < top_k[0].0 {
                            top_k[0] = (key, row_idx);
                            let mut i = 0;
                            loop {
                                let left = 2 * i + 1;
                                let right = 2 * i + 2;
                                let mut largest = i;
                                if left < k && top_k[left].0 > top_k[largest].0 { largest = left; }
                                if right < k && top_k[right].0 > top_k[largest].0 { largest = right; }
                                if largest == i { break; }
                                top_k.swap(i, largest);
                                i = largest;
                            }
                        }
                    }
                }
                
                top_k.sort_unstable_by_key(|x| x.0);
                top_k.into_iter().map(|(_, idx)| idx).collect()
            } else {
                // Fallback for non-Int64: use simple vec
                Self::topk_generic(row_count, k, deleted, no_deletes, desc)
            }
        } else {
            // ORDER BY _id: trivial case
            if desc {
                (0..row_count).rev().filter(|&i| no_deletes || !deleted.get(i)).take(k).collect()
            } else {
                (0..row_count).filter(|&i| no_deletes || !deleted.get(i)).take(k).collect()
            }
        };
        
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
        let string_data = match &columns[like_col_idx] {
            crate::table::column_table::TypedColumn::String { data, nulls } => Some((data, nulls)),
            _ => None,
        };
        
        let (data, nulls) = match string_data {
            Some(d) => d,
            None => return Ok(SqlResult::new(result_columns.to_vec(), Vec::new())),
        };
        
        let no_nulls = nulls.all_false();
        let data_len = data.len().min(row_count);
        
        let mut rows = Vec::with_capacity(limit.min(100));
        let mut found = 0usize;
        
        // Streaming scan with early termination
        for row_idx in 0..data_len {
            // Skip deleted rows
            if !no_deletes && deleted.get(row_idx) { continue; }
            // Skip null values
            if !no_nulls && nulls.get(row_idx) { continue; }
            
            // Check LIKE match
            let matches = matcher.matches(&data[row_idx]);
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
    fn execute_streaming_where_limit(
        stmt: &SelectStatement,
        result_columns: &[String],
        column_indices: &[(String, Option<usize>)],
        table: &ColumnTable,
        where_expr: &SqlExpr,
    ) -> Result<SqlResult, ApexError> {
        let offset = stmt.offset.unwrap_or(0);
        let limit = stmt.limit.unwrap_or(usize::MAX);
        
        // Convert WHERE expression to optimized filter
        let filter = Self::expr_to_filter(where_expr)?;
        
        // Use IoEngine for filtered data reading with streaming early termination
        let matching_indices = IoEngine::read_filtered_indices(table, &filter, Some(limit), offset);
        
        // Build result rows using IoEngine
        let rows: Vec<Vec<Value>> = matching_indices.iter()
            .map(|&row_idx| IoEngine::build_row_values(table, row_idx, column_indices))
            .collect();
        
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
        use std::collections::BinaryHeap;
        
        let deleted = table.deleted_ref();
        let columns = table.columns_ref();
        let schema = table.schema_ref();
        let row_count = table.get_row_count();
        let no_deletes = deleted.all_false();
        let k = stmt.offset.unwrap_or(0) + stmt.limit.unwrap_or(usize::MAX);
        
        // Convert WHERE expression to optimized filter
        let filter = Self::expr_to_filter(where_expr)?;
        let evaluator = StreamingFilterEvaluator::new(&filter, schema, columns);
        
        // Get ORDER BY info
        let first_order = &stmt.order_by[0];
        let order_col_idx = if first_order.column == "_id" { None } else { schema.get_index(&first_order.column) };
        let desc = first_order.descending;
        
        #[derive(Clone)]
        struct HeapItem { idx: usize, key: i64 }
        impl Eq for HeapItem {}
        impl PartialEq for HeapItem { fn eq(&self, other: &Self) -> bool { self.key == other.key } }
        impl Ord for HeapItem { fn cmp(&self, other: &Self) -> std::cmp::Ordering { other.key.cmp(&self.key) } }
        impl PartialOrd for HeapItem { fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> { Some(self.cmp(other)) } }
        
        let mut heap: BinaryHeap<HeapItem> = BinaryHeap::with_capacity(k + 1);
        
        // Streaming scan with top-K selection
        if let Some(idx) = order_col_idx {
            match &columns[idx] {
                crate::table::column_table::TypedColumn::Int64 { data, .. } => {
                    let data_len = data.len().min(row_count);
                    for row_idx in 0..data_len {
                        if !no_deletes && deleted.get(row_idx) { continue; }
                        if !evaluator.matches(row_idx) { continue; }
                        
                        let key = if desc { -data[row_idx] } else { data[row_idx] };
                        if heap.len() < k {
                            heap.push(HeapItem { idx: row_idx, key });
                        } else if let Some(top) = heap.peek() {
                            if key < top.key {
                                heap.pop();
                                heap.push(HeapItem { idx: row_idx, key });
                            }
                        }
                    }
                }
                crate::table::column_table::TypedColumn::Float64 { data, .. } => {
                    let data_len = data.len().min(row_count);
                    for row_idx in 0..data_len {
                        if !no_deletes && deleted.get(row_idx) { continue; }
                        if !evaluator.matches(row_idx) { continue; }
                        
                        let key = if desc { -(data[row_idx] * 1000.0) as i64 } else { (data[row_idx] * 1000.0) as i64 };
                        if heap.len() < k {
                            heap.push(HeapItem { idx: row_idx, key });
                        } else if let Some(top) = heap.peek() {
                            if key < top.key {
                                heap.pop();
                                heap.push(HeapItem { idx: row_idx, key });
                            }
                        }
                    }
                }
                _ => {
                    // Non-numeric: use row index as key
                    for row_idx in 0..row_count {
                        if !no_deletes && deleted.get(row_idx) { continue; }
                        if !evaluator.matches(row_idx) { continue; }
                        
                        let key = if desc { -(row_idx as i64) } else { row_idx as i64 };
                        if heap.len() < k {
                            heap.push(HeapItem { idx: row_idx, key });
                        } else if let Some(top) = heap.peek() {
                            if key < top.key {
                                heap.pop();
                                heap.push(HeapItem { idx: row_idx, key });
                            }
                        }
                    }
                }
            }
        } else {
            // ORDER BY _id
            for row_idx in 0..row_count {
                if !no_deletes && deleted.get(row_idx) { continue; }
                if !evaluator.matches(row_idx) { continue; }
                
                let key = if desc { -(row_idx as i64) } else { row_idx as i64 };
                if heap.len() < k {
                    heap.push(HeapItem { idx: row_idx, key });
                } else if let Some(top) = heap.peek() {
                    if key < top.key {
                        heap.pop();
                        heap.push(HeapItem { idx: row_idx, key });
                    }
                }
            }
        }
        
        // Sort heap items by key
        let mut items: Vec<_> = heap.into_vec();
        items.sort_by_key(|h| h.key);
        
        // Build result rows
        let offset = stmt.offset.unwrap_or(0);
        let limit = stmt.limit.unwrap_or(usize::MAX);
        let mut rows = Vec::with_capacity(limit.min(items.len()));
        
        for item in items.into_iter().skip(offset).take(limit) {
            let row_idx = item.idx;
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
    fn execute_streaming_distinct(
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
    fn count_matching_rows(where_expr: &SqlExpr, table: &ColumnTable) -> Result<usize, ApexError> {
        use rayon::prelude::*;
        
        let filter = Self::expr_to_filter(where_expr)?;
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
        use std::collections::BinaryHeap;
        
        let columns = table.columns_ref();
        let schema = table.schema_ref();
        let k = stmt.offset.unwrap_or(0) + stmt.limit.unwrap_or(usize::MAX);
        
        let first_order = &stmt.order_by[0];
        let order_col_idx = if first_order.column == "_id" { None } else { schema.get_index(&first_order.column) };
        let desc = first_order.descending;
        
        #[derive(Clone)]
        struct HeapItem { idx: usize, key: i64 }
        impl Eq for HeapItem {}
        impl PartialEq for HeapItem { fn eq(&self, other: &Self) -> bool { self.key == other.key } }
        impl Ord for HeapItem { fn cmp(&self, other: &Self) -> Ordering { other.key.cmp(&self.key) } }
        impl PartialOrd for HeapItem { fn partial_cmp(&self, other: &Self) -> Option<Ordering> { Some(self.cmp(other)) } }
        
        let mut heap: BinaryHeap<HeapItem> = BinaryHeap::with_capacity(k + 1);
        
        // Use pre-filtered matching_indices
        if let Some(idx) = order_col_idx {
            if let crate::table::column_table::TypedColumn::Int64 { data, .. } = &columns[idx] {
                for &row_idx in matching_indices {
                    let key = if desc { -data[row_idx] } else { data[row_idx] };
                    if heap.len() < k {
                        heap.push(HeapItem { idx: row_idx, key });
                    } else if let Some(top) = heap.peek() {
                        if key < top.key {
                            heap.pop();
                            heap.push(HeapItem { idx: row_idx, key });
                        }
                    }
                }
            } else {
                for &row_idx in matching_indices {
                    let key = if desc { -(row_idx as i64) } else { row_idx as i64 };
                    if heap.len() < k {
                        heap.push(HeapItem { idx: row_idx, key });
                    } else if let Some(top) = heap.peek() {
                        if key < top.key {
                            heap.pop();
                            heap.push(HeapItem { idx: row_idx, key });
                        }
                    }
                }
            }
        } else {
            for &row_idx in matching_indices {
                let key = if desc { -(row_idx as i64) } else { row_idx as i64 };
                if heap.len() < k {
                    heap.push(HeapItem { idx: row_idx, key });
                } else if let Some(top) = heap.peek() {
                    if key < top.key {
                        heap.pop();
                        heap.push(HeapItem { idx: row_idx, key });
                    }
                }
            }
        }
        
        let mut items: Vec<_> = heap.into_vec();
        items.sort_by_key(|h| h.key);
        
        let offset = stmt.offset.unwrap_or(0);
        let limit = stmt.limit.unwrap_or(usize::MAX);
        let mut rows = Vec::with_capacity(limit.min(items.len()));
        
        for item in items.into_iter().skip(offset).take(limit) {
            let row_idx = item.idx;
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
        let mut agg_specs: Vec<(AggregateFunc, Option<String>, String)> = Vec::new();
        for col in &stmt.columns {
            if let SelectColumn::Aggregate { func, column, alias } = col {
                let col_name = alias.clone().unwrap_or_else(|| {
                    let func_name = match func {
                        AggregateFunc::Count => "COUNT",
                        AggregateFunc::Sum => "SUM",
                        AggregateFunc::Avg => "AVG",
                        AggregateFunc::Min => "MIN",
                        AggregateFunc::Max => "MAX",
                    };
                    if let Some(c) = column {
                        format!("{}({})", func_name, c)
                    } else {
                        format!("{}(*)", func_name)
                    }
                });
                agg_specs.push((func.clone(), column.clone(), col_name));
            }
        }
        
        // Check if all aggregates use the same numeric column - enable ultra-fast path
        let same_column: Option<&str> = {
            let cols: Vec<_> = agg_specs.iter()
                .filter_map(|(_, c, _)| c.as_deref())
                .collect();
            if cols.is_empty() || cols.windows(2).all(|w| w[0] == w[1]) {
                cols.first().copied()
            } else {
                None
            }
        };
        
        // Ultra-fast path: all aggregates on same Int64 column
        if let Some(col_name) = same_column {
            if let Some(col_idx) = schema.get_index(col_name) {
                if let crate::table::column_table::TypedColumn::Int64 { data, nulls } = &columns[col_idx] {
                    return Self::compute_aggregates_int64_fused(&agg_specs, data, nulls, matching_indices);
                }
            }
        }
        
        // Fast path: compute all aggregates in single pass with accumulators
        Self::compute_aggregates_generic_fused(&agg_specs, schema, columns, matching_indices)
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
    fn execute_group_by(
        stmt: &SelectStatement,
        matching_indices: &[usize],
        table: &ColumnTable,
    ) -> Result<SqlResult, ApexError> {
        let schema = table.schema_ref();
        let columns = table.columns_ref();
        
        // Group rows by GROUP BY columns
        let mut groups: HashMap<String, Vec<usize>> = HashMap::new();
        
        for &row_idx in matching_indices {
            let mut key_parts = Vec::new();
            for group_col in &stmt.group_by {
                if let Some(col_idx) = schema.get_index(group_col) {
                    let val = columns[col_idx].get(row_idx).unwrap_or(Value::Null);
                    key_parts.push(format!("{:?}", val));
                }
            }
            let key = key_parts.join("|");
            groups.entry(key).or_insert_with(Vec::new).push(row_idx);
        }
        
        // Build result columns
        let mut result_columns = Vec::new();
        for col in &stmt.columns {
            match col {
                SelectColumn::Column(name) => result_columns.push(name.clone()),
                SelectColumn::ColumnAlias { alias, .. } => result_columns.push(alias.clone()),
                SelectColumn::Aggregate { func, column, alias } => {
                    let name = alias.clone().unwrap_or_else(|| {
                        let func_name = match func {
                            AggregateFunc::Count => "COUNT",
                            AggregateFunc::Sum => "SUM",
                            AggregateFunc::Avg => "AVG",
                            AggregateFunc::Min => "MIN",
                            AggregateFunc::Max => "MAX",
                        };
                        if let Some(c) = column {
                            format!("{}({})", func_name, c)
                        } else {
                            format!("{}(*)", func_name)
                        }
                    });
                    result_columns.push(name);
                }
                _ => {}
            }
        }
        
        // Build result rows
        let mut result_rows = Vec::new();
        
        for (_, group_indices) in groups {
            let mut row = Vec::new();
            
            for col in &stmt.columns {
                match col {
                    SelectColumn::Column(name) | SelectColumn::ColumnAlias { column: name, .. } => {
                        // Get value from first row in group
                        if let Some(&first_idx) = group_indices.first() {
                            if let Some(col_idx) = schema.get_index(name) {
                                row.push(columns[col_idx].get(first_idx).unwrap_or(Value::Null));
                            } else {
                                row.push(Value::Null);
                            }
                        }
                    }
                    SelectColumn::Aggregate { func, column, .. } => {
                        let value = Self::compute_aggregate(func, column.as_deref(), &group_indices, table)?;
                        row.push(value);
                    }
                    _ => {}
                }
            }
            
            result_rows.push(row);
        }
        
        // Apply HAVING filter
        if let Some(ref having_expr) = stmt.having {
            // TODO: Implement HAVING filter
            // For now, skip HAVING
        }
        
        Ok(SqlResult::new(result_columns, result_rows))
    }
    
    /// Build Arrow arrays directly from column data using matching indices
    /// This is much faster than row-by-row construction for large result sets
    fn build_arrow_direct(
        result_columns: &[String],
        column_indices: &[(String, Option<usize>)],
        matching_indices: &[usize],
        table: &ColumnTable,
    ) -> Result<SqlResult, ApexError> {
        use arrow::array::{ArrayRef, Int64Array, Float64Array, StringBuilder, BooleanArray};
        use arrow::datatypes::{DataType as ArrowDataType, Field, Schema};
        use arrow::record_batch::RecordBatch;
        use std::sync::Arc;
        use rayon::prelude::*;
        
        let columns = table.columns_ref();
        let num_rows = matching_indices.len();
        
        // Build Arrow arrays directly from column data
        let mut fields = Vec::with_capacity(result_columns.len());
        let mut arrays: Vec<ArrayRef> = Vec::with_capacity(result_columns.len());
        
        for (col_name, col_idx) in column_indices {
            if col_name == "_id" {
                // Build _id array from indices
                let id_values: Vec<i64> = matching_indices.iter().map(|&i| i as i64).collect();
                fields.push(Field::new("_id", ArrowDataType::Int64, false));
                arrays.push(Arc::new(Int64Array::from(id_values)));
            } else if let Some(idx) = col_idx {
                match &columns[*idx] {
                    TypedColumn::Int64 { data, nulls } => {
                        // Parallel gather for Int64
                        let values: Vec<Option<i64>> = if num_rows > 100_000 {
                            matching_indices.par_iter()
                                .map(|&i| {
                                    if i < data.len() && !nulls.get(i) {
                                        Some(data[i])
                                    } else {
                                        None
                                    }
                                })
                                .collect()
                        } else {
                            matching_indices.iter()
                                .map(|&i| {
                                    if i < data.len() && !nulls.get(i) {
                                        Some(data[i])
                                    } else {
                                        None
                                    }
                                })
                                .collect()
                        };
                        fields.push(Field::new(col_name, ArrowDataType::Int64, true));
                        arrays.push(Arc::new(Int64Array::from(values)));
                    }
                    TypedColumn::Float64 { data, nulls } => {
                        let values: Vec<Option<f64>> = if num_rows > 100_000 {
                            matching_indices.par_iter()
                                .map(|&i| {
                                    if i < data.len() && !nulls.get(i) {
                                        Some(data[i])
                                    } else {
                                        None
                                    }
                                })
                                .collect()
                        } else {
                            matching_indices.iter()
                                .map(|&i| {
                                    if i < data.len() && !nulls.get(i) {
                                        Some(data[i])
                                    } else {
                                        None
                                    }
                                })
                                .collect()
                        };
                        fields.push(Field::new(col_name, ArrowDataType::Float64, true));
                        arrays.push(Arc::new(Float64Array::from(values)));
                    }
                    TypedColumn::String { data, nulls } => {
                        // For strings, use StringBuilder for efficiency
                        let mut builder = StringBuilder::with_capacity(num_rows, num_rows * 64);
                        for &i in matching_indices {
                            if i < data.len() && !nulls.get(i) {
                                builder.append_value(&data[i]);
                            } else {
                                builder.append_null();
                            }
                        }
                        fields.push(Field::new(col_name, ArrowDataType::Utf8, true));
                        arrays.push(Arc::new(builder.finish()));
                    }
                    TypedColumn::Bool { data, nulls } => {
                        let values: Vec<Option<bool>> = matching_indices.iter()
                            .map(|&i| {
                                if i < data.len() && !nulls.get(i) {
                                    Some(data.get(i))
                                } else {
                                    None
                                }
                            })
                            .collect();
                        fields.push(Field::new(col_name, ArrowDataType::Boolean, true));
                        arrays.push(Arc::new(BooleanArray::from(values)));
                    }
                    TypedColumn::Mixed { data, nulls } => {
                        // Mixed type - convert to string representation
                        let mut builder = StringBuilder::with_capacity(num_rows, num_rows * 32);
                        for &i in matching_indices {
                            if i < data.len() && !nulls.get(i) {
                                builder.append_value(data[i].to_string_value());
                            } else {
                                builder.append_null();
                            }
                        }
                        fields.push(Field::new(col_name, ArrowDataType::Utf8, true));
                        arrays.push(Arc::new(builder.finish()));
                    }
                }
            } else {
                // Column not found - add nulls
                let null_values: Vec<Option<i64>> = vec![None; num_rows];
                fields.push(Field::new(col_name, ArrowDataType::Int64, true));
                arrays.push(Arc::new(Int64Array::from(null_values)));
            }
        }
        
        // Create RecordBatch and return SqlResult with embedded Arrow batch
        let schema = Arc::new(Schema::new(fields));
        let batch = RecordBatch::try_new(schema, arrays)
            .map_err(|e| ApexError::SerializationError(e.to_string()))?;
        
        Ok(SqlResult::with_arrow_batch(result_columns.to_vec(), batch))
    }
}
