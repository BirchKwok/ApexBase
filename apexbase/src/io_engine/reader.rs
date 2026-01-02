//! Read operations for the I/O Engine
//!
//! This module contains all read-related operations including queries,
//! retrieval, SQL execution, and internal filtered data reading.

use crate::table::ColumnTable;
use crate::table::column_table::TypedColumn;
use crate::data::Value;
use crate::query::{Filter, SqlExecutor, SqlResult};
use crate::ApexError;
use arrow::record_batch::RecordBatch;
use std::collections::HashMap;

use super::types::{QueryHints, IoStrategy};
use super::result::IoResult;
use super::filter::StreamingFilterEvaluator;

/// Read operations implementation
pub struct IoReader;

impl IoReader {
    /// Execute a filtered query with automatic strategy selection
    pub fn query(
        table: &mut ColumnTable,
        where_clause: &str,
        hints: QueryHints,
        strategy: IoStrategy,
    ) -> Result<IoResult, ApexError> {
        match strategy {
            IoStrategy::Arrow => Self::query_arrow(table, where_clause),
            IoStrategy::Streaming => {
                let limit = hints.limit_value.unwrap_or(1000);
                Self::query_streaming(table, where_clause, limit)
            }
            IoStrategy::Direct => Self::query_direct(table, where_clause),
        }
    }
    
    /// Execute query returning Arrow format (best for large results)
    pub fn query_arrow(
        table: &mut ColumnTable,
        where_clause: &str,
    ) -> Result<IoResult, ApexError> {
        let batch = table.query_to_record_batch(where_clause)?;
        Ok(IoResult::Arrow(batch))
    }
    
    /// Execute query with streaming early termination (best for LIMIT)
    pub fn query_streaming(
        table: &mut ColumnTable,
        where_clause: &str,
        limit: usize,
    ) -> Result<IoResult, ApexError> {
        let rows = table.query_with_limit(where_clause, Some(limit))?;
        
        // For small results, return directly; for larger, convert to Arrow
        if rows.len() <= 1000 {
            Ok(IoResult::Rows(rows))
        } else {
            IoResult::rows_to_arrow(&rows).map(IoResult::Arrow)
        }
    }
    
    /// Execute query returning row-based format (best for small results)
    pub fn query_direct(
        table: &mut ColumnTable,
        where_clause: &str,
    ) -> Result<IoResult, ApexError> {
        let rows = table.query(where_clause)?;
        Ok(IoResult::Rows(rows))
    }
    
    /// Retrieve all rows (always uses Arrow for efficiency)
    pub fn retrieve_all(table: &mut ColumnTable) -> Result<IoResult, ApexError> {
        // Use query with "1=1" filter to get all rows as Arrow
        let batch = table.query_to_record_batch("1=1")?;
        Ok(IoResult::Arrow(batch))
    }
    
    /// Retrieve single row by ID (uses Direct strategy)
    pub fn retrieve_by_id(
        table: &mut ColumnTable,
        id: u64,
    ) -> Result<Option<HashMap<String, Value>>, ApexError> {
        // Use query with ID filter
        let where_clause = format!("_id = {}", id);
        let rows = table.query(&where_clause)?;
        Ok(rows.into_iter().next())
    }
    
    /// Retrieve multiple rows by IDs (selects strategy based on count)
    pub fn retrieve_many(
        table: &mut ColumnTable,
        ids: &[u64],
    ) -> Result<IoResult, ApexError> {
        if ids.is_empty() {
            return Ok(IoResult::Empty);
        }
        
        // Build IN clause
        let id_list: Vec<String> = ids.iter().map(|id| id.to_string()).collect();
        let where_clause = format!("_id IN ({})", id_list.join(", "));
        
        if ids.len() <= 100 {
            // Direct for small batches
            let rows = table.query(&where_clause)?;
            Ok(IoResult::Rows(rows))
        } else {
            // Arrow for larger batches
            let batch = table.query_to_record_batch(&where_clause)?;
            Ok(IoResult::Arrow(batch))
        }
    }
    
    // ========== SQL EXECUTION ==========
    
    /// Execute a full SQL statement with automatic strategy selection
    pub fn execute_sql(table: &mut ColumnTable, sql: &str, strategy: IoStrategy) -> Result<SqlResult, ApexError> {
        // Execute through SqlExecutor (handles all SQL features)
        let result = SqlExecutor::execute(sql, table)?;
        
        // For Arrow strategy with large results, ensure we have Arrow batch
        if strategy == IoStrategy::Arrow && result.arrow_batch.is_none() && result.rows.len() > 10_000 {
            // Convert rows to Arrow for consistency
            if let Ok(batch) = result.to_record_batch() {
                return Ok(SqlResult::with_arrow_batch(result.columns, batch));
            }
        }
        
        Ok(result)
    }
    
    /// Execute SQL and return Arrow RecordBatch directly
    pub fn execute_sql_arrow(table: &mut ColumnTable, sql: &str) -> Result<RecordBatch, ApexError> {
        let result = SqlExecutor::execute(sql, table)?;
        result.to_record_batch()
    }
    
    // ========== INTERNAL FILTERED DATA READING (for SqlExecutor) ==========
    
    /// Read filtered rows with streaming early termination
    /// Returns row indices matching the filter, up to limit
    /// Internal method for SqlExecutor
    pub(crate) fn read_filtered_indices(
        table: &ColumnTable,
        filter: &Filter,
        limit: Option<usize>,
        offset: usize,
    ) -> Vec<usize> {
        let deleted = table.deleted_ref();
        let row_count = table.get_row_count();
        let no_deletes = deleted.all_false();
        let max_results = limit.unwrap_or(usize::MAX);
        let _need = offset + max_results;
        
        let schema = table.schema_ref();
        let columns = table.columns_ref();
        
        // Use filter's optimized column filtering
        if limit.is_none() && offset == 0 {
            // Full scan without limit - use Filter's parallel implementation
            return filter.filter_columns(schema, columns, row_count, deleted);
        }
        
        // Streaming with early termination for LIMIT queries
        let mut result = Vec::with_capacity(max_results.min(1000));
        let mut found = 0usize;
        
        // Create streaming evaluator
        let evaluator = StreamingFilterEvaluator::new(filter, schema, columns);
        
        for row_idx in 0..row_count {
            if !no_deletes && deleted.get(row_idx) { continue; }
            
            if evaluator.matches(row_idx) {
                found += 1;
                if found > offset {
                    result.push(row_idx);
                    if result.len() >= max_results { break; }
                }
            }
        }
        
        result
    }
    
    /// Read all non-deleted row indices (for queries without WHERE)
    /// Internal method for SqlExecutor
    pub(crate) fn read_all_indices(
        table: &ColumnTable,
        limit: Option<usize>,
        offset: usize,
    ) -> Vec<usize> {
        let deleted = table.deleted_ref();
        let row_count = table.get_row_count();
        let no_deletes = deleted.all_false();
        let max_results = limit.unwrap_or(usize::MAX);
        
        if no_deletes && offset == 0 && limit.is_none() {
            // Fast path: all rows, no deletions
            return (0..row_count).collect();
        }
        
        let mut result = Vec::with_capacity(max_results.min(row_count));
        let mut found = 0usize;
        
        for row_idx in 0..row_count {
            if !no_deletes && deleted.get(row_idx) { continue; }
            found += 1;
            if found > offset {
                result.push(row_idx);
                if result.len() >= max_results { break; }
            }
        }
        
        result
    }
    
    /// Build row values from column indices
    /// Internal method for SqlExecutor
    pub(crate) fn build_row_values(
        table: &ColumnTable,
        row_idx: usize,
        column_indices: &[(String, Option<usize>)],
    ) -> Vec<Value> {
        let columns = table.columns_ref();
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
        
        row_values
    }
    
    /// Get column value for sorting (returns i64 for comparison)
    /// Internal method for SqlExecutor ORDER BY
    #[allow(dead_code)]
    pub(crate) fn get_sort_key(
        table: &ColumnTable,
        row_idx: usize,
        col_idx: Option<usize>,
        is_id: bool,
    ) -> i64 {
        if is_id {
            return row_idx as i64;
        }
        if let Some(idx) = col_idx {
            let columns = table.columns_ref();
            match &columns[idx] {
                TypedColumn::Int64 { data, .. } => {
                    if row_idx < data.len() { data[row_idx] } else { i64::MIN }
                }
                TypedColumn::Float64 { data, .. } => {
                    if row_idx < data.len() { data[row_idx] as i64 } else { i64::MIN }
                }
                _ => i64::MIN,
            }
        } else {
            i64::MIN
        }
    }
}

/// Extract query hints from SQL string for strategy selection
pub fn extract_sql_hints(sql_upper: &str) -> QueryHints {
    let mut hints = QueryHints::new();
    
    // Check for aggregates
    if sql_upper.contains("COUNT(") || sql_upper.contains("SUM(") 
       || sql_upper.contains("AVG(") || sql_upper.contains("MIN(") 
       || sql_upper.contains("MAX(") {
        hints.has_aggregates = true;
    }
    
    // Check for GROUP BY
    if sql_upper.contains("GROUP BY") {
        hints.has_group_by = true;
    }
    
    // Check for ORDER BY
    if sql_upper.contains("ORDER BY") {
        hints.has_order_by = true;
    }
    
    // Check for LIMIT
    if sql_upper.contains("LIMIT") {
        hints.has_limit = true;
        // Try to extract limit value
        if let Some(pos) = sql_upper.find("LIMIT") {
            let after_limit = &sql_upper[pos + 5..];
            let limit_str: String = after_limit.trim().chars()
                .take_while(|c| c.is_ascii_digit())
                .collect();
            if let Ok(limit) = limit_str.parse::<usize>() {
                hints.limit_value = Some(limit);
            }
        }
    }
    
    hints
}
