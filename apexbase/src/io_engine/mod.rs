//! Unified I/O Engine for ApexBase
//! 
//! This module provides a unified interface for all data read/write operations,
//! automatically selecting the optimal strategy based on operation characteristics.
//! 
//! # Read Strategies
//! 
//! - **Arrow**: Zero-copy columnar format, best for large result sets (>1000 rows)
//! - **Streaming**: Early termination for LIMIT queries, best for partial scans
//! - **Direct**: Traditional row-based, best for small results (<100 rows) and aggregates
//! 
//! # Write Strategies
//! 
//! - **Single**: Single row insert/update/delete, best for interactive operations
//! - **Batch**: Bulk insert/update/delete, best for large data loads (>100 rows)
//! - **Transactional**: Atomic multi-row operations with rollback support
//! 
//! # Architecture
//! 
//! ```text
//! ┌─────────────────────────────────────────────────────────────────┐
//! │                        Python Bindings                          │
//! └─────────────────────────────────────────────────────────────────┘
//!                                │
//!                                ▼
//! ┌─────────────────────────────────────────────────────────────────┐
//! │                         IoEngine                                │
//! │  ┌────────────────────────────────────────────────────────────┐ │
//! │  │ READ:  Arrow | Streaming | Direct                          │ │
//! │  │ WRITE: Single | Batch | Transactional                      │ │
//! │  └────────────────────────────────────────────────────────────┘ │
//! └─────────────────────────────────────────────────────────────────┘
//!                                │
//!                                ▼
//! ┌─────────────────────────────────────────────────────────────────┐
//! │                       ColumnTable                               │
//! └─────────────────────────────────────────────────────────────────┘
//! ```

// Submodules
mod types;
mod result;
mod filter;
mod reader;
mod writer;

// Re-exports
pub use types::{QueryHints, IoStrategy, WriteStrategy, WriteHints, WriteResult};
pub use result::IoResult;
pub use reader::{IoReader, extract_sql_hints};
pub use writer::IoWriter;
pub(crate) use filter::StreamingFilterEvaluator;

use crate::table::ColumnTable;
use crate::data::Value;
use crate::query::{Filter, SqlResult};
use crate::ApexError;
use arrow::record_batch::RecordBatch;
use std::collections::HashMap;

/// Unified I/O Engine - Router Layer
/// 
/// IoEngine acts as the main entry point and router for all I/O operations.
/// Actual implementations are delegated to specialized submodules:
/// - `reader`: Read operations (query, retrieve, SQL execution)
/// - `writer`: Write operations (insert, update, delete)
pub struct IoEngine;

impl IoEngine {
    // ========== STRATEGY SELECTION ==========
    
    /// Select the optimal read I/O strategy based on query hints
    pub fn select_strategy(hints: &QueryHints) -> IoStrategy {
        // Force strategy if specified
        if let Some(strategy) = hints.force_strategy {
            return strategy;
        }
        
        // Streaming for LIMIT queries without ORDER BY (early termination)
        if hints.has_limit && !hints.has_order_by && !hints.has_aggregates {
            if let Some(limit) = hints.limit_value {
                if limit <= 10000 {
                    return IoStrategy::Streaming;
                }
            }
        }
        
        // Direct for aggregates and small results
        if hints.has_aggregates || hints.has_group_by {
            return IoStrategy::Direct;
        }
        
        // Estimate-based selection
        if let Some(estimated) = hints.estimated_rows {
            if estimated <= 100 {
                return IoStrategy::Direct;
            }
        }
        
        // Default to Arrow for large/unknown results
        IoStrategy::Arrow
    }
    
    /// Select the optimal write strategy based on hints
    #[inline]
    pub fn select_write_strategy(hints: &WriteHints) -> WriteStrategy {
        IoWriter::select_write_strategy(hints)
    }
    
    // ========== READ OPERATIONS (delegated to IoReader) ==========
    
    /// Execute a filtered query with automatic strategy selection
    #[inline]
    pub fn query(
        table: &mut ColumnTable,
        where_clause: &str,
        hints: QueryHints,
    ) -> Result<IoResult, ApexError> {
        let strategy = Self::select_strategy(&hints);
        IoReader::query(table, where_clause, hints, strategy)
    }
    
    /// Execute query returning Arrow format (best for large results)
    #[inline]
    pub fn query_arrow(
        table: &mut ColumnTable,
        where_clause: &str,
    ) -> Result<IoResult, ApexError> {
        IoReader::query_arrow(table, where_clause)
    }
    
    /// Execute query with streaming early termination (best for LIMIT)
    #[inline]
    pub fn query_streaming(
        table: &mut ColumnTable,
        where_clause: &str,
        limit: usize,
    ) -> Result<IoResult, ApexError> {
        IoReader::query_streaming(table, where_clause, limit)
    }
    
    /// Execute query returning row-based format (best for small results)
    #[inline]
    pub fn query_direct(
        table: &mut ColumnTable,
        where_clause: &str,
    ) -> Result<IoResult, ApexError> {
        IoReader::query_direct(table, where_clause)
    }
    
    /// Retrieve all rows (always uses Arrow for efficiency)
    #[inline]
    pub fn retrieve_all(table: &mut ColumnTable) -> Result<IoResult, ApexError> {
        IoReader::retrieve_all(table)
    }
    
    /// Retrieve single row by ID
    #[inline]
    pub fn retrieve_by_id(
        table: &mut ColumnTable,
        id: u64,
    ) -> Result<Option<HashMap<String, Value>>, ApexError> {
        IoReader::retrieve_by_id(table, id)
    }
    
    /// Retrieve multiple rows by IDs
    #[inline]
    pub fn retrieve_many(
        table: &mut ColumnTable,
        ids: &[u64],
    ) -> Result<IoResult, ApexError> {
        IoReader::retrieve_many(table, ids)
    }
    
    // ========== SQL EXECUTION (delegated to IoReader) ==========
    
    /// Execute a full SQL statement with automatic strategy selection
    #[inline]
    pub fn execute_sql(table: &mut ColumnTable, sql: &str) -> Result<SqlResult, ApexError> {
        let sql_upper = sql.to_uppercase();
        let hints = extract_sql_hints(&sql_upper);
        let strategy = Self::select_strategy(&hints);
        IoReader::execute_sql(table, sql, strategy)
    }
    
    /// Execute SQL and return Arrow RecordBatch directly
    #[inline]
    pub fn execute_sql_arrow(table: &mut ColumnTable, sql: &str) -> Result<RecordBatch, ApexError> {
        IoReader::execute_sql_arrow(table, sql)
    }
    
    // ========== WRITE OPERATIONS (delegated to IoWriter) ==========
    
    /// Insert a single row
    #[inline]
    pub fn insert(
        table: &mut ColumnTable,
        row: &HashMap<String, Value>,
        hints: WriteHints,
    ) -> Result<WriteResult, ApexError> {
        IoWriter::insert(table, row, hints)
    }
    
    /// Insert multiple rows
    #[inline]
    pub fn insert_many(
        table: &mut ColumnTable,
        rows: &[HashMap<String, Value>],
        hints: WriteHints,
    ) -> Result<WriteResult, ApexError> {
        IoWriter::insert_many(table, rows, hints)
    }
    
    /// Update a row by ID
    #[inline]
    pub fn update_by_id(
        table: &mut ColumnTable,
        id: u64,
        updates: &HashMap<String, Value>,
        hints: WriteHints,
    ) -> Result<WriteResult, ApexError> {
        IoWriter::update_by_id(table, id, updates, hints)
    }
    
    /// Update rows matching a WHERE clause
    #[inline]
    pub fn update_where(
        table: &mut ColumnTable,
        where_clause: &str,
        updates: &HashMap<String, Value>,
        hints: WriteHints,
    ) -> Result<WriteResult, ApexError> {
        IoWriter::update_where(table, where_clause, updates, hints)
    }
    
    /// Delete a row by ID
    #[inline]
    pub fn delete_by_id(
        table: &mut ColumnTable,
        id: u64,
        hints: WriteHints,
    ) -> Result<WriteResult, ApexError> {
        IoWriter::delete_by_id(table, id, hints)
    }
    
    /// Delete rows matching a WHERE clause
    #[inline]
    pub fn delete_where(
        table: &mut ColumnTable,
        where_clause: &str,
        hints: WriteHints,
    ) -> Result<WriteResult, ApexError> {
        IoWriter::delete_where(table, where_clause, hints)
    }
    
    // ========== INTERNAL METHODS (delegated to IoReader) ==========
    
    /// Read filtered rows with streaming early termination
    #[inline]
    pub(crate) fn read_filtered_indices(
        table: &ColumnTable,
        filter: &Filter,
        limit: Option<usize>,
        offset: usize,
    ) -> Vec<usize> {
        IoReader::read_filtered_indices(table, filter, limit, offset)
    }
    
    /// Read all non-deleted row indices
    #[inline]
    pub(crate) fn read_all_indices(
        table: &ColumnTable,
        limit: Option<usize>,
        offset: usize,
    ) -> Vec<usize> {
        IoReader::read_all_indices(table, limit, offset)
    }
    
    /// Build row values from column indices
    #[inline]
    pub(crate) fn build_row_values(
        table: &ColumnTable,
        row_idx: usize,
        column_indices: &[(String, Option<usize>)],
    ) -> Vec<Value> {
        IoReader::build_row_values(table, row_idx, column_indices)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_strategy_selection() {
        // Streaming for LIMIT without ORDER BY
        let hints = QueryHints::new().with_limit(100);
        assert_eq!(IoEngine::select_strategy(&hints), IoStrategy::Streaming);
        
        // Direct for aggregates
        let hints = QueryHints::new().with_aggregates();
        assert_eq!(IoEngine::select_strategy(&hints), IoStrategy::Direct);
        
        // Direct for small estimated results
        let hints = QueryHints::new().with_estimated_rows(50);
        assert_eq!(IoEngine::select_strategy(&hints), IoStrategy::Direct);
        
        // Arrow for large estimated results
        let hints = QueryHints::new().with_estimated_rows(10000);
        assert_eq!(IoEngine::select_strategy(&hints), IoStrategy::Arrow);
        
        // Force override
        let hints = QueryHints::new().with_estimated_rows(10).force(IoStrategy::Arrow);
        assert_eq!(IoEngine::select_strategy(&hints), IoStrategy::Arrow);
    }
}
