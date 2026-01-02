//! Write operations for the I/O Engine
//!
//! This module contains all write-related operations including insert,
//! update, and delete operations.

use crate::table::ColumnTable;
use crate::data::Value;
use crate::ApexError;
use std::collections::HashMap;

use super::types::{WriteHints, WriteStrategy, WriteResult};

/// Write operations implementation
pub struct IoWriter;

impl IoWriter {
    /// Select the optimal write strategy based on hints
    pub fn select_write_strategy(hints: &WriteHints) -> WriteStrategy {
        // Force strategy if specified
        if let Some(strategy) = hints.force_strategy {
            return strategy;
        }
        
        // Transactional for atomic operations
        if hints.atomic {
            return WriteStrategy::Transactional;
        }
        
        // Batch for large row counts
        if let Some(count) = hints.row_count {
            if count > 100 {
                return WriteStrategy::Batch;
            }
        }
        
        // Default to Single
        WriteStrategy::Single
    }
    
    /// Insert a single row
    pub fn insert(
        table: &mut ColumnTable,
        row: &HashMap<String, Value>,
        _hints: WriteHints,
    ) -> Result<WriteResult, ApexError> {
        // Single strategy: direct insert
        let id = table.insert(row)?;
        Ok(WriteResult {
            rows_affected: 1,
            inserted_ids: Some(vec![id]),
            first_id: Some(id),
        })
    }
    
    /// Insert multiple rows (auto-selects batch strategy for large inserts)
    pub fn insert_many(
        table: &mut ColumnTable,
        rows: &[HashMap<String, Value>],
        hints: WriteHints,
    ) -> Result<WriteResult, ApexError> {
        let strategy = Self::select_write_strategy(&hints);
        let count = rows.len();
        
        match strategy {
            WriteStrategy::Single => {
                // Insert one by one
                let mut ids = Vec::with_capacity(count);
                for row in rows {
                    let id = table.insert(row)?;
                    ids.push(id);
                }
                let first = ids.first().copied();
                Ok(WriteResult {
                    rows_affected: count,
                    inserted_ids: Some(ids),
                    first_id: first,
                })
            }
            WriteStrategy::Batch | WriteStrategy::Transactional => {
                // Batch insert
                let ids = table.insert_batch(rows)?;
                let first = ids.first().copied();
                Ok(WriteResult {
                    rows_affected: count,
                    inserted_ids: Some(ids),
                    first_id: first,
                })
            }
        }
    }
    
    /// Update a row by ID
    pub fn update_by_id(
        table: &mut ColumnTable,
        id: u64,
        updates: &HashMap<String, Value>,
        _hints: WriteHints,
    ) -> Result<WriteResult, ApexError> {
        let success = table.update(id, updates);
        Ok(WriteResult {
            rows_affected: if success { 1 } else { 0 },
            inserted_ids: None,
            first_id: None,
        })
    }
    
    /// Update rows matching a WHERE clause
    pub fn update_where(
        table: &mut ColumnTable,
        where_clause: &str,
        updates: &HashMap<String, Value>,
        _hints: WriteHints,
    ) -> Result<WriteResult, ApexError> {
        // First query matching IDs, then update each
        let rows = table.query(where_clause)?;
        let mut count = 0;
        for row in &rows {
            if let Some(Value::Int64(id)) = row.get("_id") {
                if table.update(*id as u64, updates) {
                    count += 1;
                }
            }
        }
        Ok(WriteResult {
            rows_affected: count,
            inserted_ids: None,
            first_id: None,
        })
    }
    
    /// Delete a row by ID
    pub fn delete_by_id(
        table: &mut ColumnTable,
        id: u64,
        _hints: WriteHints,
    ) -> Result<WriteResult, ApexError> {
        let success = table.delete(id);
        Ok(WriteResult {
            rows_affected: if success { 1 } else { 0 },
            inserted_ids: None,
            first_id: None,
        })
    }
    
    /// Delete rows matching a WHERE clause
    pub fn delete_where(
        table: &mut ColumnTable,
        where_clause: &str,
        _hints: WriteHints,
    ) -> Result<WriteResult, ApexError> {
        // First query matching IDs, then delete each
        let rows = table.query(where_clause)?;
        let ids: Vec<u64> = rows.iter()
            .filter_map(|row| row.get("_id"))
            .filter_map(|v| v.as_i64())
            .map(|id| id as u64)
            .collect();
        
        let success = table.delete_batch(&ids);
        Ok(WriteResult {
            rows_affected: if success { ids.len() } else { 0 },
            inserted_ids: None,
            first_id: None,
        })
    }
}
