//! Type definitions for the I/O Engine
//!
//! This module contains all the type definitions used by the I/O engine,
//! including query hints, strategies, and write results.

/// Query hints for strategy selection
#[derive(Debug, Clone, Default)]
pub struct QueryHints {
    /// Expected result size (if known)
    pub estimated_rows: Option<usize>,
    /// Whether query has LIMIT clause
    pub has_limit: bool,
    /// LIMIT value if present
    pub limit_value: Option<usize>,
    /// Whether query has ORDER BY
    pub has_order_by: bool,
    /// Whether query has aggregates (COUNT, SUM, etc.)
    pub has_aggregates: bool,
    /// Whether query has GROUP BY
    pub has_group_by: bool,
    /// Force a specific strategy
    pub force_strategy: Option<IoStrategy>,
}

impl QueryHints {
    pub fn new() -> Self {
        Self::default()
    }
    
    pub fn with_limit(mut self, limit: usize) -> Self {
        self.has_limit = true;
        self.limit_value = Some(limit);
        self
    }
    
    pub fn with_order_by(mut self) -> Self {
        self.has_order_by = true;
        self
    }
    
    pub fn with_aggregates(mut self) -> Self {
        self.has_aggregates = true;
        self
    }
    
    pub fn with_estimated_rows(mut self, rows: usize) -> Self {
        self.estimated_rows = Some(rows);
        self
    }
    
    pub fn force(mut self, strategy: IoStrategy) -> Self {
        self.force_strategy = Some(strategy);
        self
    }
}

/// Read I/O Strategy enum
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IoStrategy {
    /// Arrow columnar format - best for large results
    Arrow,
    /// Streaming with early termination - best for LIMIT queries
    Streaming,
    /// Direct row-based - best for small results and aggregates
    Direct,
}

/// Write I/O Strategy enum
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WriteStrategy {
    /// Single row operation - best for interactive use
    Single,
    /// Batch operation - best for bulk loads (>100 rows)
    Batch,
    /// Transactional - atomic multi-row with rollback
    Transactional,
}

/// Write hints for strategy selection
#[derive(Debug, Clone, Default)]
pub struct WriteHints {
    /// Number of rows to write
    pub row_count: Option<usize>,
    /// Whether operation requires atomicity
    pub atomic: bool,
    /// Force a specific strategy
    pub force_strategy: Option<WriteStrategy>,
}

impl WriteHints {
    pub fn new() -> Self {
        Self::default()
    }
    
    pub fn with_row_count(mut self, count: usize) -> Self {
        self.row_count = Some(count);
        self
    }
    
    pub fn atomic(mut self) -> Self {
        self.atomic = true;
        self
    }
    
    pub fn force(mut self, strategy: WriteStrategy) -> Self {
        self.force_strategy = Some(strategy);
        self
    }
}

/// Write operation result
#[derive(Debug, Clone)]
pub struct WriteResult {
    /// Number of rows affected
    pub rows_affected: usize,
    /// IDs of inserted rows (for insert operations)
    pub inserted_ids: Option<Vec<u64>>,
    /// First inserted ID (for single insert)
    pub first_id: Option<u64>,
}
