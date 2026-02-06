//! Query Planner - Routes queries to OLTP or OLAP execution paths
//!
//! Analyzes SQL queries and selects the optimal execution strategy:
//! - **OLTP path**: Index-based point lookups, single-row mutations
//! - **OLAP path**: Vectorized columnar scans with SIMD/JIT
//!
//! Architecture:
//! ```text
//! ┌─────────────────┐
//! │   SQL Query      │
//! └────────┬────────┘
//!          │
//! ┌────────▼────────┐
//! │  QueryPlanner    │
//! │  - Analyze AST   │
//! │  - Check indexes │
//! │  - Estimate cost │
//! └────────┬────────┘
//!          │
//!   ┌──────┴──────────┐
//!   │                 │
//!   ▼                 ▼
//! ┌──────────┐  ┌──────────────┐
//! │ OLTP     │  │  OLAP        │
//! │ Executor │  │  Executor    │
//! │ (index)  │  │  (vectorized)│
//! └──────────┘  └──────────────┘
//! ```

use std::collections::HashSet;

use crate::query::{SqlStatement, SelectStatement, SqlExpr, SelectColumn, AggregateFunc};
use crate::query::sql_parser::BinaryOperator;
use crate::storage::index::IndexManager;
use crate::data::Value;

// ============================================================================
// Execution Strategy
// ============================================================================

/// The chosen execution strategy for a query
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ExecutionStrategy {
    /// Use OLTP path: index-based lookups
    OltpIndexLookup {
        /// Column to use for index lookup
        column: String,
        /// Type of lookup
        lookup_type: IndexLookupType,
    },
    /// Use OLTP path: primary key lookup (_id = X)
    OltpPrimaryKey {
        /// The _id value to look up
        id_value: i64,
    },
    /// Use OLAP path: full vectorized columnar scan
    OlapFullScan,
    /// Use OLAP path: vectorized scan with filter pushdown
    OlapFilteredScan,
    /// Use OLAP path: aggregation query
    OlapAggregation,
    /// Direct write (INSERT/UPDATE/DELETE)
    DirectWrite,
    /// DDL operation (CREATE/ALTER/DROP TABLE)
    Ddl,
}

/// Type of index lookup for OLTP path
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum IndexLookupType {
    /// Exact equality: col = value
    Equality,
    /// Range: col BETWEEN low AND high
    Range,
    /// IN list: col IN (v1, v2, ...)
    InList,
}

// ============================================================================
// Query Characteristics
// ============================================================================

/// Analyzed characteristics of a query
#[derive(Debug, Clone)]
pub struct QueryCharacteristics {
    /// Whether the query has aggregation functions
    pub has_aggregation: bool,
    /// Whether the query has GROUP BY
    pub has_group_by: bool,
    /// Whether the query has ORDER BY
    pub has_order_by: bool,
    /// Whether the query has JOIN
    pub has_join: bool,
    /// Whether the query has subqueries
    pub has_subquery: bool,
    /// Whether the query has LIMIT
    pub has_limit: bool,
    /// Whether the query filters on _id (primary key)
    pub filters_on_pk: bool,
    /// Columns used in WHERE clause equality conditions
    pub equality_filter_columns: Vec<String>,
    /// Columns used in WHERE clause range conditions
    pub range_filter_columns: Vec<String>,
    /// Estimated selectivity (0.0 = no rows, 1.0 = all rows)
    pub estimated_selectivity: f64,
    /// Whether this is a write operation
    pub is_write: bool,
    /// Whether this is a DDL operation
    pub is_ddl: bool,
}

impl Default for QueryCharacteristics {
    fn default() -> Self {
        Self {
            has_aggregation: false,
            has_group_by: false,
            has_order_by: false,
            has_join: false,
            has_subquery: false,
            has_limit: false,
            filters_on_pk: false,
            equality_filter_columns: Vec::new(),
            range_filter_columns: Vec::new(),
            estimated_selectivity: 1.0,
            is_write: false,
            is_ddl: false,
        }
    }
}

// ============================================================================
// Query Planner
// ============================================================================

/// Query planner that analyzes SQL and selects execution strategy
pub struct QueryPlanner;

impl QueryPlanner {
    /// Analyze a parsed SQL statement and determine the best execution strategy
    pub fn plan(stmt: &SqlStatement, index_manager: Option<&IndexManager>) -> ExecutionStrategy {
        match stmt {
            SqlStatement::Select(select) => {
                Self::plan_select(select, index_manager)
            }
            SqlStatement::Insert { .. } => ExecutionStrategy::DirectWrite,
            SqlStatement::Update { .. } => ExecutionStrategy::DirectWrite,
            SqlStatement::Delete { .. } => ExecutionStrategy::DirectWrite,
            SqlStatement::CreateTable { .. }
            | SqlStatement::DropTable { .. }
            | SqlStatement::AlterTable { .. }
            | SqlStatement::TruncateTable { .. } => ExecutionStrategy::Ddl,
            _ => ExecutionStrategy::OlapFullScan,
        }
    }

    /// Plan a SELECT query
    fn plan_select(
        select: &SelectStatement,
        index_manager: Option<&IndexManager>,
    ) -> ExecutionStrategy {
        let chars = Self::analyze_select(select);

        // DDL/write check
        if chars.is_write {
            return ExecutionStrategy::DirectWrite;
        }
        if chars.is_ddl {
            return ExecutionStrategy::Ddl;
        }

        // Aggregation queries always go OLAP
        if chars.has_aggregation || chars.has_group_by {
            return ExecutionStrategy::OlapAggregation;
        }

        // JOIN queries always go OLAP
        if chars.has_join || chars.has_subquery {
            return ExecutionStrategy::OlapFullScan;
        }

        // Primary key lookup: _id = X
        if chars.filters_on_pk {
            if let Some(id) = Self::extract_pk_value(&select.where_clause) {
                return ExecutionStrategy::OltpPrimaryKey { id_value: id };
            }
        }

        // Check if we have an index for any equality filter column
        if let Some(idx_mgr) = index_manager {
            for col in &chars.equality_filter_columns {
                if idx_mgr.has_index_on(col) {
                    return ExecutionStrategy::OltpIndexLookup {
                        column: col.clone(),
                        lookup_type: IndexLookupType::Equality,
                    };
                }
            }
            for col in &chars.range_filter_columns {
                if idx_mgr.has_index_on(col) {
                    return ExecutionStrategy::OltpIndexLookup {
                        column: col.clone(),
                        lookup_type: IndexLookupType::Range,
                    };
                }
            }
        }

        // High selectivity with LIMIT → OLAP with filter pushdown
        if chars.estimated_selectivity < 0.1 || chars.has_limit {
            return ExecutionStrategy::OlapFilteredScan;
        }

        // Default: OLAP full scan
        ExecutionStrategy::OlapFullScan
    }

    /// Analyze a SELECT statement to extract characteristics
    fn analyze_select(select: &SelectStatement) -> QueryCharacteristics {
        let mut chars = QueryCharacteristics::default();

        // Check for aggregation in select columns
        for col in &select.columns {
            match col {
                SelectColumn::Aggregate { .. } => {
                    chars.has_aggregation = true;
                }
                _ => {}
            }
        }

        // Check GROUP BY
        if !select.group_by.is_empty() {
            chars.has_group_by = true;
        }

        // Check ORDER BY
        if !select.order_by.is_empty() {
            chars.has_order_by = true;
        }

        // Check LIMIT
        if select.limit.is_some() {
            chars.has_limit = true;
        }

        // Check JOINs
        if !select.joins.is_empty() {
            chars.has_join = true;
        }

        // Analyze WHERE clause
        if let Some(ref where_expr) = select.where_clause {
            Self::analyze_where(where_expr, &mut chars);
        }

        chars
    }

    /// Analyze WHERE clause for index-friendly patterns
    fn analyze_where(expr: &SqlExpr, chars: &mut QueryCharacteristics) {
        match expr {
            SqlExpr::BinaryOp { left, op, right } => {
                match op {
                    BinaryOperator::Eq => {
                        if let SqlExpr::Column(col) = left.as_ref() {
                            if col == "_id" {
                                chars.filters_on_pk = true;
                            }
                            chars.equality_filter_columns.push(col.clone());
                            chars.estimated_selectivity *= 0.01; // Very selective
                        }
                    }
                    BinaryOperator::Gt | BinaryOperator::Ge
                    | BinaryOperator::Lt | BinaryOperator::Le => {
                        if let SqlExpr::Column(col) = left.as_ref() {
                            chars.range_filter_columns.push(col.clone());
                            chars.estimated_selectivity *= 0.3; // Moderately selective
                        }
                    }
                    BinaryOperator::And => {
                        Self::analyze_where(left, chars);
                        Self::analyze_where(right, chars);
                    }
                    BinaryOperator::Or => {
                        Self::analyze_where(left, chars);
                        Self::analyze_where(right, chars);
                        chars.estimated_selectivity = (chars.estimated_selectivity * 2.0).min(1.0);
                    }
                    _ => {}
                }
            }
            SqlExpr::Between { column, low, high, .. } => {
                chars.range_filter_columns.push(column.clone());
                chars.estimated_selectivity *= 0.2;
            }
            SqlExpr::In { column, values, .. } => {
                chars.equality_filter_columns.push(column.clone());
                chars.estimated_selectivity *= (values.len() as f64 * 0.01).min(0.5);
            }
            _ => {}
        }
    }

    /// Extract primary key value from WHERE _id = X
    fn extract_pk_value(where_clause: &Option<SqlExpr>) -> Option<i64> {
        if let Some(SqlExpr::BinaryOp { left, op: BinaryOperator::Eq, right }) = where_clause {
            if let SqlExpr::Column(col) = left.as_ref() {
                if col == "_id" {
                    if let SqlExpr::Literal(val) = right.as_ref() {
                        return val.as_i64();
                    }
                }
            }
        }
        None
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_strategy_display() {
        let strategy = ExecutionStrategy::OltpPrimaryKey { id_value: 42 };
        assert_eq!(strategy, ExecutionStrategy::OltpPrimaryKey { id_value: 42 });
    }

    #[test]
    fn test_query_characteristics_default() {
        let chars = QueryCharacteristics::default();
        assert!(!chars.has_aggregation);
        assert!(!chars.has_group_by);
        assert_eq!(chars.estimated_selectivity, 1.0);
    }
}
