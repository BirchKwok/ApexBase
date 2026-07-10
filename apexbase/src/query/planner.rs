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

use std::collections::HashMap;
use std::hash::{Hash, Hasher};

use once_cell::sync::Lazy;
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};

use crate::query::sql_parser::BinaryOperator;
use crate::query::{SelectColumn, SelectStatement, SqlExpr, SqlStatement};
use crate::data::Value;
use crate::storage::index::IndexManager;

// ============================================================================
// Table Statistics Cache (for CBO)
// ============================================================================

/// Per-column statistics collected by ANALYZE
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ColumnStats {
    /// Number of distinct values
    pub ndv: u64,
    /// Number of null values
    pub null_count: u64,
    /// Min value (as string for universal comparison)
    pub min_value: String,
    /// Max value (as string for universal comparison)
    pub max_value: String,
    /// Typed numeric bounds used by range selectivity estimation.
    #[serde(default)]
    pub numeric_min: Option<f64>,
    #[serde(default)]
    pub numeric_max: Option<f64>,
    /// Optional equi-width histogram for numeric predicates.
    #[serde(default)]
    pub histogram: Vec<HistogramBucket>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HistogramBucket {
    pub lower: f64,
    pub upper: f64,
    pub row_count: u64,
}

/// Per-table statistics collected by ANALYZE
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TableStats {
    /// Total row count
    pub row_count: u64,
    /// Per-column statistics: column_name → stats
    pub columns: HashMap<String, ColumnStats>,
    /// Timestamp when stats were collected (epoch millis)
    pub collected_at: u64,
}

/// Global stats cache: table_path → TableStats
static STATS_CACHE: Lazy<RwLock<HashMap<String, TableStats>>> =
    Lazy::new(|| RwLock::new(HashMap::new()));

#[derive(Debug, Clone)]
struct PlanFeedback {
    strategy: ExecutionStrategy,
    estimated_rows: f64,
    actual_rows: f64,
    samples: u64,
}

static PLAN_FEEDBACK: Lazy<RwLock<HashMap<u64, PlanFeedback>>> =
    Lazy::new(|| RwLock::new(HashMap::new()));

/// Store ANALYZE results into the stats cache
pub fn store_table_stats(table_key: &str, stats: TableStats) {
    if let Ok(data) = bincode::serialize(&stats) {
        let _ = std::fs::write(stats_sidecar_path(table_key), data);
    }
    STATS_CACHE.write().insert(table_key.to_string(), stats);
}

/// Retrieve cached stats for a table
pub fn get_table_stats(table_key: &str) -> Option<TableStats> {
    if let Some(stats) = STATS_CACHE.read().get(table_key).cloned() {
        return Some(stats);
    }

    let sidecar = stats_sidecar_path(table_key);
    let data = std::fs::read(sidecar).ok()?;
    let stats: TableStats = bincode::deserialize(&data).ok()?;
    if !stats_are_fresh(table_key, &stats) {
        return None;
    }
    STATS_CACHE
        .write()
        .insert(table_key.to_string(), stats.clone());
    Some(stats)
}

/// Invalidate stats for a table (e.g., after DML)
pub fn invalidate_table_stats(table_key: &str) {
    STATS_CACHE.write().remove(table_key);
}

fn feedback_key(table_key: &str, select: &SelectStatement) -> u64 {
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    table_key.hash(&mut hasher);
    format!("{:?}", select).hash(&mut hasher);
    hasher.finish()
}

/// Record runtime cardinality feedback for EXPLAIN ANALYZE and future
/// executions of the same normalized AST shape.
pub fn record_plan_feedback(
    table_key: &str,
    select: &SelectStatement,
    strategy: &ExecutionStrategy,
    estimated_rows: f64,
    actual_rows: f64,
) {
    let key = feedback_key(table_key, select);
    let mut cache = PLAN_FEEDBACK.write();
    let entry = cache.entry(key).or_insert_with(|| PlanFeedback {
        strategy: strategy.clone(),
        estimated_rows,
        actual_rows,
        samples: 0,
    });
    entry.strategy = strategy.clone();
    entry.estimated_rows = (entry.estimated_rows * entry.samples as f64 + estimated_rows)
        / (entry.samples as f64 + 1.0);
    entry.actual_rows = (entry.actual_rows * entry.samples as f64 + actual_rows)
        / (entry.samples as f64 + 1.0);
    entry.samples = entry.samples.saturating_add(1);
}

fn stats_sidecar_path(table_key: &str) -> std::path::PathBuf {
    std::path::PathBuf::from(format!("{}.cbo_stats", table_key))
}

fn stats_are_fresh(table_key: &str, stats: &TableStats) -> bool {
    let Ok(modified) = std::fs::metadata(table_key).and_then(|meta| meta.modified()) else {
        return true;
    };
    let modified_ms = modified
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64;
    modified_ms <= stats.collected_at
}

// ============================================================================
// Cost Model
// ============================================================================

/// Cost of different operations (relative units)
const COST_SEQ_SCAN_PER_ROW: f64 = 1.0;
const COST_INDEX_LOOKUP: f64 = 4.0;
const COST_INDEX_SCAN_PER_ROW: f64 = 1.5;
const COST_INDEX_BASE_FETCH_PER_ROW: f64 = 1.0;
const COST_HASH_BUILD_PER_ROW: f64 = 2.0;
const COST_HASH_PROBE_PER_ROW: f64 = 0.5;
const COST_SORT_PER_ROW_LOG: f64 = 0.1;

/// Estimated cost of an execution plan
#[derive(Debug, Clone)]
pub struct PlanCost {
    /// Total estimated cost (lower is better)
    pub total: f64,
    /// Estimated output rows
    pub output_rows: f64,
    /// Estimated rows read from the storage layer.
    pub rows_read: f64,
}

impl PlanCost {
    fn seq_scan(row_count: f64) -> Self {
        Self {
            total: row_count * COST_SEQ_SCAN_PER_ROW,
            output_rows: row_count,
            rows_read: row_count,
        }
    }

    fn index_scan(row_count: f64, selectivity: f64) -> Self {
        let output = row_count * selectivity;
        Self {
            // Index lookup returns row ids first; materializing non-covering
            // rows has a separate random/scatter-read component.
            total: COST_INDEX_LOOKUP
                + output * (COST_INDEX_SCAN_PER_ROW + COST_INDEX_BASE_FETCH_PER_ROW),
            output_rows: output,
            rows_read: output,
        }
    }

    fn hash_join(left: &PlanCost, right: &PlanCost) -> Self {
        let build = left.output_rows * COST_HASH_BUILD_PER_ROW;
        let probe = right.output_rows * COST_HASH_PROBE_PER_ROW;
        Self {
            total: left.total + right.total + build + probe,
            output_rows: left.output_rows.min(right.output_rows),
            rows_read: left.rows_read + right.rows_read,
        }
    }
}

/// A physical candidate considered by the planner.
#[derive(Debug, Clone)]
pub struct PlanCandidate {
    pub name: String,
    pub strategy: ExecutionStrategy,
    pub cost: PlanCost,
}

/// The complete decision handed to the executor and EXPLAIN.
#[derive(Debug, Clone)]
pub struct QueryPlan {
    pub strategy: ExecutionStrategy,
    pub cost: PlanCost,
    pub candidates: Vec<PlanCandidate>,
    pub stats_available: bool,
    pub feedback_applied: bool,
}

/// Storage facts that affect physical plan legality and cost.
#[derive(Debug, Clone, Copy, Default)]
pub struct PlannerContext {
    pub mmap_only: bool,
}

// ============================================================================
// Selectivity Estimator
// ============================================================================

impl QueryPlanner {
    /// Estimate selectivity of a WHERE expression using table stats
    pub fn estimate_selectivity(expr: &SqlExpr, stats: &TableStats) -> f64 {
        match expr {
            SqlExpr::BinaryOp { left, op, right } => {
                match op {
                    BinaryOperator::Eq => {
                        let col = match (left.as_ref(), right.as_ref()) {
                            (SqlExpr::Column(col), _) => Some(col),
                            (_, SqlExpr::Column(col)) => Some(col),
                            _ => None,
                        };
                        if let Some(cs) = col.and_then(|column| stats.columns.get(column)) {
                            if cs.ndv > 0 {
                                return (1.0 / cs.ndv as f64).min(1.0);
                            }
                        }
                        0.01 // default for equality
                    }
                    BinaryOperator::Gt
                    | BinaryOperator::Ge
                    | BinaryOperator::Lt
                    | BinaryOperator::Le => {
                        let (column, literal) = match (left.as_ref(), right.as_ref()) {
                            (SqlExpr::Column(column), literal) => {
                                (Some(column), Self::literal_f64(literal))
                            }
                            (literal, SqlExpr::Column(column)) => {
                                (Some(column), Self::literal_f64(literal))
                            }
                            _ => (None, None),
                        };
                        if let (Some(column), Some(value)) = (column, literal) {
                            if let Some(cs) = stats.columns.get(column) {
                                return Self::estimate_range_selectivity(cs, op, value);
                            }
                        }
                        0.33
                    }
                    BinaryOperator::And => {
                        let s1 = Self::estimate_selectivity(left, stats);
                        let s2 = Self::estimate_selectivity(right, stats);
                        (s1 * s2).clamp(0.0, 1.0)
                    }
                    BinaryOperator::Or => {
                        let s1 = Self::estimate_selectivity(left, stats);
                        let s2 = Self::estimate_selectivity(right, stats);
                        (s1 + s2 - s1 * s2).min(1.0)
                    }
                    BinaryOperator::NotEq => {
                        if let SqlExpr::Column(col) = left.as_ref() {
                            if let Some(cs) = stats.columns.get(col) {
                                if cs.ndv > 0 {
                                    return 1.0 - 1.0 / cs.ndv as f64;
                                }
                            }
                        }
                        0.99
                    }
                    _ => 0.5,
                }
            }
            SqlExpr::Between {
                column,
                low,
                high,
                ..
            } => {
                let Some(cs) = stats.columns.get(column) else {
                    return 0.15;
                };
                let (Some(low), Some(high)) = (
                    Self::literal_f64(low),
                    Self::literal_f64(high),
                ) else {
                    return 0.15;
                };
                match (cs.numeric_min, cs.numeric_max) {
                    (Some(min), Some(max)) if max > min => {
                        let low_cdf = ((low - min) / (max - min)).clamp(0.0, 1.0);
                        let high_cdf = ((high - min) / (max - min)).clamp(0.0, 1.0);
                        (high_cdf - low_cdf).clamp(0.0, 1.0)
                    }
                    _ => 0.15,
                }
            }
            SqlExpr::In { column, values, .. } => {
                if let Some(cs) = stats.columns.get(column) {
                    if cs.ndv > 0 {
                        return (values.len() as f64 / cs.ndv as f64).min(1.0);
                    }
                }
                (values.len() as f64 * 0.01).min(0.5)
            }
            SqlExpr::Like { .. } => 0.1,
            SqlExpr::IsNull { negated, .. } => {
                if let SqlExpr::IsNull { column, .. } = expr {
                    if let Some(cs) = stats.columns.get(column) {
                        let null_fraction = if stats.row_count == 0 {
                            0.0
                        } else {
                            cs.null_count as f64 / stats.row_count as f64
                        };
                        return if *negated {
                            (1.0 - null_fraction).clamp(0.0, 1.0)
                        } else {
                            null_fraction.clamp(0.0, 1.0)
                        };
                    }
                }
                if *negated { 0.95 } else { 0.05 }
            }
            SqlExpr::UnaryOp {
                op: crate::query::sql_parser::UnaryOperator::Not,
                expr,
            } => 1.0 - Self::estimate_selectivity(expr, stats),
            _ => 0.5,
        }
    }

    fn literal_f64(expr: &SqlExpr) -> Option<f64> {
        match expr {
            SqlExpr::Literal(Value::Int64(value)) => Some(*value as f64),
            SqlExpr::Literal(Value::UInt64(value)) => Some(*value as f64),
            SqlExpr::Literal(Value::Float64(value)) => Some(*value),
            _ => None,
        }
    }

    fn estimate_range_selectivity(
        stats: &ColumnStats,
        op: &BinaryOperator,
        value: f64,
    ) -> f64 {
        let Some(min) = stats.numeric_min else {
            return 0.33;
        };
        let Some(max) = stats.numeric_max else {
            return 0.33;
        };
        if max <= min {
            return match op {
                BinaryOperator::Ge | BinaryOperator::Le => 1.0,
                BinaryOperator::Gt | BinaryOperator::Lt => 0.0,
                _ => 1.0,
            };
        }
        let cdf = |x: f64| {
            if stats.histogram.is_empty() {
                return ((x - min) / (max - min)).clamp(0.0, 1.0);
            }
            let total: u64 = stats.histogram.iter().map(|bucket| bucket.row_count).sum();
            if total == 0 {
                return ((x - min) / (max - min)).clamp(0.0, 1.0);
            }
            let mut before = 0u64;
            for bucket in &stats.histogram {
                if x >= bucket.upper {
                    before += bucket.row_count;
                    continue;
                }
                if x <= bucket.lower || bucket.upper <= bucket.lower {
                    break;
                }
                let fraction = ((x - bucket.lower) / (bucket.upper - bucket.lower))
                    .clamp(0.0, 1.0);
                return (before as f64 + fraction * bucket.row_count as f64) / total as f64;
            }
            before as f64 / total as f64
        };
        match op {
            BinaryOperator::Gt => 1.0 - cdf(value),
            BinaryOperator::Ge => 1.0 - cdf(value),
            BinaryOperator::Lt => cdf(value),
            BinaryOperator::Le => cdf(value),
            _ => 0.33,
        }
    }

    /// Determine whether to use an index or full scan based on cost
    pub fn should_use_index(col: &str, selectivity: f64, row_count: u64) -> bool {
        let scan_cost = PlanCost::seq_scan(row_count as f64);
        let index_cost = PlanCost::index_scan(row_count as f64, selectivity);
        index_cost.total < scan_cost.total
    }

    /// Optimize join order for a list of tables with estimated row counts
    /// Returns tables sorted by ascending row count (smallest table first = build side)
    pub fn optimize_join_order(tables: &[(String, u64)]) -> Vec<String> {
        let mut sorted = tables.to_vec();
        sorted.sort_by_key(|(_, rows)| *rows);
        sorted.into_iter().map(|(name, _)| name).collect()
    }
}

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
            SqlStatement::Select(select) => Self::plan_select(select, index_manager),
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
        Self::plan_select_with_stats(
            select,
            index_manager,
            None,
            PlannerContext::default(),
        )
        .strategy
    }

    /// Plan with CBO: use table stats for cost-based index/scan decisions
    pub fn plan_with_stats(
        stmt: &SqlStatement,
        index_manager: Option<&IndexManager>,
        table_key: &str,
    ) -> ExecutionStrategy {
        let stats = get_table_stats(table_key);
        match stmt {
            SqlStatement::Select(select) => {
                Self::plan_select_with_stats(
                    select,
                    index_manager,
                    stats.as_ref(),
                    PlannerContext::default(),
                )
                .strategy
            }
            _ => Self::plan(stmt, index_manager),
        }
    }

    /// Plan SELECT with CBO — takes &SelectStatement directly, avoiding a clone at the call site.
    pub fn plan_select_pub(
        select: &SelectStatement,
        index_manager: Option<&IndexManager>,
        table_key: &str,
    ) -> ExecutionStrategy {
        let stats = get_table_stats(table_key);
        Self::plan_select_with_stats(
            select,
            index_manager,
            stats.as_ref(),
            PlannerContext::default(),
        )
        .strategy
    }

    /// Plan a SELECT and retain candidate/cost information for execution and
    /// EXPLAIN.  Unlike the legacy strategy-only API, this compares all legal
    /// single-table access paths before choosing one.
    pub fn plan_select_details(
        select: &SelectStatement,
        index_manager: Option<&IndexManager>,
        table_key: &str,
        context: PlannerContext,
    ) -> QueryPlan {
        let stats = get_table_stats(table_key);
        let mut plan = Self::plan_select_with_stats(select, index_manager, stats.as_ref(), context);
        if !plan.candidates.is_empty() {
            if let Some(feedback) = PLAN_FEEDBACK.read().get(&feedback_key(table_key, select)) {
                if feedback.samples > 0 && feedback.estimated_rows > 0.0 {
                    let correction = (feedback.actual_rows / feedback.estimated_rows)
                        .clamp(0.25, 4.0);
                    for candidate in &mut plan.candidates {
                        if candidate.strategy == feedback.strategy {
                            candidate.cost.total *= correction;
                        }
                    }
                    if let Some(chosen) = plan.candidates.iter().min_by(|left, right| {
                        left.cost
                            .total
                            .partial_cmp(&right.cost.total)
                            .unwrap_or(std::cmp::Ordering::Equal)
                    }) {
                        plan.strategy = chosen.strategy.clone();
                        plan.cost = chosen.cost.clone();
                        plan.feedback_applied = true;
                    }
                }
            }
        }
        plan
    }

    /// Plan SELECT with cost-based optimization using ANALYZE stats
    fn plan_select_with_stats(
        select: &SelectStatement,
        index_manager: Option<&IndexManager>,
        stats: Option<&TableStats>,
        context: PlannerContext,
    ) -> QueryPlan {
        let chars = Self::analyze_select(select);

        let fixed = |strategy: ExecutionStrategy| QueryPlan {
            strategy,
            cost: PlanCost {
                total: 0.0,
                output_rows: 0.0,
                rows_read: 0.0,
            },
            candidates: Vec::new(),
            stats_available: stats.is_some(),
            feedback_applied: false,
        };

        if chars.is_write {
            return fixed(ExecutionStrategy::DirectWrite);
        }
        if chars.is_ddl {
            return fixed(ExecutionStrategy::Ddl);
        }
        if chars.has_aggregation || chars.has_group_by {
            return fixed(ExecutionStrategy::OlapAggregation);
        }
        if chars.has_join || chars.has_subquery {
            return fixed(ExecutionStrategy::OlapFullScan);
        }

        // Primary key lookup
        if chars.filters_on_pk {
            if let Some(id) = Self::extract_pk_value(&select.where_clause) {
                return fixed(ExecutionStrategy::OltpPrimaryKey { id_value: id });
            }
        }

        let row_count = stats.map(|s| s.row_count).unwrap_or(10_000).max(1) as f64;
        let selectivity = select
            .where_clause
            .as_ref()
            .map(|expr| {
                stats
                    .map(|s| Self::estimate_selectivity(expr, s))
                    .unwrap_or(chars.estimated_selectivity)
            })
            .unwrap_or(1.0)
            .clamp(0.0, 1.0);
        let output_rows = (row_count * selectivity).max(0.0);
        let scan_strategy = if select.where_clause.is_some()
            && (selectivity < 0.1 || chars.has_limit || context.mmap_only)
        {
            ExecutionStrategy::OlapFilteredScan
        } else {
            ExecutionStrategy::OlapFullScan
        };
        let mut scan_cost = PlanCost::seq_scan(row_count);
        scan_cost.output_rows = output_rows;
        let mut candidates = vec![PlanCandidate {
            name: if select.where_clause.is_some() {
                "sequential-filtered-scan".to_string()
            } else {
                "sequential-scan".to_string()
            },
            strategy: scan_strategy,
            cost: scan_cost,
        }];

        if !context.mmap_only {
            if let (Some(idx_mgr), Some(_where_expr)) = (index_manager, &select.where_clause) {
                for col in &chars.equality_filter_columns {
                    if !idx_mgr.has_usable_index_for_predicate(col, false) {
                        continue;
                    }
                    let col_selectivity = stats
                        .and_then(|s| s.columns.get(col))
                        .map(|cs| {
                            if cs.ndv > 0 {
                                (1.0 / cs.ndv as f64).min(1.0)
                            } else {
                                selectivity
                            }
                        })
                        .unwrap_or(selectivity);
                    let mut cost = PlanCost::index_scan(row_count, col_selectivity);
                    cost.output_rows = row_count * col_selectivity;
                    candidates.push(PlanCandidate {
                        name: format!("index-scan({})", col),
                        strategy: ExecutionStrategy::OltpIndexLookup {
                            column: col.clone(),
                            lookup_type: IndexLookupType::Equality,
                        },
                        cost,
                    });
                }
                for col in &chars.range_filter_columns {
                    if !idx_mgr.has_usable_index_for_predicate(col, true) {
                        continue;
                    }
                    // Range selectivity must use the range predicate estimate,
                    // not 1/NDV (which is only valid for equality).
                    let mut cost = PlanCost::index_scan(row_count, selectivity);
                    cost.output_rows = output_rows;
                    candidates.push(PlanCandidate {
                        name: format!("index-range-scan({})", col),
                        strategy: ExecutionStrategy::OltpIndexLookup {
                            column: col.clone(),
                            lookup_type: IndexLookupType::Range,
                        },
                        cost,
                    });
                }

                let equality_columns: Vec<String> = chars
                    .equality_filter_columns
                    .iter()
                    .cloned()
                    .collect::<std::collections::BTreeSet<_>>()
                    .into_iter()
                    .collect();
                if idx_mgr.has_composite_index(&equality_columns) {
                    let mut cost = PlanCost::index_scan(row_count, selectivity);
                    cost.output_rows = output_rows;
                    candidates.push(PlanCandidate {
                        name: format!("composite-index-scan({})", equality_columns.join(",")),
                        strategy: ExecutionStrategy::OltpIndexLookup {
                            column: equality_columns[0].clone(),
                            lookup_type: IndexLookupType::Equality,
                        },
                        cost,
                    });
                }
            }
        }

        candidates = candidates
            .into_iter()
            .map(|mut candidate| {
                if !select.order_by.is_empty() {
                    let rows = candidate.cost.output_rows.max(1.0);
                    candidate.cost.total += COST_SORT_PER_ROW_LOG * rows * rows.ln();
                }
                if let Some(limit) = select.limit {
                    let wanted = (limit + select.offset.unwrap_or(0)) as f64;
                    candidate.cost.output_rows = candidate.cost.output_rows.min(wanted);
                }
                candidate
            })
            .collect::<Vec<_>>();
        let chosen = candidates
            .iter()
            .min_by(|left, right| {
                left.cost
                    .total
                    .partial_cmp(&right.cost.total)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .cloned()
            .unwrap_or_else(|| candidates[0].clone());
        QueryPlan {
            strategy: chosen.strategy,
            cost: chosen.cost,
            candidates,
            stats_available: stats.is_some(),
            feedback_applied: false,
        }
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
                        let column = match (left.as_ref(), right.as_ref()) {
                            (SqlExpr::Column(column), _) | (_, SqlExpr::Column(column)) => {
                                Some(column)
                            }
                            _ => None,
                        };
                        if let Some(col) = column {
                            if col == "_id" {
                                chars.filters_on_pk = true;
                            }
                            chars.equality_filter_columns.push(col.clone());
                            chars.estimated_selectivity *= 0.01; // Very selective
                        }
                    }
                    BinaryOperator::Gt
                    | BinaryOperator::Ge
                    | BinaryOperator::Lt
                    | BinaryOperator::Le => {
                        let column = match (left.as_ref(), right.as_ref()) {
                            (SqlExpr::Column(column), _) | (_, SqlExpr::Column(column)) => {
                                Some(column)
                            }
                            _ => None,
                        };
                        if let Some(col) = column {
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
            SqlExpr::Between {
                column, low, high, ..
            } => {
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
        if let Some(SqlExpr::BinaryOp {
            left,
            op: BinaryOperator::Eq,
            right,
        }) = where_clause
        {
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
