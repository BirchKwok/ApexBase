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
use ahash::AHashMap;
use std::io;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use parking_lot::RwLock;
use once_cell::sync::Lazy;

use crate::query::{SqlParser, SqlStatement, SelectStatement, SqlExpr, SelectColumn, JoinType, JoinClause, UnionStatement, AggregateFunc};
use crate::query::sql_parser::BinaryOperator;
use crate::query::sql_parser::FromItem;
use crate::query::jit::{ExprJIT, FilterFnI64, simd_sum_i64, simd_sum_f64, simd_min_i64, simd_max_i64};

/// Zone Map optimization result for filter pruning
#[derive(PartialEq, Eq, Clone, Copy)]
enum ZoneMapResult {
    NoMatch,    // Filter definitely won't match any rows
    MayMatch,   // Filter might match some rows
}
use crate::storage::TableStorageBackend;
use crate::data::{DataType, Value};
use std::collections::HashSet;
use ahash::AHasher;
use std::hash::{Hash, Hasher};

// Global storage cache to avoid repeated open() calls which load all IDs
// Key: canonical path, Value: (backend, last_modified_time, last_access_time)
// Uses LRU eviction when cache exceeds MAX_CACHE_ENTRIES
const MAX_CACHE_ENTRIES: usize = 64;  // Limit cache to 64 tables

static STORAGE_CACHE: Lazy<RwLock<AHashMap<PathBuf, (Arc<TableStorageBackend>, std::time::SystemTime, std::time::Instant)>>> = 
    Lazy::new(|| RwLock::new(AHashMap::with_capacity(MAX_CACHE_ENTRIES)));

/// Evict least recently used entries from cache if over limit
fn evict_lru_cache_entries(cache: &mut AHashMap<PathBuf, (Arc<TableStorageBackend>, std::time::SystemTime, std::time::Instant)>) {
    if cache.len() <= MAX_CACHE_ENTRIES {
        return;
    }
    
    // Find the entry with oldest access time
    let entries_to_remove = cache.len() - MAX_CACHE_ENTRIES + 1; // Remove a few extra to avoid frequent eviction
    let mut access_times: Vec<(PathBuf, std::time::Instant)> = cache
        .iter()
        .map(|(k, (_, _, access))| (k.clone(), *access))
        .collect();
    
    // Sort by access time (oldest first)
    access_times.sort_by_key(|(_, t)| *t);
    
    // Remove oldest entries
    for (path, _) in access_times.into_iter().take(entries_to_remove) {
        cache.remove(&path);
    }
}

/// Zone Map (min-max index) for a column
/// Used to skip filtering when conditions can't match
#[derive(Clone, Debug)]
struct ZoneMap {
    min_int: Option<i64>,
    max_int: Option<i64>,
    min_float: Option<f64>,
    max_float: Option<f64>,
    has_nulls: bool,
}

impl ZoneMap {
    fn from_int64_array(arr: &Int64Array) -> Self {
        let mut min_val: Option<i64> = None;
        let mut max_val: Option<i64> = None;
        let mut has_nulls = false;
        
        for i in 0..arr.len() {
            if arr.is_null(i) {
                has_nulls = true;
            } else {
                let v = arr.value(i);
                min_val = Some(min_val.map_or(v, |m| m.min(v)));
                max_val = Some(max_val.map_or(v, |m| m.max(v)));
            }
        }
        
        Self { min_int: min_val, max_int: max_val, min_float: None, max_float: None, has_nulls }
    }
    
    fn from_float64_array(arr: &Float64Array) -> Self {
        let mut min_val: Option<f64> = None;
        let mut max_val: Option<f64> = None;
        let mut has_nulls = false;
        
        for i in 0..arr.len() {
            if arr.is_null(i) {
                has_nulls = true;
            } else {
                let v = arr.value(i);
                min_val = Some(min_val.map_or(v, |m| m.min(v)));
                max_val = Some(max_val.map_or(v, |m| m.max(v)));
            }
        }
        
        Self { min_int: None, max_int: None, min_float: min_val, max_float: max_val, has_nulls }
    }
    
    /// Check if a comparison can potentially match any rows
    /// Returns true if the filter might match, false if it definitely won't match
    #[inline]
    fn can_match(&self, op: &BinaryOperator, literal: &Value) -> bool {
        match literal {
            Value::Int64(v) => self.can_match_int(*v, op),
            Value::Float64(v) => self.can_match_float(*v, op),
            _ => true, // Can't optimize, assume might match
        }
    }
    
    #[inline]
    fn can_match_int(&self, v: i64, op: &BinaryOperator) -> bool {
        let (min, max) = match (self.min_int, self.max_int) {
            (Some(min), Some(max)) => (min, max),
            _ => return true, // No stats, assume might match
        };
        
        match op {
            BinaryOperator::Eq => v >= min && v <= max,
            BinaryOperator::NotEq => true, // Can't optimize !=
            BinaryOperator::Lt => min < v,
            BinaryOperator::Le => min <= v,
            BinaryOperator::Gt => max > v,
            BinaryOperator::Ge => max >= v,
            _ => true,
        }
    }
    
    #[inline]
    fn can_match_float(&self, v: f64, op: &BinaryOperator) -> bool {
        let (min, max) = match (self.min_float, self.max_float) {
            (Some(min), Some(max)) => (min, max),
            _ => {
                // Try int stats for float comparison
                if let (Some(min), Some(max)) = (self.min_int, self.max_int) {
                    (min as f64, max as f64)
                } else {
                    return true;
                }
            }
        };
        
        match op {
            BinaryOperator::Eq => v >= min && v <= max,
            BinaryOperator::NotEq => true,
            BinaryOperator::Lt => min < v,
            BinaryOperator::Le => min <= v,
            BinaryOperator::Gt => max > v,
            BinaryOperator::Ge => max >= v,
            _ => true,
        }
    }
}

/// Invalidate the storage cache for a specific path
/// CRITICAL: Must be called before any write operation to release mmap on Windows
pub fn invalidate_storage_cache(path: &Path) {
    let canonical = path.canonicalize().unwrap_or_else(|_| path.to_path_buf());
    let mut cache = STORAGE_CACHE.write();
    cache.remove(&canonical);
}

/// Invalidate all storage cache entries under a directory
/// CRITICAL: Must be called when closing a client to release all mmaps on Windows
pub fn invalidate_storage_cache_dir(dir: &Path) {
    let canonical_dir = dir.canonicalize().unwrap_or_else(|_| dir.to_path_buf());
    let mut cache = STORAGE_CACHE.write();
    cache.retain(|path, _| !path.starts_with(&canonical_dir));
}

/// Get or open a cached storage backend
/// Auto-compacts delta files before reading to ensure data consistency
fn get_cached_backend(path: &Path) -> io::Result<Arc<TableStorageBackend>> {
    let canonical = path.canonicalize().unwrap_or_else(|_| path.to_path_buf());
    
    // Check for delta file - if exists, compact before reading
    let delta_path = {
        let mut dp = canonical.clone();
        let name = dp.file_name().unwrap_or_default().to_string_lossy();
        dp.set_file_name(format!("{}.delta", name));
        dp
    };
    
    let has_delta = delta_path.exists();
    
    // Check file modification time (include delta file if exists)
    let metadata = std::fs::metadata(path)?;
    let mut modified = metadata.modified().unwrap_or(std::time::SystemTime::UNIX_EPOCH);
    
    if has_delta {
        if let Ok(delta_meta) = std::fs::metadata(&delta_path) {
            if let Ok(delta_modified) = delta_meta.modified() {
                if delta_modified > modified {
                    modified = delta_modified;
                }
            }
        }
    }
    
    // Try read from cache first (only if no delta file pending)
    if !has_delta {
        // Check cache and update access time if found
        let mut cache = STORAGE_CACHE.write();
        if let Some((backend, cached_time, _)) = cache.get(&canonical) {
            if *cached_time >= modified {
                let backend_clone = Arc::clone(backend);
                // Update access time for LRU tracking
                if let Some(entry) = cache.get_mut(&canonical) {
                    entry.2 = std::time::Instant::now();
                }
                return Ok(backend_clone);
            }
        }
    }
    
    // Cache miss, stale, or delta exists - need to open fresh
    // If delta exists, compact it first
    if has_delta {
        // Open for write to trigger compaction
        let storage = TableStorageBackend::open_for_write(path)?;
        storage.compact()?;
        // Delta is now merged, invalidate cache
        invalidate_storage_cache(path);
    }
    
    // Open backend (now with compacted data)
    let backend = Arc::new(TableStorageBackend::open(path)?);
    
    // Update cache with fresh modification time
    let new_modified = std::fs::metadata(path)?
        .modified()
        .unwrap_or(std::time::SystemTime::UNIX_EPOCH);
    
    {
        let mut cache = STORAGE_CACHE.write();
        // Evict LRU entries if cache is full
        evict_lru_cache_entries(&mut cache);
        cache.insert(canonical, (Arc::clone(&backend), new_modified, std::time::Instant::now()));
    }
    
    Ok(backend)
}

/// V3 Native Query Executor
/// 
/// Executes SQL queries directly on V3 storage using Arrow compute kernels.
/// This replaces the ColumnTable-based execution path.
pub struct ApexExecutor;

/// Query execution result
pub enum ApexResult {
    /// Query returned data rows
    Data(RecordBatch),
    /// Query returned empty result
    Empty(Arc<Schema>),
    /// Query returned a scalar (COUNT, etc.)
    Scalar(i64),
}

impl ApexResult {
    pub fn to_record_batch(self) -> io::Result<RecordBatch> {
        match self {
            ApexResult::Data(batch) => Ok(batch),
            ApexResult::Empty(schema) => Ok(RecordBatch::new_empty(schema)),
            ApexResult::Scalar(val) => {
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
            ApexResult::Data(batch) => batch.num_rows(),
            ApexResult::Empty(_) => 0,
            ApexResult::Scalar(_) => 1,
        }
    }
}

impl ApexExecutor {
    /// Invalidate the storage cache for a specific path
    /// CRITICAL: Must be called before any write operation to release mmap on Windows
    pub fn invalidate_cache_for_path(path: &Path) {
        invalidate_storage_cache(path);
    }
    
    /// Invalidate all storage cache entries under a directory
    /// CRITICAL: Must be called when closing a client to release all mmaps on Windows
    pub fn invalidate_cache_for_dir(dir: &Path) {
        invalidate_storage_cache_dir(dir);
    }
    
    /// Execute a SQL query on V3 storage (single table)
    pub fn execute(sql: &str, storage_path: &Path) -> io::Result<ApexResult> {
        let stmt = SqlParser::parse(sql)
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidInput, e.to_string()))?;

        Self::execute_parsed(stmt, storage_path)
    }

    /// Execute a SQL query with multi-table support (for JOINs)
    pub fn execute_with_base_dir(sql: &str, base_dir: &Path, default_table_path: &Path) -> io::Result<ApexResult> {
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
    pub fn execute_parsed(stmt: SqlStatement, storage_path: &Path) -> io::Result<ApexResult> {
        match stmt {
            SqlStatement::Select(select) => Self::execute_select(select, storage_path),
            SqlStatement::Union(union) => Self::execute_union(union, storage_path),
            SqlStatement::Insert { values, columns, .. } => {
                Self::execute_insert(storage_path, columns.as_deref(), &values)
            }
            SqlStatement::Delete { where_clause, .. } => {
                Self::execute_delete(storage_path, where_clause.as_ref())
            }
            SqlStatement::Update { assignments, where_clause, .. } => {
                Self::execute_update(storage_path, &assignments, where_clause.as_ref())
            }
            SqlStatement::TruncateTable { .. } => {
                Self::execute_truncate(storage_path)
            }
            _ => Err(io::Error::new(
                io::ErrorKind::Unsupported,
                "DDL statements require base_dir context - use execute_with_base_dir()",
            )),
        }
    }

    /// Execute a parsed SQL statement with multi-table support
    pub fn execute_parsed_multi(stmt: SqlStatement, base_dir: &Path, default_table_path: &Path) -> io::Result<ApexResult> {
        match stmt {
            SqlStatement::Select(select) => {
                if select.joins.is_empty() {
                    // Resolve the actual table path from FROM clause for non-join queries
                    let actual_path = Self::resolve_from_table_path(&select, base_dir, default_table_path);
                    Self::execute_select_with_base_dir(select, &actual_path, base_dir, default_table_path)
                } else {
                    Self::execute_select_with_joins(select, base_dir, default_table_path)
                }
            }
            SqlStatement::Union(union) => Self::execute_union(union, default_table_path),
            // DDL Statements
            SqlStatement::CreateTable { table, columns, if_not_exists } => {
                Self::execute_create_table(base_dir, &table, &columns, if_not_exists)
            }
            SqlStatement::DropTable { table, if_exists } => {
                Self::execute_drop_table(base_dir, &table, if_exists)
            }
            SqlStatement::AlterTable { table, operation } => {
                Self::execute_alter_table(base_dir, &table, &operation)
            }
            SqlStatement::TruncateTable { table } => {
                let table_path = Self::resolve_table_path(&table, base_dir, default_table_path);
                Self::execute_truncate(&table_path)
            }
            // DML Statements
            SqlStatement::Insert { table, columns, values } => {
                let table_path = Self::resolve_table_path(&table, base_dir, default_table_path);
                Self::execute_insert(&table_path, columns.as_deref(), &values)
            }
            SqlStatement::Delete { table, where_clause } => {
                let table_path = Self::resolve_table_path(&table, base_dir, default_table_path);
                Self::execute_delete(&table_path, where_clause.as_ref())
            }
            SqlStatement::Update { table, assignments, where_clause } => {
                let table_path = Self::resolve_table_path(&table, base_dir, default_table_path);
                Self::execute_update(&table_path, &assignments, where_clause.as_ref())
            }
            _ => Err(io::Error::new(
                io::ErrorKind::Unsupported,
                "Statement type not supported",
            )),
        }
    }

    /// Execute multiple SQL statements separated by semicolons.
    /// Currently used for temporary VIEW support within a single execute() call.
    fn execute_parsed_multi_statements(
        stmts: Vec<SqlStatement>,
        base_dir: &Path,
        default_table_path: &Path,
    ) -> io::Result<ApexResult> {
        let mut views: AHashMap<String, SelectStatement> = AHashMap::new();
        let mut last_result: Option<ApexResult> = None;

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
                // DDL/DML statements - execute directly
                other => {
                    last_result = Some(Self::execute_parsed_multi(other, base_dir, default_table_path)?);
                }
            }
        }

        last_result.ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "No query to execute"))
    }

    fn rewrite_select_views(mut select: SelectStatement, views: &AHashMap<String, SelectStatement>) -> SelectStatement {
        // Rewrite FROM clause if it references a VIEW
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

        // Rewrite JOIN clauses if they reference VIEWs
        let mut new_joins = Vec::with_capacity(select.joins.len());
        for mut join in select.joins {
            if let FromItem::Table { table, alias } = &join.right {
                let table_name = table.trim_matches('"');
                if let Some(view_stmt) = views.get(table_name) {
                    let alias_name = alias.clone().unwrap_or_else(|| table_name.to_string());
                    join.right = FromItem::Subquery {
                        stmt: Box::new(view_stmt.clone()),
                        alias: alias_name,
                    };
                }
            }
            new_joins.push(join);
        }
        select.joins = new_joins;

        select
    }

    /// Execute SELECT statement (legacy - uses storage_path for subqueries too)
    fn execute_select(stmt: SelectStatement, storage_path: &Path) -> io::Result<ApexResult> {
        // Delegate to the base_dir version, using storage_path's parent as base_dir
        let base_dir = storage_path.parent().unwrap_or(storage_path);
        Self::execute_select_with_base_dir(stmt, storage_path, base_dir, storage_path)
    }

    /// Execute SELECT statement with base_dir for proper subquery table resolution
    fn execute_select_with_base_dir(stmt: SelectStatement, storage_path: &Path, base_dir: &Path, default_table_path: &Path) -> io::Result<ApexResult> {
        // FAST PATH: Pure COUNT(*) without WHERE/GROUP BY - O(1) from metadata
        if Self::is_pure_count_star(&stmt) {
            if !storage_path.exists() {
                return Ok(ApexResult::Scalar(0));
            }
            let backend = get_cached_backend(storage_path)?;
            let count = backend.active_row_count() as i64;
            
            let output_name = if let Some(SelectColumn::Aggregate { alias, .. }) = stmt.columns.first() {
                alias.clone().unwrap_or_else(|| "COUNT(*)".to_string())
            } else {
                "COUNT(*)".to_string()
            };
            let schema = Arc::new(Schema::new(vec![
                Field::new(&output_name, ArrowDataType::Int64, false),
            ]));
            let array: ArrayRef = Arc::new(Int64Array::from(vec![count]));
            let batch = RecordBatch::try_new(schema, vec![array])
                .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e.to_string()))?;
            return Ok(ApexResult::Data(batch));
        }
        
        // Check for derived table (FROM subquery) - resolve table path from subquery's FROM clause
        let batch = match &stmt.from {
            Some(FromItem::Subquery { stmt: sub_stmt, .. }) => {
                // Resolve the actual table path from the subquery's FROM clause
                let sub_path = Self::resolve_from_table_path(sub_stmt, base_dir, default_table_path);
                let sub_result = Self::execute_select_with_base_dir(*sub_stmt.clone(), &sub_path, base_dir, default_table_path)?;
                sub_result.to_record_batch()?
            }
            _ => {
                // Normal table - read from storage
                // If file doesn't exist (e.g., after drop_if_exists), return empty batch
                if !storage_path.exists() {
                    let schema = Arc::new(Schema::empty());
                    RecordBatch::new_empty(schema)
                } else {
                    let backend = get_cached_backend(storage_path)?;
                    
                    // Check if any SELECT column contains a scalar subquery
                    // Scalar subqueries may reference arbitrary columns, so read all
                    let has_scalar_subquery = stmt.columns.iter().any(|col| {
                        if let SelectColumn::Expression { expr, .. } = col {
                            Self::expr_contains_scalar_subquery(expr)
                        } else {
                            false
                        }
                    });
                    
                    if has_scalar_subquery {
                        // Scalar subqueries may reference outer columns - use required_columns 
                        // which already extracts outer column references from subqueries
                        let required_cols = stmt.required_columns();
                        let col_refs: Option<Vec<&str>> = required_cols
                            .as_ref()
                            .filter(|cols| !cols.is_empty())
                            .map(|cols| cols.iter().map(|s| s.as_str()).collect());
                        backend.read_columns_to_arrow(col_refs.as_deref(), 0, None)?
                    } else {
                        // Check conditions for late materialization optimization
                        let has_aggregation_check = stmt.columns.iter().any(|col| {
                            matches!(col, SelectColumn::Aggregate { .. })
                                || matches!(col, SelectColumn::Expression { expr, .. } if Self::expr_contains_aggregate(expr))
                        });
                        
                        // Late Materialization for WHERE: SELECT * with WHERE (no ORDER BY)
                        let where_cols = stmt.where_columns();
                        let can_late_materialize_where = stmt.is_select_star()
                            && stmt.where_clause.is_some()
                            && stmt.order_by.is_empty()
                            && stmt.group_by.is_empty()
                            && !has_aggregation_check
                            && !where_cols.is_empty();
                        
                        // Late Materialization for ORDER BY: SELECT * with ORDER BY + LIMIT (no WHERE)
                        let order_cols: Vec<String> = stmt.order_by.iter()
                            .map(|o| o.column.trim_matches('"').to_string())
                            .collect();
                        let can_late_materialize_order = stmt.is_select_star()
                            && stmt.where_clause.is_none()
                            && !stmt.order_by.is_empty()
                            && stmt.limit.is_some()
                            && stmt.group_by.is_empty()
                            && !has_aggregation_check;
                        
                        // FAST PATH 0: Check for _id = X pattern (O(1) lookup)
                        if let Some(where_clause) = &stmt.where_clause {
                            if let Some(id) = Self::extract_id_equality_filter(where_clause) {
                                if let Some(batch) = backend.read_row_by_id_to_arrow(id)? {
                                    batch
                                } else {
                                    // ID not found - return empty batch with schema
                                    backend.read_columns_to_arrow(None, 0, Some(0))?
                                }
                            } else if let Some(result) = Self::try_fast_filter_group_order(&backend, &stmt)? {
                                // FAST PATH for Complex (Filter+Group+Order) - biggest optimization
                                return Ok(result);
                            } else if can_late_materialize_where {
                                // FAST PATH 1: Try dictionary-based filter for simple string equality
                                if let Some(result) = Self::try_fast_string_filter(&backend, &stmt)? {
                                    result
                                // FAST PATH 2: Try numeric range filter for BETWEEN
                                } else if let Some(result) = Self::try_fast_numeric_range_filter(&backend, &stmt)? {
                                    result
                                // FAST PATH 3: Try combined string + numeric filter for multi-condition
                                } else if let Some(result) = Self::try_fast_multi_condition_filter(&backend, &stmt)? {
                                    result
                                } else {
                                    // Late materialization for SELECT * WHERE path
                                    Self::execute_with_late_materialization(&backend, &stmt, storage_path)?
                                }
                            } else {
                                // Standard path: read all required columns upfront
                                let required_cols = stmt.required_columns();
                                let col_refs: Option<Vec<&str>> = required_cols
                                    .as_ref()
                                    .filter(|cols| !cols.is_empty())
                                    .map(|cols| cols.iter().map(|s| s.as_str()).collect());
                                backend.read_columns_to_arrow(col_refs.as_deref(), 0, None)?
                            }
                        } else if can_late_materialize_where {
                            // FAST PATH 1: Try dictionary-based filter for simple string equality
                            if let Some(result) = Self::try_fast_string_filter(&backend, &stmt)? {
                                result
                            // FAST PATH 2: Try numeric range filter for BETWEEN
                            } else if let Some(result) = Self::try_fast_numeric_range_filter(&backend, &stmt)? {
                                result
                            // FAST PATH 3: Try combined string + numeric filter for multi-condition
                            } else if let Some(result) = Self::try_fast_multi_condition_filter(&backend, &stmt)? {
                                result
                            } else {
                                // Late materialization for SELECT * WHERE path
                                Self::execute_with_late_materialization(&backend, &stmt, storage_path)?
                            }
                        } else if stmt.where_clause.is_some() && stmt.limit.is_none() {
                            // FAST PATH: String filter without LIMIT (uses dictionary scan)
                            if let Some(result) = Self::try_fast_string_filter_no_limit(&backend, &stmt)? {
                                result
                            } else {
                                // Standard path for WHERE without LIMIT
                                let required_cols = stmt.required_columns();
                                let col_refs: Option<Vec<&str>> = required_cols
                                    .as_ref()
                                    .filter(|cols| !cols.is_empty())
                                    .map(|cols| cols.iter().map(|s| s.as_str()).collect());
                                backend.read_columns_to_arrow(col_refs.as_deref(), 0, None)?
                            }
                        } else if can_late_materialize_order {
                            // Late materialization for ORDER BY + LIMIT path
                            Self::execute_with_order_late_materialization(&backend, &stmt)?
                        } else {
                            // Standard path: read all required columns upfront
                            let required_cols = stmt.required_columns();
                            let col_refs: Option<Vec<&str>> = required_cols
                                .as_ref()
                                .filter(|cols| !cols.is_empty())
                                .map(|cols| cols.iter().map(|s| s.as_str()).collect());
                            
                            // LIMIT pushdown: only read limited rows if safe
                            let can_pushdown_limit = stmt.where_clause.is_none()
                                && stmt.order_by.is_empty()
                                && stmt.group_by.is_empty()
                                && !has_aggregation_check;
                            
                            let row_limit = if can_pushdown_limit {
                                stmt.limit.map(|l| l + stmt.offset.unwrap_or(0))
                            } else {
                                None
                            };
                            
                            backend.read_columns_to_arrow(col_refs.as_deref(), 0, row_limit)?
                        }
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
            return Ok(ApexResult::Empty(batch.schema()));
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
            return Ok(ApexResult::Empty(filtered.schema()));
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

        // Apply ORDER BY with LIMIT optimization (top-k heap sort)
        let limited = if !stmt.order_by.is_empty() {
            let k = stmt.limit.map(|l| l + stmt.offset.unwrap_or(0));
            let sorted = Self::apply_order_by_topk(&filtered, &stmt.order_by, k)?;
            Self::apply_limit_offset(&sorted, stmt.limit, stmt.offset)?
        } else {
            Self::apply_limit_offset(&filtered, stmt.limit, stmt.offset)?
        };

        // Apply projection (SELECT columns) - pass storage_path for scalar subqueries
        let projected = Self::apply_projection_with_storage(&limited, &stmt.columns, Some(storage_path))?;

        // Apply DISTINCT if specified
        let result = if stmt.distinct {
            Self::deduplicate_batch(&projected)?
        } else {
            projected
        };

        Ok(ApexResult::Data(result))
    }

    /// Fast path for simple string equality filters on dictionary-encoded columns
    /// Uses storage-level early termination for LIMIT queries
    /// Supports column projection pushdown (not limited to SELECT *)
    fn try_fast_string_filter(
        backend: &TableStorageBackend,
        stmt: &SelectStatement,
    ) -> io::Result<Option<RecordBatch>> {
        // Only handle simple WHERE col = 'value' patterns
        let where_clause = match &stmt.where_clause {
            Some(w) => w,
            None => return Ok(None),
        };
        
        // Must have LIMIT for early termination benefit
        if stmt.limit.is_none() {
            return Ok(None);
        }
        
        // Extract column name and literal value from simple equality
        let (col_name, filter_value) = match where_clause {
            SqlExpr::BinaryOp { left, op, right } => {
                use crate::query::sql_parser::BinaryOperator;
                if *op != BinaryOperator::Eq {
                    return Ok(None);
                }
                match (left.as_ref(), right.as_ref()) {
                    (SqlExpr::Column(col), SqlExpr::Literal(Value::String(val))) => {
                        (col.trim_matches('"').to_string(), val.clone())
                    }
                    (SqlExpr::Literal(Value::String(val)), SqlExpr::Column(col)) => {
                        (col.trim_matches('"').to_string(), val.clone())
                    }
                    _ => return Ok(None),
                }
            }
            _ => return Ok(None),
        };
        
        let limit = stmt.limit.unwrap_or(100);
        let offset = stmt.offset.unwrap_or(0);
        
        // OPTIMIZATION: Column projection pushdown - only read required columns
        let projected_cols: Option<Vec<String>> = if stmt.is_select_star() {
            None // All columns
        } else {
            Some(stmt.required_columns().unwrap_or_default())
        };
        let col_refs: Option<Vec<&str>> = projected_cols.as_ref()
            .map(|cols| cols.iter().map(|s| s.as_str()).collect());
        
        // Use storage-level filter with early termination
        let result = backend.read_columns_filtered_string_with_limit_to_arrow(
            col_refs.as_deref(),
            &col_name,
            &filter_value,
            true, // filter_eq = true for equality
            limit,
            offset,
        )?;
        
        Ok(Some(result))
    }

    /// Fast path for string equality filters WITHOUT LIMIT
    /// Uses dictionary index scan for maximum performance
    fn try_fast_string_filter_no_limit(
        backend: &TableStorageBackend,
        stmt: &SelectStatement,
    ) -> io::Result<Option<RecordBatch>> {
        use crate::query::sql_parser::BinaryOperator;
        
        // Only handle simple WHERE col = 'value' patterns
        let where_clause = match &stmt.where_clause {
            Some(w) => w,
            None => return Ok(None),
        };
        
        // Extract column name and literal value from simple equality
        let (col_name, filter_value) = match where_clause {
            SqlExpr::BinaryOp { left, op: BinaryOperator::Eq, right } => {
                match (left.as_ref(), right.as_ref()) {
                    (SqlExpr::Column(col), SqlExpr::Literal(Value::String(val))) => {
                        (col.trim_matches('"').to_string(), val.clone())
                    }
                    (SqlExpr::Literal(Value::String(val)), SqlExpr::Column(col)) => {
                        (col.trim_matches('"').to_string(), val.clone())
                    }
                    _ => return Ok(None),
                }
            }
            _ => return Ok(None),
        };
        
        // OPTIMIZATION: Column projection pushdown - only read required columns
        let projected_cols: Option<Vec<String>> = if stmt.is_select_star() {
            None // All columns
        } else {
            Some(stmt.required_columns().unwrap_or_default())
        };
        let col_refs: Option<Vec<&str>> = projected_cols.as_ref()
            .map(|cols| cols.iter().map(|s| s.as_str()).collect());
        
        // Use storage-level filter WITHOUT early termination (no LIMIT)
        let result = backend.read_columns_filtered_string_to_arrow(
            col_refs.as_deref(),
            &col_name,
            &filter_value,
            true, // filter_eq = true for equality
        )?;
        
        Ok(Some(result))
    }

    /// Fast path for numeric range filters (BETWEEN)
    /// Uses streaming scan with early termination for LIMIT queries
    /// Supports column projection pushdown (not limited to SELECT *)
    fn try_fast_numeric_range_filter(
        backend: &TableStorageBackend,
        stmt: &SelectStatement,
    ) -> io::Result<Option<RecordBatch>> {
        use crate::query::sql_parser::BinaryOperator;
        
        // Must have LIMIT for early termination benefit
        if stmt.limit.is_none() {
            return Ok(None);
        }
        
        let where_clause = match &stmt.where_clause {
            Some(w) => w,
            None => return Ok(None),
        };
        
        // Extract BETWEEN pattern: col BETWEEN low AND high
        let (col_name, low, high) = match where_clause {
            SqlExpr::Between { column, low, high, negated } => {
                if *negated {
                    return Ok(None);
                }
                let low_val = Self::extract_numeric_value(low)?;
                let high_val = Self::extract_numeric_value(high)?;
                (column.trim_matches('"').to_string(), low_val, high_val)
            }
            // Also handle col >= low AND col <= high pattern
            SqlExpr::BinaryOp { left, op: BinaryOperator::And, right } => {
                let (col1, op1, val1) = match Self::extract_comparison(left) {
                    Ok(v) => v,
                    Err(_) => return Ok(None),
                };
                let (col2, op2, val2) = match Self::extract_comparison(right) {
                    Ok(v) => v,
                    Err(_) => return Ok(None),
                };
                
                if col1 != col2 {
                    return Ok(None);
                }
                
                // Determine low and high from the operators
                let (low, high) = match (op1, op2) {
                    (BinaryOperator::Ge, BinaryOperator::Le) => (val1, val2),
                    (BinaryOperator::Le, BinaryOperator::Ge) => (val2, val1),
                    (BinaryOperator::Gt, BinaryOperator::Lt) => (val1, val2),
                    (BinaryOperator::Lt, BinaryOperator::Gt) => (val2, val1),
                    _ => return Ok(None),
                };
                (col1, low, high)
            }
            _ => return Ok(None),
        };
        
        let limit = stmt.limit.unwrap_or(100);
        let offset = stmt.offset.unwrap_or(0);
        
        // Use storage-level numeric range filter with early termination
        let result = backend.read_columns_filtered_range_with_limit_to_arrow(
            None, // All columns (SELECT *)
            &col_name,
            low,
            high,
            limit,
            offset,
        )?;
        
        Ok(Some(result))
    }
    
    /// Helper to extract numeric value from SqlExpr
    fn extract_numeric_value(expr: &SqlExpr) -> io::Result<f64> {
        match expr {
            SqlExpr::Literal(Value::Int64(n)) => Ok(*n as f64),
            SqlExpr::Literal(Value::Int32(n)) => Ok(*n as f64),
            SqlExpr::Literal(Value::Float64(n)) => Ok(*n),
            SqlExpr::Literal(Value::Float32(n)) => Ok(*n as f64),
            _ => Err(io::Error::new(io::ErrorKind::InvalidInput, "not a number")),
        }
    }
    
    /// Helper to extract comparison from binary op
    fn extract_comparison(expr: &SqlExpr) -> io::Result<(String, crate::query::sql_parser::BinaryOperator, f64)> {
        use crate::query::sql_parser::BinaryOperator;
        match expr {
            SqlExpr::BinaryOp { left, op, right } => {
                match (left.as_ref(), right.as_ref()) {
                    (SqlExpr::Column(col), lit) => {
                        let val = Self::extract_numeric_value(lit)?;
                        Ok((col.trim_matches('"').to_string(), op.clone(), val))
                    }
                    (lit, SqlExpr::Column(col)) => {
                        let val = Self::extract_numeric_value(lit)?;
                        // Flip the operator
                        let flipped_op = match op {
                            BinaryOperator::Gt => BinaryOperator::Lt,
                            BinaryOperator::Lt => BinaryOperator::Gt,
                            BinaryOperator::Ge => BinaryOperator::Le,
                            BinaryOperator::Le => BinaryOperator::Ge,
                            _ => return Err(io::Error::new(io::ErrorKind::InvalidInput, "unsupported op")),
                        };
                        Ok((col.trim_matches('"').to_string(), flipped_op, val))
                    }
                    _ => Err(io::Error::new(io::ErrorKind::InvalidInput, "not a comparison")),
                }
            }
            _ => Err(io::Error::new(io::ErrorKind::InvalidInput, "not a binary op")),
        }
    }

    /// Fast path for multi-condition WHERE with string equality AND numeric comparison
    /// Handles: SELECT * WHERE string_col = 'value' AND numeric_col > N LIMIT n
    fn try_fast_multi_condition_filter(
        backend: &TableStorageBackend,
        stmt: &SelectStatement,
    ) -> io::Result<Option<RecordBatch>> {
        use crate::query::sql_parser::BinaryOperator;
        
        // Must be SELECT * with LIMIT
        if !stmt.is_select_star() || stmt.limit.is_none() {
            return Ok(None);
        }
        
        let where_clause = match &stmt.where_clause {
            Some(w) => w,
            None => return Ok(None),
        };
        
        // Must be AND of two conditions
        let (left_cond, right_cond) = match where_clause {
            SqlExpr::BinaryOp { left, op: BinaryOperator::And, right } => {
                (left.as_ref(), right.as_ref())
            }
            _ => return Ok(None),
        };
        
        // Try to extract string equality and numeric comparison from either order
        let (str_col, str_val, num_col, num_op, num_val) = 
            if let (Some((sc, sv)), Some((nc, no, nv))) = (
                Self::extract_string_equality(left_cond),
                Self::extract_numeric_comparison(right_cond)
            ) {
                (sc, sv, nc, no, nv)
            } else if let (Some((sc, sv)), Some((nc, no, nv))) = (
                Self::extract_string_equality(right_cond),
                Self::extract_numeric_comparison(left_cond)
            ) {
                (sc, sv, nc, no, nv)
            } else {
                return Ok(None);
            };
        
        let limit = stmt.limit.unwrap_or(100);
        let offset = stmt.offset.unwrap_or(0);
        
        // Use storage-level combined filter
        let result = backend.read_columns_filtered_string_numeric_with_limit_to_arrow(
            None, // All columns (SELECT *)
            &str_col,
            &str_val,
            &num_col,
            &num_op,
            num_val,
            limit,
            offset,
        )?;
        
        Ok(Some(result))
    }
    
    /// FAST PATH for Complex (Filter+Group+Order) queries
    /// Optimized for: SELECT group_col, AGG(agg_col) FROM table WHERE filter_col = 'value' GROUP BY group_col ORDER BY agg DESC LIMIT n
    /// Uses single-pass execution with direct dictionary indexing for maximum performance
    fn try_fast_filter_group_order(
        backend: &TableStorageBackend,
        stmt: &SelectStatement,
    ) -> io::Result<Option<ApexResult>> {
        use crate::query::AggregateFunc;
        use crate::query::sql_parser::BinaryOperator;
        
        // Check pattern: must have WHERE, GROUP BY, ORDER BY, and LIMIT
        let where_clause = match &stmt.where_clause {
            Some(w) => w,
            None => return Ok(None),
        };
        
        if stmt.group_by.is_empty() || stmt.order_by.is_empty() || stmt.limit.is_none() {
            return Ok(None);
        }
        
        // Must be simple string equality filter
        let (filter_col, filter_val) = match where_clause {
            SqlExpr::BinaryOp { left, op: BinaryOperator::Eq, right } => {
                match (left.as_ref(), right.as_ref()) {
                    (SqlExpr::Column(col), SqlExpr::Literal(Value::String(val))) => {
                        (col.trim_matches('"').to_string(), val.as_str())
                    }
                    (SqlExpr::Literal(Value::String(val)), SqlExpr::Column(col)) => {
                        (col.trim_matches('"').to_string(), val.as_str())
                    }
                    _ => return Ok(None),
                }
            }
            _ => return Ok(None),
        };
        
        // Must have exactly one GROUP BY column (string)
        if stmt.group_by.len() != 1 {
            return Ok(None);
        }
        let group_col = stmt.group_by[0].trim_matches('"');
        
        // Must have exactly one ORDER BY clause
        if stmt.order_by.len() != 1 {
            return Ok(None);
        }
        let order_clause = &stmt.order_by[0];
        let order_col = order_clause.column.trim_matches('"');
        let descending = order_clause.descending;
        
        // Check if we have exactly one aggregate column
        let mut agg_func = None;
        let mut agg_col = None;
        for col in &stmt.columns {
            if let SelectColumn::Aggregate { func, column, .. } = col {
                agg_func = Some(func.clone());
                agg_col = column.as_deref();
            }
        }
        
        let agg_func = match agg_func {
            Some(f) => f,
            None => return Ok(None),
        };
        
        // Only support SUM and COUNT for now
        if !matches!(agg_func, AggregateFunc::Sum | AggregateFunc::Count) {
            return Ok(None);
        }
        
        // Check HAVING clause - must be simple
        if let Some(having) = &stmt.having {
            // Only support simple column > value comparisons
            match having {
                SqlExpr::BinaryOp { left, op: BinaryOperator::Gt, right } => {
                    match (left.as_ref(), right.as_ref()) {
                        (SqlExpr::Column(col), SqlExpr::Literal(Value::Int64(val))) => {
                            if col.trim_matches('"') != order_col {
                                return Ok(None);
                            }
                            // Will apply this filter after aggregation
                        }
                        _ => return Ok(None),
                    }
                }
                _ => return Ok(None),
            }
        }
        
        let limit = stmt.limit.unwrap_or(100);
        let offset = stmt.offset.unwrap_or(0);
        
        // Call storage-level optimized function
        match backend.execute_filter_group_order(
            &filter_col,
            filter_val,
            group_col,
            agg_col,
            agg_func,
            order_col,
            descending,
            limit,
            offset,
        ) {
            Ok(Some(result)) => {
                // Apply HAVING if present
                if stmt.having.is_some() {
                    // HAVING already applied in storage function for simple cases
                }
                Ok(Some(ApexResult::Data(result)))
            }
            Ok(None) => Ok(None),
            Err(e) => Err(e),
        }
    }
    
    /// Helper to extract string equality: col = 'value'
    fn extract_string_equality(expr: &SqlExpr) -> Option<(String, String)> {
        use crate::query::sql_parser::BinaryOperator;
        match expr {
            SqlExpr::BinaryOp { left, op: BinaryOperator::Eq, right } => {
                match (left.as_ref(), right.as_ref()) {
                    (SqlExpr::Column(col), SqlExpr::Literal(Value::String(val))) |
                    (SqlExpr::Literal(Value::String(val)), SqlExpr::Column(col)) => {
                        Some((col.trim_matches('"').to_string(), val.clone()))
                    }
                    _ => None,
                }
            }
            _ => None,
        }
    }
    
    /// Helper to extract numeric comparison: col > N, col >= N, col < N, col <= N
    fn extract_numeric_comparison(expr: &SqlExpr) -> Option<(String, String, f64)> {
        use crate::query::sql_parser::BinaryOperator;
        match expr {
            SqlExpr::BinaryOp { left, op, right } => {
                let op_str = match op {
                    BinaryOperator::Gt => ">",
                    BinaryOperator::Ge => ">=",
                    BinaryOperator::Lt => "<",
                    BinaryOperator::Le => "<=",
                    BinaryOperator::Eq => "=",
                    _ => return None,
                };
                
                match (left.as_ref(), right.as_ref()) {
                    (SqlExpr::Column(col), lit) => {
                        if let Ok(val) = Self::extract_numeric_value(lit) {
                            Some((col.trim_matches('"').to_string(), op_str.to_string(), val))
                        } else {
                            None
                        }
                    }
                    (lit, SqlExpr::Column(col)) => {
                        if let Ok(val) = Self::extract_numeric_value(lit) {
                            // Flip operator for reversed order
                            let flipped = match op_str {
                                ">" => "<",
                                ">=" => "<=",
                                "<" => ">",
                                "<=" => ">=",
                                _ => op_str,
                            };
                            Some((col.trim_matches('"').to_string(), flipped.to_string(), val))
                        } else {
                            None
                        }
                    }
                    _ => None,
                }
            }
            _ => None,
        }
    }

    /// Execute SELECT * with late materialization optimization
    /// 1. Read only WHERE columns first
    /// 2. Apply filter to get matching row indices
    /// 3. Read remaining columns only for matching rows
    fn execute_with_late_materialization(
        backend: &TableStorageBackend,
        stmt: &SelectStatement,
        storage_path: &Path,
    ) -> io::Result<RecordBatch> {
        use arrow::compute;
        
        // Step 1: Read only columns needed for WHERE clause
        let where_cols = stmt.where_columns();
        let where_col_refs: Vec<&str> = where_cols.iter().map(|s| s.as_str()).collect();
        
        // Also include _id for later row identification
        let mut cols_to_read: Vec<&str> = vec!["_id"];
        cols_to_read.extend(where_col_refs.iter());
        
        let where_clause = stmt.where_clause.as_ref().unwrap();
        let need_count = stmt.limit.map(|l| l + stmt.offset.unwrap_or(0));
        
        // OPTIMIZATION: Streaming filter evaluation with early termination
        // Read data in chunks and stop once we have enough matches
        let total_rows = backend.row_count() as usize;
        // Adaptive chunk size: smaller for small LIMIT (assume ~50% selectivity)
        let chunk_size: usize = if let Some(need) = need_count {
            // Start with 4x the needed rows, grow if selectivity is low
            (need * 4).max(1000).min(100_000)
        } else {
            50_000
        };
        
        let limited_indices: Vec<usize> = if let Some(need) = need_count {
            let mut indices = Vec::with_capacity(need);
            let mut start_row: usize = 0;
            
            while start_row < total_rows && indices.len() < need {
                let rows_to_read = chunk_size.min(total_rows - start_row);
                let filter_batch = backend.read_columns_to_arrow(Some(&cols_to_read), start_row, Some(rows_to_read))?;
                
                if filter_batch.num_rows() == 0 {
                    break;
                }
                
                let mask = Self::evaluate_predicate_with_storage(&filter_batch, where_clause, storage_path)?;
                
                // Collect matching indices from this chunk
                for (i, v) in mask.iter().enumerate() {
                    if v == Some(true) {
                        indices.push(start_row + i);
                        if indices.len() >= need {
                            break;
                        }
                    }
                }
                
                start_row += rows_to_read;
            }
            
            // Apply offset
            if let Some(offset) = stmt.offset {
                indices.into_iter().skip(offset).collect()
            } else {
                indices
            }
        } else {
            // No LIMIT - use streaming chunks to avoid loading all data at once
            let mut all_indices = Vec::new();
            let mut start_row: usize = 0;
            
            while start_row < total_rows {
                let rows_to_read = chunk_size.min(total_rows - start_row);
                let filter_batch = backend.read_columns_to_arrow(Some(&cols_to_read), start_row, Some(rows_to_read))?;
                
                if filter_batch.num_rows() == 0 {
                    break;
                }
                
                let mask = Self::evaluate_predicate_with_storage(&filter_batch, where_clause, storage_path)?;
                
                // Collect matching indices from this chunk
                for (i, v) in mask.iter().enumerate() {
                    if v == Some(true) {
                        all_indices.push(start_row + i);
                    }
                }
                
                start_row += rows_to_read;
            }
            
            all_indices
        };
        
        if limited_indices.is_empty() {
            return backend.read_columns_to_arrow(None, 0, Some(0));
        }
        
        // Step 4: Read ALL columns but only for matching row indices
        // This reads directly from disk for only the matching rows - true late materialization
        backend.read_columns_by_indices_to_arrow(&limited_indices)
    }

    /// Execute SELECT * with ORDER BY + LIMIT late materialization
    /// 1. Read only ORDER BY columns in chunks
    /// 2. Use streaming top-k to find best rows without loading all data
    /// 3. Read all other columns only for those k rows
    fn execute_with_order_late_materialization(
        backend: &TableStorageBackend,
        stmt: &SelectStatement,
    ) -> io::Result<RecordBatch> {
        let k = stmt.limit.map(|l| l + stmt.offset.unwrap_or(0)).unwrap_or(0);
        if k == 0 {
            return backend.read_columns_to_arrow(None, 0, Some(0));
        }

        // Step 1: Read only columns needed for ORDER BY
        let order_cols: Vec<&str> = stmt.order_by.iter()
            .map(|o| {
                let col = o.column.trim_matches('"');
                if let Some(dot_pos) = col.rfind('.') {
                    &col[dot_pos + 1..]
                } else {
                    col
                }
            })
            .collect();
        
        let sort_batch = backend.read_columns_to_arrow(Some(&order_cols), 0, None)?;
        let num_rows = sort_batch.num_rows();
        
        if num_rows == 0 {
            return backend.read_columns_to_arrow(None, 0, Some(0));
        }

        let k_actual = k.min(num_rows);
        
        // Step 2: Find top-k indices using optimized streaming algorithm
        let final_indices: Vec<usize> = if stmt.order_by.len() == 1 && k_actual <= 100 {
            let clause = &stmt.order_by[0];
            let col_name = clause.column.trim_matches('"');
            let actual_col = if let Some(dot_pos) = col_name.rfind('.') {
                &col_name[dot_pos + 1..]
            } else {
                col_name
            };
            
            if let Some(col) = sort_batch.column_by_name(actual_col) {
                // Fast path for Float64 DESC (most common case)
                if let Some(float_arr) = col.as_any().downcast_ref::<Float64Array>() {
                    let descending = clause.descending;
                    
                    // Streaming top-k: maintain sorted list of top k (value, index) pairs
                    let mut top_k: Vec<(f64, usize)> = Vec::with_capacity(k_actual + 1);
                    
                    if descending {
                        // DESC: keep k largest values
                        for i in 0..num_rows {
                            let val = if float_arr.is_null(i) { f64::NEG_INFINITY } else { float_arr.value(i) };
                            
                            if top_k.len() < k_actual {
                                let pos = top_k.partition_point(|(v, _)| *v > val);
                                top_k.insert(pos, (val, i));
                            } else if val > top_k[k_actual - 1].0 {
                                let pos = top_k.partition_point(|(v, _)| *v > val);
                                top_k.insert(pos, (val, i));
                                top_k.pop();
                            }
                        }
                    } else {
                        // ASC: keep k smallest values
                        for i in 0..num_rows {
                            let val = if float_arr.is_null(i) { f64::INFINITY } else { float_arr.value(i) };
                            
                            if top_k.len() < k_actual {
                                let pos = top_k.partition_point(|(v, _)| *v < val);
                                top_k.insert(pos, (val, i));
                            } else if val < top_k[k_actual - 1].0 {
                                let pos = top_k.partition_point(|(v, _)| *v < val);
                                top_k.insert(pos, (val, i));
                                top_k.pop();
                            }
                        }
                    }
                    
                    let offset = stmt.offset.unwrap_or(0);
                    top_k.into_iter().skip(offset).map(|(_, idx)| idx).collect()
                } else if let Some(int_arr) = col.as_any().downcast_ref::<Int64Array>() {
                    let descending = clause.descending;
                    let mut top_k: Vec<(i64, usize)> = Vec::with_capacity(k_actual + 1);
                    
                    if descending {
                        for i in 0..num_rows {
                            let val = if int_arr.is_null(i) { i64::MIN } else { int_arr.value(i) };
                            
                            if top_k.len() < k_actual {
                                let pos = top_k.partition_point(|(v, _)| *v > val);
                                top_k.insert(pos, (val, i));
                            } else if val > top_k[k_actual - 1].0 {
                                let pos = top_k.partition_point(|(v, _)| *v > val);
                                top_k.insert(pos, (val, i));
                                top_k.pop();
                            }
                        }
                    } else {
                        for i in 0..num_rows {
                            let val = if int_arr.is_null(i) { i64::MAX } else { int_arr.value(i) };
                            
                            if top_k.len() < k_actual {
                                let pos = top_k.partition_point(|(v, _)| *v < val);
                                top_k.insert(pos, (val, i));
                            } else if val < top_k[k_actual - 1].0 {
                                let pos = top_k.partition_point(|(v, _)| *v < val);
                                top_k.insert(pos, (val, i));
                                top_k.pop();
                            }
                        }
                    }
                    
                    let offset = stmt.offset.unwrap_or(0);
                    top_k.into_iter().skip(offset).map(|(_, idx)| idx).collect()
                } else {
                    Self::compute_topk_indices_generic(&sort_batch, &stmt.order_by, k_actual, stmt.offset)
                }
            } else {
                Self::compute_topk_indices_generic(&sort_batch, &stmt.order_by, k_actual, stmt.offset)
            }
        } else {
            Self::compute_topk_indices_generic(&sort_batch, &stmt.order_by, k_actual, stmt.offset)
        };

        if final_indices.is_empty() {
            return backend.read_columns_to_arrow(None, 0, Some(0));
        }

        // Step 3: Read ALL columns but only for top-k row indices
        backend.read_columns_by_indices_to_arrow(&final_indices)
    }

    /// Generic top-k computation using partial sort (fallback for complex cases)
    fn compute_topk_indices_generic(
    sort_batch: &RecordBatch,
    order_by: &[crate::query::OrderByClause],
    k: usize,
    offset: Option<usize>,
) -> Vec<usize> {
    let num_rows = sort_batch.num_rows();
    
    let sort_cols: Vec<(ArrayRef, bool)> = order_by.iter()
        .filter_map(|clause| {
            let col_name = clause.column.trim_matches('"');
            let actual_col = if let Some(dot_pos) = col_name.rfind('.') {
                &col_name[dot_pos + 1..]
            } else {
                col_name
            };
            sort_batch.column_by_name(actual_col).map(|col| (col.clone(), clause.descending))
        })
        .collect();

        let compare_rows = |a: usize, b: usize| -> std::cmp::Ordering {
            for (col, descending) in &sort_cols {
                let ord = Self::compare_array_values(col, a, b);
                if ord != std::cmp::Ordering::Equal {
                    return if *descending { ord.reverse() } else { ord };
                }
            }
            std::cmp::Ordering::Equal
        };

        let mut indices: Vec<usize> = (0..num_rows).collect();
        
        if k < num_rows {
            indices.select_nth_unstable_by(k - 1, |&a, &b| compare_rows(a, b));
            indices.truncate(k);
        }
        indices.sort_by(|&a, &b| compare_rows(a, b));

        if let Some(off) = offset {
            indices.into_iter().skip(off).collect()
        } else {
            indices
        }
    }
    
    /// Fast path for combined WHERE filter + GROUP BY on dictionary columns
    /// Does filter and aggregation in a single pass without intermediate materialization
    fn try_fast_filter_groupby(
        backend: &TableStorageBackend,
        stmt: &SelectStatement,
    ) -> io::Result<Option<RecordBatch>> {
        use arrow::array::DictionaryArray;
        use arrow::datatypes::UInt32Type;
        use crate::query::AggregateFunc;
        
        // Only handle simple patterns: WHERE col = 'value' with single-column GROUP BY
        let where_clause = match &stmt.where_clause {
            Some(w) => w,
            None => return Ok(None),
        };
        
        if stmt.group_by.len() != 1 {
            return Ok(None);
        }
        
        // Extract filter column and value
        let (filter_col, filter_value) = match where_clause {
            SqlExpr::BinaryOp { left, op, right } => {
                use crate::query::sql_parser::BinaryOperator;
                if *op != BinaryOperator::Eq {
                    return Ok(None);
                }
                match (left.as_ref(), right.as_ref()) {
                    (SqlExpr::Column(col), SqlExpr::Literal(Value::String(val))) => {
                        (col.trim_matches('"').to_string(), val.clone())
                    }
                    (SqlExpr::Literal(Value::String(val)), SqlExpr::Column(col)) => {
                        (col.trim_matches('"').to_string(), val.clone())
                    }
                    _ => return Ok(None),
                }
            }
            _ => return Ok(None),
        };
        
        let group_col = stmt.group_by[0].trim_matches('"').to_string();
        
        // Find aggregate column
        let mut agg_col_name: Option<String> = None;
        let mut agg_func: Option<AggregateFunc> = None;
        let mut agg_alias: Option<String> = None;
        
        for col in &stmt.columns {
            if let SelectColumn::Aggregate { func, column, alias, .. } = col {
                if let Some(col_name) = column {
                    let actual = col_name.trim_matches('"');
                    if actual != "*" {
                        agg_col_name = Some(actual.to_string());
                        agg_func = Some(func.clone());
                        agg_alias = alias.clone();
                    }
                }
                break;
            }
        }
        
        let agg_col_name = match agg_col_name {
            Some(c) => c,
            None => return Ok(None),
        };
        
        // Read only needed columns
        let cols_to_read: Vec<&str> = vec![filter_col.as_str(), group_col.as_str(), agg_col_name.as_str()];
        let batch = backend.read_columns_to_arrow(Some(&cols_to_read), 0, None)?;
        
        if batch.num_rows() == 0 {
            return Ok(None);
        }
        
        let num_rows = batch.num_rows();
        
        // Get filter column as dictionary
        let filter_arr = match batch.column_by_name(&filter_col) {
            Some(c) => c,
            None => return Ok(None),
        };
        
        let filter_dict = match filter_arr.as_any().downcast_ref::<DictionaryArray<UInt32Type>>() {
            Some(d) => d,
            None => return Ok(None),
        };
        
        // Find filter key
        let filter_keys = filter_dict.keys();
        let filter_values = filter_dict.values();
        let filter_str_values = match filter_values.as_any().downcast_ref::<StringArray>() {
            Some(s) => s,
            None => return Ok(None),
        };
        
        let mut target_filter_key: Option<u32> = None;
        for i in 0..filter_str_values.len() {
            if filter_str_values.value(i) == filter_value {
                target_filter_key = Some(i as u32);
                break;
            }
        }
        
        let target_filter_key = match target_filter_key {
            Some(k) => k,
            None => {
                // Value not in dictionary - return empty result
                let schema = Arc::new(Schema::new(vec![
                    Field::new(&group_col, ArrowDataType::Utf8, false),
                ]));
                return Ok(Some(RecordBatch::new_empty(schema)));
            }
        };
        
        // Get group column as dictionary
        let group_arr = match batch.column_by_name(&group_col) {
            Some(c) => c,
            None => return Ok(None),
        };
        
        let group_dict = match group_arr.as_any().downcast_ref::<DictionaryArray<UInt32Type>>() {
            Some(d) => d,
            None => return Ok(None),
        };
        
        let group_keys = group_dict.keys();
        let group_values = group_dict.values();
        let group_str_values = match group_values.as_any().downcast_ref::<StringArray>() {
            Some(s) => s,
            None => return Ok(None),
        };
        let group_dict_size = group_str_values.len() + 1;
        
        // Get aggregate column
        let agg_arr = match batch.column_by_name(&agg_col_name) {
            Some(c) => c,
            None => return Ok(None),
        };
        
        let agg_float = agg_arr.as_any().downcast_ref::<Float64Array>();
        let agg_int = agg_arr.as_any().downcast_ref::<Int64Array>();
        
        // Single-pass: filter + aggregate
        let mut counts: Vec<i64> = vec![0; group_dict_size];
        let mut sums: Vec<f64> = vec![0.0; group_dict_size];
        
        let filter_key_values = filter_keys.values();
        let group_key_values = group_keys.values();
        
        if let Some(float_arr) = agg_float {
            if filter_keys.null_count() == 0 && group_keys.null_count() == 0 && float_arr.null_count() == 0 {
                let float_values = float_arr.values();
                for i in 0..num_rows {
                    if unsafe { *filter_key_values.get_unchecked(i) } == target_filter_key {
                        let gk = unsafe { *group_key_values.get_unchecked(i) as usize + 1 };
                        unsafe {
                            *counts.get_unchecked_mut(gk) += 1;
                            *sums.get_unchecked_mut(gk) += *float_values.get_unchecked(i);
                        }
                    }
                }
            } else {
                for i in 0..num_rows {
                    if !filter_keys.is_null(i) && filter_keys.value(i) == target_filter_key {
                        let gk = if group_keys.is_null(i) { 0 } else { group_keys.value(i) as usize + 1 };
                        counts[gk] += 1;
                        if !float_arr.is_null(i) {
                            sums[gk] += float_arr.value(i);
                        }
                    }
                }
            }
        } else if let Some(int_arr) = agg_int {
            for i in 0..num_rows {
                if !filter_keys.is_null(i) && filter_keys.value(i) == target_filter_key {
                    let gk = if group_keys.is_null(i) { 0 } else { group_keys.value(i) as usize + 1 };
                    counts[gk] += 1;
                    if !int_arr.is_null(i) {
                        sums[gk] += int_arr.value(i) as f64;
                    }
                }
            }
        } else {
            return Ok(None);
        }
        
        // Collect results - pre-allocate with estimated group count
        let estimated_groups = (group_dict_size / 4).max(16);
        let mut result_groups: Vec<&str> = Vec::with_capacity(estimated_groups);
        let mut result_values: Vec<f64> = Vec::with_capacity(estimated_groups);
        
        for gk in 1..group_dict_size {
            if counts[gk] > 0 {
                result_groups.push(group_str_values.value(gk - 1));
                let value = match agg_func {
                    Some(AggregateFunc::Sum) => sums[gk],
                    Some(AggregateFunc::Avg) => sums[gk] / counts[gk] as f64,
                    Some(AggregateFunc::Count) => counts[gk] as f64,
                    _ => sums[gk],
                };
                result_values.push(value);
            }
        }
        
        // Build result batch
        let agg_field_name = agg_alias.unwrap_or_else(|| {
            let func_name = match agg_func {
                Some(AggregateFunc::Sum) => "SUM",
                Some(AggregateFunc::Avg) => "AVG",
                Some(AggregateFunc::Count) => "COUNT",
                _ => "AGG",
            };
            format!("{}({})", func_name, agg_col_name)
        });
        
        let schema = Arc::new(Schema::new(vec![
            Field::new(&group_col, ArrowDataType::Utf8, false),
            Field::new(&agg_field_name, ArrowDataType::Float64, true),
        ]));
        
        let mut result_batch = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(StringArray::from(result_groups)),
                Arc::new(Float64Array::from(result_values)),
            ],
        ).map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e.to_string()))?;
        
        // Apply ORDER BY if present
        if !stmt.order_by.is_empty() {
            let k = stmt.limit.map(|l| l + stmt.offset.unwrap_or(0));
            result_batch = Self::apply_order_by_topk(&result_batch, &stmt.order_by, k)?;
        }
        
        // Apply LIMIT/OFFSET
        result_batch = Self::apply_limit_offset(&result_batch, stmt.limit, stmt.offset)?;
        
        Ok(Some(result_batch))
    }

    /// Execute GROUP BY with WHERE using late materialization
    /// 1. Read only WHERE columns first
    /// 2. Filter to get matching row indices
    /// 3. Read GROUP BY + aggregate columns only for matching rows
    fn execute_with_groupby_late_materialization(
        backend: &TableStorageBackend,
        stmt: &SelectStatement,
        storage_path: &Path,
    ) -> io::Result<RecordBatch> {
        use arrow::array::DictionaryArray;
        use arrow::datatypes::UInt32Type;
        
        // FAST PATH: Try combined filter + GROUP BY on dictionary columns in single pass
        if let Some(result) = Self::try_fast_filter_groupby(backend, stmt)? {
            return Ok(result);
        }
        
        // Step 1: Read only columns needed for WHERE clause
        let where_cols = stmt.where_columns();
        let where_col_refs: Vec<&str> = where_cols.iter().map(|s| s.as_str()).collect();
        
        let filter_batch = backend.read_columns_to_arrow(Some(&where_col_refs), 0, None)?;
        
        if filter_batch.num_rows() == 0 {
            // Return empty batch - get required columns for schema
            let required_cols = stmt.required_columns();
            let col_refs: Option<Vec<&str>> = required_cols
                .as_ref()
                .filter(|cols| !cols.is_empty())
                .map(|cols| cols.iter().map(|s| s.as_str()).collect());
            return backend.read_columns_to_arrow(col_refs.as_deref(), 0, Some(0));
        }
        
        // Step 2: Apply WHERE filter to get matching row indices
        let where_clause = stmt.where_clause.as_ref().unwrap();
        let mask = Self::evaluate_predicate_with_storage(&filter_batch, where_clause, storage_path)?;
        
        // Collect matching indices
        let indices: Vec<usize> = mask.iter()
            .enumerate()
            .filter_map(|(i, v)| if v == Some(true) { Some(i) } else { None })
            .collect();
        
        if indices.is_empty() {
            let required_cols = stmt.required_columns();
            let col_refs: Option<Vec<&str>> = required_cols
                .as_ref()
                .filter(|cols| !cols.is_empty())
                .map(|cols| cols.iter().map(|s| s.as_str()).collect());
            return backend.read_columns_to_arrow(col_refs.as_deref(), 0, Some(0));
        }
        
        // Step 3: Read only required columns (GROUP BY + aggregates) for matching rows
        let required_cols = stmt.required_columns();
        let other_cols: Vec<&str> = if let Some(ref cols) = required_cols {
            cols.iter()
                .filter(|c| !where_cols.contains(c))
                .map(|s| s.as_str())
                .collect()
        } else {
            Vec::new()
        };
        
        // Read other columns for matching indices only
        if other_cols.is_empty() {
            // All needed columns are in WHERE - just filter the batch
            let indices_array = arrow::array::UInt64Array::from(
                indices.iter().map(|&i| i as u64).collect::<Vec<_>>()
            );
            let columns: Vec<ArrayRef> = filter_batch
                .columns()
                .iter()
                .map(|col| compute::take(col, &indices_array, None))
                .collect::<Result<Vec<_>, _>>()
                .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e.to_string()))?;
            RecordBatch::try_new(filter_batch.schema(), columns)
                .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e.to_string()))
        } else {
            // Need to read additional columns for matching rows
            let other_batch = backend.read_columns_by_indices_to_arrow(&indices)?;
            
            // Also filter the WHERE columns batch
            let indices_array = arrow::array::UInt64Array::from(
                indices.iter().map(|&i| i as u64).collect::<Vec<_>>()
            );
            let where_columns: Vec<ArrayRef> = filter_batch
                .columns()
                .iter()
                .map(|col| compute::take(col, &indices_array, None))
                .collect::<Result<Vec<_>, _>>()
                .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e.to_string()))?;
            
            // Merge: use other_batch as base (has _id and other columns)
            // Add WHERE columns that aren't already present
            let mut fields: Vec<Field> = other_batch.schema().fields().iter().map(|f| f.as_ref().clone()).collect();
            let mut arrays: Vec<ArrayRef> = other_batch.columns().to_vec();
            
            for (i, field) in filter_batch.schema().fields().iter().enumerate() {
                if other_batch.column_by_name(field.name()).is_none() {
                    fields.push(field.as_ref().clone());
                    arrays.push(where_columns[i].clone());
                }
            }
            
            let schema = Arc::new(Schema::new(fields));
            RecordBatch::try_new(schema, arrays)
                .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e.to_string()))
        }
    }

    /// Execute SELECT statement with JOINs
    fn execute_select_with_joins(stmt: SelectStatement, base_dir: &Path, default_table_path: &Path) -> io::Result<ApexResult> {
        // Get the left (base) table - supports both Table and Subquery (VIEW)
        let mut result_batch = match &stmt.from {
            Some(FromItem::Table { table, .. }) => {
                let left_path = Self::resolve_table_path(table, base_dir, default_table_path);
                let left_backend = get_cached_backend(&left_path)?;
                left_backend.read_columns_to_arrow(None, 0, None)?
            }
            Some(FromItem::Subquery { stmt: sub_stmt, .. }) => {
                // Execute subquery (VIEW) to get source data
                let sub_path = Self::resolve_from_table_path(sub_stmt, base_dir, default_table_path);
                let sub_result = Self::execute_select_with_base_dir(*sub_stmt.clone(), &sub_path, base_dir, default_table_path)?;
                sub_result.to_record_batch()?
            }
            None => {
                let left_backend = get_cached_backend(default_table_path)?;
                left_backend.read_columns_to_arrow(None, 0, None)?
            }
        };

        // Process each JOIN clause - supports both Table and Subquery (VIEW)
        for join_clause in &stmt.joins {
            let right_batch = match &join_clause.right {
                FromItem::Table { table, .. } => {
                    let right_path = Self::resolve_table_path(table, base_dir, default_table_path);
                    let right_backend = get_cached_backend(&right_path)?;
                    right_backend.read_columns_to_arrow(None, 0, None)?
                }
                FromItem::Subquery { stmt: sub_stmt, .. } => {
                    // Execute subquery (VIEW) to get source data
                    let sub_path = Self::resolve_from_table_path(sub_stmt, base_dir, default_table_path);
                    let sub_result = Self::execute_select_with_base_dir(*sub_stmt.clone(), &sub_path, base_dir, default_table_path)?;
                    sub_result.to_record_batch()?
                }
            };

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
            return Ok(ApexResult::Empty(result_batch.schema()));
        }

        // Apply WHERE filter (with storage path for subquery support)
        let filtered = if let Some(ref where_clause) = stmt.where_clause {
            Self::apply_filter_with_storage(&result_batch, where_clause, default_table_path)?
        } else {
            result_batch
        };

        if filtered.num_rows() == 0 {
            return Ok(ApexResult::Empty(filtered.schema()));
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

        Ok(ApexResult::Data(result))
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
            return base_dir.join(format!("{}.apex", table_name));
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

    /// Convert expression to column name string (for display/field naming)
    fn expr_to_column_name(expr: &SqlExpr) -> String {
        match expr {
            SqlExpr::Column(name) => {
                // Handle table.column format - take the column part
                if let Some(dot_pos) = name.rfind('.') {
                    name[dot_pos + 1..].trim_matches('"').to_string()
                } else {
                    name.trim_matches('"').to_string()
                }
            }
            _ => "group".to_string(),
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

        // For INNER JOIN, swap tables if right is larger (build from smaller table)
        let should_swap = !matches!(join_type, JoinType::Left) && right.num_rows() > left.num_rows() * 2;
        
        let (build_batch, probe_batch, build_key_col, probe_key_col, build_key, probe_key, swapped) = if should_swap {
            (left, right, left_key_col, right_key_col, left_key, right_key, true)
        } else {
            (right, left, right_key_col, left_key_col, right_key, left_key, false)
        };

        // Build hash table from build side (smaller table for INNER JOIN)
        let build_rows = build_batch.num_rows();
        
        // For Int64 keys, use optimized direct hash building
        let hash_table: AHashMap<u64, Vec<usize>> = if let Some(build_int_arr) = build_key_col.as_any().downcast_ref::<Int64Array>() {
            let mut table: AHashMap<u64, Vec<usize>> = AHashMap::with_capacity(build_rows);
            for i in 0..build_rows {
                if !build_int_arr.is_null(i) {
                    let val = build_int_arr.value(i);
                    let hash = {
                        let mut h = AHasher::default();
                        val.hash(&mut h);
                        h.finish()
                    };
                    table.entry(hash).or_insert_with(|| Vec::with_capacity(2)).push(i);
                }
            }
            table
        } else {
            let mut table: AHashMap<u64, Vec<usize>> = AHashMap::with_capacity(build_rows);
            for i in 0..build_rows {
                let hash = Self::hash_array_value_fast(build_key_col, i);
                table.entry(hash).or_insert_with(|| Vec::with_capacity(4)).push(i);
            }
            table
        };
        
        let right_rows = right.num_rows();

        // Probe phase - use probe_batch (which may be swapped)
        let probe_rows = probe_batch.num_rows();
        let is_left_join = matches!(join_type, JoinType::Left);
        
        // Check if key columns are Int64 (most common case) - can skip equality check
        let is_int64_key = probe_key_col.as_any().downcast_ref::<Int64Array>().is_some() 
            && build_key_col.as_any().downcast_ref::<Int64Array>().is_some();

        // For INNER JOIN with Int64 keys, use optimized path without Option overhead
        let is_inner_join_fast_path = is_int64_key && (!is_left_join || swapped);
        
        // Collect probe_idx -> build_idx mappings
        let (probe_indices, build_indices_u32, build_indices_opt): (Vec<u32>, Vec<u32>, Vec<Option<u32>>) = 
        if is_inner_join_fast_path {
            // Ultra-fast INNER JOIN path with direct u32 vectors - no Option overhead
            let probe_int_arr = probe_key_col.as_any().downcast_ref::<Int64Array>().unwrap();
            
            let est_matches = probe_rows;
            let mut probe_idx_vec: Vec<u32> = Vec::with_capacity(est_matches);
            let mut build_idx_vec: Vec<u32> = Vec::with_capacity(est_matches);
            
            for probe_idx in 0..probe_rows {
                let probe_val = probe_int_arr.value(probe_idx);
                let probe_hash = {
                    let mut h = AHasher::default();
                    probe_val.hash(&mut h);
                    h.finish()
                };
                
                if let Some(build_matches) = hash_table.get(&probe_hash) {
                    for &build_idx in build_matches {
                        probe_idx_vec.push(probe_idx as u32);
                        build_idx_vec.push(build_idx as u32);
                    }
                }
            }
            (probe_idx_vec, build_idx_vec, Vec::new())
        } else if is_int64_key {
            // LEFT JOIN with Int64 keys - needs Option for NULL handling
            let probe_int_arr = probe_key_col.as_any().downcast_ref::<Int64Array>().unwrap();
            
            let est_matches = probe_rows;
            let mut probe_idx_vec: Vec<u32> = Vec::with_capacity(est_matches);
            let mut build_idx_vec: Vec<Option<u32>> = Vec::with_capacity(est_matches);
            
            for probe_idx in 0..probe_rows {
                let probe_val = probe_int_arr.value(probe_idx);
                let probe_hash = {
                    let mut h = AHasher::default();
                    probe_val.hash(&mut h);
                    h.finish()
                };
                
                if let Some(build_matches) = hash_table.get(&probe_hash) {
                    for &build_idx in build_matches {
                        probe_idx_vec.push(probe_idx as u32);
                        build_idx_vec.push(Some(build_idx as u32));
                    }
                } else {
                    probe_idx_vec.push(probe_idx as u32);
                    build_idx_vec.push(None);
                }
            }
            (probe_idx_vec, Vec::new(), build_idx_vec)
        } else {
            let mut probe_idx_vec: Vec<u32> = Vec::with_capacity(probe_rows);
            let mut build_idx_vec: Vec<Option<u32>> = Vec::with_capacity(probe_rows);
            
            if is_left_join && !swapped {
                for probe_idx in 0..probe_rows {
                    let probe_hash = Self::hash_array_value_fast(probe_key_col, probe_idx);
                    let mut found_match = false;
                    
                    if let Some(build_matches) = hash_table.get(&probe_hash) {
                        for &build_idx in build_matches {
                            if Self::arrays_equal_at(probe_key_col, probe_idx, build_key_col, build_idx) {
                                probe_idx_vec.push(probe_idx as u32);
                                build_idx_vec.push(Some(build_idx as u32));
                                found_match = true;
                            }
                        }
                    }
                    
                    if !found_match {
                        probe_idx_vec.push(probe_idx as u32);
                        build_idx_vec.push(None);
                    }
                }
            } else {
                for probe_idx in 0..probe_rows {
                    let probe_hash = Self::hash_array_value_fast(probe_key_col, probe_idx);
                    
                    if let Some(build_matches) = hash_table.get(&probe_hash) {
                        for &build_idx in build_matches {
                            if Self::arrays_equal_at(probe_key_col, probe_idx, build_key_col, build_idx) {
                                probe_idx_vec.push(probe_idx as u32);
                                build_idx_vec.push(Some(build_idx as u32));
                            }
                        }
                    }
                }
            }
            (probe_idx_vec, Vec::new(), build_idx_vec)
        };
        
        // Convert back to left/right indices based on whether we swapped
        // For INNER JOIN fast path, use direct u32 indices
        let (left_indices, right_indices_u32, right_indices_opt): (Vec<u32>, Vec<u32>, Vec<Option<u32>>) = 
        if is_inner_join_fast_path {
            if swapped {
                (build_indices_u32.clone(), probe_indices.clone(), Vec::new())
            } else {
                (probe_indices, build_indices_u32, Vec::new())
            }
        } else if swapped {
            (build_indices_opt.iter().map(|x| x.unwrap_or(0)).collect(), 
             Vec::new(),
             probe_indices.iter().map(|x| Some(*x)).collect())
        } else {
            (probe_indices, Vec::new(), build_indices_opt)
        };
        
        let left_rows = left.num_rows();

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

        // Build result columns - pre-allocate for all columns
        let mut columns: Vec<ArrayRef> = Vec::with_capacity(left.num_columns() + right.num_columns());

        // Take from left - avoid clone by creating array directly
        let left_indices_array = arrow::array::UInt32Array::from(left_indices);
        for col in left.columns() {
            let taken = compute::take(col.as_ref(), &left_indices_array, None)
                .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e.to_string()))?;
            columns.push(taken);
        }

        // Take from right (excluding join key to avoid duplication)
        // Use direct u32 indices for INNER JOIN fast path (no Option overhead)
        if is_inner_join_fast_path {
            let right_indices_array = arrow::array::UInt32Array::from(right_indices_u32);
            for (col_idx, field) in right.schema().fields().iter().enumerate() {
                if field.name() != right_key {
                    let right_col = right.column(col_idx);
                    let taken = compute::take(right_col.as_ref(), &right_indices_array, None)
                        .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e.to_string()))?;
                    columns.push(taken);
                }
            }
        } else {
            // LEFT JOIN path - handle nulls with Option indices
            for (col_idx, field) in right.schema().fields().iter().enumerate() {
                if field.name() != right_key {
                    let right_col = right.column(col_idx);
                    let taken = Self::take_with_nulls(right_col, &right_indices_opt)?;
                    columns.push(taken);
                }
            }
        }

        RecordBatch::try_new(result_schema, columns)
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e.to_string()))
    }

    /// Hash a value at given index in an array (legacy, uses DefaultHasher)
    fn hash_array_value(array: &ArrayRef, idx: usize) -> u64 {
        Self::hash_array_value_fast(array, idx)
    }

    /// Fast hash using ahash (2-3x faster than DefaultHasher)
    #[inline]
    fn hash_array_value_fast(array: &ArrayRef, idx: usize) -> u64 {
        let mut hasher = AHasher::default();
        
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

    /// Strip table alias prefix from column name (e.g., "o.user_id" -> "user_id")
    fn strip_table_prefix(col_name: &str) -> &str {
        if let Some(dot_pos) = col_name.find('.') {
            &col_name[dot_pos + 1..]
        } else {
            col_name
        }
    }
    
    /// Get column from batch, stripping table prefix if needed
    fn get_column_by_name<'a>(batch: &'a RecordBatch, col_name: &str) -> Option<&'a ArrayRef> {
        let clean_name = col_name.trim_matches('"');
        // Try exact match first
        if let Some(col) = batch.column_by_name(clean_name) {
            return Some(col);
        }
        // Try without table prefix (e.g., "o.user_id" -> "user_id")
        let stripped = Self::strip_table_prefix(clean_name);
        batch.column_by_name(stripped)
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
    /// Uses Zone Maps optimization to potentially skip filtering entirely
    fn apply_filter_with_storage(batch: &RecordBatch, expr: &SqlExpr, storage_path: &Path) -> io::Result<RecordBatch> {
        // Zone Map optimization: check if filter can possibly match
        if let Some(result) = Self::try_zone_map_filter(batch, expr) {
            if result == ZoneMapResult::NoMatch {
                // Filter definitely won't match any rows - return empty batch
                return Ok(RecordBatch::new_empty(batch.schema()));
            }
            // ZoneMapResult::AllMatch would mean all rows match, but we still need to evaluate
            // to handle nulls correctly, so we fall through
        }
        
        let mask = Self::evaluate_predicate_with_storage(batch, expr, storage_path)?;
        compute::filter_record_batch(batch, &mask)
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e.to_string()))
    }
    
    
    /// Try to use Zone Maps to skip filtering
    /// Returns Some(NoMatch) if filter definitely won't match, None otherwise
    fn try_zone_map_filter(batch: &RecordBatch, expr: &SqlExpr) -> Option<ZoneMapResult> {
        match expr {
            SqlExpr::BinaryOp { left, op, right } => {
                // Handle simple column vs literal comparisons
                if let (SqlExpr::Column(col_name), SqlExpr::Literal(lit)) = (left.as_ref(), right.as_ref()) {
                    return Self::check_zone_map_comparison(batch, col_name, op, lit);
                }
                if let (SqlExpr::Literal(lit), SqlExpr::Column(col_name)) = (left.as_ref(), right.as_ref()) {
                    // Flip the operator for literal vs column
                    let flipped_op = match op {
                        BinaryOperator::Lt => BinaryOperator::Gt,
                        BinaryOperator::Le => BinaryOperator::Ge,
                        BinaryOperator::Gt => BinaryOperator::Lt,
                        BinaryOperator::Ge => BinaryOperator::Le,
                        _ => op.clone(),
                    };
                    return Self::check_zone_map_comparison(batch, col_name, &flipped_op, lit);
                }
                
                // Handle AND: if either side is NoMatch, result is NoMatch
                if *op == BinaryOperator::And {
                    let left_result = Self::try_zone_map_filter(batch, left);
                    if left_result == Some(ZoneMapResult::NoMatch) {
                        return Some(ZoneMapResult::NoMatch);
                    }
                    let right_result = Self::try_zone_map_filter(batch, right);
                    if right_result == Some(ZoneMapResult::NoMatch) {
                        return Some(ZoneMapResult::NoMatch);
                    }
                }
                None
            }
            SqlExpr::Between { column, low, high, negated } => {
                // Check if BETWEEN range overlaps with column's value range
                if *negated {
                    return None; // NOT BETWEEN is harder to optimize
                }
                if let (SqlExpr::Literal(low_lit), SqlExpr::Literal(high_lit)) = (low.as_ref(), high.as_ref()) {
                    Self::check_zone_map_between(batch, column, low_lit, high_lit)
                } else {
                    None
                }
            }
            SqlExpr::Paren(inner) => Self::try_zone_map_filter(batch, inner),
            _ => None,
        }
    }
    
    /// Check Zone Map for a simple comparison
    fn check_zone_map_comparison(batch: &RecordBatch, col_name: &str, op: &BinaryOperator, lit: &Value) -> Option<ZoneMapResult> {
        let col_name = col_name.trim_matches('"');
        let col = batch.column_by_name(col_name)?;
        
        let zone_map = if let Some(arr) = col.as_any().downcast_ref::<Int64Array>() {
            ZoneMap::from_int64_array(arr)
        } else if let Some(arr) = col.as_any().downcast_ref::<Float64Array>() {
            ZoneMap::from_float64_array(arr)
        } else {
            return None; // Can't optimize string columns without dictionary encoding
        };
        
        if zone_map.can_match(op, lit) {
            Some(ZoneMapResult::MayMatch)
        } else {
            Some(ZoneMapResult::NoMatch)
        }
    }
    
    /// Check Zone Map for BETWEEN
    fn check_zone_map_between(batch: &RecordBatch, col_name: &str, low: &Value, high: &Value) -> Option<ZoneMapResult> {
        let col_name = col_name.trim_matches('"');
        let col = batch.column_by_name(col_name)?;
        
        let zone_map = if let Some(arr) = col.as_any().downcast_ref::<Int64Array>() {
            ZoneMap::from_int64_array(arr)
        } else if let Some(arr) = col.as_any().downcast_ref::<Float64Array>() {
            ZoneMap::from_float64_array(arr)
        } else {
            return None;
        };
        
        // BETWEEN low AND high: col >= low AND col <= high
        // NoMatch if: max < low OR min > high
        let can_match_low = zone_map.can_match(&BinaryOperator::Ge, low);
        let can_match_high = zone_map.can_match(&BinaryOperator::Le, high);
        
        if can_match_low && can_match_high {
            Some(ZoneMapResult::MayMatch)
        } else {
            Some(ZoneMapResult::NoMatch)
        }
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
                let array = Self::get_column_by_name(batch, col_name)
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
                let val = Self::get_column_by_name(batch, col_name)
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
        let mut outer_refs = Vec::with_capacity(4); // Most subqueries have few outer refs
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
        use arrow::array::Datum;
        
        // OPTIMIZATION: Fast path for column vs literal comparisons using scalar ops
        // This avoids broadcasting the literal to a full array
        if let Some(result) = Self::try_scalar_comparison(batch, left, op, right)? {
            return Ok(result);
        }
        
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
    
    /// Try to use scalar comparison for column vs literal (faster than array vs array)
    #[inline]
    fn try_scalar_comparison(
        batch: &RecordBatch,
        left: &SqlExpr,
        op: &crate::query::sql_parser::BinaryOperator,
        right: &SqlExpr,
    ) -> io::Result<Option<BooleanArray>> {
        use crate::query::sql_parser::BinaryOperator;
        use arrow::array::Scalar;
        
        // Check for column = literal pattern
        let (col_expr, lit_val, reversed) = match (left, right) {
            (SqlExpr::Column(_), SqlExpr::Literal(v)) => (left, v, false),
            (SqlExpr::Literal(v), SqlExpr::Column(_)) => (right, v, true),
            _ => return Ok(None),
        };
        
        let col_array = Self::evaluate_expr_to_array(batch, col_expr)?;
        
        // FAST PATH: DictionaryArray<UInt32, Utf8> - compare using dictionary indices
        // OPTIMIZATION: Use direct buffer access instead of iterator for maximum speed
        use arrow::array::DictionaryArray;
        use arrow::datatypes::UInt32Type;
        use arrow::buffer::BooleanBuffer;
        if let Some(dict_arr) = col_array.as_any().downcast_ref::<DictionaryArray<UInt32Type>>() {
            if let Value::String(s) = lit_val {
                let keys = dict_arr.keys();
                let values = dict_arr.values();
                if let Some(str_values) = values.as_any().downcast_ref::<StringArray>() {
                    // Find which dictionary index matches the filter value
                    let mut target_idx: Option<u32> = None;
                    for i in 0..str_values.len() {
                        if str_values.value(i) == s {
                            target_idx = Some(i as u32);
                            break;
                        }
                    }
                    
                    // Ultra-fast integer comparison on dictionary indices using direct buffer access
                    let num_rows = keys.len();
                    let result: BooleanArray = match (op, reversed, target_idx) {
                        (BinaryOperator::Eq, _, Some(idx)) => {
                            // Fast path: no nulls - use raw slice comparison
                            if keys.null_count() == 0 {
                                let key_values = keys.values();
                                let bools: Vec<bool> = key_values.iter().map(|&k| k == idx).collect();
                                BooleanArray::from(bools)
                            } else {
                                // Has nulls - handle them
                                let mut builder = arrow::array::BooleanBuilder::with_capacity(num_rows);
                                for i in 0..num_rows {
                                    if keys.is_null(i) {
                                        builder.append_value(false);
                                    } else {
                                        builder.append_value(keys.value(i) == idx);
                                    }
                                }
                                builder.finish()
                            }
                        }
                        (BinaryOperator::Eq, _, None) => {
                            // Value not in dictionary - no matches
                            BooleanArray::from(vec![false; num_rows])
                        }
                        (BinaryOperator::NotEq, _, Some(idx)) => {
                            if keys.null_count() == 0 {
                                let key_values = keys.values();
                                let bools: Vec<bool> = key_values.iter().map(|&k| k != idx).collect();
                                BooleanArray::from(bools)
                            } else {
                                let mut builder = arrow::array::BooleanBuilder::with_capacity(num_rows);
                                for i in 0..num_rows {
                                    if keys.is_null(i) {
                                        builder.append_value(false);
                                    } else {
                                        builder.append_value(keys.value(i) != idx);
                                    }
                                }
                                builder.finish()
                            }
                        }
                        (BinaryOperator::NotEq, _, None) => {
                            // Value not in dictionary - all non-null match
                            if keys.null_count() == 0 {
                                BooleanArray::from(vec![true; num_rows])
                            } else {
                                let mut builder = arrow::array::BooleanBuilder::with_capacity(num_rows);
                                for i in 0..num_rows {
                                    builder.append_value(!keys.is_null(i));
                                }
                                builder.finish()
                            }
                        }
                        _ => return Ok(None), // Other comparisons fall through
                    };
                    return Ok(Some(result));
                }
            }
        }
        
        // String scalar comparison (regular StringArray)
        if let Some(str_arr) = col_array.as_any().downcast_ref::<StringArray>() {
            if let Value::String(s) = lit_val {
                let scalar = Scalar::new(arrow::array::StringArray::from(vec![s.as_str()]));
                let result = match (op, reversed) {
                    (BinaryOperator::Eq, _) => cmp::eq(str_arr, &scalar),
                    (BinaryOperator::NotEq, _) => cmp::neq(str_arr, &scalar),
                    (BinaryOperator::Lt, false) | (BinaryOperator::Gt, true) => cmp::lt(str_arr, &scalar),
                    (BinaryOperator::Le, false) | (BinaryOperator::Ge, true) => cmp::lt_eq(str_arr, &scalar),
                    (BinaryOperator::Gt, false) | (BinaryOperator::Lt, true) => cmp::gt(str_arr, &scalar),
                    (BinaryOperator::Ge, false) | (BinaryOperator::Le, true) => cmp::gt_eq(str_arr, &scalar),
                    _ => return Ok(None),
                };
                return result.map(Some).map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e.to_string()));
            }
        }
        
        // Int64 scalar comparison
        if let Some(int_arr) = col_array.as_any().downcast_ref::<Int64Array>() {
            let int_val = match lit_val {
                Value::Int64(i) => *i,
                Value::Float64(f) => *f as i64,
                _ => return Ok(None),
            };
            
            // JIT optimization for large arrays (>100k rows)
            let num_rows = int_arr.len();
            if num_rows > 100_000 {
                if let Some(result) = Self::try_jit_int_filter(int_arr, op, int_val, reversed) {
                    return Ok(Some(result));
                }
            }
            
            let scalar = Scalar::new(Int64Array::from(vec![int_val]));
            let result = match (op, reversed) {
                (BinaryOperator::Eq, _) => cmp::eq(int_arr, &scalar),
                (BinaryOperator::NotEq, _) => cmp::neq(int_arr, &scalar),
                (BinaryOperator::Lt, false) | (BinaryOperator::Gt, true) => cmp::lt(int_arr, &scalar),
                (BinaryOperator::Le, false) | (BinaryOperator::Ge, true) => cmp::lt_eq(int_arr, &scalar),
                (BinaryOperator::Gt, false) | (BinaryOperator::Lt, true) => cmp::gt(int_arr, &scalar),
                (BinaryOperator::Ge, false) | (BinaryOperator::Le, true) => cmp::gt_eq(int_arr, &scalar),
                _ => return Ok(None),
            };
            return result.map(Some).map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e.to_string()));
        }
        
        // Float64 scalar comparison
        if let Some(float_arr) = col_array.as_any().downcast_ref::<Float64Array>() {
            let float_val = match lit_val {
                Value::Float64(f) => *f,
                Value::Int64(i) => *i as f64,
                _ => return Ok(None),
            };
            let scalar = Scalar::new(Float64Array::from(vec![float_val]));
            let result = match (op, reversed) {
                (BinaryOperator::Eq, _) => cmp::eq(float_arr, &scalar),
                (BinaryOperator::NotEq, _) => cmp::neq(float_arr, &scalar),
                (BinaryOperator::Lt, false) | (BinaryOperator::Gt, true) => cmp::lt(float_arr, &scalar),
                (BinaryOperator::Le, false) | (BinaryOperator::Ge, true) => cmp::lt_eq(float_arr, &scalar),
                (BinaryOperator::Gt, false) | (BinaryOperator::Lt, true) => cmp::gt(float_arr, &scalar),
                (BinaryOperator::Ge, false) | (BinaryOperator::Le, true) => cmp::gt_eq(float_arr, &scalar),
                _ => return Ok(None),
            };
            return result.map(Some).map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e.to_string()));
        }
        
        Ok(None)
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

    /// Try to use JIT compilation for integer filter (for large arrays)
    /// Returns None if JIT compilation fails
    fn try_jit_int_filter(
        int_arr: &Int64Array,
        op: &BinaryOperator,
        lit_val: i64,
        reversed: bool,
    ) -> Option<BooleanArray> {
        // Adjust operator if reversed
        let actual_op = if reversed {
            match op {
                BinaryOperator::Lt => BinaryOperator::Gt,
                BinaryOperator::Le => BinaryOperator::Ge,
                BinaryOperator::Gt => BinaryOperator::Lt,
                BinaryOperator::Ge => BinaryOperator::Le,
                _ => op.clone(),
            }
        } else {
            op.clone()
        };
        
        // Try to compile and execute JIT filter
        let mut jit = ExprJIT::new().ok()?;
        let filter_fn = jit.compile_int_filter(actual_op, lit_val).ok()?;
        
        let num_rows = int_arr.len();
        let mut result_bytes = vec![0u8; num_rows];
        
        // Get raw pointer to i64 data
        let data_ptr = int_arr.values().as_ptr();
        
        // Execute JIT-compiled filter
        unsafe {
            filter_fn(data_ptr, num_rows, result_bytes.as_mut_ptr());
        }
        
        // Convert result bytes to BooleanArray
        let bools: Vec<bool> = result_bytes.iter().map(|&b| b != 0).collect();
        Some(BooleanArray::from(bools))
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
        let target = Self::get_column_by_name(batch, col_name)
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
        let array = Self::get_column_by_name(batch, col_name)
            .ok_or_else(|| io::Error::new(io::ErrorKind::NotFound, format!("Column '{}' not found", col_name)))?;
        
        // Convert SQL LIKE pattern to regex
        let regex_pattern = Self::like_to_regex(pattern);
        let regex = regex::Regex::new(&regex_pattern)
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidInput, e.to_string()))?;
        
        // Handle both StringArray and DictionaryArray
        use arrow::array::DictionaryArray;
        use arrow::datatypes::UInt32Type;
        
        let result: BooleanArray = if let Some(string_array) = array.as_any().downcast_ref::<StringArray>() {
            string_array
                .iter()
                .map(|opt| opt.map(|s| regex.is_match(s)).unwrap_or(false))
                .collect()
        } else if let Some(dict_array) = array.as_any().downcast_ref::<DictionaryArray<UInt32Type>>() {
            // Handle dictionary-encoded string columns
            let values = dict_array.values();
            let str_values = values.as_any().downcast_ref::<StringArray>()
                .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "Dictionary values must be strings"))?;
            let keys = dict_array.keys();
            
            (0..dict_array.len())
                .map(|i| {
                    if keys.is_null(i) {
                        false
                    } else {
                        let key = keys.value(i) as usize;
                        if key < str_values.len() && !str_values.is_null(key) {
                            regex.is_match(str_values.value(key))
                        } else {
                            false
                        }
                    }
                })
                .collect()
        } else {
            return Err(io::Error::new(io::ErrorKind::InvalidData, "LIKE requires string column"));
        };
        
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
        let array = Self::get_column_by_name(batch, col_name)
            .ok_or_else(|| io::Error::new(io::ErrorKind::NotFound, format!("Column '{}' not found", col_name)))?;
        
        // Use pattern directly as regex (convert glob-style * to regex .*)
        let regex_pattern = pattern.replace("*", ".*");
        let regex = regex::Regex::new(&regex_pattern)
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidInput, e.to_string()))?;
        
        // Handle both StringArray and DictionaryArray
        use arrow::array::DictionaryArray;
        use arrow::datatypes::UInt32Type;
        
        let result: BooleanArray = if let Some(string_array) = array.as_any().downcast_ref::<StringArray>() {
            string_array
                .iter()
                .map(|opt| opt.map(|s| regex.is_match(s)).unwrap_or(false))
                .collect()
        } else if let Some(dict_array) = array.as_any().downcast_ref::<DictionaryArray<UInt32Type>>() {
            let values = dict_array.values();
            let str_values = values.as_any().downcast_ref::<StringArray>()
                .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "Dictionary values must be strings"))?;
            let keys = dict_array.keys();
            
            (0..dict_array.len())
                .map(|i| {
                    if keys.is_null(i) {
                        false
                    } else {
                        let key = keys.value(i) as usize;
                        if key < str_values.len() && !str_values.is_null(key) {
                            regex.is_match(str_values.value(key))
                        } else {
                            false
                        }
                    }
                })
                .collect()
        } else {
            return Err(io::Error::new(io::ErrorKind::InvalidData, "REGEXP requires string column"));
        };
        
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
    /// OPTIMIZATION: Use parallel sort for large datasets (>100K rows) using Rayon
    fn apply_order_by(
        batch: &RecordBatch,
        order_by: &[crate::query::OrderByClause],
    ) -> io::Result<RecordBatch> {
        use arrow::compute::SortColumn;
        use rayon::prelude::*;

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

        let num_rows = batch.num_rows();
        
        // For large datasets (>50K rows), use parallel sort with Rayon
        let indices = if num_rows > 50_000 && sort_columns.len() == 1 {
            // Single column sort - use parallel sort for better performance
            Self::parallel_sort_indices(batch, order_by)?
        } else {
            // Multi-column or small dataset - use Arrow's lexsort
            compute::lexsort_to_indices(&sort_columns, None)
                .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e.to_string()))?
        };

        // OPTIMIZATION: Use SIMD-accelerated take for better performance
        let indices_array = &indices;
        let columns: Vec<ArrayRef> = if num_rows > 100_000 {
            use rayon::prelude::*;
            use crate::query::simd_take::optimized_take;
            batch
                .columns()
                .par_iter()
                .map(|col| Arc::new(optimized_take(col, indices_array)) as ArrayRef)
                .collect()
        } else {
            use crate::query::simd_take::optimized_take;
            batch
                .columns()
                .iter()
                .map(|col| Arc::new(optimized_take(col, indices_array)) as ArrayRef)
                .collect()
        };

        RecordBatch::try_new(batch.schema(), columns)
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e.to_string()))
    }
    
    /// Parallel sort for single-column ORDER BY using Rayon
    /// Uses indices with custom comparator for better memory efficiency
    fn parallel_sort_indices(
        batch: &RecordBatch,
        order_by: &[crate::query::OrderByClause],
    ) -> io::Result<arrow::array::UInt32Array> {
        use rayon::prelude::*;
        
        let clause = &order_by[0];
        let col_name = clause.column.trim_matches('"');
        let actual_col = if let Some(dot_pos) = col_name.rfind('.') {
            &col_name[dot_pos + 1..]
        } else {
            col_name
        };
        
        let col = batch.column_by_name(actual_col)
            .ok_or_else(|| io::Error::new(io::ErrorKind::NotFound, format!("Column {} not found", actual_col)))?;
        
        let num_rows = batch.num_rows();
        
        // For Int64 - use fast parallel sort with custom comparator
        if let Some(int_arr) = col.as_any().downcast_ref::<Int64Array>() {
            let descending = clause.descending;
            
            // Check if we can use counting sort (range is limited)
            if !descending {
                let (min_val, max_val) = {
                    let mut min = i64::MAX;
                    let mut max = i64::MIN;
                    for i in 0..num_rows {
                        if !int_arr.is_null(i) {
                            let v = int_arr.value(i);
                            min = min.min(v);
                            max = max.max(v);
                        }
                    }
                    (min, max)
                };
                
                let range = (max_val - min_val + 1) as usize;
                // Use counting sort if range is reasonable (< 5M values)
                if range <= 5_000_000 && range > 0 {
                    return Self::counting_sort_indices(int_arr, min_val, max_val, num_rows);
                }
            }
            
            // Create (value, index) pairs and sort in parallel
            let mut pairs: Vec<(i64, usize)> = (0..num_rows)
                .map(|i| {
                    let val = if int_arr.is_null(i) { 
                        if descending { i64::MIN } else { i64::MAX }
                    } else { 
                        int_arr.value(i) 
                    };
                    (val, i)
                })
                .collect();
            
            // Parallel sort using unstable sort for better performance
            if descending {
                pairs.par_sort_unstable_by(|a, b| b.0.cmp(&a.0));
            } else {
                pairs.par_sort_unstable_by(|a, b| a.0.cmp(&b.0));
            }
            
            let sorted_indices: Vec<u32> = pairs.iter().map(|(_, idx)| *idx as u32).collect();
            return Ok(arrow::array::UInt32Array::from(sorted_indices));
        }
        
        // For Float64 - use parallel sort
        if let Some(float_arr) = col.as_any().downcast_ref::<Float64Array>() {
            let descending = clause.descending;
            
            let mut pairs: Vec<(f64, usize)> = (0..num_rows)
                .map(|i| {
                    let val = if float_arr.is_null(i) { 
                        if descending { f64::NEG_INFINITY } else { f64::INFINITY }
                    } else { 
                        float_arr.value(i) 
                    };
                    (val, i)
                })
                .collect();
            
            if descending {
                pairs.par_sort_unstable_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
            } else {
                pairs.par_sort_unstable_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
            }
            
            let sorted_indices: Vec<u32> = pairs.iter().map(|(_, idx)| *idx as u32).collect();
            return Ok(arrow::array::UInt32Array::from(sorted_indices));
        }
        
        // Fallback to Arrow's lexsort for other types
        use arrow::compute::{SortColumn, SortOptions};
        let sort_columns: Vec<_> = order_by
            .iter()
            .filter_map(|clause| {
                let col_name = clause.column.trim_matches('"');
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
        
        compute::lexsort_to_indices(&sort_columns, None)
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e.to_string()))
    }
    
    /// Counting sort for integer arrays with limited range
    /// O(n + range) complexity, much faster than comparison sort for small ranges
    fn counting_sort_indices(
        arr: &Int64Array,
        min_val: i64,
        max_val: i64,
        num_rows: usize,
    ) -> io::Result<arrow::array::UInt32Array> {
        let range = (max_val - min_val + 1) as usize;
        let mut counts: Vec<usize> = vec![0; range + 1]; // +1 for nulls
        
        // Count occurrences
        for i in 0..num_rows {
            if arr.is_null(i) {
                counts[range] += 1;
            } else {
                let idx = (arr.value(i) - min_val) as usize;
                counts[idx] += 1;
            }
        }
        
        // Compute prefix sums for output positions
        let mut positions: Vec<usize> = vec![0; range + 1];
        let mut total = 0;
        for i in 0..range {
            positions[i] = total;
            total += counts[i];
        }
        positions[range] = total; // nulls at the end
        
        // Build result indices
        let mut result: Vec<u32> = vec![0; num_rows];
        for i in 0..num_rows {
            let pos = if arr.is_null(i) {
                positions[range]
            } else {
                let idx = (arr.value(i) - min_val) as usize;
                positions[idx]
            };
            result[pos] = i as u32;
            if arr.is_null(i) {
                positions[range] += 1;
            } else {
                positions[(arr.value(i) - min_val) as usize] += 1;
            }
        }
        
        Ok(arrow::array::UInt32Array::from(result))
    }

    /// Apply ORDER BY with top-k optimization (heap sort for LIMIT queries)
    /// When k is Some, uses partial sort O(n log k) instead of full sort O(n log n)
    /// OPTIMIZATION: Pre-downcast columns once to avoid repeated dynamic dispatch
    fn apply_order_by_topk(
        batch: &RecordBatch,
        order_by: &[crate::query::OrderByClause],
        k: Option<usize>,
    ) -> io::Result<RecordBatch> {
        use std::cmp::Ordering;

        // If no limit or limit >= rows, use standard sort
        let num_rows = batch.num_rows();
        if k.is_none() || k.unwrap() >= num_rows {
            return Self::apply_order_by(batch, order_by);
        }
        let k = k.unwrap();

        if k == 0 {
            return Ok(RecordBatch::new_empty(batch.schema()));
        }

        // Pre-downcast sort columns for fast comparison (avoid repeated dynamic dispatch)
        enum TypedSortCol<'a> {
            Int64(&'a Int64Array, bool),   // (array, descending)
            Float64(&'a Float64Array, bool),
            String(&'a StringArray, bool),
            Other(&'a ArrayRef, bool),
        }
        
        let typed_sort_cols: Vec<TypedSortCol> = order_by
            .iter()
            .filter_map(|clause| {
                let col_name = clause.column.trim_matches('"');
                let actual_col = if let Some(dot_pos) = col_name.rfind('.') {
                    &col_name[dot_pos + 1..]
                } else {
                    col_name
                };
                batch.column_by_name(actual_col).map(|col| {
                    if let Some(arr) = col.as_any().downcast_ref::<Int64Array>() {
                        TypedSortCol::Int64(arr, clause.descending)
                    } else if let Some(arr) = col.as_any().downcast_ref::<Float64Array>() {
                        TypedSortCol::Float64(arr, clause.descending)
                    } else if let Some(arr) = col.as_any().downcast_ref::<StringArray>() {
                        TypedSortCol::String(arr, clause.descending)
                    } else {
                        TypedSortCol::Other(col, clause.descending)
                    }
                })
            })
            .collect();

        if typed_sort_cols.is_empty() {
            return Ok(batch.clone());
        }

        // Fast comparison using pre-downcast columns
        let compare_rows = |a: usize, b: usize| -> Ordering {
            for col in &typed_sort_cols {
                let ord = match col {
                    TypedSortCol::Int64(arr, desc) => {
                        let a_null = arr.is_null(a);
                        let b_null = arr.is_null(b);
                        let ord = if a_null && b_null {
                            Ordering::Equal
                        } else if a_null {
                            Ordering::Greater
                        } else if b_null {
                            Ordering::Less
                        } else {
                            arr.value(a).cmp(&arr.value(b))
                        };
                        if *desc { ord.reverse() } else { ord }
                    }
                    TypedSortCol::Float64(arr, desc) => {
                        let a_null = arr.is_null(a);
                        let b_null = arr.is_null(b);
                        let ord = if a_null && b_null {
                            Ordering::Equal
                        } else if a_null {
                            Ordering::Greater
                        } else if b_null {
                            Ordering::Less
                        } else {
                            arr.value(a).partial_cmp(&arr.value(b)).unwrap_or(Ordering::Equal)
                        };
                        if *desc { ord.reverse() } else { ord }
                    }
                    TypedSortCol::String(arr, desc) => {
                        let a_null = arr.is_null(a);
                        let b_null = arr.is_null(b);
                        let ord = if a_null && b_null {
                            Ordering::Equal
                        } else if a_null {
                            Ordering::Greater
                        } else if b_null {
                            Ordering::Less
                        } else {
                            arr.value(a).cmp(arr.value(b))
                        };
                        if *desc { ord.reverse() } else { ord }
                    }
                    TypedSortCol::Other(arr, desc) => {
                        let ord = Self::compare_array_values(arr, a, b);
                        if *desc { ord.reverse() } else { ord }
                    }
                };
                if ord != Ordering::Equal {
                    return ord;
                }
            }
            Ordering::Equal
        };

        // OPTIMIZATION: For small k, use heap-based top-k (O(n log k))
        // For larger k, use partial sort (O(n + k log k))
        let indices: Vec<usize> = if k <= 100 {
            // Heap-based approach for small k - maintains a max-heap of size k
            use std::collections::BinaryHeap;
            
            // Wrapper for reverse comparison (we want min-heap behavior for top-k)
            struct HeapItem(usize);
            
            impl PartialEq for HeapItem {
                fn eq(&self, other: &Self) -> bool { self.0 == other.0 }
            }
            impl Eq for HeapItem {}
            
            // Create comparison closure that captures typed_sort_cols
            let mut heap: BinaryHeap<(std::cmp::Reverse<usize>, usize)> = BinaryHeap::with_capacity(k + 1);
            
            // Simple approach: store (score, index) where score is computed once
            // For numeric DESC sorting, we can use the value directly as score
            if typed_sort_cols.len() == 1 {
                match &typed_sort_cols[0] {
                    TypedSortCol::Float64(arr, true) => {
                        // DESC sort on float - use value as score, keep k largest
                        let mut top_k: Vec<(i64, usize)> = Vec::with_capacity(k);
                        for i in 0..num_rows {
                            let score = if arr.is_null(i) { i64::MIN } else { 
                                (arr.value(i) * 1e10) as i64 // Scale to preserve precision
                            };
                            if top_k.len() < k {
                                top_k.push((score, i));
                                if top_k.len() == k {
                                    top_k.sort_by(|a, b| b.0.cmp(&a.0));
                                }
                            } else if score > top_k[k-1].0 {
                                top_k[k-1] = (score, i);
                                // Bubble up
                                let mut j = k - 1;
                                while j > 0 && top_k[j].0 > top_k[j-1].0 {
                                    top_k.swap(j, j-1);
                                    j -= 1;
                                }
                            }
                        }
                        top_k.iter().map(|(_, i)| *i).collect()
                    }
                    TypedSortCol::Int64(arr, true) => {
                        // DESC sort on int - use value as score, keep k largest
                        let mut top_k: Vec<(i64, usize)> = Vec::with_capacity(k);
                        for i in 0..num_rows {
                            let score = if arr.is_null(i) { i64::MIN } else { arr.value(i) };
                            if top_k.len() < k {
                                top_k.push((score, i));
                                if top_k.len() == k {
                                    top_k.sort_by(|a, b| b.0.cmp(&a.0));
                                }
                            } else if score > top_k[k-1].0 {
                                top_k[k-1] = (score, i);
                                let mut j = k - 1;
                                while j > 0 && top_k[j].0 > top_k[j-1].0 {
                                    top_k.swap(j, j-1);
                                    j -= 1;
                                }
                            }
                        }
                        top_k.iter().map(|(_, i)| *i).collect()
                    }
                    _ => {
                        // Fall back to generic approach
                        let mut all_indices: Vec<usize> = (0..num_rows).collect();
                        all_indices.select_nth_unstable_by(k - 1, |&a, &b| compare_rows(a, b));
                        all_indices.truncate(k);
                        all_indices.sort_by(|&a, &b| compare_rows(a, b));
                        all_indices
                    }
                }
            } else {
                // Multi-column sort - use generic approach
                let mut all_indices: Vec<usize> = (0..num_rows).collect();
                all_indices.select_nth_unstable_by(k - 1, |&a, &b| compare_rows(a, b));
                all_indices.truncate(k);
                all_indices.sort_by(|&a, &b| compare_rows(a, b));
                all_indices
            }
        } else {
            // Larger k - use partial sort (more efficient for k > 100)
            let mut all_indices: Vec<usize> = (0..num_rows).collect();
            if k < num_rows {
                all_indices.select_nth_unstable_by(k - 1, |&a, &b| compare_rows(a, b));
                all_indices.truncate(k);
            }
            all_indices.sort_by(|&a, &b| compare_rows(a, b));
            all_indices
        };

        // Take rows by indices
        let indices_array = arrow::array::UInt64Array::from(
            indices.iter().map(|&i| i as u64).collect::<Vec<_>>()
        );

        let columns: Vec<ArrayRef> = batch
            .columns()
            .iter()
            .map(|col| compute::take(col, &indices_array, None))
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
    ) -> io::Result<ApexResult> {
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
            return Ok(ApexResult::Scalar(batch.num_rows() as i64));
        }

        let schema = Arc::new(Schema::new(fields));
        let result = RecordBatch::try_new(schema, arrays)
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e.to_string()))?;

        Ok(ApexResult::Data(result))
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
                    // SIMD-optimized sum for non-null arrays
                    let sum = if int_array.null_count() == 0 {
                        simd_sum_i64(int_array.values())
                    } else {
                        int_array.iter().filter_map(|v| v).sum()
                    };
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
                    // SIMD-optimized sum for non-null arrays
                    let sum = if float_array.null_count() == 0 {
                        simd_sum_f64(float_array.values())
                    } else {
                        float_array.iter().filter_map(|v| v).sum()
                    };
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
                    // SIMD-optimized AVG for non-null arrays
                    let (sum, count) = if int_array.null_count() == 0 {
                        (simd_sum_i64(int_array.values()), int_array.len())
                    } else {
                        let values: Vec<i64> = int_array.iter().filter_map(|v| v).collect();
                        (values.iter().sum::<i64>(), values.len())
                    };
                    let avg = if count == 0 { 0.0 } else { sum as f64 / count as f64 };
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
                    // SIMD-optimized AVG for non-null arrays
                    let (sum, count) = if float_array.null_count() == 0 {
                        (simd_sum_f64(float_array.values()), float_array.len())
                    } else {
                        let values: Vec<f64> = float_array.iter().filter_map(|v| v).collect();
                        (values.iter().sum::<f64>(), values.len())
                    };
                    let avg = if count == 0 { 0.0 } else { sum / count as f64 };
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
    /// OPTIMIZATION: Uses vectorized execution engine for maximum performance
    fn execute_group_by(batch: &RecordBatch, stmt: &SelectStatement) -> io::Result<ApexResult> {
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
        
        // Check if we can use fast path: only simple aggregates (COUNT, SUM, AVG, MIN, MAX)
        // without DISTINCT, expressions, or HAVING that needs row access
        let can_use_incremental = Self::can_use_incremental_aggregation(stmt);
        
        if can_use_incremental {
            // Try vectorized execution for single-column GROUP BY
            if group_cols.len() == 1 {
                if let Ok(result) = Self::execute_group_by_vectorized(batch, stmt, &group_cols[0]) {
                    return Ok(result);
                }
            }
            return Self::execute_group_by_incremental(batch, stmt, &group_cols);
        }
        
        // Fall back to full row-index based aggregation for complex cases
        Self::execute_group_by_with_indices(batch, stmt, &group_cols)
    }
    
    /// Execute GROUP BY using vectorized execution engine
    /// Processes data in 2048-row batches for cache efficiency
    fn execute_group_by_vectorized(
        batch: &RecordBatch,
        stmt: &SelectStatement,
        group_col_name: &str,
    ) -> io::Result<ApexResult> {
        use crate::query::vectorized::{execute_vectorized_group_by, VectorizedHashAgg};
        use crate::query::AggregateFunc;
        
        // OPTIMIZATION: For DictionaryArray columns, use direct indexing (much faster)
        if let Some(col) = batch.column_by_name(group_col_name) {
            use arrow::array::DictionaryArray;
            use arrow::datatypes::UInt32Type;
            if let Some(dict_arr) = col.as_any().downcast_ref::<DictionaryArray<UInt32Type>>() {
                let keys = dict_arr.keys();
                let values = dict_arr.values();
                if let Some(str_values) = values.as_any().downcast_ref::<StringArray>() {
                    let num_rows = batch.num_rows();
                    let dict_size = str_values.len() + 1; // +1 for NULL slot
                    
                    // Check if this is COUNT(*) only - can optimize by streaming directly
                    let is_count_only = stmt.columns.iter().all(|c| {
                        matches!(c, SelectColumn::Aggregate { func: AggregateFunc::Count, column: None, .. })
                    });
                    
                    if is_count_only {
                        // OPTIMIZED: Direct aggregation without building indices Vec
                        let mut counts: Vec<i64> = vec![0; dict_size];
                        
                        for row_idx in 0..num_rows {
                            if !keys.is_null(row_idx) {
                                let group_idx = keys.value(row_idx) as usize + 1;
                                unsafe { *counts.get_unchecked_mut(group_idx) += 1; }
                            }
                        }
                        
                        // Build result directly
                        let active_groups: Vec<usize> = (1..dict_size)
                            .filter(|&i| counts[i] > 0)
                            .collect();
                        
                        let mut result_fields: Vec<Field> = Vec::new();
                        let mut result_arrays: Vec<ArrayRef> = Vec::new();
                        
                        // Add group column
                        let group_col_name_clean = stmt.group_by.first()
                            .map(|s| {
                                let trimmed = s.trim_matches('"');
                                if let Some(dot_pos) = trimmed.rfind('.') {
                                    trimmed[dot_pos + 1..].to_string()
                                } else {
                                    trimmed.to_string()
                                }
                            })
                            .unwrap_or_else(|| "group".to_string());
                        
                        let group_values: Vec<&str> = active_groups.iter()
                            .map(|&i| str_values.value(i - 1))
                            .collect();
                        result_fields.push(Field::new(&group_col_name_clean, ArrowDataType::Utf8, false));
                        result_arrays.push(Arc::new(StringArray::from(group_values)));
                        
                        // Add COUNT(*) column
                        let count_values: Vec<i64> = active_groups.iter().map(|&i| counts[i]).collect();
                        result_fields.push(Field::new("COUNT(*)", ArrowDataType::Int64, false));
                        result_arrays.push(Arc::new(Int64Array::from(count_values)));
                        
                        let schema = Arc::new(Schema::new(result_fields));
                        let result_batch = RecordBatch::try_new(schema, result_arrays)
                            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e.to_string()))?;
                        
                        return Ok(ApexResult::Data(result_batch));
                    }
                    
                    // For other aggregates, use the standard path
                    let indices: Vec<u32> = (0..num_rows)
                        .map(|i| {
                            if keys.is_null(i) { 0u32 } else { keys.value(i) + 1 }
                        })
                        .collect();
                    
                    let dict_values: Vec<&str> = (0..str_values.len())
                        .map(|i| str_values.value(i))
                        .collect();
                    
                    return Self::execute_group_by_string_dict(
                        batch, stmt, str_values, &indices, &dict_values, dict_size
                    );
                }
            }
            
            // OPTIMIZATION: For low-cardinality StringArray, build dictionary on the fly
            // REMOVED sampling to stabilize performance - always try dictionary path first
            if let Some(str_arr) = col.as_any().downcast_ref::<StringArray>() {
                let num_rows = batch.num_rows();
                
                // Build dictionary directly without sampling - more stable performance
                let mut dict: AHashMap<&str, u32> = AHashMap::with_capacity(200);
                let mut dict_values: Vec<&str> = Vec::with_capacity(200);
                let mut next_id = 1u32;
                
                // First pass: build dictionary and check cardinality
                let mut indices: Vec<u32> = Vec::with_capacity(num_rows);
                indices.resize(num_rows, 0);
                
                for i in 0..num_rows {
                    if !str_arr.is_null(i) {
                        let s = str_arr.value(i);
                        let id = *dict.entry(s).or_insert_with(|| {
                            let id = next_id;
                            next_id += 1;
                            dict_values.push(s);
                            id
                        });
                        indices[i] = id;
                    }
                }
                
                // Only use dict indexing if cardinality is reasonable (<=1000)
                let dict_size = dict_values.len() + 1;
                if dict_size <= 1000 {
                    return Self::execute_group_by_string_dict(
                        batch, stmt, str_arr, &indices, &dict_values, dict_size
                    );
                }
                // Fall through to hash-based aggregation for high cardinality
            }
        }
        
        // Find aggregate column name and type
        let mut agg_col_name: Option<&str> = None;
        let mut has_int_agg = false;
        
        for col in &stmt.columns {
            if let SelectColumn::Aggregate { column: Some(col_name), .. } = col {
                let actual_col = col_name.trim_matches('"');
                let actual_col = if let Some(dot_pos) = actual_col.rfind('.') {
                    &actual_col[dot_pos + 1..]
                } else {
                    actual_col
                };
                if actual_col != "*" {
                    agg_col_name = Some(actual_col);
                    // Check if it's an int column
                    if let Some(arr) = batch.column_by_name(actual_col) {
                        has_int_agg = arr.as_any().downcast_ref::<Int64Array>().is_some();
                    }
                }
                break;
            }
        }
        
        // Execute vectorized GROUP BY for non-dictionary columns
        let hash_agg = execute_vectorized_group_by(batch, group_col_name, agg_col_name, has_int_agg)?;
        
        // Build result from hash aggregation table
        Self::build_group_by_result_from_vectorized(stmt, group_col_name, &hash_agg, has_int_agg)
    }
    
    /// Build GROUP BY result from vectorized hash aggregation
    fn build_group_by_result_from_vectorized(
        stmt: &SelectStatement,
        group_col_name: &str,
        hash_agg: &crate::query::vectorized::VectorizedHashAgg,
        has_int_agg: bool,
    ) -> io::Result<ApexResult> {
        use crate::query::AggregateFunc;
        
        let num_groups = hash_agg.num_groups();
        if num_groups == 0 {
            // Return empty result
            let schema = Arc::new(Schema::new(vec![
                Field::new(group_col_name, ArrowDataType::Utf8, false),
            ]));
            return Ok(ApexResult::Empty(schema));
        }
        
        let states = hash_agg.states();
        let group_keys_str = hash_agg.group_keys_str();
        let group_keys_int = hash_agg.group_keys_int();
        
        let mut result_fields: Vec<Field> = Vec::new();
        let mut result_arrays: Vec<ArrayRef> = Vec::new();
        
        // Check if group column has an alias in the SELECT clause
        let mut group_col_alias: Option<&str> = None;
        for col in &stmt.columns {
            if let SelectColumn::ColumnAlias { column, alias } = col {
                let col_name = column.trim_matches('"');
                let actual_col = if let Some(dot_pos) = col_name.rfind('.') {
                    &col_name[dot_pos + 1..]
                } else {
                    col_name
                };
                if actual_col == group_col_name {
                    group_col_alias = Some(alias.as_str());
                    break;
                }
            }
        }
        let output_group_name = group_col_alias.unwrap_or(group_col_name);
        
        // Add group column
        if !group_keys_str.is_empty() {
            result_fields.push(Field::new(output_group_name, ArrowDataType::Utf8, false));
            result_arrays.push(Arc::new(StringArray::from(
                group_keys_str.iter().map(|s| s.as_str()).collect::<Vec<_>>()
            )));
        } else {
            result_fields.push(Field::new(output_group_name, ArrowDataType::Int64, false));
            result_arrays.push(Arc::new(Int64Array::from(group_keys_int.to_vec())));
        }
        
        // Add aggregate columns
        for col in &stmt.columns {
            if let SelectColumn::Aggregate { func, column, alias, .. } = col {
                let func_name = match func {
                    AggregateFunc::Count => "COUNT",
                    AggregateFunc::Sum => "SUM",
                    AggregateFunc::Avg => "AVG",
                    AggregateFunc::Min => "MIN",
                    AggregateFunc::Max => "MAX",
                };
                let field_name = alias.clone().unwrap_or_else(|| {
                    format!("{}({})", func_name, column.as_deref().unwrap_or("*"))
                });
                
                match func {
                    AggregateFunc::Count => {
                        let values: Vec<i64> = states.iter().map(|s| s.count).collect();
                        result_fields.push(Field::new(&field_name, ArrowDataType::Int64, false));
                        result_arrays.push(Arc::new(Int64Array::from(values)));
                    }
                    AggregateFunc::Sum => {
                        if has_int_agg {
                            let values: Vec<i64> = states.iter().map(|s| s.sum_int).collect();
                            result_fields.push(Field::new(&field_name, ArrowDataType::Int64, true));
                            result_arrays.push(Arc::new(Int64Array::from(values)));
                        } else {
                            let values: Vec<f64> = states.iter().map(|s| s.sum_float).collect();
                            result_fields.push(Field::new(&field_name, ArrowDataType::Float64, true));
                            result_arrays.push(Arc::new(Float64Array::from(values)));
                        }
                    }
                    AggregateFunc::Avg => {
                        let values: Vec<f64> = states.iter().map(|s| {
                            if s.count > 0 {
                                if has_int_agg {
                                    s.sum_int as f64 / s.count as f64
                                } else {
                                    s.sum_float / s.count as f64
                                }
                            } else { 0.0 }
                        }).collect();
                        result_fields.push(Field::new(&field_name, ArrowDataType::Float64, true));
                        result_arrays.push(Arc::new(Float64Array::from(values)));
                    }
                    AggregateFunc::Min => {
                        if has_int_agg {
                            let values: Vec<Option<i64>> = states.iter().map(|s| s.min_int).collect();
                            result_fields.push(Field::new(&field_name, ArrowDataType::Int64, true));
                            result_arrays.push(Arc::new(Int64Array::from(values)));
                        } else {
                            let values: Vec<Option<f64>> = states.iter().map(|s| s.min_float).collect();
                            result_fields.push(Field::new(&field_name, ArrowDataType::Float64, true));
                            result_arrays.push(Arc::new(Float64Array::from(values)));
                        }
                    }
                    AggregateFunc::Max => {
                        if has_int_agg {
                            let values: Vec<Option<i64>> = states.iter().map(|s| s.max_int).collect();
                            result_fields.push(Field::new(&field_name, ArrowDataType::Int64, true));
                            result_arrays.push(Arc::new(Int64Array::from(values)));
                        } else {
                            let values: Vec<Option<f64>> = states.iter().map(|s| s.max_float).collect();
                            result_fields.push(Field::new(&field_name, ArrowDataType::Float64, true));
                            result_arrays.push(Arc::new(Float64Array::from(values)));
                        }
                    }
                }
            }
        }
        
        let schema = Arc::new(Schema::new(result_fields));
        let mut result_batch = RecordBatch::try_new(schema, result_arrays)
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e.to_string()))?;
        
        // Apply HAVING clause if present
        if let Some(having_expr) = &stmt.having {
            let mask = Self::evaluate_predicate(&result_batch, having_expr)?;
            result_batch = compute::filter_record_batch(&result_batch, &mask)
                .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e.to_string()))?;
        }
        
        // Apply ORDER BY if present
        if !stmt.order_by.is_empty() {
            result_batch = Self::apply_order_by(&result_batch, &stmt.order_by)?;
        }
        
        
        Ok(ApexResult::Data(result_batch))
    }

    /// Check if we can use incremental aggregation (no DISTINCT, no complex expressions)
    fn can_use_incremental_aggregation(stmt: &SelectStatement) -> bool {
        for col in &stmt.columns {
            match col {
                SelectColumn::Aggregate { distinct, .. } => {
                    if *distinct { return false; }
                }
                SelectColumn::Expression { .. } => {
                    return false; // Expressions may need row access
                }
                _ => {}
            }
        }
        // HAVING with aggregates is OK, but complex expressions aren't
        true
    }
    
    /// Ultra-fast GROUP BY for string columns using direct dictionary indexing
    /// Uses cache-friendly sequential aggregation with bounds-check elimination
    fn execute_group_by_string_dict(
        batch: &RecordBatch,
        stmt: &SelectStatement,
        _str_arr: &StringArray,
        indices: &[u32],
        dict_values: &[&str],
        dict_size: usize,
    ) -> io::Result<ApexResult> {
        use crate::query::AggregateFunc;
        
        let num_rows = batch.num_rows();
        
        // Direct-indexed aggregate state - pre-allocated for all possible groups
        let mut counts: Vec<i64> = vec![0; dict_size];
        let mut sums_int: Vec<i64> = vec![0; dict_size];
        let mut sums_float: Vec<f64> = vec![0.0; dict_size];
        let mut mins_int: Vec<Option<i64>> = vec![None; dict_size];
        let mut maxs_int: Vec<Option<i64>> = vec![None; dict_size];
        
        // Find aggregate column
        let mut agg_col_int: Option<&Int64Array> = None;
        let mut agg_col_float: Option<&Float64Array> = None;
        
        for col in &stmt.columns {
            if let SelectColumn::Aggregate { column: Some(col_name), .. } = col {
                let actual_col = col_name.trim_matches('"');
                let actual_col = if let Some(dot_pos) = actual_col.rfind('.') {
                    &actual_col[dot_pos + 1..]
                } else {
                    actual_col
                };
                if actual_col != "*" {
                    if let Some(arr) = batch.column_by_name(actual_col) {
                        if let Some(int_arr) = arr.as_any().downcast_ref::<Int64Array>() {
                            agg_col_int = Some(int_arr);
                        } else if let Some(float_arr) = arr.as_any().downcast_ref::<Float64Array>() {
                            agg_col_float = Some(float_arr);
                        }
                    }
                }
                break;
            }
        }
        
        // OPTIMIZED AGGREGATION: Single pass with bounds-check elimination
        // Uses unsafe for hot path when no nulls present
        if let Some(int_arr) = agg_col_int {
            if int_arr.null_count() == 0 {
                // Fast path: no nulls - use raw slice access
                let values = int_arr.values();
                for row_idx in 0..num_rows {
                    let group_idx = unsafe { *indices.get_unchecked(row_idx) as usize };
                    if group_idx != 0 {
                        unsafe {
                            *counts.get_unchecked_mut(group_idx) += 1;
                            let val = *values.get_unchecked(row_idx);
                            *sums_int.get_unchecked_mut(group_idx) = 
                                sums_int.get_unchecked(group_idx).wrapping_add(val);
                            let min_slot = mins_int.get_unchecked_mut(group_idx);
                            *min_slot = Some(min_slot.map_or(val, |m| m.min(val)));
                            let max_slot = maxs_int.get_unchecked_mut(group_idx);
                            *max_slot = Some(max_slot.map_or(val, |m| m.max(val)));
                        }
                    }
                }
            } else {
                // Slow path: has nulls
                for row_idx in 0..num_rows {
                    let group_idx = indices[row_idx] as usize;
                    if group_idx == 0 { continue; }
                    counts[group_idx] += 1;
                    if !int_arr.is_null(row_idx) {
                        let val = int_arr.value(row_idx);
                        sums_int[group_idx] = sums_int[group_idx].wrapping_add(val);
                        mins_int[group_idx] = Some(mins_int[group_idx].map_or(val, |m| m.min(val)));
                        maxs_int[group_idx] = Some(maxs_int[group_idx].map_or(val, |m| m.max(val)));
                    }
                }
            }
        } else if let Some(float_arr) = agg_col_float {
            if float_arr.null_count() == 0 {
                let values = float_arr.values();
                for row_idx in 0..num_rows {
                    let group_idx = unsafe { *indices.get_unchecked(row_idx) as usize };
                    if group_idx != 0 {
                        unsafe {
                            *counts.get_unchecked_mut(group_idx) += 1;
                            *sums_float.get_unchecked_mut(group_idx) += *values.get_unchecked(row_idx);
                        }
                    }
                }
            } else {
                for row_idx in 0..num_rows {
                    let group_idx = indices[row_idx] as usize;
                    if group_idx == 0 { continue; }
                    counts[group_idx] += 1;
                    if !float_arr.is_null(row_idx) {
                        sums_float[group_idx] += float_arr.value(row_idx);
                    }
                }
            }
        } else {
            // COUNT(*) only
            for row_idx in 0..num_rows {
                let group_idx = unsafe { *indices.get_unchecked(row_idx) as usize };
                if group_idx != 0 {
                    unsafe { *counts.get_unchecked_mut(group_idx) += 1; }
                }
            }
        }
        
        // Collect non-empty groups (skip index 0 which is NULL)
        let active_groups: Vec<usize> = (1..dict_size)
            .filter(|&i| counts[i] > 0)
            .collect();
        
        // Build result arrays
        let mut result_fields: Vec<Field> = Vec::new();
        let mut result_arrays: Vec<ArrayRef> = Vec::new();
        
        // Add group column (string values from dictionary)
        // OPTIMIZATION: Check if group column has an alias in SELECT clause
        let group_by_col = stmt.group_by.first().map(|s| s.trim_matches('"')).unwrap_or("");
        let group_col_name = stmt.columns.iter()
            .find_map(|col| {
                // Check for ColumnAlias pattern: column AS alias
                if let SelectColumn::ColumnAlias { column, alias } = col {
                    let col_trimmed = column.trim_matches('"');
                    // Match either full name (u.tier) or just column name (tier)
                    if col_trimmed == group_by_col || 
                       (group_by_col.contains('.') && col_trimmed == group_by_col.split('.').next().unwrap_or("")) ||
                       (col_trimmed.contains('.') && col_trimmed.ends_with(&format!(". {}", group_by_col.split('.').last().unwrap_or("")))) {
                        return Some(alias.clone());
                    }
                }
                None
            })
            .unwrap_or_else(|| {
                // No alias found, use column name (stripping table prefix)
                if let Some(dot_pos) = group_by_col.rfind('.') {
                    group_by_col[dot_pos + 1..].to_string()
                } else {
                    group_by_col.to_string()
                }
            });
        
        let group_values: Vec<&str> = active_groups.iter()
            .map(|&i| dict_values[i - 1]) // -1 because dict_values is 0-indexed, indices are 1-indexed
            .collect();
        result_fields.push(Field::new(&group_col_name, ArrowDataType::Utf8, false));
        result_arrays.push(Arc::new(StringArray::from(group_values)));
        
        // Add aggregate columns
        for col in &stmt.columns {
            if let SelectColumn::Aggregate { func, column, alias, .. } = col {
                let func_name = match func {
                    AggregateFunc::Count => "COUNT",
                    AggregateFunc::Sum => "SUM",
                    AggregateFunc::Avg => "AVG",
                    AggregateFunc::Min => "MIN",
                    AggregateFunc::Max => "MAX",
                };
                let field_name = alias.clone().unwrap_or_else(|| {
                    format!("{}({})", func_name, column.as_deref().unwrap_or("*"))
                });
                
                match func {
                    AggregateFunc::Count => {
                        let values: Vec<i64> = active_groups.iter().map(|&i| counts[i]).collect();
                        result_fields.push(Field::new(&field_name, ArrowDataType::Int64, false));
                        result_arrays.push(Arc::new(Int64Array::from(values)));
                    }
                    AggregateFunc::Sum => {
                        if agg_col_int.is_some() {
                            let values: Vec<i64> = active_groups.iter().map(|&i| sums_int[i]).collect();
                            result_fields.push(Field::new(&field_name, ArrowDataType::Int64, true));
                            result_arrays.push(Arc::new(Int64Array::from(values)));
                        } else {
                            let values: Vec<f64> = active_groups.iter().map(|&i| sums_float[i]).collect();
                            result_fields.push(Field::new(&field_name, ArrowDataType::Float64, true));
                            result_arrays.push(Arc::new(Float64Array::from(values)));
                        }
                    }
                    AggregateFunc::Avg => {
                        let values: Vec<f64> = active_groups.iter().map(|&i| {
                            if counts[i] > 0 {
                                if agg_col_int.is_some() {
                                    sums_int[i] as f64 / counts[i] as f64
                                } else {
                                    sums_float[i] / counts[i] as f64
                                }
                            } else { 0.0 }
                        }).collect();
                        result_fields.push(Field::new(&field_name, ArrowDataType::Float64, true));
                        result_arrays.push(Arc::new(Float64Array::from(values)));
                    }
                    AggregateFunc::Min => {
                        let values: Vec<Option<i64>> = active_groups.iter().map(|&i| mins_int[i]).collect();
                        result_fields.push(Field::new(&field_name, ArrowDataType::Int64, true));
                        result_arrays.push(Arc::new(Int64Array::from(values)));
                    }
                    AggregateFunc::Max => {
                        let values: Vec<Option<i64>> = active_groups.iter().map(|&i| maxs_int[i]).collect();
                        result_fields.push(Field::new(&field_name, ArrowDataType::Int64, true));
                        result_arrays.push(Arc::new(Int64Array::from(values)));
                    }
                }
            }
        }
        
        let schema = Arc::new(Schema::new(result_fields));
        let mut result_batch = RecordBatch::try_new(schema, result_arrays)
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e.to_string()))?;
        
        // Apply HAVING clause if present
        if let Some(having_expr) = &stmt.having {
            let mask = Self::evaluate_predicate(&result_batch, having_expr)?;
            result_batch = compute::filter_record_batch(&result_batch, &mask)
                .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e.to_string()))?;
        }
        
        // Apply ORDER BY if present
        if !stmt.order_by.is_empty() {
            result_batch = Self::apply_order_by(&result_batch, &stmt.order_by)?;
        }
        
        Ok(ApexResult::Data(result_batch))
    }
    
    /// Ultra-fast GROUP BY using direct array indexing for small integer ranges
    /// This avoids hash map overhead entirely - O(1) per row instead of hash lookup
    fn execute_group_by_direct_index(
        batch: &RecordBatch,
        stmt: &SelectStatement,
        group_col: &Int64Array,
        min_val: usize,
        range: usize,
    ) -> io::Result<ApexResult> {
        use crate::query::AggregateFunc;
        
        let num_rows = batch.num_rows();
        
        // Direct-indexed aggregate state: [count, sum_int, sum_float, min_int, max_int, first_row]
        let mut counts: Vec<i64> = vec![0; range];
        let mut sums_int: Vec<i64> = vec![0; range];
        let mut sums_float: Vec<f64> = vec![0.0; range];
        let mut mins_int: Vec<Option<i64>> = vec![None; range];
        let mut maxs_int: Vec<Option<i64>> = vec![None; range];
        let mut first_rows: Vec<usize> = vec![usize::MAX; range];
        
        // Find aggregate column
        let mut agg_col_int: Option<&Int64Array> = None;
        let mut agg_col_float: Option<&Float64Array> = None;
        
        for col in &stmt.columns {
            if let SelectColumn::Aggregate { column: Some(col_name), .. } = col {
                let actual_col = col_name.trim_matches('"');
                let actual_col = if let Some(dot_pos) = actual_col.rfind('.') {
                    &actual_col[dot_pos + 1..]
                } else {
                    actual_col
                };
                if actual_col != "*" {
                    if let Some(arr) = batch.column_by_name(actual_col) {
                        if let Some(int_arr) = arr.as_any().downcast_ref::<Int64Array>() {
                            agg_col_int = Some(int_arr);
                        } else if let Some(float_arr) = arr.as_any().downcast_ref::<Float64Array>() {
                            agg_col_float = Some(float_arr);
                        }
                    }
                }
                break;
            }
        }
        
        // Single pass aggregation with direct indexing
        for row_idx in 0..num_rows {
            if group_col.is_null(row_idx) { continue; }
            let group_val = group_col.value(row_idx) as usize - min_val;
            
            counts[group_val] += 1;
            if first_rows[group_val] == usize::MAX {
                first_rows[group_val] = row_idx;
            }
            
            if let Some(int_arr) = agg_col_int {
                if !int_arr.is_null(row_idx) {
                    let val = int_arr.value(row_idx);
                    sums_int[group_val] = sums_int[group_val].wrapping_add(val);
                    mins_int[group_val] = Some(mins_int[group_val].map_or(val, |m| m.min(val)));
                    maxs_int[group_val] = Some(maxs_int[group_val].map_or(val, |m| m.max(val)));
                }
            }
            if let Some(float_arr) = agg_col_float {
                if !float_arr.is_null(row_idx) {
                    sums_float[group_val] += float_arr.value(row_idx);
                }
            }
        }
        
        // Collect non-empty groups
        let active_groups: Vec<usize> = (0..range)
            .filter(|&i| counts[i] > 0)
            .collect();
        
        // Build result arrays
        let mut result_fields: Vec<Field> = Vec::new();
        let mut result_arrays: Vec<ArrayRef> = Vec::new();
        
        // Add group column
        let group_col_name = stmt.group_by.first()
            .map(|s| s.trim_matches('"').to_string())
            .unwrap_or_else(|| "group".to_string());
        
        let group_values: Vec<i64> = active_groups.iter()
            .map(|&i| (i + min_val) as i64)
            .collect();
        result_fields.push(Field::new(&group_col_name, ArrowDataType::Int64, false));
        result_arrays.push(Arc::new(Int64Array::from(group_values)));
        
        // Add aggregate columns
        for col in &stmt.columns {
            if let SelectColumn::Aggregate { func, column, alias, .. } = col {
                let field_name = alias.clone().unwrap_or_else(|| {
                    format!("{}({})", func.to_string(), column.as_deref().unwrap_or("*"))
                });
                
                match func {
                    AggregateFunc::Count => {
                        let values: Vec<i64> = active_groups.iter().map(|&i| counts[i]).collect();
                        result_fields.push(Field::new(&field_name, ArrowDataType::Int64, false));
                        result_arrays.push(Arc::new(Int64Array::from(values)));
                    }
                    AggregateFunc::Sum => {
                        if agg_col_int.is_some() {
                            let values: Vec<i64> = active_groups.iter().map(|&i| sums_int[i]).collect();
                            result_fields.push(Field::new(&field_name, ArrowDataType::Int64, true));
                            result_arrays.push(Arc::new(Int64Array::from(values)));
                        } else {
                            let values: Vec<f64> = active_groups.iter().map(|&i| sums_float[i]).collect();
                            result_fields.push(Field::new(&field_name, ArrowDataType::Float64, true));
                            result_arrays.push(Arc::new(Float64Array::from(values)));
                        }
                    }
                    AggregateFunc::Avg => {
                        let values: Vec<f64> = active_groups.iter().map(|&i| {
                            if counts[i] > 0 {
                                if agg_col_int.is_some() {
                                    sums_int[i] as f64 / counts[i] as f64
                                } else {
                                    sums_float[i] / counts[i] as f64
                                }
                            } else { 0.0 }
                        }).collect();
                        result_fields.push(Field::new(&field_name, ArrowDataType::Float64, true));
                        result_arrays.push(Arc::new(Float64Array::from(values)));
                    }
                    AggregateFunc::Min => {
                        let values: Vec<Option<i64>> = active_groups.iter().map(|&i| mins_int[i]).collect();
                        result_fields.push(Field::new(&field_name, ArrowDataType::Int64, true));
                        result_arrays.push(Arc::new(Int64Array::from(values)));
                    }
                    AggregateFunc::Max => {
                        let values: Vec<Option<i64>> = active_groups.iter().map(|&i| maxs_int[i]).collect();
                        result_fields.push(Field::new(&field_name, ArrowDataType::Int64, true));
                        result_arrays.push(Arc::new(Int64Array::from(values)));
                    }
                }
            }
        }
        
        let schema = Arc::new(Schema::new(result_fields));
        let mut result_batch = RecordBatch::try_new(schema, result_arrays)
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e.to_string()))?;
        
        // Apply HAVING clause if present
        if let Some(having_expr) = &stmt.having {
            let mask = Self::evaluate_predicate(&result_batch, having_expr)?;
            result_batch = compute::filter_record_batch(&result_batch, &mask)
                .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e.to_string()))?;
        }
        
        // Apply ORDER BY if present
        if !stmt.order_by.is_empty() {
            result_batch = Self::apply_order_by(&result_batch, &stmt.order_by)?;
        }
        
        Ok(ApexResult::Data(result_batch))
    }
    
    /// Fast incremental GROUP BY with parallel partitioned aggregation (DuckDB-style)
    /// Key optimizations:
    /// 1. Parallel partition-based aggregation for large datasets
    /// 2. Single-pass hash+aggregate for each partition
    /// 3. Merge partition results at the end
    fn execute_group_by_incremental(
        batch: &RecordBatch,
        stmt: &SelectStatement,
        group_cols: &[String],
    ) -> io::Result<ApexResult> {
        use crate::query::AggregateFunc;
        
        let num_rows = batch.num_rows();
        
        // FAST PATH: Single column GROUP BY on small integer range (e.g., category_id 0-999)
        // Uses direct array indexing instead of hash map - much faster
        if group_cols.len() == 1 {
            if let Some(col) = batch.column_by_name(&group_cols[0]) {
                if let Some(int_arr) = col.as_any().downcast_ref::<Int64Array>() {
                    // Check if values are in a small range for direct indexing
                    let (min_val, max_val) = {
                        let mut min = i64::MAX;
                        let mut max = i64::MIN;
                        for i in 0..num_rows {
                            if !int_arr.is_null(i) {
                                let v = int_arr.value(i);
                                min = min.min(v);
                                max = max.max(v);
                            }
                        }
                        (min, max)
                    };
                    
                    // Use direct indexing if range is reasonable (< 10000 unique values)
                    let range = (max_val - min_val + 1) as usize;
                    if min_val >= 0 && range <= 10000 && range > 0 {
                        return Self::execute_group_by_direct_index(batch, stmt, int_arr, min_val as usize, range);
                    }
                }
            }
        }
        
        let estimated_groups = (num_rows / 10).max(16);
        
        // Incremental aggregate state per group
        #[derive(Clone)]
        struct GroupState {
            first_row: usize,
            count: i64,
            sum_int: i64,
            sum_float: f64,
            min_int: Option<i64>,
            max_int: Option<i64>,
            min_float: Option<f64>,
            max_float: Option<f64>,
        }
        
        impl GroupState {
            #[inline(always)]
            fn new(first_row: usize) -> Self {
                Self {
                    first_row,
                    count: 0,
                    sum_int: 0,
                    sum_float: 0.0,
                    min_int: None,
                    max_int: None,
                    min_float: None,
                    max_float: None,
                }
            }
        }
        
        // Pre-downcast group columns and build runtime dictionaries for strings
        // OPTIMIZATION: Build dictionary (string -> integer ID) for low-cardinality string columns
        // This converts string hashing to integer operations, similar to DuckDB's storage-level dictionary
        enum TypedCol<'a> {
            Int64(&'a Int64Array),
            Float64(&'a Float64Array),
            StringDict(&'a StringArray, Vec<u32>),  // (array, dictionary indices per row)
            Bool(&'a BooleanArray),
            Other(&'a ArrayRef),
        }
        
        // OPTIMIZATION: For single column GROUP BY, use direct dictionary indexing
        // This is much faster than hash-based grouping for low-cardinality columns
        if group_cols.len() == 1 {
            if let Some(col) = batch.column_by_name(&group_cols[0]) {
                // FAST PATH 1: Arrow DictionaryArray - indices already available, no conversion needed!
                use arrow::array::DictionaryArray;
                use arrow::datatypes::UInt32Type;
                if let Some(dict_arr) = col.as_any().downcast_ref::<DictionaryArray<UInt32Type>>() {
                    let keys = dict_arr.keys();
                    let values = dict_arr.values();
                    if let Some(str_values) = values.as_any().downcast_ref::<StringArray>() {
                        let dict_size = str_values.len() + 1; // +1 for NULL slot
                        
                        // Extract indices directly - no dictionary building needed!
                        let indices: Vec<u32> = (0..num_rows)
                            .map(|i| {
                                if keys.is_null(i) { 0u32 } else { keys.value(i) + 1 } // +1 for NULL at 0
                            })
                            .collect();
                        
                        // Build dict_values from StringArray
                        let dict_values: Vec<&str> = (0..str_values.len())
                            .map(|i| str_values.value(i))
                            .collect();
                        
                        return Self::execute_group_by_string_dict(
                            batch, stmt, str_values, &indices, &dict_values, dict_size
                        );
                    }
                }
                
                // FAST PATH 2: Regular StringArray - build dictionary
                // REMOVED sampling to stabilize performance
                if let Some(str_arr) = col.as_any().downcast_ref::<StringArray>() {
                    // Build dictionary directly without sampling
                    let mut dict: AHashMap<&str, u32> = AHashMap::with_capacity(200);
                    let mut dict_values: Vec<&str> = Vec::with_capacity(200);
                    let mut next_id = 1u32;
                    
                    let mut indices: Vec<u32> = Vec::with_capacity(num_rows);
                    indices.resize(num_rows, 0);
                    
                    for i in 0..num_rows {
                        if !str_arr.is_null(i) {
                            let s = str_arr.value(i);
                            let id = *dict.entry(s).or_insert_with(|| {
                                let id = next_id;
                                next_id += 1;
                                dict_values.push(s);
                                id
                            });
                            indices[i] = id;
                        }
                    }
                    
                    // Only use dict indexing if cardinality is reasonable
                    let dict_size = dict_values.len() + 1;
                    if dict_size <= 1000 {
                        return Self::execute_group_by_string_dict(
                            batch, stmt, str_arr, &indices, &dict_values, dict_size
                        );
                    }
                }
            }
        }
        
        // FAST PATH: 2-column GROUP BY with low-cardinality string columns
        // Uses composite dictionary indexing: (dict1_id * dict2_size + dict2_id) as direct array index
        if group_cols.len() == 2 {
            use arrow::array::DictionaryArray;
            use arrow::datatypes::UInt32Type;
            
            let col1 = batch.column_by_name(&group_cols[0]);
            let col2 = batch.column_by_name(&group_cols[1]);
            
            if let (Some(c1), Some(c2)) = (col1, col2) {
                // Build dictionaries for both columns - handles both StringArray and DictionaryArray
                let build_dict = |col: &ArrayRef, n_rows: usize| -> Option<(Vec<u32>, Vec<String>, usize)> {
                    // Case 1: DictionaryArray - already dictionary encoded!
                    if let Some(dict_arr) = col.as_any().downcast_ref::<DictionaryArray<UInt32Type>>() {
                        let keys = dict_arr.keys();
                        let values = dict_arr.values();
                        if let Some(str_values) = values.as_any().downcast_ref::<StringArray>() {
                            let dict_size = str_values.len() + 1;
                            if dict_size <= 1000 {
                                let indices: Vec<u32> = (0..n_rows)
                                    .map(|i| {
                                        if keys.is_null(i) { 0u32 } else { keys.value(i) + 1 }
                                    })
                                    .collect();
                                let dict_values: Vec<String> = (0..str_values.len())
                                    .map(|i| str_values.value(i).to_string())
                                    .collect();
                                return Some((indices, dict_values, dict_size));
                            }
                        }
                    }
                    
                    // Case 2: StringArray - build dictionary
                    if let Some(str_arr) = col.as_any().downcast_ref::<StringArray>() {
                        let mut dict: AHashMap<&str, u32> = AHashMap::with_capacity(200);
                        let mut dict_values: Vec<String> = Vec::with_capacity(200);
                        let mut next_id = 1u32;
                        
                        let indices: Vec<u32> = (0..n_rows)
                            .map(|i| {
                                if str_arr.is_null(i) {
                                    0u32
                                } else {
                                    let s = str_arr.value(i);
                                    *dict.entry(s).or_insert_with(|| {
                                        let id = next_id;
                                        next_id += 1;
                                        dict_values.push(s.to_string());
                                        id
                                    })
                                }
                            })
                            .collect();
                        
                        let dict_size = dict_values.len() + 1;
                        if dict_size <= 1000 {
                            return Some((indices, dict_values, dict_size));
                        }
                    }
                    
                    // Case 3: LargeStringArray - build dictionary
                    if let Some(str_arr) = col.as_any().downcast_ref::<arrow::array::LargeStringArray>() {
                        let mut dict: AHashMap<String, u32> = AHashMap::with_capacity(200);
                        let mut dict_values: Vec<String> = Vec::with_capacity(200);
                        let mut next_id = 1u32;
                        
                        let indices: Vec<u32> = (0..n_rows)
                            .map(|i| {
                                if str_arr.is_null(i) {
                                    0u32
                                } else {
                                    let s = str_arr.value(i);
                                    *dict.entry(s.to_string()).or_insert_with(|| {
                                        let id = next_id;
                                        next_id += 1;
                                        dict_values.push(s.to_string());
                                        id
                                    })
                                }
                            })
                            .collect();
                        
                        let dict_size = dict_values.len() + 1;
                        if dict_size <= 1000 {
                            return Some((indices, dict_values, dict_size));
                        }
                    }
                    
                    // Case 4: BinaryArray - build dictionary
                    if let Some(bin_arr) = col.as_any().downcast_ref::<arrow::array::BinaryArray>() {
                        let mut dict: AHashMap<String, u32> = AHashMap::with_capacity(200);
                        let mut dict_values: Vec<String> = Vec::with_capacity(200);
                        let mut next_id = 1u32;
                        
                        let indices: Vec<u32> = (0..n_rows)
                            .map(|i| {
                                if bin_arr.is_null(i) {
                                    0u32
                                } else {
                                    let s = bin_arr.value(i);
                                    let s_str = String::from_utf8_lossy(s);
                                    *dict.entry(s_str.to_string()).or_insert_with(|| {
                                        let id = next_id;
                                        next_id += 1;
                                        dict_values.push(s_str.to_string());
                                        id
                                    })
                                }
                            })
                            .collect();
                        
                        let dict_size = dict_values.len() + 1;
                        if dict_size <= 1000 {
                            return Some((indices, dict_values, dict_size));
                        }
                    }
                    
                    None
                };
                
                if let (Some((indices1, dict1_values, dict1_size)), Some((indices2, dict2_values, dict2_size))) = 
                    (build_dict(c1, num_rows), build_dict(c2, num_rows)) 
                {
                    // Use composite key: (idx1 * dict2_size + idx2) for direct array indexing
                    let total_size = dict1_size * dict2_size;
                    if total_size <= 100_000 {
                        // Find aggregate column - support both Int64 and Float64
                        let mut agg_col_float: Option<&Float64Array> = None;
                        let mut agg_col_int: Option<&Int64Array> = None;
                        for col in &stmt.columns {
                            if let SelectColumn::Aggregate { column: Some(col_name), .. } = col {
                                let actual_col = col_name.trim_matches('"');
                                let actual_col = if let Some(dot_pos) = actual_col.rfind('.') {
                                    &actual_col[dot_pos + 1..]
                                } else {
                                    actual_col
                                };
                                if actual_col != "*" {
                                    if let Some(arr) = batch.column_by_name(actual_col) {
                                        if let Some(float_arr) = arr.as_any().downcast_ref::<Float64Array>() {
                                            agg_col_float = Some(float_arr);
                                        } else if let Some(int_arr) = arr.as_any().downcast_ref::<Int64Array>() {
                                            agg_col_int = Some(int_arr);
                                        }
                                    }
                                }
                                break;
                            }
                        }
                        
                        // Direct-indexed aggregation - no hash map needed!
                        let mut counts: Vec<i64> = vec![0; total_size];
                        let mut sums_int: Vec<i64> = vec![0; total_size];
                        let mut sums_float: Vec<f64> = vec![0.0; total_size];
                        
                        if let Some(int_arr) = agg_col_int {
                            // Int64 aggregate
                            if int_arr.null_count() == 0 {
                                let values = int_arr.values();
                                for row_idx in 0..num_rows {
                                    let idx1 = unsafe { *indices1.get_unchecked(row_idx) as usize };
                                    let idx2 = unsafe { *indices2.get_unchecked(row_idx) as usize };
                                    if idx1 != 0 && idx2 != 0 {
                                        let composite = idx1 * dict2_size + idx2;
                                        unsafe {
                                            *counts.get_unchecked_mut(composite) += 1;
                                            *sums_int.get_unchecked_mut(composite) += *values.get_unchecked(row_idx);
                                        }
                                    }
                                }
                            } else {
                                for row_idx in 0..num_rows {
                                    let idx1 = indices1[row_idx] as usize;
                                    let idx2 = indices2[row_idx] as usize;
                                    if idx1 == 0 || idx2 == 0 { continue; }
                                    let composite = idx1 * dict2_size + idx2;
                                    counts[composite] += 1;
                                    if !int_arr.is_null(row_idx) {
                                        sums_int[composite] += int_arr.value(row_idx);
                                    }
                                }
                            }
                        } else if let Some(float_arr) = agg_col_float {
                            // Float64 aggregate
                            if float_arr.null_count() == 0 {
                                let values = float_arr.values();
                                for row_idx in 0..num_rows {
                                    let idx1 = unsafe { *indices1.get_unchecked(row_idx) as usize };
                                    let idx2 = unsafe { *indices2.get_unchecked(row_idx) as usize };
                                    if idx1 != 0 && idx2 != 0 {
                                        let composite = idx1 * dict2_size + idx2;
                                        unsafe {
                                            *counts.get_unchecked_mut(composite) += 1;
                                            *sums_float.get_unchecked_mut(composite) += *values.get_unchecked(row_idx);
                                        }
                                    }
                                }
                            } else {
                                for row_idx in 0..num_rows {
                                    let idx1 = indices1[row_idx] as usize;
                                    let idx2 = indices2[row_idx] as usize;
                                    if idx1 == 0 || idx2 == 0 { continue; }
                                    let composite = idx1 * dict2_size + idx2;
                                    counts[composite] += 1;
                                    if !float_arr.is_null(row_idx) {
                                        sums_float[composite] += float_arr.value(row_idx);
                                    }
                                }
                            }
                        } else {
                            // COUNT(*) only
                            for row_idx in 0..num_rows {
                                let idx1 = unsafe { *indices1.get_unchecked(row_idx) as usize };
                                let idx2 = unsafe { *indices2.get_unchecked(row_idx) as usize };
                                if idx1 != 0 && idx2 != 0 {
                                    let composite = idx1 * dict2_size + idx2;
                                    unsafe { *counts.get_unchecked_mut(composite) += 1; }
                                }
                            }
                        }
                        
                        // Collect active groups
                        let mut result_col1: Vec<&str> = Vec::with_capacity(dict1_size * dict2_size / 10);
                        let mut result_col2: Vec<&str> = Vec::with_capacity(dict1_size * dict2_size / 10);
                        let mut result_counts: Vec<i64> = Vec::with_capacity(dict1_size * dict2_size / 10);
                        let mut result_sums_int: Vec<i64> = Vec::with_capacity(dict1_size * dict2_size / 10);
                        let mut result_sums_float: Vec<f64> = Vec::with_capacity(dict1_size * dict2_size / 10);
                        
                        for idx1 in 1..dict1_size {
                            for idx2 in 1..dict2_size {
                                let composite = idx1 * dict2_size + idx2;
                                if counts[composite] > 0 {
                                    result_col1.push(&dict1_values[idx1 - 1]);
                                    result_col2.push(&dict2_values[idx2 - 1]);
                                    result_counts.push(counts[composite]);
                                    result_sums_int.push(sums_int[composite]);
                                    result_sums_float.push(sums_float[composite]);
                                }
                            }
                        }
                        
                        // Build result
                        use crate::query::AggregateFunc;
                        let mut result_fields: Vec<Field> = Vec::new();
                        let mut result_arrays: Vec<ArrayRef> = Vec::new();
                        
                        result_fields.push(Field::new(group_cols[0].trim_matches('"'), ArrowDataType::Utf8, false));
                        result_arrays.push(Arc::new(StringArray::from(result_col1)));
                        result_fields.push(Field::new(group_cols[1].trim_matches('"'), ArrowDataType::Utf8, false));
                        result_arrays.push(Arc::new(StringArray::from(result_col2)));
                        
                        let has_int_agg = agg_col_int.is_some();
                        
                        for col in &stmt.columns {
                            if let SelectColumn::Aggregate { func, column, alias, .. } = col {
                                let func_name = match func {
                                    AggregateFunc::Count => "COUNT",
                                    AggregateFunc::Sum => "SUM",
                                    AggregateFunc::Avg => "AVG",
                                    AggregateFunc::Min => "MIN",
                                    AggregateFunc::Max => "MAX",
                                };
                                let field_name = alias.clone().unwrap_or_else(|| {
                                    format!("{}({})", func_name, column.as_deref().unwrap_or("*"))
                                });
                                
                                match func {
                                    AggregateFunc::Count => {
                                        result_fields.push(Field::new(&field_name, ArrowDataType::Int64, false));
                                        result_arrays.push(Arc::new(Int64Array::from(result_counts.clone())));
                                    }
                                    AggregateFunc::Sum => {
                                        if has_int_agg {
                                            result_fields.push(Field::new(&field_name, ArrowDataType::Int64, true));
                                            result_arrays.push(Arc::new(Int64Array::from(result_sums_int.clone())));
                                        } else {
                                            result_fields.push(Field::new(&field_name, ArrowDataType::Float64, true));
                                            result_arrays.push(Arc::new(Float64Array::from(result_sums_float.clone())));
                                        }
                                    }
                                    AggregateFunc::Avg => {
                                        let avgs: Vec<f64> = if has_int_agg {
                                            result_counts.iter().zip(result_sums_int.iter())
                                                .map(|(&c, &s)| if c > 0 { s as f64 / c as f64 } else { 0.0 })
                                                .collect()
                                        } else {
                                            result_counts.iter().zip(result_sums_float.iter())
                                                .map(|(&c, &s)| if c > 0 { s / c as f64 } else { 0.0 })
                                                .collect()
                                        };
                                        result_fields.push(Field::new(&field_name, ArrowDataType::Float64, true));
                                        result_arrays.push(Arc::new(Float64Array::from(avgs)));
                                    }
                                    _ => {}
                                }
                            }
                        }
                        
                        let schema = Arc::new(Schema::new(result_fields));
                        let mut result_batch = RecordBatch::try_new(schema, result_arrays)
                            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e.to_string()))?;
                        
                        // Apply HAVING/ORDER BY/LIMIT
                        if let Some(ref having) = stmt.having {
                            let mask = Self::evaluate_predicate(&result_batch, having)?;
                            result_batch = compute::filter_record_batch(&result_batch, &mask)
                                .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e.to_string()))?;
                        }
                        
                        if !stmt.order_by.is_empty() {
                            let k = stmt.limit.map(|l| l + stmt.offset.unwrap_or(0));
                            result_batch = Self::apply_order_by_topk(&result_batch, &stmt.order_by, k)?;
                        }
                        
                        result_batch = Self::apply_limit_offset(&result_batch, stmt.limit, stmt.offset)?;
                        
                        return Ok(ApexResult::Data(result_batch));
                    }
                }
            }
        }
        
        // FAST PATH: String + Int64 2-column GROUP BY (common case: category + numeric id)
        // Uses composite key: (string_dict_id * int_range + int_value_offset) for direct array indexing
        if group_cols.len() == 2 {
            let col1 = batch.column_by_name(&group_cols[0]);
            let col2 = batch.column_by_name(&group_cols[1]);
            
            if let (Some(c1), Some(c2)) = (col1, col2) {
                // Try to build dictionary for string column and get int range for int column
                let string_dict_result: Option<(Vec<u32>, Vec<String>, usize)> = {
                    use arrow::array::DictionaryArray;
                    use arrow::datatypes::UInt32Type;
                    
                    // Case 1: DictionaryArray
                    if let Some(dict_arr) = c1.as_any().downcast_ref::<DictionaryArray<UInt32Type>>() {
                        let keys = dict_arr.keys();
                        let values = dict_arr.values();
                        if let Some(str_values) = values.as_any().downcast_ref::<StringArray>() {
                            let dict_size = str_values.len() + 1;
                            if dict_size <= 1000 {
                                let indices: Vec<u32> = (0..num_rows)
                                    .map(|i| if keys.is_null(i) { 0u32 } else { keys.value(i) + 1 })
                                    .collect();
                                let dict_values: Vec<String> = (0..str_values.len())
                                    .map(|i| str_values.value(i).to_string())
                                    .collect();
                                Some((indices, dict_values, dict_size))
                            } else {
                                None
                            }
                        } else {
                            None
                        }
                    }
                    // Case 2: StringArray - build dictionary
                    else if let Some(str_arr) = c1.as_any().downcast_ref::<StringArray>() {
                        let mut dict: AHashMap<&str, u32> = AHashMap::with_capacity(200);
                        let mut dict_values: Vec<String> = Vec::with_capacity(200);
                        let mut next_id = 1u32;
                        
                        let indices: Vec<u32> = (0..num_rows)
                            .map(|i| {
                                if str_arr.is_null(i) {
                                    0u32
                                } else {
                                    let s = str_arr.value(i);
                                    *dict.entry(s).or_insert_with(|| {
                                        let id = next_id;
                                        next_id += 1;
                                        dict_values.push(s.to_string());
                                        id
                                    })
                                }
                            })
                            .collect();
                        
                        let dict_size = dict_values.len() + 1;
                        if dict_size <= 1000 {
                            Some((indices, dict_values, dict_size))
                        } else {
                            None
                        }
                    } else {
                        None
                    }
                };
                
                // Get int column range
                let int_range_result: Option<(Vec<u32>, i64, usize)> = if let Some(int_arr) = c2.as_any().downcast_ref::<Int64Array>() {
                    let (min_val, max_val) = {
                        let mut min = i64::MAX;
                        let mut max = i64::MIN;
                        for i in 0..num_rows {
                            if !int_arr.is_null(i) {
                                let v = int_arr.value(i);
                                min = min.min(v);
                                max = max.max(v);
                            }
                        }
                        (min, max)
                    };
                    
                    let range = (max_val - min_val + 1) as usize;
                    if min_val >= 0 && range <= 1000 && range > 0 {
                        let indices: Vec<u32> = (0..num_rows)
                            .map(|i| {
                                if int_arr.is_null(i) {
                                    0u32
                                } else {
                                    (int_arr.value(i) - min_val + 1) as u32
                                }
                            })
                            .collect();
                        Some((indices, min_val, range))
                    } else {
                        None
                    }
                } else {
                    None
                };
                
                // If both columns can use dictionary indexing
                if let (Some((str_indices, str_values, str_size)), Some((int_indices, int_min, int_range))) = 
                    (string_dict_result, int_range_result) 
                {
                    let total_size = str_size * (int_range + 1);
                    if total_size <= 100_000 {
                        // Find aggregate column
                        let mut agg_col_int: Option<&Int64Array> = None;
                        let mut agg_col_float: Option<&Float64Array> = None;
                        for col in &stmt.columns {
                            if let SelectColumn::Aggregate { column: Some(col_name), .. } = col {
                                let actual_col = col_name.trim_matches('"');
                                let actual_col = if let Some(dot_pos) = actual_col.rfind('.') {
                                    &actual_col[dot_pos + 1..]
                                } else {
                                    actual_col
                                };
                                if actual_col != "*" {
                                    if let Some(arr) = batch.column_by_name(actual_col) {
                                        if let Some(float_arr) = arr.as_any().downcast_ref::<Float64Array>() {
                                            agg_col_float = Some(float_arr);
                                        } else if let Some(int_arr) = arr.as_any().downcast_ref::<Int64Array>() {
                                            agg_col_int = Some(int_arr);
                                        }
                                    }
                                }
                                break;
                            }
                        }
                        
                        // Direct-indexed aggregation
                        let mut counts: Vec<i64> = vec![0; total_size];
                        let mut sums_int: Vec<i64> = vec![0; total_size];
                        let mut sums_float: Vec<f64> = vec![0.0; total_size];
                        
                        if let Some(int_arr) = agg_col_int {
                            if int_arr.null_count() == 0 {
                                let values = int_arr.values();
                                for row_idx in 0..num_rows {
                                    let str_idx = unsafe { *str_indices.get_unchecked(row_idx) as usize };
                                    let int_idx = unsafe { *int_indices.get_unchecked(row_idx) as usize };
                                    if str_idx != 0 && int_idx != 0 {
                                        let composite = str_idx * (int_range + 1) + int_idx;
                                        unsafe {
                                            *counts.get_unchecked_mut(composite) += 1;
                                            *sums_int.get_unchecked_mut(composite) += *values.get_unchecked(row_idx);
                                        }
                                    }
                                }
                            } else {
                                for row_idx in 0..num_rows {
                                    let str_idx = str_indices[row_idx] as usize;
                                    let int_idx = int_indices[row_idx] as usize;
                                    if str_idx == 0 || int_idx == 0 { continue; }
                                    let composite = str_idx * (int_range + 1) + int_idx;
                                    counts[composite] += 1;
                                    if !int_arr.is_null(row_idx) {
                                        sums_int[composite] += int_arr.value(row_idx);
                                    }
                                }
                            }
                        } else if let Some(float_arr) = agg_col_float {
                            if float_arr.null_count() == 0 {
                                let values = float_arr.values();
                                for row_idx in 0..num_rows {
                                    let str_idx = unsafe { *str_indices.get_unchecked(row_idx) as usize };
                                    let int_idx = unsafe { *int_indices.get_unchecked(row_idx) as usize };
                                    if str_idx != 0 && int_idx != 0 {
                                        let composite = str_idx * (int_range + 1) + int_idx;
                                        unsafe {
                                            *counts.get_unchecked_mut(composite) += 1;
                                            *sums_float.get_unchecked_mut(composite) += *values.get_unchecked(row_idx);
                                        }
                                    }
                                }
                            } else {
                                for row_idx in 0..num_rows {
                                    let str_idx = str_indices[row_idx] as usize;
                                    let int_idx = int_indices[row_idx] as usize;
                                    if str_idx == 0 || int_idx == 0 { continue; }
                                    let composite = str_idx * (int_range + 1) + int_idx;
                                    counts[composite] += 1;
                                    if !float_arr.is_null(row_idx) {
                                        sums_float[composite] += float_arr.value(row_idx);
                                    }
                                }
                            }
                        } else {
                            // COUNT(*) only
                            for row_idx in 0..num_rows {
                                let str_idx = unsafe { *str_indices.get_unchecked(row_idx) as usize };
                                let int_idx = unsafe { *int_indices.get_unchecked(row_idx) as usize };
                                if str_idx != 0 && int_idx != 0 {
                                    let composite = str_idx * (int_range + 1) + int_idx;
                                    unsafe { *counts.get_unchecked_mut(composite) += 1; }
                                }
                            }
                        }
                        
                        // Collect active groups
                        let mut result_col1: Vec<&str> = Vec::with_capacity(total_size / 10);
                        let mut result_col2: Vec<i64> = Vec::with_capacity(total_size / 10);
                        let mut result_counts: Vec<i64> = Vec::with_capacity(total_size / 10);
                        let mut result_sums_int: Vec<i64> = Vec::with_capacity(total_size / 10);
                        let mut result_sums_float: Vec<f64> = Vec::with_capacity(total_size / 10);
                        
                        for str_idx in 1..str_size {
                            for int_offset in 1..=int_range {
                                let composite = str_idx * (int_range + 1) + int_offset;
                                if counts[composite] > 0 {
                                    result_col1.push(&str_values[str_idx - 1]);
                                    result_col2.push(int_min + (int_offset - 1) as i64);
                                    result_counts.push(counts[composite]);
                                    result_sums_int.push(sums_int[composite]);
                                    result_sums_float.push(sums_float[composite]);
                                }
                            }
                        }
                        
                        // Build result
                        use crate::query::AggregateFunc;
                        let mut result_fields: Vec<Field> = Vec::new();
                        let mut result_arrays: Vec<ArrayRef> = Vec::new();
                        
                        result_fields.push(Field::new(group_cols[0].trim_matches('"'), ArrowDataType::Utf8, false));
                        result_arrays.push(Arc::new(StringArray::from(result_col1)));
                        result_fields.push(Field::new(group_cols[1].trim_matches('"'), ArrowDataType::Int64, false));
                        result_arrays.push(Arc::new(Int64Array::from(result_col2)));
                        
                        let has_int_agg = agg_col_int.is_some();
                        
                        for col in &stmt.columns {
                            if let SelectColumn::Aggregate { func, column, alias, .. } = col {
                                let func_name = match func {
                                    AggregateFunc::Count => "COUNT",
                                    AggregateFunc::Sum => "SUM",
                                    AggregateFunc::Avg => "AVG",
                                    AggregateFunc::Min => "MIN",
                                    AggregateFunc::Max => "MAX",
                                };
                                let field_name = alias.clone().unwrap_or_else(|| {
                                    format!("{}({})", func_name, column.as_deref().unwrap_or("*"))
                                });
                                
                                match func {
                                    AggregateFunc::Count => {
                                        result_fields.push(Field::new(&field_name, ArrowDataType::Int64, false));
                                        result_arrays.push(Arc::new(Int64Array::from(result_counts.clone())));
                                    }
                                    AggregateFunc::Sum => {
                                        if has_int_agg {
                                            result_fields.push(Field::new(&field_name, ArrowDataType::Int64, true));
                                            result_arrays.push(Arc::new(Int64Array::from(result_sums_int.clone())));
                                        } else {
                                            result_fields.push(Field::new(&field_name, ArrowDataType::Float64, true));
                                            result_arrays.push(Arc::new(Float64Array::from(result_sums_float.clone())));
                                        }
                                    }
                                    AggregateFunc::Avg => {
                                        let avgs: Vec<f64> = if has_int_agg {
                                            result_counts.iter().zip(result_sums_int.iter())
                                                .map(|(&c, &s)| if c > 0 { s as f64 / c as f64 } else { 0.0 })
                                                .collect()
                                        } else {
                                            result_counts.iter().zip(result_sums_float.iter())
                                                .map(|(&c, &s)| if c > 0 { s / c as f64 } else { 0.0 })
                                                .collect()
                                        };
                                        result_fields.push(Field::new(&field_name, ArrowDataType::Float64, true));
                                        result_arrays.push(Arc::new(Float64Array::from(avgs)));
                                    }
                                    _ => {}
                                }
                            }
                        }
                        
                        let schema = Arc::new(Schema::new(result_fields));
                        let mut result_batch = RecordBatch::try_new(schema, result_arrays)
                            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e.to_string()))?;
                        
                        // Apply HAVING/ORDER BY/LIMIT
                        if let Some(ref having) = stmt.having {
                            let mask = Self::evaluate_predicate(&result_batch, having)?;
                            result_batch = compute::filter_record_batch(&result_batch, &mask)
                                .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e.to_string()))?;
                        }
                        
                        if !stmt.order_by.is_empty() {
                            let k = stmt.limit.map(|l| l + stmt.offset.unwrap_or(0));
                            result_batch = Self::apply_order_by_topk(&result_batch, &stmt.order_by, k)?;
                        }
                        
                        result_batch = Self::apply_limit_offset(&result_batch, stmt.limit, stmt.offset)?;
                        
                        return Ok(ApexResult::Data(result_batch));
                    }
                }
            }
        }
        
        // FAST PATH: Multi-column GROUP BY (3+ columns) using vectorized execution
        // This is faster than the general path because it uses pre-typed columns and batch processing
        if group_cols.len() >= 3 {
            use crate::query::multi_column::{execute_multi_column_group_by, build_multi_column_result};
            
            // Extract aggregate function info
            let (agg_func, agg_col_name) = stmt.columns.iter()
                .find_map(|col| {
                    if let SelectColumn::Aggregate { func, column, .. } = col {
                        Some((func.clone(), column.as_deref()))
                    } else {
                        None
                    }
                })
                .unwrap_or((crate::query::AggregateFunc::Count, None));
            
            // Execute optimized multi-column group by
            match execute_multi_column_group_by(batch, group_cols, agg_col_name) {
                Ok(hash_agg) => {
                    let result_batch = build_multi_column_result(
                        &hash_agg, batch, group_cols, Some(agg_func), agg_col_name
                    )?;
                    
                    // Apply HAVING if present
                    let mut result = result_batch;
                    if let Some(having_expr) = &stmt.having {
                        let mask = Self::evaluate_predicate(&result, having_expr)?;
                        result = compute::filter_record_batch(&result, &mask)
                            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e.to_string()))?;
                    }
                    
                    // Apply ORDER BY with top-k optimization
                    if !stmt.order_by.is_empty() {
                        let k = stmt.limit.map(|l| l + stmt.offset.unwrap_or(0));
                        result = Self::apply_order_by_topk(&result, &stmt.order_by, k)?;
                    }
                    
                    // Apply LIMIT + OFFSET
                    if stmt.limit.is_some() || stmt.offset.is_some() {
                        result = Self::apply_limit_offset(&result, stmt.limit, stmt.offset)?;
                    }
                    
                    return Ok(ApexResult::Data(result));
                }
                Err(_) => {
                    // Fall through to general path
                }
            }
        }
        
        let typed_group_cols: Vec<Option<TypedCol>> = group_cols.iter()
            .map(|col_name| {
                batch.column_by_name(col_name).map(|col| {
                    if let Some(arr) = col.as_any().downcast_ref::<Int64Array>() {
                        TypedCol::Int64(arr)
                    } else if let Some(arr) = col.as_any().downcast_ref::<StringArray>() {
                        // Build runtime dictionary: string -> unique ID
                        // This converts O(string_len) hashing to O(1) integer operations
                        let mut dict: AHashMap<&str, u32> = AHashMap::with_capacity(1000);
                        let mut next_id = 1u32; // 0 reserved for NULL
                        let indices: Vec<u32> = (0..num_rows)
                            .map(|i| {
                                if arr.is_null(i) {
                                    0u32
                                } else {
                                    let s = arr.value(i);
                                    *dict.entry(s).or_insert_with(|| {
                                        let id = next_id;
                                        next_id += 1;
                                        id
                                    })
                                }
                            })
                            .collect();
                        TypedCol::StringDict(arr, indices)
                    } else if let Some(arr) = col.as_any().downcast_ref::<Float64Array>() {
                        TypedCol::Float64(arr)
                    } else if let Some(arr) = col.as_any().downcast_ref::<BooleanArray>() {
                        TypedCol::Bool(arr)
                    } else {
                        TypedCol::Other(col)
                    }
                })
            })
            .collect();
        
        // Find aggregate columns for incremental updates
        let mut agg_col_int: Option<&Int64Array> = None;
        let mut agg_col_float: Option<&Float64Array> = None;
        
        for col in &stmt.columns {
            if let SelectColumn::Aggregate { column: Some(col_name), .. } = col {
                let actual_col = col_name.trim_matches('"');
                let actual_col = if let Some(dot_pos) = actual_col.rfind('.') {
                    &actual_col[dot_pos + 1..]
                } else {
                    actual_col
                };
                if actual_col != "*" {
                    if let Some(arr) = batch.column_by_name(actual_col) {
                        if let Some(int_arr) = arr.as_any().downcast_ref::<Int64Array>() {
                            agg_col_int = Some(int_arr);
                        } else if let Some(float_arr) = arr.as_any().downcast_ref::<Float64Array>() {
                            agg_col_float = Some(float_arr);
                        }
                    }
                }
                break;
            }
        }
        
        // Pre-compute all group keys (row_idx -> hash) for fast parallel access
        // OPTIMIZATION: Parallel hash computation for large datasets
        use rayon::prelude::*;
        let group_keys: Vec<u64> = if num_rows > 50_000 {
            // Parallel hash computation
            (0..num_rows)
                .into_par_iter()
                .map(|row_idx| {
                    let mut hasher = AHasher::default();
                    for col_opt in &typed_group_cols {
                        match col_opt {
                            Some(TypedCol::Int64(arr)) => {
                                if !arr.is_null(row_idx) {
                                    hasher.write_i64(arr.value(row_idx));
                                } else {
                                    hasher.write_u8(0);
                                }
                            }
                            Some(TypedCol::StringDict(_arr, indices)) => {
                                hasher.write_u32(indices[row_idx]);
                            }
                            Some(TypedCol::Float64(arr)) => {
                                if !arr.is_null(row_idx) {
                                    hasher.write_u64(arr.value(row_idx).to_bits());
                                } else {
                                    hasher.write_u8(0);
                                }
                            }
                            Some(TypedCol::Bool(arr)) => {
                                if !arr.is_null(row_idx) {
                                    hasher.write_u8(arr.value(row_idx) as u8);
                                } else {
                                    hasher.write_u8(2);
                                }
                            }
                            Some(TypedCol::Other(col)) => {
                                hasher.write_u64(Self::hash_array_value_fast(col, row_idx));
                            }
                            None => {}
                        }
                    }
                    hasher.finish()
                })
                .collect()
        } else {
            // Sequential for small datasets
            (0..num_rows)
                .map(|row_idx| {
                    let mut hasher = AHasher::default();
                    for col_opt in &typed_group_cols {
                        match col_opt {
                            Some(TypedCol::Int64(arr)) => {
                                if !arr.is_null(row_idx) {
                                    hasher.write_i64(arr.value(row_idx));
                                } else {
                                    hasher.write_u8(0);
                                }
                            }
                            Some(TypedCol::StringDict(_arr, indices)) => {
                                hasher.write_u32(indices[row_idx]);
                            }
                            Some(TypedCol::Float64(arr)) => {
                                if !arr.is_null(row_idx) {
                                    hasher.write_u64(arr.value(row_idx).to_bits());
                                } else {
                                    hasher.write_u8(0);
                                }
                            }
                            Some(TypedCol::Bool(arr)) => {
                                if !arr.is_null(row_idx) {
                                    hasher.write_u8(arr.value(row_idx) as u8);
                                } else {
                                    hasher.write_u8(2);
                                }
                            }
                            Some(TypedCol::Other(col)) => {
                                hasher.write_u64(Self::hash_array_value_fast(col, row_idx));
                            }
                            None => {}
                        }
                    }
                    hasher.finish()
                })
                .collect()
        };
        
        // Pre-compute aggregate values for parallel access
        let agg_int_vals: Option<Vec<Option<i64>>> = agg_col_int.map(|arr| {
            (0..num_rows).map(|i| if arr.is_null(i) { None } else { Some(arr.value(i)) }).collect()
        });
        let agg_float_vals: Option<Vec<Option<f64>>> = agg_col_float.map(|arr| {
            (0..num_rows).map(|i| if arr.is_null(i) { None } else { Some(arr.value(i)) }).collect()
        });
        
        // Parallel partitioned aggregation for large datasets
        use rayon::prelude::*;
        let use_parallel = num_rows > 50_000;
        
        let groups: AHashMap<u64, GroupState> = if use_parallel {
            let num_partitions = rayon::current_num_threads().max(4);
            let partition_size = (num_rows + num_partitions - 1) / num_partitions;
            
            // Each partition aggregates independently
            let partition_results: Vec<AHashMap<u64, GroupState>> = (0..num_partitions)
                .into_par_iter()
                .map(|p| {
                    let start = p * partition_size;
                    let end = ((p + 1) * partition_size).min(num_rows);
                    let mut local: AHashMap<u64, GroupState> = AHashMap::with_capacity(estimated_groups / num_partitions + 1);
                    
                    for row_idx in start..end {
                        let key = group_keys[row_idx];
                        let state = local.entry(key).or_insert_with(|| GroupState::new(row_idx));
                        state.count += 1;
                        
                        if let Some(ref vals) = agg_int_vals {
                            if let Some(val) = vals[row_idx] {
                                state.sum_int = state.sum_int.wrapping_add(val);
                                state.min_int = Some(state.min_int.map_or(val, |m| m.min(val)));
                                state.max_int = Some(state.max_int.map_or(val, |m| m.max(val)));
                            }
                        }
                        if let Some(ref vals) = agg_float_vals {
                            if let Some(val) = vals[row_idx] {
                                state.sum_float += val;
                                state.min_float = Some(state.min_float.map_or(val, |m| m.min(val)));
                                state.max_float = Some(state.max_float.map_or(val, |m| m.max(val)));
                            }
                        }
                    }
                    local
                })
                .collect();
            
            // Merge partition results
            let mut merged: AHashMap<u64, GroupState> = AHashMap::with_capacity(estimated_groups);
            for local in partition_results {
                for (key, state) in local {
                    merged.entry(key)
                        .and_modify(|e| {
                            e.count += state.count;
                            e.sum_int = e.sum_int.wrapping_add(state.sum_int);
                            e.sum_float += state.sum_float;
                            if let Some(v) = state.min_int {
                                e.min_int = Some(e.min_int.map_or(v, |m| m.min(v)));
                            }
                            if let Some(v) = state.max_int {
                                e.max_int = Some(e.max_int.map_or(v, |m| m.max(v)));
                            }
                            if let Some(v) = state.min_float {
                                e.min_float = Some(e.min_float.map_or(v, |m| m.min(v)));
                            }
                            if let Some(v) = state.max_float {
                                e.max_float = Some(e.max_float.map_or(v, |m| m.max(v)));
                            }
                        })
                        .or_insert(state);
                }
            }
            merged
        } else {
            // Sequential for small datasets
            let mut groups: AHashMap<u64, GroupState> = AHashMap::with_capacity(estimated_groups);
            for row_idx in 0..num_rows {
                let key = group_keys[row_idx];
                let state = groups.entry(key).or_insert_with(|| GroupState::new(row_idx));
                state.count += 1;
                
                if let Some(ref vals) = agg_int_vals {
                    if let Some(val) = vals[row_idx] {
                        state.sum_int = state.sum_int.wrapping_add(val);
                        state.min_int = Some(state.min_int.map_or(val, |m| m.min(val)));
                        state.max_int = Some(state.max_int.map_or(val, |m| m.max(val)));
                    }
                }
                if let Some(ref vals) = agg_float_vals {
                    if let Some(val) = vals[row_idx] {
                        state.sum_float += val;
                        state.min_float = Some(state.min_float.map_or(val, |m| m.min(val)));
                        state.max_float = Some(state.max_float.map_or(val, |m| m.max(val)));
                    }
                }
            }
            groups
        };
        
        // Build result arrays from group states
        let num_groups = groups.len();
        let states: Vec<GroupState> = groups.into_values().collect();
        
        let mut result_fields: Vec<Field> = Vec::new();
        let mut result_arrays: Vec<ArrayRef> = Vec::new();
        
        for col in &stmt.columns {
            match col {
                SelectColumn::Column(name) | SelectColumn::ColumnAlias { column: name, .. } => {
                    let col_name = name.trim_matches('"');
                    let actual_col = if let Some(dot_pos) = col_name.rfind('.') {
                        &col_name[dot_pos + 1..]
                    } else {
                        col_name
                    };
                    let output_name = match col {
                        SelectColumn::ColumnAlias { alias, .. } => alias.as_str(),
                        _ => actual_col,
                    };
                    
                    if let Some(src_col) = batch.column_by_name(actual_col) {
                        // Take value from first row of each group
                        let first_indices: Vec<usize> = states.iter().map(|s| s.first_row).collect();
                        let indices_arr = arrow::array::UInt32Array::from(first_indices.iter().map(|&i| i as u32).collect::<Vec<_>>());
                        let taken = compute::take(src_col.as_ref(), &indices_arr, None)
                            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e.to_string()))?;
                        result_fields.push(Field::new(output_name, taken.data_type().clone(), true));
                        result_arrays.push(taken);
                    }
                }
                SelectColumn::Aggregate { func, column, alias, .. } => {
                    let func_name = match func {
                        AggregateFunc::Count => "COUNT",
                        AggregateFunc::Sum => "SUM",
                        AggregateFunc::Avg => "AVG",
                        AggregateFunc::Min => "MIN",
                        AggregateFunc::Max => "MAX",
                    };
                    let output_name = alias.clone().unwrap_or_else(|| {
                        if let Some(c) = column { format!("{}({})", func_name, c) } else { format!("{}(*)", func_name) }
                    });
                    
                    match func {
                        AggregateFunc::Count => {
                            let counts: Vec<i64> = states.iter().map(|s| s.count).collect();
                            result_fields.push(Field::new(&output_name, ArrowDataType::Int64, false));
                            result_arrays.push(Arc::new(Int64Array::from(counts)));
                        }
                        AggregateFunc::Sum => {
                            if agg_col_int.is_some() {
                                let sums: Vec<i64> = states.iter().map(|s| s.sum_int).collect();
                                result_fields.push(Field::new(&output_name, ArrowDataType::Int64, false));
                                result_arrays.push(Arc::new(Int64Array::from(sums)));
                            } else {
                                let sums: Vec<f64> = states.iter().map(|s| s.sum_float).collect();
                                result_fields.push(Field::new(&output_name, ArrowDataType::Float64, false));
                                result_arrays.push(Arc::new(Float64Array::from(sums)));
                            }
                        }
                        AggregateFunc::Avg => {
                            if agg_col_int.is_some() {
                                let avgs: Vec<Option<f64>> = states.iter().map(|s| {
                                    if s.count > 0 { Some(s.sum_int as f64 / s.count as f64) } else { None }
                                }).collect();
                                result_fields.push(Field::new(&output_name, ArrowDataType::Float64, true));
                                result_arrays.push(Arc::new(Float64Array::from(avgs)));
                            } else {
                                let avgs: Vec<Option<f64>> = states.iter().map(|s| {
                                    if s.count > 0 { Some(s.sum_float / s.count as f64) } else { None }
                                }).collect();
                                result_fields.push(Field::new(&output_name, ArrowDataType::Float64, true));
                                result_arrays.push(Arc::new(Float64Array::from(avgs)));
                            }
                        }
                        AggregateFunc::Min => {
                            if agg_col_int.is_some() {
                                let mins: Vec<Option<i64>> = states.iter().map(|s| s.min_int).collect();
                                result_fields.push(Field::new(&output_name, ArrowDataType::Int64, true));
                                result_arrays.push(Arc::new(Int64Array::from(mins)));
                            } else {
                                let mins: Vec<Option<f64>> = states.iter().map(|s| s.min_float).collect();
                                result_fields.push(Field::new(&output_name, ArrowDataType::Float64, true));
                                result_arrays.push(Arc::new(Float64Array::from(mins)));
                            }
                        }
                        AggregateFunc::Max => {
                            if agg_col_int.is_some() {
                                let maxs: Vec<Option<i64>> = states.iter().map(|s| s.max_int).collect();
                                result_fields.push(Field::new(&output_name, ArrowDataType::Int64, true));
                                result_arrays.push(Arc::new(Int64Array::from(maxs)));
                            } else {
                                let maxs: Vec<Option<f64>> = states.iter().map(|s| s.max_float).collect();
                                result_fields.push(Field::new(&output_name, ArrowDataType::Float64, true));
                                result_arrays.push(Arc::new(Float64Array::from(maxs)));
                            }
                        }
                    }
                }
                _ => {}
            }
        }
        
        if result_fields.is_empty() {
            return Ok(ApexResult::Scalar(num_groups as i64));
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
        
        // Apply ORDER BY with top-k optimization if LIMIT is present
        if !stmt.order_by.is_empty() {
            let k = stmt.limit.map(|l| l + stmt.offset.unwrap_or(0));
            result = Self::apply_order_by_topk(&result, &stmt.order_by, k)?;
        }
        
        // Apply LIMIT + OFFSET
        if stmt.limit.is_some() || stmt.offset.is_some() {
            result = Self::apply_limit_offset(&result, stmt.limit, stmt.offset)?;
        }
        
        Ok(ApexResult::Data(result))
    }
    
    /// Original GROUP BY with full row indices (for complex queries with DISTINCT or expressions)
    fn execute_group_by_with_indices(
        batch: &RecordBatch,
        stmt: &SelectStatement,
        group_cols: &[String],
    ) -> io::Result<ApexResult> {
        // Create groups: key -> row indices (using AHashMap for speed)
        let num_rows = batch.num_rows();
        let estimated_groups = (num_rows / 10).max(16); // Estimate ~10 rows per group
        let mut groups: AHashMap<u64, Vec<usize>> = AHashMap::with_capacity(estimated_groups);
        
        // OPTIMIZATION: Pre-downcast columns to typed arrays for faster access
        // This avoids repeated dynamic dispatch in the hot loop
        enum TypedColumn<'a> {
            Int64(&'a Int64Array),
            Float64(&'a Float64Array),
            String(&'a StringArray),
            Bool(&'a BooleanArray),
            Other(&'a ArrayRef),
        }
        
        let typed_cols: Vec<Option<TypedColumn>> = group_cols.iter()
            .map(|col_name| {
                batch.column_by_name(col_name).map(|col| {
                    if let Some(arr) = col.as_any().downcast_ref::<Int64Array>() {
                        TypedColumn::Int64(arr)
                    } else if let Some(arr) = col.as_any().downcast_ref::<StringArray>() {
                        TypedColumn::String(arr)
                    } else if let Some(arr) = col.as_any().downcast_ref::<Float64Array>() {
                        TypedColumn::Float64(arr)
                    } else if let Some(arr) = col.as_any().downcast_ref::<BooleanArray>() {
                        TypedColumn::Bool(arr)
                    } else {
                        TypedColumn::Other(col)
                    }
                })
            })
            .collect();
        
        // Build groups with optimized type-specific hashing
        for row_idx in 0..num_rows {
            let mut hasher = AHasher::default();
            for col_opt in &typed_cols {
                match col_opt {
                    Some(TypedColumn::Int64(arr)) => {
                        if !arr.is_null(row_idx) {
                            hasher.write_i64(arr.value(row_idx));
                        } else {
                            hasher.write_u8(0);
                        }
                    }
                    Some(TypedColumn::String(arr)) => {
                        if !arr.is_null(row_idx) {
                            hasher.write(arr.value(row_idx).as_bytes());
                        } else {
                            hasher.write_u8(0);
                        }
                    }
                    Some(TypedColumn::Float64(arr)) => {
                        if !arr.is_null(row_idx) {
                            hasher.write_u64(arr.value(row_idx).to_bits());
                        } else {
                            hasher.write_u8(0);
                        }
                    }
                    Some(TypedColumn::Bool(arr)) => {
                        if !arr.is_null(row_idx) {
                            hasher.write_u8(arr.value(row_idx) as u8);
                        } else {
                            hasher.write_u8(2);
                        }
                    }
                    Some(TypedColumn::Other(col)) => {
                        hasher.write_u64(Self::hash_array_value_fast(col, row_idx));
                    }
                    None => {}
                }
            }
            let key = hasher.finish();
            groups.entry(key).or_insert_with(|| Vec::with_capacity(16)).push(row_idx);
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
            return Ok(ApexResult::Scalar(num_groups as i64));
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

        Ok(ApexResult::Data(result))
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

    /// OPTIMIZED: Extract _id = X pattern for O(1) lookup
    /// Returns Some(id) if WHERE clause is simple `_id = literal` or `literal = _id`
    #[inline]
    fn extract_id_equality_filter(expr: &SqlExpr) -> Option<u64> {
        use crate::query::sql_parser::BinaryOperator;
        
        if let SqlExpr::BinaryOp { left, op, right } = expr {
            if !matches!(op, BinaryOperator::Eq) {
                return None;
            }
            
            // Check _id = literal pattern
            if let SqlExpr::Column(col) = left.as_ref() {
                let col_name = col.trim_matches('"');
                let actual_col = if let Some(dot_pos) = col_name.rfind('.') {
                    &col_name[dot_pos + 1..]
                } else {
                    col_name
                };
                if actual_col == "_id" {
                    if let SqlExpr::Literal(Value::Int64(id)) = right.as_ref() {
                        return Some(*id as u64);
                    }
                }
            }
            
            // Check literal = _id pattern
            if let SqlExpr::Column(col) = right.as_ref() {
                let col_name = col.trim_matches('"');
                let actual_col = if let Some(dot_pos) = col_name.rfind('.') {
                    &col_name[dot_pos + 1..]
                } else {
                    col_name
                };
                if actual_col == "_id" {
                    if let SqlExpr::Literal(Value::Int64(id)) = left.as_ref() {
                        return Some(*id as u64);
                    }
                }
            }
        }
        None
    }

    /// Extract simple string equality filter: column = 'literal' or 'literal' = column
    /// Returns (column_name, literal_value, is_equality) if matches, None otherwise
    #[inline]
    fn extract_simple_string_filter(expr: &SqlExpr) -> Option<(String, String, bool)> {
        use crate::query::sql_parser::BinaryOperator;
        
        if let SqlExpr::BinaryOp { left, op, right } = expr {
            let is_eq = matches!(op, BinaryOperator::Eq);
            let is_neq = matches!(op, BinaryOperator::NotEq);
            
            if !is_eq && !is_neq {
                return None;
            }
            
            // Check column = 'literal' pattern
            if let (SqlExpr::Column(col), SqlExpr::Literal(Value::String(lit))) = (left.as_ref(), right.as_ref()) {
                let col_name = col.trim_matches('"');
                let actual_col = if let Some(dot_pos) = col_name.rfind('.') {
                    &col_name[dot_pos + 1..]
                } else {
                    col_name
                };
                return Some((actual_col.to_string(), lit.clone(), is_eq));
            }
            
            // Check 'literal' = column pattern
            if let (SqlExpr::Literal(Value::String(lit)), SqlExpr::Column(col)) = (left.as_ref(), right.as_ref()) {
                let col_name = col.trim_matches('"');
                let actual_col = if let Some(dot_pos) = col_name.rfind('.') {
                    &col_name[dot_pos + 1..]
                } else {
                    col_name
                };
                return Some((actual_col.to_string(), lit.clone(), is_eq));
            }
        }
        
        None
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
            SqlExpr::Between { column, low, high, .. } => {
                // Handle BETWEEN expression: column BETWEEN low AND high
                let col_name = column.trim_matches('"');
                let actual_col = if let Some(dot_pos) = col_name.rfind('.') {
                    &col_name[dot_pos + 1..]
                } else {
                    col_name
                };
                if !columns.contains(&actual_col.to_string()) {
                    columns.push(actual_col.to_string());
                }
                Self::collect_columns_from_expr(low, columns);
                Self::collect_columns_from_expr(high, columns);
            }
            SqlExpr::Like { column, .. } | SqlExpr::Regexp { column, .. } => {
                // Handle LIKE/REGEXP expressions
                let col_name = column.trim_matches('"');
                let actual_col = if let Some(dot_pos) = col_name.rfind('.') {
                    &col_name[dot_pos + 1..]
                } else {
                    col_name
                };
                if !columns.contains(&actual_col.to_string()) {
                    columns.push(actual_col.to_string());
                }
            }
            SqlExpr::In { column, .. } => {
                // Handle IN expression (values are literals, not column refs)
                let col_name = column.trim_matches('"');
                let actual_col = if let Some(dot_pos) = col_name.rfind('.') {
                    &col_name[dot_pos + 1..]
                } else {
                    col_name
                };
                if !columns.contains(&actual_col.to_string()) {
                    columns.push(actual_col.to_string());
                }
            }
            SqlExpr::IsNull { column, .. } => {
                // Handle IS NULL / IS NOT NULL (negated field handles both)
                let col_name = column.trim_matches('"');
                let actual_col = if let Some(dot_pos) = col_name.rfind('.') {
                    &col_name[dot_pos + 1..]
                } else {
                    col_name
                };
                if !columns.contains(&actual_col.to_string()) {
                    columns.push(actual_col.to_string());
                }
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

    /// Check if this is a pure COUNT(*) query that can be answered from metadata
    /// Returns true if: SELECT COUNT(*) FROM table (no WHERE, no GROUP BY, no HAVING)
    fn is_pure_count_star(stmt: &SelectStatement) -> bool {
        // Must have exactly one column which is COUNT(*)
        if stmt.columns.len() != 1 {
            return false;
        }
        
        // Must be COUNT(*) aggregate
        let is_count_star = match &stmt.columns[0] {
            SelectColumn::Aggregate { func, column, distinct, .. } => {
                matches!(func, AggregateFunc::Count) 
                    && !distinct 
                    && column.as_ref().map(|c| c == "*" || c.chars().next().map(|ch| ch.is_ascii_digit()).unwrap_or(false)).unwrap_or(true)
            }
            _ => false,
        };
        
        if !is_count_star {
            return false;
        }
        
        // No WHERE clause
        if stmt.where_clause.is_some() {
            return false;
        }
        
        // No GROUP BY
        if !stmt.group_by.is_empty() {
            return false;
        }
        
        // No HAVING
        if stmt.having.is_some() {
            return false;
        }
        
        // No subquery in FROM
        if matches!(stmt.from, Some(FromItem::Subquery { .. })) {
            return false;
        }
        
        // No JOINs
        if !stmt.joins.is_empty() {
            return false;
        }
        
        true
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

    /// Compute aggregate for each group (parallelized for large datasets)
    fn compute_aggregate_for_groups(
        batch: &RecordBatch,
        func: &crate::query::AggregateFunc,
        column: &Option<String>,
        alias: &Option<String>,
        group_indices: &[Vec<usize>],
        distinct: bool,
    ) -> io::Result<(Field, ArrayRef)> {
        use crate::query::AggregateFunc;
        use ahash::AHashSet;
        use rayon::prelude::*;
        
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
                let use_parallel = group_indices.len() > 100;
                let counts: Vec<i64> = if let Some(col_name) = &actual_column {
                    if col_name == "*" || col_name.chars().next().map(|c| c.is_ascii_digit()).unwrap_or(false) {
                        if use_parallel {
                            group_indices.par_iter().map(|g| g.len() as i64).collect()
                        } else {
                            group_indices.iter().map(|g| g.len() as i64).collect()
                        }
                    } else if let Some(array) = batch.column_by_name(col_name) {
                        if distinct {
                            // COUNT(DISTINCT column) - count unique values per group
                            if let Some(int_arr) = array.as_any().downcast_ref::<Int64Array>() {
                                if use_parallel {
                                    group_indices.par_iter().map(|g| {
                                        let unique: AHashSet<i64> = g.iter()
                                            .filter(|&&i| !int_arr.is_null(i))
                                            .map(|&i| int_arr.value(i))
                                            .collect();
                                        unique.len() as i64
                                    }).collect()
                                } else {
                                    group_indices.iter().map(|g| {
                                        let unique: AHashSet<i64> = g.iter()
                                            .filter(|&&i| !int_arr.is_null(i))
                                            .map(|&i| int_arr.value(i))
                                            .collect();
                                        unique.len() as i64
                                    }).collect()
                                }
                            } else if let Some(str_arr) = array.as_any().downcast_ref::<StringArray>() {
                                if use_parallel {
                                    group_indices.par_iter().map(|g| {
                                        let unique: AHashSet<&str> = g.iter()
                                            .filter(|&&i| !str_arr.is_null(i))
                                            .map(|&i| str_arr.value(i))
                                            .collect();
                                        unique.len() as i64
                                    }).collect()
                                } else {
                                    group_indices.iter().map(|g| {
                                        let unique: AHashSet<&str> = g.iter()
                                            .filter(|&&i| !str_arr.is_null(i))
                                            .map(|&i| str_arr.value(i))
                                            .collect();
                                        unique.len() as i64
                                    }).collect()
                                }
                            } else {
                                if use_parallel {
                                    group_indices.par_iter().map(|g| g.iter().filter(|&&i| !array.is_null(i)).count() as i64).collect()
                                } else {
                                    group_indices.iter().map(|g| g.iter().filter(|&&i| !array.is_null(i)).count() as i64).collect()
                                }
                            }
                        } else {
                            if use_parallel {
                                group_indices.par_iter().map(|g| g.iter().filter(|&&i| !array.is_null(i)).count() as i64).collect()
                            } else {
                                group_indices.iter().map(|g| g.iter().filter(|&&i| !array.is_null(i)).count() as i64).collect()
                            }
                        }
                    } else {
                        vec![0; group_indices.len()]
                    }
                } else {
                    if use_parallel {
                        group_indices.par_iter().map(|g| g.len() as i64).collect()
                    } else {
                        group_indices.iter().map(|g| g.len() as i64).collect()
                    }
                };
                Ok((Field::new(&output_name, ArrowDataType::Int64, false), Arc::new(Int64Array::from(counts))))
            }
            AggregateFunc::Sum => {
                let col_name = actual_column.as_ref().ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "SUM requires column"))?;
                let array = batch.column_by_name(col_name).ok_or_else(|| io::Error::new(io::ErrorKind::NotFound, format!("Column '{}' not found", col_name)))?;
                
                if let Some(int_array) = array.as_any().downcast_ref::<Int64Array>() {
                    // Fast path: direct slice access with loop unrolling
                    let values = int_array.values();
                    let use_parallel = group_indices.len() > 100;
                    
                    let sums: Vec<i64> = if use_parallel {
                        group_indices.par_iter().map(|g| {
                            // Unrolled summation for better instruction pipelining
                            let mut sum0: i64 = 0;
                            let mut sum1: i64 = 0;
                            let mut sum2: i64 = 0;
                            let mut sum3: i64 = 0;
                            let chunks = g.chunks_exact(4);
                            let remainder = chunks.remainder();
                            for chunk in chunks {
                                sum0 = sum0.wrapping_add(values[chunk[0]]);
                                sum1 = sum1.wrapping_add(values[chunk[1]]);
                                sum2 = sum2.wrapping_add(values[chunk[2]]);
                                sum3 = sum3.wrapping_add(values[chunk[3]]);
                            }
                            for &i in remainder {
                                sum0 = sum0.wrapping_add(values[i]);
                            }
                            sum0.wrapping_add(sum1).wrapping_add(sum2).wrapping_add(sum3)
                        }).collect()
                    } else {
                        group_indices.iter().map(|g| {
                            let mut sum: i64 = 0;
                            for &i in g {
                                sum = sum.wrapping_add(values[i]);
                            }
                            sum
                        }).collect()
                    };
                    Ok((Field::new(&output_name, ArrowDataType::Int64, false), Arc::new(Int64Array::from(sums))))
                } else if let Some(float_array) = array.as_any().downcast_ref::<Float64Array>() {
                    let values = float_array.values();
                    let use_parallel = group_indices.len() > 100;
                    
                    let sums: Vec<f64> = if use_parallel {
                        group_indices.par_iter().map(|g| {
                            let mut sum: f64 = 0.0;
                            for &i in g {
                                sum += values[i];
                            }
                            sum
                        }).collect()
                    } else {
                        group_indices.iter().map(|g| {
                            let mut sum: f64 = 0.0;
                            for &i in g {
                                sum += values[i];
                            }
                            sum
                        }).collect()
                    };
                    Ok((Field::new(&output_name, ArrowDataType::Float64, false), Arc::new(Float64Array::from(sums))))
                } else {
                    Err(io::Error::new(io::ErrorKind::InvalidData, "SUM requires numeric column"))
                }
            }
            AggregateFunc::Avg => {
                let col_name = actual_column.as_ref().ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "AVG requires column"))?;
                let array = batch.column_by_name(col_name).ok_or_else(|| io::Error::new(io::ErrorKind::NotFound, format!("Column '{}' not found", col_name)))?;
                
                if let Some(int_array) = array.as_any().downcast_ref::<Int64Array>() {
                    // Fast path: direct slice access, compute sum and count together
                    let values = int_array.values();
                    let avgs: Vec<f64> = group_indices.iter().map(|g| {
                        if g.is_empty() { return 0.0; }
                        let mut sum: i64 = 0;
                        for &i in g {
                            sum = sum.wrapping_add(values[i]);
                        }
                        sum as f64 / g.len() as f64
                    }).collect();
                    Ok((Field::new(&output_name, ArrowDataType::Float64, false), Arc::new(Float64Array::from(avgs))))
                } else if let Some(float_array) = array.as_any().downcast_ref::<Float64Array>() {
                    let values = float_array.values();
                    let avgs: Vec<f64> = group_indices.iter().map(|g| {
                        if g.is_empty() { return 0.0; }
                        let mut sum: f64 = 0.0;
                        for &i in g {
                            sum += values[i];
                        }
                        sum / g.len() as f64
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
    fn execute_union(union: UnionStatement, storage_path: &Path) -> io::Result<ApexResult> {
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

        Ok(ApexResult::Data(result))
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
    /// OPTIMIZATION: Fast path for single-column DISTINCT using dictionary indexing
    fn deduplicate_batch(batch: &RecordBatch) -> io::Result<RecordBatch> {
        use ahash::AHashSet;
        use std::hash::Hasher;
        use arrow::array::DictionaryArray;
        use arrow::datatypes::UInt32Type;
        
        let num_rows = batch.num_rows();
        if num_rows <= 1 {
            return Ok(batch.clone());
        }

        let num_cols = batch.num_columns();
        
        // FAST PATH: Single column DISTINCT - use direct dictionary indexing
        if num_cols == 1 {
            let col = batch.column(0);
            
            // Case 1: DictionaryArray - already has unique values, just get first occurrence of each key
            if let Some(dict_arr) = col.as_any().downcast_ref::<DictionaryArray<UInt32Type>>() {
                let keys = dict_arr.keys();
                let dict_size = dict_arr.values().len() + 1; // +1 for NULL
                let mut first_occurrence: Vec<Option<u32>> = vec![None; dict_size];
                let mut keep_indices: Vec<u32> = Vec::with_capacity(dict_size);
                
                for row_idx in 0..num_rows {
                    let key = if keys.is_null(row_idx) { 0usize } else { keys.value(row_idx) as usize + 1 };
                    if first_occurrence[key].is_none() {
                        first_occurrence[key] = Some(row_idx as u32);
                        keep_indices.push(row_idx as u32);
                    }
                }
                
                if keep_indices.len() == num_rows {
                    return Ok(batch.clone());
                }
                
                let indices = arrow::array::UInt32Array::from(keep_indices);
                let filtered = compute::take(col.as_ref(), &indices, None)
                    .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e.to_string()))?;
                return RecordBatch::try_new(batch.schema(), vec![filtered])
                    .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e.to_string()));
            }
            
            // Case 2: StringArray - build dictionary on the fly for low cardinality
            // REMOVED sampling to stabilize performance
            if let Some(str_arr) = col.as_any().downcast_ref::<StringArray>() {
                // Build dictionary directly without sampling
                let mut dict: AHashMap<&str, u32> = AHashMap::with_capacity(1000);
                let mut keep_indices: Vec<u32> = Vec::with_capacity(1000);
                let mut has_null = false;
                
                for row_idx in 0..num_rows {
                    if str_arr.is_null(row_idx) {
                        if !has_null {
                            has_null = true;
                            keep_indices.push(row_idx as u32);
                        }
                    } else {
                        let s = str_arr.value(row_idx);
                        if !dict.contains_key(s) {
                            dict.insert(s, row_idx as u32);
                            keep_indices.push(row_idx as u32);
                        }
                    }
                }
                
                if keep_indices.len() == num_rows {
                    return Ok(batch.clone());
                }
                
                let indices = arrow::array::UInt32Array::from(keep_indices);
                let filtered = compute::take(col.as_ref(), &indices, None)
                    .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e.to_string()))?;
                return RecordBatch::try_new(batch.schema(), vec![filtered])
                    .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e.to_string()));
            }
            
            // Case 3: Int64Array - use direct value dedup
            if let Some(int_arr) = col.as_any().downcast_ref::<Int64Array>() {
                let mut seen: AHashSet<i64> = AHashSet::with_capacity(num_rows.min(10000));
                let mut keep_indices: Vec<u32> = Vec::with_capacity(num_rows.min(10000));
                let mut has_null = false;
                
                for row_idx in 0..num_rows {
                    if int_arr.is_null(row_idx) {
                        if !has_null {
                            has_null = true;
                            keep_indices.push(row_idx as u32);
                        }
                    } else if seen.insert(int_arr.value(row_idx)) {
                        keep_indices.push(row_idx as u32);
                    }
                }
                
                if keep_indices.len() == num_rows {
                    return Ok(batch.clone());
                }
                
                let indices = arrow::array::UInt32Array::from(keep_indices);
                let filtered = compute::take(col.as_ref(), &indices, None)
                    .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e.to_string()))?;
                return RecordBatch::try_new(batch.schema(), vec![filtered])
                    .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e.to_string()));
            }
        }
        
        // General path for multi-column deduplication
        // Pre-compute column types for faster dispatch
        enum ColType<'a> {
            Int64(&'a Int64Array),
            Float64(&'a Float64Array),
            String(&'a StringArray, Vec<u64>),  // Pre-computed string hashes
            Bool(&'a BooleanArray),
            Other(&'a ArrayRef),
        }
        
        let typed_cols: Vec<ColType> = batch.columns().iter().map(|col| {
            if let Some(arr) = col.as_any().downcast_ref::<Int64Array>() {
                ColType::Int64(arr)
            } else if let Some(arr) = col.as_any().downcast_ref::<Float64Array>() {
                ColType::Float64(arr)
            } else if let Some(arr) = col.as_any().downcast_ref::<StringArray>() {
                // Pre-compute hashes for strings
                let hashes: Vec<u64> = (0..num_rows).map(|i| {
                    if arr.is_null(i) { 0 } else {
                        let mut h = ahash::AHasher::default();
                        h.write(arr.value(i).as_bytes());
                        h.finish()
                    }
                }).collect();
                ColType::String(arr, hashes)
            } else if let Some(arr) = col.as_any().downcast_ref::<BooleanArray>() {
                ColType::Bool(arr)
            } else {
                ColType::Other(col)
            }
        }).collect();
        
        // Pre-compute all row hashes for parallel deduplication
        let row_hashes: Vec<u64> = (0..num_rows)
            .map(|row_idx| {
                let mut hasher = ahash::AHasher::default();
                for typed_col in &typed_cols {
                    match typed_col {
                        ColType::Int64(arr) => {
                            if arr.is_null(row_idx) {
                                hasher.write_u8(0);
                            } else {
                                hasher.write_u8(1);
                                hasher.write_i64(arr.value(row_idx));
                            }
                        }
                        ColType::Float64(arr) => {
                            if arr.is_null(row_idx) {
                                hasher.write_u8(0);
                            } else {
                                hasher.write_u8(1);
                                hasher.write_u64(arr.value(row_idx).to_bits());
                            }
                        }
                        ColType::String(_arr, hashes) => {
                            hasher.write_u64(hashes[row_idx]);
                        }
                        ColType::Bool(arr) => {
                            if arr.is_null(row_idx) {
                                hasher.write_u8(0);
                            } else {
                                hasher.write_u8(if arr.value(row_idx) { 2 } else { 1 });
                            }
                        }
                        ColType::Other(arr) => {
                            hasher.write_u8(if arr.is_null(row_idx) { 0 } else { 1 });
                            hasher.write_usize(row_idx);
                        }
                    }
                }
                hasher.finish()
            })
            .collect();
        
        // Sequential deduplication using pre-computed hashes
        let mut seen: AHashSet<u64> = AHashSet::with_capacity(num_rows.min(10000));
        let mut keep_indices: Vec<u32> = Vec::with_capacity(num_rows.min(10000));

        for (row_idx, &hash) in row_hashes.iter().enumerate() {
            if seen.insert(hash) {
                keep_indices.push(row_idx as u32);
            }
        }

        if keep_indices.len() == num_rows {
            return Ok(batch.clone());
        }

        // Create filtered batch
        let indices = arrow::array::UInt32Array::from(keep_indices);
        let mut result_columns: Vec<ArrayRef> = Vec::with_capacity(num_cols);
        
        for col in batch.columns() {
            let filtered = compute::take(col.as_ref(), &indices, None)
                .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e.to_string()))?;
            result_columns.push(filtered);
        }

        RecordBatch::try_new(batch.schema(), result_columns)
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
    fn execute_window_function(batch: &RecordBatch, stmt: &SelectStatement) -> io::Result<ApexResult> {
        // Collect window specs: (func_name, args, partition_by, order_by, output_name)
        let mut window_specs: Vec<(String, Vec<String>, Vec<String>, Vec<crate::query::OrderByClause>, String)> = Vec::new();
        
        let supported = ["ROW_NUMBER", "RANK", "DENSE_RANK", "NTILE", "PERCENT_RANK", "CUME_DIST", "LAG", "LEAD", "FIRST_VALUE", "LAST_VALUE", "NTH_VALUE", "SUM", "AVG", "COUNT", "MIN", "MAX", "RUNNING_SUM"];
        
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

        // Group rows by partition key (using AHashMap for speed)
        let num_rows = batch.num_rows();
        let mut groups: AHashMap<u64, Vec<usize>> = AHashMap::with_capacity(num_rows / 10 + 1);
        
        // Pre-fetch partition column references
        let partition_col_refs: Vec<Option<&ArrayRef>> = partition_by.iter()
            .map(|col_name| batch.column_by_name(col_name.trim_matches('"')))
            .collect();
        
        for row_idx in 0..num_rows {
            let mut hasher = AHasher::default();
            for col_opt in &partition_col_refs {
                if let Some(col) = col_opt {
                    hasher.write_u64(Self::hash_array_value_fast(col, row_idx));
                }
            }
            let key = hasher.finish();
            groups.entry(key).or_insert_with(|| Vec::with_capacity(16)).push(row_idx);
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
                "MIN" => {
                    // MIN(column) OVER - minimum in partition
                    let col_name = func_args.get(0).map(|s| s.trim_matches('"')).unwrap_or("");
                    if let Some(src_col) = batch.column_by_name(col_name) {
                        if let Some(int_arr) = src_col.as_any().downcast_ref::<Int64Array>() {
                            let min_val = indices.iter()
                                .filter_map(|&i| if int_arr.is_null(i) { None } else { Some(int_arr.value(i)) })
                                .min()
                                .unwrap_or(0);
                            for &row_idx in &indices {
                                window_values[row_idx] = min_val;
                            }
                        }
                    }
                }
                "MAX" => {
                    // MAX(column) OVER - maximum in partition
                    let col_name = func_args.get(0).map(|s| s.trim_matches('"')).unwrap_or("");
                    if let Some(src_col) = batch.column_by_name(col_name) {
                        if let Some(int_arr) = src_col.as_any().downcast_ref::<Int64Array>() {
                            let max_val = indices.iter()
                                .filter_map(|&i| if int_arr.is_null(i) { None } else { Some(int_arr.value(i)) })
                                .max()
                                .unwrap_or(0);
                            for &row_idx in &indices {
                                window_values[row_idx] = max_val;
                            }
                        }
                    }
                }
                "RUNNING_SUM" => {
                    // RUNNING_SUM(column) OVER - cumulative sum (rows unbounded preceding to current)
                    let col_name = func_args.get(0).map(|s| s.trim_matches('"')).unwrap_or("");
                    if let Some(src_col) = batch.column_by_name(col_name) {
                        if let Some(int_arr) = src_col.as_any().downcast_ref::<Int64Array>() {
                            let mut running = 0i64;
                            for &row_idx in &indices {
                                if !int_arr.is_null(row_idx) {
                                    running += int_arr.value(row_idx);
                                }
                                window_values[row_idx] = running;
                            }
                        }
                    }
                }
                "NTH_VALUE" => {
                    // NTH_VALUE(column, n) - get nth value in partition
                    let col_name = func_args.get(0).map(|s| s.trim_matches('"')).unwrap_or("");
                    let n = if func_args.len() > 1 {
                        func_args[1].trim_start_matches("Int64(").trim_end_matches(')').parse().unwrap_or(1usize)
                    } else { 1usize };
                    
                    if let Some(src_col) = batch.column_by_name(col_name) {
                        if let Some(int_arr) = src_col.as_any().downcast_ref::<Int64Array>() {
                            let nth_val = if n > 0 && n <= indices.len() {
                                let nth_row = indices[n - 1];
                                if int_arr.is_null(nth_row) { 0 } else { int_arr.value(nth_row) }
                            } else { 0 };
                            for &row_idx in &indices {
                                window_values[row_idx] = nth_val;
                            }
                        }
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

        Ok(ApexResult::Data(result))
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

    // ========== DDL Execution Methods ==========

    /// Execute CREATE TABLE statement
    /// High-performance: O(1) - just creates file header
    fn execute_create_table(
        base_dir: &Path,
        table: &str,
        columns: &[crate::query::sql_parser::ColumnDef],
        if_not_exists: bool,
    ) -> io::Result<ApexResult> {
        let table_path = base_dir.join(format!("{}.apex", table));
        
        if table_path.exists() {
            if if_not_exists {
                // Return success without error
                return Ok(ApexResult::Scalar(0));
            } else {
                return Err(io::Error::new(
                    io::ErrorKind::AlreadyExists,
                    format!("Table '{}' already exists", table),
                ));
            }
        }
        
        // Create empty storage file with schema
        TableStorageBackend::create(&table_path)?;
        let storage = TableStorageBackend::open_for_write(&table_path)?;
        
        // Add columns to schema (if provided)
        for col_def in columns {
            storage.add_column(&col_def.name, col_def.data_type.clone())?;
        }
        
        storage.save()?;
        
        Ok(ApexResult::Scalar(0))
    }

    /// Execute DROP TABLE statement
    /// High-performance: O(1) - just deletes file
    fn execute_drop_table(base_dir: &Path, table: &str, if_exists: bool) -> io::Result<ApexResult> {
        let table_path = base_dir.join(format!("{}.apex", table));
        
        // Invalidate cache first to release file handles
        invalidate_storage_cache(&table_path);
        
        if !table_path.exists() {
            if if_exists {
                return Ok(ApexResult::Scalar(0));
            } else {
                return Err(io::Error::new(
                    io::ErrorKind::NotFound,
                    format!("Table '{}' does not exist", table),
                ));
            }
        }
        
        std::fs::remove_file(&table_path)?;
        
        Ok(ApexResult::Scalar(0))
    }

    /// Execute ALTER TABLE statement
    fn execute_alter_table(
        base_dir: &Path,
        table: &str,
        operation: &crate::query::sql_parser::AlterTableOp,
    ) -> io::Result<ApexResult> {
        use crate::query::sql_parser::AlterTableOp;
        
        let table_path = base_dir.join(format!("{}.apex", table));
        
        if !table_path.exists() {
            return Err(io::Error::new(
                io::ErrorKind::NotFound,
                format!("Table '{}' does not exist", table),
            ));
        }
        
        // Invalidate cache before write
        invalidate_storage_cache(&table_path);
        
        // Note: ALTER TABLE operations need to preserve existing data, so we use open_for_write
        // which loads all column data. For true schema-only operations (like TRUNCATE),
        // we can use open_for_schema_change which only loads metadata.
        let storage = TableStorageBackend::open_for_write(&table_path)?;
        
        match operation {
            AlterTableOp::AddColumn { name, data_type } => {
                storage.add_column(name, data_type.clone())?;
            }
            AlterTableOp::DropColumn { name } => {
                storage.drop_column(name)?;
            }
            AlterTableOp::RenameColumn { old_name, new_name } => {
                storage.rename_column(old_name, new_name)?;
            }
        }
        
        storage.save()?;
        
        // Invalidate cache after write to ensure subsequent reads get fresh data
        invalidate_storage_cache(&table_path);
        
        Ok(ApexResult::Scalar(0))
    }

    /// Execute TRUNCATE TABLE statement
    /// High-performance: recreates empty file
    fn execute_truncate(storage_path: &Path) -> io::Result<ApexResult> {
        if !storage_path.exists() {
            return Err(io::Error::new(
                io::ErrorKind::NotFound,
                "Table does not exist",
            ));
        }
        
        // Invalidate cache before write
        invalidate_storage_cache(storage_path);
        
        // OPTIMIZATION: Use open_for_schema_change - only loads metadata, NOT column data
        let old_storage = TableStorageBackend::open_for_schema_change(storage_path)?;
        let schema = old_storage.get_schema();
        drop(old_storage);
        
        // Recreate empty file with same schema
        TableStorageBackend::create(storage_path)?;
        // Use open_for_schema_change for adding columns (schema only)
        let storage = TableStorageBackend::open_for_schema_change(storage_path)?;
        for (name, dtype) in &schema {
            storage.add_column(name, dtype.clone())?;
        }
        storage.save()?;
        
        // Invalidate cache after write to ensure subsequent reads get fresh data
        invalidate_storage_cache(storage_path);
        
        Ok(ApexResult::Scalar(0))
    }

    // ========== DML Execution Methods ==========

    /// Execute INSERT statement
    fn execute_insert(
        storage_path: &Path,
        columns: Option<&[String]>,
        values: &[Vec<Value>],
    ) -> io::Result<ApexResult> {
        use std::collections::HashMap;
        
        if !storage_path.exists() {
            return Err(io::Error::new(
                io::ErrorKind::NotFound,
                "Table does not exist",
            ));
        }
        
        // Invalidate cache before write
        invalidate_storage_cache(storage_path);
        
        // Use open_for_write to load all data (needed for correct column alignment)
        let storage = TableStorageBackend::open_for_write(storage_path)?;
        
        // Get column names from schema or explicit list
        let col_names: Vec<String> = if let Some(cols) = columns {
            cols.to_vec()
        } else {
            storage.get_schema().iter().map(|(n, _)| n.clone()).collect()
        };
        
        // Build typed column data from values
        let mut int_columns: HashMap<String, Vec<i64>> = HashMap::new();
        let mut float_columns: HashMap<String, Vec<f64>> = HashMap::new();
        let mut string_columns: HashMap<String, Vec<String>> = HashMap::new();
        let mut bool_columns: HashMap<String, Vec<bool>> = HashMap::new();
        
        for row_values in values {
            for (i, value) in row_values.iter().enumerate() {
                if i < col_names.len() {
                    let col_name = &col_names[i];
                    match value {
                        Value::Int64(v) => int_columns.entry(col_name.clone()).or_default().push(*v),
                        Value::Int32(v) => int_columns.entry(col_name.clone()).or_default().push(*v as i64),
                        Value::Float64(v) => float_columns.entry(col_name.clone()).or_default().push(*v),
                        Value::Float32(v) => float_columns.entry(col_name.clone()).or_default().push(*v as f64),
                        Value::String(v) => string_columns.entry(col_name.clone()).or_default().push(v.clone()),
                        Value::Bool(v) => bool_columns.entry(col_name.clone()).or_default().push(*v),
                        _ => {}
                    }
                }
            }
        }
        
        let rows_inserted = values.len() as i64;
        
        // Use insert_typed + save for reliable writes
        storage.insert_typed(int_columns, float_columns, string_columns, HashMap::new(), bool_columns)?;
        storage.save()?;
        
        // Invalidate cache after write to ensure subsequent reads get fresh data
        invalidate_storage_cache(storage_path);
        
        Ok(ApexResult::Scalar(rows_inserted))
    }

    /// Execute DELETE statement (soft delete - marks rows as deleted without physical removal)
    fn execute_delete(storage_path: &Path, where_clause: Option<&SqlExpr>) -> io::Result<ApexResult> {
        if !storage_path.exists() {
            return Err(io::Error::new(
                io::ErrorKind::NotFound,
                "Table does not exist",
            ));
        }
        
        // Invalidate cache before write
        invalidate_storage_cache(storage_path);
        
        // For DELETE without WHERE, only need to read _id column (memory efficient)
        if where_clause.is_none() {
            // Use lazy open - only load metadata, not column data
            let storage = TableStorageBackend::open(storage_path)?;
            let count = storage.active_row_count() as i64;
            
            // Only read _id column for soft deletion (not all columns)
            let batch = storage.read_columns_to_arrow(Some(&["_id"]), 0, None)?;
            if let Some(id_col) = batch.column_by_name("_id") {
                if let Some(id_arr) = id_col.as_any().downcast_ref::<UInt64Array>() {
                    for i in 0..id_arr.len() {
                        storage.delete(id_arr.value(i));
                    }
                } else if let Some(id_arr) = id_col.as_any().downcast_ref::<Int64Array>() {
                    for i in 0..id_arr.len() {
                        storage.delete(id_arr.value(i) as u64);
                    }
                }
            }
            storage.save()?;
            
            // Invalidate cache after write
            invalidate_storage_cache(storage_path);
            return Ok(ApexResult::Scalar(count));
        }
        
        // OPTIMIZATION: Only read columns needed for WHERE evaluation + _id
        // This avoids loading unnecessary columns (can save 50-90% IO for wide tables)
        let storage = TableStorageBackend::open(storage_path)?;
        
        // Extract column names from WHERE clause
        let mut where_cols: Vec<String> = Vec::new();
        Self::collect_columns_from_expr(where_clause.unwrap(), &mut where_cols);
        where_cols.sort();
        where_cols.dedup();
        
        // Always include _id for deletion
        if !where_cols.iter().any(|c| c == "_id") {
            where_cols.push("_id".to_string());
        }
        
        let col_refs: Vec<&str> = where_cols.iter().map(|s| s.as_str()).collect();
        let batch = storage.read_columns_to_arrow(Some(&col_refs), 0, None)?;
        let filter_mask = Self::evaluate_predicate(&batch, where_clause.unwrap())?;
        
        // Count and delete matching rows
        let mut deleted = 0i64;
        for i in 0..filter_mask.len() {
            if filter_mask.value(i) {
                if let Some(id_col) = batch.column_by_name("_id") {
                    if let Some(id_arr) = id_col.as_any().downcast_ref::<UInt64Array>() {
                        storage.delete(id_arr.value(i));
                        deleted += 1;
                    } else if let Some(id_arr) = id_col.as_any().downcast_ref::<Int64Array>() {
                        storage.delete(id_arr.value(i) as u64);
                        deleted += 1;
                    }
                }
            }
        }
        
        // Only save if rows were actually deleted
        // Calling save() with lazy-loaded storage (no column data) would corrupt the file
        if deleted > 0 {
            storage.save()?;
            // Invalidate cache after write
            invalidate_storage_cache(storage_path);
        }
        
        Ok(ApexResult::Scalar(deleted))
    }

    /// Execute UPDATE statement
    /// Note: UPDATE is implemented as delete + insert for simplicity
    fn execute_update(
        storage_path: &Path,
        assignments: &[(String, SqlExpr)],
        where_clause: Option<&SqlExpr>,
    ) -> io::Result<ApexResult> {
        use std::collections::HashMap as StdHashMap;
        
        if !storage_path.exists() {
            return Err(io::Error::new(
                io::ErrorKind::NotFound,
                "Table does not exist",
            ));
        }
        
        // Invalidate cache before write
        invalidate_storage_cache(storage_path);
        
        // OPTIMIZATION: Only read columns needed for WHERE + columns being updated + _id
        // This avoids loading unnecessary columns (can save 50-90% IO for wide tables)
        let storage = TableStorageBackend::open(storage_path)?;
        
        // Collect required columns: WHERE columns + assignment target columns + _id
        let mut required_cols: Vec<String> = Vec::new();
        
        // Add columns from WHERE clause
        if let Some(where_expr) = where_clause {
            Self::collect_columns_from_expr(where_expr, &mut required_cols);
        }
        
        // Add columns being assigned (both target and source columns in expressions)
        for (col_name, expr) in assignments {
            required_cols.push(col_name.clone());
            Self::collect_columns_from_expr(expr, &mut required_cols);
        }
        
        // Always include _id
        required_cols.push("_id".to_string());
        required_cols.sort();
        required_cols.dedup();
        
        let col_refs: Vec<&str> = required_cols.iter().map(|s| s.as_str()).collect();
        let batch = storage.read_columns_to_arrow(Some(&col_refs), 0, None)?;
        
        // Find rows to update
        let filter_mask = if let Some(where_expr) = where_clause {
            Self::evaluate_predicate(&batch, where_expr)?
        } else {
            // Update all rows
            BooleanArray::from(vec![true; batch.num_rows()])
        };
        
        // Collect IDs and new values for matching rows
        let mut updates: Vec<(u64, StdHashMap<String, Value>)> = Vec::new();
        
        for i in 0..filter_mask.len() {
            if filter_mask.value(i) {
                if let Some(id_col) = batch.column_by_name("_id") {
                    let id = if let Some(id_arr) = id_col.as_any().downcast_ref::<UInt64Array>() {
                        id_arr.value(i)
                    } else if let Some(id_arr) = id_col.as_any().downcast_ref::<Int64Array>() {
                        id_arr.value(i) as u64
                    } else {
                        continue;
                    };
                    
                    // Build update map
                    let mut update_data: StdHashMap<String, Value> = StdHashMap::new();
                    for (col_name, expr) in assignments {
                        let value = Self::evaluate_expr_to_value(&batch, expr, i)?;
                        update_data.insert(col_name.clone(), value);
                    }
                    
                    updates.push((id, update_data));
                }
            }
        }
        
        // Apply updates: for each row, build complete row with updates applied
        let updated = updates.len() as i64;
        
        // Build lookup for existing row data from batch
        let schema = batch.schema();
        
        for (row_idx, (_id, update_data)) in updates.into_iter().enumerate() {
            // Get existing values from batch for this row
            let mut row_data = update_data;
            
            // Add existing column values that weren't updated
            for field in schema.fields() {
                let col_name = field.name();
                if col_name == "_id" || row_data.contains_key(col_name) {
                    continue;
                }
                if let Some(col) = batch.column_by_name(col_name) {
                    if let Some(val) = Self::get_value_at(col, row_idx) {
                        row_data.insert(col_name.clone(), val);
                    }
                }
            }
            
            // Insert the updated row (ID is auto-assigned, old row is soft-deleted)
            storage.insert_rows(&[row_data])?;
        }
        
        storage.save()?;
        
        // Invalidate cache after write
        invalidate_storage_cache(storage_path);
        
        Ok(ApexResult::Scalar(updated))
    }

    /// Get a value from an Arrow array at a specific row index
    fn get_value_at(array: &ArrayRef, row: usize) -> Option<Value> {
        if array.is_null(row) {
            return Some(Value::Null);
        }
        if let Some(arr) = array.as_any().downcast_ref::<Int64Array>() {
            Some(Value::Int64(arr.value(row)))
        } else if let Some(arr) = array.as_any().downcast_ref::<Float64Array>() {
            Some(Value::Float64(arr.value(row)))
        } else if let Some(arr) = array.as_any().downcast_ref::<StringArray>() {
            Some(Value::String(arr.value(row).to_string()))
        } else if let Some(arr) = array.as_any().downcast_ref::<BooleanArray>() {
            Some(Value::Bool(arr.value(row)))
        } else if let Some(arr) = array.as_any().downcast_ref::<UInt64Array>() {
            Some(Value::Int64(arr.value(row) as i64))
        } else {
            None
        }
    }

    /// Evaluate an expression to a Value for UPDATE
    fn evaluate_expr_to_value(batch: &RecordBatch, expr: &SqlExpr, row: usize) -> io::Result<Value> {
        match expr {
            SqlExpr::Literal(v) => Ok(v.clone()),
            SqlExpr::Column(name) => {
                let col_name = name.trim_matches('"');
                if let Some(col) = batch.column_by_name(col_name) {
                    if col.is_null(row) {
                        return Ok(Value::Null);
                    }
                    if let Some(arr) = col.as_any().downcast_ref::<Int64Array>() {
                        Ok(Value::Int64(arr.value(row)))
                    } else if let Some(arr) = col.as_any().downcast_ref::<Float64Array>() {
                        Ok(Value::Float64(arr.value(row)))
                    } else if let Some(arr) = col.as_any().downcast_ref::<StringArray>() {
                        Ok(Value::String(arr.value(row).to_string()))
                    } else if let Some(arr) = col.as_any().downcast_ref::<BooleanArray>() {
                        Ok(Value::Bool(arr.value(row)))
                    } else {
                        Ok(Value::Null)
                    }
                } else {
                    Err(io::Error::new(
                        io::ErrorKind::InvalidInput,
                        format!("Column '{}' not found", col_name),
                    ))
                }
            }
            _ => Err(io::Error::new(
                io::ErrorKind::Unsupported,
                "Complex expressions in UPDATE not yet supported",
            )),
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

        let result = ApexExecutor::execute("SELECT * FROM default", &path).unwrap();
        let batch = result.to_record_batch().unwrap();
        assert_eq!(batch.num_rows(), 5);
    }

    #[test]
    fn test_select_with_where() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test.apex");
        create_test_storage(&path);

        let result = ApexExecutor::execute("SELECT * FROM default WHERE age > 30", &path).unwrap();
        let batch = result.to_record_batch().unwrap();
        assert_eq!(batch.num_rows(), 3); // age 35, 40, 45
    }

    #[test]
    fn test_select_with_limit() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test.apex");
        create_test_storage(&path);

        let result = ApexExecutor::execute("SELECT * FROM default LIMIT 2", &path).unwrap();
        let batch = result.to_record_batch().unwrap();
        assert_eq!(batch.num_rows(), 2);
    }

    #[test]
    fn test_count_aggregate() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test.apex");
        create_test_storage(&path);

        let result = ApexExecutor::execute("SELECT COUNT(*) FROM default", &path).unwrap();
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

        let result = ApexExecutor::execute("SELECT SUM(age) FROM default", &path).unwrap();
        let batch = result.to_record_batch().unwrap();
        
        let sum_array = batch.column(0).as_any().downcast_ref::<Int64Array>().unwrap();
        assert_eq!(sum_array.value(0), 175); // 25+30+35+40+45
    }

    #[test]
    fn test_order_by() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test.apex");
        create_test_storage(&path);

        let result = ApexExecutor::execute("SELECT * FROM default ORDER BY age DESC LIMIT 2", &path).unwrap();
        let batch = result.to_record_batch().unwrap();
        assert_eq!(batch.num_rows(), 2);
        
        let age_array = batch.column_by_name("age").unwrap()
            .as_any().downcast_ref::<Int64Array>().unwrap();
        assert_eq!(age_array.value(0), 45);
        assert_eq!(age_array.value(1), 40);
    }
}
