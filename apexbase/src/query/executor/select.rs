// SELECT execution: fast paths, index scans, late materialization

impl ApexExecutor {
    /// Execute SELECT statement with base_dir for proper subquery table resolution
    fn execute_select_with_base_dir(mut stmt: SelectStatement, storage_path: &Path, base_dir: &Path, default_table_path: &Path) -> io::Result<ApexResult> {
        // Resolve MATCH()/FUZZY_MATCH() predicates to _id IN (...) before anything else
        if let Some(ref wc) = stmt.where_clause {
            if Self::expr_has_fts_match(wc) {
                let (_, table_name) = crate::query::executor::base_dir_and_table_pub(storage_path);
                let resolved = Self::resolve_fts_in_expr(stmt.where_clause.take().unwrap(), base_dir, &table_name)?;
                stmt.where_clause = Some(resolved);
            }
        }

        // FAST PATH: Pure COUNT(*) without WHERE/GROUP BY - O(1) from metadata
        if Self::is_pure_count_star(&stmt) {
            if !storage_path.exists() {
                let tbl = storage_path.file_stem().unwrap_or_default().to_string_lossy();
                return Err(io::Error::new(io::ErrorKind::NotFound, format!("Table '{}' does not exist", tbl)));
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
                .map_err(|e| err_data( e.to_string()))?;
            return Ok(ApexResult::Data(batch));
        }
        
        // Check for derived table (FROM subquery) - resolve table path from subquery's FROM clause
        let batch = match &stmt.from {
            Some(FromItem::TableFunction { func, file, options, .. }) => {
                Self::read_table_function(func, file, options)?
            }
            Some(FromItem::Subquery { stmt: sub_stmt, .. }) => {
                // Resolve the actual table path from the subquery's FROM clause
                let sub_path = Self::resolve_from_table_path(sub_stmt, base_dir, default_table_path);
                let sub_result = Self::execute_select_with_base_dir(*sub_stmt.clone(), &sub_path, base_dir, default_table_path)?;
                sub_result.to_record_batch()?
            }
            None => {
                // No FROM clause (e.g., SELECT 1, 1) — create a single-row virtual batch
                let schema = Arc::new(Schema::new(vec![
                    Field::new("_dummy", ArrowDataType::Int64, false),
                ]));
                RecordBatch::try_new(schema, vec![Arc::new(Int64Array::from(vec![0i64])) as ArrayRef])
                    .map_err(|e| err_data(e.to_string()))?
            }
            Some(FromItem::Table { .. }) => {
                // Normal table - read from storage
                if !storage_path.exists() {
                    let tbl = storage_path.file_stem().unwrap_or_default().to_string_lossy();
                    return Err(io::Error::new(io::ErrorKind::NotFound, format!("Table '{}' does not exist", tbl)));
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
                        let col_refs = Self::get_col_refs(&stmt);
                        backend.read_columns_to_arrow(col_refs.as_ref().map(|v| v.iter().map(|s| s.as_str()).collect::<Vec<_>>()).as_deref(), 0, None)?
                    } else {
                        // Check conditions for late materialization optimization
                        let has_aggregation_check = stmt.columns.iter().any(|col| {
                            matches!(col, SelectColumn::Aggregate { .. })
                                || matches!(col, SelectColumn::Expression { expr, .. } if Self::expr_contains_aggregate(expr))
                        });

                        // FAST PATH: Direct aggregation for simple numeric aggregates
                        // Compute COUNT/SUM/AVG/MIN/MAX directly from V4 columns (mmap or in-memory)
                        if has_aggregation_check
                            && stmt.where_clause.is_none()
                            && stmt.group_by.is_empty()
                            && stmt.joins.is_empty()
                            && !backend.has_pending_deltas()
                        {
                            if let Some(result) = Self::try_mmap_aggregation(&backend, &stmt)? {
                                return Ok(result);
                            }
                        }
                        
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
                        
                        // CBO: Use plan_with_stats() to decide execution strategy.
                        // Skip expensive index checks when CBO recommends full scan or aggregation.
                        // Also skip CBO entirely for trivial queries (no WHERE = no index possible).
                        let cbo_skip_index = if stmt.where_clause.is_none() {
                            true
                        } else {
                            let cbo_strategy = {
                                let table_key = storage_path.to_string_lossy();
                                let (bd, tname) = base_dir_and_table(storage_path);
                                let idx_mgr_arc = get_index_manager(&bd, &tname);
                                let idx_mgr = idx_mgr_arc.lock();
                                let sql_stmt = SqlStatement::Select(stmt.clone());
                                QueryPlanner::plan_with_stats(&sql_stmt, Some(&*idx_mgr), &table_key)
                            };
                            matches!(
                                cbo_strategy,
                                ExecutionStrategy::OlapFullScan
                                | ExecutionStrategy::OlapAggregation
                                | ExecutionStrategy::OlapFilteredScan
                            )
                        };

                        // FAST PATH INDEX: Check if WHERE clause can use a secondary index
                        // (skipped when CBO says full scan/aggregation is cheaper)
                        if !cbo_skip_index {
                            if let Some(ref where_clause) = stmt.where_clause {
                                if let Some(result) = Self::try_index_accelerated_read(
                                    &backend, &stmt, where_clause, base_dir, storage_path,
                                )? {
                                    return Ok(result);
                                }
                            }
                        }

                    // FAST PATH 0: Check for _id = X pattern (O(1) lookup)
                        if let Some(where_clause) = &stmt.where_clause {
                            if let Some(id) = Self::extract_id_equality_filter(where_clause) {
                                if let Some(batch) = backend.read_row_by_id_to_arrow(id)? {
                                    batch
                                } else {
                                    // Not in memory — fall through to general mmap → Arrow → WHERE filter path
                                    let batch = backend.read_columns_to_arrow(None, 0, None)?;
                                    if batch.num_rows() == 0 {
                                        backend.read_columns_to_arrow(None, 0, Some(0))?
                                    } else {
                                        // Apply WHERE filter on the mmap-read batch
                                        let filtered = Self::apply_filter_with_storage(&batch, where_clause, storage_path)?;
                                        if filtered.num_rows() == 0 {
                                            return Ok(ApexResult::Empty(filtered.schema()));
                                        }
                                        return Ok(ApexResult::Data(Self::apply_projection_with_storage(&filtered, &stmt.columns, Some(storage_path))?));
                                    }
                                }
                            } else if let Some(result) = Self::try_fast_filter_group_order(&backend, &stmt)? {
                                // FAST PATH for Complex (Filter+Group+Order) - biggest optimization
                                return Ok(result);
                            } else if can_late_materialize_where {
                                // FAST PATH 1: Try dictionary-based filter for simple string equality (with LIMIT)
                                if let Some(result) = Self::try_fast_string_filter(&backend, &stmt)? {
                                    result
                                // FAST PATH 1b: String equality without LIMIT - storage-level scan
                                } else if let Some(result) = Self::try_fast_string_filter_no_limit(&backend, &stmt)? {
                                    return Ok(ApexResult::Data(result));
                                // FAST PATH 2: Try numeric range filter for BETWEEN
                                } else if let Some(result) = Self::try_fast_numeric_range_filter(&backend, &stmt)? {
                                    result
                                // FAST PATH 3: Try combined string + numeric filter for multi-condition
                                } else if let Some(result) = Self::try_fast_multi_condition_filter(&backend, &stmt)? {
                                    result
                                } else if backend.is_mmap_only() && !backend.has_pending_deltas() {
                                    // MMAP FAST PATH: byte-level scan + point lookups
                                    if let Some(where_clause) = &stmt.where_clause {
                                        let matching_indices = if let Some((col, val)) = Self::extract_string_equality(where_clause) {
                                            backend.scan_string_filter_mmap(&col, &val, stmt.limit.map(|l| l + stmt.offset.unwrap_or(0)))?
                                        } else if let Some((col, low, high)) = Self::extract_between_range(where_clause) {
                                            backend.scan_numeric_range_mmap(&col, low, high, stmt.limit.map(|l| l + stmt.offset.unwrap_or(0)))?
                                        } else {
                                            None
                                        };
                                        if let Some(indices) = matching_indices {
                                            if indices.is_empty() {
                                                return Ok(ApexResult::Empty(Arc::new(Schema::empty())));
                                            }
                                            // Use index-based extraction for all result sizes (avoids full table scan)
                                            let batch = backend.read_columns_by_indices_to_arrow(&indices)?;
                                            return Ok(ApexResult::Data(batch));
                                        }
                                    }
                                    let filtered = Self::execute_with_late_materialization(&backend, &stmt, storage_path)?;
                                    if filtered.num_rows() == 0 {
                                        return Ok(ApexResult::Empty(filtered.schema()));
                                    }
                                    return Ok(ApexResult::Data(filtered));
                                } else {
                                    // Late materialization for SELECT * WHERE path
                                    // Return directly to avoid applying WHERE filter twice
                                    let filtered = Self::execute_with_late_materialization(&backend, &stmt, storage_path)?;
                                    if filtered.num_rows() == 0 {
                                        return Ok(ApexResult::Empty(filtered.schema()));
                                    }
                                    return Ok(ApexResult::Data(filtered));
                                }
                            } else if !stmt.group_by.is_empty() {
                                // GROUP BY with WHERE: use dict-encoded path for faster string aggregation
                                let col_refs = Self::get_col_refs(&stmt);
                                backend.read_columns_to_arrow_dict(col_refs.as_ref().map(|v| v.iter().map(|s| s.as_str()).collect::<Vec<_>>()).as_deref())?
                            } else if let Some(batch) = Self::try_numeric_predicate_pushdown(&backend, &stmt)? {
                                batch
                            } else {
                                let col_refs = Self::get_col_refs(&stmt);
                                backend.read_columns_to_arrow(col_refs.as_ref().map(|v| v.iter().map(|s| s.as_str()).collect::<Vec<_>>()).as_deref(), 0, None)?
                            }
                        } else if can_late_materialize_where {
                            // FAST PATH 1: Try dictionary-based filter for simple string equality
                            if let Some(result) = Self::try_fast_string_filter(&backend, &stmt)? {
                                result
                            // FAST PATH 1b: No-LIMIT string filter (uses mmap scan + late materialization)
                            } else if let Some(result) = Self::try_fast_string_filter_no_limit(&backend, &stmt)? {
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
                            } else if let Some(batch) = Self::try_numeric_predicate_pushdown(&backend, &stmt)? {
                                batch
                            } else {
                                let col_refs = Self::get_col_refs(&stmt);
                                backend.read_columns_to_arrow(col_refs.as_ref().map(|v| v.iter().map(|s| s.as_str()).collect::<Vec<_>>()).as_deref(), 0, None)?
                            }
                        } else if can_late_materialize_order {
                            // Late materialization for ORDER BY + LIMIT path
                            Self::execute_with_order_late_materialization(&backend, &stmt)?
                        } else {
                            let col_refs = Self::get_col_refs(&stmt);
                            let col_refs_vec: Option<Vec<&str>> = col_refs.as_ref().map(|v| v.iter().map(|s| s.as_str()).collect());
                            let can_pushdown_limit = stmt.where_clause.is_none()
                                && stmt.order_by.is_empty()
                                && stmt.group_by.is_empty()
                                && !has_aggregation_check;
                            
                            // Note: V4 fast agg disabled - Arrow clone+SIMD outperforms due to cache warming
                            
                            if !stmt.group_by.is_empty() {
                                // V4 FAST PATH: Cached GROUP BY
                                if let Some(result) = Self::try_fast_cached_group_by(&backend, &stmt)? {
                                    return Ok(result);
                                }
                                // Fallback: dict-encoded Arrow path
                                backend.read_columns_to_arrow_dict(col_refs_vec.as_deref())?
                            } else {
                                let _row_limit = if can_pushdown_limit {
                                    stmt.limit.map(|l| l + stmt.offset.unwrap_or(0))
                                } else {
                                    None
                                };
                                backend.read_columns_to_arrow(col_refs_vec.as_deref(), 0, _row_limit)?
                            }
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

        // For DISTINCT: sort without top-k limit, project, deduplicate, then limit
        // For non-DISTINCT: apply top-k sort + limit, then project
        let result = if stmt.distinct {
            let sorted = if !stmt.order_by.is_empty() {
                Self::apply_order_by(&filtered, &stmt.order_by)?
            } else {
                filtered
            };
            let projected = Self::apply_projection_with_storage(&sorted, &stmt.columns, Some(storage_path))?;
            let deduped = Self::deduplicate_batch(&projected)?;
            Self::apply_limit_offset(&deduped, stmt.limit, stmt.offset)?
        } else {
            // Apply ORDER BY with LIMIT optimization (top-k heap sort)
            let limited = if !stmt.order_by.is_empty() {
                let k = stmt.limit.map(|l| l + stmt.offset.unwrap_or(0));
                let sorted = Self::apply_order_by_topk(&filtered, &stmt.order_by, k)?;
                Self::apply_limit_offset(&sorted, stmt.limit, stmt.offset)?
            } else {
                Self::apply_limit_offset(&filtered, stmt.limit, stmt.offset)?
            };
            Self::apply_projection_with_storage(&limited, &stmt.columns, Some(storage_path))?
        };

        Ok(ApexResult::Data(result))
    }

    /// Try to use a secondary index to accelerate a SELECT query.
    /// Returns Some(result) if an index was used, None to fall through to scan paths.
    /// Only used for simple equality WHERE clauses on indexed columns (no GROUP BY/aggregation).
    fn try_index_accelerated_read(
        backend: &TableStorageBackend,
        stmt: &SelectStatement,
        where_clause: &SqlExpr,
        base_dir: &Path,
        storage_path: &Path,
    ) -> io::Result<Option<ApexResult>> {
        use crate::storage::index::index_manager::PredicateHint;
        use crate::query::sql_parser::BinaryOperator;

        // Skip index path for mmap-only: per-row reads would each do a full mmap scan
        if backend.is_mmap_only() { return Ok(None); }

        // Only use index for simple queries: no GROUP BY, no aggregation, no JOIN
        if !stmt.group_by.is_empty() || !stmt.joins.is_empty() {
            return Ok(None);
        }
        let has_agg = stmt.columns.iter().any(|c| matches!(c, SelectColumn::Aggregate { .. }));
        if has_agg {
            return Ok(None);
        }

        // Extract predicate(s): single or AND-combined predicates
        // Flatten AND chains into individual (col_name, hint) pairs
        let mut predicates: Vec<(String, PredicateHint)> = Vec::new();
        Self::extract_index_predicates(where_clause, &mut predicates);

        if predicates.is_empty() {
            return Ok(None);
        }

        // Check which predicates have indexes available
        let (bd, tname) = base_dir_and_table(storage_path);
        let idx_mgr_arc = get_index_manager(&bd, &tname);
        let mut idx_mgr = idx_mgr_arc.lock();

        // Filter to predicates that have indexes
        let indexed_preds: Vec<(String, PredicateHint)> = predicates.into_iter()
            .filter(|(col, _)| idx_mgr.has_index_on(col))
            .collect();

        if indexed_preds.is_empty() {
            return Ok(None);
        }

        // CBO: Pre-estimate selectivity using ANALYZE stats before expensive index lookup
        let table_key = storage_path.to_string_lossy();
        if let Some(stats) = get_table_stats(&table_key) {
            let selectivity = QueryPlanner::estimate_selectivity(where_clause, &stats);
            if stats.row_count > 0 && !QueryPlanner::should_use_index("", selectivity, stats.row_count) {
                return Ok(None); // CBO says full scan is cheaper, skip index lookup
            }
        }

        // Look up each indexed predicate and intersect row ID sets
        let mut row_ids: Option<Vec<u64>> = None;
        for (col_name, hint) in &indexed_preds {
            let lookup_result = idx_mgr.lookup(col_name, hint)?;
            match lookup_result {
                Some(r) => {
                    row_ids = Some(match row_ids {
                        None => r.row_ids,
                        Some(existing) => {
                            // Intersect: keep only IDs in both sets
                            let set: std::collections::HashSet<u64> = r.row_ids.into_iter().collect();
                            existing.into_iter().filter(|id| set.contains(id)).collect()
                        }
                    });
                }
                None => {
                    // Index couldn't satisfy this predicate, skip it
                    // (still use other index results if available)
                }
            }
        }

        let row_ids = match row_ids {
            Some(ids) => ids,
            None => return Ok(None),
        };

        if row_ids.is_empty() {
            let empty = backend.read_columns_to_arrow(None, 0, Some(0))?;
            return Ok(Some(ApexResult::Empty(empty.schema())));
        }

        // Read matching rows by their _ids
        // CBO: use ANALYZE stats to decide index vs full scan cost
        let total_rows = backend.active_row_count();
        let selectivity = if total_rows > 0 { row_ids.len() as f64 / total_rows as f64 } else { 1.0 };
        if !QueryPlanner::should_use_index("", selectivity, total_rows as u64) {
            return Ok(None); // Cost model says full scan is cheaper
        }

        // Covering index (index-only scan): if all SELECT columns are covered by
        // the index (_id + indexed columns), build result directly without reading
        // the base table — avoids expensive per-row table lookups.
        if let Some(covered) = Self::try_index_only_scan(stmt, &indexed_preds, &row_ids)? {
            return Ok(Some(covered));
        }

        // Read matching rows one by one and concat
        let mut batches: Vec<RecordBatch> = Vec::with_capacity(row_ids.len().min(1024));
        for &rid in &row_ids {
            if let Some(batch) = backend.read_row_by_id_to_arrow(rid)? {
                batches.push(batch);
            }
        }

        if batches.is_empty() {
            let empty = backend.read_columns_to_arrow(None, 0, Some(0))?;
            return Ok(Some(ApexResult::Empty(empty.schema())));
        }

        // Concat all batches into one
        let schema = batches[0].schema();
        let combined = arrow::compute::concat_batches(&schema, &batches)
            .map_err(|e| err_data(e.to_string()))?;

        // Apply ORDER BY if present
        let sorted = if !stmt.order_by.is_empty() {
            Self::apply_order_by(&combined, &stmt.order_by)?
        } else {
            combined
        };

        // Apply OFFSET + LIMIT
        let result = {
            let offset = stmt.offset.unwrap_or(0);
            let total = sorted.num_rows();
            if offset >= total {
                sorted.slice(0, 0)
            } else if let Some(limit) = stmt.limit {
                let end = (offset + limit).min(total);
                sorted.slice(offset, end - offset)
            } else if offset > 0 {
                sorted.slice(offset, total - offset)
            } else {
                sorted
            }
        };

        // Apply column projection if not SELECT *
        if !stmt.is_select_star() {
            let projected = Self::apply_projection(&result, &stmt.columns)?;
            if projected.num_rows() == 0 {
                return Ok(Some(ApexResult::Empty(projected.schema())));
            }
            return Ok(Some(ApexResult::Data(projected)));
        }

        if result.num_rows() == 0 {
            return Ok(Some(ApexResult::Empty(result.schema())));
        }
        Ok(Some(ApexResult::Data(result)))
    }

    /// Extract index-usable predicates from an expression (flattens AND chains).
    /// Each predicate is (column_name, PredicateHint).
    fn extract_index_predicates(expr: &SqlExpr, out: &mut Vec<(String, crate::storage::index::index_manager::PredicateHint)>) {
        use crate::query::sql_parser::BinaryOperator;
        use crate::storage::index::index_manager::PredicateHint;
        match expr {
            // AND chain: recurse into both sides
            SqlExpr::BinaryOp { left, op: BinaryOperator::And, right } => {
                Self::extract_index_predicates(left, out);
                Self::extract_index_predicates(right, out);
            }
            // col OP literal or literal OP col
            SqlExpr::BinaryOp { left, op, right } => {
                if let SqlExpr::Column(col) = left.as_ref() {
                    if col != "_id" {
                        if let Some(val) = Self::expr_to_value(right) {
                            let h = match op {
                                BinaryOperator::Eq => Some(PredicateHint::Eq(val)),
                                BinaryOperator::Gt => Some(PredicateHint::Gt(val)),
                                BinaryOperator::Ge => Some(PredicateHint::Gte(val)),
                                BinaryOperator::Lt => Some(PredicateHint::Lt(val)),
                                BinaryOperator::Le => Some(PredicateHint::Lte(val)),
                                _ => None,
                            };
                            if let Some(hint) = h { out.push((col.clone(), hint)); }
                        }
                    }
                } else if let SqlExpr::Column(col) = right.as_ref() {
                    if col != "_id" {
                        if let Some(val) = Self::expr_to_value(left) {
                            let h = match op {
                                BinaryOperator::Eq => Some(PredicateHint::Eq(val)),
                                BinaryOperator::Gt => Some(PredicateHint::Lt(val)),
                                BinaryOperator::Ge => Some(PredicateHint::Lte(val)),
                                BinaryOperator::Lt => Some(PredicateHint::Gt(val)),
                                BinaryOperator::Le => Some(PredicateHint::Gte(val)),
                                _ => None,
                            };
                            if let Some(hint) = h { out.push((col.clone(), hint)); }
                        }
                    }
                }
            }
            SqlExpr::Between { column, low, high, negated } => {
                if !negated && column != "_id" {
                    if let (Some(low_val), Some(high_val)) = (Self::expr_to_value(low), Self::expr_to_value(high)) {
                        out.push((column.clone(), PredicateHint::Range { low: low_val, high: high_val }));
                    }
                }
            }
            SqlExpr::In { column, values, negated } => {
                if !negated && column != "_id" {
                    out.push((column.clone(), PredicateHint::In(values.clone())));
                }
            }
            _ => {}
        }
    }

    /// Covering index (index-only scan): build result directly from index data
    /// when all SELECT columns are covered by {_id, indexed_columns}.
    /// For equality predicates, the column value is known from the predicate itself.
    /// Returns None if the query needs columns not available from the index.
    fn try_index_only_scan(
        stmt: &SelectStatement,
        indexed_preds: &[(String, crate::storage::index::index_manager::PredicateHint)],
        row_ids: &[u64],
    ) -> io::Result<Option<ApexResult>> {
        use std::collections::HashMap;
        // Only for non-* queries (SELECT * needs all columns)
        if stmt.is_select_star() {
            return Ok(None);
        }
        // Only for simple equality predicates (we know the exact value)
        // Collect indexed column names and their known values
        use crate::storage::index::index_manager::PredicateHint;
        let mut known_values: HashMap<String, Value> = HashMap::new();
        for (col, hint) in indexed_preds {
            match hint {
                PredicateHint::Eq(val) => { known_values.insert(col.clone(), val.clone()); }
                PredicateHint::In(_) if row_ids.len() <= 1 => {
                    return Ok(None);
                }
                _ => { return Ok(None); } // Range predicates: values vary per row
            }
        }
        if known_values.is_empty() {
            return Ok(None);
        }

        // Check if all SELECT columns are covered by {_id} ∪ {indexed columns}
        let mut need_id = false;
        let mut need_cols: Vec<String> = Vec::new();
        for col in &stmt.columns {
            match col {
                SelectColumn::Column(name) => {
                    let clean = name.trim_matches('"');
                    if clean == "_id" {
                        need_id = true;
                    } else if known_values.contains_key(clean) {
                        need_cols.push(clean.to_string());
                    } else {
                        return Ok(None); // Need a column not in index → can't cover
                    }
                }
                SelectColumn::ColumnAlias { column, .. } => {
                    let clean = column.trim_matches('"');
                    if clean == "_id" {
                        need_id = true;
                    } else if known_values.contains_key(clean) {
                        need_cols.push(clean.to_string());
                    } else {
                        return Ok(None);
                    }
                }
                SelectColumn::All => { return Ok(None); }
                _ => { return Ok(None); } // Aggregate, expression, etc.
            }
        }

        // Build Arrow RecordBatch directly from index data
        let n = row_ids.len();
        let mut fields: Vec<Field> = Vec::new();
        let mut arrays: Vec<ArrayRef> = Vec::new();

        if need_id {
            fields.push(Field::new("_id", arrow::datatypes::DataType::Int64, false));
            let id_arr: Int64Array = row_ids.iter().map(|&id| id as i64).collect();
            arrays.push(Arc::new(id_arr) as ArrayRef);
        }

        for col_name in &need_cols {
            let val = &known_values[col_name];
            match val {
                Value::Int64(v) => {
                    fields.push(Field::new(col_name, arrow::datatypes::DataType::Int64, true));
                    let arr = Int64Array::from(vec![*v; n]);
                    arrays.push(Arc::new(arr) as ArrayRef);
                }
                Value::Float64(f) => {
                    fields.push(Field::new(col_name, arrow::datatypes::DataType::Float64, true));
                    let arr = Float64Array::from(vec![*f; n]);
                    arrays.push(Arc::new(arr) as ArrayRef);
                }
                Value::String(s) => {
                    fields.push(Field::new(col_name, arrow::datatypes::DataType::Utf8, true));
                    let arr = StringArray::from(vec![s.as_str(); n]);
                    arrays.push(Arc::new(arr) as ArrayRef);
                }
                Value::Bool(b) => {
                    fields.push(Field::new(col_name, arrow::datatypes::DataType::Boolean, true));
                    let arr = BooleanArray::from(vec![*b; n]);
                    arrays.push(Arc::new(arr) as ArrayRef);
                }
                _ => { return Ok(None); } // Unsupported value type for index-only scan
            }
        }

        if fields.is_empty() {
            return Ok(None);
        }

        let schema = Arc::new(Schema::new(fields));
        let batch = RecordBatch::try_new(schema, arrays)
            .map_err(|e| err_data(e.to_string()))?;

        // Apply ORDER BY if present
        let sorted = if !stmt.order_by.is_empty() {
            Self::apply_order_by(&batch, &stmt.order_by)?
        } else {
            batch
        };

        // Apply OFFSET + LIMIT
        let result = Self::apply_limit_offset(&sorted, stmt.limit, stmt.offset)?;

        if result.num_rows() == 0 {
            return Ok(Some(ApexResult::Empty(result.schema())));
        }
        Ok(Some(ApexResult::Data(result)))
    }

    /// Convert a SqlExpr literal to a Value (for index lookup)
    fn expr_to_value(expr: &SqlExpr) -> Option<Value> {
        match expr {
            SqlExpr::Literal(v) => Some(v.clone()),
            _ => None,
        }
    }

    /// Fast path for simple string equality filters on dictionary-encoded columns
    /// Uses storage-level early termination for LIMIT queries when limit is Some
    /// Supports column projection pushdown (not limited to SELECT *)
    fn try_fast_string_filter(
        backend: &TableStorageBackend,
        stmt: &SelectStatement,
    ) -> io::Result<Option<RecordBatch>> {
        // Must have LIMIT for early termination benefit
        if stmt.limit.is_none() {
            return Ok(None);
        }
        Self::try_fast_string_filter_impl(backend, stmt, stmt.limit)
    }

    /// Fast path for string equality filters WITHOUT LIMIT
    fn try_fast_string_filter_no_limit(
        backend: &TableStorageBackend,
        stmt: &SelectStatement,
    ) -> io::Result<Option<RecordBatch>> {
        Self::try_fast_string_filter_impl(backend, stmt, None)
    }
    
    /// Unified implementation for string equality filter fast path
    fn try_fast_string_filter_impl(
        backend: &TableStorageBackend,
        stmt: &SelectStatement,
        limit: Option<usize>,
    ) -> io::Result<Option<RecordBatch>> {
        // Skip fast path if delta store has pending updates — the in-memory scan
        // bypasses DeltaMerger and would return stale data. Fall through to the
        // standard mmap read path which applies DeltaMerger overlay.
        if backend.has_pending_deltas() {
            return Ok(None);
        }

        let where_clause = match &stmt.where_clause {
            Some(w) => w,
            None => return Ok(None),
        };
        
        let (col_name, filter_value) = match Self::extract_string_equality(where_clause) {
            Some(v) => v,
            None => return Ok(None),
        };
        
        // Column projection pushdown
        let projected_cols: Option<Vec<String>> = if stmt.is_select_star() {
            None
        } else {
            Some(stmt.required_columns().unwrap_or_default())
        };
        let col_refs: Option<Vec<&str>> = projected_cols.as_ref()
            .map(|cols| cols.iter().map(|s| s.as_str()).collect());
        
        let result = if let Some(lim) = limit {
            backend.read_columns_filtered_string_with_limit_to_arrow(
                col_refs.as_deref(),
                &col_name,
                &filter_value,
                true,
                lim,
                stmt.offset.unwrap_or(0),
            )?
        } else {
            backend.read_columns_filtered_string_to_arrow(
                col_refs.as_deref(),
                &col_name,
                &filter_value,
                true,
            )?
        };
        
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
        
        if backend.has_pending_deltas() || backend.is_mmap_only() {
            return Ok(None);
        }

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
            _ => Err(err_input( "not a number")),
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
                            _ => return Err(err_input( "unsupported op")),
                        };
                        Ok((col.trim_matches('"').to_string(), flipped_op, val))
                    }
                    _ => Err(err_input( "not a comparison")),
                }
            }
            _ => Err(err_input( "not a binary op")),
        }
    }

    /// Fast path for multi-condition WHERE with string equality AND numeric comparison
    /// Handles: SELECT * WHERE string_col = 'value' AND numeric_col > N LIMIT n
    fn try_fast_multi_condition_filter(
        backend: &TableStorageBackend,
        stmt: &SelectStatement,
    ) -> io::Result<Option<RecordBatch>> {
        use crate::query::sql_parser::BinaryOperator;
        
        if backend.has_pending_deltas() || backend.is_mmap_only() { return Ok(None); }

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
        if backend.has_pending_deltas() { return Ok(None); }

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
        
        // Support: string equality (col = 'val') OR BETWEEN (col BETWEEN low AND high)
        enum FilterType<'a> {
            StringEq(String, &'a str),
            Between(String, f64, f64),
        }
        
        let filter = match where_clause {
            SqlExpr::BinaryOp { left, op: BinaryOperator::Eq, right } => {
                match (left.as_ref(), right.as_ref()) {
                    (SqlExpr::Column(col), SqlExpr::Literal(Value::String(val))) => {
                        FilterType::StringEq(col.trim_matches('"').to_string(), val.as_str())
                    }
                    (SqlExpr::Literal(Value::String(val)), SqlExpr::Column(col)) => {
                        FilterType::StringEq(col.trim_matches('"').to_string(), val.as_str())
                    }
                    _ => return Ok(None),
                }
            }
            SqlExpr::Between { column, low, high, negated } if !negated => {
                let low_val = Self::extract_numeric_value(low).ok();
                let high_val = Self::extract_numeric_value(high).ok();
                if let (Some(lo), Some(hi)) = (low_val, high_val) {
                    FilterType::Between(column.trim_matches('"').to_string(), lo, hi)
                } else {
                    return Ok(None);
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
        
        // Support SUM, COUNT, and AVG
        if !matches!(agg_func, AggregateFunc::Sum | AggregateFunc::Count | AggregateFunc::Avg) {
            return Ok(None);
        }
        
        // Check HAVING clause - must be simple
        if let Some(having) = &stmt.having {
            match having {
                SqlExpr::BinaryOp { left, op: BinaryOperator::Gt, right } => {
                    match (left.as_ref(), right.as_ref()) {
                        (SqlExpr::Column(_col), SqlExpr::Literal(Value::Int64(_val))) => {}
                        _ => return Ok(None),
                    }
                }
                _ => return Ok(None),
            }
        }
        
        let limit = stmt.limit.unwrap_or(100);
        let offset = stmt.offset.unwrap_or(0);
        
        // For string equality filter, use existing storage-level path
        match &filter {
            FilterType::StringEq(filter_col, filter_val) => {
                // Only SUM/COUNT for the storage-level string eq path
                if !matches!(agg_func, AggregateFunc::Sum | AggregateFunc::Count) {
                    return Ok(None);
                }
                match backend.execute_filter_group_order(
                    filter_col,
                    filter_val,
                    group_col,
                    agg_col,
                    agg_func,
                    order_col,
                    descending,
                    limit,
                    offset,
                ) {
                    Ok(Some(result)) => Ok(Some(ApexResult::Data(result))),
                    Ok(None) => Ok(None),
                    Err(e) => Err(e),
                }
            }
            FilterType::Between(filter_col, lo, hi) => {
                // OPTIMIZED: Use global cached dict for O(1) group lookup
                let raw_results = if let Some(dict_arc) = crate::storage::backend::get_global_dict_cache(
                    backend.path(), group_col, &backend.storage,
                )? {
                    backend.execute_between_group_agg_cached(
                        filter_col, *lo, *hi, &dict_arc.0, &dict_arc.1, agg_col,
                    )?
                } else {
                    backend.storage.execute_between_group_agg(
                        filter_col, *lo, *hi, group_col, agg_col,
                    )?
                };
                
                let raw = match raw_results {
                    Some(r) if !r.is_empty() => r,
                    _ => return Ok(None),
                };
                
                // Compute final aggregated values
                let mut results: Vec<(String, f64)> = raw.into_iter().map(|(k, sum, count)| {
                    let val = match agg_func {
                        AggregateFunc::Sum => sum,
                        AggregateFunc::Count => count as f64,
                        AggregateFunc::Avg => if count > 0 { sum / count as f64 } else { 0.0 },
                        _ => sum,
                    };
                    (k, val)
                }).collect();
                
                // Sort
                if descending {
                    results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
                } else {
                    results.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
                }
                let results: Vec<_> = results.into_iter().skip(offset).take(limit).collect();
                
                if results.is_empty() {
                    return Ok(None);
                }
                
                // Build Arrow result
                let group_values: Vec<&str> = results.iter().map(|(k, _)| k.as_str()).collect();
                let agg_values: Vec<f64> = results.iter().map(|(_, v)| *v).collect();
                
                let group_col_name = group_col.to_string();
                let agg_col_name = stmt.columns.iter().find_map(|c| {
                    if let SelectColumn::Aggregate { alias, .. } = c {
                        alias.clone().or_else(|| Some(order_col.to_string()))
                    } else { None }
                }).unwrap_or_else(|| order_col.to_string());
                
                let schema = Arc::new(Schema::new(vec![
                    Field::new(&group_col_name, ArrowDataType::Utf8, false),
                    Field::new(&agg_col_name, ArrowDataType::Float64, false),
                ]));
                let arrays: Vec<ArrayRef> = vec![
                    Arc::new(StringArray::from(group_values)),
                    Arc::new(Float64Array::from(agg_values)),
                ];
                let result = RecordBatch::try_new(schema, arrays)
                    .map_err(|e| err_data(e.to_string()))?;
                
                Ok(Some(ApexResult::Data(result)))
            }
        }
    }
    
    /// V4 FAST PATH for GROUP BY queries without WHERE
    /// Handles: SELECT group_col, AGG1(col1), AGG2(col2) FROM table GROUP BY group_col
    fn try_fast_v4_group_by(
        backend: &TableStorageBackend,
        stmt: &SelectStatement,
    ) -> io::Result<Option<ApexResult>> {
        if backend.has_pending_deltas() { return Ok(None); }

        use crate::query::AggregateFunc;
        
        // Must be single GROUP BY column, no WHERE, no ORDER BY
        if stmt.group_by.len() != 1 || stmt.where_clause.is_some() || !stmt.order_by.is_empty() {
            return Ok(None);
        }
        
        let group_col = stmt.group_by[0].trim_matches('"');
        
        // Extract aggregate columns: (col_name_or_"*", is_count_star, func, alias)
        let mut agg_info: Vec<(&str, bool, AggregateFunc, Option<String>)> = Vec::new();
        
        for col in &stmt.columns {
            match col {
                SelectColumn::Aggregate { func, column, alias, .. } => {
                    let is_count_star = matches!(func, AggregateFunc::Count) && column.is_none();
                    let col_name = column.as_deref().unwrap_or("*");
                    agg_info.push((col_name, is_count_star, func.clone(), alias.clone()));
                }
                SelectColumn::Column(name) => {
                    if name.trim_matches('"') == group_col { continue; }
                    return Ok(None);
                }
                SelectColumn::ColumnAlias { column, .. } => {
                    if column.trim_matches('"') == group_col { continue; }
                    return Ok(None);
                }
                _ => return Ok(None),
            }
        }
        
        if agg_info.is_empty() {
            return Ok(None);
        }
        
        // Build agg_cols for storage call
        let agg_cols: Vec<(&str, bool)> = agg_info.iter()
            .map(|(col, is_count, _, _)| (*col, *is_count))
            .collect();
        
        let raw = match backend.execute_group_agg(group_col, &agg_cols)? {
            Some(r) if !r.is_empty() => r,
            _ => return Ok(None),
        };
        
        // Build result: group_col + one column per aggregate
        let num_groups = raw.len();
        let group_values: Vec<&str> = raw.iter().map(|(k, _)| k.as_str()).collect();
        
        let mut fields: Vec<Field> = vec![
            Field::new(group_col, ArrowDataType::Utf8, false),
        ];
        let mut arrays: Vec<ArrayRef> = vec![
            Arc::new(StringArray::from(group_values)),
        ];
        
        for (ai, (_, _, func, alias)) in agg_info.iter().enumerate() {
            let col_name = alias.as_deref().unwrap_or(match func {
                AggregateFunc::Count => "COUNT(*)",
                AggregateFunc::Avg => "AVG",
                AggregateFunc::Sum => "SUM",
                AggregateFunc::Min => "MIN",
                AggregateFunc::Max => "MAX",
            });
            
            let values: Vec<f64> = raw.iter().map(|(_, aggs)| {
                let (sum, count) = aggs[ai];
                match func {
                    AggregateFunc::Count => count as f64,
                    AggregateFunc::Avg => if count > 0 { sum / count as f64 } else { 0.0 },
                    AggregateFunc::Sum => sum,
                    _ => sum,
                }
            }).collect();
            
            // Use Int64 for COUNT, Float64 for others
            if matches!(func, AggregateFunc::Count) {
                let int_values: Vec<i64> = values.iter().map(|v| *v as i64).collect();
                fields.push(Field::new(col_name, ArrowDataType::Int64, false));
                arrays.push(Arc::new(Int64Array::from(int_values)));
            } else {
                fields.push(Field::new(col_name, ArrowDataType::Float64, false));
                arrays.push(Arc::new(Float64Array::from(values)));
            }
        }
        
        // Apply HAVING if present
        let schema = Arc::new(Schema::new(fields));
        let batch = RecordBatch::try_new(schema, arrays)
            .map_err(|e| err_data(e.to_string()))?;
        
        let mut result = if let Some(having) = &stmt.having {
            let mask = Self::evaluate_predicate(&batch, having)?;
            let filtered = arrow::compute::filter_record_batch(&batch, &mask)
                .map_err(|e| err_data(e.to_string()))?;
            if filtered.num_rows() == 0 {
                return Ok(Some(ApexResult::Empty(filtered.schema())));
            }
            filtered
        } else {
            batch
        };

        // Apply ORDER BY with aggregate expression resolver
        if !stmt.order_by.is_empty() {
            let resolved_ob = Self::resolve_order_by_cols(&stmt.columns, &stmt.order_by);
            let k = stmt.limit.map(|l| l + stmt.offset.unwrap_or(0));
            result = Self::apply_order_by_topk(&result, &resolved_ob, k)?;
        }

        // Apply LIMIT + OFFSET
        if stmt.limit.is_some() || stmt.offset.is_some() {
            result = Self::apply_limit_offset(&result, stmt.limit, stmt.offset)?;
        }

        Ok(Some(ApexResult::Data(result)))
    }

    /// V4 FAST PATH: Simple aggregation (no GROUP BY, no WHERE)
    /// Handles: SELECT COUNT(*), AVG(col), SUM(col), MIN(col), MAX(col) FROM table
    fn try_fast_simple_agg(
        backend: &TableStorageBackend,
        stmt: &SelectStatement,
    ) -> io::Result<Option<ApexResult>> {
        if backend.has_pending_deltas() || backend.is_mmap_only() { return Ok(None); }

        use crate::query::AggregateFunc;
        
        // Collect unique column names needed for aggregation
        let mut unique_cols: Vec<String> = Vec::new();
        for col in &stmt.columns {
            if let SelectColumn::Aggregate { func, column, distinct, .. } = col {
                if *distinct { return Ok(None); } // DISTINCT needs full scan
                let name = column.as_deref().unwrap_or("*");
                if name == "_id" { return Ok(None); } // _id stored separately
                // COUNT(col) and AVG(col) must exclude NULLs: the storage fast path
                // returns total-row-count which would be wrong when NULLs are present.
                // Fall back to the full Arrow scan which respects null_count().
                let is_star_or_const = name == "*" || name.chars().next().map(|c| c.is_ascii_digit()).unwrap_or(false);
                if !is_star_or_const {
                    if matches!(func, AggregateFunc::Count | AggregateFunc::Avg) {
                        return Ok(None);
                    }
                }
                if !unique_cols.contains(&name.to_string()) {
                    unique_cols.push(name.to_string());
                }
            } else {
                return Ok(None); // Non-aggregate column present
            }
        }
        if unique_cols.is_empty() { return Ok(None); }
        
        let col_refs: Vec<&str> = unique_cols.iter().map(|s| s.as_str()).collect();
        let raw = match backend.execute_simple_agg(&col_refs)? {
            Some(r) => r,
            None => return Ok(None),
        };
        
        // Build result
        let mut fields: Vec<Field> = Vec::new();
        let mut arrays: Vec<ArrayRef> = Vec::new();
        
        for col in &stmt.columns {
            if let SelectColumn::Aggregate { func, column, alias, .. } = col {
                let col_name = column.as_deref().unwrap_or("*");
                let fn_name = match func { AggregateFunc::Count => "COUNT", AggregateFunc::Sum => "SUM", AggregateFunc::Avg => "AVG", AggregateFunc::Min => "MIN", AggregateFunc::Max => "MAX" };
                let output_name = alias.clone().unwrap_or_else(|| if let Some(c) = column { format!("{}({})", fn_name, c) } else { format!("{}(*)", fn_name) });
                
                let idx = unique_cols.iter().position(|s| s == col_name).unwrap_or(0);
                let (count, sum, min_v, max_v, is_int) = raw[idx];
                
                match func {
                    AggregateFunc::Count => {
                        fields.push(Field::new(&output_name, ArrowDataType::Int64, false));
                        arrays.push(Arc::new(Int64Array::from(vec![count])));
                    }
                    AggregateFunc::Sum => {
                        if is_int {
                            fields.push(Field::new(&output_name, ArrowDataType::Int64, false));
                            arrays.push(Arc::new(Int64Array::from(vec![sum as i64])));
                        } else {
                            fields.push(Field::new(&output_name, ArrowDataType::Float64, false));
                            arrays.push(Arc::new(Float64Array::from(vec![sum])));
                        }
                    }
                    AggregateFunc::Avg => {
                        let avg = if count > 0 { sum / count as f64 } else { 0.0 };
                        fields.push(Field::new(&output_name, ArrowDataType::Float64, false));
                        arrays.push(Arc::new(Float64Array::from(vec![avg])));
                    }
                    AggregateFunc::Min => {
                        if is_int {
                            fields.push(Field::new(&output_name, ArrowDataType::Int64, false));
                            arrays.push(Arc::new(Int64Array::from(vec![min_v as i64])));
                        } else {
                            fields.push(Field::new(&output_name, ArrowDataType::Float64, false));
                            arrays.push(Arc::new(Float64Array::from(vec![min_v])));
                        }
                    }
                    AggregateFunc::Max => {
                        if is_int {
                            fields.push(Field::new(&output_name, ArrowDataType::Int64, false));
                            arrays.push(Arc::new(Int64Array::from(vec![max_v as i64])));
                        } else {
                            fields.push(Field::new(&output_name, ArrowDataType::Float64, false));
                            arrays.push(Arc::new(Float64Array::from(vec![max_v])));
                        }
                    }
                }
            }
        }
        
        let schema = Arc::new(Schema::new(fields));
        let batch = RecordBatch::try_new(schema, arrays)
            .map_err(|e| err_data(e.to_string()))?;
        Ok(Some(ApexResult::Data(batch)))
    }

    /// V4 FAST PATH: Cached GROUP BY (builds dict cache on first call, reuses on subsequent calls)
    fn try_fast_cached_group_by(
        backend: &TableStorageBackend,
        stmt: &SelectStatement,
    ) -> io::Result<Option<ApexResult>> {
        if backend.has_pending_deltas() { return Ok(None); }

        use crate::query::AggregateFunc;
        
        // Must be single GROUP BY column, no WHERE
        if stmt.group_by.len() != 1 || stmt.where_clause.is_some() || !stmt.order_by.is_empty() {
            return Ok(None);
        }
        
        let group_col = stmt.group_by[0].trim_matches('"');
        
        // Extract aggregate info
        let mut agg_info: Vec<(&str, bool, AggregateFunc, Option<String>)> = Vec::new();
        for col in &stmt.columns {
            match col {
                SelectColumn::Aggregate { func, column, alias, .. } => {
                    let is_count_star = matches!(func, AggregateFunc::Count) && column.is_none();
                    let col_name = column.as_deref().unwrap_or("*");
                    agg_info.push((col_name, is_count_star, func.clone(), alias.clone()));
                }
                SelectColumn::Column(name) => {
                    if name.trim_matches('"') == group_col { continue; }
                    return Ok(None);
                }
                SelectColumn::ColumnAlias { column, .. } => {
                    if column.trim_matches('"') == group_col { continue; }
                    return Ok(None);
                }
                _ => return Ok(None),
            }
        }
        if agg_info.is_empty() { return Ok(None); }
        
        // Get or build cached dict (global cache — survives backend reopens)
        let dict_arc = match crate::storage::backend::get_global_dict_cache(
            backend.path(), group_col, &backend.storage,
        )? {
            Some(c) => c,
            None => return Ok(None),
        };
        let (dict_strings, group_ids) = (dict_arc.0.as_slice(), dict_arc.1.as_slice());
        
        let agg_cols: Vec<(&str, bool)> = agg_info.iter()
            .map(|(col, is_count, _, _)| (*col, *is_count))
            .collect();
        
        let raw = match backend.execute_group_agg_cached(dict_strings, group_ids, &agg_cols)? {
            Some(r) if !r.is_empty() => r,
            _ => return Ok(None),
        };
        
        // Build result
        let num_groups = raw.len();
        let group_values: Vec<&str> = raw.iter().map(|(k, _)| k.as_str()).collect();
        
        let mut fields: Vec<Field> = vec![Field::new(group_col, ArrowDataType::Utf8, false)];
        let mut arrays: Vec<ArrayRef> = vec![Arc::new(StringArray::from(group_values))];
        
        for (ai, (_, _, func, alias)) in agg_info.iter().enumerate() {
            let col_name = alias.as_deref().unwrap_or(match func {
                AggregateFunc::Count => "COUNT(*)", AggregateFunc::Avg => "AVG",
                AggregateFunc::Sum => "SUM", AggregateFunc::Min => "MIN", AggregateFunc::Max => "MAX",
            });
            let values: Vec<f64> = raw.iter().map(|(_, aggs)| {
                let (sum, count) = aggs[ai];
                match func {
                    AggregateFunc::Count => count as f64,
                    AggregateFunc::Avg => if count > 0 { sum / count as f64 } else { 0.0 },
                    _ => sum,
                }
            }).collect();
            if matches!(func, AggregateFunc::Count) {
                let int_values: Vec<i64> = values.iter().map(|v| *v as i64).collect();
                fields.push(Field::new(col_name, ArrowDataType::Int64, false));
                arrays.push(Arc::new(Int64Array::from(int_values)));
            } else {
                fields.push(Field::new(col_name, ArrowDataType::Float64, false));
                arrays.push(Arc::new(Float64Array::from(values)));
            }
        }
        
        let schema = Arc::new(Schema::new(fields));
        let batch = RecordBatch::try_new(schema, arrays)
            .map_err(|e| err_data(e.to_string()))?;
        
        // Apply HAVING
        if let Some(having) = &stmt.having {
            let mask = Self::evaluate_predicate(&batch, having)?;
            let filtered = arrow::compute::filter_record_batch(&batch, &mask)
                .map_err(|e| err_data(e.to_string()))?;
            if filtered.num_rows() == 0 {
                return Ok(Some(ApexResult::Empty(filtered.schema())));
            }
            return Ok(Some(ApexResult::Data(filtered)));
        }
        
        Ok(Some(ApexResult::Data(batch)))
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
    
    /// Helper to extract BETWEEN range: col BETWEEN low AND high
    fn extract_between_range(expr: &SqlExpr) -> Option<(String, f64, f64)> {
        match expr {
            SqlExpr::Between { column, low, high, negated } => {
                if *negated { return None; }
                let col = column.trim_matches('"').to_string();
                let low_val = Self::extract_numeric_value(low).ok()?;
                let high_val = Self::extract_numeric_value(high).ok()?;
                Some((col, low_val, high_val))
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

    /// Systematic predicate pushdown: extract simple numeric comparison from WHERE
    /// and use storage-level filtered read instead of full table scan.
    /// Handles: col > N, col >= N, col < N, col <= N, col = N, col != N
    /// Returns Some(batch) if pushdown succeeded, None to fall through.
    fn try_numeric_predicate_pushdown(
        backend: &TableStorageBackend,
        stmt: &SelectStatement,
    ) -> io::Result<Option<RecordBatch>> {
        if backend.has_pending_deltas() || backend.is_mmap_only() {
            return Ok(None);
        }
        let where_clause = match &stmt.where_clause {
            Some(w) => w,
            None => return Ok(None),
        };
        // Try to extract a simple numeric comparison (col op literal)
        let (col_name, op_str, value) = match Self::extract_numeric_comparison(where_clause) {
            Some(v) => v,
            None => return Ok(None),
        };
        // Column projection pushdown
        let col_refs = Self::get_col_refs(stmt);
        let col_refs_vec: Option<Vec<&str>> = col_refs.as_ref()
            .map(|v| v.iter().map(|s| s.as_str()).collect());
        let batch = backend.read_columns_filtered_to_arrow(
            col_refs_vec.as_deref(),
            &col_name,
            &op_str,
            value,
        )?;
        Ok(Some(batch))
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

        let where_clause = stmt.where_clause.as_ref().unwrap();
        let need_count = stmt.limit.map(|l| l + stmt.offset.unwrap_or(0));

        // FAST PATH: no LIMIT → full sequential read + vectorized Arrow filter
        // This beats chunked reads + random index access (400K random seeks > 1 sequential scan).
        if need_count.is_none() {
            let full_batch = backend.read_columns_to_arrow(None, 0, None)?;
            if full_batch.num_rows() > 0 {
                let mask = Self::evaluate_predicate_with_storage(&full_batch, where_clause, storage_path)?;
                return compute::filter_record_batch(&full_batch, &mask)
                    .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e.to_string()));
            }
            return Ok(full_batch);
        }

        // Step 1: Read only columns needed for WHERE clause
        let where_cols = stmt.where_columns();
        let where_col_refs: Vec<&str> = where_cols.iter().map(|s| s.as_str()).collect();
        
        // Also include _id for later row identification
        let mut cols_to_read: Vec<&str> = vec!["_id"];
        cols_to_read.extend(where_col_refs.iter());
        
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
                
                #[cfg(test)]
                {
                    let true_count = mask.iter().filter(|v| *v == Some(true)).count();
                    eprintln!("DEBUG late_mat chunk: mask true_count={}", true_count);
                }
                
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
        
        // Step 4: Read ALL columns but only for matching row indices.
        // NOTE: limited_indices are positions in the ACTIVE row sequence (deleted rows excluded).
        // For V4 mmap-only backends, read_columns_by_indices_to_arrow delegates to
        // extract_rows_by_indices_to_arrow which uses PHYSICAL row positions — causing
        // wrong results when deletions shift active vs physical positions.
        // For mmap-only: use full active read + Arrow take (active indices match active batch).
        // For in-memory (data loaded): read_columns_by_indices_to_arrow falls back to the
        // same full-read + take path, so physical==active there too.
        if backend.is_mmap_only() {
            use arrow::array::ArrayRef;
            let full_batch = backend.read_columns_to_arrow(None, 0, None)?;
            let indices_arr = arrow::array::UInt32Array::from(
                limited_indices.iter().map(|&i| i as u32).collect::<Vec<_>>()
            );
            let taken_cols: Vec<ArrayRef> = full_batch.columns().iter()
                .map(|col| arrow::compute::take(col.as_ref(), &indices_arr, None)
                    .map_err(|e| err_data(e.to_string())))
                .collect::<io::Result<Vec<_>>>()?;
            arrow::record_batch::RecordBatch::try_new(full_batch.schema(), taken_cols)
                .map_err(|e| err_data(e.to_string()))
        } else {
            backend.read_columns_by_indices_to_arrow(&limited_indices)
        }
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

        // MMAP FAST PATH: single ORDER BY column + mmap-only → direct top-K scan without Arrow
        if backend.is_mmap_only() && stmt.order_by.len() == 1 && !backend.has_pending_deltas() {
            let clause = &stmt.order_by[0];
            let col_name = clause.column.trim_matches('"');
            let actual_col = if let Some(p) = col_name.rfind('.') { &col_name[p+1..] } else { col_name };
            if let Some(heap) = backend.scan_top_k_indices_mmap(actual_col, k, clause.descending)? {
                let offset = stmt.offset.unwrap_or(0);
                let final_indices: Vec<usize> = heap.into_iter().skip(offset).map(|(idx, _)| idx).collect();
                if !final_indices.is_empty() {
                    return backend.read_columns_by_indices_to_arrow(&final_indices);
                }
            }
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
        if backend.has_pending_deltas() || backend.is_mmap_only() { return Ok(None); }

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
        ).map_err(|e| err_data( e.to_string()))?;
        
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
            let col_refs = Self::get_col_refs(stmt);
            let col_refs_vec: Option<Vec<&str>> = col_refs.as_ref().map(|v| v.iter().map(|s| s.as_str()).collect());
            return backend.read_columns_to_arrow(col_refs_vec.as_deref(), 0, Some(0));
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
            let col_refs = Self::get_col_refs(stmt);
            let col_refs_vec: Option<Vec<&str>> = col_refs.as_ref().map(|v| v.iter().map(|s| s.as_str()).collect());
            return backend.read_columns_to_arrow(col_refs_vec.as_deref(), 0, Some(0));
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
                .map_err(|e| err_data( e.to_string()))?;
            RecordBatch::try_new(filter_batch.schema(), columns)
                .map_err(|e| err_data( e.to_string()))
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
                .map_err(|e| err_data( e.to_string()))?;
            
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
                .map_err(|e| err_data( e.to_string()))
        }
    }

    // ========== FTS Helper: resolve MATCH()/FUZZY_MATCH() to _id IN (...) ==========

    /// Recursively replace every `FtsMatch { query, fuzzy }` node in an expression with
    /// `In { column: "_id", values: [matching doc ids] }`.  Requires the FtsManager to
    /// be registered for `base_dir` (via `register_fts_manager` or `CREATE FTS INDEX`).
    fn resolve_fts_in_expr(expr: SqlExpr, base_dir: &Path, table_name: &str) -> io::Result<SqlExpr> {
        match expr {
            SqlExpr::FtsMatch { query, fuzzy } => {
                let mgr = crate::query::executor::get_fts_manager(base_dir)
                    .ok_or_else(|| io::Error::new(
                        io::ErrorKind::Other,
                        format!("FTS not initialised for this database. Run CREATE FTS INDEX ON {} first.", table_name),
                    ))?;
                let engine = mgr.get_engine(table_name)
                    .map_err(|e| io::Error::new(io::ErrorKind::Other, e.to_string()))?;
                let ids: Vec<u64> = if fuzzy {
                    engine.fuzzy_search(&query, 1)
                        .map(|r| r.iter().map(|id| id as u64).collect())
                        .unwrap_or_default()
                } else {
                    engine.search_ids(&query)
                        .unwrap_or_default()
                };
                if ids.is_empty() {
                    // _id < 0  — guaranteed empty (all valid _id are >= 0)
                    Ok(SqlExpr::BinaryOp {
                        left:  Box::new(SqlExpr::Column("_id".to_string())),
                        op:    crate::query::sql_parser::BinaryOperator::Lt,
                        right: Box::new(SqlExpr::Literal(crate::data::Value::Int64(0))),
                    })
                } else {
                    let values: Vec<crate::data::Value> = ids.into_iter()
                        .map(|id| crate::data::Value::Int64(id as i64))
                        .collect();
                    Ok(SqlExpr::In { column: "_id".to_string(), values, negated: false })
                }
            }
            SqlExpr::BinaryOp { left, op, right } => {
                Ok(SqlExpr::BinaryOp {
                    left:  Box::new(Self::resolve_fts_in_expr(*left,  base_dir, table_name)?),
                    op,
                    right: Box::new(Self::resolve_fts_in_expr(*right, base_dir, table_name)?),
                })
            }
            SqlExpr::UnaryOp { op, expr } => {
                Ok(SqlExpr::UnaryOp { op, expr: Box::new(Self::resolve_fts_in_expr(*expr, base_dir, table_name)?) })
            }
            SqlExpr::Paren(inner) => {
                Ok(SqlExpr::Paren(Box::new(Self::resolve_fts_in_expr(*inner, base_dir, table_name)?)))
            }
            // All other variants have no nested SqlExpr that could contain FtsMatch
            other => Ok(other),
        }
    }

    /// Return true iff `expr` contains at least one `FtsMatch` node.
    fn expr_has_fts_match(expr: &SqlExpr) -> bool {
        match expr {
            SqlExpr::FtsMatch { .. } => true,
            SqlExpr::BinaryOp { left, right, .. } =>
                Self::expr_has_fts_match(left) || Self::expr_has_fts_match(right),
            SqlExpr::UnaryOp { expr, .. } => Self::expr_has_fts_match(expr),
            SqlExpr::Paren(inner) => Self::expr_has_fts_match(inner),
            _ => false,
        }
    }
}
