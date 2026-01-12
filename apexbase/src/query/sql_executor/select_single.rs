use super::*;

impl SqlExecutor {
    /// Execute SELECT statement - ULTRA-OPTIMIZED
    pub(crate) fn execute_select(stmt: SelectStatement, table: &mut ColumnTable) -> Result<SqlResult, ApexError> {
        if !stmt.joins.is_empty() {
            return Err(ApexError::QueryParseError(
                "JOIN requires multi-table execution".to_string(),
            ));
        }
        // Table name validation is handled at the binding layer
        // which can access all tables and route to the correct one
        
        // Only flush if there might be pending writes (lazy flush)
        if table.has_pending_writes() {
            table.flush_write_buffer();
        }
        
        // Pre-compute flags
        let has_aggregates = stmt.columns.iter().any(|c| matches!(c, SelectColumn::Aggregate { .. }));
        let has_window = stmt.columns.iter().any(|c| matches!(c, SelectColumn::WindowFunction { .. }));
        let no_group_by = stmt.group_by.is_empty();

        // ============ UNIFIED SIMPLE PIPELINE (GUARDED) ============
        // Only handle a very small subset here to avoid regressions.
        if !has_aggregates
            && !has_window
            && no_group_by
            && stmt.having.is_none()
            && simple_pipeline::is_eligible(&stmt)
        {
            // Pipeline expects an immutable table reference (no mutation)
            match simple_pipeline::execute(&stmt, table) {
                Ok(r) => return Ok(r),
                Err(ApexError::QueryParseError(msg))
                    if msg == "not eligible for simple select plan" =>
                {
                    // Fall back to legacy paths (supports complex WHERE expressions).
                }
                Err(e) => return Err(e),
            }
        }

        match crate::query::engine::PlanOptimizer::try_build_plan(&stmt, table) {
            Ok(Some(plan)) => return crate::query::engine::PlanExecutor::execute_sql_result(table, &plan),
            Ok(None) => {}
            Err(ApexError::QueryParseError(msg)) if msg == "not eligible for select plan" => {
                // Fall back to legacy paths (supports complex WHERE expressions).
            }
            Err(e) => return Err(e),
        }

        // ============ PATHS REQUIRING INDEX COLLECTION ============
        
        // Step 1: Get matching row indices based on WHERE clause
        let matching_indices: Vec<usize> = if let Some(ref where_expr) = stmt.where_clause {
            Self::evaluate_where(where_expr, table)?
        } else {
            // All non-deleted rows (only reached for complex queries)
            let deleted = table.deleted_ref();
            let row_count = table.get_row_count();
            (0..row_count).filter(|&i| !deleted.get(i)).collect()
        };

        // Window functions (minimal support): row_number() over(partition by ... order by ...)
        if has_window {
            if has_aggregates {
                return Err(ApexError::QueryParseError("Window functions with aggregates are not supported".to_string()));
            }
            if !no_group_by {
                return Err(ApexError::QueryParseError("Window functions with GROUP BY are not supported".to_string()));
            }
            if stmt.distinct {
                return Err(ApexError::QueryParseError("Window functions with DISTINCT are not supported".to_string()));
            }

            return Self::execute_window_row_number(&stmt, &matching_indices, table);
        }
        
        // Step 2: Aggregates / GROUP BY remain on legacy paths for now.
        // (Non-aggregate single-table SELECTs are handled by PlanOptimizer above.)
        let (result_columns, column_indices) = Self::resolve_columns(&stmt.columns, table)?;

        if has_aggregates && no_group_by {
            return Self::execute_aggregate(&stmt, &matching_indices, table);
        }

        if !no_group_by {
            let plan = crate::query::engine::PlanBuilder::build_group_by_plan(&stmt)?;
            return crate::query::engine::PlanExecutor::execute_sql_result(table, &plan);
        }

        // Fallback: row materialization for legacy-only features.
        // (Kept minimal; should be eliminated as remaining features are plan-ified.)
        let mut final_indices: Vec<usize> = matching_indices;

        // Preserve ORDER BY + LIMIT/OFFSET semantics even on legacy fallback.
        if !stmt.order_by.is_empty() {
            let offset = stmt.offset.unwrap_or(0);
            let limit = stmt.limit.unwrap_or(usize::MAX);
            let k = offset.saturating_add(limit);
            final_indices = Self::sort_indices_by_columns_topk(&final_indices, &stmt.order_by, table, k)?;
            final_indices = final_indices.into_iter().skip(offset).take(limit).collect();
        } else if stmt.limit.is_some() || stmt.offset.is_some() {
            let offset = stmt.offset.unwrap_or(0);
            let limit = stmt.limit.unwrap_or(usize::MAX);
            final_indices = final_indices.into_iter().skip(offset).take(limit).collect();
        }

        let mut rows: Vec<Vec<Value>> = Vec::with_capacity(final_indices.len().min(10000));
        for row_idx in final_indices.iter() {
            let mut row_values = Vec::with_capacity(result_columns.len());
            for (col_name, col_idx) in column_indices.iter() {
                if col_name == "_id" {
                    row_values.push(Value::Int64(*row_idx as i64));
                } else if let Some(idx) = col_idx {
                    row_values.push(table.columns_ref()[*idx].get(*row_idx).unwrap_or(Value::Null));
                } else {
                    row_values.push(Value::Null);
                }
            }
            rows.push(row_values);
        }
        Ok(SqlResult::new(result_columns, rows))
    }
    
    /// Partial sort indices by ORDER BY columns - only get top K elements
    /// Uses BinaryHeap for O(n log k) time complexity
    pub(crate) fn sort_indices_by_columns_topk(
        indices: &[usize],
        order_by: &[OrderByClause],
        table: &ColumnTable,
        k: usize,
    ) -> Result<Vec<usize>, ApexError> {
        crate::query::engine::ops::sort_indices_by_columns_topk(indices, order_by, table, k)
    }
    
    /// Resolve column names and indices from SELECT clause
    pub(crate) fn resolve_columns(
        columns: &[SelectColumn],
        table: &ColumnTable,
    ) -> Result<(Vec<String>, Vec<(String, Option<usize>)>), ApexError> {
        let (cols, idxs, _exprs) = crate::query::engine::ops::resolve_columns(columns, table)?;
        Ok((cols, idxs))
    }
    
    /// Evaluate WHERE clause and return matching row indices
    pub(crate) fn evaluate_where(expr: &SqlExpr, table: &ColumnTable) -> Result<Vec<usize>, ApexError> {
        // Fast path: compile to optimized Filter.
        if let Ok(filter) = sql_expr_to_filter(expr) {
            let schema = table.schema_ref();
            let columns = table.columns_ref();
            let row_count = table.get_row_count();
            let deleted = table.deleted_ref();
            return Ok(filter.filter_columns(schema, columns, row_count, deleted));
        }

        // Fallback: row-wise predicate evaluation for complex expressions
        // (e.g. WHERE a + b > 10).
        let ctx = crate::query::engine::ops::new_eval_context();
        let deleted = table.deleted_ref();
        let row_count = table.get_row_count();
        let no_deletes = deleted.all_false();

        let mut out = Vec::new();
        out.reserve(row_count.min(1024));
        for row_idx in 0..row_count {
            if !no_deletes && deleted.get(row_idx) {
                continue;
            }
            if Self::eval_where_predicate(expr, table, row_idx, &ctx)? {
                out.push(row_idx);
            }
        }
        Ok(out)
    }

    fn eval_where_predicate(
        expr: &SqlExpr,
        table: &ColumnTable,
        row_idx: usize,
        ctx: &crate::query::engine::ops::EvalContext,
    ) -> Result<bool, ApexError> {
        use crate::data::Value;

        #[inline]
        fn to_bool(v: Value) -> bool {
            v.as_bool().unwrap_or(false)
        }

        match expr {
            SqlExpr::Paren(inner) => Self::eval_where_predicate(inner, table, row_idx, ctx),
            SqlExpr::Literal(v) => Ok(v.as_bool().unwrap_or(false)),
            SqlExpr::UnaryOp { op: UnaryOperator::Not, expr } => {
                Ok(!Self::eval_where_predicate(expr, table, row_idx, ctx)?)
            }
            SqlExpr::BinaryOp { left, op, right } => match op {
                BinaryOperator::And => Ok(
                    Self::eval_where_predicate(left, table, row_idx, ctx)?
                        && Self::eval_where_predicate(right, table, row_idx, ctx)?,
                ),
                BinaryOperator::Or => Ok(
                    Self::eval_where_predicate(left, table, row_idx, ctx)?
                        || Self::eval_where_predicate(right, table, row_idx, ctx)?,
                ),
                BinaryOperator::Eq
                | BinaryOperator::NotEq
                | BinaryOperator::Lt
                | BinaryOperator::Le
                | BinaryOperator::Gt
                | BinaryOperator::Ge => {
                    let lv = crate::query::engine::ops::eval_scalar_expr(left, table, row_idx, ctx)?;
                    let rv = crate::query::engine::ops::eval_scalar_expr(right, table, row_idx, ctx)?;
                    if lv.is_null() || rv.is_null() {
                        return Ok(false);
                    }
                    let ord = lv.partial_cmp(&rv).unwrap_or(Ordering::Equal);
                    Ok(match op {
                        BinaryOperator::Eq => ord == Ordering::Equal,
                        BinaryOperator::NotEq => ord != Ordering::Equal,
                        BinaryOperator::Lt => ord == Ordering::Less,
                        BinaryOperator::Le => ord != Ordering::Greater,
                        BinaryOperator::Gt => ord == Ordering::Greater,
                        BinaryOperator::Ge => ord != Ordering::Less,
                        _ => false,
                    })
                }
                // For arithmetic (or other) operators used as predicate, evaluate as scalar and coerce to bool.
                _ => {
                    let v = crate::query::engine::ops::eval_scalar_expr(expr, table, row_idx, ctx)?;
                    Ok(to_bool(v))
                }
            },
            SqlExpr::Like {
                column,
                pattern,
                negated,
            } => {
                let v = crate::query::engine::ops::eval_scalar_expr(
                    &SqlExpr::Column(column.clone()),
                    table,
                    row_idx,
                    ctx,
                )?;
                let s = match v {
                    Value::String(s) => s,
                    _ => return Ok(false),
                };
                let m = LikeMatcher::new(pattern);
                let ok = m.matches(s.as_str());
                Ok(if *negated { !ok } else { ok })
            }
            SqlExpr::Regexp {
                column,
                pattern,
                negated,
            } => {
                let v = crate::query::engine::ops::eval_scalar_expr(
                    &SqlExpr::Column(column.clone()),
                    table,
                    row_idx,
                    ctx,
                )?;
                let s = match v {
                    Value::String(s) => s,
                    _ => return Ok(false),
                };
                let m = RegexpMatcher::new(pattern);
                let ok = m.matches(s.as_str());
                Ok(if *negated { !ok } else { ok })
            }
            SqlExpr::In {
                column,
                values,
                negated,
            } => {
                let v = crate::query::engine::ops::eval_scalar_expr(
                    &SqlExpr::Column(column.clone()),
                    table,
                    row_idx,
                    ctx,
                )?;
                let ok = values.iter().any(|x| &v == x);
                Ok(if *negated { !ok } else { ok })
            }
            SqlExpr::Between {
                column,
                low,
                high,
                negated,
            } => {
                let v = crate::query::engine::ops::eval_scalar_expr(
                    &SqlExpr::Column(column.clone()),
                    table,
                    row_idx,
                    ctx,
                )?;
                if v.is_null() {
                    return Ok(false);
                }
                let lv = crate::query::engine::ops::eval_scalar_expr(low, table, row_idx, ctx)?;
                let hv = crate::query::engine::ops::eval_scalar_expr(high, table, row_idx, ctx)?;
                if lv.is_null() || hv.is_null() {
                    return Ok(false);
                }
                let ok_low = v.partial_cmp(&lv).map(|o| o != Ordering::Less).unwrap_or(false);
                let ok_high = v.partial_cmp(&hv).map(|o| o != Ordering::Greater).unwrap_or(false);
                let ok = ok_low && ok_high;
                Ok(if *negated { !ok } else { ok })
            }
            SqlExpr::IsNull { column, negated } => {
                let v = crate::query::engine::ops::eval_scalar_expr(
                    &SqlExpr::Column(column.clone()),
                    table,
                    row_idx,
                    ctx,
                )?;
                let ok = v.is_null();
                Ok(if *negated { !ok } else { ok })
            }
            // Fallback: treat scalar expression as boolean
            _ => {
                let v = crate::query::engine::ops::eval_scalar_expr(expr, table, row_idx, ctx)?;
                Ok(to_bool(v))
            }
        }
    }

    /// Minimal window function execution: supports only row_number() OVER (PARTITION BY <col> ORDER BY <col>)
    /// Applies WHERE first (matching_indices), then computes row_number per partition.
    pub(crate) fn execute_window_row_number(
        stmt: &SelectStatement,
        matching_indices: &[usize],
        table: &ColumnTable,
    ) -> Result<SqlResult, ApexError> {
        #[derive(Clone, Eq)]
        enum PartKey {
            Null,
            Int(i64),
            Str(String),
            Other(String),
        }

        impl PartialEq for PartKey {
            fn eq(&self, other: &Self) -> bool {
                match (self, other) {
                    (PartKey::Null, PartKey::Null) => true,
                    (PartKey::Int(a), PartKey::Int(b)) => a == b,
                    (PartKey::Str(a), PartKey::Str(b)) => a == b,
                    (PartKey::Other(a), PartKey::Other(b)) => a == b,
                    _ => false,
                }
            }
        }

        impl Hash for PartKey {
            fn hash<H: Hasher>(&self, state: &mut H) {
                match self {
                    PartKey::Null => 0u8.hash(state),
                    PartKey::Int(v) => {
                        1u8.hash(state);
                        v.hash(state);
                    }
                    PartKey::Str(s) => {
                        2u8.hash(state);
                        s.hash(state);
                    }
                    PartKey::Other(s) => {
                        3u8.hash(state);
                        s.hash(state);
                    }
                }
            }
        }

        // Collect window specs (only row_number supported)
        let mut window_specs = Vec::new();
        for col in &stmt.columns {
            if let SelectColumn::WindowFunction { name, partition_by, order_by, alias } = col {
                if name.to_uppercase() != "ROW_NUMBER" {
                    return Err(ApexError::QueryParseError(format!("Unsupported window function: {}", name)));
                }
                if partition_by.len() != 1 {
                    return Err(ApexError::QueryParseError("row_number() requires exactly 1 PARTITION BY column".to_string()));
                }
                if order_by.len() != 1 {
                    return Err(ApexError::QueryParseError("row_number() requires exactly 1 ORDER BY column in OVER()".to_string()));
                }
                window_specs.push((partition_by[0].clone(), order_by[0].clone(), alias.clone().unwrap_or_else(|| "row_number".to_string())));
            }
        }
        if window_specs.len() != 1 {
            return Err(ApexError::QueryParseError("Only a single row_number() window function is supported".to_string()));
        }
        let (partition_col, order_clause, window_alias) = window_specs.remove(0);

        let schema = table.schema_ref();
        let columns = table.columns_ref();

        // Expand SELECT list (including `*`) into a concrete output schema.
        // For window execution we only support a single window column, which will
        // be represented as a synthetic column name with no underlying index.
        let (result_columns, column_indices, _exprs) =
            crate::query::engine::ops::resolve_columns(&stmt.columns, table)?;

        let part_idx = schema.get_index(&partition_col)
            .ok_or_else(|| ApexError::QueryParseError(format!("Unknown PARTITION BY column: {}", partition_col)))?;
        let order_idx = if order_clause.column == "_id" {
            None
        } else {
            schema.get_index(&order_clause.column)
        };

        // Group matching indices by partition key
        let mut groups: HashMap<PartKey, Vec<usize>> = HashMap::new();
        for &row_idx in matching_indices {
            let key = match columns[part_idx].get(row_idx).unwrap_or(Value::Null) {
                Value::Null => PartKey::Null,
                Value::Int64(v) => PartKey::Int(v),
                Value::String(s) => PartKey::Str(s),
                other => PartKey::Other(other.to_string_value()),
            };
            groups.entry(key).or_insert_with(Vec::new).push(row_idx);
        }

        // Build output rows
        let mut out_rows: Vec<Vec<Value>> = Vec::with_capacity(matching_indices.len().min(10000));

        for (_, mut idxs) in groups {
            // Sort within partition by ORDER BY
            if let Some(oidx) = order_idx {
                let desc = order_clause.descending;
                idxs.sort_by(|&a, &b| {
                    let av = columns[oidx].get(a);
                    let bv = columns[oidx].get(b);
                    let cmp = Self::compare_values(av.as_ref(), bv.as_ref(), None);
                    if desc { cmp.reverse() } else { cmp }
                });
            } else {
                // ORDER BY _id
                if order_clause.descending {
                    idxs.sort_by(|a, b| b.cmp(a));
                } else {
                    idxs.sort_unstable();
                }
            }

            for (pos, row_idx) in idxs.into_iter().enumerate() {
                let rn = (pos + 1) as i64;
                let mut row = Vec::with_capacity(result_columns.len());
                for (col_name, col_idx) in column_indices.iter() {
                    if col_name == &window_alias {
                        row.push(Value::Int64(rn));
                    } else if col_name == "_id" {
                        row.push(Value::Int64(row_idx as i64));
                    } else if let Some(ci) = col_idx {
                        row.push(columns[*ci].get(row_idx).unwrap_or(Value::Null));
                    } else {
                        // Synthetic / unsupported expression columns (not expected in window path)
                        row.push(Value::Null);
                    }
                }
                out_rows.push(row);
            }
        }

        Ok(SqlResult::new(result_columns, out_rows))
    }
    
    /// Apply DISTINCT to result rows
    pub(crate) fn apply_distinct(rows: Vec<Vec<Value>>) -> Vec<Vec<Value>> {
        crate::query::engine::ops::apply_distinct(rows)
    }
    
    /// Apply ORDER BY to result rows
    pub(crate) fn apply_order_by(
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
    pub(crate) fn compare_values(a: Option<&Value>, b: Option<&Value>, nulls_first: Option<bool>) -> Ordering {
        crate::query::engine::ops::compare_values(a, b, nulls_first)
    }
    
    pub(crate) fn compare_non_null(a: &Value, b: &Value) -> Ordering {
        crate::query::engine::ops::compare_non_null(a, b)
    }
    
    // ========================================================================
    // ULTRA-FAST PATH IMPLEMENTATIONS
    // ========================================================================
    
    /// O(1) COUNT(*) without WHERE clause
    pub(crate) fn try_fast_count_star(stmt: &SelectStatement, table: &ColumnTable) -> Option<SqlResult> {
        // Check if this is a simple COUNT(*) / COUNT(constant) query
        if stmt.columns.len() == 1 {
            if let SelectColumn::Aggregate { func: AggregateFunc::Count, column, distinct, alias } = &stmt.columns[0] {
                if !*distinct && Self::is_count_star_like(column) {
                    let col_name = alias.clone().unwrap_or_else(|| {
                        column
                            .as_ref()
                            .map(|c| format!("COUNT({})", c))
                            .unwrap_or_else(|| "COUNT(*)".to_string())
                    });
                    let count = table.row_count() as i64;
                    return Some(SqlResult::new(vec![col_name], vec![vec![Value::Int64(count)]]));
                }
            }
        }
        None
    }

    #[inline]
    pub(crate) fn is_count_star_like(column: &Option<String>) -> bool {
        match column {
            None => true,
            Some(c) => Self::is_count_constant_arg(c),
        }
    }

    #[inline]
    fn is_count_constant_arg(arg: &str) -> bool {
        // Parser stores COUNT(constant) as textual representation (e.g. "1", "3.14", "'x'", "true").
        let s = arg.trim();
        if s.is_empty() {
            return false;
        }

        let sl = s.to_ascii_lowercase();
        if sl == "true" || sl == "false" || sl == "null" {
            return true;
        }
        if s.starts_with('\'') && s.ends_with('\'') && s.len() >= 2 {
            return true;
        }

        // Numeric literal (int/float). Be permissive: accept leading +/- and a single dot.
        let mut saw_digit = false;
        let mut saw_dot = false;
        for (i, ch) in s.chars().enumerate() {
            if (ch == '+' || ch == '-') && i == 0 {
                continue;
            }
            if ch.is_ascii_digit() {
                saw_digit = true;
                continue;
            }
            if ch == '.' && !saw_dot {
                saw_dot = true;
                continue;
            }
            return false;
        }
        saw_digit
    }
    
    /// Direct aggregate computation without index collection - ultra-fast for full table scans
    pub(crate) fn execute_aggregate_direct(stmt: &SelectStatement, table: &ColumnTable) -> Result<SqlResult, ApexError> {
        let schema = table.schema_ref();
        let columns = table.columns_ref();
        let deleted = table.deleted_ref();
        let row_count = table.get_row_count();
        
        // Collect aggregate specs
        let mut agg_specs: Vec<(AggregateFunc, Option<String>, bool, String)> = Vec::new();
        for col in &stmt.columns {
            if let SelectColumn::Aggregate { func, column, distinct, alias } = col {
                let col_name = alias.clone().unwrap_or_else(|| {
                    let func_name = match func {
                        AggregateFunc::Count => "COUNT",
                        AggregateFunc::Sum => "SUM",
                        AggregateFunc::Avg => "AVG",
                        AggregateFunc::Min => "MIN",
                        AggregateFunc::Max => "MAX",
                    };
                    if let Some(c) = column {
                        if *distinct {
                            format!("{}(DISTINCT {})", func_name, c)
                        } else {
                            format!("{}({})", func_name, c)
                        }
                    } else {
                        format!("{}(*)", func_name)
                    }
                });
                agg_specs.push((func.clone(), column.clone(), *distinct, col_name));
            }
        }

        // Fast path: mixed aggregates on internal _id plus COUNT(*)/COUNT(constant)
        // Example: SELECT MIN(_id), MAX(_id), COUNT(1) FROM t
        if agg_specs.iter().all(|(_, _, d, _)| !*d) {
            if Self::agg_specs_id_and_count_star_like_only(&agg_specs.iter().map(|(f,c,_,n)| (f.clone(), c.clone(), n.clone())).collect::<Vec<_>>()) {
                return Self::compute_aggregates_id_mixed_direct(&agg_specs.iter().map(|(f,c,_,n)| (f.clone(), c.clone(), n.clone())).collect::<Vec<_>>(), deleted, row_count);
            }
        }
        
        // Check if all aggregates use same numeric column
        let same_column: Option<&str> = {
            let cols: Vec<_> = agg_specs.iter().filter_map(|(_, c, d, _)| if *d { None } else { c.as_deref() }).collect();
            if cols.is_empty() || cols.windows(2).all(|w| w[0] == w[1]) { cols.first().copied() }
            else { None }
        };

        // Ultra-fast path: aggregates on internal _id (row index)
        if let Some("_id") = same_column {
            return Self::compute_aggregates_id_direct(
                &agg_specs
                    .iter()
                    .map(|(f, c, _, n)| (f.clone(), c.clone(), n.clone()))
                    .collect::<Vec<_>>(),
                deleted,
                row_count,
            );
        }
        
        // Ultra-fast path: direct Int64 column scan
        if let Some(col_name) = same_column {
            if let Some(col_idx) = schema.get_index(col_name) {
                if let crate::table::column_table::TypedColumn::Int64 { data, nulls } = &columns[col_idx] {
                    return Self::compute_aggregates_int64_direct(
                        &agg_specs
                            .iter()
                            .map(|(f, c, _, n)| (f.clone(), c.clone(), n.clone()))
                            .collect::<Vec<_>>(),
                        data,
                        nulls,
                        deleted,
                        row_count,
                    );
                }
            }
        }
        
        // Fast path: direct scan without index collection
        // Generic direct path (supports COUNT(DISTINCT))
        Self::compute_aggregates_direct_with_distinct(&agg_specs, schema, columns, deleted, row_count)
    }

    fn compute_aggregates_direct_with_distinct(
        agg_specs: &[(AggregateFunc, Option<String>, bool, String)],
        schema: &crate::table::column_table::ColumnSchema,
        columns: &[crate::table::column_table::TypedColumn],
        deleted: &crate::table::column_table::BitVec,
        row_count: usize,
    ) -> Result<SqlResult, ApexError> {
        use std::collections::HashSet;

        let mut result_columns = Vec::with_capacity(agg_specs.len());
        let mut result_values = Vec::with_capacity(agg_specs.len());

        // Pre-resolve column indices
        let col_indices: Vec<Option<usize>> = agg_specs
            .iter()
            .map(|(_, c, _, _)| c.as_ref().and_then(|cc| schema.get_index(cc)))
            .collect();

        // For COUNT(DISTINCT), prepare sets
        let mut distinct_sets: Vec<Option<HashSet<Vec<u8>>>> = agg_specs
            .iter()
            .map(|(f, _, d, _)| {
                if matches!(f, AggregateFunc::Count) && *d {
                    Some(HashSet::new())
                } else {
                    None
                }
            })
            .collect();

        // Accumulators for non-distinct aggregates
        let mut count_star: Vec<i64> = vec![0; agg_specs.len()];
        let mut count_col: Vec<i64> = vec![0; agg_specs.len()];
        let mut sum: Vec<f64> = vec![0.0; agg_specs.len()];
        let mut sum_count: Vec<i64> = vec![0; agg_specs.len()];
        let mut minv: Vec<Option<Value>> = vec![None; agg_specs.len()];
        let mut maxv: Vec<Option<Value>> = vec![None; agg_specs.len()];

        for row_idx in 0..row_count {
            if deleted.get(row_idx) {
                continue;
            }
            for (i, (func, _, distinct, _)) in agg_specs.iter().enumerate() {
                let ci = col_indices[i];
                match func {
                    AggregateFunc::Count => {
                        if *distinct {
                            if let Some(col_idx) = ci {
                                if let Some(v) = columns[col_idx].get(row_idx) {
                                    if !v.is_null() {
                                        if let Some(set) = distinct_sets[i].as_mut() {
                                            set.insert(v.to_bytes());
                                        }
                                    }
                                }
                            }
                        } else if ci.is_none() {
                            count_star[i] += 1;
                        } else if let Some(col_idx) = ci {
                            if let Some(v) = columns[col_idx].get(row_idx) {
                                if !v.is_null() {
                                    count_col[i] += 1;
                                }
                            }
                        }
                    }
                    AggregateFunc::Sum | AggregateFunc::Avg => {
                        if *distinct {
                            // Not supported by parser
                        } else if let Some(col_idx) = ci {
                            if let Some(v) = columns[col_idx].get(row_idx) {
                                if let Some(n) = v.as_f64() {
                                    sum[i] += n;
                                    sum_count[i] += 1;
                                }
                            }
                        }
                    }
                    AggregateFunc::Min => {
                        if *distinct {
                            // Not supported by parser
                        } else if let Some(col_idx) = ci {
                            if let Some(v) = columns[col_idx].get(row_idx) {
                                if v.is_null() {
                                    continue;
                                }
                                let cur = minv[i].take();
                                minv[i] = Some(match cur {
                                    None => v,
                                    Some(cv) => {
                                        if SqlExecutor::compare_non_null(&cv, &v) == Ordering::Greater {
                                            v
                                        } else {
                                            cv
                                        }
                                    }
                                });
                            }
                        }
                    }
                    AggregateFunc::Max => {
                        if *distinct {
                            // Not supported by parser
                        } else if let Some(col_idx) = ci {
                            if let Some(v) = columns[col_idx].get(row_idx) {
                                if v.is_null() {
                                    continue;
                                }
                                let cur = maxv[i].take();
                                maxv[i] = Some(match cur {
                                    None => v,
                                    Some(cv) => {
                                        if SqlExecutor::compare_non_null(&cv, &v) == Ordering::Less {
                                            v
                                        } else {
                                            cv
                                        }
                                    }
                                });
                            }
                        }
                    }
                }
            }
        }

        for (i, (func, _, distinct, name)) in agg_specs.iter().enumerate() {
            result_columns.push(name.clone());
            let v = match func {
                AggregateFunc::Count => {
                    if *distinct {
                        let n = distinct_sets[i].as_ref().map(|s| s.len()).unwrap_or(0);
                        Value::Int64(n as i64)
                    } else if col_indices[i].is_none() {
                        Value::Int64(count_star[i])
                    } else {
                        Value::Int64(count_col[i])
                    }
                }
                AggregateFunc::Sum => Value::Float64(sum[i]),
                AggregateFunc::Avg => {
                    if sum_count[i] > 0 {
                        Value::Float64(sum[i] / sum_count[i] as f64)
                    } else {
                        Value::Null
                    }
                }
                AggregateFunc::Min => minv[i].clone().unwrap_or(Value::Null),
                AggregateFunc::Max => maxv[i].clone().unwrap_or(Value::Null),
            };
            result_values.push(v);
        }

        Ok(SqlResult::new(result_columns, vec![result_values]))
    }

    /// Ultra-fast aggregates for internal _id without index collection.
    /// Treats _id as the row index (Int64), skipping deleted rows.
    fn compute_aggregates_id_direct(
        agg_specs: &[(AggregateFunc, Option<String>, String)],
        deleted: &crate::table::column_table::BitVec,
        row_count: usize,
    ) -> Result<SqlResult, ApexError> {
        let no_deletes = deleted.all_false();

        // O(1) when there are no deleted rows
        let (count_star, sum, min_val, max_val) = if no_deletes {
            if row_count == 0 {
                (0i64, 0i64, i64::MAX, i64::MIN)
            } else {
                let n = row_count as i64;
                let minv = 0i64;
                let maxv = n - 1;
                let sumv = (n - 1) * n / 2;
                (n, sumv, minv, maxv)
            }
        } else {
            let mut count_star = 0i64;
            let mut sum = 0i64;
            let mut min_val = i64::MAX;
            let mut max_val = i64::MIN;

            for row_idx in 0..row_count {
                if deleted.get(row_idx) {
                    continue;
                }
                let v = row_idx as i64;
                count_star += 1;
                sum += v;
                if v < min_val {
                    min_val = v;
                }
                if v > max_val {
                    max_val = v;
                }
            }
            (count_star, sum, min_val, max_val)
        };

        let mut result_columns = Vec::with_capacity(agg_specs.len());
        let mut result_values = Vec::with_capacity(agg_specs.len());

        for (func, column, name) in agg_specs {
            result_columns.push(name.clone());
            let value = match func {
                AggregateFunc::Count => {
                    if column.is_none() {
                        Value::Int64(count_star)
                    } else {
                        // _id is never NULL for non-deleted rows
                        Value::Int64(count_star)
                    }
                }
                AggregateFunc::Sum => Value::Float64(sum as f64),
                AggregateFunc::Avg => {
                    if count_star > 0 {
                        Value::Float64(sum as f64 / count_star as f64)
                    } else {
                        Value::Null
                    }
                }
                AggregateFunc::Min => {
                    if count_star > 0 {
                        Value::Int64(min_val)
                    } else {
                        Value::Null
                    }
                }
                AggregateFunc::Max => {
                    if count_star > 0 {
                        Value::Int64(max_val)
                    } else {
                        Value::Null
                    }
                }
            };
            result_values.push(value);
        }

        Ok(SqlResult::new(result_columns, vec![result_values]))
    }

    #[inline]
    fn agg_specs_id_and_count_star_like_only(agg_specs: &[(AggregateFunc, Option<String>, String)]) -> bool {
        agg_specs.iter().all(|(func, col, _)| {
            match func {
                // Any aggregate on _id is allowed
                AggregateFunc::Min | AggregateFunc::Max | AggregateFunc::Sum | AggregateFunc::Avg => {
                    matches!(col.as_deref(), Some("_id"))
                }
                AggregateFunc::Count => {
                    // COUNT(*) / COUNT(constant) / COUNT(_id)
                    matches!(col.as_deref(), Some("_id")) || Self::is_count_star_like(col)
                }
            }
        })
    }

    /// Mixed aggregate computation for `_id` plus COUNT(*)/COUNT(constant).
    /// Keeps user-visible column names (e.g. COUNT(1)).
    fn compute_aggregates_id_mixed_direct(
        agg_specs: &[(AggregateFunc, Option<String>, String)],
        deleted: &crate::table::column_table::BitVec,
        row_count: usize,
    ) -> Result<SqlResult, ApexError> {
        // We can reuse the id-direct computation because it already produces all functions,
        // but we need COUNT(constant) to behave like COUNT(*) in value semantics.
        let no_deletes = deleted.all_false();

        // O(1) when there are no deleted rows
        let (count_star, sum, min_val, max_val) = if no_deletes {
            if row_count == 0 {
                (0i64, 0i64, i64::MAX, i64::MIN)
            } else {
                let n = row_count as i64;
                let minv = 0i64;
                let maxv = n - 1;
                let sumv = (n - 1) * n / 2;
                (n, sumv, minv, maxv)
            }
        } else {
            let mut count_star = 0i64;
            let mut sum = 0i64;
            let mut min_val = i64::MAX;
            let mut max_val = i64::MIN;
            for row_idx in 0..row_count {
                if deleted.get(row_idx) {
                    continue;
                }
                let v = row_idx as i64;
                count_star += 1;
                sum += v;
                if v < min_val {
                    min_val = v;
                }
                if v > max_val {
                    max_val = v;
                }
            }
            (count_star, sum, min_val, max_val)
        };

        let mut result_columns = Vec::with_capacity(agg_specs.len());
        let mut result_values = Vec::with_capacity(agg_specs.len());
        for (func, column, name) in agg_specs {
            result_columns.push(name.clone());
            let value = match func {
                AggregateFunc::Count => {
                    // COUNT(_id) / COUNT(*) / COUNT(constant) all equal number of non-deleted rows
                    Value::Int64(count_star)
                }
                AggregateFunc::Sum => Value::Float64(sum as f64),
                AggregateFunc::Avg => {
                    if count_star > 0 {
                        Value::Float64(sum as f64 / count_star as f64)
                    } else {
                        Value::Null
                    }
                }
                AggregateFunc::Min => {
                    if count_star > 0 {
                        Value::Int64(min_val)
                    } else {
                        Value::Null
                    }
                }
                AggregateFunc::Max => {
                    if count_star > 0 {
                        Value::Int64(max_val)
                    } else {
                        Value::Null
                    }
                }
            };
            // Keep column name as-is (COUNT(1) etc.)
            let _ = column;
            result_values.push(value);
        }

        Ok(SqlResult::new(result_columns, vec![result_values]))
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
            is_id: bool,
            count: i64,
            sum: f64,
            min: Option<Value>,
            max: Option<Value>,
        }
        
        let no_deletes = deleted.all_false();
        let mut accumulators: Vec<Accumulator> = agg_specs.iter()
            .map(|(_, col, _)| {
                let is_id = matches!(col.as_deref(), Some("_id"));
                let col_idx = if is_id { None } else { col.as_ref().and_then(|c| schema.get_index(c)) };
                Accumulator { col_idx, is_id, count: 0, sum: 0.0, min: None, max: None }
            })
            .collect();
        
        for row_idx in 0..row_count {
            if !no_deletes && deleted.get(row_idx) { continue; }
            
            for (i, (func, _, _)) in agg_specs.iter().enumerate() {
                let acc = &mut accumulators[i];
                if acc.is_id {
                    let val = Value::Int64(row_idx as i64);
                    acc.count += 1;
                    if let Value::Int64(n) = &val {
                        acc.sum += *n as f64;
                    }
                    if matches!(func, AggregateFunc::Min | AggregateFunc::Max) {
                        match &acc.min {
                            None => {
                                acc.min = Some(val.clone());
                                acc.max = Some(val);
                            }
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
                } else if let Some(col_idx) = acc.col_idx {
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
                    // COUNT(*)
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
    pub(crate) fn execute_streaming_limit(
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
    pub(crate) fn execute_streaming_topk(
        stmt: &SelectStatement,
        result_columns: &[String],
        column_indices: &[(String, Option<usize>)],
        table: &ColumnTable,
    ) -> Result<SqlResult, ApexError> {
        let deleted = table.deleted_ref();
        let row_count = table.get_row_count();
        let no_deletes = deleted.all_false();
        let k = stmt.offset.unwrap_or(0) + stmt.limit.unwrap_or(usize::MAX);

        // Collect candidate indices, then reuse unified engine sort semantics.
        let mut candidates: Vec<usize> = Vec::with_capacity(row_count.min(k.saturating_mul(4)).max(1024));
        for row_idx in 0..row_count {
            if !no_deletes && deleted.get(row_idx) {
                continue;
            }
            candidates.push(row_idx);
        }

        let top_indices = crate::query::engine::ops::sort_indices_by_columns_topk(
            &candidates,
            &stmt.order_by,
            table,
            k,
        )?;
        let columns = table.columns_ref();
        
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
        let string_col = match &columns[like_col_idx] {
            crate::table::column_table::TypedColumn::String(col) => Some(col),
            _ => None,
        };
        
        let col = match string_col {
            Some(c) => c,
            None => return Ok(SqlResult::new(result_columns.to_vec(), Vec::new())),
        };
        
        let data_len = col.len().min(row_count);
        
        let mut rows = Vec::with_capacity(limit.min(100));
        let mut found = 0usize;
        
        // Streaming scan with early termination
        for row_idx in 0..data_len {
            // Skip deleted rows
            if !no_deletes && deleted.get(row_idx) { continue; }
            // Skip null values
            if col.is_null(row_idx) { continue; }
            
            // Check LIKE match
            let matches = col.get(row_idx).map(|s| matcher.matches(s)).unwrap_or(false);
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
    pub(crate) fn execute_streaming_where_limit(
        stmt: &SelectStatement,
        result_columns: &[String],
        column_indices: &[(String, Option<usize>)],
        table: &ColumnTable,
        where_expr: &SqlExpr,
    ) -> Result<SqlResult, ApexError> {
        let offset = stmt.offset.unwrap_or(0);
        let limit = stmt.limit.unwrap_or(usize::MAX);
        
        // Convert WHERE expression to optimized filter
        if let Ok(filter) = sql_expr_to_filter(where_expr) {
            // Use IoEngine for filtered data reading with streaming early termination
            let matching_indices =
                IoEngine::read_filtered_indices(table, &filter, Some(limit), offset);

            // Build result rows using IoEngine
            let rows: Vec<Vec<Value>> = matching_indices
                .iter()
                .map(|&row_idx| IoEngine::build_row_values(table, row_idx, column_indices))
                .collect();

            return Ok(SqlResult::new(result_columns.to_vec(), rows));
        }

        // Fallback: row-wise predicate evaluation with early termination
        let ctx = crate::query::engine::ops::new_eval_context();
        let deleted = table.deleted_ref();
        let row_count = table.get_row_count();
        let no_deletes = deleted.all_false();

        let mut rows: Vec<Vec<Value>> = Vec::with_capacity(limit.min(1024));
        let mut seen_matches = 0usize;
        for row_idx in 0..row_count {
            if !no_deletes && deleted.get(row_idx) {
                continue;
            }
            if !Self::eval_where_predicate(where_expr, table, row_idx, &ctx)? {
                continue;
            }
            seen_matches += 1;
            if seen_matches <= offset {
                continue;
            }
            rows.push(IoEngine::build_row_values(table, row_idx, column_indices));
            if rows.len() >= limit {
                break;
            }
        }

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
        let deleted = table.deleted_ref();
        let columns = table.columns_ref();
        let schema = table.schema_ref();
        let row_count = table.get_row_count();
        let no_deletes = deleted.all_false();
        let k = stmt.offset.unwrap_or(0) + stmt.limit.unwrap_or(usize::MAX);

        let mut candidates: Vec<usize> = Vec::new();
        if let Ok(filter) = sql_expr_to_filter(where_expr) {
            let evaluator = StreamingFilterEvaluator::new(&filter, schema, columns);
            for row_idx in 0..row_count {
                if !no_deletes && deleted.get(row_idx) {
                    continue;
                }
                if !evaluator.matches(row_idx) {
                    continue;
                }
                candidates.push(row_idx);
            }
        } else {
            let ctx = crate::query::engine::ops::new_eval_context();
            for row_idx in 0..row_count {
                if !no_deletes && deleted.get(row_idx) {
                    continue;
                }
                if !Self::eval_where_predicate(where_expr, table, row_idx, &ctx)? {
                    continue;
                }
                candidates.push(row_idx);
            }
        }

        let top_indices = crate::query::engine::ops::sort_indices_by_columns_topk(
            &candidates,
            &stmt.order_by,
            table,
            k,
        )?;

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
    
    /// Streaming DISTINCT + LIMIT - stops early when we have enough unique values
    /// Memory-optimized: uses u64 hash instead of String for deduplication
    pub(crate) fn execute_streaming_distinct(
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
    pub(crate) fn count_matching_rows(where_expr: &SqlExpr, table: &ColumnTable) -> Result<usize, ApexError> {
        use rayon::prelude::*;
        
        let filter = sql_expr_to_filter(where_expr)?;
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
        let k = stmt.offset.unwrap_or(0) + stmt.limit.unwrap_or(usize::MAX);
        let top_indices = crate::query::engine::ops::sort_indices_by_columns_topk(
            matching_indices,
            &stmt.order_by,
            table,
            k,
        )?;

        let columns = table.columns_ref();
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
        let mut agg_specs: Vec<(AggregateFunc, Option<String>, bool, String)> = Vec::new();
        for col in &stmt.columns {
            if let SelectColumn::Aggregate { func, column, distinct, alias } = col {
                let col_name = alias.clone().unwrap_or_else(|| {
                    let func_name = match func {
                        AggregateFunc::Count => "COUNT",
                        AggregateFunc::Sum => "SUM",
                        AggregateFunc::Avg => "AVG",
                        AggregateFunc::Min => "MIN",
                        AggregateFunc::Max => "MAX",
                    };
                    if let Some(c) = column {
                        if *distinct {
                            format!("{}(DISTINCT {})", func_name, c)
                        } else {
                            format!("{}({})", func_name, c)
                        }
                    } else {
                        format!("{}(*)", func_name)
                    }
                });
                agg_specs.push((func.clone(), column.clone(), *distinct, col_name));
            }
        }

        let has_distinct = agg_specs.iter().any(|(f, c, d, _)| matches!(f, AggregateFunc::Count) && *d && c.is_some());
        
        // Check if all aggregates use the same numeric column - enable ultra-fast path
        let same_column: Option<&str> = {
            let cols: Vec<_> = agg_specs.iter()
                .filter_map(|(_, c, d, _)| if *d { None } else { c.as_deref() })
                .collect();
            if cols.is_empty() || cols.windows(2).all(|w| w[0] == w[1]) {
                cols.first().copied()
            } else {
                None
            }
        };

        // Ultra-fast path: all aggregates on internal _id (row index)
        if !has_distinct {
            if let Some("_id") = same_column {
                return Self::compute_aggregates_id_fused(&agg_specs.iter().map(|(f,c,_,n)| (f.clone(), c.clone(), n.clone())).collect::<Vec<_>>(), matching_indices);
            }
        }
        
        // Ultra-fast path: all aggregates on same Int64 column
        if !has_distinct {
            if let Some(col_name) = same_column {
                if let Some(col_idx) = schema.get_index(col_name) {
                    if let crate::table::column_table::TypedColumn::Int64 { data, nulls } = &columns[col_idx] {
                        return Self::compute_aggregates_int64_fused(&agg_specs.iter().map(|(f,c,_,n)| (f.clone(), c.clone(), n.clone())).collect::<Vec<_>>(), data, nulls, matching_indices);
                    }
                }
            }
        }
        
        // Fast path: compute all aggregates in single pass with accumulators
        Self::compute_aggregates_generic_fused_with_distinct(&agg_specs, schema, columns, matching_indices)
    }

    fn compute_aggregates_generic_fused_with_distinct(
        agg_specs: &[(AggregateFunc, Option<String>, bool, String)],
        schema: &crate::table::column_table::ColumnSchema,
        columns: &[crate::table::column_table::TypedColumn],
        indices: &[usize],
    ) -> Result<SqlResult, ApexError> {
        use std::collections::HashSet;

        struct Accumulator {
            col_idx: Option<usize>,
            distinct: bool,
            seen: Option<HashSet<Vec<u8>>>,
            count: i64,
            sum: f64,
            min: Option<Value>,
            max: Option<Value>,
            sum_count: i64,
        }

        let mut accumulators: Vec<Accumulator> = agg_specs
            .iter()
            .map(|(func, col, distinct, _)| {
                let col_idx = col.as_ref().and_then(|c| schema.get_index(c));
                let seen = if matches!(func, AggregateFunc::Count) && *distinct {
                    Some(HashSet::new())
                } else {
                    None
                };
                Accumulator {
                    col_idx,
                    distinct: *distinct,
                    seen,
                    count: 0,
                    sum: 0.0,
                    min: None,
                    max: None,
                    sum_count: 0,
                }
            })
            .collect();

        for &row_idx in indices {
            for (i, (func, _, _, _)) in agg_specs.iter().enumerate() {
                let acc = &mut accumulators[i];
                match func {
                    AggregateFunc::Count => {
                        if acc.distinct {
                            if let Some(ci) = acc.col_idx {
                                if let Some(v) = columns[ci].get(row_idx) {
                                    if !v.is_null() {
                                        if let Some(set) = acc.seen.as_mut() {
                                            set.insert(v.to_bytes());
                                        }
                                    }
                                }
                            }
                        } else if let Some(ci) = acc.col_idx {
                            if let Some(v) = columns[ci].get(row_idx) {
                                if !v.is_null() {
                                    acc.count += 1;
                                }
                            }
                        } else {
                            acc.count += 1;
                        }
                    }
                    AggregateFunc::Sum | AggregateFunc::Avg => {
                        if let Some(ci) = acc.col_idx {
                            if let Some(v) = columns[ci].get(row_idx) {
                                if let Some(n) = v.as_f64() {
                                    acc.sum += n;
                                    acc.sum_count += 1;
                                }
                            }
                        }
                    }
                    AggregateFunc::Min => {
                        if let Some(ci) = acc.col_idx {
                            if let Some(v) = columns[ci].get(row_idx) {
                                if v.is_null() {
                                    continue;
                                }
                                match &acc.min {
                                    None => {
                                        acc.min = Some(v.clone());
                                        acc.max = Some(v);
                                    }
                                    Some(curr_min) => {
                                        if SqlExecutor::compare_non_null(curr_min, &v) == Ordering::Greater {
                                            acc.min = Some(v.clone());
                                        }
                                        if let Some(curr_max) = &acc.max {
                                            if SqlExecutor::compare_non_null(curr_max, &v) == Ordering::Less {
                                                acc.max = Some(v);
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                    AggregateFunc::Max => {
                        // handled in Min block via shared logic above
                        if let Some(ci) = acc.col_idx {
                            if let Some(v) = columns[ci].get(row_idx) {
                                if v.is_null() {
                                    continue;
                                }
                                match &acc.max {
                                    None => acc.max = Some(v),
                                    Some(curr_max) => {
                                        if SqlExecutor::compare_non_null(curr_max, &v) == Ordering::Less {
                                            acc.max = Some(v);
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        let mut result_columns = Vec::with_capacity(agg_specs.len());
        let mut result_values = Vec::with_capacity(agg_specs.len());

        for (i, (func, _, _, name)) in agg_specs.iter().enumerate() {
            result_columns.push(name.clone());
            let acc = &accumulators[i];
            let value = match func {
                AggregateFunc::Count => {
                    if acc.distinct {
                        Value::Int64(acc.seen.as_ref().map(|s| s.len()).unwrap_or(0) as i64)
                    } else {
                        Value::Int64(acc.count)
                    }
                }
                AggregateFunc::Sum => Value::Float64(acc.sum),
                AggregateFunc::Avg => {
                    if acc.sum_count > 0 {
                        Value::Float64(acc.sum / acc.sum_count as f64)
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

    /// Ultra-fast fused aggregates for internal _id using matching row indices.
    /// Treats _id as the row index (Int64). Note that `_id` is never NULL.
    fn compute_aggregates_id_fused(
        agg_specs: &[(AggregateFunc, Option<String>, String)],
        indices: &[usize],
    ) -> Result<SqlResult, ApexError> {
        let mut count_star = 0i64;
        let mut sum = 0i64;
        let mut min_val = i64::MAX;
        let mut max_val = i64::MIN;

        for &row_idx in indices {
            let v = row_idx as i64;
            count_star += 1;
            sum += v;
            if v < min_val {
                min_val = v;
            }
            if v > max_val {
                max_val = v;
            }
        }

        let mut result_columns = Vec::with_capacity(agg_specs.len());
        let mut result_values = Vec::with_capacity(agg_specs.len());
        for (func, column, name) in agg_specs {
            result_columns.push(name.clone());
            let value = match func {
                AggregateFunc::Count => {
                    if column.is_none() {
                        Value::Int64(count_star)
                    } else {
                        Value::Int64(count_star)
                    }
                }
                AggregateFunc::Sum => Value::Float64(sum as f64),
                AggregateFunc::Avg => {
                    if count_star > 0 {
                        Value::Float64(sum as f64 / count_star as f64)
                    } else {
                        Value::Null
                    }
                }
                AggregateFunc::Min => {
                    if count_star > 0 {
                        Value::Int64(min_val)
                    } else {
                        Value::Null
                    }
                }
                AggregateFunc::Max => {
                    if count_star > 0 {
                        Value::Int64(max_val)
                    } else {
                        Value::Null
                    }
                }
            };
            result_values.push(value);
        }

        Ok(SqlResult::new(result_columns, vec![result_values]))
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
    pub(crate) fn execute_group_by(
        stmt: &SelectStatement,
        matching_indices: &[usize],
        table: &ColumnTable,
    ) -> Result<SqlResult, ApexError> {
        let schema = table.schema_ref();
        let columns = table.columns_ref();

        #[derive(Clone, Eq)]
        enum KeyPart {
            Null,
            Int(i64),
            Float(u64),
            Bool(bool),
            StrId(u32),
            Other(String),
        }

        impl PartialEq for KeyPart {
            fn eq(&self, other: &Self) -> bool {
                match (self, other) {
                    (KeyPart::Null, KeyPart::Null) => true,
                    (KeyPart::Int(a), KeyPart::Int(b)) => a == b,
                    (KeyPart::Float(a), KeyPart::Float(b)) => a == b,
                    (KeyPart::Bool(a), KeyPart::Bool(b)) => a == b,
                    (KeyPart::StrId(a), KeyPart::StrId(b)) => a == b,
                    (KeyPart::Other(a), KeyPart::Other(b)) => a == b,
                    _ => false,
                }
            }
        }

        impl Hash for KeyPart {
            fn hash<H: Hasher>(&self, state: &mut H) {
                match self {
                    KeyPart::Null => 0u8.hash(state),
                    KeyPart::Int(v) => {
                        1u8.hash(state);
                        v.hash(state);
                    }
                    KeyPart::Float(bits) => {
                        2u8.hash(state);
                        bits.hash(state);
                    }
                    KeyPart::Bool(b) => {
                        3u8.hash(state);
                        b.hash(state);
                    }
                    KeyPart::StrId(id) => {
                        4u8.hash(state);
                        id.hash(state);
                    }
                    KeyPart::Other(s) => {
                        5u8.hash(state);
                        s.hash(state);
                    }
                }
            }
        }

        #[derive(Clone, Eq)]
        struct GroupKey {
            parts: Vec<KeyPart>,
        }

        impl PartialEq for GroupKey {
            fn eq(&self, other: &Self) -> bool {
                self.parts == other.parts
            }
        }

        impl Hash for GroupKey {
            fn hash<H: Hasher>(&self, state: &mut H) {
                self.parts.hash(state)
            }
        }

        // Local string interner for GROUP BY keys: each distinct string allocated once.
        let mut string_intern: HashMap<String, u32> = HashMap::new();
        let mut next_str_id: u32 = 0;

        #[derive(Clone)]
        struct AggState {
            func: AggregateFunc,
            col_idx: Option<usize>,
            distinct: bool,
            seen: Option<std::collections::HashSet<Vec<u8>>>,
            count_star: i64,
            count_non_null: i64,
            sum: f64,
            sum_count: i64,
            min: Option<Value>,
            max: Option<Value>,
        }

        impl AggState {
            fn new(func: AggregateFunc, col_idx: Option<usize>, distinct: bool) -> Self {
                let seen = if matches!(func, AggregateFunc::Count) && distinct {
                    Some(std::collections::HashSet::new())
                } else {
                    None
                };
                Self {
                    func,
                    col_idx,
                    distinct,
                    seen,
                    count_star: 0,
                    count_non_null: 0,
                    sum: 0.0,
                    sum_count: 0,
                    min: None,
                    max: None,
                }
            }

            fn update(&mut self, columns: &[TypedColumn], row_idx: usize) {
                match self.func {
                    AggregateFunc::Count => {
                        if self.distinct {
                            if let Some(ci) = self.col_idx {
                                if let Some(v) = columns[ci].get(row_idx) {
                                    if !v.is_null() {
                                        if let Some(set) = self.seen.as_mut() {
                                            set.insert(v.to_bytes());
                                        }
                                    }
                                }
                            }
                        } else if self.col_idx.is_none() {
                            self.count_star += 1;
                        } else if let Some(ci) = self.col_idx {
                            if let Some(v) = columns[ci].get(row_idx) {
                                if !v.is_null() {
                                    self.count_non_null += 1;
                                }
                            }
                        }
                    }
                    AggregateFunc::Sum | AggregateFunc::Avg => {
                        if let Some(ci) = self.col_idx {
                            if let Some(v) = columns[ci].get(row_idx) {
                                if v.is_null() {
                                    return;
                                }
                                if let Some(n) = v.as_f64() {
                                    self.sum += n;
                                    self.sum_count += 1;
                                }
                            }
                        }
                    }
                    AggregateFunc::Min => {
                        if let Some(ci) = self.col_idx {
                            if let Some(v) = columns[ci].get(row_idx) {
                                if v.is_null() {
                                    return;
                                }
                                self.min = Some(match &self.min {
                                    None => v,
                                    Some(curr) => {
                                        if SqlExecutor::compare_non_null(curr, &v) == Ordering::Greater {
                                            v
                                        } else {
                                            curr.clone()
                                        }
                                    }
                                });
                            }
                        }
                    }
                    AggregateFunc::Max => {
                        if let Some(ci) = self.col_idx {
                            if let Some(v) = columns[ci].get(row_idx) {
                                if v.is_null() {
                                    return;
                                }
                                self.max = Some(match &self.max {
                                    None => v,
                                    Some(curr) => {
                                        if SqlExecutor::compare_non_null(curr, &v) == Ordering::Less {
                                            v
                                        } else {
                                            curr.clone()
                                        }
                                    }
                                });
                            }
                        }
                    }
                }
            }

            fn finalize(&self) -> Value {
                match self.func {
                    AggregateFunc::Count => {
                        if self.distinct {
                            Value::Int64(self.seen.as_ref().map(|s| s.len()).unwrap_or(0) as i64)
                        } else if self.col_idx.is_none() {
                            Value::Int64(self.count_star)
                        } else {
                            Value::Int64(self.count_non_null)
                        }
                    }
                    AggregateFunc::Sum => Value::Float64(self.sum),
                    AggregateFunc::Avg => {
                        if self.sum_count > 0 {
                            Value::Float64(self.sum / self.sum_count as f64)
                        } else {
                            Value::Null
                        }
                    }
                    AggregateFunc::Min => self.min.clone().unwrap_or(Value::Null),
                    AggregateFunc::Max => self.max.clone().unwrap_or(Value::Null),
                }
            }
        }

        #[derive(Clone)]
        struct GroupState {
            first_row_idx: usize,
            aggs: Vec<AggState>,
        }

        // Resolve GROUP BY column indices once
        let mut group_col_indices = Vec::with_capacity(stmt.group_by.len());
        for col in &stmt.group_by {
            let idx = schema
                .get_index(col)
                .ok_or_else(|| ApexError::QueryParseError(format!("Unknown GROUP BY column: {}", col)))?;
            group_col_indices.push(idx);
        }

        // Collect aggregate specs in SELECT order
        let mut agg_specs: Vec<(AggregateFunc, Option<usize>, bool)> = Vec::new();

        fn add_agg_spec(
            out: &mut Vec<(AggregateFunc, Option<usize>, bool)>,
            func: AggregateFunc,
            col_idx: Option<usize>,
            distinct: bool,
        ) {
            if !out.iter().any(|(f, c, d)| *f == func && *c == col_idx && *d == distinct) {
                out.push((func, col_idx, distinct));
            }
        }

        // 1) Explicit aggregates in SELECT list
        for col in &stmt.columns {
            if let SelectColumn::Aggregate { func, column, distinct, .. } = col {
                let col_idx = column.as_ref().and_then(|c| schema.get_index(c));
                add_agg_spec(&mut agg_specs, func.clone(), col_idx, *distinct);
            }
        }

        // 2) Aggregates inside expressions (e.g. CASE WHEN SUM(amount) ...)
        fn collect_aggs_in_expr(
            expr: &SqlExpr,
            schema: &crate::table::column_table::ColumnSchema,
            out: &mut Vec<(AggregateFunc, Option<usize>, bool)>,
        ) {
            match expr {
                SqlExpr::Paren(inner) => collect_aggs_in_expr(inner, schema, out),
                SqlExpr::BinaryOp { left, right, .. } => {
                    collect_aggs_in_expr(left, schema, out);
                    collect_aggs_in_expr(right, schema, out);
                }
                SqlExpr::UnaryOp { expr, .. } => collect_aggs_in_expr(expr, schema, out),
                SqlExpr::Between { low, high, .. } => {
                    collect_aggs_in_expr(low, schema, out);
                    collect_aggs_in_expr(high, schema, out);
                }
                SqlExpr::Function { name, args } => {
                    let func = match name.to_uppercase().as_str() {
                        "COUNT" => Some(AggregateFunc::Count),
                        "SUM" => Some(AggregateFunc::Sum),
                        "AVG" => Some(AggregateFunc::Avg),
                        "MIN" => Some(AggregateFunc::Min),
                        "MAX" => Some(AggregateFunc::Max),
                        _ => None,
                    };
                    if let Some(func) = func {
                        let col_idx = if args.is_empty() {
                            None
                        } else {
                            match &args[0] {
                                SqlExpr::Column(c) => schema.get_index(c),
                                _ => None,
                            }
                        };
                        // DISTINCT inside expressions is not parsed/represented currently; default false.
                        add_agg_spec(out, func, col_idx, false);
                    }
                    for a in args {
                        collect_aggs_in_expr(a, schema, out);
                    }
                }
                SqlExpr::Case { when_then, else_expr } => {
                    for (c, v) in when_then {
                        collect_aggs_in_expr(c, schema, out);
                        collect_aggs_in_expr(v, schema, out);
                    }
                    if let Some(e) = else_expr {
                        collect_aggs_in_expr(e, schema, out);
                    }
                }
                SqlExpr::ScalarSubquery { .. } => {
                    // Scalar subqueries inside GROUP BY expressions are not supported yet.
                }
                _ => {}
            }
        }

        for col in &stmt.columns {
            if let SelectColumn::Expression { expr, .. } = col {
                collect_aggs_in_expr(expr, schema, &mut agg_specs);
            }
        }
        if let Some(h) = stmt.having.as_ref() {
            collect_aggs_in_expr(h, schema, &mut agg_specs);
        }

        // Map SELECT aggregate aliases -> agg_specs index, so HAVING can reference them
        // e.g. SELECT COUNT(*) AS c ... HAVING c > 1
        let mut agg_alias_to_spec_idx: HashMap<String, usize> = HashMap::new();
        for col in &stmt.columns {
            if let SelectColumn::Aggregate { func, column, distinct, alias } = col {
                if let Some(alias) = alias.as_ref() {
                    let col_idx = column.as_ref().and_then(|c| schema.get_index(c));
                    if let Some(pos) = agg_specs
                        .iter()
                        .position(|(f, c, d)| *f == *func && *c == col_idx && *d == *distinct)
                    {
                        agg_alias_to_spec_idx.insert(alias.clone(), pos);
                    }
                }
            }
        }

        fn eval_having_scalar(
            expr: &SqlExpr,
            schema: &crate::table::column_table::ColumnSchema,
            columns: &[TypedColumn],
            group_state: &GroupState,
            agg_specs: &[(AggregateFunc, Option<usize>, bool)],
            agg_alias_to_spec_idx: &HashMap<String, usize>,
        ) -> Result<Value, ApexError> {
            fn cast_value(v: Value, target: crate::data::DataType) -> Result<Value, ApexError> {
                use crate::data::DataType;

                if v.is_null() {
                    return Ok(Value::Null);
                }

                match target {
                    DataType::String => Ok(Value::String(v.to_string_value())),
                    DataType::Bool => match v {
                        Value::Bool(b) => Ok(Value::Bool(b)),
                        Value::Int8(i) => Ok(Value::Bool(i != 0)),
                        Value::Int16(i) => Ok(Value::Bool(i != 0)),
                        Value::Int32(i) => Ok(Value::Bool(i != 0)),
                        Value::Int64(i) => Ok(Value::Bool(i != 0)),
                        Value::UInt8(i) => Ok(Value::Bool(i != 0)),
                        Value::UInt16(i) => Ok(Value::Bool(i != 0)),
                        Value::UInt32(i) => Ok(Value::Bool(i != 0)),
                        Value::UInt64(i) => Ok(Value::Bool(i != 0)),
                        Value::Float32(f) => Ok(Value::Bool(f != 0.0)),
                        Value::Float64(f) => Ok(Value::Bool(f != 0.0)),
                        Value::String(s) => {
                            let ls = s.trim().to_lowercase();
                            match ls.as_str() {
                                "true" | "1" | "t" | "yes" | "y" => Ok(Value::Bool(true)),
                                "false" | "0" | "f" | "no" | "n" => Ok(Value::Bool(false)),
                                _ => Err(ApexError::QueryParseError(format!(
                                    "Cannot CAST value '{}' to BOOLEAN",
                                    s
                                ))),
                            }
                        }
                        other => Err(ApexError::QueryParseError(format!(
                            "Cannot CAST {:?} to BOOLEAN",
                            other
                        ))),
                    },
                    DataType::Int8
                    | DataType::Int16
                    | DataType::Int32
                    | DataType::Int64
                    | DataType::UInt8
                    | DataType::UInt16
                    | DataType::UInt32
                    | DataType::UInt64 => {
                        if let Some(i) = v.as_i64() {
                            Ok(Value::Int64(i))
                        } else if let Value::String(s) = v {
                            let t = s.trim();
                            let parsed: i64 = t.parse().map_err(|_| {
                                ApexError::QueryParseError(format!(
                                    "Cannot CAST value '{}' to INTEGER",
                                    s
                                ))
                            })?;
                            Ok(Value::Int64(parsed))
                        } else {
                            Err(ApexError::QueryParseError(format!(
                                "Cannot CAST value '{}' to INTEGER",
                                v.to_string_value()
                            )))
                        }
                    }
                    DataType::Float32 | DataType::Float64 => {
                        if let Some(f) = v.as_f64() {
                            Ok(Value::Float64(f))
                        } else if let Value::String(s) = v {
                            let t = s.trim();
                            let parsed: f64 = t.parse().map_err(|_| {
                                ApexError::QueryParseError(format!(
                                    "Cannot CAST value '{}' to DOUBLE",
                                    s
                                ))
                            })?;
                            Ok(Value::Float64(parsed))
                        } else {
                            Err(ApexError::QueryParseError(format!(
                                "Cannot CAST value '{}' to DOUBLE",
                                v.to_string_value()
                            )))
                        }
                    }
                    _ => Err(ApexError::QueryParseError(format!(
                        "Unsupported CAST target type: {}",
                        target
                    ))),
                }
            }

            match expr {
                SqlExpr::Paren(inner) => {
                    eval_having_scalar(inner, schema, columns, group_state, agg_specs, agg_alias_to_spec_idx)
                }
                SqlExpr::Literal(v) => Ok(v.clone()),
                SqlExpr::Cast { expr, data_type } => {
                    let v = eval_having_scalar(
                        expr,
                        schema,
                        columns,
                        group_state,
                        agg_specs,
                        agg_alias_to_spec_idx,
                    )?;
                    cast_value(v, *data_type)
                }
                SqlExpr::ScalarSubquery { .. } => Err(ApexError::QueryParseError(
                    "Scalar subquery is not supported in GROUP BY/HAVING expressions".to_string(),
                )),
                SqlExpr::Case { when_then, else_expr } => {
                    for (cond, val) in when_then {
                        let cv = eval_having_scalar(
                            cond,
                            schema,
                            columns,
                            group_state,
                            agg_specs,
                            agg_alias_to_spec_idx,
                        )?;
                        if cv.as_bool().unwrap_or(false) {
                            return eval_having_scalar(
                                val,
                                schema,
                                columns,
                                group_state,
                                agg_specs,
                                agg_alias_to_spec_idx,
                            );
                        }
                    }
                    if let Some(e) = else_expr {
                        eval_having_scalar(e, schema, columns, group_state, agg_specs, agg_alias_to_spec_idx)
                    } else {
                        Ok(Value::Null)
                    }
                }
                SqlExpr::Column(name) => {
                    let row_idx = group_state.first_row_idx;
                    if name == "_id" {
                        Ok(Value::Int64(row_idx as i64))
                    } else if let Some(ci) = schema.get_index(name) {
                        Ok(columns[ci].get(row_idx).unwrap_or(Value::Null))
                    } else {
                        // Not a base column: try resolve as SELECT aggregate alias
                        if let Some(&spec_idx) = agg_alias_to_spec_idx.get(name) {
                            Ok(group_state.aggs[spec_idx].finalize())
                        } else {
                            Ok(Value::Null)
                        }
                    }
                }
                SqlExpr::Function { name, args } => {
                    if name.eq_ignore_ascii_case("rand") {
                        if !args.is_empty() {
                            return Err(ApexError::QueryParseError(
                                "RAND() does not accept arguments".to_string(),
                            ));
                        }
                        return Ok(Value::Float64(rand::random::<f64>()));
                    }
                    let func = match name.to_uppercase().as_str() {
                        "COUNT" => AggregateFunc::Count,
                        "SUM" => AggregateFunc::Sum,
                        "AVG" => AggregateFunc::Avg,
                        "MIN" => AggregateFunc::Min,
                        "MAX" => AggregateFunc::Max,
                        _ => {
                            return Err(ApexError::QueryParseError(
                                format!("Unsupported function in HAVING: {}", name),
                            ))
                        }
                    };

                    let col_idx = if args.is_empty() {
                        None
                    } else {
                        match &args[0] {
                            SqlExpr::Column(c) => schema.get_index(c),
                            _ => None,
                        }
                    };

                    for (i, (sf, sc, _sd)) in agg_specs.iter().enumerate() {
                        if *sf == func && *sc == col_idx {
                            return Ok(group_state.aggs[i].finalize());
                        }
                    }
                    Ok(Value::Null)
                }
                SqlExpr::UnaryOp { op, expr } => {
                    match op {
                        crate::query::sql_parser::UnaryOperator::Not => {
                            let v = eval_having_scalar(
                                expr,
                                schema,
                                columns,
                                group_state,
                                agg_specs,
                                agg_alias_to_spec_idx,
                            )?;
                            Ok(Value::Bool(!v.as_bool().unwrap_or(false)))
                        }
                        crate::query::sql_parser::UnaryOperator::Minus => {
                            let v = eval_having_scalar(
                                expr,
                                schema,
                                columns,
                                group_state,
                                agg_specs,
                                agg_alias_to_spec_idx,
                            )?;
                            if let Some(i) = v.as_i64() {
                                Ok(Value::Int64(-i))
                            } else if let Some(f) = v.as_f64() {
                                Ok(Value::Float64(-f))
                            } else {
                                Ok(Value::Null)
                            }
                        }
                    }
                }
                SqlExpr::BinaryOp { left, op, right } => {
                    let lv = eval_having_scalar(
                        left,
                        schema,
                        columns,
                        group_state,
                        agg_specs,
                        agg_alias_to_spec_idx,
                    )?;
                    let rv = eval_having_scalar(
                        right,
                        schema,
                        columns,
                        group_state,
                        agg_specs,
                        agg_alias_to_spec_idx,
                    )?;
                    match op {
                        BinaryOperator::And => Ok(Value::Bool(lv.as_bool().unwrap_or(false) && rv.as_bool().unwrap_or(false))),
                        BinaryOperator::Or => Ok(Value::Bool(lv.as_bool().unwrap_or(false) || rv.as_bool().unwrap_or(false))),
                        BinaryOperator::Eq => Ok(Value::Bool(lv == rv)),
                        BinaryOperator::NotEq => Ok(Value::Bool(lv != rv)),
                        BinaryOperator::Lt | BinaryOperator::Le | BinaryOperator::Gt | BinaryOperator::Ge => {
                            if lv.is_null() || rv.is_null() {
                                return Ok(Value::Bool(false));
                            }
                            let ord = match (lv.as_f64(), rv.as_f64()) {
                                (Some(a), Some(b)) => a.partial_cmp(&b).unwrap_or(Ordering::Equal),
                                _ => SqlExecutor::compare_non_null(&lv, &rv),
                            };
                            let b = match op {
                                BinaryOperator::Lt => ord == Ordering::Less,
                                BinaryOperator::Le => ord != Ordering::Greater,
                                BinaryOperator::Gt => ord == Ordering::Greater,
                                BinaryOperator::Ge => ord != Ordering::Less,
                                _ => false,
                            };
                            Ok(Value::Bool(b))
                        }
                        BinaryOperator::Add | BinaryOperator::Sub | BinaryOperator::Mul | BinaryOperator::Div | BinaryOperator::Mod => {
                            let lf = lv.as_f64();
                            let rf = rv.as_f64();
                            if let (Some(a), Some(b)) = (lf, rf) {
                                let out = match op {
                                    BinaryOperator::Add => a + b,
                                    BinaryOperator::Sub => a - b,
                                    BinaryOperator::Mul => a * b,
                                    BinaryOperator::Div => a / b,
                                    BinaryOperator::Mod => a % b,
                                    _ => a,
                                };
                                Ok(Value::Float64(out))
                            } else {
                                Ok(Value::Null)
                            }
                        }
                    }
                }
                SqlExpr::Like { .. }
                | SqlExpr::Regexp { .. }
                | SqlExpr::In { .. }
                | SqlExpr::InSubquery { .. }
                | SqlExpr::ExistsSubquery { .. }
                | SqlExpr::Between { .. }
                | SqlExpr::IsNull { .. } => Err(ApexError::QueryParseError(
                    "Unsupported predicate in HAVING".to_string(),
                )),
            }
        }

        fn eval_having_predicate(
            expr: &SqlExpr,
            schema: &crate::table::column_table::ColumnSchema,
            columns: &[TypedColumn],
            group_state: &GroupState,
            agg_specs: &[(AggregateFunc, Option<usize>, bool)],
            agg_alias_to_spec_idx: &HashMap<String, usize>,
        ) -> Result<bool, ApexError> {
            let v = eval_having_scalar(
                expr,
                schema,
                columns,
                group_state,
                agg_specs,
                agg_alias_to_spec_idx,
            )?;
            Ok(v.as_bool().unwrap_or(false))
        }

        // Group rows and compute aggregates in a single pass
        let mut groups: HashMap<GroupKey, GroupState> = HashMap::new();
        for &row_idx in matching_indices {
            let mut parts = Vec::with_capacity(group_col_indices.len());
            for &col_idx in &group_col_indices {
                let kp = match &columns[col_idx] {
                    TypedColumn::String(col) => {
                        match col.get(row_idx) {
                            None => KeyPart::Null,
                            Some(s) => {
                                if let Some(&id) = string_intern.get(s) {
                                    KeyPart::StrId(id)
                                } else {
                                    let id = next_str_id;
                                    next_str_id = next_str_id.wrapping_add(1);
                                    string_intern.insert(s.to_string(), id);
                                    KeyPart::StrId(id)
                                }
                            }
                        }
                    }
                    other => {
                        // Fallback via Value for non-string types
                        let v = other.get(row_idx).unwrap_or(Value::Null);
                        match v {
                            Value::Null => KeyPart::Null,
                            Value::Bool(b) => KeyPart::Bool(b),
                            Value::Int64(i) => KeyPart::Int(i),
                            Value::Int32(i) => KeyPart::Int(i as i64),
                            Value::Int16(i) => KeyPart::Int(i as i64),
                            Value::Int8(i) => KeyPart::Int(i as i64),
                            Value::UInt64(u) => KeyPart::Int(u as i64),
                            Value::UInt32(u) => KeyPart::Int(u as i64),
                            Value::UInt16(u) => KeyPart::Int(u as i64),
                            Value::UInt8(u) => KeyPart::Int(u as i64),
                            Value::Float64(f) => KeyPart::Float(f.to_bits()),
                            Value::Float32(f) => KeyPart::Float((f as f64).to_bits()),
                            Value::String(s) => {
                                // Should not happen (handled above), but keep safe
                                if let Some(&id) = string_intern.get(&s) {
                                    KeyPart::StrId(id)
                                } else {
                                    let id = next_str_id;
                                    next_str_id = next_str_id.wrapping_add(1);
                                    string_intern.insert(s, id);
                                    KeyPart::StrId(id)
                                }
                            }
                            other => KeyPart::Other(other.to_string_value()),
                        }
                    }
                };
                parts.push(kp);
            }

            let key = GroupKey { parts };
            let entry = groups.entry(key).or_insert_with(|| GroupState {
                first_row_idx: row_idx,
                aggs: agg_specs
                    .iter()
                    .map(|(func, col_idx, distinct)| AggState::new(func.clone(), *col_idx, *distinct))
                    .collect(),
            });

            for agg in &mut entry.aggs {
                agg.update(columns, row_idx);
            }
        }
        
        // Build result columns
        let mut result_columns = Vec::new();
        for col in &stmt.columns {
            match col {
                SelectColumn::Column(name) => result_columns.push(name.clone()),
                SelectColumn::ColumnAlias { alias, .. } => result_columns.push(alias.clone()),
                SelectColumn::Aggregate { func, column, distinct, alias } => {
                    let name = alias.clone().unwrap_or_else(|| {
                        let func_name = match func {
                            AggregateFunc::Count => "COUNT",
                            AggregateFunc::Sum => "SUM",
                            AggregateFunc::Avg => "AVG",
                            AggregateFunc::Min => "MIN",
                            AggregateFunc::Max => "MAX",
                        };
                        if let Some(c) = column {
                            if *distinct {
                                format!("{}(DISTINCT {})", func_name, c)
                            } else {
                                format!("{}({})", func_name, c)
                            }
                        } else {
                            format!("{}(*)", func_name)
                        }
                    });
                    result_columns.push(name);
                }
                SelectColumn::Expression { alias, .. } => {
                    result_columns.push(alias.clone().unwrap_or_else(|| "expr".to_string()));
                }
                _ => {}
            }
        }
        
        // Build result rows
        use arrow::array::{ArrayRef, BooleanBuilder, Float64Builder, Int64Builder, StringBuilder};
        use arrow::datatypes::{DataType as ArrowDataType, Field};
        use std::sync::Arc;

        enum ColBuilder {
            Int64(Int64Builder),
            Float64(Float64Builder),
            Bool(BooleanBuilder),
            Utf8(StringBuilder),
        }

        impl ColBuilder {
            fn append_value(&mut self, v: &Value) {
                match self {
                    ColBuilder::Int64(b) => {
                        if let Some(i) = v.as_i64() { b.append_value(i); } else { b.append_null(); }
                    }
                    ColBuilder::Float64(b) => {
                        if let Some(f) = v.as_f64() { b.append_value(f); } else { b.append_null(); }
                    }
                    ColBuilder::Bool(b) => {
                        if let Some(x) = v.as_bool() { b.append_value(x); } else { b.append_null(); }
                    }
                    ColBuilder::Utf8(b) => {
                        if v.is_null() { b.append_null(); } else { b.append_value(v.to_string_value()); }
                    }
                }
            }

            fn finish(self) -> ArrayRef {
                match self {
                    ColBuilder::Int64(mut b) => Arc::new(b.finish()),
                    ColBuilder::Float64(mut b) => Arc::new(b.finish()),
                    ColBuilder::Bool(mut b) => Arc::new(b.finish()),
                    ColBuilder::Utf8(mut b) => Arc::new(b.finish()),
                }
            }
        }

        // Pre-build Arrow schema + builders from SELECT list
        let mut fields: Vec<Field> = Vec::with_capacity(result_columns.len());
        let mut builders: Vec<ColBuilder> = Vec::with_capacity(result_columns.len());

        for col in &stmt.columns {
            match col {
                SelectColumn::Column(name) => {
                    let out_name = name.clone();
                    if name == "_id" {
                        fields.push(Field::new(&out_name, ArrowDataType::Int64, true));
                        builders.push(ColBuilder::Int64(Int64Builder::new()));
                    } else if let Some(ci) = schema.get_index(name) {
                        match &columns[ci] {
                            TypedColumn::Int64 { .. } => {
                                fields.push(Field::new(&out_name, ArrowDataType::Int64, true));
                                builders.push(ColBuilder::Int64(Int64Builder::new()));
                            }
                            TypedColumn::Float64 { .. } => {
                                fields.push(Field::new(&out_name, ArrowDataType::Float64, true));
                                builders.push(ColBuilder::Float64(Float64Builder::new()));
                            }
                            TypedColumn::Bool { .. } => {
                                fields.push(Field::new(&out_name, ArrowDataType::Boolean, true));
                                builders.push(ColBuilder::Bool(BooleanBuilder::new()));
                            }
                            TypedColumn::String(_) => {
                                fields.push(Field::new(&out_name, ArrowDataType::Utf8, true));
                                builders.push(ColBuilder::Utf8(StringBuilder::new()));
                            }
                            TypedColumn::Mixed { .. } => {
                                fields.push(Field::new(&out_name, ArrowDataType::Utf8, true));
                                builders.push(ColBuilder::Utf8(StringBuilder::new()));
                            }
                        }
                    } else {
                        fields.push(Field::new(&out_name, ArrowDataType::Utf8, true));
                        builders.push(ColBuilder::Utf8(StringBuilder::new()));
                    }
                }
                SelectColumn::ColumnAlias { column, alias } => {
                    let out_name = alias.clone();
                    if column == "_id" {
                        fields.push(Field::new(&out_name, ArrowDataType::Int64, true));
                        builders.push(ColBuilder::Int64(Int64Builder::new()));
                    } else if let Some(ci) = schema.get_index(column) {
                        match &columns[ci] {
                            TypedColumn::Int64 { .. } => {
                                fields.push(Field::new(&out_name, ArrowDataType::Int64, true));
                                builders.push(ColBuilder::Int64(Int64Builder::new()));
                            }
                            TypedColumn::Float64 { .. } => {
                                fields.push(Field::new(&out_name, ArrowDataType::Float64, true));
                                builders.push(ColBuilder::Float64(Float64Builder::new()));
                            }
                            TypedColumn::Bool { .. } => {
                                fields.push(Field::new(&out_name, ArrowDataType::Boolean, true));
                                builders.push(ColBuilder::Bool(BooleanBuilder::new()));
                            }
                            TypedColumn::String(_) => {
                                fields.push(Field::new(&out_name, ArrowDataType::Utf8, true));
                                builders.push(ColBuilder::Utf8(StringBuilder::new()));
                            }
                            TypedColumn::Mixed { .. } => {
                                fields.push(Field::new(&out_name, ArrowDataType::Utf8, true));
                                builders.push(ColBuilder::Utf8(StringBuilder::new()));
                            }
                        }
                    } else {
                        fields.push(Field::new(&out_name, ArrowDataType::Utf8, true));
                        builders.push(ColBuilder::Utf8(StringBuilder::new()));
                    }
                }
                SelectColumn::Aggregate { func, column, distinct, alias } => {
                    let out_name = alias.clone().unwrap_or_else(|| {
                        let func_name = match func {
                            AggregateFunc::Count => "COUNT",
                            AggregateFunc::Sum => "SUM",
                            AggregateFunc::Avg => "AVG",
                            AggregateFunc::Min => "MIN",
                            AggregateFunc::Max => "MAX",
                        };
                        if let Some(c) = column {
                            if *distinct {
                                format!("{}(DISTINCT {})", func_name, c)
                            } else {
                                format!("{}({})", func_name, c)
                            }
                        } else {
                            format!("{}(*)", func_name)
                        }
                    });

                    match func {
                        AggregateFunc::Count => {
                            fields.push(Field::new(&out_name, ArrowDataType::Int64, true));
                            builders.push(ColBuilder::Int64(Int64Builder::new()));
                        }
                        AggregateFunc::Sum | AggregateFunc::Avg => {
                            fields.push(Field::new(&out_name, ArrowDataType::Float64, true));
                            builders.push(ColBuilder::Float64(Float64Builder::new()));
                        }
                        AggregateFunc::Min | AggregateFunc::Max => {
                            // Use source column type when possible, else Utf8 fallback
                            if let Some(c) = column.as_ref() {
                                if let Some(ci) = schema.get_index(c) {
                                    match &columns[ci] {
                                        TypedColumn::Int64 { .. } => {
                                            fields.push(Field::new(&out_name, ArrowDataType::Int64, true));
                                            builders.push(ColBuilder::Int64(Int64Builder::new()));
                                        }
                                        TypedColumn::Float64 { .. } => {
                                            fields.push(Field::new(&out_name, ArrowDataType::Float64, true));
                                            builders.push(ColBuilder::Float64(Float64Builder::new()));
                                        }
                                        TypedColumn::Bool { .. } => {
                                            fields.push(Field::new(&out_name, ArrowDataType::Boolean, true));
                                            builders.push(ColBuilder::Bool(BooleanBuilder::new()));
                                        }
                                        TypedColumn::String(_) => {
                                            fields.push(Field::new(&out_name, ArrowDataType::Utf8, true));
                                            builders.push(ColBuilder::Utf8(StringBuilder::new()));
                                        }
                                        TypedColumn::Mixed { .. } => {
                                            fields.push(Field::new(&out_name, ArrowDataType::Utf8, true));
                                            builders.push(ColBuilder::Utf8(StringBuilder::new()));
                                        }
                                    }
                                } else {
                                    fields.push(Field::new(&out_name, ArrowDataType::Utf8, true));
                                    builders.push(ColBuilder::Utf8(StringBuilder::new()));
                                }
                            } else {
                                fields.push(Field::new(&out_name, ArrowDataType::Utf8, true));
                                builders.push(ColBuilder::Utf8(StringBuilder::new()));
                            }
                        }
                    }
                }
                SelectColumn::Expression { alias, .. } => {
                    let out_name = alias.clone().unwrap_or_else(|| "expr".to_string());
                    fields.push(Field::new(&out_name, ArrowDataType::Utf8, true));
                    builders.push(ColBuilder::Utf8(StringBuilder::new()));
                }
                _ => {}
            }
        }

        let having_expr = stmt.having.as_ref();

        // Pre-map explicit Aggregate columns to agg_specs indices (cannot rely on sequential order
        // once expressions also contribute aggregate specs).
        let mut select_agg_to_spec: Vec<Option<usize>> = Vec::with_capacity(stmt.columns.len());
        for col in &stmt.columns {
            if let SelectColumn::Aggregate { func, column, distinct, .. } = col {
                let col_idx = column.as_ref().and_then(|c| schema.get_index(c));
                let idx = agg_specs
                    .iter()
                    .position(|(f, c, d)| f == func && *c == col_idx && *d == *distinct);
                select_agg_to_spec.push(idx);
            } else {
                select_agg_to_spec.push(None);
            }
        }

        // Build rows (apply HAVING), then ORDER BY/LIMIT/OFFSET, then append into Arrow builders.
        // Note: groups is a HashMap so iteration order is nondeterministic.
        let mut out_rows: Vec<Vec<Value>> = Vec::with_capacity(groups.len());
        for (_key, group_state) in groups {
            if let Some(h) = having_expr {
                if !eval_having_predicate(h, schema, columns, &group_state, &agg_specs, &agg_alias_to_spec_idx)? {
                    continue;
                }
            }

            let mut row_values: Vec<Value> = Vec::with_capacity(result_columns.len());
            for (pos, col) in stmt.columns.iter().enumerate() {
                match col {
                    SelectColumn::Column(name) => {
                        let first_idx = group_state.first_row_idx;
                        if name == "_id" {
                            row_values.push(Value::Int64(first_idx as i64));
                        } else if let Some(ci) = schema.get_index(name) {
                            row_values.push(columns[ci].get(first_idx).unwrap_or(Value::Null));
                        } else {
                            row_values.push(Value::Null);
                        }
                    }
                    SelectColumn::ColumnAlias { column, .. } => {
                        let first_idx = group_state.first_row_idx;
                        if column == "_id" {
                            row_values.push(Value::Int64(first_idx as i64));
                        } else if let Some(ci) = schema.get_index(column) {
                            row_values.push(columns[ci].get(first_idx).unwrap_or(Value::Null));
                        } else {
                            row_values.push(Value::Null);
                        }
                    }
                    SelectColumn::Aggregate { .. } => {
                        let idx = select_agg_to_spec.get(pos).and_then(|x| *x);
                        if let Some(i) = idx {
                            row_values.push(group_state.aggs[i].finalize());
                        } else {
                            row_values.push(Value::Null);
                        }
                    }
                    SelectColumn::Expression { expr, .. } => {
                        row_values.push(eval_having_scalar(
                            expr,
                            schema,
                            columns,
                            &group_state,
                            &agg_specs,
                            &agg_alias_to_spec_idx,
                        )?);
                    }
                    SelectColumn::All | SelectColumn::WindowFunction { .. } => {
                        return Err(ApexError::QueryParseError(
                            "Unsupported SELECT item in GROUP BY".to_string(),
                        ));
                    }
                }
            }
            out_rows.push(row_values);
        }

        if !stmt.order_by.is_empty() {
            out_rows = Self::apply_order_by(out_rows, &result_columns, &stmt.order_by)?;
        }

        let off = stmt.offset.unwrap_or(0);
        let lim = stmt.limit.unwrap_or(usize::MAX);
        let out_rows = out_rows.into_iter().skip(off).take(lim).collect::<Vec<_>>();

        Ok(SqlResult::new(result_columns, out_rows))
    }
    
    /// Build Arrow arrays directly from column data using matching indices
    /// This is much faster than row-by-row construction for large result sets
    pub(crate) fn build_arrow_direct(
        result_columns: &[String],
        column_indices: &[(String, Option<usize>)],
        matching_indices: &[usize],
        table: &ColumnTable,
    ) -> Result<SqlResult, ApexError> {
        let projected_exprs = vec![None; column_indices.len()];
        let ctx = crate::query::engine::ops::new_eval_context();
        crate::query::engine::ops::build_arrow_direct(
            result_columns,
            column_indices,
            &projected_exprs,
            matching_indices,
            table,
            &ctx,
        )
    }
}
