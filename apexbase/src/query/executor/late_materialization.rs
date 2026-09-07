// Late materialization scan adaptation: filter-first column reads for SELECT *, ORDER BY, GROUP BY.

impl ApexExecutor {
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

        // FAST PATH: no LIMIT.
        // Fall back to full sequential read + vectorized Arrow filter.
        // Highly selective shapes are handled by dedicated mmap fast paths above.
        if need_count.is_none() {
            // Fallback: full sequential read + vectorized Arrow filter
            // Need both SELECT columns and WHERE columns (WHERE is applied on this batch)
            // For SELECT *, required_columns() returns None → read all columns
            let col_refs_vec: Option<Vec<String>> = stmt.required_columns().map(|mut sel_cols| {
                for wc in stmt.where_columns() {
                    if !sel_cols.iter().any(|c| c.eq_ignore_ascii_case(&wc)) {
                        sel_cols.push(wc);
                    }
                }
                sel_cols
            });
            let col_refs_strs: Option<Vec<&str>> = col_refs_vec
                .as_ref()
                .map(|v| v.iter().map(|s| s.as_str()).collect());
            let full_batch = backend.read_columns_to_arrow(col_refs_strs.as_deref(), 0, None)?;
            if full_batch.num_rows() > 0 {
                let mask =
                    Self::evaluate_predicate_with_storage(&full_batch, where_clause, storage_path)?;
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
                let filter_batch = backend.read_columns_to_arrow(
                    Some(&cols_to_read),
                    start_row,
                    Some(rows_to_read),
                )?;

                if filter_batch.num_rows() == 0 {
                    break;
                }

                let mask = Self::evaluate_predicate_with_storage(
                    &filter_batch,
                    where_clause,
                    storage_path,
                )?;

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
                let filter_batch = backend.read_columns_to_arrow(
                    Some(&cols_to_read),
                    start_row,
                    Some(rows_to_read),
                )?;

                if filter_batch.num_rows() == 0 {
                    break;
                }

                let mask = Self::evaluate_predicate_with_storage(
                    &filter_batch,
                    where_clause,
                    storage_path,
                )?;

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
            let col_refs = Self::get_col_refs(stmt);
            let col_refs_vec: Option<Vec<&str>> = col_refs
                .as_ref()
                .map(|v| v.iter().map(|s| s.as_str()).collect());
            let full_batch = backend.read_columns_to_arrow(col_refs_vec.as_deref(), 0, None)?;
            let indices_arr = arrow::array::UInt32Array::from(
                limited_indices
                    .iter()
                    .map(|&i| i as u32)
                    .collect::<Vec<_>>(),
            );
            let taken_cols: Vec<ArrayRef> = full_batch
                .columns()
                .iter()
                .map(|col| {
                    arrow::compute::take(col.as_ref(), &indices_arr, None)
                        .map_err(|e| err_data(e.to_string()))
                })
                .collect::<io::Result<Vec<_>>>()?;
            arrow::record_batch::RecordBatch::try_new(full_batch.schema(), taken_cols)
                .map_err(|e| err_data(e.to_string()))
        } else {
            let col_refs = Self::get_col_refs(stmt);
            let col_refs_vec: Option<Vec<&str>> = col_refs
                .as_ref()
                .map(|v| v.iter().map(|s| s.as_str()).collect());
            backend.read_columns_by_indices_to_arrow(&limited_indices, col_refs_vec.as_deref())
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
        let limit = stmt.limit.unwrap_or(0);
        let offset = stmt.offset.unwrap_or(0);
        let k = limit + offset;
        if k == 0 {
            return backend.read_columns_to_arrow(None, 0, Some(0));
        }

        // IN-MEMORY FAST PATH: 2-col ORDER BY (string, float64) — skip Arrow string conversion.
        // Uses global dict cache (u16 group_ids) + raw float64 column for typed sort keys.
        if stmt.order_by.len() == 2 && !backend.has_pending_deltas() {
            let o0 = &stmt.order_by[0];
            let o1 = &stmt.order_by[1];
            let c0 = {
                let c = o0.column.trim_matches('"');
                if let Some(p) = c.rfind('.') {
                    &c[p + 1..]
                } else {
                    c
                }
            };
            let c1 = {
                let c = o1.column.trim_matches('"');
                if let Some(p) = c.rfind('.') {
                    &c[p + 1..]
                } else {
                    c
                }
            };
            if let Some(indices) =
                backend.order_topk_str_float64(c0, !o0.descending, c1, !o1.descending, k, offset)?
            {
                if !indices.is_empty() {
                    return backend.read_columns_by_indices_to_arrow(&indices, None);
                }
                return backend.read_columns_to_arrow(None, 0, Some(0));
            }
        }

        // MMAP FAST PATH: single ORDER BY column + mmap-only → direct top-K scan without Arrow
        if backend.is_mmap_only()
            && !backend.has_delta()
            && stmt.order_by.len() == 1
            && !backend.has_pending_deltas()
        {
            let clause = &stmt.order_by[0];
            let col_name = clause.column.trim_matches('"');
            let actual_col = if let Some(p) = col_name.rfind('.') {
                &col_name[p + 1..]
            } else {
                col_name
            };
            if let Some(heap) = backend.scan_top_k_indices_mmap(actual_col, k, clause.descending)? {
                let final_indices: Vec<usize> =
                    heap.into_iter().skip(offset).map(|(idx, _)| idx).collect();
                if !final_indices.is_empty() {
                    return backend.read_columns_by_indices_to_arrow(&final_indices, None);
                }
                return backend.read_columns_to_arrow(None, 0, Some(0));
            }
        }

        // Step 1: Read only columns needed for ORDER BY
        let mut order_cols: Vec<&str> = stmt
            .order_by
            .iter()
            .map(|o| {
                let col = o.column.trim_matches('"');
                if let Some(dot_pos) = col.rfind('.') {
                    &col[dot_pos + 1..]
                } else {
                    col
                }
            })
            .collect();
        if backend.has_delta() {
            order_cols.push("_id");
        }

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
                            let val = if float_arr.is_null(i) {
                                f64::NEG_INFINITY
                            } else {
                                float_arr.value(i)
                            };

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
                            let val = if float_arr.is_null(i) {
                                f64::INFINITY
                            } else {
                                float_arr.value(i)
                            };

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

                    top_k.into_iter().skip(offset).map(|(_, idx)| idx).collect()
                } else if let Some(int_arr) = col.as_any().downcast_ref::<Int64Array>() {
                    let descending = clause.descending;
                    let mut top_k: Vec<(i64, usize)> = Vec::with_capacity(k_actual + 1);

                    if descending {
                        for i in 0..num_rows {
                            let val = if int_arr.is_null(i) {
                                i64::MIN
                            } else {
                                int_arr.value(i)
                            };

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
                            let val = if int_arr.is_null(i) {
                                i64::MAX
                            } else {
                                int_arr.value(i)
                            };

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

                    top_k.into_iter().skip(offset).map(|(_, idx)| idx).collect()
                } else {
                    Self::compute_topk_indices_generic(
                        &sort_batch,
                        &stmt.order_by,
                        k_actual,
                        stmt.offset,
                    )
                }
            } else {
                Self::compute_topk_indices_generic(
                    &sort_batch,
                    &stmt.order_by,
                    k_actual,
                    stmt.offset,
                )
            }
        } else {
            Self::compute_topk_indices_generic(&sort_batch, &stmt.order_by, k_actual, stmt.offset)
        };

        if final_indices.is_empty() {
            return backend.read_columns_to_arrow(None, 0, Some(0));
        }

        // Step 3: Read ALL columns but only for top-k rows. Combined base +
        // append-delta batches use logical IDs because their Arrow positions do
        // not map one-to-one to physical base-file indices.
        if backend.has_delta() {
            let id_col = sort_batch
                .column_by_name("_id")
                .ok_or_else(|| err_data("ORDER BY delta batch missing _id"))?;
            let ids = if let Some(array) = id_col.as_any().downcast_ref::<Int64Array>() {
                final_indices
                    .iter()
                    .map(|&index| array.value(index) as u64)
                    .collect::<Vec<_>>()
            } else if let Some(array) = id_col
                .as_any()
                .downcast_ref::<arrow::array::UInt64Array>()
            {
                final_indices
                    .iter()
                    .map(|&index| array.value(index))
                    .collect::<Vec<_>>()
            } else {
                return Err(err_data("ORDER BY delta batch has invalid _id type"));
            };
            backend.read_rows_by_ids_to_arrow(&ids)
        } else {
            backend.read_columns_by_indices_to_arrow(&final_indices, None)
        }
    }


    /// Fast path for combined WHERE filter + GROUP BY on dictionary columns
    /// Does filter and aggregation in a single pass without intermediate materialization
    fn try_fast_filter_groupby(
        backend: &TableStorageBackend,
        stmt: &SelectStatement,
    ) -> io::Result<Option<RecordBatch>> {
        if backend.has_pending_deltas() || backend.is_mmap_only() {
            return Ok(None);
        }

        use crate::query::AggregateFunc;
        use arrow::array::DictionaryArray;
        use arrow::datatypes::UInt32Type;

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
            if let SelectColumn::Aggregate {
                func,
                column,
                alias,
                ..
            } = col
            {
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
        let cols_to_read: Vec<&str> = vec![
            filter_col.as_str(),
            group_col.as_str(),
            agg_col_name.as_str(),
        ];
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

        let filter_dict = match filter_arr
            .as_any()
            .downcast_ref::<DictionaryArray<UInt32Type>>()
        {
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
                let schema = Arc::new(Schema::new(vec![Field::new(
                    &group_col,
                    ArrowDataType::Utf8,
                    false,
                )]));
                return Ok(Some(RecordBatch::new_empty(schema)));
            }
        };

        // Get group column as dictionary
        let group_arr = match batch.column_by_name(&group_col) {
            Some(c) => c,
            None => return Ok(None),
        };

        let group_dict = match group_arr
            .as_any()
            .downcast_ref::<DictionaryArray<UInt32Type>>()
        {
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
            if filter_keys.null_count() == 0
                && group_keys.null_count() == 0
                && float_arr.null_count() == 0
            {
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
                        let gk = if group_keys.is_null(i) {
                            0
                        } else {
                            group_keys.value(i) as usize + 1
                        };
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
                    let gk = if group_keys.is_null(i) {
                        0
                    } else {
                        group_keys.value(i) as usize + 1
                    };
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
        )
        .map_err(|e| err_data(e.to_string()))?;

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
            let col_refs_vec: Option<Vec<&str>> = col_refs
                .as_ref()
                .map(|v| v.iter().map(|s| s.as_str()).collect());
            return backend.read_columns_to_arrow(col_refs_vec.as_deref(), 0, Some(0));
        }

        // Step 2: Apply WHERE filter to get matching row indices
        let where_clause = stmt.where_clause.as_ref().unwrap();
        let mask =
            Self::evaluate_predicate_with_storage(&filter_batch, where_clause, storage_path)?;

        // Collect matching indices
        let indices: Vec<usize> = mask
            .iter()
            .enumerate()
            .filter_map(|(i, v)| if v == Some(true) { Some(i) } else { None })
            .collect();

        if indices.is_empty() {
            let col_refs = Self::get_col_refs(stmt);
            let col_refs_vec: Option<Vec<&str>> = col_refs
                .as_ref()
                .map(|v| v.iter().map(|s| s.as_str()).collect());
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
                indices.iter().map(|&i| i as u64).collect::<Vec<_>>(),
            );
            let columns: Vec<ArrayRef> = filter_batch
                .columns()
                .iter()
                .map(|col| compute::take(col, &indices_array, None))
                .collect::<Result<Vec<_>, _>>()
                .map_err(|e| err_data(e.to_string()))?;
            RecordBatch::try_new(filter_batch.schema(), columns)
                .map_err(|e| err_data(e.to_string()))
        } else {
            // Need to read additional columns for matching rows
            let other_batch = backend.read_columns_by_indices_to_arrow(&indices, None)?;

            // Also filter the WHERE columns batch
            let indices_array = arrow::array::UInt64Array::from(
                indices.iter().map(|&i| i as u64).collect::<Vec<_>>(),
            );
            let where_columns: Vec<ArrayRef> = filter_batch
                .columns()
                .iter()
                .map(|col| compute::take(col, &indices_array, None))
                .collect::<Result<Vec<_>, _>>()
                .map_err(|e| err_data(e.to_string()))?;

            // Merge: use other_batch as base (has _id and other columns)
            // Add WHERE columns that aren't already present
            let mut fields: Vec<Field> = other_batch
                .schema()
                .fields()
                .iter()
                .map(|f| f.as_ref().clone())
                .collect();
            let mut arrays: Vec<ArrayRef> = other_batch.columns().to_vec();

            for (i, field) in filter_batch.schema().fields().iter().enumerate() {
                if other_batch.column_by_name(field.name()).is_none() {
                    fields.push(field.as_ref().clone());
                    arrays.push(where_columns[i].clone());
                }
            }

            let schema = Arc::new(Schema::new(fields));
            RecordBatch::try_new(schema, arrays).map_err(|e| err_data(e.to_string()))
        }
    }
}
