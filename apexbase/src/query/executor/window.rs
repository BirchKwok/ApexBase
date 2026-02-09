// Window functions, UNION execution

impl ApexExecutor {
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
                .map_err(|e| err_data( e.to_string()))?;
            columns.push(concatenated);
        }

        RecordBatch::try_new(left.schema(), columns)
            .map_err(|e| err_data( e.to_string()))
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
                    .map_err(|e| err_data( e.to_string()))?;
                return RecordBatch::try_new(batch.schema(), vec![filtered])
                    .map_err(|e| err_data( e.to_string()));
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
                    .map_err(|e| err_data( e.to_string()))?;
                return RecordBatch::try_new(batch.schema(), vec![filtered])
                    .map_err(|e| err_data( e.to_string()));
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
                    .map_err(|e| err_data( e.to_string()))?;
                return RecordBatch::try_new(batch.schema(), vec![filtered])
                    .map_err(|e| err_data( e.to_string()));
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
                .map_err(|e| err_data( e.to_string()))?;
            result_columns.push(filtered);
        }

        RecordBatch::try_new(batch.schema(), result_columns)
            .map_err(|e| err_data( e.to_string()))
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
                    return Err(err_input( 
                        format!("Unsupported window function: {}", name)));
                }
                let out_name = alias.clone().unwrap_or_else(|| name.to_lowercase());
                window_specs.push((upper, args.clone(), partition_by.clone(), order_by.clone(), out_name));
            }
        }

        if window_specs.is_empty() {
            return Err(err_input( "No window function found"));
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

        // Build window function result arrays (one for int, one for float)
        let mut window_values: Vec<i64> = vec![0; batch.num_rows()];
        let mut window_float_values: Vec<f64> = vec![0.0; batch.num_rows()];
        // Determine if the window function operates on a float column
        let value_funcs = ["SUM", "AVG", "MIN", "MAX", "LAG", "LEAD", "FIRST_VALUE", "LAST_VALUE", "NTH_VALUE", "RUNNING_SUM"];
        let is_value_func = value_funcs.contains(&func_name.as_str());
        let src_col_name = if is_value_func { func_args.get(0).map(|s| s.trim_matches('"').to_string()) } else { None };
        // AVG always returns float regardless of source column type
        let use_float = if func_name == "AVG" {
            true
        } else if let Some(ref cn) = src_col_name {
            batch.column_by_name(cn).map(|c| c.as_any().downcast_ref::<Float64Array>().is_some()).unwrap_or(false)
        } else {
            false
        };
        
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
                        if let Some(float_arr) = src_col.as_any().downcast_ref::<Float64Array>() {
                            for (pos, &row_idx) in indices.iter().enumerate() {
                                if pos >= offset {
                                    let prev_row = indices[pos - offset];
                                    window_float_values[row_idx] = if float_arr.is_null(prev_row) { 0.0 } else { float_arr.value(prev_row) };
                                } else {
                                    window_float_values[row_idx] = 0.0;
                                }
                            }
                        } else if let Some(int_arr) = src_col.as_any().downcast_ref::<Int64Array>() {
                            for (pos, &row_idx) in indices.iter().enumerate() {
                                if pos >= offset {
                                    let prev_row = indices[pos - offset];
                                    window_values[row_idx] = if int_arr.is_null(prev_row) { 0 } else { int_arr.value(prev_row) };
                                } else {
                                    window_values[row_idx] = 0;
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
                        if let Some(float_arr) = src_col.as_any().downcast_ref::<Float64Array>() {
                            let len = indices.len();
                            for (pos, &row_idx) in indices.iter().enumerate() {
                                if pos + offset < len {
                                    let next_row = indices[pos + offset];
                                    window_float_values[row_idx] = if float_arr.is_null(next_row) { 0.0 } else { float_arr.value(next_row) };
                                } else {
                                    window_float_values[row_idx] = 0.0;
                                }
                            }
                        } else if let Some(int_arr) = src_col.as_any().downcast_ref::<Int64Array>() {
                            let len = indices.len();
                            for (pos, &row_idx) in indices.iter().enumerate() {
                                if pos + offset < len {
                                    let next_row = indices[pos + offset];
                                    window_values[row_idx] = if int_arr.is_null(next_row) { 0 } else { int_arr.value(next_row) };
                                } else {
                                    window_values[row_idx] = 0;
                                }
                            }
                        }
                    }
                }
                "FIRST_VALUE" => {
                    // FIRST_VALUE(column) - get first value in partition
                    let col_name = func_args.get(0).map(|s| s.trim_matches('"')).unwrap_or("");
                    if let Some(src_col) = batch.column_by_name(col_name) {
                        if let Some(float_arr) = src_col.as_any().downcast_ref::<Float64Array>() {
                            let first_row = indices[0];
                            let first_val = if float_arr.is_null(first_row) { 0.0 } else { float_arr.value(first_row) };
                            for &row_idx in &indices {
                                window_float_values[row_idx] = first_val;
                            }
                        } else if let Some(int_arr) = src_col.as_any().downcast_ref::<Int64Array>() {
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
                        if let Some(float_arr) = src_col.as_any().downcast_ref::<Float64Array>() {
                            let last_row = indices[indices.len() - 1];
                            let last_val = if float_arr.is_null(last_row) { 0.0 } else { float_arr.value(last_row) };
                            for &row_idx in &indices {
                                window_float_values[row_idx] = last_val;
                            }
                        } else if let Some(int_arr) = src_col.as_any().downcast_ref::<Int64Array>() {
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
                        if let Some(float_arr) = src_col.as_any().downcast_ref::<Float64Array>() {
                            let total: f64 = indices.iter()
                                .filter_map(|&i| if float_arr.is_null(i) { None } else { Some(float_arr.value(i)) })
                                .sum();
                            for &row_idx in &indices {
                                window_float_values[row_idx] = total;
                            }
                        } else if let Some(int_arr) = src_col.as_any().downcast_ref::<Int64Array>() {
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
                    // AVG(column) OVER - average in partition (always returns float)
                    let col_name = func_args.get(0).map(|s| s.trim_matches('"')).unwrap_or("");
                    if let Some(src_col) = batch.column_by_name(col_name) {
                        if let Some(float_arr) = src_col.as_any().downcast_ref::<Float64Array>() {
                            let vals: Vec<f64> = indices.iter()
                                .filter_map(|&i| if float_arr.is_null(i) { None } else { Some(float_arr.value(i)) })
                                .collect();
                            let avg = if vals.is_empty() { 0.0 } else { vals.iter().sum::<f64>() / vals.len() as f64 };
                            for &row_idx in &indices {
                                window_float_values[row_idx] = avg;
                            }
                        } else if let Some(int_arr) = src_col.as_any().downcast_ref::<Int64Array>() {
                            let vals: Vec<i64> = indices.iter()
                                .filter_map(|&i| if int_arr.is_null(i) { None } else { Some(int_arr.value(i)) })
                                .collect();
                            let avg = if vals.is_empty() { 0.0 } else { vals.iter().sum::<i64>() as f64 / vals.len() as f64 };
                            for &row_idx in &indices {
                                window_float_values[row_idx] = avg;
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
                        if let Some(float_arr) = src_col.as_any().downcast_ref::<Float64Array>() {
                            let min_val = indices.iter()
                                .filter_map(|&i| if float_arr.is_null(i) { None } else { Some(float_arr.value(i)) })
                                .fold(f64::INFINITY, f64::min);
                            let min_val = if min_val == f64::INFINITY { 0.0 } else { min_val };
                            for &row_idx in &indices {
                                window_float_values[row_idx] = min_val;
                            }
                        } else if let Some(int_arr) = src_col.as_any().downcast_ref::<Int64Array>() {
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
                        if let Some(float_arr) = src_col.as_any().downcast_ref::<Float64Array>() {
                            let max_val = indices.iter()
                                .filter_map(|&i| if float_arr.is_null(i) { None } else { Some(float_arr.value(i)) })
                                .fold(f64::NEG_INFINITY, f64::max);
                            let max_val = if max_val == f64::NEG_INFINITY { 0.0 } else { max_val };
                            for &row_idx in &indices {
                                window_float_values[row_idx] = max_val;
                            }
                        } else if let Some(int_arr) = src_col.as_any().downcast_ref::<Int64Array>() {
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
                        if let Some(float_arr) = src_col.as_any().downcast_ref::<Float64Array>() {
                            let mut running = 0.0f64;
                            for &row_idx in &indices {
                                if !float_arr.is_null(row_idx) {
                                    running += float_arr.value(row_idx);
                                }
                                window_float_values[row_idx] = running;
                            }
                        } else if let Some(int_arr) = src_col.as_any().downcast_ref::<Int64Array>() {
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
                        if let Some(float_arr) = src_col.as_any().downcast_ref::<Float64Array>() {
                            let nth_val = if n > 0 && n <= indices.len() {
                                let nth_row = indices[n - 1];
                                if float_arr.is_null(nth_row) { 0.0 } else { float_arr.value(nth_row) }
                            } else { 0.0 };
                            for &row_idx in &indices {
                                window_float_values[row_idx] = nth_val;
                            }
                        } else if let Some(int_arr) = src_col.as_any().downcast_ref::<Int64Array>() {
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
                    if use_float {
                        result_fields.push(Field::new(&out_name, ArrowDataType::Float64, false));
                        result_arrays.push(Arc::new(Float64Array::from(window_float_values.clone())));
                    } else {
                        result_fields.push(Field::new(&out_name, ArrowDataType::Int64, false));
                        result_arrays.push(Arc::new(Int64Array::from(window_values.clone())));
                    }
                }
                _ => {}
            }
        }

        let schema = Arc::new(Schema::new(result_fields));
        let result = RecordBatch::try_new(schema, result_arrays)
            .map_err(|e| err_data( e.to_string()))?;

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

}
