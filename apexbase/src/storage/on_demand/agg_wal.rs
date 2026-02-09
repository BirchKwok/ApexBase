// Aggregation fast paths, WAL/durability, accessors, insert_rows, flush, close, Drop

impl OnDemandStorage {
    /// Execute simple aggregation (no GROUP BY, no WHERE) directly on V4 columns.
    /// Supports both in-memory and mmap-only paths.
    /// Returns (count, sum, min, max, is_int) for each requested column
    pub fn execute_simple_agg(
        &self,
        agg_cols: &[&str],
    ) -> io::Result<Option<Vec<(i64, f64, f64, f64, bool)>>> {
        use arrow::array::PrimitiveArray;
        use arrow::buffer::{Buffer, ScalarBuffer};
        use arrow::datatypes::{Int64Type, Float64Type};
        use std::sync::Arc;
        
        // Check if in-memory data is available for fast path
        let columns = self.columns.read();
        let has_in_memory = !columns.is_empty() && columns.iter().any(|c| c.len() > 0);
        
        if !has_in_memory {
            drop(columns);
            // MMAP PATH: scan columns from disk without loading into memory
            return self.execute_simple_agg_mmap(agg_cols);
        }
        
        let schema = self.schema.read();
        let deleted = self.deleted.read();
        let total_rows = columns.first().map(|c| c.len()).unwrap_or(0);
        
        let has_deleted = deleted.iter().any(|&b| b != 0);
        // Bail to Arrow path if there are deleted rows (need filtered arrays)
        if has_deleted { return Ok(None); }
        
        let active_count = total_rows as i64;
        let mut results: Vec<(i64, f64, f64, f64, bool)> = Vec::with_capacity(agg_cols.len());
        
        for &col_name in agg_cols {
            if col_name == "*" || col_name == "1" {
                results.push((active_count, 0.0, 0.0, 0.0, false));
                continue;
            }
            
            let col_idx = match schema.get_index(col_name) {
                Some(idx) => idx,
                None => { results.push((0, 0.0, 0.0, 0.0, false)); continue; }
            };
            if col_idx >= columns.len() { results.push((0, 0.0, 0.0, 0.0, false)); continue; }
            
            match &columns[col_idx] {
                ColumnData::Int64(vals) => {
                    let byte_len = vals.len() * std::mem::size_of::<i64>();
                    let buffer = unsafe {
                        Buffer::from_custom_allocation(
                            std::ptr::NonNull::new_unchecked(vals.as_ptr() as *mut u8),
                            byte_len,
                            Arc::new(()),
                        )
                    };
                    let arr = PrimitiveArray::<Int64Type>::new(
                        ScalarBuffer::new(buffer, 0, vals.len()), None,
                    );
                    let sum = arrow::compute::sum(&arr).unwrap_or(0);
                    let min_v = arrow::compute::min(&arr).unwrap_or(i64::MAX);
                    let max_v = arrow::compute::max(&arr).unwrap_or(i64::MIN);
                    drop(arr);
                    results.push((vals.len() as i64, sum as f64, min_v as f64, max_v as f64, true));
                }
                ColumnData::Float64(vals) => {
                    let byte_len = vals.len() * std::mem::size_of::<f64>();
                    let buffer = unsafe {
                        Buffer::from_custom_allocation(
                            std::ptr::NonNull::new_unchecked(vals.as_ptr() as *mut u8),
                            byte_len,
                            Arc::new(()),
                        )
                    };
                    let arr = PrimitiveArray::<Float64Type>::new(
                        ScalarBuffer::new(buffer, 0, vals.len()), None,
                    );
                    let sum = arrow::compute::sum(&arr).unwrap_or(0.0);
                    let min_v = arrow::compute::min(&arr).unwrap_or(f64::INFINITY);
                    let max_v = arrow::compute::max(&arr).unwrap_or(f64::NEG_INFINITY);
                    drop(arr);
                    results.push((vals.len() as i64, sum, min_v, max_v, false));
                }
                _ => { results.push((active_count, 0.0, 0.0, 0.0, false)); }
            }
        }
        
        Ok(Some(results))
    }

    /// MMAP PATH: Execute simple aggregation by scanning V4 RGs via mmap
    fn execute_simple_agg_mmap(
        &self,
        agg_cols: &[&str],
    ) -> io::Result<Option<Vec<(i64, f64, f64, f64, bool)>>> {
        let footer = match self.get_or_load_footer()? {
            Some(f) => f,
            None => return Ok(None),
        };

        let schema = &footer.schema;
        // Resolve which columns need scanning (skip * and 1)
        let col_indices: Vec<usize> = agg_cols.iter()
            .filter(|&&n| n != "*" && n != "1")
            .filter_map(|&n| schema.get_index(n))
            .collect();

        // Count active rows from footer metadata (O(1), no scan needed)
        let total_active: i64 = footer.row_groups.iter()
            .map(|rg| rg.active_rows() as i64)
            .sum();

        if col_indices.is_empty() {
            // COUNT(*) only — no column scan needed
            return Ok(Some(agg_cols.iter().map(|_| (total_active, 0.0, 0.0, 0.0, false)).collect()));
        }

        let (scanned_cols, del_bytes) = self.scan_columns_mmap(&col_indices, &footer)?;
        let has_deleted = del_bytes.iter().any(|&b| b != 0);

        // Build results
        let mut results: Vec<(i64, f64, f64, f64, bool)> = Vec::with_capacity(agg_cols.len());
        let mut scan_idx = 0usize;
        for &col_name in agg_cols {
            if col_name == "*" || col_name == "1" {
                results.push((total_active, 0.0, 0.0, 0.0, false));
                continue;
            }
            if scan_idx >= scanned_cols.len() {
                results.push((0, 0.0, 0.0, 0.0, false));
                continue;
            }
            let col = &scanned_cols[scan_idx];
            scan_idx += 1;
            match col {
                ColumnData::Int64(vals) => {
                    if has_deleted {
                        let mut count = 0i64; let mut sum = 0i64;
                        let mut min_v = i64::MAX; let mut max_v = i64::MIN;
                        for (i, &v) in vals.iter().enumerate() {
                            let b = i / 8; let bit = i % 8;
                            if b < del_bytes.len() && (del_bytes[b] >> bit) & 1 != 0 { continue; }
                            count += 1; sum += v;
                            if v < min_v { min_v = v; }
                            if v > max_v { max_v = v; }
                        }
                        results.push((count, sum as f64, min_v as f64, max_v as f64, true));
                    } else {
                        let n = vals.len() as i64;
                        let sum: i64 = vals.iter().sum();
                        let min_v = vals.iter().copied().min().unwrap_or(i64::MAX);
                        let max_v = vals.iter().copied().max().unwrap_or(i64::MIN);
                        results.push((n, sum as f64, min_v as f64, max_v as f64, true));
                    }
                }
                ColumnData::Float64(vals) => {
                    if has_deleted {
                        let mut count = 0i64; let mut sum = 0.0f64;
                        let mut min_v = f64::INFINITY; let mut max_v = f64::NEG_INFINITY;
                        for (i, &v) in vals.iter().enumerate() {
                            let b = i / 8; let bit = i % 8;
                            if b < del_bytes.len() && (del_bytes[b] >> bit) & 1 != 0 { continue; }
                            count += 1; sum += v;
                            if v < min_v { min_v = v; }
                            if v > max_v { max_v = v; }
                        }
                        results.push((count, sum, min_v, max_v, false));
                    } else {
                        let n = vals.len() as i64;
                        let sum: f64 = vals.iter().sum();
                        let min_v = vals.iter().copied().fold(f64::INFINITY, f64::min);
                        let max_v = vals.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                        results.push((n, sum, min_v, max_v, false));
                    }
                }
                _ => { results.push((total_active, 0.0, 0.0, 0.0, false)); }
            }
        }
        Ok(Some(results))
    }

    /// Compute numeric column aggregates from in-memory V4 columns.
    /// Returns (count, sum, min, max) for the specified column.
    pub fn compute_column_stats_inmemory(&self, col_name: &str) -> io::Result<Option<(u64, f64, f64, f64)>> {
        if !self.has_v4_in_memory_data() { return Ok(None); }
        
        let schema = self.schema.read();
        let columns = self.columns.read();
        let deleted = self.deleted.read();
        let total_rows = self.ids.read().len();
        
        let col_idx = match schema.get_index(col_name) {
            Some(idx) => idx,
            None => return Ok(None),
        };
        if col_idx >= columns.len() { return Ok(None); }
        
        let has_deleted = deleted.iter().any(|&b| b != 0);
        
        match &columns[col_idx] {
            ColumnData::Int64(vals) => {
                let count = vals.len().min(total_rows);
                if !has_deleted {
                    let mut sum = 0i64;
                    let mut min_v = i64::MAX;
                    let mut max_v = i64::MIN;
                    for i in 0..count {
                        let v = unsafe { *vals.get_unchecked(i) };
                        sum += v;
                        if v < min_v { min_v = v; }
                        if v > max_v { max_v = v; }
                    }
                    Ok(Some((count as u64, sum as f64, min_v as f64, max_v as f64)))
                } else {
                    let mut c = 0u64;
                    let mut sum = 0i64;
                    let mut min_v = i64::MAX;
                    let mut max_v = i64::MIN;
                    for i in 0..count {
                        let b = i / 8; let bit = i % 8;
                        if b < deleted.len() && (deleted[b] >> bit) & 1 != 0 { continue; }
                        let v = vals[i];
                        c += 1; sum += v;
                        if v < min_v { min_v = v; }
                        if v > max_v { max_v = v; }
                    }
                    Ok(Some((c, sum as f64, min_v as f64, max_v as f64)))
                }
            }
            ColumnData::Float64(vals) => {
                let count = vals.len().min(total_rows);
                if !has_deleted {
                    let mut sum = 0.0f64;
                    let mut min_v = f64::INFINITY;
                    let mut max_v = f64::NEG_INFINITY;
                    for i in 0..count {
                        let v = unsafe { *vals.get_unchecked(i) };
                        sum += v;
                        if v < min_v { min_v = v; }
                        if v > max_v { max_v = v; }
                    }
                    Ok(Some((count as u64, sum, min_v, max_v)))
                } else {
                    let mut c = 0u64;
                    let mut sum = 0.0f64;
                    let mut min_v = f64::INFINITY;
                    let mut max_v = f64::NEG_INFINITY;
                    for i in 0..count {
                        let b = i / 8; let bit = i % 8;
                        if b < deleted.len() && (deleted[b] >> bit) & 1 != 0 { continue; }
                        let v = vals[i];
                        c += 1; sum += v;
                        if v < min_v { min_v = v; }
                        if v > max_v { max_v = v; }
                    }
                    Ok(Some((c, sum, min_v, max_v)))
                }
            }
            _ => Ok(None),
        }
    }

    /// Build cached string dictionary indices for a column (row→group_id mapping)
    /// Returns (dict_strings, group_ids) where group_ids[row] = index into dict_strings.
    /// Supports both in-memory and mmap-only paths.
    pub fn build_string_dict_cache(
        &self,
        col_name: &str,
    ) -> io::Result<Option<(Vec<String>, Vec<u16>)>> {
        if self.has_v4_in_memory_data() {
            // In-memory fast path
            let schema = self.schema.read();
            let columns = self.columns.read();
            let total_rows = self.ids.read().len();
            
            let col_idx = match schema.get_index(col_name) {
                Some(idx) => idx,
                None => return Ok(None),
            };
            if col_idx >= columns.len() { return Ok(None); }
            
            let (offsets, data) = match &columns[col_idx] {
                ColumnData::String { offsets, data } => (offsets, data),
                _ => return Ok(None),
            };
            let count = offsets.len().saturating_sub(1).min(total_rows);
            
            let mut dict_map: ahash::AHashMap<&[u8], u16> = ahash::AHashMap::with_capacity(64);
            let mut dict_strings: Vec<String> = Vec::with_capacity(64);
            let mut group_ids: Vec<u16> = Vec::with_capacity(count);
            
            for i in 0..count {
                let s = offsets[i] as usize;
                let e = offsets[i + 1] as usize;
                let key = &data[s..e];
                let gid = match dict_map.get(key) {
                    Some(&id) => id,
                    None => {
                        let id = dict_strings.len() as u16;
                        dict_map.insert(key, id);
                        dict_strings.push(std::str::from_utf8(key).unwrap_or("").to_string());
                        id
                    }
                };
                group_ids.push(gid);
            }
            
            return Ok(Some((dict_strings, group_ids)));
        }
        
        // MMAP PATH: scan string column from V4 RGs
        self.build_string_dict_cache_mmap(col_name)
    }
    
    /// MMAP PATH: build string dict cache by scanning V4 RGs
    fn build_string_dict_cache_mmap(
        &self,
        col_name: &str,
    ) -> io::Result<Option<(Vec<String>, Vec<u16>)>> {
        let footer = match self.get_or_load_footer()? {
            Some(f) => f,
            None => return Ok(None),
        };
        let col_idx = match footer.schema.get_index(col_name) {
            Some(idx) => idx,
            None => return Ok(None),
        };
        
        let (scanned_cols, _del_bytes) = self.scan_columns_mmap(&[col_idx], &footer)?;
        if scanned_cols.is_empty() { return Ok(None); }
        
        let (offsets, data) = match &scanned_cols[0] {
            ColumnData::String { offsets, data } => (offsets, data),
            _ => return Ok(None),
        };
        let count = offsets.len().saturating_sub(1);
        
        let mut dict_map: ahash::AHashMap<&[u8], u16> = ahash::AHashMap::with_capacity(64);
        let mut dict_strings: Vec<String> = Vec::with_capacity(64);
        let mut group_ids: Vec<u16> = Vec::with_capacity(count);
        
        for i in 0..count {
            let s = offsets[i] as usize;
            let e = offsets[i + 1] as usize;
            let key = &data[s..e];
            let gid = match dict_map.get(key) {
                Some(&id) => id,
                None => {
                    let id = dict_strings.len() as u16;
                    dict_map.insert(key, id);
                    dict_strings.push(std::str::from_utf8(key).unwrap_or("").to_string());
                    id
                }
            };
            group_ids.push(gid);
        }
        
        Ok(Some((dict_strings, group_ids)))
    }

    /// Execute GROUP BY + aggregate using pre-built dict cache.
    /// Supports both in-memory and mmap-only paths.
    pub fn execute_group_agg_cached(
        &self,
        dict_strings: &[String],
        group_ids: &[u16],
        agg_cols: &[(&str, bool)], // (col_name, is_count_star)
    ) -> io::Result<Option<Vec<(String, Vec<(f64, i64)>)>>> {
        if !self.has_v4_in_memory_data() {
            // MMAP PATH: scan agg columns from disk, then aggregate with pre-built dict
            return self.execute_group_agg_cached_mmap(dict_strings, group_ids, agg_cols);
        }
        
        let schema = self.schema.read();
        let columns = self.columns.read();
        let deleted = self.deleted.read();
        let total_rows = self.ids.read().len();
        
        let has_deleted = deleted.iter().any(|&b| b != 0);
        let scan_rows = total_rows.min(group_ids.len());
        let num_groups = dict_strings.len();
        let num_aggs = agg_cols.len();
        
        struct AggSlice<'a> { i64_vals: Option<&'a [i64]>, f64_vals: Option<&'a [f64]>, is_count: bool }
        let agg_slices: Vec<AggSlice> = agg_cols.iter().map(|(name, is_count)| {
            if *is_count {
                AggSlice { i64_vals: None, f64_vals: None, is_count: true }
            } else if let Some(idx) = schema.get_index(name) {
                if idx < columns.len() {
                    match &columns[idx] {
                        ColumnData::Int64(v) => AggSlice { i64_vals: Some(v.as_slice()), f64_vals: None, is_count: false },
                        ColumnData::Float64(v) => AggSlice { i64_vals: None, f64_vals: Some(v.as_slice()), is_count: false },
                        _ => AggSlice { i64_vals: None, f64_vals: None, is_count: true },
                    }
                } else { AggSlice { i64_vals: None, f64_vals: None, is_count: true } }
            } else { AggSlice { i64_vals: None, f64_vals: None, is_count: true } }
        }).collect();
        
        let flat_len = num_groups * num_aggs;
        let mut flat_sums = vec![0.0f64; flat_len];
        let mut flat_counts = vec![0i64; flat_len];
        
        // Single-pass aggregation with O(1) group lookup via cached group_ids
        if has_deleted {
            for i in 0..scan_rows {
                let b = i / 8; let bit = i % 8;
                if b < deleted.len() && (deleted[b] >> bit) & 1 != 0 { continue; }
                let base = group_ids[i] as usize * num_aggs;
                for (ai, agg) in agg_slices.iter().enumerate() {
                    flat_counts[base + ai] += 1;
                    if !agg.is_count {
                        if let Some(vals) = agg.f64_vals { if i < vals.len() { flat_sums[base + ai] += vals[i]; } }
                        else if let Some(vals) = agg.i64_vals { if i < vals.len() { flat_sums[base + ai] += vals[i] as f64; } }
                    }
                }
            }
        } else {
            for i in 0..scan_rows {
                let base = group_ids[i] as usize * num_aggs;
                for (ai, agg) in agg_slices.iter().enumerate() {
                    flat_counts[base + ai] += 1;
                    if !agg.is_count {
                        if let Some(vals) = agg.f64_vals { if i < vals.len() { unsafe { *flat_sums.get_unchecked_mut(base + ai) += *vals.get_unchecked(i); } } }
                        else if let Some(vals) = agg.i64_vals { if i < vals.len() { unsafe { *flat_sums.get_unchecked_mut(base + ai) += *vals.get_unchecked(i) as f64; } } }
                    }
                }
            }
        }
        
        let results: Vec<(String, Vec<(f64, i64)>)> = (0..num_groups)
            .filter(|&gid| flat_counts[gid * num_aggs] > 0)
            .map(|gid| {
                let aggs: Vec<(f64, i64)> = (0..num_aggs)
                    .map(|ai| (flat_sums[gid * num_aggs + ai], flat_counts[gid * num_aggs + ai]))
                    .collect();
                (dict_strings[gid].clone(), aggs)
            })
            .collect();
        
        Ok(Some(results))
    }

    /// MMAP PATH: execute GROUP BY + aggregate using pre-built dict cache, scanning agg columns from disk.
    fn execute_group_agg_cached_mmap(
        &self,
        dict_strings: &[String],
        group_ids: &[u16],
        agg_cols: &[(&str, bool)],
    ) -> io::Result<Option<Vec<(String, Vec<(f64, i64)>)>>> {
        let footer = match self.get_or_load_footer()? {
            Some(f) => f,
            None => return Ok(None),
        };
        let schema = &footer.schema;
        
        // Resolve agg column indices (skip count-star which doesn't need a column)
        let agg_col_indices: Vec<Option<usize>> = agg_cols.iter()
            .map(|(name, is_count)| if *is_count { None } else { schema.get_index(name) })
            .collect();
        let needed: Vec<usize> = agg_col_indices.iter().filter_map(|&x| x).collect();
        
        let (scanned, del_bytes) = if needed.is_empty() {
            (Vec::new(), Vec::new())
        } else {
            self.scan_columns_mmap(&needed, &footer)?
        };
        let has_deleted = del_bytes.iter().any(|&b| b != 0);
        
        // Map needed index → scanned position
        let needed_to_pos: HashMap<usize, usize> = needed.iter().enumerate()
            .map(|(pos, &idx)| (idx, pos))
            .collect();
        
        let num_groups = dict_strings.len();
        let num_aggs = agg_cols.len();
        let scan_rows = group_ids.len();
        let flat_len = num_groups * num_aggs;
        let mut flat_sums = vec![0.0f64; flat_len];
        let mut flat_counts = vec![0i64; flat_len];
        
        for i in 0..scan_rows {
            if has_deleted {
                let b = i / 8; let bit = i % 8;
                if b < del_bytes.len() && (del_bytes[b] >> bit) & 1 != 0 { continue; }
            }
            let base = group_ids[i] as usize * num_aggs;
            if base + num_aggs > flat_len { continue; }
            for (ai, opt_idx) in agg_col_indices.iter().enumerate() {
                flat_counts[base + ai] += 1;
                if let Some(&col_idx) = opt_idx.as_ref() {
                    if let Some(&pos) = needed_to_pos.get(&col_idx) {
                        if pos < scanned.len() {
                            match &scanned[pos] {
                                ColumnData::Int64(v) => { if i < v.len() { flat_sums[base + ai] += v[i] as f64; } }
                                ColumnData::Float64(v) => { if i < v.len() { flat_sums[base + ai] += v[i]; } }
                                _ => {}
                            }
                        }
                    }
                }
            }
        }
        
        let results: Vec<(String, Vec<(f64, i64)>)> = (0..num_groups)
            .filter(|&gid| flat_counts[gid * num_aggs] > 0)
            .map(|gid| {
                let aggs: Vec<(f64, i64)> = (0..num_aggs)
                    .map(|ai| (flat_sums[gid * num_aggs + ai], flat_counts[gid * num_aggs + ai]))
                    .collect();
                (dict_strings[gid].clone(), aggs)
            })
            .collect();
        
        Ok(Some(results))
    }

    /// Execute BETWEEN + GROUP BY using pre-built dict cache.
    /// Supports both in-memory and mmap-only paths.
    pub fn execute_between_group_agg_cached(
        &self,
        filter_col: &str,
        lo: f64,
        hi: f64,
        dict_strings: &[String],
        group_ids: &[u16],
        agg_col: Option<&str>,
    ) -> io::Result<Option<Vec<(String, f64, i64)>>> {
        if !self.has_v4_in_memory_data() {
            // MMAP PATH: scan filter+agg columns from disk
            return self.execute_between_group_agg_cached_mmap(filter_col, lo, hi, dict_strings, group_ids, agg_col);
        }
        
        let schema = self.schema.read();
        let columns = self.columns.read();
        let deleted = self.deleted.read();
        let total_rows = self.ids.read().len();
        
        let filter_idx = match schema.get_index(filter_col) {
            Some(idx) => idx,
            None => return Ok(None),
        };
        if filter_idx >= columns.len() { return Ok(None); }
        
        let has_deleted = deleted.iter().any(|&b| b != 0);
        let lo_i64 = lo as i64;
        let hi_i64 = hi as i64;
        let scan_rows = total_rows.min(group_ids.len());
        let num_groups = dict_strings.len();
        
        let mut group_sums = vec![0.0f64; num_groups];
        let mut group_counts = vec![0i64; num_groups];
        
        let agg_idx = agg_col.and_then(|ac| schema.get_index(ac));
        let agg_f64 = agg_idx.and_then(|idx| {
            if idx < columns.len() { match &columns[idx] { ColumnData::Float64(v) => Some(v.as_slice()), _ => None } } else { None }
        });
        let agg_i64 = agg_idx.and_then(|idx| {
            if idx < columns.len() { match &columns[idx] { ColumnData::Int64(v) => Some(v.as_slice()), _ => None } } else { None }
        });
        
        // Branchless accumulation: eliminates branch misprediction at ~50% BETWEEN hit rate
        // mask = (in_range) as i64 → 0 or 1, multiply with agg value so non-matching adds 0
        macro_rules! between_agg_branchy {
            ($filter_vals:expr, $lo_cmp:expr, $hi_cmp:expr, $limit:expr) => {{
                for i in 0..$limit {
                    let b = i / 8; let bit = i % 8;
                    if b < deleted.len() && (deleted[b] >> bit) & 1 != 0 { continue; }
                    let fv = unsafe { *$filter_vals.get_unchecked(i) };
                    if fv >= $lo_cmp && fv <= $hi_cmp {
                        let gid = unsafe { *group_ids.get_unchecked(i) } as usize;
                        unsafe { *group_counts.get_unchecked_mut(gid) += 1; }
                        if let Some(av) = agg_f64 { unsafe { *group_sums.get_unchecked_mut(gid) += *av.get_unchecked(i); } }
                        else if let Some(av) = agg_i64 { unsafe { *group_sums.get_unchecked_mut(gid) += *av.get_unchecked(i) as f64; } }
                    }
                }
            }};
        }
        
        if let Some(filter_vals) = match &columns[filter_idx] { ColumnData::Int64(v) => Some(v.as_slice()), _ => None } {
            let limit = scan_rows.min(filter_vals.len()).min(group_ids.len());
            let limit = limit.min(agg_f64.map_or(usize::MAX, |a| a.len())).min(agg_i64.map_or(usize::MAX, |a| a.len()));
            if has_deleted {
                between_agg_branchy!(filter_vals, lo_i64, hi_i64, limit);
            } else if let Some(av) = agg_f64 {
                // HOT PATH: branchless i64 filter + f64 agg (no deleted)
                for i in 0..limit {
                    let fv = unsafe { *filter_vals.get_unchecked(i) };
                    let mask = (fv >= lo_i64 && fv <= hi_i64) as i64;
                    let gid = unsafe { *group_ids.get_unchecked(i) } as usize;
                    unsafe {
                        *group_counts.get_unchecked_mut(gid) += mask;
                        *group_sums.get_unchecked_mut(gid) += mask as f64 * *av.get_unchecked(i);
                    }
                }
            } else if let Some(av) = agg_i64 {
                for i in 0..limit {
                    let fv = unsafe { *filter_vals.get_unchecked(i) };
                    let mask = (fv >= lo_i64 && fv <= hi_i64) as i64;
                    let gid = unsafe { *group_ids.get_unchecked(i) } as usize;
                    unsafe {
                        *group_counts.get_unchecked_mut(gid) += mask;
                        *group_sums.get_unchecked_mut(gid) += mask as f64 * (*av.get_unchecked(i) as f64);
                    }
                }
            } else {
                // COUNT-only: branchless
                for i in 0..limit {
                    let fv = unsafe { *filter_vals.get_unchecked(i) };
                    let mask = (fv >= lo_i64 && fv <= hi_i64) as i64;
                    let gid = unsafe { *group_ids.get_unchecked(i) } as usize;
                    unsafe { *group_counts.get_unchecked_mut(gid) += mask; }
                }
            }
        } else if let Some(filter_vals) = match &columns[filter_idx] { ColumnData::Float64(v) => Some(v.as_slice()), _ => None } {
            let limit = scan_rows.min(filter_vals.len()).min(group_ids.len());
            let limit = limit.min(agg_f64.map_or(usize::MAX, |a| a.len())).min(agg_i64.map_or(usize::MAX, |a| a.len()));
            if has_deleted {
                between_agg_branchy!(filter_vals, lo, hi, limit);
            } else if let Some(av) = agg_f64 {
                for i in 0..limit {
                    let fv = unsafe { *filter_vals.get_unchecked(i) };
                    let mask = (fv >= lo && fv <= hi) as i64;
                    let gid = unsafe { *group_ids.get_unchecked(i) } as usize;
                    unsafe {
                        *group_counts.get_unchecked_mut(gid) += mask;
                        *group_sums.get_unchecked_mut(gid) += mask as f64 * *av.get_unchecked(i);
                    }
                }
            } else if let Some(av) = agg_i64 {
                for i in 0..limit {
                    let fv = unsafe { *filter_vals.get_unchecked(i) };
                    let mask = (fv >= lo && fv <= hi) as i64;
                    let gid = unsafe { *group_ids.get_unchecked(i) } as usize;
                    unsafe {
                        *group_counts.get_unchecked_mut(gid) += mask;
                        *group_sums.get_unchecked_mut(gid) += mask as f64 * (*av.get_unchecked(i) as f64);
                    }
                }
            } else {
                for i in 0..limit {
                    let fv = unsafe { *filter_vals.get_unchecked(i) };
                    let mask = (fv >= lo && fv <= hi) as i64;
                    let gid = unsafe { *group_ids.get_unchecked(i) } as usize;
                    unsafe { *group_counts.get_unchecked_mut(gid) += mask; }
                }
            }
        }
        
        let results: Vec<(String, f64, i64)> = (0..num_groups)
            .filter(|&gid| group_counts[gid] > 0)
            .map(|gid| (dict_strings[gid].clone(), group_sums[gid], group_counts[gid]))
            .collect();
        
        Ok(Some(results))
    }

    /// MMAP PATH: execute BETWEEN + GROUP BY using pre-built dict cache, scanning filter+agg from disk.
    fn execute_between_group_agg_cached_mmap(
        &self,
        filter_col: &str,
        lo: f64,
        hi: f64,
        dict_strings: &[String],
        group_ids: &[u16],
        agg_col: Option<&str>,
    ) -> io::Result<Option<Vec<(String, f64, i64)>>> {
        let footer = match self.get_or_load_footer()? {
            Some(f) => f,
            None => return Ok(None),
        };
        let schema = &footer.schema;
        
        let filter_idx = match schema.get_index(filter_col) {
            Some(idx) => idx,
            None => return Ok(None),
        };
        let agg_idx = agg_col.and_then(|ac| schema.get_index(ac));
        
        let mut needed: Vec<usize> = vec![filter_idx];
        if let Some(ai) = agg_idx {
            if ai != filter_idx { needed.push(ai); }
        }
        
        let (scanned, del_bytes) = self.scan_columns_mmap(&needed, &footer)?;
        let has_deleted = del_bytes.iter().any(|&b| b != 0);
        
        let filter_col_data = &scanned[0];
        let agg_col_data = agg_idx.map(|ai| {
            if ai == filter_idx { &scanned[0] } else { &scanned[1] }
        });
        
        let num_groups = dict_strings.len();
        let scan_rows = group_ids.len();
        let mut group_sums = vec![0.0f64; num_groups];
        let mut group_counts = vec![0i64; num_groups];
        let lo_i64 = lo as i64;
        let hi_i64 = hi as i64;
        
        macro_rules! between_mmap {
            ($fvals:expr, $lo_c:expr, $hi_c:expr) => {
                let limit = scan_rows.min($fvals.len()).min(group_ids.len());
                for i in 0..limit {
                    if has_deleted {
                        let b = i / 8; let bit = i % 8;
                        if b < del_bytes.len() && (del_bytes[b] >> bit) & 1 != 0 { continue; }
                    }
                    let fv = $fvals[i];
                    if fv >= $lo_c && fv <= $hi_c {
                        let gid = group_ids[i] as usize;
                        if gid < num_groups {
                            group_counts[gid] += 1;
                            if let Some(acd) = agg_col_data {
                                match acd {
                                    ColumnData::Int64(v) => { if i < v.len() { group_sums[gid] += v[i] as f64; } }
                                    ColumnData::Float64(v) => { if i < v.len() { group_sums[gid] += v[i]; } }
                                    _ => {}
                                }
                            }
                        }
                    }
                }
            };
        }
        
        match filter_col_data {
            ColumnData::Int64(vals) => { between_mmap!(vals, lo_i64, hi_i64); }
            ColumnData::Float64(vals) => { between_mmap!(vals, lo, hi); }
            _ => return Ok(None),
        }
        
        let results: Vec<(String, f64, i64)> = (0..num_groups)
            .filter(|&gid| group_counts[gid] > 0)
            .map(|gid| (dict_strings[gid].clone(), group_sums[gid], group_counts[gid]))
            .collect();
        
        Ok(Some(results))
    }

    /// Execute BETWEEN + GROUP BY + aggregate directly on V4 columns.
    /// Supports both in-memory and mmap-only paths.
    pub fn execute_between_group_agg(
        &self,
        filter_col: &str,
        lo: f64,
        hi: f64,
        group_col: &str,
        agg_col: Option<&str>,
    ) -> io::Result<Option<Vec<(String, f64, i64)>>> {
        if !self.has_v4_in_memory_data() {
            // MMAP PATH: build dict + scan filter+agg from disk
            if let Some((dict_strings, group_ids)) = self.build_string_dict_cache(group_col)? {
                return self.execute_between_group_agg_cached_mmap(filter_col, lo, hi, &dict_strings, &group_ids, agg_col);
            }
            return Ok(None);
        }
        
        let schema = self.schema.read();
        let columns = self.columns.read();
        let deleted = self.deleted.read();
        let total_rows = self.ids.read().len();
        
        let filter_idx = match schema.get_index(filter_col) {
            Some(idx) => idx,
            None => return Ok(None),
        };
        let group_idx = match schema.get_index(group_col) {
            Some(idx) => idx,
            None => return Ok(None),
        };
        let agg_idx = agg_col.and_then(|ac| schema.get_index(ac));
        
        if filter_idx >= columns.len() || group_idx >= columns.len() {
            return Ok(None);
        }
        
        let has_deleted = deleted.iter().any(|&b| b != 0);
        let lo_i64 = lo as i64;
        let hi_i64 = hi as i64;
        
        let (group_offsets, group_bytes) = match &columns[group_idx] {
            ColumnData::String { offsets, data } => (offsets, data),
            _ => return Ok(None),
        };
        let group_count = group_offsets.len().saturating_sub(1);
        let scan_rows = total_rows.min(group_count);
        
        // Single-pass: build group dict + filter + aggregate simultaneously
        // Use small linear-scan dict for ≤64 groups (faster than hash map for short strings)
        let mut dict_entries: Vec<(u32, u32)> = Vec::with_capacity(32); // (start, end) in group_bytes
        let mut dict_strings: Vec<String> = Vec::with_capacity(32);
        let mut group_sums = vec![0.0f64; 64]; // pre-alloc for up to 64 groups
        let mut group_counts = vec![0i64; 64];
        
        let filter_i64 = match &columns[filter_idx] {
            ColumnData::Int64(v) => Some(v.as_slice()),
            _ => None,
        };
        let filter_f64 = match &columns[filter_idx] {
            ColumnData::Float64(v) => Some(v.as_slice()),
            _ => None,
        };
        let agg_i64 = agg_idx.and_then(|idx| {
            if idx < columns.len() { match &columns[idx] { ColumnData::Int64(v) => Some(v.as_slice()), _ => None } } else { None }
        });
        let agg_f64 = agg_idx.and_then(|idx| {
            if idx < columns.len() { match &columns[idx] { ColumnData::Float64(v) => Some(v.as_slice()), _ => None } } else { None }
        });
        
        // Macro for the inner aggregation to avoid code duplication
        macro_rules! agg_row {
            ($gid:expr, $i:expr) => {
                group_counts[$gid] += 1;
                if let Some(av) = agg_f64 { if $i < av.len() { group_sums[$gid] += av[$i]; } }
                else if let Some(av) = agg_i64 { if $i < av.len() { group_sums[$gid] += av[$i] as f64; } }
            }
        }
        
        // Inline group lookup: linear scan of ≤64 entries
        #[inline(always)]
        fn find_group(dict: &[(u32, u32)], group_bytes: &[u8], s: usize, e: usize) -> Option<usize> {
            let needle = &group_bytes[s..e];
            let needle_len = (e - s) as u32;
            for (idx, &(ds, de)) in dict.iter().enumerate() {
                if de - ds == needle_len && &group_bytes[ds as usize..de as usize] == needle {
                    return Some(idx);
                }
            }
            None
        }
        
        // Single-pass: filter + group + aggregate
        if let Some(vals) = filter_i64 {
            let limit = scan_rows.min(vals.len());
            if has_deleted {
                for i in 0..limit {
                    let b = i / 8; let bit = i % 8;
                    if b < deleted.len() && (deleted[b] >> bit) & 1 != 0 { continue; }
                    if vals[i] >= lo_i64 && vals[i] <= hi_i64 && i < group_count {
                        let s = group_offsets[i] as usize;
                        let e = group_offsets[i + 1] as usize;
                        let gid = if let Some(g) = find_group(&dict_entries, group_bytes, s, e) { g }
                        else {
                            let g = dict_entries.len();
                            dict_entries.push((s as u32, e as u32));
                            dict_strings.push(std::str::from_utf8(&group_bytes[s..e]).unwrap_or("").to_string());
                            if g >= group_sums.len() { group_sums.resize(g + 16, 0.0); group_counts.resize(g + 16, 0); }
                            g
                        };
                        agg_row!(gid, i);
                    }
                }
            } else {
                for i in 0..limit {
                    if vals[i] >= lo_i64 && vals[i] <= hi_i64 && i < group_count {
                        let s = group_offsets[i] as usize;
                        let e = group_offsets[i + 1] as usize;
                        let gid = if let Some(g) = find_group(&dict_entries, group_bytes, s, e) { g }
                        else {
                            let g = dict_entries.len();
                            dict_entries.push((s as u32, e as u32));
                            dict_strings.push(std::str::from_utf8(&group_bytes[s..e]).unwrap_or("").to_string());
                            if g >= group_sums.len() { group_sums.resize(g + 16, 0.0); group_counts.resize(g + 16, 0); }
                            g
                        };
                        agg_row!(gid, i);
                    }
                }
            }
        } else if let Some(vals) = filter_f64 {
            let limit = scan_rows.min(vals.len());
            for i in 0..limit {
                if has_deleted {
                    let b = i / 8; let bit = i % 8;
                    if b < deleted.len() && (deleted[b] >> bit) & 1 != 0 { continue; }
                }
                if vals[i] >= lo && vals[i] <= hi && i < group_count {
                    let s = group_offsets[i] as usize;
                    let e = group_offsets[i + 1] as usize;
                    let gid = if let Some(g) = find_group(&dict_entries, group_bytes, s, e) { g }
                    else {
                        let g = dict_entries.len();
                        dict_entries.push((s as u32, e as u32));
                        dict_strings.push(std::str::from_utf8(&group_bytes[s..e]).unwrap_or("").to_string());
                        if g >= group_sums.len() { group_sums.resize(g + 16, 0.0); group_counts.resize(g + 16, 0); }
                        g
                    };
                    agg_row!(gid, i);
                }
            }
        }
        
        let num_groups = dict_entries.len();
        
        let results: Vec<(String, f64, i64)> = (0..num_groups)
            .filter(|&gid| group_counts[gid] > 0)
            .map(|gid| (dict_strings[gid].clone(), group_sums[gid], group_counts[gid]))
            .collect();
        
        Ok(Some(results))
    }

    /// Execute GROUP BY + aggregate directly on V4 columns (no WHERE filter).
    /// Supports both in-memory and mmap-only paths.
    pub fn execute_group_agg(
        &self,
        group_col: &str,
        agg_cols: &[(&str, bool)],
    ) -> io::Result<Option<Vec<(String, Vec<(f64, i64)>)>>> {
        if !self.has_v4_in_memory_data() {
            // MMAP PATH: build dict cache from disk, then aggregate
            if let Some((dict_strings, group_ids)) = self.build_string_dict_cache(group_col)? {
                return self.execute_group_agg_cached_mmap(&dict_strings, &group_ids, agg_cols);
            }
            return Ok(None);
        }
        
        let schema = self.schema.read();
        let columns = self.columns.read();
        let deleted = self.deleted.read();
        let total_rows = self.ids.read().len();
        
        let group_idx = match schema.get_index(group_col) {
            Some(idx) => idx,
            None => return Ok(None),
        };
        if group_idx >= columns.len() { return Ok(None); }
        
        let has_deleted = deleted.iter().any(|&b| b != 0);
        
        let (group_offsets, group_bytes) = match &columns[group_idx] {
            ColumnData::String { offsets, data } => (offsets, data),
            _ => return Ok(None),
        };
        let group_count = group_offsets.len().saturating_sub(1);
        let scan_rows = total_rows.min(group_count);
        let num_aggs = agg_cols.len();
        
        // Resolve agg column slices
        struct AggSlice<'a> { i64_vals: Option<&'a [i64]>, f64_vals: Option<&'a [f64]>, is_count: bool }
        let agg_slices: Vec<AggSlice> = agg_cols.iter().map(|(name, is_count)| {
            if *is_count {
                AggSlice { i64_vals: None, f64_vals: None, is_count: true }
            } else if let Some(idx) = schema.get_index(name) {
                if idx < columns.len() {
                    match &columns[idx] {
                        ColumnData::Int64(v) => AggSlice { i64_vals: Some(v.as_slice()), f64_vals: None, is_count: false },
                        ColumnData::Float64(v) => AggSlice { i64_vals: None, f64_vals: Some(v.as_slice()), is_count: false },
                        _ => AggSlice { i64_vals: None, f64_vals: None, is_count: true },
                    }
                } else { AggSlice { i64_vals: None, f64_vals: None, is_count: true } }
            } else { AggSlice { i64_vals: None, f64_vals: None, is_count: true } }
        }).collect();
        
        // Linear-scan dictionary for ≤64 groups (faster than hash map for short strings)
        let mut dict_entries: Vec<(u32, u32)> = Vec::with_capacity(32);
        let mut dict_strings: Vec<String> = Vec::with_capacity(32);
        let max_flat = 64 * num_aggs;
        let mut flat_sums = vec![0.0f64; max_flat];
        let mut flat_counts = vec![0i64; max_flat];
        
        #[inline(always)]
        fn find_group_ga(dict: &[(u32, u32)], gb: &[u8], s: usize, e: usize) -> Option<usize> {
            let needle = &gb[s..e];
            let nlen = (e - s) as u32;
            for (idx, &(ds, de)) in dict.iter().enumerate() {
                if de - ds == nlen && &gb[ds as usize..de as usize] == needle { return Some(idx); }
            }
            None
        }
        
        // Single-pass: group + aggregate
        for i in 0..scan_rows {
            if has_deleted {
                let b = i / 8; let bit = i % 8;
                if b < deleted.len() && (deleted[b] >> bit) & 1 != 0 { continue; }
            }
            let s = group_offsets[i] as usize;
            let e = group_offsets[i + 1] as usize;
            let gid = if let Some(g) = find_group_ga(&dict_entries, group_bytes, s, e) { g }
            else {
                let g = dict_entries.len();
                dict_entries.push((s as u32, e as u32));
                dict_strings.push(std::str::from_utf8(&group_bytes[s..e]).unwrap_or("").to_string());
                if (g + 1) * num_aggs > flat_sums.len() {
                    flat_sums.resize((g + 16) * num_aggs, 0.0);
                    flat_counts.resize((g + 16) * num_aggs, 0);
                }
                g
            };
            let base = gid * num_aggs;
            for (ai, agg) in agg_slices.iter().enumerate() {
                flat_counts[base + ai] += 1;
                if !agg.is_count {
                    if let Some(vals) = agg.f64_vals { if i < vals.len() { flat_sums[base + ai] += vals[i]; } }
                    else if let Some(vals) = agg.i64_vals { if i < vals.len() { flat_sums[base + ai] += vals[i] as f64; } }
                }
            }
        }
        
        let num_groups = dict_entries.len();
        let results: Vec<(String, Vec<(f64, i64)>)> = (0..num_groups)
            .filter(|&gid| flat_counts[gid * num_aggs] > 0)
            .map(|gid| {
                let aggs: Vec<(f64, i64)> = (0..num_aggs)
                    .map(|ai| (flat_sums[gid * num_aggs + ai], flat_counts[gid * num_aggs + ai]))
                    .collect();
                (dict_strings[gid].clone(), aggs)
            })
            .collect();
        
        Ok(Some(results))
    }

    /// Get the current durability level
    pub fn durability(&self) -> super::DurabilityLevel {
        self.durability
    }
    
    /// Set the durability level
    /// 
    /// Note: This only affects future operations. Existing buffered data
    /// is not automatically synced when changing to a higher durability level.
    pub fn set_durability(&mut self, level: super::DurabilityLevel) {
        self.durability = level;
    }
    
    /// Checkpoint: merge WAL records into main file and clear WAL
    /// 
    /// This is called automatically on save() for safe/max modes.
    /// After checkpoint, all data is in the main file and WAL is cleared.
    /// This improves read performance by eliminating WAL merge overhead.
    pub fn checkpoint(&self) -> io::Result<()> {
        if self.durability == super::DurabilityLevel::Fast {
            return Ok(()); // No WAL in fast mode
        }
        
        let wal_buffer = self.wal_buffer.read();
        if wal_buffer.is_empty() {
            return Ok(()); // Nothing to checkpoint
        }
        drop(wal_buffer);
        
        // Save main file (this persists all in-memory data including WAL records)
        self.save()?;
        
        // Clear WAL after successful save
        {
            let mut wal_buffer = self.wal_buffer.write();
            let mut wal_writer = self.wal_writer.write();
            
            wal_buffer.clear();
            
            // Create fresh WAL file
            if let Some(_) = wal_writer.take() {
                let wal_path = Self::wal_path(&self.path);
                *wal_writer = Some(super::incremental::WalWriter::create(
                    &wal_path, 
                    self.next_id.load(Ordering::SeqCst)
                )?);
            }
        }
        
        Ok(())
    }
    
    /// Get number of pending WAL records
    pub fn wal_record_count(&self) -> usize {
        self.wal_buffer.read().len()
    }
    
    /// Check if WAL needs checkpoint (has pending records)
    pub fn needs_checkpoint(&self) -> bool {
        !self.wal_buffer.read().is_empty()
    }

    /// Get constraints for a column by name (returns default if none set)
    pub fn get_column_constraints(&self, name: &str) -> ColumnConstraints {
        let schema = self.schema.read();
        schema.get_constraints(name).cloned().unwrap_or_default()
    }

    /// Check if any column has constraints defined
    pub fn has_constraints(&self) -> bool {
        let schema = self.schema.read();
        schema.constraints.iter().any(|c| c.not_null || c.primary_key || c.unique || c.check_expr_sql.is_some() || c.foreign_key.is_some())
    }

    /// Set constraints for a column by name
    pub fn set_column_constraints(&self, name: &str, cons: ColumnConstraints) {
        let mut schema = self.schema.write();
        if let Some(idx) = schema.get_index(name) {
            if idx < schema.constraints.len() {
                schema.constraints[idx] = cons;
            }
        }
    }

    /// Write a transactional INSERT record to WAL (P0-4: WAL-first for crash recovery)
    pub fn wal_write_txn_insert(&self, txn_id: u64, id: u64, data: HashMap<String, super::on_demand::ColumnValue>) -> io::Result<()> {
        let mut wal_writer = self.wal_writer.write();
        if let Some(writer) = wal_writer.as_mut() {
            let record = super::incremental::WalRecord::Insert { id, data, txn_id };
            writer.append(&record)?;
        }
        Ok(())
    }

    /// Write a transactional DELETE record to WAL (P0-4: WAL-first for crash recovery)
    pub fn wal_write_txn_delete(&self, txn_id: u64, id: u64) -> io::Result<()> {
        let mut wal_writer = self.wal_writer.write();
        if let Some(writer) = wal_writer.as_mut() {
            let record = super::incremental::WalRecord::Delete { id, txn_id };
            writer.append(&record)?;
        }
        Ok(())
    }

    /// Write a transaction BEGIN marker to WAL (for crash recovery)
    pub fn wal_write_txn_begin(&self, txn_id: u64) -> io::Result<()> {
        let mut wal_writer = self.wal_writer.write();
        if let Some(writer) = wal_writer.as_mut() {
            let record = super::incremental::WalRecord::TxnBegin { txn_id };
            writer.append(&record)?;
        }
        Ok(())
    }

    /// Write a transaction COMMIT marker to WAL (for crash recovery)
    pub fn wal_write_txn_commit(&self, txn_id: u64) -> io::Result<()> {
        let mut wal_writer = self.wal_writer.write();
        if let Some(writer) = wal_writer.as_mut() {
            let record = super::incremental::WalRecord::TxnCommit { txn_id };
            writer.append(&record)?;
            writer.flush()?;
            if self.durability == super::DurabilityLevel::Max {
                writer.sync()?;
            }
        }
        Ok(())
    }

    /// Write a transaction ROLLBACK marker to WAL (for crash recovery)
    pub fn wal_write_txn_rollback(&self, txn_id: u64) -> io::Result<()> {
        let mut wal_writer = self.wal_writer.write();
        if let Some(writer) = wal_writer.as_mut() {
            let record = super::incremental::WalRecord::TxnRollback { txn_id };
            writer.append(&record)?;
            writer.flush()?;
        }
        Ok(())
    }

    /// Sync the WAL to disk (fsync)
    pub fn wal_sync(&self) -> io::Result<()> {
        let mut wal_writer = self.wal_writer.write();
        if let Some(writer) = wal_writer.as_mut() {
            writer.sync()?;
        }
        Ok(())
    }

    // ========================================================================
    // Query APIs
    // ========================================================================

    /// Get row count (includes both base file and delta rows)
    pub fn row_count(&self) -> u64 {
        let base_rows = self.header.read().row_count;
        let delta_rows = self.delta_row_count() as u64;
        base_rows + delta_rows
    }
    
    /// Get base file row count only (without delta)
    pub fn base_row_count(&self) -> u64 {
        self.header.read().row_count
    }

    /// Get column names
    pub fn column_names(&self) -> Vec<String> {
        let mut names = vec!["_id".to_string()];
        names.extend(self.schema.read().columns.iter().map(|(name, _)| name.clone()));
        names
    }

    /// Get schema
    pub fn get_schema(&self) -> Vec<(String, ColumnType)> {
        self.schema.read().columns.clone()
    }

    /// Get header info: (footer_offset, row_count)
    #[inline]
    pub fn header_info(&self) -> (u64, u64) {
        let h = self.header.read();
        (h.footer_offset, h.row_count)
    }

    /// Get the next available ID value
    #[inline]
    pub fn next_id_value(&self) -> u64 {
        self.next_id.load(std::sync::atomic::Ordering::Relaxed)
    }

    // ========================================================================
    // Compatibility APIs (matching ColumnarStorage interface)
    // ========================================================================

    /// Insert rows using generic value type (compatibility with ColumnarStorage)
    /// Optimized with single-pass column collection
    /// 
    /// For safe/max durability modes, rows are written to WAL first for crash recovery.
    /// - Safe mode: WAL is flushed but fsync is deferred to flush() call
    /// - Max mode: WAL is fsync'd immediately after each insert for strongest guarantee
    pub fn insert_rows(&self, rows: &[HashMap<String, ColumnValue>]) -> io::Result<Vec<u64>> {
        if rows.is_empty() {
            return Ok(Vec::new());
        }
        
        // For safe/max durability with batch writes: use WAL for efficiency
        // Single-row writes skip WAL (original fsync-on-save behavior is faster)
        // WAL benefit: single I/O for many rows; WAL overhead: extra I/O for single rows
        let start_id = self.next_id.load(Ordering::SeqCst);
        let use_wal = self.durability != super::DurabilityLevel::Fast && rows.len() > 1;
        
        if use_wal {
            // Batch writes: use WAL for efficiency (single I/O for all rows)
            let mut wal_writer = self.wal_writer.write();
            
            if let Some(writer) = wal_writer.as_mut() {
                let record = super::incremental::WalRecord::BatchInsert { 
                    start_id, 
                    rows: rows.to_vec(),
                    txn_id: 0,
                };
                writer.append(&record)?;
                writer.flush()?;
                
                // For max durability: fsync WAL immediately
                if self.durability == super::DurabilityLevel::Max {
                    writer.sync()?;
                }
            }
        }
        // Note: For single-row writes, fsync happens in save() based on durability level
        
        // Handle case where all rows are empty dicts - still create rows with just _id
        let all_empty = rows.iter().all(|r| r.is_empty());
        if all_empty {
            let row_count = rows.len();
            let start_id = self.next_id.fetch_add(row_count as u64, Ordering::SeqCst);
            let ids: Vec<u64> = (start_id..start_id + row_count as u64).collect();
            
            // Add IDs
            self.ids.write().extend_from_slice(&ids);
            
            // Update header
            {
                let mut header = self.header.write();
                header.row_count = self.ids.read().len() as u64;
            }
            
            // Update id_to_idx mapping only if it's already built
            {
                let ids_guard = self.ids.read();
                let mut id_to_idx = self.id_to_idx.write();
                if let Some(map) = id_to_idx.as_mut() {
                    let start_idx = ids_guard.len() - ids.len();
                    for (i, &id) in ids.iter().enumerate() {
                        map.insert(id, start_idx + i);
                    }
                }
            }
            
            // Extend deleted bitmap
            {
                let mut deleted = self.deleted.write();
                let new_len = (self.ids.read().len() + 7) / 8;
                deleted.resize(new_len, 0);
            }
            
            // Update active count
            self.active_count.fetch_add(row_count as u64, Ordering::Relaxed);
            
            // Update pending rows counter and check auto-flush
            self.pending_rows.fetch_add(row_count as u64, Ordering::Relaxed);
            self.maybe_auto_flush()?;
            
            return Ok(ids);
        }

        // Single-pass optimized: determine column types from first non-empty row
        // and pre-allocate all vectors
        let num_rows = rows.len();
        let mut int_columns: HashMap<String, Vec<i64>> = HashMap::new();
        let mut float_columns: HashMap<String, Vec<f64>> = HashMap::new();
        let mut string_columns: HashMap<String, Vec<String>> = HashMap::new();
        let mut binary_columns: HashMap<String, Vec<Vec<u8>>> = HashMap::new();
        let mut bool_columns: HashMap<String, Vec<bool>> = HashMap::new();
        let mut null_positions: HashMap<String, Vec<bool>> = HashMap::new();

        // CRITICAL: Include ALL existing schema columns to ensure proper alignment
        // This fixes the partial column insert bug where missing columns don't get padded
        {
            let schema = self.schema.read();
            for (col_name, col_type) in &schema.columns {
                match col_type {
                    ColumnType::Int64 | ColumnType::Int8 | ColumnType::Int16 | ColumnType::Int32 |
                    ColumnType::UInt8 | ColumnType::UInt16 | ColumnType::UInt32 | ColumnType::UInt64 |
                    ColumnType::Timestamp | ColumnType::Date => {
                        int_columns.insert(col_name.clone(), Vec::with_capacity(num_rows));
                    }
                    ColumnType::Float64 | ColumnType::Float32 => {
                        float_columns.insert(col_name.clone(), Vec::with_capacity(num_rows));
                    }
                    ColumnType::String | ColumnType::StringDict | ColumnType::Null => {
                        string_columns.insert(col_name.clone(), Vec::with_capacity(num_rows));
                    }
                    ColumnType::Binary => {
                        binary_columns.insert(col_name.clone(), Vec::with_capacity(num_rows));
                    }
                    ColumnType::Bool => {
                        bool_columns.insert(col_name.clone(), Vec::with_capacity(num_rows));
                    }
                }
                null_positions.insert(col_name.clone(), Vec::with_capacity(num_rows));
            }
        }

        // Also determine schema from input rows for NEW columns
        let sample_size = std::cmp::min(10, num_rows);
        for row in rows.iter().take(sample_size) {
            for (key, val) in row {
                if int_columns.contains_key(key) || float_columns.contains_key(key) 
                    || string_columns.contains_key(key) || binary_columns.contains_key(key)
                    || bool_columns.contains_key(key) {
                    continue;
                }
                match val {
                    ColumnValue::Int64(_) => { int_columns.insert(key.clone(), Vec::with_capacity(num_rows)); }
                    ColumnValue::Float64(_) => { float_columns.insert(key.clone(), Vec::with_capacity(num_rows)); }
                    ColumnValue::String(_) => { string_columns.insert(key.clone(), Vec::with_capacity(num_rows)); }
                    ColumnValue::Binary(_) => { binary_columns.insert(key.clone(), Vec::with_capacity(num_rows)); }
                    ColumnValue::Bool(_) => { bool_columns.insert(key.clone(), Vec::with_capacity(num_rows)); }
                    ColumnValue::Null => { string_columns.insert(key.clone(), Vec::with_capacity(num_rows)); }
                }
                null_positions.insert(key.clone(), Vec::with_capacity(num_rows));
            }
        }

        // Pre-allocate NULL string to avoid repeated allocation
        static NULL_MARKER: &str = "\x00__NULL__\x00";
        
        // Single pass: collect all values and track NULLs
        // Note: For homogeneous data (common case), new columns won't be discovered mid-stream
        for row in rows {
            // Handle new columns discovered mid-stream (rare case for heterogeneous data)
            for (key, val) in row {
                if !int_columns.contains_key(key) && !float_columns.contains_key(key) 
                    && !string_columns.contains_key(key) && !binary_columns.contains_key(key)
                    && !bool_columns.contains_key(key) {
                    match val {
                        ColumnValue::Int64(_) => { int_columns.insert(key.clone(), Vec::with_capacity(num_rows)); }
                        ColumnValue::Float64(_) => { float_columns.insert(key.clone(), Vec::with_capacity(num_rows)); }
                        ColumnValue::String(_) => { string_columns.insert(key.clone(), Vec::with_capacity(num_rows)); }
                        ColumnValue::Binary(_) => { binary_columns.insert(key.clone(), Vec::with_capacity(num_rows)); }
                        ColumnValue::Bool(_) => { bool_columns.insert(key.clone(), Vec::with_capacity(num_rows)); }
                        ColumnValue::Null => { string_columns.insert(key.clone(), Vec::with_capacity(num_rows)); }
                    }
                    null_positions.insert(key.clone(), Vec::with_capacity(num_rows));
                }
            }
            
            // Collect values for all columns and track NULL positions
            for (key, col) in int_columns.iter_mut() {
                let (val, is_null) = match row.get(key) {
                    Some(ColumnValue::Int64(v)) => (*v, false),
                    Some(ColumnValue::Null) | None => (0, true),
                    _ => (0, true),
                };
                col.push(val);
                null_positions.entry(key.clone()).or_default().push(is_null);
            }
            for (key, col) in float_columns.iter_mut() {
                let (val, is_null) = match row.get(key) {
                    Some(ColumnValue::Float64(v)) => (*v, false),
                    Some(ColumnValue::Null) | None => (0.0, true),
                    _ => (0.0, true),
                };
                col.push(val);
                null_positions.entry(key.clone()).or_default().push(is_null);
            }
            for (key, col) in string_columns.iter_mut() {
                let (val, is_null) = match row.get(key) {
                    Some(ColumnValue::String(v)) => (v.clone(), false),
                    Some(ColumnValue::Null) => (NULL_MARKER.to_string(), true),
                    None => (String::new(), true),
                    _ => (String::new(), true),
                };
                col.push(val);
                null_positions.entry(key.clone()).or_default().push(is_null);
            }
            for (key, col) in binary_columns.iter_mut() {
                let (val, is_null) = match row.get(key) {
                    Some(ColumnValue::Binary(v)) => (v.clone(), false),
                    Some(ColumnValue::Null) | None => (Vec::new(), true),
                    _ => (Vec::new(), true),
                };
                col.push(val);
                null_positions.entry(key.clone()).or_default().push(is_null);
            }
            for (key, col) in bool_columns.iter_mut() {
                let (val, is_null) = match row.get(key) {
                    Some(ColumnValue::Bool(v)) => (*v, false),
                    Some(ColumnValue::Null) | None => (false, true),
                    _ => (false, true),
                };
                col.push(val);
                null_positions.entry(key.clone()).or_default().push(is_null);
            }
        }

        let result = self.insert_typed_with_nulls(int_columns, float_columns, string_columns, binary_columns, bool_columns, null_positions)?;
        
        // Update pending rows counter and check auto-flush
        self.pending_rows.fetch_add(result.len() as u64, Ordering::Relaxed);
        self.maybe_auto_flush()?;
        
        Ok(result)
    }

    /// Insert typed columns and immediately persist to disk
    /// 
    /// This is the preferred method for V3 direct writes - data is immediately
    /// visible to V3Executor after this call returns.
    pub fn insert_typed_and_persist(
        &self,
        int_columns: HashMap<String, Vec<i64>>,
        float_columns: HashMap<String, Vec<f64>>,
        string_columns: HashMap<String, Vec<String>>,
        bool_columns: HashMap<String, Vec<bool>>,
    ) -> io::Result<Vec<u64>> {
        let ids = self.insert_typed(int_columns, float_columns, string_columns, HashMap::new(), bool_columns)?;
        if !ids.is_empty() {
            self.save()?;
        }
        Ok(ids)
    }

    /// Append delta (for compatibility - just calls insert_rows + save)
    pub fn append_delta(&self, rows: &[HashMap<String, ColumnValue>]) -> io::Result<Vec<u64>> {
        let ids = self.insert_rows(rows)?;
        self.save()?;
        Ok(ids)
    }

    /// Fast delta append (same as append_delta for this format)
    pub fn append_delta_fast(&self, rows: &[HashMap<String, ColumnValue>]) -> io::Result<Vec<u64>> {
        self.append_delta(rows)
    }

    /// Check if compaction is needed (true if delta file exists)
    pub fn needs_compaction(&self) -> bool {
        self.has_delta()
    }

    /// Flush changes to disk
    pub fn flush(&self) -> io::Result<()> {
        self.save()
    }

    /// Close storage and release all resources
    /// IMPORTANT: On Windows, mmap must be released before temp directory cleanup
    pub fn close(&self) -> io::Result<()> {
        // Save any pending changes first
        self.save()?;
        
        // Release mmap cache BEFORE closing file (critical for Windows)
        self.mmap_cache.write().invalidate();
        
        // Close file handle
        *self.file.write() = None;
        
        Ok(())
    }
    
    /// Release mmap without saving (for cleanup scenarios)
    pub fn release_mmap(&self) {
        self.mmap_cache.write().invalidate();
    }
}

impl Drop for OnDemandStorage {
    fn drop(&mut self) {
        // Release mmap first (critical for Windows)
        // parking_lot's try_write returns Option, not Result
        if let Some(mut cache) = self.mmap_cache.try_write() {
            cache.invalidate();
        }
        // File handle will be dropped automatically after mmap is released
    }
}


