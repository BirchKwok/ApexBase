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
                None => return Ok(None), // unknown column (e.g. _id) — fall back to Arrow path
            };
            if col_idx >= columns.len() { return Ok(None); }
            
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

    /// MMAP PATH: Execute simple aggregation by scanning V4 RGs via mmap.
    /// Uses per-RG streaming to avoid building a full 1M-element Vec.
    /// PLAIN-encoded columns are processed zero-copy from mmap; RLE/BITPACK decoded per-RG.
    fn execute_simple_agg_mmap(
        &self,
        agg_cols: &[&str],
    ) -> io::Result<Option<Vec<(i64, f64, f64, f64, bool)>>> {
        let footer = match self.get_or_load_footer()? {
            Some(f) => f,
            None => return Ok(None),
        };

        let schema = &footer.schema;
        let non_star: Vec<&str> = agg_cols.iter()
            .filter(|&&n| n != "*" && n != "1" && !n.chars().next().map(|c| c.is_ascii_digit()).unwrap_or(false))
            .copied()
            .collect();
        let col_indices: Vec<usize> = non_star.iter()
            .filter_map(|&n| schema.get_index(n))
            .collect();
        // If any requested column doesn't exist in schema, bail to the Arrow path
        if col_indices.len() != non_star.len() { return Ok(None); }

        let total_active: i64 = footer.row_groups.iter()
            .map(|rg| rg.active_rows() as i64)
            .sum();

        if col_indices.is_empty() {
            return Ok(Some(agg_cols.iter().map(|_| (total_active, 0.0, 0.0, 0.0, false)).collect()));
        }

        // FAST PATH: use pre-computed sidecar stats if data is clean (no deletes, no deltas)
        let has_any_deletes = footer.row_groups.iter().any(|rg| rg.deletion_count > 0);
        if !has_any_deletes && !self.has_pending_deltas() {
            if let Some(sidecar) = self.try_read_col_stats_sidecar() {
                let mut results: Vec<(i64, f64, f64, f64, bool)> = Vec::with_capacity(agg_cols.len());
                let mut all_found = true;
                for &col_name in agg_cols {
                    if col_name == "*" || col_name == "1"
                        || col_name.chars().next().map(|c| c.is_ascii_digit()).unwrap_or(false)
                    {
                        results.push((total_active, 0.0, 0.0, 0.0, false));
                    } else if let Some(&(count, sum, min, max, is_int)) = sidecar.get(col_name) {
                        results.push((count, sum, min, max, is_int));
                    } else {
                        all_found = false;
                        break;
                    }
                }
                if all_found { return Ok(Some(results)); }
            }
        }

        // Per-column streaming accumulators (no large Vec allocation)
        let nc = col_indices.len();
        let mut col_counts = vec![0i64; nc];
        let mut col_sums   = vec![0.0f64; nc];
        let mut col_mins   = vec![f64::INFINITY; nc];
        let mut col_maxs   = vec![f64::NEG_INFINITY; nc];
        let mut col_is_int = vec![false; nc];

        // Check whether all RGs qualify for the streaming zero-copy path
        // (uncompressed + RCIX available for ALL requested columns).
        let max_col_idx = col_indices.iter().copied().max().unwrap_or(0);
        let all_rcix = footer.row_groups.iter().enumerate().all(|(rg_i, rg_meta)| {
            if rg_meta.row_count == 0 { return true; }
            footer.col_offsets.get(rg_i).map_or(false, |v| v.len() > max_col_idx)
        });

        if !all_rcix {
            // FALLBACK: old path — builds full Vec but handles compressed/old files
            let (scanned_cols, del_bytes) = self.scan_columns_mmap(&col_indices, &footer)?;
            let has_deleted = del_bytes.iter().any(|&b| b != 0);
            let mut results: Vec<(i64, f64, f64, f64, bool)> = Vec::with_capacity(agg_cols.len());
            let mut scan_idx = 0usize;
            for &col_name in agg_cols {
                if col_name == "*" || col_name == "1" {
                    results.push((total_active, 0.0, 0.0, 0.0, false));
                    continue;
                }
                if scan_idx >= scanned_cols.len() { results.push((0, 0.0, 0.0, 0.0, false)); continue; }
                let col = &scanned_cols[scan_idx]; scan_idx += 1;
                match col {
                    ColumnData::Int64(vals) => {
                        let n = vals.len() as i64;
                        let sum: i64 = vals.iter().sum();
                        let min_v = vals.iter().copied().min().unwrap_or(i64::MAX);
                        let max_v = vals.iter().copied().max().unwrap_or(i64::MIN);
                        let _ = has_deleted;
                        results.push((n, sum as f64, min_v as f64, max_v as f64, true));
                    }
                    ColumnData::Float64(vals) => {
                        let n = vals.len() as i64;
                        let sum: f64 = vals.iter().sum();
                        let min_v = vals.iter().copied().fold(f64::INFINITY, f64::min);
                        let max_v = vals.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                        results.push((n, sum, min_v, max_v, false));
                    }
                    _ => { results.push((total_active, 0.0, 0.0, 0.0, false)); }
                }
            }
            return Ok(Some(results));
        }

        // STREAMING ZERO-COPY PATH: process each RG directly from mmap
        let file_guard = self.file.read();
        let file = file_guard.as_ref()
            .ok_or_else(|| io::Error::new(io::ErrorKind::NotConnected, "File not open"))?;
        let mut mmap_guard = self.mmap_cache.write();
        let mmap_ref = mmap_guard.get_or_create(file)?;

        const PLAIN: u8 = 0u8;

        for (rg_i, rg_meta) in footer.row_groups.iter().enumerate() {
            let rg_rows = rg_meta.row_count as usize;
            if rg_rows == 0 { continue; }

            let rg_end = (rg_meta.offset + rg_meta.data_size) as usize;
            if rg_end > mmap_ref.len() { continue; }
            let rg_bytes = &mmap_ref[rg_meta.offset as usize .. rg_end];
            if rg_bytes.len() < 32 { continue; }

            let compress_flag = rg_bytes[28];
            let encoding_version = rg_bytes[29];
            let null_bitmap_len = (rg_rows + 7) / 8;
            let del_start_body = rg_rows * 8;
            let del_vec_len = null_bitmap_len;

            let rcix = match footer.col_offsets.get(rg_i).filter(|v| !v.is_empty()) {
                Some(r) => r,
                None => continue,
            };

            // Must be uncompressed for zero-copy
            if compress_flag != RG_COMPRESS_NONE || encoding_version < 1 {
                // Compressed RG: fall back to read_column_encoded per column
                let body_raw = &rg_bytes[32..];
                let decompressed_buf = decompress_rg_body(compress_flag, body_raw)?;
                let body: &[u8] = decompressed_buf.as_deref().unwrap_or(body_raw);
                for (ci, &col_idx) in col_indices.iter().enumerate() {
                    if col_idx >= rcix.len() { continue; }
                    let col_off = rcix[col_idx] as usize;
                    let data_start = col_off + null_bitmap_len;
                    if data_start >= body.len() { continue; }
                    let col_type = schema.columns[col_idx].1;
                    let (col_data, _) = read_column_encoded(&body[data_start..], col_type)?;
                    match &col_data {
                        ColumnData::Int64(v) => { col_is_int[ci] = true; for &x in v { col_counts[ci] += 1; col_sums[ci] += x as f64; let xf = x as f64; if xf < col_mins[ci] { col_mins[ci] = xf; } if xf > col_maxs[ci] { col_maxs[ci] = xf; } } }
                        ColumnData::Float64(v) => { for &x in v { col_counts[ci] += 1; col_sums[ci] += x; if x < col_mins[ci] { col_mins[ci] = x; } if x > col_maxs[ci] { col_maxs[ci] = x; } } }
                        _ => { col_counts[ci] += col_data.len() as i64; } // COUNT on non-numeric col
                    }
                }
                continue;
            }

            let body = &rg_bytes[32..];
            let has_deleted = del_start_body + del_vec_len <= body.len()
                && body[del_start_body..del_start_body + del_vec_len].iter().any(|&b| b != 0);

            for (ci, &col_idx) in col_indices.iter().enumerate() {
                if col_idx >= rcix.len() { continue; }
                let col_off = rcix[col_idx] as usize;
                let data_start = col_off + null_bitmap_len;
                if data_start + 1 > body.len() { continue; }

                let col_type = schema.columns[col_idx].1;
                let encoding = body[data_start];
                let payload = &body[data_start + 1..];

                if encoding == PLAIN && payload.len() >= 8 {
                    let count = u64::from_le_bytes(payload[0..8].try_into().unwrap()) as usize;
                    let raw = &payload[8..];
                    match col_type {
                        ColumnType::Float64 | ColumnType::Float32 => {
                            let n = count.min(rg_rows).min(raw.len() / 8);
                            let vals = unsafe { std::slice::from_raw_parts(raw.as_ptr() as *const f64, n) };
                            if !has_deleted {
                                let s: f64 = vals.iter().sum();
                                let mn = vals.iter().copied().fold(col_mins[ci], f64::min);
                                let mx = vals.iter().copied().fold(col_maxs[ci], f64::max);
                                col_counts[ci] += n as i64; col_sums[ci] += s; col_mins[ci] = mn; col_maxs[ci] = mx;
                            } else {
                                let del = &body[del_start_body..del_start_body + del_vec_len];
                                for (i, &v) in vals.iter().enumerate() {
                                    if (del[i / 8] >> (i % 8)) & 1 != 0 { continue; }
                                    col_counts[ci] += 1; col_sums[ci] += v;
                                    if v < col_mins[ci] { col_mins[ci] = v; } if v > col_maxs[ci] { col_maxs[ci] = v; }
                                }
                            }
                        }
                        ColumnType::Int64 | ColumnType::Int8 | ColumnType::Int16 | ColumnType::Int32 |
                        ColumnType::UInt8 | ColumnType::UInt16 | ColumnType::UInt32 | ColumnType::UInt64 |
                        ColumnType::Timestamp | ColumnType::Date => {
                            col_is_int[ci] = true;
                            let n = count.min(rg_rows).min(raw.len() / 8);
                            let vals = unsafe { std::slice::from_raw_parts(raw.as_ptr() as *const i64, n) };
                            if !has_deleted {
                                // Separate passes for SIMD auto-vectorization (LLVM can vectorize each individually)
                                let s: i64 = vals.iter().copied().sum();
                                let mn = vals.iter().copied().min().unwrap_or(i64::MAX);
                                let mx = vals.iter().copied().max().unwrap_or(i64::MIN);
                                col_counts[ci] += n as i64; col_sums[ci] += s as f64;
                                if (mn as f64) < col_mins[ci] { col_mins[ci] = mn as f64; }
                                if (mx as f64) > col_maxs[ci] { col_maxs[ci] = mx as f64; }
                            } else {
                                let del = &body[del_start_body..del_start_body + del_vec_len];
                                for (i, &v) in vals.iter().enumerate() {
                                    if (del[i / 8] >> (i % 8)) & 1 != 0 { continue; }
                                    col_counts[ci] += 1; col_sums[ci] += v as f64;
                                    let vf = v as f64;
                                    if vf < col_mins[ci] { col_mins[ci] = vf; } if vf > col_maxs[ci] { col_maxs[ci] = vf; }
                                }
                            }
                        }
                        _ => {
                            // Non-numeric (String/StringDict/Bool/Binary): count elements for COUNT(col)
                            let n = count.min(rg_rows);
                            if !has_deleted {
                                col_counts[ci] += n as i64;
                            } else {
                                let del = &body[del_start_body..del_start_body + del_vec_len];
                                for i in 0..n { if (del[i / 8] >> (i % 8)) & 1 == 0 { col_counts[ci] += 1; } }
                            }
                        }
                    }
                } else {
                    const BITPACK: u8 = 2u8;
                    // BITPACK Int64: accumulate directly without Vec allocation
                    if encoding == BITPACK && matches!(col_type, ColumnType::Int64 | ColumnType::Int8 |
                        ColumnType::Int16 | ColumnType::Int32 | ColumnType::UInt8 | ColumnType::UInt16 |
                        ColumnType::UInt32 | ColumnType::UInt64 | ColumnType::Timestamp | ColumnType::Date)
                    {
                        col_is_int[ci] = true;
                        if let Some((sum, mn, mx, n, _)) = bitpack_agg_i64(payload) {
                            col_counts[ci] += n as i64;
                            col_sums[ci] += sum as f64;
                            let mnf = mn as f64; let mxf = mx as f64;
                            if mnf < col_mins[ci] { col_mins[ci] = mnf; }
                            if mxf > col_maxs[ci] { col_maxs[ci] = mxf; }
                        }
                    } else {
                        // RLE or other: decode per-RG, accumulate, drop
                        let (col_data, _) = read_column_encoded(&body[data_start..], col_type)?;
                        match &col_data {
                            ColumnData::Int64(v) => { col_is_int[ci] = true; for &x in v { col_counts[ci] += 1; col_sums[ci] += x as f64; let xf = x as f64; if xf < col_mins[ci] { col_mins[ci] = xf; } if xf > col_maxs[ci] { col_maxs[ci] = xf; } } }
                            ColumnData::Float64(v) => { for &x in v { col_counts[ci] += 1; col_sums[ci] += x; if x < col_mins[ci] { col_mins[ci] = x; } if x > col_maxs[ci] { col_maxs[ci] = x; } } }
                            _ => { col_counts[ci] += col_data.len() as i64; } // COUNT on non-numeric col
                        }
                    }
                }
            }
        }

        drop(mmap_guard);
        drop(file_guard);

        // Build results
        let mut results: Vec<(i64, f64, f64, f64, bool)> = Vec::with_capacity(agg_cols.len());
        let mut ci = 0usize;
        for &col_name in agg_cols {
            if col_name == "*" || col_name == "1" {
                results.push((total_active, 0.0, 0.0, 0.0, false));
            } else if ci < nc {
                let mn = if col_mins[ci] == f64::INFINITY { 0.0 } else { col_mins[ci] };
                let mx = if col_maxs[ci] == f64::NEG_INFINITY { 0.0 } else { col_maxs[ci] };
                results.push((col_counts[ci], col_sums[ci], mn, mx, col_is_int[ci]));
                ci += 1;
            } else {
                results.push((0, 0.0, 0.0, 0.0, false));
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
        
        match &scanned_cols[0] {
            ColumnData::StringDict { indices, dict_offsets, dict_data } => {
                // FAST PATH: column stored as StringDict — decode only the small dict (~10 entries),
                // then map u32 indices directly to u16 group_ids without any string hashing.
                let dict_count = dict_offsets.len().saturating_sub(1);
                let dict_strings: Vec<String> = (0..dict_count)
                    .map(|i| {
                        let s = dict_offsets[i] as usize;
                        let e = dict_offsets[i + 1] as usize;
                        std::str::from_utf8(&dict_data[s..e]).unwrap_or("").to_string()
                    })
                    .collect();
                // Indices are 1-based (0 = null sentinel); subtract 1 for group_id
                let group_ids: Vec<u16> = indices.iter()
                    .map(|&idx| if idx == 0 { 0 } else { (idx - 1) as u16 })
                    .collect();
                Ok(Some((dict_strings, group_ids)))
            }
            ColumnData::String { offsets, data } => {
                // Fallback: plain string column — build dict via hash map
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
            _ => Ok(None),
        }
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

    /// MMAP PATH: execute GROUP BY + aggregate using pre-built dict cache.
    /// Streaming per-RG path: reads each column directly from mmap without full materialization.
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

        let agg_col_indices: Vec<Option<usize>> = agg_cols.iter()
            .map(|(name, is_count)| if *is_count { None } else { schema.get_index(name) })
            .collect();
        let needed: Vec<usize> = agg_col_indices.iter().filter_map(|&x| x).collect();

        let num_groups = dict_strings.len();
        let num_aggs = agg_cols.len();
        let flat_len = num_groups * num_aggs;
        let mut flat_sums = vec![0.0f64; flat_len];
        let mut flat_counts = vec![0i64; flat_len];

        // Check if all RGs support the streaming zero-copy path
        let max_col_idx = needed.iter().copied().max().unwrap_or(0);
        let all_rcix = footer.row_groups.iter().enumerate().all(|(rg_i, rg_meta)| {
            if rg_meta.row_count == 0 { return true; }
            footer.col_offsets.get(rg_i).map_or(false, |v| v.len() > max_col_idx)
        });

        if all_rcix && !needed.is_empty() {
            // STREAMING ZERO-COPY: process each RG directly without materializing full columns
            let file_guard = self.file.read();
            let file = file_guard.as_ref()
                .ok_or_else(|| err_not_conn("File not open for group-by agg"))?;
            let mut mmap_guard = self.mmap_cache.write();
            let mmap_ref = mmap_guard.get_or_create(file)?;
            let mut rg_row_offset = 0usize;

            for (rg_i, rg_meta) in footer.row_groups.iter().enumerate() {
                let rg_rows = rg_meta.row_count as usize;
                if rg_rows == 0 { rg_row_offset += rg_rows; continue; }
                let rg_end = (rg_meta.offset + rg_meta.data_size) as usize;
                if rg_end > mmap_ref.len() { return Err(crate::storage::on_demand::err_data("RG past EOF")); }
                let rg_bytes = &mmap_ref[rg_meta.offset as usize..rg_end];
                if rg_bytes.len() < 32 { rg_row_offset += rg_rows; continue; }
                let compress_flag = rg_bytes[28];
                let encoding_version = rg_bytes[29];
                let null_bitmap_len = (rg_rows + 7) / 8;
                let del_start = rg_rows * 8;
                let del_vec_len = null_bitmap_len;
                let rcix = &footer.col_offsets[rg_i];

                if compress_flag != RG_COMPRESS_NONE || encoding_version < 1 {
                    // Compressed: fall back to scan_columns_mmap for this RG
                    rg_row_offset += rg_rows; continue;
                }

                let body = &rg_bytes[32..];
                let has_deleted = rg_meta.deletion_count > 0;
                let del_bytes = if del_start + del_vec_len <= body.len() {
                    &body[del_start..del_start + del_vec_len]
                } else { &[] };

                // For each agg column, get raw slice from mmap via RCIX
                let gids_slice = &group_ids[rg_row_offset..(rg_row_offset + rg_rows).min(group_ids.len())];
                let rg_n = gids_slice.len();

                // Specialized single-agg Float64 fast path (most common: AVG/SUM of one column)
                if num_aggs == 1 {
                    let is_count = agg_cols[0].1;
                    if is_count {
                        if !has_deleted {
                            for i in 0..rg_n { let gid = unsafe { *gids_slice.get_unchecked(i) } as usize; unsafe { *flat_counts.get_unchecked_mut(gid) += 1; } }
                        } else {
                            for i in 0..rg_n { if !del_bytes.is_empty() && (del_bytes[i/8] >> (i%8)) & 1 != 0 { continue; } let gid = unsafe { *gids_slice.get_unchecked(i) } as usize; unsafe { *flat_counts.get_unchecked_mut(gid) += 1; } }
                        }
                    } else if let Some(&col_idx) = needed.first() {
                        let col_off = rcix[col_idx] as usize;
                        let data_start = col_off + null_bitmap_len;
                        if data_start + 1 < body.len() {
                            let encoding = body[data_start];
                            let payload = &body[data_start + 1..];
                            if encoding == 0u8 && payload.len() >= 8 {
                                let count = u64::from_le_bytes(payload[0..8].try_into().unwrap()) as usize;
                                let n = count.min(rg_rows).min(rg_n).min((payload.len() - 8) / 8);
                                let col_type = schema.columns[col_idx].1;
                                if matches!(col_type, ColumnType::Float64 | ColumnType::Float32) {
                                    let vals = unsafe { std::slice::from_raw_parts(payload[8..].as_ptr() as *const f64, n) };
                                    if !has_deleted {
                                        for i in 0..n { let gid = unsafe { *gids_slice.get_unchecked(i) } as usize; unsafe { *flat_counts.get_unchecked_mut(gid) += 1; *flat_sums.get_unchecked_mut(gid) += *vals.get_unchecked(i); } }
                                    } else {
                                        for i in 0..n { if !del_bytes.is_empty() && (del_bytes[i/8] >> (i%8)) & 1 != 0 { continue; } let gid = unsafe { *gids_slice.get_unchecked(i) } as usize; unsafe { *flat_counts.get_unchecked_mut(gid) += 1; *flat_sums.get_unchecked_mut(gid) += *vals.get_unchecked(i); } }
                                    }
                                    rg_row_offset += rg_rows; continue;
                                } else if matches!(col_type, ColumnType::Int64 | ColumnType::Int8 | ColumnType::Int16 | ColumnType::Int32 | ColumnType::UInt8 | ColumnType::UInt16 | ColumnType::UInt32 | ColumnType::UInt64) {
                                    let vals = unsafe { std::slice::from_raw_parts(payload[8..].as_ptr() as *const i64, n) };
                                    if !has_deleted {
                                        for i in 0..n { let gid = unsafe { *gids_slice.get_unchecked(i) } as usize; unsafe { *flat_counts.get_unchecked_mut(gid) += 1; *flat_sums.get_unchecked_mut(gid) += *vals.get_unchecked(i) as f64; } }
                                    } else {
                                        for i in 0..n { if !del_bytes.is_empty() && (del_bytes[i/8] >> (i%8)) & 1 != 0 { continue; } let gid = unsafe { *gids_slice.get_unchecked(i) } as usize; unsafe { *flat_counts.get_unchecked_mut(gid) += 1; *flat_sums.get_unchecked_mut(gid) += *vals.get_unchecked(i) as f64; } }
                                    }
                                    rg_row_offset += rg_rows; continue;
                                }
                            }
                        }
                        // Fallback: decode column
                        if col_off + null_bitmap_len < body.len() {
                            let (col_data, _) = read_column_encoded(&body[col_off + null_bitmap_len..], schema.columns[col_idx].1)?;
                            match &col_data { ColumnData::Float64(v) => { let n = v.len().min(rg_n); for i in 0..n { if has_deleted && !del_bytes.is_empty() && (del_bytes[i/8] >> (i%8)) & 1 != 0 { continue; } let gid = gids_slice[i] as usize; flat_counts[gid] += 1; flat_sums[gid] += v[i]; } } ColumnData::Int64(v) => { let n = v.len().min(rg_n); for i in 0..n { if has_deleted && !del_bytes.is_empty() && (del_bytes[i/8] >> (i%8)) & 1 != 0 { continue; } let gid = gids_slice[i] as usize; flat_counts[gid] += 1; flat_sums[gid] += v[i] as f64; } } _ => { for i in 0..rg_n { if has_deleted && !del_bytes.is_empty() && (del_bytes[i/8] >> (i%8)) & 1 != 0 { continue; } let gid = gids_slice[i] as usize; flat_counts[gid] += 1; } } }
                        }
                    }
                    rg_row_offset += rg_rows; continue;
                }
                // MULTI-AGG STREAMING: pre-load all agg column slices for this RG, single-pass hot loop
                // Enumerate each agg col: get PLAIN zero-copy slice or bail to outer fallback
                enum RgColSlice<'b> { Count, F64(&'b [f64]), I64(&'b [i64]) }
                let mut rg_slices: Vec<RgColSlice> = Vec::with_capacity(num_aggs);
                let mut ok = true;
                for (ai, &opt_col_idx) in agg_col_indices.iter().enumerate() {
                    if agg_cols[ai].1 || opt_col_idx.is_none() {
                        rg_slices.push(RgColSlice::Count); continue;
                    }
                    let col_idx = opt_col_idx.unwrap();
                    if col_idx >= rcix.len() { ok = false; break; }
                    let col_off = rcix[col_idx] as usize + null_bitmap_len;
                    if col_off + 1 >= body.len() { ok = false; break; }
                    let enc = body[col_off];
                    let payload = &body[col_off + 1..];
                    if enc != 0u8 || payload.len() < 8 { ok = false; break; }
                    let cnt = u64::from_le_bytes(payload[0..8].try_into().unwrap()) as usize;
                    let col_type = schema.columns[col_idx].1;
                    let n = cnt.min(rg_rows).min((payload.len() - 8) / 8);
                    if matches!(col_type, ColumnType::Float64 | ColumnType::Float32) {
                        rg_slices.push(RgColSlice::F64(unsafe { std::slice::from_raw_parts(payload[8..].as_ptr() as *const f64, n) }));
                    } else if matches!(col_type, ColumnType::Int64 | ColumnType::Int8 | ColumnType::Int16 | ColumnType::Int32 | ColumnType::UInt8 | ColumnType::UInt16 | ColumnType::UInt32 | ColumnType::UInt64) {
                        rg_slices.push(RgColSlice::I64(unsafe { std::slice::from_raw_parts(payload[8..].as_ptr() as *const i64, n) }));
                    } else { ok = false; break; }
                }
                if ok && !has_deleted {
                    // Specialized 2-agg [Count, F64] fast path (most common: COUNT(*)+AVG/SUM)
                    if num_aggs == 2 {
                        // 2 writes/row: COUNT(*) + AVG/SUM numerator; defer AVG denominator fixup O(groups)
                        if let (RgColSlice::Count, RgColSlice::F64(vals)) = (&rg_slices[0], &rg_slices[1]) {
                            let n = rg_n.min(vals.len());
                            for i in 0..n {
                                let gid = unsafe { *gids_slice.get_unchecked(i) } as usize;
                                let base = gid * 2;
                                unsafe { *flat_counts.get_unchecked_mut(base) += 1; *flat_sums.get_unchecked_mut(base + 1) += *vals.get_unchecked(i); }
                            }
                            for g in 0..num_groups { flat_counts[g * 2 + 1] = flat_counts[g * 2]; }
                            rg_row_offset += rg_rows; continue;
                        }
                        if let (RgColSlice::Count, RgColSlice::I64(vals)) = (&rg_slices[0], &rg_slices[1]) {
                            let n = rg_n.min(vals.len());
                            for i in 0..n {
                                let gid = unsafe { *gids_slice.get_unchecked(i) } as usize;
                                let base = gid * 2;
                                unsafe { *flat_counts.get_unchecked_mut(base) += 1; *flat_sums.get_unchecked_mut(base + 1) += *vals.get_unchecked(i) as f64; }
                            }
                            for g in 0..num_groups { flat_counts[g * 2 + 1] = flat_counts[g * 2]; }
                            rg_row_offset += rg_rows; continue;
                        }
                    }
                    // General multi-agg path
                    let n_limit: usize = rg_slices.iter().map(|sl| match sl { RgColSlice::F64(v) => v.len(), RgColSlice::I64(v) => v.len(), RgColSlice::Count => usize::MAX }).min().unwrap_or(rg_n).min(rg_n);
                    for i in 0..n_limit {
                        let gid = unsafe { *gids_slice.get_unchecked(i) } as usize;
                        let base = gid * num_aggs;
                        for (ai, sl) in rg_slices.iter().enumerate() {
                            unsafe { *flat_counts.get_unchecked_mut(base + ai) += 1; }
                            match sl {
                                RgColSlice::Count => {}
                                RgColSlice::F64(v) => { unsafe { *flat_sums.get_unchecked_mut(base + ai) += *v.get_unchecked(i); } }
                                RgColSlice::I64(v) => { unsafe { *flat_sums.get_unchecked_mut(base + ai) += *v.get_unchecked(i) as f64; } }
                            }
                        }
                    }
                    rg_row_offset += rg_rows; continue;
                }
                rg_row_offset += rg_rows;
            }

            // Handle multi-agg or any RGs that fell through: collect remaining via old path
            // (most common case handled above; this is just a safety fallback)
            let results: Vec<(String, Vec<(f64, i64)>)> = (0..num_groups)
                .filter(|&gid| flat_counts[gid * num_aggs] > 0 || agg_col_indices.iter().all(|x| x.is_none()))
                .filter(|&gid| flat_counts[gid] > 0)
                .map(|gid| {
                    let aggs: Vec<(f64, i64)> = (0..num_aggs).map(|ai| (flat_sums[gid * num_aggs + ai], flat_counts[gid * num_aggs + ai])).collect();
                    (dict_strings[gid].clone(), aggs)
                })
                .collect();
            return Ok(Some(results));
        }

        // FALLBACK: full materialization via scan_columns_mmap (compressed or no RCIX)
        let (scanned, del_bytes) = if needed.is_empty() {
            (Vec::new(), Vec::new())
        } else {
            self.scan_columns_mmap(&needed, &footer)?
        };
        let has_deleted = del_bytes.iter().any(|&b| b != 0);
        
        // Pre-resolve column data slices — resolve HashMap once, not per row
        let needed_to_pos: Vec<(usize, usize)> = needed.iter().enumerate()
            .map(|(pos, &idx)| (idx, pos))
            .collect();
        
        struct AggSlice<'a> { i64_vals: Option<&'a [i64]>, f64_vals: Option<&'a [f64]>, is_count: bool }
        let agg_slices: Vec<AggSlice> = agg_col_indices.iter().map(|opt_idx| {
            if opt_idx.is_none() { return AggSlice { i64_vals: None, f64_vals: None, is_count: true }; }
            let col_idx = opt_idx.unwrap();
            let pos = needed_to_pos.iter().find(|&&(idx, _)| idx == col_idx).map(|&(_, p)| p);
            if let Some(pos) = pos {
                if pos < scanned.len() {
                    match &scanned[pos] {
                        ColumnData::Int64(v)   => return AggSlice { i64_vals: Some(v.as_slice()), f64_vals: None, is_count: false },
                        ColumnData::Float64(v) => return AggSlice { i64_vals: None, f64_vals: Some(v.as_slice()), is_count: false },
                        _ => {}
                    }
                }
            }
            AggSlice { i64_vals: None, f64_vals: None, is_count: true }
        }).collect();
        
        let num_groups = dict_strings.len();
        let num_aggs = agg_cols.len();
        let scan_rows = group_ids.len();
        let flat_len = num_groups * num_aggs;
        let mut flat_sums = vec![0.0f64; flat_len];
        let mut flat_counts = vec![0i64; flat_len];
        
        // Specialized fast path for single agg column (most common) — no inner loop
        if num_aggs == 1 && !has_deleted {
            match &agg_slices[0] {
                AggSlice { f64_vals: Some(vals), .. } => {
                    let limit = scan_rows.min(vals.len());
                    for i in 0..limit {
                        let gid = unsafe { *group_ids.get_unchecked(i) } as usize;
                        unsafe { *flat_counts.get_unchecked_mut(gid) += 1; }
                        unsafe { *flat_sums.get_unchecked_mut(gid) += *vals.get_unchecked(i); }
                    }
                }
                AggSlice { i64_vals: Some(vals), .. } => {
                    let limit = scan_rows.min(vals.len());
                    for i in 0..limit {
                        let gid = unsafe { *group_ids.get_unchecked(i) } as usize;
                        unsafe { *flat_counts.get_unchecked_mut(gid) += 1; }
                        unsafe { *flat_sums.get_unchecked_mut(gid) += *vals.get_unchecked(i) as f64; }
                    }
                }
                _ => {
                    for i in 0..scan_rows {
                        let gid = unsafe { *group_ids.get_unchecked(i) } as usize;
                        unsafe { *flat_counts.get_unchecked_mut(gid) += 1; }
                    }
                }
            }
        } else {
            for i in 0..scan_rows {
                if has_deleted {
                    let b = i / 8; let bit = i % 8;
                    if b < del_bytes.len() && (del_bytes[b] >> bit) & 1 != 0 { continue; }
                }
                let base = group_ids[i] as usize * num_aggs;
                if base + num_aggs > flat_len { continue; }
                for (ai, agg) in agg_slices.iter().enumerate() {
                    unsafe { *flat_counts.get_unchecked_mut(base + ai) += 1; }
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

        // Check RCIX availability
        let max_col_idx = [Some(filter_idx), agg_idx].iter().filter_map(|&x| x).max().unwrap_or(0);
        let all_rcix = footer.row_groups.iter().enumerate().all(|(rg_i, rg_meta)| {
            if rg_meta.row_count == 0 { return true; }
            footer.col_offsets.get(rg_i).map_or(false, |v| v.len() > max_col_idx)
        });

        let num_groups = dict_strings.len();
        let mut group_sums = vec![0.0f64; num_groups];
        let mut group_counts = vec![0i64; num_groups];
        let lo_i64 = lo as i64;
        let hi_i64 = hi as i64;

        if all_rcix {
            // STREAMING: process per-RG without materializing full columns
            let file_guard = self.file.read();
            let file = file_guard.as_ref().ok_or_else(|| err_not_conn("File not open"))?;
            let mut mmap_guard = self.mmap_cache.write();
            let mmap_ref = mmap_guard.get_or_create(file)?;
            let mut rg_row_offset = 0usize;

            for (rg_i, rg_meta) in footer.row_groups.iter().enumerate() {
                let rg_rows = rg_meta.row_count as usize;
                if rg_rows == 0 { rg_row_offset += rg_rows; continue; }
                let rg_end = (rg_meta.offset + rg_meta.data_size) as usize;
                if rg_end > mmap_ref.len() { return Err(crate::storage::on_demand::err_data("RG past EOF")); }
                let rg_bytes = &mmap_ref[rg_meta.offset as usize..rg_end];
                if rg_bytes.len() < 32 { rg_row_offset += rg_rows; continue; }
                let compress_flag = rg_bytes[28];
                let encoding_version = rg_bytes[29];
                let null_bitmap_len = (rg_rows + 7) / 8;
                let rcix = &footer.col_offsets[rg_i];
                let has_deleted = rg_meta.deletion_count > 0;
                let del_start = rg_rows * 8;
                let del_vec_len = null_bitmap_len;

                // Decompress if needed
                let decompressed_buf = decompress_rg_body(compress_flag, &rg_bytes[32..])?;
                let body: &[u8] = decompressed_buf.as_deref().unwrap_or(&rg_bytes[32..]);
                let del_bytes: &[u8] = if del_start + del_vec_len <= body.len() {
                    &body[del_start..del_start + del_vec_len]
                } else { &[] };

                let gids_slice = &group_ids[rg_row_offset..(rg_row_offset + rg_rows).min(group_ids.len())];
                let rg_n = gids_slice.len();

                // Get filter column data — zero-copy for BITPACK, full decode for others
                let f_col_off = rcix[filter_idx] as usize;
                let f_data_start = f_col_off + null_bitmap_len;
                if f_data_start >= body.len() { rg_row_offset += rg_rows; continue; }
                let f_bytes = &body[f_data_start..];
                let f_encoding = if encoding_version >= 1 && !f_bytes.is_empty() { f_bytes[0] } else { 0 };
                // Check for BITPACK filter: inline decode to avoid 500KB Vec<i64> per RG
                let filter_is_bitpack = f_encoding == 2u8 && f_bytes.len() >= 18; // 1 enc + 17 header
                let (bp_count, bp_bit_width, bp_min_val, bp_packed): (usize, usize, i64, &[u8]) = if filter_is_bitpack {
                    let d = &f_bytes[1..];
                    let cnt = u64::from_le_bytes(d[0..8].try_into().unwrap()) as usize;
                    let bw = d[8] as usize;
                    let mv = i64::from_le_bytes(d[9..17].try_into().unwrap());
                    let pb = (cnt * bw + 7) / 8;
                    let packed = if 17 + pb <= d.len() { &d[17..17+pb] } else { &[] as &[u8] };
                    (cnt, bw, mv, packed)
                } else { (0, 0, 0, &[]) };
                let filter_data_owned: Option<ColumnData> = if filter_is_bitpack { None } else {
                    Some(if encoding_version >= 1 {
                        read_column_encoded(f_bytes, schema.columns[filter_idx].1)?.0
                    } else {
                        ColumnData::from_bytes_typed(f_bytes, schema.columns[filter_idx].1)?.0
                    })
                };

                // Get agg column via zero-copy PLAIN slice or fallback decode
                enum AggBuf { None, F64ZC(*const f64, usize), I64ZC(*const i64, usize), Owned(ColumnData) }
                let agg_buf: AggBuf = match agg_idx {
                    None => AggBuf::None,
                    Some(ai) if ai == filter_idx => AggBuf::None,
                    Some(ai) => {
                        let a_col_off = rcix[ai] as usize;
                        let a_data_start = a_col_off + null_bitmap_len;
                        if a_data_start + 1 < body.len() && encoding_version >= 1 {
                            let enc = body[a_data_start];
                            let payload = &body[a_data_start + 1..];
                            if enc == 0u8 && payload.len() >= 8 {
                                let count = u64::from_le_bytes(payload[0..8].try_into().unwrap()) as usize;
                                let col_type = schema.columns[ai].1;
                                if matches!(col_type, ColumnType::Float64 | ColumnType::Float32) {
                                    let n = count.min(rg_rows).min((payload.len() - 8) / 8);
                                    AggBuf::F64ZC(payload[8..].as_ptr() as *const f64, n)
                                } else if matches!(col_type, ColumnType::Int64 | ColumnType::Int8 | ColumnType::Int16 | ColumnType::Int32 | ColumnType::UInt8 | ColumnType::UInt16 | ColumnType::UInt32 | ColumnType::UInt64) {
                                    let n = count.min(rg_rows).min((payload.len() - 8) / 8);
                                    AggBuf::I64ZC(payload[8..].as_ptr() as *const i64, n)
                                } else {
                                    let (ad, _) = read_column_encoded(&body[a_data_start..], col_type)?;
                                    AggBuf::Owned(ad)
                                }
                            } else {
                                let (ad, _) = read_column_encoded(&body[a_data_start..], schema.columns[ai].1)?;
                                AggBuf::Owned(ad)
                            }
                        } else if a_data_start < body.len() {
                            let (ad, _) = ColumnData::from_bytes_typed(&body[a_data_start..], schema.columns[ai].1)?;
                            AggBuf::Owned(ad)
                        } else { AggBuf::None }
                    }
                };
                // Expose as slices (zero-copy or owned)
                let (agg_f64_zc_ptr, agg_f64_zc_n): (Option<*const f64>, usize) = if let AggBuf::F64ZC(p, n) = &agg_buf { (Some(*p), *n) } else { (None, 0) };
                let (agg_i64_zc_ptr, agg_i64_zc_n): (Option<*const i64>, usize) = if let AggBuf::I64ZC(p, n) = &agg_buf { (Some(*p), *n) } else { (None, 0) };
                let agg_data_owned = if let AggBuf::Owned(ref cd) = agg_buf { Some(cd) } else { None };
                let agg_f64_own = agg_data_owned.and_then(|acd| match acd { ColumnData::Float64(v) => Some(v.as_slice()), _ => None });
                let agg_i64_own = agg_data_owned.and_then(|acd| match acd { ColumnData::Int64(v) => Some(v.as_slice()), _ => None });
                let agg_f64: Option<&[f64]> = if let Some(p) = agg_f64_zc_ptr {
                    Some(unsafe { std::slice::from_raw_parts(p, agg_f64_zc_n) })
                } else { agg_f64_own };
                let agg_i64: Option<&[i64]> = if let Some(p) = agg_i64_zc_ptr {
                    Some(unsafe { std::slice::from_raw_parts(p, agg_i64_zc_n) })
                } else { agg_i64_own };
                let (agg_f64, agg_i64) = if agg_idx == Some(filter_idx) {
                    // Same column as filter — derive from decoded data (non-BITPACK fallback)
                    match filter_data_owned.as_ref() {
                        Some(ColumnData::Float64(v)) => (Some(v.as_slice()), None::<&[i64]>),
                        Some(ColumnData::Int64(v)) => (None, Some(v.as_slice())),
                        _ => (agg_f64, agg_i64)
                    }
                } else { (agg_f64, agg_i64) };

                macro_rules! between_rg {
                    ($fvals:expr, $lo_c:expr, $hi_c:expr) => {{
                        let limit = rg_n.min($fvals.len());
                        if !has_deleted {
                            if let Some(av) = agg_f64 {
                                let limit = limit.min(av.len());
                                for i in 0..limit {
                                    let fv = unsafe { *$fvals.get_unchecked(i) };
                                    let mask = (fv >= $lo_c && fv <= $hi_c) as i64;
                                    let gid = unsafe { *gids_slice.get_unchecked(i) } as usize;
                                    unsafe { *group_counts.get_unchecked_mut(gid) += mask; *group_sums.get_unchecked_mut(gid) += mask as f64 * *av.get_unchecked(i); }
                                }
                            } else if let Some(av) = agg_i64 {
                                let limit = limit.min(av.len());
                                for i in 0..limit {
                                    let fv = unsafe { *$fvals.get_unchecked(i) };
                                    let mask = (fv >= $lo_c && fv <= $hi_c) as i64;
                                    let gid = unsafe { *gids_slice.get_unchecked(i) } as usize;
                                    unsafe { *group_counts.get_unchecked_mut(gid) += mask; *group_sums.get_unchecked_mut(gid) += mask as f64 * (*av.get_unchecked(i) as f64); }
                                }
                            } else {
                                for i in 0..limit {
                                    let fv = unsafe { *$fvals.get_unchecked(i) };
                                    let mask = (fv >= $lo_c && fv <= $hi_c) as i64;
                                    let gid = unsafe { *gids_slice.get_unchecked(i) } as usize;
                                    unsafe { *group_counts.get_unchecked_mut(gid) += mask; }
                                }
                            }
                        } else {
                            for i in 0..limit {
                                if !del_bytes.is_empty() && (del_bytes[i/8] >> (i%8)) & 1 != 0 { continue; }
                                let fv = unsafe { *$fvals.get_unchecked(i) };
                                if fv >= $lo_c && fv <= $hi_c {
                                    let gid = unsafe { *gids_slice.get_unchecked(i) } as usize;
                                    if gid < num_groups {
                                        unsafe { *group_counts.get_unchecked_mut(gid) += 1; }
                                        if let Some(av) = agg_f64 { unsafe { *group_sums.get_unchecked_mut(gid) += *av.get_unchecked(i); } }
                                        else if let Some(av) = agg_i64 { unsafe { *group_sums.get_unchecked_mut(gid) += *av.get_unchecked(i) as f64; } }
                                    }
                                }
                            }
                        }
                    }};
                }

                if filter_is_bitpack && !bp_packed.is_empty() {
                    // BITPACK zero-copy: 64-bit wide unaligned reads — no Vec allocation
                    let delta_lo = (lo_i64.saturating_sub(bp_min_val)).max(0) as u64;
                    let delta_hi = if hi_i64 >= bp_min_val { (hi_i64 - bp_min_val) as u64 } else { u64::MAX };
                    let bp_mask = if bp_bit_width == 0 { 0u64 } else { (1u64 << bp_bit_width) - 1 };
                    let n = bp_count.min(rg_n);
                    // Inline extractor: reads 8-byte window, shifts, masks
                    macro_rules! bp_delta { ($i:expr) => {{
                        let bit_off = $i * bp_bit_width;
                        let byte_idx = bit_off / 8;
                        let bit_shift = bit_off % 8;
                        let raw = if byte_idx + 8 <= bp_packed.len() {
                            unsafe { (bp_packed.as_ptr().add(byte_idx) as *const u64).read_unaligned() }
                        } else {
                            let mut buf = [0u8; 8];
                            let avail = bp_packed.len().saturating_sub(byte_idx);
                            unsafe { std::ptr::copy_nonoverlapping(bp_packed.as_ptr().add(byte_idx), buf.as_mut_ptr(), avail); }
                            u64::from_le_bytes(buf)
                        };
                        (raw >> bit_shift) & bp_mask
                    }}}
                    if !has_deleted {
                        if let Some(av) = agg_f64 {
                            let n = n.min(av.len());
                            if bp_bit_width == 0 {
                                let passes = (bp_min_val >= lo_i64 && bp_min_val <= hi_i64) as i64;
                                for i in 0..n { let gid = unsafe { *gids_slice.get_unchecked(i) } as usize; unsafe { *group_counts.get_unchecked_mut(gid) += passes; *group_sums.get_unchecked_mut(gid) += passes as f64 * *av.get_unchecked(i); } }
                            } else if bp_bit_width == 6 {
                                // Specialized 6-bit path: 8 values per 6-byte (48-bit) chunk, fully inlined
                                macro_rules! do_scatter6 { ($w:expr, $sh:expr, $ii:expr) => {{
                                    let d = ($w >> $sh) & 63u64;
                                    let mask = (d >= delta_lo && d <= delta_hi) as i64;
                                    let gid = unsafe { *gids_slice.get_unchecked($ii) } as usize;
                                    unsafe { *group_counts.get_unchecked_mut(gid) += mask; *group_sums.get_unchecked_mut(gid) += mask as f64 * *av.get_unchecked($ii); }
                                }}}
                                let chunks = n / 8;
                                for c in 0..chunks {
                                    let base_byte = c * 6;
                                    let base_i = c * 8;
                                    let word = unsafe { (bp_packed.as_ptr().add(base_byte) as *const u64).read_unaligned() };
                                    do_scatter6!(word,  0, base_i+0);
                                    do_scatter6!(word,  6, base_i+1);
                                    do_scatter6!(word, 12, base_i+2);
                                    do_scatter6!(word, 18, base_i+3);
                                    do_scatter6!(word, 24, base_i+4);
                                    do_scatter6!(word, 30, base_i+5);
                                    do_scatter6!(word, 36, base_i+6);
                                    do_scatter6!(word, 42, base_i+7);
                                }
                                for i in (chunks * 8)..n {
                                    let delta = bp_delta!(i);
                                    let mask = (delta >= delta_lo && delta <= delta_hi) as i64;
                                    let gid = unsafe { *gids_slice.get_unchecked(i) } as usize;
                                    unsafe { *group_counts.get_unchecked_mut(gid) += mask; *group_sums.get_unchecked_mut(gid) += mask as f64 * *av.get_unchecked(i); }
                                }
                            } else {
                                for i in 0..n {
                                    let delta = bp_delta!(i);
                                    let mask = (delta >= delta_lo && delta <= delta_hi) as i64;
                                    let gid = unsafe { *gids_slice.get_unchecked(i) } as usize;
                                    unsafe { *group_counts.get_unchecked_mut(gid) += mask; *group_sums.get_unchecked_mut(gid) += mask as f64 * *av.get_unchecked(i); }
                                }
                            }
                        } else if let Some(av) = agg_i64 {
                            let n = n.min(av.len());
                            for i in 0..n {
                                let delta = if bp_bit_width == 0 { 0 } else { bp_delta!(i) };
                                let mask = (bp_bit_width == 0 || (delta >= delta_lo && delta <= delta_hi)) as i64;
                                let gid = unsafe { *gids_slice.get_unchecked(i) } as usize;
                                unsafe { *group_counts.get_unchecked_mut(gid) += mask; *group_sums.get_unchecked_mut(gid) += mask as f64 * *av.get_unchecked(i) as f64; }
                            }
                        } else if agg_idx == Some(filter_idx) {
                            // agg col == filter col (BITPACK): sum the filtered BITPACK values inline
                            for i in 0..n {
                                let delta = if bp_bit_width == 0 { 0u64 } else { bp_delta!(i) };
                                let mask = (bp_bit_width == 0 || (delta >= delta_lo && delta <= delta_hi)) as i64;
                                let gid = unsafe { *gids_slice.get_unchecked(i) } as usize;
                                let val = bp_min_val + delta as i64;
                                unsafe { *group_counts.get_unchecked_mut(gid) += mask; *group_sums.get_unchecked_mut(gid) += mask as f64 * val as f64; }
                            }
                        } else {
                            for i in 0..n {
                                let delta = if bp_bit_width == 0 { 0u64 } else { bp_delta!(i) };
                                let mask = (bp_bit_width == 0 || (delta >= delta_lo && delta <= delta_hi)) as i64;
                                let gid = unsafe { *gids_slice.get_unchecked(i) } as usize;
                                unsafe { *group_counts.get_unchecked_mut(gid) += mask; }
                            }
                        }
                    }
                } else if let Some(ref filter_data) = filter_data_owned {
                    match filter_data {
                        ColumnData::Int64(vals) => { between_rg!(vals, lo_i64, hi_i64); }
                        ColumnData::Float64(vals) => { between_rg!(vals, lo, hi); }
                        _ => {}
                    }
                }
                rg_row_offset += rg_rows;
            }

            let results: Vec<(String, f64, i64)> = (0..num_groups)
                .filter(|&gid| group_counts[gid] > 0)
                .map(|gid| (dict_strings[gid].clone(), group_sums[gid], group_counts[gid]))
                .collect();
            return Ok(Some(results));
        }

        // FALLBACK: full materialization (no RCIX or old format)
        let mut needed: Vec<usize> = vec![filter_idx];
        if let Some(ai) = agg_idx { if ai != filter_idx { needed.push(ai); } }
        let (scanned, del_bytes) = self.scan_columns_mmap(&needed, &footer)?;
        let has_deleted = del_bytes.iter().any(|&b| b != 0);

        let filter_col_data = &scanned[0];
        let agg_col_data = agg_idx.map(|ai| {
            if ai == filter_idx { &scanned[0] } else { &scanned[1] }
        });
        
        let scan_rows = group_ids.len();
        
        // Pre-resolve agg data slice — avoids match/option check inside hot loop
        let agg_f64 = agg_col_data.and_then(|acd| match acd { ColumnData::Float64(v) => Some(v.as_slice()), _ => None });
        let agg_i64 = agg_col_data.and_then(|acd| match acd { ColumnData::Int64(v) => Some(v.as_slice()), _ => None });
        
        macro_rules! between_hot {
            ($fvals:expr, $lo_c:expr, $hi_c:expr) => {{
                let limit = scan_rows.min($fvals.len()).min(group_ids.len());
                if !has_deleted {
                    if let Some(av) = agg_f64 {
                        let limit = limit.min(av.len());
                        for i in 0..limit {
                            let fv = unsafe { *$fvals.get_unchecked(i) };
                            let mask = (fv >= $lo_c && fv <= $hi_c) as i64;
                            let gid = unsafe { *group_ids.get_unchecked(i) } as usize;
                            unsafe {
                                *group_counts.get_unchecked_mut(gid) += mask;
                                *group_sums.get_unchecked_mut(gid) += mask as f64 * *av.get_unchecked(i);
                            }
                        }
                    } else if let Some(av) = agg_i64 {
                        let limit = limit.min(av.len());
                        for i in 0..limit {
                            let fv = unsafe { *$fvals.get_unchecked(i) };
                            let mask = (fv >= $lo_c && fv <= $hi_c) as i64;
                            let gid = unsafe { *group_ids.get_unchecked(i) } as usize;
                            unsafe {
                                *group_counts.get_unchecked_mut(gid) += mask;
                                *group_sums.get_unchecked_mut(gid) += mask as f64 * (*av.get_unchecked(i) as f64);
                            }
                        }
                    } else {
                        for i in 0..limit {
                            let fv = unsafe { *$fvals.get_unchecked(i) };
                            let mask = (fv >= $lo_c && fv <= $hi_c) as i64;
                            let gid = unsafe { *group_ids.get_unchecked(i) } as usize;
                            unsafe { *group_counts.get_unchecked_mut(gid) += mask; }
                        }
                    }
                } else {
                    for i in 0..limit {
                        let b = i / 8; let bit = i % 8;
                        if b < del_bytes.len() && (del_bytes[b] >> bit) & 1 != 0 { continue; }
                        let fv = unsafe { *$fvals.get_unchecked(i) };
                        if fv >= $lo_c && fv <= $hi_c {
                            let gid = unsafe { *group_ids.get_unchecked(i) } as usize;
                            if gid < num_groups {
                                unsafe { *group_counts.get_unchecked_mut(gid) += 1; }
                                if let Some(av) = agg_f64 { unsafe { *group_sums.get_unchecked_mut(gid) += *av.get_unchecked(i); } }
                                else if let Some(av) = agg_i64 { unsafe { *group_sums.get_unchecked_mut(gid) += *av.get_unchecked(i) as f64; } }
                            }
                        }
                    }
                }
            }};
        }
        
        match filter_col_data {
            ColumnData::Int64(vals) => { between_hot!(vals, lo_i64, hi_i64); }
            ColumnData::Float64(vals) => { between_hot!(vals, lo, hi); }
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

    fn try_read_col_stats_sidecar(&self) -> Option<std::collections::HashMap<String, (i64, f64, f64, f64, bool)>> {
        let sidecar_path = std::path::PathBuf::from(format!("{}.stats", self.path.display()));
        let data = std::fs::read(&sidecar_path).ok()?;
        if data.len() < 12 || &data[..8] != b"APEXSTAT" { return None; }
        let num_cols = u32::from_le_bytes(data[8..12].try_into().ok()?) as usize;
        let mut pos = 12usize;
        let mut result = std::collections::HashMap::with_capacity(num_cols);
        for _ in 0..num_cols {
            if pos + 2 > data.len() { return None; }
            let name_len = u16::from_le_bytes(data[pos..pos+2].try_into().ok()?) as usize;
            pos += 2;
            if pos + name_len + 33 > data.len() { return None; }
            let name = std::str::from_utf8(&data[pos..pos+name_len]).ok()?.to_string();
            pos += name_len;
            let count = i64::from_le_bytes(data[pos..pos+8].try_into().ok()?); pos += 8;
            let sum = f64::from_bits(u64::from_le_bytes(data[pos..pos+8].try_into().ok()?)); pos += 8;
            let min = f64::from_bits(u64::from_le_bytes(data[pos..pos+8].try_into().ok()?)); pos += 8;
            let max = f64::from_bits(u64::from_le_bytes(data[pos..pos+8].try_into().ok()?)); pos += 8;
            let is_int = data[pos] != 0; pos += 1;
            result.insert(name, (count, sum, min, max, is_int));
        }
        Some(result)
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


