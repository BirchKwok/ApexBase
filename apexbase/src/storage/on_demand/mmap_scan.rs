// Mmap scanning, footer loading, column range readers, filter+group+order fast path

impl OnDemandStorage {
    /// Get the V4 footer, using cached version when file hasn't changed.
    /// Returns None for V3 files.
    /// Cache is invalidated when file size changes (another instance appended data)
    /// or explicitly via `invalidate_footer_cache()` after writes.
    pub(crate) fn get_or_load_footer(&self) -> io::Result<Option<V4Footer>> {
        let file_len = std::fs::metadata(&self.path)
            .map(|m| m.len())
            .unwrap_or(0);
        if file_len < HEADER_SIZE_V3 as u64 {
            return Ok(None);
        }

        // Fast path: return cached footer if file size hasn't changed
        {
            let cached = self.v4_footer.read();
            if let Some(ref footer) = *cached {
                let mc = self.mmap_cache.read();
                if mc.file_size == file_len {
                    return Ok(Some(footer.clone()));
                }
            }
        }

        // Invalidate mmap if file size changed (another instance appended data)
        {
            let mut mc = self.mmap_cache.write();
            if mc.file_size != 0 && mc.file_size != file_len {
                mc.invalidate();
            }
        }

        // Ensure file handle is open (reopen if needed after save_v4 replaced file)
        {
            let fg = self.file.read();
            if fg.is_none() {
                drop(fg);
                if let Ok(f) = File::open(&self.path) {
                    *self.file.write() = Some(f);
                }
            }
        }

        let file_guard = self.file.read();
        let file_handle = match file_guard.as_ref() {
            Some(f) => f,
            None => return Ok(None),
        };
        let mut mmap = self.mmap_cache.write();

        // Read the on-disk header fresh to get current footer_offset
        let mut header_bytes = [0u8; HEADER_SIZE_V3];
        mmap.read_at(file_handle, &mut header_bytes, 0)?;
        let on_disk_header = OnDemandHeader::from_bytes(&header_bytes)?;

        if on_disk_header.footer_offset == 0 || on_disk_header.version != FORMAT_VERSION_V4 {
            return Ok(None);
        }
        let footer_offset = on_disk_header.footer_offset;
        if footer_offset >= file_len {
            return Ok(None);
        }

        let footer_byte_count = (file_len - footer_offset) as usize;
        let mut footer_bytes = vec![0u8; footer_byte_count];
        mmap.read_at(file_handle, &mut footer_bytes, footer_offset)?;
        drop(mmap);
        drop(file_guard);

        let footer = V4Footer::from_bytes(&footer_bytes)?;
        // Cache the footer for subsequent reads
        *self.v4_footer.write() = Some(footer.clone());
        Ok(Some(footer))
    }

    /// Invalidate the cached V4 footer (call after writes that change the footer).
    fn invalidate_footer_cache(&self) {
        *self.v4_footer.write() = None;
    }

    /// Read columns from on-disk V4 Row Groups directly via mmap → Arrow RecordBatch.
    /// Only materializes the requested columns; skips others with zero allocation.
    /// This is the core on-demand reading function that avoids loading all data into memory.
    ///
    /// # Arguments
    /// * `column_names` - Which columns to read (None = all)
    /// * `include_id` - Whether to include the _id column
    /// * `row_limit` - Maximum number of active rows to return (None = all)
    /// * `dict_encode_strings` - Whether to produce DictionaryArray for low-cardinality strings
    pub fn to_arrow_batch_mmap(
        &self,
        column_names: Option<&[&str]>,
        include_id: bool,
        row_limit: Option<usize>,
        dict_encode_strings: bool,
    ) -> io::Result<Option<RecordBatch>> {
        use arrow::array::{Int64Array, Float64Array, StringArray, BooleanArray, PrimitiveArray};
        use arrow::buffer::{Buffer, NullBuffer, BooleanBuffer, ScalarBuffer};
        use arrow::datatypes::{Schema, Field, DataType as ArrowDataType, Int64Type, Float64Type};
        use std::sync::Arc;

        let footer = match self.get_or_load_footer()? {
            Some(f) => f,
            None => return Ok(None), // V3 file — caller should use legacy path
        };

        let schema = &footer.schema;
        let col_count = schema.column_count();

        // Determine which columns to read (indices into schema)
        let col_indices: Vec<usize> = if let Some(names) = column_names {
            names.iter()
                .filter(|&&n| n != "_id")
                .filter_map(|&name| schema.get_index(name))
                .collect()
        } else {
            (0..col_count).collect()
        };

        // Compute total active rows across all RGs
        let total_active: usize = footer.row_groups.iter()
            .map(|rg| rg.active_rows() as usize)
            .sum();

        if total_active == 0 {
            // Build empty schema and return empty batch
            let mut fields: Vec<Field> = Vec::new();
            if include_id {
                fields.push(Field::new("_id", ArrowDataType::Int64, false));
            }
            for &ci in &col_indices {
                let (name, ct) = &schema.columns[ci];
                let dt = match ct {
                    ColumnType::Int64 | ColumnType::Int8 | ColumnType::Int16 |
                    ColumnType::Int32 | ColumnType::UInt8 | ColumnType::UInt16 |
                    ColumnType::UInt32 | ColumnType::UInt64 => ArrowDataType::Int64,
                    ColumnType::Float64 | ColumnType::Float32 => ArrowDataType::Float64,
                    ColumnType::Bool => ArrowDataType::Boolean,
                    _ => ArrowDataType::Utf8,
                };
                fields.push(Field::new(name, dt, true));
            }
            let arrow_schema = Arc::new(Schema::new(fields));
            return Ok(Some(RecordBatch::new_empty(arrow_schema)));
        }

        let effective_limit = row_limit.unwrap_or(total_active).min(total_active);

        // Get mmap for the file
        let file_guard = self.file.read();
        let file = file_guard.as_ref()
            .ok_or_else(|| err_not_conn("File not open for mmap read"))?;
        let mut mmap_guard = self.mmap_cache.write();
        let mmap_ref = mmap_guard.get_or_create(file)?;

        // Accumulators for each output column + _id
        let mut all_ids: Vec<i64> = Vec::with_capacity(effective_limit);
        // For each requested column, accumulate ColumnData across RGs
        let mut col_accumulators: Vec<ColumnData> = col_indices.iter()
            .map(|&ci| {
                let ct = schema.columns[ci].1;
                // StringDict is decoded to String before accumulation,
                // so accumulator must be String type
                let acc_type = if ct == ColumnType::StringDict { ColumnType::String } else { ct };
                ColumnData::new(acc_type)
            })
            .collect();
        let mut null_accumulators: Vec<Vec<bool>> = vec![Vec::new(); col_indices.len()];
        let mut rows_collected: usize = 0;

        for (rg_idx, rg_meta) in footer.row_groups.iter().enumerate() {
            if rows_collected >= effective_limit {
                break;
            }
            if rg_meta.row_count == 0 {
                continue;
            }

            let rg_rows = rg_meta.row_count as usize;
            let rg_end = (rg_meta.offset + rg_meta.data_size) as usize;
            if rg_end > mmap_ref.len() {
                return Err(err_data("RG extends past EOF"));
            }
            let rg_bytes = &mmap_ref[rg_meta.offset as usize .. rg_end];

            // Check compression flag at RG header byte 28
            let compress_flag = if rg_bytes.len() >= 32 { rg_bytes[28] } else { RG_COMPRESS_NONE };
            let encoding_version = if rg_bytes.len() >= 32 { rg_bytes[29] } else { 0 };
            let has_deletes = rg_meta.deletion_count > 0;
            let rg_active = rg_meta.active_rows() as usize;
            let rows_to_take = (effective_limit - rows_collected).min(rg_active);
            let null_bitmap_len = (rg_rows + 7) / 8;

            // === RCIX fast path: O(1) direct seeks for no-compression, no-deletes ===
            // Skips sequential column scanning — jumps directly to each column via footer index.
            // For LIMIT 100 with 65536-row RG: touches ~800B of IDs + targeted column pages
            // instead of scanning 512KB IDs + full column sequence.
            if compress_flag == RG_COMPRESS_NONE
                && !has_deletes
                && encoding_version >= 1
                && rg_idx < footer.col_offsets.len()
                && !footer.col_offsets[rg_idx].is_empty()
            {
                let rg_body_abs = (rg_meta.offset + 32) as usize;
                let col_offsets = &footer.col_offsets[rg_idx];

                // Read only first rows_to_take IDs directly from mmap (avoids touching rest)
                {
                    let id_end = rg_body_abs + rows_to_take * 8;
                    if id_end <= mmap_ref.len() {
                        let id_bytes = &mmap_ref[rg_body_abs..id_end];
                        for i in 0..rows_to_take {
                            let id = u64::from_le_bytes(
                                id_bytes[i * 8..(i + 1) * 8].try_into().unwrap()
                            );
                            all_ids.push(id as i64);
                        }
                    }
                }

                // Direct column reads via RCIX — no sequential scan of preceding columns
                for (out_pos, &col_idx) in col_indices.iter().enumerate() {
                    if col_idx >= col_offsets.len() {
                        // Schema evolution: column added after this RG was written
                        let col_type = schema.columns[col_idx].1;
                        let default_col = Self::create_default_column(col_type, rows_to_take);
                        col_accumulators[out_pos].append(&default_col);
                        null_accumulators[out_pos].extend(std::iter::repeat(true).take(rows_to_take));
                        continue;
                    }
                    let col_abs = rg_body_abs + col_offsets[col_idx] as usize;
                    if col_abs + null_bitmap_len > mmap_ref.len() {
                        continue;
                    }
                    let null_bytes = &mmap_ref[col_abs..col_abs + null_bitmap_len];
                    let data_abs = col_abs + null_bitmap_len;
                    if data_abs >= mmap_ref.len() {
                        continue;
                    }
                    let col_type = schema.columns[col_idx].1;
                    let (col_data, _) = if rows_to_take < rg_rows {
                        read_column_encoded_partial(&mmap_ref[data_abs..], col_type, rows_to_take)?
                    } else {
                        read_column_encoded(&mmap_ref[data_abs..], col_type)?
                    };
                    let col_data = if matches!(&col_data, ColumnData::StringDict { .. }) {
                        col_data.decode_string_dict()
                    } else {
                        col_data
                    };
                    col_accumulators[out_pos].append(&col_data);
                    for i in 0..rows_to_take {
                        null_accumulators[out_pos].push((null_bytes[i / 8] >> (i % 8)) & 1 == 1);
                    }
                }

                rows_collected += rows_to_take;
                continue; // skip sequential scan path below
            }
            // === End RCIX fast path ===

            // Get the body bytes (after 32-byte RG header), decompressing if needed
            let decompressed_buf = decompress_rg_body(compress_flag, &rg_bytes[32..])?;
            let body: &[u8] = decompressed_buf.as_deref().unwrap_or(&rg_bytes[32..]);
            let mut pos: usize = 0;

            // Read IDs
            let id_byte_len = rg_rows * 8;
            if pos + id_byte_len > body.len() {
                return Err(err_data("RG IDs truncated"));
            }
            let id_slice = &body[pos..pos + id_byte_len];
            pos += id_byte_len;

            // Read deletion vector
            let del_vec_len = (rg_rows + 7) / 8;
            if pos + del_vec_len > body.len() {
                return Err(err_data("RG deletion vector truncated"));
            }
            let del_bytes = &body[pos..pos + del_vec_len];
            pos += del_vec_len;

            // Always collect active IDs from this RG (needed for DeltaMerger overlay)
            {
                let mut taken = 0;
                for i in 0..rg_rows {
                    if has_deletes && (del_bytes[i / 8] >> (i % 8)) & 1 == 1 {
                        continue; // deleted
                    }
                    let id = u64::from_le_bytes(
                        id_slice[i * 8..(i + 1) * 8].try_into().unwrap()
                    );
                    all_ids.push(id as i64);
                    taken += 1;
                    if taken >= rows_to_take { break; }
                }
            }

            // Parse columns — read requested, skip others
            // Build mapping: disk col_idx → output position in col_accumulators
            // This ensures correct data placement regardless of column ordering
            // between the footer schema and the requested column list.
            let col_idx_to_out: HashMap<usize, usize> = col_indices.iter()
                .enumerate()
                .map(|(out_pos, &col_idx)| (col_idx, out_pos))
                .collect();
            // Track which output columns got data from this RG
            let mut rg_filled: Vec<bool> = vec![false; col_indices.len()];
            for col_idx in 0..col_count {
                // Schema evolution: RG may have fewer columns than footer schema.
                // If we've exhausted the RG data, remaining columns get defaults.
                if pos + null_bitmap_len > body.len() {
                    break;
                }
                let null_bytes = &body[pos..pos + null_bitmap_len];
                pos += null_bitmap_len;

                let col_type = schema.columns[col_idx].1;

                if let Some(&out_pos) = col_idx_to_out.get(&col_idx) {
                    // OPTIMIZATION: For LIMIT queries without deletes, use partial column read
                    // to avoid allocating/copying full column data (e.g., 1M rows → only 100)
                    if !has_deletes && rows_to_take < rg_rows && encoding_version >= 1 {
                        let (col_data, consumed) = read_column_encoded_partial(&body[pos..], col_type, rows_to_take)?;
                        pos += consumed;
                        let col_data = if matches!(&col_data, ColumnData::StringDict { .. }) {
                            col_data.decode_string_dict()
                        } else {
                            col_data
                        };
                        col_accumulators[out_pos].append(&col_data);
                        for i in 0..rows_to_take {
                            let is_null = (null_bytes[i / 8] >> (i % 8)) & 1 == 1;
                            null_accumulators[out_pos].push(is_null);
                        }
                    } else {
                        // Full column read path
                        let (col_data, consumed) = if encoding_version >= 1 {
                            read_column_encoded(&body[pos..], col_type)?
                        } else {
                            ColumnData::from_bytes_typed(&body[pos..], col_type)?
                        };
                        pos += consumed;

                        let col_data = if matches!(&col_data, ColumnData::StringDict { .. }) {
                            col_data.decode_string_dict()
                        } else {
                            col_data
                        };

                        if has_deletes {
                            let active_indices: Vec<usize> = (0..rg_rows)
                                .filter(|&i| (del_bytes[i / 8] >> (i % 8)) & 1 == 0)
                                .take(rows_to_take)
                                .collect();
                            let filtered = col_data.filter_by_indices(&active_indices);
                            col_accumulators[out_pos].append(&filtered);

                            for &old_idx in &active_indices {
                                let ob = old_idx / 8;
                                let obit = old_idx % 8;
                                let is_null = ob < null_bytes.len() && (null_bytes[ob] >> obit) & 1 == 1;
                                null_accumulators[out_pos].push(is_null);
                            }
                        } else {
                            if rows_to_take < rg_rows {
                                let range_data = col_data.slice_range(0, rows_to_take);
                                col_accumulators[out_pos].append(&range_data);
                                for i in 0..rows_to_take {
                                    let is_null = (null_bytes[i / 8] >> (i % 8)) & 1 == 1;
                                    null_accumulators[out_pos].push(is_null);
                                }
                            } else {
                                col_accumulators[out_pos].append(&col_data);
                                for i in 0..rg_rows {
                                    let is_null = (null_bytes[i / 8] >> (i % 8)) & 1 == 1;
                                    null_accumulators[out_pos].push(is_null);
                                }
                            }
                        }
                    }
                    rg_filled[out_pos] = true;
                } else {
                    // Skip this column (no allocation, encoding-aware)
                    let consumed = if encoding_version >= 1 {
                        skip_column_encoded(&body[pos..], col_type)?
                    } else {
                        ColumnData::skip_bytes_typed(&body[pos..], col_type)?
                    };
                    pos += consumed;
                }
            }
            // Fill default values for columns that weren't in this RG (schema evolution)
            for (out_pos, filled) in rg_filled.iter().enumerate() {
                if !filled {
                    let col_type = schema.columns[col_indices[out_pos]].1;
                    let default_col = Self::create_default_column(col_type, rows_to_take);
                    col_accumulators[out_pos].append(&default_col);
                    // All rows are null for this missing column
                    null_accumulators[out_pos].extend(std::iter::repeat(true).take(rows_to_take));
                }
            }
            rows_collected += rows_to_take;
        }

        drop(mmap_guard);
        drop(file_guard);

        // Build Arrow RecordBatch from accumulated data
        let active_count = rows_collected;
        let mut fields: Vec<Field> = Vec::with_capacity(col_indices.len() + 1);
        let mut arrays: Vec<ArrayRef> = Vec::with_capacity(col_indices.len() + 1);

        // Save row IDs for potential DeltaMerger overlay
        let row_ids_for_delta: Vec<u64> = all_ids.iter().map(|&id| id as u64).collect();

        // _id column
        if include_id {
            fields.push(Field::new("_id", ArrowDataType::Int64, false));
            arrays.push(Arc::new(Int64Array::from(all_ids)));
        }

        // Data columns
        for (out_idx, &col_idx) in col_indices.iter().enumerate() {
            let (col_name, schema_col_type_ref) = &schema.columns[col_idx];
            let schema_col_type = *schema_col_type_ref;
            let col_data = &col_accumulators[out_idx];
            let null_bitmap = &null_accumulators[out_idx];

            // Build Arrow null buffer from per-bit bool accumulator
            let null_buf: Option<NullBuffer> = if !null_bitmap.is_empty() && null_bitmap.iter().any(|&b| b) {
                let mut validity_bytes = vec![0xFFu8; (active_count + 7) / 8];
                for (i, &is_null) in null_bitmap.iter().enumerate() {
                    if is_null {
                        // Clear validity bit (Arrow: 1=valid, 0=null)
                        validity_bytes[i / 8] &= !(1u8 << (i % 8));
                    }
                }
                Some(NullBuffer::new(BooleanBuffer::new(Buffer::from(validity_bytes), 0, active_count)))
            } else {
                None
            };

            let (arrow_dt, array): (ArrowDataType, ArrayRef) = match col_data {
                ColumnData::Int64(values) => {
                    match schema_col_type {
                        ColumnType::Timestamp => {
                            use arrow::datatypes::TimestampMicrosecondType;
                            let arr = PrimitiveArray::<TimestampMicrosecondType>::new(
                                ScalarBuffer::from(values.clone()), null_buf,
                            );
                            (ArrowDataType::Timestamp(arrow::datatypes::TimeUnit::Microsecond, None), Arc::new(arr) as ArrayRef)
                        }
                        ColumnType::Date => {
                            use arrow::datatypes::Date32Type;
                            let data_i32: Vec<i32> = values.iter().map(|&v| v as i32).collect();
                            let arr = PrimitiveArray::<Date32Type>::new(
                                ScalarBuffer::from(data_i32), null_buf,
                            );
                            (ArrowDataType::Date32, Arc::new(arr) as ArrayRef)
                        }
                        _ => {
                            let arr = PrimitiveArray::<Int64Type>::new(
                                ScalarBuffer::from(values.clone()), null_buf,
                            );
                            (ArrowDataType::Int64, Arc::new(arr) as ArrayRef)
                        }
                    }
                }
                ColumnData::Float64(values) => {
                    let arr = PrimitiveArray::<Float64Type>::new(
                        ScalarBuffer::from(values.clone()), null_buf,
                    );
                    (ArrowDataType::Float64, Arc::new(arr) as ArrayRef)
                }
                ColumnData::String { offsets, data } => {
                    let count = offsets.len().saturating_sub(1);
                    if null_buf.is_some() {
                        let strings: Vec<Option<&str>> = (0..count.min(active_count)).map(|i| {
                            if i < null_bitmap.len() && null_bitmap[i] {
                                None
                            } else {
                                let start = offsets[i] as usize;
                                let end = offsets[i + 1] as usize;
                                std::str::from_utf8(&data[start..end]).ok()
                            }
                        }).collect();
                        (ArrowDataType::Utf8, Arc::new(StringArray::from(strings)))
                    } else {
                        let strings: Vec<&str> = (0..count.min(active_count)).map(|i| {
                            let start = offsets[i] as usize;
                            let end = offsets[i + 1] as usize;
                            std::str::from_utf8(&data[start..end]).unwrap_or("")
                        }).collect();
                        (ArrowDataType::Utf8, Arc::new(StringArray::from(strings)))
                    }
                }
                ColumnData::Bool { data: bool_data, len: bool_len } => {
                    let bools: Vec<Option<bool>> = (0..*bool_len.min(&active_count)).map(|i| {
                        if null_buf.is_some() {
                            if i < null_bitmap.len() && null_bitmap[i] {
                                return None;
                            }
                        }
                        let byte_idx = i / 8;
                        let bit_idx = i % 8;
                        let val = byte_idx < bool_data.len() && (bool_data[byte_idx] >> bit_idx) & 1 == 1;
                        Some(val)
                    }).collect();
                    (ArrowDataType::Boolean, Arc::new(BooleanArray::from(bools)))
                }
                ColumnData::Binary { offsets, data } => {
                    use arrow::array::BinaryArray;
                    let count = offsets.len().saturating_sub(1);
                    if null_buf.is_some() {
                        let bins: Vec<Option<&[u8]>> = (0..count.min(active_count)).map(|i| {
                            if i < null_bitmap.len() && null_bitmap[i] {
                                None
                            } else {
                                let start = offsets[i] as usize;
                                let end = offsets[i + 1] as usize;
                                Some(&data[start..end])
                            }
                        }).collect();
                        (ArrowDataType::Binary, Arc::new(BinaryArray::from(bins)))
                    } else {
                        let bins: Vec<&[u8]> = (0..count.min(active_count)).map(|i| {
                            let start = offsets[i] as usize;
                            let end = offsets[i + 1] as usize;
                            &data[start..end]
                        }).collect();
                        (ArrowDataType::Binary, Arc::new(BinaryArray::from(bins)))
                    }
                }
                _ => {
                    // Fallback: null array
                    let arr = arrow::array::new_null_array(&ArrowDataType::Utf8, active_count);
                    (ArrowDataType::Utf8, arr)
                }
            };

            fields.push(Field::new(col_name, arrow_dt, true));
            arrays.push(array);
        }

        let arrow_schema = Arc::new(Schema::new(fields));
        let batch = RecordBatch::try_new(arrow_schema, arrays)
            .map_err(|e| err_data(e.to_string()))?;

        // Apply DeltaStore overlay if there are pending cell-level updates
        let ds = self.delta_store.read();
        if !ds.is_empty() && batch.num_rows() > 0 {
            let merged = super::delta::DeltaMerger::merge(&batch, &ds, &row_ids_for_delta)?;
            return Ok(Some(merged));
        }

        Ok(Some(batch))
    }
    
    /// Read a single column for a contiguous row range, V3 or V4.
    /// For V4 (in-memory): uses slice_range on loaded columns.
    /// For V3 (mmap): uses read_column_range_mmap.
    fn read_column_auto(
        &self,
        col_idx: usize,
        col_type: ColumnType,
        start: usize,
        count: usize,
        total_rows: usize,
        is_v4: bool,
    ) -> io::Result<ColumnData> {
        if is_v4 {
            let columns = self.columns.read();
            if col_idx < columns.len() && columns[col_idx].len() > 0 {
                Ok(columns[col_idx].slice_range(start, start + count))
            } else {
                Ok(Self::create_default_column(col_type, count))
            }
        } else {
            let column_index = self.column_index.read();
            if col_idx >= column_index.len() {
                return Ok(Self::create_default_column(col_type, count));
            }
            let entry = column_index[col_idx].clone();
            drop(column_index);
            
            let file_guard = self.file.read();
            let file = file_guard.as_ref()
                .ok_or_else(|| err_not_conn("File not open"))?;
            let mut mmap = self.mmap_cache.write();
            self.read_column_range_mmap(&mut mmap, file, &entry, col_type, start, count, total_rows)
        }
    }
    
    /// Read a single column for scattered row indices, V3 or V4.
    fn read_column_scattered_auto(
        &self,
        col_idx: usize,
        col_type: ColumnType,
        indices: &[usize],
        total_rows: usize,
        is_v4: bool,
    ) -> io::Result<ColumnData> {
        if is_v4 {
            let columns = self.columns.read();
            if col_idx < columns.len() && columns[col_idx].len() > 0 {
                Ok(columns[col_idx].filter_by_indices(indices))
            } else {
                Ok(Self::create_default_column(col_type, indices.len()))
            }
        } else {
            let column_index = self.column_index.read();
            if col_idx >= column_index.len() {
                return Ok(Self::create_default_column(col_type, indices.len()));
            }
            let entry = column_index[col_idx].clone();
            drop(column_index);
            
            let file_guard = self.file.read();
            let file = file_guard.as_ref()
                .ok_or_else(|| err_not_conn("File not open"))?;
            let mut mmap = self.mmap_cache.write();
            self.read_column_scattered_mmap(&mut mmap, file, &entry, col_type, indices, total_rows)
        }
    }

    /// Ensure IDs are loaded into memory (lazy loading optimization)
    /// This is called on-demand when IDs are actually needed.
    /// For V4: loads IDs from Row Groups via mmap (lightweight — only IDs, not column data).
    fn ensure_ids_loaded(&self) -> io::Result<()> {
        // Quick check without write lock
        if !self.ids.read().is_empty() {
            return Ok(());
        }
        
        let header = self.header.read();
        let id_count = header.row_count as usize;
        
        if id_count == 0 {
            return Ok(());
        }
        
        if self.cached_footer_offset.load(Ordering::Relaxed) > 0 {
            // V4: Load IDs from Row Groups via mmap (no column data loaded)
            drop(header);
            return self.ensure_ids_loaded_v4();
        }
        
        let file_guard = self.file.read();
        let file = file_guard.as_ref().ok_or_else(|| {
            err_not_conn("File not open")
        })?;
        
        let mut mmap_cache = self.mmap_cache.write();
        let mut ids = self.ids.write();
        
        // Double-check after acquiring write lock
        if !ids.is_empty() {
            return Ok(());
        }
        
        // Load IDs from disk (V3)
        let mut id_bytes = vec![0u8; id_count * 8];
        mmap_cache.read_at(file, &mut id_bytes, header.id_column_offset)?;
        
        ids.reserve(id_count);
        for i in 0..id_count {
            let id = u64::from_le_bytes(id_bytes[i * 8..(i + 1) * 8].try_into().unwrap());
            ids.push(id);
        }
        
        Ok(())
    }

    /// V4-specific: Load IDs + deletion vector from Row Groups via mmap.
    /// This is lightweight — only reads IDs and deletion bitmaps, NOT column data.
    /// Enables delete/exists/id_index operations without loading full table into memory.
    fn ensure_ids_loaded_v4(&self) -> io::Result<()> {
        // Double-check under write lock
        let mut ids = self.ids.write();
        if !ids.is_empty() {
            return Ok(());
        }

        let footer = match self.get_or_load_footer()? {
            Some(f) => f,
            None => return Ok(()),
        };

        let file_guard = self.file.read();
        let file = file_guard.as_ref()
            .ok_or_else(|| err_not_conn("File not open for V4 ID load"))?;
        let mut mmap_guard = self.mmap_cache.write();
        let mmap_ref = mmap_guard.get_or_create(file)?;

        let total_active: usize = footer.row_groups.iter()
            .map(|rg| rg.row_count as usize)
            .sum();
        ids.reserve(total_active);
        let mut deleted_acc: Vec<u8> = Vec::new();
        let mut max_id: u64 = 0;

        for rg_meta in &footer.row_groups {
            let rg_rows = rg_meta.row_count as usize;
            if rg_rows == 0 { continue; }

            let rg_end = (rg_meta.offset + rg_meta.data_size) as usize;
            if rg_end > mmap_ref.len() {
                return Err(err_data("RG extends past EOF"));
            }
            let rg_bytes = &mmap_ref[rg_meta.offset as usize .. rg_end];
            
            // Check compression flag at RG header byte 28
            let compress_flag = if rg_bytes.len() >= 32 { rg_bytes[28] } else { RG_COMPRESS_NONE };
            
            // Get the body bytes (after 32-byte RG header), decompressing if needed
            let decompressed_buf = decompress_rg_body(compress_flag, &rg_bytes[32..])?;
            let body: &[u8] = decompressed_buf.as_deref().unwrap_or(&rg_bytes[32..]);
            let mut pos: usize = 0;

            // Read IDs
            let id_byte_len = rg_rows * 8;
            if pos + id_byte_len > body.len() {
                return Err(err_data("RG IDs truncated"));
            }
            let id_slice = &body[pos..pos + id_byte_len];
            pos += id_byte_len;

            for i in 0..rg_rows {
                let id = u64::from_le_bytes(
                    id_slice[i * 8..(i + 1) * 8].try_into().unwrap()
                );
                ids.push(id);
                if id > max_id { max_id = id; }
            }

            // Read deletion vector
            let del_vec_len = (rg_rows + 7) / 8;
            if pos + del_vec_len > body.len() {
                return Err(err_data("RG deletion vector truncated"));
            }
            deleted_acc.extend_from_slice(&body[pos..pos + del_vec_len]);
        }

        drop(mmap_guard);
        drop(file_guard);

        // Update deletion bitmap
        {
            let mut deleted = self.deleted.write();
            if deleted.is_empty() && !deleted_acc.is_empty() {
                *deleted = deleted_acc;
            }
        }

        // Update next_id
        let current_next = self.next_id.load(Ordering::SeqCst);
        if max_id + 1 > current_next {
            self.next_id.store(max_id + 1, Ordering::SeqCst);
        }

        Ok(())
    }

    /// Scan specific columns from V4 Row Groups via mmap WITHOUT loading all data.
    /// Returns (Vec<ColumnData>, deletion_bitmap, Vec<null_bitmap_per_output_col>).
    /// This is the core building block for mmap-based fast paths (agg, filter, GROUP BY).
    fn scan_columns_mmap(
        &self,
        col_indices: &[usize],
        footer: &V4Footer,
    ) -> io::Result<(Vec<ColumnData>, Vec<u8>)> {
        let (cols, del, _nulls) = self.scan_columns_mmap_with_nulls(col_indices, footer)?;
        Ok((cols, del))
    }

    /// Same as scan_columns_mmap but also returns per-column null bitmaps.
    fn scan_columns_mmap_with_nulls(
        &self,
        col_indices: &[usize],
        footer: &V4Footer,
    ) -> io::Result<(Vec<ColumnData>, Vec<u8>, Vec<Vec<u8>>)> {
        let schema = &footer.schema;
        let col_count = schema.column_count();

        let file_guard = self.file.read();
        let file = file_guard.as_ref()
            .ok_or_else(|| err_not_conn("File not open for mmap scan"))?;
        let mut mmap_guard = self.mmap_cache.write();
        let mmap_ref = mmap_guard.get_or_create(file)?;

        // Pre-allocate accumulators to total row count — eliminates 16+ reallocations during RG iteration
        let total_rows: usize = footer.row_groups.iter().map(|rg| rg.row_count as usize).sum();
        let total_del_bytes = (total_rows + 7) / 8;
        let mut col_accumulators: Vec<ColumnData> = col_indices.iter()
            .map(|&ci| {
                let ct = schema.columns[ci].1;
                let acc_type = if ct == ColumnType::StringDict { ColumnType::String } else { ct };
                ColumnData::new(acc_type)
            })
            .collect();
        let mut all_del_bytes: Vec<u8> = Vec::with_capacity(total_del_bytes);
        let mut col_null_bitmaps: Vec<Vec<u8>> = col_indices.iter()
            .map(|_| Vec::with_capacity(total_del_bytes))
            .collect();

        let col_idx_to_out: HashMap<usize, usize> = col_indices.iter()
            .enumerate()
            .map(|(out_pos, &col_idx)| (col_idx, out_pos))
            .collect();

        for (rg_i, rg_meta) in footer.row_groups.iter().enumerate() {
            let rg_rows = rg_meta.row_count as usize;
            if rg_rows == 0 { continue; }

            let rg_end = (rg_meta.offset + rg_meta.data_size) as usize;
            if rg_end > mmap_ref.len() {
                return Err(err_data("RG extends past EOF"));
            }
            let rg_bytes = &mmap_ref[rg_meta.offset as usize .. rg_end];
            
            // Check compression flag at RG header byte 28, encoding version at byte 29
            let compress_flag = if rg_bytes.len() >= 32 { rg_bytes[28] } else { RG_COMPRESS_NONE };
            let encoding_version = if rg_bytes.len() >= 32 { rg_bytes[29] } else { 0 };
            
            let null_bitmap_len = (rg_rows + 7) / 8;
            let del_vec_len = (rg_rows + 7) / 8;

            // RCIX FAST PATH: uncompressed + RCIX available → jump directly to each column
            // Eliminates sequential skip of unneeded columns (O(1) per column seek).
            let rcix = footer.col_offsets.get(rg_i).filter(|v| !v.is_empty());
            if compress_flag == RG_COMPRESS_NONE && encoding_version >= 1 && rcix.is_some() {
                let body = &rg_bytes[32..];
                let rg_col_offsets = rcix.unwrap();

                // Deletion vector starts after IDs
                let del_start = rg_rows * 8;
                if del_start + del_vec_len <= body.len() {
                    all_del_bytes.extend_from_slice(&body[del_start..del_start + del_vec_len]);
                }

                // Read only requested columns using RCIX offsets
                for (&col_idx, &out_pos) in &col_idx_to_out {
                    if col_idx >= rg_col_offsets.len() { continue; }
                    let col_start = rg_col_offsets[col_idx] as usize;
                    if col_start + null_bitmap_len > body.len() { continue; }

                    col_null_bitmaps[out_pos]
                        .extend_from_slice(&body[col_start..col_start + null_bitmap_len]);

                    let data_start = col_start + null_bitmap_len;
                    if data_start > body.len() { continue; }
                    let col_type = schema.columns[col_idx].1;
                    let (col_data, _) = read_column_encoded(&body[data_start..], col_type)?;
                    let col_data = if matches!(&col_data, ColumnData::StringDict { .. }) {
                        col_data.decode_string_dict()
                    } else {
                        col_data
                    };
                    col_accumulators[out_pos].append(&col_data);
                }
                continue; // Skip sequential path
            }

            // SEQUENTIAL PATH: compressed or no RCIX — decompress and scan all columns
            let decompressed_buf = decompress_rg_body(compress_flag, &rg_bytes[32..])?;
            let body: &[u8] = decompressed_buf.as_deref().unwrap_or(&rg_bytes[32..]);
            let mut pos: usize = 0;

            // Skip IDs
            pos += rg_rows * 8;

            // Read deletion vector
            if pos + del_vec_len > body.len() {
                return Err(err_data("RG deletion vector truncated"));
            }
            all_del_bytes.extend_from_slice(&body[pos..pos + del_vec_len]);
            pos += del_vec_len;

            // Parse columns — read requested, skip others
            for col_idx in 0..col_count {
                if pos + null_bitmap_len > body.len() { break; }
                let null_bytes = &body[pos..pos + null_bitmap_len];
                pos += null_bitmap_len;

                let col_type = schema.columns[col_idx].1;

                if let Some(&out_pos) = col_idx_to_out.get(&col_idx) {
                    col_null_bitmaps[out_pos].extend_from_slice(null_bytes);

                    let (col_data, consumed) = if encoding_version >= 1 {
                        read_column_encoded(&body[pos..], col_type)?
                    } else {
                        ColumnData::from_bytes_typed(&body[pos..], col_type)?
                    };
                    pos += consumed;
                    let col_data = if matches!(&col_data, ColumnData::StringDict { .. }) {
                        col_data.decode_string_dict()
                    } else {
                        col_data
                    };
                    col_accumulators[out_pos].append(&col_data);
                } else {
                    let consumed = if encoding_version >= 1 {
                        skip_column_encoded(&body[pos..], col_type)?
                    } else {
                        ColumnData::skip_bytes_typed(&body[pos..], col_type)?
                    };
                    pos += consumed;
                }
            }
        }

        Ok((col_accumulators, all_del_bytes, col_null_bitmaps))
    }

    /// Determine which Row Groups can be skipped based on per-RG zone maps.
    /// Returns a set of RG indices that definitely won't match the filter.
    fn zone_map_prune_rgs(
        footer: &V4Footer,
        filter_col_idx: usize,
        filter_op: &str,
        filter_value: f64,
    ) -> HashSet<usize> {
        let mut skip: HashSet<usize> = HashSet::new();
        let filter_val_i64 = filter_value as i64;
        for (rg_idx, rg_zmaps) in footer.zone_maps.iter().enumerate() {
            for zm in rg_zmaps {
                if zm.col_idx as usize != filter_col_idx { continue; }
                let dominated = if zm.is_float {
                    let mn = f64::from_bits(zm.min_bits as u64);
                    let mx = f64::from_bits(zm.max_bits as u64);
                    match filter_op {
                        ">"  => mx <= filter_value,
                        ">=" => mx < filter_value,
                        "<"  => mn >= filter_value,
                        "<=" => mn > filter_value,
                        "=" | "==" => filter_value < mn || filter_value > mx,
                        _ => false,
                    }
                } else {
                    let mn = zm.min_bits;
                    let mx = zm.max_bits;
                    match filter_op {
                        ">"  => mx <= filter_val_i64,
                        ">=" => mx < filter_val_i64,
                        "<"  => mn >= filter_val_i64,
                        "<=" => mn > filter_val_i64,
                        "=" | "==" => filter_val_i64 < mn || filter_val_i64 > mx,
                        "!=" | "<>" => mn == mx && mn == filter_val_i64,
                        _ => false,
                    }
                };
                if dominated { skip.insert(rg_idx); }
                break; // only one zone map per column per RG
            }
        }
        skip
    }

    /// Like scan_columns_mmap but skips Row Groups in the `skip_rgs` set.
    /// Used for zone-map-pruned filtered scans.
    fn scan_columns_mmap_skip_rgs(
        &self,
        col_indices: &[usize],
        footer: &V4Footer,
        skip_rgs: &HashSet<usize>,
    ) -> io::Result<(Vec<ColumnData>, Vec<u8>)> {
        if skip_rgs.is_empty() {
            // No pruning needed — delegate to normal scan
            return self.scan_columns_mmap(col_indices, footer);
        }

        let schema = &footer.schema;
        let col_count = schema.column_count();

        let file_guard = self.file.read();
        let file = file_guard.as_ref()
            .ok_or_else(|| err_not_conn("File not open for mmap scan"))?;
        let mut mmap_guard = self.mmap_cache.write();
        let mmap_ref = mmap_guard.get_or_create(file)?;

        let mut col_accumulators: Vec<ColumnData> = col_indices.iter()
            .map(|&ci| {
                let ct = schema.columns[ci].1;
                let acc_type = if ct == ColumnType::StringDict { ColumnType::String } else { ct };
                ColumnData::new(acc_type)
            })
            .collect();
        let mut all_del_bytes: Vec<u8> = Vec::new();

        let col_idx_to_out: HashMap<usize, usize> = col_indices.iter()
            .enumerate()
            .map(|(out_pos, &col_idx)| (col_idx, out_pos))
            .collect();

        for (rg_idx, rg_meta) in footer.row_groups.iter().enumerate() {
            let rg_rows = rg_meta.row_count as usize;
            if rg_rows == 0 { continue; }
            if skip_rgs.contains(&rg_idx) { continue; } // Zone map pruned!

            let rg_end = (rg_meta.offset + rg_meta.data_size) as usize;
            if rg_end > mmap_ref.len() {
                return Err(err_data("RG extends past EOF"));
            }
            let rg_bytes = &mmap_ref[rg_meta.offset as usize .. rg_end];
            let compress_flag = if rg_bytes.len() >= 32 { rg_bytes[28] } else { RG_COMPRESS_NONE };
            let encoding_version = if rg_bytes.len() >= 32 { rg_bytes[29] } else { 0 };
            let decompressed_buf = decompress_rg_body(compress_flag, &rg_bytes[32..])?;
            let body: &[u8] = decompressed_buf.as_deref().unwrap_or(&rg_bytes[32..]);
            let mut pos: usize = 0;

            pos += rg_rows * 8; // skip IDs

            let del_vec_len = (rg_rows + 7) / 8;
            if pos + del_vec_len > body.len() {
                return Err(err_data("RG deletion vector truncated"));
            }
            all_del_bytes.extend_from_slice(&body[pos..pos + del_vec_len]);
            pos += del_vec_len;

            let null_bitmap_len = (rg_rows + 7) / 8;
            for col_idx in 0..col_count {
                if pos + null_bitmap_len > body.len() { break; }
                pos += null_bitmap_len; // skip null bitmap

                let col_type = schema.columns[col_idx].1;
                if let Some(&out_pos) = col_idx_to_out.get(&col_idx) {
                    let (col_data, consumed) = if encoding_version >= 1 {
                        read_column_encoded(&body[pos..], col_type)?
                    } else {
                        ColumnData::from_bytes_typed(&body[pos..], col_type)?
                    };
                    pos += consumed;
                    let col_data = if matches!(&col_data, ColumnData::StringDict { .. }) {
                        col_data.decode_string_dict()
                    } else {
                        col_data
                    };
                    col_accumulators[out_pos].append(&col_data);
                } else {
                    let consumed = if encoding_version >= 1 {
                        skip_column_encoded(&body[pos..], col_type)?
                    } else {
                        ColumnData::skip_bytes_typed(&body[pos..], col_type)?
                    };
                    pos += consumed;
                }
            }
        }

        Ok((col_accumulators, all_del_bytes))
    }

    /// Read IDs for a row range (lazy loads IDs if not already loaded)
    pub fn read_ids(&self, start_row: usize, row_count: Option<usize>) -> io::Result<Vec<u64>> {
        // Ensure IDs are loaded (lazy loading)
        self.ensure_ids_loaded()?;
        
        let ids = self.ids.read();
        let total = ids.len();
        let start = start_row.min(total);
        let count = row_count.map(|c| c.min(total - start)).unwrap_or(total - start);
        Ok(ids[start..start + count].to_vec())
    }

    /// Read IDs for specific row indices (optimized for scattered access, lazy loads)
    pub fn read_ids_by_indices(&self, row_indices: &[usize]) -> io::Result<Vec<i64>> {
        // Ensure IDs are loaded (lazy loading)
        self.ensure_ids_loaded()?;
        
        let ids = self.ids.read();
        let total = ids.len();
        Ok(row_indices.iter()
            .map(|&i| if i < total { ids[i] as i64 } else { 0 })
            .collect())
    }

    /// Execute Complex (Filter+Group+Order) query with single-pass optimization
    /// Key optimization for: SELECT group_col, AGG(agg_col) FROM table WHERE filter_col = 'value'
    /// GROUP BY group_col ORDER BY total DESC LIMIT n
    pub fn execute_filter_group_order(
        &self,
        filter_col: &str,
        filter_val: &str,
        group_col: &str,
        agg_col: Option<&str>,
        agg_func: crate::query::AggregateFunc,
        descending: bool,
        limit: usize,
        offset: usize,
    ) -> io::Result<Option<RecordBatch>> {
        use crate::query::AggregateFunc;
        use arrow::array::{Int64Array, Float64Array, StringArray, UInt32Array};
        use arrow::datatypes::{Field, Schema, DataType as ArrowDataType};
        use std::collections::BinaryHeap;
        use std::cmp::Ordering;
        use std::sync::Arc;

        // V4: mmap dict scan is V3-specific, let caller use general query path
        // LOCK-FREE: use cached_footer_offset for V4 detection
        if self.cached_footer_offset.load(std::sync::atomic::Ordering::Relaxed) > 0 {
            return Ok(None);
        }

        let schema = self.schema.read();
        let column_index = self.column_index.read();
        let header = self.header.read();

        let total_rows = header.row_count as usize;
        if total_rows == 0 {
            return Ok(Some(RecordBatch::new_empty(Arc::new(Schema::empty()))));
        }

        // Get column indices
        let filter_idx = match schema.get_index(filter_col) {
            Some(idx) => idx,
            None => return Ok(None),
        };
        let group_idx = match schema.get_index(group_col) {
            Some(idx) => idx,
            None => return Ok(None),
        };

        let file_guard = self.file.read();
        let file = file_guard.as_ref().ok_or_else(|| {
            err_not_conn("File not open")
        })?;
        
        let mut mmap_cache = self.mmap_cache.write();

        // Read filter column to find matching rows
        let filter_index = &column_index[filter_idx];
        let (_, filter_col_type) = &schema.columns[filter_idx];
        
        if *filter_col_type != ColumnType::StringDict {
            return Ok(None); // Only optimized for dictionary-encoded strings
        }

        // Read filter column header and dictionary
        let filter_base = filter_index.data_offset;
        let mut header_buf = [0u8; 16];
        mmap_cache.read_at(file, &mut header_buf, filter_base)?;
        let stored_rows = u64::from_le_bytes(header_buf[0..8].try_into().unwrap()) as usize;
        let dict_size = u64::from_le_bytes(header_buf[8..16].try_into().unwrap()) as usize;
        
        // Read dictionary
        let dict_offsets_offset = filter_base + 16 + (stored_rows * 4) as u64;
        let mut dict_offsets_buf = vec![0u8; dict_size * 4];
        mmap_cache.read_at(file, &mut dict_offsets_buf, dict_offsets_offset)?;
        let dict_offsets: Vec<u32> = (0..dict_size)
            .map(|i| u32::from_le_bytes(dict_offsets_buf[i*4..(i+1)*4].try_into().unwrap()))
            .collect();
        
        let data_len_offset = dict_offsets_offset + (dict_size * 4) as u64;
        let mut data_len_buf = [0u8; 8];
        mmap_cache.read_at(file, &mut data_len_buf, data_len_offset)?;
        let dict_data_len = u64::from_le_bytes(data_len_buf) as usize;
        
        let dict_data_offset = data_len_offset + 8;
        let mut dict_data = vec![0u8; dict_data_len];
        if dict_data_len > 0 {
            mmap_cache.read_at(file, &mut dict_data, dict_data_offset)?;
        }
        
        // Find filter value in dictionary
        let filter_bytes = filter_val.as_bytes();
        let mut target_dict_idx: Option<u32> = None;
        for i in 0..dict_size.saturating_sub(1) {
            let start = dict_offsets[i] as usize;
            let end = if i + 1 < dict_size { dict_offsets[i + 1] as usize } else { dict_data_len };
            if &dict_data[start..end] == filter_bytes {
                target_dict_idx = Some((i + 1) as u32);
                break;
            }
        }
        
        let target_idx = match target_dict_idx {
            Some(idx) => idx,
            None => return Ok(Some(RecordBatch::new_empty(Arc::new(Schema::empty())))),
        };
        
        // Read group column dictionary
        let group_index = &column_index[group_idx];
        let (_, group_col_type) = &schema.columns[group_idx];
        
        if *group_col_type != ColumnType::StringDict {
            return Ok(None);
        }
        
        let group_base = group_index.data_offset;
        let mut group_header = [0u8; 16];
        mmap_cache.read_at(file, &mut group_header, group_base)?;
        let group_rows = u64::from_le_bytes(group_header[0..8].try_into().unwrap()) as usize;
        let group_dict_size = u64::from_le_bytes(group_header[8..16].try_into().unwrap()) as usize;
        
        let group_dict_offsets_offset = group_base + 16 + (group_rows * 4) as u64;
        let mut group_dict_offsets_buf = vec![0u8; group_dict_size * 4];
        mmap_cache.read_at(file, &mut group_dict_offsets_buf, group_dict_offsets_offset)?;
        let group_dict_offsets: Vec<u32> = (0..group_dict_size)
            .map(|i| u32::from_le_bytes(group_dict_offsets_buf[i*4..(i+1)*4].try_into().unwrap()))
            .collect();
        
        let group_data_len_offset = group_dict_offsets_offset + (group_dict_size * 4) as u64;
        let mut group_data_len_buf = [0u8; 8];
        mmap_cache.read_at(file, &mut group_data_len_buf, group_data_len_offset)?;
        let group_dict_data_len = u64::from_le_bytes(group_data_len_buf) as usize;
        
        let group_dict_data_offset = group_data_len_offset + 8;
        let mut group_dict_data = vec![0u8; group_dict_data_len];
        if group_dict_data_len > 0 {
            mmap_cache.read_at(file, &mut group_dict_data, group_dict_data_offset)?;
        }
        
        // Aggregate: group_idx -> (count, sum)
        let mut group_counts: Vec<i64> = vec![0; group_dict_size];
        let mut group_sums: Vec<f64> = vec![0.0; group_dict_size];
        
        // Read filter indices and aggregate
        let filter_indices_offset = filter_base + 16;
        let group_indices_offset = group_base + 16;
        
        // Read agg column if specified
        let agg_idx = agg_col.and_then(|name| schema.get_index(name));
        let agg_values: Option<Vec<f64>> = if let Some(idx) = agg_idx {
            let agg_index = &column_index[idx];
            let (_, agg_col_type) = &schema.columns[idx];
            if *agg_col_type == ColumnType::Float64 || *agg_col_type == ColumnType::Int64 {
                let agg_base = agg_index.data_offset + 8; // Skip count header
                let mut agg_buf = vec![0u8; stored_rows * 8];
                mmap_cache.read_at(file, &mut agg_buf, agg_base)?;
                let values: Vec<f64> = (0..stored_rows)
                    .map(|i| {
                        let bytes = &agg_buf[i*8..(i+1)*8];
                        f64::from_le_bytes(bytes.try_into().unwrap())
                    })
                    .collect();
                Some(values)
            } else {
                None
            }
        } else {
            None
        };
        
        // Single-pass aggregation
        const CHUNK_SIZE: usize = 8192;
        let mut filter_chunk = vec![0u32; CHUNK_SIZE];
        let mut group_chunk = vec![0u32; CHUNK_SIZE];
        
        let mut row = 0;
        while row < stored_rows {
            let chunk_rows = CHUNK_SIZE.min(stored_rows - row);
            
            // Read filter indices chunk
            let filter_bytes = unsafe {
                std::slice::from_raw_parts_mut(filter_chunk.as_mut_ptr() as *mut u8, chunk_rows * 4)
            };
            mmap_cache.read_at(file, filter_bytes, filter_indices_offset + (row * 4) as u64)?;
            
            // Read group indices chunk
            let group_bytes = unsafe {
                std::slice::from_raw_parts_mut(group_chunk.as_mut_ptr() as *mut u8, chunk_rows * 4)
            };
            mmap_cache.read_at(file, group_bytes, group_indices_offset + (row * 4) as u64)?;
            
            // Aggregate
            for i in 0..chunk_rows {
                if filter_chunk[i] == target_idx {
                    let g_idx = group_chunk[i] as usize;
                    if g_idx > 0 && g_idx < group_dict_size {
                        group_counts[g_idx] += 1;
                        if let Some(ref vals) = agg_values {
                            group_sums[g_idx] += vals[row + i];
                        }
                    }
                }
            }
            row += chunk_rows;
        }
        
        // Build result with top-k
        #[derive(Clone, Copy)]
        struct HeapItem { idx: usize, count: i64, sum: f64 }
        
        impl Ord for HeapItem {
            fn cmp(&self, other: &Self) -> Ordering {
                let self_val = self.count;
                let other_val = other.count;
                self_val.cmp(&other_val)
            }
        }
        
        impl PartialOrd for HeapItem {
            fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
                Some(self.cmp(other))
            }
        }
        
        impl Eq for HeapItem {}
        impl PartialEq for HeapItem {
            fn eq(&self, other: &Self) -> bool {
                self.count == other.count
            }
        }
        
        let mut heap = BinaryHeap::new();
        for i in 1..group_dict_size {
            if group_counts[i] > 0 {
                if heap.len() < limit + offset {
                    heap.push(HeapItem { idx: i, count: group_counts[i], sum: group_sums[i] });
                } else if let Some(min) = heap.peek() {
                    if group_counts[i] > min.count {
                        heap.pop();
                        heap.push(HeapItem { idx: i, count: group_counts[i], sum: group_sums[i] });
                    }
                }
            }
        }
        
        // Extract results
        let mut results: Vec<HeapItem> = heap.into_vec();
        results.sort_by(|a, b| {
            if descending {
                b.count.cmp(&a.count)
            } else {
                a.count.cmp(&b.count)
            }
        });
        
        // Apply offset and limit
        let final_results: Vec<HeapItem> = results.into_iter().skip(offset).take(limit).collect();
        
        if final_results.is_empty() {
            return Ok(Some(RecordBatch::new_empty(Arc::new(Schema::empty()))));
        }
        
        // Build Arrow arrays
        let group_strings: Vec<&str> = final_results.iter()
            .map(|item| {
                let dict_idx = item.idx - 1;
                let start = group_dict_offsets[dict_idx] as usize;
                let end = if dict_idx + 1 < group_dict_size { group_dict_offsets[dict_idx + 1] as usize } else { group_dict_data_len };
                std::str::from_utf8(&group_dict_data[start..end]).unwrap_or("")
            })
            .collect();
        
        let counts: Vec<i64> = final_results.iter().map(|item| item.count).collect();
        
        let result_schema = Arc::new(Schema::new(vec![
            Field::new(group_col, ArrowDataType::Utf8, false),
            Field::new("total", ArrowDataType::Int64, false),
        ]));
        
        let result_batch = RecordBatch::try_new(
            result_schema,
            vec![
                Arc::new(StringArray::from(group_strings)) as ArrayRef,
                Arc::new(Int64Array::from(counts)) as ArrayRef,
            ],
        ).map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e.to_string()))?;
        
        Ok(Some(result_batch))
    }

    // ========================================================================
    // Internal helpers
    // ========================================================================

    /// Ensure id_to_idx AHashMap is built (lazy load)
    /// Called automatically by delete/exists/get_row_idx operations
    fn ensure_id_index(&self) {
        // First ensure IDs are loaded (since we lazy-load them now)
        let _ = self.ensure_ids_loaded();
        
        let mut id_to_idx = self.id_to_idx.write();
        if id_to_idx.is_none() {
            let ids = self.ids.read();
            let mut map = ahash::AHashMap::with_capacity(ids.len());
            for (idx, &id) in ids.iter().enumerate() {
                map.insert(id, idx);
            }
            *id_to_idx = Some(map);
        }
    }

    // ========================================================================
    // Internal read helpers (mmap-based for cross-platform zero-copy reads)
    // ========================================================================

    fn read_column_range_mmap(
        &self,
        mmap_cache: &mut MmapCache,
        file: &File,
        index: &ColumnIndexEntry,
        dtype: ColumnType,
        start_row: usize,
        row_count: usize,
        _total_rows: usize,
    ) -> io::Result<ColumnData> {
        // ColumnData format has an 8-byte count header for all types
        // Format: [count:u64][data...]
        const HEADER_SIZE: u64 = 8;
        
        match dtype {
            ColumnType::Int64 | ColumnType::Int8 | ColumnType::Int16 | ColumnType::Int32 |
            ColumnType::UInt8 | ColumnType::UInt16 | ColumnType::UInt32 | ColumnType::UInt64 |
            ColumnType::Timestamp | ColumnType::Date => {
                // Format: [count:u64][values:i64*]
                // Zero-copy optimization: read directly into i64 buffer
                let byte_offset = HEADER_SIZE + (start_row * 8) as u64;
                
                let mut values: Vec<i64> = vec![0i64; row_count];
                // SAFETY: i64 has the same memory layout as [u8; 8] on little-endian systems
                // We read directly into the Vec's backing memory to avoid byte-by-byte parsing
                let bytes_slice = unsafe {
                    std::slice::from_raw_parts_mut(
                        values.as_mut_ptr() as *mut u8,
                        row_count * 8
                    )
                };
                mmap_cache.read_at(file, bytes_slice, index.data_offset + byte_offset)?;
                
                // Handle endianness: convert from LE if on BE system
                #[cfg(target_endian = "big")]
                for v in &mut values {
                    *v = i64::from_le(*v);
                }
                
                Ok(ColumnData::Int64(values))
            }
            ColumnType::Float64 | ColumnType::Float32 => {
                // Format: [count:u64][values:f64*]
                // Zero-copy optimization: read directly into f64 buffer
                let byte_offset = HEADER_SIZE + (start_row * 8) as u64;
                
                let mut values: Vec<f64> = vec![0f64; row_count];
                // SAFETY: f64 has the same memory layout as [u8; 8] on little-endian systems
                let bytes_slice = unsafe {
                    std::slice::from_raw_parts_mut(
                        values.as_mut_ptr() as *mut u8,
                        row_count * 8
                    )
                };
                mmap_cache.read_at(file, bytes_slice, index.data_offset + byte_offset)?;
                
                // Handle endianness: convert from LE if on BE system
                #[cfg(target_endian = "big")]
                for v in &mut values {
                    *v = f64::from_le_bytes(v.to_ne_bytes());
                }
                
                Ok(ColumnData::Float64(values))
            }
            ColumnType::Bool => {
                // Format: [len:u64][packed_bits...]
                let start_byte = start_row / 8;
                let end_byte = (start_row + row_count + 7) / 8;
                let byte_count = end_byte - start_byte;
                
                let mut packed = vec![0u8; byte_count];
                mmap_cache.read_at(file, &mut packed, index.data_offset + HEADER_SIZE + start_byte as u64)?;
                
                Ok(ColumnData::Bool { data: packed, len: row_count })
            }
            ColumnType::String | ColumnType::Binary => {
                // Variable-length type: need to read offsets first
                self.read_variable_column_range_mmap(mmap_cache, file, index, dtype, start_row, row_count)
            }
            ColumnType::StringDict => {
                // Native dictionary-encoded string reading
                self.read_string_dict_column_range_mmap(mmap_cache, file, index, start_row, row_count)
            }
            ColumnType::Null => {
                Ok(ColumnData::Int64(vec![0; row_count]))
            }
        }
    }

    fn read_variable_column_range_mmap(
        &self,
        mmap_cache: &mut MmapCache,
        file: &File,
        index: &ColumnIndexEntry,
        dtype: ColumnType,
        start_row: usize,
        row_count: usize,
    ) -> io::Result<ColumnData> {
        // Variable-length format: [count:u64][offsets:u32*][data_len:u64][data:bytes]
        // Read header to get total count
        let mut header_buf = [0u8; 8];
        mmap_cache.read_at(file, &mut header_buf, index.data_offset)?;
        let total_count = u64::from_le_bytes(header_buf) as usize;
        
        if start_row >= total_count {
            return Ok(ColumnData::String { offsets: vec![0], data: Vec::new() });
        }
        
        let actual_count = row_count.min(total_count - start_row);
        
        // OPTIMIZATION: Read offsets directly into u32 Vec using bulk read
        let offset_start = 8 + start_row * 4; // skip count header
        let offset_count = actual_count + 1;
        let mut offsets: Vec<u32> = vec![0u32; offset_count];
        // SAFETY: u32 slice can be safely viewed as bytes for reading
        let offset_bytes = unsafe {
            std::slice::from_raw_parts_mut(offsets.as_mut_ptr() as *mut u8, offset_count * 4)
        };
        mmap_cache.read_at(file, offset_bytes, index.data_offset + offset_start as u64)?;
        
        // Handle endianness on big-endian systems
        #[cfg(target_endian = "big")]
        for off in &mut offsets {
            *off = u32::from_le(*off);
        }
        
        // Calculate data range
        let data_start = offsets[0];
        let data_end = offsets[actual_count];
        let data_len = (data_end - data_start) as usize;
        
        // Read data portion
        // Data starts after: 8 (count) + (total_count+1)*4 (offsets) + 8 (data_len)
        let data_offset_in_file = index.data_offset + 8 + (total_count + 1) as u64 * 4 + 8 + data_start as u64;
        let mut data = vec![0u8; data_len];
        if data_len > 0 {
            mmap_cache.read_at(file, &mut data, data_offset_in_file)?;
        }
        
        // Normalize offsets to start at 0 using SIMD-friendly subtraction
        let base = offsets[0];
        if base != 0 {
            for off in &mut offsets {
                *off -= base;
            }
        }
        
        match dtype {
            ColumnType::String => Ok(ColumnData::String { offsets, data }),
            ColumnType::Binary => Ok(ColumnData::Binary { offsets, data }),
            _ => Err(io::Error::new(io::ErrorKind::InvalidData, "Not a variable type")),
        }
    }

    /// Read StringDict column with native format
    /// Format: [row_count:u64][dict_size:u64][indices:u32*row_count][dict_offsets:u32*dict_size][dict_data_len:u64][dict_data]
    /// OPTIMIZED: Uses bulk read for u32 arrays instead of per-element parsing
    fn read_string_dict_column_range_mmap(
        &self,
        mmap_cache: &mut MmapCache,
        file: &File,
        index: &ColumnIndexEntry,
        start_row: usize,
        row_count: usize,
    ) -> io::Result<ColumnData> {
        let base_offset = index.data_offset;
        
        // Read header: [row_count:u64][dict_size:u64]
        let mut header = [0u8; 16];
        mmap_cache.read_at(file, &mut header, base_offset)?;
        let total_rows = u64::from_le_bytes(header[0..8].try_into().unwrap()) as usize;
        let dict_size = u64::from_le_bytes(header[8..16].try_into().unwrap()) as usize;
        
        if start_row >= total_rows {
            return Ok(ColumnData::StringDict {
                indices: Vec::new(),
                dict_offsets: vec![0],
                dict_data: Vec::new(),
            });
        }
        
        let actual_count = row_count.min(total_rows - start_row);
        
        // OPTIMIZATION: Read indices directly into Vec<u32>
        let indices_offset = base_offset + 16 + (start_row * 4) as u64;
        let mut indices: Vec<u32> = vec![0u32; actual_count];
        let indices_bytes = unsafe {
            std::slice::from_raw_parts_mut(indices.as_mut_ptr() as *mut u8, actual_count * 4)
        };
        mmap_cache.read_at(file, indices_bytes, indices_offset)?;
        
        #[cfg(target_endian = "big")]
        for idx in &mut indices {
            *idx = u32::from_le(*idx);
        }
        
        // OPTIMIZATION: Read dict_offsets directly into Vec<u32>
        let dict_offsets_offset = base_offset + 16 + (total_rows * 4) as u64;
        let mut dict_offsets: Vec<u32> = vec![0u32; dict_size];
        let dict_offsets_bytes = unsafe {
            std::slice::from_raw_parts_mut(dict_offsets.as_mut_ptr() as *mut u8, dict_size * 4)
        };
        mmap_cache.read_at(file, dict_offsets_bytes, dict_offsets_offset)?;
        
        #[cfg(target_endian = "big")]
        for off in &mut dict_offsets {
            *off = u32::from_le(*off);
        }
        
        // Read dict_data_len and dict_data
        let dict_data_len_offset = dict_offsets_offset + (dict_size * 4) as u64;
        let mut data_len_buf = [0u8; 8];
        mmap_cache.read_at(file, &mut data_len_buf, dict_data_len_offset)?;
        let dict_data_len = u64::from_le_bytes(data_len_buf) as usize;
        
        let dict_data_offset = dict_data_len_offset + 8;
        let mut dict_data = vec![0u8; dict_data_len];
        if dict_data_len > 0 {
            mmap_cache.read_at(file, &mut dict_data, dict_data_offset)?;
        }
        
        Ok(ColumnData::StringDict {
            indices,
            dict_offsets,
            dict_data,
        })
    }

    /// Read StringDict column with scattered row indices
    /// OPTIMIZED: Only reads the specific indices needed, not all indices
    fn read_string_dict_column_scattered_mmap(
        &self,
        mmap_cache: &mut MmapCache,
        file: &File,
        index: &ColumnIndexEntry,
        row_indices: &[usize],
    ) -> io::Result<ColumnData> {
        if row_indices.is_empty() {
            return Ok(ColumnData::StringDict {
                indices: Vec::new(),
                dict_offsets: vec![0],
                dict_data: Vec::new(),
            });
        }
        
        let base_offset = index.data_offset;
        
        // Read header: [row_count:u64][dict_size:u64]
        let mut header = [0u8; 16];
        mmap_cache.read_at(file, &mut header, base_offset)?;
        let total_rows = u64::from_le_bytes(header[0..8].try_into().unwrap()) as usize;
        let dict_size = u64::from_le_bytes(header[8..16].try_into().unwrap()) as usize;
        
        let all_indices_offset = base_offset + 16;
        let n = row_indices.len();
        
        // OPTIMIZED: Read only the specific indices we need instead of all indices
        // For small scattered reads, read individually; for dense reads, read a range
        let mut indices = Vec::with_capacity(n);
        
        if n <= 256 {
            // Small number of indices - read each one individually using thread-local buffer
            SCATTERED_READ_BUF.with(|buf| {
                let mut buf = buf.borrow_mut();
                buf.resize(4, 0);
                for &row_idx in row_indices {
                    if row_idx < total_rows {
                        mmap_cache.read_at(file, &mut buf[..4], all_indices_offset + (row_idx * 4) as u64)?;
                        indices.push(u32::from_le_bytes([buf[0], buf[1], buf[2], buf[3]]));
                    } else {
                        indices.push(0);
                    }
                }
                Ok::<(), io::Error>(())
            })?;
        } else {
            // For larger reads, find min/max and read that range
            let min_idx = *row_indices.iter().min().unwrap_or(&0);
            let max_idx = *row_indices.iter().max().unwrap_or(&0);
            let range_size = max_idx - min_idx + 1;
            
            // OPTIMIZATION: If range is reasonably dense, read whole range as Vec<u32>
            if range_size <= n * 4 && range_size <= total_rows {
                let mut range_values: Vec<u32> = vec![0u32; range_size];
                let range_bytes = unsafe {
                    std::slice::from_raw_parts_mut(range_values.as_mut_ptr() as *mut u8, range_size * 4)
                };
                mmap_cache.read_at(file, range_bytes, all_indices_offset + (min_idx * 4) as u64)?;
                
                #[cfg(target_endian = "big")]
                for v in &mut range_values {
                    *v = u32::from_le(*v);
                }
                
                for &row_idx in row_indices {
                    if row_idx < total_rows {
                        let local_idx = row_idx - min_idx;
                        indices.push(range_values[local_idx]);
                    } else {
                        indices.push(0);
                    }
                }
            } else {
                // Sparse - read individually using thread-local buffer
                SCATTERED_READ_BUF.with(|buf| {
                    let mut buf = buf.borrow_mut();
                    buf.resize(4, 0);
                    for &row_idx in row_indices {
                        if row_idx < total_rows {
                            mmap_cache.read_at(file, &mut buf[..4], all_indices_offset + (row_idx * 4) as u64)?;
                            indices.push(u32::from_le_bytes([buf[0], buf[1], buf[2], buf[3]]));
                        } else {
                            indices.push(0);
                        }
                    }
                    Ok::<(), io::Error>(())
                })?;
            }
        }
        
        // OPTIMIZATION: Read dict_offsets directly into Vec<u32>
        let dict_offsets_offset = base_offset + 16 + (total_rows * 4) as u64;
        let mut dict_offsets: Vec<u32> = vec![0u32; dict_size];
        let dict_offsets_bytes = unsafe {
            std::slice::from_raw_parts_mut(dict_offsets.as_mut_ptr() as *mut u8, dict_size * 4)
        };
        mmap_cache.read_at(file, dict_offsets_bytes, dict_offsets_offset)?;
        
        #[cfg(target_endian = "big")]
        for off in &mut dict_offsets {
            *off = u32::from_le(*off);
        }
        
        // Read dict_data_len and dict_data
        let dict_data_len_offset = dict_offsets_offset + (dict_size * 4) as u64;
        let mut data_len_buf = [0u8; 8];
        mmap_cache.read_at(file, &mut data_len_buf, dict_data_len_offset)?;
        let dict_data_len = u64::from_le_bytes(data_len_buf) as usize;
        
        let dict_data_offset = dict_data_len_offset + 8;
        let mut dict_data = vec![0u8; dict_data_len];
        if dict_data_len > 0 {
            mmap_cache.read_at(file, &mut dict_data, dict_data_offset)?;
        }
        
        Ok(ColumnData::StringDict {
            indices,
            dict_offsets,
            dict_data,
        })
    }

    /// Optimized scattered read for variable-length columns (String/Binary) using mmap
    fn read_variable_column_scattered_mmap(
        &self,
        mmap_cache: &mut MmapCache,
        file: &File,
        index: &ColumnIndexEntry,
        dtype: ColumnType,
        row_indices: &[usize],
    ) -> io::Result<ColumnData> {
        if row_indices.is_empty() {
            return Ok(match dtype {
                ColumnType::String => ColumnData::String { offsets: vec![0], data: Vec::new() },
                _ => ColumnData::Binary { offsets: vec![0], data: Vec::new() },
            });
        }

        // Variable-length format: [count:u64][offsets:u32*(count+1)][data_len:u64][data:bytes]
        // Read header to get total count
        let mut header_buf = [0u8; 8];
        mmap_cache.read_at(file, &mut header_buf, index.data_offset)?;
        let total_count = u64::from_le_bytes(header_buf) as usize;

        // Read only the offsets we need (need idx and idx+1 for each row)
        // Collect unique offset indices needed
        let mut offset_indices: Vec<usize> = Vec::with_capacity(row_indices.len() * 2);
        for &idx in row_indices {
            if idx < total_count {
                offset_indices.push(idx);
                offset_indices.push(idx + 1);
            }
        }
        offset_indices.sort_unstable();
        offset_indices.dedup();

        if offset_indices.is_empty() {
            return Ok(match dtype {
                ColumnType::String => ColumnData::String { offsets: vec![0], data: Vec::new() },
                _ => ColumnData::Binary { offsets: vec![0], data: Vec::new() },
            });
        }

        // Read required offsets in batches (optimize for contiguous ranges)
        let mut offset_map: HashMap<usize, u32> = HashMap::with_capacity(offset_indices.len());
        let offset_base = index.data_offset + 8; // skip count header
        
        // For small number of indices, read individually
        // For larger sets, read a range that covers all needed offsets
        let min_idx = *offset_indices.first().unwrap();
        let max_idx = *offset_indices.last().unwrap();
        
        if max_idx - min_idx < offset_indices.len() * 4 {
            // Indices are sparse enough - read range
            let range_count = max_idx - min_idx + 1;
            let mut offset_buf = vec![0u8; range_count * 4];
            mmap_cache.read_at(file, &mut offset_buf, offset_base + (min_idx * 4) as u64)?;
            
            for &idx in &offset_indices {
                let local_idx = idx - min_idx;
                let off = u32::from_le_bytes(offset_buf[local_idx * 4..(local_idx + 1) * 4].try_into().unwrap());
                offset_map.insert(idx, off);
            }
        } else {
            // Very sparse - read individually
            let mut buf = [0u8; 4];
            for &idx in &offset_indices {
                mmap_cache.read_at(file, &mut buf, offset_base + (idx * 4) as u64)?;
                offset_map.insert(idx, u32::from_le_bytes(buf));
            }
        }

        // Calculate data offset base: skip count(8) + offsets((total_count+1)*4) + data_len(8)
        let data_base = index.data_offset + 8 + (total_count + 1) as u64 * 4 + 8;

        // Read data for each requested row and build result
        let mut result_offsets = vec![0u32];
        let mut result_data = Vec::new();

        for &idx in row_indices {
            if idx < total_count {
                let start = *offset_map.get(&idx).unwrap_or(&0);
                let end = *offset_map.get(&(idx + 1)).unwrap_or(&start);
                let len = (end - start) as usize;
                
                if len > 0 {
                    let mut chunk = vec![0u8; len];
                    mmap_cache.read_at(file, &mut chunk, data_base + start as u64)?;
                    result_data.extend_from_slice(&chunk);
                }
                result_offsets.push(result_data.len() as u32);
            } else {
                // Out of bounds - push empty
                result_offsets.push(result_data.len() as u32);
            }
        }

        match dtype {
            ColumnType::String => Ok(ColumnData::String { offsets: result_offsets, data: result_data }),
            ColumnType::Binary => Ok(ColumnData::Binary { offsets: result_offsets, data: result_data }),
            _ => Err(io::Error::new(io::ErrorKind::InvalidData, "Not a variable type")),
        }
    }
    
    /// Optimized scattered read for numeric types using row-group based I/O
    /// Reads data in larger chunks (row-groups) to reduce number of I/O operations
    fn read_numeric_scattered_optimized<T: Copy + Default + 'static>(
        mmap_cache: &mut MmapCache,
        file: &File,
        index: &ColumnIndexEntry,
        row_indices: &[usize],
        header_size: u64,
    ) -> io::Result<Vec<T>> {
        if row_indices.is_empty() {
            return Ok(Vec::new());
        }
        
        let n = row_indices.len();
        let elem_size = std::mem::size_of::<T>();
        
        // For small numbers, simple sequential read without sorting is faster
        // Typical LIMIT queries (100-500 rows) benefit from avoiding sort overhead
        if n <= 256 {
            let mut values = Vec::with_capacity(n);
            SCATTERED_READ_BUF.with(|buf| {
                let mut buf = buf.borrow_mut();
                buf.resize(8, 0);
                for &idx in row_indices {
                    mmap_cache.read_at(file, &mut buf[..elem_size], index.data_offset + header_size + (idx * elem_size) as u64)?;
                    let val: T = unsafe { std::ptr::read(buf.as_ptr() as *const T) };
                    values.push(val);
                }
                Ok::<(), io::Error>(())
            })?;
            return Ok(values);
        }
        
        // ROW-GROUP BASED READING for larger scattered reads
        const ROW_GROUP_SIZE: usize = 8192;
        
        // Sort indices and track original positions
        let mut indexed: Vec<(usize, usize)> = row_indices.iter().enumerate().map(|(i, &idx)| (idx, i)).collect();
        indexed.sort_unstable_by_key(|&(idx, _)| idx);
        
        let mut result: Vec<T> = vec![T::default(); n];
        let mut i = 0;
        
        // Process by row-groups
        while i < indexed.len() {
            let first_idx = indexed[i].0;
            let group_start = (first_idx / ROW_GROUP_SIZE) * ROW_GROUP_SIZE;
            let group_end = group_start + ROW_GROUP_SIZE;
            
            // Find all indices within this row-group
            let mut group_indices = Vec::new();
            while i < indexed.len() && indexed[i].0 < group_end {
                group_indices.push(indexed[i]);
                i += 1;
            }
            
            // Decide read strategy based on density within group
            let indices_in_group = group_indices.len();
            let span = group_indices.last().unwrap().0 - group_indices.first().unwrap().0 + 1;
            
            // If indices are dense enough, read the span; otherwise read full group
            if indices_in_group * 4 >= span || span <= 256 {
                // Dense or small span: read just the span
                let read_start = group_indices.first().unwrap().0;
                let read_len = span;
                let mut buf: Vec<u8> = vec![0u8; read_len * elem_size];
                mmap_cache.read_at(file, &mut buf, index.data_offset + header_size + (read_start * elem_size) as u64)?;
                
                for (idx, orig_pos) in group_indices {
                    let offset = idx - read_start;
                    let val: T = unsafe { std::ptr::read(buf.as_ptr().add(offset * elem_size) as *const T) };
                    result[orig_pos] = val;
                }
            } else {
                // Sparse: read individual values (but they're sorted so still sequential-ish)
                let mut buf = [0u8; 8];
                for (idx, orig_pos) in group_indices {
                    mmap_cache.read_at(file, &mut buf[..elem_size], index.data_offset + header_size + (idx * elem_size) as u64)?;
                    let val: T = unsafe { std::ptr::read(buf.as_ptr() as *const T) };
                    result[orig_pos] = val;
                }
            }
        }
        
        Ok(result)
    }

    fn read_column_scattered_mmap(
        &self,
        mmap_cache: &mut MmapCache,
        file: &File,
        index: &ColumnIndexEntry,
        dtype: ColumnType,
        row_indices: &[usize],
        _total_rows: usize,
    ) -> io::Result<ColumnData> {
        // ColumnData format has an 8-byte count header
        const HEADER_SIZE: u64 = 8;
        
        match dtype {
            ColumnType::Int64 | ColumnType::Int8 | ColumnType::Int16 | ColumnType::Int32 |
            ColumnType::UInt8 | ColumnType::UInt16 | ColumnType::UInt32 | ColumnType::UInt64 |
            ColumnType::Timestamp | ColumnType::Date => {
                Self::read_numeric_scattered_optimized::<i64>(mmap_cache, file, index, row_indices, HEADER_SIZE)
                    .map(ColumnData::Int64)
            }
            ColumnType::Float64 | ColumnType::Float32 => {
                Self::read_numeric_scattered_optimized::<f64>(mmap_cache, file, index, row_indices, HEADER_SIZE)
                    .map(ColumnData::Float64)
            }
            ColumnType::String | ColumnType::Binary => {
                // Optimized scattered read for variable-length types
                self.read_variable_column_scattered_mmap(mmap_cache, file, index, dtype, row_indices)
            }
            ColumnType::Bool => {
                // Bool is stored as packed bits: [count:u64][packed_bits...]
                // Read the packed bits and extract specific indices
                let packed_len = (index.data_length as usize - 8 + 7) / 8;
                let mut packed = vec![0u8; packed_len.max(1)];
                if packed_len > 0 {
                    mmap_cache.read_at(file, &mut packed, index.data_offset + HEADER_SIZE)?;
                }
                
                // Extract the specific bits for requested indices
                let mut result_packed = vec![0u8; (row_indices.len() + 7) / 8];
                for (result_idx, &src_idx) in row_indices.iter().enumerate() {
                    let src_byte = src_idx / 8;
                    let src_bit = src_idx % 8;
                    let bit_value = if src_byte < packed.len() {
                        (packed[src_byte] >> src_bit) & 1
                    } else {
                        0
                    };
                    
                    let dst_byte = result_idx / 8;
                    let dst_bit = result_idx % 8;
                    if bit_value == 1 {
                        result_packed[dst_byte] |= 1 << dst_bit;
                    }
                }
                
                Ok(ColumnData::Bool { data: result_packed, len: row_indices.len() })
            }
            ColumnType::StringDict => {
                // Native dictionary-encoded string scattered read
                self.read_string_dict_column_scattered_mmap(mmap_cache, file, index, row_indices)
            }
            ColumnType::Null => {
                // Null column - return empty Int64 as placeholder
                Ok(ColumnData::Int64(vec![0i64; row_indices.len()]))
            }
        }
    }

    /// Mmap-level string equality scan: find rows where col_name = target_value.
    /// Scans raw bytes without creating Arrow arrays. Returns global row indices of matches.
    pub fn scan_string_filter_mmap(&self, col_name: &str, target: &str, limit: Option<usize>) -> io::Result<Option<Vec<usize>>> {
        let footer = match self.get_or_load_footer()? {
            Some(f) => f,
            None => return Ok(None),
        };
        let schema = &footer.schema;
        let col_idx = match schema.get_index(col_name) {
            Some(i) => i,
            None => return Ok(None),
        };
        let col_type = schema.columns[col_idx].1;
        if !matches!(col_type, ColumnType::String | ColumnType::StringDict) { return Ok(None); }
        let col_count = schema.column_count();
        let file_guard = self.file.read();
        let file = file_guard.as_ref()
            .ok_or_else(|| err_not_conn("File not open for string scan"))?;
        let mut mmap_guard = self.mmap_cache.write();
        let mmap_ref = mmap_guard.get_or_create(file)?;
        let target_bytes = target.as_bytes();
        let max_matches = limit.unwrap_or(usize::MAX);
        let mut matches: Vec<usize> = Vec::new();
        let mut global_row_offset: usize = 0;
        // Build the memmem Finder once (precomputes Boyer-Moore table from target_bytes)
        let memmem_finder = memchr::memmem::Finder::new(target_bytes);

        // ── PARALLEL FAST PATH ───────────────────────────────────────────────
        // For no-limit StringDict scans on uncompressed+RCIX data, scan all RGs
        // in parallel using Rayon to exploit all CPU cores simultaneously.
        if limit.is_none() && footer.row_groups.len() > 1
            && matches!(col_type, ColumnType::StringDict)
        {
            // Check whether every RG qualifies for the parallel fast path
            let all_fast = footer.row_groups.iter().enumerate().all(|(rg_i, rg_meta)| {
                let rg_end = (rg_meta.offset + rg_meta.data_size) as usize;
                if rg_end > mmap_ref.len() { return false; }
                let rg_bytes = &mmap_ref[rg_meta.offset as usize..rg_end];
                let compress_flag = rg_bytes.get(28).copied().unwrap_or(RG_COMPRESS_NONE);
                let enc_ver = rg_bytes.get(29).copied().unwrap_or(0);
                compress_flag == RG_COMPRESS_NONE && enc_ver >= 1
                    && footer.col_offsets.get(rg_i).map_or(false, |v| v.len() > col_idx)
            });

            if all_fast {
                // Cast pointer to usize (Send+Sync) — safe because mmap_guard keeps Mmap
                // alive for the entire scope and all parallel tasks are read-only.
                let mmap_ptr: usize = mmap_ref.as_ptr() as usize;
                let mmap_len: usize = mmap_ref.len();

                // Build per-RG scan descriptors upfront
                struct RgDesc {
                    rg_offset: usize, rg_data_size: usize, rg_rows: usize,
                    global_off: usize, col_rcix: usize, has_deletes: bool,
                }
                let mut rg_descs: Vec<RgDesc> = Vec::with_capacity(footer.row_groups.len());
                let mut off = 0usize;
                for (rg_i, rg_meta) in footer.row_groups.iter().enumerate() {
                    rg_descs.push(RgDesc {
                        rg_offset: rg_meta.offset as usize,
                        rg_data_size: rg_meta.data_size as usize,
                        rg_rows: rg_meta.row_count as usize,
                        global_off: off,
                        col_rcix: footer.col_offsets[rg_i][col_idx] as usize,
                        has_deletes: rg_meta.deletion_count > 0,
                    });
                    off += rg_meta.row_count as usize;
                }

                let target_len = target_bytes.len();
                let all_rg_matches: Vec<Vec<usize>> = rg_descs.par_iter().map(|desc| {
                    let mmap = unsafe { std::slice::from_raw_parts(mmap_ptr as *const u8, mmap_len) };
                    let rg_end = desc.rg_offset + desc.rg_data_size;
                    if rg_end > mmap.len() || rg_end < desc.rg_offset + 32 { return vec![]; }
                    let rg_bytes = &mmap[desc.rg_offset..rg_end];
                    let body = &rg_bytes[32..];
                    let rg_rows = desc.rg_rows;
                    let null_bitmap_len = (rg_rows + 7) / 8;
                    let del_vec_len = null_bitmap_len;
                    let del_start = rg_rows * 8;

                    let col_off = desc.col_rcix;
                    if col_off + null_bitmap_len > body.len() { return vec![]; }
                    let col_bytes = &body[col_off + null_bitmap_len..];
                    if col_bytes.is_empty() { return vec![]; }
                    let encoding = col_bytes[0];
                    if encoding != COL_ENCODING_PLAIN { return vec![]; }
                    let data = &col_bytes[1..];
                    if data.len() < 16 { return vec![]; }

                    let row_count = u64::from_le_bytes(data[0..8].try_into().unwrap_or([0;8])) as usize;
                    let dict_size = u64::from_le_bytes(data[8..16].try_into().unwrap_or([0;8])) as usize;
                    if dict_size == 0 { return vec![]; }
                    let indices_start = 16usize;
                    let indices_len = row_count * 4;
                    let dict_off_start = indices_start + indices_len;
                    let dict_offsets_len = dict_size * 4;
                    let dict_data_len_off = dict_off_start + dict_offsets_len;
                    if dict_data_len_off + 8 > data.len() { return vec![]; }
                    let dict_data_len = u64::from_le_bytes(
                        data[dict_data_len_off..dict_data_len_off+8].try_into().unwrap_or([0;8])
                    ) as usize;
                    let dict_data_start = dict_data_len_off + 8;

                    let dict_offsets: &[u32] = unsafe {
                        std::slice::from_raw_parts(
                            data[dict_off_start..].as_ptr() as *const u32, dict_size)
                    };
                    let indices: &[u32] = unsafe {
                        std::slice::from_raw_parts(
                            data[indices_start..].as_ptr() as *const u32, row_count)
                    };

                    // SIMD search target in raw dict data
                    let raw_end = (dict_data_start + dict_data_len).min(data.len());
                    let raw_dict = &data[dict_data_start..raw_end];
                    let finder = memchr::memmem::Finder::new(target_bytes);
                    let mut target_dict_idx: Option<u32> = None;
                    let mut search_from = 0usize;
                    while let Some(rel) = finder.find(&raw_dict[search_from..]) {
                        let abs = search_from + rel;
                        if let Ok(di) = dict_offsets.binary_search(&(abs as u32)) {
                            let de = if di + 1 < dict_size { dict_offsets[di+1] as usize } else { dict_data_len };
                            if de - abs == target_len {
                                target_dict_idx = Some((di + 1) as u32);
                                break;
                            }
                        }
                        search_from += rel + 1;
                        if search_from >= raw_dict.len() { break; }
                    }

                    let Some(tdi) = target_dict_idx else { return vec![]; };

                    let n = row_count.min(rg_rows);
                    let mut local: Vec<usize> = Vec::new();
                    if !desc.has_deletes {
                        for i in 0..n {
                            if indices[i] == tdi { local.push(desc.global_off + i); }
                        }
                    } else {
                        if del_start + del_vec_len > body.len() { return local; }
                        let del_bytes = &body[del_start..del_start + del_vec_len];
                        for i in 0..n {
                            if (del_bytes[i/8] >> (i%8)) & 1 == 1 { continue; }
                            if indices[i] == tdi { local.push(desc.global_off + i); }
                        }
                    }
                    local
                }).collect();

                // Merge results in RG order (already ordered since RGs are enumerated in order)
                matches = all_rg_matches.into_iter().flatten().collect();
                return Ok(Some(matches));
            }
        }
        // ── END PARALLEL FAST PATH ───────────────────────────────────────────

        for (rg_i, rg_meta) in footer.row_groups.iter().enumerate() {
            if matches.len() >= max_matches { break; }
            let rg_rows = rg_meta.row_count as usize;
            if rg_rows == 0 { global_row_offset += rg_rows; continue; }
            let rg_end = (rg_meta.offset + rg_meta.data_size) as usize;
            if rg_end > mmap_ref.len() { return Err(err_data("RG extends past EOF")); }
            let rg_bytes = &mmap_ref[rg_meta.offset as usize .. rg_end];
            let compress_flag = if rg_bytes.len() >= 32 { rg_bytes[28] } else { RG_COMPRESS_NONE };
            let encoding_version = if rg_bytes.len() >= 32 { rg_bytes[29] } else { 0 };
            let decompressed = decompress_rg_body(compress_flag, &rg_bytes[32..])?;
            let body: &[u8] = decompressed.as_deref().unwrap_or(&rg_bytes[32..]);
            let del_vec_len = (rg_rows + 7) / 8;
            let del_start = rg_rows * 8;
            if del_start + del_vec_len > body.len() { return Err(err_data("RG del vec truncated")); }
            let del_bytes = &body[del_start..del_start + del_vec_len];
            let has_deletes = rg_meta.deletion_count > 0;
            let null_bitmap_len = (rg_rows + 7) / 8;

            // Zone-map skip: if RG has a string-length zone map for this column,
            // skip the entire RG when target_len is outside [min_len, max_len].
            // This eliminates scanning 15/16 RGs for typical queries (e.g. 9-char target
            // in a dataset where only RG0 has 9-char strings).
            if let Some(zmaps) = footer.zone_maps.get(rg_i) {
                if let Some(zm) = zmaps.iter().find(|z| z.col_idx as usize == col_idx && !z.is_float) {
                    let tlen = target_bytes.len() as i64;
                    if tlen < zm.min_bits || tlen > zm.max_bits {
                        global_row_offset += rg_rows;
                        continue;
                    }
                }
            }

            // RCIX fast path: jump directly to target column without scanning all columns
            let rcix = if compress_flag == RG_COMPRESS_NONE && encoding_version >= 1 {
                footer.col_offsets.get(rg_i).filter(|v| v.len() > col_idx)
            } else { None };

            if let Some(rcix) = rcix {
                let col_off = rcix[col_idx] as usize;
                if col_off + null_bitmap_len > body.len() { global_row_offset += rg_rows; continue; }
                let null_bytes = &body[col_off..col_off + null_bitmap_len];
                let ct = schema.columns[col_idx].1;
                let col_bytes = &body[col_off + null_bitmap_len..];
                {
                    let enc_offset = if encoding_version >= 1 { 1 } else { 0 };
                    let encoding = if encoding_version >= 1 { col_bytes[0] } else { COL_ENCODING_PLAIN };
                    let data = &col_bytes[enc_offset..];

                    if encoding == COL_ENCODING_PLAIN && matches!(ct, ColumnType::String) && data.len() >= 8 {
                        let count = u64::from_le_bytes(data[0..8].try_into().unwrap()) as usize;
                        let all_offsets_len = (count + 1) * 4;
                        if 8 + all_offsets_len <= data.len() {
                            let data_len_off = 8 + all_offsets_len;
                            if data_len_off + 8 <= data.len() {
                                let data_start = data_len_off + 8;
                                // FAST: cast offset bytes to &[u32] slice (avoids 2M u32::from_le_bytes calls)
                                let offsets: &[u32] = unsafe {
                                    std::slice::from_raw_parts(
                                        data[8..].as_ptr() as *const u32,
                                        count + 1,
                                    )
                                };
                                let target_len = target_bytes.len();
                                let n = count.min(rg_rows);
                                // FAST: memmem scan raw string data + binary search boundary check
                                // Replaces O(n) sequential scan with O(scan + k·log n) where k = rare hits
                                let data_len_val = u64::from_le_bytes(
                                    data[data_len_off..data_len_off+8].try_into().unwrap_or([0;8])
                                ) as usize;
                                let raw_end = (data_start + data_len_val).min(data.len());
                                let raw_str = &data[data_start..raw_end];
                                let mut search_from = 0usize;
                                while let Some(rel) = memmem_finder.find(&raw_str[search_from..]) {
                                    if matches.len() >= max_matches { break; }
                                    let abs = search_from + rel;
                                    // Binary search: find if abs is a valid string start offset
                                    if let Ok(di) = offsets[..count].binary_search(&(abs as u32)) {
                                        let end_off = offsets[di + 1] as usize;
                                        if end_off - abs == target_len && di < n {
                                            // Verify not deleted / null
                                            let skip = if has_deletes {
                                                (del_bytes[di / 8] >> (di % 8)) & 1 == 1
                                            } else { false };
                                            if !skip && (null_bytes[di / 8] >> (di % 8)) & 1 == 0 {
                                                matches.push(global_row_offset + di);
                                            }
                                        }
                                    }
                                    search_from += rel + 1;
                                    if search_from >= raw_str.len() { break; }
                                }
                            }
                        }
                    } else if encoding != COL_ENCODING_PLAIN && matches!(ct, ColumnType::String) {
                        // Non-PLAIN encoded String: decode then scan
                        let (col_data, _consumed) = if encoding_version >= 1 {
                            read_column_encoded(col_bytes, ct)?
                        } else {
                            ColumnData::from_bytes_typed(col_bytes, ct)?
                        };
                        if let ColumnData::String { offsets, data: str_data } = &col_data {
                            let count = offsets.len().saturating_sub(1);
                            for i in 0..count.min(rg_rows) {
                                if matches.len() >= max_matches { break; }
                                if has_deletes && (del_bytes[i / 8] >> (i % 8)) & 1 == 1 { continue; }
                                if (null_bytes[i / 8] >> (i % 8)) & 1 == 1 { continue; }
                                let s = offsets[i] as usize;
                                let e = offsets[i + 1] as usize;
                                if e - s == target_bytes.len() && &str_data[s..e] == target_bytes {
                                    matches.push(global_row_offset + i);
                                }
                            }
                        }
                    } else if encoding == COL_ENCODING_PLAIN && matches!(ct, ColumnType::StringDict) && data.len() >= 16 {
                        // StringDict: find target in dict, then scan indices
                        let row_count = u64::from_le_bytes(data[0..8].try_into().unwrap()) as usize;
                        let dict_size = u64::from_le_bytes(data[8..16].try_into().unwrap()) as usize;
                        let indices_start = 16usize;
                        let indices_len = row_count * 4;
                        let dict_off_start = indices_start + indices_len;
                        let dict_offsets_len = dict_size * 4;
                        let dict_data_len_off = dict_off_start + dict_offsets_len;
                        if dict_data_len_off + 8 <= data.len() {
                            let dict_data_len = u64::from_le_bytes(data[dict_data_len_off..dict_data_len_off+8].try_into().unwrap()) as usize;
                            let dict_data_start = dict_data_len_off + 8;
                            // FAST: cast to &[u32] slices
                            let dict_offsets: &[u32] = unsafe {
                                std::slice::from_raw_parts(
                                    data[dict_off_start..].as_ptr() as *const u32,
                                    dict_size,
                                )
                            };
                            let indices: &[u32] = unsafe {
                                std::slice::from_raw_parts(
                                    data[indices_start..].as_ptr() as *const u32,
                                    row_count,
                                )
                            };
                            // Find target in dictionary using SIMD memmem + binary search boundary check
                            let target_len = target_bytes.len();
                            let mut target_dict_idx: Option<u32> = None;
                            let raw_end = (dict_data_start + dict_data_len).min(data.len());
                            let raw_dict = &data[dict_data_start..raw_end];
                            let mut search_from = 0usize;
                            while let Some(rel) = memmem_finder.find(&raw_dict[search_from..]) {
                                let abs = search_from + rel;
                                // Verify exact boundary: binary search for abs in dict_offsets
                                if let Ok(di) = dict_offsets.binary_search(&(abs as u32)) {
                                    let de = if di + 1 < dict_size { dict_offsets[di + 1] as usize } else { dict_data_len };
                                    if de - abs == target_len {
                                        target_dict_idx = Some((di + 1) as u32);
                                        break;
                                    }
                                }
                                search_from += rel + 1;
                                if search_from >= raw_dict.len() { break; }
                            }
                            if let Some(tdi) = target_dict_idx {
                                let n = row_count.min(rg_rows);
                                if !has_deletes {
                                    for i in 0..n {
                                        if matches.len() >= max_matches { break; }
                                        if indices[i] == tdi { matches.push(global_row_offset + i); }
                                    }
                                } else {
                                    for i in 0..n {
                                        if matches.len() >= max_matches { break; }
                                        if (del_bytes[i / 8] >> (i % 8)) & 1 == 1 { continue; }
                                        if indices[i] == tdi { matches.push(global_row_offset + i); }
                                    }
                                }
                            }
                        }
                    }
                }
            } else {
                // Fallback: sequential pos scan for compressed or pre-RCIX row groups
                let mut pos = del_start + del_vec_len;
                for ci in 0..col_count {
                    if pos + null_bitmap_len > body.len() { break; }
                    let null_bytes = &body[pos..pos + null_bitmap_len];
                    pos += null_bitmap_len;
                    let ct = schema.columns[ci].1;
                    if ci == col_idx {
                        let col_bytes = &body[pos..];
                        let enc_offset = if encoding_version >= 1 { 1 } else { 0 };
                        let encoding = if encoding_version >= 1 && !col_bytes.is_empty() { col_bytes[0] } else { COL_ENCODING_PLAIN };
                        let data = if enc_offset <= col_bytes.len() { &col_bytes[enc_offset..] } else { &[] };
                        if encoding == COL_ENCODING_PLAIN && matches!(ct, ColumnType::String) && data.len() >= 8 {
                            let count = u64::from_le_bytes(data[0..8].try_into().unwrap()) as usize;
                            let all_offsets_len = (count + 1) * 4;
                            if 8 + all_offsets_len <= data.len() {
                                let data_len_off = 8 + all_offsets_len;
                                if data_len_off + 8 <= data.len() {
                                    let data_start = data_len_off + 8;
                                    let offsets: &[u32] = unsafe { std::slice::from_raw_parts(data[8..].as_ptr() as *const u32, count + 1) };
                                    let tlen = target_bytes.len();
                                    for i in 0..count.min(rg_rows) {
                                        if matches.len() >= max_matches { break; }
                                        if has_deletes && (del_bytes[i / 8] >> (i % 8)) & 1 == 1 { continue; }
                                        let s = offsets[i] as usize; let e = offsets[i + 1] as usize;
                                        if e - s == tlen && data_start + e <= data.len() && &data[data_start + s..data_start + e] == target_bytes {
                                            matches.push(global_row_offset + i);
                                        }
                                    }
                                }
                            }
                        } else if encoding == COL_ENCODING_PLAIN && matches!(ct, ColumnType::StringDict) && data.len() >= 16 {
                            let row_count = u64::from_le_bytes(data[0..8].try_into().unwrap()) as usize;
                            let dict_size = u64::from_le_bytes(data[8..16].try_into().unwrap()) as usize;
                            let dict_off_start = 16 + row_count * 4;
                            let dict_data_len_off = dict_off_start + dict_size * 4;
                            if dict_data_len_off + 8 <= data.len() {
                                let dict_data_len = u64::from_le_bytes(data[dict_data_len_off..dict_data_len_off+8].try_into().unwrap()) as usize;
                                let dict_data_start = dict_data_len_off + 8;
                                let dict_offsets: &[u32] = unsafe { std::slice::from_raw_parts(data[dict_off_start..].as_ptr() as *const u32, dict_size) };
                                let indices: &[u32] = unsafe { std::slice::from_raw_parts(data[16..].as_ptr() as *const u32, row_count) };
                                let tlen = target_bytes.len();
                                let mut tdi: Option<u32> = None;
                                for di in 0..dict_size {
                                    let ds = dict_offsets[di] as usize;
                                    let de = if di + 1 < dict_size { dict_offsets[di + 1] as usize } else { dict_data_len };
                                    if de - ds == tlen && dict_data_start + de <= data.len() && &data[dict_data_start + ds..dict_data_start + de] == target_bytes {
                                        tdi = Some((di + 1) as u32); break;
                                    }
                                }
                                if let Some(tdi) = tdi {
                                    for i in 0..row_count.min(rg_rows) {
                                        if matches.len() >= max_matches { break; }
                                        if has_deletes && (del_bytes[i / 8] >> (i % 8)) & 1 == 1 { continue; }
                                        if indices[i] == tdi { matches.push(global_row_offset + i); }
                                    }
                                }
                            }
                        }
                    }
                    let consumed = if encoding_version >= 1 {
                        skip_column_encoded(&body[pos..], ct)?
                    } else {
                        ColumnData::skip_bytes_typed(&body[pos..], ct)?
                    };
                    pos += consumed;
                }
            }
            global_row_offset += rg_rows;
        }
        drop(mmap_guard);
        drop(file_guard);
        Ok(Some(matches))
    }

    /// Mmap-level numeric range scan: find rows where col_name BETWEEN low AND high.
    /// Returns global row indices of matches.
    pub fn scan_numeric_range_mmap(&self, col_name: &str, low: f64, high: f64, limit: Option<usize>) -> io::Result<Option<Vec<usize>>> {
        let footer = match self.get_or_load_footer()? {
            Some(f) => f,
            None => return Ok(None),
        };
        let schema = &footer.schema;
        let col_idx = match schema.get_index(col_name) {
            Some(i) => i,
            None => return Ok(None),
        };
        let col_type = schema.columns[col_idx].1;
        let is_int = matches!(col_type, ColumnType::Int64 | ColumnType::Int8 | ColumnType::Int16 |
            ColumnType::Int32 | ColumnType::UInt8 | ColumnType::UInt16 | ColumnType::UInt32 |
            ColumnType::UInt64 | ColumnType::Timestamp | ColumnType::Date);
        let is_float = matches!(col_type, ColumnType::Float64 | ColumnType::Float32);
        if !is_int && !is_float { return Ok(None); }
        let col_count = schema.column_count();
        let file_guard = self.file.read();
        let file = file_guard.as_ref()
            .ok_or_else(|| err_not_conn("File not open for range scan"))?;
        let mut mmap_guard = self.mmap_cache.write();
        let mmap_ref = mmap_guard.get_or_create(file)?;
        let max_matches = limit.unwrap_or(usize::MAX);
        let mut matches: Vec<usize> = Vec::new();
        let mut global_row_offset: usize = 0;

        for (rg_i, rg_meta) in footer.row_groups.iter().enumerate() {
            if matches.len() >= max_matches { break; }
            let rg_rows = rg_meta.row_count as usize;
            if rg_rows == 0 { global_row_offset += rg_rows; continue; }

            // Zone map pruning: skip RG if filter range can't overlap
            if rg_i < footer.zone_maps.len() {
                if let Some(zm) = footer.zone_maps[rg_i].iter().find(|z| z.col_idx as usize == col_idx) {
                    let skip = if zm.is_float {
                        !zm.may_overlap_float_range(low, high)
                    } else {
                        !zm.may_overlap_int_range(low.ceil() as i64, high.floor() as i64)
                    };
                    if skip { global_row_offset += rg_rows; continue; }
                }
            }

            let rg_end = (rg_meta.offset + rg_meta.data_size) as usize;
            if rg_end > mmap_ref.len() { return Err(err_data("RG extends past EOF")); }
            let rg_bytes = &mmap_ref[rg_meta.offset as usize .. rg_end];
            let compress_flag = if rg_bytes.len() >= 32 { rg_bytes[28] } else { RG_COMPRESS_NONE };
            let encoding_version = if rg_bytes.len() >= 32 { rg_bytes[29] } else { 0 };
            let decompressed = decompress_rg_body(compress_flag, &rg_bytes[32..])?;
            let body: &[u8] = decompressed.as_deref().unwrap_or(&rg_bytes[32..]);
            let del_vec_len = (rg_rows + 7) / 8;
            let id_section = rg_rows * 8;
            if id_section + del_vec_len > body.len() { return Err(err_data("RG del vec truncated")); }
            let del_bytes = &body[id_section..id_section + del_vec_len];
            let has_deletes = rg_meta.deletion_count > 0;
            let null_bitmap_len = (rg_rows + 7) / 8;

            // RCIX fast path: jump directly to target column offset
            let rcix_available = rg_i < footer.col_offsets.len()
                && col_idx < footer.col_offsets[rg_i].len()
                && compress_flag == RG_COMPRESS_NONE;
            if rcix_available {
                let col_body_off = footer.col_offsets[rg_i][col_idx] as usize;
                if col_body_off + null_bitmap_len > body.len() { global_row_offset += rg_rows; continue; }
                let null_bytes = &body[col_body_off..col_body_off + null_bitmap_len];
                let data_start = col_body_off + null_bitmap_len;
                let col_bytes = &body[data_start..];
                let enc_offset = if encoding_version >= 1 { 1 } else { 0 };
                let encoding = if encoding_version >= 1 && !col_bytes.is_empty() { col_bytes[0] } else { COL_ENCODING_PLAIN };
                if encoding == COL_ENCODING_PLAIN && col_bytes.len() > enc_offset + 8 {
                    let payload = &col_bytes[enc_offset..];
                    let count = u64::from_le_bytes(payload[0..8].try_into().unwrap()) as usize;
                    let n = count.min(rg_rows).min((payload.len() - 8) / 8);
                    if is_int {
                        let low_i = low.ceil() as i64;
                        let high_i = high.floor() as i64;
                        let vals = unsafe { std::slice::from_raw_parts(payload[8..].as_ptr() as *const i64, n) };
                        for i in 0..n {
                            if matches.len() >= max_matches { break; }
                            if has_deletes && (del_bytes[i/8] >> (i%8)) & 1 == 1 { continue; }
                            if (null_bytes[i/8] >> (i%8)) & 1 == 1 { continue; }
                            if vals[i] >= low_i && vals[i] <= high_i { matches.push(global_row_offset + i); }
                        }
                    } else {
                        let vals = unsafe { std::slice::from_raw_parts(payload[8..].as_ptr() as *const f64, n) };
                        for i in 0..n {
                            if matches.len() >= max_matches { break; }
                            if has_deletes && (del_bytes[i/8] >> (i%8)) & 1 == 1 { continue; }
                            if (null_bytes[i/8] >> (i%8)) & 1 == 1 { continue; }
                            if vals[i] >= low && vals[i] <= high { matches.push(global_row_offset + i); }
                        }
                    }
                    global_row_offset += rg_rows;
                    continue;
                }
            }

            // Fallback: sequential column scan (compressed or no RCIX)
            let mut pos = id_section + del_vec_len;
            for ci in 0..col_count {
                if pos + null_bitmap_len > body.len() { break; }
                let null_bytes = &body[pos..pos + null_bitmap_len];
                pos += null_bitmap_len;
                let ct = schema.columns[ci].1;

                if ci == col_idx {
                    let col_bytes = &body[pos..];
                    let enc_offset = if encoding_version >= 1 { 1 } else { 0 };
                    let encoding = if encoding_version >= 1 { col_bytes[0] } else { COL_ENCODING_PLAIN };
                    let data_slice = &col_bytes[enc_offset..];

                    if encoding == COL_ENCODING_PLAIN && data_slice.len() >= 8 {
                        let count = u64::from_le_bytes(data_slice[0..8].try_into().unwrap()) as usize;
                        let values_start = 8usize;
                        let n = count.min(rg_rows);
                        if is_int {
                            let low_i = low.ceil() as i64;
                            let high_i = high.floor() as i64;
                            let vals: &[i64] = unsafe {
                                std::slice::from_raw_parts(
                                    data_slice[values_start..].as_ptr() as *const i64,
                                    n.min((data_slice.len() - values_start) / 8),
                                )
                            };
                            if !has_deletes {
                                for i in 0..vals.len() {
                                    if matches.len() >= max_matches { break; }
                                    if vals[i] >= low_i && vals[i] <= high_i { matches.push(global_row_offset + i); }
                                }
                            } else {
                                for i in 0..vals.len() {
                                    if matches.len() >= max_matches { break; }
                                    if (del_bytes[i / 8] >> (i % 8)) & 1 == 1 { continue; }
                                    if vals[i] >= low_i && vals[i] <= high_i { matches.push(global_row_offset + i); }
                                }
                            }
                        } else {
                            let vals: &[f64] = unsafe {
                                std::slice::from_raw_parts(
                                    data_slice[values_start..].as_ptr() as *const f64,
                                    n.min((data_slice.len() - values_start) / 8),
                                )
                            };
                            if !has_deletes {
                                for i in 0..vals.len() {
                                    if matches.len() >= max_matches { break; }
                                    if vals[i] >= low && vals[i] <= high { matches.push(global_row_offset + i); }
                                }
                            } else {
                                for i in 0..vals.len() {
                                    if matches.len() >= max_matches { break; }
                                    if (del_bytes[i / 8] >> (i % 8)) & 1 == 1 { continue; }
                                    if vals[i] >= low && vals[i] <= high { matches.push(global_row_offset + i); }
                                }
                            }
                        }
                    } else {
                        // Non-PLAIN encoding (RLE, BITPACK, etc.): decode then scan
                        let (col_data, _consumed) = if encoding_version >= 1 {
                            read_column_encoded(col_bytes, ct)?
                        } else {
                            ColumnData::from_bytes_typed(col_bytes, ct)?
                        };
                        match &col_data {
                            ColumnData::Int64(vals) => {
                                let low_i = low.ceil() as i64;
                                let high_i = high.floor() as i64;
                                for i in 0..vals.len().min(rg_rows) {
                                    if matches.len() >= max_matches { break; }
                                    if has_deletes && (del_bytes[i / 8] >> (i % 8)) & 1 == 1 { continue; }
                                    if (null_bytes[i / 8] >> (i % 8)) & 1 == 1 { continue; }
                                    if vals[i] >= low_i && vals[i] <= high_i { matches.push(global_row_offset + i); }
                                }
                            }
                            ColumnData::Float64(vals) => {
                                for i in 0..vals.len().min(rg_rows) {
                                    if matches.len() >= max_matches { break; }
                                    if has_deletes && (del_bytes[i / 8] >> (i % 8)) & 1 == 1 { continue; }
                                    if (null_bytes[i / 8] >> (i % 8)) & 1 == 1 { continue; }
                                    if vals[i] >= low && vals[i] <= high { matches.push(global_row_offset + i); }
                                }
                            }
                            _ => {}
                        }
                    }
                }
                let consumed = if encoding_version >= 1 {
                    skip_column_encoded(&body[pos..], ct)?
                } else {
                    ColumnData::skip_bytes_typed(&body[pos..], ct)?
                };
                pos += consumed;
            }
            global_row_offset += rg_rows;
        }
        drop(mmap_guard);
        drop(file_guard);
        Ok(Some(matches))
    }

    /// Direct mmap top-K scan: finds top-k row indices by a numeric column without materializing
    /// the full Arrow array. Uses RCIX + zone maps for O(N_rows) with O(k) heap in L1 cache.
    /// Returns Vec<(global_row_idx, value)> sorted in the requested order.
    pub fn scan_top_k_indices_mmap(
        &self,
        col_name: &str,
        k: usize,
        descending: bool,
    ) -> io::Result<Option<Vec<(usize, f64)>>> {
        if k == 0 { return Ok(Some(vec![])); }
        let footer = match self.get_or_load_footer()? {
            Some(f) => f,
            None => return Ok(None),
        };
        let schema = &footer.schema;
        let col_idx = match schema.get_index(col_name) {
            Some(i) => i,
            None => return Ok(None),
        };
        let col_type = schema.columns[col_idx].1;
        let is_int = matches!(col_type, ColumnType::Int64 | ColumnType::Int8 | ColumnType::Int16 |
            ColumnType::Int32 | ColumnType::UInt8 | ColumnType::UInt16 | ColumnType::UInt32 |
            ColumnType::UInt64 | ColumnType::Timestamp | ColumnType::Date);
        let is_float = matches!(col_type, ColumnType::Float64 | ColumnType::Float32);
        if !is_int && !is_float { return Ok(None); }

        let file_guard = self.file.read();
        let file = file_guard.as_ref()
            .ok_or_else(|| err_not_conn("File not open for top-k scan"))?;
        let mut mmap_guard = self.mmap_cache.write();
        let mmap_ref = mmap_guard.get_or_create(file)?;

        // heap: sorted Vec<(value, global_idx)>; descending → keep k largest
        let mut heap: Vec<(f64, usize)> = Vec::with_capacity(k + 1);
        let mut global_offset: usize = 0;

        for (rg_i, rg_meta) in footer.row_groups.iter().enumerate() {
            let rg_rows = rg_meta.row_count as usize;
            if rg_rows == 0 { global_offset += rg_rows; continue; }

            let rg_end = (rg_meta.offset + rg_meta.data_size) as usize;
            if rg_end > mmap_ref.len() { return Err(err_data("RG extends past EOF")); }
            let rg_bytes = &mmap_ref[rg_meta.offset as usize..rg_end];
            let compress_flag = if rg_bytes.len() >= 32 { rg_bytes[28] } else { RG_COMPRESS_NONE };
            let encoding_version = if rg_bytes.len() >= 32 { rg_bytes[29] } else { 0 };
            let decompressed = decompress_rg_body(compress_flag, &rg_bytes[32..])?;
            let body: &[u8] = decompressed.as_deref().unwrap_or(&rg_bytes[32..]);
            let id_section = rg_rows * 8;
            let del_vec_len = (rg_rows + 7) / 8;
            let null_bitmap_len = (rg_rows + 7) / 8;
            let has_deletes = rg_meta.deletion_count > 0;
            let del_bytes = if id_section + del_vec_len <= body.len() {
                &body[id_section..id_section + del_vec_len]
            } else { &[] };

            // Get pointer to column data via RCIX if available
            let col_bytes: &[u8] = if rg_i < footer.col_offsets.len()
                && col_idx < footer.col_offsets[rg_i].len()
                && compress_flag == RG_COMPRESS_NONE
            {
                let col_body_off = footer.col_offsets[rg_i][col_idx] as usize;
                let data_start = col_body_off + null_bitmap_len;
                if data_start > body.len() { global_offset += rg_rows; continue; }
                &body[data_start..]
            } else {
                // Fallback: sequential column scan
                let mut pos = id_section + del_vec_len;
                let mut found: &[u8] = &[];
                for ci in 0..schema.column_count() {
                    if pos + null_bitmap_len > body.len() { break; }
                    pos += null_bitmap_len;
                    if ci == col_idx { found = &body[pos..]; break; }
                    let consumed = if encoding_version >= 1 {
                        skip_column_encoded(&body[pos..], schema.columns[ci].1)?
                    } else {
                        ColumnData::skip_bytes_typed(&body[pos..], schema.columns[ci].1)?
                    };
                    pos += consumed;
                }
                found
            };

            if col_bytes.is_empty() { global_offset += rg_rows; continue; }

            let enc_offset = if encoding_version >= 1 { 1 } else { 0 };
            let encoding = if encoding_version >= 1 && !col_bytes.is_empty() { col_bytes[0] } else { COL_ENCODING_PLAIN };

            if encoding == COL_ENCODING_PLAIN && col_bytes.len() > enc_offset + 8 {
                let payload = &col_bytes[enc_offset..];
                let count = u64::from_le_bytes(payload[0..8].try_into().unwrap()) as usize;
                let n = count.min(rg_rows).min((payload.len() - 8) / 8);

                macro_rules! topk_scan {
                    ($vals:expr) => {{
                        if descending {
                            // Keep k largest: heap sorted descending, threshold = heap[k-1]
                            for i in 0..n {
                                if has_deletes && !del_bytes.is_empty() && (del_bytes[i/8] >> (i%8)) & 1 == 1 { continue; }
                                let val = $vals[i];
                                if heap.len() < k {
                                    let pos = heap.partition_point(|(v, _)| *v > val);
                                    heap.insert(pos, (val, global_offset + i));
                                } else if val > heap[k - 1].0 {
                                    let pos = heap.partition_point(|(v, _)| *v > val);
                                    heap.insert(pos, (val, global_offset + i));
                                    heap.pop();
                                }
                            }
                        } else {
                            // Keep k smallest: heap sorted ascending, threshold = heap[k-1]
                            for i in 0..n {
                                if has_deletes && !del_bytes.is_empty() && (del_bytes[i/8] >> (i%8)) & 1 == 1 { continue; }
                                let val = $vals[i];
                                if heap.len() < k {
                                    let pos = heap.partition_point(|(v, _)| *v < val);
                                    heap.insert(pos, (val, global_offset + i));
                                } else if val < heap[k - 1].0 {
                                    let pos = heap.partition_point(|(v, _)| *v < val);
                                    heap.insert(pos, (val, global_offset + i));
                                    heap.pop();
                                }
                            }
                        }
                    }};
                }

                if is_float {
                    let vals = unsafe { std::slice::from_raw_parts(payload[8..].as_ptr() as *const f64, n) };
                    topk_scan!(vals);
                } else {
                    let vals = unsafe { std::slice::from_raw_parts(payload[8..].as_ptr() as *const i64, n) };
                    let fvals: Vec<f64> = vals.iter().map(|&v| v as f64).collect();
                    topk_scan!(fvals);
                }
            } else {
                // Non-PLAIN: decode and scan
                let (col_data, _) = if encoding_version >= 1 {
                    read_column_encoded(col_bytes, col_type)?
                } else {
                    ColumnData::from_bytes_typed(col_bytes, col_type)?
                };
                let fvals: Vec<f64> = match &col_data {
                    ColumnData::Float64(v) => v.iter().map(|&x| x).collect(),
                    ColumnData::Int64(v) => v.iter().map(|&x| x as f64).collect(),
                    _ => { global_offset += rg_rows; continue; }
                };
                let n = fvals.len().min(rg_rows);
                macro_rules! topk_scan2 {
                    ($vals:expr) => {{
                        if descending {
                            for i in 0..n {
                                if has_deletes && !del_bytes.is_empty() && (del_bytes[i/8] >> (i%8)) & 1 == 1 { continue; }
                                let val = $vals[i];
                                if heap.len() < k {
                                    let pos = heap.partition_point(|(v, _)| *v > val);
                                    heap.insert(pos, (val, global_offset + i));
                                } else if val > heap[k - 1].0 {
                                    let pos = heap.partition_point(|(v, _)| *v > val);
                                    heap.insert(pos, (val, global_offset + i));
                                    heap.pop();
                                }
                            }
                        } else {
                            for i in 0..n {
                                if has_deletes && !del_bytes.is_empty() && (del_bytes[i/8] >> (i%8)) & 1 == 1 { continue; }
                                let val = $vals[i];
                                if heap.len() < k {
                                    let pos = heap.partition_point(|(v, _)| *v < val);
                                    heap.insert(pos, (val, global_offset + i));
                                } else if val < heap[k - 1].0 {
                                    let pos = heap.partition_point(|(v, _)| *v < val);
                                    heap.insert(pos, (val, global_offset + i));
                                    heap.pop();
                                }
                            }
                        }
                    }};
                }
                topk_scan2!(fvals);
            }
            global_offset += rg_rows;
        }
        Ok(Some(heap.into_iter().map(|(v, i)| (i, v)).collect()))
    }

    /// Compute numeric column aggregates directly from mmap without Arrow arrays.
    /// Returns (count, sum, min, max) for the specified column.
    /// Only works for Int64/Float64 columns in V4 mmap-only mode.
    pub fn compute_column_stats_mmap(&self, col_name: &str) -> io::Result<Option<(u64, f64, f64, f64)>> {
        let footer = match self.get_or_load_footer()? {
            Some(f) => f,
            None => return Ok(None),
        };
        let schema = &footer.schema;
        let col_idx = match schema.get_index(col_name) {
            Some(i) => i,
            None => return Ok(None),
        };
        let col_type = schema.columns[col_idx].1;
        let is_int = matches!(col_type, ColumnType::Int64 | ColumnType::Int8 | ColumnType::Int16 |
            ColumnType::Int32 | ColumnType::UInt8 | ColumnType::UInt16 | ColumnType::UInt32 |
            ColumnType::UInt64 | ColumnType::Timestamp | ColumnType::Date);
        let is_float = matches!(col_type, ColumnType::Float64 | ColumnType::Float32);
        if !is_int && !is_float { return Ok(None); }

        let col_count = schema.column_count();
        let file_guard = self.file.read();
        let file = file_guard.as_ref()
            .ok_or_else(|| err_not_conn("File not open for mmap agg"))?;
        let mut mmap_guard = self.mmap_cache.write();
        let mmap_ref = mmap_guard.get_or_create(file)?;

        let mut total_count: u64 = 0;
        let mut total_sum: f64 = 0.0;
        let mut total_min: f64 = f64::INFINITY;
        let mut total_max: f64 = f64::NEG_INFINITY;

        for rg_meta in &footer.row_groups {
            let rg_rows = rg_meta.row_count as usize;
            if rg_rows == 0 { continue; }
            let rg_end = (rg_meta.offset + rg_meta.data_size) as usize;
            if rg_end > mmap_ref.len() { return Err(err_data("RG extends past EOF")); }
            let rg_bytes = &mmap_ref[rg_meta.offset as usize .. rg_end];
            let compress_flag = if rg_bytes.len() >= 32 { rg_bytes[28] } else { RG_COMPRESS_NONE };
            let encoding_version = if rg_bytes.len() >= 32 { rg_bytes[29] } else { 0 };
            let decompressed = decompress_rg_body(compress_flag, &rg_bytes[32..])?;
            let body: &[u8] = decompressed.as_deref().unwrap_or(&rg_bytes[32..]);
            let mut pos = rg_rows * 8; // skip IDs
            let del_vec_len = (rg_rows + 7) / 8;
            if pos + del_vec_len > body.len() { return Err(err_data("RG del vec truncated")); }
            let del_bytes = &body[pos..pos + del_vec_len];
            let has_deletes = rg_meta.deletion_count > 0;
            pos += del_vec_len;

            let null_bitmap_len = (rg_rows + 7) / 8;
            for ci in 0..col_count {
                if pos + null_bitmap_len > body.len() { break; }
                let null_bytes = &body[pos..pos + null_bitmap_len];
                pos += null_bitmap_len;

                let ct = schema.columns[ci].1;
                if ci == col_idx {
                    let col_bytes = &body[pos..];
                    let enc_offset = if encoding_version >= 1 { 1 } else { 0 };
                    let encoding = if encoding_version >= 1 { col_bytes[0] } else { COL_ENCODING_PLAIN };

                    if encoding == COL_ENCODING_PLAIN {
                        let data = &col_bytes[enc_offset..];
                        if data.len() >= 8 {
                            let count = u64::from_le_bytes(data[0..8].try_into().unwrap()) as usize;
                            let values_start = 8usize;
                            if is_int {
                                for i in 0..count.min(rg_rows) {
                                    if has_deletes && (del_bytes[i / 8] >> (i % 8)) & 1 == 1 { continue; }
                                    if (null_bytes[i / 8] >> (i % 8)) & 1 == 1 { continue; }
                                    let off = values_start + i * 8;
                                    if off + 8 > data.len() { break; }
                                    let v = i64::from_le_bytes(data[off..off+8].try_into().unwrap()) as f64;
                                    total_count += 1;
                                    total_sum += v;
                                    if v < total_min { total_min = v; }
                                    if v > total_max { total_max = v; }
                                }
                            } else {
                                for i in 0..count.min(rg_rows) {
                                    if has_deletes && (del_bytes[i / 8] >> (i % 8)) & 1 == 1 { continue; }
                                    if (null_bytes[i / 8] >> (i % 8)) & 1 == 1 { continue; }
                                    let off = values_start + i * 8;
                                    if off + 8 > data.len() { break; }
                                    let v = f64::from_le_bytes(data[off..off+8].try_into().unwrap());
                                    if !v.is_nan() {
                                        total_count += 1;
                                        total_sum += v;
                                        if v < total_min { total_min = v; }
                                        if v > total_max { total_max = v; }
                                    }
                                }
                            }
                        }
                    } else {
                        // Encoded column: fallback to full decode
                        let (col_data, _) = if encoding_version >= 1 {
                            read_column_encoded(col_bytes, ct)?
                        } else {
                            ColumnData::from_bytes_typed(col_bytes, ct)?
                        };
                        match &col_data {
                            ColumnData::Int64(vals) => {
                                for (i, &v) in vals.iter().enumerate() {
                                    if has_deletes && i < rg_rows && (del_bytes[i / 8] >> (i % 8)) & 1 == 1 { continue; }
                                    if i < rg_rows && (null_bytes[i / 8] >> (i % 8)) & 1 == 1 { continue; }
                                    let fv = v as f64;
                                    total_count += 1;
                                    total_sum += fv;
                                    if fv < total_min { total_min = fv; }
                                    if fv > total_max { total_max = fv; }
                                }
                            }
                            ColumnData::Float64(vals) => {
                                for (i, &v) in vals.iter().enumerate() {
                                    if has_deletes && i < rg_rows && (del_bytes[i / 8] >> (i % 8)) & 1 == 1 { continue; }
                                    if i < rg_rows && (null_bytes[i / 8] >> (i % 8)) & 1 == 1 { continue; }
                                    if !v.is_nan() {
                                        total_count += 1;
                                        total_sum += v;
                                        if v < total_min { total_min = v; }
                                        if v > total_max { total_max = v; }
                                    }
                                }
                            }
                            _ => {}
                        }
                    }
                }
                // Skip column data to advance pos
                let consumed = if encoding_version >= 1 {
                    skip_column_encoded(&body[pos..], ct)?
                } else {
                    ColumnData::skip_bytes_typed(&body[pos..], ct)?
                };
                pos += consumed;
            }
        }

        if total_count == 0 {
            return Ok(Some((0, 0.0, 0.0, 0.0)));
        }
        Ok(Some((total_count, total_sum, total_min, total_max)))
    }

    /// Extract specific rows by global indices from mmap, returning an Arrow RecordBatch.
    /// Navigates each RG body once and extracts only values at target positions.
    /// Much faster than full-table scan + take for small result sets.
    pub fn extract_rows_by_indices_to_arrow(&self, indices: &[usize]) -> io::Result<Option<arrow::record_batch::RecordBatch>> {
        use arrow::array::{ArrayRef, Int64Array, Float64Array, StringBuilder, BooleanBuilder};
        use arrow::datatypes::{Schema, Field, DataType as ArrowDataType};
        use std::sync::Arc;

        if indices.is_empty() {
            return Ok(Some(arrow::record_batch::RecordBatch::new_empty(Arc::new(Schema::empty()))));
        }

        let footer = match self.get_or_load_footer()? {
            Some(f) => f,
            None => return Ok(None),
        };
        let schema = &footer.schema;
        let col_count = schema.column_count();

        // Build RG cumulative bounds
        let mut rg_bounds: Vec<(usize, usize)> = Vec::new();
        let mut cumulative = 0usize;
        for rg in &footer.row_groups {
            let n = rg.row_count as usize;
            rg_bounds.push((cumulative, cumulative + n));
            cumulative += n;
        }

        // Group indices by RG: (output_position, local_index_within_rg)
        let mut rg_local_indices: Vec<Vec<(usize, usize)>> = vec![Vec::new(); footer.row_groups.len()];
        for (out_idx, &global_idx) in indices.iter().enumerate() {
            for (rg_i, &(start, end)) in rg_bounds.iter().enumerate() {
                if global_idx >= start && global_idx < end {
                    rg_local_indices[rg_i].push((out_idx, global_idx - start));
                    break;
                }
            }
        }

        let n_out = indices.len();
        let mut out_ids: Vec<i64> = vec![0i64; n_out];
        let mut out_cols: Vec<Vec<Option<crate::data::Value>>> = (0..col_count).map(|_| vec![None; n_out]).collect();

        let file_guard = self.file.read();
        let file = file_guard.as_ref()
            .ok_or_else(|| err_not_conn("File not open for batch extract"))?;
        let mut mmap_guard = self.mmap_cache.write();
        let mmap_ref = mmap_guard.get_or_create(file)?;

        for (rg_i, local_pairs) in rg_local_indices.iter().enumerate() {
            if local_pairs.is_empty() { continue; }
            let rg_meta = &footer.row_groups[rg_i];
            let rg_rows = rg_meta.row_count as usize;
            let rg_end = (rg_meta.offset + rg_meta.data_size) as usize;
            if rg_end > mmap_ref.len() { return Err(err_data("RG extends past EOF")); }
            let rg_bytes = &mmap_ref[rg_meta.offset as usize .. rg_end];
            let compress_flag = if rg_bytes.len() >= 32 { rg_bytes[28] } else { RG_COMPRESS_NONE };
            let encoding_version = if rg_bytes.len() >= 32 { rg_bytes[29] } else { 0 };
            let decompressed = decompress_rg_body(compress_flag, &rg_bytes[32..])?;
            let body: &[u8] = decompressed.as_deref().unwrap_or(&rg_bytes[32..]);
            let null_bitmap_len = (rg_rows + 7) / 8;

            // Extract IDs for target rows
            if rg_rows * 8 <= body.len() {
                for &(out_idx, local_idx) in local_pairs {
                    let id_off = local_idx * 8;
                    if id_off + 8 <= rg_rows * 8 {
                        out_ids[out_idx] = i64::from_le_bytes(body[id_off..id_off+8].try_into().unwrap());
                    }
                }
            }

            // Use RCIX if available for O(1) per-column access (skip sequential pos scan)
            let rcix = if compress_flag == RG_COMPRESS_NONE && encoding_version >= 1
                && rg_i < footer.col_offsets.len() && footer.col_offsets[rg_i].len() >= col_count
            { Some(&footer.col_offsets[rg_i]) } else { None };

            // Fallback sequential pos (only used when RCIX not available)
            let mut pos = rg_rows * 8 + (rg_rows + 7) / 8;

            for ci in 0..col_count {
                let ct = schema.columns[ci].1;
                let (null_bytes, col_bytes) = if let Some(rcix) = rcix {
                    let col_off = rcix[ci] as usize;
                    if col_off + null_bitmap_len > body.len() { continue; }
                    (&body[col_off..col_off + null_bitmap_len], &body[col_off + null_bitmap_len..])
                } else {
                    if pos + null_bitmap_len > body.len() { break; }
                    let nb = &body[pos..pos + null_bitmap_len];
                    pos += null_bitmap_len;
                    (nb, &body[pos..])
                };
                let col_bytes = col_bytes;
                let enc_offset = if encoding_version >= 1 { 1 } else { 0 };
                let encoding = if encoding_version >= 1 && !col_bytes.is_empty() { col_bytes[0] } else { COL_ENCODING_PLAIN };
                let data_bytes = if enc_offset <= col_bytes.len() { &col_bytes[enc_offset..] } else { &[] as &[u8] };

                let extracted = match (encoding, ct) {
                    (COL_ENCODING_PLAIN, ColumnType::Int64 | ColumnType::Int8 | ColumnType::Int16 |
                     ColumnType::Int32 | ColumnType::UInt8 | ColumnType::UInt16 | ColumnType::UInt32 |
                     ColumnType::UInt64 | ColumnType::Timestamp | ColumnType::Date) if data_bytes.len() >= 8 => {
                        for &(out_idx, local_idx) in local_pairs {
                            if (null_bytes[local_idx / 8] >> (local_idx % 8)) & 1 == 1 { continue; }
                            let off = 8 + local_idx * 8;
                            if off + 8 <= data_bytes.len() {
                                out_cols[ci][out_idx] = Some(crate::data::Value::Int64(
                                    i64::from_le_bytes(data_bytes[off..off+8].try_into().unwrap())
                                ));
                            }
                        }
                        true
                    }
                    (COL_ENCODING_PLAIN, ColumnType::Float64 | ColumnType::Float32) if data_bytes.len() >= 8 => {
                        for &(out_idx, local_idx) in local_pairs {
                            if (null_bytes[local_idx / 8] >> (local_idx % 8)) & 1 == 1 { continue; }
                            let off = 8 + local_idx * 8;
                            if off + 8 <= data_bytes.len() {
                                out_cols[ci][out_idx] = Some(crate::data::Value::Float64(
                                    f64::from_le_bytes(data_bytes[off..off+8].try_into().unwrap())
                                ));
                            }
                        }
                        true
                    }
                    (COL_ENCODING_PLAIN, ColumnType::String) if data_bytes.len() >= 8 => {
                        let count = u64::from_le_bytes(data_bytes[0..8].try_into().unwrap()) as usize;
                        let offsets_end = 8 + (count + 1) * 4;
                        let data_len_off = offsets_end;
                        if data_len_off + 8 <= data_bytes.len() {
                            let data_start = data_len_off + 8;
                            for &(out_idx, local_idx) in local_pairs {
                                if (null_bytes[local_idx / 8] >> (local_idx % 8)) & 1 == 1 { continue; }
                                if local_idx < count {
                                    let s_off = 8 + local_idx * 4;
                                    let e_off = 8 + (local_idx + 1) * 4;
                                    if e_off + 4 <= data_bytes.len() {
                                        let s = u32::from_le_bytes(data_bytes[s_off..s_off+4].try_into().unwrap()) as usize;
                                        let e = u32::from_le_bytes(data_bytes[e_off..e_off+4].try_into().unwrap()) as usize;
                                        if data_start + e <= data_bytes.len() {
                                            out_cols[ci][out_idx] = Some(crate::data::Value::String(
                                                std::str::from_utf8(&data_bytes[data_start + s..data_start + e]).unwrap_or("").to_string()
                                            ));
                                        }
                                    }
                                }
                            }
                            true
                        } else { false }
                    }
                    (COL_ENCODING_PLAIN, ColumnType::StringDict) if data_bytes.len() >= 16 => {
                        let row_count = u64::from_le_bytes(data_bytes[0..8].try_into().unwrap()) as usize;
                        let dict_size = u64::from_le_bytes(data_bytes[8..16].try_into().unwrap()) as usize;
                        let indices_start = 16usize;
                        let dict_off_start = indices_start + row_count * 4;
                        let dict_data_len_off = dict_off_start + dict_size * 4;
                        if dict_data_len_off + 8 <= data_bytes.len() {
                            let dict_data_len = u64::from_le_bytes(data_bytes[dict_data_len_off..dict_data_len_off+8].try_into().unwrap()) as usize;
                            let dict_data_start = dict_data_len_off + 8;
                            for &(out_idx, local_idx) in local_pairs {
                                if (null_bytes[local_idx / 8] >> (local_idx % 8)) & 1 == 1 { continue; }
                                if local_idx < row_count {
                                    let idx_off = indices_start + local_idx * 4;
                                    if idx_off + 4 <= data_bytes.len() {
                                        let dict_idx = u32::from_le_bytes(data_bytes[idx_off..idx_off+4].try_into().unwrap());
                                        if dict_idx == 0 { continue; }
                                        let di = (dict_idx - 1) as usize;
                                        if di < dict_size {
                                            let ds = u32::from_le_bytes(data_bytes[dict_off_start + di*4..dict_off_start + di*4+4].try_into().unwrap()) as usize;
                                            let de = if di + 1 < dict_size {
                                                u32::from_le_bytes(data_bytes[dict_off_start + (di+1)*4..dict_off_start + (di+1)*4+4].try_into().unwrap()) as usize
                                            } else { dict_data_len };
                                            if dict_data_start + de <= data_bytes.len() {
                                                out_cols[ci][out_idx] = Some(crate::data::Value::String(
                                                    std::str::from_utf8(&data_bytes[dict_data_start + ds..dict_data_start + de]).unwrap_or("").to_string()
                                                ));
                                            }
                                        }
                                    }
                                }
                            }
                            true
                        } else { false }
                    }
                    (2u8 /* COL_ENCODING_BITPACK */, ColumnType::Int64 | ColumnType::Int8 | ColumnType::Int16 |
                     ColumnType::Int32 | ColumnType::UInt8 | ColumnType::UInt16 | ColumnType::UInt32 |
                     ColumnType::UInt64 | ColumnType::Timestamp | ColumnType::Date) => {
                        for &(out_idx, local_idx) in local_pairs {
                            if (null_bytes[local_idx / 8] >> (local_idx % 8)) & 1 == 1 { continue; }
                            if let Some(v) = crate::storage::on_demand::bitpack_decode_at_idx(data_bytes, local_idx) {
                                out_cols[ci][out_idx] = Some(crate::data::Value::Int64(v));
                            }
                        }
                        true
                    }
                    (COL_ENCODING_PLAIN, ColumnType::Bool) if data_bytes.len() >= 8 => {
                        for &(out_idx, local_idx) in local_pairs {
                            if (null_bytes[local_idx / 8] >> (local_idx % 8)) & 1 == 1 { continue; }
                            let byte_off = 8 + local_idx / 8;
                            let bit = local_idx % 8;
                            if byte_off < data_bytes.len() {
                                out_cols[ci][out_idx] = Some(crate::data::Value::Bool(
                                    (data_bytes[byte_off] >> bit) & 1 == 1
                                ));
                            }
                        }
                        true
                    }
                    _ => false,
                };

                if !extracted {
                    // Fallback: decode full column and pick values at target indices
                    let (col_data, _) = if encoding_version >= 1 {
                        read_column_encoded(col_bytes, ct)?
                    } else {
                        ColumnData::from_bytes_typed(col_bytes, ct)?
                    };
                    for &(out_idx, local_idx) in local_pairs {
                        if (null_bytes[local_idx / 8] >> (local_idx % 8)) & 1 == 1 { continue; }
                        let val = match &col_data {
                            ColumnData::Int64(v) => if local_idx < v.len() { Some(crate::data::Value::Int64(v[local_idx])) } else { None },
                            ColumnData::Float64(v) => if local_idx < v.len() { Some(crate::data::Value::Float64(v[local_idx])) } else { None },
                            ColumnData::Bool { data, len } => if local_idx < *len {
                                Some(crate::data::Value::Bool((data[local_idx / 8] >> (local_idx % 8)) & 1 == 1))
                            } else { None },
                            ColumnData::String { offsets, data } => {
                                let cnt = offsets.len().saturating_sub(1);
                                if local_idx < cnt {
                                    let s = offsets[local_idx] as usize;
                                    let e = offsets[local_idx + 1] as usize;
                                    Some(crate::data::Value::String(std::str::from_utf8(&data[s..e]).unwrap_or("").to_string()))
                                } else { None }
                            }
                            ColumnData::StringDict { indices: idx_arr, dict_offsets, dict_data, .. } => {
                                if local_idx < idx_arr.len() {
                                    let di = idx_arr[local_idx];
                                    if di == 0 { None } else {
                                        let d = (di - 1) as usize;
                                        if d < dict_offsets.len() {
                                            let s = dict_offsets[d] as usize;
                                            let e = if d + 1 < dict_offsets.len() { dict_offsets[d + 1] as usize } else { dict_data.len() };
                                            Some(crate::data::Value::String(std::str::from_utf8(&dict_data[s..e]).unwrap_or("").to_string()))
                                        } else { None }
                                    }
                                } else { None }
                            }
                            _ => None,
                        };
                        if let Some(v) = val { out_cols[ci][out_idx] = Some(v); }
                    }
                }

                // Advance pos past this column (only needed for fallback sequential path)
                if rcix.is_none() {
                    let consumed = if encoding_version >= 1 {
                        skip_column_encoded(&body[pos..], ct)?
                    } else {
                        ColumnData::skip_bytes_typed(&body[pos..], ct)?
                    };
                    pos += consumed;
                }
            }
        }

        drop(mmap_guard);
        drop(file_guard);

        // Build Arrow RecordBatch from collected values
        let mut fields: Vec<Field> = Vec::with_capacity(col_count + 1);
        let mut arrays: Vec<ArrayRef> = Vec::with_capacity(col_count + 1);

        fields.push(Field::new("_id", ArrowDataType::Int64, false));
        arrays.push(Arc::new(Int64Array::from(out_ids)));

        for ci in 0..col_count {
            let col_name = &schema.columns[ci].0;
            let ct = schema.columns[ci].1;
            let vals = &out_cols[ci];
            match ct {
                ColumnType::Int64 | ColumnType::Int8 | ColumnType::Int16 | ColumnType::Int32 |
                ColumnType::UInt8 | ColumnType::UInt16 | ColumnType::UInt32 | ColumnType::UInt64 |
                ColumnType::Timestamp | ColumnType::Date => {
                    let arr: Vec<Option<i64>> = vals.iter().map(|v| match v {
                        Some(crate::data::Value::Int64(n)) => Some(*n),
                        _ => None,
                    }).collect();
                    fields.push(Field::new(col_name, ArrowDataType::Int64, true));
                    arrays.push(Arc::new(Int64Array::from(arr)));
                }
                ColumnType::Float64 | ColumnType::Float32 => {
                    let arr: Vec<Option<f64>> = vals.iter().map(|v| match v {
                        Some(crate::data::Value::Float64(n)) => Some(*n),
                        _ => None,
                    }).collect();
                    fields.push(Field::new(col_name, ArrowDataType::Float64, true));
                    arrays.push(Arc::new(Float64Array::from(arr)));
                }
                ColumnType::String | ColumnType::StringDict => {
                    let mut builder = StringBuilder::with_capacity(n_out, n_out * 16);
                    for v in vals {
                        match v {
                            Some(crate::data::Value::String(s)) => builder.append_value(s),
                            _ => builder.append_null(),
                        }
                    }
                    fields.push(Field::new(col_name, ArrowDataType::Utf8, true));
                    arrays.push(Arc::new(builder.finish()) as ArrayRef);
                }
                ColumnType::Bool => {
                    let mut builder = BooleanBuilder::with_capacity(n_out);
                    for v in vals {
                        match v {
                            Some(crate::data::Value::Bool(b)) => builder.append_value(*b),
                            _ => builder.append_null(),
                        }
                    }
                    fields.push(Field::new(col_name, ArrowDataType::Boolean, true));
                    arrays.push(Arc::new(builder.finish()) as ArrayRef);
                }
                _ => {
                    let mut builder = StringBuilder::with_capacity(n_out, n_out * 16);
                    for v in vals {
                        match v {
                            Some(crate::data::Value::String(s)) => builder.append_value(s),
                            _ => builder.append_null(),
                        }
                    }
                    fields.push(Field::new(col_name, ArrowDataType::Utf8, true));
                    arrays.push(Arc::new(builder.finish()) as ArrayRef);
                }
            }
        }

        let batch_schema = Arc::new(Schema::new(fields));
        let batch = arrow::record_batch::RecordBatch::try_new(batch_schema, arrays)
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e.to_string()))?;
        Ok(Some(batch))
    }

}
