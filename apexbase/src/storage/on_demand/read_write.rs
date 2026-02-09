// Insert, delete, read_row_by_id, column management, save, persist

impl OnDemandStorage {
    /// Read IDs for specific global row indices from mmap.
    /// Returns Vec<u64> of IDs corresponding to the given indices.
    pub fn get_ids_for_global_indices_mmap(&self, indices: &[usize]) -> io::Result<Vec<u64>> {
        let footer = match self.get_or_load_footer()? {
            Some(f) => f,
            None => return Ok(vec![]),
        };
        // Build RG bounds
        let mut rg_bounds: Vec<(usize, usize)> = Vec::new();
        let mut cumulative = 0usize;
        for rg in &footer.row_groups {
            let n = rg.row_count as usize;
            rg_bounds.push((cumulative, cumulative + n));
            cumulative += n;
        }

        let file_guard = self.file.read();
        let file = file_guard.as_ref()
            .ok_or_else(|| err_not_conn("File not open for ID read"))?;
        let mut mmap_guard = self.mmap_cache.write();
        let mmap_ref = mmap_guard.get_or_create(file)?;

        let mut result = Vec::with_capacity(indices.len());
        for &idx in indices {
            let mut found = false;
            for (rg_i, &(start, end)) in rg_bounds.iter().enumerate() {
                if idx >= start && idx < end {
                    let local = idx - start;
                    let rg_meta = &footer.row_groups[rg_i];
                    let rg_end = (rg_meta.offset + rg_meta.data_size) as usize;
                    if rg_end <= mmap_ref.len() {
                        let rg_bytes = &mmap_ref[rg_meta.offset as usize .. rg_end];
                        let compress_flag = if rg_bytes.len() >= 32 { rg_bytes[28] } else { RG_COMPRESS_NONE };
                        let decompressed = decompress_rg_body(compress_flag, &rg_bytes[32..])?;
                        let body: &[u8] = decompressed.as_deref().unwrap_or(&rg_bytes[32..]);
                        let off = local * 8;
                        if off + 8 <= body.len() {
                            let id = u64::from_le_bytes(body[off..off+8].try_into().unwrap());
                            result.push(id);
                            found = true;
                        }
                    }
                    break;
                }
            }
            if !found { result.push(0); }
        }
        Ok(result)
    }
    // ========================================================================
    // Write APIs
    // ========================================================================

    /// Insert typed columns directly
    pub fn insert_typed(
        &self,
        int_columns: HashMap<String, Vec<i64>>,
        float_columns: HashMap<String, Vec<f64>>,
        string_columns: HashMap<String, Vec<String>>,
        binary_columns: HashMap<String, Vec<Vec<u8>>>,
        bool_columns: HashMap<String, Vec<bool>>,
    ) -> io::Result<Vec<u64>> {
        // Determine row count as maximum across all columns (for heterogeneous schemas)
        let row_count = int_columns.values().map(|v| v.len()).max().unwrap_or(0)
            .max(float_columns.values().map(|v| v.len()).max().unwrap_or(0))
            .max(string_columns.values().map(|v| v.len()).max().unwrap_or(0))
            .max(binary_columns.values().map(|v| v.len()).max().unwrap_or(0))
            .max(bool_columns.values().map(|v| v.len()).max().unwrap_or(0));

        if row_count == 0 {
            return Ok(Vec::new());
        }

        // Allocate IDs atomically
        let start_id = self.next_id.fetch_add(row_count as u64, Ordering::SeqCst);
        let ids: Vec<u64> = (start_id..start_id + row_count as u64).collect();

        // Ensure schema has all columns, padding existing rows with defaults for new columns
        {
            let mut schema = self.schema.write();
            let mut columns = self.columns.write();
            let mut nulls = self.nulls.write();
            let existing_row_count = self.ids.read().len();

            // First, add any new columns to schema
            for name in int_columns.keys() {
                schema.add_column(name, ColumnType::Int64);
            }
            for name in float_columns.keys() {
                schema.add_column(name, ColumnType::Float64);
            }
            for name in string_columns.keys() {
                schema.add_column(name, ColumnType::String);
            }
            for name in binary_columns.keys() {
                schema.add_column(name, ColumnType::Binary);
            }
            for name in bool_columns.keys() {
                schema.add_column(name, ColumnType::Bool);
            }

            // Then, ensure columns vector matches schema (using correct types from schema)
            while columns.len() < schema.column_count() {
                let col_idx = columns.len();
                let (_, col_type) = &schema.columns[col_idx];
                let mut col = ColumnData::new(*col_type);
                // Pad with defaults for existing rows
                if existing_row_count > 0 {
                    match &mut col {
                        ColumnData::Int64(v) => v.resize(existing_row_count, 0),
                        ColumnData::Float64(v) => v.resize(existing_row_count, 0.0),
                        ColumnData::String { offsets, .. } => {
                            for _ in 0..existing_row_count {
                                offsets.push(0);
                            }
                        }
                        ColumnData::Binary { offsets, .. } => {
                            for _ in 0..existing_row_count {
                                offsets.push(0);
                            }
                        }
                        ColumnData::Bool { len, .. } => {
                            *len = existing_row_count;
                        }
                        _ => {}
                    }
                }
                columns.push(col);
                nulls.push(Vec::new());
            }
        }

        // OPTIMIZATION: combine ID append + column append + metadata updates
        // to minimize lock acquire/release overhead
        let col_count_for_header;
        {
            let schema = self.schema.read();
            col_count_for_header = schema.column_count() as u32;
            let mut ids_guard = self.ids.write();
            let start_idx = ids_guard.len();
            ids_guard.extend_from_slice(&ids);
            let total_rows_after = ids_guard.len();
            drop(ids_guard);

            // Append column data (schema read lock still held)
            let mut columns = self.columns.write();
            for (name, values) in int_columns {
                if let Some(idx) = schema.get_index(&name) {
                    columns[idx].extend_i64(&values);
                }
            }
            for (name, values) in float_columns {
                if let Some(idx) = schema.get_index(&name) {
                    columns[idx].extend_f64(&values);
                }
            }
            for (name, values) in string_columns {
                if let Some(idx) = schema.get_index(&name) {
                    if idx < columns.len() {
                        match &columns[idx] {
                            ColumnData::String { .. } => {
                                columns[idx].extend_strings(&values);
                            }
                            ColumnData::StringDict { indices, dict_offsets, dict_data } => {
                                let mut new_offsets = vec![0u32];
                                let mut new_data = Vec::new();
                                for &dict_idx in indices {
                                    if dict_idx == 0 {
                                        new_offsets.push(new_data.len() as u32);
                                    } else {
                                        let actual_idx = (dict_idx - 1) as usize;
                                        if actual_idx + 1 < dict_offsets.len() {
                                            let start = dict_offsets[actual_idx] as usize;
                                            let end = dict_offsets[actual_idx + 1] as usize;
                                            new_data.extend_from_slice(&dict_data[start..end]);
                                        }
                                        new_offsets.push(new_data.len() as u32);
                                    }
                                }
                                columns[idx] = ColumnData::String { 
                                    offsets: new_offsets, 
                                    data: new_data 
                                };
                                columns[idx].extend_strings(&values);
                            }
                            _ => {
                                columns[idx] = ColumnData::new(ColumnType::String);
                                columns[idx].extend_strings(&values);
                            }
                        }
                    }
                }
            }
            for (name, values) in binary_columns {
                if let Some(idx) = schema.get_index(&name) {
                    columns[idx].extend_bytes(&values);
                }
            }
            for (name, values) in bool_columns {
                if let Some(idx) = schema.get_index(&name) {
                    columns[idx].extend_bools(&values);
                }
            }
            drop(columns);
            drop(schema);

            // Update id_to_idx if already built (avoid rebuilding)
            {
                let mut id_to_idx = self.id_to_idx.write();
                if let Some(map) = id_to_idx.as_mut() {
                    for (i, &id) in ids.iter().enumerate() {
                        map.insert(id, start_idx + i);
                    }
                }
            }

            // Extend deleted bitmap
            {
                let mut deleted = self.deleted.write();
                let new_len = (total_rows_after + 7) / 8;
                deleted.resize(new_len, 0);
            }
        }

        // Update header
        {
            let mut header = self.header.write();
            header.row_count += row_count as u64;
            header.column_count = col_count_for_header;
            header.modified_at = chrono::Utc::now().timestamp();
        }
        
        // Update active count (new rows are not deleted)
        self.active_count.fetch_add(row_count as u64, Ordering::Relaxed);

        Ok(ids)
    }

    /// Insert typed columns with EXPLICIT IDs (used during delta compaction)
    /// This preserves the original IDs from delta file instead of generating new ones
    fn insert_typed_with_ids(
        &self,
        ids: &[u64],
        int_columns: HashMap<String, Vec<i64>>,
        float_columns: HashMap<String, Vec<f64>>,
        string_columns: HashMap<String, Vec<String>>,
        binary_columns: HashMap<String, Vec<Vec<u8>>>,
        bool_columns: HashMap<String, Vec<bool>>,
    ) -> io::Result<()> {
        let row_count = ids.len();
        if row_count == 0 {
            return Ok(());
        }

        // Update next_id to be greater than any provided ID
        for &id in ids {
            let current = self.next_id.load(Ordering::SeqCst);
            if id >= current {
                self.next_id.store(id + 1, Ordering::SeqCst);
            }
        }

        // Ensure schema has all columns
        {
            let mut schema = self.schema.write();
            let mut columns = self.columns.write();
            let mut nulls = self.nulls.write();
            let existing_row_count = self.ids.read().len();

            for name in int_columns.keys() {
                schema.add_column(name, ColumnType::Int64);
            }
            for name in float_columns.keys() {
                schema.add_column(name, ColumnType::Float64);
            }
            for name in string_columns.keys() {
                schema.add_column(name, ColumnType::String);
            }
            for name in binary_columns.keys() {
                schema.add_column(name, ColumnType::Binary);
            }
            for name in bool_columns.keys() {
                schema.add_column(name, ColumnType::Bool);
            }

            while columns.len() < schema.column_count() {
                let col_idx = columns.len();
                let (_, col_type) = &schema.columns[col_idx];
                let mut col = ColumnData::new(*col_type);
                if existing_row_count > 0 {
                    match &mut col {
                        ColumnData::Int64(v) => v.resize(existing_row_count, 0),
                        ColumnData::Float64(v) => v.resize(existing_row_count, 0.0),
                        ColumnData::String { offsets, .. } => {
                            for _ in 0..existing_row_count {
                                offsets.push(0);
                            }
                        }
                        ColumnData::Binary { offsets, .. } => {
                            for _ in 0..existing_row_count {
                                offsets.push(0);
                            }
                        }
                        ColumnData::Bool { len, .. } => {
                            *len = existing_row_count;
                        }
                        ColumnData::StringDict { indices, .. } => {
                            indices.resize(existing_row_count, 0);
                        }
                    }
                }
                columns.push(col);
                nulls.push(Vec::new());
            }
        }

        // Append IDs
        {
            let mut ids_vec = self.ids.write();
            ids_vec.extend_from_slice(ids);
        }

        // Append column data
        {
            let schema = self.schema.read();
            let mut columns = self.columns.write();

            for (name, values) in int_columns {
                if let Some(idx) = schema.get_index(&name) {
                    if idx < columns.len() {
                        columns[idx].extend_i64(&values);
                    }
                }
            }

            for (name, values) in float_columns {
                if let Some(idx) = schema.get_index(&name) {
                    if idx < columns.len() {
                        columns[idx].extend_f64(&values);
                    }
                }
            }

            for (name, values) in string_columns {
                if let Some(idx) = schema.get_index(&name) {
                    if idx < columns.len() {
                        match &columns[idx] {
                            ColumnData::String { .. } => {
                                columns[idx].extend_strings(&values);
                            }
                            ColumnData::StringDict { indices, dict_offsets, dict_data } => {
                                let mut new_offsets = vec![0u32];
                                let mut new_data = Vec::new();
                                for &dict_idx in indices {
                                    if dict_idx == 0 {
                                        new_offsets.push(new_data.len() as u32);
                                    } else {
                                        let actual_idx = (dict_idx - 1) as usize;
                                        if actual_idx + 1 < dict_offsets.len() {
                                            let start = dict_offsets[actual_idx] as usize;
                                            let end = dict_offsets[actual_idx + 1] as usize;
                                            new_data.extend_from_slice(&dict_data[start..end]);
                                        }
                                        new_offsets.push(new_data.len() as u32);
                                    }
                                }
                                columns[idx] = ColumnData::String { 
                                    offsets: new_offsets, 
                                    data: new_data 
                                };
                                columns[idx].extend_strings(&values);
                            }
                            _ => {
                                columns[idx] = ColumnData::new(ColumnType::String);
                                columns[idx].extend_strings(&values);
                            }
                        }
                    }
                }
            }

            for (name, values) in bool_columns {
                if let Some(idx) = schema.get_index(&name) {
                    if idx < columns.len() {
                        columns[idx].extend_bools(&values);
                    }
                }
            }

            // Pad columns that don't have new data
            for col_idx in 0..columns.len() {
                let expected_len = self.ids.read().len();
                let current_len = columns[col_idx].len();
                if current_len < expected_len {
                    let pad_count = expected_len - current_len;
                    match &mut columns[col_idx] {
                        ColumnData::Int64(v) => v.extend(std::iter::repeat(0).take(pad_count)),
                        ColumnData::Float64(v) => v.extend(std::iter::repeat(0.0).take(pad_count)),
                        ColumnData::String { offsets, .. } => {
                            for _ in 0..pad_count {
                                offsets.push(*offsets.last().unwrap_or(&0));
                            }
                        }
                        ColumnData::Binary { offsets, .. } => {
                            for _ in 0..pad_count {
                                offsets.push(*offsets.last().unwrap_or(&0));
                            }
                        }
                        ColumnData::Bool { data, len } => {
                            for _ in 0..pad_count {
                                let byte_idx = *len / 8;
                                if byte_idx >= data.len() { data.push(0); }
                                *len += 1;
                            }
                        }
                        ColumnData::StringDict { indices, .. } => {
                            indices.extend(std::iter::repeat(0).take(pad_count));
                        }
                    }
                }
            }
        }

        // Update header
        {
            let mut header = self.header.write();
            header.row_count = self.ids.read().len() as u64;
            header.column_count = self.schema.read().column_count() as u32;
        }

        // Extend deleted bitmap
        {
            let mut deleted = self.deleted.write();
            let new_len = (self.ids.read().len() + 7) / 8;
            deleted.resize(new_len, 0);
        }
        
        self.active_count.fetch_add(row_count as u64, Ordering::Relaxed);
        Ok(())
    }

    /// Insert typed columns with explicit NULL tracking for heterogeneous schemas
    pub fn insert_typed_with_nulls(
        &self,
        int_columns: HashMap<String, Vec<i64>>,
        float_columns: HashMap<String, Vec<f64>>,
        string_columns: HashMap<String, Vec<String>>,
        binary_columns: HashMap<String, Vec<Vec<u8>>>,
        bool_columns: HashMap<String, Vec<bool>>,
        null_positions: HashMap<String, Vec<bool>>,
    ) -> io::Result<Vec<u64>> {
        // Determine row count as maximum across all columns
        let row_count = int_columns.values().map(|v| v.len()).max().unwrap_or(0)
            .max(float_columns.values().map(|v| v.len()).max().unwrap_or(0))
            .max(string_columns.values().map(|v| v.len()).max().unwrap_or(0))
            .max(binary_columns.values().map(|v| v.len()).max().unwrap_or(0))
            .max(bool_columns.values().map(|v| v.len()).max().unwrap_or(0));

        if row_count == 0 {
            return Ok(Vec::new());
        }

        // Allocate IDs atomically
        let start_id = self.next_id.fetch_add(row_count as u64, Ordering::SeqCst);
        let ids: Vec<u64> = (start_id..start_id + row_count as u64).collect();

        // Ensure schema has all columns and track column indices
        // For new columns, pad existing rows with defaults (NULL-like values)
        let mut col_name_to_idx: HashMap<String, usize> = HashMap::new();
        {
            let mut schema = self.schema.write();
            let mut columns = self.columns.write();
            let mut nulls = self.nulls.write();
            let existing_row_count = self.ids.read().len();

            for name in int_columns.keys() {
                let idx = schema.add_column(name, ColumnType::Int64);
                col_name_to_idx.insert(name.clone(), idx);
                while columns.len() <= idx {
                    let mut col = ColumnData::new(ColumnType::Int64);
                    // Pad with defaults for existing rows
                    if let ColumnData::Int64(v) = &mut col {
                        v.resize(existing_row_count, 0);
                    }
                    columns.push(col);
                    // Mark all existing rows as NULL for new column
                    nulls.push(Vec::new());
                }
            }
            for name in float_columns.keys() {
                let idx = schema.add_column(name, ColumnType::Float64);
                col_name_to_idx.insert(name.clone(), idx);
                while columns.len() <= idx {
                    let mut col = ColumnData::new(ColumnType::Float64);
                    if let ColumnData::Float64(v) = &mut col {
                        v.resize(existing_row_count, 0.0);
                    }
                    columns.push(col);
                    nulls.push(Vec::new());
                }
            }
            for name in string_columns.keys() {
                let idx = schema.add_column(name, ColumnType::String);
                col_name_to_idx.insert(name.clone(), idx);
                while columns.len() <= idx {
                    let mut col = ColumnData::new(ColumnType::String);
                    // Pad with empty strings for existing rows
                    if let ColumnData::String { offsets, .. } = &mut col {
                        for _ in 0..existing_row_count {
                            offsets.push(0);
                        }
                    }
                    columns.push(col);
                    nulls.push(Vec::new());
                }
            }
            for name in binary_columns.keys() {
                let idx = schema.add_column(name, ColumnType::Binary);
                col_name_to_idx.insert(name.clone(), idx);
                while columns.len() <= idx {
                    let mut col = ColumnData::new(ColumnType::Binary);
                    if let ColumnData::Binary { offsets, .. } = &mut col {
                        for _ in 0..existing_row_count {
                            offsets.push(0);
                        }
                    }
                    columns.push(col);
                    nulls.push(Vec::new());
                }
            }
            for name in bool_columns.keys() {
                let idx = schema.add_column(name, ColumnType::Bool);
                col_name_to_idx.insert(name.clone(), idx);
                while columns.len() <= idx {
                    let mut col = ColumnData::new(ColumnType::Bool);
                    if let ColumnData::Bool { len, .. } = &mut col {
                        *len = existing_row_count;
                    }
                    columns.push(col);
                    nulls.push(Vec::new());
                }
            }
        }

        // Append IDs
        self.ids.write().extend_from_slice(&ids);

        // Append column data
        {
            let schema = self.schema.read();
            let mut columns = self.columns.write();

            for (name, values) in int_columns {
                if let Some(idx) = schema.get_index(&name) {
                    columns[idx].extend_i64(&values);
                }
            }
            for (name, values) in float_columns {
                if let Some(idx) = schema.get_index(&name) {
                    columns[idx].extend_f64(&values);
                }
            }
            for (name, values) in string_columns {
                if let Some(idx) = schema.get_index(&name) {
                    for v in &values {
                        columns[idx].push_string(v);
                    }
                }
            }
            for (name, values) in binary_columns {
                if let Some(idx) = schema.get_index(&name) {
                    for v in &values {
                        columns[idx].push_bytes(v);
                    }
                }
            }
            for (name, values) in bool_columns {
                if let Some(idx) = schema.get_index(&name) {
                    for v in values {
                        columns[idx].push_bool(v);
                    }
                }
            }
        }

        // Update null bitmaps for each column
        {
            let mut nulls = self.nulls.write();
            let base_row = self.ids.read().len() - row_count;
            
            for (col_name, is_null_vec) in null_positions {
                if let Some(&col_idx) = col_name_to_idx.get(&col_name) {
                    if col_idx < nulls.len() {
                        // Extend null bitmap for this column
                        let null_bitmap = &mut nulls[col_idx];
                        for (i, &is_null) in is_null_vec.iter().enumerate() {
                            if is_null {
                                let row_idx = base_row + i;
                                let byte_idx = row_idx / 8;
                                let bit_idx = row_idx % 8;
                                while null_bitmap.len() <= byte_idx {
                                    null_bitmap.push(0);
                                }
                                null_bitmap[byte_idx] |= 1 << bit_idx;
                            }
                        }
                    }
                }
            }
        }

        // Update header
        {
            let mut header = self.header.write();
            header.row_count += row_count as u64;
            header.column_count = self.schema.read().column_count() as u32;
            header.modified_at = chrono::Utc::now().timestamp();
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
        
        // Extend deleted bitmap with zeros for new rows
        {
            let mut deleted = self.deleted.write();
            let new_len = (self.ids.read().len() + 7) / 8;
            deleted.resize(new_len, 0);
        }
        
        // Update active count (new rows are not deleted)
        self.active_count.fetch_add(row_count as u64, Ordering::Relaxed);

        Ok(ids)
    }

    // ========================================================================
    // Delete/Update APIs
    // ========================================================================

    /// Delete a row by ID (soft delete)
    /// Returns true if the row was found and deleted
    pub fn delete(&self, id: u64) -> bool {
        self.ensure_id_index();
        let id_to_idx = self.id_to_idx.read();
        let map = id_to_idx.as_ref().unwrap();
        if let Some(&row_idx) = map.get(&id) {
            drop(id_to_idx);  // Release read lock before write
            let mut deleted = self.deleted.write();
            let byte_idx = row_idx / 8;
            let bit_idx = row_idx % 8;
            
            // Ensure bitmap is large enough
            if byte_idx >= deleted.len() {
                deleted.resize(byte_idx + 1, 0);
            }
            
            // Only decrement if not already deleted
            let was_deleted = (deleted[byte_idx] >> bit_idx) & 1 == 1;
            if !was_deleted {
                self.active_count.fetch_sub(1, Ordering::Relaxed);
            }
            
            // Set the deleted bit
            deleted[byte_idx] |= 1 << bit_idx;
            true
        } else {
            false
        }
    }

    /// Delete multiple rows by IDs (soft delete)
    /// Returns true if all rows were found and deleted
    pub fn delete_batch(&self, ids: &[u64]) -> bool {
        self.ensure_id_index();
        let id_to_idx = self.id_to_idx.read();
        let map = id_to_idx.as_ref().unwrap();
        let mut deleted = self.deleted.write();
        let mut all_found = true;
        let mut deleted_count = 0u64;
        
        for &id in ids {
            if let Some(&row_idx) = map.get(&id) {
                let byte_idx = row_idx / 8;
                let bit_idx = row_idx % 8;
                
                if byte_idx >= deleted.len() {
                    deleted.resize(byte_idx + 1, 0);
                }
                
                // Only count if not already deleted
                let was_deleted = (deleted[byte_idx] >> bit_idx) & 1 == 1;
                if !was_deleted {
                    deleted_count += 1;
                }
                
                deleted[byte_idx] |= 1 << bit_idx;
            } else {
                all_found = false;
            }
        }
        
        // Update active count
        if deleted_count > 0 {
            self.active_count.fetch_sub(deleted_count, Ordering::Relaxed);
        }
        
        all_found
    }

    /// Check if a row is deleted
    pub fn is_deleted(&self, row_idx: usize) -> bool {
        let deleted = self.deleted.read();
        let byte_idx = row_idx / 8;
        let bit_idx = row_idx % 8;
        
        if byte_idx < deleted.len() {
            (deleted[byte_idx] >> bit_idx) & 1 == 1
        } else {
            false
        }
    }

    /// Check if an ID exists and is not deleted
    /// Also checks delta file for IDs not yet merged into base
    pub fn exists(&self, id: u64) -> bool {
        // First check base file IDs
        self.ensure_id_index();
        let id_to_idx = self.id_to_idx.read();
        let map = id_to_idx.as_ref().unwrap();
        if let Some(&row_idx) = map.get(&id) {
            if !self.is_deleted(row_idx) {
                return true;
            }
        }
        
        // Check delta file for IDs not yet merged
        if let Ok(Some((delta_ids, _))) = self.read_delta_data() {
            return delta_ids.contains(&id);
        }
        
        false
    }

    /// Get row index for an ID (None if not found or deleted)
    pub fn get_row_idx(&self, id: u64) -> Option<usize> {
        self.ensure_id_index();
        let id_to_idx = self.id_to_idx.read();
        let map = id_to_idx.as_ref().unwrap();
        if let Some(&row_idx) = map.get(&id) {
            if !self.is_deleted(row_idx) {
                Some(row_idx)
            } else {
                None
            }
        } else {
            None
        }
    }

    /// OPTIMIZED: Read a single row by ID using O(1) index lookup.
    /// Returns HashMap of column_name -> ColumnData (single element).
    /// Supports both in-memory and mmap-only paths.
    pub fn read_row_by_id(&self, id: u64, column_names: Option<&[&str]>) -> io::Result<Option<HashMap<String, ColumnData>>> {
        let is_v4 = self.is_v4_format();
        
        // O(1) lookup using id_to_idx index (works for both in-memory and mmap-loaded IDs)
        let row_idx = match self.get_row_idx(id) {
            Some(idx) => idx,
            None => return Ok(None),
        };
        
        let indices = vec![row_idx];
        let schema = self.schema.read();
        
        // Get columns to read
        let cols_to_read: Vec<(usize, String, ColumnType)> = if let Some(names) = column_names {
            names.iter()
                .filter_map(|&name| {
                    if name == "_id" {
                        None
                    } else {
                        schema.get_index(name).map(|idx| {
                            (idx, name.to_string(), schema.columns[idx].1)
                        })
                    }
                })
                .collect()
        } else {
            schema.columns.iter().enumerate()
                .map(|(idx, (name, dtype))| (idx, name.clone(), *dtype))
                .collect()
        };
        
        let total_rows = self.header.read().row_count as usize;
        drop(schema);
        
        let mut result = HashMap::new();
        
        // Add _id if requested or no column filter
        let include_id = column_names.map(|cols| cols.contains(&"_id")).unwrap_or(true);
        if include_id {
            result.insert("_id".to_string(), ColumnData::Int64(vec![id as i64]));
        }
        
        if is_v4 && !self.has_v4_in_memory_data() {
            // MMAP PATH: Find the target RG by cumulative row count, scan only that RG
            let col_indices: Vec<usize> = cols_to_read.iter().map(|(idx, _, _)| *idx).collect();
            if let Some(footer) = self.get_or_load_footer()? {
                // Find which RG contains row_idx
                let mut cumulative = 0usize;
                let mut target_rg_idx = None;
                let mut local_idx = row_idx;
                for (rg_i, rg_meta) in footer.row_groups.iter().enumerate() {
                    let rg_rows = rg_meta.row_count as usize;
                    if row_idx < cumulative + rg_rows {
                        local_idx = row_idx - cumulative;
                        target_rg_idx = Some(rg_i);
                        break;
                    }
                    cumulative += rg_rows;
                }
                if let Some(rg_i) = target_rg_idx {
                    // Create a single-RG footer view for scan
                    let single_rg_footer = V4Footer {
                        schema: footer.schema.clone(),
                        row_groups: vec![footer.row_groups[rg_i].clone()],
                        zone_maps: if rg_i < footer.zone_maps.len() {
                            vec![footer.zone_maps[rg_i].clone()]
                        } else {
                            vec![]
                        },
                    };
                    let (scanned, _del, col_nulls) = self.scan_columns_mmap_with_nulls(&col_indices, &single_rg_footer)?;
                    let local_indices = vec![local_idx];
                    for (out_pos, (_, col_name, _)) in cols_to_read.iter().enumerate() {
                        if out_pos < scanned.len() {
                            let mut col = scanned[out_pos].clone();
                            if out_pos < col_nulls.len() && !col_nulls[out_pos].is_empty() {
                                col.apply_null_bitmap(&col_nulls[out_pos]);
                            }
                            result.insert(col_name.clone(), col.filter_by_indices(&local_indices));
                        }
                    }
                }
            }
        } else {
            for (col_idx, col_name, col_type) in cols_to_read {
                let col_data = self.read_column_scattered_auto(col_idx, col_type, &indices, total_rows, is_v4)?;
                result.insert(col_name, col_data);
            }
        }
        
        Ok(Some(result))
    }

    /// Ultra-fast point lookup: returns Vec<(col_name, Value)> directly from V4 columns
    /// Bypasses Arrow conversion and HashMap overhead
    pub fn read_row_by_id_values(&self, id: u64) -> io::Result<Option<Vec<(String, crate::data::Value)>>> {
        use crate::data::Value;
        
        let is_v4 = self.is_v4_format();
        if !is_v4 { return Ok(None); }
        if !self.has_v4_in_memory_data() {
            // DIRECT MMAP PATH: Navigate RG body bytes, extract single row without
            // deserializing entire RG. O(columns) instead of O(rows * columns).
            let row_idx = match self.get_row_idx(id) {
                Some(idx) => idx,
                None => return Ok(None),
            };
            let footer = match self.get_or_load_footer()? {
                Some(f) => f,
                None => return Ok(None),
            };
            // Find which RG contains row_idx
            let mut cumulative = 0usize;
            let mut target_rg_i = None;
            let mut local_idx = row_idx;
            for (i, rg) in footer.row_groups.iter().enumerate() {
                let rg_rows = rg.row_count as usize;
                if row_idx < cumulative + rg_rows {
                    local_idx = row_idx - cumulative;
                    target_rg_i = Some(i);
                    break;
                }
                cumulative += rg_rows;
            }
            let rg_i = match target_rg_i {
                Some(i) => i,
                None => return Ok(None),
            };
            let rg_meta = &footer.row_groups[rg_i];
            let rg_rows = rg_meta.row_count as usize;

            // Get mmap bytes for this RG
            let file_guard = self.file.read();
            let file = file_guard.as_ref()
                .ok_or_else(|| err_not_conn("File not open for point lookup"))?;
            let mut mmap_guard = self.mmap_cache.write();
            let mmap_ref = mmap_guard.get_or_create(file)?;
            let rg_end = (rg_meta.offset + rg_meta.data_size) as usize;
            if rg_end > mmap_ref.len() {
                return Err(err_data("RG extends past EOF"));
            }
            let rg_bytes = &mmap_ref[rg_meta.offset as usize .. rg_end];
            let compress_flag = if rg_bytes.len() >= 32 { rg_bytes[28] } else { RG_COMPRESS_NONE };
            let encoding_version = if rg_bytes.len() >= 32 { rg_bytes[29] } else { 0 };
            let decompressed = decompress_rg_body(compress_flag, &rg_bytes[32..])?;
            let body: &[u8] = decompressed.as_deref().unwrap_or(&rg_bytes[32..]);
            let mut pos = 0usize;

            // Read target ID
            let id_offset = local_idx * 8;
            if pos + rg_rows * 8 > body.len() { return Err(err_data("RG IDs truncated")); }
            let read_id = u64::from_le_bytes(body[id_offset..id_offset+8].try_into().unwrap());
            pos += rg_rows * 8;

            // Check deletion
            let del_vec_len = (rg_rows + 7) / 8;
            if pos + del_vec_len > body.len() { return Err(err_data("RG del vec truncated")); }
            let is_deleted = (body[pos + local_idx / 8] >> (local_idx % 8)) & 1 == 1;
            if is_deleted { return Ok(None); }
            pos += del_vec_len;

            let schema = &footer.schema;
            let col_count = schema.column_count();
            let null_bitmap_len = (rg_rows + 7) / 8;
            let mut result = Vec::with_capacity(col_count + 1);
            result.push(("_id".to_string(), Value::Int64(read_id as i64)));

            for col_idx in 0..col_count {
                if pos + null_bitmap_len > body.len() { break; }
                let null_bytes = &body[pos..pos + null_bitmap_len];
                pos += null_bitmap_len;
                let is_null = (null_bytes[local_idx / 8] >> (local_idx % 8)) & 1 == 1;
                let col_type = schema.columns[col_idx].1;
                let col_name = &schema.columns[col_idx].0;

                if is_null {
                    // Skip column data
                    let consumed = if encoding_version >= 1 {
                        skip_column_encoded(&body[pos..], col_type)?
                    } else {
                        ColumnData::skip_bytes_typed(&body[pos..], col_type)?
                    };
                    pos += consumed;
                    result.push((col_name.clone(), Value::Null));
                    continue;
                }

                // Extract single value from column data at local_idx
                let col_bytes = &body[pos..];
                let enc_offset = if encoding_version >= 1 { 1 } else { 0 };
                let encoding = if encoding_version >= 1 { col_bytes[0] } else { COL_ENCODING_PLAIN };
                let data_bytes = &col_bytes[enc_offset..];

                let val = match (encoding, col_type) {
                    (COL_ENCODING_PLAIN, ColumnType::Int64 | ColumnType::Int8 | ColumnType::Int16 |
                     ColumnType::Int32 | ColumnType::UInt8 | ColumnType::UInt16 | ColumnType::UInt32 |
                     ColumnType::UInt64 | ColumnType::Timestamp | ColumnType::Date) => {
                        // Plain Int64: [count:u64][data: count*8]
                        let off = 8 + local_idx * 8;
                        if off + 8 <= data_bytes.len() {
                            Value::Int64(i64::from_le_bytes(data_bytes[off..off+8].try_into().unwrap()))
                        } else { Value::Null }
                    }
                    (COL_ENCODING_PLAIN, ColumnType::Float64 | ColumnType::Float32) => {
                        let off = 8 + local_idx * 8;
                        if off + 8 <= data_bytes.len() {
                            Value::Float64(f64::from_le_bytes(data_bytes[off..off+8].try_into().unwrap()))
                        } else { Value::Null }
                    }
                    (COL_ENCODING_PLAIN, ColumnType::Bool) => {
                        // Plain Bool: [len:u64][packed bits]
                        let byte_off = 8 + local_idx / 8;
                        let bit = local_idx % 8;
                        if byte_off < data_bytes.len() {
                            Value::Bool((data_bytes[byte_off] >> bit) & 1 == 1)
                        } else { Value::Null }
                    }
                    (COL_ENCODING_PLAIN, ColumnType::String) => {
                        // Plain String: [count:u64][offsets:(count+1)*4][data_len:u64][data]
                        if data_bytes.len() >= 8 {
                            let count = u64::from_le_bytes(data_bytes[0..8].try_into().unwrap()) as usize;
                            let off_start = 8 + local_idx * 4;
                            let off_end = 8 + (local_idx + 1) * 4;
                            if off_end + 4 <= data_bytes.len() && local_idx < count {
                                let s = u32::from_le_bytes(data_bytes[off_start..off_start+4].try_into().unwrap()) as usize;
                                let e = u32::from_le_bytes(data_bytes[off_end..off_end+4].try_into().unwrap()) as usize;
                                let offsets_end = 8 + (count + 1) * 4;
                                let data_len_off = offsets_end;
                                if data_len_off + 8 <= data_bytes.len() {
                                    let data_start = data_len_off + 8;
                                    if data_start + e <= data_bytes.len() {
                                        Value::String(std::str::from_utf8(&data_bytes[data_start + s..data_start + e]).unwrap_or("").to_string())
                                    } else { Value::Null }
                                } else { Value::Null }
                            } else { Value::Null }
                        } else { Value::Null }
                    }
                    (COL_ENCODING_PLAIN, ColumnType::Binary) => {
                        // Plain Binary: same layout as String: [count:u64][offsets:(count+1)*4][data_len:u64][data]
                        if data_bytes.len() >= 8 {
                            let count = u64::from_le_bytes(data_bytes[0..8].try_into().unwrap()) as usize;
                            let off_start = 8 + local_idx * 4;
                            let off_end = 8 + (local_idx + 1) * 4;
                            if off_end + 4 <= data_bytes.len() && local_idx < count {
                                let s = u32::from_le_bytes(data_bytes[off_start..off_start+4].try_into().unwrap()) as usize;
                                let e = u32::from_le_bytes(data_bytes[off_end..off_end+4].try_into().unwrap()) as usize;
                                let offsets_end = 8 + (count + 1) * 4;
                                let data_len_off = offsets_end;
                                if data_len_off + 8 <= data_bytes.len() {
                                    let data_start = data_len_off + 8;
                                    if data_start + e <= data_bytes.len() {
                                        Value::Binary(data_bytes[data_start + s..data_start + e].to_vec())
                                    } else { Value::Null }
                                } else { Value::Null }
                            } else { Value::Null }
                        } else { Value::Null }
                    }
                    (COL_ENCODING_PLAIN, ColumnType::StringDict) => {
                        // StringDict: [row_count:u64][dict_size:u64][indices:row_count*4][dict_offsets:dict_size*4][dict_data_len:u64][dict_data]
                        if data_bytes.len() >= 16 {
                            let row_count = u64::from_le_bytes(data_bytes[0..8].try_into().unwrap()) as usize;
                            let dict_size = u64::from_le_bytes(data_bytes[8..16].try_into().unwrap()) as usize;
                            let idx_off = 16 + local_idx * 4;
                            if idx_off + 4 <= data_bytes.len() && local_idx < row_count {
                                let dict_idx = u32::from_le_bytes(data_bytes[idx_off..idx_off+4].try_into().unwrap());
                                if dict_idx == 0 { Value::Null } else {
                                    let di = (dict_idx - 1) as usize;
                                    let dict_off_start = 16 + row_count * 4;
                                    let do_off = dict_off_start + di * 4;
                                    let do_off_next = dict_off_start + (di + 1) * 4;
                                    if do_off_next + 4 <= data_bytes.len() && di < dict_size {
                                        let ds = u32::from_le_bytes(data_bytes[do_off..do_off+4].try_into().unwrap()) as usize;
                                        let de = u32::from_le_bytes(data_bytes[do_off_next..do_off_next+4].try_into().unwrap()) as usize;
                                        let dict_data_len_off = dict_off_start + dict_size * 4;
                                        if dict_data_len_off + 8 <= data_bytes.len() {
                                            let dict_data_start = dict_data_len_off + 8;
                                            if dict_data_start + de <= data_bytes.len() {
                                                Value::String(std::str::from_utf8(&data_bytes[dict_data_start + ds..dict_data_start + de]).unwrap_or("").to_string())
                                            } else { Value::Null }
                                        } else { Value::Null }
                                    } else { Value::Null }
                                }
                            } else { Value::Null }
                        } else { Value::Null }
                    }
                    _ => {
                        // RLE/Bitpack/other: fallback to full decode, extract single value
                        let (col_data, _consumed) = if encoding_version >= 1 {
                            read_column_encoded(col_bytes, col_type)?
                        } else {
                            ColumnData::from_bytes_typed(col_bytes, col_type)?
                        };
                        match &col_data {
                            ColumnData::Int64(v) => if local_idx < v.len() { Value::Int64(v[local_idx]) } else { Value::Null },
                            ColumnData::Float64(v) => if local_idx < v.len() { Value::Float64(v[local_idx]) } else { Value::Null },
                            ColumnData::Bool { data, len } => {
                                if local_idx < *len {
                                    Value::Bool((data[local_idx / 8] >> (local_idx % 8)) & 1 == 1)
                                } else { Value::Null }
                            }
                            ColumnData::String { offsets, data } => {
                                let count = offsets.len().saturating_sub(1);
                                if local_idx < count {
                                    let s = offsets[local_idx] as usize;
                                    let e = offsets[local_idx + 1] as usize;
                                    Value::String(std::str::from_utf8(&data[s..e]).unwrap_or("").to_string())
                                } else { Value::Null }
                            }
                            ColumnData::Binary { offsets, data } => {
                                let count = offsets.len().saturating_sub(1);
                                if local_idx < count {
                                    let s = offsets[local_idx] as usize;
                                    let e = offsets[local_idx + 1] as usize;
                                    Value::Binary(data[s..e].to_vec())
                                } else { Value::Null }
                            }
                            ColumnData::StringDict { indices, dict_offsets, dict_data, .. } => {
                                if local_idx < indices.len() {
                                    let di = indices[local_idx];
                                    if di == 0 { Value::Null } else {
                                        let idx = (di - 1) as usize;
                                        if idx < dict_offsets.len() {
                                            let s = dict_offsets[idx] as usize;
                                            let e = if idx + 1 < dict_offsets.len() { dict_offsets[idx + 1] as usize } else { dict_data.len() };
                                            Value::String(std::str::from_utf8(&dict_data[s..e]).unwrap_or("").to_string())
                                        } else { Value::Null }
                                    }
                                } else { Value::Null }
                            }
                            _ => Value::Null,
                        }
                    }
                };

                // Skip full column to advance pos
                let consumed = if encoding_version >= 1 {
                    skip_column_encoded(&body[pos..], col_type)?
                } else {
                    ColumnData::skip_bytes_typed(&body[pos..], col_type)?
                };
                pos += consumed;
                result.push((col_name.clone(), val));
            }

            drop(mmap_guard);
            drop(file_guard);
            return Ok(Some(result));
        }
        
        let row_idx = match self.get_row_idx(id) {
            Some(idx) => idx,
            None => return Ok(None),
        };
        
        let schema = self.schema.read();
        let columns = self.columns.read();
        let nulls = self.nulls.read();
        
        let mut result = Vec::with_capacity(schema.column_count() + 1);
        result.push(("_id".to_string(), Value::Int64(id as i64)));
        
        for (col_idx, (col_name, _)) in schema.columns.iter().enumerate() {
            // Check null
            if col_idx < nulls.len() && !nulls[col_idx].is_empty() {
                let b = row_idx / 8; let bit = row_idx % 8;
                if b < nulls[col_idx].len() && (nulls[col_idx][b] >> bit) & 1 == 1 {
                    result.push((col_name.clone(), Value::Null));
                    continue;
                }
            }
            
            if col_idx >= columns.len() {
                result.push((col_name.clone(), Value::Null));
                continue;
            }
            
            let val = match &columns[col_idx] {
                ColumnData::Int64(v) => {
                    if row_idx < v.len() { Value::Int64(v[row_idx]) } else { Value::Null }
                }
                ColumnData::Float64(v) => {
                    if row_idx < v.len() { Value::Float64(v[row_idx]) } else { Value::Null }
                }
                ColumnData::String { offsets, data } => {
                    let count = offsets.len().saturating_sub(1);
                    if row_idx < count {
                        let s = offsets[row_idx] as usize;
                        let e = offsets[row_idx + 1] as usize;
                        Value::String(std::str::from_utf8(&data[s..e]).unwrap_or("").to_string())
                    } else { Value::Null }
                }
                ColumnData::Bool { data, len } => {
                    if row_idx < *len {
                        let b = row_idx / 8; let bit = row_idx % 8;
                        if b < data.len() {
                            Value::Bool((data[b] >> bit) & 1 == 1)
                        } else { Value::Null }
                    } else { Value::Null }
                }
                ColumnData::Binary { offsets, data } => {
                    let count = offsets.len().saturating_sub(1);
                    if row_idx < count {
                        let s = offsets[row_idx] as usize;
                        let e = offsets[row_idx + 1] as usize;
                        Value::Binary(data[s..e].to_vec())
                    } else { Value::Null }
                }
                ColumnData::StringDict { indices, dict_offsets, dict_data } => {
                    if row_idx < indices.len() {
                        let idx = indices[row_idx];
                        if idx == 0 { Value::Null } else {
                            let di = (idx - 1) as usize;
                            if di + 1 < dict_offsets.len() {
                                let s = dict_offsets[di] as usize;
                                let e = dict_offsets[di + 1] as usize;
                                Value::String(std::str::from_utf8(&dict_data[s..e]).unwrap_or("").to_string())
                            } else { Value::Null }
                        }
                    } else { Value::Null }
                }
                _ => Value::Null,
            };
            result.push((col_name.clone(), val));
        }
        
        Ok(Some(result))
    }

    /// Fast SELECT * LIMIT N: read first N non-deleted rows directly from V4 columns
    /// Returns (column_names, rows) where each row is Vec<Value>
    /// Bypasses SQL parsing and Arrow conversion entirely
    pub fn read_rows_limit_values(&self, limit: usize) -> io::Result<Option<(Vec<String>, Vec<Vec<crate::data::Value>>)>> {
        use crate::data::Value;
        
        let is_v4 = self.is_v4_format();
        if !is_v4 { return Ok(None); }
        
        // MMAP PATH: scan from disk if no in-memory data
        if !self.has_v4_in_memory_data() {
            return self.read_rows_limit_values_mmap(limit);
        }
        
        let schema = self.schema.read();
        let columns = self.columns.read();
        let nulls = self.nulls.read();
        let ids = self.ids.read();
        let deleted = self.deleted.read();
        let total_rows = ids.len();
        let has_deleted = deleted.iter().any(|&b| b != 0);
        
        // Build column names
        let mut col_names = Vec::with_capacity(schema.column_count() + 1);
        col_names.push("_id".to_string());
        for (name, _) in &schema.columns {
            col_names.push(name.clone());
        }
        
        let actual_limit = limit.min(total_rows);
        let mut rows: Vec<Vec<Value>> = Vec::with_capacity(actual_limit);
        let mut emitted = 0usize;
        
        for row_idx in 0..total_rows {
            if emitted >= limit { break; }
            // Skip deleted
            if has_deleted {
                let b = row_idx / 8; let bit = row_idx % 8;
                if b < deleted.len() && (deleted[b] >> bit) & 1 != 0 { continue; }
            }
            
            let mut row = Vec::with_capacity(col_names.len());
            // _id
            row.push(if row_idx < ids.len() { Value::Int64(ids[row_idx] as i64) } else { Value::Null });
            
            for col_idx in 0..schema.column_count() {
                // Null check
                if col_idx < nulls.len() && !nulls[col_idx].is_empty() {
                    let b = row_idx / 8; let bit = row_idx % 8;
                    if b < nulls[col_idx].len() && (nulls[col_idx][b] >> bit) & 1 == 1 {
                        row.push(Value::Null);
                        continue;
                    }
                }
                if col_idx >= columns.len() { row.push(Value::Null); continue; }
                
                let val = match &columns[col_idx] {
                    ColumnData::Int64(v) => {
                        if row_idx < v.len() { Value::Int64(v[row_idx]) } else { Value::Null }
                    }
                    ColumnData::Float64(v) => {
                        if row_idx < v.len() { Value::Float64(v[row_idx]) } else { Value::Null }
                    }
                    ColumnData::String { offsets, data } => {
                        let count = offsets.len().saturating_sub(1);
                        if row_idx < count {
                            let s = offsets[row_idx] as usize;
                            let e = offsets[row_idx + 1] as usize;
                            Value::String(std::str::from_utf8(&data[s..e]).unwrap_or("").to_string())
                        } else { Value::Null }
                    }
                    ColumnData::Bool { data, len } => {
                        if row_idx < *len {
                            let b = row_idx / 8; let bit = row_idx % 8;
                            if b < data.len() { Value::Bool((data[b] >> bit) & 1 == 1) } else { Value::Null }
                        } else { Value::Null }
                    }
                    ColumnData::Binary { offsets, data } => {
                        let count = offsets.len().saturating_sub(1);
                        if row_idx < count {
                            let s = offsets[row_idx] as usize;
                            let e = offsets[row_idx + 1] as usize;
                            Value::Binary(data[s..e].to_vec())
                        } else { Value::Null }
                    }
                    ColumnData::StringDict { indices, dict_offsets, dict_data } => {
                        if row_idx < indices.len() {
                            let idx = indices[row_idx];
                            if idx == 0 { Value::Null } else {
                                let di = (idx - 1) as usize;
                                if di + 1 < dict_offsets.len() {
                                    let s = dict_offsets[di] as usize;
                                    let e = dict_offsets[di + 1] as usize;
                                    Value::String(std::str::from_utf8(&dict_data[s..e]).unwrap_or("").to_string())
                                } else { Value::Null }
                            }
                        } else { Value::Null }
                    }
                    _ => Value::Null,
                };
                row.push(val);
            }
            rows.push(row);
            emitted += 1;
        }
        
        Ok(Some((col_names, rows)))
    }

    /// MMAP PATH: Fast SELECT * LIMIT N
    /// For small limits on mmap-only data, return None to let SQL path use arrow_batch_cache
    fn read_rows_limit_values_mmap(&self, _limit: usize) -> io::Result<Option<(Vec<String>, Vec<Vec<crate::data::Value>>)>> {
        // Return None to fall through to SQL execution path which uses
        // arrow_batch_cache (populated on warmup, subsequent reads are O(1) slice)
        Ok(None)
    }

    /// OPTIMIZED: Read multiple rows by IDs using O(1) index lookups
    /// Returns Vec of (id, row_data) for found rows
    pub fn read_rows_by_ids(&self, ids: &[u64], column_names: Option<&[&str]>) -> io::Result<Vec<(u64, HashMap<String, ColumnData>)>> {
        if ids.is_empty() {
            return Ok(Vec::new());
        }
        
        let is_v4 = self.is_v4_format();
        if is_v4 && !self.has_v4_in_memory_data() {
            // MMAP PATH: delegate to read_row_by_id per ID
            let mut results = Vec::with_capacity(ids.len());
            for &id in ids {
                if let Some(row) = self.read_row_by_id(id, column_names)? {
                    results.push((id, row));
                }
            }
            return Ok(results);
        }
        
        // Build id_to_idx if needed
        self.ensure_id_index();
        let id_to_idx = self.id_to_idx.read();
        let map = id_to_idx.as_ref().unwrap();
        
        // Collect valid row indices
        let mut valid_ids_indices: Vec<(u64, usize)> = Vec::with_capacity(ids.len());
        for &id in ids {
            if let Some(&row_idx) = map.get(&id) {
                if !self.is_deleted(row_idx) {
                    valid_ids_indices.push((id, row_idx));
                }
            }
        }
        
        if valid_ids_indices.is_empty() {
            return Ok(Vec::new());
        }
        
        let indices: Vec<usize> = valid_ids_indices.iter().map(|(_, idx)| *idx).collect();
        drop(id_to_idx);
        
        // Read columns
        let schema = self.schema.read();
        
        let cols_to_read: Vec<(usize, String, ColumnType)> = if let Some(names) = column_names {
            names.iter()
                .filter_map(|&name| {
                    if name == "_id" {
                        None
                    } else {
                        schema.get_index(name).map(|idx| {
                            (idx, name.to_string(), schema.columns[idx].1)
                        })
                    }
                })
                .collect()
        } else {
            schema.columns.iter().enumerate()
                .map(|(idx, (name, dtype))| (idx, name.clone(), *dtype))
                .collect()
        };
        
        let total_rows = self.header.read().row_count as usize;
        drop(schema);
        
        // Read all columns for all indices
        let mut column_data: HashMap<String, ColumnData> = HashMap::new();
        let include_id = column_names.map(|cols| cols.contains(&"_id")).unwrap_or(true);
        
        for (col_idx, col_name, col_type) in cols_to_read {
            let col_data = self.read_column_scattered_auto(col_idx, col_type, &indices, total_rows, is_v4)?;
            column_data.insert(col_name, col_data);
        }
        
        // Split into per-row results
        let mut results = Vec::with_capacity(valid_ids_indices.len());
        for (i, (id, _)) in valid_ids_indices.iter().enumerate() {
            let mut row_data = HashMap::new();
            if include_id {
                row_data.insert("_id".to_string(), ColumnData::Int64(vec![*id as i64]));
            }
            for (col_name, col_data) in &column_data {
                let single_val = col_data.filter_by_indices(&[i]);
                row_data.insert(col_name.clone(), single_val);
            }
            results.push((*id, row_data));
        }
        
        Ok(results)
    }

    /// Get the count of non-deleted rows (includes delta rows)
    pub fn active_row_count(&self) -> u64 {
        let base_active = self.active_count.load(std::sync::atomic::Ordering::Relaxed);
        let delta_rows = self.delta_row_count() as u64;
        base_active + delta_rows
    }

    /// Drop a column from schema (logical delete - data stays but column is removed from schema)
    /// When save() is called, only columns in schema will be written to file
    pub fn drop_column(&self, name: &str) -> io::Result<()> {
        let mut schema = self.schema.write();
        
        // Find column index
        let idx = match schema.get_index(name) {
            Some(idx) => idx,
            None => return Err(io::Error::new(io::ErrorKind::NotFound, format!("Column '{}' not found", name))),
        };
        
        // Remove from schema (logical delete)
        schema.columns.remove(idx);
        schema.name_to_idx.remove(name);
        
        // Rebuild name_to_idx with updated indices
        // Collect names first to avoid borrow conflict
        let names: Vec<String> = schema.columns.iter().map(|(n, _)| n.clone()).collect();
        schema.name_to_idx.clear();
        for (i, n) in names.into_iter().enumerate() {
            schema.name_to_idx.insert(n, i);
        }
        
        // Also remove from in-memory structures to keep them in sync with schema
        // This ensures save() writes correct data
        {
            let mut columns = self.columns.write();
            let mut nulls = self.nulls.write();
            let mut column_index = self.column_index.write();
            
            if idx < columns.len() {
                columns.remove(idx);
            }
            if idx < nulls.len() {
                nulls.remove(idx);
            }
            if idx < column_index.len() {
                column_index.remove(idx);
            }
        }
        
        // Update header column count
        {
            let mut header = self.header.write();
            header.column_count = schema.column_count() as u32;
        }
        
        Ok(())
    }

    /// Add a new column to schema and storage with padding for existing rows
    pub fn add_column_with_padding(&self, name: &str, dtype: crate::data::DataType) -> io::Result<()> {
        use crate::data::DataType;
        
        // For V4, schema is updated via footer; data stays on disk (mmap)
        self.load_all_columns_into_memory()?;
        
        let col_type = match dtype {
            DataType::Int64 | DataType::Int32 | DataType::Int16 | DataType::Int8 => ColumnType::Int64,
            DataType::Float64 | DataType::Float32 => ColumnType::Float64,
            DataType::String => ColumnType::String,
            DataType::Bool => ColumnType::Bool,
            DataType::Binary => ColumnType::Binary,
            DataType::Timestamp => ColumnType::Timestamp,
            DataType::Date => ColumnType::Date,
            _ => ColumnType::String,
        };
        
        let mut schema = self.schema.write();
        let mut columns = self.columns.write();
        let mut nulls = self.nulls.write();
        // Use header.row_count for V4 (IDs may not be loaded in mmap-only mode)
        let existing_row_count = {
            let header = self.header.read();
            let from_header = header.row_count as usize;
            drop(header);
            let ids = self.ids.read();
            let from_ids = ids.len();
            drop(ids);
            from_header.max(from_ids)
        };
        
        // Add to schema
        let idx = schema.add_column(name, col_type);
        
        // Ensure columns vector is large enough
        while columns.len() <= idx {
            let mut col = ColumnData::new(col_type);
            // Pad with defaults for existing rows
            match &mut col {
                ColumnData::Int64(v) => v.resize(existing_row_count, 0),
                ColumnData::Float64(v) => v.resize(existing_row_count, 0.0),
                ColumnData::String { offsets, .. } => {
                    for _ in 0..existing_row_count {
                        offsets.push(0);
                    }
                }
                ColumnData::Binary { offsets, .. } => {
                    for _ in 0..existing_row_count {
                        offsets.push(0);
                    }
                }
                ColumnData::Bool { len, .. } => {
                    *len = existing_row_count;
                }
                ColumnData::StringDict { indices, .. } => {
                    indices.resize(existing_row_count, 0);
                }
            }
            columns.push(col);
            nulls.push(Vec::new());
        }
        
        // Update header
        {
            let mut header = self.header.write();
            header.column_count = schema.column_count() as u32;
        }
        
        Ok(())
    }

    /// Replace a row by ID (delete old row, insert new with SAME ID)
    /// Returns true if successful
    pub fn replace(&self, id: u64, data: &HashMap<String, ColumnValue>) -> io::Result<bool> {
        // Check if ID exists
        if !self.exists(id) {
            return Ok(false);
        }
        
        // Delete the old row (soft delete)
        self.delete(id);
        
        // Convert data to typed columns for insert_typed
        let mut int_columns: HashMap<String, Vec<i64>> = HashMap::new();
        let mut float_columns: HashMap<String, Vec<f64>> = HashMap::new();
        let mut string_columns: HashMap<String, Vec<String>> = HashMap::new();
        let mut binary_columns: HashMap<String, Vec<Vec<u8>>> = HashMap::new();
        let mut bool_columns: HashMap<String, Vec<bool>> = HashMap::new();
        
        for (name, val) in data {
            match val {
                ColumnValue::Int64(v) => { int_columns.insert(name.clone(), vec![*v]); }
                ColumnValue::Float64(v) => { float_columns.insert(name.clone(), vec![*v]); }
                ColumnValue::String(v) => { string_columns.insert(name.clone(), vec![v.clone()]); }
                ColumnValue::Binary(v) => { binary_columns.insert(name.clone(), vec![v.clone()]); }
                ColumnValue::Bool(v) => { bool_columns.insert(name.clone(), vec![*v]); }
                ColumnValue::Null => {}
            }
        }
        
        // Use insert_typed but override the ID
        // First, determine row count (should be 1)
        let row_count = 1;
        
        // Instead of using next_id, we'll use the original ID
        let ids = vec![id];
        
        // Ensure schema has all columns and pad new columns with defaults
        {
            let mut schema = self.schema.write();
            let mut columns = self.columns.write();
            let mut nulls = self.nulls.write();
            let ids = self.ids.read();
            let existing_row_count = ids.len();
            drop(ids);
            
            for name in int_columns.keys() {
                let idx = schema.add_column(name, ColumnType::Int64);
                while columns.len() <= idx {
                    // New column - pad with defaults for existing rows
                    let mut col = ColumnData::new(ColumnType::Int64);
                    if let ColumnData::Int64(v) = &mut col {
                        v.resize(existing_row_count, 0);
                    }
                    columns.push(col);
                    nulls.push(Vec::new());
                }
            }
            for name in float_columns.keys() {
                let idx = schema.add_column(name, ColumnType::Float64);
                while columns.len() <= idx {
                    let mut col = ColumnData::new(ColumnType::Float64);
                    if let ColumnData::Float64(v) = &mut col {
                        v.resize(existing_row_count, 0.0);
                    }
                    columns.push(col);
                    nulls.push(Vec::new());
                }
            }
            for name in string_columns.keys() {
                let idx = schema.add_column(name, ColumnType::String);
                while columns.len() <= idx {
                    let mut col = ColumnData::new(ColumnType::String);
                    if let ColumnData::String { offsets, .. } = &mut col {
                        // For strings, push empty string offsets for existing rows
                        for _ in 0..existing_row_count {
                            offsets.push(0);
                        }
                    }
                    columns.push(col);
                    nulls.push(Vec::new());
                }
            }
            for name in binary_columns.keys() {
                let idx = schema.add_column(name, ColumnType::Binary);
                while columns.len() <= idx {
                    let mut col = ColumnData::new(ColumnType::Binary);
                    if let ColumnData::Binary { offsets, .. } = &mut col {
                        for _ in 0..existing_row_count {
                            offsets.push(0);
                        }
                    }
                    columns.push(col);
                    nulls.push(Vec::new());
                }
            }
            for name in bool_columns.keys() {
                let idx = schema.add_column(name, ColumnType::Bool);
                while columns.len() <= idx {
                    let mut col = ColumnData::new(ColumnType::Bool);
                    if let ColumnData::Bool { len, .. } = &mut col {
                        *len = existing_row_count;
                    }
                    columns.push(col);
                    nulls.push(Vec::new());
                }
            }
        }
        
        // Append ID
        self.ids.write().extend_from_slice(&ids);
        
        // Append column data
        {
            let schema = self.schema.read();
            let mut columns = self.columns.write();
            
            for (name, values) in int_columns {
                if let Some(idx) = schema.get_index(&name) {
                    columns[idx].extend_i64(&values);
                }
            }
            for (name, values) in float_columns {
                if let Some(idx) = schema.get_index(&name) {
                    columns[idx].extend_f64(&values);
                }
            }
            for (name, values) in string_columns {
                if let Some(idx) = schema.get_index(&name) {
                    for v in &values {
                        columns[idx].push_string(v);
                    }
                }
            }
            for (name, values) in binary_columns {
                if let Some(idx) = schema.get_index(&name) {
                    for v in &values {
                        columns[idx].push_bytes(v);
                    }
                }
            }
            for (name, values) in bool_columns {
                if let Some(idx) = schema.get_index(&name) {
                    for v in values {
                        columns[idx].push_bool(v);
                    }
                }
            }
            
            // Pad any schema columns not in the replacement data with defaults + null
            let expected_len = self.ids.read().len();
            let mut nulls = self.nulls.write();
            for col_idx in 0..schema.column_count() {
                if col_idx < columns.len() && columns[col_idx].len() < expected_len {
                    // This column wasn't in the replacement — pad with default
                    let deficit = expected_len - columns[col_idx].len();
                    for _ in 0..deficit {
                        match &mut columns[col_idx] {
                            ColumnData::Int64(v) => v.push(0),
                            ColumnData::Float64(v) => v.push(0.0),
                            ColumnData::String { offsets, .. } => {
                                offsets.push(*offsets.last().unwrap_or(&0));
                            }
                            ColumnData::Binary { offsets, .. } => {
                                offsets.push(*offsets.last().unwrap_or(&0));
                            }
                            ColumnData::Bool { data, len } => {
                                let byte_idx = *len / 8;
                                if byte_idx >= data.len() { data.push(0); }
                                *len += 1;
                            }
                            ColumnData::StringDict { indices, .. } => indices.push(0),
                        }
                    }
                    // Mark padded rows as null
                    if col_idx >= nulls.len() {
                        nulls.resize(col_idx + 1, Vec::new());
                    }
                    let total_rows = expected_len;
                    let null_len = (total_rows + 7) / 8;
                    nulls[col_idx].resize(null_len, 0);
                    for row in (total_rows - deficit)..total_rows {
                        nulls[col_idx][row / 8] |= 1 << (row % 8);
                    }
                }
            }
        }
        
        // Update header
        {
            let mut header = self.header.write();
            header.row_count = self.ids.read().len() as u64;
            header.column_count = self.schema.read().column_count() as u32;
        }
        
        // Update id_to_idx mapping only if it's already built
        {
            let ids_guard = self.ids.read();
            let mut id_to_idx = self.id_to_idx.write();
            if let Some(map) = id_to_idx.as_mut() {
                let row_idx = ids_guard.len() - 1;
                map.insert(id, row_idx);
            }
        }
        
        // Extend deleted bitmap
        {
            let mut deleted = self.deleted.write();
            let new_len = (self.ids.read().len() + 7) / 8;
            deleted.resize(new_len, 0);
        }
        
        Ok(true)
    }

    // ========================================================================
    // Persistence
    // ========================================================================

    /// Check if a string column should use dictionary encoding
    /// Returns true if unique values < 20% of row count and row count > 1000
    fn should_dict_encode(col: &ColumnData) -> bool {
        if let ColumnData::String { offsets, data } = col {
            let row_count = offsets.len().saturating_sub(1);
            if row_count < 1000 {
                return false;
            }
            // Estimate unique values by sampling
            use ahash::AHashSet;
            let sample_size = (row_count / 10).min(1000);
            let mut unique: AHashSet<&[u8]> = AHashSet::with_capacity(sample_size);
            for i in 0..sample_size {
                let idx = i * 10; // Sample every 10th row
                if idx < row_count {
                    let start = offsets[idx] as usize;
                    let end = offsets[idx + 1] as usize;
                    unique.insert(&data[start..end]);
                }
            }
            // Use dictionary if cardinality < 20% of sampled rows
            unique.len() < sample_size / 5
        } else {
            false
        }
    }

    /// Save to file (full rewrite with V3 format)
    /// 
    /// MEMORY OPTIMIZED: Processes one column at a time using placeholder + seek-back.
    /// Peak memory = original columns (already in memory) + 1 filtered column copy,
    /// instead of original columns + ALL filtered column copies.
    /// 
    /// Automatically converts low-cardinality string columns to dictionary encoding.
    pub fn save(&self) -> io::Result<()> {
        // OPTIMIZATION: For existing V4 files with only deletions (no new rows,
        // no schema changes), update deletion vectors in-place instead of full rewrite.
        // All other cases use the proven save_v4() full-rewrite path.
        // Note: append optimization is handled at engine level (write_typed→append_row_group).
        let header = self.header.read();
        let is_v4 = header.version == FORMAT_VERSION_V4 && header.footer_offset > 0;
        drop(header);

        if is_v4 {
            let on_disk_rows = self.persisted_row_count.load(Ordering::SeqCst) as usize;
            let ids = self.ids.read();
            let in_memory_ids = ids.len();
            let has_new_rows = in_memory_ids > 0;
            let base_loaded = self.v4_base_loaded.load(Ordering::SeqCst);
            let has_unloaded_base = on_disk_rows > 0 && in_memory_ids > 0 && !base_loaded;
            drop(ids);

            // If base data isn't loaded but we have new rows, append incrementally
            if has_unloaded_base {
                let ids = self.ids.read();
                let new_ids: Vec<u64> = ids.clone();
                drop(ids);
                let cols = self.columns.read();
                let new_cols: Vec<ColumnData> = cols.clone();
                drop(cols);
                let nulls = self.nulls.read();
                let new_nulls: Vec<Vec<u8>> = nulls.clone();
                drop(nulls);
                self.pending_rows.store(0, Ordering::SeqCst);
                return self.append_row_group(&new_ids, &new_cols, &new_nulls);
            }

            if !has_new_rows && !base_loaded && on_disk_rows > 0 {
                // Schema-only change (add/drop/rename column) on V4 mmap-only.
                // Base data is NOT in memory — must NOT call save_v4() which would
                // rewrite with empty data and lose everything.
                // Instead, update just the footer schema on disk.
                return self.update_v4_footer_schema();
            }

            if !has_new_rows {
                let deleted = self.deleted.read();
                let has_deletes = deleted.iter().any(|&b| b != 0);
                if has_deletes {
                    // Count deleted rows for compaction threshold
                    let del_count = (0..on_disk_rows).filter(|&i| {
                        let byte_idx = i / 8;
                        let bit_idx = i % 8;
                        byte_idx < deleted.len() && (deleted[byte_idx] >> bit_idx) & 1 == 1
                    }).count();
                    drop(deleted);
                    let ratio = if on_disk_rows > 0 { del_count as f64 / on_disk_rows as f64 } else { 0.0 };

                    if ratio <= 0.5 {
                        // Low deletion ratio → update deletion vectors in-place
                        self.pending_rows.store(0, Ordering::SeqCst);
                        // Also persist delta store if it has pending changes
                        if self.has_pending_deltas() {
                            let _ = self.save_delta_store();
                        }
                        return self.save_deletion_vectors();
                    }
                    // High deletion ratio → full rewrite to reclaim space (fall through)
                }
            }
        }

        self.pending_rows.store(0, Ordering::SeqCst);
        let result = self.save_v4();
        // After full rewrite, clear delta store (deltas are now in the base file)
        if result.is_ok() {
            let _ = self.clear_delta_store();
            // WAL checkpoint: all data is persisted, truncate WAL to prevent unbounded growth
            self.checkpoint_wal();
        }
        result
    }
    
    /// Checkpoint WAL: truncate the WAL file after a successful save.
    /// All WAL records are now redundant since data is fully persisted to .apex.
    fn checkpoint_wal(&self) {
        let mut wal_writer = self.wal_writer.write();
        if wal_writer.is_some() {
            // Drop the existing writer to release file handle
            *wal_writer = None;
            // Recreate WAL file (truncates old content)
            let wal_path = Self::wal_path(&self.path);
            let next_id = self.next_id.load(Ordering::SeqCst);
            if let Ok(writer) = super::incremental::WalWriter::create(&wal_path, next_id) {
                *wal_writer = Some(writer);
            }
        }
        // Clear WAL buffer
        let mut wal_buffer = self.wal_buffer.write();
        wal_buffer.clear();
    }
    
    // ========================================================================
    // V4 Row Group Format — Save / Open / Append
    // ========================================================================
    
    /// Slice a null bitmap for a contiguous row range [start, end).
    /// OPTIMIZATION: uses bulk memcpy when start is byte-aligned.
    fn slice_null_bitmap(nulls: &[u8], start: usize, end: usize) -> Vec<u8> {
        let count = end.saturating_sub(start);
        if count == 0 || nulls.is_empty() {
            return vec![0u8; (count + 7) / 8];
        }
        let result_len = (count + 7) / 8;
        if start % 8 == 0 {
            let src_byte = start / 8;
            let copy_len = result_len.min(nulls.len().saturating_sub(src_byte));
            let mut result = vec![0u8; result_len];
            if copy_len > 0 {
                result[..copy_len].copy_from_slice(&nulls[src_byte..src_byte + copy_len]);
            }
            let tail_bits = count % 8;
            if tail_bits > 0 && result_len > 0 {
                result[result_len - 1] &= (1u8 << tail_bits) - 1;
            }
            return result;
        }
        let mut result = vec![0u8; result_len];
        for i in 0..count {
            let ob = (start + i) / 8;
            let obit = (start + i) % 8;
            if ob < nulls.len() && (nulls[ob] >> obit) & 1 == 1 {
                result[i / 8] |= 1 << (i % 8);
            }
        }
        result
    }
    
    /// Save in V4 Row Group format.
    /// Splits data into Row Groups of DEFAULT_ROW_GROUP_SIZE rows each.
    /// Each RG is self-contained with IDs, deletion vector, and per-column data.
    ///
    /// V4 File Layout:
    /// ```text
    /// [Header 256B] [RG0] [RG1] ... [V4Footer]
    /// ```
    pub fn save_v4(&self) -> io::Result<()> {
        self.mmap_cache.write().invalidate();
        *self.file.write() = None;
        // On Windows, active mmaps prevent file truncate/write (OS error 1224).
        // Must invalidate ALL caches (engine cache + insert_cache + schema_cache + executor STORAGE_CACHE).
        // On Unix/Linux, only executor cache needs invalidation (mmaps don't block writes).
        #[cfg(target_os = "windows")]
        super::engine::engine().invalidate(&self.path);
        #[cfg(not(target_os = "windows"))]
        crate::query::ApexExecutor::invalidate_cache_for_path(&self.path);
        
        // Atomic write: write to .tmp file, then rename over the original.
        // If crash occurs mid-write, only the .tmp file is corrupted; original is intact.
        let tmp_path = self.path.with_extension("apex.tmp");
        let file = OpenOptions::new()
            .write(true).create(true).truncate(true)
            .open(&tmp_path)?;
        let mut writer = BufWriter::with_capacity(256 * 1024, file);
        
        // Phase 1: Build filtered (active) data under read guards.
        // This produces clean flat columns/ids/nulls with deleted rows removed
        // and missing columns padded. Used for both disk write and in-memory state.
        let active_ids: Vec<u64>;
        let mut active_columns: Vec<ColumnData>;
        let mut active_nulls: Vec<Vec<u8>>;
        let active_count: usize;
        let col_count: usize;
        let schema_clone: OnDemandSchema;
        
        {
            let schema = self.schema.read();
            let ids = self.ids.read();
            let columns = self.columns.read();
            let nulls = self.nulls.read();
            let deleted = self.deleted.read();
            
            col_count = schema.column_count();
            schema_clone = schema.clone();
            let has_deleted = deleted.iter().any(|&b| b != 0);
            
            if has_deleted {
                let indices: Vec<usize> = (0..ids.len())
                    .filter(|&i| {
                        let byte_idx = i / 8;
                        let bit_idx = i % 8;
                        byte_idx >= deleted.len() || (deleted[byte_idx] >> bit_idx) & 1 == 0
                    })
                    .collect();
                active_ids = indices.iter().map(|&i| ids[i]).collect();
                active_count = indices.len();
                
                active_columns = Vec::with_capacity(col_count);
                active_nulls = Vec::with_capacity(col_count);
                for col_idx in 0..col_count {
                    // Filter column data
                    if col_idx < columns.len() {
                        active_columns.push(columns[col_idx].filter_by_indices(&indices));
                    } else {
                        active_columns.push(Self::create_default_column(schema.columns[col_idx].1, active_count));
                    }
                    // Filter null bitmap
                    let orig_nulls = nulls.get(col_idx).map(|v| v.as_slice()).unwrap_or(&[]);
                    let null_len = (active_count + 7) / 8;
                    let mut nb = vec![0u8; null_len];
                    for (new_idx, &old_idx) in indices.iter().enumerate() {
                        let ob = old_idx / 8;
                        let obit = old_idx % 8;
                        if ob < orig_nulls.len() && (orig_nulls[ob] >> obit) & 1 == 1 {
                            nb[new_idx / 8] |= 1 << (new_idx % 8);
                        }
                    }
                    active_nulls.push(nb);
                }
            } else {
                active_ids = ids.to_vec();
                active_count = ids.len();
                
                active_columns = Vec::with_capacity(col_count);
                active_nulls = Vec::with_capacity(col_count);
                for col_idx in 0..col_count {
                    if col_idx < columns.len() {
                        active_columns.push(columns[col_idx].clone());
                    } else {
                        active_columns.push(Self::create_default_column(schema.columns[col_idx].1, active_count));
                    }
                    active_nulls.push(nulls.get(col_idx).map(|v| v.to_vec()).unwrap_or_default());
                }
            }
        } // All read guards dropped here
        
        // Phase 2: Write V4 format from active data (no lock contention).
        let rg_size = DEFAULT_ROW_GROUP_SIZE as usize;
        
        // Write placeholder header
        writer.write_all(&[0u8; HEADER_SIZE_V3])?;
        
        // Write Row Groups
        let mut rg_metas: Vec<RowGroupMeta> = Vec::new();
        let mut all_zone_maps: RgZoneMaps = Vec::new();
        let mut actual_col_types: Vec<ColumnType> = Vec::new();
        let mut chunk_start = 0;
        
        while chunk_start < active_count || (active_count == 0 && rg_metas.is_empty()) {
            let chunk_end = (chunk_start + rg_size).min(active_count);
            let chunk_rows = chunk_end - chunk_start;
            
            // Handle empty table — write one empty RG
            if active_count == 0 && rg_metas.is_empty() {
                let rg_offset = writer.stream_position()?;
                writer.write_all(MAGIC_ROW_GROUP)?;
                writer.write_all(&0u32.to_le_bytes())?;
                writer.write_all(&(col_count as u32).to_le_bytes())?;
                writer.write_all(&0u64.to_le_bytes())?;
                writer.write_all(&0u64.to_le_bytes())?;
                writer.write_all(&[0u8; 4])?;
                let rg_end = writer.stream_position()?;
                rg_metas.push(RowGroupMeta {
                    offset: rg_offset, data_size: rg_end - rg_offset,
                    row_count: 0, min_id: 0, max_id: 0, deletion_count: 0,
                });
                break;
            }
            
            let rg_offset = writer.stream_position()?;
            let chunk_ids = &active_ids[chunk_start..chunk_end];
            let min_id = chunk_ids.iter().copied().min().unwrap_or(0);
            let max_id = chunk_ids.iter().copied().max().unwrap_or(0);
            
            // Serialize RG body to buffer (IDs + deletion vector + columns)
            let is_single_rg = chunk_start == 0 && chunk_end == active_count;
            let null_bitmap_len = (chunk_rows + 7) / 8;
            let mut body_buf: Vec<u8> = Vec::with_capacity(chunk_rows * 8 + chunk_rows * col_count);
            {
                let mut body_writer = std::io::Cursor::new(&mut body_buf);
                
                // IDs — bulk write via unsafe slice cast
                let id_bytes = unsafe {
                    std::slice::from_raw_parts(chunk_ids.as_ptr() as *const u8, chunk_ids.len() * 8)
                };
                body_writer.write_all(id_bytes)?;
                
                // Deletion vector (all zeros — fresh save, no deletes)
                let del_vec_len = (chunk_rows + 7) / 8;
                body_writer.write_all(&vec![0u8; del_vec_len])?;
                
                // Columns
                for col_idx in 0..col_count {
                    let chunk_col_owned;
                    let chunk_col_ref: &ColumnData = if is_single_rg {
                        &active_columns[col_idx]
                    } else {
                        chunk_col_owned = active_columns[col_idx].slice_range(chunk_start, chunk_end);
                        &chunk_col_owned
                    };
                    
                    // Dict-encode low-cardinality string columns for disk
                    let dict_encoded;
                    let processed: &ColumnData = if Self::should_dict_encode(chunk_col_ref) {
                        dict_encoded = chunk_col_ref.to_dict_encoded().unwrap_or_else(|| chunk_col_ref.clone());
                        &dict_encoded
                    } else {
                        chunk_col_ref
                    };
                    
                    // Track actual type for footer schema
                    if rg_metas.is_empty() {
                        let actual_type = match processed {
                            ColumnData::StringDict { .. } => ColumnType::StringDict,
                            _ => schema_clone.columns[col_idx].1,
                        };
                        actual_col_types.push(actual_type);
                    }
                    
                    // Null bitmap
                    if is_single_rg && active_nulls[col_idx].len() == null_bitmap_len {
                        body_writer.write_all(&active_nulls[col_idx])?;
                    } else {
                        let chunk_nulls = Self::slice_null_bitmap(
                            &active_nulls[col_idx], chunk_start, chunk_end,
                        );
                        body_writer.write_all(&chunk_nulls)?;
                    }
                    write_column_encoded(processed, schema_clone.columns[col_idx].1, &mut body_writer)?;
                }
            }
            
            // Compress body using configured compression algorithm
            let (compress_flag, disk_body) = compress_rg_body(body_buf, self.compression());
            
            // RG header (32 bytes) — byte 28 = compression flag, byte 29 = encoding version
            writer.write_all(MAGIC_ROW_GROUP)?;
            writer.write_all(&(chunk_rows as u32).to_le_bytes())?;
            writer.write_all(&(col_count as u32).to_le_bytes())?;
            writer.write_all(&min_id.to_le_bytes())?;
            writer.write_all(&max_id.to_le_bytes())?;
            writer.write_all(&[compress_flag, 1, 0, 0])?; // encoding_version=1: per-column encoding prefix
            
            // RG body (possibly compressed)
            writer.write_all(&disk_body)?;
            
            // Compute zone maps for this RG's numeric columns
            let mut rg_zmaps: Vec<RgColumnZoneMap> = Vec::new();
            for col_idx in 0..col_count {
                let chunk_col_ref: &ColumnData = if is_single_rg {
                    &active_columns[col_idx]
                } else {
                    // Already sliced above — re-slice for zone map
                    // Use active_columns directly since we only need min/max
                    &active_columns[col_idx]
                };
                match chunk_col_ref {
                    ColumnData::Int64(data) => {
                        if !data.is_empty() {
                            let slice = if is_single_rg {
                                &data[..]
                            } else {
                                &data[chunk_start..chunk_end]
                            };
                            let (mut mn, mut mx) = (i64::MAX, i64::MIN);
                            for &v in slice { mn = mn.min(v); mx = mx.max(v); }
                            rg_zmaps.push(RgColumnZoneMap {
                                col_idx: col_idx as u16, min_bits: mn, max_bits: mx,
                                has_nulls: false, is_float: false,
                            });
                        }
                    }
                    ColumnData::Float64(data) => {
                        if !data.is_empty() {
                            let slice = if is_single_rg {
                                &data[..]
                            } else {
                                &data[chunk_start..chunk_end]
                            };
                            let (mut mn, mut mx) = (f64::INFINITY, f64::NEG_INFINITY);
                            for &v in slice { if v < mn { mn = v; } if v > mx { mx = v; } }
                            rg_zmaps.push(RgColumnZoneMap {
                                col_idx: col_idx as u16,
                                min_bits: mn.to_bits() as i64,
                                max_bits: mx.to_bits() as i64,
                                has_nulls: false, is_float: true,
                            });
                        }
                    }
                    _ => {}
                }
            }
            all_zone_maps.push(rg_zmaps);

            let rg_end = writer.stream_position()?;
            rg_metas.push(RowGroupMeta {
                offset: rg_offset, data_size: rg_end - rg_offset,
                row_count: chunk_rows as u32, min_id, max_id, deletion_count: 0,
            });
            
            chunk_start = chunk_end;
        }
        
        // Build modified schema with actual types (StringDict if dict-encoded)
        // IMPORTANT: preserve constraints from the original schema
        let modified_schema = if !actual_col_types.is_empty() {
            let mut ms = OnDemandSchema::new();
            for (col_idx, (col_name, _)) in schema_clone.columns.iter().enumerate() {
                ms.add_column(col_name, actual_col_types[col_idx]);
            }
            // Copy constraints from original schema
            ms.constraints = schema_clone.constraints.clone();
            ms
        } else {
            schema_clone.clone()
        };
        
        // Write V4 footer
        let footer_offset = writer.stream_position()?;
        let footer = V4Footer {
            schema: modified_schema,
            row_groups: rg_metas.clone(),
            zone_maps: all_zone_maps,
        };
        writer.write_all(&footer.to_bytes())?;
        writer.flush()?;
        
        if self.durability == super::DurabilityLevel::Max {
            writer.get_ref().sync_all()?;
        }
        
        // Seek back to fix header
        {
            let mut header = self.header.write();
            header.version = FORMAT_VERSION_V4;
            header.row_count = active_count as u64;
            header.column_count = col_count as u32;
            header.footer_offset = footer_offset;
            header.row_group_count = rg_metas.len() as u32;
            header.schema_offset = 0;
            header.column_index_offset = 0;
            header.id_column_offset = 0;
        }
        self.cached_footer_offset.store(footer_offset, Ordering::Release);
        let header = self.header.read();
        let writer_inner = writer.get_mut();
        writer_inner.seek(SeekFrom::Start(0))?;
        writer_inner.write_all(&header.to_bytes())?;
        writer_inner.flush()?;
        
        // Ensure all data is on disk before the atomic rename
        if self.durability != super::DurabilityLevel::Fast {
            writer_inner.sync_all()?;
        }
        
        // Phase 3: Atomic rename .tmp → .apex
        // POSIX rename is atomic; on crash the original file remains intact.
        drop(header);
        drop(writer);
        std::fs::rename(&tmp_path, &self.path)?;
        
        // OPTIMIZATION: compute max_id BEFORE moving active_ids (avoids re-reading after write)
        let max_active_id = active_ids.iter().max().copied().unwrap_or(0);
        
        *self.column_index.write() = Vec::new();
        *self.ids.write() = active_ids;
        *self.columns.write() = active_columns;
        *self.nulls.write() = active_nulls;
        let del_len = (active_count + 7) / 8;
        *self.deleted.write() = vec![0u8; del_len];
        *self.id_to_idx.write() = None;
        self.mmap_cache.write().invalidate();
        
        self.active_count.store(active_count as u64, Ordering::SeqCst);
        // save_v4 physically removes deleted rows; persisted = active
        self.persisted_row_count.store(active_count as u64, Ordering::SeqCst);
        // Mark base as loaded — all data is now in memory after full rewrite
        self.v4_base_loaded.store(true, Ordering::SeqCst);
        let candidate = max_active_id + 1;
        let current = self.next_id.load(Ordering::SeqCst);
        if candidate > current {
            self.next_id.store(candidate, Ordering::SeqCst);
        }
        
        let file = File::open(&self.path)?;
        *self.file.write() = Some(file);
        
        // On Linux, eagerly create the mmap so the next read avoids lazy-creation overhead.
        // This is safe because the file was just written and is in a consistent state.
        #[cfg(target_os = "linux")]
        {
            let file_guard = self.file.read();
            if let Some(f) = file_guard.as_ref() {
                let _ = self.mmap_cache.write().get_or_create(f);
            }
        }
        
        Ok(())
    }
    
    /// Open a V4 file: read footer, then load all RG data into flat columns.
    /// Used by write operations (drop_column, etc.) that need full data in memory,
    /// and by tests. Production reads use mmap on-demand reading instead.
    pub fn open_v4_data(&self) -> io::Result<()> {
        let header = self.header.read();
        if header.footer_offset == 0 {
            return Err(err_data("V4 file has no footer"));
        }
        let footer_offset = header.footer_offset;
        drop(header);
        
        // Read footer from file
        let file_guard = self.file.read();
        let file = file_guard.as_ref()
            .ok_or_else(|| err_not_conn("File not open for V4 read"))?;
        let mut mmap = self.mmap_cache.write();
        
        // Read footer
        let file_len = std::fs::metadata(&self.path)?.len();
        let footer_byte_count = (file_len - footer_offset) as usize;
        let mut footer_bytes = vec![0u8; footer_byte_count];
        mmap.read_at(file, &mut footer_bytes, footer_offset)?;
        let footer = V4Footer::from_bytes(&footer_bytes)?;
        
        // Update schema from footer
        *self.schema.write() = footer.schema.clone();
        let col_count = footer.schema.column_count();
        
        // Compute total rows from RG metadata (header.row_count stores active count,
        // but RGs may contain deleted rows that are still physically present)
        let total_rows: usize = footer.row_groups.iter().map(|rg| rg.row_count as usize).sum();
        
        // Allocate flat columns
        let mut all_ids: Vec<u64> = Vec::with_capacity(total_rows);
        let mut all_columns: Vec<ColumnData> = (0..col_count)
            .map(|i| ColumnData::new(footer.schema.columns[i].1))
            .collect();
        let mut all_nulls: Vec<Vec<u8>> = vec![Vec::new(); col_count];
        let mut all_deleted: Vec<u8> = Vec::new(); // flat deletion bitmap
        
        // Read each Row Group as a byte buffer, parse sequentially
        let mut max_id_seen: u64 = 0;
        let mut total_deleted: u64 = 0;
        for rg_meta in &footer.row_groups {
            if rg_meta.row_count == 0 {
                continue;
            }
            let rg_rows = rg_meta.row_count as usize;
            let rg_size = rg_meta.data_size as usize;
            
            // Read entire RG into buffer
            let mut rg_buf = vec![0u8; rg_size];
            mmap.read_at(file, &mut rg_buf, rg_meta.offset)?;
            
            // Check compression flag at RG header byte 28, encoding version at byte 29
            let compress_flag = if rg_buf.len() >= 32 { rg_buf[28] } else { RG_COMPRESS_NONE };
            let encoding_version = if rg_buf.len() >= 32 { rg_buf[29] } else { 0 };
            
            // Get the body bytes (after 32-byte RG header), decompressing if needed
            let decompressed_buf = decompress_rg_body(compress_flag, &rg_buf[32..])?;
            let body: &[u8] = decompressed_buf.as_deref().unwrap_or(&rg_buf[32..]);
            let mut pos: usize = 0;
            
            // Parse IDs — OPTIMIZATION: bulk memcpy instead of per-element loop
            let ids_before = all_ids.len();
            let id_byte_len = rg_rows * 8;
            all_ids.resize(ids_before + rg_rows, 0);
            unsafe {
                std::ptr::copy_nonoverlapping(
                    body[pos..].as_ptr(),
                    all_ids[ids_before..].as_mut_ptr() as *mut u8,
                    id_byte_len,
                );
            }
            if rg_meta.max_id > max_id_seen {
                max_id_seen = rg_meta.max_id;
            }
            pos += id_byte_len;
            
            // Read deletion vector and merge into flat bitmap
            let del_vec_len = (rg_rows + 7) / 8;
            let del_bytes = &body[pos..pos + del_vec_len];
            let needed_len = (ids_before + rg_rows + 7) / 8;
            if all_deleted.len() < needed_len {
                all_deleted.resize(needed_len, 0);
            }
            if ids_before % 8 == 0 {
                let dest_byte = ids_before / 8;
                let copy_len = del_vec_len.min(all_deleted.len() - dest_byte);
                all_deleted[dest_byte..dest_byte + copy_len]
                    .copy_from_slice(&del_bytes[..copy_len]);
            } else {
                for i in 0..rg_rows {
                    if (del_bytes[i / 8] >> (i % 8)) & 1 == 1 {
                        let flat_idx = ids_before + i;
                        all_deleted[flat_idx / 8] |= 1 << (flat_idx % 8);
                    }
                }
            }
            total_deleted += rg_meta.deletion_count as u64;
            pos += del_vec_len;
            
            // Parse columns
            let null_bitmap_len = (rg_rows + 7) / 8;
            for col_idx in 0..col_count {
                // Read null bitmap
                let null_bytes = &body[pos..pos + null_bitmap_len];
                
                // Merge into flat nulls
                let flat_start = ids_before;
                let needed_len = (flat_start + rg_rows + 7) / 8;
                if all_nulls[col_idx].len() < needed_len {
                    all_nulls[col_idx].resize(needed_len, 0);
                }
                // OPTIMIZATION: bulk copy when flat_start is byte-aligned
                if flat_start % 8 == 0 {
                    let dest_byte = flat_start / 8;
                    let copy_len = null_bitmap_len.min(all_nulls[col_idx].len() - dest_byte);
                    all_nulls[col_idx][dest_byte..dest_byte + copy_len]
                        .copy_from_slice(&null_bytes[..copy_len]);
                } else {
                    for i in 0..rg_rows {
                        if (null_bytes[i / 8] >> (i % 8)) & 1 == 1 {
                            let flat_idx = flat_start + i;
                            all_nulls[col_idx][flat_idx / 8] |= 1 << (flat_idx % 8);
                        }
                    }
                }
                pos += null_bitmap_len;
                
                // Parse column data (encoding-aware for version 1, plain for version 0)
                let col_type = footer.schema.columns[col_idx].1;
                let (col_data, consumed) = if encoding_version >= 1 {
                    read_column_encoded(&body[pos..], col_type)?
                } else {
                    ColumnData::from_bytes_typed(&body[pos..], col_type)?
                };
                pos += consumed;
                
                // Append to flat column
                all_columns[col_idx].append(&col_data);
            }
        }
        
        drop(mmap);
        drop(file_guard);
        
        // Decode StringDict columns back to plain String for in-memory use.
        // Dict encoding is a disk-only optimization; push_string/extend_strings
        // only work on ColumnData::String, so we must normalize here.
        {
            let mut schema_w = self.schema.write();
            for col_idx in 0..all_columns.len() {
                if matches!(&all_columns[col_idx], ColumnData::StringDict { .. }) {
                    let col = std::mem::replace(&mut all_columns[col_idx], ColumnData::new(ColumnType::String));
                    all_columns[col_idx] = col.decode_string_dict();
                    // Update schema type from StringDict → String
                    if col_idx < schema_w.columns.len() {
                        schema_w.columns[col_idx].1 = ColumnType::String;
                    }
                }
            }
        }
        
        // OPTIMIZATION: compute next_id from tracked max before moving all_ids
        let next_id = if max_id_seen > 0 {
            max_id_seen + 1
        } else {
            all_ids.iter().max().map(|&id| id + 1).unwrap_or(0)
        };
        
        // Store flat data
        *self.ids.write() = all_ids;
        *self.columns.write() = all_columns;
        *self.nulls.write() = all_nulls;
        
        // Use deletion vectors read from disk (not all-zeros)
        let deleted_len = (total_rows + 7) / 8;
        if all_deleted.len() < deleted_len {
            all_deleted.resize(deleted_len, 0);
        }
        *self.deleted.write() = all_deleted;
        
        self.next_id.store(next_id, Ordering::SeqCst);
        self.active_count.store(total_rows as u64 - total_deleted, Ordering::SeqCst);
        // Track actual on-disk row count (total rows in RGs, including deleted)
        self.persisted_row_count.store(total_rows as u64, Ordering::SeqCst);
        self.v4_base_loaded.store(true, Ordering::SeqCst);
        *self.id_to_idx.write() = None;
        
        Ok(())
    }
    
    /// Update only the V4 footer schema on disk (no data rewrite).
    /// Used for DDL operations (add/drop/rename column) when base data
    /// is not loaded into memory (mmap-only mode).
    pub fn update_v4_footer_schema(&self) -> io::Result<()> {
        let header = self.header.read();
        if header.version != FORMAT_VERSION_V4 || header.footer_offset == 0 {
            return Err(err_data("update_v4_footer_schema requires V4 format file"));
        }
        let footer_offset = header.footer_offset;
        drop(header);

        // Read existing footer from disk
        let file_len = std::fs::metadata(&self.path)?.len();
        let footer_bytes = {
            let file_guard = self.file.read();
            let file = file_guard.as_ref()
                .ok_or_else(|| err_not_conn("File not open"))?;
            let mut mmap = self.mmap_cache.write();
            let size = (file_len - footer_offset) as usize;
            let mut buf = vec![0u8; size];
            mmap.read_at(file, &mut buf, footer_offset)?;
            buf
        };
        let mut footer = V4Footer::from_bytes(&footer_bytes)?;

        // Update footer schema from current in-memory schema
        let schema = self.schema.read();
        footer.schema = schema.clone();
        drop(schema);

        // Release mmap before writing
        self.mmap_cache.write().invalidate();
        *self.file.write() = None;
        crate::query::ApexExecutor::invalidate_cache_for_path(&self.path);

        // Write updated footer at same offset (overwrite old footer)
        let mut file = OpenOptions::new().write(true).open(&self.path)?;
        let new_footer_bytes = footer.to_bytes();
        file.seek(SeekFrom::Start(footer_offset))?;
        file.write_all(&new_footer_bytes)?;
        // Write footer size + magic trailer
        file.write_all(&(new_footer_bytes.len() as u64).to_le_bytes())?;
        file.write_all(b"APXFOOT\0")?;
        file.flush()?;

        // Truncate file to remove any trailing data from old (possibly larger) footer
        let new_file_len = footer_offset + new_footer_bytes.len() as u64 + 16;
        file.set_len(new_file_len)?;

        // Update header column count (both in-memory and on-disk)
        {
            let mut header = self.header.write();
            header.column_count = footer.schema.column_count() as u32;
            // Write updated header to disk
            let mut hfile = OpenOptions::new().write(true).open(&self.path)?;
            hfile.write_all(&header.to_bytes())?;
            hfile.flush()?;
        }

        // Reopen file handle
        drop(file);
        let file = File::open(&self.path)?;
        *self.file.write() = Some(file);

        Ok(())
    }

    /// Update only the deletion vectors in existing Row Groups on disk.
    /// O(num_RGs) random writes instead of O(all_data) full rewrite.
    /// Also updates the footer's per-RG deletion_count and the header's row_count.
    fn save_deletion_vectors(&self) -> io::Result<()> {
        let header = self.header.read();
        if header.version != FORMAT_VERSION_V4 || header.footer_offset == 0 {
            return Err(err_data("save_deletion_vectors requires V4 format file"));
        }
        let footer_offset = header.footer_offset;
        drop(header);

        // Read existing footer
        let file_len = std::fs::metadata(&self.path)?.len();
        let footer_bytes = {
            let file_guard = self.file.read();
            let file = file_guard.as_ref()
                .ok_or_else(|| err_not_conn("File not open"))?;
            let mut mmap = self.mmap_cache.write();
            let size = (file_len - footer_offset) as usize;
            let mut buf = vec![0u8; size];
            mmap.read_at(file, &mut buf, footer_offset)?;
            buf
        };
        let mut footer = V4Footer::from_bytes(&footer_bytes)?;

        // Check if any RG is compressed — if so, we cannot do in-place deletion
        // vector updates (the deletion vector is inside the compressed body).
        // Fall back to full rewrite via save_v4().
        {
            let file_guard = self.file.read();
            let file = file_guard.as_ref()
                .ok_or_else(|| err_not_conn("File not open"))?;
            let mut mmap = self.mmap_cache.write();
            let mmap_ref = mmap.get_or_create(file)?;
            for rg_meta in &footer.row_groups {
                if rg_meta.row_count == 0 { continue; }
                let rg_end = (rg_meta.offset + rg_meta.data_size) as usize;
                if rg_end <= mmap_ref.len() {
                    let rg_bytes = &mmap_ref[rg_meta.offset as usize..rg_end];
                    if rg_bytes.len() >= 32 && rg_bytes[28] != RG_COMPRESS_NONE {
                        // Compressed RG detected — must do full rewrite
                        drop(mmap);
                        drop(file_guard);
                        return self.save_v4();
                    }
                }
            }
        }

        // Release mmap before writing
        self.mmap_cache.write().invalidate();
        *self.file.write() = None;
        crate::query::ApexExecutor::invalidate_cache_for_path(&self.path);

        let deleted = self.deleted.read();
        let mut file = OpenOptions::new().write(true).open(&self.path)?;

        // For each RG, write the updated deletion vector at its known offset
        let mut flat_row_start: usize = 0;
        let mut total_active: u64 = 0;
        for rg_meta in footer.row_groups.iter_mut() {
            let rg_rows = rg_meta.row_count as usize;
            if rg_rows == 0 {
                continue;
            }

            // Deletion vector starts after RG header (32 bytes) + IDs (rg_rows * 8)
            let del_vec_offset = rg_meta.offset + 32 + (rg_rows as u64 * 8);
            let del_vec_len = (rg_rows + 7) / 8;

            // Extract this RG's slice from the flat deleted bitmap
            let rg_del_vec = Self::slice_null_bitmap(
                &deleted, flat_row_start, flat_row_start + rg_rows,
            );

            // Count deleted rows in this RG
            let mut del_count: u32 = 0;
            for i in 0..rg_rows {
                if (rg_del_vec[i / 8] >> (i % 8)) & 1 == 1 {
                    del_count += 1;
                }
            }
            rg_meta.deletion_count = del_count;
            total_active += (rg_rows as u32 - del_count) as u64;

            // Write deletion vector to disk
            file.seek(SeekFrom::Start(del_vec_offset))?;
            file.write_all(&rg_del_vec[..del_vec_len])?;

            flat_row_start += rg_rows;
        }
        drop(deleted);

        // Rewrite footer with updated deletion_counts
        file.seek(SeekFrom::Start(footer_offset))?;
        let new_footer_bytes = footer.to_bytes();
        file.write_all(&new_footer_bytes)?;
        // Truncate in case new footer is shorter (shouldn't happen, but safety)
        let new_end = footer_offset + new_footer_bytes.len() as u64;
        file.set_len(new_end)?;
        file.flush()?;

        // Update header: row_count = active rows (matches save_v4 convention)
        {
            let mut header = self.header.write();
            header.row_count = total_active;
            file.seek(SeekFrom::Start(0))?;
            file.write_all(&header.to_bytes())?;
        }
        file.flush()?;
        drop(file);

        // Reopen file handle
        let new_file = File::open(&self.path)?;
        *self.file.write() = Some(new_file);

        Ok(())
    }

    /// Write a new Row Group to disk without modifying in-memory state.
    /// Called by save() when rows are already in memory and only need persisting.
    /// Also called by append_row_group() which additionally updates memory.
    fn write_row_group_to_disk(
        &self,
        new_ids: &[u64],
        new_columns: &[ColumnData],
        new_nulls: &[Vec<u8>],
    ) -> io::Result<()> {
        let header = self.header.read();
        if header.version != FORMAT_VERSION_V4 || header.footer_offset == 0 {
            return Err(err_data("write_row_group_to_disk requires V4 format file"));
        }
        let footer_offset = header.footer_offset;
        drop(header);
        
        // Read existing footer
        let file_len = std::fs::metadata(&self.path)?.len();
        let footer_bytes = {
            let file_guard = self.file.read();
            let file = file_guard.as_ref()
                .ok_or_else(|| err_not_conn("File not open"))?;
            let mut mmap = self.mmap_cache.write();
            let size = (file_len - footer_offset) as usize;
            let mut buf = vec![0u8; size];
            mmap.read_at(file, &mut buf, footer_offset)?;
            buf
        };
        let mut footer = V4Footer::from_bytes(&footer_bytes)?;

        // Schema evolution: merge any new columns from in-memory schema into footer
        {
            let mem_schema = self.schema.read();
            for (name, ct) in &mem_schema.columns {
                if footer.schema.get_index(name).is_none() {
                    footer.schema.add_column(name, *ct);
                }
            }
        }
        let col_count = footer.schema.column_count();
        
        // Release mmap before writing
        self.mmap_cache.write().invalidate();
        *self.file.write() = None;
        crate::query::ApexExecutor::invalidate_cache_for_path(&self.path);
        
        // Open file for append — seek to old footer position (overwrite it)
        let mut file = OpenOptions::new().write(true).open(&self.path)?;
        file.seek(SeekFrom::Start(footer_offset))?;
        let mut writer = BufWriter::with_capacity(64 * 1024, file);
        
        let rg_rows = new_ids.len();
        let rg_offset = footer_offset;
        let min_id = new_ids.iter().copied().min().unwrap_or(0);
        let max_id = new_ids.iter().copied().max().unwrap_or(0);
        
        // Serialize RG body to buffer (IDs + deletion vector + columns)
        let null_bitmap_len = (rg_rows + 7) / 8;
        let mut body_buf: Vec<u8> = Vec::with_capacity(rg_rows * 8 + rg_rows * col_count);
        {
            let mut body_writer = std::io::Cursor::new(&mut body_buf);
            
            // IDs
            for &id in new_ids {
                body_writer.write_all(&id.to_le_bytes())?;
            }
            
            // Deletion vector (all zeros)
            let del_vec_len = (rg_rows + 7) / 8;
            body_writer.write_all(&vec![0u8; del_vec_len])?;
            
            // Columns
            for col_idx in 0..col_count {
                // Null bitmap
                let col_nulls = new_nulls.get(col_idx).map(|v| v.as_slice()).unwrap_or(&[]);
                let padded = if col_nulls.len() < null_bitmap_len {
                    let mut v = vec![0u8; null_bitmap_len];
                    let copy = col_nulls.len().min(null_bitmap_len);
                    v[..copy].copy_from_slice(&col_nulls[..copy]);
                    v
                } else {
                    col_nulls[..null_bitmap_len].to_vec()
                };
                body_writer.write_all(&padded)?;
                
                // Column data — dict-encode if footer schema expects StringDict
                if col_idx < new_columns.len() {
                    let col = &new_columns[col_idx];
                    let col_type = if col_idx < footer.schema.columns.len() {
                        footer.schema.columns[col_idx].1
                    } else {
                        ColumnType::Int64
                    };
                    if col_type == ColumnType::StringDict
                        && matches!(col, ColumnData::String { .. })
                    {
                        if let Some(dict) = col.to_dict_encoded() {
                            write_column_encoded(&dict, col_type, &mut body_writer)?;
                        } else {
                            write_column_encoded(col, col_type, &mut body_writer)?;
                        }
                    } else {
                        write_column_encoded(col, col_type, &mut body_writer)?;
                    }
                }
            }
        }
        
        // Compress body using configured compression algorithm
        let (compress_flag, disk_body) = compress_rg_body(body_buf, self.compression());
        
        // Write RG header (32 bytes) — byte 28 = compression flag
        writer.write_all(MAGIC_ROW_GROUP)?;
        writer.write_all(&(rg_rows as u32).to_le_bytes())?;
        writer.write_all(&(col_count as u32).to_le_bytes())?;
        writer.write_all(&min_id.to_le_bytes())?;
        writer.write_all(&max_id.to_le_bytes())?;
        writer.write_all(&[compress_flag, 1, 0, 0])?; // encoding_version=1: per-column encoding prefix
        
        // RG body (possibly compressed)
        writer.write_all(&disk_body)?;
        
        let rg_end = writer.stream_position()?;
        
        // Update footer with new RG
        footer.row_groups.push(RowGroupMeta {
            offset: rg_offset,
            data_size: rg_end - rg_offset,
            row_count: rg_rows as u32,
            min_id,
            max_id,
            deletion_count: 0,
        });
        
        // Write updated footer + trailer (footer_size + magic)
        let new_footer_offset = rg_end;
        let footer_bytes = footer.to_bytes();
        writer.write_all(&footer_bytes)?;
        writer.write_all(&(footer_bytes.len() as u64).to_le_bytes())?;
        writer.write_all(MAGIC_V4_FOOTER)?;
        writer.flush()?;
        
        // Fix header
        let new_persisted = self.persisted_row_count.load(Ordering::SeqCst) + rg_rows as u64;
        let writer_inner = writer.get_mut();
        {
            let mut header = self.header.write();
            header.row_count = new_persisted;
            header.footer_offset = new_footer_offset;
            header.row_group_count = footer.row_groups.len() as u32;
        }
        self.cached_footer_offset.store(new_footer_offset, Ordering::Release);
        let header = self.header.read();
        writer_inner.seek(SeekFrom::Start(0))?;
        writer_inner.write_all(&header.to_bytes())?;
        writer_inner.flush()?;
        
        drop(header);
        drop(writer);
        
        // Reopen file
        let new_file = File::open(&self.path)?;
        *self.file.write() = Some(new_file);
        
        // Update persisted count (disk now has more rows)
        self.persisted_row_count.store(new_persisted, Ordering::SeqCst);
        
        Ok(())
    }

    /// Append a new Row Group to an existing V4 file without rewriting.
    /// Overwrites old footer, writes new RG + updated footer, fixes header.
    /// Also updates in-memory state (IDs, active_count).
    /// Use this when adding NEW data that is NOT already in memory.
    pub fn append_row_group(
        &self,
        new_ids: &[u64],
        new_columns: &[ColumnData],
        new_nulls: &[Vec<u8>],
    ) -> io::Result<()> {
        let rg_rows = new_ids.len();
        self.write_row_group_to_disk(new_ids, new_columns, new_nulls)?;
        
        // Update in-memory state (caller hasn't added these rows yet)
        {
            let mut ids = self.ids.write();
            ids.extend_from_slice(new_ids);
        }
        let next_id = new_ids.iter().max().map(|&id| id + 1).unwrap_or(0);
        let current_next = self.next_id.load(Ordering::SeqCst);
        if next_id > current_next {
            self.next_id.store(next_id, Ordering::SeqCst);
        }
        self.active_count.fetch_add(rg_rows as u64, Ordering::SeqCst);
        *self.id_to_idx.write() = None;
        
        Ok(())
    }

    /// Explicitly sync data to disk (fsync)
    /// 
    /// This ensures all buffered data is written to persistent storage.
    /// For safe/max durability modes, also syncs the WAL file.
    /// Called automatically for Safe/Max durability levels on save().
    /// For Fast durability, call this manually when you need durability guarantees.
    pub fn sync(&self) -> io::Result<()> {
        // Sync WAL first (for safe/max modes)
        if self.durability != super::DurabilityLevel::Fast {
            let mut wal_writer = self.wal_writer.write();
            if let Some(writer) = wal_writer.as_mut() {
                writer.sync()?;
            }
        }
        
        // Sync main data file
        // On Windows, sync_all() requires write access. Since save() already flushes
        // data via BufWriter and does fsync for Max durability, we need to open
        // the file with write access specifically for syncing.
        if self.path.exists() {
            // Open with write access for fsync (append mode to avoid truncation)
            let file = OpenOptions::new()
                .write(true)
                .append(true)
                .open(&self.path)?;
            file.sync_all()?;
        }
        Ok(())
    }
    
}
