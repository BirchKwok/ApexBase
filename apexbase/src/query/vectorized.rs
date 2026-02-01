//! Vectorized Execution Engine for GROUP BY
//!
//! This module implements DuckDB-style vectorized execution:
//! - Process data in small batches (vectors) of 2048 rows
//! - Stream data through pipeline instead of loading all at once
//! - Use efficient hash aggregation with pre-computed hashes
//!
//! Key optimizations:
//! 1. Cache-friendly batch processing
//! 2. Pre-computed hash values for grouping
//! 3. SIMD-friendly aggregation loops
//! 4. Minimal memory allocations

use arrow::array::{Array, ArrayRef, Int64Array, Float64Array, StringArray, BooleanArray, UInt64Array};
use arrow::datatypes::{DataType as ArrowDataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use ahash::{AHashMap, AHasher};
use std::hash::{Hash, Hasher};
use std::io;
use std::sync::Arc;

/// Vector size for batch processing (DuckDB uses 2048)
pub const VECTOR_SIZE: usize = 2048;

/// Pre-computed hash for a group key
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct GroupHash(u64);

impl GroupHash {
    #[inline(always)]
    pub fn from_i64(val: i64) -> Self {
        // OPTIMIZATION: Use direct bit pattern as hash for integers
        // This avoids AHasher creation overhead for integer keys
        GroupHash(val as u64)
    }
    
    #[inline(always)]
    pub fn from_u32(val: u32) -> Self {
        // For dictionary indices, use direct value as hash (perfect hash)
        GroupHash(val as u64)
    }
    
    #[inline(always)]
    pub fn from_str(val: &str) -> Self {
        let mut hasher = AHasher::default();
        val.hash(&mut hasher);
        GroupHash(hasher.finish())
    }
    
    #[inline(always)]
    pub fn combine(self, other: GroupHash) -> Self {
        // Combine two hashes using XOR and rotation
        GroupHash(self.0.rotate_left(5) ^ other.0)
    }
}

/// Aggregate state for a single group
#[derive(Clone)]
pub struct AggregateState {
    pub count: i64,
    pub sum_int: i64,
    pub sum_float: f64,
    pub min_int: Option<i64>,
    pub max_int: Option<i64>,
    pub min_float: Option<f64>,
    pub max_float: Option<f64>,
    pub first_row_idx: usize,
}

impl AggregateState {
    #[inline(always)]
    pub fn new(first_row_idx: usize) -> Self {
        Self {
            count: 0,
            sum_int: 0,
            sum_float: 0.0,
            min_int: None,
            max_int: None,
            min_float: None,
            max_float: None,
            first_row_idx,
        }
    }
    
    #[inline(always)]
    pub fn update_int(&mut self, val: i64) {
        self.count += 1;
        self.sum_int = self.sum_int.wrapping_add(val);
        self.min_int = Some(self.min_int.map_or(val, |m| m.min(val)));
        self.max_int = Some(self.max_int.map_or(val, |m| m.max(val)));
    }
    
    #[inline(always)]
    pub fn update_float(&mut self, val: f64) {
        self.count += 1;
        self.sum_float += val;
        self.min_float = Some(self.min_float.map_or(val, |m| m.min(val)));
        self.max_float = Some(self.max_float.map_or(val, |m| m.max(val)));
    }
    
    #[inline(always)]
    pub fn update_count(&mut self) {
        self.count += 1;
    }
    
    /// Merge another state into this one
    #[inline(always)]
    pub fn merge(&mut self, other: &AggregateState) {
        self.count += other.count;
        self.sum_int = self.sum_int.wrapping_add(other.sum_int);
        self.sum_float += other.sum_float;
        if let Some(other_min) = other.min_int {
            self.min_int = Some(self.min_int.map_or(other_min, |m| m.min(other_min)));
        }
        if let Some(other_max) = other.max_int {
            self.max_int = Some(self.max_int.map_or(other_max, |m| m.max(other_max)));
        }
        if let Some(other_min) = other.min_float {
            self.min_float = Some(self.min_float.map_or(other_min, |m| m.min(other_min)));
        }
        if let Some(other_max) = other.max_float {
            self.max_float = Some(self.max_float.map_or(other_max, |m| m.max(other_max)));
        }
    }
}

/// Vectorized hash aggregation table
/// Uses a two-level structure for efficiency:
/// 1. Hash table mapping GroupHash -> group_id
/// 2. Flat vector of AggregateState indexed by group_id
pub struct VectorizedHashAgg {
    /// Hash -> group_id mapping
    hash_table: AHashMap<GroupHash, u32>,
    /// Aggregate states indexed by group_id
    states: Vec<AggregateState>,
    /// Group keys (for result building)
    group_keys_int: Vec<i64>,
    group_keys_str: Vec<String>,
}

impl VectorizedHashAgg {
    pub fn new(is_int_key: bool, estimated_groups: usize) -> Self {
        // OPTIMIZATION: Pre-allocate with 2x estimated capacity to reduce rehashing
        let capacity = (estimated_groups * 2).max(64);
        Self {
            hash_table: AHashMap::with_capacity_and_hasher(capacity, Default::default()),
            states: Vec::with_capacity(estimated_groups),
            group_keys_int: if is_int_key { Vec::with_capacity(estimated_groups) } else { Vec::new() },
            group_keys_str: if !is_int_key { Vec::with_capacity(estimated_groups) } else { Vec::new() },
        }
    }
    
    /// Get or create a group, returns the group_id
    #[inline(always)]
    pub fn get_or_create_group_int(&mut self, key: i64, row_idx: usize) -> u32 {
        let hash = GroupHash::from_i64(key);
        if let Some(&group_id) = self.hash_table.get(&hash) {
            group_id
        } else {
            let group_id = self.states.len() as u32;
            self.hash_table.insert(hash, group_id);
            self.states.push(AggregateState::new(row_idx));
            self.group_keys_int.push(key);
            group_id
        }
    }
    
    /// Get or create a group for string key
    #[inline(always)]
    pub fn get_or_create_group_str(&mut self, key: &str, row_idx: usize) -> u32 {
        let hash = GroupHash::from_str(key);
        if let Some(&group_id) = self.hash_table.get(&hash) {
            group_id
        } else {
            let group_id = self.states.len() as u32;
            self.hash_table.insert(hash, group_id);
            self.states.push(AggregateState::new(row_idx));
            self.group_keys_str.push(key.to_string());
            group_id
        }
    }
    
    /// Get or create a group for dictionary index (perfect hash)
    #[inline(always)]
    pub fn get_or_create_group_dict(&mut self, dict_idx: u32, key_str: &str, row_idx: usize) -> u32 {
        let hash = GroupHash::from_u32(dict_idx);
        if let Some(&group_id) = self.hash_table.get(&hash) {
            group_id
        } else {
            let group_id = self.states.len() as u32;
            self.hash_table.insert(hash, group_id);
            self.states.push(AggregateState::new(row_idx));
            self.group_keys_str.push(key_str.to_string());
            group_id
        }
    }
    
    /// Update aggregate state for a group
    #[inline(always)]
    pub fn update_int(&mut self, group_id: u32, val: i64) {
        unsafe {
            self.states.get_unchecked_mut(group_id as usize).update_int(val);
        }
    }
    
    #[inline(always)]
    pub fn update_float(&mut self, group_id: u32, val: f64) {
        unsafe {
            self.states.get_unchecked_mut(group_id as usize).update_float(val);
        }
    }
    
    #[inline(always)]
    pub fn update_count(&mut self, group_id: u32) {
        unsafe {
            self.states.get_unchecked_mut(group_id as usize).update_count();
        }
    }
    
    /// Get number of groups
    pub fn num_groups(&self) -> usize {
        self.states.len()
    }
    
    /// Get aggregate states
    pub fn states(&self) -> &[AggregateState] {
        &self.states
    }
    
    /// Get group keys (int)
    pub fn group_keys_int(&self) -> &[i64] {
        &self.group_keys_int
    }
    
    /// Get group keys (string)
    pub fn group_keys_str(&self) -> &[String] {
        &self.group_keys_str
    }
}

/// Process a vector (batch) of rows for GROUP BY aggregation
/// This is the core vectorized execution function
pub fn process_vector_group_by(
    hash_agg: &mut VectorizedHashAgg,
    // Group column data
    group_col_int: Option<&[i64]>,
    group_col_str: Option<&StringArray>,
    group_col_dict_indices: Option<&[u32]>,
    group_col_dict_values: Option<&[&str]>,
    // Aggregate column data
    agg_col_int: Option<&[i64]>,
    agg_col_float: Option<&[f64]>,
    // Row range in the vector
    start_row: usize,
    end_row: usize,
    // Whether to only count (COUNT(*))
    count_only: bool,
) {
    // FAST PATH 1: Dictionary-encoded string column (perfect hash)
    if let (Some(dict_indices), Some(dict_values)) = (group_col_dict_indices, group_col_dict_values) {
        if count_only {
            for row_idx in start_row..end_row {
                let dict_idx = unsafe { *dict_indices.get_unchecked(row_idx) };
                if dict_idx == 0 { continue; } // NULL
                let key_str = unsafe { *dict_values.get_unchecked((dict_idx - 1) as usize) };
                let group_id = hash_agg.get_or_create_group_dict(dict_idx, key_str, row_idx);
                hash_agg.update_count(group_id);
            }
        } else if let Some(vals) = agg_col_int {
            for row_idx in start_row..end_row {
                let dict_idx = unsafe { *dict_indices.get_unchecked(row_idx) };
                if dict_idx == 0 { continue; }
                let key_str = unsafe { *dict_values.get_unchecked((dict_idx - 1) as usize) };
                let group_id = hash_agg.get_or_create_group_dict(dict_idx, key_str, row_idx);
                let val = unsafe { *vals.get_unchecked(row_idx) };
                hash_agg.update_int(group_id, val);
            }
        } else if let Some(vals) = agg_col_float {
            for row_idx in start_row..end_row {
                let dict_idx = unsafe { *dict_indices.get_unchecked(row_idx) };
                if dict_idx == 0 { continue; }
                let key_str = unsafe { *dict_values.get_unchecked((dict_idx - 1) as usize) };
                let group_id = hash_agg.get_or_create_group_dict(dict_idx, key_str, row_idx);
                let val = unsafe { *vals.get_unchecked(row_idx) };
                hash_agg.update_float(group_id, val);
            }
        }
        return;
    }
    
    // FAST PATH 2: Integer group column
    if let Some(group_vals) = group_col_int {
        if count_only {
            for row_idx in start_row..end_row {
                let key = unsafe { *group_vals.get_unchecked(row_idx) };
                let group_id = hash_agg.get_or_create_group_int(key, row_idx);
                hash_agg.update_count(group_id);
            }
        } else if let Some(vals) = agg_col_int {
            for row_idx in start_row..end_row {
                let key = unsafe { *group_vals.get_unchecked(row_idx) };
                let group_id = hash_agg.get_or_create_group_int(key, row_idx);
                let val = unsafe { *vals.get_unchecked(row_idx) };
                hash_agg.update_int(group_id, val);
            }
        } else if let Some(vals) = agg_col_float {
            for row_idx in start_row..end_row {
                let key = unsafe { *group_vals.get_unchecked(row_idx) };
                let group_id = hash_agg.get_or_create_group_int(key, row_idx);
                let val = unsafe { *vals.get_unchecked(row_idx) };
                hash_agg.update_float(group_id, val);
            }
        }
        return;
    }
    
    // FAST PATH 3: Regular string column
    if let Some(str_arr) = group_col_str {
        if count_only {
            for row_idx in start_row..end_row {
                if str_arr.is_null(row_idx) { continue; }
                let key = str_arr.value(row_idx);
                let group_id = hash_agg.get_or_create_group_str(key, row_idx);
                hash_agg.update_count(group_id);
            }
        } else if let Some(vals) = agg_col_int {
            for row_idx in start_row..end_row {
                if str_arr.is_null(row_idx) { continue; }
                let key = str_arr.value(row_idx);
                let group_id = hash_agg.get_or_create_group_str(key, row_idx);
                let val = unsafe { *vals.get_unchecked(row_idx) };
                hash_agg.update_int(group_id, val);
            }
        } else if let Some(vals) = agg_col_float {
            for row_idx in start_row..end_row {
                if str_arr.is_null(row_idx) { continue; }
                let key = str_arr.value(row_idx);
                let group_id = hash_agg.get_or_create_group_str(key, row_idx);
                let val = unsafe { *vals.get_unchecked(row_idx) };
                hash_agg.update_float(group_id, val);
            }
        }
    }
}

/// Execute vectorized GROUP BY on a RecordBatch
/// Processes data in VECTOR_SIZE batches for cache efficiency
pub fn execute_vectorized_group_by(
    batch: &RecordBatch,
    group_col_name: &str,
    agg_col_name: Option<&str>,
    _has_int_agg: bool,
) -> io::Result<VectorizedHashAgg> {
    let num_rows = batch.num_rows();
    let estimated_groups = (num_rows / 100).max(16).min(10000);
    
    // Get group column
    let group_col = batch.column_by_name(group_col_name)
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "Group column not found"))?;
    
    // Get aggregate column if specified
    let agg_col = agg_col_name.and_then(|name| batch.column_by_name(name));
    
    // Extract aggregate column data first (needed for all paths)
    let agg_col_int: Option<&[i64]> = agg_col.and_then(|c| {
        c.as_any().downcast_ref::<Int64Array>().map(|a| a.values().as_ref())
    });
    let agg_col_float: Option<&[f64]> = agg_col.and_then(|c| {
        c.as_any().downcast_ref::<Float64Array>().map(|a| a.values().as_ref())
    });
    let count_only = agg_col_int.is_none() && agg_col_float.is_none();
    
    // Determine group column type and extract data
    let group_col_int: Option<&[i64]>;
    let group_col_str: Option<&StringArray>;
    let mut group_col_dict_indices: Option<Vec<u32>> = None;
    let mut group_col_dict_values: Option<Vec<&str>> = None;
    let is_int_key: bool;
    
    // Try DictionaryArray first
    use arrow::array::DictionaryArray;
    use arrow::datatypes::UInt32Type;
    
    if let Some(dict_arr) = group_col.as_any().downcast_ref::<DictionaryArray<UInt32Type>>() {
        let keys = dict_arr.keys();
        let values = dict_arr.values();
        if let Some(str_values) = values.as_any().downcast_ref::<StringArray>() {
            // Extract dictionary indices
            let indices: Vec<u32> = (0..num_rows)
                .map(|i| if keys.is_null(i) { 0u32 } else { keys.value(i) + 1 })
                .collect();
            let dict_vals: Vec<&str> = (0..str_values.len())
                .map(|i| str_values.value(i))
                .collect();
            group_col_dict_indices = Some(indices);
            group_col_dict_values = Some(dict_vals);
            group_col_int = None;
            group_col_str = None;
            is_int_key = false;
        } else {
            return Err(io::Error::new(io::ErrorKind::InvalidInput, "Unsupported dictionary value type"));
        }
    } else if let Some(int_arr) = group_col.as_any().downcast_ref::<Int64Array>() {
        group_col_int = Some(int_arr.values());
        group_col_str = None;
        is_int_key = true;
    } else if let Some(str_arr) = group_col.as_any().downcast_ref::<StringArray>() {
        group_col_int = None;
        group_col_str = Some(str_arr);
        is_int_key = false;
    } else {
        return Err(io::Error::new(io::ErrorKind::InvalidInput, "Unsupported group column type"));
    }
    
    // Create hash aggregation table
    let mut hash_agg = VectorizedHashAgg::new(is_int_key, estimated_groups);
    
    // Process in vectors (batches) for cache efficiency
    let dict_indices_ref = group_col_dict_indices.as_deref();
    let dict_values_ref: Option<Vec<&str>> = group_col_dict_values;
    let dict_values_slice: Option<&[&str]> = dict_values_ref.as_deref();
    
    for batch_start in (0..num_rows).step_by(VECTOR_SIZE) {
        let batch_end = (batch_start + VECTOR_SIZE).min(num_rows);
        
        process_vector_group_by(
            &mut hash_agg,
            group_col_int,
            group_col_str,
            dict_indices_ref,
            dict_values_slice,
            agg_col_int,
            agg_col_float,
            batch_start,
            batch_end,
            count_only,
        );
    }
    
    Ok(hash_agg)
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_group_hash() {
        let h1 = GroupHash::from_i64(42);
        let h2 = GroupHash::from_i64(42);
        let h3 = GroupHash::from_i64(43);
        
        assert_eq!(h1, h2);
        assert_ne!(h1, h3);
    }
    
    #[test]
    fn test_aggregate_state() {
        let mut state = AggregateState::new(0);
        state.update_int(10);
        state.update_int(20);
        state.update_int(5);
        
        assert_eq!(state.count, 3);
        assert_eq!(state.sum_int, 35);
        assert_eq!(state.min_int, Some(5));
        assert_eq!(state.max_int, Some(20));
    }
}
