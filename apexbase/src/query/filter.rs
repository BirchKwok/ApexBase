//! Query filter implementation

use crate::data::{Row, Value};
use crate::table::column_table::{TypedColumn, ColumnSchema, BitVec};

/// Comparison operator
#[derive(Debug, Clone, PartialEq)]
pub enum CompareOp {
    Equal,
    NotEqual,
    LessThan,
    LessEqual,
    GreaterThan,
    GreaterEqual,
    Like,
    In,
}

/// A filter condition
#[derive(Debug, Clone)]
pub enum Filter {
    /// Always true
    True,
    /// Always false
    False,
    /// Compare field to value
    Compare {
        field: String,
        op: CompareOp,
        value: Value,
    },
    /// LIKE pattern match
    Like {
        field: String,
        pattern: String,
    },
    /// IN list
    In {
        field: String,
        values: Vec<Value>,
    },
    /// AND combination
    And(Vec<Filter>),
    /// OR combination
    Or(Vec<Filter>),
    /// NOT
    Not(Box<Filter>),
}

impl Filter {
    /// Check if a row matches this filter
    #[inline]
    pub fn matches(&self, row: &Row) -> bool {
        match self {
            Filter::True => true,
            Filter::False => false,
            Filter::Compare { field, op, value } => {
                if let Some(row_value) = row.get(field) {
                    Self::compare_fast(row_value, op, value)
                } else {
                    false
                }
            }
            Filter::Like { field, pattern } => {
                if let Some(Value::String(s)) = row.get(field) {
                    Self::like_match(s, pattern)
                } else {
                    false
                }
            }
            Filter::In { field, values } => {
                if let Some(row_value) = row.get(field) {
                    values.iter().any(|v| row_value == v)
                } else {
                    false
                }
            }
            Filter::And(filters) => filters.iter().all(|f| f.matches(row)),
            Filter::Or(filters) => filters.iter().any(|f| f.matches(row)),
            Filter::Not(filter) => !filter.matches(row),
        }
    }

    /// Fast comparison for common types (optimized hot path)
    #[inline(always)]
    fn compare_fast(left: &Value, op: &CompareOp, right: &Value) -> bool {
        // Fast path for same-type Int64 comparisons (most common case)
        if let (Value::Int64(l), Value::Int64(r)) = (left, right) {
            return match op {
                CompareOp::Equal => l == r,
                CompareOp::NotEqual => l != r,
                CompareOp::LessThan => l < r,
                CompareOp::LessEqual => l <= r,
                CompareOp::GreaterThan => l > r,
                CompareOp::GreaterEqual => l >= r,
                _ => false,
            };
        }
        
        // Fast path for Float64 comparisons
        if let (Value::Float64(l), Value::Float64(r)) = (left, right) {
            return match op {
                CompareOp::Equal => (l - r).abs() < f64::EPSILON,
                CompareOp::NotEqual => (l - r).abs() >= f64::EPSILON,
                CompareOp::LessThan => l < r,
                CompareOp::LessEqual => l <= r,
                CompareOp::GreaterThan => l > r,
                CompareOp::GreaterEqual => l >= r,
                _ => false,
            };
        }
        
        // Fast path for String comparisons
        if let (Value::String(l), Value::String(r)) = (left, right) {
            return match op {
                CompareOp::Equal => l == r,
                CompareOp::NotEqual => l != r,
                CompareOp::LessThan => l < r,
                CompareOp::LessEqual => l <= r,
                CompareOp::GreaterThan => l > r,
                CompareOp::GreaterEqual => l >= r,
                CompareOp::Like => Self::like_match(l, r),
                _ => false,
            };
        }
        
        // Fallback to generic comparison
        Self::compare(left, op, right)
    }

    /// Compare two values (generic fallback)
    fn compare(left: &Value, op: &CompareOp, right: &Value) -> bool {
        match op {
            CompareOp::Equal => left == right,
            CompareOp::NotEqual => left != right,
            CompareOp::LessThan => left.partial_cmp(right) == Some(std::cmp::Ordering::Less),
            CompareOp::LessEqual => {
                matches!(left.partial_cmp(right), Some(std::cmp::Ordering::Less | std::cmp::Ordering::Equal))
            }
            CompareOp::GreaterThan => left.partial_cmp(right) == Some(std::cmp::Ordering::Greater),
            CompareOp::GreaterEqual => {
                matches!(left.partial_cmp(right), Some(std::cmp::Ordering::Greater | std::cmp::Ordering::Equal))
            }
            CompareOp::Like => {
                if let (Value::String(l), Value::String(r)) = (left, right) {
                    Self::like_match(l, r)
                } else {
                    false
                }
            }
            CompareOp::In => false, // Handled separately
        }
    }

    /// SQL LIKE pattern matching
    fn like_match(s: &str, pattern: &str) -> bool {
        let pattern = pattern.replace('%', ".*").replace('_', ".");
        if let Ok(re) = regex::Regex::new(&format!("^{}$", pattern)) {
            re.is_match(s)
        } else {
            // Fallback to simple contains
            let clean_pattern = pattern.replace(".*", "").replace('.', "");
            s.contains(&clean_pattern)
        }
    }

    // ========================================================================
    // Column-based filtering (high-performance path)
    // ========================================================================

    /// Filter columns and return matching row indices
    /// This is the fastest path for column-oriented storage
    pub fn filter_columns(
        &self,
        schema: &ColumnSchema,
        columns: &[TypedColumn],
        row_count: usize,
        deleted: &BitVec,
    ) -> Vec<usize> {
        match self {
            Filter::True => {
                // All non-deleted rows
                (0..row_count).filter(|&i| !deleted.get(i)).collect()
            }
            Filter::False => Vec::new(),
            Filter::Compare { field, op, value } => {
                self.filter_compare_column(schema, columns, row_count, deleted, field, op, value)
            }
            Filter::And(filters) => {
                // Start with all rows
                let mut result: Vec<usize> = (0..row_count).filter(|&i| !deleted.get(i)).collect();
                for filter in filters {
                    let matching = filter.filter_columns(schema, columns, row_count, deleted);
                    result.retain(|idx| matching.contains(idx));
                    if result.is_empty() {
                        break;
                    }
                }
                result
            }
            Filter::Or(filters) => {
                let mut result_set = std::collections::HashSet::new();
                for filter in filters {
                    for idx in filter.filter_columns(schema, columns, row_count, deleted) {
                        result_set.insert(idx);
                    }
                }
                let mut result: Vec<usize> = result_set.into_iter().collect();
                result.sort_unstable();
                result
            }
            Filter::Not(filter) => {
                let matching = filter.filter_columns(schema, columns, row_count, deleted);
                let matching_set: std::collections::HashSet<_> = matching.into_iter().collect();
                (0..row_count)
                    .filter(|&i| !deleted.get(i) && !matching_set.contains(&i))
                    .collect()
            }
            Filter::Like { field, pattern } => {
                self.filter_like_column(schema, columns, row_count, deleted, field, pattern)
            }
            Filter::In { field, values } => {
                self.filter_in_column(schema, columns, row_count, deleted, field, values)
            }
        }
    }

    /// Filter a single Compare condition directly on column data
    /// OPTIMIZED: Uses parallel processing for large datasets
    #[inline]
    fn filter_compare_column(
        &self,
        schema: &ColumnSchema,
        columns: &[TypedColumn],
        row_count: usize,
        deleted: &BitVec,
        field: &str,
        op: &CompareOp,
        value: &Value,
    ) -> Vec<usize> {
        use rayon::prelude::*;
        
        let col_idx = match schema.get_index(field) {
            Some(idx) => idx,
            None => return Vec::new(),
        };

        let column = &columns[col_idx];
        let no_deletes = deleted.all_false();
        
        // Use parallel processing for large datasets (> 100K rows)
        let use_parallel = row_count >= 100_000;

        match (column, value) {
            // Fast path: Int64 column with Int64 value
            (TypedColumn::Int64 { data, nulls }, Value::Int64(target)) => {
                let no_nulls = nulls.all_false();
                let target = *target;
                let data_len = data.len().min(row_count);
                
                if use_parallel {
                    // Parallel filtering using rayon
                    (0..data_len).into_par_iter()
                        .filter(|&i| {
                            let skip = (!no_deletes && deleted.get(i)) || (!no_nulls && nulls.get(i));
                            if skip { return false; }
                            let val = data[i];
                            match op {
                                CompareOp::Equal => val == target,
                                CompareOp::NotEqual => val != target,
                                CompareOp::LessThan => val < target,
                                CompareOp::LessEqual => val <= target,
                                CompareOp::GreaterThan => val > target,
                                CompareOp::GreaterEqual => val >= target,
                                _ => false,
                            }
                        })
                        .collect()
                } else {
                    // Sequential for smaller datasets
                    let mut result = Vec::with_capacity(row_count / 4);
                    for i in 0..data_len {
                        let skip = (!no_deletes && deleted.get(i)) || (!no_nulls && nulls.get(i));
                        if skip { continue; }
                        let val = data[i];
                        let matches = match op {
                            CompareOp::Equal => val == target,
                            CompareOp::NotEqual => val != target,
                            CompareOp::LessThan => val < target,
                            CompareOp::LessEqual => val <= target,
                            CompareOp::GreaterThan => val > target,
                            CompareOp::GreaterEqual => val >= target,
                            _ => false,
                        };
                        if matches { result.push(i); }
                    }
                    result
                }
            }
            // Fast path: Float64 column with Float64 value
            (TypedColumn::Float64 { data, nulls }, Value::Float64(target)) => {
                let no_nulls = nulls.all_false();
                let target = *target;
                let data_len = data.len().min(row_count);
                
                if use_parallel {
                    (0..data_len).into_par_iter()
                        .filter(|&i| {
                            let skip = (!no_deletes && deleted.get(i)) || (!no_nulls && nulls.get(i));
                            if skip { return false; }
                            let val = data[i];
                            match op {
                                CompareOp::Equal => (val - target).abs() < f64::EPSILON,
                                CompareOp::NotEqual => (val - target).abs() >= f64::EPSILON,
                                CompareOp::LessThan => val < target,
                                CompareOp::LessEqual => val <= target,
                                CompareOp::GreaterThan => val > target,
                                CompareOp::GreaterEqual => val >= target,
                                _ => false,
                            }
                        })
                        .collect()
                } else {
                    let mut result = Vec::with_capacity(row_count / 4);
                    for i in 0..data_len {
                        let skip = (!no_deletes && deleted.get(i)) || (!no_nulls && nulls.get(i));
                        if skip { continue; }
                        let val = data[i];
                        let matches = match op {
                            CompareOp::Equal => (val - target).abs() < f64::EPSILON,
                            CompareOp::NotEqual => (val - target).abs() >= f64::EPSILON,
                            CompareOp::LessThan => val < target,
                            CompareOp::LessEqual => val <= target,
                            CompareOp::GreaterThan => val > target,
                            CompareOp::GreaterEqual => val >= target,
                            _ => false,
                        };
                        if matches { result.push(i); }
                    }
                    result
                }
            }
            // Fast path: String column with String value
            (TypedColumn::String { data, nulls }, Value::String(target)) => {
                let no_nulls = nulls.all_false();
                let data_len = data.len().min(row_count);
                
                if use_parallel {
                    (0..data_len).into_par_iter()
                        .filter(|&i| {
                            let skip = (!no_deletes && deleted.get(i)) || (!no_nulls && nulls.get(i));
                            if skip { return false; }
                            let val = &data[i];
                            match op {
                                CompareOp::Equal => val == target,
                                CompareOp::NotEqual => val != target,
                                CompareOp::LessThan => val < target,
                                CompareOp::LessEqual => val <= target,
                                CompareOp::GreaterThan => val > target,
                                CompareOp::GreaterEqual => val >= target,
                                CompareOp::Like => Self::like_match(val, target),
                                _ => false,
                            }
                        })
                        .collect()
                } else {
                    let mut result = Vec::with_capacity(row_count / 4);
                    for i in 0..data_len {
                        let skip = (!no_deletes && deleted.get(i)) || (!no_nulls && nulls.get(i));
                        if skip { continue; }
                        let val = &data[i];
                        let matches = match op {
                            CompareOp::Equal => val == target,
                            CompareOp::NotEqual => val != target,
                            CompareOp::LessThan => val < target,
                            CompareOp::LessEqual => val <= target,
                            CompareOp::GreaterThan => val > target,
                            CompareOp::GreaterEqual => val >= target,
                            CompareOp::Like => Self::like_match(val, target),
                            _ => false,
                        };
                        if matches { result.push(i); }
                    }
                    result
                }
            }
            // Fallback: use generic Value comparison
            _ => {
                let mut result = Vec::with_capacity(row_count / 4);
                for i in 0..row_count {
                    if !deleted.get(i) {
                        if let Some(row_value) = column.get(i) {
                            if !row_value.is_null() && Self::compare_fast(&row_value, op, value) {
                                result.push(i);
                            }
                        }
                    }
                }
                result
            }
        }
    }

    /// Filter LIKE condition on column data
    fn filter_like_column(
        &self,
        schema: &ColumnSchema,
        columns: &[TypedColumn],
        row_count: usize,
        deleted: &BitVec,
        field: &str,
        pattern: &str,
    ) -> Vec<usize> {
        let col_idx = match schema.get_index(field) {
            Some(idx) => idx,
            None => return Vec::new(),
        };

        let column = &columns[col_idx];
        let mut result = Vec::new();

        if let TypedColumn::String { data, nulls } = column {
            for (i, val) in data.iter().enumerate() {
                if i < row_count && !deleted.get(i) && !nulls.get(i) {
                    if Self::like_match(val, pattern) {
                        result.push(i);
                    }
                }
            }
        }

        result
    }

    /// Filter IN condition on column data
    fn filter_in_column(
        &self,
        schema: &ColumnSchema,
        columns: &[TypedColumn],
        row_count: usize,
        deleted: &BitVec,
        field: &str,
        values: &[Value],
    ) -> Vec<usize> {
        let col_idx = match schema.get_index(field) {
            Some(idx) => idx,
            None => return Vec::new(),
        };

        let column = &columns[col_idx];
        let mut result = Vec::new();

        for i in 0..row_count {
            if !deleted.get(i) {
                if let Some(row_value) = column.get(i) {
                    if !row_value.is_null() && values.iter().any(|v| &row_value == v) {
                        result.push(i);
                    }
                }
            }
        }

        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_row(id: u64, name: &str, age: i64) -> Row {
        let mut row = Row::new(id);
        row.set("name", name);
        row.set("age", Value::Int64(age));
        row
    }

    #[test]
    fn test_compare_filter() {
        let row = make_row(1, "John", 30);

        let filter = Filter::Compare {
            field: "age".to_string(),
            op: CompareOp::GreaterThan,
            value: Value::Int64(25),
        };
        assert!(filter.matches(&row));

        let filter = Filter::Compare {
            field: "age".to_string(),
            op: CompareOp::LessThan,
            value: Value::Int64(25),
        };
        assert!(!filter.matches(&row));
    }

    #[test]
    fn test_like_filter() {
        let row = make_row(1, "John Smith", 30);

        let filter = Filter::Like {
            field: "name".to_string(),
            pattern: "John%".to_string(),
        };
        assert!(filter.matches(&row));

        let filter = Filter::Like {
            field: "name".to_string(),
            pattern: "%Smith".to_string(),
        };
        assert!(filter.matches(&row));
    }

    #[test]
    fn test_and_filter() {
        let row = make_row(1, "John", 30);

        let filter = Filter::And(vec![
            Filter::Compare {
                field: "age".to_string(),
                op: CompareOp::GreaterThan,
                value: Value::Int64(25),
            },
            Filter::Compare {
                field: "name".to_string(),
                op: CompareOp::Equal,
                value: Value::String("John".to_string()),
            },
        ]);
        assert!(filter.matches(&row));
    }

    #[test]
    fn test_or_filter() {
        let row = make_row(1, "John", 30);

        let filter = Filter::Or(vec![
            Filter::Compare {
                field: "age".to_string(),
                op: CompareOp::LessThan,
                value: Value::Int64(25),
            },
            Filter::Compare {
                field: "name".to_string(),
                op: CompareOp::Equal,
                value: Value::String("John".to_string()),
            },
        ]);
        assert!(filter.matches(&row));
    }
}

