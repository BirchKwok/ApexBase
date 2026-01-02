//! Column definitions and column block storage

use super::{DataType, Value};
use serde::{Deserialize, Serialize};

/// Column definition
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ColumnDef {
    /// Column ID (unique within a table)
    pub id: u16,
    /// Column name
    pub name: String,
    /// Data type
    pub data_type: DataType,
    /// Whether the column can contain null values
    pub nullable: bool,
    /// Whether this column is indexed
    pub indexed: bool,
    /// Default value (optional)
    pub default_value: Option<Value>,
    /// Ordinal position in the table
    pub ordinal_position: u16,
}

impl ColumnDef {
    /// Create a new column definition
    pub fn new(id: u16, name: impl Into<String>, data_type: DataType) -> Self {
        Self {
            id,
            name: name.into(),
            data_type,
            nullable: true,
            indexed: false,
            default_value: None,
            ordinal_position: id,
        }
    }

    /// Set nullable flag
    pub fn nullable(mut self, nullable: bool) -> Self {
        self.nullable = nullable;
        self
    }

    /// Set indexed flag
    pub fn indexed(mut self, indexed: bool) -> Self {
        self.indexed = indexed;
        self
    }

    /// Set default value
    pub fn with_default(mut self, value: Value) -> Self {
        self.default_value = Some(value);
        self
    }

    /// Set ordinal position
    pub fn position(mut self, pos: u16) -> Self {
        self.ordinal_position = pos;
        self
    }
}

/// A column of values (columnar storage)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Column {
    /// Column definition
    pub def: ColumnDef,
    /// Values in this column
    pub values: Vec<Value>,
    /// Null bitmap (bit i is 1 if row i is null)
    pub null_bitmap: Vec<u8>,
}

impl Column {
    /// Create a new column
    pub fn new(def: ColumnDef) -> Self {
        Self {
            def,
            values: Vec::new(),
            null_bitmap: Vec::new(),
        }
    }

    /// Create a column with preallocated capacity
    pub fn with_capacity(def: ColumnDef, capacity: usize) -> Self {
        Self {
            def,
            values: Vec::with_capacity(capacity),
            null_bitmap: vec![0; (capacity + 7) / 8],
        }
    }

    /// Get the number of values
    pub fn len(&self) -> usize {
        self.values.len()
    }

    /// Check if the column is empty
    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }

    /// Get a value at the given index
    pub fn get(&self, index: usize) -> Option<&Value> {
        self.values.get(index)
    }

    /// Check if a value at the given index is null
    pub fn is_null(&self, index: usize) -> bool {
        if index >= self.values.len() {
            return true;
        }
        let byte_index = index / 8;
        let bit_index = index % 8;
        if byte_index >= self.null_bitmap.len() {
            return false;
        }
        (self.null_bitmap[byte_index] >> bit_index) & 1 == 1
    }

    /// Set a value as null
    fn set_null(&mut self, index: usize, is_null: bool) {
        let byte_index = index / 8;
        let bit_index = index % 8;
        
        // Ensure the bitmap is large enough
        while self.null_bitmap.len() <= byte_index {
            self.null_bitmap.push(0);
        }
        
        if is_null {
            self.null_bitmap[byte_index] |= 1 << bit_index;
        } else {
            self.null_bitmap[byte_index] &= !(1 << bit_index);
        }
    }

    /// Push a value to the column
    pub fn push(&mut self, value: Value) {
        let is_null = value.is_null();
        let index = self.values.len();
        self.values.push(value);
        self.set_null(index, is_null);
    }

    /// Push a null value
    pub fn push_null(&mut self) {
        let index = self.values.len();
        self.values.push(Value::Null);
        self.set_null(index, true);
    }

    /// Set a value at the given index
    pub fn set(&mut self, index: usize, value: Value) {
        if index >= self.values.len() {
            // Extend with nulls
            while self.values.len() <= index {
                self.push_null();
            }
        }
        let is_null = value.is_null();
        self.values[index] = value;
        self.set_null(index, is_null);
    }

    /// Get the data type
    pub fn data_type(&self) -> DataType {
        self.def.data_type
    }

    /// Get the column name
    pub fn name(&self) -> &str {
        &self.def.name
    }

    /// Iterate over values
    pub fn iter(&self) -> impl Iterator<Item = &Value> {
        self.values.iter()
    }

    /// Count non-null values
    pub fn count_non_null(&self) -> usize {
        self.values.iter().filter(|v| !v.is_null()).count()
    }

    /// Get all non-null values
    pub fn non_null_values(&self) -> Vec<&Value> {
        self.values.iter().filter(|v| !v.is_null()).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_column_def() {
        let col = ColumnDef::new(1, "name", DataType::String)
            .nullable(false)
            .indexed(true);

        assert_eq!(col.name, "name");
        assert_eq!(col.data_type, DataType::String);
        assert!(!col.nullable);
        assert!(col.indexed);
    }

    #[test]
    fn test_column_operations() {
        let def = ColumnDef::new(1, "age", DataType::Int64);
        let mut col = Column::new(def);

        col.push(Value::Int64(30));
        col.push(Value::Null);
        col.push(Value::Int64(25));

        assert_eq!(col.len(), 3);
        assert_eq!(col.get(0), Some(&Value::Int64(30)));
        assert!(!col.is_null(0));
        assert!(col.is_null(1));
        assert!(!col.is_null(2));
    }

    #[test]
    fn test_null_bitmap() {
        let def = ColumnDef::new(1, "value", DataType::Int64);
        let mut col = Column::new(def);

        // Push 10 values with some nulls
        for i in 0..10 {
            if i % 3 == 0 {
                col.push_null();
            } else {
                col.push(Value::Int64(i as i64));
            }
        }

        // Check nulls
        for i in 0..10 {
            assert_eq!(col.is_null(i), i % 3 == 0);
        }
    }
}

