//! IoResult type and conversion helpers
//!
//! This module contains the unified query result type that can hold
//! different output formats (Arrow, Rows, Scalar).

use crate::data::Value;
use crate::ApexError;
use arrow::record_batch::RecordBatch;
use std::collections::HashMap;

/// Unified query result that can hold different output formats
pub enum IoResult {
    /// Arrow RecordBatch for large columnar results
    Arrow(RecordBatch),
    /// Row-based results for small queries
    Rows(Vec<HashMap<String, Value>>),
    /// Single scalar value for aggregates
    Scalar(Value),
    /// Empty result
    Empty,
}

impl IoResult {
    /// Get number of rows in result
    pub fn num_rows(&self) -> usize {
        match self {
            IoResult::Arrow(batch) => batch.num_rows(),
            IoResult::Rows(rows) => rows.len(),
            IoResult::Scalar(_) => 1,
            IoResult::Empty => 0,
        }
    }
    
    /// Check if result is empty
    pub fn is_empty(&self) -> bool {
        self.num_rows() == 0
    }
    
    /// Convert to Arrow RecordBatch (may involve conversion)
    pub fn to_arrow(&self) -> Result<RecordBatch, ApexError> {
        match self {
            IoResult::Arrow(batch) => Ok(batch.clone()),
            IoResult::Rows(rows) => Self::rows_to_arrow(rows),
            IoResult::Scalar(val) => Self::scalar_to_arrow(val),
            IoResult::Empty => Self::empty_arrow(),
        }
    }
    
    /// Convert to row-based format (may involve conversion)
    pub fn to_rows(&self) -> Vec<HashMap<String, Value>> {
        match self {
            IoResult::Arrow(batch) => Self::arrow_to_rows(batch),
            IoResult::Rows(rows) => rows.clone(),
            IoResult::Scalar(val) => {
                let mut row = HashMap::new();
                row.insert("result".to_string(), val.clone());
                vec![row]
            }
            IoResult::Empty => vec![],
        }
    }
    
    /// Convert rows to Arrow RecordBatch
    pub(crate) fn rows_to_arrow(rows: &[HashMap<String, Value>]) -> Result<RecordBatch, ApexError> {
        use arrow::array::{ArrayRef, StringBuilder, Int64Array, Float64Array, BooleanArray};
        use arrow::datatypes::{DataType as ArrowDataType, Field, Schema};
        use std::sync::Arc;
        
        if rows.is_empty() {
            return Self::empty_arrow();
        }
        
        let first_row = &rows[0];
        let mut fields = Vec::new();
        let mut arrays: Vec<ArrayRef> = Vec::new();
        
        for (col_name, sample_val) in first_row {
            let (arrow_type, array): (ArrowDataType, ArrayRef) = match sample_val {
                Value::Int64(_) | Value::Int32(_) | Value::Int16(_) | Value::Int8(_) => {
                    let values: Vec<Option<i64>> = rows.iter()
                        .map(|row| row.get(col_name).and_then(|v| v.as_i64()))
                        .collect();
                    (ArrowDataType::Int64, Arc::new(Int64Array::from(values)))
                }
                Value::Float64(_) | Value::Float32(_) => {
                    let values: Vec<Option<f64>> = rows.iter()
                        .map(|row| row.get(col_name).and_then(|v| v.as_f64()))
                        .collect();
                    (ArrowDataType::Float64, Arc::new(Float64Array::from(values)))
                }
                Value::Bool(_) => {
                    let values: Vec<Option<bool>> = rows.iter()
                        .map(|row| row.get(col_name).and_then(|v| v.as_bool()))
                        .collect();
                    (ArrowDataType::Boolean, Arc::new(BooleanArray::from(values)))
                }
                _ => {
                    let mut builder = StringBuilder::with_capacity(rows.len(), rows.len() * 32);
                    for row in rows {
                        match row.get(col_name) {
                            Some(v) if !v.is_null() => builder.append_value(v.to_string_value()),
                            _ => builder.append_null(),
                        }
                    }
                    (ArrowDataType::Utf8, Arc::new(builder.finish()))
                }
            };
            fields.push(Field::new(col_name, arrow_type, true));
            arrays.push(array);
        }
        
        let schema = Arc::new(Schema::new(fields));
        RecordBatch::try_new(schema, arrays)
            .map_err(|e| ApexError::SerializationError(e.to_string()))
    }
    
    /// Convert scalar value to Arrow RecordBatch
    fn scalar_to_arrow(val: &Value) -> Result<RecordBatch, ApexError> {
        use arrow::array::{ArrayRef, Int64Array, Float64Array, StringBuilder};
        use arrow::datatypes::{DataType as ArrowDataType, Field, Schema};
        use std::sync::Arc;
        
        let (arrow_type, array): (ArrowDataType, ArrayRef) = match val {
            Value::Int64(v) => (ArrowDataType::Int64, Arc::new(Int64Array::from(vec![*v]))),
            Value::Float64(v) => (ArrowDataType::Float64, Arc::new(Float64Array::from(vec![*v]))),
            _ => {
                let mut builder = StringBuilder::new();
                builder.append_value(val.to_string_value());
                (ArrowDataType::Utf8, Arc::new(builder.finish()))
            }
        };
        
        let schema = Arc::new(Schema::new(vec![Field::new("result", arrow_type, true)]));
        RecordBatch::try_new(schema, vec![array])
            .map_err(|e| ApexError::SerializationError(e.to_string()))
    }
    
    /// Create empty Arrow RecordBatch
    fn empty_arrow() -> Result<RecordBatch, ApexError> {
        use arrow::datatypes::Schema;
        use std::sync::Arc;
        
        let schema = Arc::new(Schema::empty());
        RecordBatch::try_new(schema, vec![])
            .map_err(|e| ApexError::SerializationError(e.to_string()))
    }
    
    /// Convert Arrow RecordBatch to rows
    fn arrow_to_rows(batch: &RecordBatch) -> Vec<HashMap<String, Value>> {
        use arrow::array::{Array, Int64Array, Float64Array, StringArray, BooleanArray};
        use arrow::datatypes::DataType as ArrowDataType;
        
        let mut rows = Vec::with_capacity(batch.num_rows());
        let schema = batch.schema();
        
        for row_idx in 0..batch.num_rows() {
            let mut row = HashMap::new();
            for (col_idx, field) in schema.fields().iter().enumerate() {
                let col = batch.column(col_idx);
                let value = if col.is_null(row_idx) {
                    Value::Null
                } else {
                    match field.data_type() {
                        ArrowDataType::Int64 => {
                            let arr = col.as_any().downcast_ref::<Int64Array>().unwrap();
                            Value::Int64(arr.value(row_idx))
                        }
                        ArrowDataType::Float64 => {
                            let arr = col.as_any().downcast_ref::<Float64Array>().unwrap();
                            Value::Float64(arr.value(row_idx))
                        }
                        ArrowDataType::Boolean => {
                            let arr = col.as_any().downcast_ref::<BooleanArray>().unwrap();
                            Value::Bool(arr.value(row_idx))
                        }
                        ArrowDataType::Utf8 => {
                            let arr = col.as_any().downcast_ref::<StringArray>().unwrap();
                            Value::String(arr.value(row_idx).to_string())
                        }
                        _ => Value::Null,
                    }
                };
                row.insert(field.name().clone(), value);
            }
            rows.push(row);
        }
        rows
    }
}
