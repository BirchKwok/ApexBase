use super::*;

pub struct SqlResult {
    /// Column names in result
    pub columns: Vec<String>,
    /// Row data (each row is a vector of values)
    pub rows: Vec<Vec<Value>>,
    /// Number of rows affected (for non-SELECT)
    pub rows_affected: usize,
    /// Pre-built Arrow batch for fast path (bypasses row conversion)
    pub arrow_batch: Option<arrow::record_batch::RecordBatch>,
}

impl SqlResult {
    pub fn new(columns: Vec<String>, rows: Vec<Vec<Value>>) -> Self {
        let rows_affected = rows.len();
        Self { columns, rows, rows_affected, arrow_batch: None }
    }

    pub fn empty() -> Self {
        Self { columns: Vec::new(), rows: Vec::new(), rows_affected: 0, arrow_batch: None }
    }
    
    /// Create SqlResult with pre-built Arrow batch (fast path)
    pub fn with_arrow_batch(columns: Vec<String>, batch: arrow::record_batch::RecordBatch) -> Self {
        let rows_affected = batch.num_rows();
        Self { columns, rows: Vec::new(), rows_affected, arrow_batch: Some(batch) }
    }
    
    /// Convert SqlResult to Arrow RecordBatch for fast Python transfer
    pub fn to_record_batch(&self) -> Result<arrow::record_batch::RecordBatch, ApexError> {
        use arrow::array::{ArrayRef, Float64Array, Int64Array, StringBuilder, BooleanArray};
        use arrow::datatypes::{DataType as ArrowDataType, Field, Schema};
        use std::sync::Arc;
        
        // Fast path: return pre-built Arrow batch if available
        if let Some(ref batch) = self.arrow_batch {
            return Ok(batch.clone());
        }
        
        if self.rows.is_empty() {
            use arrow::array::new_null_array;

            let fields: Vec<Field> = self.columns.iter()
                .map(|name| Field::new(name, ArrowDataType::Utf8, true))
                .collect();
            let schema = Arc::new(Schema::new(fields.clone()));
            let arrays: Vec<ArrayRef> = fields
                .iter()
                .map(|f| new_null_array(f.data_type(), 0))
                .collect();

            return arrow::record_batch::RecordBatch::try_new(schema, arrays)
                .map_err(|e| ApexError::SerializationError(e.to_string()));
        }
        
        // Infer types from first row
        let first_row = &self.rows[0];
        let mut fields = Vec::with_capacity(self.columns.len());
        let mut arrays: Vec<ArrayRef> = Vec::with_capacity(self.columns.len());
        
        for (col_idx, col_name) in self.columns.iter().enumerate() {
            let sample_val = first_row.get(col_idx);
            let (arrow_type, array) = match sample_val {
                Some(Value::Int64(_)) | Some(Value::Int32(_)) | Some(Value::Int16(_)) | Some(Value::Int8(_)) => {
                    let values: Vec<Option<i64>> = self.rows.iter()
                        .map(|row| row.get(col_idx).and_then(|v| v.as_i64()))
                        .collect();
                    (ArrowDataType::Int64, Arc::new(Int64Array::from(values)) as ArrayRef)
                }
                Some(Value::Float64(_)) | Some(Value::Float32(_)) => {
                    let values: Vec<Option<f64>> = self.rows.iter()
                        .map(|row| row.get(col_idx).and_then(|v| v.as_f64()))
                        .collect();
                    (ArrowDataType::Float64, Arc::new(Float64Array::from(values)) as ArrayRef)
                }
                Some(Value::Bool(_)) => {
                    let values: Vec<Option<bool>> = self.rows.iter()
                        .map(|row| row.get(col_idx).and_then(|v| v.as_bool()))
                        .collect();
                    (ArrowDataType::Boolean, Arc::new(BooleanArray::from(values)) as ArrayRef)
                }
                _ => {
                    // String or mixed - convert to string
                    let mut builder = StringBuilder::with_capacity(self.rows.len(), self.rows.len() * 32);
                    for row in &self.rows {
                        match row.get(col_idx) {
                            Some(v) if !v.is_null() => builder.append_value(v.to_string_value()),
                            _ => builder.append_null(),
                        }
                    }
                    (ArrowDataType::Utf8, Arc::new(builder.finish()) as ArrayRef)
                }
            };
            fields.push(Field::new(col_name, arrow_type, true));
            arrays.push(array);
        }
        
        let schema = Arc::new(Schema::new(fields));
        arrow::record_batch::RecordBatch::try_new(schema, arrays)
            .map_err(|e| ApexError::SerializationError(e.to_string()))
    }
}

