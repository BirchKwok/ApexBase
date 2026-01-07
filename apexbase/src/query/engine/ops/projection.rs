use crate::query::sql_parser::SelectColumn;
use crate::query::{AggregateFunc, SqlResult};
use crate::table::column_table::TypedColumn;
use crate::table::ColumnTable;
use crate::ApexError;

pub(crate) fn resolve_columns(
    columns: &[SelectColumn],
    table: &ColumnTable,
) -> Result<(Vec<String>, Vec<(String, Option<usize>)>), ApexError> {
    let mut result_names = Vec::new();
    let mut column_indices = Vec::new();
    let mut seen = std::collections::HashSet::new();
    let schema = table.schema_ref();

    let explicit_id_requested = columns.iter().any(|c| match c {
        SelectColumn::Column(name) => name == "_id",
        SelectColumn::ColumnAlias { column, .. } => column == "_id",
        _ => false,
    });

    for col in columns {
        match col {
            SelectColumn::All => {
                if !explicit_id_requested {
                    if seen.insert("_id".to_string()) {
                        result_names.push("_id".to_string());
                        column_indices.push(("_id".to_string(), None));
                    }
                }

                for (name, _) in &schema.columns {
                    if seen.insert(name.clone()) {
                        result_names.push(name.clone());
                        let idx = schema.get_index(name);
                        column_indices.push((name.clone(), idx));
                    }
                }
            }
            SelectColumn::Column(name) => {
                let (display_name, lookup_name) = if name.contains('.') {
                    let last = name.rsplit('.').next().unwrap_or(name.as_str()).to_string();
                    (last.clone(), last)
                } else {
                    (name.clone(), name.clone())
                };

                if seen.insert(display_name.clone()) {
                    result_names.push(display_name.clone());
                    if lookup_name == "_id" {
                        column_indices.push((lookup_name, None));
                    } else {
                        let idx = schema.get_index(&lookup_name);
                        column_indices.push((lookup_name, idx));
                    }
                }
            }
            SelectColumn::ColumnAlias { column, alias } => {
                if seen.insert(alias.clone()) {
                    result_names.push(alias.clone());
                    if column == "_id" {
                        column_indices.push((column.clone(), None));
                    } else {
                        let lookup = if column.contains('.') {
                            column.rsplit('.').next().unwrap_or(column.as_str())
                        } else {
                            column
                        };
                        let idx = schema.get_index(lookup);
                        column_indices.push((lookup.to_string(), idx));
                    }
                }
            }
            SelectColumn::Aggregate {
                func,
                column,
                distinct,
                alias,
            } => {
                let name = alias.clone().unwrap_or_else(|| {
                    let func_name = match func {
                        AggregateFunc::Count => "COUNT",
                        AggregateFunc::Sum => "SUM",
                        AggregateFunc::Avg => "AVG",
                        AggregateFunc::Min => "MIN",
                        AggregateFunc::Max => "MAX",
                    };
                    if let Some(col) = column {
                        if *distinct {
                            format!("{}(DISTINCT {})", func_name, col)
                        } else {
                            format!("{}({})", func_name, col)
                        }
                    } else {
                        format!("{}(*)", func_name)
                    }
                });
                if seen.insert(name.clone()) {
                    result_names.push(name.clone());
                    column_indices.push((name, None));
                }
            }
            SelectColumn::Expression { alias, .. } => {
                let name = alias.clone().unwrap_or_else(|| "expr".to_string());
                if seen.insert(name.clone()) {
                    result_names.push(name.clone());
                    column_indices.push((name, None));
                }
            }
            SelectColumn::WindowFunction { alias, name, .. } => {
                let col_name = alias.clone().unwrap_or_else(|| name.clone());
                if seen.insert(col_name.clone()) {
                    result_names.push(col_name.clone());
                    column_indices.push((col_name, None));
                }
            }
        }
    }

    Ok((result_names, column_indices))
}

pub(crate) fn build_arrow_direct(
    result_columns: &[String],
    column_indices: &[(String, Option<usize>)],
    matching_indices: &[usize],
    table: &ColumnTable,
) -> Result<SqlResult, ApexError> {
    use arrow::array::{ArrayRef, BooleanArray, Float64Array, Int64Array, StringBuilder};
    use arrow::datatypes::{DataType as ArrowDataType, Field, Schema};
    use arrow::record_batch::RecordBatch;
    use rayon::prelude::*;
    use std::sync::Arc;

    let columns = table.columns_ref();
    let num_rows = matching_indices.len();

    let mut fields = Vec::with_capacity(result_columns.len());
    let mut arrays: Vec<ArrayRef> = Vec::with_capacity(result_columns.len());

    for (col_name, col_idx) in column_indices {
        if col_name == "_id" {
            let id_values: Vec<i64> = matching_indices.iter().map(|&i| i as i64).collect();
            fields.push(Field::new("_id", ArrowDataType::Int64, false));
            arrays.push(Arc::new(Int64Array::from(id_values)));
        } else if let Some(idx) = col_idx {
            match &columns[*idx] {
                TypedColumn::Int64 { data, nulls } => {
                    let values: Vec<Option<i64>> = if num_rows > 100_000 {
                        matching_indices
                            .par_iter()
                            .map(|&i| if i < data.len() && !nulls.get(i) { Some(data[i]) } else { None })
                            .collect()
                    } else {
                        matching_indices
                            .iter()
                            .map(|&i| if i < data.len() && !nulls.get(i) { Some(data[i]) } else { None })
                            .collect()
                    };
                    fields.push(Field::new(col_name, ArrowDataType::Int64, true));
                    arrays.push(Arc::new(Int64Array::from(values)));
                }
                TypedColumn::Float64 { data, nulls } => {
                    let values: Vec<Option<f64>> = if num_rows > 100_000 {
                        matching_indices
                            .par_iter()
                            .map(|&i| if i < data.len() && !nulls.get(i) { Some(data[i]) } else { None })
                            .collect()
                    } else {
                        matching_indices
                            .iter()
                            .map(|&i| if i < data.len() && !nulls.get(i) { Some(data[i]) } else { None })
                            .collect()
                    };
                    fields.push(Field::new(col_name, ArrowDataType::Float64, true));
                    arrays.push(Arc::new(Float64Array::from(values)));
                }
                TypedColumn::String(col) => {
                    let dtype = if col.is_dictionary_enabled() {
                        ArrowDataType::Dictionary(
                            Box::new(ArrowDataType::Int32),
                            Box::new(ArrowDataType::Utf8),
                        )
                    } else {
                        ArrowDataType::Utf8
                    };
                    fields.push(Field::new(col_name, dtype, true));
                    arrays.push(col.to_arrow_array_indexed(matching_indices));
                }
                TypedColumn::Bool { data, nulls } => {
                    let values: Vec<Option<bool>> = matching_indices
                        .iter()
                        .map(|&i| if i < data.len() && !nulls.get(i) { Some(data.get(i)) } else { None })
                        .collect();
                    fields.push(Field::new(col_name, ArrowDataType::Boolean, true));
                    arrays.push(Arc::new(BooleanArray::from(values)));
                }
                TypedColumn::Mixed { data, nulls } => {
                    let mut builder = StringBuilder::with_capacity(num_rows, num_rows * 32);
                    for &i in matching_indices {
                        if i < data.len() && !nulls.get(i) {
                            builder.append_value(data[i].to_string_value());
                        } else {
                            builder.append_null();
                        }
                    }
                    fields.push(Field::new(col_name, ArrowDataType::Utf8, true));
                    arrays.push(Arc::new(builder.finish()));
                }
            }
        } else {
            let null_values: Vec<Option<i64>> = vec![None; num_rows];
            fields.push(Field::new(col_name, ArrowDataType::Int64, true));
            arrays.push(Arc::new(Int64Array::from(null_values)));
        }
    }

    let schema = Arc::new(Schema::new(fields));
    let batch = RecordBatch::try_new(schema, arrays)
        .map_err(|e| ApexError::SerializationError(e.to_string()))?;

    Ok(SqlResult::with_arrow_batch(result_columns.to_vec(), batch))
}
