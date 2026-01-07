use crate::data::Value;
use crate::query::sql_parser::{BinaryOperator, SelectColumn, SqlExpr, UnaryOperator};
use crate::query::{AggregateFunc, SqlResult};
use crate::table::column_table::TypedColumn;
use crate::table::ColumnTable;
use crate::ApexError;
use arrow::array::{ArrayRef, BooleanArray, Float64Array, Int64Array, StringBuilder};
use arrow::datatypes::{DataType as ArrowDataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use rayon::prelude::*;
use std::sync::Arc;

pub(crate) struct EvalContext {
    pub(crate) now_str: String,
}

pub(crate) fn new_eval_context() -> EvalContext {
    EvalContext {
        now_str: chrono::Local::now()
            .format("%Y-%m-%d %H:%M:%S")
            .to_string(),
    }
}

pub(crate) fn resolve_columns(
    columns: &[SelectColumn],
    table: &ColumnTable,
) -> Result<(Vec<String>, Vec<(String, Option<usize>)>, Vec<Option<SqlExpr>>), ApexError> {
    let mut result_names = Vec::new();
    let mut column_indices = Vec::new();
    let mut projected_exprs: Vec<Option<SqlExpr>> = Vec::new();
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
                        projected_exprs.push(None);
                    }
                }

                for (name, _) in &schema.columns {
                    if seen.insert(name.clone()) {
                        result_names.push(name.clone());
                        let idx = schema.get_index(name);
                        column_indices.push((name.clone(), idx));
                        projected_exprs.push(None);
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
                    projected_exprs.push(None);
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
                    projected_exprs.push(None);
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
                    projected_exprs.push(None);
                }
            }
            SelectColumn::Expression { expr, alias } => {
                let name = alias.clone().unwrap_or_else(|| "expr".to_string());
                if seen.insert(name.clone()) {
                    result_names.push(name.clone());
                    column_indices.push((name, None));
                    projected_exprs.push(Some(expr.clone()));
                }
            }
            SelectColumn::WindowFunction { alias, name, .. } => {
                let col_name = alias.clone().unwrap_or_else(|| name.clone());
                if seen.insert(col_name.clone()) {
                    result_names.push(col_name.clone());
                    column_indices.push((col_name, None));
                    projected_exprs.push(None);
                }
            }
        }
    }

    Ok((result_names, column_indices, projected_exprs))
}

pub(crate) fn eval_scalar_expr(
    expr: &SqlExpr,
    table: &ColumnTable,
    row_idx: usize,
    ctx: &EvalContext,
) -> Result<Value, ApexError> {
    fn get_col_value(table: &ColumnTable, col_ref: &str, row_idx: usize) -> Value {
        let name = if col_ref.contains('.') {
            col_ref.rsplit('.').next().unwrap_or(col_ref)
        } else {
            col_ref
        };
        if name == "_id" {
            return Value::Int64(row_idx as i64);
        }
        let schema = table.schema_ref();
        if let Some(ci) = schema.get_index(name) {
            table.columns_ref()[ci].get(row_idx).unwrap_or(Value::Null)
        } else {
            Value::Null
        }
    }

    fn coalesce_values(mut vs: Vec<Value>) -> Value {
        for v in vs.drain(..) {
            if !v.is_null() {
                return v;
            }
        }
        Value::Null
    }

    match expr {
        SqlExpr::Paren(inner) => eval_scalar_expr(inner, table, row_idx, ctx),
        SqlExpr::Literal(v) => Ok(v.clone()),
        SqlExpr::Column(c) => Ok(get_col_value(table, c, row_idx)),
        SqlExpr::UnaryOp { op: UnaryOperator::Minus, expr } => {
            let v = eval_scalar_expr(expr, table, row_idx, ctx)?;
            if let Some(i) = v.as_i64() {
                Ok(Value::Int64(-i))
            } else if let Some(f) = v.as_f64() {
                Ok(Value::Float64(-f))
            } else {
                Ok(Value::Null)
            }
        }
        SqlExpr::BinaryOp { left, op, right } => {
            let lv = eval_scalar_expr(left, table, row_idx, ctx)?;
            let rv = eval_scalar_expr(right, table, row_idx, ctx)?;
            match op {
                BinaryOperator::Add
                | BinaryOperator::Sub
                | BinaryOperator::Mul
                | BinaryOperator::Div
                | BinaryOperator::Mod => {
                    if let (Some(a), Some(b)) = (lv.as_f64(), rv.as_f64()) {
                        let out = match op {
                            BinaryOperator::Add => a + b,
                            BinaryOperator::Sub => a - b,
                            BinaryOperator::Mul => a * b,
                            BinaryOperator::Div => a / b,
                            BinaryOperator::Mod => a % b,
                            _ => a,
                        };
                        Ok(Value::Float64(out))
                    } else {
                        Ok(Value::Null)
                    }
                }
                _ => Ok(Value::Null),
            }
        }
        SqlExpr::Function { name, args } => {
            let uname = name.to_uppercase();

            match uname.as_str() {
                "RAND" => {
                    if !args.is_empty() {
                        return Err(ApexError::QueryParseError(
                            "RAND() does not accept arguments".to_string(),
                        ));
                    }
                    Ok(Value::Float64(rand::random::<f64>()))
                }
                "NOW" => {
                    if !args.is_empty() {
                        return Err(ApexError::QueryParseError(
                            "NOW() does not accept arguments".to_string(),
                        ));
                    }
                    Ok(Value::String(ctx.now_str.clone()))
                }
                "SQRT" => {
                    if args.len() != 1 {
                        return Err(ApexError::QueryParseError(
                            "SQRT() expects 1 argument".to_string(),
                        ));
                    }
                    let v = eval_scalar_expr(&args[0], table, row_idx, ctx)?;
                    if v.is_null() {
                        return Ok(Value::Null);
                    }
                    let x = v.as_f64().unwrap_or(f64::NAN);
                    if !x.is_finite() || x < 0.0 {
                        Ok(Value::Null)
                    } else {
                        Ok(Value::Float64(x.sqrt()))
                    }
                }
                "ROUND" => {
                    if args.is_empty() || args.len() > 2 {
                        return Err(ApexError::QueryParseError(
                            "ROUND() expects 1 or 2 arguments".to_string(),
                        ));
                    }
                    let v = eval_scalar_expr(&args[0], table, row_idx, ctx)?;
                    if v.is_null() {
                        return Ok(Value::Null);
                    }
                    let x = v.as_f64().unwrap_or(f64::NAN);
                    if !x.is_finite() {
                        return Ok(Value::Null);
                    }
                    let d: i32 = if args.len() == 2 {
                        let dv = eval_scalar_expr(&args[1], table, row_idx, ctx)?;
                        dv.as_i64().unwrap_or(0) as i32
                    } else {
                        0
                    };
                    let factor = 10f64.powi(d);
                    Ok(Value::Float64((x * factor).round() / factor))
                }
                "CONCAT" => {
                    if args.is_empty() {
                        return Err(ApexError::QueryParseError(
                            "CONCAT() expects at least 1 argument".to_string(),
                        ));
                    }
                    let mut any_non_null = false;
                    let mut out = String::new();
                    for a in args {
                        let v = eval_scalar_expr(a, table, row_idx, ctx)?;
                        if !v.is_null() {
                            any_non_null = true;
                            out.push_str(&v.to_string_value());
                        }
                    }
                    if any_non_null {
                        Ok(Value::String(out))
                    } else {
                        Ok(Value::Null)
                    }
                }
                "ISNULL" | "NVL" | "IFNULL" => {
                    if args.len() != 2 {
                        return Err(ApexError::QueryParseError(
                            format!("{}() expects 2 arguments", uname),
                        ));
                    }
                    let a = eval_scalar_expr(&args[0], table, row_idx, ctx)?;
                    if !a.is_null() {
                        Ok(a)
                    } else {
                        eval_scalar_expr(&args[1], table, row_idx, ctx)
                    }
                }
                "COALESCE" => {
                    if args.is_empty() {
                        return Err(ApexError::QueryParseError(
                            "COALESCE() expects at least 1 argument".to_string(),
                        ));
                    }
                    let mut vs = Vec::with_capacity(args.len());
                    for a in args {
                        vs.push(eval_scalar_expr(a, table, row_idx, ctx)?);
                    }
                    Ok(coalesce_values(vs))
                }
                "LEN" => {
                    if args.len() != 1 {
                        return Err(ApexError::QueryParseError(
                            "LEN() expects 1 argument".to_string(),
                        ));
                    }
                    let v = eval_scalar_expr(&args[0], table, row_idx, ctx)?;
                    if v.is_null() {
                        return Ok(Value::Null);
                    }
                    let s = v.to_string_value();
                    Ok(Value::Int64(s.chars().count() as i64))
                }
                "MID" => {
                    if args.len() != 2 && args.len() != 3 {
                        return Err(ApexError::QueryParseError(
                            "MID() expects 2 or 3 arguments".to_string(),
                        ));
                    }
                    let s0 = eval_scalar_expr(&args[0], table, row_idx, ctx)?;
                    let start0 = eval_scalar_expr(&args[1], table, row_idx, ctx)?;
                    if s0.is_null() || start0.is_null() {
                        return Ok(Value::Null);
                    }
                    let s = s0.to_string_value();
                    let mut start = start0.as_i64().unwrap_or(1);
                    if start < 1 {
                        start = 1;
                    }
                    let start_idx = (start - 1) as usize;
                    let chars: Vec<char> = s.chars().collect();
                    if start_idx >= chars.len() {
                        return Ok(Value::String(String::new()));
                    }

                    let end_idx = if args.len() == 3 {
                        let len0 = eval_scalar_expr(&args[2], table, row_idx, ctx)?;
                        if len0.is_null() {
                            return Ok(Value::Null);
                        }
                        let mut l = len0.as_i64().unwrap_or(0);
                        if l < 0 {
                            l = 0;
                        }
                        (start_idx + l as usize).min(chars.len())
                    } else {
                        chars.len()
                    };
                    let out: String = chars[start_idx..end_idx].iter().collect();
                    Ok(Value::String(out))
                }
                "REPLACE" => {
                    if args.len() != 3 {
                        return Err(ApexError::QueryParseError(
                            "REPLACE() expects 3 arguments".to_string(),
                        ));
                    }
                    let s0 = eval_scalar_expr(&args[0], table, row_idx, ctx)?;
                    let from0 = eval_scalar_expr(&args[1], table, row_idx, ctx)?;
                    let to0 = eval_scalar_expr(&args[2], table, row_idx, ctx)?;
                    if s0.is_null() || from0.is_null() || to0.is_null() {
                        return Ok(Value::Null);
                    }
                    let s = s0.to_string_value();
                    let from = from0.to_string_value();
                    let to = to0.to_string_value();
                    Ok(Value::String(s.replace(&from, &to)))
                }
                "TRIM" => {
                    if args.len() != 1 {
                        return Err(ApexError::QueryParseError(
                            "TRIM() expects 1 argument".to_string(),
                        ));
                    }
                    let v = eval_scalar_expr(&args[0], table, row_idx, ctx)?;
                    if v.is_null() {
                        return Ok(Value::Null);
                    }
                    Ok(Value::String(v.to_string_value().trim().to_string()))
                }
                "UPPER" => {
                    if args.len() != 1 {
                        return Err(ApexError::QueryParseError(
                            "UPPER() expects 1 argument".to_string(),
                        ));
                    }
                    let v = eval_scalar_expr(&args[0], table, row_idx, ctx)?;
                    if v.is_null() {
                        return Ok(Value::Null);
                    }
                    Ok(Value::String(v.to_string_value().to_uppercase()))
                }
                "LOWER" => {
                    if args.len() != 1 {
                        return Err(ApexError::QueryParseError(
                            "LOWER() expects 1 argument".to_string(),
                        ));
                    }
                    let v = eval_scalar_expr(&args[0], table, row_idx, ctx)?;
                    if v.is_null() {
                        return Ok(Value::Null);
                    }
                    Ok(Value::String(v.to_string_value().to_lowercase()))
                }
                _ => Err(ApexError::QueryParseError(format!(
                    "Unsupported function in SELECT list: {}",
                    name
                ))),
            }
        }
        _ => Err(ApexError::QueryParseError(
            "Unsupported expression in SELECT list".to_string(),
        )),
    }
}

pub(crate) fn build_arrow_direct(
    result_columns: &[String],
    column_indices: &[(String, Option<usize>)],
    projected_exprs: &[Option<SqlExpr>],
    matching_indices: &[usize],
    table: &ColumnTable,
    ctx: &EvalContext,
) -> Result<SqlResult, ApexError> {
    let columns = table.columns_ref();
    let num_rows = matching_indices.len();

    let mut fields = Vec::with_capacity(result_columns.len());
    let mut arrays: Vec<ArrayRef> = Vec::with_capacity(result_columns.len());

    for (pos, (col_name, col_idx)) in column_indices.iter().enumerate() {
        let expr = projected_exprs.get(pos).and_then(|e| e.clone());
        if col_name == "_id" {
            let id_values: Vec<i64> = matching_indices.iter().map(|&i| i as i64).collect();
            fields.push(Field::new("_id", ArrowDataType::Int64, false));
            arrays.push(Arc::new(Int64Array::from(id_values)));
        } else if let Some(expr) = expr {
            // Evaluate expression per row, infer output type.
            let mut values: Vec<Value> = Vec::with_capacity(num_rows);
            for &i in matching_indices {
                values.push(eval_scalar_expr(&expr, table, i, ctx)?);
            }

            let sample_kind = values.iter().find(|v| !v.is_null()).map(|v| v.data_type());

            match sample_kind {
                Some(crate::data::DataType::Bool) => {
                    let out: Vec<Option<bool>> = values
                        .into_iter()
                        .map(|v| if v.is_null() { None } else { Some(v.as_bool().unwrap_or(false)) })
                        .collect();
                    fields.push(Field::new(col_name, ArrowDataType::Boolean, true));
                    arrays.push(Arc::new(BooleanArray::from(out)));
                }
                Some(crate::data::DataType::Int8)
                | Some(crate::data::DataType::Int16)
                | Some(crate::data::DataType::Int32)
                | Some(crate::data::DataType::Int64)
                | Some(crate::data::DataType::UInt8)
                | Some(crate::data::DataType::UInt16)
                | Some(crate::data::DataType::UInt32)
                | Some(crate::data::DataType::UInt64)
                | Some(crate::data::DataType::Timestamp)
                | Some(crate::data::DataType::Date) => {
                    let out: Vec<Option<i64>> = values
                        .into_iter()
                        .map(|v| {
                            if v.is_null() {
                                None
                            } else if let Value::Timestamp(t) = v {
                                Some(t)
                            } else if let Value::Date(d) = v {
                                Some(d as i64)
                            } else {
                                v.as_i64()
                            }
                        })
                        .collect();
                    fields.push(Field::new(col_name, ArrowDataType::Int64, true));
                    arrays.push(Arc::new(Int64Array::from(out)));
                }
                Some(crate::data::DataType::Float32) | Some(crate::data::DataType::Float64) => {
                    let out: Vec<Option<f64>> = values
                        .into_iter()
                        .map(|v| if v.is_null() { None } else { v.as_f64() })
                        .collect();
                    fields.push(Field::new(col_name, ArrowDataType::Float64, true));
                    arrays.push(Arc::new(Float64Array::from(out)));
                }
                Some(crate::data::DataType::String)
                | Some(crate::data::DataType::Binary)
                | Some(crate::data::DataType::Json)
                | Some(crate::data::DataType::Array) => {
                    let mut builder = StringBuilder::with_capacity(num_rows, num_rows * 16);
                    for v in values {
                        if v.is_null() {
                            builder.append_null();
                        } else {
                            builder.append_value(v.to_string_value());
                        }
                    }
                    fields.push(Field::new(col_name, ArrowDataType::Utf8, true));
                    arrays.push(Arc::new(builder.finish()));
                }
                None => {
                    let null_values: Vec<Option<i64>> = vec![None; num_rows];
                    fields.push(Field::new(col_name, ArrowDataType::Int64, true));
                    arrays.push(Arc::new(Int64Array::from(null_values)));
                }
                Some(_other) => {
                    let mut builder = StringBuilder::with_capacity(num_rows, num_rows * 16);
                    for v in values {
                        if v.is_null() {
                            builder.append_null();
                        } else {
                            builder.append_value(v.to_string_value());
                        }
                    }
                    fields.push(Field::new(col_name, ArrowDataType::Utf8, true));
                    arrays.push(Arc::new(builder.finish()));
                }
            }
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
