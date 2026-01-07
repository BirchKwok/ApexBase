use crate::io_engine::IoEngine;
use crate::io_engine::StreamingFilterEvaluator;
use crate::query::engine::logical_plan::LogicalPlan;
use crate::query::sql_expr_to_filter;
use crate::query::SqlExecutor;
use crate::query::SqlResult;
use crate::table::ColumnTable;
use crate::ApexError;
use arrow::array::{
    ArrayRef, BooleanBuilder, Float64Builder, Int64Builder, StringBuilder,
};
use arrow::datatypes::{DataType as ArrowDataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use std::sync::Arc;

pub(crate) struct PlanExecutor;

enum ColBuilder {
    Int64(Int64Builder),
    Float64(Float64Builder),
    Boolean(BooleanBuilder),
    Utf8(StringBuilder),
}

impl ColBuilder {
    fn finish(self) -> ArrayRef {
        match self {
            ColBuilder::Int64(mut b) => Arc::new(b.finish()) as ArrayRef,
            ColBuilder::Float64(mut b) => Arc::new(b.finish()) as ArrayRef,
            ColBuilder::Boolean(mut b) => Arc::new(b.finish()) as ArrayRef,
            ColBuilder::Utf8(mut b) => Arc::new(b.finish()) as ArrayRef,
        }
    }
}

impl PlanExecutor {
    fn try_execute_project_scan_streaming_arrow(
        table: &ColumnTable,
        scan_filter: &Option<crate::query::Filter>,
        scan_limit: Option<usize>,
        scan_offset: usize,
        result_columns: &[String],
        column_indices: &[(String, Option<usize>)],
        projected_exprs: &[Option<crate::query::sql_parser::SqlExpr>],
    ) -> Result<Option<SqlResult>, ApexError> {
        // Only support pure column projection (no expressions) in streaming path.
        if projected_exprs.iter().any(|e| e.is_some()) {
            return Ok(None);
        }

        let schema = table.schema_ref();
        let columns = table.columns_ref();
        let deleted = table.deleted_ref();
        let row_count = table.get_row_count();
        let no_deletes = deleted.all_false();

        let max_results = scan_limit.unwrap_or(usize::MAX);

        // Prepare Arrow schema + builders.
        let mut fields: Vec<Field> = Vec::with_capacity(result_columns.len());
        let mut builders: Vec<ColBuilder> = Vec::with_capacity(result_columns.len());

        // Conservative initial capacity to reduce reallocations but avoid over-reserving.
        let cap = max_results.min(8192).max(1024);

        for (col_name, col_idx) in column_indices {
            if col_name == "_id" {
                fields.push(Field::new("_id", ArrowDataType::Int64, false));
                builders.push(ColBuilder::Int64(Int64Builder::with_capacity(cap)));
                continue;
            }

            let idx = if let Some(i) = col_idx {
                *i
            } else if let Some(i) = schema.get_index(col_name) {
                i
            } else {
                // Unknown column -> return NULL int64 column (keeps behavior consistent with legacy)
                fields.push(Field::new(col_name, ArrowDataType::Int64, true));
                builders.push(ColBuilder::Int64(Int64Builder::with_capacity(cap)));
                continue;
            };

            match &columns[idx] {
                crate::table::column_table::TypedColumn::Int64 { .. }
                | crate::table::column_table::TypedColumn::Mixed { .. } => {
                    // Mixed will be stringified.
                    if matches!(
                        &columns[idx],
                        crate::table::column_table::TypedColumn::Int64 { .. }
                    ) {
                        fields.push(Field::new(col_name, ArrowDataType::Int64, true));
                        builders.push(ColBuilder::Int64(Int64Builder::with_capacity(cap)));
                    } else {
                        fields.push(Field::new(col_name, ArrowDataType::Utf8, true));
                        builders.push(ColBuilder::Utf8(StringBuilder::with_capacity(cap, cap * 16)));
                    }
                }
                crate::table::column_table::TypedColumn::Float64 { .. } => {
                    fields.push(Field::new(col_name, ArrowDataType::Float64, true));
                    builders.push(ColBuilder::Float64(Float64Builder::with_capacity(cap)));
                }
                crate::table::column_table::TypedColumn::Bool { .. } => {
                    fields.push(Field::new(col_name, ArrowDataType::Boolean, true));
                    builders.push(ColBuilder::Boolean(BooleanBuilder::with_capacity(cap)));
                }
                crate::table::column_table::TypedColumn::String(_) => {
                    fields.push(Field::new(col_name, ArrowDataType::Utf8, true));
                    builders.push(ColBuilder::Utf8(StringBuilder::with_capacity(cap, cap * 16)));
                }
            }
        }

        let evaluator = scan_filter
            .as_ref()
            .map(|f| StreamingFilterEvaluator::new(f, schema, columns));

        let mut seen_non_deleted = 0usize;
        let mut seen_matches = 0usize;
        let mut produced = 0usize;

        for row_idx in 0..row_count {
            if !no_deletes && deleted.get(row_idx) {
                continue;
            }

            // Apply filter if present.
            if let Some(ev) = evaluator.as_ref() {
                if !ev.matches(row_idx) {
                    continue;
                }

                seen_matches += 1;
                if seen_matches <= scan_offset {
                    continue;
                }
            } else {
                // No filter: offset/limit are applied on non-deleted rows.
                seen_non_deleted += 1;
                if seen_non_deleted <= scan_offset {
                    continue;
                }
            }

            // Emit row.
            for (pos, (col_name, col_idx)) in column_indices.iter().enumerate() {
                match &mut builders[pos] {
                    ColBuilder::Int64(b) => {
                        if col_name == "_id" {
                            b.append_value(row_idx as i64);
                            continue;
                        }
                        let idx = if let Some(i) = col_idx {
                            *i
                        } else if let Some(i) = schema.get_index(col_name) {
                            i
                        } else {
                            b.append_null();
                            continue;
                        };
                        match &columns[idx] {
                            crate::table::column_table::TypedColumn::Int64 { data, nulls } => {
                                if row_idx < data.len() && !nulls.get(row_idx) {
                                    b.append_value(data[row_idx]);
                                } else {
                                    b.append_null();
                                }
                            }
                            _ => {
                                // Unknown/unexpected -> NULL
                                b.append_null();
                            }
                        }
                    }
                    ColBuilder::Float64(b) => {
                        let idx = if let Some(i) = col_idx {
                            *i
                        } else if let Some(i) = schema.get_index(col_name) {
                            i
                        } else {
                            b.append_null();
                            continue;
                        };
                        match &columns[idx] {
                            crate::table::column_table::TypedColumn::Float64 { data, nulls } => {
                                if row_idx < data.len() && !nulls.get(row_idx) {
                                    b.append_value(data[row_idx]);
                                } else {
                                    b.append_null();
                                }
                            }
                            _ => b.append_null(),
                        }
                    }
                    ColBuilder::Boolean(b) => {
                        let idx = if let Some(i) = col_idx {
                            *i
                        } else if let Some(i) = schema.get_index(col_name) {
                            i
                        } else {
                            b.append_null();
                            continue;
                        };
                        match &columns[idx] {
                            crate::table::column_table::TypedColumn::Bool { data, nulls } => {
                                if row_idx < data.len() && !nulls.get(row_idx) {
                                    b.append_value(data.get(row_idx));
                                } else {
                                    b.append_null();
                                }
                            }
                            _ => b.append_null(),
                        }
                    }
                    ColBuilder::Utf8(b) => {
                        let idx = if let Some(i) = col_idx {
                            *i
                        } else if let Some(i) = schema.get_index(col_name) {
                            i
                        } else {
                            b.append_null();
                            continue;
                        };
                        match &columns[idx] {
                            crate::table::column_table::TypedColumn::String(col) => {
                                if let Some(s) = col.get(row_idx) {
                                    b.append_value(s);
                                } else {
                                    b.append_null();
                                }
                            }
                            crate::table::column_table::TypedColumn::Mixed { data, nulls } => {
                                if row_idx < data.len() && !nulls.get(row_idx) {
                                    b.append_value(data[row_idx].to_string_value());
                                } else {
                                    b.append_null();
                                }
                            }
                            _ => {
                                // Fallback: use generic Value get to string.
                                match columns[idx].get(row_idx) {
                                    Some(v) if !v.is_null() => b.append_value(v.to_string_value()),
                                    _ => b.append_null(),
                                }
                            }
                        }
                    }
                }
            }

            produced += 1;
            if produced >= max_results {
                break;
            }
        }

        let schema = Arc::new(Schema::new(fields));
        let arrays = builders.into_iter().map(|b| b.finish()).collect::<Vec<_>>();
        let batch = RecordBatch::try_new(schema, arrays)
            .map_err(|e| ApexError::SerializationError(e.to_string()))?;

        Ok(Some(SqlResult::with_arrow_batch(result_columns.to_vec(), batch)))
    }
}

impl PlanExecutor {
    pub(crate) fn execute_indices(table: &ColumnTable, plan: &LogicalPlan) -> Vec<usize> {
        match plan {
            LogicalPlan::Scan {
                filter,
                limit,
                offset,
            } => match filter {
                Some(f) => IoEngine::read_filtered_indices(table, f, *limit, *offset),
                None => IoEngine::read_all_indices(table, *limit, *offset),
            },

            LogicalPlan::Sort { input, order_by, k } => {
                let indices = Self::execute_indices(table, input);
                if indices.is_empty() || order_by.is_empty() {
                    return indices;
                }
                let kk = k.unwrap_or(indices.len()).min(indices.len());
                crate::query::engine::ops::sort_indices_by_columns_topk(&indices, order_by, table, kk)
                    .unwrap_or(indices)
            }

            LogicalPlan::Limit { input, limit, offset } => {
                let indices = Self::execute_indices(table, input);
                let off = *offset;
                let lim = limit.unwrap_or(usize::MAX);
                indices.into_iter().skip(off).take(lim).collect()
            }

            // These nodes materialize rows/arrow, so indices-only execution is not applicable.
            LogicalPlan::Project { .. }
            | LogicalPlan::Distinct { .. }
            | LogicalPlan::DistinctLimit { .. }
            | LogicalPlan::Aggregate { .. }
            | LogicalPlan::AggregateDirect { .. }
            | LogicalPlan::WindowRowNumber { .. }
            | LogicalPlan::Union { .. }
            | LogicalPlan::Count { .. } => Vec::new(),
        }
    }

    pub(crate) fn execute_sql_result(table: &ColumnTable, plan: &LogicalPlan) -> Result<SqlResult, ApexError> {
        match plan {
            LogicalPlan::Project {
                input,
                result_columns,
                column_indices,
                projected_exprs,
                prefer_arrow,
            } => {
                // Streaming Arrow fast path: Project over Scan (no indices Vec materialization).
                if *prefer_arrow {
                    if let LogicalPlan::Scan { filter, limit, offset } = input.as_ref() {
                        if let Some(res) = Self::try_execute_project_scan_streaming_arrow(
                            table,
                            filter,
                            *limit,
                            *offset,
                            result_columns,
                            column_indices,
                            projected_exprs,
                        )? {
                            return Ok(res);
                        }
                    }
                }

                let indices = Self::execute_indices(table, input);

                let ctx = crate::query::engine::ops::new_eval_context();

                if *prefer_arrow {
                    return crate::query::engine::ops::build_arrow_direct(
                        result_columns,
                        column_indices,
                        projected_exprs,
                        &indices,
                        table,
                        &ctx,
                    );
                }

                let schema = table.schema_ref();
                let cols = table.columns_ref();
                let mut rows: Vec<Vec<crate::data::Value>> = Vec::with_capacity(indices.len());
                for row_idx in indices {
                    let mut row = Vec::with_capacity(column_indices.len());
                    for (pos, (col_name, col_idx)) in column_indices.iter().enumerate() {
                        if let Some(expr) = projected_exprs.get(pos).and_then(|e| e.clone()) {
                            row.push(crate::query::engine::ops::eval_scalar_expr(
                                &expr,
                                table,
                                row_idx,
                                &ctx,
                            )?);
                            continue;
                        }
                        if col_name == "_id" {
                            row.push(crate::data::Value::Int64(row_idx as i64));
                        } else if let Some(idx) = col_idx {
                            row.push(cols[*idx].get(row_idx).unwrap_or(crate::data::Value::Null));
                        } else if let Some(idx) = schema.get_index(col_name) {
                            row.push(cols[idx].get(row_idx).unwrap_or(crate::data::Value::Null));
                        } else {
                            row.push(crate::data::Value::Null);
                        }
                    }
                    rows.push(row);
                }

                Ok(SqlResult::new(result_columns.clone(), rows))
            }

            LogicalPlan::Distinct { input } => {
                let mut res = Self::execute_sql_result(table, input)?;
                if res.arrow_batch.is_some() {
                    // For now DISTINCT is implemented for row-based results.
                    // PlanBuilder should set prefer_arrow=false when DISTINCT is requested.
                    return Err(ApexError::QueryParseError(
                        "DISTINCT is not supported on Arrow results yet".to_string(),
                    ));
                }
                res.rows = crate::query::engine::ops::apply_distinct(res.rows);
                res.rows_affected = res.rows.len();
                Ok(res)
            }

            LogicalPlan::DistinctLimit {
                input,
                limit,
                offset,
                result_columns,
                column_indices,
            } => {
                let need = offset.saturating_add(*limit);
                let indices = Self::execute_indices(table, input);

                let schema = table.schema_ref();
                let cols = table.columns_ref();

                let mut seen: std::collections::HashSet<String> = std::collections::HashSet::new();
                let mut rows: Vec<Vec<crate::data::Value>> = Vec::with_capacity((*limit).min(10000));

                for row_idx in indices {
                    let mut row = Vec::with_capacity(column_indices.len());
                    for (col_name, col_idx) in column_indices {
                        if col_name == "_id" {
                            row.push(crate::data::Value::Int64(row_idx as i64));
                        } else if let Some(idx) = col_idx {
                            row.push(cols[*idx].get(row_idx).unwrap_or(crate::data::Value::Null));
                        } else if let Some(idx) = schema.get_index(col_name) {
                            row.push(cols[idx].get(row_idx).unwrap_or(crate::data::Value::Null));
                        } else {
                            row.push(crate::data::Value::Null);
                        }
                    }

                    let key = format!("{:?}", row);
                    if !seen.insert(key) {
                        continue;
                    }

                    rows.push(row);
                    if seen.len() >= need {
                        break;
                    }
                }

                let rows = rows.into_iter().skip(*offset).take(*limit).collect::<Vec<_>>();
                Ok(SqlResult::new(result_columns.clone(), rows))
            }

            LogicalPlan::Aggregate { input, stmt } => {
                let matching_indices = Self::execute_indices(table, input);
                SqlExecutor::execute_group_by(stmt, &matching_indices, table)
            }

            LogicalPlan::AggregateDirect { stmt } => SqlExecutor::execute_aggregate_direct(stmt, table),

            LogicalPlan::WindowRowNumber { stmt } => {
                let matching_indices: Vec<usize> = if let Some(ref where_expr) = stmt.where_clause {
                    let filter = sql_expr_to_filter(where_expr)?;
                    IoEngine::read_filtered_indices(table, &filter, None, 0)
                } else {
                    IoEngine::read_all_indices(table, None, 0)
                };
                SqlExecutor::execute_window_row_number(stmt, &matching_indices, table)
            }

            LogicalPlan::Union {
                left,
                right,
                all,
                order_by,
                limit,
                offset,
            } => {
                let left = Self::execute_sql_result(table, left)?;
                let right = Self::execute_sql_result(table, right)?;

                // For now UNION is implemented on row-based results.
                // If a child produced only Arrow, we would need to materialize.
                if left.rows.is_empty() && left.arrow_batch.is_some() {
                    return Err(ApexError::QueryParseError(
                        "UNION is not supported on Arrow-only results yet".to_string(),
                    ));
                }
                if right.rows.is_empty() && right.arrow_batch.is_some() {
                    return Err(ApexError::QueryParseError(
                        "UNION is not supported on Arrow-only results yet".to_string(),
                    ));
                }

                SqlExecutor::merge_union_results(left, right, *all, order_by, *limit, *offset)
            }

            LogicalPlan::Count { stmt } => {
                // COUNT fast paths: keep semantics identical by reusing SqlExecutor helpers.
                // IMPORTANT: try_fast_count_star ignores WHERE, so only use it when WHERE is absent.
                if stmt.where_clause.is_none() {
                    if let Some(res) = SqlExecutor::try_fast_count_star(stmt, table) {
                        return Ok(res);
                    }
                }

                if stmt.columns.len() != 1 {
                    return Err(ApexError::QueryParseError(
                        "COUNT plan expects a single COUNT() expression".to_string(),
                    ));
                }

                let (count_target, out_name) = match &stmt.columns[0] {
                    crate::query::sql_parser::SelectColumn::Aggregate { column, alias, .. } => {
                        let out = alias.clone().unwrap_or_else(|| {
                            column
                                .as_ref()
                                .map(|c| format!("COUNT({})", c))
                                .unwrap_or_else(|| "COUNT(*)".to_string())
                        });
                        (column.clone(), out)
                    }
                    _ => (None, "COUNT(*)".to_string()),
                };

                match stmt.where_clause.as_ref() {
                    None => {
                        // No WHERE: delegate to direct aggregate implementation for consistent semantics.
                        SqlExecutor::execute_aggregate_direct(stmt, table)
                    }
                    Some(where_expr) => {
                        // COUNT(*) / COUNT(constant)
                        if SqlExecutor::is_count_star_like(&count_target) {
                            let count = SqlExecutor::count_matching_rows(where_expr, table)?;
                            return Ok(SqlResult::new(
                                vec![out_name],
                                vec![vec![crate::data::Value::Int64(count as i64)]],
                            ));
                        }

                        // COUNT(column) with WHERE: count non-null values among matching rows.
                        let filter = sql_expr_to_filter(where_expr)?;
                        let matching = IoEngine::read_filtered_indices(table, &filter, None, 0);

                        let schema = table.schema_ref();
                        let cols = table.columns_ref();
                        let deleted = table.deleted_ref();

                        let col_name = count_target.unwrap_or_else(|| "".to_string());
                        let ci = schema.get_index(&col_name).ok_or_else(|| {
                            ApexError::QueryParseError(format!("Unknown column '{}' in COUNT()", col_name))
                        })?;

                        let mut cnt: usize = 0;
                        for row_idx in matching {
                            if deleted.get(row_idx) {
                                continue;
                            }
                            if let Some(v) = cols[ci].get(row_idx) {
                                if !v.is_null() {
                                    cnt += 1;
                                }
                            }
                        }

                        Ok(SqlResult::new(
                            vec![out_name],
                            vec![vec![crate::data::Value::Int64(cnt as i64)]],
                        ))
                    }
                }
            }

            // For scan-only plans
            LogicalPlan::Scan { .. } | LogicalPlan::Sort { .. } | LogicalPlan::Limit { .. } => {
                let indices = Self::execute_indices(table, plan);
                Ok(SqlResult::new(Vec::new(), indices.into_iter().map(|_| Vec::new()).collect()))
            }
        }
    }
}
