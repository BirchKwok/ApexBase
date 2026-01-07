use crate::io_engine::IoEngine;
use crate::query::engine::logical_plan::LogicalPlan;
use crate::query::sql_expr_to_filter;
use crate::query::SqlExecutor;
use crate::query::SqlResult;
use crate::table::ColumnTable;
use crate::ApexError;

pub(crate) struct PlanExecutor;

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
                prefer_arrow,
            } => {
                let indices = Self::execute_indices(table, input);

                if *prefer_arrow && indices.len() > 10_000 {
                    return crate::query::engine::ops::build_arrow_direct(result_columns, column_indices, &indices, table);
                }

                let schema = table.schema_ref();
                let cols = table.columns_ref();
                let mut rows: Vec<Vec<crate::data::Value>> = Vec::with_capacity(indices.len());
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
