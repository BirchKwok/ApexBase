use crate::query::engine::tables_plan::TablesPlan;
use crate::query::engine::ops::{aggregate_join_rows, build_join_rows, project_join_rows_plain, sort_join_rows, try_aggregate_join_rows_fast_path, JoinContext, JoinRow};
use crate::query::SqlResult;
use crate::table::ColumnTable;
use crate::ApexError;
use std::collections::HashMap;

pub(crate) struct PlanExecutorTables;

impl PlanExecutorTables {
    fn stmt_has_aggregates(stmt: &crate::query::sql_parser::SelectStatement) -> bool {
        use crate::query::sql_parser::SelectColumn;
        stmt.columns
            .iter()
            .any(|c| matches!(c, SelectColumn::Aggregate { .. }))
    }

    fn execute_join_rows_plan(
        tables: &mut HashMap<String, ColumnTable>,
        plan: &TablesPlan,
    ) -> Result<(JoinContext, Vec<JoinRow>), ApexError> {
        match plan {
            TablesPlan::JoinScan { stmt } => build_join_rows(stmt, tables),
            TablesPlan::Sort { input, order_by, .. } => {
                let (ctx, mut rows) = Self::execute_join_rows_plan(tables, input)?;
                sort_join_rows(&ctx, &mut rows, order_by, tables);
                Ok((ctx, rows))
            }
            TablesPlan::Limit { input, limit, offset } => {
                let (ctx, rows) = Self::execute_join_rows_plan(tables, input)?;
                let off = *offset;
                let lim = limit.unwrap_or(usize::MAX);
                let rows = rows.into_iter().skip(off).take(lim).collect::<Vec<_>>();
                Ok((ctx, rows))
            }
            TablesPlan::Project { input, .. } => Self::execute_join_rows_plan(tables, input),
        }
    }

    pub(crate) fn execute_tables_plan(
        tables: &mut HashMap<String, ColumnTable>,
        default_table: &str,
        plan: &TablesPlan,
    ) -> Result<SqlResult, ApexError> {
        match plan {
            // Scaffolding: until JoinScan/Sort/Limit/Aggregate/Project are executed natively,
            // fall back to the existing JOIN executor with the original statement.
            TablesPlan::Project { stmt, input } => {
                // Aggregate/group-by/having: now executed natively (still on JoinRow references).
                if Self::stmt_has_aggregates(stmt) || !stmt.group_by.is_empty() || stmt.having.is_some() {
                    if let Some(res) = try_aggregate_join_rows_fast_path(stmt, tables)? {
                        return Ok(res);
                    }
                    let (ctx, rows) = build_join_rows(stmt, tables)?;
                    return aggregate_join_rows(stmt, &ctx, &rows, tables);
                }

                let (ctx, rows) = Self::execute_join_rows_plan(tables, input)?;
                project_join_rows_plain(stmt, &ctx, &rows, tables)
            }
            TablesPlan::Sort { input, .. }
            | TablesPlan::Limit { input, .. } => Self::execute_tables_plan(tables, default_table, input),
            TablesPlan::JoinScan { .. } => Err(ApexError::QueryParseError(
                "multi-table plan must start at Project".to_string(),
            )),
        }
    }
}
