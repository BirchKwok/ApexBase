use crate::query::engine::logical_plan::LogicalPlan;
use crate::query::sql_parser::{SelectColumn, SelectStatement};
use crate::query::AggregateFunc;
use crate::table::ColumnTable;
use crate::ApexError;

pub(crate) struct PlanOptimizer;

impl PlanOptimizer {
    pub(crate) fn try_build_plan(stmt: &SelectStatement, table: &ColumnTable) -> Result<Option<LogicalPlan>, ApexError> {
        let has_aggregates = stmt
            .columns
            .iter()
            .any(|c| matches!(c, SelectColumn::Aggregate { .. }));
        let has_window = stmt
            .columns
            .iter()
            .any(|c| matches!(c, SelectColumn::WindowFunction { .. }));
        let no_where = stmt.where_clause.is_none();
        let no_group_by = stmt.group_by.is_empty();

        if has_window {
            // Only support row_number() OVER (...) for now, keeping the existing semantics.
            // Constraints match the legacy implementation.
            if stmt.joins.is_empty()
                && !stmt.distinct
                && stmt.having.is_none()
                && no_group_by
                && !has_aggregates
                && stmt.columns.iter().all(|c| match c {
                    SelectColumn::Column(_) => true,
                    SelectColumn::ColumnAlias { .. } => true,
                    SelectColumn::WindowFunction { name, .. } => name.eq_ignore_ascii_case("row_number"),
                    _ => false,
                })
            {
                return Ok(Some(LogicalPlan::WindowRowNumber { stmt: stmt.clone() }));
            }

            return Ok(None);
        }

        // COUNT(*) / COUNT(constant) without GROUP BY => dedicated plan node.
        if has_aggregates && no_group_by && stmt.columns.len() == 1 {
            if let SelectColumn::Aggregate {
                func: AggregateFunc::Count,
                distinct,
                ..
            } = &stmt.columns[0]
            {
                if !*distinct {
                    return Ok(Some(LogicalPlan::Count { stmt: stmt.clone() }));
                }
            }
        }

        // Scalar aggregates without WHERE/GROUP BY: keep existing direct aggregate behavior.
        if has_aggregates && no_group_by && no_where {
            return Ok(Some(LogicalPlan::AggregateDirect { stmt: stmt.clone() }));
        }

        // For single-table non-aggregate queries, go through unified plan.
        if !has_aggregates && no_group_by {
            let plan = crate::query::engine::PlanBuilder::build_select_plan(stmt, table)?;
            return Ok(Some(plan));
        }

        // Keep other fast paths as execution-time optimizations for now.
        // They can be moved into dedicated plan nodes later.
        Ok(None)
    }
}
