use crate::query::engine::logical_plan::LogicalPlan;
use crate::query::engine::TablesPlan;
use crate::query::sql_expr_to_filter;
use crate::query::sql_parser::{SelectColumn, SelectStatement, SqlStatement, UnionStatement};
use crate::table::ColumnTable;
use crate::ApexError;

pub(crate) struct PlanBuilder;

impl PlanBuilder {
    pub(crate) fn build_plan(stmt: &SqlStatement, table: &ColumnTable) -> Result<LogicalPlan, ApexError> {
        match stmt {
            SqlStatement::Select(select) => PlanBuilder::build_select_plan(select, table),
            SqlStatement::Union(u) => PlanBuilder::build_union_plan(u, table),
            SqlStatement::CreateView { .. } | SqlStatement::DropView { .. } => Err(ApexError::QueryParseError(
                "CREATE/DROP VIEW must be handled before planning".to_string(),
            )),
        }
    }

    pub(crate) fn build_union_plan(union: &UnionStatement, table: &ColumnTable) -> Result<LogicalPlan, ApexError> {
        fn build_one(stmt: &SqlStatement, table: &ColumnTable) -> Result<LogicalPlan, ApexError> {
            match stmt {
                SqlStatement::Select(sel) => {
                    let has_aggs = sel.columns.iter().any(|c| matches!(c, SelectColumn::Aggregate { .. }));
                    if !sel.group_by.is_empty() || has_aggs {
                        // GROUP BY (and aggregate) path already evaluates HAVING/ORDER/LIMIT internally.
                        PlanBuilder::build_group_by_plan(sel)
                    } else {
                        PlanBuilder::build_simple_select_plan(sel, table)
                    }
                }
                SqlStatement::Union(u) => PlanBuilder::build_union_plan(u, table),
                SqlStatement::CreateView { .. } | SqlStatement::DropView { .. } => Err(ApexError::QueryParseError(
                    "CREATE/DROP VIEW must be handled before planning".to_string(),
                )),
            }
        }

        let left = build_one(&union.left, table)?;
        let right = build_one(&union.right, table)?;

        Ok(LogicalPlan::Union {
            left: Box::new(left),
            right: Box::new(right),
            all: union.all,
            order_by: union.order_by.clone(),
            limit: union.limit,
            offset: union.offset,
        })
    }

    pub(crate) fn build_select_plan(stmt: &SelectStatement, table: &ColumnTable) -> Result<LogicalPlan, ApexError> {
        if !stmt.joins.is_empty() {
            return Err(ApexError::QueryParseError(
                "select plan does not support JOIN".to_string(),
            ));
        }
        if !stmt.group_by.is_empty() {
            return Err(ApexError::QueryParseError(
                "select plan does not support GROUP BY".to_string(),
            ));
        }
        if stmt.having.is_some() {
            return Err(ApexError::QueryParseError(
                "select plan does not support HAVING".to_string(),
            ));
        }

        let filter = match stmt.where_clause.as_ref() {
            Some(expr) => Some(sql_expr_to_filter(expr)?),
            None => None,
        };

        let mut plan = LogicalPlan::Scan {
            filter,
            limit: None,
            offset: 0,
        };

        // ORDER BY handled via Sort + optional top-k hint.
        if !stmt.order_by.is_empty() {
            let k = stmt
                .limit
                .map(|lim| stmt.offset.unwrap_or(0).saturating_add(lim));
            plan = LogicalPlan::Sort {
                input: Box::new(plan),
                order_by: stmt.order_by.clone(),
                k,
            };
        }

        // LIMIT/OFFSET pushed down as indices trimming.
        if stmt.limit.is_some() || stmt.offset.is_some() {
            plan = LogicalPlan::Limit {
                input: Box::new(plan),
                limit: stmt.limit,
                offset: stmt.offset.unwrap_or(0),
            };
        }

        let (result_columns, column_indices, projected_exprs) =
            crate::query::engine::ops::resolve_columns(&stmt.columns, table)?;

        // DISTINCT + LIMIT (no ORDER BY) early stop.
        if stmt.distinct && stmt.order_by.is_empty() && stmt.limit.is_some() {
            return Ok(LogicalPlan::DistinctLimit {
                input: Box::new(plan),
                limit: stmt.limit.unwrap_or(usize::MAX),
                offset: stmt.offset.unwrap_or(0),
                result_columns,
                column_indices,
            });
        }

        plan = LogicalPlan::Project {
            input: Box::new(plan),
            result_columns,
            column_indices,
            projected_exprs,
            prefer_arrow: !stmt.distinct,
        };

        if stmt.distinct {
            plan = LogicalPlan::Distinct {
                input: Box::new(plan),
            };
        }

        Ok(plan)
    }

    pub(crate) fn build_simple_select_plan(
        stmt: &SelectStatement,
        table: &ColumnTable,
    ) -> Result<LogicalPlan, ApexError> {
        if !crate::query::engine::is_simple_select_eligible(stmt) {
            return Err(ApexError::QueryParseError(
                "not eligible for simple select plan".to_string(),
            ));
        }

        let filter = match stmt.where_clause.as_ref() {
            Some(expr) => Some(sql_expr_to_filter(expr)?),
            None => None,
        };

        let mut plan = LogicalPlan::Scan {
            filter,
            limit: None,
            offset: 0,
        };

        if !stmt.order_by.is_empty() {
            let k = stmt
                .limit
                .map(|lim| stmt.offset.unwrap_or(0).saturating_add(lim));
            plan = LogicalPlan::Sort {
                input: Box::new(plan),
                order_by: stmt.order_by.clone(),
                k,
            };
        }

        if stmt.limit.is_some() || stmt.offset.is_some() {
            plan = LogicalPlan::Limit {
                input: Box::new(plan),
                limit: stmt.limit,
                offset: stmt.offset.unwrap_or(0),
            };
        }

        let (result_columns, column_indices, projected_exprs) =
            crate::query::engine::ops::resolve_columns(&stmt.columns, table)?;
        plan = LogicalPlan::Project {
            input: Box::new(plan),
            result_columns,
            column_indices,
            projected_exprs,
            // DISTINCT is implemented on rows for now.
            prefer_arrow: !stmt.distinct,
        };

        if stmt.distinct {
            plan = LogicalPlan::Distinct {
                input: Box::new(plan),
            };
        }

        Ok(plan)
    }

    pub(crate) fn build_group_by_plan(stmt: &SelectStatement) -> Result<LogicalPlan, ApexError> {
        if stmt.joins.iter().len() != 0 {
            return Err(ApexError::QueryParseError(
                "GROUP BY plan does not support JOIN yet".to_string(),
            ));
        }

        let filter = match stmt.where_clause.as_ref() {
            Some(expr) => Some(sql_expr_to_filter(expr)?),
            None => None,
        };

        let scan = LogicalPlan::Scan {
            filter,
            // LIMIT/OFFSET should be applied after aggregation.
            limit: None,
            offset: 0,
        };

        Ok(LogicalPlan::Aggregate {
            input: Box::new(scan),
            stmt: stmt.clone(),
        })
    }

    pub(crate) fn build_join_tables_plan(stmt: &SelectStatement) -> Result<TablesPlan, ApexError> {
        if stmt.joins.is_empty() {
            return Err(ApexError::QueryParseError(
                "not a JOIN select".to_string(),
            ));
        }

        // Build a composable multi-table plan pipeline.
        // NOTE: the executor will initially fall back to the legacy join executor at the outer
        // Project boundary to keep behavior/perf stable. Subsequent refactors will progressively
        // execute these nodes natively.
        let mut plan = TablesPlan::JoinScan { stmt: stmt.clone() };

        // ORDER BY as a separate node (with optional top-k hint).
        if !stmt.order_by.is_empty() {
            plan = TablesPlan::Sort {
                input: Box::new(plan),
                order_by: stmt.order_by.clone(),
            };
        }

        // LIMIT/OFFSET.
        if stmt.limit.is_some() || stmt.offset.is_some() {
            plan = TablesPlan::Limit {
                input: Box::new(plan),
                limit: stmt.limit,
                offset: stmt.offset.unwrap_or(0),
            };
        }

        // Projection boundary (current fallback point).
        Ok(TablesPlan::Project {
            input: Box::new(plan),
            stmt: stmt.clone(),
        })
    }
}
