mod executor;
mod logical_plan;
mod optimizer;
pub(crate) mod ops;
mod plan_builder;
mod simple_select;
mod tables_plan;
mod tables_executor;

pub(crate) use executor::PlanExecutor;
pub(crate) use optimizer::PlanOptimizer;
use logical_plan::LogicalPlan;
pub(crate) use plan_builder::PlanBuilder;
pub(crate) use simple_select::is_simple_select_eligible;
pub(crate) use tables_plan::TablesPlan;
pub(crate) use tables_executor::PlanExecutorTables;

use crate::query::sql_expr_to_filter;
use crate::query::sql_parser::SqlParser;
use crate::table::ColumnTable;
use crate::ApexError;

pub fn plan_query_indices(
    table: &ColumnTable,
    where_clause: &str,
    limit: Option<usize>,
    offset: usize,
) -> Result<Vec<usize>, ApexError> {
    let where_clause = where_clause.trim();

    let filter = if where_clause.is_empty() || where_clause == "1=1" {
        None
    } else {
        let expr = SqlParser::parse_expression(where_clause)?;
        Some(sql_expr_to_filter(&expr)?)
    };

    let plan = LogicalPlan::Scan { filter, limit, offset };
    Ok(PlanExecutor::execute_indices(table, &plan))
}
