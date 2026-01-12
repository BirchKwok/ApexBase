use super::*;
use super::*;
use crate::query::engine::{PlanBuilder, PlanExecutor};

pub(super) fn is_eligible(stmt: &SelectStatement) -> bool {
    crate::query::engine::is_simple_select_eligible(stmt)
}

pub(super) fn execute(stmt: &SelectStatement, table: &ColumnTable) -> Result<SqlResult, ApexError> {
    let plan = PlanBuilder::build_simple_select_plan(stmt, table)?;
    PlanExecutor::execute_sql_result(table, &plan)
}
