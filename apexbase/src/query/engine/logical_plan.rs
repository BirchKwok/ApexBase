use crate::query::Filter;
use crate::query::sql_parser::OrderByClause;
use crate::query::sql_parser::SelectStatement;

#[derive(Debug, Clone)]
pub(crate) enum LogicalPlan {
    Scan {
        filter: Option<Filter>,
        limit: Option<usize>,
        offset: usize,
    },

    Sort {
        input: Box<LogicalPlan>,
        order_by: Vec<OrderByClause>,
        k: Option<usize>,
    },

    Limit {
        input: Box<LogicalPlan>,
        limit: Option<usize>,
        offset: usize,
    },

    Project {
        input: Box<LogicalPlan>,
        result_columns: Vec<String>,
        column_indices: Vec<(String, Option<usize>)>,
        prefer_arrow: bool,
    },

    Distinct {
        input: Box<LogicalPlan>,
    },

    DistinctLimit {
        input: Box<LogicalPlan>,
        limit: usize,
        offset: usize,
        result_columns: Vec<String>,
        column_indices: Vec<(String, Option<usize>)>,
    },

    Aggregate {
        input: Box<LogicalPlan>,
        stmt: SelectStatement,
    },

    AggregateDirect {
        stmt: SelectStatement,
    },

    WindowRowNumber {
        stmt: SelectStatement,
    },

    Union {
        left: Box<LogicalPlan>,
        right: Box<LogicalPlan>,
        all: bool,
        order_by: Vec<OrderByClause>,
        limit: Option<usize>,
        offset: Option<usize>,
    },

    Count {
        stmt: SelectStatement,
    },
}
