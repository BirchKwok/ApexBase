use crate::query::sql_parser::OrderByClause;
use crate::query::sql_parser::SelectStatement;

#[derive(Debug, Clone)]
pub(crate) enum TablesPlan {
    JoinScan {
        stmt: SelectStatement,
    },

    Sort {
        input: Box<TablesPlan>,
        order_by: Vec<OrderByClause>,
    },

    Limit {
        input: Box<TablesPlan>,
        limit: Option<usize>,
        offset: usize,
    },

    Project {
        input: Box<TablesPlan>,
        stmt: SelectStatement,
    },
}
