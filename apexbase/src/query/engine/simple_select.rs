use crate::query::sql_parser::{SelectColumn, SelectStatement};

pub(crate) fn is_simple_select_eligible(stmt: &SelectStatement) -> bool {
    if !stmt.joins.is_empty() {
        return false;
    }
    if !stmt.group_by.is_empty() {
        return false;
    }
    if stmt.having.is_some() {
        return false;
    }

    for c in &stmt.columns {
        match c {
            SelectColumn::All | SelectColumn::Column(_) | SelectColumn::ColumnAlias { .. } => {}
            _ => return false,
        }
    }

    true
}
