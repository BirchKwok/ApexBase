//! Query executor - parses and executes queries

use super::filter::Filter;
use crate::{ApexError, Result};
use crate::query::sql_expr_to_filter;
use crate::query::sql_parser::SqlParser;

/// Query executor
pub struct QueryExecutor;

impl QueryExecutor {
    /// Create a new query executor
    pub fn new() -> Self {
        Self
    }

    /// Parse a WHERE clause into a Filter
    pub fn parse(&self, where_clause: &str) -> Result<Filter> {
        let where_clause = where_clause.trim();

        // Handle empty or always true
        if where_clause.is_empty() || where_clause == "1=1" {
            return Ok(Filter::True);
        }

        // Unified SQL semantics: parse as SQL expression and compile to Filter
        let expr = SqlParser::parse_expression(where_clause)?;
        sql_expr_to_filter(&expr).map_err(|e| ApexError::QueryParseError(e.to_string()))
    }
}

impl Default for QueryExecutor {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::Row;
    use crate::data::Value;

    fn make_row(id: u64, name: &str, age: i64, city: &str) -> Row {
        let mut row = Row::new(id);
        row.set("name", name);
        row.set("age", Value::Int64(age));
        row.set("city", city);
        row
    }

    #[test]
    fn test_parse_simple_comparison() {
        let executor = QueryExecutor::new();

        let filter = executor.parse("age > 25").unwrap();
        let row = make_row(1, "John", 30, "NYC");
        assert!(filter.matches(&row));

        let row = make_row(2, "Jane", 20, "LA");
        assert!(!filter.matches(&row));
    }

    #[test]
    fn test_parse_string_comparison() {
        let executor = QueryExecutor::new();

        let filter = executor.parse("name = 'John'").unwrap();
        let row = make_row(1, "John", 30, "NYC");
        assert!(filter.matches(&row));
    }

    #[test]
    fn test_parse_and_condition() {
        let executor = QueryExecutor::new();

        let filter = executor.parse("age > 25 AND city = 'NYC'").unwrap();

        let row1 = make_row(1, "John", 30, "NYC");
        assert!(filter.matches(&row1));

        let row2 = make_row(2, "Jane", 30, "LA");
        assert!(!filter.matches(&row2));
    }

    #[test]
    fn test_parse_like() {
        let executor = QueryExecutor::new();

        let filter = executor.parse("name LIKE 'J%'").unwrap();

        let row1 = make_row(1, "John", 30, "NYC");
        assert!(filter.matches(&row1));

        let row2 = make_row(2, "Bob", 25, "LA");
        assert!(!filter.matches(&row2));
    }

    #[test]
    fn test_parse_in() {
        let executor = QueryExecutor::new();

        let filter = executor.parse("age IN (25, 30, 35)").unwrap();

        let row1 = make_row(1, "John", 30, "NYC");
        assert!(filter.matches(&row1));

        let row2 = make_row(2, "Jane", 28, "LA");
        assert!(!filter.matches(&row2));
    }

    #[test]
    fn test_always_true() {
        let executor = QueryExecutor::new();

        let filter = executor.parse("1=1").unwrap();
        let row = make_row(1, "John", 30, "NYC");
        assert!(filter.matches(&row));

        let filter = executor.parse("").unwrap();
        assert!(filter.matches(&row));
    }
}

