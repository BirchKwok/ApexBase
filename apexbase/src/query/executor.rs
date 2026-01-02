//! Query executor - parses and executes queries

use super::filter::{CompareOp, Filter};
use crate::data::Value;
use crate::{ApexError, Result};

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

        // Handle AND/OR at top level
        if let Some(filter) = self.try_parse_logical(where_clause)? {
            return Ok(filter);
        }

        // Parse single condition
        self.parse_condition(where_clause)
    }

    /// Try to parse logical operators (AND/OR)
    fn try_parse_logical(&self, clause: &str) -> Result<Option<Filter>> {
        // Split by AND (case insensitive)
        let and_parts: Vec<&str> = self.split_by_and(clause);
        if and_parts.len() > 1 {
            let filters: Result<Vec<Filter>> = and_parts
                .iter()
                .map(|part| self.parse_condition(part.trim()))
                .collect();
            return Ok(Some(Filter::And(filters?)));
        }

        // Split by OR (case insensitive)
        let or_parts: Vec<&str> = self.split_by_or(clause);
        if or_parts.len() > 1 {
            let filters: Result<Vec<Filter>> = or_parts
                .iter()
                .map(|part| self.parse_condition(part.trim()))
                .collect();
            return Ok(Some(Filter::Or(filters?)));
        }

        Ok(None)
    }

    /// Split by AND (respecting quotes and parentheses)
    fn split_by_and<'a>(&self, s: &'a str) -> Vec<&'a str> {
        self.split_by_keyword(s, " AND ")
    }

    /// Split by OR (respecting quotes and parentheses)
    fn split_by_or<'a>(&self, s: &'a str) -> Vec<&'a str> {
        self.split_by_keyword(s, " OR ")
    }

    /// Split by keyword (case insensitive)
    fn split_by_keyword<'a>(&self, s: &'a str, keyword: &str) -> Vec<&'a str> {
        let upper = s.to_uppercase();
        let mut parts = Vec::new();
        let mut last = 0;
        let mut in_quote = false;
        let mut paren_depth = 0;

        let keyword_upper = keyword.to_uppercase();
        
        for (i, c) in s.char_indices() {
            match c {
                '\'' | '"' => in_quote = !in_quote,
                '(' if !in_quote => paren_depth += 1,
                ')' if !in_quote => paren_depth -= 1,
                _ => {}
            }

            if !in_quote && paren_depth == 0 {
                if upper[i..].starts_with(&keyword_upper) {
                    parts.push(&s[last..i]);
                    last = i + keyword.len();
                }
            }
        }

        if last < s.len() {
            parts.push(&s[last..]);
        }

        if parts.is_empty() {
            vec![s]
        } else {
            parts
        }
    }

    /// Parse a single condition
    fn parse_condition(&self, cond: &str) -> Result<Filter> {
        let cond = cond.trim();

        // Handle LIKE
        if let Some(filter) = self.try_parse_like(cond)? {
            return Ok(filter);
        }

        // Handle IN
        if let Some(filter) = self.try_parse_in(cond)? {
            return Ok(filter);
        }

        // Handle comparison operators
        self.parse_comparison(cond)
    }

    /// Try to parse LIKE condition
    fn try_parse_like(&self, cond: &str) -> Result<Option<Filter>> {
        let upper = cond.to_uppercase();
        if let Some(pos) = upper.find(" LIKE ") {
            let field = cond[..pos].trim().to_string();
            let pattern = cond[pos + 6..].trim();
            let pattern = self.unquote(pattern);
            return Ok(Some(Filter::Like { field, pattern }));
        }
        Ok(None)
    }

    /// Try to parse IN condition
    fn try_parse_in(&self, cond: &str) -> Result<Option<Filter>> {
        let upper = cond.to_uppercase();
        if let Some(pos) = upper.find(" IN ") {
            let field = cond[..pos].trim().to_string();
            let values_str = cond[pos + 4..].trim();
            
            // Parse values list: (val1, val2, ...)
            if values_str.starts_with('(') && values_str.ends_with(')') {
                let inner = &values_str[1..values_str.len() - 1];
                let values: Vec<Value> = inner
                    .split(',')
                    .map(|v| self.parse_value(v.trim()))
                    .collect();
                return Ok(Some(Filter::In { field, values }));
            }
        }
        Ok(None)
    }

    /// Parse comparison condition
    fn parse_comparison(&self, cond: &str) -> Result<Filter> {
        // Try different operators in order of specificity
        let operators = [
            (">=", CompareOp::GreaterEqual),
            ("<=", CompareOp::LessEqual),
            ("!=", CompareOp::NotEqual),
            ("<>", CompareOp::NotEqual),
            (">", CompareOp::GreaterThan),
            ("<", CompareOp::LessThan),
            ("=", CompareOp::Equal),
        ];

        for (op_str, op) in operators {
            if let Some(pos) = cond.find(op_str) {
                let field = cond[..pos].trim().to_string();
                let value_str = cond[pos + op_str.len()..].trim();
                let value = self.parse_value(value_str);

                return Ok(Filter::Compare { field, op, value });
            }
        }

        Err(ApexError::QueryParseError(format!(
            "Invalid condition: {}",
            cond
        )))
    }

    /// Parse a value from string
    fn parse_value(&self, s: &str) -> Value {
        let s = s.trim();

        // Handle quoted strings
        if (s.starts_with('\'') && s.ends_with('\''))
            || (s.starts_with('"') && s.ends_with('"'))
        {
            return Value::String(self.unquote(s));
        }

        // Handle boolean
        let upper = s.to_uppercase();
        if upper == "TRUE" {
            return Value::Bool(true);
        }
        if upper == "FALSE" {
            return Value::Bool(false);
        }
        if upper == "NULL" {
            return Value::Null;
        }

        // Handle numbers
        if let Ok(i) = s.parse::<i64>() {
            return Value::Int64(i);
        }
        if let Ok(f) = s.parse::<f64>() {
            return Value::Float64(f);
        }

        // Default to string
        Value::String(s.to_string())
    }

    /// Remove quotes from a string
    fn unquote(&self, s: &str) -> String {
        let s = s.trim();
        if (s.starts_with('\'') && s.ends_with('\''))
            || (s.starts_with('"') && s.ends_with('"'))
        {
            s[1..s.len() - 1].to_string()
        } else {
            s.to_string()
        }
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

