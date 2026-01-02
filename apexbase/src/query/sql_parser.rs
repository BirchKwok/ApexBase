//! SQL:2023 Parser for ApexBase
//! 
//! Supports standard SQL SELECT statements with:
//! - SELECT columns or SELECT *
//! - FROM table
//! - WHERE conditions (with LIKE, IN, AND, OR, NOT, comparison operators)
//! - ORDER BY column [ASC|DESC]
//! - LIMIT n [OFFSET m]
//! - DISTINCT
//! - Column aliases (AS)
//! - Aggregate functions (COUNT, SUM, AVG, MIN, MAX)
//! - GROUP BY / HAVING

use crate::ApexError;
use crate::data::Value;

/// SQL Statement types
#[derive(Debug, Clone)]
pub enum SqlStatement {
    Select(SelectStatement),
}

/// SELECT statement structure
#[derive(Debug, Clone)]
pub struct SelectStatement {
    pub distinct: bool,
    pub columns: Vec<SelectColumn>,
    pub from: Option<String>,
    pub where_clause: Option<SqlExpr>,
    pub group_by: Vec<String>,
    pub having: Option<SqlExpr>,
    pub order_by: Vec<OrderByClause>,
    pub limit: Option<usize>,
    pub offset: Option<usize>,
}

/// Column selection in SELECT clause
#[derive(Debug, Clone)]
pub enum SelectColumn {
    /// SELECT *
    All,
    /// SELECT column_name
    Column(String),
    /// SELECT column_name AS alias
    ColumnAlias { column: String, alias: String },
    /// SELECT COUNT(*), SUM(col), etc.
    Aggregate { func: AggregateFunc, column: Option<String>, alias: Option<String> },
    /// SELECT expression AS alias
    Expression { expr: SqlExpr, alias: Option<String> },
}

/// Aggregate functions
#[derive(Debug, Clone, PartialEq)]
pub enum AggregateFunc {
    Count,
    Sum,
    Avg,
    Min,
    Max,
}

/// ORDER BY clause
#[derive(Debug, Clone)]
pub struct OrderByClause {
    pub column: String,
    pub descending: bool,
    pub nulls_first: Option<bool>,  // SQL:2023 NULLS FIRST/LAST
}

/// SQL Expression (for WHERE, HAVING, etc.)
#[derive(Debug, Clone)]
pub enum SqlExpr {
    /// Column reference
    Column(String),
    /// Literal value
    Literal(Value),
    /// Binary operation: expr op expr
    BinaryOp { left: Box<SqlExpr>, op: BinaryOperator, right: Box<SqlExpr> },
    /// Unary operation: NOT expr
    UnaryOp { op: UnaryOperator, expr: Box<SqlExpr> },
    /// LIKE pattern matching
    Like { column: String, pattern: String, negated: bool },
    /// IN list: column IN (v1, v2, ...)
    In { column: String, values: Vec<Value>, negated: bool },
    /// BETWEEN: column BETWEEN low AND high
    Between { column: String, low: Box<SqlExpr>, high: Box<SqlExpr>, negated: bool },
    /// IS NULL / IS NOT NULL
    IsNull { column: String, negated: bool },
    /// Function call
    Function { name: String, args: Vec<SqlExpr> },
    /// Parenthesized expression
    Paren(Box<SqlExpr>),
}

/// Binary operators
#[derive(Debug, Clone, PartialEq)]
pub enum BinaryOperator {
    // Comparison
    Eq,         // =
    NotEq,      // != or <>
    Lt,         // <
    Le,         // <=
    Gt,         // >
    Ge,         // >=
    // Logical
    And,
    Or,
    // Arithmetic (for expressions)
    Add,
    Sub,
    Mul,
    Div,
    Mod,
}

/// Unary operators
#[derive(Debug, Clone, PartialEq)]
pub enum UnaryOperator {
    Not,
    Minus,
}

/// SQL Parser
pub struct SqlParser {
    tokens: Vec<Token>,
    pos: usize,
}

/// Token types for SQL lexer
#[derive(Debug, Clone, PartialEq)]
enum Token {
    // Keywords
    Select, From, Where, And, Or, Not, As, Distinct,
    Order, By, Asc, Desc, Limit, Offset, Nulls, First, Last,
    Like, In, Between, Is, Null,
    Group, Having,
    Count, Sum, Avg, Min, Max,
    True, False,
    // Symbols
    Star,           // *
    Comma,          // ,
    Dot,            // .
    LParen,         // (
    RParen,         // )
    Eq,             // =
    NotEq,          // != or <>
    Lt,             // <
    Le,             // <=
    Gt,             // >
    Ge,             // >=
    Plus,           // +
    Minus,          // -
    Slash,          // /
    Percent,        // %
    // Literals
    Identifier(String),
    StringLit(String),
    IntLit(i64),
    FloatLit(f64),
    // End
    Eof,
}

impl SqlParser {
    /// Parse a SQL statement
    pub fn parse(sql: &str) -> Result<SqlStatement, ApexError> {
        let tokens = Self::tokenize(sql)?;
        let mut parser = SqlParser { tokens, pos: 0 };
        parser.parse_statement()
    }

    /// Tokenize SQL string
    fn tokenize(sql: &str) -> Result<Vec<Token>, ApexError> {
        let mut tokens = Vec::new();
        let chars: Vec<char> = sql.chars().collect();
        let len = chars.len();
        let mut i = 0;

        while i < len {
            let c = chars[i];

            // Skip whitespace
            if c.is_whitespace() {
                i += 1;
                continue;
            }

            // Single character tokens
            match c {
                '*' => { tokens.push(Token::Star); i += 1; continue; }
                ',' => { tokens.push(Token::Comma); i += 1; continue; }
                '.' => { tokens.push(Token::Dot); i += 1; continue; }
                '(' => { tokens.push(Token::LParen); i += 1; continue; }
                ')' => { tokens.push(Token::RParen); i += 1; continue; }
                '+' => { tokens.push(Token::Plus); i += 1; continue; }
                '-' => { tokens.push(Token::Minus); i += 1; continue; }
                '/' => { tokens.push(Token::Slash); i += 1; continue; }
                '%' => { tokens.push(Token::Percent); i += 1; continue; }
                _ => {}
            }

            // Multi-character operators
            if c == '=' {
                tokens.push(Token::Eq);
                i += 1;
                continue;
            }
            if c == '!' && i + 1 < len && chars[i + 1] == '=' {
                tokens.push(Token::NotEq);
                i += 2;
                continue;
            }
            if c == '<' {
                if i + 1 < len && chars[i + 1] == '=' {
                    tokens.push(Token::Le);
                    i += 2;
                } else if i + 1 < len && chars[i + 1] == '>' {
                    tokens.push(Token::NotEq);
                    i += 2;
                } else {
                    tokens.push(Token::Lt);
                    i += 1;
                }
                continue;
            }
            if c == '>' {
                if i + 1 < len && chars[i + 1] == '=' {
                    tokens.push(Token::Ge);
                    i += 2;
                } else {
                    tokens.push(Token::Gt);
                    i += 1;
                }
                continue;
            }

            // String literals
            if c == '\'' || c == '"' {
                let quote = c;
                i += 1;
                let start = i;
                while i < len && chars[i] != quote {
                    if chars[i] == '\\' && i + 1 < len {
                        i += 2; // Skip escaped char
                    } else {
                        i += 1;
                    }
                }
                let s: String = chars[start..i].iter().collect();
                tokens.push(Token::StringLit(s));
                if i < len { i += 1; } // Skip closing quote
                continue;
            }

            // Numbers
            if c.is_ascii_digit() || (c == '.' && i + 1 < len && chars[i + 1].is_ascii_digit()) {
                let start = i;
                let mut has_dot = c == '.';
                i += 1;
                while i < len && (chars[i].is_ascii_digit() || (!has_dot && chars[i] == '.')) {
                    if chars[i] == '.' { has_dot = true; }
                    i += 1;
                }
                let num_str: String = chars[start..i].iter().collect();
                if has_dot {
                    let f: f64 = num_str.parse().map_err(|_| 
                        ApexError::QueryParseError(format!("Invalid number: {}", num_str)))?;
                    tokens.push(Token::FloatLit(f));
                } else {
                    let n: i64 = num_str.parse().map_err(|_| 
                        ApexError::QueryParseError(format!("Invalid number: {}", num_str)))?;
                    tokens.push(Token::IntLit(n));
                }
                continue;
            }

            // Identifiers and keywords
            if c.is_alphabetic() || c == '_' {
                let start = i;
                while i < len && (chars[i].is_alphanumeric() || chars[i] == '_') {
                    i += 1;
                }
                let word: String = chars[start..i].iter().collect();
                let upper = word.to_uppercase();
                let token = match upper.as_str() {
                    "SELECT" => Token::Select,
                    "FROM" => Token::From,
                    "WHERE" => Token::Where,
                    "AND" => Token::And,
                    "OR" => Token::Or,
                    "NOT" => Token::Not,
                    "AS" => Token::As,
                    "DISTINCT" => Token::Distinct,
                    "ORDER" => Token::Order,
                    "BY" => Token::By,
                    "ASC" => Token::Asc,
                    "DESC" => Token::Desc,
                    "LIMIT" => Token::Limit,
                    "OFFSET" => Token::Offset,
                    "NULLS" => Token::Nulls,
                    "FIRST" => Token::First,
                    "LAST" => Token::Last,
                    "LIKE" => Token::Like,
                    "IN" => Token::In,
                    "BETWEEN" => Token::Between,
                    "IS" => Token::Is,
                    "NULL" => Token::Null,
                    "GROUP" => Token::Group,
                    "HAVING" => Token::Having,
                    "COUNT" => Token::Count,
                    "SUM" => Token::Sum,
                    "AVG" => Token::Avg,
                    "MIN" => Token::Min,
                    "MAX" => Token::Max,
                    "TRUE" => Token::True,
                    "FALSE" => Token::False,
                    _ => Token::Identifier(word),
                };
                tokens.push(token);
                continue;
            }

            return Err(ApexError::QueryParseError(format!("Unexpected character: {}", c)));
        }

        tokens.push(Token::Eof);
        Ok(tokens)
    }

    fn current(&self) -> &Token {
        &self.tokens[self.pos]
    }

    fn advance(&mut self) -> &Token {
        let tok = &self.tokens[self.pos];
        if self.pos < self.tokens.len() - 1 {
            self.pos += 1;
        }
        tok
    }

    fn expect(&mut self, expected: Token) -> Result<(), ApexError> {
        if std::mem::discriminant(self.current()) == std::mem::discriminant(&expected) {
            self.advance();
            Ok(())
        } else {
            Err(ApexError::QueryParseError(format!(
                "Expected {:?}, got {:?}", expected, self.current()
            )))
        }
    }

    fn parse_statement(&mut self) -> Result<SqlStatement, ApexError> {
        match self.current() {
            Token::Select => self.parse_select().map(SqlStatement::Select),
            _ => Err(ApexError::QueryParseError("Expected SELECT statement".to_string())),
        }
    }

    fn parse_select(&mut self) -> Result<SelectStatement, ApexError> {
        self.expect(Token::Select)?;

        // DISTINCT
        let distinct = if matches!(self.current(), Token::Distinct) {
            self.advance();
            true
        } else {
            false
        };

        // Columns
        let columns = self.parse_select_columns()?;

        // FROM (optional for simple queries)
        let from = if matches!(self.current(), Token::From) {
            self.advance();
            if let Token::Identifier(name) = self.current().clone() {
                self.advance();
                Some(name)
            } else {
                return Err(ApexError::QueryParseError("Expected table name after FROM".to_string()));
            }
        } else {
            None
        };

        // WHERE
        let where_clause = if matches!(self.current(), Token::Where) {
            self.advance();
            Some(self.parse_expr()?)
        } else {
            None
        };

        // GROUP BY
        let group_by = if matches!(self.current(), Token::Group) {
            self.advance();
            self.expect(Token::By)?;
            self.parse_column_list()?
        } else {
            Vec::new()
        };

        // HAVING
        let having = if matches!(self.current(), Token::Having) {
            self.advance();
            Some(self.parse_expr()?)
        } else {
            None
        };

        // ORDER BY
        let order_by = if matches!(self.current(), Token::Order) {
            self.advance();
            self.expect(Token::By)?;
            self.parse_order_by()?
        } else {
            Vec::new()
        };

        // LIMIT
        let limit = if matches!(self.current(), Token::Limit) {
            self.advance();
            if let Token::IntLit(n) = self.current().clone() {
                self.advance();
                Some(n as usize)
            } else {
                return Err(ApexError::QueryParseError("Expected number after LIMIT".to_string()));
            }
        } else {
            None
        };

        // OFFSET
        let offset = if matches!(self.current(), Token::Offset) {
            self.advance();
            if let Token::IntLit(n) = self.current().clone() {
                self.advance();
                Some(n as usize)
            } else {
                return Err(ApexError::QueryParseError("Expected number after OFFSET".to_string()));
            }
        } else {
            None
        };

        Ok(SelectStatement {
            distinct,
            columns,
            from,
            where_clause,
            group_by,
            having,
            order_by,
            limit,
            offset,
        })
    }

    fn parse_select_columns(&mut self) -> Result<Vec<SelectColumn>, ApexError> {
        let mut columns = Vec::new();

        loop {
            // Check for *
            if matches!(self.current(), Token::Star) {
                self.advance();
                columns.push(SelectColumn::All);
            }
            // Check for aggregate functions
            else if matches!(self.current(), Token::Count | Token::Sum | Token::Avg | Token::Min | Token::Max) {
                let func = match self.current() {
                    Token::Count => AggregateFunc::Count,
                    Token::Sum => AggregateFunc::Sum,
                    Token::Avg => AggregateFunc::Avg,
                    Token::Min => AggregateFunc::Min,
                    Token::Max => AggregateFunc::Max,
                    _ => unreachable!(),
                };
                self.advance();
                self.expect(Token::LParen)?;
                
                let column = if matches!(self.current(), Token::Star) {
                    self.advance();
                    None  // COUNT(*)
                } else if let Token::Identifier(name) = self.current().clone() {
                    self.advance();
                    Some(name)
                } else {
                    None
                };
                
                self.expect(Token::RParen)?;
                
                // Optional alias
                let alias = if matches!(self.current(), Token::As) {
                    self.advance();
                    if let Token::Identifier(name) = self.current().clone() {
                        self.advance();
                        Some(name)
                    } else {
                        None
                    }
                } else {
                    None
                };
                
                columns.push(SelectColumn::Aggregate { func, column, alias });
            }
            // Column name
            else if let Token::Identifier(name) = self.current().clone() {
                self.advance();
                
                // Check for alias
                if matches!(self.current(), Token::As) {
                    self.advance();
                    if let Token::Identifier(alias) = self.current().clone() {
                        self.advance();
                        columns.push(SelectColumn::ColumnAlias { column: name, alias });
                    } else {
                        columns.push(SelectColumn::Column(name));
                    }
                } else {
                    columns.push(SelectColumn::Column(name));
                }
            }
            else {
                break;
            }

            // Check for comma
            if matches!(self.current(), Token::Comma) {
                self.advance();
            } else {
                break;
            }
        }

        if columns.is_empty() {
            return Err(ApexError::QueryParseError("Expected column list after SELECT".to_string()));
        }

        Ok(columns)
    }

    fn parse_column_list(&mut self) -> Result<Vec<String>, ApexError> {
        let mut columns = Vec::new();
        
        loop {
            if let Token::Identifier(name) = self.current().clone() {
                self.advance();
                columns.push(name);
            } else {
                break;
            }
            
            if matches!(self.current(), Token::Comma) {
                self.advance();
            } else {
                break;
            }
        }
        
        Ok(columns)
    }

    fn parse_order_by(&mut self) -> Result<Vec<OrderByClause>, ApexError> {
        let mut clauses = Vec::new();

        loop {
            if let Token::Identifier(column) = self.current().clone() {
                self.advance();
                
                let descending = if matches!(self.current(), Token::Desc) {
                    self.advance();
                    true
                } else if matches!(self.current(), Token::Asc) {
                    self.advance();
                    false
                } else {
                    false
                };
                
                // SQL:2023 NULLS FIRST/LAST
                let nulls_first = if matches!(self.current(), Token::Nulls) {
                    self.advance();
                    if matches!(self.current(), Token::First) {
                        self.advance();
                        Some(true)
                    } else if matches!(self.current(), Token::Last) {
                        self.advance();
                        Some(false)
                    } else {
                        None
                    }
                } else {
                    None
                };
                
                clauses.push(OrderByClause { column, descending, nulls_first });
            } else {
                break;
            }

            if matches!(self.current(), Token::Comma) {
                self.advance();
            } else {
                break;
            }
        }

        Ok(clauses)
    }

    fn parse_expr(&mut self) -> Result<SqlExpr, ApexError> {
        self.parse_or_expr()
    }

    fn parse_or_expr(&mut self) -> Result<SqlExpr, ApexError> {
        let mut left = self.parse_and_expr()?;

        while matches!(self.current(), Token::Or) {
            self.advance();
            let right = self.parse_and_expr()?;
            left = SqlExpr::BinaryOp {
                left: Box::new(left),
                op: BinaryOperator::Or,
                right: Box::new(right),
            };
        }

        Ok(left)
    }

    fn parse_and_expr(&mut self) -> Result<SqlExpr, ApexError> {
        let mut left = self.parse_not_expr()?;

        while matches!(self.current(), Token::And) {
            self.advance();
            let right = self.parse_not_expr()?;
            left = SqlExpr::BinaryOp {
                left: Box::new(left),
                op: BinaryOperator::And,
                right: Box::new(right),
            };
        }

        Ok(left)
    }

    fn parse_not_expr(&mut self) -> Result<SqlExpr, ApexError> {
        if matches!(self.current(), Token::Not) {
            self.advance();
            let expr = self.parse_not_expr()?;
            Ok(SqlExpr::UnaryOp {
                op: UnaryOperator::Not,
                expr: Box::new(expr),
            })
        } else {
            self.parse_comparison()
        }
    }

    fn parse_comparison(&mut self) -> Result<SqlExpr, ApexError> {
        let left = self.parse_primary()?;

        // Check for comparison operators
        match self.current() {
            Token::Eq => {
                self.advance();
                let right = self.parse_primary()?;
                Ok(SqlExpr::BinaryOp {
                    left: Box::new(left),
                    op: BinaryOperator::Eq,
                    right: Box::new(right),
                })
            }
            Token::NotEq => {
                self.advance();
                let right = self.parse_primary()?;
                Ok(SqlExpr::BinaryOp {
                    left: Box::new(left),
                    op: BinaryOperator::NotEq,
                    right: Box::new(right),
                })
            }
            Token::Lt => {
                self.advance();
                let right = self.parse_primary()?;
                Ok(SqlExpr::BinaryOp {
                    left: Box::new(left),
                    op: BinaryOperator::Lt,
                    right: Box::new(right),
                })
            }
            Token::Le => {
                self.advance();
                let right = self.parse_primary()?;
                Ok(SqlExpr::BinaryOp {
                    left: Box::new(left),
                    op: BinaryOperator::Le,
                    right: Box::new(right),
                })
            }
            Token::Gt => {
                self.advance();
                let right = self.parse_primary()?;
                Ok(SqlExpr::BinaryOp {
                    left: Box::new(left),
                    op: BinaryOperator::Gt,
                    right: Box::new(right),
                })
            }
            Token::Ge => {
                self.advance();
                let right = self.parse_primary()?;
                Ok(SqlExpr::BinaryOp {
                    left: Box::new(left),
                    op: BinaryOperator::Ge,
                    right: Box::new(right),
                })
            }
            Token::Like | Token::Not => {
                // Handle LIKE and NOT LIKE
                let negated = if matches!(self.current(), Token::Not) {
                    self.advance();
                    true
                } else {
                    false
                };
                
                if matches!(self.current(), Token::Like) {
                    self.advance();
                    let pattern = if let Token::StringLit(s) = self.current().clone() {
                        self.advance();
                        s
                    } else {
                        return Err(ApexError::QueryParseError("Expected pattern after LIKE".to_string()));
                    };
                    
                    let column = match left {
                        SqlExpr::Column(name) => name,
                        _ => return Err(ApexError::QueryParseError("LIKE requires column name".to_string())),
                    };
                    
                    Ok(SqlExpr::Like { column, pattern, negated })
                } else if matches!(self.current(), Token::In) {
                    // NOT IN
                    self.advance();
                    let (column, values) = self.parse_in_list(&left)?;
                    Ok(SqlExpr::In { column, values, negated })
                } else if matches!(self.current(), Token::Between) {
                    // NOT BETWEEN
                    self.advance();
                    let (column, low, high) = self.parse_between(&left)?;
                    Ok(SqlExpr::Between { column, low, high, negated })
                } else {
                    Ok(left)
                }
            }
            Token::In => {
                self.advance();
                let (column, values) = self.parse_in_list(&left)?;
                Ok(SqlExpr::In { column, values, negated: false })
            }
            Token::Between => {
                self.advance();
                let (column, low, high) = self.parse_between(&left)?;
                Ok(SqlExpr::Between { column, low, high, negated: false })
            }
            Token::Is => {
                self.advance();
                let negated = if matches!(self.current(), Token::Not) {
                    self.advance();
                    true
                } else {
                    false
                };
                self.expect(Token::Null)?;
                let column = match left {
                    SqlExpr::Column(name) => name,
                    _ => return Err(ApexError::QueryParseError("IS NULL requires column name".to_string())),
                };
                Ok(SqlExpr::IsNull { column, negated })
            }
            _ => Ok(left),
        }
    }

    fn parse_in_list(&mut self, left: &SqlExpr) -> Result<(String, Vec<Value>), ApexError> {
        let column = match left {
            SqlExpr::Column(name) => name.clone(),
            _ => return Err(ApexError::QueryParseError("IN requires column name".to_string())),
        };
        
        self.expect(Token::LParen)?;
        let mut values = Vec::new();
        
        loop {
            let val = match self.current().clone() {
                Token::StringLit(s) => { self.advance(); Value::String(s) }
                Token::IntLit(n) => { self.advance(); Value::Int64(n) }
                Token::FloatLit(f) => { self.advance(); Value::Float64(f) }
                Token::True => { self.advance(); Value::Bool(true) }
                Token::False => { self.advance(); Value::Bool(false) }
                Token::Null => { self.advance(); Value::Null }
                _ => break,
            };
            values.push(val);
            
            if matches!(self.current(), Token::Comma) {
                self.advance();
            } else {
                break;
            }
        }
        
        self.expect(Token::RParen)?;
        Ok((column, values))
    }

    fn parse_between(&mut self, left: &SqlExpr) -> Result<(String, Box<SqlExpr>, Box<SqlExpr>), ApexError> {
        let column = match left {
            SqlExpr::Column(name) => name.clone(),
            _ => return Err(ApexError::QueryParseError("BETWEEN requires column name".to_string())),
        };
        
        let low = self.parse_primary()?;
        self.expect(Token::And)?;
        let high = self.parse_primary()?;
        
        Ok((column, Box::new(low), Box::new(high)))
    }

    fn parse_primary(&mut self) -> Result<SqlExpr, ApexError> {
        match self.current().clone() {
            Token::LParen => {
                self.advance();
                let expr = self.parse_expr()?;
                self.expect(Token::RParen)?;
                Ok(SqlExpr::Paren(Box::new(expr)))
            }
            Token::StringLit(s) => {
                self.advance();
                Ok(SqlExpr::Literal(Value::String(s)))
            }
            Token::IntLit(n) => {
                self.advance();
                Ok(SqlExpr::Literal(Value::Int64(n)))
            }
            Token::FloatLit(f) => {
                self.advance();
                Ok(SqlExpr::Literal(Value::Float64(f)))
            }
            Token::True => {
                self.advance();
                Ok(SqlExpr::Literal(Value::Bool(true)))
            }
            Token::False => {
                self.advance();
                Ok(SqlExpr::Literal(Value::Bool(false)))
            }
            Token::Null => {
                self.advance();
                Ok(SqlExpr::Literal(Value::Null))
            }
            Token::Identifier(name) => {
                self.advance();
                Ok(SqlExpr::Column(name))
            }
            Token::Minus => {
                self.advance();
                let expr = self.parse_primary()?;
                Ok(SqlExpr::UnaryOp {
                    op: UnaryOperator::Minus,
                    expr: Box::new(expr),
                })
            }
            _ => Err(ApexError::QueryParseError(format!(
                "Unexpected token in expression: {:?}", self.current()
            ))),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simple_select() {
        let sql = "SELECT * FROM users";
        let stmt = SqlParser::parse(sql).unwrap();
        if let SqlStatement::Select(s) = stmt {
            assert!(!s.distinct);
            assert_eq!(s.columns.len(), 1);
            assert!(matches!(s.columns[0], SelectColumn::All));
            assert_eq!(s.from, Some("users".to_string()));
        }
    }

    #[test]
    fn test_select_with_where() {
        let sql = "SELECT name, age FROM users WHERE age > 18 AND name LIKE 'John%'";
        let stmt = SqlParser::parse(sql).unwrap();
        if let SqlStatement::Select(s) = stmt {
            assert_eq!(s.columns.len(), 2);
            assert!(s.where_clause.is_some());
        }
    }

    #[test]
    fn test_select_with_order_limit() {
        let sql = "SELECT * FROM users ORDER BY age DESC LIMIT 10 OFFSET 5";
        let stmt = SqlParser::parse(sql).unwrap();
        if let SqlStatement::Select(s) = stmt {
            assert_eq!(s.order_by.len(), 1);
            assert!(s.order_by[0].descending);
            assert_eq!(s.limit, Some(10));
            assert_eq!(s.offset, Some(5));
        }
    }
}
