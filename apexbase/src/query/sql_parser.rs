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
    /// SELECT row_number() OVER (PARTITION BY ... ORDER BY ...) AS alias
    WindowFunction {
        name: String,
        partition_by: Vec<String>,
        order_by: Vec<OrderByClause>,
        alias: Option<String>,
    },
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
    /// REGEXP pattern matching
    Regexp { column: String, pattern: String, negated: bool },
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
    sql_chars: Vec<char>,
    tokens: Vec<SpannedToken>,
    pos: usize,
}

#[derive(Debug, Clone, PartialEq)]
struct SpannedToken {
    token: Token,
    start: usize,
    end: usize,
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
    Regexp,
    Over,
    Partition,
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
        let mut parser = SqlParser {
            sql_chars: sql.chars().collect(),
            tokens,
            pos: 0,
        };
        parser.parse_statement()
    }

    /// Tokenize SQL string
    fn tokenize(sql: &str) -> Result<Vec<SpannedToken>, ApexError> {
        let mut tokens: Vec<SpannedToken> = Vec::new();
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
                '*' => { tokens.push(SpannedToken { token: Token::Star, start: i, end: i + 1 }); i += 1; continue; }
                ',' => { tokens.push(SpannedToken { token: Token::Comma, start: i, end: i + 1 }); i += 1; continue; }
                '.' => { tokens.push(SpannedToken { token: Token::Dot, start: i, end: i + 1 }); i += 1; continue; }
                '(' => { tokens.push(SpannedToken { token: Token::LParen, start: i, end: i + 1 }); i += 1; continue; }
                ')' => { tokens.push(SpannedToken { token: Token::RParen, start: i, end: i + 1 }); i += 1; continue; }
                '+' => { tokens.push(SpannedToken { token: Token::Plus, start: i, end: i + 1 }); i += 1; continue; }
                '-' => { tokens.push(SpannedToken { token: Token::Minus, start: i, end: i + 1 }); i += 1; continue; }
                '/' => { tokens.push(SpannedToken { token: Token::Slash, start: i, end: i + 1 }); i += 1; continue; }
                '%' => { tokens.push(SpannedToken { token: Token::Percent, start: i, end: i + 1 }); i += 1; continue; }
                _ => {}
            }

            // Multi-character operators
            if c == '=' {
                tokens.push(SpannedToken { token: Token::Eq, start: i, end: i + 1 });
                i += 1;
                continue;
            }

            // Double-quoted identifier: "identifier"
            if c == '"' {
                let start0 = i;
                i += 1; // skip opening quote
                let start = i;
                while i < len && chars[i] != '"' {
                    i += 1;
                }
                if i >= len {
                    return Err(ApexError::QueryParseError(format!(
                        "Syntax error at byte {}: Unterminated double-quoted identifier",
                        start0
                    )));
                }
                let ident: String = chars[start..i].iter().collect();
                i += 1; // skip closing quote
                tokens.push(SpannedToken { token: Token::Identifier(ident), start: start0, end: i });
                continue;
            }
            if c == '\'' {
                let start0 = i;
                i += 1; // skip opening quote
                let start = i;
                while i < len && chars[i] != '\'' {
                    i += 1;
                }
                if i >= len {
                    return Err(ApexError::QueryParseError(format!(
                        "Syntax error at byte {}: Unterminated string literal",
                        start0
                    )));
                }
                let s: String = chars[start..i].iter().collect();
                i += 1; // skip closing quote
                tokens.push(SpannedToken { token: Token::StringLit(s), start: start0, end: i });
                continue;
            }
            if c == '!' && i + 1 < len && chars[i + 1] == '=' {
                tokens.push(SpannedToken { token: Token::NotEq, start: i, end: i + 2 });
                i += 2;
                continue;
            }
            if c == '<' {
                if i + 1 < len && chars[i + 1] == '=' {
                    tokens.push(SpannedToken { token: Token::Le, start: i, end: i + 2 });
                    i += 2;
                } else if i + 1 < len && chars[i + 1] == '>' {
                    tokens.push(SpannedToken { token: Token::NotEq, start: i, end: i + 2 });
                    i += 2;
                } else {
                    tokens.push(SpannedToken { token: Token::Lt, start: i, end: i + 1 });
                    i += 1;
                }
                continue;
            }
            if c == '>' {
                if i + 1 < len && chars[i + 1] == '=' {
                    tokens.push(SpannedToken { token: Token::Ge, start: i, end: i + 2 });
                    i += 2;
                } else {
                    tokens.push(SpannedToken { token: Token::Gt, start: i, end: i + 1 });
                    i += 1;
                }
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
                        ApexError::QueryParseError(format!("Syntax error at byte {}: Invalid number: {}", start, num_str)))?;
                    tokens.push(SpannedToken { token: Token::FloatLit(f), start, end: i });
                } else {
                    let n: i64 = num_str.parse().map_err(|_| 
                        ApexError::QueryParseError(format!("Syntax error at byte {}: Invalid number: {}", start, num_str)))?;
                    tokens.push(SpannedToken { token: Token::IntLit(n), start, end: i });
                }
                continue;
            }

            // Identifiers and keywords
            if c.is_ascii_alphabetic() || c == '_' {
                let start = i;
                i += 1;
                while i < len && (chars[i].is_ascii_alphanumeric() || chars[i] == '_') {
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
                    "REGEXP" => Token::Regexp,
                    "OVER" => Token::Over,
                    "PARTITION" => Token::Partition,
                    _ => Token::Identifier(word),
                };
                tokens.push(SpannedToken { token, start, end: i });
                continue;
            }

            return Err(ApexError::QueryParseError(format!(
                "Syntax error at byte {}: Unexpected character: {}",
                i, c
            )));
        }

        tokens.push(SpannedToken { token: Token::Eof, start: len, end: len });
        Ok(tokens)
    }

    fn current(&self) -> &Token {
        &self.tokens[self.pos].token
    }

    fn current_span(&self) -> (usize, usize) {
        let t = &self.tokens[self.pos];
        (t.start, t.end)
    }

    fn format_near(&self, at: usize) -> String {
        if self.sql_chars.is_empty() {
            return String::new();
        }
        let start = at.saturating_sub(16);
        let end = (at + 16).min(self.sql_chars.len());
        let snippet: String = self.sql_chars[start..end].iter().collect();
        snippet.replace('\n', " ")
    }

    fn line_col(&self, at: usize) -> (usize, usize) {
        // 1-based line/col
        let mut line = 1usize;
        let mut col = 1usize;
        let end = at.min(self.sql_chars.len());
        for ch in self.sql_chars.iter().take(end) {
            if *ch == '\n' {
                line += 1;
                col = 1;
            } else {
                col += 1;
            }
        }
        (line, col)
    }

    fn syntax_error(&self, at: usize, msg: String) -> ApexError {
        let near = self.format_near(at);
        let (line, col) = self.line_col(at);
        ApexError::QueryParseError(format!(
            "Syntax error at {}:{} (pos {}): {} (near: {})",
            line, col, at, msg, near
        ))
    }

    fn keyword_suggestion(&self) -> Option<String> {
        match self.current().clone() {
            Token::Identifier(s) => {
                let u = s.to_uppercase();
                // Keep list small and stable; used only for human-friendly hints.
                const KWS: [&str; 10] = [
                    "SELECT", "FROM", "WHERE", "LIKE", "LIMIT", "OFFSET", "ORDER", "GROUP", "HAVING", "DISTINCT",
                ];

                // Fast path for common "plural" / extra trailing char typos: FROMs, WHEREs, LIKEs, LIMITs
                for kw in KWS {
                    if u.len() == kw.len() + 1 && u.starts_with(kw) {
                        return Some(kw.to_string());
                    }
                    if u.ends_with('S') && &u[..u.len() - 1] == kw {
                        return Some(kw.to_string());
                    }
                }

                // Fuzzy match: allow small edit distance (e.g., SELECTE -> SELECT)
                let mut best: Option<(&str, usize)> = None;
                for kw in KWS {
                    let dist = Self::edit_distance(&u, kw);
                    if dist <= 2 {
                        match best {
                            None => best = Some((kw, dist)),
                            Some((_, best_dist)) if dist < best_dist => best = Some((kw, dist)),
                            _ => {}
                        }
                    }
                }
                best.map(|(kw, _)| kw.to_string())
            }
            _ => None,
        }
    }

    fn edit_distance(a: &str, b: &str) -> usize {
        // Classic DP Levenshtein distance. Inputs are short keywords; performance is irrelevant.
        let a: Vec<char> = a.chars().collect();
        let b: Vec<char> = b.chars().collect();
        let n = a.len();
        let m = b.len();

        if n == 0 {
            return m;
        }
        if m == 0 {
            return n;
        }

        let mut dp = vec![vec![0usize; m + 1]; n + 1];
        for i in 0..=n {
            dp[i][0] = i;
        }
        for j in 0..=m {
            dp[0][j] = j;
        }

        for i in 1..=n {
            for j in 1..=m {
                let cost = if a[i - 1] == b[j - 1] { 0 } else { 1 };
                dp[i][j] = (dp[i - 1][j] + 1)
                    .min(dp[i][j - 1] + 1)
                    .min(dp[i - 1][j - 1] + cost);
            }
        }
        dp[n][m]
    }

    fn advance(&mut self) -> &Token {
        let tok = &self.tokens[self.pos].token;
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
            let (start, _) = self.current_span();
            Err(self.syntax_error(
                start,
                format!("Expected {:?}, got {:?}", expected, self.current()),
            ))
        }
    }

    fn parse_statement(&mut self) -> Result<SqlStatement, ApexError> {
        match self.current() {
            Token::Select => {
                let stmt = self.parse_select().map(SqlStatement::Select)?;
                // Reject trailing tokens, so typos like FROMs/WHEREs don't get silently ignored.
                if !matches!(self.current(), Token::Eof) {
                    let (start, _) = self.current_span();
                    let mut msg = format!("Unexpected token {:?} after end of statement", self.current());
                    if let Some(kw) = self.keyword_suggestion() {
                        msg = format!("{} (did you mean {}?)", msg, kw);
                    }
                    return Err(self.syntax_error(start, msg));
                }
                Ok(stmt)
            }
            _ => {
                let (start, _) = self.current_span();
                let mut msg = "Expected SELECT statement".to_string();
                if let Some(kw) = self.keyword_suggestion() {
                    if kw == "SELECT" {
                        msg = format!("{} (did you mean SELECT?)", msg);
                    }
                }
                Err(self.syntax_error(start, msg))
            }
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

    /// Parse a column reference, supporting qualified names like t.col.
    ///
    /// Currently we normalize to the last identifier segment (e.g. "t._id" => "_id").
    fn parse_column_ref(&mut self) -> Result<String, ApexError> {
        let mut name = if let Token::Identifier(n) = self.current().clone() {
            self.advance();
            n
        } else {
            return Err(ApexError::QueryParseError("Expected column identifier".to_string()));
        };

        while matches!(self.current(), Token::Dot) {
            self.advance();
            if let Token::Identifier(n) = self.current().clone() {
                self.advance();
                name = n;
            } else {
                return Err(ApexError::QueryParseError("Expected identifier after '.'".to_string()));
            }
        }

        Ok(name)
    }

    fn parse_select_columns(&mut self) -> Result<Vec<SelectColumn>, ApexError> {
        let mut columns = Vec::new();

        loop {
            // SELECT *
            if matches!(self.current(), Token::Star) {
                self.advance();
                columns.push(SelectColumn::All);
            }
            // Aggregate functions
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
                    None
                } else if matches!(self.current(), Token::Identifier(_)) {
                    Some(self.parse_column_ref()?)
                } else {
                    None
                };

                self.expect(Token::RParen)?;

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
            // Column or window function name
            else if matches!(self.current(), Token::Identifier(_)) {
                let name = self.parse_column_ref()?;

                // Only window function supported: row_number() OVER (...)
                if matches!(self.current(), Token::LParen) {
                    self.advance();
                    self.expect(Token::RParen)?;

                    if !matches!(self.current(), Token::Over) {
                        return Err(ApexError::QueryParseError(
                            format!("Unsupported function in SELECT list: {}", name)
                        ));
                    }

                    self.advance();
                    self.expect(Token::LParen)?;

                    let mut partition_by = Vec::new();
                    if matches!(self.current(), Token::Partition) {
                        self.advance();
                        self.expect(Token::By)?;
                        partition_by = self.parse_column_list()?;
                    }

                    let order_by = if matches!(self.current(), Token::Order) {
                        self.advance();
                        self.expect(Token::By)?;
                        self.parse_order_by()?
                    } else {
                        Vec::new()
                    };

                    self.expect(Token::RParen)?;

                    let alias = if matches!(self.current(), Token::As) {
                        self.advance();
                        if let Token::Identifier(alias) = self.current().clone() {
                            self.advance();
                            Some(alias)
                        } else {
                            None
                        }
                    } else {
                        None
                    };

                    columns.push(SelectColumn::WindowFunction {
                        name,
                        partition_by,
                        order_by,
                        alias,
                    });
                } else {
                    // Regular column with optional alias
                    if matches!(self.current(), Token::As) {
                        self.advance();
                        if let Token::Identifier(alias) = self.current().clone() {
                            self.advance();
                            columns.push(SelectColumn::ColumnAlias { column: name, alias });
                        } else {
                            return Err(ApexError::QueryParseError("Expected alias after AS".to_string()));
                        }
                    } else {
                        columns.push(SelectColumn::Column(name));
                    }
                }
            } else {
                break;
            }

            if matches!(self.current(), Token::Comma) {
                self.advance();
            } else {
                break;
            }
        }

    if columns.is_empty() {
        let (start, _) = self.current_span();
        return Err(self.syntax_error(start, "Expected column list after SELECT".to_string()));
    }

    Ok(columns)
}

fn parse_column_list(&mut self) -> Result<Vec<String>, ApexError> {
    let mut columns = Vec::new();
    
    loop {
        if matches!(self.current(), Token::Identifier(_)) {
            let name = self.parse_column_ref()?;
            columns.push(name);
        } else {
            return Err(ApexError::QueryParseError("Expected column name".to_string()));
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
        if matches!(self.current(), Token::Identifier(_)) {
            let column = self.parse_column_ref()?;
            
            let descending = if matches!(self.current(), Token::Desc) {
                self.advance();
                true
            } else if matches!(self.current(), Token::Asc) {
                self.advance();
                false
            } else {
                // Default ASC
                if matches!(self.current(), Token::Asc) {
                    self.advance();
                }
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
        self.parse_or()
    }

    fn parse_or(&mut self) -> Result<SqlExpr, ApexError> {
        let mut left = self.parse_and()?;
        while matches!(self.current(), Token::Or) {
            self.advance();
            let right = self.parse_and()?;
            left = SqlExpr::BinaryOp {
                left: Box::new(left),
                op: BinaryOperator::Or,
                right: Box::new(right),
            };
        }
        Ok(left)
    }

    fn parse_and(&mut self) -> Result<SqlExpr, ApexError> {
        let mut left = self.parse_not()?;
        while matches!(self.current(), Token::And) {
            self.advance();
            let right = self.parse_not()?;
            left = SqlExpr::BinaryOp {
                left: Box::new(left),
                op: BinaryOperator::And,
                right: Box::new(right),
            };
        }
        Ok(left)
    }

    fn parse_not(&mut self) -> Result<SqlExpr, ApexError> {
        if matches!(self.current(), Token::Not) {
            self.advance();
            let expr = self.parse_not()?;
            return Ok(SqlExpr::UnaryOp {
                op: UnaryOperator::Not,
                expr: Box::new(expr),
            });
        }
        self.parse_comparison()
    }

    fn parse_comparison(&mut self) -> Result<SqlExpr, ApexError> {
        let left = self.parse_add_sub()?;

        // Special forms only supported when left is a column
        let left_col = if let SqlExpr::Column(ref c) = left {
            Some(c.clone())
        } else {
            None
        };

        if matches!(self.current(), Token::Like) {
            let column = left_col.ok_or_else(|| ApexError::QueryParseError("LIKE requires column on left side".to_string()))?;
            self.advance();
            let pattern = match self.current().clone() {
                Token::StringLit(s) => {
                    self.advance();
                    s
                }
                _ => return Err(ApexError::QueryParseError("LIKE pattern must be a string literal".to_string())),
            };
            return Ok(SqlExpr::Like { column, pattern, negated: false });
        }

        if matches!(self.current(), Token::Regexp) {
            let column = left_col.ok_or_else(|| ApexError::QueryParseError("REGEXP requires column on left side".to_string()))?;
            self.advance();
            let pattern = match self.current().clone() {
                Token::StringLit(s) => {
                    self.advance();
                    s
                }
                _ => return Err(ApexError::QueryParseError("REGEXP pattern must be a string literal".to_string())),
            };
            return Ok(SqlExpr::Regexp { column, pattern, negated: false });
        }

        if matches!(self.current(), Token::In) {
            let column = left_col.ok_or_else(|| ApexError::QueryParseError("IN requires column on left side".to_string()))?;
            self.advance();
            self.expect(Token::LParen)?;
            let mut values = Vec::new();
            loop {
                match self.current() {
                    Token::StringLit(_) | Token::IntLit(_) | Token::FloatLit(_) | Token::True | Token::False | Token::Null => {
                        values.push(self.parse_literal_value()?);
                    }
                    _ => break,
                }
                if matches!(self.current(), Token::Comma) {
                    self.advance();
                } else {
                    break;
                }
            }
            self.expect(Token::RParen)?;
            return Ok(SqlExpr::In { column, values, negated: false });
        }

        if matches!(self.current(), Token::Between) {
            let column = left_col.ok_or_else(|| ApexError::QueryParseError("BETWEEN requires column on left side".to_string()))?;
            self.advance();
            let low = Box::new(self.parse_add_sub()?);
            self.expect(Token::And)?;
            let high = Box::new(self.parse_add_sub()?);
            return Ok(SqlExpr::Between { column, low, high, negated: false });
        }

        if matches!(self.current(), Token::Is) {
            let column = left_col.ok_or_else(|| ApexError::QueryParseError("IS NULL requires column on left side".to_string()))?;
            self.advance();
            let negated = if matches!(self.current(), Token::Not) {
                self.advance();
                true
            } else {
                false
            };
            self.expect(Token::Null)?;
            return Ok(SqlExpr::IsNull { column, negated });
        }

        let op = match self.current() {
            Token::Eq => Some(BinaryOperator::Eq),
            Token::NotEq => Some(BinaryOperator::NotEq),
            Token::Lt => Some(BinaryOperator::Lt),
            Token::Le => Some(BinaryOperator::Le),
            Token::Gt => Some(BinaryOperator::Gt),
            Token::Ge => Some(BinaryOperator::Ge),
            _ => None,
        };
        if let Some(op) = op {
            self.advance();
            let right = self.parse_add_sub()?;
            return Ok(SqlExpr::BinaryOp {
                left: Box::new(left),
                op,
                right: Box::new(right),
            });
        }

        Ok(left)
    }

    fn parse_add_sub(&mut self) -> Result<SqlExpr, ApexError> {
        let mut left = self.parse_mul_div()?;
        loop {
            let op = match self.current() {
                Token::Plus => Some(BinaryOperator::Add),
                Token::Minus => Some(BinaryOperator::Sub),
                _ => None,
            };
            if let Some(op) = op {
                self.advance();
                let right = self.parse_mul_div()?;
                left = SqlExpr::BinaryOp {
                    left: Box::new(left),
                    op,
                    right: Box::new(right),
                };
            } else {
                break;
            }
        }
        Ok(left)
    }

    fn parse_mul_div(&mut self) -> Result<SqlExpr, ApexError> {
        let mut left = self.parse_primary()?;
        loop {
            let op = match self.current() {
                Token::Star => Some(BinaryOperator::Mul),
                Token::Slash => Some(BinaryOperator::Div),
                Token::Percent => Some(BinaryOperator::Mod),
                _ => None,
            };
            if let Some(op) = op {
                self.advance();
                let right = self.parse_primary()?;
                left = SqlExpr::BinaryOp {
                    left: Box::new(left),
                    op,
                    right: Box::new(right),
                };
            } else {
                break;
            }
        }
        Ok(left)
    }

    fn parse_literal_value(&mut self) -> Result<Value, ApexError> {
        match self.current().clone() {
            Token::StringLit(s) => {
                self.advance();
                Ok(Value::String(s))
            }
            Token::IntLit(n) => {
                self.advance();
                Ok(Value::Int64(n))
            }
            Token::FloatLit(f) => {
                self.advance();
                Ok(Value::Float64(f))
            }
            Token::True => {
                self.advance();
                Ok(Value::Bool(true))
            }
            Token::False => {
                self.advance();
                Ok(Value::Bool(false))
            }
            Token::Null => {
                self.advance();
                Ok(Value::Null)
            }
            _ => Err(ApexError::QueryParseError("Expected literal".to_string())),
        }
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
            Token::Identifier(_) => {
                let name = self.parse_column_ref()?;
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
            _ => Err(ApexError::QueryParseError(
                format!("Unexpected token in expression: {:?}", self.current())
            )),
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
        let SqlStatement::Select(s) = stmt;
        assert!(!s.distinct);
        assert_eq!(s.columns.len(), 1);
        assert!(matches!(s.columns[0], SelectColumn::All));
        assert_eq!(s.from, Some("users".to_string()));
    }

    #[test]
    fn test_select_with_where() {
        let sql = "SELECT name, age FROM users WHERE age > 18 AND name LIKE 'John%'";
        let stmt = SqlParser::parse(sql).unwrap();
        let SqlStatement::Select(s) = stmt;
        assert_eq!(s.columns.len(), 2);
        assert!(s.where_clause.is_some());
    }

    #[test]
    fn test_select_with_order_limit() {
        let sql = "SELECT * FROM users ORDER BY age DESC LIMIT 10 OFFSET 5";
        let stmt = SqlParser::parse(sql).unwrap();
        let SqlStatement::Select(s) = stmt;
        assert_eq!(s.order_by.len(), 1);
        assert!(s.order_by[0].descending);
        assert_eq!(s.limit, Some(10));
        assert_eq!(s.offset, Some(5));
    }

    #[test]
    fn test_select_qualified_id() {
        let sql = "SELECT default._id, name FROM default ORDER BY default._id";
        let stmt = SqlParser::parse(sql).unwrap();
        let SqlStatement::Select(s) = stmt;
        assert_eq!(s.columns.len(), 2);
        match &s.columns[0] {
            SelectColumn::Column(c) => assert_eq!(c, "_id"),
            other => panic!("unexpected column: {:?}", other),
        }
        match &s.columns[1] {
            SelectColumn::Column(c) => assert_eq!(c, "name"),
            other => panic!("unexpected column: {:?}", other),
        }
        assert_eq!(s.order_by.len(), 1);
        assert_eq!(s.order_by[0].column, "_id");
    }

    #[test]
    fn test_select_quoted_id() {
        let sql = "SELECT \"_id\", name FROM default ORDER BY \"_id\"";
        let stmt = SqlParser::parse(sql).unwrap();
        let SqlStatement::Select(s) = stmt;
        assert_eq!(s.columns.len(), 2);
        match &s.columns[0] {
            SelectColumn::Column(c) => assert_eq!(c, "_id"),
            other => panic!("unexpected column: {:?}", other),
        }
        assert_eq!(s.order_by.len(), 1);
        assert_eq!(s.order_by[0].column, "_id");
    }

    #[test]
    fn test_syntax_error_missing_select_list() {
        let err = SqlParser::parse("SELECT FROM t").unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("Syntax error"));
        assert!(msg.contains("Expected column list"));
    }

    #[test]
    fn test_syntax_error_unterminated_string() {
        let err = SqlParser::parse("SELECT * FROM t WHERE name = 'abc").unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("Unterminated string literal"));
        assert!(msg.contains("Syntax error"));
    }

    #[test]
    fn test_syntax_error_unexpected_character() {
        let err = SqlParser::parse("SELECT * FROM t WHERE a = @").unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("Unexpected character"));
        assert!(msg.contains("Syntax error"));
    }

    #[test]
    fn test_syntax_error_misspelled_keywords_like_froms() {
        let sql = "select * froms default wheres title likes 'Python%' limits 10";
        let err = SqlParser::parse(sql).unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("Syntax error"));
        assert!(msg.contains("did you mean FROM") || msg.contains("did you mean WHERE") || msg.contains("did you mean LIKE") || msg.contains("did you mean LIMIT"));
    }

    #[test]
    fn test_syntax_error_misspelled_select_keyword() {
        let sql = "selecte * from default";
        let err = SqlParser::parse(sql).unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("Syntax error"));
        assert!(msg.contains("did you mean SELECT"));
    }
}
