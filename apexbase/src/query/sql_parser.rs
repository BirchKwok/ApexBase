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
use crate::data::DataType;
use crate::data::Value;

/// Column definition for CREATE TABLE
#[derive(Debug, Clone)]
pub struct ColumnDef {
    pub name: String,
    pub data_type: DataType,
}

/// ALTER TABLE operation types
#[derive(Debug, Clone)]
pub enum AlterTableOp {
    AddColumn { name: String, data_type: DataType },
    DropColumn { name: String },
    RenameColumn { old_name: String, new_name: String },
}

/// SQL Statement types
#[derive(Debug, Clone)]
pub enum SqlStatement {
    Select(SelectStatement),
    Union(UnionStatement),
    CreateView { name: String, stmt: SelectStatement },
    DropView { name: String },
    // DDL Statements
    CreateTable { table: String, columns: Vec<ColumnDef>, if_not_exists: bool },
    DropTable { table: String, if_exists: bool },
    AlterTable { table: String, operation: AlterTableOp },
    TruncateTable { table: String },
    // DML Statements
    Insert { table: String, columns: Option<Vec<String>>, values: Vec<Vec<Value>> },
    Delete { table: String, where_clause: Option<SqlExpr> },
    Update { table: String, assignments: Vec<(String, SqlExpr)>, where_clause: Option<SqlExpr> },
}

#[derive(Debug, Clone)]
pub struct UnionStatement {
    pub left: Box<SqlStatement>,
    pub right: Box<SqlStatement>,
    pub all: bool,
    pub order_by: Vec<OrderByClause>,
    pub limit: Option<usize>,
    pub offset: Option<usize>,
}

#[derive(Debug, Clone)]
pub enum FromItem {
    Table {
        table: String,
        alias: Option<String>,
    },
    Subquery {
        stmt: Box<SelectStatement>,
        alias: String,
    },
}

#[derive(Debug, Clone, PartialEq)]
pub enum JoinType {
    Inner,
    Left,
}

#[derive(Debug, Clone)]
pub struct JoinClause {
    pub join_type: JoinType,
    pub right: FromItem,
    pub on: SqlExpr,
}

/// SELECT statement structure
#[derive(Debug, Clone)]
pub struct SelectStatement {
    pub distinct: bool,
    pub columns: Vec<SelectColumn>,
    pub from: Option<FromItem>,
    pub joins: Vec<JoinClause>,
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
    Aggregate { func: AggregateFunc, column: Option<String>, distinct: bool, alias: Option<String> },
    /// SELECT expression AS alias
    Expression { expr: SqlExpr, alias: Option<String> },
    /// SELECT row_number() OVER (PARTITION BY ... ORDER BY ...) AS alias
    /// Also supports LAG(col, offset, default) and LEAD(col, offset, default)
    WindowFunction {
        name: String,
        args: Vec<String>,  // Function arguments (column, offset, default)
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

impl std::fmt::Display for AggregateFunc {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AggregateFunc::Count => write!(f, "COUNT"),
            AggregateFunc::Sum => write!(f, "SUM"),
            AggregateFunc::Avg => write!(f, "AVG"),
            AggregateFunc::Min => write!(f, "MIN"),
            AggregateFunc::Max => write!(f, "MAX"),
        }
    }
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
    /// IN subquery: column IN (SELECT ...)
    InSubquery { column: String, stmt: Box<SelectStatement>, negated: bool },
    /// EXISTS subquery: EXISTS (SELECT ...)
    ExistsSubquery { stmt: Box<SelectStatement> },
    /// Scalar subquery: (SELECT ...)
    ScalarSubquery { stmt: Box<SelectStatement> },
    /// CASE WHEN <cond> THEN <expr> [WHEN <cond> THEN <expr>]* [ELSE <expr>] END
    Case {
        when_then: Vec<(SqlExpr, SqlExpr)>,
        else_expr: Option<Box<SqlExpr>>,
    },
    /// BETWEEN: column BETWEEN low AND high
    Between { column: String, low: Box<SqlExpr>, high: Box<SqlExpr>, negated: bool },
    /// IS NULL / IS NOT NULL
    IsNull { column: String, negated: bool },
    /// Function call
    Function { name: String, args: Vec<SqlExpr> },
    /// CAST(expr AS TYPE)
    Cast { expr: Box<SqlExpr>, data_type: DataType },
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

// ============================================================================
// Column Extraction for On-Demand Reading
// ============================================================================

impl SelectStatement {
    /// Extract columns needed only for WHERE clause evaluation
    /// Used for late materialization optimization
    pub fn where_columns(&self) -> Vec<String> {
        let mut columns = Vec::new();
        if let Some(ref expr) = self.where_clause {
            Self::extract_columns_from_expr(expr, &mut columns);
        }
        columns.sort();
        columns.dedup();
        columns
    }
    
    /// Check if this query uses SELECT * (needs all columns)
    pub fn is_select_star(&self) -> bool {
        self.columns.iter().any(|col| matches!(col, SelectColumn::All))
    }
    
    /// Extract all column names required by this SELECT statement
    /// Returns None if SELECT * is used (meaning all columns needed)
    pub fn required_columns(&self) -> Option<Vec<String>> {
        let mut columns = Vec::new();
        let mut has_star = false;
        let mut has_explicit_id = false;
        
        // Extract from SELECT clause
        for col in &self.columns {
            match col {
                SelectColumn::All => {
                    has_star = true;
                }
                SelectColumn::Column(name) => {
                    // Strip table prefix if present (e.g., "default._id" -> "_id")
                    let actual_name = if let Some(dot_pos) = name.rfind('.') {
                        &name[dot_pos + 1..]
                    } else {
                        name.as_str()
                    };
                    if actual_name == "_id" {
                        has_explicit_id = true;
                    } else {
                        columns.push(actual_name.to_string());
                    }
                }
                SelectColumn::ColumnAlias { column, .. } => {
                    if column == "_id" {
                        has_explicit_id = true;
                    } else {
                        columns.push(column.clone());
                    }
                }
                SelectColumn::Aggregate { column, .. } => {
                    if let Some(col) = column {
                        if col == "_id" {
                            has_explicit_id = true;
                        } else if col != "*" && !col.chars().next().map(|c| c.is_ascii_digit()).unwrap_or(false) {
                            // Skip constants like "1", "2" and "*" - only add real column names
                            columns.push(col.clone());
                        }
                    }
                }
                SelectColumn::Expression { expr, .. } => {
                    Self::extract_columns_from_expr(expr, &mut columns);
                }
                SelectColumn::WindowFunction { args, partition_by, order_by, .. } => {
                    // Add columns from args (for LAG/LEAD)
                    for arg in args {
                        if !arg.starts_with("Int") && !arg.starts_with("Float") && arg != "_id" {
                            columns.push(arg.clone());
                        }
                    }
                    for col in partition_by {
                        if col != "_id" {
                            columns.push(col.clone());
                        }
                    }
                    for ob in order_by {
                        if ob.column != "_id" {
                            columns.push(ob.column.clone());
                        }
                    }
                }
            }
        }
        
        // Extract from WHERE clause
        if let Some(ref expr) = self.where_clause {
            Self::extract_columns_from_expr(expr, &mut columns);
        }
        
        // Extract from ORDER BY
        for ob in &self.order_by {
            if ob.column != "_id" {
                columns.push(ob.column.clone());
            }
        }
        
        // Extract from GROUP BY
        for col in &self.group_by {
            if col != "_id" {
                columns.push(col.clone());
            }
        }
        
        // Extract from HAVING
        if let Some(ref expr) = self.having {
            Self::extract_columns_from_expr(expr, &mut columns);
        }
        
        if has_star {
            None  // SELECT * means all columns
        } else {
            // Include _id if explicitly requested
            if has_explicit_id {
                columns.push("_id".to_string());
            }
            // Deduplicate
            columns.sort();
            columns.dedup();
            // If no columns needed (e.g., COUNT(*)), return None to get all columns
            // so the batch has correct row count for aggregation
            if columns.is_empty() {
                None
            } else {
                Some(columns)
            }
        }
    }
    
    fn extract_columns_from_expr(expr: &SqlExpr, columns: &mut Vec<String>) {
        match expr {
            SqlExpr::Column(name) => {
                // Strip table prefix if present (e.g., "o.user_id" -> "user_id")
                let actual_name = if let Some(dot_pos) = name.rfind('.') {
                    &name[dot_pos + 1..]
                } else {
                    name.as_str()
                };
                if actual_name != "_id" {
                    columns.push(actual_name.to_string());
                }
            }
            SqlExpr::BinaryOp { left, right, .. } => {
                Self::extract_columns_from_expr(left, columns);
                Self::extract_columns_from_expr(right, columns);
            }
            SqlExpr::UnaryOp { expr, .. } => {
                Self::extract_columns_from_expr(expr, columns);
            }
            SqlExpr::Like { column, .. } | 
            SqlExpr::Regexp { column, .. } |
            SqlExpr::In { column, .. } |
            SqlExpr::Between { column, .. } |
            SqlExpr::IsNull { column, .. } |
            SqlExpr::InSubquery { column, .. } => {
                // Strip table prefix if present
                let actual_name = if let Some(dot_pos) = column.rfind('.') {
                    &column[dot_pos + 1..]
                } else {
                    column.as_str()
                };
                if actual_name != "_id" {
                    columns.push(actual_name.to_string());
                }
            }
            SqlExpr::Case { when_then, else_expr } => {
                for (cond, then_expr) in when_then {
                    Self::extract_columns_from_expr(cond, columns);
                    Self::extract_columns_from_expr(then_expr, columns);
                }
                if let Some(else_e) = else_expr {
                    Self::extract_columns_from_expr(else_e, columns);
                }
            }
            SqlExpr::Function { args, .. } => {
                for arg in args {
                    Self::extract_columns_from_expr(arg, columns);
                }
            }
            SqlExpr::Cast { expr, .. } => {
                Self::extract_columns_from_expr(expr, columns);
            }
            SqlExpr::Paren(inner) => {
                Self::extract_columns_from_expr(inner, columns);
            }
            SqlExpr::ExistsSubquery { stmt } | SqlExpr::ScalarSubquery { stmt } => {
                // For correlated subqueries, extract outer column references from WHERE clause
                // These are columns like "u.user_id" or "outer_table.col" that reference the outer query
                if let Some(ref where_clause) = stmt.where_clause {
                    Self::extract_outer_refs_from_subquery(where_clause, columns);
                }
            }
            SqlExpr::InSubquery { column, stmt, .. } => {
                // The column being compared (e.g., "user_id" in "user_id IN (SELECT ...)")
                if column != "_id" {
                    columns.push(column.clone());
                }
                // Also extract outer references from subquery WHERE clause
                if let Some(ref where_clause) = stmt.where_clause {
                    Self::extract_outer_refs_from_subquery(where_clause, columns);
                }
            }
            _ => {}
        }
    }
    
    /// Extract outer column references from subquery expressions
    /// These are qualified column names like "u.col" or "table.col" that reference outer tables
    fn extract_outer_refs_from_subquery(expr: &SqlExpr, columns: &mut Vec<String>) {
        match expr {
            SqlExpr::Column(name) => {
                let clean_name = name.trim_matches('"');
                // Check for qualified names like "u.user_id" or "users.user_id"
                if let Some(dot_pos) = clean_name.find('.') {
                    let col_part = &clean_name[dot_pos + 1..];
                    if col_part != "_id" && !columns.contains(&col_part.to_string()) {
                        columns.push(col_part.to_string());
                    }
                }
            }
            SqlExpr::BinaryOp { left, right, .. } => {
                Self::extract_outer_refs_from_subquery(left, columns);
                Self::extract_outer_refs_from_subquery(right, columns);
            }
            SqlExpr::UnaryOp { expr: inner, .. } | SqlExpr::Paren(inner) => {
                Self::extract_outer_refs_from_subquery(inner, columns);
            }
            _ => {}
        }
    }
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
    Join, Left, Right, Full, Inner, Outer, On,
    Union, All,
    Exists,
    Cast,
    Case, When, Then, Else, End,
    Create, Drop, View,
    // DDL keywords
    Table, Alter, Add, Column, Rename, To, If, Truncate,
    // DML keywords
    Insert, Into, Values, Delete, Update, Set,
    // Symbols
    Star,           // *
    Comma,          // ,
    Dot,            // .
    LParen,         // (
    RParen,         // )
    Semicolon,      // ;
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

    /// Parse multiple SQL statements separated by semicolons.
    pub fn parse_multi(sql: &str) -> Result<Vec<SqlStatement>, ApexError> {
        let tokens = Self::tokenize(sql)?;
        let mut parser = SqlParser {
            sql_chars: sql.chars().collect(),
            tokens,
            pos: 0,
        };
        parser.parse_statements()
    }

    /// Parse a standalone SQL expression (same grammar as WHERE/HAVING).
    ///
    /// This is used to unify the non-SQL query language with SQL semantics.
    pub fn parse_expression(expr: &str) -> Result<SqlExpr, ApexError> {
        let tokens = Self::tokenize(expr)?;
        let mut parser = SqlParser {
            sql_chars: expr.chars().collect(),
            tokens,
            pos: 0,
        };
        let e = parser.parse_expr()?;

        if !matches!(parser.current(), Token::Eof) {
            let (start, _) = parser.current_span();
            return Err(parser.syntax_error(
                start,
                format!("Unexpected token {:?} after end of expression", parser.current()),
            ));
        }

        Ok(e)
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
                ';' => { tokens.push(SpannedToken { token: Token::Semicolon, start: i, end: i + 1 }); i += 1; continue; }
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
                    "JOIN" => Token::Join,
                    "LEFT" => Token::Left,
                    "RIGHT" => Token::Right,
                    "FULL" => Token::Full,
                    "INNER" => Token::Inner,
                    "OUTER" => Token::Outer,
                    "ON" => Token::On,
                    "UNION" => Token::Union,
                    "ALL" => Token::All,
                    "EXISTS" => Token::Exists,
                    "CAST" => Token::Cast,
                    "CASE" => Token::Case,
                    "WHEN" => Token::When,
                    "THEN" => Token::Then,
                    "ELSE" => Token::Else,
                    "END" => Token::End,
                    "CREATE" => Token::Create,
                    "DROP" => Token::Drop,
                    "VIEW" => Token::View,
                    // DDL keywords
                    "TABLE" => Token::Table,
                    "ALTER" => Token::Alter,
                    "ADD" => Token::Add,
                    "COLUMN" => Token::Column,
                    "RENAME" => Token::Rename,
                    "TO" => Token::To,
                    "IF" => Token::If,
                    "TRUNCATE" => Token::Truncate,
                    // DML keywords
                    "INSERT" => Token::Insert,
                    "INTO" => Token::Into,
                    "VALUES" => Token::Values,
                    "DELETE" => Token::Delete,
                    "UPDATE" => Token::Update,
                    "SET" => Token::Set,
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

    fn parse_statements(&mut self) -> Result<Vec<SqlStatement>, ApexError> {
        let mut out = Vec::new();
        while !matches!(self.current(), Token::Eof) {
            while matches!(self.current(), Token::Semicolon) {
                self.advance();
            }
            if matches!(self.current(), Token::Eof) {
                break;
            }
            let stmt = self.parse_statement()?;
            out.push(stmt);
            while matches!(self.current(), Token::Semicolon) {
                self.advance();
            }
        }
        Ok(out)
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
                const KWS: [&str; 61] = [
                    "SELECT",
                    "FROM",
                    "WHERE",
                    "AND",
                    "OR",
                    "NOT",
                    "AS",
                    "DISTINCT",
                    "ORDER",
                    "BY",
                    "ASC",
                    "DESC",
                    "LIMIT",
                    "OFFSET",
                    "NULLS",
                    "FIRST",
                    "LAST",
                    "LIKE",
                    "IN",
                    "BETWEEN",
                    "IS",
                    "NULL",
                    "GROUP",
                    "HAVING",
                    "COUNT",
                    "SUM",
                    "AVG",
                    "MIN",
                    "MAX",
                    "TRUE",
                    "FALSE",
                    "REGEXP",
                    "OVER",
                    "PARTITION",
                    "JOIN",
                    "LEFT",
                    "RIGHT",
                    "FULL",
                    "INNER",
                    "OUTER",
                    "ON",
                    "UNION",
                    "ALL",
                    "EXISTS",
                    "CASE",
                    "WHEN",
                    "THEN",
                    "ELSE",
                    "END",
                    // DDL keywords
                    "TABLE",
                    "ALTER",
                    "ADD",
                    "COLUMN",
                    "RENAME",
                    "TRUNCATE",
                    // DML keywords
                    "INSERT",
                    "INTO",
                    "VALUES",
                    "DELETE",
                    "UPDATE",
                    "SET",
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
            let mut msg = format!("Expected {:?}, got {:?}", expected, self.current());
            if let Some(kw) = self.keyword_suggestion() {
                msg = format!("{} (did you mean {}?)", msg, kw);
            }
            Err(self.syntax_error(start, msg))
        }
    }

    fn parse_statement(&mut self) -> Result<SqlStatement, ApexError> {
        match self.current() {
            Token::Select => {
                // Parse the first SELECT part without consuming ORDER/LIMIT/OFFSET.
                // Those trailing clauses belong to UNION result if a UNION follows.
                let mut stmt = SqlStatement::Select(self.parse_select_part()?);

                // UNION chain
                while matches!(self.current(), Token::Union) {
                    self.advance();
                    let all = if matches!(self.current(), Token::All) {
                        self.advance();
                        true
                    } else {
                        false
                    };

                    if !matches!(self.current(), Token::Select) {
                        let (start, _) = self.current_span();
                        return Err(self.syntax_error(start, "Expected SELECT after UNION".to_string()));
                    }
                    let right = SqlStatement::Select(self.parse_select_part()?);

                    stmt = SqlStatement::Union(UnionStatement {
                        left: Box::new(stmt),
                        right: Box::new(right),
                        all,
                        order_by: Vec::new(),
                        limit: None,
                        offset: None,
                    });
                }

                // Trailing clauses apply to the final result (SELECT or UNION)
                let order_by = if matches!(self.current(), Token::Order) {
                    self.advance();
                    self.expect(Token::By)?;
                    self.parse_order_by()?
                } else {
                    Vec::new()
                };

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

                if !order_by.is_empty() || limit.is_some() || offset.is_some() {
                    match stmt {
                        SqlStatement::Union(mut u) => {
                            u.order_by = order_by;
                            u.limit = limit;
                            u.offset = offset;
                            stmt = SqlStatement::Union(u);
                        }
                        SqlStatement::Select(mut s) => {
                            s.order_by = order_by;
                            s.limit = limit;
                            s.offset = offset;
                            stmt = SqlStatement::Select(s);
                        }
                        _ => {}
                    }
                }

                Ok(stmt)
            }
            Token::Create => {
                self.advance();
                match self.current() {
                    Token::View => {
                        self.advance();
                        let name = self.parse_identifier()?;
                        self.expect(Token::As)?;
                        if !matches!(self.current(), Token::Select) {
                            let (start, _) = self.current_span();
                            return Err(self.syntax_error(start, "Expected SELECT after AS".to_string()));
                        }
                        let stmt = self.parse_select_internal(true)?;
                        Ok(SqlStatement::CreateView { name, stmt })
                    }
                    Token::Table => {
                        self.advance();
                        // Check for IF NOT EXISTS
                        let if_not_exists = self.parse_if_not_exists()?;
                        let table = self.parse_identifier()?;
                        self.expect(Token::LParen)?;
                        let columns = self.parse_column_defs()?;
                        self.expect(Token::RParen)?;
                        Ok(SqlStatement::CreateTable { table, columns, if_not_exists })
                    }
                    _ => {
                        let (start, _) = self.current_span();
                        Err(self.syntax_error(start, "Expected TABLE or VIEW after CREATE".to_string()))
                    }
                }
            }
            Token::Drop => {
                self.advance();
                match self.current() {
                    Token::View => {
                        self.advance();
                        let name = self.parse_identifier()?;
                        Ok(SqlStatement::DropView { name })
                    }
                    Token::Table => {
                        self.advance();
                        // Check for IF EXISTS
                        let if_exists = self.parse_if_exists()?;
                        let table = self.parse_identifier()?;
                        Ok(SqlStatement::DropTable { table, if_exists })
                    }
                    _ => {
                        let (start, _) = self.current_span();
                        Err(self.syntax_error(start, "Expected TABLE or VIEW after DROP".to_string()))
                    }
                }
            }
            Token::Alter => {
                self.advance();
                self.expect(Token::Table)?;
                let table = self.parse_identifier()?;
                let operation = self.parse_alter_operation()?;
                Ok(SqlStatement::AlterTable { table, operation })
            }
            Token::Truncate => {
                self.advance();
                self.expect(Token::Table)?;
                let table = self.parse_identifier()?;
                Ok(SqlStatement::TruncateTable { table })
            }
            Token::Insert => {
                self.advance();
                self.expect(Token::Into)?;
                let table = self.parse_identifier()?;
                // Optional column list
                let columns = if matches!(self.current(), Token::LParen) {
                    self.advance();
                    let cols = self.parse_identifier_list()?;
                    self.expect(Token::RParen)?;
                    Some(cols)
                } else {
                    None
                };
                self.expect(Token::Values)?;
                let values = self.parse_values_list()?;
                Ok(SqlStatement::Insert { table, columns, values })
            }
            Token::Delete => {
                self.advance();
                self.expect(Token::From)?;
                let table = self.parse_identifier()?;
                let where_clause = if matches!(self.current(), Token::Where) {
                    self.advance();
                    Some(self.parse_expr()?)
                } else {
                    None
                };
                Ok(SqlStatement::Delete { table, where_clause })
            }
            Token::Update => {
                self.advance();
                let table = self.parse_identifier()?;
                self.expect(Token::Set)?;
                let assignments = self.parse_assignments()?;
                let where_clause = if matches!(self.current(), Token::Where) {
                    self.advance();
                    Some(self.parse_expr()?)
                } else {
                    None
                };
                Ok(SqlStatement::Update { table, assignments, where_clause })
            }
            _ => {
                let (start, _) = self.current_span();
                Err(self.syntax_error(start, "Expected SQL statement".to_string()))
            }
        }
    }

    // Parse a SELECT used as a UNION operand: does not consume ORDER BY / LIMIT / OFFSET.
    pub fn parse_select_statement(&mut self) -> Result<SelectStatement, ApexError> {
        self.parse_select_internal(false)
    }

    fn parse_select_part(&mut self) -> Result<SelectStatement, ApexError> {
        self.parse_select_internal(false)
    }

    #[allow(dead_code)]
    fn parse_select(&mut self) -> Result<SelectStatement, ApexError> {
        self.parse_select_internal(true)
    }

    fn parse_select_internal(&mut self, parse_tail: bool) -> Result<SelectStatement, ApexError> {
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
            match self.current().clone() {
                Token::Identifier(table) => {
                    self.advance();
                    let alias = if let Token::Identifier(a) = self.current().clone() {
                        self.advance();
                        Some(a)
                    } else {
                        None
                    };
                    Some(FromItem::Table { table, alias })
                }
                Token::LParen => {
                    self.advance();

                    // Derived table: FROM (SELECT ...) alias
                    // For now, only allow a SELECT (no UNION) inside.
                    if !matches!(self.current(), Token::Select) {
                        let (start, _) = self.current_span();
                        return Err(self.syntax_error(start, "Expected SELECT after FROM (".to_string()));
                    }
                    let sub = self.parse_select_internal(true)?;
                    self.expect(Token::RParen)?;

                    let alias = if let Token::Identifier(a) = self.current().clone() {
                        self.advance();
                        a
                    } else {
                        return Err(ApexError::QueryParseError(
                            "Derived table in FROM requires an alias".to_string(),
                        ));
                    };

                    Some(FromItem::Subquery {
                        stmt: Box::new(sub),
                        alias,
                    })
                }
                _ => {
                    return Err(ApexError::QueryParseError("Expected table name after FROM".to_string()));
                }
            }
        } else {
            None
        };

        // JOIN clauses
        let mut joins: Vec<JoinClause> = Vec::new();
        loop {
            let join_type = if matches!(self.current(), Token::Join) {
                JoinType::Inner
            } else if matches!(self.current(), Token::Left) {
                self.advance();
                if matches!(self.current(), Token::Outer) {
                    self.advance();
                }
                self.expect(Token::Join)?;
                JoinType::Left
            } else if matches!(self.current(), Token::Inner) {
                self.advance();
                self.expect(Token::Join)?;
                JoinType::Inner
            } else {
                break;
            };

            if matches!(self.current(), Token::Join) {
                self.advance();
            }

            let right = match self.current().clone() {
                Token::Identifier(table) => {
                    self.advance();
                    let alias = if let Token::Identifier(a) = self.current().clone() {
                        self.advance();
                        Some(a)
                    } else {
                        None
                    };
                    FromItem::Table { table, alias }
                }
                _ => {
                    return Err(ApexError::QueryParseError("Expected table name after JOIN".to_string()));
                }
            };

            self.expect(Token::On)?;
            let on = self.parse_expr()?;
            joins.push(JoinClause { join_type, right, on });
        }

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

        // ORDER BY / LIMIT / OFFSET
        let order_by = if parse_tail && matches!(self.current(), Token::Order) {
            self.advance();
            self.expect(Token::By)?;
            self.parse_order_by()?
        } else {
            Vec::new()
        };

        let limit = if parse_tail && matches!(self.current(), Token::Limit) {
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

        let offset = if parse_tail && matches!(self.current(), Token::Offset) {
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
            joins,
            where_clause,
            group_by,
            having,
            order_by,
            limit,
            offset,
        })
    }

    fn parse_alias_identifier(&mut self) -> Option<String> {
        match self.current() {
            Token::Identifier(name) => {
                let name = name.clone();
                self.advance();
                Some(name)
            }
            // Allow keyword aliases, e.g. COUNT(1) count
            Token::Count => { self.advance(); Some("count".to_string()) }
            Token::Sum => { self.advance(); Some("sum".to_string()) }
            Token::Avg => { self.advance(); Some("avg".to_string()) }
            Token::Min => { self.advance(); Some("min".to_string()) }
            Token::Max => { self.advance(); Some("max".to_string()) }
            _ => None,
        }
    }

    /// Parse a column reference, supporting qualified names like t.col.
    ///
    /// We preserve the full qualified name (e.g. "t._id"). Execution may
    /// normalize this as needed.
    fn parse_column_ref(&mut self) -> Result<String, ApexError> {
        let mut full = if let Token::Identifier(n) = self.current().clone() {
            self.advance();
            n
        } else {
            return Err(ApexError::QueryParseError("Expected column identifier".to_string()));
        };

        while matches!(self.current(), Token::Dot) {
            self.advance();
            if let Token::Identifier(n) = self.current().clone() {
                self.advance();
                full.push('.');
                full.push_str(&n);
            } else {
                return Err(ApexError::QueryParseError("Expected identifier after '.'".to_string()));
            }
        }

        Ok(full)
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

                let distinct = if matches!(self.current(), Token::Distinct) {
                    self.advance();
                    true
                } else {
                    false
                };

                let column = if matches!(self.current(), Token::Star) {
                    if distinct {
                        let (start, _) = self.current_span();
                        return Err(self.syntax_error(start, "COUNT(DISTINCT *) is not supported".to_string()));
                    }
                    self.advance();
                    None
                } else if matches!(self.current(), Token::Identifier(_)) {
                    if distinct && func != AggregateFunc::Count {
                        let (start, _) = self.current_span();
                        return Err(self.syntax_error(start, "DISTINCT is only supported for COUNT".to_string()));
                    }
                    Some(self.parse_column_ref()?)
                } else if func == AggregateFunc::Count
                    && matches!(
                        self.current(),
                        Token::IntLit(_)
                            | Token::FloatLit(_)
                            | Token::StringLit(_)
                            | Token::True
                            | Token::False
                            | Token::Null
                    )
                {
                    // COUNT(1) / COUNT(constant) are commonly used and semantically equivalent to COUNT(*)
                    // for our execution engine.
                    let arg = match self.current().clone() {
                        Token::IntLit(n) => n.to_string(),
                        Token::FloatLit(f) => f.to_string(),
                        Token::StringLit(s) => format!("'{}'", s),
                        Token::True => "true".to_string(),
                        Token::False => "false".to_string(),
                        Token::Null => "null".to_string(),
                        _ => "1".to_string(),
                    };
                    self.advance();
                    Some(arg)
                } else {
                    None
                };

                self.expect(Token::RParen)?;

                // Check if this is a window function (has OVER clause)
                if matches!(self.current(), Token::Over) {
                    // Convert aggregate to window function
                    let func_name = format!("{}", func);
                    let args: Vec<String> = column.clone().into_iter().collect();
                    
                    self.advance(); // consume OVER
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
                        self.parse_alias_identifier()
                    } else {
                        self.parse_alias_identifier()
                    };
                    
                    columns.push(SelectColumn::WindowFunction {
                        name: func_name,
                        args,
                        partition_by,
                        order_by,
                        alias,
                    });
                } else {
                    let alias = if matches!(self.current(), Token::As) {
                        self.advance();
                        self.parse_alias_identifier()
                    } else {
                        // Allow implicit aliases: MIN(x) min_x
                        // Also allow keyword aliases like COUNT(1) count
                        self.parse_alias_identifier()
                    };

                    columns.push(SelectColumn::Aggregate { func, column, distinct, alias });
                }
            }
            // Column or window function name
            else if matches!(self.current(), Token::Identifier(_)) {
                let name = self.parse_column_ref()?;

                // Only window function supported: row_number() OVER (...)
                if matches!(self.current(), Token::LParen) {
                    // Parse function call.
                    // If it is followed by OVER, treat it as a window function.
                    // Otherwise, treat it as a scalar expression in the SELECT list.
                    let func_expr = self.parse_function_call_from_name(name.clone())?;

                    if matches!(self.current(), Token::Over) {
                        // Window function: func(args...) OVER (PARTITION BY ... ORDER BY ...)
                        // Extract args from the function expression
                        let args = if let SqlExpr::Function { args: func_args, .. } = &func_expr {
                            func_args.iter().filter_map(|a| {
                                if let SqlExpr::Column(c) = a { Some(c.clone()) }
                                else if let SqlExpr::Literal(v) = a { Some(format!("{:?}", v)) }
                                else { None }
                            }).collect()
                        } else {
                            Vec::new()
                        };
                        
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
                            self.parse_alias_identifier()
                        } else {
                            self.parse_alias_identifier()
                        };

                        columns.push(SelectColumn::WindowFunction {
                            name,
                            args,
                            partition_by,
                            order_by,
                            alias,
                        });
                    } else {
                        let alias = if matches!(self.current(), Token::As) {
                            self.advance();
                            self.parse_alias_identifier()
                        } else {
                            self.parse_alias_identifier()
                        };

                        columns.push(SelectColumn::Expression { expr: func_expr, alias });
                    }
                } else {
                    // Regular column with optional alias
                    if matches!(self.current(), Token::As) {
                        self.advance();
                        if let Some(alias) = self.parse_alias_identifier() {
                            columns.push(SelectColumn::ColumnAlias { column: name, alias });
                        } else {
                            return Err(ApexError::QueryParseError("Expected alias after AS".to_string()));
                        }
                    } else {
                        // Allow implicit aliases: col alias
                        if let Some(alias) = self.parse_alias_identifier() {
                            columns.push(SelectColumn::ColumnAlias { column: name, alias });
                        } else {
                            columns.push(SelectColumn::Column(name));
                        }
                    }
                }
            } else {
                // Fallback: allow expression/literal select items like `SELECT 1`.
                // This is commonly used in EXISTS subqueries.
                if matches!(
                    self.current(),
                    Token::IntLit(_)
                        | Token::FloatLit(_)
                        | Token::StringLit(_)
                        | Token::True
                        | Token::False
                        | Token::Null
                        | Token::LParen
                        | Token::Exists
                        | Token::Cast
                        | Token::Case
                        | Token::Not
                        | Token::Minus
                ) {
                    let expr = self.parse_expr()?;
                    let alias = if matches!(self.current(), Token::As) {
                        self.advance();
                        self.parse_alias_identifier()
                    } else {
                        self.parse_alias_identifier()
                    };
                    columns.push(SelectColumn::Expression { expr, alias });
                } else {
                    break;
                }
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

    fn parse_unary(&mut self) -> Result<SqlExpr, ApexError> {
        match self.current() {
            Token::Minus => {
                self.advance();
                let expr = self.parse_unary()?;
                Ok(SqlExpr::UnaryOp {
                    op: UnaryOperator::Minus,
                    expr: Box::new(expr),
                })
            }
            _ => self.parse_primary(),
        }
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
            other => Err(ApexError::QueryParseError(format!(
                "Expected literal value, got {:?}",
                other
            ))),
        }
    }

    fn parse_function_call_from_name(&mut self, name: String) -> Result<SqlExpr, ApexError> {
        self.expect(Token::LParen)?;
        let mut args = Vec::new();
        // Special-case COUNT(*) in expression contexts (e.g. HAVING COUNT(*) > 1).
        // In SELECT list we have separate aggregate parsing that already handles COUNT(*),
        // but expressions go through this generic function-call parser.
        if matches!(self.current(), Token::Star) && name.eq_ignore_ascii_case("count") {
            self.advance();
            self.expect(Token::RParen)?;
            return Ok(SqlExpr::Function { name, args });
        }

        if !matches!(self.current(), Token::RParen) {
            loop {
                args.push(self.parse_expr()?);
                if matches!(self.current(), Token::Comma) {
                    self.advance();
                } else {
                    break;
                }
            }
        }
        self.expect(Token::RParen)?;
        Ok(SqlExpr::Function { name, args })
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

        // Support infix negation: <col> NOT LIKE/IN/BETWEEN/REGEXP ...
        // Note: Unary NOT is already handled in parse_not(). If we see NOT here,
        // it must be part of one of the supported infix forms.
        let mut negated = false;
        if matches!(self.current(), Token::Not) {
            self.advance();
            negated = true;
        }

        if matches!(self.current(), Token::Like) {
            let column = left_col.ok_or_else(|| ApexError::QueryParseError("LIKE requires column on left side".to_string()))?;
            self.advance();
            let pattern = match self.current().clone() {
                Token::StringLit(s) => {
                    self.advance();
                    s
                }
                Token::Identifier(s) => {
                    // Support double-quoted patterns like LIKE "foo%" which tokenize as Identifier.
                    self.advance();
                    s
                }
                _ => return Err(ApexError::QueryParseError("LIKE pattern must be a string literal".to_string())),
            };
            return Ok(SqlExpr::Like { column, pattern, negated });
        }

        if matches!(self.current(), Token::Regexp) {
            let column = left_col.ok_or_else(|| ApexError::QueryParseError("REGEXP requires column on left side".to_string()))?;
            self.advance();
            let pattern = match self.current().clone() {
                Token::StringLit(s) => {
                    self.advance();
                    s
                }
                Token::Identifier(s) => {
                    // Support double-quoted patterns like REGEXP "test*" which tokenize as Identifier.
                    self.advance();
                    s
                }
                _ => return Err(ApexError::QueryParseError("REGEXP pattern must be a string literal".to_string())),
            };
            return Ok(SqlExpr::Regexp { column, pattern, negated });
        }

        if matches!(self.current(), Token::In) {
            let column = left_col.ok_or_else(|| ApexError::QueryParseError("IN requires column on left side".to_string()))?;
            self.advance();
            self.expect(Token::LParen)?;
            if matches!(self.current(), Token::Select) {
                let sub = self.parse_select_internal(true)?;
                self.expect(Token::RParen)?;
                return Ok(SqlExpr::InSubquery {
                    column,
                    stmt: Box::new(sub),
                    negated,
                });
            }
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
            return Ok(SqlExpr::In { column, values, negated });
        }

        if matches!(self.current(), Token::Between) {
            let column = left_col.ok_or_else(|| ApexError::QueryParseError("BETWEEN requires column on left side".to_string()))?;
            self.advance();
            let low = Box::new(self.parse_add_sub()?);
            self.expect(Token::And)?;
            let high = Box::new(self.parse_add_sub()?);
            return Ok(SqlExpr::Between { column, low, high, negated });
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

        if negated {
            let (start, _) = self.current_span();
            return Err(self.syntax_error(
                start,
                "Expected LIKE/IN/BETWEEN/REGEXP after NOT".to_string(),
            ));
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
        let mut left = self.parse_unary()?;
        loop {
            let op = match self.current() {
                Token::Star => Some(BinaryOperator::Mul),
                Token::Slash => Some(BinaryOperator::Div),
                Token::Percent => Some(BinaryOperator::Mod),
                _ => None,
            };
            if let Some(op) = op {
                self.advance();
                let right = self.parse_unary()?;
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

    fn parse_primary(&mut self) -> Result<SqlExpr, ApexError> {
        match self.current().clone() {
            Token::LParen => {
                self.advance();
                if matches!(self.current(), Token::Select) {
                    let sub = self.parse_select_internal(true)?;
                    self.expect(Token::RParen)?;
                    Ok(SqlExpr::ScalarSubquery { stmt: Box::new(sub) })
                } else {
                    let expr = self.parse_expr()?;
                    self.expect(Token::RParen)?;
                    Ok(SqlExpr::Paren(Box::new(expr)))
                }
            }
            Token::Exists => {
                self.advance();
                self.expect(Token::LParen)?;
                if !matches!(self.current(), Token::Select) {
                    return Err(ApexError::QueryParseError(
                        "EXISTS requires a SELECT subquery".to_string(),
                    ));
                }
                let sub = self.parse_select_internal(true)?;
                self.expect(Token::RParen)?;
                Ok(SqlExpr::ExistsSubquery {
                    stmt: Box::new(sub),
                })
            }
            Token::Cast => {
                self.advance();
                self.expect(Token::LParen)?;
                let expr = self.parse_expr()?;
                self.expect(Token::As)?;
                let ty = match self.current().clone() {
                    Token::Identifier(t) => {
                        self.advance();
                        t
                    }
                    other => {
                        return Err(ApexError::QueryParseError(format!(
                            "Expected type name after AS in CAST(), got {:?}",
                            other
                        )))
                    }
                };

                // Be conservative but tolerant here: some callers rely on `CAST(expr AS TYPE) AS alias`.
                // In rare cases we may see `AS` immediately after the type token (alias),
                // so avoid failing with a confusing "Expected RParen, got As" error.
                if matches!(self.current(), Token::RParen) {
                    self.advance();
                } else if !matches!(self.current(), Token::As) {
                    self.expect(Token::RParen)?;
                }
                Ok(SqlExpr::Cast {
                    expr: Box::new(expr),
                    data_type: DataType::from_sql_type(&ty),
                })
            }
            Token::Case => {
                self.advance();

                let mut when_then: Vec<(SqlExpr, SqlExpr)> = Vec::new();
                let mut else_expr: Option<Box<SqlExpr>> = None;

                if !matches!(self.current(), Token::When) {
                    let (start, _) = self.current_span();
                    return Err(self.syntax_error(start, "CASE must start with WHEN".to_string()));
                }

                while matches!(self.current(), Token::When) {
                    self.advance();
                    let cond = self.parse_expr()?;
                    self.expect(Token::Then)?;
                    let val = self.parse_expr()?;
                    when_then.push((cond, val));
                }

                if matches!(self.current(), Token::Else) {
                    self.advance();
                    let v = self.parse_expr()?;
                    else_expr = Some(Box::new(v));
                }

                self.expect(Token::End)?;

                Ok(SqlExpr::Case { when_then, else_expr })
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

                if matches!(self.current(), Token::LParen) {
                    return self.parse_function_call_from_name(name);
                }

                let mut full = name;
                while matches!(self.current(), Token::Dot) {
                    self.advance();
                    if let Token::Identifier(n) = self.current().clone() {
                        self.advance();
                        full.push('.');
                        full.push_str(&n);
                    } else {
                        return Err(ApexError::QueryParseError("Expected identifier after '.'".to_string()));
                    }
                }

                Ok(SqlExpr::Column(full))
            }

            Token::Count => {
                self.advance();
                self.parse_function_call_from_name("count".to_string())
            }
            Token::Sum => {
                self.advance();
                self.parse_function_call_from_name("sum".to_string())
            }
            Token::Avg => {
                self.advance();
                self.parse_function_call_from_name("avg".to_string())
            }
            Token::Min => {
                self.advance();
                self.parse_function_call_from_name("min".to_string())
            }
            Token::Max => {
                self.advance();
                self.parse_function_call_from_name("max".to_string())
            }
            _ => Err(ApexError::QueryParseError(
                format!("Unexpected token in expression: {:?}", self.current())
            )),
        }
    }

    // ========== DDL Helper Methods ==========

    /// Parse an identifier (table name, column name, etc.)
    fn parse_identifier(&mut self) -> Result<String, ApexError> {
        match self.current().clone() {
            Token::Identifier(s) => {
                self.advance();
                Ok(s)
            }
            _ => {
                let (start, _) = self.current_span();
                Err(self.syntax_error(start, "Expected identifier".to_string()))
            }
        }
    }

    /// Parse IF NOT EXISTS clause
    fn parse_if_not_exists(&mut self) -> Result<bool, ApexError> {
        if matches!(self.current(), Token::If) {
            self.advance();
            self.expect(Token::Not)?;
            self.expect(Token::Exists)?;
            Ok(true)
        } else {
            Ok(false)
        }
    }

    /// Parse IF EXISTS clause
    fn parse_if_exists(&mut self) -> Result<bool, ApexError> {
        if matches!(self.current(), Token::If) {
            self.advance();
            self.expect(Token::Exists)?;
            Ok(true)
        } else {
            Ok(false)
        }
    }

    /// Parse column definitions for CREATE TABLE
    fn parse_column_defs(&mut self) -> Result<Vec<ColumnDef>, ApexError> {
        let mut columns = Vec::new();
        
        loop {
            let name = self.parse_identifier()?;
            let data_type = self.parse_data_type()?;
            columns.push(ColumnDef { name, data_type });
            
            if matches!(self.current(), Token::Comma) {
                self.advance();
            } else {
                break;
            }
        }
        
        Ok(columns)
    }

    /// Parse data type for column definition
    fn parse_data_type(&mut self) -> Result<DataType, ApexError> {
        match self.current().clone() {
            Token::Identifier(s) => {
                self.advance();
                match s.to_uppercase().as_str() {
                    "INT" | "INT64" | "INTEGER" | "BIGINT" => Ok(DataType::Int64),
                    "FLOAT" | "FLOAT64" | "DOUBLE" | "REAL" => Ok(DataType::Float64),
                    "STRING" | "TEXT" | "VARCHAR" => Ok(DataType::String),
                    "BOOL" | "BOOLEAN" => Ok(DataType::Bool),
                    // Bytes type not supported yet, treat as String
                    "BYTES" | "BLOB" | "BINARY" => Ok(DataType::String),
                    _ => {
                        let (start, _) = self.current_span();
                        Err(self.syntax_error(start, format!("Unknown data type: {}", s)))
                    }
                }
            }
            _ => {
                let (start, _) = self.current_span();
                Err(self.syntax_error(start, "Expected data type".to_string()))
            }
        }
    }

    /// Parse ALTER TABLE operation
    fn parse_alter_operation(&mut self) -> Result<AlterTableOp, ApexError> {
        match self.current() {
            Token::Add => {
                self.advance();
                // Optional COLUMN keyword
                if matches!(self.current(), Token::Column) {
                    self.advance();
                }
                let name = self.parse_identifier()?;
                let data_type = self.parse_data_type()?;
                Ok(AlterTableOp::AddColumn { name, data_type })
            }
            Token::Drop => {
                self.advance();
                // Optional COLUMN keyword
                if matches!(self.current(), Token::Column) {
                    self.advance();
                }
                let name = self.parse_identifier()?;
                Ok(AlterTableOp::DropColumn { name })
            }
            Token::Rename => {
                self.advance();
                // Optional COLUMN keyword
                if matches!(self.current(), Token::Column) {
                    self.advance();
                }
                let old_name = self.parse_identifier()?;
                self.expect(Token::To)?;
                let new_name = self.parse_identifier()?;
                Ok(AlterTableOp::RenameColumn { old_name, new_name })
            }
            _ => {
                let (start, _) = self.current_span();
                Err(self.syntax_error(start, "Expected ADD, DROP, or RENAME".to_string()))
            }
        }
    }

    /// Parse comma-separated list of identifiers
    fn parse_identifier_list(&mut self) -> Result<Vec<String>, ApexError> {
        let mut list = Vec::new();
        
        loop {
            list.push(self.parse_identifier()?);
            
            if matches!(self.current(), Token::Comma) {
                self.advance();
            } else {
                break;
            }
        }
        
        Ok(list)
    }

    /// Parse VALUES clause for INSERT
    fn parse_values_list(&mut self) -> Result<Vec<Vec<Value>>, ApexError> {
        let mut rows = Vec::new();
        
        loop {
            self.expect(Token::LParen)?;
            let mut row = Vec::new();
            
            loop {
                let value = self.parse_literal_value()?;
                row.push(value);
                
                if matches!(self.current(), Token::Comma) {
                    self.advance();
                } else {
                    break;
                }
            }
            
            self.expect(Token::RParen)?;
            rows.push(row);
            
            if matches!(self.current(), Token::Comma) {
                self.advance();
            } else {
                break;
            }
        }
        
        Ok(rows)
    }

    /// Parse SET clause for UPDATE (column = value pairs)
    fn parse_assignments(&mut self) -> Result<Vec<(String, SqlExpr)>, ApexError> {
        let mut assignments = Vec::new();
        
        loop {
            let column = self.parse_identifier()?;
            self.expect(Token::Eq)?;
            let value = self.parse_expr()?;
            assignments.push((column, value));
            
            if matches!(self.current(), Token::Comma) {
                self.advance();
            } else {
                break;
            }
        }
        
        Ok(assignments)
    }

}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simple_select() {
        let sql = "SELECT * FROM users";
        let stmt = SqlParser::parse(sql).unwrap();
        let SqlStatement::Select(s) = stmt else {
            panic!("expected select");
        };
        assert!(!s.distinct);
        assert_eq!(s.columns.len(), 1);
        assert!(matches!(s.columns[0], SelectColumn::All));
        assert!(matches!(
            s.from,
            Some(FromItem::Table {
                table,
                alias: None
            }) if table == "users"
        ));
    }

    #[test]
    fn test_select_with_where() {
        let sql = "SELECT name, age FROM users WHERE age > 18 AND name LIKE 'John%'";
        let stmt = SqlParser::parse(sql).unwrap();
        let SqlStatement::Select(s) = stmt else {
            panic!("expected select");
        };
        assert_eq!(s.columns.len(), 2);
        assert!(s.where_clause.is_some());
    }

    #[test]
    fn test_select_with_order_limit() {
        let sql = "SELECT * FROM users ORDER BY age DESC LIMIT 10 OFFSET 5";
        let stmt = SqlParser::parse(sql).unwrap();
        let SqlStatement::Select(s) = stmt else {
            panic!("expected select");
        };
        assert_eq!(s.order_by.len(), 1);
        assert!(s.order_by[0].descending);
        assert_eq!(s.limit, Some(10));
        assert_eq!(s.offset, Some(5));
    }

    #[test]
    fn test_select_qualified_id() {
        let sql = "SELECT default._id, name FROM default ORDER BY default._id";
        let stmt = SqlParser::parse(sql).unwrap();
        let SqlStatement::Select(s) = stmt else {
            panic!("expected select");
        };
        assert_eq!(s.columns.len(), 2);
        match &s.columns[0] {
            SelectColumn::Column(c) => assert_eq!(c, "default._id"),
            other => panic!("unexpected column: {:?}", other),
        }
        match &s.columns[1] {
            SelectColumn::Column(c) => assert_eq!(c, "name"),
            other => panic!("unexpected column: {:?}", other),
        }
        assert_eq!(s.order_by.len(), 1);
        assert_eq!(s.order_by[0].column, "default._id");
    }

    #[test]
    fn test_select_quoted_id() {
        let sql = "SELECT \"_id\", name FROM default ORDER BY \"_id\"";
        let stmt = SqlParser::parse(sql).unwrap();
        let SqlStatement::Select(s) = stmt else {
            panic!("expected select");
        };
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

    #[test]
    fn test_syntax_error_misspelled_select_keyword_selects() {
        let sql = "selects max(_id) from default";
        let err = SqlParser::parse(sql).unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("Syntax error"));
        assert!(msg.contains("did you mean SELECT"));
    }

    #[test]
    fn test_syntax_error_misspelled_join_keyword() {
        let sql = "select * from t1 joinn t2 on t1.id = t2.id";
        let err = SqlParser::parse(sql).unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("Syntax error"));
        assert!(msg.contains("did you mean JOIN"));
    }
}
