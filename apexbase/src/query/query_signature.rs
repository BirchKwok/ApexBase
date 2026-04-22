//! Query Signature Classifier
//!
//! Single-point-of-truth for SQL query classification. All layers (Python client,
//! PyO3 bindings, executor) share this classification — no duplicate pattern matching.
//!
//! The classifier operates on raw SQL text (~2-5µs) and produces a `QuerySignature`
//! enum that determines the optimal execution path. This replaces the previous
//! architecture where 83 fast-path checks were scattered across 4 layers.

/// DDL sub-kind — pre-extracted table name avoids re-uppercasing in bindings.
#[derive(Debug, Clone, PartialEq)]
pub enum DdlKind {
    CreateTable { name: String },
    DropTable { name: String },
    Other,
}

/// Query signature — lightweight classification of a SQL statement.
///
/// Produced ONCE by `classify()`, then consumed by all layers without re-parsing.
/// Variants are ordered from most-specific (cheapest to execute) to least-specific.
#[derive(Debug, Clone, PartialEq)]
pub enum QuerySignature {
    /// `SELECT COUNT(*) FROM <table>` — O(1) metadata read
    CountStar { table: String },

    /// `SELECT * ... WHERE _id = N` — O(1) point lookup by primary key
    PointLookup { id: u64, table: Option<String> },

    /// `SELECT col1, col2 ... WHERE _id = N` — projected point lookup by primary key
    ProjectedPointLookup {
        id: u64,
        table: Option<String>,
        columns: Vec<String>,
    },

    /// `SELECT * ... WHERE _id IN (...)` — batch point lookup by primary key
    IdBatchLookup {
        ids: Vec<u64>,
        table: Option<String>,
    },

    /// `SELECT col1, col2 ... WHERE _id IN (...)` — projected batch primary-key lookup
    ProjectedIdBatchLookup {
        ids: Vec<u64>,
        table: Option<String>,
        columns: Vec<String>,
    },

    /// `SELECT * FROM <table>` — full table scan without parser/planner overhead
    FullScan { table: Option<String> },

    /// `SELECT col1, col2 FROM <table>` — projected full scan without parser/planner overhead
    ProjectedFullScan {
        table: Option<String>,
        columns: Vec<String>,
    },

    /// `SELECT * FROM <table> LIMIT N` — sequential scan with early termination
    SimpleScanLimit { limit: usize, table: Option<String> },

    /// `SELECT col1, col2 FROM <table> LIMIT N` — projected sequential scan
    ProjectedScanLimit {
        limit: usize,
        table: Option<String>,
        columns: Vec<String>,
    },

    /// `SELECT * FROM <table> WHERE col = 'val'` — mmap string equality scan
    StringEqualityFilter {
        table: Option<String>,
        column: String,
        value: String,
    },

    /// `SELECT * FROM <table> WHERE col = 'val' LIMIT N [OFFSET M]` — early-terminating string equality scan
    StringEqualityFilterLimit {
        table: Option<String>,
        column: String,
        value: String,
        limit: usize,
        offset: usize,
    },

    /// `SELECT col1, col2 FROM <table> WHERE filter_col = 'val'` — projected string equality scan
    ProjectedStringEqualityFilter {
        table: Option<String>,
        columns: Vec<String>,
        column: String,
        value: String,
    },

    /// `SELECT col1, col2 FROM <table> WHERE filter_col = 'val' LIMIT N [OFFSET M]` — projected early-terminating string equality scan
    ProjectedStringEqualityFilterLimit {
        table: Option<String>,
        columns: Vec<String>,
        column: String,
        value: String,
        limit: usize,
        offset: usize,
    },

    /// `SELECT * FROM <table> WHERE col LIKE 'pattern'` — mmap LIKE scan
    LikeFilter {
        table: Option<String>,
        column: String,
        pattern: String,
    },

    /// `SELECT ... FROM read_csv(...) / read_parquet(...) / read_json(...)` — table function
    TableFunction,

    /// DDL: CREATE TABLE, DROP TABLE, ALTER TABLE, CREATE INDEX, etc.
    Ddl { kind: DdlKind },

    /// DML write: INSERT, UPDATE, DELETE, TRUNCATE, COPY IMPORT
    DmlWrite,

    /// Transaction control: BEGIN, COMMIT, ROLLBACK, SAVEPOINT, RELEASE
    Transaction,

    /// Multi-statement SQL (contains ';' separator)
    MultiStatement,

    /// SET / RESET variable
    SessionCommand,

    /// EXPLAIN / ANALYZE
    Explain,

    /// CTE: WITH ... AS (...)
    Cte,

    /// Everything else — full parse + plan + execute
    Complex,
}

impl QuerySignature {
    /// Returns true if this signature represents a read-only query.
    #[inline]
    pub fn is_read_only(&self) -> bool {
        matches!(
            self,
            QuerySignature::CountStar { .. }
                | QuerySignature::PointLookup { .. }
                | QuerySignature::ProjectedPointLookup { .. }
                | QuerySignature::IdBatchLookup { .. }
                | QuerySignature::ProjectedIdBatchLookup { .. }
                | QuerySignature::FullScan { .. }
                | QuerySignature::ProjectedFullScan { .. }
                | QuerySignature::SimpleScanLimit { .. }
                | QuerySignature::ProjectedScanLimit { .. }
                | QuerySignature::StringEqualityFilter { .. }
                | QuerySignature::StringEqualityFilterLimit { .. }
                | QuerySignature::ProjectedStringEqualityFilter { .. }
                | QuerySignature::ProjectedStringEqualityFilterLimit { .. }
                | QuerySignature::LikeFilter { .. }
                | QuerySignature::TableFunction
                | QuerySignature::Explain
                | QuerySignature::Cte
                | QuerySignature::Complex
        )
    }

    /// Returns true if this is a write operation that needs locking.
    #[inline]
    pub fn needs_write_lock(&self) -> bool {
        matches!(self, QuerySignature::DmlWrite)
    }

    /// Returns true if the Python layer needs to hold its threading lock.
    #[inline]
    pub fn needs_python_lock(&self) -> bool {
        matches!(
            self,
            QuerySignature::DmlWrite
                | QuerySignature::Ddl { .. }
                | QuerySignature::Transaction
                | QuerySignature::MultiStatement
        )
    }

    /// Returns true if this signature can bypass SQL parsing entirely.
    #[inline]
    pub fn can_skip_parse(&self) -> bool {
        matches!(
            self,
            QuerySignature::CountStar { .. }
                | QuerySignature::PointLookup { .. }
                | QuerySignature::ProjectedPointLookup { .. }
                | QuerySignature::IdBatchLookup { .. }
                | QuerySignature::ProjectedIdBatchLookup { .. }
                | QuerySignature::FullScan { .. }
                | QuerySignature::ProjectedFullScan { .. }
                | QuerySignature::SimpleScanLimit { .. }
                | QuerySignature::ProjectedScanLimit { .. }
                | QuerySignature::StringEqualityFilter { .. }
                | QuerySignature::StringEqualityFilterLimit { .. }
                | QuerySignature::ProjectedStringEqualityFilter { .. }
                | QuerySignature::ProjectedStringEqualityFilterLimit { .. }
                | QuerySignature::LikeFilter { .. }
        )
    }
}

/// Classify a SQL string into a `QuerySignature`.
///
/// This is the SINGLE point where SQL text pattern matching happens.
/// Cost: ~2-5µs (one uppercase pass + a handful of prefix/contains checks).
///
/// The function takes a pre-computed uppercase SQL to avoid redundant allocations
/// when the caller already has it.
pub fn classify(sql: &str) -> QuerySignature {
    let s = sql.trim();
    // We work on a bounded prefix for safety. 4 KiB comfortably covers common
    // `IN (...)` lookup lists from benchmarks while keeping classification cheap.
    let upper_buf: String;
    let su = if s.len() <= 4096 {
        upper_buf = s.to_ascii_uppercase();
        &upper_buf
    } else {
        // For very long SQL, only uppercase the first 4 KiB for classification
        upper_buf = s[..4096].to_ascii_uppercase();
        &upper_buf
    };

    // ── Multi-statement detection (must be first — overrides everything) ──
    {
        let trimmed = s.trim_end_matches(';').trim();
        if trimmed.contains(';') {
            return QuerySignature::MultiStatement;
        }
    }

    // ── Transaction commands ──
    if su.starts_with("BEGIN")
        || su == "COMMIT"
        || su == "COMMIT;"
        || su == "ROLLBACK"
        || su == "ROLLBACK;"
        || su.starts_with("SAVEPOINT ")
        || su.starts_with("ROLLBACK TO")
        || su.starts_with("RELEASE")
    {
        return QuerySignature::Transaction;
    }

    // ── Session commands ──
    if su.starts_with("SET ") || su.starts_with("RESET ") {
        return QuerySignature::SessionCommand;
    }

    // ── DDL ──
    if su.starts_with("CREATE ") || su.starts_with("DROP ") || su.starts_with("ALTER ") {
        return QuerySignature::Ddl {
            kind: extract_ddl_kind(s, su),
        };
    }

    // ── DML writes ──
    if su.starts_with("INSERT")
        || su.starts_with("UPDATE")
        || su.starts_with("DELETE")
        || su.starts_with("TRUNCATE")
    {
        // Special case: DELETE can still use pre-parse fast path inside executor
        return QuerySignature::DmlWrite;
    }

    // ── COPY (can be read or write) ──
    if su.starts_with("COPY ") {
        // COPY ... FROM is a write, COPY ... TO is a read
        return QuerySignature::DmlWrite;
    }

    // ── EXPLAIN ──
    if su.starts_with("EXPLAIN") {
        return QuerySignature::Explain;
    }

    // ── CTE ──
    if su.starts_with("WITH ") {
        return QuerySignature::Cte;
    }

    // ── REINDEX / PRAGMA ──
    if su.starts_with("REINDEX") || su.starts_with("PRAGMA") {
        return QuerySignature::Ddl {
            kind: DdlKind::Other,
        };
    }

    // ── SHOW / FTS DDL ──
    if su.starts_with("SHOW ") {
        return QuerySignature::Ddl {
            kind: DdlKind::Other,
        };
    }

    // ── SELECT queries — classify further ──
    if !su.starts_with("SELECT") {
        return QuerySignature::Complex;
    }

    if contains_unquoted_keyword(s, "UNION")
        || contains_unquoted_keyword(s, "INTERSECT")
        || contains_unquoted_keyword(s, "EXCEPT")
    {
        return QuerySignature::Complex;
    }

    // Guard flags for modifier keywords
    let has_where = su.contains("WHERE");
    let has_group = su.contains("GROUP");
    let has_having = su.contains("HAVING");
    let has_join = su.contains("JOIN");
    let has_order = su.contains("ORDER");
    let has_limit = su.contains("LIMIT");
    let has_distinct = su.contains("DISTINCT");

    // ── Table function: FROM READ_CSV / READ_PARQUET / READ_JSON ──
    if su.contains("FROM READ_CSV(")
        || su.contains("FROM READ_PARQUET(")
        || su.contains("FROM READ_JSON(")
    {
        return QuerySignature::TableFunction;
    }

    // ── COUNT(*) — no WHERE/GROUP/HAVING/JOIN/DISTINCT ──
    if su.starts_with("SELECT COUNT(*) FROM ")
        && !has_where
        && !has_group
        && !has_having
        && !has_join
        && !has_distinct
    {
        let after_from = su["SELECT COUNT(*) FROM ".len()..].trim();
        let tname = after_from.trim_end_matches(';').trim();
        if !tname.is_empty() && !tname.contains(' ') {
            return QuerySignature::CountStar {
                table: tname.to_lowercase(),
            };
        }
    }

    let is_exact_star_select = has_exact_star_projection(s, su);
    let simple_projection = extract_simple_projection_columns(s, su);

    if let Some(columns) = simple_projection.clone() {
        if !has_limit && !has_order && !has_group && !has_join {
            if let Some(id) = extract_simple_id_equality(s, su) {
                let table = extract_from_table(s, su);
                return QuerySignature::ProjectedPointLookup { id, table, columns };
            }
        }

        if !has_limit && !has_order && !has_group && !has_join {
            if let Some(ids) = extract_simple_id_in_list(s, su) {
                let table = extract_from_table(s, su);
                return QuerySignature::ProjectedIdBatchLookup {
                    ids,
                    table,
                    columns,
                };
            }
        }

        if has_limit && !has_where && !has_order && !has_group && !has_join {
            if let Some(limit) = extract_limit_from_upper(su) {
                return QuerySignature::ProjectedScanLimit {
                    limit,
                    table: extract_from_table(s, su),
                    columns,
                };
            }
        }

        if has_where
            && !has_order
            && !has_group
            && !has_join
            && !su.contains("BETWEEN")
            && !su.contains(" IN ")
            && !su.contains('>')
            && !su.contains('<')
            && !su.contains(" LIKE ")
            && s.contains('\'')
        {
            if has_limit {
                if let Some((column, value, limit, offset)) =
                    extract_string_equality_with_limit(s, su)
                {
                    return QuerySignature::ProjectedStringEqualityFilterLimit {
                        table: extract_from_table(s, su),
                        columns,
                        column,
                        value,
                        limit,
                        offset,
                    };
                }
            }
            if let Some((column, value)) = extract_string_equality(s, su) {
                return QuerySignature::ProjectedStringEqualityFilter {
                    table: extract_from_table(s, su),
                    columns,
                    column,
                    value,
                };
            }
        }

        if !has_where && !has_order && !has_group && !has_join && !has_limit && !has_distinct {
            let table = extract_from_table(s, su);
            return QuerySignature::ProjectedFullScan { table, columns };
        }
    }

    // ── Point lookup: SELECT * ... WHERE _ID = N ──
    if is_exact_star_select && !has_limit && !has_order && !has_group && !has_join {
        if let Some(id) = extract_simple_id_equality(s, su) {
            let table = extract_from_table(s, su);
            return QuerySignature::PointLookup { id, table };
        }
    }

    // ── Batch point lookup: SELECT * ... WHERE _ID IN (...) ──
    if is_exact_star_select && !has_limit && !has_order && !has_group && !has_join {
        if let Some(ids) = extract_simple_id_in_list(s, su) {
            let table = extract_from_table(s, su);
            return QuerySignature::IdBatchLookup { ids, table };
        }
    }

    // ── Simple scan: SELECT * ... LIMIT N (no WHERE/ORDER/GROUP/JOIN) ──
    if is_exact_star_select && has_limit && !has_where && !has_order && !has_group && !has_join {
        if let Some(limit) = extract_limit_from_upper(su) {
            return QuerySignature::SimpleScanLimit {
                limit,
                table: extract_from_table(s, su),
            };
        }
    }

    // ── String equality: SELECT * ... WHERE col = 'val' (no LIMIT/ORDER/GROUP/JOIN/BETWEEN/IN) ──
    if is_exact_star_select
        && has_where
        && !has_order
        && !has_group
        && !has_join
        && !su.contains("BETWEEN")
        && !su.contains(" IN ")
        && !su.contains('>')
        && !su.contains('<')
        && !su.contains(" LIKE ")
        && s.contains('\'')
    {
        if has_limit {
            if let Some((col, val, limit, offset)) = extract_string_equality_with_limit(s, su) {
                return QuerySignature::StringEqualityFilterLimit {
                    table: extract_from_table(s, su),
                    column: col,
                    value: val,
                    limit,
                    offset,
                };
            }
        }
        if let Some((col, val)) = extract_string_equality(s, su) {
            return QuerySignature::StringEqualityFilter {
                table: extract_from_table(s, su),
                column: col,
                value: val,
            };
        }
    }

    // ── LIKE filter: SELECT * ... WHERE col LIKE 'pattern' (simple, no AND/OR/NOT) ──
    if is_exact_star_select
        && su.contains(" LIKE ")
        && has_where
        && !su.contains("NOT LIKE")
        && !has_limit
        && !has_order
        && !has_group
        && !has_join
        && !su.contains(" AND ")
        && !su.contains(" OR ")
        && s.contains('\'')
    {
        if let Some((col, pattern)) = extract_like_pattern(s, su) {
            return QuerySignature::LikeFilter {
                table: extract_from_table(s, su),
                column: col,
                pattern,
            };
        }
    }

    // ── Full scan: SELECT * FROM <table> (no WHERE/LIMIT/ORDER/GROUP/JOIN/DISTINCT) ──
    if is_exact_star_select
        && !has_where
        && !has_order
        && !has_group
        && !has_join
        && !has_limit
        && !has_distinct
    {
        let table = extract_from_table(s, su);
        return QuerySignature::FullScan { table };
    }

    QuerySignature::Complex
}

/// Extract the integer ID from a simple `WHERE _id = N` clause.
/// Returns None when the WHERE clause contains anything beyond the equality.
fn extract_simple_id_equality(sql: &str, su: &str) -> Option<u64> {
    let where_pos = su.find("WHERE")?;
    let after_where = sql[where_pos + 5..].trim().trim_end_matches(';').trim();
    let after_where_upper = after_where.to_ascii_uppercase();

    let id_prefix = if after_where_upper.starts_with("_ID =") {
        "_ID ="
    } else if after_where_upper.starts_with("_ID=") {
        "_ID="
    } else if after_where_upper.starts_with("\"_ID\" =") {
        "\"_ID\" ="
    } else if after_where_upper.starts_with("\"_ID\"=") {
        "\"_ID\"="
    } else {
        return None;
    };

    let rhs = after_where[id_prefix.len()..].trim_start();
    let num_end = rhs.find(|c: char| !c.is_ascii_digit()).unwrap_or(rhs.len());
    if num_end == 0 {
        return None;
    }
    let rest = rhs[num_end..].trim();
    if !rest.is_empty() {
        return None;
    }
    rhs[..num_end].parse::<u64>().ok()
}

/// Extract IDs from a simple `WHERE _id IN (...)` clause.
/// Returns None when the WHERE clause contains anything beyond the IN list.
fn extract_simple_id_in_list(sql: &str, su: &str) -> Option<Vec<u64>> {
    let where_pos = su.find("WHERE")?;
    let after_where = sql[where_pos + 5..].trim().trim_end_matches(';').trim();
    let after_where_upper = after_where.to_ascii_uppercase();

    let id_prefix = if after_where_upper.starts_with("_ID IN") {
        "_ID IN"
    } else if after_where_upper.starts_with("\"_ID\" IN") {
        "\"_ID\" IN"
    } else {
        return None;
    };

    let rhs = after_where[id_prefix.len()..].trim_start();
    if !rhs.starts_with('(') {
        return None;
    }
    let end_pos = rhs.find(')')?;
    let list = rhs[1..end_pos].trim();
    let rest = rhs[end_pos + 1..].trim();
    if !rest.is_empty() || list.is_empty() {
        return None;
    }

    let mut ids = Vec::new();
    for part in list.split(',') {
        let id = part.trim().parse::<u64>().ok()?;
        ids.push(id);
    }
    if ids.is_empty() {
        return None;
    }
    Some(ids)
}

/// Extract LIMIT value from uppercased SQL.
fn extract_limit_from_upper(su: &str) -> Option<usize> {
    let after_limit = su.rsplit("LIMIT").next()?;
    after_limit
        .trim()
        .trim_end_matches(';')
        .parse::<usize>()
        .ok()
}

/// Extract a simple comma-separated projection list containing only plain column references.
/// Rejects `*`, expressions, aliases, and mixed `table.*` forms.
fn extract_simple_projection_columns(sql: &str, su: &str) -> Option<Vec<String>> {
    if !su.starts_with("SELECT") {
        return None;
    }
    let from_pos = su.find(" FROM ")?;
    let projection = sql["SELECT".len()..from_pos].trim();
    if projection.is_empty() || projection == "*" {
        return None;
    }

    let mut columns = Vec::new();
    let mut seen = std::collections::HashSet::new();
    for raw_part in projection.split(',') {
        let raw = raw_part.trim();
        if raw.is_empty() {
            return None;
        }
        let raw_upper = raw.to_ascii_uppercase();
        if raw == "*"
            || raw.ends_with(".*")
            || raw.contains('(')
            || raw.contains(')')
            || raw.contains('+')
            || raw.contains('-')
            || raw.contains('/')
            || raw.contains('\'')
            || raw_upper.contains(" AS ")
            || raw.chars().any(|c| c.is_whitespace())
        {
            return None;
        }

        let normalized = raw
            .rsplit('.')
            .next()?
            .trim_matches(|c| c == '"' || c == '`');
        if normalized.is_empty() || normalized == "*" {
            return None;
        }
        if !seen.insert(normalized.to_string()) {
            return None;
        }
        columns.push(normalized.to_string());
    }

    if columns.is_empty() {
        None
    } else {
        Some(columns)
    }
}

/// Returns true only for exact `SELECT * FROM ...` projections.
/// Rejects forms like `SELECT *, _id ...`, `SELECT _id, * ...`, `SELECT t.* ...`.
fn has_exact_star_projection(sql: &str, su: &str) -> bool {
    if !su.starts_with("SELECT") {
        return false;
    }
    let from_pos = match su.find(" FROM ") {
        Some(pos) => pos,
        None => return false,
    };
    let projection = sql["SELECT".len()..from_pos].trim();
    projection == "*"
}

/// Extract (column, value) from `WHERE col = 'val'` in original-case SQL,
/// guided by the uppercased version for keyword positions.
fn extract_string_equality(sql: &str, su: &str) -> Option<(String, String)> {
    let where_pos = su.find("WHERE")?;
    let after_where = sql[where_pos + 5..].trim().trim_end_matches(';');
    parse_string_equality_clause(after_where)
}

/// Extract (column, value, limit, offset) from
/// `WHERE col = 'val' LIMIT N [OFFSET M]`.
fn extract_string_equality_with_limit(
    sql: &str,
    su: &str,
) -> Option<(String, String, usize, usize)> {
    let where_pos = su.find("WHERE")?;
    let limit_pos = su.rfind("LIMIT")?;
    if limit_pos <= where_pos {
        return None;
    }

    let where_clause = sql[where_pos + 5..limit_pos].trim();
    let (column, value) = parse_string_equality_clause(where_clause)?;
    let after_limit = sql[limit_pos + "LIMIT".len()..]
        .trim()
        .trim_end_matches(';')
        .trim();
    let (limit, offset) = parse_limit_offset_clause(after_limit)?;
    Some((column, value, limit, offset))
}

fn parse_string_equality_clause(clause: &str) -> Option<(String, String)> {
    let after_where = clause.trim();
    let eq_pos = after_where.find('=')?;
    let col = after_where[..eq_pos].trim().trim_matches('"').to_string();
    if col.contains(' ') || col.contains('(') {
        return None;
    }
    let rhs = after_where[eq_pos + 1..].trim();
    if !rhs.starts_with('\'') {
        return None;
    }
    let val_end = rhs[1..].find('\'')?;
    let val = rhs[1..1 + val_end].to_string();
    let rest = rhs[1 + val_end + 1..].trim();
    if !rest.is_empty() {
        return None;
    }
    Some((col, val))
}

fn parse_limit_offset_clause(clause: &str) -> Option<(usize, usize)> {
    let mut parts = clause.split_whitespace();
    let limit = parts.next()?.parse::<usize>().ok()?;
    let offset = match parts.next() {
        None => 0,
        Some(keyword) if keyword.eq_ignore_ascii_case("OFFSET") => {
            let parsed = parts.next()?.parse::<usize>().ok()?;
            if parts.next().is_some() {
                return None;
            }
            parsed
        }
        Some(_) => return None,
    };
    Some((limit, offset))
}

/// Extract table name from `FROM <table>` clause using original-case SQL + uppercased version.
fn extract_from_table(sql: &str, su: &str) -> Option<String> {
    let fp = su.find(" FROM ")?;
    let after_from = sql[fp + 6..].trim_start();
    let tn_end = after_from
        .find(|c: char| c.is_whitespace() || c == ';')
        .unwrap_or(after_from.len());
    let tname = after_from[..tn_end].trim_matches('"').to_lowercase();
    if tname.is_empty() {
        None
    } else {
        Some(tname)
    }
}

/// Extract DDL sub-kind from original-case SQL + uppercased version.
/// Extracts table name for CREATE TABLE / DROP TABLE; returns Other for everything else.
fn extract_ddl_kind(sql: &str, su: &str) -> DdlKind {
    if su.starts_with("CREATE TABLE") {
        let rest = &sql["CREATE TABLE".len()..].trim_start();
        // Skip "IF NOT EXISTS"
        let rest = if rest.len() >= 13 && rest[..13].eq_ignore_ascii_case("IF NOT EXISTS") {
            rest[13..].trim_start()
        } else {
            rest
        };
        if let Some(name) = rest
            .split(|c: char| c.is_whitespace() || c == '(' || c == ';')
            .next()
        {
            let tbl = name
                .trim_matches(|c: char| c == '"' || c == '\'' || c == '`')
                .to_lowercase();
            if !tbl.is_empty() {
                return DdlKind::CreateTable { name: tbl };
            }
        }
    } else if su.starts_with("DROP TABLE") {
        let rest = &sql["DROP TABLE".len()..].trim_start();
        // Skip "IF EXISTS"
        let rest = if rest.len() >= 9 && rest[..9].eq_ignore_ascii_case("IF EXISTS") {
            rest[9..].trim_start()
        } else {
            rest
        };
        if let Some(name) = rest.split(|c: char| c.is_whitespace() || c == ';').next() {
            let tbl = name
                .trim_matches(|c: char| c == '"' || c == '\'' || c == '`')
                .to_lowercase();
            if !tbl.is_empty() {
                return DdlKind::DropTable { name: tbl };
            }
        }
    }
    DdlKind::Other
}

/// Extract (column, pattern) from `WHERE col LIKE 'pattern'`.
fn extract_like_pattern(sql: &str, su: &str) -> Option<(String, String)> {
    let where_pos = su.find("WHERE")?;
    let after_where = sql[where_pos + 5..].trim().trim_end_matches(';');
    let after_where_upper = after_where.to_uppercase();
    let like_pos = after_where_upper.find(" LIKE ")?;
    let col = after_where[..like_pos].trim().trim_matches('"').to_string();
    if col.contains(' ') || col.contains('(') {
        return None;
    }
    let rhs = after_where[like_pos + 6..].trim();
    if !rhs.starts_with('\'') {
        return None;
    }
    let val_end = rhs[1..].find('\'')?;
    let pattern = rhs[1..1 + val_end].to_string();
    let rest = rhs[1 + val_end + 1..].trim();
    if !rest.is_empty() {
        return None;
    }
    Some((col, pattern))
}

fn contains_unquoted_keyword(sql: &str, keyword: &str) -> bool {
    let bytes = sql.as_bytes();
    let keyword_bytes = keyword.as_bytes();
    let kw_len = keyword_bytes.len();
    let mut i = 0usize;
    let mut in_single_quote = false;

    while i < bytes.len() {
        let b = bytes[i];
        if b == b'\'' {
            if in_single_quote && i + 1 < bytes.len() && bytes[i + 1] == b'\'' {
                i += 2;
                continue;
            }
            in_single_quote = !in_single_quote;
            i += 1;
            continue;
        }

        if !in_single_quote
            && i + kw_len <= bytes.len()
            && bytes[i..i + kw_len].eq_ignore_ascii_case(keyword_bytes)
        {
            let prev_ok = i == 0 || !is_ident_byte(bytes[i - 1]);
            let next_ok = i + kw_len == bytes.len() || !is_ident_byte(bytes[i + kw_len]);
            if prev_ok && next_ok {
                return true;
            }
        }

        i += 1;
    }

    false
}

#[inline]
fn is_ident_byte(b: u8) -> bool {
    b.is_ascii_alphanumeric() || b == b'_'
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_count_star() {
        assert_eq!(
            classify("SELECT COUNT(*) FROM users"),
            QuerySignature::CountStar {
                table: "users".to_string()
            }
        );
        // With WHERE — should be Complex
        assert_eq!(
            classify("SELECT COUNT(*) FROM users WHERE age > 10"),
            QuerySignature::Complex
        );
    }

    #[test]
    fn test_point_lookup() {
        assert_eq!(
            classify("SELECT * FROM t WHERE _id = 42"),
            QuerySignature::PointLookup {
                id: 42,
                table: Some("t".to_string())
            }
        );
        assert_eq!(
            classify("SELECT * FROM t WHERE _id=100"),
            QuerySignature::PointLookup {
                id: 100,
                table: Some("t".to_string())
            }
        );
        assert_eq!(
            classify("SELECT name FROM t WHERE _id = 42"),
            QuerySignature::Complex
        );
        assert_eq!(
            classify("SELECT * FROM t WHERE _id = 42 AND age = 1"),
            QuerySignature::Complex
        );
        assert_eq!(
            classify("SELECT name, age FROM t WHERE _id = 42"),
            QuerySignature::ProjectedPointLookup {
                id: 42,
                table: Some("t".to_string()),
                columns: vec!["name".to_string(), "age".to_string()],
            }
        );
    }

    #[test]
    fn test_id_batch_lookup() {
        assert_eq!(
            classify("SELECT * FROM t WHERE _id IN (1, 5, 9)"),
            QuerySignature::IdBatchLookup {
                ids: vec![1, 5, 9],
                table: Some("t".to_string()),
            }
        );
        assert_eq!(
            classify("SELECT * FROM t WHERE _id IN (1, 5, 9) AND age > 1"),
            QuerySignature::Complex
        );
        assert_eq!(
            classify("SELECT name FROM t WHERE _id IN (1, 5, 9)"),
            QuerySignature::ProjectedIdBatchLookup {
                ids: vec![1, 5, 9],
                table: Some("t".to_string()),
                columns: vec!["name".to_string()],
            }
        );
    }

    #[test]
    fn test_simple_scan_limit() {
        assert_eq!(
            classify("SELECT * FROM t LIMIT 100"),
            QuerySignature::SimpleScanLimit {
                limit: 100,
                table: Some("t".to_string()),
            }
        );
        // With WHERE — not a simple scan
        assert_eq!(
            classify("SELECT * FROM t WHERE x > 1 LIMIT 100"),
            QuerySignature::Complex
        );
        assert_eq!(
            classify("SELECT name, age FROM t LIMIT 100"),
            QuerySignature::ProjectedScanLimit {
                limit: 100,
                table: Some("t".to_string()),
                columns: vec!["name".to_string(), "age".to_string()],
            }
        );
    }

    #[test]
    fn test_full_scan() {
        assert_eq!(
            classify("SELECT * FROM t"),
            QuerySignature::FullScan {
                table: Some("t".to_string())
            }
        );
        assert_eq!(classify("SELECT *, _id FROM t"), QuerySignature::Complex);
        assert_eq!(
            classify("SELECT name, age FROM t"),
            QuerySignature::ProjectedFullScan {
                table: Some("t".to_string()),
                columns: vec!["name".to_string(), "age".to_string()],
            }
        );
        assert_eq!(
            classify("SELECT name FROM t UNION ALL SELECT name FROM t"),
            QuerySignature::Complex
        );
    }

    #[test]
    fn test_string_equality() {
        assert_eq!(
            classify("SELECT * FROM t WHERE city = 'NYC'"),
            QuerySignature::StringEqualityFilter {
                table: Some("t".to_string()),
                column: "city".to_string(),
                value: "NYC".to_string(),
            }
        );
        assert_eq!(
            classify("SELECT * FROM t WHERE city = 'NYC' AND age = 20"),
            QuerySignature::Complex
        );
        assert_eq!(
            classify("SELECT name FROM t WHERE city = 'NYC'"),
            QuerySignature::ProjectedStringEqualityFilter {
                table: Some("t".to_string()),
                columns: vec!["name".to_string()],
                column: "city".to_string(),
                value: "NYC".to_string(),
            }
        );
        assert_eq!(
            classify("SELECT * FROM t WHERE city = 'NYC' LIMIT 1"),
            QuerySignature::StringEqualityFilterLimit {
                table: Some("t".to_string()),
                column: "city".to_string(),
                value: "NYC".to_string(),
                limit: 1,
                offset: 0,
            }
        );
        assert_eq!(
            classify("SELECT name FROM t WHERE city = 'NYC' LIMIT 5 OFFSET 2"),
            QuerySignature::ProjectedStringEqualityFilterLimit {
                table: Some("t".to_string()),
                columns: vec!["name".to_string()],
                column: "city".to_string(),
                value: "NYC".to_string(),
                limit: 5,
                offset: 2,
            }
        );
    }

    #[test]
    fn test_like_filter() {
        assert_eq!(
            classify("SELECT * FROM t WHERE name LIKE '%smith%'"),
            QuerySignature::LikeFilter {
                table: Some("t".to_string()),
                column: "name".to_string(),
                pattern: "%smith%".to_string(),
            }
        );
    }

    #[test]
    fn test_table_function() {
        assert_eq!(
            classify("SELECT * FROM read_csv('/tmp/data.csv')"),
            QuerySignature::TableFunction
        );
    }

    #[test]
    fn test_ddl() {
        assert_eq!(
            classify("CREATE TABLE t (id INT)"),
            QuerySignature::Ddl {
                kind: DdlKind::CreateTable {
                    name: "t".to_string()
                }
            }
        );
        assert_eq!(
            classify("DROP TABLE t"),
            QuerySignature::Ddl {
                kind: DdlKind::DropTable {
                    name: "t".to_string()
                }
            }
        );
        assert!(matches!(
            classify("ALTER TABLE t ADD COLUMN x INT"),
            QuerySignature::Ddl {
                kind: DdlKind::Other
            }
        ));
    }

    #[test]
    fn test_dml_write() {
        assert_eq!(
            classify("INSERT INTO t VALUES (1)"),
            QuerySignature::DmlWrite
        );
        assert_eq!(
            classify("DELETE FROM t WHERE id = 1"),
            QuerySignature::DmlWrite
        );
        assert_eq!(classify("UPDATE t SET x = 1"), QuerySignature::DmlWrite);
    }

    #[test]
    fn test_transaction() {
        assert_eq!(classify("BEGIN"), QuerySignature::Transaction);
        assert_eq!(classify("COMMIT"), QuerySignature::Transaction);
        assert_eq!(classify("ROLLBACK"), QuerySignature::Transaction);
        assert_eq!(classify("SAVEPOINT sp1"), QuerySignature::Transaction);
    }

    #[test]
    fn test_multi_statement() {
        assert_eq!(
            classify("INSERT INTO t VALUES (1); SELECT * FROM t"),
            QuerySignature::MultiStatement
        );
    }

    #[test]
    fn test_cte() {
        assert_eq!(
            classify("WITH cte AS (SELECT 1) SELECT * FROM cte"),
            QuerySignature::Cte
        );
    }

    #[test]
    fn test_complex() {
        assert_eq!(
            classify("SELECT a, b FROM t WHERE x > 1 ORDER BY a LIMIT 10"),
            QuerySignature::Complex
        );
        assert_eq!(
            classify("SELECT * FROM t JOIN u ON t.id = u.id"),
            QuerySignature::Complex
        );
    }

    #[test]
    fn test_session_command() {
        assert_eq!(classify("SET x = 1"), QuerySignature::SessionCommand);
        assert_eq!(classify("RESET x"), QuerySignature::SessionCommand);
    }
}
