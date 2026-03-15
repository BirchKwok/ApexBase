//! Query Signature Classifier
//!
//! Single-point-of-truth for SQL query classification. All layers (Python client,
//! PyO3 bindings, executor) share this classification — no duplicate pattern matching.
//!
//! The classifier operates on raw SQL text (~2-5µs) and produces a `QuerySignature`
//! enum that determines the optimal execution path. This replaces the previous
//! architecture where 83 fast-path checks were scattered across 4 layers.

/// Query signature — lightweight classification of a SQL statement.
///
/// Produced ONCE by `classify()`, then consumed by all layers without re-parsing.
/// Variants are ordered from most-specific (cheapest to execute) to least-specific.
#[derive(Debug, Clone, PartialEq)]
pub enum QuerySignature {
    /// `SELECT COUNT(*) FROM <table>` — O(1) metadata read
    CountStar { table: String },

    /// `SELECT ... WHERE _id = N` — O(1) point lookup by primary key
    PointLookup { id: u64 },

    /// `SELECT * FROM <table> LIMIT N` — sequential scan with early termination
    SimpleScanLimit { limit: usize },

    /// `SELECT * FROM <table> WHERE col = 'val'` — mmap string equality scan
    StringEqualityFilter { column: String, value: String },

    /// `SELECT * FROM <table> WHERE col LIKE 'pattern'` — mmap LIKE scan
    LikeFilter { column: String, pattern: String },

    /// `SELECT ... FROM read_csv(...) / read_parquet(...) / read_json(...)` — table function
    TableFunction,

    /// DDL: CREATE TABLE, DROP TABLE, ALTER TABLE, CREATE INDEX, etc.
    Ddl,

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
                | QuerySignature::SimpleScanLimit { .. }
                | QuerySignature::StringEqualityFilter { .. }
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
                | QuerySignature::Ddl
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
                | QuerySignature::SimpleScanLimit { .. }
                | QuerySignature::StringEqualityFilter { .. }
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
    // We work on a bounded prefix for safety — no query keyword detection needs >300 chars
    let upper_buf: String;
    let su = if s.len() <= 512 {
        upper_buf = s.to_ascii_uppercase();
        &upper_buf
    } else {
        // For very long SQL, only uppercase the first 512 chars for classification
        upper_buf = s[..512].to_ascii_uppercase();
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
        || su == "COMMIT" || su == "COMMIT;"
        || su == "ROLLBACK" || su == "ROLLBACK;"
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
        return QuerySignature::Ddl;
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
        return QuerySignature::Ddl;
    }

    // ── SHOW / FTS DDL ──
    if su.starts_with("SHOW ") {
        return QuerySignature::Ddl;
    }

    // ── SELECT queries — classify further ──
    if !su.starts_with("SELECT") {
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
        && !has_where && !has_group && !has_having && !has_join && !has_distinct
    {
        let after_from = su["SELECT COUNT(*) FROM ".len()..].trim();
        let tname = after_from.trim_end_matches(';').trim();
        if !tname.is_empty() && !tname.contains(' ') {
            return QuerySignature::CountStar {
                table: tname.to_lowercase(),
            };
        }
    }

    // ── Point lookup: WHERE _ID = N ──
    if (su.contains("WHERE _ID =") || su.contains("WHERE _ID="))
        && !has_limit && !has_order && !has_group && !has_join
    {
        if let Some(id) = extract_id_from_upper(su) {
            return QuerySignature::PointLookup { id };
        }
    }

    // ── Simple scan: SELECT * ... LIMIT N (no WHERE/ORDER/GROUP/JOIN) ──
    if su.starts_with("SELECT *") && has_limit
        && !has_where && !has_order && !has_group && !has_join
    {
        if let Some(limit) = extract_limit_from_upper(su) {
            return QuerySignature::SimpleScanLimit { limit };
        }
    }

    // ── String equality: SELECT * ... WHERE col = 'val' (no LIMIT/ORDER/GROUP/JOIN/BETWEEN/IN) ──
    if su.starts_with("SELECT *") && has_where
        && !has_limit && !has_order && !has_group && !has_join
        && !su.contains("BETWEEN") && !su.contains(" IN ")
        && !su.contains('>') && !su.contains('<')
        && !su.contains(" LIKE ")
        && sql.contains('\'')
    {
        if let Some((col, val)) = extract_string_equality(sql, su) {
            return QuerySignature::StringEqualityFilter { column: col, value: val };
        }
    }

    // ── LIKE filter: SELECT * ... WHERE col LIKE 'pattern' (simple, no AND/OR/NOT) ──
    if su.starts_with("SELECT *") && su.contains(" LIKE ") && has_where
        && !su.contains("NOT LIKE")
        && !has_limit && !has_order && !has_group && !has_join
        && !su.contains(" AND ") && !su.contains(" OR ")
        && sql.contains('\'')
    {
        if let Some((col, pattern)) = extract_like_pattern(sql, su) {
            return QuerySignature::LikeFilter { column: col, pattern };
        }
    }

    QuerySignature::Complex
}

/// Extract the integer ID from "WHERE _ID = N" or "WHERE _ID=N" in an uppercased SQL.
fn extract_id_from_upper(su: &str) -> Option<u64> {
    let eq_pos = su.find("WHERE _ID =").map(|p| p + "WHERE ".len())
        .or_else(|| su.find("WHERE _ID=").map(|p| p + "WHERE ".len()))?;
    let skip = if su[eq_pos..].starts_with("_ID =") { 5 } else { 4 };
    let after_eq = su[eq_pos + skip..].trim_start();
    let num_end = after_eq.find(|c: char| !c.is_ascii_digit()).unwrap_or(after_eq.len());
    if num_end == 0 { return None; }
    after_eq[..num_end].parse::<u64>().ok()
}

/// Extract LIMIT value from uppercased SQL.
fn extract_limit_from_upper(su: &str) -> Option<usize> {
    let after_limit = su.rsplit("LIMIT").next()?;
    after_limit.trim().trim_end_matches(';').parse::<usize>().ok()
}

/// Extract (column, value) from `WHERE col = 'val'` in original-case SQL,
/// guided by the uppercased version for keyword positions.
fn extract_string_equality(sql: &str, su: &str) -> Option<(String, String)> {
    let where_pos = su.find("WHERE")?;
    let after_where = sql[where_pos + 5..].trim().trim_end_matches(';');
    let eq_pos = after_where.find('=')?;
    let col = after_where[..eq_pos].trim().trim_matches('"').to_string();
    if col.contains(' ') || col.contains('(') { return None; }
    let rhs = after_where[eq_pos + 1..].trim();
    if !rhs.starts_with('\'') { return None; }
    let val_end = rhs[1..].find('\'')?;
    let val = rhs[1..1 + val_end].to_string();
    Some((col, val))
}

/// Extract (column, pattern) from `WHERE col LIKE 'pattern'`.
fn extract_like_pattern(sql: &str, su: &str) -> Option<(String, String)> {
    let where_pos = su.find("WHERE")?;
    let after_where = sql[where_pos + 5..].trim().trim_end_matches(';');
    let after_where_upper = after_where.to_uppercase();
    let like_pos = after_where_upper.find(" LIKE ")?;
    let col = after_where[..like_pos].trim().trim_matches('"').to_string();
    if col.contains(' ') || col.contains('(') { return None; }
    let rhs = after_where[like_pos + 6..].trim();
    if !rhs.starts_with('\'') { return None; }
    let val_end = rhs[1..].find('\'')?;
    let pattern = rhs[1..1 + val_end].to_string();
    Some((col, pattern))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_count_star() {
        assert_eq!(
            classify("SELECT COUNT(*) FROM users"),
            QuerySignature::CountStar { table: "users".to_string() }
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
            QuerySignature::PointLookup { id: 42 }
        );
        assert_eq!(
            classify("SELECT * FROM t WHERE _id=100"),
            QuerySignature::PointLookup { id: 100 }
        );
    }

    #[test]
    fn test_simple_scan_limit() {
        assert_eq!(
            classify("SELECT * FROM t LIMIT 100"),
            QuerySignature::SimpleScanLimit { limit: 100 }
        );
        // With WHERE — not a simple scan
        assert_eq!(
            classify("SELECT * FROM t WHERE x > 1 LIMIT 100"),
            QuerySignature::Complex
        );
    }

    #[test]
    fn test_string_equality() {
        assert_eq!(
            classify("SELECT * FROM t WHERE city = 'NYC'"),
            QuerySignature::StringEqualityFilter {
                column: "city".to_string(),
                value: "NYC".to_string(),
            }
        );
    }

    #[test]
    fn test_like_filter() {
        assert_eq!(
            classify("SELECT * FROM t WHERE name LIKE '%smith%'"),
            QuerySignature::LikeFilter {
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
        assert_eq!(classify("CREATE TABLE t (id INT)"), QuerySignature::Ddl);
        assert_eq!(classify("DROP TABLE t"), QuerySignature::Ddl);
        assert_eq!(classify("ALTER TABLE t ADD COLUMN x INT"), QuerySignature::Ddl);
    }

    #[test]
    fn test_dml_write() {
        assert_eq!(classify("INSERT INTO t VALUES (1)"), QuerySignature::DmlWrite);
        assert_eq!(classify("DELETE FROM t WHERE id = 1"), QuerySignature::DmlWrite);
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
