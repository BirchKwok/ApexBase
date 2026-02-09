"""
Tests for multi-statement SQL execution (semicolon-separated).
Verifies that multiple SQL statements in a single execute() call
work correctly across DML, DDL, index, PRAGMA, REINDEX, and mixed scenarios.
"""

import pytest
import tempfile
import shutil

from apexbase import ApexClient


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def client():
    """Create a temporary client with a test table."""
    tmpdir = tempfile.mkdtemp(prefix="apexbase_multi_")
    c = ApexClient(dirpath=tmpdir)
    c.create_table("t1", {"name": "string", "age": "int", "city": "string"})
    c.use_table("t1")
    yield c
    c.close()
    shutil.rmtree(tmpdir, ignore_errors=True)


@pytest.fixture
def client_with_data(client):
    """Client with pre-loaded test data."""
    client.store([
        {"name": "Alice", "age": 25, "city": "NYC"},
        {"name": "Bob", "age": 30, "city": "LA"},
        {"name": "Charlie", "age": 35, "city": "NYC"},
    ])
    return client


# ============================================================================
# 1. Multiple DML Statements
# ============================================================================

class TestMultiStatementDML:
    """Multiple DML statements in a single execute() call."""

    def test_multiple_inserts(self, client):
        """Multiple INSERT statements separated by semicolons."""
        client.execute("""
            INSERT INTO t1 (name, age, city) VALUES ('Alice', 25, 'NYC');
            INSERT INTO t1 (name, age, city) VALUES ('Bob', 30, 'LA');
            INSERT INTO t1 (name, age, city) VALUES ('Charlie', 35, 'SF')
        """)
        assert client.count_rows() == 3
        df = client.execute("SELECT name FROM t1 ORDER BY name").to_pandas()
        assert list(df["name"]) == ["Alice", "Bob", "Charlie"]

    def test_five_inserts_then_count(self, client):
        """Five INSERTs followed by a SELECT COUNT."""
        result = client.execute("""
            INSERT INTO t1 (name, age, city) VALUES ('A', 1, 'X');
            INSERT INTO t1 (name, age, city) VALUES ('B', 2, 'X');
            INSERT INTO t1 (name, age, city) VALUES ('C', 3, 'X');
            INSERT INTO t1 (name, age, city) VALUES ('D', 4, 'X');
            INSERT INTO t1 (name, age, city) VALUES ('E', 5, 'X');
            SELECT COUNT(*) as cnt FROM t1
        """)
        df = result.to_pandas()
        assert df.iloc[0]["cnt"] == 5

    def test_insert_then_delete(self, client_with_data):
        """INSERT + DELETE in one call."""
        client_with_data.execute("""
            INSERT INTO t1 (name, age, city) VALUES ('Dave', 40, 'SF');
            DELETE FROM t1 WHERE name = 'Alice'
        """)
        assert client_with_data.count_rows() == 3
        df = client_with_data.execute("SELECT name FROM t1 ORDER BY name").to_pandas()
        assert "Alice" not in list(df["name"])
        assert "Dave" in list(df["name"])

    def test_insert_then_update(self, client_with_data):
        """INSERT + UPDATE in one call."""
        client_with_data.execute("""
            INSERT INTO t1 (name, age, city) VALUES ('Dave', 40, 'SF');
            UPDATE t1 SET age = 99 WHERE name = 'Alice'
        """)
        assert client_with_data.count_rows() == 4
        df = client_with_data.execute("SELECT age FROM t1 WHERE name = 'Alice'").to_pandas()
        assert df.iloc[0]["age"] == 99

    def test_update_then_select(self, client_with_data):
        """UPDATE + SELECT returns the SELECT result."""
        result = client_with_data.execute("""
            UPDATE t1 SET city = 'SF' WHERE name = 'Alice';
            SELECT name, city FROM t1 WHERE name = 'Alice'
        """)
        df = result.to_pandas()
        assert len(df) == 1
        assert df.iloc[0]["city"] == "SF"

    def test_delete_then_select(self, client_with_data):
        """DELETE + SELECT returns the SELECT result."""
        result = client_with_data.execute("""
            DELETE FROM t1 WHERE name = 'Bob';
            SELECT name FROM t1 ORDER BY name
        """)
        df = result.to_pandas()
        assert len(df) == 2
        assert list(df["name"]) == ["Alice", "Charlie"]

    def test_truncate_insert_select(self, client_with_data):
        """TRUNCATE + INSERT + SELECT in one call."""
        result = client_with_data.execute("""
            TRUNCATE TABLE t1;
            INSERT INTO t1 (name, age, city) VALUES ('Fresh', 1, 'NEW');
            SELECT COUNT(*) as cnt FROM t1
        """)
        df = result.to_pandas()
        assert df.iloc[0]["cnt"] == 1

    def test_multiple_deletes(self, client_with_data):
        """Multiple DELETE statements."""
        client_with_data.execute("""
            DELETE FROM t1 WHERE name = 'Alice';
            DELETE FROM t1 WHERE name = 'Bob'
        """)
        assert client_with_data.count_rows() == 1
        df = client_with_data.execute("SELECT name FROM t1").to_pandas()
        assert list(df["name"]) == ["Charlie"]

    def test_multiple_updates(self, client_with_data):
        """Multiple UPDATE statements."""
        client_with_data.execute("""
            UPDATE t1 SET age = 100 WHERE name = 'Alice';
            UPDATE t1 SET age = 200 WHERE name = 'Bob';
            UPDATE t1 SET age = 300 WHERE name = 'Charlie'
        """)
        df = client_with_data.execute("SELECT name, age FROM t1 ORDER BY name").to_pandas()
        assert list(df["age"]) == [100, 200, 300]


# ============================================================================
# 2. DDL + DML Mixed Statements
# ============================================================================

class TestMultiStatementDDLDML:
    """Mixed DDL and DML in a single execute() call."""

    def test_create_tables(self):
        """Multiple CREATE TABLE statements."""
        with tempfile.TemporaryDirectory() as tmpdir:
            c = ApexClient(tmpdir)
            c.create_table("default")
            c.execute("""
                CREATE TABLE IF NOT EXISTS users;
                CREATE TABLE IF NOT EXISTS orders;
                CREATE TABLE IF NOT EXISTS products
            """)
            tables = c.list_tables()
            assert "users" in tables
            assert "orders" in tables
            assert "products" in tables
            c.close()

    def test_create_drop_tables(self):
        """CREATE + DROP TABLE in one call."""
        with tempfile.TemporaryDirectory() as tmpdir:
            c = ApexClient(tmpdir)
            c.create_table("default")
            c.execute("""
                CREATE TABLE IF NOT EXISTS temp_table;
                DROP TABLE IF EXISTS temp_table
            """)
            tables = c.list_tables()
            assert "temp_table" not in tables
            c.close()

    def test_alter_then_insert_then_select(self):
        """ALTER TABLE + INSERT + SELECT."""
        with tempfile.TemporaryDirectory() as tmpdir:
            c = ApexClient(tmpdir)
            c.create_table("t2", {"x": "int"})
            c.use_table("t2")
            c.store([{"x": 1}])
            result = c.execute("""
                ALTER TABLE t2 ADD COLUMN y STRING;
                INSERT INTO t2 (x, y) VALUES (2, 'hello');
                SELECT x, y FROM t2 ORDER BY x
            """)
            df = result.to_pandas()
            assert len(df) == 2
            assert df.iloc[1]["y"] == "hello"
            c.close()

    def test_truncate_then_insert(self, client_with_data):
        """TRUNCATE + multiple INSERTs."""
        client_with_data.execute("""
            TRUNCATE TABLE t1;
            INSERT INTO t1 (name, age, city) VALUES ('X', 10, 'A');
            INSERT INTO t1 (name, age, city) VALUES ('Y', 20, 'B')
        """)
        assert client_with_data.count_rows() == 2


# ============================================================================
# 3. Index Operations Multi-Statement
# ============================================================================

class TestMultiStatementIndex:
    """Index DDL + DML in a single execute() call."""

    def test_create_multiple_indexes(self, client_with_data):
        """Multiple CREATE INDEX statements."""
        client_with_data.execute("""
            CREATE INDEX idx_name ON t1 (name);
            CREATE INDEX idx_age ON t1 (age) USING BTREE
        """)
        # Verify both indexes work
        r1 = client_with_data.execute("SELECT * FROM t1 WHERE name = 'Alice'").to_pandas()
        assert len(r1) == 1
        r2 = client_with_data.execute("SELECT * FROM t1 WHERE age = 30").to_pandas()
        assert len(r2) == 1

    def test_create_index_then_query(self, client_with_data):
        """CREATE INDEX + query using that index."""
        result = client_with_data.execute("""
            CREATE INDEX idx_city ON t1 (city);
            SELECT name FROM t1 WHERE city = 'NYC' ORDER BY name
        """)
        df = result.to_pandas()
        assert list(df["name"]) == ["Alice", "Charlie"]

    def test_create_composite_index_then_query(self, client_with_data):
        """CREATE composite INDEX + query."""
        result = client_with_data.execute("""
            CREATE INDEX idx_city_age ON t1 (city, age);
            SELECT name FROM t1 WHERE city = 'NYC' AND age = 25
        """)
        df = result.to_pandas()
        assert len(df) == 1
        assert df.iloc[0]["name"] == "Alice"

    def test_drop_and_recreate_index(self, client_with_data):
        """DROP INDEX + CREATE INDEX."""
        client_with_data.execute("CREATE INDEX idx_name ON t1 (name)")
        client_with_data.execute("""
            DROP INDEX idx_name ON t1;
            CREATE INDEX idx_name_v2 ON t1 (name) USING BTREE
        """)
        r = client_with_data.execute("SELECT * FROM t1 WHERE name = 'Alice'").to_pandas()
        assert len(r) == 1

    def test_create_index_insert_query(self, client_with_data):
        """CREATE INDEX + INSERT + query."""
        result = client_with_data.execute("""
            CREATE INDEX idx_city ON t1 (city);
            INSERT INTO t1 (name, age, city) VALUES ('Dave', 40, 'SF');
            SELECT name FROM t1 WHERE city = 'SF'
        """)
        df = result.to_pandas()
        assert len(df) == 1
        assert df.iloc[0]["name"] == "Dave"

    def test_create_index_then_reindex(self, client_with_data):
        """CREATE INDEX + REINDEX in one call."""
        client_with_data.execute("""
            CREATE INDEX idx_name ON t1 (name);
            REINDEX t1
        """)
        r = client_with_data.execute("SELECT * FROM t1 WHERE name = 'Bob'").to_pandas()
        assert len(r) == 1


# ============================================================================
# 4. PRAGMA Multi-Statement
# ============================================================================

class TestMultiStatementPragma:
    """PRAGMA commands in multi-statement SQL."""

    def test_multiple_pragmas(self, client_with_data):
        """Multiple PRAGMA statements, last result returned."""
        result = client_with_data.execute("""
            PRAGMA version;
            PRAGMA table_info(t1)
        """)
        df = result.to_pandas()
        # Last statement's result: table_info
        assert "cid" in df.columns
        assert "name" in df.columns

    def test_analyze_then_pragma_stats(self, client_with_data):
        """ANALYZE + PRAGMA stats in one call."""
        result = client_with_data.execute("""
            ANALYZE t1;
            PRAGMA stats(t1)
        """)
        df = result.to_pandas()
        assert df.iloc[0]["row_count"] == 3

    def test_insert_then_pragma_integrity(self, client_with_data):
        """INSERT + PRAGMA integrity_check."""
        result = client_with_data.execute("""
            INSERT INTO t1 (name, age, city) VALUES ('Dave', 40, 'SF');
            PRAGMA integrity_check(t1)
        """)
        df = result.to_pandas()
        assert "check" in df.columns
        # All checks should pass
        for _, row in df.iterrows():
            assert "FAIL" not in row["status"]

    def test_pragma_version_standalone(self, client):
        """PRAGMA version as part of multi-statement."""
        result = client.execute("""
            PRAGMA version;
            PRAGMA version
        """)
        df = result.to_pandas()
        assert "ApexBase" in df.iloc[0]["version"]


# ============================================================================
# 5. REINDEX Multi-Statement
# ============================================================================

class TestMultiStatementReindex:
    """REINDEX in multi-statement SQL."""

    def test_insert_then_reindex(self, client_with_data):
        """INSERT + REINDEX after index exists."""
        client_with_data.execute("CREATE INDEX idx_name ON t1 (name)")
        client_with_data.execute("""
            INSERT INTO t1 (name, age, city) VALUES ('Dave', 40, 'SF');
            REINDEX t1
        """)
        r = client_with_data.execute("SELECT * FROM t1 WHERE name = 'Dave'").to_pandas()
        assert len(r) == 1

    def test_delete_then_reindex(self, client_with_data):
        """DELETE + REINDEX."""
        client_with_data.execute("CREATE INDEX idx_name ON t1 (name)")
        client_with_data.execute("""
            DELETE FROM t1 WHERE name = 'Alice';
            REINDEX t1
        """)
        r = client_with_data.execute("SELECT * FROM t1 WHERE name = 'Alice'").to_pandas()
        assert len(r) == 0

    def test_update_then_reindex_then_select(self, client_with_data):
        """UPDATE + REINDEX + SELECT."""
        client_with_data.execute("CREATE INDEX idx_city ON t1 (city)")
        result = client_with_data.execute("""
            UPDATE t1 SET city = 'SF' WHERE name = 'Alice';
            REINDEX t1;
            SELECT name FROM t1 WHERE city = 'SF'
        """)
        df = result.to_pandas()
        assert len(df) == 1
        assert df.iloc[0]["name"] == "Alice"

    def test_reindex_then_pragma_integrity(self, client_with_data):
        """REINDEX + PRAGMA integrity_check."""
        client_with_data.execute("CREATE INDEX idx_name ON t1 (name)")
        result = client_with_data.execute("""
            REINDEX t1;
            PRAGMA integrity_check(t1)
        """)
        df = result.to_pandas()
        for _, row in df.iterrows():
            assert "FAIL" not in row["status"]


# ============================================================================
# 6. Complex Multi-Statement Scenarios
# ============================================================================

class TestMultiStatementComplex:
    """Complex multi-statement scenarios combining many operations."""

    def test_full_lifecycle_in_one_call(self):
        """CREATE TABLE + ALTER + INSERT + SELECT all in one call."""
        with tempfile.TemporaryDirectory() as tmpdir:
            c = ApexClient(tmpdir)
            c.create_table("default")
            result = c.execute("""
                CREATE TABLE IF NOT EXISTS lifecycle;
                ALTER TABLE lifecycle ADD COLUMN name STRING;
                ALTER TABLE lifecycle ADD COLUMN value INT;
                INSERT INTO lifecycle (name, value) VALUES ('key1', 100);
                INSERT INTO lifecycle (name, value) VALUES ('key2', 200);
                SELECT name, value FROM lifecycle ORDER BY name
            """)
            df = result.to_pandas()
            assert len(df) == 2
            assert list(df["name"]) == ["key1", "key2"]
            assert list(df["value"]) == [100, 200]
            c.close()

    def test_insert_update_delete_select(self, client):
        """INSERT + UPDATE + DELETE + SELECT chain."""
        result = client.execute("""
            INSERT INTO t1 (name, age, city) VALUES ('Alice', 25, 'NYC');
            INSERT INTO t1 (name, age, city) VALUES ('Bob', 30, 'LA');
            INSERT INTO t1 (name, age, city) VALUES ('Charlie', 35, 'SF');
            UPDATE t1 SET age = 99 WHERE name = 'Bob';
            DELETE FROM t1 WHERE name = 'Charlie';
            SELECT name, age FROM t1 ORDER BY name
        """)
        df = result.to_pandas()
        assert len(df) == 2
        assert list(df["name"]) == ["Alice", "Bob"]
        assert list(df["age"]) == [25, 99]

    def test_index_lifecycle_in_one_call(self, client_with_data):
        """CREATE INDEX + INSERT + REINDEX + SELECT."""
        result = client_with_data.execute("""
            CREATE INDEX idx_city ON t1 (city);
            INSERT INTO t1 (name, age, city) VALUES ('Dave', 40, 'SF');
            REINDEX t1;
            SELECT name FROM t1 WHERE city = 'SF'
        """)
        df = result.to_pandas()
        assert len(df) == 1
        assert df.iloc[0]["name"] == "Dave"

    def test_analyze_stats_integrity_chain(self, client_with_data):
        """ANALYZE + PRAGMA stats + PRAGMA integrity_check."""
        result = client_with_data.execute("""
            ANALYZE t1;
            PRAGMA stats(t1);
            PRAGMA integrity_check(t1)
        """)
        df = result.to_pandas()
        # Last result: integrity_check
        assert "check" in df.columns
        for _, row in df.iterrows():
            assert "FAIL" not in row["status"]

    def test_semicolon_in_string_literal(self, client):
        """Semicolons inside string literals should not split statements."""
        client.execute("""
            INSERT INTO t1 (name, age, city) VALUES ('semi;colon', 1, 'test;city');
            SELECT name FROM t1
        """)
        df = client.execute("SELECT name, city FROM t1").to_pandas()
        assert len(df) == 1
        assert df.iloc[0]["name"] == "semi;colon"
        assert df.iloc[0]["city"] == "test;city"

    def test_trailing_semicolons(self, client):
        """Extra trailing semicolons should not cause errors."""
        client.execute("""
            INSERT INTO t1 (name, age, city) VALUES ('A', 1, 'X');;;
        """)
        assert client.count_rows() == 1

    def test_empty_statements_between_semicolons(self, client):
        """Empty statements between semicolons should be tolerated."""
        client.execute("""
            INSERT INTO t1 (name, age, city) VALUES ('A', 1, 'X');
            ;
            INSERT INTO t1 (name, age, city) VALUES ('B', 2, 'Y');
        """)
        assert client.count_rows() == 2

    def test_last_result_is_returned(self, client_with_data):
        """Multi-statement returns the LAST statement's result."""
        # When last stmt is SELECT, result should be the SELECT data
        result = client_with_data.execute("""
            INSERT INTO t1 (name, age, city) VALUES ('Dave', 40, 'SF');
            SELECT name FROM t1 ORDER BY name
        """)
        df = result.to_pandas()
        assert len(df) == 4
        assert list(df["name"]) == ["Alice", "Bob", "Charlie", "Dave"]

    def test_ten_inserts(self, client):
        """10 INSERT statements in one call."""
        stmts = ";\n".join(
            f"INSERT INTO t1 (name, age, city) VALUES ('user_{i}', {20+i}, 'city_{i}')"
            for i in range(10)
        )
        client.execute(stmts)
        assert client.count_rows() == 10

    def test_insert_delete_insert_count(self, client):
        """INSERT + DELETE + INSERT — net effect check."""
        result = client.execute("""
            INSERT INTO t1 (name, age, city) VALUES ('A', 1, 'X');
            INSERT INTO t1 (name, age, city) VALUES ('B', 2, 'Y');
            DELETE FROM t1 WHERE name = 'A';
            INSERT INTO t1 (name, age, city) VALUES ('C', 3, 'Z');
            SELECT COUNT(*) as cnt FROM t1
        """)
        df = result.to_pandas()
        assert df.iloc[0]["cnt"] == 2  # B + C remain


# ============================================================================
# 7. Multi-Statement with Transactions (separate execute calls for BEGIN/COMMIT)
# ============================================================================

class TestMultiStatementWithTransactions:
    """Multi-statement DML inside a transaction context."""

    def test_multi_insert_in_txn(self, client):
        """BEGIN (separate) + multi-INSERT (one call) + COMMIT (separate)."""
        client.execute("BEGIN")
        client.execute("""
            INSERT INTO t1 (name, age, city) VALUES ('A', 1, 'X');
            INSERT INTO t1 (name, age, city) VALUES ('B', 2, 'Y');
            INSERT INTO t1 (name, age, city) VALUES ('C', 3, 'Z')
        """)
        client.execute("COMMIT")
        assert client.count_rows() == 3

    def test_multi_insert_rollback(self, client):
        """BEGIN + multi-INSERT + ROLLBACK — all discarded."""
        client.store([{"name": "Base", "age": 0, "city": "X"}])
        client.execute("BEGIN")
        client.execute("""
            INSERT INTO t1 (name, age, city) VALUES ('A', 1, 'X');
            INSERT INTO t1 (name, age, city) VALUES ('B', 2, 'Y')
        """)
        client.execute("ROLLBACK")
        assert client.count_rows() == 1

    def test_multi_dml_in_txn(self, client_with_data):
        """BEGIN + multi-DML (INSERT+UPDATE+DELETE) + COMMIT."""
        client_with_data.execute("BEGIN")
        client_with_data.execute("""
            INSERT INTO t1 (name, age, city) VALUES ('Dave', 40, 'SF');
            UPDATE t1 SET age = 99 WHERE name = 'Alice';
            DELETE FROM t1 WHERE name = 'Bob'
        """)
        client_with_data.execute("COMMIT")
        assert client_with_data.count_rows() == 3  # Alice + Charlie + Dave
        df = client_with_data.execute("SELECT age FROM t1 WHERE name = 'Alice'").to_pandas()
        assert df.iloc[0]["age"] == 99


# ============================================================================
# 8. Edge Cases
# ============================================================================

class TestMultiStatementEdgeCases:
    """Edge cases for multi-statement SQL."""

    def test_single_statement_with_semicolon(self, client):
        """Single statement ending with semicolon should work."""
        client.execute("INSERT INTO t1 (name, age, city) VALUES ('A', 1, 'X');")
        assert client.count_rows() == 1

    def test_whitespace_between_statements(self, client):
        """Whitespace and newlines between statements."""
        client.execute("""
        
            INSERT INTO t1 (name, age, city) VALUES ('A', 1, 'X')
            
            ;
            
            INSERT INTO t1 (name, age, city) VALUES ('B', 2, 'Y')
            
        """)
        assert client.count_rows() == 2

    def test_mixed_case_keywords(self, client):
        """Mixed case SQL keywords in multi-statement."""
        client.execute("""
            insert INTO t1 (name, age, city) values ('A', 1, 'X');
            INSERT into t1 (name, age, city) VALUES ('B', 2, 'Y')
        """)
        assert client.count_rows() == 2

    def test_begin_insert_commit_single_call(self, client):
        """BEGIN + INSERT + COMMIT in one execute() call (SQLite-like)."""
        client.store([{"name": "Base", "age": 0, "city": "X"}])
        client.execute("""
            BEGIN;
            INSERT INTO t1 (name, age, city) VALUES ('TxnRow', 99, 'TxnCity');
            COMMIT
        """)
        assert client.count_rows() == 2
        assert client._in_txn == False
        df = client.execute("SELECT name, age FROM t1 WHERE name = 'TxnRow'").to_pandas()
        assert df.iloc[0]["age"] == 99

    def test_begin_multi_dml_commit_single_call(self, client):
        """BEGIN + multiple DML + COMMIT in one execute() call."""
        client.execute("""
            BEGIN;
            INSERT INTO t1 (name, age, city) VALUES ('A', 1, 'X');
            INSERT INTO t1 (name, age, city) VALUES ('B', 2, 'Y');
            INSERT INTO t1 (name, age, city) VALUES ('C', 3, 'Z');
            COMMIT
        """)
        assert client.count_rows() == 3
        assert client._in_txn == False

    def test_begin_insert_rollback_single_call(self, client):
        """BEGIN + INSERT + ROLLBACK in one call — no data committed."""
        client.store([{"name": "Base", "age": 0, "city": "X"}])
        client.execute("""
            BEGIN;
            INSERT INTO t1 (name, age, city) VALUES ('Gone', 99, 'Nowhere');
            ROLLBACK
        """)
        assert client.count_rows() == 1
        assert client._in_txn == False

    def test_upsert_in_multi_statement(self, client):
        """INSERT ON CONFLICT in multi-statement context."""
        client.execute("""
            CREATE UNIQUE INDEX idx_name ON t1 (name);
            INSERT INTO t1 (name, age, city) VALUES ('Alice', 25, 'NYC')
        """)
        result = client.execute("""
            INSERT INTO t1 (name, age, city) VALUES ('Alice', 30, 'LA')
                ON CONFLICT (name) DO UPDATE SET age = 30, city = 'LA';
            SELECT age, city FROM t1 WHERE name = 'Alice'
        """)
        df = result.to_pandas()
        assert df.iloc[0]["age"] == 30
        assert df.iloc[0]["city"] == "LA"
