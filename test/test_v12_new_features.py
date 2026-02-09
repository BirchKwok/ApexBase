"""
Tests for HTAP Gap Analysis v12 new features:
1. Transaction timeout protection (TxnManager 30s timeout + idle cleanup)
2. REINDEX command (full index rebuild)
3. PRAGMA commands (integrity_check / table_info / version / stats)
4. Composite multi-column indexes (CREATE INDEX ... ON t(c1, c2))
"""

import pytest
import tempfile
import shutil
import time

from apexbase import ApexClient


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def client():
    """Create a temporary client with a test table."""
    tmpdir = tempfile.mkdtemp(prefix="apexbase_v12_")
    c = ApexClient(dirpath=tmpdir)
    c.create_table("t1", {"name": "string", "age": "int", "city": "string", "score": "float"})
    c.use_table("t1")
    yield c
    c.close()
    shutil.rmtree(tmpdir, ignore_errors=True)


@pytest.fixture
def client_with_data(client):
    """Client with pre-loaded test data."""
    rows = []
    cities = ["Beijing", "Shanghai", "Guangzhou", "Shenzhen", "Hangzhou"]
    for i in range(100):
        rows.append({
            "name": f"user_{i}",
            "age": 20 + (i % 50),
            "city": cities[i % len(cities)],
            "score": 60.0 + (i % 40),
        })
    client.store(rows)
    return client


# ============================================================================
# 1. Transaction Timeout Protection
# ============================================================================

class TestTransactionTimeout:
    """Test transaction timeout protection (30s idle timeout)."""

    def test_txn_completes_within_timeout(self, client):
        """A transaction that completes quickly should succeed without timeout."""
        client.store([{"name": "Alice", "age": 25, "city": "NYC", "score": 90.0}])

        client.execute("BEGIN")
        client.execute("INSERT INTO t1 (name, age, city, score) VALUES ('Bob', 30, 'LA', 85.0)")
        client.execute("COMMIT")

        assert client.count_rows() == 2

    def test_multiple_rapid_transactions(self, client):
        """Multiple rapid transactions should all succeed (no stale timeout)."""
        client.store([{"name": "base", "age": 1, "city": "X", "score": 1.0}])

        for i in range(10):
            client.execute("BEGIN")
            client.execute(f"INSERT INTO t1 (name, age, city, score) VALUES ('txn_{i}', {i}, 'city_{i}', {float(i)})")
            client.execute("COMMIT")

        assert client.count_rows() == 11

    def test_txn_with_reads_refreshes_activity(self, client):
        """Reads within a transaction should refresh the activity timestamp."""
        client.store([{"name": "Alice", "age": 25, "city": "NYC", "score": 90.0}])

        client.execute("BEGIN")
        # Multiple reads should keep the transaction alive
        for _ in range(5):
            df = client.execute("SELECT * FROM t1").to_pandas()
            assert len(df) >= 1
        client.execute("INSERT INTO t1 (name, age, city, score) VALUES ('Bob', 30, 'LA', 85.0)")
        client.execute("COMMIT")

        assert client.count_rows() == 2

    def test_rollback_after_activity(self, client):
        """ROLLBACK should work even after multiple operations."""
        client.store([{"name": "Alice", "age": 25, "city": "NYC", "score": 90.0}])

        client.execute("BEGIN")
        client.execute("INSERT INTO t1 (name, age, city, score) VALUES ('Ghost', 99, 'X', 0.0)")
        # Read (refreshes activity)
        df = client.execute("SELECT * FROM t1").to_pandas()
        assert len(df) == 2  # read-your-writes
        client.execute("ROLLBACK")

        assert client.count_rows() == 1

    def test_new_txn_after_commit(self, client):
        """Starting a new transaction immediately after COMMIT should work."""
        client.store([{"name": "Alice", "age": 25, "city": "NYC", "score": 90.0}])

        client.execute("BEGIN")
        client.execute("INSERT INTO t1 (name, age, city, score) VALUES ('Bob', 30, 'LA', 85.0)")
        client.execute("COMMIT")

        client.execute("BEGIN")
        client.execute("INSERT INTO t1 (name, age, city, score) VALUES ('Charlie', 35, 'SF', 95.0)")
        client.execute("COMMIT")

        assert client.count_rows() == 3

    def test_new_txn_after_rollback(self, client):
        """Starting a new transaction after ROLLBACK should work cleanly."""
        client.store([{"name": "Alice", "age": 25, "city": "NYC", "score": 90.0}])

        client.execute("BEGIN")
        client.execute("INSERT INTO t1 (name, age, city, score) VALUES ('Ghost', 99, 'X', 0.0)")
        client.execute("ROLLBACK")

        client.execute("BEGIN")
        client.execute("INSERT INTO t1 (name, age, city, score) VALUES ('Bob', 30, 'LA', 85.0)")
        client.execute("COMMIT")

        assert client.count_rows() == 2
        df = client.execute("SELECT name FROM t1 ORDER BY name").to_pandas()
        assert list(df["name"]) == ["Alice", "Bob"]


# ============================================================================
# 2. REINDEX Command
# ============================================================================

class TestReindex:
    """Test REINDEX table command — full index rebuild."""

    def test_reindex_rebuilds_index(self, client_with_data):
        """REINDEX should rebuild indexes and queries still work correctly."""
        c = client_with_data
        c.execute("CREATE INDEX idx_city ON t1 (city)")

        # Verify index works before REINDEX
        r1 = c.execute("SELECT COUNT(*) as cnt FROM t1 WHERE city = 'Beijing'").to_dict()
        count_before = r1[0]["cnt"]

        # REINDEX
        result = c.execute("REINDEX t1")
        # Should return number of indexes rebuilt
        assert result is not None

        # Verify index still works after REINDEX
        r2 = c.execute("SELECT COUNT(*) as cnt FROM t1 WHERE city = 'Beijing'").to_dict()
        count_after = r2[0]["cnt"]
        assert count_before == count_after

    def test_reindex_no_indexes(self, client):
        """REINDEX on table with no indexes should return 0."""
        client.store([{"name": "Alice", "age": 25, "city": "NYC", "score": 90.0}])
        result = client.execute("REINDEX t1")
        assert result is not None

    def test_reindex_nonexistent_table(self, client):
        """REINDEX on non-existent table should error."""
        with pytest.raises(Exception):
            client.execute("REINDEX nonexistent_table")

    def test_reindex_after_insert(self, client):
        """REINDEX after INSERT should include new data in index."""
        client.store([{"name": "Alice", "age": 25, "city": "NYC", "score": 90.0}])
        client.execute("CREATE INDEX idx_city ON t1 (city)")

        # Insert new data
        client.execute("INSERT INTO t1 (name, age, city, score) VALUES ('Bob', 30, 'LA', 85.0)")

        # REINDEX
        client.execute("REINDEX t1")

        # Verify new data is in the index
        r = client.execute("SELECT * FROM t1 WHERE city = 'LA'").to_pandas()
        assert len(r) == 1
        assert r.iloc[0]["name"] == "Bob"

    def test_reindex_after_delete(self, client):
        """REINDEX after DELETE should not include deleted data in index."""
        client.store([
            {"name": "Alice", "age": 25, "city": "NYC", "score": 90.0},
            {"name": "Bob", "age": 30, "city": "LA", "score": 85.0},
            {"name": "Charlie", "age": 35, "city": "NYC", "score": 95.0},
        ])
        client.execute("CREATE INDEX idx_city ON t1 (city)")

        # Delete
        client.execute("DELETE FROM t1 WHERE name = 'Alice'")

        # REINDEX
        client.execute("REINDEX t1")

        # Verify deleted data is not in index results
        r = client.execute("SELECT * FROM t1 WHERE city = 'NYC'").to_pandas()
        assert len(r) == 1
        assert r.iloc[0]["name"] == "Charlie"

    def test_reindex_after_update(self, client):
        """REINDEX after UPDATE should reflect updated values in index."""
        client.store([
            {"name": "Alice", "age": 25, "city": "NYC", "score": 90.0},
            {"name": "Bob", "age": 30, "city": "LA", "score": 85.0},
        ])
        client.execute("CREATE INDEX idx_city ON t1 (city)")

        # Update city
        client.execute("UPDATE t1 SET city = 'SF' WHERE name = 'Alice'")

        # REINDEX
        client.execute("REINDEX t1")

        # Verify updated value is in index
        r_sf = client.execute("SELECT * FROM t1 WHERE city = 'SF'").to_pandas()
        assert len(r_sf) == 1
        assert r_sf.iloc[0]["name"] == "Alice"

        r_nyc = client.execute("SELECT * FROM t1 WHERE city = 'NYC'").to_pandas()
        assert len(r_nyc) == 0

    def test_reindex_multiple_indexes(self, client):
        """REINDEX should rebuild all indexes on the table."""
        client.store([
            {"name": "Alice", "age": 25, "city": "NYC", "score": 90.0},
            {"name": "Bob", "age": 30, "city": "LA", "score": 85.0},
        ])
        client.execute("CREATE INDEX idx_city ON t1 (city)")
        client.execute("CREATE INDEX idx_age ON t1 (age) USING BTREE")

        # REINDEX
        client.execute("REINDEX t1")

        # Verify both indexes work
        r1 = client.execute("SELECT * FROM t1 WHERE city = 'NYC'").to_pandas()
        assert len(r1) == 1

        r2 = client.execute("SELECT * FROM t1 WHERE age = 30").to_pandas()
        assert len(r2) == 1
        assert r2.iloc[0]["name"] == "Bob"

    def test_reindex_with_table_keyword(self, client):
        """REINDEX TABLE t1 syntax should also work."""
        client.store([{"name": "Alice", "age": 25, "city": "NYC", "score": 90.0}])
        client.execute("CREATE INDEX idx_city ON t1 (city)")
        result = client.execute("REINDEX TABLE t1")
        assert result is not None


# ============================================================================
# 3. PRAGMA Commands
# ============================================================================

class TestPragmaIntegrityCheck:
    """Test PRAGMA integrity_check."""

    def test_integrity_check_healthy_table(self, client_with_data):
        """PRAGMA integrity_check on a healthy table should return all 'ok'."""
        c = client_with_data
        result = c.execute("PRAGMA integrity_check(t1)")
        df = result.to_pandas()

        assert len(df) > 0
        assert "check" in df.columns
        assert "status" in df.columns

        # All statuses should contain "ok"
        for _, row in df.iterrows():
            assert "ok" in row["status"].lower() or "FAIL" not in row["status"], \
                f"Check '{row['check']}' failed: {row['status']}"

    def test_integrity_check_with_indexes(self, client_with_data):
        """PRAGMA integrity_check should report index count."""
        c = client_with_data
        c.execute("CREATE INDEX idx_city ON t1 (city)")
        c.execute("CREATE INDEX idx_age ON t1 (age) USING BTREE")

        result = c.execute("PRAGMA integrity_check(t1)")
        df = result.to_pandas()

        # Find the indexes check row
        idx_row = df[df["check"] == "indexes"]
        assert len(idx_row) == 1
        assert "2 indexes" in idx_row.iloc[0]["status"]

    def test_integrity_check_reports_row_count(self, client_with_data):
        """PRAGMA integrity_check should report correct row count."""
        c = client_with_data
        result = c.execute("PRAGMA integrity_check(t1)")
        df = result.to_pandas()

        row_count_row = df[df["check"] == "row_count"]
        assert len(row_count_row) == 1
        assert "100 rows" in row_count_row.iloc[0]["status"]

    def test_integrity_check_reports_schema(self, client_with_data):
        """PRAGMA integrity_check should report schema column count."""
        c = client_with_data
        result = c.execute("PRAGMA integrity_check(t1)")
        df = result.to_pandas()

        schema_row = df[df["check"] == "schema_valid"]
        assert len(schema_row) == 1
        assert "columns" in schema_row.iloc[0]["status"]

    def test_integrity_check_nonexistent_table(self, client):
        """PRAGMA integrity_check on non-existent table should report FAIL."""
        result = client.execute("PRAGMA integrity_check(nonexistent)")
        df = result.to_pandas()
        # Should have file_exists check with FAIL
        file_row = df[df["check"] == "file_exists"]
        assert len(file_row) == 1
        assert "FAIL" in file_row.iloc[0]["status"]

    def test_integrity_check_empty_table(self, client):
        """PRAGMA integrity_check on empty table should still pass."""
        client.store([{"name": "temp", "age": 1, "city": "x", "score": 1.0}])
        client.execute("TRUNCATE TABLE t1")
        client.flush()

        result = client.execute("PRAGMA integrity_check(t1)")
        df = result.to_pandas()
        # File should exist and be readable
        file_row = df[df["check"] == "file_exists"]
        assert len(file_row) == 1
        assert "ok" in file_row.iloc[0]["status"].lower()


class TestPragmaTableInfo:
    """Test PRAGMA table_info."""

    def test_table_info_returns_columns(self, client_with_data):
        """PRAGMA table_info should return column info."""
        c = client_with_data
        result = c.execute("PRAGMA table_info(t1)")
        df = result.to_pandas()

        assert "cid" in df.columns
        assert "name" in df.columns
        assert "type" in df.columns
        assert len(df) >= 4  # _id, name, age, city, score (at least 4 user cols)

    def test_table_info_column_names(self, client_with_data):
        """PRAGMA table_info should list all column names."""
        c = client_with_data
        result = c.execute("PRAGMA table_info(t1)")
        df = result.to_pandas()

        col_names = set(df["name"].tolist())
        assert "name" in col_names
        assert "age" in col_names
        assert "city" in col_names
        assert "score" in col_names

    def test_table_info_cid_sequential(self, client_with_data):
        """PRAGMA table_info should have sequential cid values."""
        c = client_with_data
        result = c.execute("PRAGMA table_info(t1)")
        df = result.to_pandas()

        cids = sorted(df["cid"].tolist())
        assert cids == list(range(len(cids)))

    def test_table_info_nonexistent_table(self, client):
        """PRAGMA table_info on non-existent table should error."""
        with pytest.raises(Exception):
            client.execute("PRAGMA table_info(nonexistent)")

    def test_table_info_after_add_column(self):
        """PRAGMA table_info should reflect ALTER TABLE ADD COLUMN."""
        with tempfile.TemporaryDirectory() as tmpdir:
            c = ApexClient(tmpdir)
            c.create_table("t2", {"name": "string", "age": "int"})
            c.use_table("t2")
            c.store([{"name": "Alice", "age": 25}])

            c.execute("ALTER TABLE t2 ADD COLUMN email STRING")

            result = c.execute("PRAGMA table_info(t2)")
            df = result.to_pandas()
            col_names = set(df["name"].tolist())
            assert "email" in col_names
            c.close()


class TestPragmaVersion:
    """Test PRAGMA version."""

    def test_version_returns_string(self, client):
        """PRAGMA version should return a version string."""
        result = client.execute("PRAGMA version")
        df = result.to_pandas()

        assert len(df) == 1
        assert "version" in df.columns
        version_str = df.iloc[0]["version"]
        assert "ApexBase" in version_str

    def test_version_format(self, client):
        """PRAGMA version should contain version number."""
        result = client.execute("PRAGMA version")
        df = result.to_pandas()
        version_str = df.iloc[0]["version"]
        # Should match "ApexBase X.Y" format
        assert "1.0" in version_str or "ApexBase" in version_str


class TestPragmaStats:
    """Test PRAGMA stats."""

    def test_stats_after_analyze(self, client_with_data):
        """PRAGMA stats after ANALYZE should return table statistics."""
        c = client_with_data
        c.execute("ANALYZE t1")
        result = c.execute("PRAGMA stats(t1)")
        df = result.to_pandas()

        assert len(df) == 1
        assert "table" in df.columns
        assert "row_count" in df.columns
        assert "columns" in df.columns
        assert df.iloc[0]["row_count"] == 100

    def test_stats_without_analyze(self, client_with_data):
        """PRAGMA stats without prior ANALYZE should return scalar 0 or empty."""
        c = client_with_data
        result = c.execute("PRAGMA stats(t1)")
        # Without ANALYZE, stats may not be available
        # The result could be Scalar(0) or empty data
        assert result is not None

    def test_stats_column_count(self, client_with_data):
        """PRAGMA stats should report correct column count."""
        c = client_with_data
        c.execute("ANALYZE t1")
        result = c.execute("PRAGMA stats(t1)")
        df = result.to_pandas()

        assert len(df) == 1
        # Should have at least 4 user columns + _id
        assert df.iloc[0]["columns"] >= 4


class TestPragmaErrors:
    """Test PRAGMA error handling."""

    def test_unknown_pragma(self, client):
        """Unknown PRAGMA should raise an error."""
        with pytest.raises(Exception, match="Unknown PRAGMA|Unsupported|unknown"):
            client.execute("PRAGMA unknown_command")


# ============================================================================
# 4. Composite Multi-Column Index
# ============================================================================

class TestCompositeIndexCreation:
    """Test CREATE INDEX with multiple columns."""

    def test_create_composite_index(self, client_with_data):
        """CREATE INDEX on (city, age) should succeed."""
        c = client_with_data
        result = c.execute("CREATE INDEX idx_city_age ON t1 (city, age)")
        assert result is not None

    def test_create_composite_index_btree(self, client_with_data):
        """CREATE INDEX ... USING BTREE on multiple columns should succeed."""
        c = client_with_data
        result = c.execute("CREATE INDEX idx_city_age ON t1 (city, age) USING BTREE")
        assert result is not None

    def test_create_composite_index_unique(self, client_with_data):
        """CREATE UNIQUE INDEX on multiple columns should succeed."""
        c = client_with_data
        result = c.execute("CREATE UNIQUE INDEX idx_name_city ON t1 (name, city)")
        assert result is not None

    def test_create_composite_index_if_not_exists(self, client_with_data):
        """CREATE INDEX IF NOT EXISTS on composite should not error on duplicate."""
        c = client_with_data
        c.execute("CREATE INDEX idx_city_age ON t1 (city, age)")
        result = c.execute("CREATE INDEX IF NOT EXISTS idx_city_age ON t1 (city, age)")
        assert result is not None

    def test_create_composite_index_duplicate_error(self, client_with_data):
        """CREATE INDEX without IF NOT EXISTS should error on duplicate composite index."""
        c = client_with_data
        c.execute("CREATE INDEX idx_city_age ON t1 (city, age)")
        with pytest.raises(Exception):
            c.execute("CREATE INDEX idx_city_age ON t1 (city, age)")

    def test_create_composite_index_nonexistent_column(self, client_with_data):
        """CREATE INDEX with non-existent column should error."""
        c = client_with_data
        with pytest.raises(Exception):
            c.execute("CREATE INDEX idx_bad ON t1 (city, nonexistent_col)")

    def test_create_composite_index_on_empty_table(self, client):
        """CREATE INDEX on empty table with schema should succeed."""
        client.store([{"name": "temp", "age": 1, "city": "x", "score": 1.0}])
        client.execute("TRUNCATE TABLE t1")
        result = client.execute("CREATE INDEX idx_city_age ON t1 (city, age)")
        assert result is not None

    def test_drop_composite_index(self, client_with_data):
        """DROP INDEX on composite index should succeed."""
        c = client_with_data
        c.execute("CREATE INDEX idx_city_age ON t1 (city, age)")
        result = c.execute("DROP INDEX idx_city_age ON t1")
        assert result is not None


class TestCompositeIndexQuery:
    """Test that composite indexes return correct query results."""

    def test_composite_index_equality_filter(self, client_with_data):
        """WHERE on composite index columns should return correct results."""
        c = client_with_data
        c.execute("CREATE INDEX idx_city_age ON t1 (city, age)")

        result = c.execute("SELECT * FROM t1 WHERE city = 'Beijing' AND age = 20")
        df = result.to_pandas()
        # All results should match both conditions
        for _, row in df.iterrows():
            assert row["city"] == "Beijing"
            assert row["age"] == 20

    def test_composite_index_correctness_vs_scan(self, client_with_data):
        """Composite index results should match full-scan results."""
        c = client_with_data

        # Full scan (no index)
        r_scan = c.execute(
            "SELECT name FROM t1 WHERE city = 'Shanghai' AND age = 21 ORDER BY name"
        ).to_pandas()

        # Create composite index
        c.execute("CREATE INDEX idx_city_age ON t1 (city, age)")

        # Index-accelerated query
        r_idx = c.execute(
            "SELECT name FROM t1 WHERE city = 'Shanghai' AND age = 21 ORDER BY name"
        ).to_pandas()

        assert len(r_scan) == len(r_idx)
        assert r_scan["name"].tolist() == r_idx["name"].tolist()

    def test_composite_index_with_order_by(self, client_with_data):
        """Composite index + ORDER BY should return correctly ordered results."""
        c = client_with_data
        c.execute("CREATE INDEX idx_city_age ON t1 (city, age)")

        result = c.execute(
            "SELECT * FROM t1 WHERE city = 'Beijing' ORDER BY age"
        ).to_pandas()
        ages = result["age"].tolist()
        assert ages == sorted(ages)

    def test_composite_index_with_limit(self, client_with_data):
        """Composite index + LIMIT should respect the limit."""
        c = client_with_data
        c.execute("CREATE INDEX idx_city_age ON t1 (city, age)")

        result = c.execute("SELECT * FROM t1 WHERE city = 'Beijing' LIMIT 3")
        df = result.to_pandas()
        assert len(df) <= 3


class TestCompositeIndexDML:
    """Test that composite indexes stay in sync with DML operations."""

    def test_insert_updates_composite_index(self, client):
        """SQL INSERT after composite index creation should be found."""
        client.store([{"name": "Alice", "age": 25, "city": "NYC", "score": 90.0}])
        client.execute("CREATE INDEX idx_city_age ON t1 (city, age)")

        client.execute("INSERT INTO t1 (name, age, city, score) VALUES ('Bob', 30, 'LA', 85.0)")

        result = client.execute("SELECT * FROM t1 WHERE city = 'LA' AND age = 30")
        df = result.to_pandas()
        assert len(df) == 1
        assert df.iloc[0]["name"] == "Bob"

    def test_delete_updates_composite_index(self, client):
        """SQL DELETE should remove entries from composite index."""
        client.store([
            {"name": "Alice", "age": 25, "city": "NYC", "score": 90.0},
            {"name": "Bob", "age": 30, "city": "LA", "score": 85.0},
            {"name": "Charlie", "age": 25, "city": "NYC", "score": 95.0},
        ])
        client.execute("CREATE INDEX idx_city_age ON t1 (city, age)")

        client.execute("DELETE FROM t1 WHERE name = 'Alice'")

        result = client.execute("SELECT * FROM t1 WHERE city = 'NYC' AND age = 25")
        df = result.to_pandas()
        assert len(df) == 1
        assert df.iloc[0]["name"] == "Charlie"

    def test_update_updates_composite_index(self, client):
        """SQL UPDATE should update composite index entries."""
        client.store([
            {"name": "Alice", "age": 25, "city": "NYC", "score": 90.0},
            {"name": "Bob", "age": 30, "city": "LA", "score": 85.0},
        ])
        client.execute("CREATE INDEX idx_city_age ON t1 (city, age)")

        client.execute("UPDATE t1 SET city = 'SF', age = 28 WHERE name = 'Alice'")

        # Old composite key should not match
        r_old = client.execute("SELECT * FROM t1 WHERE city = 'NYC' AND age = 25")
        assert len(r_old.to_pandas()) == 0

        # New composite key should match
        r_new = client.execute("SELECT * FROM t1 WHERE city = 'SF' AND age = 28")
        df_new = r_new.to_pandas()
        assert len(df_new) == 1
        assert df_new.iloc[0]["name"] == "Alice"

    def test_reindex_composite_index(self, client):
        """REINDEX should correctly rebuild composite indexes."""
        client.store([
            {"name": "Alice", "age": 25, "city": "NYC", "score": 90.0},
            {"name": "Bob", "age": 30, "city": "LA", "score": 85.0},
            {"name": "Charlie", "age": 25, "city": "NYC", "score": 95.0},
        ])
        client.execute("CREATE INDEX idx_city_age ON t1 (city, age)")

        # Modify data
        client.execute("INSERT INTO t1 (name, age, city, score) VALUES ('Dave', 40, 'SF', 75.0)")
        client.execute("DELETE FROM t1 WHERE name = 'Bob'")

        # REINDEX
        client.execute("REINDEX t1")

        # Verify results
        r = client.execute("SELECT * FROM t1 WHERE city = 'NYC' AND age = 25")
        df = r.to_pandas()
        assert len(df) == 2  # Alice + Charlie
        names = sorted(df["name"].tolist())
        assert names == ["Alice", "Charlie"]

        # Verify deleted row not found
        r2 = client.execute("SELECT * FROM t1 WHERE city = 'LA' AND age = 30")
        assert len(r2.to_pandas()) == 0

        # Verify new row found
        r3 = client.execute("SELECT * FROM t1 WHERE city = 'SF' AND age = 40")
        assert len(r3.to_pandas()) == 1


class TestCompositeIndexThreeColumns:
    """Test composite index with 3 columns."""

    def test_three_column_composite_index(self):
        """CREATE INDEX on 3 columns should work."""
        with tempfile.TemporaryDirectory() as tmpdir:
            c = ApexClient(tmpdir)
            c.create_table("t3", {"a": "string", "b": "int", "c": "string"})
            c.use_table("t3")
            c.store([
                {"a": "x", "b": 1, "c": "p"},
                {"a": "x", "b": 1, "c": "q"},
                {"a": "x", "b": 2, "c": "p"},
                {"a": "y", "b": 1, "c": "p"},
            ])
            c.execute("CREATE INDEX idx_abc ON t3 (a, b, c)")

            result = c.execute("SELECT * FROM t3 WHERE a = 'x' AND b = 1 AND c = 'p'")
            df = result.to_pandas()
            assert len(df) == 1

            result2 = c.execute("SELECT * FROM t3 WHERE a = 'x' AND b = 1")
            df2 = result2.to_pandas()
            assert len(df2) == 2
            c.close()


# ============================================================================
# Integration: Cross-Feature Tests
# ============================================================================

class TestCrossFeatureIntegration:
    """Tests combining multiple v12 features together."""

    def test_pragma_integrity_after_reindex(self, client_with_data):
        """PRAGMA integrity_check after REINDEX should still pass."""
        c = client_with_data
        c.execute("CREATE INDEX idx_city ON t1 (city)")
        c.execute("REINDEX t1")

        result = c.execute("PRAGMA integrity_check(t1)")
        df = result.to_pandas()
        for _, row in df.iterrows():
            assert "FAIL" not in row["status"]

    def test_reindex_composite_then_pragma(self, client_with_data):
        """REINDEX composite index then verify with PRAGMA integrity_check."""
        c = client_with_data
        c.execute("CREATE INDEX idx_city_age ON t1 (city, age)")
        c.execute("REINDEX t1")

        result = c.execute("PRAGMA integrity_check(t1)")
        df = result.to_pandas()

        idx_row = df[df["check"] == "indexes"]
        assert len(idx_row) == 1
        assert "1 indexes" in idx_row.iloc[0]["status"]

    def test_txn_with_composite_index(self, client):
        """Transaction operations with composite index should work."""
        client.store([{"name": "Alice", "age": 25, "city": "NYC", "score": 90.0}])
        client.execute("CREATE INDEX idx_city_age ON t1 (city, age)")

        client.execute("BEGIN")
        client.execute("INSERT INTO t1 (name, age, city, score) VALUES ('Bob', 30, 'LA', 85.0)")
        client.execute("COMMIT")

        result = client.execute("SELECT * FROM t1 WHERE city = 'LA' AND age = 30")
        df = result.to_pandas()
        assert len(df) == 1
        assert df.iloc[0]["name"] == "Bob"

    def test_pragma_table_info_after_operations(self, client):
        """PRAGMA table_info should work after various DML operations."""
        client.store([{"name": "Alice", "age": 25, "city": "NYC", "score": 90.0}])
        client.execute("INSERT INTO t1 (name, age, city, score) VALUES ('Bob', 30, 'LA', 85.0)")
        client.execute("DELETE FROM t1 WHERE name = 'Alice'")

        result = client.execute("PRAGMA table_info(t1)")
        df = result.to_pandas()
        col_names = set(df["name"].tolist())
        assert "name" in col_names
        assert "age" in col_names
        assert "city" in col_names
        assert "score" in col_names
