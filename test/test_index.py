"""Tests for CREATE INDEX / DROP INDEX and index-accelerated SELECT."""

import os
import shutil
import tempfile
import time

import pytest

from apexbase import ApexClient


@pytest.fixture
def client():
    """Create a temporary client for each test."""
    tmpdir = tempfile.mkdtemp(prefix="apexbase_idx_")
    c = ApexClient(dirpath=tmpdir)
    c.create_table("idx_test")
    yield c
    c.close()
    shutil.rmtree(tmpdir, ignore_errors=True)


class TestCreateDropIndex:
    """Test CREATE INDEX / DROP INDEX DDL statements."""

    def test_create_hash_index(self, client):
        """CREATE INDEX with default HASH type."""
        # Insert data first
        rows = [{"city": "Beijing", "age": 30}, {"city": "Shanghai", "age": 25}]
        client.store(rows)
        result = client.execute("CREATE INDEX idx_city ON idx_test (city)")
        assert result is not None

    def test_create_btree_index(self, client):
        """CREATE INDEX ... USING BTREE."""
        rows = [{"score": 90}, {"score": 80}]
        client.store(rows)
        result = client.execute("CREATE INDEX idx_score ON idx_test (score) USING BTREE")
        assert result is not None

    def test_create_unique_index(self, client):
        """CREATE UNIQUE INDEX."""
        rows = [{"email": "a@b.com"}, {"email": "c@d.com"}]
        client.store(rows)
        result = client.execute("CREATE UNIQUE INDEX idx_email ON idx_test (email) USING HASH")
        assert result is not None

    def test_create_index_if_not_exists(self, client):
        """CREATE INDEX IF NOT EXISTS should not error on duplicate."""
        rows = [{"city": "Beijing"}]
        client.store(rows)
        client.execute("CREATE INDEX idx_city ON idx_test (city)")
        # Second create with IF NOT EXISTS should succeed
        result = client.execute("CREATE INDEX IF NOT EXISTS idx_city ON idx_test (city)")
        assert result is not None

    def test_create_index_duplicate_error(self, client):
        """CREATE INDEX without IF NOT EXISTS should error on duplicate."""
        rows = [{"city": "Beijing"}]
        client.store(rows)
        client.execute("CREATE INDEX idx_city ON idx_test (city)")
        with pytest.raises(Exception):
            client.execute("CREATE INDEX idx_city ON idx_test (city)")

    def test_drop_index(self, client):
        """DROP INDEX should remove the index."""
        rows = [{"city": "Beijing"}]
        client.store(rows)
        client.execute("CREATE INDEX idx_city ON idx_test (city)")
        result = client.execute("DROP INDEX idx_city ON idx_test")
        assert result is not None

    def test_drop_index_if_exists(self, client):
        """DROP INDEX IF EXISTS should not error on missing index."""
        result = client.execute("DROP INDEX IF EXISTS idx_nonexist ON idx_test")
        assert result is not None

    def test_drop_index_not_found_error(self, client):
        """DROP INDEX without IF EXISTS should error on missing index."""
        with pytest.raises(Exception):
            client.execute("DROP INDEX idx_nonexist ON idx_test")

    def test_create_index_on_empty_table(self, client):
        """CREATE INDEX on empty table (with schema) should succeed."""
        # Store a row then truncate to have schema but no data
        client.store([{"city": "test"}])
        client.execute("TRUNCATE TABLE idx_test")
        result = client.execute("CREATE INDEX idx_city ON idx_test (city)")
        assert result is not None

    def test_create_index_nonexistent_column(self, client):
        """CREATE INDEX on nonexistent column should error."""
        rows = [{"city": "Beijing"}]
        client.store(rows)
        with pytest.raises(Exception):
            client.execute("CREATE INDEX idx_foo ON idx_test (nonexistent_col)")


class TestIndexAcceleratedSelect:
    """Test that index-accelerated SELECT returns correct results."""

    def _setup_data(self, client, n=100):
        """Insert N rows with city and age columns."""
        cities = ["Beijing", "Shanghai", "Guangzhou", "Shenzhen", "Hangzhou"]
        rows = []
        for i in range(n):
            rows.append({"city": cities[i % len(cities)], "age": 20 + (i % 50), "name": f"user_{i}"})
        client.store(rows)
        return rows

    def test_index_equality_filter(self, client):
        """SELECT with WHERE city = 'X' should return correct results with index."""
        self._setup_data(client, 100)
        client.execute("CREATE INDEX idx_city ON idx_test (city)")

        result = client.execute("SELECT * FROM idx_test WHERE city = 'Beijing'")
        df = result.to_pandas()
        assert len(df) == 20  # 100 / 5 cities
        assert all(df["city"] == "Beijing")

    def test_index_equality_filter_no_match(self, client):
        """SELECT with WHERE on non-existent value should return empty."""
        self._setup_data(client, 50)
        client.execute("CREATE INDEX idx_city ON idx_test (city)")

        result = client.execute("SELECT * FROM idx_test WHERE city = 'NonExistent'")
        df = result.to_pandas()
        assert len(df) == 0

    def test_index_in_filter(self, client):
        """SELECT with WHERE city IN (...) should work with index."""
        self._setup_data(client, 100)
        client.execute("CREATE INDEX idx_city ON idx_test (city)")

        result = client.execute("SELECT * FROM idx_test WHERE city IN ('Beijing', 'Shanghai')")
        df = result.to_pandas()
        assert len(df) == 40  # 20 Beijing + 20 Shanghai

    def test_index_with_order_by(self, client):
        """Index SELECT + ORDER BY should return correctly ordered results."""
        self._setup_data(client, 100)
        client.execute("CREATE INDEX idx_city ON idx_test (city)")

        result = client.execute("SELECT * FROM idx_test WHERE city = 'Beijing' ORDER BY age DESC")
        df = result.to_pandas()
        assert len(df) == 20
        ages = df["age"].tolist()
        assert ages == sorted(ages, reverse=True)

    def test_index_with_limit(self, client):
        """Index SELECT + LIMIT should respect the limit."""
        self._setup_data(client, 100)
        client.execute("CREATE INDEX idx_city ON idx_test (city)")

        result = client.execute("SELECT * FROM idx_test WHERE city = 'Beijing' LIMIT 5")
        df = result.to_pandas()
        assert len(df) == 5

    def test_index_correctness_vs_scan(self, client):
        """Index path results should match full-scan results."""
        self._setup_data(client, 200)

        # Query WITHOUT index (full scan)
        result_scan = client.execute("SELECT name FROM idx_test WHERE city = 'Guangzhou' ORDER BY name")
        df_scan = result_scan.to_pandas()

        # Create index and query WITH index
        client.execute("CREATE INDEX idx_city ON idx_test (city)")
        result_idx = client.execute("SELECT name FROM idx_test WHERE city = 'Guangzhou' ORDER BY name")
        df_idx = result_idx.to_pandas()

        # Same row count
        assert len(df_scan) == len(df_idx)
        # Same name values
        assert df_scan["name"].tolist() == df_idx["name"].tolist()

    def test_index_drop_fallback_to_scan(self, client):
        """After DROP INDEX, queries should still work via full scan."""
        self._setup_data(client, 50)
        client.execute("CREATE INDEX idx_city ON idx_test (city)")
        client.execute("DROP INDEX idx_city ON idx_test")

        # Should still work without index
        result = client.execute("SELECT * FROM idx_test WHERE city = 'Beijing'")
        df = result.to_pandas()
        assert len(df) == 10


class TestIndexWithDML:
    """Test that index stays in sync with INSERT/UPDATE/DELETE via SQL."""

    def test_sql_insert_updates_index(self, client):
        """SQL INSERT after CREATE INDEX should be found via index."""
        client.store([{"city": "Beijing", "age": 30}])
        client.execute("CREATE INDEX idx_city ON idx_test (city)")

        # SQL INSERT adds a new row
        client.execute("INSERT INTO idx_test (city, age) VALUES ('Shanghai', 40)")

        # Index should find the new row
        result = client.execute("SELECT * FROM idx_test WHERE city = 'Shanghai'")
        df = result.to_pandas()
        assert len(df) == 1
        assert df.iloc[0]["age"] == 40

    def test_sql_delete_updates_index(self, client):
        """SQL DELETE should remove entries from the index."""
        cities = ["Beijing", "Shanghai", "Guangzhou"]
        client.store([{"city": c, "age": 20 + i} for i, c in enumerate(cities)])
        client.execute("CREATE INDEX idx_city ON idx_test (city)")

        # Verify Beijing exists
        r1 = client.execute("SELECT * FROM idx_test WHERE city = 'Beijing'")
        assert len(r1.to_pandas()) == 1

        # Delete Beijing row
        client.execute("DELETE FROM idx_test WHERE city = 'Beijing'")

        # Index-accelerated query should return empty
        r2 = client.execute("SELECT * FROM idx_test WHERE city = 'Beijing'")
        df2 = r2.to_pandas()
        assert len(df2) == 0

        # Other cities still findable
        r3 = client.execute("SELECT * FROM idx_test WHERE city = 'Shanghai'")
        assert len(r3.to_pandas()) == 1

    def test_sql_update_updates_index(self, client):
        """SQL UPDATE should update index entries."""
        client.store([{"city": "Beijing", "age": 30}, {"city": "Shanghai", "age": 25}])
        client.execute("CREATE INDEX idx_city ON idx_test (city)")

        # Update Beijing -> Chengdu
        client.execute("UPDATE idx_test SET city = 'Chengdu' WHERE city = 'Beijing'")

        # Beijing should no longer be found
        r1 = client.execute("SELECT * FROM idx_test WHERE city = 'Beijing'")
        assert len(r1.to_pandas()) == 0

        # Chengdu should now be found
        r2 = client.execute("SELECT * FROM idx_test WHERE city = 'Chengdu'")
        df2 = r2.to_pandas()
        assert len(df2) == 1
        assert df2.iloc[0]["age"] == 30

    def test_delete_all_then_insert(self, client):
        """DELETE all rows then INSERT new rows - index should reflect."""
        client.store([{"city": "Beijing"}, {"city": "Shanghai"}])
        client.execute("CREATE INDEX idx_city ON idx_test (city)")

        # Delete all
        client.execute("DELETE FROM idx_test")
        r1 = client.execute("SELECT * FROM idx_test WHERE city = 'Beijing'")
        assert len(r1.to_pandas()) == 0

        # Insert new
        client.execute("INSERT INTO idx_test (city) VALUES ('Hangzhou')")
        r2 = client.execute("SELECT * FROM idx_test WHERE city = 'Hangzhou'")
        assert len(r2.to_pandas()) == 1

    def test_store_api_after_index(self, client):
        """Python store() API after CREATE INDEX - index won't auto-update (store uses bindings path)."""
        client.store([{"city": "Beijing"}])
        client.execute("CREATE INDEX idx_city ON idx_test (city)")

        # store() goes through bindings, not SQL executor - index may be stale
        # This test documents the current behavior
        client.store([{"city": "Shanghai"}])

        # Full scan (COUNT with WHERE uses scan path) should find both
        result = client.execute("SELECT COUNT(*) FROM idx_test WHERE city = 'Shanghai'")
        df = result.to_pandas()
        assert df.iloc[0, 0] >= 1  # At minimum scan path finds it


class TestIndexPerformance:
    """Sanity check that index path doesn't cause performance regression."""

    def test_no_regression_without_index(self, client):
        """Queries without indexes should not be slower."""
        cities = ["Beijing", "Shanghai", "Guangzhou"]
        rows = [{"city": cities[i % 3], "age": i} for i in range(1000)]
        client.store(rows)

        start = time.time()
        for _ in range(10):
            client.execute("SELECT * FROM idx_test WHERE city = 'Beijing'")
        elapsed_no_idx = time.time() - start

        # Should complete in reasonable time (< 5s for 10 queries on 1K rows)
        assert elapsed_no_idx < 5.0, f"No-index queries took {elapsed_no_idx:.2f}s"
