"""
OLTP Test Suite for ApexBase

Tests transactional workloads:
- Single/batch insert
- Point lookup by _id
- Update single/multiple rows
- Delete single/multiple rows
- String equality filter with LIMIT
- Numeric range filter with LIMIT
- Combined string + numeric filter
- Insert-then-read consistency
- Delete-then-count consistency
"""

import pytest
import tempfile
import shutil
from pathlib import Path
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'apexbase', 'python'))

try:
    from apexbase import ApexClient
except ImportError as e:
    pytest.skip(f"ApexBase not available: {e}", allow_module_level=True)

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False


@pytest.fixture
def oltp_client():
    """Create a client with 1000 rows of OLTP-style data."""
    tmp = tempfile.mkdtemp()
    client = ApexClient(os.path.join(tmp, "oltp_test"))
    cities = ["Beijing", "Shanghai", "Shenzhen", "Guangzhou", "Hangzhou"]
    rows = []
    for i in range(1000):
        rows.append({
            "user_id": i + 1,
            "age": 20 + (i % 50),
            "balance": 100.0 + i * 1.5,
            "city": cities[i % 5],
            "active": i % 3 != 0,
        })
    client.store(rows)
    yield client
    shutil.rmtree(tmp, ignore_errors=True)


@pytest.fixture
def empty_client():
    """Create an empty client."""
    tmp = tempfile.mkdtemp()
    client = ApexClient(os.path.join(tmp, "empty_test"))
    yield client
    shutil.rmtree(tmp, ignore_errors=True)


def _count(client):
    """Helper to get row count via SQL."""
    result = client.execute("SELECT COUNT(*) as cnt FROM default")
    return result[0]["cnt"]


class TestOltpInsert:
    def test_insert_single_row(self, empty_client):
        empty_client.store({"name": "Alice", "age": 30})
        assert _count(empty_client) == 1

    def test_insert_batch_rows(self, empty_client):
        rows = [{"name": f"user_{i}", "age": 20 + i} for i in range(100)]
        empty_client.store(rows)
        assert _count(empty_client) == 100

    def test_insert_columnar(self, empty_client):
        data = {
            "name": [f"user_{i}" for i in range(50)],
            "score": [float(i) for i in range(50)],
        }
        empty_client.store(data)
        assert _count(empty_client) == 50

    def test_insert_preserves_types(self, empty_client):
        empty_client.store({
            "int_val": 42,
            "float_val": 3.14,
            "str_val": "hello",
            "bool_val": True,
        })
        result = empty_client.execute("SELECT * FROM default")
        assert len(result) == 1
        row = result[0]
        assert row["int_val"] == 42
        assert abs(row["float_val"] - 3.14) < 0.01
        assert row["str_val"] == "hello"
        assert row["bool_val"] is True

    def test_insert_incremental(self, oltp_client):
        assert _count(oltp_client) == 1000
        oltp_client.store([{"user_id": 1001, "age": 99, "balance": 0.0, "city": "Chengdu", "active": True}])
        assert _count(oltp_client) == 1001

    @pytest.mark.skipif(not PANDAS_AVAILABLE, reason="pandas not available")
    def test_insert_dataframe(self, empty_client):
        df = pd.DataFrame({"x": [1, 2, 3], "y": [4.0, 5.0, 6.0]})
        empty_client.store(df)
        assert _count(empty_client) == 3


class TestOltpPointLookup:
    def test_lookup_by_id(self, oltp_client):
        # _id is auto-generated; just verify the lookup returns exactly 1 row
        result = oltp_client.execute("SELECT * FROM default WHERE _id = 1")
        assert len(result) == 1

    def test_lookup_nonexistent_id(self, oltp_client):
        result = oltp_client.execute("SELECT * FROM default WHERE _id = 999999")
        assert len(result) == 0

    def test_retrieve_by_id(self, oltp_client):
        row = oltp_client.retrieve(1)
        assert row is not None

    def test_retrieve_nonexistent(self, oltp_client):
        row = oltp_client.retrieve(999999)
        assert row is None


class TestOltpUpdate:
    def test_update_single_field(self, oltp_client):
        oltp_client.execute("UPDATE default SET age = 99 WHERE user_id = 1")
        result = oltp_client.execute("SELECT age FROM default WHERE user_id = 1")
        assert len(result) >= 1
        ages = [r["age"] for r in result]
        assert 99 in ages

    def test_update_multiple_rows(self, oltp_client):
        oltp_client.execute("UPDATE default SET balance = 0.0 WHERE city = 'Shanghai'")
        result = oltp_client.execute("SELECT COUNT(*) as cnt FROM default WHERE balance = 0.0")
        assert result[0]["cnt"] == 200  # Shanghai = every 5th row = 200

    def test_update_no_match(self, oltp_client):
        oltp_client.execute("UPDATE default SET age = 0 WHERE city = 'NonExistent'")
        result = oltp_client.execute("SELECT COUNT(*) as cnt FROM default WHERE age = 0")
        assert result[0]["cnt"] == 0


class TestOltpDelete:
    def test_delete_single_row(self, oltp_client):
        oltp_client.execute("DELETE FROM default WHERE user_id = 1")
        result = oltp_client.execute("SELECT COUNT(*) as cnt FROM default WHERE user_id = 1")
        assert result[0]["cnt"] == 0

    def test_delete_by_string_filter(self, oltp_client):
        oltp_client.execute("DELETE FROM default WHERE city = 'Beijing'")
        result = oltp_client.execute("SELECT COUNT(*) as cnt FROM default")
        assert result[0]["cnt"] == 800  # 200 Beijing rows deleted

    def test_delete_no_match(self, oltp_client):
        oltp_client.execute("DELETE FROM default WHERE city = 'NonExistent'")
        assert _count(oltp_client) == 1000


class TestOltpStringFilter:
    def test_string_eq_with_limit(self, oltp_client):
        result = oltp_client.execute("SELECT * FROM default WHERE city = 'Shenzhen' LIMIT 10")
        assert len(result) == 10
        for row in result:
            assert row["city"] == "Shenzhen"

    def test_string_eq_full_scan(self, oltp_client):
        result = oltp_client.execute("SELECT * FROM default WHERE city = 'Beijing'")
        assert len(result) == 200

    def test_string_eq_no_match(self, oltp_client):
        result = oltp_client.execute("SELECT * FROM default WHERE city = 'Tokyo'")
        assert len(result) == 0


class TestOltpNumericFilter:
    def test_range_filter_with_limit(self, oltp_client):
        result = oltp_client.execute(
            "SELECT * FROM default WHERE age BETWEEN 30 AND 35 LIMIT 50"
        )
        assert 0 < len(result) <= 50
        for row in result:
            assert 30 <= row["age"] <= 35

    def test_equality_filter(self, oltp_client):
        result = oltp_client.execute("SELECT * FROM default WHERE age = 25")
        assert len(result) > 0
        for row in result:
            assert row["age"] == 25

    def test_comparison_filter(self, oltp_client):
        result = oltp_client.execute("SELECT * FROM default WHERE age > 65")
        assert len(result) > 0
        for row in result:
            assert row["age"] > 65


class TestOltpCombinedFilter:
    def test_string_and_numeric(self, oltp_client):
        result = oltp_client.execute(
            "SELECT * FROM default WHERE city = 'Beijing' AND age > 50"
        )
        assert len(result) > 0
        for row in result:
            assert row["city"] == "Beijing"
            assert row["age"] > 50

    def test_or_condition(self, oltp_client):
        result = oltp_client.execute(
            "SELECT * FROM default WHERE age < 21 OR age > 68"
        )
        assert len(result) > 0
        for row in result:
            assert row["age"] < 21 or row["age"] > 68


class TestOltpConsistency:
    def test_insert_then_count(self, empty_client):
        for i in range(5):
            empty_client.store([{"val": j} for j in range(10)])
        assert _count(empty_client) == 50

    def test_delete_then_insert_then_count(self, oltp_client):
        oltp_client.execute("DELETE FROM default WHERE city = 'Beijing'")
        oltp_client.store([{"user_id": 9999, "age": 1, "balance": 0.0, "city": "Beijing", "active": True}])
        result = oltp_client.execute("SELECT COUNT(*) as cnt FROM default WHERE city = 'Beijing'")
        assert result[0]["cnt"] == 1

    @pytest.mark.skipif(not PANDAS_AVAILABLE, reason="pandas not available")
    def test_query_to_pandas(self, oltp_client):
        result = oltp_client.execute("SELECT * FROM default WHERE city = 'Shanghai' LIMIT 5")
        df = result.to_pandas()
        assert len(df) == 5
        assert "city" in df.columns
        assert all(df["city"] == "Shanghai")
