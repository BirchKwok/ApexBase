"""
Test DuckDB-compatible SELECT clause features:
- SELECT * EXCLUDE (...)
- SELECT * REPLACE (expr AS col)
- SELECT COLUMNS('regex')
- SELECT DISTINCT ON (...)
"""
import pytest
import tempfile
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'apexbase', 'python'))

try:
    from apexbase import ApexClient
except ImportError as e:
    pytest.skip(f"ApexBase not available: {e}", allow_module_level=True)


@pytest.fixture
def client():
    """Create a client with test data having multiple columns."""
    with tempfile.TemporaryDirectory() as temp_dir:
        c = ApexClient(dirpath=temp_dir)
        c.create_table("addresses")
        c.use_table("addresses")
        c.store([
            {"city": "Beijing", "name": "Alice", "age": 30, "score": 85.5},
            {"city": "Shanghai", "name": "Bob", "age": 25, "score": 90.0},
            {"city": "Beijing", "name": "Charlie", "age": 35, "score": 78.0},
            {"city": "Guangzhou", "name": "Diana", "age": 28, "score": 92.5},
        ])
        c.flush()
        yield c
        c.close()


class TestExclude:
    """Test SELECT * EXCLUDE (...) syntax."""

    def test_exclude_single_column(self, client):
        result = client.execute("SELECT * EXCLUDE (name) FROM addresses ORDER BY age").to_dict()
        assert len(result) == 4
        for row in result:
            assert "name" not in row
            assert "city" in row
            assert "age" in row
            assert "score" in row

    def test_exclude_multiple_columns(self, client):
        result = client.execute("SELECT * EXCLUDE (name, score) FROM addresses ORDER BY age").to_dict()
        assert len(result) == 4
        for row in result:
            assert "name" not in row
            assert "score" not in row
            assert "city" in row
            assert "age" in row

    def test_exclude_with_where(self, client):
        result = client.execute("SELECT * EXCLUDE (name) FROM addresses WHERE age = 30").to_dict()
        assert len(result) == 1
        assert "name" not in result[0]
        assert result[0]["city"] == "Beijing"


class TestReplace:
    """Test SELECT * REPLACE (expr AS col) syntax."""

    def test_replace_single_column(self, client):
        result = client.execute("SELECT * REPLACE (UPPER(city) AS city) FROM addresses ORDER BY city").to_dict()
        assert len(result) == 4
        # All cities should be uppercase
        for row in result:
            assert row["city"] == row["city"].upper()

    def test_replace_with_arithmetic(self, client):
        result = client.execute(
            "SELECT * REPLACE (age + 1 AS age) FROM addresses WHERE name = 'Bob'"
        ).to_dict()
        assert result[0]["age"] == 26

    def test_replace_preserves_other_columns(self, client):
        result = client.execute(
            "SELECT * REPLACE (LOWER(name) AS name) FROM addresses WHERE name = 'Alice'"
        ).to_dict()
        assert result[0]["name"] == "alice"
        assert "city" in result[0]
        assert result[0]["city"] == "Beijing"


class TestColumns:
    """Test SELECT COLUMNS('regex') syntax."""

    def test_columns_regex_match_single(self, client):
        result = client.execute("SELECT COLUMNS('age') FROM addresses ORDER BY age").to_dict()
        assert len(result) == 4
        assert len(result[0]) == 1
        assert "age" in result[0]

    def test_columns_regex_match_multiple(self, client):
        # '^ag' matches columns starting with 'ag' (only 'age')
        result = client.execute("SELECT COLUMNS('^ag') FROM addresses ORDER BY age").to_dict()
        assert len(result) == 4
        for row in result:
            assert "age" in row
            assert "city" not in row
            assert "name" not in row
            assert "score" not in row

    def test_columns_regex_contains_e(self, client):
        # 'e' matches columns containing 'e': name, age, score
        result = client.execute("SELECT COLUMNS('e') FROM addresses").to_dict()
        assert len(result) == 4
        for row in result:
            assert "city" not in row  # 'city' does NOT contain 'e'


class TestDistinctOn:
    """Test SELECT DISTINCT ON (...) syntax."""

    def test_distinct_on_single_column(self, client):
        result = client.execute(
            "SELECT DISTINCT ON(city) city, name, age FROM addresses ORDER BY age"
        ).to_dict()
        # Should return one row per city, the one with smallest age (first in ORDER BY)
        cities = {r["city"] for r in result}
        assert cities == {"Beijing", "Shanghai", "Guangzhou"}
        # Beijing should have the younger person (Alice, age 30) not Charlie (age 35)
        beijing_rows = [r for r in result if r["city"] == "Beijing"]
        assert len(beijing_rows) == 1
        assert beijing_rows[0]["name"] == "Alice"

    def test_distinct_on_multiple_columns(self, client):
        # DISTINCT ON(city) - each city appears once
        result = client.execute(
            "SELECT DISTINCT ON(city) city, name FROM addresses ORDER BY age"
        ).to_dict()
        assert len(result) == 3

    def test_distinct_on_with_order_desc(self, client):
        result = client.execute(
            "SELECT DISTINCT ON(city) city, age FROM addresses ORDER BY age DESC"
        ).to_dict()
        # First row per city after ordering by age DESC -> highest age per city
        beijing_rows = [r for r in result if r["city"] == "Beijing"]
        assert beijing_rows[0]["age"] == 35  # Charlie is older

    def test_distinct_on_with_limit(self, client):
        result = client.execute(
            "SELECT DISTINCT ON(city) city, name, age FROM addresses ORDER BY age LIMIT 2"
        ).to_dict()
        assert len(result) <= 2

    def test_distinct_on_plain_select(self, client):
        """Plain DISTINCT still works alongside DISTINCT ON."""
        result = client.execute("SELECT DISTINCT city FROM addresses ORDER BY city").to_dict()
        cities = {r["city"] for r in result}
        assert cities == {"Beijing", "Shanghai", "Guangzhou"}
