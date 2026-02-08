"""Tests for CTE (WITH...AS), INSERT...SELECT, and EXPLAIN SQL features."""
import pytest
import tempfile
import os
import shutil
from apexbase import ApexStorage


def get_rows(result):
    """Extract rows from execute() result dict."""
    return result.get('rows', [])

def get_columns(result):
    """Extract column names from execute() result dict."""
    return result.get('columns', [])

def get_scalar(result):
    """Extract scalar value (first cell) from result."""
    rows = get_rows(result)
    if rows and rows[0]:
        return rows[0][0]
    return None

def get_col_values(result, col_name):
    """Extract all values from a specific column."""
    cols = get_columns(result)
    if col_name not in cols:
        return []
    idx = cols.index(col_name)
    return [row[idx] for row in get_rows(result)]


@pytest.fixture
def db_with_data():
    """Create a database with sample data in a unique temp directory."""
    tmpdir = tempfile.mkdtemp(prefix="apextest_cte_")
    storage = ApexStorage(tmpdir)
    storage.execute("CREATE TABLE IF NOT EXISTS users (name TEXT, age INT, city TEXT)")
    storage.execute("TRUNCATE TABLE users")
    storage.execute("INSERT INTO users (name, age, city) VALUES ('Alice', 30, 'NYC')")
    storage.execute("INSERT INTO users (name, age, city) VALUES ('Bob', 25, 'LA')")
    storage.execute("INSERT INTO users (name, age, city) VALUES ('Charlie', 35, 'NYC')")
    storage.execute("INSERT INTO users (name, age, city) VALUES ('Diana', 28, 'LA')")
    storage.execute("INSERT INTO users (name, age, city) VALUES ('Eve', 32, 'NYC')")
    yield storage
    storage.close()
    shutil.rmtree(tmpdir, ignore_errors=True)


# ========== EXPLAIN Tests ==========

class TestExplain:
    def test_explain_simple_select(self, db_with_data):
        result = db_with_data.execute("EXPLAIN SELECT * FROM users")
        plan = get_rows(result)[0][0]
        assert 'Query Plan' in plan
        assert 'Scan' in plan
        assert 'users' in plan

    def test_explain_select_with_where(self, db_with_data):
        result = db_with_data.execute("EXPLAIN SELECT name FROM users WHERE age > 30")
        plan = get_rows(result)[0][0]
        assert 'Filter' in plan
        assert 'WHERE' in plan

    def test_explain_select_with_group_by(self, db_with_data):
        result = db_with_data.execute("EXPLAIN SELECT city, COUNT(*) FROM users GROUP BY city")
        plan = get_rows(result)[0][0]
        assert 'GroupBy' in plan
        assert 'city' in plan

    def test_explain_select_with_order_limit(self, db_with_data):
        result = db_with_data.execute("EXPLAIN SELECT * FROM users ORDER BY age DESC LIMIT 3")
        plan = get_rows(result)[0][0]
        assert 'Sort' in plan
        assert 'DESC' in plan
        assert 'Limit: 3' in plan

    def test_explain_analyze(self, db_with_data):
        result = db_with_data.execute("EXPLAIN ANALYZE SELECT * FROM users WHERE age > 25")
        plan = get_rows(result)[0][0]
        assert 'Actual Time' in plan
        assert 'Actual Rows' in plan

    def test_explain_insert(self, db_with_data):
        result = db_with_data.execute("EXPLAIN INSERT INTO users (name, age, city) VALUES ('Frank', 40, 'SF')")
        plan = get_rows(result)[0][0]
        assert 'Insert' in plan
        assert 'users' in plan

    def test_explain_shows_row_count(self, db_with_data):
        result = db_with_data.execute("EXPLAIN SELECT * FROM users")
        plan = get_rows(result)[0][0]
        assert 'Rows: ~' in plan
        assert 'Columns: 3' in plan


# ========== INSERT...SELECT Tests ==========

class TestInsertSelect:
    def test_insert_select_same_table(self, db_with_data):
        db_with_data.execute("CREATE TABLE IF NOT EXISTS users_backup (name TEXT, age INT, city TEXT)")
        db_with_data.execute("TRUNCATE TABLE users_backup")
        db_with_data.execute("INSERT INTO users_backup (name, age, city) SELECT name, age, city FROM users")
        result = db_with_data.execute("SELECT COUNT(*) FROM users_backup")
        assert get_scalar(result) == 5

    def test_insert_select_with_filter(self, db_with_data):
        db_with_data.execute("CREATE TABLE IF NOT EXISTS nyc_users (name TEXT, age INT, city TEXT)")
        db_with_data.execute("TRUNCATE TABLE nyc_users")
        db_with_data.execute("INSERT INTO nyc_users (name, age, city) SELECT name, age, city FROM users WHERE city = 'NYC'")
        result = db_with_data.execute("SELECT COUNT(*) FROM nyc_users")
        assert get_scalar(result) == 3

    def test_insert_select_with_order_limit(self, db_with_data):
        db_with_data.execute("CREATE TABLE IF NOT EXISTS top_users (name TEXT, age INT)")
        db_with_data.execute("TRUNCATE TABLE top_users")
        db_with_data.execute("INSERT INTO top_users (name, age) SELECT name, age FROM users ORDER BY age DESC LIMIT 2")
        result = db_with_data.execute("SELECT * FROM top_users ORDER BY age DESC")
        rows = get_rows(result)
        assert len(rows) == 2
        ages = get_col_values(result, 'age')
        assert ages[0] == 35  # Charlie
        assert ages[1] == 32  # Eve

    def test_insert_select_aggregation(self, db_with_data):
        db_with_data.execute("CREATE TABLE IF NOT EXISTS city_stats (city TEXT, cnt INT)")
        db_with_data.execute("TRUNCATE TABLE city_stats")
        db_with_data.execute("INSERT INTO city_stats (city, cnt) SELECT city, COUNT(*) FROM users GROUP BY city")
        result = db_with_data.execute("SELECT * FROM city_stats ORDER BY city")
        rows = get_rows(result)
        assert len(rows) == 2
        cities = get_col_values(result, 'city')
        cnts = get_col_values(result, 'cnt')
        la_idx = cities.index('LA')
        nyc_idx = cities.index('NYC')
        assert cnts[la_idx] == 2
        assert cnts[nyc_idx] == 3

    def test_insert_select_empty_result(self, db_with_data):
        db_with_data.execute("CREATE TABLE IF NOT EXISTS empty_table (name TEXT, age INT)")
        db_with_data.execute("TRUNCATE TABLE empty_table")
        db_with_data.execute("INSERT INTO empty_table (name, age) SELECT name, age FROM users WHERE age > 100")
        result = db_with_data.execute("SELECT COUNT(*) FROM empty_table")
        assert get_scalar(result) == 0


# ========== CTE (WITH...AS) Tests ==========

class TestCTE:
    def test_simple_cte(self, db_with_data):
        result = db_with_data.execute("""
            WITH seniors AS (SELECT name, age FROM users WHERE age >= 30)
            SELECT * FROM seniors ORDER BY age
        """)
        rows = get_rows(result)
        assert len(rows) == 3
        names = get_col_values(result, 'name')
        assert names == ['Alice', 'Eve', 'Charlie']

    def test_cte_with_aggregation(self, db_with_data):
        result = db_with_data.execute("""
            WITH city_counts AS (SELECT city, COUNT(*) AS cnt FROM users GROUP BY city)
            SELECT * FROM city_counts ORDER BY cnt DESC
        """)
        rows = get_rows(result)
        assert len(rows) == 2
        cnts = get_col_values(result, 'cnt')
        assert cnts[0] == 3  # NYC
        assert cnts[1] == 2  # LA

    def test_cte_with_filter_on_cte(self, db_with_data):
        result = db_with_data.execute("""
            WITH young AS (SELECT name, age, city FROM users WHERE age < 30)
            SELECT name FROM young WHERE city = 'LA'
        """)
        rows = get_rows(result)
        assert len(rows) == 2
        names = set(get_col_values(result, 'name'))
        assert 'Bob' in names
        assert 'Diana' in names

    def test_cte_preserves_original_data(self, db_with_data):
        """CTE should not modify the original table."""
        db_with_data.execute("""
            WITH temp AS (SELECT name FROM users WHERE age > 30)
            SELECT * FROM temp
        """)
        result = db_with_data.execute("SELECT COUNT(*) FROM users")
        assert get_scalar(result) == 5  # Original data unchanged
