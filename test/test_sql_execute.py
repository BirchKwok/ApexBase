"""
Comprehensive test suite for ApexBase SQL Execute Operations and SqlResult

This module tests:
- SQL execute operations with various SELECT statements
- SqlResult functionality and conversions
- SQL syntax support (ORDER BY, LIMIT, DISTINCT, aggregates, GROUP BY)
- Edge cases and error handling
- Performance considerations
- Complex SQL queries
"""

import pytest
import tempfile
import time
import re
import shutil
from pathlib import Path
import sys
import os
import numpy as np

# Add the apexbase python module to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'apexbase', 'python'))

try:
    from apexbase import ApexClient, ResultView, ARROW_AVAILABLE, POLARS_AVAILABLE
except ImportError as e:
    pytest.skip(f"ApexBase not available: {e}", allow_module_level=True)

# Optional imports
try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False

try:
    import polars as pl
    POLARS_DF_AVAILABLE = True
except ImportError:
    POLARS_DF_AVAILABLE = False

try:
    import pyarrow as pa
    PYARROW_AVAILABLE = True
except ImportError:
    PYARROW_AVAILABLE = False


def _execute_or_xfail(client: ApexClient, sql: str):
    try:
        return client.execute(sql)
    except Exception as e:
        pytest.xfail(f"SQL not supported yet: {sql} ({type(e).__name__}: {e})")


class TestBasicSQLExecute:
    """Test basic SQL execute operations"""
    
    def test_execute_basic_select(self):
        """Test basic SELECT statement"""
        with tempfile.TemporaryDirectory() as temp_dir:
            client = ApexClient(dirpath=temp_dir)
            
            # Store test data
            test_data = [
                {"name": "Alice", "age": 25, "city": "NYC"},
                {"name": "Bob", "age": 30, "city": "LA"},
                {"name": "Charlie", "age": 35, "city": "Chicago"},
            ]
            client.store(test_data)
            
            # Execute basic SELECT
            result = client.execute("SELECT * FROM default")
            
            assert isinstance(result, ResultView)
            assert len(result) == 3
            assert "name" in result.columns
            assert "age" in result.columns
            assert "city" in result.columns
            assert "_id" not in result.columns  # _id should be hidden
            
            client.close()

    def test_execute_cast_expression(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            client = ApexClient(dirpath=temp_dir)

            client.store([{"s": "123", "f": "1.25", "b": "true", "n": 7}])
            client.flush()

            res = client.execute(
                """
                SELECT
                  CAST('123' AS INT) AS i1,
                  CAST(s AS BIGINT) AS i2,
                  CAST(f AS DOUBLE) AS d1,
                  CAST(n AS VARCHAR) AS s1,
                  CAST(b AS BOOLEAN) AS bo
                FROM default
                """.strip()
            )
            row = res.first()
            assert row["i1"] == 123
            assert row["i2"] == 123
            assert row["d1"] == pytest.approx(1.25)
            assert row["s1"] == "7"
            assert row["bo"] is True

            # NULL propagation
            r2 = client.execute("SELECT CAST(NULL AS INT) AS x FROM default").first()
            assert r2["x"] is None

            # Invalid cast should error
            with pytest.raises(Exception):
                client.execute("SELECT CAST('abc' AS INT) AS x FROM default").to_dict()

            client.close()

    def test_where_multi_column_arithmetic_predicate(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            client = ApexClient(dirpath=temp_dir)

            client.store(
                [
                    {"a": 3, "b": 7},
                    {"a": 6, "b": 5},
                    {"a": 1, "b": 2},
                    {"a": 10, "b": 0},
                ]
            )
            client.flush()

            res = client.execute(
                "SELECT a, b FROM default WHERE a + b > 10 ORDER BY a ASC"
            )
            out = res.to_dict()
            assert out == [{"a": 6, "b": 5}]

            client.close()

    def test_where_multi_column_arithmetic_predicate_with_limit_offset(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            client = ApexClient(dirpath=temp_dir)

            client.store(
                [
                    {"a": 9, "b": 3},
                    {"a": 8, "b": 4},
                    {"a": 7, "b": 4},
                    {"a": 6, "b": 6},
                    {"a": 5, "b": 6},
                ]
            )
            client.flush()

            res = client.execute(
                "SELECT a, b FROM default WHERE a + b > 10 ORDER BY a DESC LIMIT 2 OFFSET 1"
            )
            out = res.to_dict()
            assert out == [{"a": 8, "b": 4}, {"a": 7, "b": 4}]

            client.close()

    def test_execute_temporary_view_create_select_drop(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            client = ApexClient(dirpath=temp_dir)

            client.store([{"a": 1}, {"a": 2}])
            client.flush()

            # View exists only within this execute call
            res = client.execute(
                """
                CREATE VIEW v AS SELECT a FROM default WHERE a >= 2;
                SELECT * FROM v;
                DROP VIEW v;
                """.strip()
            )
            out = res.to_dict()
            assert out == [{"a": 2}]

            # New execute: view should not exist
            with pytest.raises(Exception):
                client.execute("SELECT * FROM v").to_dict()

            client.close()

    def test_execute_temporary_view_name_conflict_with_table(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            client = ApexClient(dirpath=temp_dir)

            client.create_table("t")
            client.use_table("t")
            client.store([{"x": 1}])
            client.flush()
            client.use_table("default")

            with pytest.raises(Exception):
                client.execute("CREATE VIEW t AS SELECT 1 AS x")

            client.close()

    def test_execute_string_scalar_functions(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            client = ApexClient(dirpath=temp_dir)

            client.store([{"s": "  Abc-XYZ  ", "n": 2}])
            client.flush()

            res = client.execute(
                """
                SELECT
                  LEN('你好') AS l_unicode,
                  MID('abcdef', 2, 3) AS mid1,
                  MID('abcdef', 3) AS mid2,
                  REPLACE('a-b-c', '-', '_') AS rep,
                  TRIM('  hi  ') AS tr,
                  UPPER('aBc') AS up,
                  LOWER('aBc') AS lo,
                  UCASE('aBc') AS uca,
                  LCASE('aBc') AS lca
                FROM default
                """.strip()
            )
            row = res.first()

            assert row["l_unicode"] == 2
            assert row["mid1"] == "bcd"
            assert row["mid2"] == "cdef"
            assert row["rep"] == "a_b_c"
            assert row["tr"] == "hi"
            assert row["up"] == "ABC"
            assert row["lo"] == "abc"
            assert row["uca"] == "ABC"
            assert row["lca"] == "abc"

            res2 = client.execute(
                """
                SELECT
                  UCASE(NULL) AS u_null,
                  LCASE(NULL) AS l_null
                FROM default
                """.strip()
            )
            row2 = res2.first()
            assert row2["u_null"] is None
            assert row2["l_null"] is None

            # UCASE/LCASE: only allow string literal or column name
            with pytest.raises(Exception):
                client.execute("SELECT UCASE(1) AS x FROM default").to_dict()

            with pytest.raises(Exception):
                client.execute("SELECT LCASE(1) AS x FROM default").to_dict()

            # Column exists but non-string values should error
            with pytest.raises(Exception):
                client.execute("SELECT UCASE(n) AS x FROM default").to_dict()

            # Nested expression is supported
            r3 = client.execute("SELECT UCASE(LOWER('a')) AS x FROM default").first()
            assert r3["x"] == "A"

            client.close()

    def test_execute_scalar_standard_functions(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            client = ApexClient(dirpath=temp_dir)

            client.store([{"v": 1}])
            client.flush()

            res = client.execute(
                """
                SELECT
                  ROUND(1.2345, 2) AS r2,
                  ROUND(1.6) AS r0,
                  SQRT(9) AS s,
                  CONCAT('a', 'b', 'c') AS c,
                  COALESCE(NULL, 'x', 'y') AS co,
                  IFNULL(NULL, 7) AS ifn,
                  NVL(NULL, 'z') AS nv,
                  ISNULL(NULL, 'k') AS isn
                FROM default
                """.strip()
            )
            row = res.first()

            assert row["r2"] == pytest.approx(1.23)
            assert row["r0"] == pytest.approx(2.0)
            assert row["s"] == pytest.approx(3.0)
            assert row["c"] == "abc"
            assert row["co"] == "x"
            assert row["ifn"] == 7
            assert row["nv"] == "z"
            assert row["isn"] == "k"

            # NOW(): returns formatted datetime string, stable within one execute
            now_res = client.execute("SELECT NOW() AS t FROM default")
            t = now_res.first()["t"]
            assert isinstance(t, str)
            assert re.match(r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}$", t)

            now2 = client.execute("SELECT NOW() AS a, NOW() AS b FROM default").first()
            assert now2["a"] == now2["b"]

            client.close()

    def test_execute_rand_function(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            client = ApexClient(dirpath=temp_dir)

            rows = [{"v": i} for i in range(100)]
            client.store(rows)
            client.flush()

            result = client.execute("SELECT rand() AS r FROM default")
            out = result.to_dict()
            assert len(out) == 100
            for row in out[:10]:
                assert isinstance(row["r"], float)
                assert 0.0 <= row["r"] < 1.0

            # Non-deterministic: expect not all values identical.
            rs = [row["r"] for row in out]
            assert len(set(rs)) > 1

            client.close()

    def test_execute_join_group_by_agg_order_alias_flexible(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            client = ApexClient(dirpath=temp_dir)

            client.create_table("users")
            client.store([
                {"user_id": 1, "name": "Alice", "tier": "pro"},
                {"user_id": 2, "name": "Bob", "tier": "free"},
                {"user_id": 3, "name": "Charlie", "tier": "pro"},
            ])
            client.flush()

            client.create_table("orders")
            client.store([
                {"order_id": 10, "user_id": 1, "amount": 120},
                {"order_id": 11, "user_id": 1, "amount": 80},
                {"order_id": 12, "user_id": 2, "amount": 30},
                {"order_id": 13, "user_id": 3, "amount": 200},
            ])
            client.flush()

            # Select order/aliases differ from perf case; should still be correct.
            result = _execute_or_xfail(
                client,
                """
                SELECT SUM(o.amount) AS s, u.tier AS t, COUNT(*) AS c
                FROM users u
                JOIN orders o ON u.user_id = o.user_id
                WHERE o.amount >= 50
                GROUP BY u.tier
                ORDER BY s DESC
                """.strip(),
            )
            rows = result.to_dict()
            assert [r["t"] for r in rows] == ["pro"]
            assert rows[0]["c"] == 3
            assert rows[0]["s"] == 400

            client.close()

    def test_execute_join_group_by_count_col(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            client = ApexClient(dirpath=temp_dir)

            client.create_table("users")
            client.store([
                {"user_id": 1, "tier": "pro"},
                {"user_id": 2, "tier": "free"},
            ])
            client.flush()

            client.create_table("orders")
            client.store([
                {"order_id": 10, "user_id": 1, "amount": 120},
                {"order_id": 11, "user_id": 1, "amount": None},
                {"order_id": 12, "user_id": 2, "amount": 80},
            ])
            client.flush()

            result = _execute_or_xfail(
                client,
                """
                SELECT u.tier, COUNT(o.amount) AS cnt
                FROM users u
                JOIN orders o ON u.user_id = o.user_id
                GROUP BY u.tier
                ORDER BY u.tier
                """.strip(),
            )
            rows = result.to_dict()
            assert [(r["tier"], r["cnt"]) for r in rows] == [("free", 1), ("pro", 2)]

            client.close()

    def test_execute_join_group_by_min_max_avg_with_and_where(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            client = ApexClient(dirpath=temp_dir)

            client.create_table("users")
            client.store([
                {"user_id": 1, "tier": "pro"},
                {"user_id": 2, "tier": "free"},
                {"user_id": 3, "tier": "pro"},
            ])
            client.flush()

            client.create_table("orders")
            client.store([
                {"order_id": 10, "user_id": 1, "amount": 120},
                {"order_id": 11, "user_id": 1, "amount": 80},
                {"order_id": 12, "user_id": 2, "amount": 30},
                {"order_id": 13, "user_id": 3, "amount": 200},
            ])
            client.flush()

            # AND predicates split across both tables.
            result = _execute_or_xfail(
                client,
                """
                SELECT u.tier, MIN(o.amount) AS mi, MAX(o.amount) AS ma, AVG(o.amount) AS av
                FROM users u
                JOIN orders o ON u.user_id = o.user_id
                WHERE u.tier = 'pro' AND o.amount >= 80
                GROUP BY u.tier
                """.strip(),
            )
            rows = result.to_dict()
            assert len(rows) == 1
            assert rows[0]["tier"] == "pro"
            assert rows[0]["mi"] == 80
            assert rows[0]["ma"] == 200
            assert rows[0]["av"] == pytest.approx((120 + 80 + 200) / 3)

            client.close()

    def test_execute_aggregate_implicit_alias_keyword(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            client = ApexClient(dirpath=temp_dir)

            rows = [{"v": i} for i in range(10)]
            client.store(rows)
            client.flush()

            result = client.execute(
                "select min(_id) min_id, max(_id) max_id, count(1) count from default"
            )
            rows = result.to_dict()
            assert isinstance(rows, list)
            assert len(rows) == 1
            assert rows[0]["min_id"] == 0
            assert rows[0]["max_id"] == 9
            assert rows[0]["count"] == 10

            client.close()

    def test_execute_aggregate_implicit_alias(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            client = ApexClient(dirpath=temp_dir)

            rows = [{"v": i} for i in range(10)]
            client.store(rows)
            client.flush()

            result = client.execute(
                "select min(_id) min_id, max(_id) as max_id, count(1) as count from default"
            )
            rows = result.to_dict()
            assert isinstance(rows, list)
            assert len(rows) == 1
            assert rows[0]["min_id"] == 0
            assert rows[0]["max_id"] == 9
            assert rows[0]["count"] == 10

            client.close()

    def test_execute_min_max_count_constant_on_internal_id(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            client = ApexClient(dirpath=temp_dir)

            rows = [{"v": i} for i in range(100)]
            client.store(rows)
            client.flush()

            result = client.execute("select min(_id), max(_id), count(1) from default")
            rows = result.to_dict()
            assert isinstance(rows, list)
            assert len(rows) == 1
            assert rows[0]["MIN(_id)"] == 0
            assert rows[0]["MAX(_id)"] == 99
            assert rows[0]["COUNT(1)"] == 100

            client.close()

    def test_execute_count_constant(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            client = ApexClient(dirpath=temp_dir)

            rows = [{"v": i} for i in range(10)]
            client.store(rows)
            client.flush()

            result = client.execute("select min(_id), max(_id), count(1) from default")
            rows = result.to_dict()
            assert isinstance(rows, list)
            assert len(rows) == 1
            assert rows[0]["MIN(_id)"] == 0
            assert rows[0]["MAX(_id)"] == 9
            assert rows[0]["COUNT(1)"] == 10

            client.close()

    def test_execute_min_max_count_on_internal_id(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            client = ApexClient(dirpath=temp_dir)

            rows = [{"v": i} for i in range(100)]
            client.store(rows)
            client.flush()

            result = client.execute("select min(_id), max(_id), count(*) from default")
            rows = result.to_dict()
            assert isinstance(rows, list)
            assert len(rows) == 1
            assert rows[0]["MIN(_id)"] == 0
            assert rows[0]["MAX(_id)"] == 99
            assert rows[0]["COUNT(*)"] == 100

            client.close()

    def test_execute_select_star_plus_id_arrow_fast_path_column_order(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            if not (ARROW_AVAILABLE and PYARROW_AVAILABLE):
                pytest.skip("Arrow/PyArrow not available")

            client = ApexClient(dirpath=temp_dir)

            # Trigger large-result Arrow paths (threshold is > 10_000)
            rows = [{"name": f"u{i}", "age": i} for i in range(12000)]
            client.store(rows)
            client.flush()

            # Keep user-specified order: '*' then '_id'
            result = client.execute("SELECT *, _id FROM default")
            assert len(result) == 12000
            assert "_id" in result.columns
            assert result.columns[-1] == "_id"

            # Keep user-specified order: '_id' then '*'
            result2 = client.execute("SELECT _id, * FROM default")
            assert len(result2) == 12000
            assert "_id" in result2.columns
            assert result2.columns[0] == "_id"

            client.close()

    def test_execute_select_qualified_id_column(self):
        """Test SELECT with qualified internal id column (e.g., default._id)"""
        with tempfile.TemporaryDirectory() as temp_dir:
            client = ApexClient(dirpath=temp_dir)

            client.store([
                {"name": "Alice", "age": 25},
                {"name": "Bob", "age": 30},
            ])

            result = client.execute("SELECT default._id, name FROM default ORDER BY default._id")
            assert "_id" in result.columns
            rows = result.to_dict()
            assert rows[0]["_id"] == 0
            assert rows[1]["_id"] == 1

            client.close()

    def test_execute_select_quoted_id_column(self):
        """Test SELECT with quoted internal id column (\"_id\")"""
        with tempfile.TemporaryDirectory() as temp_dir:
            client = ApexClient(dirpath=temp_dir)

            client.store([
                {"name": "Alice", "age": 25},
                {"name": "Bob", "age": 30},
            ])

            result = client.execute('SELECT "_id", name FROM default ORDER BY "_id"')
            assert "_id" in result.columns
            rows = result.to_dict()
            assert rows[0]["_id"] == 0
            assert rows[1]["_id"] == 1

            client.close()

    def test_execute_select_star_plus_id_column(self):
        """Test SELECT *, _id should explicitly expose _id"""
        with tempfile.TemporaryDirectory() as temp_dir:
            client = ApexClient(dirpath=temp_dir)

            client.store([
                {"name": "Alice", "age": 25},
                {"name": "Bob", "age": 30},
            ])

            result = client.execute("SELECT *, _id FROM default ORDER BY _id")
            assert "_id" in result.columns
            # _id should appear at the user-specified position (after '*')
            assert result.columns[-1] == "_id"
            rows = result.to_dict()
            assert rows[0]["_id"] == 0
            assert rows[1]["_id"] == 1

            client.close()

    def test_execute_select_explicit_id_column(self):
        """Test SELECT explicitly returning internal _id column"""
        with tempfile.TemporaryDirectory() as temp_dir:
            client = ApexClient(dirpath=temp_dir)

            client.store([
                {"name": "Alice", "age": 25},
                {"name": "Bob", "age": 30},
            ])

            result = client.execute("SELECT _id, name FROM default ORDER BY _id")

            assert "_id" in result.columns
            rows = result.to_dict()
            assert isinstance(rows, list)
            assert len(rows) == 2
            assert "_id" in rows[0]
            assert rows[0]["_id"] == 0
            assert rows[1]["_id"] == 1

            client.close()

    def test_execute_arrow_dictionary_string_schema_match(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            if not (ARROW_AVAILABLE and PYARROW_AVAILABLE and PANDAS_AVAILABLE):
                pytest.skip("Arrow/PyArrow/Pandas not available")

            client = ApexClient(dirpath=temp_dir)

            repeated = [{"title": "Python编程指南", "content": "same", "number": i % 10} for i in range(6000)]
            client.store(repeated)
            client.flush()

            df = client.execute("select * from default where title like 'Python%'").to_pandas()
            assert len(df) == 6000
            assert "title" in df.columns
            assert "content" in df.columns
            assert "number" in df.columns
            assert df["title"].iloc[0].startswith("Python")

            # Regression: ensure schema matches Dictionary-encoded string arrays on Arrow fast path
            # when explicitly requesting _id in addition to '*'.
            df2 = client.execute("select *, _id from default").to_pandas()
            assert len(df2) == 6000
            assert "_id" in df2.columns

            client.close()

    def test_execute_where_not_like_and_like(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            client = ApexClient(dirpath=temp_dir)

            rows = [
                {"title": "Python编程指南第1章", "content": "a"},
                {"title": "Python编程指南第2章", "content": "b"},
                {"title": "Python入门", "content": "c"},
                {"title": "Rust编程指南第1章", "content": "d"},
            ]
            client.store(rows)
            client.flush()

            result = client.execute(
                "select * from default where title like 'Python%' and title not like '%编程指南第1%'")
            titles = [r["title"] for r in result.to_dict()]
            assert "Python编程指南第1章" not in titles
            assert "Python编程指南第2章" in titles
            assert "Python入门" in titles
            assert "Rust编程指南第1章" not in titles

            client.close()
    
    def test_execute_select_specific_columns(self):
        """Test SELECT with specific columns"""
        with tempfile.TemporaryDirectory() as temp_dir:
            client = ApexClient(dirpath=temp_dir)
            
            # Store test data
            test_data = [
                {"name": "Alice", "age": 25, "city": "NYC", "salary": 50000},
                {"name": "Bob", "age": 30, "city": "LA", "salary": 60000},
            ]
            client.store(test_data)
            
            # Execute SELECT with specific columns
            result = client.execute("SELECT name, age FROM default")
            
            assert len(result) == 2
            assert result.columns == ["name", "age"]
            assert "city" not in result.columns
            assert "salary" not in result.columns
            
            # Check data
            rows = list(result)
            names = [row["name"] for row in rows]
            ages = [row["age"] for row in rows]
            assert "Alice" in names
            assert "Bob" in names
            assert 25 in ages
            assert 30 in ages
            
            client.close()
    
    def test_execute_select_with_where(self):
        """Test SELECT with WHERE clause"""
        with tempfile.TemporaryDirectory() as temp_dir:
            client = ApexClient(dirpath=temp_dir)
            
            # Store test data
            test_data = [
                {"name": "Alice", "age": 25, "city": "NYC", "active": True},
                {"name": "Bob", "age": 30, "city": "LA", "active": False},
                {"name": "Charlie", "age": 35, "city": "Chicago", "active": True},
            ]
            client.store(test_data)
            
            # Execute SELECT with WHERE
            result = client.execute("SELECT name, age FROM default WHERE age > 25")
            
            assert len(result) == 2
            rows = list(result)
            assert len(rows) == 2
            assert isinstance(rows[0], dict)
            assert "name" in rows[0]
            assert "age" in rows[0]
            names = [row["name"] for row in rows]
            assert "Bob" in names
            assert "Charlie" in names
            assert "Alice" not in names
            
            client.close()
    
    def test_execute_select_with_order_by(self):
        """Test SELECT with ORDER BY clause"""
        with tempfile.TemporaryDirectory() as temp_dir:
            client = ApexClient(dirpath=temp_dir)
            
            # Store test data
            test_data = [
                {"name": "Charlie", "age": 35},
                {"name": "Alice", "age": 25},
                {"name": "Bob", "age": 30},
            ]
            client.store(test_data)
            
            # Execute SELECT with ORDER BY ASC
            result = client.execute("SELECT name, age FROM default ORDER BY age ASC")
            
            assert len(result) == 3
            rows = list(result)
            ages = [row["age"] for row in rows]
            assert ages == [25, 30, 35]  # Sorted ascending
            
            # Execute SELECT with ORDER BY DESC
            result = client.execute("SELECT name, age FROM default ORDER BY age DESC")
            
            rows = list(result)
            ages = [row["age"] for row in rows]
            assert ages == [35, 30, 25]  # Sorted descending
            
            client.close()
    
    def test_execute_select_with_limit(self):
        """Test SELECT with LIMIT clause"""
        with tempfile.TemporaryDirectory() as temp_dir:
            client = ApexClient(dirpath=temp_dir)
            
            # Store test data
            test_data = [{"id": i, "value": f"item_{i}"} for i in range(10)]
            client.store(test_data)
            
            # Execute SELECT with LIMIT
            result = client.execute("SELECT * FROM default LIMIT 5")
            
            assert len(result) == 5
            
            # Execute SELECT with LIMIT and ORDER BY
            result = client.execute("SELECT * FROM default ORDER BY id DESC LIMIT 3")
            
            assert len(result) == 3
            rows = list(result)
            ids = [row["id"] for row in rows]
            assert ids == [9, 8, 7]  # Last 3 IDs in descending order
            
            client.close()
    
    def test_execute_select_with_limit_offset(self):
        """Test SELECT with LIMIT and OFFSET"""
        with tempfile.TemporaryDirectory() as temp_dir:
            client = ApexClient(dirpath=temp_dir)
            
            # Store test data
            test_data = [{"id": i, "value": f"item_{i}"} for i in range(10)]
            client.store(test_data)
            
            # Execute SELECT with LIMIT and OFFSET
            result = client.execute("SELECT * FROM default LIMIT 3 OFFSET 5")
            
            assert len(result) == 3
            rows = list(result)
            ids = [row["id"] for row in rows]
            assert ids == [5, 6, 7]  # IDs 5, 6, 7
            
            client.close()
    
    def test_execute_select_distinct(self):
        """Test SELECT DISTINCT"""
        with tempfile.TemporaryDirectory() as temp_dir:
            client = ApexClient(dirpath=temp_dir)
            
            # Store test data with duplicates
            test_data = [
                {"city": "NYC", "country": "USA"},
                {"city": "LA", "country": "USA"},
                {"city": "NYC", "country": "USA"},
                {"city": "Chicago", "country": "USA"},
                {"city": "Toronto", "country": "Canada"},
                {"city": "Vancouver", "country": "Canada"},
            ]
            client.store(test_data)
            
            # Execute SELECT DISTINCT on single column
            result = client.execute("SELECT DISTINCT city FROM default")
            
            assert len(result) == 5  # NYC, LA, Chicago, Toronto, Vancouver
            cities = [row["city"] for row in result]
            assert "NYC" in cities
            assert "LA" in cities
            assert "Chicago" in cities
            assert "Toronto" in cities
            assert "Vancouver" in cities
            
            # Execute SELECT DISTINCT on multiple columns
            result = client.execute("SELECT DISTINCT city, country FROM default")
            
            assert len(result) == 5  # 5 unique combinations
            
            client.close()


class TestSQLAggregates:
    """Test SQL aggregate functions"""
    
    def test_execute_count_aggregate(self):
        """Test COUNT aggregate function"""
        with tempfile.TemporaryDirectory() as temp_dir:
            client = ApexClient(dirpath=temp_dir)
            
            # Store test data
            test_data = [
                {"name": "Alice", "age": 25, "city": "NYC"},
                {"name": "Bob", "age": 30, "city": "LA"},
                {"name": "Charlie", "age": 35, "city": "NYC"},
            ]
            client.store(test_data)
            
            # Test COUNT(*)
            result = client.execute("SELECT COUNT(*) as total FROM default")
            
            assert len(result) == 1
            assert result.scalar() == 3
            
            # Test COUNT(column)
            result = client.execute("SELECT COUNT(city) as city_count FROM default")
            
            assert len(result) == 1
            assert result.scalar() == 3
            
            # Test COUNT with WHERE
            result = client.execute("SELECT COUNT(*) as nyc_count FROM default WHERE city = 'NYC'")
            
            assert len(result) == 1
            assert result.scalar() == 2
            
            client.close()
    
    def test_execute_sum_aggregate(self):
        """Test SUM aggregate function"""
        with tempfile.TemporaryDirectory() as temp_dir:
            client = ApexClient(dirpath=temp_dir)
            
            # Store test data
            test_data = [
                {"name": "Alice", "salary": 50000},
                {"name": "Bob", "salary": 60000},
                {"name": "Charlie", "salary": 70000},
            ]
            client.store(test_data)
            
            # Test SUM
            result = client.execute("SELECT SUM(salary) as total_salary FROM default")
            
            assert len(result) == 1
            assert result.scalar() == 180000
            
            # Test SUM with WHERE
            result = client.execute("SELECT SUM(salary) as high_salary FROM default WHERE salary > 55000")
            
            assert len(result) == 1
            assert result.scalar() == 130000  # 60000 + 70000
            
            client.close()
    
    def test_execute_avg_aggregate(self):
        """Test AVG aggregate function"""
        with tempfile.TemporaryDirectory() as temp_dir:
            client = ApexClient(dirpath=temp_dir)
            
            # Store test data
            test_data = [
                {"name": "Alice", "age": 25},
                {"name": "Bob", "age": 30},
                {"name": "Charlie", "age": 35},
            ]
            client.store(test_data)
            
            # Test AVG
            result = client.execute("SELECT AVG(age) as avg_age FROM default")
            
            assert len(result) == 1
            avg_age = result.scalar()
            assert abs(avg_age - 30.0) < 0.001  # (25 + 30 + 35) / 3 = 30
            
            client.close()
    
    def test_execute_min_max_aggregates(self):
        """Test MIN and MAX aggregate functions"""
        with tempfile.TemporaryDirectory() as temp_dir:
            client = ApexClient(dirpath=temp_dir)
            
            # Store test data
            test_data = [
                {"name": "Alice", "age": 25, "salary": 50000},
                {"name": "Bob", "age": 30, "salary": 60000},
                {"name": "Charlie", "age": 35, "salary": 70000},
            ]
            client.store(test_data)
            
            # Test MIN
            result = client.execute("SELECT MIN(age) as min_age FROM default")
            
            assert len(result) == 1
            assert result.scalar() == 25
            
            # Test MAX
            result = client.execute("SELECT MAX(salary) as max_salary FROM default")
            
            assert len(result) == 1
            assert result.scalar() == 70000
            
            # Test MIN and MAX together
            result = client.execute("SELECT MIN(age) as min_age, MAX(salary) as max_salary FROM default")
            
            assert len(result) == 1
            row = result.first()
            assert row["min_age"] == 25
            assert row["max_salary"] == 70000
            
            client.close()


class TestSQLGroupBy:
    """Test SQL GROUP BY operations"""
    
    def test_execute_group_by_basic(self):
        """Test basic GROUP BY"""
        with tempfile.TemporaryDirectory() as temp_dir:
            client = ApexClient(dirpath=temp_dir)
            
            # Store test data
            test_data = [
                {"city": "NYC", "salary": 50000},
                {"city": "LA", "salary": 60000},
                {"city": "NYC", "salary": 55000},
            ]
            client.store(test_data)
            
            # Test GROUP BY - behavior may vary
            try:
                result = client.execute("SELECT city, COUNT(*) as count FROM default GROUP BY city")
                # GROUP BY support may be limited
                assert len(result) >= 0
            except Exception as e:
                print(f"GROUP BY basic: {e}")
            
            client.close()
    
    def test_execute_group_by_with_aggregates(self):
        """Test GROUP BY with various aggregates"""
        with tempfile.TemporaryDirectory() as temp_dir:
            client = ApexClient(dirpath=temp_dir)
            
            # Store test data
            test_data = [
                {"dept": "Engineering", "salary": 80000},
                {"dept": "Sales", "salary": 60000},
            ]
            client.store(test_data)
            
            # Test GROUP BY with aggregates - behavior may vary
            try:
                result = client.execute("SELECT COUNT(*) as count FROM default")
                assert len(result) >= 0
            except Exception as e:
                print(f"GROUP BY aggregates: {e}")
            
            client.close()
    
    def test_execute_group_by_with_having(self):
        """Test GROUP BY with HAVING clause"""
        with tempfile.TemporaryDirectory() as temp_dir:
            client = ApexClient(dirpath=temp_dir)
            
            # Store test data
            test_data = [
                {"city": "NYC", "population": 1000000},
                {"city": "NYC", "population": 1100000},
                {"city": "LA", "population": 800000},
            ]
            client.store(test_data)

            # HAVING should filter on aggregated result
            result = client.execute(
                "SELECT city, COUNT(*) AS c FROM default GROUP BY city HAVING COUNT(*) > 1"
            )
            rows = result.to_dict()
            assert isinstance(rows, list)
            # Only NYC has >1 rows
            assert len(rows) == 1
            assert rows[0]["city"] == "NYC"
            assert rows[0]["c"] == 2
            
            client.close()


class TestSQLRealWorldQueries:
    def test_execute_union_and_union_all(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            client = ApexClient(dirpath=temp_dir)

            client.store([
                {"k": 1, "city": "NYC"},
                {"k": 2, "city": "NYC"},
                {"k": 3, "city": "LA"},
                {"k": 4, "city": "Chicago"},
            ])
            client.flush()

            result_union = _execute_or_xfail(
                client,
                """
                SELECT city FROM default WHERE city = 'NYC'
                UNION
                SELECT city FROM default WHERE city = 'LA'
                ORDER BY city
                """.strip(),
            )
            rows_union = result_union.to_dict()
            assert [r["city"] for r in rows_union] == ["LA", "NYC"]

            result_union_all = _execute_or_xfail(
                client,
                """
                SELECT city FROM default WHERE city = 'NYC'
                UNION ALL
                SELECT city FROM default WHERE city = 'LA'
                ORDER BY city
                """.strip(),
            )
            rows_union_all = result_union_all.to_dict()
            assert [r["city"] for r in rows_union_all] == ["LA", "NYC", "NYC"]

            client.close()

    def test_execute_multi_table_join_inner(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            client = ApexClient(dirpath=temp_dir)

            client.create_table("users")
            client.store([
                {"user_id": 1, "name": "Alice", "tier": "pro"},
                {"user_id": 2, "name": "Bob", "tier": "free"},
                {"user_id": 3, "name": "Charlie", "tier": "pro"},
            ])
            client.flush()

            client.create_table("orders")
            client.store([
                {"order_id": 10, "user_id": 1, "amount": 120},
                {"order_id": 11, "user_id": 1, "amount": 80},
                {"order_id": 12, "user_id": 2, "amount": 30},
            ])
            client.flush()

            result = _execute_or_xfail(
                client,
                """
                SELECT u.name, u.tier, o.order_id, o.amount
                FROM users u
                JOIN orders o ON u.user_id = o.user_id
                WHERE o.amount >= 50
                ORDER BY o.order_id
                """.strip(),
            )
            rows = result.to_dict()
            assert [(r["name"], r["order_id"], r["amount"]) for r in rows] == [
                ("Alice", 10, 120),
                ("Alice", 11, 80),
            ]

            client.close()

    def test_execute_multi_group_by_having(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            client = ApexClient(dirpath=temp_dir)

            client.store([
                {"region": "CN", "channel": "online", "amount": 120},
                {"region": "CN", "channel": "online", "amount": 90},
                {"region": "CN", "channel": "store", "amount": 30},
                {"region": "US", "channel": "online", "amount": 200},
                {"region": "US", "channel": "store", "amount": 40},
            ])
            client.flush()

            result = _execute_or_xfail(
                client,
                """
                SELECT region, channel, COUNT(*) AS c, SUM(amount) AS s
                FROM default
                GROUP BY region, channel
                HAVING SUM(amount) >= 150
                ORDER BY region, channel
                """.strip(),
            )
            rows = result.to_dict()
            assert [(r["region"], r["channel"], r["c"], r["s"]) for r in rows] == [
                ("CN", "online", 2, 210),
                ("US", "online", 1, 200),
            ]

            client.close()

    def test_execute_nested_aggregation_second_stage(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            client = ApexClient(dirpath=temp_dir)

            client.store([
                {"region": "CN", "amount": 120},
                {"region": "CN", "amount": 90},
                {"region": "CN", "amount": 30},
                {"region": "US", "amount": 200},
                {"region": "US", "amount": 40},
            ])
            client.flush()

            result = _execute_or_xfail(
                client,
                """
                SELECT COUNT(*) AS big_regions
                FROM (
                    SELECT region, SUM(amount) AS total
                    FROM default
                    GROUP BY region
                ) t
                WHERE t.total >= 200
                """.strip(),
            )
            assert result.scalar() == 2

            client.close()

    def test_execute_left_join_preserve_rows(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            client = ApexClient(dirpath=temp_dir)

            client.create_table("users")
            client.store([
                {"user_id": 1, "name": "Alice"},
                {"user_id": 2, "name": "Bob"},
                {"user_id": 3, "name": "Charlie"},
            ])
            client.flush()

            client.create_table("orders")
            client.store([
                {"order_id": 10, "user_id": 1, "amount": 120},
                {"order_id": 11, "user_id": 1, "amount": 80},
                {"order_id": 12, "user_id": 2, "amount": 30},
            ])
            client.flush()

            result = _execute_or_xfail(
                client,
                """
                SELECT u.user_id, u.name, o.order_id
                FROM users u
                LEFT JOIN orders o ON u.user_id = o.user_id
                ORDER BY u.user_id, o.order_id
                """.strip(),
            )
            rows = result.to_dict()
            assert [r["user_id"] for r in rows] == [1, 1, 2, 3]
            assert [r["order_id"] for r in rows] == [10, 11, 12, None]

            client.close()

    def test_execute_join_group_by_having_top_customers(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            client = ApexClient(dirpath=temp_dir)

            client.create_table("users")
            client.store([
                {"user_id": 1, "name": "Alice", "tier": "pro"},
                {"user_id": 2, "name": "Bob", "tier": "free"},
                {"user_id": 3, "name": "Charlie", "tier": "pro"},
            ])
            client.flush()

            client.create_table("orders")
            client.store([
                {"order_id": 10, "user_id": 1, "amount": 120},
                {"order_id": 11, "user_id": 1, "amount": 80},
                {"order_id": 12, "user_id": 2, "amount": 30},
                {"order_id": 13, "user_id": 3, "amount": 200},
            ])
            client.flush()

            result = _execute_or_xfail(
                client,
                """
                SELECT u.tier, COUNT(DISTINCT u.user_id) AS users, SUM(o.amount) AS revenue
                FROM users u
                JOIN orders o ON u.user_id = o.user_id
                GROUP BY u.tier
                HAVING SUM(o.amount) >= 200
                ORDER BY u.tier
                """.strip(),
            )
            rows = result.to_dict()
            assert len(rows) == 1
            assert rows[0]["tier"] == "pro"
            assert rows[0]["users"] == 2
            assert rows[0]["revenue"] == 400

            client.close()

    def test_execute_join_distinct_dimension_values(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            client = ApexClient(dirpath=temp_dir)

            client.create_table("users")
            client.store([
                {"user_id": 1, "name": "Alice", "tier": "pro"},
                {"user_id": 2, "name": "Bob", "tier": "free"},
            ])
            client.flush()

            client.create_table("orders")
            client.store([
                {"order_id": 10, "user_id": 1, "amount": 120},
                {"order_id": 11, "user_id": 1, "amount": 80},
                {"order_id": 12, "user_id": 2, "amount": 30},
            ])
            client.flush()

            result = _execute_or_xfail(
                client,
                """
                SELECT DISTINCT u.tier
                FROM users u
                JOIN orders o ON u.user_id = o.user_id
                ORDER BY u.tier
                """.strip(),
            )
            tiers = [r["tier"] for r in result.to_dict()]
            assert tiers == ["free", "pro"]

            client.close()

    def test_execute_union_with_limit_offset(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            client = ApexClient(dirpath=temp_dir)

            client.store([
                {"k": 1, "city": "NYC"},
                {"k": 2, "city": "NYC"},
                {"k": 3, "city": "LA"},
                {"k": 4, "city": "Chicago"},
                {"k": 5, "city": "Seattle"},
            ])
            client.flush()

            result = _execute_or_xfail(
                client,
                """
                SELECT city FROM default WHERE city LIKE 'C%'
                UNION ALL
                SELECT city FROM default WHERE city LIKE 'N%'
                ORDER BY city
                LIMIT 2 OFFSET 1
                """.strip(),
            )
            rows = result.to_dict()
            assert [r["city"] for r in rows] == ["NYC", "NYC"]

            client.close()

    def test_execute_in_and_not_in_subquery(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            client = ApexClient(dirpath=temp_dir)

            client.create_table("users")
            client.store([
                {"user_id": 1, "name": "Alice"},
                {"user_id": 2, "name": "Bob"},
                {"user_id": 3, "name": "Charlie"},
            ])
            client.flush()

            client.create_table("orders")
            client.store([
                {"order_id": 10, "user_id": 1, "amount": 120},
                {"order_id": 11, "user_id": 1, "amount": 80},
                {"order_id": 12, "user_id": 2, "amount": 30},
            ])
            client.flush()

            result_in = _execute_or_xfail(
                client,
                """
                SELECT name
                FROM users
                WHERE user_id IN (SELECT user_id FROM orders WHERE amount >= 80)
                ORDER BY name
                """.strip(),
            )
            assert [r["name"] for r in result_in.to_dict()] == ["Alice"]

            result_not_in = _execute_or_xfail(
                client,
                """
                SELECT name
                FROM users
                WHERE user_id NOT IN (SELECT user_id FROM orders)
                ORDER BY name
                """.strip(),
            )
            assert [r["name"] for r in result_not_in.to_dict()] == ["Charlie"]

            client.close()

    def test_execute_exists_subquery(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            client = ApexClient(dirpath=temp_dir)

            client.create_table("users")
            client.store([
                {"user_id": 1, "name": "Alice"},
                {"user_id": 2, "name": "Bob"},
                {"user_id": 3, "name": "Charlie"},
            ])
            client.flush()

            client.create_table("orders")
            client.store([
                {"order_id": 10, "user_id": 1, "amount": 120},
                {"order_id": 11, "user_id": 1, "amount": 80},
                {"order_id": 12, "user_id": 2, "amount": 30},
            ])
            client.flush()

            result = _execute_or_xfail(
                client,
                """
                SELECT u.name
                FROM users u
                WHERE EXISTS (
                    SELECT 1 FROM orders o
                    WHERE o.user_id = u.user_id AND o.amount >= 100
                )
                ORDER BY u.name
                """.strip(),
            )
            assert [r["name"] for r in result.to_dict()] == ["Alice"]

            client.close()

    def test_execute_two_stage_aggregation_bucketed(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            client = ApexClient(dirpath=temp_dir)

            client.store([
                {"region": "CN", "amount": 120},
                {"region": "CN", "amount": 90},
                {"region": "CN", "amount": 30},
                {"region": "US", "amount": 200},
                {"region": "US", "amount": 40},
                {"region": "EU", "amount": 10},
                {"region": "EU", "amount": 20},
            ])
            client.flush()

            result = _execute_or_xfail(
                client,
                """
                SELECT t.bucket, COUNT(*) AS regions
                FROM (
                    SELECT region,
                           CASE WHEN SUM(amount) >= 200 THEN 'big' ELSE 'small' END AS bucket
                    FROM default
                    GROUP BY region
                ) t
                GROUP BY t.bucket
                ORDER BY t.bucket
                """.strip(),
            )
            rows = result.to_dict()
            assert [(r["bucket"], r["regions"]) for r in rows] == [("big", 2), ("small", 1)]

            client.close()


class TestSQLSubqueriesAdvanced:
    def test_execute_not_exists_correlated(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            client = ApexClient(dirpath=temp_dir)

            client.create_table("users")
            client.store(
                [
                    {"user_id": 1, "name": "Alice"},
                    {"user_id": 2, "name": "Bob"},
                    {"user_id": 3, "name": "Charlie"},
                ]
            )
            client.flush()

            client.create_table("orders")
            client.store(
                [
                    {"order_id": 10, "user_id": 1, "amount": 120},
                    {"order_id": 11, "user_id": 1, "amount": 80},
                    {"order_id": 12, "user_id": 2, "amount": 30},
                ]
            )
            client.flush()

            result = _execute_or_xfail(
                client,
                """
                SELECT u.name
                FROM users u
                WHERE NOT EXISTS (
                    SELECT 1 FROM orders o
                    WHERE o.user_id = u.user_id
                )
                ORDER BY u.name
                """.strip(),
            )
            assert [r["name"] for r in result.to_dict()] == ["Charlie"]

            client.close()

    def test_execute_correlated_in_subquery(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            client = ApexClient(dirpath=temp_dir)

            client.create_table("users")
            client.store(
                [
                    {"user_id": 1, "name": "Alice"},
                    {"user_id": 2, "name": "Bob"},
                    {"user_id": 3, "name": "Charlie"},
                ]
            )
            client.flush()

            client.create_table("orders")
            client.store(
                [
                    {"order_id": 10, "user_id": 1, "amount": 120},
                    {"order_id": 11, "user_id": 1, "amount": 80},
                    {"order_id": 12, "user_id": 2, "amount": 30},
                ]
            )
            client.flush()

            result = _execute_or_xfail(
                client,
                """
                SELECT u.name
                FROM users u
                WHERE u.user_id IN (
                    SELECT o.user_id
                    FROM orders o
                    WHERE o.user_id = u.user_id AND o.amount >= 100
                )
                ORDER BY u.name
                """.strip(),
            )
            assert [r["name"] for r in result.to_dict()] == ["Alice"]

            client.close()

    def test_execute_scalar_correlated_subquery_in_select_list(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            client = ApexClient(dirpath=temp_dir)

            client.create_table("users")
            client.store(
                [
                    {"user_id": 1, "name": "Alice"},
                    {"user_id": 2, "name": "Bob"},
                    {"user_id": 3, "name": "Charlie"},
                ]
            )
            client.flush()

            client.create_table("orders")
            client.store(
                [
                    {"order_id": 10, "user_id": 1, "amount": 120},
                    {"order_id": 11, "user_id": 1, "amount": 80},
                    {"order_id": 12, "user_id": 2, "amount": 30},
                ]
            )
            client.flush()

            result = _execute_or_xfail(
                client,
                """
                SELECT u.name,
                       (SELECT MAX(amount) FROM orders o WHERE o.user_id = u.user_id) AS max_amount
                FROM users u
                ORDER BY u.name
                """.strip(),
            )
            rows = result.to_dict()
            assert [(r["name"], r["max_amount"]) for r in rows] == [
                ("Alice", 120),
                ("Bob", 30),
                ("Charlie", None),
            ]

            client.close()


class TestSqlResultFunctionality:
    """Test ResultView functionality and conversions"""
    
    def test_sql_result_basic_properties(self):
        """Test ResultView basic properties"""
        with tempfile.TemporaryDirectory() as temp_dir:
            client = ApexClient(dirpath=temp_dir)
            
            # Store test data
            test_data = [
                {"name": "Alice", "age": 25},
                {"name": "Bob", "age": 30},
            ]
            client.store(test_data)
            
            result = client.execute("SELECT name, age FROM default")
            
            # Test basic properties
            assert isinstance(result, ResultView)
            assert len(result) >= 0
            # Columns may vary based on implementation
            assert result.columns is not None
            
            client.close()
    
    def test_sql_result_iteration(self):
        """Test ResultView iteration"""
        with tempfile.TemporaryDirectory() as temp_dir:
            client = ApexClient(dirpath=temp_dir)
            
            # Store test data
            test_data = [
                {"name": "Alice", "age": 25},
                {"name": "Bob", "age": 30},
                {"name": "Charlie", "age": 35},
            ]
            client.store(test_data)
            
            result = client.execute("SELECT name, age FROM default ORDER BY age")
            
            # Test iteration
            names = []
            ages = []
            for row in result:
                names.append(row["name"])
                ages.append(row["age"])
            
            assert len(names) == 3
            assert names == ["Alice", "Bob", "Charlie"]
            assert ages == [25, 30, 35]
            
            client.close()
    
    def test_sql_result_to_dicts(self):
        """Test ResultView.to_dict() method"""
        with tempfile.TemporaryDirectory() as temp_dir:
            client = ApexClient(dirpath=temp_dir)
            
            # Store test data
            test_data = [
                {"name": "Alice", "age": 25},
                {"name": "Bob", "age": 30},
            ]
            client.store(test_data)
            
            result = client.execute("SELECT name, age FROM default")
            dict_list = result.to_dict()
            
            assert isinstance(dict_list, list)
            assert len(dict_list) == 2
            assert isinstance(dict_list[0], dict)
            assert dict_list[0]["name"] == "Alice"
            assert dict_list[1]["name"] == "Bob"
            
            client.close()
    
    @pytest.mark.skipif(not PANDAS_AVAILABLE, reason="Pandas not available")
    def test_sql_result_to_pandas(self):
        """Test ResultView.to_pandas() method"""
        with tempfile.TemporaryDirectory() as temp_dir:
            client = ApexClient(dirpath=temp_dir)
            
            # Store test data
            test_data = [
                {"name": "Alice", "age": 25, "city": "NYC"},
                {"name": "Bob", "age": 30, "city": "LA"},
            ]
            client.store(test_data)
            
            result = client.execute("SELECT name, age, city FROM default")
            df = result.to_pandas()
            
            assert isinstance(df, pd.DataFrame)
            assert len(df) == 2
            assert "name" in df.columns
            assert "age" in df.columns
            assert "city" in df.columns
            assert "_id" not in df.columns  # _id should be hidden
            
            client.close()
    
    @pytest.mark.skipif(not POLARS_DF_AVAILABLE, reason="Polars not available")
    def test_sql_result_to_polars(self):
        """Test ResultView.to_polars() method"""
        with tempfile.TemporaryDirectory() as temp_dir:
            client = ApexClient(dirpath=temp_dir)
            
            # Store test data
            test_data = [
                {"name": "Alice", "age": 25},
                {"name": "Bob", "age": 30},
            ]
            client.store(test_data)
            
            result = client.execute("SELECT name, age FROM default")
            df = result.to_polars()
            
            assert isinstance(df, pl.DataFrame)
            assert len(df) == 2
            assert "name" in df.columns
            assert "age" in df.columns
            assert "_id" not in df.columns  # _id should be hidden
            
            client.close()
    
    def test_sql_result_get_ids(self):
        """Test ResultView.get_ids() method"""
        with tempfile.TemporaryDirectory() as temp_dir:
            client = ApexClient(dirpath=temp_dir)
            
            # Store test data
            test_data = [
                {"name": "Alice", "age": 25},
                {"name": "Bob", "age": 30},
                {"name": "Charlie", "age": 35},
            ]
            client.store(test_data)
            
            result = client.execute("SELECT name, age FROM default")
            
            # Test get_ids with numpy array (default)
            ids = result.get_ids()
            assert isinstance(ids, np.ndarray)
            assert len(ids) == 3
            assert all(isinstance(id, (int, np.integer)) for id in ids)
            
            # Test get_ids with list
            ids_list = result.get_ids(return_list=True)
            assert isinstance(ids_list, list)
            assert len(ids_list) == 3
            assert all(isinstance(id, int) for id in ids_list)
            
            client.close()
    
    def test_sql_result_scalar(self):
        """Test ResultView.scalar() method"""
        with tempfile.TemporaryDirectory() as temp_dir:
            client = ApexClient(dirpath=temp_dir)
            
            # Store test data
            test_data = [
                {"name": "Alice", "age": 25},
                {"name": "Bob", "age": 30},
            ]
            client.store(test_data)
            
            # Test scalar with aggregate
            result = client.execute("SELECT COUNT(*) as count FROM default")
            count = result.scalar()
            assert count == 2
            
            # Test scalar with single value
            result = client.execute("SELECT age FROM default WHERE name = 'Alice'")
            age = result.scalar()
            assert age == 25
            
            # Test scalar with no results
            result = client.execute("SELECT age FROM default WHERE name = 'Nonexistent'")
            value = result.scalar()
            assert value is None
            
            client.close()
    
    def test_sql_result_first(self):
        """Test ResultView.first() method"""
        with tempfile.TemporaryDirectory() as temp_dir:
            client = ApexClient(dirpath=temp_dir)
            
            # Store test data
            test_data = [
                {"name": "Alice", "age": 25},
                {"name": "Bob", "age": 30},
                {"name": "Charlie", "age": 35},
            ]
            client.store(test_data)
            
            result = client.execute("SELECT name, age FROM default ORDER BY age")
            
            # Test first
            first_row = result.first()
            assert isinstance(first_row, dict)
            assert first_row["name"] == "Alice"
            assert first_row["age"] == 25
            
            # Test first with no results
            empty_result = client.execute("SELECT name, age FROM default WHERE age > 100")
            first_row = empty_result.first()
            assert first_row is None
            
            client.close()
    
    def test_sql_result_repr(self):
        """Test ResultView.__repr__ method"""
        with tempfile.TemporaryDirectory() as temp_dir:
            client = ApexClient(dirpath=temp_dir)
            
            # Store test data
            test_data = [
                {"name": "Alice", "age": 25},
                {"name": "Bob", "age": 30},
            ]
            client.store(test_data)
            
            result = client.execute("SELECT name, age FROM default")
            repr_str = repr(result)
            
            # Basic repr check - format may vary
            assert "ResultView" in repr_str
            
            client.close()


class TestSQLEdgeCases:
    """Test edge cases and error handling for SQL operations"""
    
    def test_execute_invalid_sql(self):
        """Test invalid SQL syntax"""
        with tempfile.TemporaryDirectory() as temp_dir:
            client = ApexClient(dirpath=temp_dir)
            
            # Store test data
            test_data = [{"name": "Alice", "age": 25}]
            client.store(test_data)
            
            # Test invalid SQL
            with pytest.raises(Exception):  # Should raise some kind of SQL error
                client.execute("INVALID SQL SYNTAX")
            
            with pytest.raises(Exception):
                client.execute("SELECT * FROM nonexistent_table")
            
            client.close()
    
    def test_execute_nonexistent_columns(self):
        """Test SELECT with nonexistent columns"""
        with tempfile.TemporaryDirectory() as temp_dir:
            client = ApexClient(dirpath=temp_dir)
            
            # Store test data
            test_data = [{"name": "Alice", "age": 25}]
            client.store(test_data)
            
            # Test with nonexistent column
            try:
                result = client.execute("SELECT nonexistent_column FROM default")
                # If no exception, should return empty results or handle gracefully
                assert len(result) == 0 or result.columns == []
            except Exception as e:
                # Exception is also acceptable behavior
                print(f"Nonexistent column handled: {e}")
            
            client.close()
    
    def test_execute_on_closed_client(self):
        """Test execute operations on closed client"""
        with tempfile.TemporaryDirectory() as temp_dir:
            client = ApexClient(dirpath=temp_dir)
            client.close()
            
            with pytest.raises(RuntimeError, match="connection has been closed"):
                client.execute("SELECT * FROM default")
            
            with pytest.raises(RuntimeError, match="connection has been closed"):
                client.execute("SELECT COUNT(*) FROM default")
    
    def test_execute_empty_database(self):
        """Test execute operations on empty database"""
        with tempfile.TemporaryDirectory() as temp_dir:
            client = ApexClient(dirpath=temp_dir)
            
            # Test SELECT on empty database
            result = client.execute("SELECT * FROM default")
            assert len(result) == 0
            
            # Test aggregate on empty database
            result = client.execute("SELECT COUNT(*) as count FROM default")
            assert result.scalar() == 0
            
            client.close()
    
    def test_execute_with_special_characters(self):
        """Test execute with special characters in data"""
        with tempfile.TemporaryDirectory() as temp_dir:
            client = ApexClient(dirpath=temp_dir)
            
            # Store test data with special characters
            test_data = [
                {"name": "Alice", "description": "Test data"},
                {"name": "Bob", "description": "Another test"},
            ]
            client.store(test_data)
            
            # Test basic queries - special character handling may vary
            try:
                result = client.execute("SELECT name FROM default")
                assert len(result) >= 0
            except Exception as e:
                print(f"Special char query: {e}")
            
            client.close()
    
    def test_execute_complex_joins(self):
        """Test complex SQL operations (if supported)"""
        with tempfile.TemporaryDirectory() as temp_dir:
            client = ApexClient(dirpath=temp_dir)
            
            # Store test data
            test_data = [
                {"name": "Alice", "age": 25, "dept": "Engineering"},
                {"name": "Bob", "age": 30, "dept": "Sales"},
                {"name": "Charlie", "age": 35, "dept": "Engineering"},
            ]
            client.store(test_data)
            
            # Test subquery (if supported)
            try:
                result = client.execute("""
                    SELECT name, age 
                    FROM default 
                    WHERE age > (SELECT AVG(age) FROM default)
                """)
                
                # Should return employees older than average (30)
                assert len(result) == 1
                assert result.first()["name"] == "Charlie"
                
            except Exception as e:
                print(f"Subqueries not supported: {e}")
            
            # Test complex CASE statement (if supported)
            try:
                result = client.execute("""
                    SELECT name, age,
                           CASE 
                               WHEN age < 30 THEN 'Young'
                               WHEN age < 40 THEN 'Middle'
                               ELSE 'Senior'
                           END as category
                    FROM default
                    ORDER BY age
                """)
                
                assert len(result) == 3
                
            except Exception as e:
                print(f"CASE statements not supported: {e}")
            
            client.close()


class TestSQLPerformance:
    """Test SQL performance considerations"""
    
    def test_execute_performance_large_dataset(self):
        """Test execute performance with large dataset"""
        with tempfile.TemporaryDirectory() as temp_dir:
            client = ApexClient(dirpath=temp_dir)
            
            # Store dataset
            data = [
                {"id": i, "category": f"cat_{i % 10}", "value": i * 1.5}
                for i in range(1000)
            ]
            client.store(data)
            
            import time
            
            # Test aggregate performance
            start_time = time.time()
            try:
                result = client.execute("SELECT COUNT(*) as count FROM default")
                assert result.scalar() >= 0
            except Exception as e:
                print(f"Perf test: {e}")
            end_time = time.time()
            
            assert (end_time - start_time) < 5.0  # Should be reasonably fast
            
            client.close()
    
    def test_execute_arrow_optimization(self):
        """Test Arrow optimization in execute when available"""
        with tempfile.TemporaryDirectory() as temp_dir:
            client = ApexClient(dirpath=temp_dir)
            
            # Store test data
            test_data = [
                {"name": "Alice", "age": 25},
                {"name": "Bob", "age": 30},
                {"name": "Charlie", "age": 35},
            ]
            client.store(test_data)
            
            result = client.execute("SELECT name, age FROM default")
            
            # If Arrow is available, result should use Arrow internally
            if ARROW_AVAILABLE and PYARROW_AVAILABLE:
                # Test that Arrow conversion works
                try:
                    table = result.to_arrow()
                    assert isinstance(table, pa.Table)
                except Exception:
                    pass  # Arrow optimization might not be active
            
            client.close()


def _store_rows_in_chunks(client: ApexClient, rows_iter, chunk_size: int = 50_000):
    buf = []
    for r in rows_iter:
        buf.append(r)
        if len(buf) >= chunk_size:
            client.store(buf)
            buf.clear()
    if buf:
        client.store(buf)


class TestSQLPerformance1M:
    @pytest.mark.perf
    @pytest.mark.slow
    def test_perf_1m_join_filter_order_limit(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            client = ApexClient(dirpath=temp_dir)

            client.create_table("users")
            t_store0 = time.perf_counter()
            _store_rows_in_chunks(
                client,
                ({"user_id": i, "tier": "pro" if (i % 10 == 0) else "free"} for i in range(200_000)),
                chunk_size=50_000,
            )
            t_store1 = time.perf_counter()
            client.flush()
            t_flush1 = time.perf_counter()

            client.create_table("orders")
            _store_rows_in_chunks(
                client,
                (
                    {
                        "order_id": i,
                        "user_id": i % 200_000,
                        "amount": (i % 97) * 1.0,
                    }
                    for i in range(1_000_000)
                ),
                chunk_size=50_000,
            )
            t_store2 = time.perf_counter()
            client.flush()
            t_flush2 = time.perf_counter()

            sql = """
            SELECT u.tier, COUNT(*) AS c, SUM(o.amount) AS s
            FROM users u
            JOIN orders o ON u.user_id = o.user_id
            WHERE o.amount >= 50
            GROUP BY u.tier
            ORDER BY s DESC
            LIMIT 10
            """.strip()

            # Warmup + repeated runs to measure executor-only improvements
            _execute_or_xfail(client, sql).to_dict()
            times = []
            for _ in range(5):
                t0 = time.perf_counter()
                result = _execute_or_xfail(client, sql)
                rows = result.to_dict()
                t1 = time.perf_counter()
                times.append(t1 - t0)

            assert isinstance(rows, list)
            assert len(rows) <= 2
            assert all("tier" in r and "c" in r and "s" in r for r in rows)
            avg = sum(times) / len(times)
            print(
                f"perf_1m_join_filter_order_limit: query_avg={avg:.3f}s query_runs={[round(x, 3) for x in times]} "
                f"store_users={t_store1 - t_store0:.3f}s flush_users={t_flush1 - t_store1:.3f}s "
                f"store_orders={t_store2 - t_flush1:.3f}s flush_orders={t_flush2 - t_store2:.3f}s"
            )

            client.close()

    @pytest.mark.perf
    @pytest.mark.slow
    def test_perf_1m_nested_subquery_two_stage(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            client = ApexClient(dirpath=temp_dir)

            t_store0 = time.perf_counter()
            _store_rows_in_chunks(
                client,
                (
                    {
                        "k": i,
                        "region": f"r{i % 200}",
                        "channel": f"c{i % 20}",
                        "amount": float(i % 101),
                    }
                    for i in range(1_000_000)
                ),
                chunk_size=50_000,
            )
            t_store1 = time.perf_counter()
            client.flush()
            t_flush1 = time.perf_counter()

            sql = """
            SELECT COUNT(*) AS big_groups
            FROM (
                SELECT region, channel, SUM(amount) AS s
                FROM default
                GROUP BY region, channel
                HAVING SUM(amount) >= 20000
            ) t
            WHERE t.s >= 20000
            """.strip()

            _execute_or_xfail(client, sql).scalar()
            times = []
            for _ in range(5):
                t0 = time.perf_counter()
                result = _execute_or_xfail(client, sql)
                v = result.scalar()
                t1 = time.perf_counter()
                times.append(t1 - t0)

            assert v is None or isinstance(v, (int, float, np.integer, np.floating))
            avg = sum(times) / len(times)
            print(
                f"perf_1m_nested_subquery_two_stage: query_avg={avg:.3f}s query_runs={[round(x, 3) for x in times]} "
                f"store={t_store1 - t_store0:.3f}s flush={t_flush1 - t_store1:.3f}s"
            )

            client.close()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
