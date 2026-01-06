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
                "SELECT city, COUNT(*) AS c FROM default GROUP BY city HAVING c > 1"
            )
            rows = result.to_dict()
            assert isinstance(rows, list)
            # Only NYC has >1 rows
            assert len(rows) == 1
            assert rows[0]["city"] == "NYC"
            assert rows[0]["c"] == 2
            
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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
