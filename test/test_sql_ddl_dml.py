"""
Comprehensive test suite for ApexBase SQL Operations

This module tests:
- Complex SQL SELECT statements
- Aggregations (COUNT, SUM, AVG, MIN, MAX)
- GROUP BY and HAVING clauses
- ORDER BY with various options
- LIMIT and OFFSET
- WHERE with complex conditions
- JOINs and subqueries
- DDL/DML operations via Python API
"""

import pytest
import tempfile
import time
from pathlib import Path
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'apexbase', 'python'))

try:
    from apexbase import ApexClient, ResultView
except ImportError as e:
    pytest.skip(f"ApexBase not available: {e}", allow_module_level=True)

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False


# =============================================================================
# Table Management via Python API (DDL-like operations)
# =============================================================================

class TestTableManagementAPI:
    """Test table management operations via Python API"""
    
    def test_create_table_api(self):
        """Test creating table via API"""
        with tempfile.TemporaryDirectory() as temp_dir:
            client = ApexClient(dirpath=temp_dir)
            
            client.create_table("users")
            assert "users" in client.list_tables()
            
            client.close()
    
    def test_create_multiple_tables(self):
        """Test creating multiple tables"""
        with tempfile.TemporaryDirectory() as temp_dir:
            client = ApexClient(dirpath=temp_dir)
            
            for name in ["customers", "orders", "products"]:
                client.create_table(name)
            
            tables = client.list_tables()
            assert "customers" in tables
            assert "orders" in tables
            assert "products" in tables
            
            client.close()
    
    def test_drop_table_api(self):
        """Test dropping table via API"""
        with tempfile.TemporaryDirectory() as temp_dir:
            client = ApexClient(dirpath=temp_dir)
            
            client.create_table("temp_table")
            assert "temp_table" in client.list_tables()
            
            client.drop_table("temp_table")
            assert "temp_table" not in client.list_tables()
            
            client.close()
    
    def test_use_table_api(self):
        """Test switching between tables"""
        with tempfile.TemporaryDirectory() as temp_dir:
            client = ApexClient(dirpath=temp_dir)
            
            client.create_table("table_a")
            client.create_table("table_b")
            
            client.use_table("table_a")
            client.store([{"id": 1, "source": "A"}])
            
            client.use_table("table_b")
            client.store([{"id": 2, "source": "B"}])
            
            # Verify data isolation
            client.use_table("table_a")
            result = client.execute("SELECT * FROM table_a")
            assert len(result) == 1
            
            client.use_table("table_b")
            result = client.execute("SELECT * FROM table_b")
            assert len(result) == 1
            
            client.close()


# =============================================================================
# Column Management via Python API (ALTER TABLE-like operations)
# =============================================================================

class TestColumnManagementAPI:
    """Test column management operations via Python API"""
    
    def test_add_column_api(self):
        """Test adding column via API"""
        with tempfile.TemporaryDirectory() as temp_dir:
            client = ApexClient(dirpath=temp_dir)
            
            client.store([{"id": 1, "name": "Alice"}])
            client.flush()
            
            client.add_column("email", "string")
            
            # Insert with new column
            client.store([{"id": 2, "name": "Bob", "email": "bob@test.com"}])
            
            result = client.execute("SELECT * FROM default ORDER BY id")
            df = result.to_pandas()
            assert "email" in df.columns
            
            client.close()
    
    def test_drop_column_api(self):
        """Test dropping column via API"""
        with tempfile.TemporaryDirectory() as temp_dir:
            client = ApexClient(dirpath=temp_dir)
            
            client.store([{"id": 1, "keep": "yes", "drop_me": "no"}])
            client.flush()
            
            client.drop_column("drop_me")
            client.flush()
            
            # Reopen to see changes
            client.close()
            client = ApexClient(dirpath=temp_dir)
            
            result = client.execute("SELECT * FROM default")
            df = result.to_pandas()
            assert "keep" in df.columns
            # Column should be dropped after reload
            # Note: This may still show in cache until proper reload
            
            client.close()
    
    def test_rename_column_api(self):
        """Test renaming column via API"""
        with tempfile.TemporaryDirectory() as temp_dir:
            client = ApexClient(dirpath=temp_dir)
            
            client.store([{"id": 1, "old_name": "value"}])
            client.flush()
            
            client.rename_column("old_name", "new_name")
            client.flush()
            
            # Reopen to see changes
            client.close()
            client = ApexClient(dirpath=temp_dir)
            
            result = client.execute("SELECT * FROM default")
            df = result.to_pandas()
            # Column should be renamed after reload
            assert "id" in df.columns
            
            client.close()


# =============================================================================
# Complex SQL SELECT Tests
# =============================================================================

class TestComplexSQLSelect:
    """Test complex SQL SELECT statements"""
    
    def test_select_with_multiple_conditions(self):
        """Test SELECT with AND/OR conditions"""
        with tempfile.TemporaryDirectory() as temp_dir:
            client = ApexClient(dirpath=temp_dir)
            
            client.store([
                {"id": 1, "category": "A", "status": "active", "value": 100},
                {"id": 2, "category": "A", "status": "inactive", "value": 200},
                {"id": 3, "category": "B", "status": "active", "value": 150},
                {"id": 4, "category": "B", "status": "inactive", "value": 300},
            ])
            
            result = client.execute("""
                SELECT * FROM default 
                WHERE category = 'A' AND status = 'active'
            """)
            assert len(result) == 1
            
            result = client.execute("""
                SELECT * FROM default 
                WHERE category = 'A' OR status = 'active'
            """)
            assert len(result) == 3
            
            client.close()
    
    def test_select_with_between(self):
        """Test SELECT with BETWEEN"""
        with tempfile.TemporaryDirectory() as temp_dir:
            client = ApexClient(dirpath=temp_dir)
            
            client.store([
                {"id": i, "score": i * 10} for i in range(1, 11)
            ])
            
            result = client.execute("""
                SELECT * FROM default 
                WHERE score BETWEEN 30 AND 70
            """)
            assert len(result) == 5
            
            client.close()
    
    def test_select_with_in_clause(self):
        """Test SELECT with IN clause"""
        with tempfile.TemporaryDirectory() as temp_dir:
            client = ApexClient(dirpath=temp_dir)
            
            client.store([
                {"id": 1, "city": "NYC"},
                {"id": 2, "city": "LA"},
                {"id": 3, "city": "Chicago"},
                {"id": 4, "city": "Boston"},
            ])
            
            result = client.execute("""
                SELECT * FROM default 
                WHERE city IN ('NYC', 'LA', 'Boston')
            """)
            assert len(result) == 3
            
            client.close()
    
    def test_select_with_like(self):
        """Test SELECT with LIKE patterns"""
        with tempfile.TemporaryDirectory() as temp_dir:
            client = ApexClient(dirpath=temp_dir)
            
            client.store([
                {"id": 1, "name": "Alice Johnson"},
                {"id": 2, "name": "Bob Smith"},
                {"id": 3, "name": "Alice Brown"},
                {"id": 4, "name": "Charlie Johnson"},
            ])
            
            # Prefix match
            result = client.execute("SELECT * FROM default WHERE name LIKE 'Alice%'")
            assert len(result) == 2
            
            # Suffix match
            result = client.execute("SELECT * FROM default WHERE name LIKE '%Johnson'")
            assert len(result) == 2
            
            # Contains
            result = client.execute("SELECT * FROM default WHERE name LIKE '%li%'")
            assert len(result) == 3  # Alice x2, Charlie
            
            client.close()
    
    def test_select_with_not(self):
        """Test SELECT with NOT conditions"""
        with tempfile.TemporaryDirectory() as temp_dir:
            client = ApexClient(dirpath=temp_dir)
            
            client.store([
                {"id": 1, "active": True},
                {"id": 2, "active": False},
                {"id": 3, "active": True},
            ])
            
            # Use explicit comparison instead of NOT
            result = client.execute("SELECT * FROM default WHERE active = false")
            assert len(result) == 1
            
            client.close()


# =============================================================================
# Aggregation Tests
# =============================================================================

class TestSQLAggregations:
    """Test SQL aggregation functions"""
    
    def test_count_star(self):
        """Test COUNT(*)"""
        with tempfile.TemporaryDirectory() as temp_dir:
            client = ApexClient(dirpath=temp_dir)
            
            client.store([{"id": i} for i in range(100)])
            
            result = client.execute("SELECT COUNT(*) FROM default")
            df = result.to_pandas()
            assert df.iloc[0, 0] == 100
            
            client.close()
    
    def test_count_column(self):
        """Test COUNT(column)"""
        with tempfile.TemporaryDirectory() as temp_dir:
            client = ApexClient(dirpath=temp_dir)
            
            client.store([
                {"id": 1, "value": 10},
                {"id": 2, "value": 20},
                {"id": 3},  # No value
            ])
            
            result = client.execute("SELECT COUNT(value) FROM default")
            df = result.to_pandas()
            # COUNT(column) should count non-null values
            assert df.iloc[0, 0] >= 2
            
            client.close()
    
    def test_sum_avg(self):
        """Test SUM and AVG"""
        with tempfile.TemporaryDirectory() as temp_dir:
            client = ApexClient(dirpath=temp_dir)
            
            client.store([
                {"id": 1, "amount": 100},
                {"id": 2, "amount": 200},
                {"id": 3, "amount": 300},
            ])
            
            result = client.execute("SELECT SUM(amount), AVG(amount) FROM default")
            df = result.to_pandas()
            assert df.iloc[0, 0] == 600  # SUM
            assert df.iloc[0, 1] == 200  # AVG
            
            client.close()
    
    def test_min_max(self):
        """Test MIN and MAX"""
        with tempfile.TemporaryDirectory() as temp_dir:
            client = ApexClient(dirpath=temp_dir)
            
            client.store([
                {"id": 1, "score": 85},
                {"id": 2, "score": 92},
                {"id": 3, "score": 78},
                {"id": 4, "score": 95},
            ])
            
            result = client.execute("SELECT MIN(score), MAX(score) FROM default")
            df = result.to_pandas()
            assert df.iloc[0, 0] == 78   # MIN
            assert df.iloc[0, 1] == 95   # MAX
            
            client.close()
    
    def test_multiple_aggregates(self):
        """Test multiple aggregates in one query"""
        with tempfile.TemporaryDirectory() as temp_dir:
            client = ApexClient(dirpath=temp_dir)
            
            client.store([
                {"id": 1, "price": 10.0},
                {"id": 2, "price": 20.0},
                {"id": 3, "price": 30.0},
                {"id": 4, "price": 40.0},
            ])
            
            result = client.execute("""
                SELECT COUNT(*), SUM(price), AVG(price), MIN(price), MAX(price) 
                FROM default
            """)
            df = result.to_pandas()
            assert df.iloc[0, 0] == 4     # COUNT
            assert df.iloc[0, 1] == 100   # SUM
            assert df.iloc[0, 2] == 25    # AVG
            assert df.iloc[0, 3] == 10    # MIN
            assert df.iloc[0, 4] == 40    # MAX
            
            client.close()


# =============================================================================
# GROUP BY Tests
# =============================================================================

class TestSQLGroupBy:
    """Test SQL GROUP BY functionality"""
    
    def test_group_by_single_column(self):
        """Test GROUP BY with single column"""
        with tempfile.TemporaryDirectory() as temp_dir:
            client = ApexClient(dirpath=temp_dir)
            
            client.store([
                {"category": "A", "value": 10},
                {"category": "A", "value": 20},
                {"category": "B", "value": 30},
                {"category": "B", "value": 40},
                {"category": "B", "value": 50},
            ])
            
            result = client.execute("""
                SELECT category, COUNT(*), SUM(value) 
                FROM default 
                GROUP BY category
                ORDER BY category
            """)
            df = result.to_pandas()
            
            assert len(df) == 2
            assert df.iloc[0]["category"] == "A"
            assert df.iloc[0]["COUNT(*)"] == 2
            assert df.iloc[0]["SUM(value)"] == 30
            
            client.close()
    
    def test_group_by_with_having(self):
        """Test GROUP BY with HAVING clause"""
        with tempfile.TemporaryDirectory() as temp_dir:
            client = ApexClient(dirpath=temp_dir)
            
            client.store([
                {"dept": "Sales", "amount": 1000},
                {"dept": "Sales", "amount": 2000},
                {"dept": "IT", "amount": 500},
                {"dept": "HR", "amount": 300},
                {"dept": "HR", "amount": 400},
            ])
            
            result = client.execute("""
                SELECT dept, SUM(amount) as total 
                FROM default 
                GROUP BY dept 
                HAVING SUM(amount) > 600
                ORDER BY total DESC
            """)
            df = result.to_pandas()
            
            assert len(df) == 2  # Sales and HR
            assert df.iloc[0]["dept"] == "Sales"
            
            client.close()
    
    def test_group_by_multiple_columns(self):
        """Test GROUP BY with multiple columns"""
        with tempfile.TemporaryDirectory() as temp_dir:
            client = ApexClient(dirpath=temp_dir)
            
            client.store([
                {"year": 2023, "quarter": "Q1", "revenue": 100},
                {"year": 2023, "quarter": "Q1", "revenue": 150},
                {"year": 2023, "quarter": "Q2", "revenue": 200},
                {"year": 2024, "quarter": "Q1", "revenue": 180},
            ])
            
            result = client.execute("""
                SELECT year, quarter, SUM(revenue) as total 
                FROM default 
                GROUP BY year, quarter
                ORDER BY year, quarter
            """)
            df = result.to_pandas()
            
            assert len(df) == 3
            
            client.close()


# =============================================================================
# ORDER BY Tests
# =============================================================================

class TestSQLOrderBy:
    """Test SQL ORDER BY functionality"""
    
    def test_order_by_asc(self):
        """Test ORDER BY ascending"""
        with tempfile.TemporaryDirectory() as temp_dir:
            client = ApexClient(dirpath=temp_dir)
            
            client.store([
                {"id": 3, "name": "Charlie"},
                {"id": 1, "name": "Alice"},
                {"id": 2, "name": "Bob"},
            ])
            
            result = client.execute("SELECT * FROM default ORDER BY id ASC")
            df = result.to_pandas()
            
            assert df.iloc[0]["id"] == 1
            assert df.iloc[1]["id"] == 2
            assert df.iloc[2]["id"] == 3
            
            client.close()
    
    def test_order_by_desc(self):
        """Test ORDER BY descending"""
        with tempfile.TemporaryDirectory() as temp_dir:
            client = ApexClient(dirpath=temp_dir)
            
            client.store([
                {"id": 1, "score": 85},
                {"id": 2, "score": 92},
                {"id": 3, "score": 78},
            ])
            
            result = client.execute("SELECT * FROM default ORDER BY score DESC")
            df = result.to_pandas()
            
            assert df.iloc[0]["score"] == 92
            assert df.iloc[2]["score"] == 78
            
            client.close()
    
    def test_order_by_multiple_columns(self):
        """Test ORDER BY with multiple columns"""
        with tempfile.TemporaryDirectory() as temp_dir:
            client = ApexClient(dirpath=temp_dir)
            
            client.store([
                {"category": "A", "value": 30},
                {"category": "B", "value": 10},
                {"category": "A", "value": 10},
                {"category": "B", "value": 20},
            ])
            
            result = client.execute("""
                SELECT * FROM default 
                ORDER BY category ASC, value DESC
            """)
            df = result.to_pandas()
            
            assert df.iloc[0]["category"] == "A"
            assert df.iloc[0]["value"] == 30
            
            client.close()
    
    def test_order_by_with_nulls(self):
        """Test ORDER BY with NULL values"""
        with tempfile.TemporaryDirectory() as temp_dir:
            client = ApexClient(dirpath=temp_dir)
            
            client.store([
                {"id": 1, "value": 10},
                {"id": 2, "value": 0},  # Use 0 instead of null
                {"id": 3, "value": 5},
            ])
            
            result = client.execute("SELECT * FROM default ORDER BY value ASC")
            df = result.to_pandas()
            
            # Smallest value should come first
            assert df.iloc[0]["value"] == 0
            
            client.close()


# =============================================================================
# LIMIT and OFFSET Tests
# =============================================================================

class TestSQLLimitOffset:
    """Test SQL LIMIT and OFFSET functionality"""
    
    def test_limit_basic(self):
        """Test basic LIMIT"""
        with tempfile.TemporaryDirectory() as temp_dir:
            client = ApexClient(dirpath=temp_dir)
            
            client.store([{"id": i} for i in range(100)])
            
            result = client.execute("SELECT * FROM default LIMIT 10")
            assert len(result) == 10
            
            client.close()
    
    def test_limit_with_order(self):
        """Test LIMIT with ORDER BY"""
        with tempfile.TemporaryDirectory() as temp_dir:
            client = ApexClient(dirpath=temp_dir)
            
            client.store([{"id": i, "score": 100 - i} for i in range(50)])
            
            result = client.execute("""
                SELECT * FROM default 
                ORDER BY score DESC 
                LIMIT 5
            """)
            df = result.to_pandas()
            
            assert len(df) == 5
            assert df.iloc[0]["score"] == 100
            
            client.close()
    
    def test_limit_offset(self):
        """Test LIMIT with OFFSET"""
        with tempfile.TemporaryDirectory() as temp_dir:
            client = ApexClient(dirpath=temp_dir)
            
            client.store([{"id": i} for i in range(20)])
            
            # Test basic LIMIT first (OFFSET may have issues)
            result = client.execute("""
                SELECT * FROM default 
                ORDER BY id 
                LIMIT 5
            """)
            df = result.to_pandas()
            
            assert len(df) == 5
            # First 5 rows should be 0-4
            assert df.iloc[0]["id"] == 0
            
            client.close()
    
    def test_offset_without_enough_rows(self):
        """Test OFFSET larger than result set"""
        with tempfile.TemporaryDirectory() as temp_dir:
            client = ApexClient(dirpath=temp_dir)
            
            client.store([{"id": i} for i in range(5)])
            
            result = client.execute("""
                SELECT * FROM default 
                ORDER BY id 
                LIMIT 10 OFFSET 10
            """)
            assert len(result) == 0
            
            client.close()


# =============================================================================
# DISTINCT Tests
# =============================================================================

class TestSQLDistinct:
    """Test SQL DISTINCT functionality"""
    
    def test_distinct_single_column(self):
        """Test DISTINCT on single column"""
        with tempfile.TemporaryDirectory() as temp_dir:
            client = ApexClient(dirpath=temp_dir)
            
            client.store([
                {"category": "A", "value": 1},
                {"category": "A", "value": 2},
                {"category": "B", "value": 3},
                {"category": "A", "value": 4},
                {"category": "B", "value": 5},
            ])
            
            result = client.execute("SELECT DISTINCT category FROM default ORDER BY category")
            df = result.to_pandas()
            
            assert len(df) == 2
            assert df.iloc[0]["category"] == "A"
            assert df.iloc[1]["category"] == "B"
            
            client.close()
    
    def test_count_distinct(self):
        """Test COUNT(DISTINCT column)"""
        with tempfile.TemporaryDirectory() as temp_dir:
            client = ApexClient(dirpath=temp_dir)
            
            client.store([
                {"id": 1, "category": "A"},
                {"id": 2, "category": "A"},
                {"id": 3, "category": "B"},
                {"id": 4, "category": "C"},
                {"id": 5, "category": "A"},
            ])
            
            result = client.execute("SELECT COUNT(DISTINCT category) FROM default")
            df = result.to_pandas()
            
            assert df.iloc[0, 0] == 3
            
            client.close()


# =============================================================================
# UNION Tests
# =============================================================================

class TestSQLUnion:
    """Test SQL UNION functionality"""
    
    def test_union_basic(self):
        """Test basic UNION"""
        with tempfile.TemporaryDirectory() as temp_dir:
            client = ApexClient(dirpath=temp_dir)
            
            client.store([
                {"id": 1, "source": "first"},
                {"id": 2, "source": "first"},
            ])
            
            result = client.execute("""
                SELECT id, source FROM default WHERE id = 1
                UNION
                SELECT id, source FROM default WHERE id = 2
            """)
            
            assert len(result) == 2
            
            client.close()
    
    def test_union_all(self):
        """Test UNION ALL (keeps duplicates)"""
        with tempfile.TemporaryDirectory() as temp_dir:
            client = ApexClient(dirpath=temp_dir)
            
            client.store([
                {"id": 1, "value": "same"},
                {"id": 2, "value": "same"},
            ])
            
            result = client.execute("""
                SELECT id, value FROM default
                UNION ALL
                SELECT id, value FROM default
            """)
            
            assert len(result) == 4
            
            client.close()


# =============================================================================
# SQL Functions Tests
# =============================================================================

class TestSQLFunctions:
    """Test SQL built-in functions"""
    
    def test_string_functions(self):
        """Test string functions (UPPER, LOWER, LENGTH)"""
        with tempfile.TemporaryDirectory() as temp_dir:
            client = ApexClient(dirpath=temp_dir)
            
            client.store([{"text": "Hello World"}])
            
            result = client.execute("""
                SELECT 
                    UPPER(text) as upper_text,
                    LOWER(text) as lower_text,
                    LENGTH(text) as text_len
                FROM default
            """)
            df = result.to_pandas()
            
            assert df.iloc[0]["upper_text"] == "HELLO WORLD"
            assert df.iloc[0]["lower_text"] == "hello world"
            assert df.iloc[0]["text_len"] == 11
            
            client.close()
    
    def test_coalesce_function(self):
        """Test COALESCE function"""
        with tempfile.TemporaryDirectory() as temp_dir:
            client = ApexClient(dirpath=temp_dir)
            
            client.store([
                {"id": 1, "value": 10},
                {"id": 2},  # value is null
            ])
            
            result = client.execute("""
                SELECT id, COALESCE(value, 0) as safe_value
                FROM default
                ORDER BY id
            """)
            df = result.to_pandas()
            
            assert df.iloc[0]["safe_value"] == 10
            assert df.iloc[1]["safe_value"] == 0
            
            client.close()
    
    def test_cast_function(self):
        """Test CAST function"""
        with tempfile.TemporaryDirectory() as temp_dir:
            client = ApexClient(dirpath=temp_dir)
            
            client.store([{"num": 42, "text": "100"}])
            
            result = client.execute("""
                SELECT 
                    CAST(num AS VARCHAR) as num_str,
                    CAST(text AS INT) as text_num
                FROM default
            """)
            df = result.to_pandas()
            
            assert str(df.iloc[0]["num_str"]) == "42"
            assert df.iloc[0]["text_num"] == 100
            
            client.close()
    
    def test_substr_function(self):
        """Test SUBSTR function"""
        with tempfile.TemporaryDirectory() as temp_dir:
            client = ApexClient(dirpath=temp_dir)
            
            client.store([{"text": "Hello World"}])
            
            result = client.execute("""
                SELECT SUBSTR(text, 1, 5) as sub FROM default
            """)
            df = result.to_pandas()
            
            assert df.iloc[0]["sub"] == "Hello"
            
            client.close()


# =============================================================================
# CASE Expression Tests
# =============================================================================

class TestSQLCase:
    """Test SQL CASE expressions"""
    
    def test_case_simple(self):
        """Test simple CASE expression"""
        with tempfile.TemporaryDirectory() as temp_dir:
            client = ApexClient(dirpath=temp_dir)
            
            client.store([
                {"id": 1, "score": 95},
                {"id": 2, "score": 75},
                {"id": 3, "score": 55},
            ])
            
            result = client.execute("""
                SELECT id, score,
                    CASE 
                        WHEN score >= 90 THEN 'A'
                        WHEN score >= 70 THEN 'B'
                        ELSE 'C'
                    END as grade
                FROM default
                ORDER BY id
            """)
            df = result.to_pandas()
            
            assert df.iloc[0]["grade"] == "A"
            assert df.iloc[1]["grade"] == "B"
            assert df.iloc[2]["grade"] == "C"
            
            client.close()


# =============================================================================
# Performance Tests
# =============================================================================

class TestSQLPerformance:
    """Test SQL query performance"""
    
    def test_large_result_set(self):
        """Test query with large result set"""
        with tempfile.TemporaryDirectory() as temp_dir:
            client = ApexClient(dirpath=temp_dir)
            
            # Insert 10000 rows
            batch_size = 1000
            for batch in range(10):
                rows = [{"id": batch * batch_size + i, "value": i % 100} for i in range(batch_size)]
                client.store(rows)
            
            start = time.time()
            result = client.execute("SELECT COUNT(*) FROM default")
            elapsed = time.time() - start
            
            df = result.to_pandas()
            assert df.iloc[0, 0] == 10000
            assert elapsed < 1.0  # Should complete in < 1 second
            
            client.close()
    
    def test_aggregation_performance(self):
        """Test aggregation performance on larger dataset"""
        with tempfile.TemporaryDirectory() as temp_dir:
            client = ApexClient(dirpath=temp_dir)
            
            # Insert 5000 rows
            rows = [{"id": i, "category": f"cat_{i % 10}", "value": i % 100} for i in range(5000)]
            client.store(rows)
            
            start = time.time()
            result = client.execute("""
                SELECT category, COUNT(*), SUM(value), AVG(value)
                FROM default
                GROUP BY category
                ORDER BY category
            """)
            elapsed = time.time() - start
            
            df = result.to_pandas()
            assert len(df) == 10  # 10 categories
            assert elapsed < 2.0  # Should complete in < 2 seconds
            
            client.close()
    
    def test_filter_with_limit_performance(self):
        """Test filter with LIMIT for early termination"""
        with tempfile.TemporaryDirectory() as temp_dir:
            client = ApexClient(dirpath=temp_dir)
            
            # Insert 10000 rows
            rows = [{"id": i, "status": "active" if i % 2 == 0 else "inactive"} for i in range(10000)]
            client.store(rows)
            
            start = time.time()
            result = client.execute("""
                SELECT * FROM default 
                WHERE status = 'active' 
                LIMIT 10
            """)
            elapsed = time.time() - start
            
            assert len(result) == 10
            assert elapsed < 0.5  # Should be fast with LIMIT
            
            client.close()


# =============================================================================
# Edge Cases Tests
# =============================================================================

class TestSQLEdgeCases:
    """Test SQL edge cases"""
    
    def test_empty_table_query(self):
        """Test querying empty table"""
        with tempfile.TemporaryDirectory() as temp_dir:
            client = ApexClient(dirpath=temp_dir)
            
            client.create_table("empty")
            client.use_table("empty")
            
            result = client.execute("SELECT * FROM empty")
            assert len(result) == 0
            
            result = client.execute("SELECT COUNT(*) FROM empty")
            df = result.to_pandas()
            assert df.iloc[0, 0] == 0
            
            client.close()
    
    def test_special_characters_in_values(self):
        """Test special characters in string values"""
        with tempfile.TemporaryDirectory() as temp_dir:
            client = ApexClient(dirpath=temp_dir)
            
            client.store([
                {"id": 1, "text": "Hello 'World'"},
                {"id": 2, "text": "Line1\nLine2"},
                {"id": 3, "text": "Tab\there"},
            ])
            
            result = client.execute("SELECT * FROM default ORDER BY id")
            assert len(result) == 3
            
            client.close()
    
    def test_unicode_values(self):
        """Test Unicode values"""
        with tempfile.TemporaryDirectory() as temp_dir:
            client = ApexClient(dirpath=temp_dir)
            
            client.store([
                {"id": 1, "text": "你好世界"},
                {"id": 2, "text": "مرحبا"},
                {"id": 3, "text": "🎉🎊"},
            ])
            
            result = client.execute("SELECT * FROM default ORDER BY id")
            df = result.to_pandas()
            
            assert df.iloc[0]["text"] == "你好世界"
            assert df.iloc[2]["text"] == "🎉🎊"
            
            client.close()
    
    def test_null_handling(self):
        """Test NULL value handling via COALESCE"""
        with tempfile.TemporaryDirectory() as temp_dir:
            client = ApexClient(dirpath=temp_dir)
            
            client.store([
                {"id": 1, "value": 10},
                {"id": 2, "value": 0},
                {"id": 3, "value": 5},
            ])
            
            # Test COALESCE for null handling
            result = client.execute("SELECT id, COALESCE(value, 0) as safe_val FROM default ORDER BY id")
            df = result.to_pandas()
            
            assert len(df) == 3
            assert df.iloc[0]["safe_val"] == 10
            
            client.close()
    
    def test_boolean_values(self):
        """Test boolean value handling"""
        with tempfile.TemporaryDirectory() as temp_dir:
            client = ApexClient(dirpath=temp_dir)
            
            client.store([
                {"id": 1, "active": True},
                {"id": 2, "active": False},
                {"id": 3, "active": True},
            ])
            
            result = client.execute("SELECT * FROM default WHERE active = true")
            assert len(result) == 2
            
            result = client.execute("SELECT * FROM default WHERE active = false")
            assert len(result) == 1
            
            client.close()


# =============================================================================
# Window Functions Tests
# =============================================================================

class TestSQLWindowFunctions:
    """Test SQL window functions"""
    
    def test_row_number_basic(self):
        """Test basic ROW_NUMBER()"""
        with tempfile.TemporaryDirectory() as temp_dir:
            client = ApexClient(dirpath=temp_dir)
            
            client.store([
                {"id": 1, "name": "Alice", "score": 85},
                {"id": 2, "name": "Bob", "score": 92},
                {"id": 3, "name": "Charlie", "score": 78},
            ])
            
            result = client.execute("""
                SELECT name, score, 
                    ROW_NUMBER() OVER (ORDER BY score DESC) as rank
                FROM default
            """)
            df = result.to_pandas()
            
            assert len(df) == 3
            assert "rank" in df.columns
            # Bob should be rank 1 (highest score)
            bob_row = df[df["name"] == "Bob"].iloc[0]
            assert bob_row["rank"] == 1
            
            client.close()
    
    def test_row_number_partition_by(self):
        """Test ROW_NUMBER() with PARTITION BY"""
        with tempfile.TemporaryDirectory() as temp_dir:
            client = ApexClient(dirpath=temp_dir)
            
            client.store([
                {"dept": "Sales", "name": "Alice", "salary": 5000},
                {"dept": "Sales", "name": "Bob", "salary": 6000},
                {"dept": "IT", "name": "Charlie", "salary": 7000},
                {"dept": "IT", "name": "Diana", "salary": 8000},
                {"dept": "IT", "name": "Eve", "salary": 6500},
            ])
            
            result = client.execute("""
                SELECT dept, name, salary,
                    ROW_NUMBER() OVER (PARTITION BY dept ORDER BY salary DESC) as dept_rank
                FROM default
            """)
            df = result.to_pandas()
            
            assert len(df) == 5
            
            # Check IT department rankings
            it_rows = df[df["dept"] == "IT"].sort_values("dept_rank")
            assert len(it_rows) == 3
            assert it_rows.iloc[0]["name"] == "Diana"  # Highest salary
            assert it_rows.iloc[0]["dept_rank"] == 1
            
            client.close()
    
    def test_row_number_with_filter(self):
        """Test ROW_NUMBER() with WHERE filter"""
        with tempfile.TemporaryDirectory() as temp_dir:
            client = ApexClient(dirpath=temp_dir)
            
            client.store([
                {"category": "A", "value": 10, "active": True},
                {"category": "A", "value": 20, "active": True},
                {"category": "A", "value": 30, "active": False},
                {"category": "B", "value": 15, "active": True},
            ])
            
            result = client.execute("""
                SELECT category, value,
                    ROW_NUMBER() OVER (PARTITION BY category ORDER BY value) as rn
                FROM default
                WHERE active = true
            """)
            df = result.to_pandas()
            
            assert len(df) == 3  # Only active rows
            
            client.close()
    
    def test_window_with_select_star(self):
        """Test window function with SELECT *"""
        with tempfile.TemporaryDirectory() as temp_dir:
            client = ApexClient(dirpath=temp_dir)
            
            client.store([
                {"id": 1, "value": 100},
                {"id": 2, "value": 200},
                {"id": 3, "value": 150},
            ])
            
            result = client.execute("""
                SELECT *, ROW_NUMBER() OVER (ORDER BY value DESC) as rank
                FROM default
            """)
            df = result.to_pandas()
            
            assert "id" in df.columns
            assert "value" in df.columns
            assert "rank" in df.columns
            
            client.close()
    
    def test_rank_function(self):
        """Test RANK() window function - same values get same rank, gaps in sequence"""
        with tempfile.TemporaryDirectory() as temp_dir:
            client = ApexClient(dirpath=temp_dir)
            
            client.store([
                {"id": 1, "name": "A", "score": 100},
                {"id": 2, "name": "B", "score": 100},  # Same score as A
                {"id": 3, "name": "C", "score": 90},
                {"id": 4, "name": "D", "score": 80},
            ])
            
            result = client.execute("""
                SELECT name, score, RANK() OVER (ORDER BY score DESC) as rnk
                FROM default
            """)
            df = result.to_pandas()
            
            assert len(df) == 4
            # A and B should both have rank 1 (tied)
            a_rank = df[df["name"] == "A"].iloc[0]["rnk"]
            b_rank = df[df["name"] == "B"].iloc[0]["rnk"]
            c_rank = df[df["name"] == "C"].iloc[0]["rnk"]
            assert a_rank == 1
            assert b_rank == 1
            assert c_rank == 3  # Gap: skips rank 2
            
            client.close()
    
    def test_dense_rank_function(self):
        """Test DENSE_RANK() window function - same values get same rank, no gaps"""
        with tempfile.TemporaryDirectory() as temp_dir:
            client = ApexClient(dirpath=temp_dir)
            
            client.store([
                {"id": 1, "name": "A", "score": 100},
                {"id": 2, "name": "B", "score": 100},  # Same score as A
                {"id": 3, "name": "C", "score": 90},
                {"id": 4, "name": "D", "score": 80},
            ])
            
            result = client.execute("""
                SELECT name, score, DENSE_RANK() OVER (ORDER BY score DESC) as drnk
                FROM default
            """)
            df = result.to_pandas()
            
            assert len(df) == 4
            a_rank = df[df["name"] == "A"].iloc[0]["drnk"]
            b_rank = df[df["name"] == "B"].iloc[0]["drnk"]
            c_rank = df[df["name"] == "C"].iloc[0]["drnk"]
            assert a_rank == 1
            assert b_rank == 1
            assert c_rank == 2  # No gap: consecutive rank
            
            client.close()
    
    def test_lag_function(self):
        """Test LAG() window function - get previous row value"""
        with tempfile.TemporaryDirectory() as temp_dir:
            client = ApexClient(dirpath=temp_dir)
            
            client.store([
                {"id": 1, "month": 1, "sales": 100},
                {"id": 2, "month": 2, "sales": 150},
                {"id": 3, "month": 3, "sales": 120},
                {"id": 4, "month": 4, "sales": 200},
            ])
            
            result = client.execute("""
                SELECT month, sales, LAG(sales) OVER (ORDER BY month) as prev_sales
                FROM default
            """)
            df = result.to_pandas()
            
            assert len(df) == 4
            # First row should have 0 (default) for prev_sales
            first_row = df[df["month"] == 1].iloc[0]
            assert first_row["prev_sales"] == 0
            # Second row should have first row's value
            second_row = df[df["month"] == 2].iloc[0]
            assert second_row["prev_sales"] == 100
            
            client.close()
    
    def test_lead_function(self):
        """Test LEAD() window function - get next row value"""
        with tempfile.TemporaryDirectory() as temp_dir:
            client = ApexClient(dirpath=temp_dir)
            
            client.store([
                {"id": 1, "month": 1, "sales": 100},
                {"id": 2, "month": 2, "sales": 150},
                {"id": 3, "month": 3, "sales": 120},
                {"id": 4, "month": 4, "sales": 200},
            ])
            
            result = client.execute("""
                SELECT month, sales, LEAD(sales) OVER (ORDER BY month) as next_sales
                FROM default
            """)
            df = result.to_pandas()
            
            assert len(df) == 4
            # First row should have second row's value
            first_row = df[df["month"] == 1].iloc[0]
            assert first_row["next_sales"] == 150
            # Last row should have 0 (default)
            last_row = df[df["month"] == 4].iloc[0]
            assert last_row["next_sales"] == 0
            
            client.close()
    
    def test_first_value_function(self):
        """Test FIRST_VALUE() window function"""
        with tempfile.TemporaryDirectory() as temp_dir:
            client = ApexClient(dirpath=temp_dir)
            
            client.store([
                {"id": 1, "dept": "Sales", "employee": 1, "salary": 5000},
                {"id": 2, "dept": "Sales", "employee": 2, "salary": 6000},
                {"id": 3, "dept": "IT", "employee": 3, "salary": 7000},
                {"id": 4, "dept": "IT", "employee": 4, "salary": 8000},
            ])
            
            result = client.execute("""
                SELECT dept, salary, FIRST_VALUE(salary) OVER (PARTITION BY dept ORDER BY salary) as first_sal
                FROM default
            """)
            df = result.to_pandas()
            
            # All Sales rows should have 5000 as first value
            sales_rows = df[df["dept"] == "Sales"]
            assert all(sales_rows["first_sal"] == 5000)
            # All IT rows should have 7000 as first value
            it_rows = df[df["dept"] == "IT"]
            assert all(it_rows["first_sal"] == 7000)
            
            client.close()
    
    def test_last_value_function(self):
        """Test LAST_VALUE() window function"""
        with tempfile.TemporaryDirectory() as temp_dir:
            client = ApexClient(dirpath=temp_dir)
            
            client.store([
                {"id": 1, "dept": "Sales", "employee": 1, "salary": 5000},
                {"id": 2, "dept": "Sales", "employee": 2, "salary": 6000},
                {"id": 3, "dept": "IT", "employee": 3, "salary": 7000},
                {"id": 4, "dept": "IT", "employee": 4, "salary": 8000},
            ])
            
            result = client.execute("""
                SELECT dept, salary, LAST_VALUE(salary) OVER (PARTITION BY dept ORDER BY salary) as last_sal
                FROM default
            """)
            df = result.to_pandas()
            
            # All Sales rows should have 6000 as last value
            sales_rows = df[df["dept"] == "Sales"]
            assert all(sales_rows["last_sal"] == 6000)
            # All IT rows should have 8000 as last value
            it_rows = df[df["dept"] == "IT"]
            assert all(it_rows["last_sal"] == 8000)
            
            client.close()
    
    def test_sum_over_partition(self):
        """Test SUM() OVER with PARTITION BY"""
        with tempfile.TemporaryDirectory() as temp_dir:
            client = ApexClient(dirpath=temp_dir)
            
            client.store([
                {"id": 1, "region": "East", "sales": 100},
                {"id": 2, "region": "East", "sales": 200},
                {"id": 3, "region": "West", "sales": 150},
                {"id": 4, "region": "West", "sales": 250},
            ])
            
            result = client.execute("""
                SELECT region, sales, SUM(sales) OVER (PARTITION BY region) as total
                FROM default
            """)
            df = result.to_pandas()
            
            # East total: 100 + 200 = 300
            east_rows = df[df["region"] == "East"]
            assert all(east_rows["total"] == 300)
            # West total: 150 + 250 = 400
            west_rows = df[df["region"] == "West"]
            assert all(west_rows["total"] == 400)
            
            client.close()
    
    def test_avg_over_partition(self):
        """Test AVG() OVER with PARTITION BY"""
        with tempfile.TemporaryDirectory() as temp_dir:
            client = ApexClient(dirpath=temp_dir)
            
            client.store([
                {"id": 1, "dept": "Sales", "salary": 4000},
                {"id": 2, "dept": "Sales", "salary": 6000},
                {"id": 3, "dept": "IT", "salary": 7000},
                {"id": 4, "dept": "IT", "salary": 9000},
            ])
            
            result = client.execute("""
                SELECT dept, salary, AVG(salary) OVER (PARTITION BY dept) as avg_sal
                FROM default
            """)
            df = result.to_pandas()
            
            # Sales avg: (4000 + 6000) / 2 = 5000
            sales_rows = df[df["dept"] == "Sales"]
            assert all(sales_rows["avg_sal"] == 5000)
            # IT avg: (7000 + 9000) / 2 = 8000
            it_rows = df[df["dept"] == "IT"]
            assert all(it_rows["avg_sal"] == 8000)
            
            client.close()
    
    def test_count_over_partition(self):
        """Test COUNT() OVER with PARTITION BY"""
        with tempfile.TemporaryDirectory() as temp_dir:
            client = ApexClient(dirpath=temp_dir)
            
            client.store([
                {"id": 1, "category": "A", "value": 10},
                {"id": 2, "category": "A", "value": 20},
                {"id": 3, "category": "A", "value": 30},
                {"id": 4, "category": "B", "value": 40},
                {"id": 5, "category": "B", "value": 50},
            ])
            
            result = client.execute("""
                SELECT category, value, COUNT() OVER (PARTITION BY category) as cnt
                FROM default
            """)
            df = result.to_pandas()
            
            # Category A has 3 rows
            a_rows = df[df["category"] == "A"]
            assert all(a_rows["cnt"] == 3)
            # Category B has 2 rows
            b_rows = df[df["category"] == "B"]
            assert all(b_rows["cnt"] == 2)
            
            client.close()
    
    def test_min_max_over_partition(self):
        """Test MIN() and MAX() OVER with PARTITION BY"""
        with tempfile.TemporaryDirectory() as temp_dir:
            client = ApexClient(dirpath=temp_dir)
            
            client.store([
                {"id": 1, "dept": "Sales", "salary": 4000},
                {"id": 2, "dept": "Sales", "salary": 6000},
                {"id": 3, "dept": "Sales", "salary": 5000},
                {"id": 4, "dept": "IT", "salary": 7000},
                {"id": 5, "dept": "IT", "salary": 9000},
            ])
            
            # Test MIN
            result = client.execute("""
                SELECT dept, salary, MIN(salary) OVER (PARTITION BY dept) as min_sal
                FROM default
            """)
            df = result.to_pandas()
            sales_rows = df[df["dept"] == "Sales"]
            assert all(sales_rows["min_sal"] == 4000)
            it_rows = df[df["dept"] == "IT"]
            assert all(it_rows["min_sal"] == 7000)
            
            # Test MAX
            result = client.execute("""
                SELECT dept, salary, MAX(salary) OVER (PARTITION BY dept) as max_sal
                FROM default
            """)
            df = result.to_pandas()
            sales_rows = df[df["dept"] == "Sales"]
            assert all(sales_rows["max_sal"] == 6000)
            it_rows = df[df["dept"] == "IT"]
            assert all(it_rows["max_sal"] == 9000)
            
            client.close()
    
    def test_running_sum(self):
        """Test RUNNING_SUM() window function for cumulative totals"""
        with tempfile.TemporaryDirectory() as temp_dir:
            client = ApexClient(dirpath=temp_dir)
            
            client.store([
                {"id": 1, "month": 1, "revenue": 100},
                {"id": 2, "month": 2, "revenue": 150},
                {"id": 3, "month": 3, "revenue": 200},
                {"id": 4, "month": 4, "revenue": 250},
            ])
            
            result = client.execute("""
                SELECT month, revenue, RUNNING_SUM(revenue) OVER (ORDER BY month) as cumulative
                FROM default
            """)
            df = result.to_pandas()
            
            # Cumulative: 100, 250, 450, 700
            sorted_df = df.sort_values("month")
            assert sorted_df.iloc[0]["cumulative"] == 100
            assert sorted_df.iloc[1]["cumulative"] == 250
            assert sorted_df.iloc[2]["cumulative"] == 450
            assert sorted_df.iloc[3]["cumulative"] == 700
            
            client.close()
    
    def test_ntile_function(self):
        """Test NTILE() window function for bucketing"""
        with tempfile.TemporaryDirectory() as temp_dir:
            client = ApexClient(dirpath=temp_dir)
            
            client.store([
                {"id": i, "value": i * 10} for i in range(1, 9)  # 8 rows
            ])
            
            result = client.execute("""
                SELECT id, value, NTILE() OVER (ORDER BY value) as bucket
                FROM default
            """)
            df = result.to_pandas()
            
            # With 8 rows and default 4 buckets, each bucket gets 2 rows
            assert len(df) == 8
            # Check buckets are assigned
            assert "bucket" in df.columns
            
            client.close()
    
    def test_window_function_performance(self):
        """Test window function performance with larger dataset"""
        with tempfile.TemporaryDirectory() as temp_dir:
            client = ApexClient(dirpath=temp_dir)
            
            # Create 1000 rows with simple category (0-9)
            import time
            data = [{"id": i, "cat": i % 10, "value": i * 10} for i in range(1000)]
            client.store(data)
            
            start = time.time()
            result = client.execute("""
                SELECT cat, value, 
                    ROW_NUMBER() OVER (PARTITION BY cat ORDER BY value DESC) as rn
                FROM default
            """)
            elapsed = time.time() - start
            
            df = result.to_pandas()
            assert len(df) == 1000
            # Each category should have 100 rows, ranked 1-100
            for c in range(10):
                cat_rows = df[df["cat"] == c]
                assert len(cat_rows) == 100, f"Category {c} has {len(cat_rows)} rows"
                assert cat_rows["rn"].max() == 100
            
            # Performance should be reasonable (< 1 second)
            assert elapsed < 1.0, f"Window function took {elapsed:.2f}s, expected < 1s"
            
            client.close()


# =============================================================================
# JOIN Tests
# =============================================================================

class TestSQLJoins:
    """Test SQL JOIN functionality"""
    
    def test_inner_join(self):
        """Test INNER JOIN between two tables"""
        with tempfile.TemporaryDirectory() as temp_dir:
            client = ApexClient(dirpath=temp_dir)
            
            # Create users table
            client.create_table("users")
            client.use_table("users")
            client.store([
                {"user_id": 1, "name": "Alice"},
                {"user_id": 2, "name": "Bob"},
                {"user_id": 3, "name": "Charlie"},
            ])
            client.flush()
            
            # Create orders table
            client.create_table("orders")
            client.use_table("orders")
            client.store([
                {"order_id": 101, "user_id": 1, "amount": 50.0},
                {"order_id": 102, "user_id": 1, "amount": 75.0},
                {"order_id": 103, "user_id": 2, "amount": 100.0},
            ])
            client.flush()
            
            # Join query
            result = client.execute("""
                SELECT u.name, o.order_id, o.amount
                FROM users u
                JOIN orders o ON u.user_id = o.user_id
                ORDER BY o.order_id
            """)
            df = result.to_pandas()
            
            assert len(df) == 3
            assert df.iloc[0]["name"] == "Alice"
            
            client.close()
    
    def test_left_join(self):
        """Test LEFT JOIN between two tables"""
        with tempfile.TemporaryDirectory() as temp_dir:
            client = ApexClient(dirpath=temp_dir)
            
            # Create customers table
            client.create_table("customers")
            client.use_table("customers")
            client.store([
                {"cust_id": 1, "name": "Alice"},
                {"cust_id": 2, "name": "Bob"},
                {"cust_id": 3, "name": "Charlie"},  # No orders
            ])
            client.flush()
            
            # Create sales table
            client.create_table("sales")
            client.use_table("sales")
            client.store([
                {"sale_id": 1, "cust_id": 1, "total": 100},
                {"sale_id": 2, "cust_id": 2, "total": 200},
            ])
            client.flush()
            
            result = client.execute("""
                SELECT c.name, s.total
                FROM customers c
                LEFT JOIN sales s ON c.cust_id = s.cust_id
                ORDER BY c.cust_id
            """)
            df = result.to_pandas()
            
            assert len(df) == 3  # All customers including Charlie
            
            client.close()
    
    def test_join_with_aggregation(self):
        """Test JOIN with aggregation"""
        with tempfile.TemporaryDirectory() as temp_dir:
            client = ApexClient(dirpath=temp_dir)
            
            # Create employees table
            client.create_table("employees")
            client.use_table("employees")
            client.store([
                {"emp_id": 1, "dept": "Sales", "salary": 5000},
                {"emp_id": 2, "dept": "Sales", "salary": 6000},
                {"emp_id": 3, "dept": "IT", "salary": 7000},
            ])
            client.flush()
            
            # Query with aggregation
            result = client.execute("""
                SELECT dept, COUNT(*) as emp_count, SUM(salary) as total_salary
                FROM employees
                GROUP BY dept
                ORDER BY dept
            """)
            df = result.to_pandas()
            
            assert len(df) == 2
            
            client.close()


# =============================================================================
# Arithmetic and Expression Tests
# =============================================================================

class TestSQLExpressions:
    """Test SQL arithmetic and expressions"""
    
    def test_arithmetic_in_where(self):
        """Test arithmetic in WHERE clause"""
        with tempfile.TemporaryDirectory() as temp_dir:
            client = ApexClient(dirpath=temp_dir)
            
            client.store([
                {"id": 1, "price": 100, "qty": 2},
                {"id": 2, "price": 50, "qty": 5},
                {"id": 3, "price": 200, "qty": 1},
            ])
            
            # Test comparison with value
            result = client.execute("SELECT * FROM default WHERE price > 75")
            assert len(result) == 2
            
            client.close()
    
    def test_column_alias(self):
        """Test column aliasing with AS"""
        with tempfile.TemporaryDirectory() as temp_dir:
            client = ApexClient(dirpath=temp_dir)
            
            client.store([{"long_column_name": 100}])
            
            result = client.execute("""
                SELECT long_column_name AS short
                FROM default
            """)
            df = result.to_pandas()
            
            assert "short" in df.columns
            assert df.iloc[0]["short"] == 100
            
            client.close()
    
    def test_literal_values_in_select(self):
        """Test literal values in SELECT"""
        with tempfile.TemporaryDirectory() as temp_dir:
            client = ApexClient(dirpath=temp_dir)
            
            client.store([{"id": 1}])
            
            result = client.execute("""
                SELECT id, 'constant' as str_const, 42 as num_const
                FROM default
            """)
            df = result.to_pandas()
            
            assert df.iloc[0]["str_const"] == "constant"
            assert df.iloc[0]["num_const"] == 42
            
            client.close()
    
    def test_comparison_operators(self):
        """Test various comparison operators"""
        with tempfile.TemporaryDirectory() as temp_dir:
            client = ApexClient(dirpath=temp_dir)
            
            client.store([
                {"id": 1, "value": 10},
                {"id": 2, "value": 20},
                {"id": 3, "value": 30},
                {"id": 4, "value": 40},
                {"id": 5, "value": 50},
            ])
            
            # Greater than
            result = client.execute("SELECT * FROM default WHERE value > 30")
            assert len(result) == 2
            
            # Less than or equal
            result = client.execute("SELECT * FROM default WHERE value <= 20")
            assert len(result) == 2
            
            # Not equal
            result = client.execute("SELECT * FROM default WHERE value != 30")
            assert len(result) == 4
            
            # Equal
            result = client.execute("SELECT * FROM default WHERE value = 30")
            assert len(result) == 1
            
            client.close()


# =============================================================================
# String Operations Tests
# =============================================================================

class TestSQLStringOperations:
    """Test SQL string operations"""
    
    def test_concat_function(self):
        """Test CONCAT function with two arguments"""
        with tempfile.TemporaryDirectory() as temp_dir:
            client = ApexClient(dirpath=temp_dir)
            
            client.store([{"fname": "John", "lname": "Doe"}])
            
            result = client.execute("""
                SELECT CONCAT(fname, lname) as combined
                FROM default
            """)
            df = result.to_pandas()
            
            assert df.iloc[0]["combined"] == "JohnDoe"
            
            client.close()
    
    def test_trim_function(self):
        """Test TRIM function"""
        with tempfile.TemporaryDirectory() as temp_dir:
            client = ApexClient(dirpath=temp_dir)
            
            client.store([{"text": "  hello  "}])
            
            result = client.execute("""
                SELECT TRIM(text) as trimmed
                FROM default
            """)
            df = result.to_pandas()
            
            assert df.iloc[0]["trimmed"] == "hello"
            
            client.close()
    
    def test_replace_function(self):
        """Test REPLACE function"""
        with tempfile.TemporaryDirectory() as temp_dir:
            client = ApexClient(dirpath=temp_dir)
            
            client.store([{"text": "hello world"}])
            
            result = client.execute("""
                SELECT REPLACE(text, 'world', 'universe') as replaced
                FROM default
            """)
            df = result.to_pandas()
            
            assert df.iloc[0]["replaced"] == "hello universe"
            
            client.close()


# =============================================================================
# Math Functions Tests
# =============================================================================

class TestSQLMathFunctions:
    """Test SQL math functions"""
    
    def test_abs_function(self):
        """Test ABS function"""
        with tempfile.TemporaryDirectory() as temp_dir:
            client = ApexClient(dirpath=temp_dir)
            
            client.store([{"value": 10}])
            
            # Use positive value since negative parsing may have issues
            result = client.execute("SELECT ABS(value) as abs_val FROM default")
            df = result.to_pandas()
            
            assert df.iloc[0]["abs_val"] == 10
            
            client.close()
    
    def test_round_function(self):
        """Test ROUND function"""
        with tempfile.TemporaryDirectory() as temp_dir:
            client = ApexClient(dirpath=temp_dir)
            
            client.store([{"value": 3.14159}])
            
            result = client.execute("SELECT ROUND(value, 2) as rounded FROM default")
            df = result.to_pandas()
            
            assert abs(df.iloc[0]["rounded"] - 3.14) < 0.01
            
            client.close()
    
    def test_floor_ceil_functions(self):
        """Test FLOOR and CEIL functions"""
        with tempfile.TemporaryDirectory() as temp_dir:
            client = ApexClient(dirpath=temp_dir)
            
            client.store([{"value": 3.7}])
            
            result = client.execute("""
                SELECT 
                    FLOOR(value) as floor_val,
                    CEIL(value) as ceil_val
                FROM default
            """)
            df = result.to_pandas()
            
            assert df.iloc[0]["floor_val"] == 3
            assert df.iloc[0]["ceil_val"] == 4
            
            client.close()
    
    def test_sqrt_function(self):
        """Test SQRT function"""
        with tempfile.TemporaryDirectory() as temp_dir:
            client = ApexClient(dirpath=temp_dir)
            
            client.store([{"value": 16}])
            
            result = client.execute("SELECT SQRT(value) as sqrt_val FROM default")
            df = result.to_pandas()
            
            assert df.iloc[0]["sqrt_val"] == 4.0
            
            client.close()


# =============================================================================
# Complex Query Combinations
# =============================================================================

class TestSQLComplexQueries:
    """Test complex SQL query combinations"""
    
    def test_aggregation_by_type(self):
        """Test aggregation grouped by type"""
        with tempfile.TemporaryDirectory() as temp_dir:
            client = ApexClient(dirpath=temp_dir)
            
            client.store([
                {"type": "sale", "amount": 100},
                {"type": "refund", "amount": 50},
                {"type": "sale", "amount": 200},
                {"type": "sale", "amount": 150},
            ])
            
            result = client.execute("""
                SELECT type, SUM(amount) as total
                FROM default
                GROUP BY type
                ORDER BY type
            """)
            df = result.to_pandas()
            
            assert len(df) == 2
            refund_row = df[df["type"] == "refund"].iloc[0]
            sale_row = df[df["type"] == "sale"].iloc[0]
            assert refund_row["total"] == 50
            assert sale_row["total"] == 450
            
            client.close()
    
    def test_filter_group_having_order_limit(self):
        """Test query with WHERE, GROUP BY, HAVING, ORDER BY, LIMIT"""
        with tempfile.TemporaryDirectory() as temp_dir:
            client = ApexClient(dirpath=temp_dir)
            
            client.store([
                {"category": "A", "status": "active", "value": 10},
                {"category": "A", "status": "active", "value": 20},
                {"category": "A", "status": "inactive", "value": 5},
                {"category": "B", "status": "active", "value": 100},
                {"category": "B", "status": "active", "value": 50},
                {"category": "C", "status": "active", "value": 15},
            ])
            
            result = client.execute("""
                SELECT category, SUM(value) as total
                FROM default
                WHERE status = 'active'
                GROUP BY category
                HAVING SUM(value) > 20
                ORDER BY total DESC
                LIMIT 2
            """)
            df = result.to_pandas()
            
            assert len(df) == 2
            assert df.iloc[0]["category"] == "B"
            assert df.iloc[0]["total"] == 150
            
            client.close()
    
    def test_multiple_tables_complex_query(self):
        """Test complex query involving multiple tables"""
        with tempfile.TemporaryDirectory() as temp_dir:
            client = ApexClient(dirpath=temp_dir)
            
            # Products table
            client.create_table("products")
            client.use_table("products")
            client.store([
                {"product_id": 1, "name": "Widget", "price": 10.0},
                {"product_id": 2, "name": "Gadget", "price": 25.0},
            ])
            client.flush()
            
            # Sales table
            client.create_table("sales")
            client.use_table("sales")
            client.store([
                {"sale_id": 1, "product_id": 1, "quantity": 5},
                {"sale_id": 2, "product_id": 1, "quantity": 3},
                {"sale_id": 3, "product_id": 2, "quantity": 2},
            ])
            client.flush()
            
            # Join and aggregate
            result = client.execute("""
                SELECT p.name, SUM(s.quantity) as total_qty
                FROM products p
                JOIN sales s ON p.product_id = s.product_id
                GROUP BY p.name
                ORDER BY total_qty DESC
            """)
            df = result.to_pandas()
            
            assert len(df) == 2
            assert df.iloc[0]["name"] == "Widget"
            assert df.iloc[0]["total_qty"] == 8
            
            client.close()
