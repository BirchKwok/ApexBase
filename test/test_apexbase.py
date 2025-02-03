import pytest
import os
import tempfile
import shutil
import pandas as pd
import pyarrow as pa
import polars as pl
from pathlib import Path
from apexbase import ApexClient
import random

@pytest.fixture
def temp_dir():
    """Create a temporary directory"""
    temp_path = tempfile.mkdtemp()
    yield temp_path
    shutil.rmtree(temp_path)

def test_initialization(temp_dir):
    """Test initialization parameters"""
    client = ApexClient()
    assert Path("apexbase.db").exists()
    client.close()  # Close the connection before removing the file
    os.remove("apexbase.db")  # Clean up
    
    # Test specified directory
    client = ApexClient(temp_dir)
    db_path = Path(temp_dir) / "apexbase.db"
    assert db_path.exists()
    client.close()  # Close the connection
    
    # Test drop_if_exists
    client = ApexClient(temp_dir, drop_if_exists=True)  # Should delete and recreate database
    assert db_path.exists()
    client.close()  # Close the connection
    
    # Test directory auto-creation
    nested_dir = os.path.join(temp_dir, "nested", "path")
    client = ApexClient(nested_dir)
    assert Path(nested_dir).exists()
    assert (Path(nested_dir) / "apexbase.db").exists()
    client.close()  # Close the connection

def test_basic_operations(temp_dir):
    """Test basic operations: store, query, retrieve"""
    client = ApexClient(temp_dir)
    
    # Test single record storage
    record = {"name": "John", "age": 30, "tags": ["python", "rust"]}
    id_ = client.store(record)
    assert id_ is not None
    assert isinstance(id_, int)
    
    # Test record retrieval
    retrieved = client.retrieve(id_)
    assert retrieved is not None
    assert retrieved["name"] == record["name"]
    assert retrieved["age"] == record["age"]
    assert retrieved["tags"] == record["tags"]
    
    # Test query
    results = client.query("age = 30")
    assert len(results) == 1
    assert results[0]["name"] == "John"

def test_batch_operations(temp_dir):
    """Test batch operations: batch store, batch retrieve"""
    client = ApexClient(temp_dir)
    
    # Test batch store
    records = [
        {"name": "John", "age": 30},
        {"name": "Jane", "age": 25},
        {"name": "Bob", "age": 35}
    ]
    ids = client.store(records)
    assert ids is not None
    assert len(ids) == 3
    
    # Test batch retrieve
    retrieved = client.retrieve_many(ids)
    assert len(retrieved) == 3
    assert all(r["name"] in ["John", "Jane", "Bob"] for r in retrieved)
    
    # Test query
    results = client.query("age > 25")
    assert len(results) == 2
    names = [r["name"] for r in results]
    assert "John" in names
    assert "Bob" in names

def test_count_rows(temp_dir):
    """Test row count statistics"""
    client = ApexClient(temp_dir)
    
    # Test empty table
    assert client.count_rows() == 0
    
    # Test single record
    client.store({"name": "John"})
    assert client.count_rows() == 1
    
    # Test multiple records
    records = [
        {"name": "Jane"},
        {"name": "Bob"},
        {"name": "Alice"}
    ]
    client.store(records)
    assert client.count_rows() == 4
    
    # Test multiple tables
    client.create_table("test_table")
    client.use_table("test_table")
    assert client.count_rows() == 0  # New table should be empty
    
    client.store({"name": "Test"})
    assert client.count_rows() == 1  # New table should have one record
    
    client.use_table("default")
    assert client.count_rows() == 4  # Default table should remain with 4 records
    
    # Test non-existent table
    with pytest.raises(ValueError):
        client.count_rows("nonexistent_table")

def test_delete_operations(temp_dir):
    """Test delete operations: single delete, batch delete"""
    client = ApexClient(temp_dir)
    
    # Prepare test data
    records = [
        {"name": "John", "age": 30},
        {"name": "Jane", "age": 25},
        {"name": "Bob", "age": 35}
    ]
    ids = client.store(records)
    assert ids is not None
    assert len(ids) == 3
    
    # Test single delete
    assert client.delete(ids[0]) is True
    assert client.retrieve(ids[0]) is None
    assert client.count_rows() == 2  # Verify record count
    
    # Test batch delete
    deleted_ids = client.delete(ids[1:])
    assert deleted_ids is True
    assert client.count_rows() == 0  # Verify record count

def test_replace_operations(temp_dir):
    """Test replace operations: single replace, batch replace"""
    client = ApexClient(temp_dir)
    
    # Prepare test data
    records = [
        {"name": "John", "age": 30},
        {"name": "Jane", "age": 25}
    ]
    ids = client.store(records)
    assert ids is not None
    assert len(ids) == 2
    
    # Test single replace
    new_data = {"name": "John Doe", "age": 31}
    assert client.replace(ids[0], new_data) is True
    updated = client.retrieve(ids[0])
    assert updated["name"] == "John Doe"
    assert updated["age"] == 31
    
    # Test batch replace
    batch_data = {
        ids[0]: {"name": "John Smith", "age": 32},
        ids[1]: {"name": "Jane Smith", "age": 26}
    }
    success_ids = client.batch_replace(batch_data)
    assert len(success_ids) == 2
    assert all(client.retrieve(id_)["name"].endswith("Smith") for id_ in success_ids)

def test_data_import(temp_dir):
    """Test data import: Pandas, PyArrow, Polars"""
    client = ApexClient(temp_dir)
    
    # Prepare test data
    data = {
        "name": ["John", "Jane", "Bob"],
        "age": [30, 25, 35],
        "city": ["New York", "London", "Paris"]
    }
    
    # Test Pandas import
    df_pandas = pd.DataFrame(data)
    client.from_pandas(df_pandas)
    results = client.query("age > 0")
    assert len(results.ids) == 3
    
    # Test PyArrow import
    table = pa.Table.from_pandas(df_pandas)
    client.from_pyarrow(table)
    results = client.query("age > 0")
    assert len(results.ids) == 3
    
    # Test Polars import
    df_polars = pl.DataFrame(data)
    client.from_polars(df_polars)
    results = client.query("age > 0")
    assert len(results.ids) == 3

def test_concurrent_access(temp_dir):
    """Test concurrent access"""
    import threading
    import random
    
    client = ApexClient(temp_dir)
    num_threads = 4
    records_per_thread = 100
    
    def worker():
        for _ in range(records_per_thread):
            record = {
                "value": random.randint(1, 1000),
                "thread_id": threading.get_ident()
            }
            id_ = client.store(record)
            assert id_ is not None
    
    threads = [threading.Thread(target=worker) for _ in range(num_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    
    # Verify all records are correctly stored
    results = client.query("1=1")
    assert len(results) == num_threads * records_per_thread

    # Verify record count for each thread
    for thread_id in set(r["thread_id"] for r in results):
        thread_results = client.query(f"thread_id = {thread_id}")
        assert len(thread_results) == records_per_thread

def test_error_handling(temp_dir):
    """Test error handling"""
    client = ApexClient(temp_dir)
    
    # Test invalid data storage
    with pytest.raises(ValueError):
        client.store("invalid data")
    
    # Test non-existent record retrieval
    assert client.retrieve(999) is None
    
    # Test non-existent record deletion
    assert client.delete(999) is False
    
    # Test non-existent record replace
    assert client.replace(999, {"name": "test"}) is False
    
    # Test invalid query syntax
    with pytest.raises(ValueError):
        client.query("INVALID SYNTAX !@#").ids
        client.query("non_existent_field = 'value'").ids

def test_large_batch_operations(temp_dir):
    """Test large batch operations performance"""
    client = ApexClient(temp_dir)
    
    # Create large test data
    num_records = 10000
    records = [
        {"id": i, "value": i * 2}
        for i in range(num_records)
    ]
    
    # Test batch storage performance
    ids = client.store(records)
    assert ids is not None
    assert len(ids) == num_records
    
    # Test query performance
    results = client.query("value >= 0")
    assert len(results) == num_records
    
    # Test range query performance
    results = client.query("value >= 1000 AND value <= 2000")
    assert len(results) == 501  # (2000 - 1000) / 2 + 1
    
    # Test batch retrieval performance
    sample_size = 1000
    sample_ids = random.sample(ids, sample_size)
    retrieved = client.retrieve_many(sample_ids)
    assert len(retrieved) == sample_size

def test_complex_queries(temp_dir):
    """Test complex queries"""
    client = ApexClient(temp_dir)
    
    # Prepare test data
    records = [
        {"name": "John", "age": 30, "city": "New York", "score": 85.5},
        {"name": "Jane", "age": 25, "city": "London", "score": 92.0},
        {"name": "Bob", "age": 35, "city": "New York", "score": 78.5},
        {"name": "Alice", "age": 28, "city": "Paris", "score": 88.0}
    ]
    ids = client.store(records)
    assert ids is not None
    assert len(ids) == 4
    
    # Test multi-condition query
    results = client.query("age > 25 AND city = 'New York'")
    assert len(results) == 2
    assert all(r["city"] == "New York" and r["age"] > 25 for r in results)
    
    # Test range query
    results = client.query("score >= 85.0 AND score <= 90.0")
    assert len(results) == 2
    assert all(85.0 <= r["score"] <= 90.0 for r in results)
    
    # Test LIKE query
    results = client.query("name LIKE 'J%'")
    assert len(results) == 2
    assert all(r["name"].startswith("J") for r in results)

def test_case_insensitive_search(temp_dir):
    """Test case-insensitive search"""
    client = ApexClient(temp_dir)
    
    # Prepare test data
    records = [
        {"name": "John Smith", "email": "JOHN@example.com"},
        {"name": "JANE DOE", "email": "jane@EXAMPLE.com"},
        {"name": "Bob Wilson", "email": "bob@Example.COM"}
    ]
    ids = client.store(records)
    assert ids is not None
    assert len(ids) == 3
    
    # Set fields to be searchable
    client.set_searchable("name", True)
    client.set_searchable("email", True)
    
    # 1. Test full-text search case-insensitive
    test_cases = [
        ("john", 1),      # Lowercase search uppercase content
        ("JANE", 1),      # Uppercase search uppercase content
        ("Bob", 1),       # Capitalize first letter search
        ("EXAMPLE", 3),   # Uppercase search mixed case
        ("example", 3),   # Lowercase search mixed case
        ("COM", 3),       # Uppercase search domain name
        ("com", 3)        # Lowercase search domain name
    ]
    
    for search_term, expected_count in test_cases:
        results = client.search_text(search_term)
        assert len(results) == expected_count, \
            f"The search term '{search_term}' should return {expected_count} results, but returned {len(results)} results"
    
    # 2. Test SQL query case-insensitive
    sql_test_cases = [
        # LIKE operator case-insensitive
        ("name LIKE '%JOHN%'", 1),
        ("name like '%john%'", 1),
        ("email LIKE '%COM%'", 3),
        ("email like '%com%'", 3),
        
        # Logical operator case-insensitive
        ("name LIKE '%John%' AND email LIKE '%example%'", 1),
        ("name like '%John%' and email like '%example%'", 1),
        ("name LIKE '%John%' OR name LIKE '%Jane%'", 2),
        ("name like '%John%' or name like '%Jane%'", 2)
    ]
    
    for query, expected_count in sql_test_cases:
        results = client.query(query)
        assert len(results) == expected_count, \
            f"The query '{query}' should return {expected_count} results, but returned {len(results)} results" 
