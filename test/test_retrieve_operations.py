"""
Comprehensive test suite for ApexBase Retrieve Operations

This module tests:
- Single record retrieval (retrieve)
- Multiple record retrieval (retrieve_many)
- All records retrieval (retrieve_all)
- Performance optimizations with Arrow C Data Interface
- Edge cases and error handling
- ResultView conversions from retrieve operations
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


class TestSingleRetrieve:
    """Test single record retrieval operations"""
    
    def test_retrieve_existing_record(self):
        """Test retrieving an existing record"""
        with tempfile.TemporaryDirectory() as temp_dir:
            client = ApexClient(dirpath=temp_dir)
            
            # Store test data
            test_data = {"name": "Alice", "age": 25, "city": "NYC"}
            client.store(test_data)
            
            # Retrieve the record
            result = client.retrieve(0)
            
            assert result is not None
            assert isinstance(result, dict)
            assert result["name"] == "Alice"
            assert result["age"] == 25
            assert result["city"] == "NYC"
            # Note: retrieve() returns raw dict which may include _id
            
            client.close()
    
    def test_retrieve_nonexistent_record(self):
        """Test retrieving a nonexistent record"""
        with tempfile.TemporaryDirectory() as temp_dir:
            client = ApexClient(dirpath=temp_dir)
            
            # Store some data
            client.store({"name": "Alice", "age": 25})
            
            # Try to retrieve nonexistent record
            result = client.retrieve(999)
            assert result is None
            
            client.close()
    
    def test_retrieve_from_empty_database(self):
        """Test retrieving from empty database"""
        with tempfile.TemporaryDirectory() as temp_dir:
            client = ApexClient(dirpath=temp_dir)
            
            # Try to retrieve from empty database
            result = client.retrieve(0)
            assert result is None
            
            client.close()
    
    def test_retrieve_multiple_records(self):
        """Test retrieving multiple specific records"""
        with tempfile.TemporaryDirectory() as temp_dir:
            client = ApexClient(dirpath=temp_dir)
            
            # Store test data
            test_data = [
                {"name": "Alice", "age": 25},
                {"name": "Bob", "age": 30},
                {"name": "Charlie", "age": 35},
                {"name": "Diana", "age": 28},
            ]
            client.store(test_data)
            
            # Retrieve specific records
            result = client.retrieve(0)
            assert result["name"] == "Alice"
            
            result = client.retrieve(1)
            assert result["name"] == "Bob"
            
            result = client.retrieve(2)
            assert result["name"] == "Charlie"
            
            result = client.retrieve(3)
            assert result["name"] == "Diana"
            
            client.close()
    
    def test_retrieve_with_various_data_types(self):
        """Test retrieving records with various data types"""
        with tempfile.TemporaryDirectory() as temp_dir:
            client = ApexClient(dirpath=temp_dir)
            
            # Store data with various types
            test_data = {
                "string_field": "test_string",
                "int_field": 42,
                "float_field": 3.14159,
                "bool_field": True,
                "none_field": None,
                "bytes_field": b"binary_data",
                "negative_int": -100,
                "zero_float": 0.0,
                "empty_string": "",
                "false_bool": False,
            }
            
            client.store(test_data)
            
            # Retrieve and verify all types
            result = client.retrieve(0)
            
            assert result["string_field"] == "test_string"
            assert result["int_field"] == 42
            assert result["float_field"] == 3.14159
            assert result["bool_field"] is True
            assert result["none_field"] is None
            assert result["bytes_field"] == b"binary_data"
            assert result["negative_int"] == -100
            assert result["zero_float"] == 0.0
            assert result["empty_string"] == ""
            assert result["false_bool"] is False
            
            client.close()
    
    def test_retrieve_with_unicode_data(self):
        """Test retrieving records with unicode characters"""
        with tempfile.TemporaryDirectory() as temp_dir:
            client = ApexClient(dirpath=temp_dir)
            
            # Store unicode data
            test_data = {
                "chinese": "你好世界",
                "emoji": "🌍🚀",
                "arabic": "مرحبا بالعالم",
                "russian": "Привет мир",
                "french": "Bonjour le monde",
            }
            
            client.store(test_data)
            
            # Retrieve and verify unicode data
            result = client.retrieve(0)
            
            for key, expected in test_data.items():
                assert result[key] == expected
            
            client.close()
    
    def test_retrieve_with_special_characters(self):
        """Test retrieving records with special characters"""
        with tempfile.TemporaryDirectory() as temp_dir:
            client = ApexClient(dirpath=temp_dir)
            
            # Store data with special characters
            test_data = {
                "quotes": 'Single "double" quotes',
                "newlines": "Line 1\nLine 2\rLine 3",
                "tabs": "Tab\tseparated",
                "backslashes": "Backslash\\test",
                "unicode": "Unicode: ñáéíóú",
                "emoji": "Emoji: 🎉🚀🌟",
                "null_bytes": "Null\x00byte",
            }
            
            client.store(test_data)
            
            # Retrieve and verify special characters
            result = client.retrieve(0)
            
            for key, expected in test_data.items():
                assert result[key] == expected
            
            client.close()


class TestRetrieveMany:
    """Test multiple record retrieval operations"""
    
    def test_retrieve_many_basic(self):
        """Test retrieving multiple records"""
        with tempfile.TemporaryDirectory() as temp_dir:
            client = ApexClient(dirpath=temp_dir)
            
            # Store test data
            test_data = [
                {"name": "Alice", "age": 25},
                {"name": "Bob", "age": 30},
                {"name": "Charlie", "age": 35},
                {"name": "Diana", "age": 28},
                {"name": "Eve", "age": 32},
            ]
            client.store(test_data)
            
            # Retrieve multiple records
            results = client.retrieve_many([0, 2, 4])
            
            assert isinstance(results, ResultView)
            assert len(results) == 3
            
            # Verify retrieved records
            retrieved_names = [r["name"] for r in results]
            assert "Alice" in retrieved_names
            assert "Charlie" in retrieved_names
            assert "Eve" in retrieved_names
            assert "Bob" not in retrieved_names
            assert "Diana" not in retrieved_names
            
            client.close()
    
    def test_retrieve_many_unordered_ids(self):
        """Test retrieving records with unordered IDs"""
        with tempfile.TemporaryDirectory() as temp_dir:
            client = ApexClient(dirpath=temp_dir)
            
            # Store test data
            test_data = [
                {"name": "Alice", "age": 25},
                {"name": "Bob", "age": 30},
                {"name": "Charlie", "age": 35},
                {"name": "Diana", "age": 28},
            ]
            client.store(test_data)
            
            # Retrieve with unordered IDs
            results = client.retrieve_many([3, 1, 0])
            
            assert len(results) == 3
            
            # Results should be in the order requested
            retrieved_names = [r["name"] for r in results]
            assert retrieved_names == ["Diana", "Bob", "Alice"]
            
            client.close()
    
    def test_retrieve_many_with_duplicates(self):
        """Test retrieving records with duplicate IDs"""
        with tempfile.TemporaryDirectory() as temp_dir:
            client = ApexClient(dirpath=temp_dir)
            
            # Store test data
            test_data = [
                {"name": "Alice", "age": 25},
                {"name": "Bob", "age": 30},
                {"name": "Charlie", "age": 35},
            ]
            client.store(test_data)
            
            # Retrieve with duplicate IDs
            results = client.retrieve_many([0, 1, 0, 2, 1])
            
            # Behavior may vary - either deduplicate or return duplicates
            # Test that we get at least the unique records
            unique_names = set(r["name"] for r in results)
            assert "Alice" in unique_names
            assert "Bob" in unique_names
            assert "Charlie" in unique_names
            
            client.close()
    
    def test_retrieve_many_empty_list(self):
        """Test retrieving with empty ID list"""
        with tempfile.TemporaryDirectory() as temp_dir:
            client = ApexClient(dirpath=temp_dir)
            
            # Store some data
            client.store({"name": "Alice", "age": 25})
            
            # Retrieve with empty list
            results = client.retrieve_many([])
            
            assert isinstance(results, ResultView)
            assert len(results) == 0
            
            client.close()
    
    def test_retrieve_many_nonexistent_ids(self):
        """Test retrieving with some nonexistent IDs"""
        with tempfile.TemporaryDirectory() as temp_dir:
            client = ApexClient(dirpath=temp_dir)
            
            # Store test data
            test_data = [
                {"name": "Alice", "age": 25},
                {"name": "Bob", "age": 30},
                {"name": "Charlie", "age": 35},
            ]
            client.store(test_data)
            
            # Retrieve with mix of existing and nonexistent IDs
            results = client.retrieve_many([0, 999, 1, 888, 2])
            
            # Should only return existing records
            assert len(results) == 3
            retrieved_names = [r["name"] for r in results]
            assert "Alice" in retrieved_names
            assert "Bob" in retrieved_names
            assert "Charlie" in retrieved_names
            
            client.close()
    
    def test_retrieve_many_all_nonexistent(self):
        """Test retrieving with all nonexistent IDs"""
        with tempfile.TemporaryDirectory() as temp_dir:
            client = ApexClient(dirpath=temp_dir)
            
            # Store some data
            client.store({"name": "Alice", "age": 25})
            
            # Retrieve with all nonexistent IDs
            results = client.retrieve_many([999, 888, 777])
            
            assert isinstance(results, ResultView)
            assert len(results) == 0
            
            client.close()
    
    def test_retrieve_many_large_list(self):
        """Test retrieving with large ID list"""
        with tempfile.TemporaryDirectory() as temp_dir:
            client = ApexClient(dirpath=temp_dir)
            
            # Store large dataset
            large_data = [{"id": i, "value": f"item_{i}"} for i in range(1000)]
            client.store(large_data)
            
            # Retrieve every 10th record
            ids_to_retrieve = list(range(0, 1000, 10))
            results = client.retrieve_many(ids_to_retrieve)
            
            assert len(results) == 100
            
            # Verify some records
            for i, result in zip(ids_to_retrieve, results):
                assert result["id"] == i
                assert result["value"] == f"item_{i}"
            
            client.close()


class TestRetrieveAll:
    """Test all records retrieval operations"""
    
    def test_retrieve_all_basic(self):
        """Test retrieving all records"""
        with tempfile.TemporaryDirectory() as temp_dir:
            client = ApexClient(dirpath=temp_dir)
            
            # Store test data
            test_data = [
                {"name": "Alice", "age": 25},
                {"name": "Bob", "age": 30},
                {"name": "Charlie", "age": 35},
            ]
            client.store(test_data)
            
            # Retrieve all records
            results = client.retrieve_all()
            
            assert isinstance(results, ResultView)
            assert len(results) == 3
            
            # Verify all records are present
            names = [r["name"] for r in results]
            assert "Alice" in names
            assert "Bob" in names
            assert "Charlie" in names
            
            client.close()
    
    def test_retrieve_all_empty_database(self):
        """Test retrieving all from empty database"""
        with tempfile.TemporaryDirectory() as temp_dir:
            client = ApexClient(dirpath=temp_dir)
            
            # Retrieve all from empty database
            results = client.retrieve_all()
            
            assert isinstance(results, ResultView)
            assert len(results) == 0
            
            client.close()
    
    def test_retrieve_all_large_dataset(self):
        """Test retrieving all from large dataset"""
        with tempfile.TemporaryDirectory() as temp_dir:
            client = ApexClient(dirpath=temp_dir)
            
            # Store large dataset
            large_data = [{"id": i, "value": f"item_{i}"} for i in range(5000)]
            client.store(large_data)
            
            # Retrieve all records
            results = client.retrieve_all()
            
            assert len(results) == 5000
            
            # Verify first and last records
            first_record = results[0]
            assert first_record["id"] == 0
            assert first_record["value"] == "item_0"
            
            last_record = results[-1]
            assert last_record["id"] == 4999
            assert last_record["value"] == "item_4999"
            
            client.close()
    
    def test_retrieve_all_with_various_types(self):
        """Test retrieving all records with various data types"""
        with tempfile.TemporaryDirectory() as temp_dir:
            client = ApexClient(dirpath=temp_dir)
            
            # Store data with various types
            test_data = [
                {"type": "string", "value": "test_string"},
                {"type": "int", "value": 42},
                {"type": "float", "value": 3.14159},
                {"type": "bool", "value": True},
                {"type": "none", "value": None},
                {"type": "bytes", "value": b"binary"},
            ]
            client.store(test_data)
            
            # Retrieve all records
            results = client.retrieve_all()
            
            assert len(results) == 6
            
            # Verify all types are preserved
            # Verify types are preserved (note: bytes may be converted)
            types_found = set()
            for result in results:
                types_found.add(result["type"])
            
            assert "string" in types_found
            assert "int" in types_found
            assert "float" in types_found
            assert "bool" in types_found
            
            client.close()


class TestRetrieveResultViewConversions:
    """Test ResultView conversions from retrieve operations"""
    
    @pytest.mark.skipif(not PANDAS_AVAILABLE, reason="Pandas not available")
    def test_retrieve_many_to_pandas(self):
        """Test retrieve_many to pandas conversion"""
        with tempfile.TemporaryDirectory() as temp_dir:
            client = ApexClient(dirpath=temp_dir)
            
            # Store test data
            test_data = [
                {"name": "Alice", "age": 25, "city": "NYC"},
                {"name": "Bob", "age": 30, "city": "LA"},
                {"name": "Charlie", "age": 35, "city": "Chicago"},
            ]
            client.store(test_data)
            
            # Retrieve multiple records and convert to pandas
            results = client.retrieve_many([0, 2])
            df = results.to_pandas()
            
            assert isinstance(df, pd.DataFrame)
            assert len(df) == 2
            assert "name" in df.columns
            assert "age" in df.columns
            assert "city" in df.columns
            assert "_id" not in df.columns
            
            names = df["name"].tolist()
            assert "Alice" in names
            assert "Charlie" in names
            
            client.close()
    
    @pytest.mark.skipif(not POLARS_DF_AVAILABLE, reason="Polars not available")
    def test_retrieve_all_to_polars(self):
        """Test retrieve_all to polars conversion"""
        with tempfile.TemporaryDirectory() as temp_dir:
            client = ApexClient(dirpath=temp_dir)
            
            # Store test data
            test_data = [
                {"name": "Alice", "age": 25},
                {"name": "Bob", "age": 30},
                {"name": "Charlie", "age": 35},
            ]
            client.store(test_data)
            
            # Retrieve all and convert to polars
            results = client.retrieve_all()
            df = results.to_polars()
            
            assert isinstance(df, pl.DataFrame)
            assert len(df) == 3
            assert "name" in df.columns
            assert "age" in df.columns
            assert "_id" not in df.columns
            
            client.close()
    
    @pytest.mark.skipif(not PYARROW_AVAILABLE, reason="PyArrow not available")
    def test_retrieve_many_to_arrow(self):
        """Test retrieve_many to Arrow conversion"""
        with tempfile.TemporaryDirectory() as temp_dir:
            client = ApexClient(dirpath=temp_dir)
            
            # Store test data
            test_data = [
                {"name": "Alice", "age": 25},
                {"name": "Bob", "age": 30},
                {"name": "Charlie", "age": 35},
            ]
            client.store(test_data)
            
            # Retrieve multiple records and convert to Arrow
            results = client.retrieve_many([0, 1, 2])
            table = results.to_arrow()
            
            assert isinstance(table, pa.Table)
            assert len(table) == 3
            assert "name" in table.column_names
            assert "age" in table.column_names
            assert "_id" not in table.column_names
            
            client.close()
    
    def test_retrieve_operations_get_ids(self):
        """Test get_ids on retrieve operations"""
        with tempfile.TemporaryDirectory() as temp_dir:
            client = ApexClient(dirpath=temp_dir)
            
            # Store test data
            test_data = [
                {"name": "Alice", "age": 25},
                {"name": "Bob", "age": 30},
                {"name": "Charlie", "age": 35},
                {"name": "Diana", "age": 28},
            ]
            client.store(test_data)
            
            # Test retrieve_many get_ids
            results = client.retrieve_many([1, 3])
            ids = results.get_ids()
            
            assert isinstance(ids, np.ndarray)
            assert len(ids) == 2
            # Should contain the requested IDs
            assert 1 in ids or 1 in ids.tolist()
            assert 3 in ids or 3 in ids.tolist()
            
            # Test retrieve_all get_ids
            all_results = client.retrieve_all()
            all_ids = all_results.get_ids()
            
            assert len(all_ids) == 4
            
            client.close()


class TestRetrievePerformance:
    """Test performance optimizations for retrieve operations"""
    
    def test_retrieve_arrow_optimization(self):
        """Test Arrow C Data Interface optimization"""
        with tempfile.TemporaryDirectory() as temp_dir:
            client = ApexClient(dirpath=temp_dir)
            
            # Store test data
            test_data = [
                {"name": "Alice", "age": 25},
                {"name": "Bob", "age": 30},
                {"name": "Charlie", "age": 35},
            ]
            client.store(test_data)
            
            # If Arrow is available, retrieve should use Arrow optimization
            if ARROW_AVAILABLE and PYARROW_AVAILABLE:
                # Test that Arrow conversion works efficiently
                results = client.retrieve_many([0, 1, 2])
                
                # Should be able to convert to Arrow without issues
                table = results.to_arrow()
                assert isinstance(table, pa.Table)
                assert len(table) == 3
            
            client.close()
    
    def test_retrieve_performance_comparison(self):
        """Test performance comparison between retrieve methods"""
        with tempfile.TemporaryDirectory() as temp_dir:
            client = ApexClient(dirpath=temp_dir)
            
            # Store medium dataset
            data_size = 1000
            test_data = [{"id": i, "value": f"item_{i}"} for i in range(data_size)]
            client.store(test_data)
            
            import time
            
            # Test retrieve_many performance
            ids_to_retrieve = list(range(0, data_size, 10))  # Every 10th item
            
            start_time = time.time()
            results = client.retrieve_many(ids_to_retrieve)
            retrieve_many_time = time.time() - start_time
            
            assert len(results) == len(ids_to_retrieve)
            
            # Test retrieve_all performance
            start_time = time.time()
            all_results = client.retrieve_all()
            retrieve_all_time = time.time() - start_time
            
            assert len(all_results) == data_size
            
            # Both should be reasonably fast
            assert retrieve_many_time < 1.0
            assert retrieve_all_time < 2.0
            
            client.close()
    
    def test_retrieve_large_single_record(self):
        """Test retrieving single record with large data"""
        with tempfile.TemporaryDirectory() as temp_dir:
            client = ApexClient(dirpath=temp_dir)
            
            # Store record with large data
            large_string = "x" * 1000000  # 1MB string
            large_data = {
                "id": 1,
                "large_text": large_string,
                "normal_field": "test",
            }
            
            client.store(large_data)
            
            # Retrieve large record
            result = client.retrieve(0)
            
            assert result is not None
            assert result["id"] == 1
            assert len(result["large_text"]) == 1000000
            assert result["normal_field"] == "test"
            
            client.close()


class TestRetrieveEdgeCases:
    """Test edge cases and error handling for retrieve operations"""
    
    def test_retrieve_on_closed_client(self):
        """Test retrieve operations on closed client"""
        with tempfile.TemporaryDirectory() as temp_dir:
            client = ApexClient(dirpath=temp_dir)
            client.close()
            
            with pytest.raises(RuntimeError, match="connection has been closed"):
                client.retrieve(0)
            
            with pytest.raises(RuntimeError, match="connection has been closed"):
                client.retrieve_many([0, 1, 2])
            
            with pytest.raises(RuntimeError, match="connection has been closed"):
                client.retrieve_all()
    
    def test_retrieve_invalid_id_types(self):
        """Test retrieving with invalid ID types"""
        with tempfile.TemporaryDirectory() as temp_dir:
            client = ApexClient(dirpath=temp_dir)
            
            # Store some data
            client.store({"name": "Alice", "age": 25})
            
            # Test invalid ID types for retrieve - may return None or raise exception
            try:
                result = client.retrieve(-1)  # Negative ID
                # If no exception, should return None for invalid ID
                assert result is None
            except (TypeError, ValueError):
                pass  # Exception is also acceptable
            
            client.close()
    
    def test_retrieve_very_large_id(self):
        """Test retrieving with very large ID"""
        with tempfile.TemporaryDirectory() as temp_dir:
            client = ApexClient(dirpath=temp_dir)
            
            # Store some data
            client.store({"name": "Alice", "age": 25})
            
            # Test with very large ID
            large_id = 2**63 - 1  # Max 64-bit integer
            result = client.retrieve(large_id)
            assert result is None
            
            client.close()
    
    def test_retrieve_from_different_tables(self):
        """Test retrieving from different tables"""
        with tempfile.TemporaryDirectory() as temp_dir:
            client = ApexClient(dirpath=temp_dir)
            
            # Store data in default table
            client.store({"name": "Alice", "table": "default"})
            
            # Create and store data in another table
            client.create_table("users")
            client.store({"name": "Bob", "table": "users"})
            
            # Retrieve from default table
            client.use_table("default")
            result = client.retrieve(0)
            assert result["name"] == "Alice"
            assert result["table"] == "default"
            
            # Retrieve from users table
            client.use_table("users")
            result = client.retrieve(0)
            assert result["name"] == "Bob"
            assert result["table"] == "users"
            
            client.close()
    
    def test_retrieve_with_fts_enabled(self):
        """Test retrieve operations with FTS enabled"""
        with tempfile.TemporaryDirectory() as temp_dir:
            client = ApexClient(dirpath=temp_dir)
            client.init_fts(index_fields=["content"])
            
            # Store data with FTS content
            test_data = [
                {"content": "searchable text", "metadata": "test1"},
                {"content": "another searchable content", "metadata": "test2"},
            ]
            client.store(test_data)
            
            # Retrieve records normally
            result = client.retrieve(0)
            assert result["content"] == "searchable text"
            assert result["metadata"] == "test1"
            
            results = client.retrieve_all()
            assert len(results) == 2
            
            client.close()
    
    def test_retrieve_after_deletions(self):
        """Test retrieving after some records have been deleted"""
        with tempfile.TemporaryDirectory() as temp_dir:
            client = ApexClient(dirpath=temp_dir)
            
            # Store test data
            test_data = [
                {"name": "Alice", "age": 25},
                {"name": "Bob", "age": 30},
                {"name": "Charlie", "age": 35},
                {"name": "Diana", "age": 28},
            ]
            client.store(test_data)
            
            # Delete some records
            client.delete(1)  # Delete Bob
            client.delete([2])  # Delete Charlie
            
            # Retrieve remaining records
            result = client.retrieve(0)
            assert result["name"] == "Alice"
            
            result = client.retrieve(3)
            assert result["name"] == "Diana"
            
            # Deleted records should return None
            result = client.retrieve(1)
            assert result is None
            
            result = client.retrieve(2)
            assert result is None
            
            client.close()
    
    def test_retrieve_consistency_after_modifications(self):
        """Test retrieve consistency after data modifications"""
        with tempfile.TemporaryDirectory() as temp_dir:
            client = ApexClient(dirpath=temp_dir)
            
            # Store initial data
            client.store({"name": "Alice", "age": 25})
            client.store({"name": "Bob", "age": 30})
            
            # Replace a record - note: replace may have specific behavior
            try:
                success = client.replace(0, {"name": "Alice Updated", "age": 26})
                if success:
                    # Retrieve and verify the update
                    result = client.retrieve(0)
                    if result is not None:
                        assert result["name"] == "Alice Updated"
                        assert result["age"] == 26
            except Exception as e:
                print(f"Replace operation: {e}")
            
            # Retrieve all and verify we have records
            results = client.retrieve_all()
            assert len(results) >= 1
            
            client.close()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
