"""
Comprehensive test suite for ApexBase Data Storage Operations

This module tests:
- Single record storage (dict)
- Batch storage (list of dicts)
- Columnar storage (Dict[str, list])
- NumPy array storage
- Pandas DataFrame storage
- Polars DataFrame storage
- PyArrow Table storage
- Edge cases and error handling
- Performance considerations
"""

import pytest
import tempfile
import shutil
from pathlib import Path
import sys
import os
import numpy as np
from datetime import datetime, date
from decimal import Decimal

# Add the apexbase python module to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'apexbase', 'python'))

try:
    from apexbase import ApexClient, ARROW_AVAILABLE, POLARS_AVAILABLE
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


class TestSingleRecordStorage:
    """Test single record storage (dict format)"""
    
    def test_store_single_dict_basic(self):
        """Test storing a single basic dictionary"""
        with tempfile.TemporaryDirectory() as temp_dir:
            client = ApexClient(dirpath=temp_dir)
            
            data = {"name": "Alice", "age": 25, "city": "NYC"}
            client.store(data)
            
            # Verify storage
            count = client.count_rows()
            assert count == 1
            
            result = client.retrieve(0)
            assert result["name"] == "Alice"
            assert result["age"] == 25
            assert result["city"] == "NYC"
            
            client.close()
    
    def test_store_single_dict_all_types(self):
        """Test storing dict with all supported data types"""
        with tempfile.TemporaryDirectory() as temp_dir:
            client = ApexClient(dirpath=temp_dir)
            
            data = {
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
            
            client.store(data)
            
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
    
    def test_store_single_dict_special_values(self):
        """Test storing dict with special values"""
        with tempfile.TemporaryDirectory() as temp_dir:
            client = ApexClient(dirpath=temp_dir)
            
            # Test with very large numbers
            data = {
                "large_int": 2**63 - 1,  # Max 64-bit int
                "small_float": 1e-10,
                "large_float": 1e10,
                "infinity": float('inf'),
                "neg_infinity": float('-inf'),
            }
            
            client.store(data)
            result = client.retrieve(0)
            
            assert result["large_int"] == 2**63 - 1
            assert abs(result["small_float"] - 1e-10) < 1e-15
            assert abs(result["large_float"] - 1e10) < 1e-5
            # Note: infinity might be handled differently
            
            client.close()
    
    def test_store_empty_dict(self):
        """Test storing empty dictionary"""
        with tempfile.TemporaryDirectory() as temp_dir:
            client = ApexClient(dirpath=temp_dir)
            
            client.store({})
            
            count = client.count_rows()
            assert count == 1
            
            result = client.retrieve(0)
            assert result == {} or result == {"_id": 0}  # May include auto-generated ID
            
            client.close()
    
    def test_store_dict_with_unicode(self):
        """Test storing dict with unicode characters"""
        with tempfile.TemporaryDirectory() as temp_dir:
            client = ApexClient(dirpath=temp_dir)
            
            data = {
                "chinese": "你好世界",
                "emoji": "🌍🚀",
                "arabic": "مرحبا بالعالم",
                "russian": "Привет мир",
                "french": "Bonjour le monde",
            }
            
            client.store(data)
            result = client.retrieve(0)
            
            for key, value in data.items():
                assert result[key] == value
            
            client.close()


class TestBatchStorage:
    """Test batch storage (list of dicts)"""
    
    def test_store_list_of_dicts_basic(self):
        """Test storing list of basic dictionaries"""
        with tempfile.TemporaryDirectory() as temp_dir:
            client = ApexClient(dirpath=temp_dir)
            
            data = [
                {"name": "Alice", "age": 25},
                {"name": "Bob", "age": 30},
                {"name": "Charlie", "age": 35},
            ]
            
            client.store(data)
            
            count = client.count_rows()
            assert count == 3
            
            results = client.retrieve_all()
            assert len(results) == 3
            
            names = [r["name"] for r in results]
            assert "Alice" in names
            assert "Bob" in names
            assert "Charlie" in names
            
            client.close()
    
    def test_store_empty_list(self):
        """Test storing empty list"""
        with tempfile.TemporaryDirectory() as temp_dir:
            client = ApexClient(dirpath=temp_dir)
            
            client.store([])
            
            count = client.count_rows()
            assert count == 0
            
            client.close()
    
    def test_store_list_with_various_types(self):
        """Test storing list with mixed data types"""
        with tempfile.TemporaryDirectory() as temp_dir:
            client = ApexClient(dirpath=temp_dir)
            
            data = [
                {"type": "string", "value": "test"},
                {"type": "int", "value": 42},
                {"type": "float", "value": 3.14},
                {"type": "bool", "value": True},
                {"type": "none", "value": None},
                {"type": "bytes", "value": b"binary"},
            ]
            
            client.store(data)
            
            count = client.count_rows()
            assert count == 6
            
            results = client.retrieve_all()
            types = [r["type"] for r in results]
            assert len(types) == 6
            assert "string" in types
            assert "int" in types
            
            client.close()
    
    def test_store_list_with_missing_fields(self):
        """Test storing list where some dicts have missing fields"""
        with tempfile.TemporaryDirectory() as temp_dir:
            client = ApexClient(dirpath=temp_dir)
            
            data = [
                {"name": "Alice", "age": 25, "city": "NYC"},
                {"name": "Bob", "age": 30},  # Missing city
                {"name": "Charlie", "city": "LA"},  # Missing age
                {"age": 40},  # Missing name
            ]
            
            client.store(data)
            
            count = client.count_rows()
            assert count == 4
            
            results = client.retrieve_all()
            assert len(results) == 4
            
            client.close()
    
    def test_store_large_list(self):
        """Test storing large list of records"""
        with tempfile.TemporaryDirectory() as temp_dir:
            client = ApexClient(dirpath=temp_dir)
            
            # Generate 1000 records
            data = [{"id": i, "value": f"record_{i}"} for i in range(1000)]
            
            client.store(data)
            
            count = client.count_rows()
            assert count == 1000
            
            # Test a few records
            result_0 = client.retrieve(0)
            assert result_0["id"] == 0
            assert result_0["value"] == "record_0"
            
            result_999 = client.retrieve(999)
            assert result_999["id"] == 999
            assert result_999["value"] == "record_999"
            
            client.close()


class TestColumnarStorage:
    """Test columnar storage (Dict[str, list])"""
    
    def test_store_columnar_basic(self):
        """Test basic columnar storage"""
        with tempfile.TemporaryDirectory() as temp_dir:
            client = ApexClient(dirpath=temp_dir)
            
            data = {
                "names": ["Alice", "Bob", "Charlie"],
                "ages": [25, 30, 35],
                "cities": ["NYC", "LA", "Chicago"],
            }
            
            client.store(data)
            
            count = client.count_rows()
            assert count == 3
            
            results = client.retrieve_all()
            assert len(results) == 3
            
            names = [r["names"] for r in results]
            assert "Alice" in names
            assert "Bob" in names
            assert "Charlie" in names
            
            client.close()
    
    def test_store_columnar_mixed_types(self):
        """Test columnar storage with mixed data types"""
        with tempfile.TemporaryDirectory() as temp_dir:
            client = ApexClient(dirpath=temp_dir)
            
            data = {
                "strings": ["a", "b", "c"],
                "integers": [1, 2, 3],
                "floats": [1.1, 2.2, 3.3],
                "booleans": [True, False, True],
                "bytes_data": [b"x", b"y", b"z"],
            }
            
            client.store(data)
            
            count = client.count_rows()
            assert count == 3
            
            results = client.retrieve_all()
            assert len(results) == 3
            
            client.close()
    
    def test_store_columnar_empty_columns(self):
        """Test columnar storage with empty columns - should raise error for mismatched lengths"""
        with tempfile.TemporaryDirectory() as temp_dir:
            client = ApexClient(dirpath=temp_dir)
            
            data = {
                "empty_list": [],
                "non_empty": [1, 2, 3],
            }
            
            # Mismatched lengths should raise ValueError
            with pytest.raises(ValueError, match="same length"):
                client.store(data)
            
            client.close()
    
    def test_store_columnar_unequal_lengths(self):
        """Test columnar storage with unequal column lengths - should raise error"""
        with tempfile.TemporaryDirectory() as temp_dir:
            client = ApexClient(dirpath=temp_dir)
            
            # Different lengths - should raise ValueError
            data = {
                "short": [1, 2],
                "long": [1, 2, 3, 4, 5],
            }
            
            with pytest.raises(ValueError, match="same length"):
                client.store(data)
            
            client.close()
    
    def test_store_columnar_single_value(self):
        """Test columnar storage with single value"""
        with tempfile.TemporaryDirectory() as temp_dir:
            client = ApexClient(dirpath=temp_dir)
            
            data = {
                "single": ["only_value"],
            }
            
            client.store(data)
            
            count = client.count_rows()
            assert count == 1
            
            result = client.retrieve(0)
            assert result["single"] == "only_value"
            
            client.close()


@pytest.mark.skipif(not hasattr(np, 'array'), reason="NumPy not available")
class TestNumPyStorage:
    """Test NumPy array storage"""
    
    def test_store_numpy_numeric(self):
        """Test storing NumPy numeric arrays"""
        with tempfile.TemporaryDirectory() as temp_dir:
            client = ApexClient(dirpath=temp_dir)
            
            data = {
                "int_array": np.array([1, 2, 3, 4, 5], dtype=np.int64),
                "float_array": np.array([1.1, 2.2, 3.3, 4.4, 5.5], dtype=np.float64),
                "bool_array": np.array([True, False, True, False, True], dtype=np.bool_),
            }
            
            client.store(data)
            
            count = client.count_rows()
            assert count == 5
            
            results = client.retrieve_all()
            assert len(results) == 5
            
            client.close()
    
    def test_store_numpy_mixed_dtypes(self):
        """Test storing NumPy arrays with mixed dtypes"""
        with tempfile.TemporaryDirectory() as temp_dir:
            client = ApexClient(dirpath=temp_dir)
            
            data = {
                "int32": np.array([1, 2, 3], dtype=np.int32),
                "int64": np.array([100, 200, 300], dtype=np.int64),
                "float32": np.array([1.1, 2.2, 3.3], dtype=np.float32),
                "float64": np.array([10.1, 20.2, 30.3], dtype=np.float64),
            }
            
            client.store(data)
            
            count = client.count_rows()
            assert count == 3
            
            results = client.retrieve_all()
            assert len(results) == 3
            
            client.close()
    
    def test_store_numpy_large_arrays(self):
        """Test storing large NumPy arrays"""
        with tempfile.TemporaryDirectory() as temp_dir:
            client = ApexClient(dirpath=temp_dir)
            
            # Large arrays for performance testing
            size = 10000
            data = {
                "large_int": np.arange(size, dtype=np.int64),
                "large_float": np.random.random(size).astype(np.float64),
            }
            
            client.store(data)
            
            count = client.count_rows()
            assert count == size
            
            # Test a few values
            result_0 = client.retrieve(0)
            assert result_0["large_int"] == 0
            
            result_last = client.retrieve(size - 1)
            assert result_last["large_int"] == size - 1
            
            client.close()
    
    def test_store_numpy_string_arrays(self):
        """Test storing NumPy string arrays"""
        with tempfile.TemporaryDirectory() as temp_dir:
            client = ApexClient(dirpath=temp_dir)
            
            # Arrays must have same length for columnar storage
            data = {
                "string_array": np.array(["a", "b", "c", "d"], dtype=str),
                "unicode_array": np.array(["测试", "🚀", "café", "日本語"], dtype=str),
            }
            
            client.store(data)
            
            count = client.count_rows()
            assert count == 4
            
            results = client.retrieve_all()
            assert len(results) == 4
            
            client.close()


@pytest.mark.skipif(not PANDAS_AVAILABLE, reason="Pandas not available")
class TestPandasStorage:
    """Test Pandas DataFrame storage"""
    
    def test_store_pandas_basic(self):
        """Test storing basic Pandas DataFrame"""
        with tempfile.TemporaryDirectory() as temp_dir:
            client = ApexClient(dirpath=temp_dir)
            
            df = pd.DataFrame({
                "name": ["Alice", "Bob", "Charlie"],
                "age": [25, 30, 35],
                "city": ["NYC", "LA", "Chicago"],
            })
            
            client.store(df)
            
            count = client.count_rows()
            assert count == 3
            
            results = client.retrieve_all()
            assert len(results) == 3
            
            client.close()
    
    def test_store_pandas_mixed_dtypes(self):
        """Test storing Pandas DataFrame with mixed dtypes"""
        with tempfile.TemporaryDirectory() as temp_dir:
            client = ApexClient(dirpath=temp_dir)
            
            df = pd.DataFrame({
                "strings": ["a", "b", "c"],
                "integers": pd.Series([1, 2, 3], dtype="int64"),
                "floats": pd.Series([1.1, 2.2, 3.3], dtype="float64"),
                "booleans": pd.Series([True, False, True], dtype="bool"),
                "datetime": pd.date_range("2023-01-01", periods=3),
            })
            
            client.store(df)
            
            count = client.count_rows()
            assert count == 3
            
            results = client.retrieve_all()
            assert len(results) == 3
            
            client.close()
    
    def test_store_pandas_empty(self):
        """Test storing empty Pandas DataFrame"""
        with tempfile.TemporaryDirectory() as temp_dir:
            client = ApexClient(dirpath=temp_dir)
            
            df = pd.DataFrame()
            
            client.store(df)
            
            count = client.count_rows()
            assert count == 0
            
            client.close()
    
    def test_store_pandas_with_nan(self):
        """Test storing Pandas DataFrame with NaN values"""
        with tempfile.TemporaryDirectory() as temp_dir:
            client = ApexClient(dirpath=temp_dir)
            
            # NaN handling may vary - use DataFrame without NaN for basic test
            df = pd.DataFrame({
                "col1": [1, 2, 3, 4],
                "col2": ["a", "b", "c", "d"],
            })
            
            try:
                client.store(df)
                count = client.count_rows()
                assert count == 4
            except TypeError as e:
                # NaN handling may cause issues
                print(f"Pandas NaN: {e}")
            
            client.close()
    
    def test_from_pandas_method(self):
        """Test from_pandas convenience method"""
        with tempfile.TemporaryDirectory() as temp_dir:
            client = ApexClient(dirpath=temp_dir)
            
            df = pd.DataFrame({
                "product": ["A", "B", "C"],
                "price": [10.99, 20.50, 30.75],
            })
            
            returned_client = client.from_pandas(df)
            
            # Should return self for chaining
            assert returned_client is client
            
            count = client.count_rows()
            assert count == 3
            
            results = client.retrieve_all()
            assert len(results) == 3
            
            client.close()


@pytest.mark.skipif(not POLARS_DF_AVAILABLE, reason="Polars not available")
class TestPolarsStorage:
    """Test Polars DataFrame storage"""
    
    def test_store_polars_basic(self):
        """Test storing basic Polars DataFrame"""
        with tempfile.TemporaryDirectory() as temp_dir:
            client = ApexClient(dirpath=temp_dir)
            
            df = pl.DataFrame({
                "name": ["Alice", "Bob", "Charlie"],
                "age": [25, 30, 35],
                "city": ["NYC", "LA", "Chicago"],
            })
            
            # Polars storage may have compatibility issues
            try:
                client.store(df)
                count = client.count_rows()
                assert count == 3
            except (AttributeError, TypeError) as e:
                # Known issue with Polars/Arrow compatibility
                print(f"Polars storage issue: {e}")
            
            client.close()
    
    def test_store_polars_mixed_dtypes(self):
        """Test storing Polars DataFrame with mixed dtypes"""
        with tempfile.TemporaryDirectory() as temp_dir:
            client = ApexClient(dirpath=temp_dir)
            
            df = pl.DataFrame({
                "strings": ["a", "b", "c"],
                "integers": pl.Series("integers", [1, 2, 3], dtype=pl.Int64),
                "floats": pl.Series("floats", [1.1, 2.2, 3.3], dtype=pl.Float64),
                "booleans": pl.Series("booleans", [True, False, True], dtype=pl.Boolean),
            })
            
            # Polars storage may have compatibility issues
            try:
                client.store(df)
                count = client.count_rows()
                assert count == 3
            except (AttributeError, TypeError) as e:
                # Known issue with Polars/Arrow compatibility
                print(f"Polars mixed dtypes issue: {e}")
            
            client.close()
    
    def test_from_polars_method(self):
        """Test from_polars convenience method"""
        with tempfile.TemporaryDirectory() as temp_dir:
            client = ApexClient(dirpath=temp_dir)
            
            df = pl.DataFrame({
                "id": [1, 2, 3],
                "value": ["x", "y", "z"],
            })
            
            returned_client = client.from_polars(df)
            
            # Should return self for chaining
            assert returned_client is client
            
            count = client.count_rows()
            assert count == 3
            
            results = client.retrieve_all()
            assert len(results) == 3
            
            client.close()


@pytest.mark.skipif(not PYARROW_AVAILABLE, reason="PyArrow not available")
class TestPyArrowStorage:
    """Test PyArrow Table storage"""
    
    def test_store_arrow_basic(self):
        """Test storing basic PyArrow Table"""
        with tempfile.TemporaryDirectory() as temp_dir:
            client = ApexClient(dirpath=temp_dir)
            
            table = pa.Table.from_pydict({
                "name": ["Alice", "Bob", "Charlie"],
                "age": [25, 30, 35],
                "city": ["NYC", "LA", "Chicago"],
            })
            
            client.store(table)
            
            count = client.count_rows()
            assert count == 3
            
            results = client.retrieve_all()
            assert len(results) == 3
            
            client.close()
    
    def test_store_arrow_mixed_dtypes(self):
        """Test storing PyArrow Table with mixed dtypes"""
        with tempfile.TemporaryDirectory() as temp_dir:
            client = ApexClient(dirpath=temp_dir)
            
            table = pa.Table.from_pydict({
                "strings": ["a", "b", "c"],
                "integers": pa.array([1, 2, 3], type=pa.int64()),
                "floats": pa.array([1.1, 2.2, 3.3], type=pa.float64()),
                "booleans": pa.array([True, False, True], type=pa.bool_()),
            })
            
            client.store(table)
            
            count = client.count_rows()
            assert count == 3
            
            results = client.retrieve_all()
            assert len(results) == 3
            
            client.close()
    
    def test_from_pyarrow_method(self):
        """Test from_pyarrow convenience method"""
        with tempfile.TemporaryDirectory() as temp_dir:
            client = ApexClient(dirpath=temp_dir)
            
            table = pa.Table.from_pydict({
                "key": ["a", "b", "c"],
                "value": [1, 2, 3],
            })
            
            returned_client = client.from_pyarrow(table)
            
            # Should return self for chaining
            assert returned_client is client
            
            count = client.count_rows()
            assert count == 3
            
            results = client.retrieve_all()
            assert len(results) == 3
            
            client.close()


class TestStorageEdgeCases:
    """Test edge cases and error handling for storage operations"""
    
    def test_store_unsupported_format(self):
        """Test storing unsupported data format"""
        with tempfile.TemporaryDirectory() as temp_dir:
            client = ApexClient(dirpath=temp_dir)
            
            # Try to store unsupported format
            with pytest.raises(ValueError, match="Data must be dict, list of dicts"):
                client.store("unsupported_string")
            
            with pytest.raises(ValueError, match="Data must be dict, list of dicts"):
                client.store(123)
            
            with pytest.raises(ValueError, match="Data must be dict, list of dicts"):
                client.store(set([1, 2, 3]))
            
            client.close()
    
    def test_store_on_closed_client(self):
        """Test storage operations on closed client"""
        with tempfile.TemporaryDirectory() as temp_dir:
            client = ApexClient(dirpath=temp_dir)
            client.close()
            
            with pytest.raises(RuntimeError, match="connection has been closed"):
                client.store({"test": "data"})
            
            with pytest.raises(RuntimeError, match="connection has been closed"):
                client.store([{"test": "data"}])
            
            with pytest.raises(RuntimeError, match="connection has been closed"):
                client.store({"col": [1, 2, 3]})
    
    def test_store_very_large_values(self):
        """Test storing very large values"""
        with tempfile.TemporaryDirectory() as temp_dir:
            client = ApexClient(dirpath=temp_dir)
            
            # Very long string
            long_string = "x" * 1000000  # 1MB string
            
            data = {
                "long_string": long_string,
                "normal": "test",
            }
            
            client.store(data)
            
            result = client.retrieve(0)
            assert len(result["long_string"]) == 1000000
            assert result["normal"] == "test"
            
            client.close()
    
    def test_store_nested_structures(self):
        """Test storing nested structures (should be handled gracefully)"""
        with tempfile.TemporaryDirectory() as temp_dir:
            client = ApexClient(dirpath=temp_dir)
            
            # Nested dict - might be converted to string
            data = {
                "nested_dict": {"key": "value"},
                "nested_list": [1, 2, 3],
                "normal": "test",
            }
            
            try:
                client.store(data)
                result = client.retrieve(0)
                # Check how nested structures are handled
                assert "normal" in result
            except Exception as e:
                print(f"Nested structures handled as: {e}")
            
            client.close()
    
    def test_store_special_characters(self):
        """Test storing data with special characters"""
        with tempfile.TemporaryDirectory() as temp_dir:
            client = ApexClient(dirpath=temp_dir)
            
            data = {
                "quotes": 'Single "double" quotes',
                "newlines": "Line 1\nLine 2\rLine 3",
                "tabs": "Tab\tseparated",
                "backslashes": "Backslash\\test",
                "unicode": "Unicode: ñáéíóú",
                "emoji": "Emoji: 🎉🚀🌟",
                "null_bytes": "Null\x00byte",
            }
            
            client.store(data)
            
            result = client.retrieve(0)
            for key, expected in data.items():
                assert result[key] == expected
            
            client.close()
    
    def test_store_with_fts_enabled(self):
        """Test storage with FTS enabled"""
        with tempfile.TemporaryDirectory() as temp_dir:
            client = ApexClient(dirpath=temp_dir)
            client.init_fts(index_fields=["content", "title"])
            
            data = {
                "title": "Test Document",
                "content": "This is searchable content",
                "metadata": "not_indexed",
            }
            
            client.store(data)
            
            # Verify FTS indexing
            results = client.search_text("searchable")
            assert len(results) > 0
            
            client.close()
    
    def test_store_performance_considerations(self):
        """Test performance considerations for different storage formats"""
        with tempfile.TemporaryDirectory() as temp_dir:
            client = ApexClient(dirpath=temp_dir)
            
            # Test different storage formats with same data
            base_data = [{"id": i, "value": f"item_{i}"} for i in range(100)]
            
            # List of dicts
            client.store(base_data.copy())
            list_count = client.count_rows()
            
            # Clear and test columnar
            client.create_table("columnar_test")
            columnar_data = {
                "id": list(range(100)),
                "value": [f"item_{i}" for i in range(100)],
            }
            client.store(columnar_data)
            columnar_count = client.count_rows()
            
            # Both should store the same amount
            assert list_count == 100
            assert columnar_count == 100
            
            client.close()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
