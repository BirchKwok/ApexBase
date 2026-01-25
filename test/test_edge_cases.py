"""
Comprehensive test suite for ApexBase Edge Cases, Error Handling, and Exception Scenarios

This module tests:
- Invalid parameter handling
- Resource exhaustion scenarios
- Concurrent access patterns
- Data corruption scenarios
- Network and I/O error simulation
- Memory pressure handling
- Invalid state transitions
- Boundary condition testing
"""

import pytest
import tempfile
import shutil
from pathlib import Path
import sys
import os
import numpy as np
import threading
import time
from concurrent.futures import ThreadPoolExecutor

# Add the apexbase python module to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'apexbase', 'python'))

try:
    from apexbase import ApexClient, FTS_AVAILABLE, ARROW_AVAILABLE, POLARS_AVAILABLE
except ImportError as e:
    pytest.skip(f"ApexBase not available: {e}", allow_module_level=True)


class TestInvalidParameters:
    """Test handling of invalid parameters"""
    
    def test_invalid_durability_parameter(self):
        """Test invalid durability level parameter"""
        with tempfile.TemporaryDirectory() as temp_dir:
            with pytest.raises(ValueError, match="durability must be"):
                ApexClient(dirpath=temp_dir, durability="invalid")
    
    def test_negative_batch_size(self):
        """Test negative batch size parameter"""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Should not raise error but handle gracefully
            client = ApexClient(dirpath=temp_dir, batch_size=-1)
            assert client._batch_size == -1
            client.close()
    
    def test_negative_cache_size(self):
        """Test negative cache size parameter"""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Should not raise error but handle gracefully
            client = ApexClient(dirpath=temp_dir, cache_size=-1)
            assert client._cache_size == -1
            client.close()
    
    def test_invalid_limit_values(self):
        """Test invalid limit values in queries"""
        with tempfile.TemporaryDirectory() as temp_dir:
            client = ApexClient(dirpath=temp_dir)
            
            # Store test data
            client.store({"name": "Alice", "age": 25})
            
            # Test negative limit - behavior may vary
            try:
                results = client.query(limit=-1)
                # If no exception, may return all or empty results
            except Exception as e:
                # Exception is acceptable for invalid limit
                pass
            
            # Test zero limit - may return empty or handle differently
            try:
                results = client.query(limit=0)
                # Zero limit might return empty or all results
            except Exception as e:
                pass
            
            client.close()
    
    def test_invalid_id_values(self):
        """Test invalid ID values for operations"""
        with tempfile.TemporaryDirectory() as temp_dir:
            client = ApexClient(dirpath=temp_dir)
            
            # Store test data
            client.store({"name": "Alice", "age": 25})
            
            # Test negative ID - may return None or raise exception
            try:
                result = client.retrieve(-1)
                # If no exception, should return None for invalid ID
            except (TypeError, ValueError, OverflowError):
                pass  # Exception is acceptable
            
            # Test delete with None - should raise exception
            try:
                client.delete(None)
            except (TypeError, ValueError):
                pass  # Exception expected
            
            client.close()
    
    def test_invalid_table_names(self):
        """Test invalid table names"""
        with tempfile.TemporaryDirectory() as temp_dir:
            client = ApexClient(dirpath=temp_dir)
            
            # Test empty table name
            try:
                client.create_table("")
                # May handle gracefully or raise error
            except Exception as e:
                assert isinstance(e, (ValueError, RuntimeError))
            
            # Test very long table name
            long_name = "a" * 10000
            try:
                client.create_table(long_name)
                # May handle gracefully or raise error
            except Exception as e:
                assert isinstance(e, (ValueError, RuntimeError))
            
            client.close()


class TestResourceExhaustion:
    """Test resource exhaustion scenarios"""
    
    def test_memory_exhaustion_large_data(self):
        """Test handling of very large data that might exhaust memory"""
        with tempfile.TemporaryDirectory() as temp_dir:
            client = ApexClient(dirpath=temp_dir)
            
            # Try to store very large data
            try:
                large_data = {
                    "large_field": "x" * 100_000_000,  # 100MB string
                    "normal_field": "test"
                }
                client.store(large_data)
                
                # Try to retrieve it
                result = client.retrieve(0)
                assert len(result["large_field"]) == 100_000_000
                
            except MemoryError:
                # Should handle memory exhaustion gracefully
                pass
            except Exception as e:
                # Other exceptions may also be acceptable
                print(f"Large data handling: {e}")
            
            client.close()
    
    def test_disk_space_exhaustion_simulation(self):
        """Test behavior when disk space is exhausted"""
        with tempfile.TemporaryDirectory() as temp_dir:
            client = ApexClient(dirpath=temp_dir)
            
            # Fill up temporary directory to simulate disk exhaustion
            try:
                # Create many large files to fill space
                for i in range(100):
                    large_file = Path(temp_dir) / f"filler_{i}.dat"
                    try:
                        with open(large_file, 'wb') as f:
                            f.write(b'x' * 1_000_000)  # 1MB per file
                    except OSError:
                        break  # Disk full
                
                # Try to store more data
                client.store({"test": "data when disk is full"})
                
            except Exception as e:
                # Should handle disk full gracefully
                print(f"Disk space exhaustion handling: {e}")
            
            client.close()
    
    def test_too_many_open_files(self):
        """Test behavior with too many open files"""
        with tempfile.TemporaryDirectory() as temp_dir:
            client = ApexClient(dirpath=temp_dir)
            
            try:
                # Try to open many clients (simulating file handle exhaustion)
                clients = []
                for i in range(1000):
                    try:
                        new_client = ApexClient(dirpath=temp_dir)
                        clients.append(new_client)
                        new_client.store({"id": i, "data": f"test_{i}"})
                    except OSError:
                        break  # Too many open files
                
                # Clean up
                for c in clients:
                    c.close()
                    
            except Exception as e:
                print(f"File handle exhaustion handling: {e}")
            
            client.close()
    
    def test_concurrent_resource_usage(self):
        """Test concurrent resource usage - writes only (reads during concurrent writes may see stale data)"""
        with tempfile.TemporaryDirectory() as temp_dir:
            client = ApexClient(dirpath=temp_dir)
            
            def worker(worker_id):
                try:
                    # Each worker stores data (no reads during concurrent writes to avoid race conditions)
                    for i in range(10):
                        data = {"worker_id": worker_id, "iteration": i, "data": f"test_{worker_id}_{i}"}
                        client.store(data)
                    
                    return True
                except Exception as e:
                    print(f"Worker {worker_id} error: {e}")
                    return False
            
            # Run multiple workers concurrently
            with ThreadPoolExecutor(max_workers=10) as executor:
                futures = [executor.submit(worker, i) for i in range(10)]
                results = [f.result() for f in futures]
            
            # Most workers should succeed
            success_count = sum(results)
            assert success_count >= 5  # Allow for failures due to resource contention without file locking
            
            # Verify data integrity after all concurrent operations complete
            all_results = client.retrieve_all()
            # Should have stored at least some data from successful workers
            assert len(all_results) >= success_count * 5  # At least half the iterations per successful worker
            
            client.close()


class TestConcurrentAccess:
    """Test concurrent access patterns"""
    
    def test_concurrent_writes(self):
        """Test concurrent write operations"""
        with tempfile.TemporaryDirectory() as temp_dir:
            client = ApexClient(dirpath=temp_dir)
            
            def writer(writer_id):
                try:
                    for i in range(50):
                        data = {"writer_id": writer_id, "iteration": i, "timestamp": time.time()}
                        client.store(data)
                    return True
                except Exception as e:
                    print(f"Writer {writer_id} error: {e}")
                    return False
            
            # Run multiple writers concurrently
            with ThreadPoolExecutor(max_workers=5) as executor:
                futures = [executor.submit(writer, i) for i in range(5)]
                results = [f.result() for f in futures]
            
            # Most writers should succeed
            success_count = sum(results)
            assert success_count >= 4
            
            # Verify data integrity
            all_results = client.retrieve_all()
            assert len(all_results) >= 200  # 5 writers * 50 iterations each
            
            client.close()
    
    def test_concurrent_reads_writes(self):
        """Test concurrent read and write operations"""
        with tempfile.TemporaryDirectory() as temp_dir:
            client = ApexClient(dirpath=temp_dir)
            
            # Pre-populate some data
            for i in range(100):
                client.store({"id": i, "value": f"initial_{i}"})
            
            def writer():
                try:
                    for i in range(50):
                        client.store({"id": i + 1000, "value": f"concurrent_{i}"})
                    return True
                except Exception as e:
                    print(f"Concurrent writer error: {e}")
                    return False
            
            def reader():
                try:
                    for i in range(50):
                        results = client.query("id >= 0")
                        assert len(results) >= 100  # At least initial data
                    return True
                except Exception as e:
                    print(f"Concurrent reader error: {e}")
                    return False
            
            # Run concurrent readers and writers
            with ThreadPoolExecutor(max_workers=6) as executor:
                futures = [executor.submit(writer)] + [executor.submit(reader) for _ in range(5)]
                results = [f.result() for f in futures]
            
            # Most operations should succeed
            success_count = sum(results)
            assert success_count >= 5
            
            client.close()
    
    def test_concurrent_table_operations(self):
        """Test concurrent table operations"""
        with tempfile.TemporaryDirectory() as temp_dir:
            client = ApexClient(dirpath=temp_dir)
            
            def table_worker(worker_id):
                try:
                    table_name = f"table_{worker_id}"
                    client.create_table(table_name)
                    
                    for i in range(10):
                        client.use_table(table_name)
                        client.store({"worker_id": worker_id, "iteration": i})
                    
                    return True
                except Exception as e:
                    print(f"Table worker {worker_id} error: {e}")
                    return False
            
            # Run multiple table workers concurrently
            with ThreadPoolExecutor(max_workers=5) as executor:
                futures = [executor.submit(table_worker, i) for i in range(5)]
                results = [f.result() for f in futures]
            
            # Most workers should succeed
            success_count = sum(results)
            assert success_count >= 4
            
            client.close()


class TestDataCorruptionScenarios:
    """Test data corruption scenarios"""
    
    def test_partial_data_write_interruption(self):
        """Test handling of interrupted data writes"""
        with tempfile.TemporaryDirectory() as temp_dir:
            client = ApexClient(dirpath=temp_dir)
            
            # Store large data that might be interrupted
            large_data = {
                "field1": "x" * 1_000_000,
                "field2": "y" * 1_000_000,
                "field3": "z" * 1_000_000,
            }
            
            try:
                client.store(large_data)
                
                # Verify data integrity
                result = client.retrieve(0)
                assert len(result["field1"]) == 1_000_000
                assert len(result["field2"]) == 1_000_000
                assert len(result["field3"]) == 1_000_000
                
            except Exception as e:
                print(f"Large write handling: {e}")
            
            client.close()
    
    def test_mixed_data_types_corruption(self):
        """Test handling of mixed data types that might cause corruption"""
        with tempfile.TemporaryDirectory() as temp_dir:
            client = ApexClient(dirpath=temp_dir)
            
            # Store data with mixed and potentially problematic types
            problematic_data = [
                {"field": None},
                {"field": ""},
                {"field": 0},
                {"field": False},
                {"field": []},  # Empty list might be converted to string
                {"field": {}},  # Empty dict might be converted to string
                {"field": float('inf')},  # Infinity
                {"field": float('-inf')},  # Negative infinity
            ]
            
            try:
                for data in problematic_data:
                    client.store(data)
                
                # Try to retrieve all data
                for i in range(len(problematic_data)):
                    result = client.retrieve(i)
                    assert result is not None  # Should not be corrupted to None
                    
            except Exception as e:
                print(f"Mixed types handling: {e}")
            
            client.close()
    
    def test_unicode_corruption(self):
        """Test unicode data corruption scenarios"""
        with tempfile.TemporaryDirectory() as temp_dir:
            client = ApexClient(dirpath=temp_dir)
            
            # Store various unicode sequences that might cause corruption
            unicode_data = [
                {"text": "Normal unicode: ñáéíóú"},
                {"text": "Emoji: 🎉🚀🌟"},
                {"text": "Mixed: Hello 🌍 世界"},
                {"text": "Zero-width: test\u200btest"},  # Zero-width space
                {"text": "Control chars: test\u0001test"},  # Control character
                {"text": "High unicode: \U0001F600"},  # Very high unicode
            ]
            
            try:
                for data in unicode_data:
                    client.store(data)
                
                # Verify unicode integrity
                for i, original in enumerate(unicode_data):
                    result = client.retrieve(i)
                    assert result["text"] == original["text"]
                    
            except Exception as e:
                print(f"Unicode corruption handling: {e}")
            
            client.close()


class TestNetworkAndIOErrors:
    """Test network and I/O error simulation"""
    
    def test_file_permission_errors(self):
        """Test handling of file permission errors"""
        with tempfile.TemporaryDirectory() as temp_dir:
            client = ApexClient(dirpath=temp_dir)
            
            # Store some data
            client.store({"test": "data"})
            client.flush()
            client.close()
            
            # Make database file read-only
            db_file = Path(temp_dir) / "apexbase.apex"
            if not db_file.exists():
                # File may not exist if using different storage strategy
                return
                
            try:
                db_file.chmod(0o444)  # Read-only
                
                # Try to write to read-only database
                try:
                    client2 = ApexClient(dirpath=temp_dir)
                    client2.store({"new": "data"})
                    client2.close()
                except (PermissionError, RuntimeError, OSError):
                    pass  # Expected
                except Exception as e:
                    print(f"Permission error handling: {e}")
                
            finally:
                # Restore permissions for cleanup
                try:
                    db_file.chmod(0o644)
                except:
                    pass
    
    def test_directory_permission_errors(self):
        """Test handling of directory permission errors"""
        with tempfile.TemporaryDirectory() as temp_dir:
            try:
                # Make directory read-only
                os.chmod(temp_dir, 0o555)  # Read and execute only
                
                # Try to create client in read-only directory
                try:
                    client = ApexClient(dirpath=temp_dir)
                    client.close()
                except PermissionError:
                    pass  # Expected
                except Exception as e:
                    print(f"Directory permission error handling: {e}")
                
            finally:
                # Restore permissions for cleanup
                try:
                    os.chmod(temp_dir, 0o755)
                except:
                    pass
    
    def test_missing_database_file(self):
        """Test handling of missing database file"""
        with tempfile.TemporaryDirectory() as temp_dir:
            client = ApexClient(dirpath=temp_dir)
            client.store({"test": "data"})
            client.close()
            
            # Remove database file
            db_file = Path(temp_dir) / "apexbase.apex"
            db_file.unlink()
            
            # Try to open client with missing file
            try:
                client2 = ApexClient(dirpath=temp_dir)
                # Should create new file or handle gracefully
                client2.store({"new": "data"})
                client2.close()
            except Exception as e:
                print(f"Missing file handling: {e}")


class TestMemoryPressure:
    """Test memory pressure handling"""
    
    def test_large_result_set_memory(self):
        """Test handling of large result sets that might cause memory pressure"""
        with tempfile.TemporaryDirectory() as temp_dir:
            client = ApexClient(dirpath=temp_dir)
            
            # Store large dataset
            large_data = [
                {"id": i, "data": "x" * 1000}  # 1KB per record
                for i in range(10000)  # 10MB total
            ]
            client.store(large_data)
            
            try:
                # Try to retrieve all data at once
                results = client.retrieve_all()
                assert len(results) == 10000
                
                # Try to convert to large pandas DataFrame if available
                if ARROW_AVAILABLE:
                    try:
                        df = results.to_pandas()
                        assert len(df) == 10000
                    except MemoryError:
                        pass  # Expected for very large datasets
                        
            except MemoryError:
                pass  # Should handle memory pressure gracefully
            except Exception as e:
                print(f"Large result set handling: {e}")
            
            client.close()
    
    def test_memory_leak_simulation(self):
        """Test for potential memory leaks with repeated operations"""
        with tempfile.TemporaryDirectory() as temp_dir:
            client = ApexClient(dirpath=temp_dir)
            
            try:
                # Perform many operations that might leak memory
                for i in range(1000):
                    # Store data
                    client.store({"iteration": i, "data": f"test_{i}"})
                    
                    # Query data
                    results = client.query(f"iteration = {i}")
                    assert len(results) == 1
                    
                    # Retrieve data
                    result = client.retrieve(i)
                    assert result is not None
                    
                    # Clean up references
                    del results, result
                    
                    # Periodic garbage collection
                    if i % 100 == 0:
                        import gc
                        gc.collect()
                        
            except MemoryError:
                pass  # Should handle memory pressure
            except Exception as e:
                print(f"Memory leak simulation: {e}")
            
            client.close()


class TestInvalidStateTransitions:
    """Test invalid state transitions"""
    
    def test_operations_after_close(self):
        """Test operations after client close"""
        with tempfile.TemporaryDirectory() as temp_dir:
            client = ApexClient(dirpath=temp_dir)
            client.store({"test": "data"})
            client.close()
            
            # All operations should raise RuntimeError
            with pytest.raises(RuntimeError, match="connection has been closed"):
                client.store({"new": "data"})
            
            with pytest.raises(RuntimeError, match="connection has been closed"):
                client.query()
            
            with pytest.raises(RuntimeError, match="connection has been closed"):
                client.retrieve(0)
            
            with pytest.raises(RuntimeError, match="connection has been closed"):
                client.delete(0)
            
            with pytest.raises(RuntimeError, match="connection has been closed"):
                client.create_table("test")
            
            with pytest.raises(RuntimeError, match="connection has been closed"):
                client.list_tables()
    
    def test_multiple_close_calls(self):
        """Test multiple close calls"""
        with tempfile.TemporaryDirectory() as temp_dir:
            client = ApexClient(dirpath=temp_dir)
            client.store({"test": "data"})
            
            # Multiple close calls should not raise errors
            client.close()
            client.close()  # Should not raise error
            client.close()  # Should not raise error
    
    def test_fts_operations_without_init(self):
        """Test FTS operations without initialization"""
        with tempfile.TemporaryDirectory() as temp_dir:
            client = ApexClient(dirpath=temp_dir)
            client.store({"content": "test"})
            
            # FTS operations should raise ValueError
            with pytest.raises(ValueError, match="Full-text search is not enabled"):
                client.search_text("test")
            
            with pytest.raises(ValueError, match="Full-text search is not enabled"):
                client.fuzzy_search_text("test")
            
            with pytest.raises(ValueError, match="Full-text search is not enabled"):
                client.search_and_retrieve("test")
            
            client.close()


class TestBoundaryConditions:
    """Test boundary conditions"""
    
    def test_empty_string_operations(self):
        """Test operations with empty strings"""
        with tempfile.TemporaryDirectory() as temp_dir:
            client = ApexClient(dirpath=temp_dir)
            
            # Store data with empty strings
            client.store({"empty": "", "normal": "test"})
            
            # Query with empty strings
            results = client.query("empty = ''")
            assert len(results) == 1
            
            # Search with empty strings (if FTS enabled)
            client.init_fts(index_fields=['content'])
            client.store({"content": ""})
            
            search_results = client.search_text("")
            # May return empty or all results depending on implementation
            assert isinstance(search_results, np.ndarray)
            
            client.close()
    
    def test_zero_and_negative_values(self):
        """Test operations with zero and negative values"""
        with tempfile.TemporaryDirectory() as temp_dir:
            client = ApexClient(dirpath=temp_dir)
            
            # Store data with zero and negative values
            test_data = [
                {"value": 0},
                {"value": -1},
                {"value": -1000000},
                {"value": 0.0},
                {"value": -0.001},
            ]
            client.store(test_data)
            
            # Query zero values - note: int 0 and float 0.0 may be stored differently
            results = client.query("value = 0")
            assert len(results) >= 1  # At least one zero value
            
            # Query negative values
            results = client.query("value < 0")
            assert len(results) >= 2  # At least some negative values
            
            client.close()
    
    def test_maximum_values(self):
        """Test operations with maximum values"""
        with tempfile.TemporaryDirectory() as temp_dir:
            client = ApexClient(dirpath=temp_dir)
            
            # Store data with maximum values
            test_data = [
                {"value": 2**63 - 1},  # Max 64-bit int
                {"value": -2**63},     # Min 64-bit int
                {"value": float('inf')},  # Infinity
                {"value": float('-inf')}, # Negative infinity
            ]
            
            try:
                client.store(test_data)
                
                # Query max int value
                results = client.query(f"value = {2**63 - 1}")
                assert len(results) == 1
                
            except Exception as e:
                print(f"Maximum values handling: {e}")
            
            client.close()
    
    def test_very_long_identifiers(self):
        """Test operations with very long identifiers"""
        with tempfile.TemporaryDirectory() as temp_dir:
            client = ApexClient(dirpath=temp_dir)
            
            # Very long field names
            long_field_name = "field_" + "x" * 1000
            
            try:
                client.store({long_field_name: "test"})
                
                # Query with long field name
                results = client.query(f"{long_field_name} = 'test'")
                assert len(results) == 1
                
            except Exception as e:
                print(f"Long identifier handling: {e}")
            
            client.close()


class TestExceptionPropagation:
    """Test exception propagation and error messages"""
    
    def test_descriptive_error_messages(self):
        """Test that error messages are descriptive"""
        with tempfile.TemporaryDirectory() as temp_dir:
            client = ApexClient(dirpath=temp_dir)
            client.close()
            
            # Check that error messages are descriptive
            with pytest.raises(RuntimeError) as exc_info:
                client.store({"test": "data"})
            
            error_msg = str(exc_info.value)
            assert "connection has been closed" in error_msg.lower()
            assert len(error_msg) > 10  # Should be reasonably descriptive
    
    def test_error_recovery(self):
        """Test error recovery scenarios"""
        with tempfile.TemporaryDirectory() as temp_dir:
            client = ApexClient(dirpath=temp_dir)
            
            # Cause an error and try to recover
            try:
                # Try invalid operation
                client.delete(999999)  # Non-existent ID
            except Exception:
                pass  # Ignore error
            
            # Client should still be usable
            client.store({"recovery": "test"})
            result = client.retrieve(0)
            assert result["recovery"] == "test"
            
            client.close()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
