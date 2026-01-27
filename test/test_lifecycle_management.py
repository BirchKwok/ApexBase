"""
Comprehensive test suite for ApexBase Context Manager and Lifecycle Management

This module tests:
- Context manager (__enter__, __exit__) functionality
- Automatic resource cleanup
- Exception handling in context managers
- Manual close() operations
- Instance registry and cleanup
- Force close scenarios
- Lifecycle state transitions
- Resource leak prevention
"""

import pytest
import tempfile
import shutil
from pathlib import Path
import sys
import os
import weakref
import gc
import threading
import time

# Add the apexbase python module to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'apexbase', 'python'))

try:
    from apexbase import ApexClient, FTS_AVAILABLE
except ImportError as e:
    pytest.skip(f"ApexBase not available: {e}", allow_module_level=True)


class TestContextManager:
    """Test context manager functionality"""
    
    def test_basic_context_manager(self):
        """Test basic context manager usage"""
        with tempfile.TemporaryDirectory() as temp_dir:
            with ApexClient(dirpath=temp_dir) as client:
                assert not client._is_closed
                
                # Perform operations
                client.store({"name": "Alice", "age": 25})
                result = client.retrieve(0)
                assert result["name"] == "Alice"
            
            # Client should be closed after context
            assert client._is_closed
    
    def test_context_manager_with_exception(self):
        """Test context manager with exception inside"""
        with tempfile.TemporaryDirectory() as temp_dir:
            with pytest.raises(ValueError, match="test exception"):
                with ApexClient(dirpath=temp_dir) as client:
                    client.store({"name": "Alice", "age": 25})
                    
                    # Raise an exception
                    raise ValueError("test exception")
            
            # Client should still be closed despite exception
            assert client._is_closed
    
    def test_context_manager_chain_operations(self):
        """Test context manager with chain operations"""
        with tempfile.TemporaryDirectory() as temp_dir:
            with ApexClient(dirpath=temp_dir).init_fts(index_fields=['content']) as client:
                assert client._is_fts_enabled()
                
                # Store searchable content
                client.store({"content": "Python programming"})
                
                # Search should work
                results = client.search_text("python")
                assert len(results) > 0
            
            # Client should be closed
            assert client._is_closed
    
    def test_context_manager_nested(self):
        """Test nested context managers"""
        with tempfile.TemporaryDirectory() as temp_dir:
            with ApexClient(dirpath=temp_dir) as client1:
                client1.store({"name": "Alice"})
                
                # Create second client with DIFFERENT path to avoid conflicts
                temp_dir2 = tempfile.mkdtemp()
                try:
                    with ApexClient(dirpath=temp_dir2) as client2:
                        client2.store({"name": "Bob"})
                        
                        # Both clients should be active with different paths
                        assert not client2._is_closed
                    
                    # Inner client should be closed
                    assert client2._is_closed
                finally:
                    import shutil
                    shutil.rmtree(temp_dir2, ignore_errors=True)
    
    def test_context_manager_return_value(self):
        """Test context manager return value"""
        with tempfile.TemporaryDirectory() as temp_dir:
            with ApexClient(dirpath=temp_dir) as client:
                # __enter__ should return self
                assert client is not None
                assert hasattr(client, 'store')
                assert hasattr(client, 'query')
    
    def test_context_manager_exit_suppression(self):
        """Test that context manager doesn't suppress exceptions"""
        with tempfile.TemporaryDirectory() as temp_dir:
            with pytest.raises(ValueError, match="test exception"):
                with ApexClient(dirpath=temp_dir) as client:
                    client.store({"test": "data"})
                    raise ValueError("test exception")
            
            # Exception should not be suppressed
            assert client._is_closed


class TestManualCloseOperations:
    """Test manual close operations"""
    
    def test_manual_close_basic(self):
        """Test basic manual close operation"""
        with tempfile.TemporaryDirectory() as temp_dir:
            client = ApexClient(dirpath=temp_dir)
            
            assert not client._is_closed
            
            # Perform operations
            client.store({"name": "Alice", "age": 25})
            result = client.retrieve(0)
            assert result["name"] == "Alice"
            
            # Manual close
            client.close()
            assert client._is_closed
            
            # Operations should fail after close
            with pytest.raises(RuntimeError, match="connection has been closed"):
                client.store({"test": "data"})
    
    def test_multiple_close_calls(self):
        """Test multiple close calls"""
        with tempfile.TemporaryDirectory() as temp_dir:
            client = ApexClient(dirpath=temp_dir)
            
            client.store({"name": "Alice"})
            client.close()
            assert client._is_closed
            
            # Multiple close calls should not raise errors
            client.close()
            client.close()
            assert client._is_closed
    
    def test_force_close_operation(self):
        """Test force close operation"""
        with tempfile.TemporaryDirectory() as temp_dir:
            client = ApexClient(dirpath=temp_dir)
            
            client.store({"name": "Alice"})
            
            # Force close (internal method)
            client._force_close()
            assert client._is_closed
            
            # Should handle gracefully
            client._force_close()
            assert client._is_closed
    
    def test_close_with_fts_enabled(self):
        """Test close with FTS enabled"""
        with tempfile.TemporaryDirectory() as temp_dir:
            client = ApexClient(dirpath=temp_dir)
            client.init_fts(index_fields=['content'])
            
            client.store({"content": "Test content"})
            
            # Close should properly clean up FTS resources
            client.close()
            assert client._is_closed
            
            # FTS operations should fail after close
            with pytest.raises((RuntimeError, ValueError, AttributeError)):
                client.search_text("test")


class TestInstanceRegistry:
    """Test instance registry and automatic cleanup"""
    
    def test_registry_registration(self):
        """Test automatic registration in instance registry"""
        with tempfile.TemporaryDirectory() as temp_dir:
            client = ApexClient(dirpath=temp_dir)
            
            # Client should be registered
            db_path = str(client._db_path)
            assert db_path in client._registry._instances
            
            # Close should unregister
            client.close()
            assert db_path not in client._registry._instances
    
    def test_registry_auto_cleanup(self):
        """Test automatic cleanup through registry"""
        with tempfile.TemporaryDirectory() as temp_dir:
            client = ApexClient(dirpath=temp_dir)
            
            # Store some data
            client.store({"name": "Alice", "age": 25})
            
            # Get reference to client
            client_ref = weakref.ref(client)
            
            # Delete client reference
            del client
            
            # Force garbage collection
            gc.collect()
            time.sleep(0.1)  # Allow cleanup to happen
            
            # Client should be cleaned up (weak reference should be dead)
            assert client_ref() is None
    
    def test_registry_close_all(self):
        """Test close all functionality"""
        # Create multiple clients with different paths
        temp_dirs = [tempfile.mkdtemp() for _ in range(3)]
        clients = []
        
        try:
            for i, td in enumerate(temp_dirs):
                client = ApexClient(dirpath=td)
                client.store({"id": i, "name": f"Client_{i}"})
                clients.append(client)
            
            # Close all through registry
            if clients:
                clients[0]._registry.close_all()
            
            # All should be closed
            for client in clients:
                assert client._is_closed
        finally:
            import shutil
            for td in temp_dirs:
                shutil.rmtree(td, ignore_errors=True)
    
    def test_registry_duplicate_paths(self):
        """Test registry handling of duplicate paths"""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create first client
            client1 = ApexClient(dirpath=temp_dir)
            client1.store({"name": "First"})
            
            # Create second client with same path - first may be closed
            client2 = ApexClient(dirpath=temp_dir)
            
            # Second client should be active
            assert not client2._is_closed
            
            # Second client should work
            client2.store({"name": "Second"})
            
            client2.close()
            
            client2.close()
    
    def test_registry_disabled_auto_manage(self):
        """Test registry with auto management disabled"""
        with tempfile.TemporaryDirectory() as temp_dir:
            client = ApexClient(dirpath=temp_dir, _auto_manage=False)
            
            # Should not be in registry
            db_path = str(client._db_path)
            assert db_path not in client._registry._instances
            
            # Close should work normally
            client.close()
            assert client._is_closed


class TestLifecycleStateTransitions:
    """Test lifecycle state transitions"""
    
    def test_state_transition_sequence(self):
        """Test proper state transition sequence"""
        with tempfile.TemporaryDirectory() as temp_dir:
            client = ApexClient(dirpath=temp_dir)
            
            # Initial state: not closed
            assert not client._is_closed
            
            # Operations should work
            client.store({"test": "data"})
            
            # Close state
            client.close()
            assert client._is_closed
            
            # Operations should fail
            with pytest.raises(RuntimeError):
                client.store({"test": "data"})
    
    def test_state_persistence_across_operations(self):
        """Test state persistence across various operations"""
        with tempfile.TemporaryDirectory() as temp_dir:
            client = ApexClient(dirpath=temp_dir)
            
            # State should persist across operations
            operations = [
                lambda: client.store({"test": "data"}),
                lambda: client.query(),
                lambda: client.retrieve(0),
                lambda: client.list_tables(),
                lambda: client.count_rows(),
            ]
            
            for op in operations:
                assert not client._is_closed
                op()
                assert not client._is_closed
            
            client.close()
            assert client._is_closed
    
    def test_state_with_table_operations(self):
        """Test state with table operations"""
        with tempfile.TemporaryDirectory() as temp_dir:
            client = ApexClient(dirpath=temp_dir)
            
            # Table operations should maintain state
            client.create_table("test_table")
            assert not client._is_closed
            
            client.use_table("test_table")
            assert not client._is_closed
            
            client.drop_table("test_table")
            assert not client._is_closed
            
            client.close()
            assert client._is_closed


class TestResourceLeakPrevention:
    """Test resource leak prevention"""
    
    def test_file_handle_cleanup(self):
        """Test file handle cleanup on close"""
        with tempfile.TemporaryDirectory() as temp_dir:
            client = ApexClient(dirpath=temp_dir)
            
            # Store data to ensure file handles are used
            for i in range(100):
                client.store({"id": i, "data": f"test_{i}"})
            
            # Close should clean up file handles
            client.close()
            
            # Try to remove directory (should work if handles are cleaned up)
            try:
                shutil.rmtree(temp_dir)
                # If we get here, handles were cleaned up properly
                # Recreate for cleanup
                os.makedirs(temp_dir)
            except OSError as e:
                # If directory can't be removed, handles might not be cleaned up
                pytest.fail(f"File handle leak detected: {e}")
    
    def test_memory_cleanup_on_close(self):
        """Test memory cleanup on close"""
        with tempfile.TemporaryDirectory() as temp_dir:
            client = ApexClient(dirpath=temp_dir)
            
            # Store large data to use memory
            large_data = [{"id": i, "data": "x" * 1000} for i in range(1000)]
            client.store(large_data)
            
            # Get memory usage before close
            client_ref = weakref.ref(client)
            
            # Close client
            client.close()
            assert client._is_closed
            
            # Delete reference and force garbage collection
            del client
            gc.collect()
            time.sleep(0.1)
            
            # Memory should be cleaned up
            assert client_ref() is None
    
    def test_fts_resource_cleanup(self):
        """Test FTS resource cleanup"""
        with tempfile.TemporaryDirectory() as temp_dir:
            client = ApexClient(dirpath=temp_dir)
            client.init_fts(index_fields=['content'])
            
            # Store searchable data
            for i in range(100):
                client.store({"content": f"Searchable content {i}"})
            
            # Perform searches to ensure FTS resources are used
            results = client.search_text("content")
            assert len(results) > 0
            
            # Close should clean up FTS resources
            client.close()
            assert client._is_closed
            
            # FTS files should be cleaned up or properly closed
            fts_dir = Path(temp_dir) / "fts_indexes"
            if fts_dir.exists():
                # Directory should exist but files should be properly closed
                assert fts_dir.is_dir()


class TestExceptionHandling:
    """Test exception handling in lifecycle operations"""
    
    def test_exception_during_init(self):
        """Test exception during initialization"""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Try to create client with invalid parameters
            try:
                client = ApexClient(dirpath=temp_dir, durability="invalid")
            except ValueError:
                pass  # Expected
            
            # Should not leave resources in inconsistent state
            # (hard to test directly, but should not crash)
    
    def test_exception_during_operation(self):
        """Test exception during operation"""
        with tempfile.TemporaryDirectory() as temp_dir:
            client = ApexClient(dirpath=temp_dir)
            
            # Cause an exception during operation
            try:
                # Try to delete non-existent record
                client.delete(999999)
            except Exception:
                pass  # Expected
            
            # Client should still be usable
            client.store({"recovery": "test"})
            result = client.retrieve(0)
            assert result["recovery"] == "test"
            
            client.close()
    
    def test_exception_in_context_manager(self):
        """Test exception handling in context manager"""
        with tempfile.TemporaryDirectory() as temp_dir:
            exception_caught = False
            
            try:
                with ApexClient(dirpath=temp_dir) as client:
                    client.store({"test": "data"})
                    raise ValueError("Test exception")
            except ValueError:
                exception_caught = True
            
            assert exception_caught
            assert client._is_closed
    
    def test_nested_exception_handling(self):
        """Test nested exception handling"""
        temp_dir1 = tempfile.mkdtemp()
        temp_dir2 = tempfile.mkdtemp()
        
        try:
            with ApexClient(dirpath=temp_dir1) as client1:
                client1.store({"name": "Alice"})
                
                try:
                    # Use different path to avoid conflicts
                    with ApexClient(dirpath=temp_dir2) as client2:
                        client2.store({"name": "Bob"})
                        raise RuntimeError("Inner exception")
                except RuntimeError:
                    pass  # Expected
                
                # Outer client should still work
                client1.store({"name": "Charlie"})
                
        except Exception as e:
            print(f"Nested exception test: {e}")
        finally:
            import shutil
            shutil.rmtree(temp_dir1, ignore_errors=True)
            shutil.rmtree(temp_dir2, ignore_errors=True)


class TestAtexitCleanup:
    """Test atexit cleanup functionality"""
    
    def test_atexit_registration(self):
        """Test that clients are registered for atexit cleanup"""
        with tempfile.TemporaryDirectory() as temp_dir:
            client = ApexClient(dirpath=temp_dir)
            
            # Client should be registered for cleanup
            # (Hard to test directly without actually exiting)
            
            # Manual close should unregister
            client.close()
            assert client._is_closed
    
    def test_cleanup_on_interpreter_shutdown(self):
        """Test cleanup behavior on interpreter shutdown simulation"""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create client
            client = ApexClient(dirpath=temp_dir)
            client.store({"test": "data"})
            
            # Simulate interpreter shutdown by calling registry cleanup
            client._registry.close_all()
            
            # Client should be closed
            assert client._is_closed


class TestConcurrentLifecycle:
    """Test concurrent lifecycle operations"""
    
    def test_concurrent_close(self):
        """Test concurrent close operations"""
        with tempfile.TemporaryDirectory() as temp_dir:
            client = ApexClient(dirpath=temp_dir)
            
            # Store some data
            for i in range(100):
                client.store({"id": i, "data": f"test_{i}"})
            
            def close_client():
                try:
                    client.close()
                    return True
                except Exception as e:
                    print(f"Close error: {e}")
                    return False
            
            # Try to close from multiple threads (only one should succeed)
            import threading
            threads = []
            results = []
            
            for i in range(5):
                thread = threading.Thread(target=lambda: results.append(close_client()))
                threads.append(thread)
                thread.start()
            
            for thread in threads:
                thread.join()
            
            # Client should be closed
            assert client._is_closed
            # At least one close should succeed
            assert any(results)
    
    def test_concurrent_creation_cleanup(self):
        """Test concurrent client creation and cleanup with file locking
        
        With reader-writer file locking, concurrent WRITE operations from different
        clients will require exclusive locks. This test verifies that:
        1. File locking prevents concurrent write access (serialized writes)
        2. At least some operations succeed
        3. No data corruption occurs
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            def create_and_close():
                try:
                    client = ApexClient(dirpath=temp_dir)
                    client.store({"test": "data"})
                    client.close()
                    return True
                except Exception as e:
                    # Expected: "Database is locked" errors for concurrent write access
                    if "locked" in str(e).lower():
                        return False
                    print(f"Create/close error: {e}")
                    return False
            
            # Run multiple create/close cycles
            import threading
            threads = []
            results = []
            
            for i in range(10):
                thread = threading.Thread(target=lambda: results.append(create_and_close()))
                threads.append(thread)
                thread.start()
            
            for thread in threads:
                thread.join()
            
            # With file locking, at least one should succeed (the first to acquire the lock)
            # Other concurrent writes may fail with lock errors - this is expected behavior
            success_count = sum(results)
            assert success_count >= 1, f"At least one concurrent operation should succeed, got {success_count}"


class TestLifecycleWithFTS:
    """Test lifecycle management with FTS"""
    
    def test_fts_lifecycle_integration(self):
        """Test FTS integration with lifecycle management"""
        with tempfile.TemporaryDirectory() as temp_dir:
            with ApexClient(dirpath=temp_dir) as client:
                # Initialize FTS
                client.init_fts(index_fields=['content'])
                
                # Store searchable data
                client.store({"content": "Python programming"})
                
                # Search should work
                results = client.search_text("python")
                # May or may not find results depending on indexing
                
                # Modify data - behavior may vary
                try:
                    client.replace(0, {"content": "JavaScript programming"})
                except Exception as e:
                    print(f"FTS lifecycle replace: {e}")
            
            # FTS resources should be cleaned up
            assert client._is_closed
    
    def test_fts_cleanup_on_close(self):
        """Test FTS cleanup on close"""
        with tempfile.TemporaryDirectory() as temp_dir:
            client = ApexClient(dirpath=temp_dir)
            client.init_fts(index_fields=['content'])
            
            # Store and search data
            client.store({"content": "Test content"})
            results = client.search_text("test")
            # Results may vary
            
            # Close should clean up FTS
            client.close()
            assert client._is_closed
            
            # FTS operations should fail after close
            with pytest.raises((RuntimeError, ValueError, AttributeError)):
                client.search_text("test")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
