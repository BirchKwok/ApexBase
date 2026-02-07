"""
Test cases for delete operations with id and where parameters.
"""
import pytest
import tempfile
import os
from apexbase import ApexClient


class TestDeleteOperations:
    """Test delete functionality with various parameters."""
    
    @pytest.fixture
    def client_with_data(self):
        """Create a client with test data."""
        tmpdir = tempfile.mkdtemp()
        client = ApexClient(os.path.join(tmpdir, 'test_delete'))
        client.create_table("default")
        
        # Insert test data
        client.store([
            {'name': 'Alice', 'age': 25, 'status': 'active'},
            {'name': 'Bob', 'age': 35, 'status': 'active'},
            {'name': 'Charlie', 'age': 45, 'status': 'inactive'},
            {'name': 'David', 'age': 30, 'status': 'inactive'},
            {'name': 'Eve', 'age': 28, 'status': 'active'},
        ])
        
        yield client
        
        # Cleanup
        client.close()
        import shutil
        shutil.rmtree(tmpdir, ignore_errors=True)
    
    def test_delete_by_single_id(self, client_with_data):
        """Test deleting a single record by ID."""
        client = client_with_data
        initial_count = client.count_rows()
        
        # Delete record with id=1
        result = client.delete(id=1)
        
        assert result == True
        assert client.count_rows() == initial_count - 1
    
    def test_delete_by_multiple_ids(self, client_with_data):
        """Test deleting multiple records by list of IDs."""
        client = client_with_data
        initial_count = client.count_rows()
        
        # Delete records with ids 1, 2, 3
        result = client.delete(id=[1, 2, 3])
        
        assert result == True
        assert client.count_rows() == initial_count - 3
    
    def test_delete_by_where_clause(self, client_with_data):
        """Test deleting records using WHERE clause."""
        client = client_with_data
        
        # Delete records where age > 30
        deleted_count = client.delete(where="age > 30")
        
        assert deleted_count == 2  # Bob (35) and Charlie (45)
        assert client.count_rows() == 3
    
    def test_delete_by_where_string_comparison(self, client_with_data):
        """Test deleting records with string comparison in WHERE clause."""
        client = client_with_data
        
        # Delete inactive records
        deleted_count = client.delete(where="status = 'inactive'")
        
        assert deleted_count == 2  # Charlie and David
        assert client.count_rows() == 3
    
    def test_delete_by_where_complex_condition(self, client_with_data):
        """Test deleting records with complex WHERE clause."""
        client = client_with_data
        
        # Delete active records with age < 30
        deleted_count = client.delete(where="status = 'active' AND age < 30")
        
        assert deleted_count == 2  # Alice (25) and Eve (28)
        assert client.count_rows() == 3
    
    def test_delete_no_args_raises_error(self, client_with_data):
        """Test that delete() without arguments raises ValueError for safety."""
        client = client_with_data
        initial_count = client.count_rows()
        
        # Should raise ValueError
        with pytest.raises(ValueError) as excinfo:
            client.delete()
        
        # Verify error message mentions safety
        assert "requires at least one argument" in str(excinfo.value)
        assert "'id' or 'where'" in str(excinfo.value)
        
        # Verify no data was deleted
        assert client.count_rows() == initial_count
    
    def test_delete_all_explicit(self, client_with_data):
        """Test explicit deletion of all records using where='1=1'."""
        client = client_with_data
        initial_count = client.count_rows()
        assert initial_count > 0
        
        # Explicitly delete all records
        deleted_count = client.delete(where="1=1")
        
        assert deleted_count == initial_count
        assert client.count_rows() == 0
    
    def test_delete_nonexistent_id(self, client_with_data):
        """Test deleting a non-existent ID returns False."""
        client = client_with_data
        initial_count = client.count_rows()
        
        # Try to delete non-existent ID
        result = client.delete(id=99999)
        
        assert result == False
        assert client.count_rows() == initial_count
    
    def test_delete_where_no_match(self, client_with_data):
        """Test deleting with WHERE clause that matches nothing."""
        client = client_with_data
        initial_count = client.count_rows()
        
        # WHERE clause that matches nothing
        deleted_count = client.delete(where="age > 100")
        
        assert deleted_count == 0
        assert client.count_rows() == initial_count
    
    def test_delete_where_between(self, client_with_data):
        """Test deleting with BETWEEN in WHERE clause."""
        client = client_with_data
        
        # Delete records where age is between 30 and 40
        deleted_count = client.delete(where="age BETWEEN 30 AND 40")
        
        assert deleted_count == 2  # Bob (35) and David (30)
        assert client.count_rows() == 3
    
    def test_delete_invalid_id_type_raises_error(self, client_with_data):
        """Test that invalid id type raises ValueError."""
        client = client_with_data
        
        with pytest.raises(ValueError) as excinfo:
            client.delete(id="invalid")
        
        assert "int or a list of ints" in str(excinfo.value)


class TestDeleteEdgeCases:
    """Test edge cases for delete operations."""
    
    @pytest.fixture
    def empty_client(self):
        """Create an empty client."""
        tmpdir = tempfile.mkdtemp()
        client = ApexClient(os.path.join(tmpdir, 'test_empty'))
        client.create_table("default")
        
        yield client
        
        client.close()
        import shutil
        shutil.rmtree(tmpdir, ignore_errors=True)
    
    def test_delete_from_empty_table(self, empty_client):
        """Test deleting from empty table."""
        client = empty_client
        
        # Should not raise error, just return 0
        deleted_count = client.delete(where="1=1")
        assert deleted_count == 0
    
    def test_delete_empty_id_list(self):
        """Test deleting with empty ID list."""
        tmpdir = tempfile.mkdtemp()
        try:
            client = ApexClient(os.path.join(tmpdir, 'test'))
            client.create_table("default")
            client.store([{'name': 'test'}])
            
            # Empty list should succeed but delete nothing
            result = client.delete(id=[])
            assert result == True
            assert client.count_rows() == 1
            
            client.close()
        finally:
            import shutil
            shutil.rmtree(tmpdir, ignore_errors=True)
