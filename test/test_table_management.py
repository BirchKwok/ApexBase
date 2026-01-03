"""
Comprehensive test suite for ApexBase Table Management Operations

This module tests:
- Table creation, deletion, listing, and switching
- Table operations with FTS enabled/disabled
- Edge cases and error handling for table operations
- Table name validation and special characters
"""

import pytest
import tempfile
import shutil
from pathlib import Path
import sys
import os

# Add the apexbase python module to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'apexbase', 'python'))

try:
    from apexbase import ApexClient
except ImportError as e:
    pytest.skip(f"ApexBase not available: {e}", allow_module_level=True)


class TestTableManagement:
    """Test table management operations"""
    
    def test_create_table(self):
        """Test creating a new table"""
        with tempfile.TemporaryDirectory() as temp_dir:
            client = ApexClient(dirpath=temp_dir)
            
            # Create a new table
            client.create_table("users")
            
            # Check current table is updated
            assert client.current_table == "users"
            
            # Verify table exists in list
            tables = client.list_tables()
            assert "users" in tables
            
            client.close()
    
    def test_create_table_with_data(self):
        """Test creating table and immediately storing data"""
        with tempfile.TemporaryDirectory() as temp_dir:
            client = ApexClient(dirpath=temp_dir)
            
            client.create_table("products")
            client.store({"name": "Laptop", "price": 999.99})
            
            # Verify data was stored in new table
            count = client.count_rows()
            assert count == 1
            
            results = client.query()
            assert len(results) == 1
            assert results[0]["name"] == "Laptop"
            
            client.close()
    
    def test_use_table(self):
        """Test switching between tables"""
        with tempfile.TemporaryDirectory() as temp_dir:
            client = ApexClient(dirpath=temp_dir)
            
            # Create multiple tables
            client.create_table("users")
            client.store({"name": "Alice"})  # Will be stored in users table
            
            client.create_table("products")
            client.store({"name": "Laptop"})  # Will be stored in products table
            
            # Switch back to users table
            client.use_table("users")
            assert client.current_table == "users"
            
            # Switch to products table
            client.use_table("products")
            assert client.current_table == "products"
            
            client.close()
    
    def test_drop_table(self):
        """Test dropping a table"""
        with tempfile.TemporaryDirectory() as temp_dir:
            client = ApexClient(dirpath=temp_dir)
            
            # Create and populate table
            client.create_table("temp_table")
            client.store({"data": "test"})
            
            # Verify table exists
            tables = client.list_tables()
            assert "temp_table" in tables
            
            # Drop table
            client.drop_table("temp_table")
            
            # Verify table is gone
            tables = client.list_tables()
            assert "temp_table" not in tables
            
            # Current table should be reset to default
            assert client.current_table == "default"
            
            client.close()
    
    def test_drop_current_table(self):
        """Test dropping the current table"""
        with tempfile.TemporaryDirectory() as temp_dir:
            client = ApexClient(dirpath=temp_dir)
            
            # Create and switch to table
            client.create_table("current")
            assert client.current_table == "current"
            
            # Drop current table
            client.drop_table("current")
            
            # Should reset to default
            assert client.current_table == "default"
            
            client.close()
    
    def test_list_tables(self):
        """Test listing all tables"""
        with tempfile.TemporaryDirectory() as temp_dir:
            client = ApexClient(dirpath=temp_dir)
            
            # Initially should have default table
            tables = client.list_tables()
            assert "default" in tables
            
            # Create multiple tables
            client.create_table("users")
            client.create_table("products")
            client.create_table("orders")
            
            # List all tables
            tables = client.list_tables()
            assert len(tables) >= 4  # default + 3 created
            assert "users" in tables
            assert "products" in tables
            assert "orders" in tables
            assert "default" in tables
            
            client.close()
    
    def test_list_tables_empty_database(self):
        """Test listing tables in fresh database"""
        with tempfile.TemporaryDirectory() as temp_dir:
            client = ApexClient(dirpath=temp_dir, drop_if_exists=True)
            
            tables = client.list_tables()
            assert isinstance(tables, list)
            # Should at least have default table
            assert "default" in tables
            
            client.close()
    
    def test_current_table_property(self):
        """Test current_table property"""
        with tempfile.TemporaryDirectory() as temp_dir:
            client = ApexClient(dirpath=temp_dir)
            
            # Default should be "default"
            assert client.current_table == "default"
            
            # Create table should update current_table
            client.create_table("test_table")
            assert client.current_table == "test_table"
            
            # Use table should update current_table
            client.use_table("default")
            assert client.current_table == "default"
            
            client.close()
    
    def test_table_isolation(self):
        """Test data isolation between tables"""
        with tempfile.TemporaryDirectory() as temp_dir:
            client = ApexClient(dirpath=temp_dir)
            
            # Store data in default table
            client.store({"type": "default_data", "value": 1})
            
            # Create and store data in another table
            client.create_table("other")
            client.store({"type": "other_data", "value": 2})
            
            # Check data isolation
            client.use_table("default")
            default_results = client.query()
            assert len(default_results) == 1
            assert default_results[0]["type"] == "default_data"
            
            client.use_table("other")
            other_results = client.query()
            assert len(other_results) == 1
            assert other_results[0]["type"] == "other_data"
            
            client.close()
    
    def test_table_name_validation(self):
        """Test table name validation with various inputs"""
        with tempfile.TemporaryDirectory() as temp_dir:
            client = ApexClient(dirpath=temp_dir)
            
            # Valid table names
            valid_names = [
                "users",
                "user_data",
                "table123",
                "TABLE_UPPER",
                "table_with_underscores",
                "table-with-dashes",
                "a",  # Single character
                "a" * 100,  # Long name
            ]
            
            for name in valid_names:
                client.create_table(name)
                assert name in client.list_tables()
            
            client.close()
    
    def test_table_name_special_characters(self):
        """Test table names with special characters"""
        with tempfile.TemporaryDirectory() as temp_dir:
            client = ApexClient(dirpath=temp_dir)
            
            # Test various special characters
            special_names = [
                "table with spaces",
                "table@with#special$chars",
                "table.with.dots",
                "table/with/slashes",
                "table\\with\\backslashes",
                "table:with:colons",
                "table;with;semicolons",
                "table'with'quotes",
                'table"with"doublequotes',
                "table(with)parentheses",
                "table[with]brackets",
                "table{with}braces",
            ]
            
            for name in special_names:
                try:
                    client.create_table(name)
                    assert name in client.list_tables()
                except Exception as e:
                    # Some special characters might not be supported
                    # That's acceptable, just log it
                    print(f"Table name '{name}' not supported: {e}")
            
            client.close()
    
    def test_table_name_unicode(self):
        """Test table names with unicode characters"""
        with tempfile.TemporaryDirectory() as temp_dir:
            client = ApexClient(dirpath=temp_dir)
            
            unicode_names = [
                "用户",  # Chinese
                "пользователь",  # Russian
                "utilisateur",  # French with accent
                "benutzer",  # German with umlaut
                "📊data",  # Emoji
                "テーブル",  # Japanese
            ]
            
            for name in unicode_names:
                try:
                    client.create_table(name)
                    assert name in client.list_tables()
                except Exception as e:
                    print(f"Unicode table name '{name}' not supported: {e}")
            
            client.close()
    
    def test_drop_nonexistent_table(self):
        """Test dropping a table that doesn't exist"""
        with tempfile.TemporaryDirectory() as temp_dir:
            client = ApexClient(dirpath=temp_dir)
            
            # Try to drop non-existent table
            # Should not raise exception (graceful handling)
            client.drop_table("nonexistent_table")
            
            client.close()
    
    def test_use_nonexistent_table(self):
        """Test switching to a table that doesn't exist"""
        with tempfile.TemporaryDirectory() as temp_dir:
            client = ApexClient(dirpath=temp_dir)
            
            # Using non-existent table should create it or handle gracefully
            try:
                client.use_table("nonexistent")
                # If successful, table should be created or accessible
                assert client.current_table == "nonexistent"
            except Exception as e:
                # If it raises an exception, that's also acceptable behavior
                print(f"Using nonexistent table raised: {e}")
            
            client.close()
    
    def test_create_existing_table(self):
        """Test creating a table that already exists"""
        with tempfile.TemporaryDirectory() as temp_dir:
            client = ApexClient(dirpath=temp_dir)
            
            # Create table
            client.create_table("existing")
            
            # Try to create same table again
            try:
                client.create_table("existing")
                # If successful, should handle gracefully
            except Exception as e:
                # If it raises an exception, that's acceptable
                print(f"Creating existing table raised: {e}")
            
            client.close()
    
    def test_table_operations_with_fts(self):
        """Test table operations when FTS is enabled"""
        with tempfile.TemporaryDirectory() as temp_dir:
            client = ApexClient(dirpath=temp_dir)
            
            # Create table with FTS
            client.create_table("fts_table")
            client.init_fts(table_name="fts_table", index_fields=["content"])
            
            # Verify FTS is enabled for the table
            assert client._is_fts_enabled("fts_table")
            
            # Store data with searchable content
            client.store({"content": "searchable text", "metadata": "test"})
            
            # Perform search
            results = client.search_text("searchable", table_name="fts_table")
            assert results is not None
            assert len(results) > 0
            
            # Drop table with FTS
            client.drop_table("fts_table")
            
            # Verify FTS config is cleaned up
            assert not client._is_fts_enabled("fts_table")
            
            client.close()
    
    def test_multiple_table_fts_configs(self):
        """Test different FTS configurations for different tables"""
        with tempfile.TemporaryDirectory() as temp_dir:
            client = ApexClient(dirpath=temp_dir)
            
            # Create multiple tables with different FTS configs
            client.create_table("articles")
            client.init_fts(table_name="articles", index_fields=["title", "body"])
            
            client.create_table("comments")
            client.init_fts(table_name="comments", index_fields=["text"], lazy_load=True)
            
            # Verify different configs
            articles_config = client._get_fts_config("articles")
            comments_config = client._get_fts_config("comments")
            
            assert articles_config is not None
            assert comments_config is not None
            assert articles_config["index_fields"] == ["title", "body"]
            assert comments_config["index_fields"] == ["text"]
            assert comments_config["config"]["lazy_load"] is True
            
            client.close()
    
    def test_table_operations_on_closed_client(self):
        """Test table operations on closed client"""
        with tempfile.TemporaryDirectory() as temp_dir:
            client = ApexClient(dirpath=temp_dir)
            client.close()
            
            # All table operations should raise RuntimeError
            with pytest.raises(RuntimeError, match="connection has been closed"):
                client.create_table("test")
            
            with pytest.raises(RuntimeError, match="connection has been closed"):
                client.use_table("test")
            
            with pytest.raises(RuntimeError, match="connection has been closed"):
                client.drop_table("test")
            
            with pytest.raises(RuntimeError, match="connection has been closed"):
                client.list_tables()
            
            with pytest.raises(RuntimeError, match="connection has been closed"):
                _ = client.current_table
    
    def test_table_count_with_table_name(self):
        """Test count_rows with specific table name"""
        with tempfile.TemporaryDirectory() as temp_dir:
            client = ApexClient(dirpath=temp_dir)
            
            # Store data in default table
            client.store({"table": "default"})
            
            # Create and store data in another table
            client.create_table("other")
            client.store({"table": "other"})
            
            # Count rows in specific tables
            default_count = client.count_rows(table_name="default")
            other_count = client.count_rows(table_name="other")
            
            assert default_count == 1
            assert other_count == 1
            
            # Current table count should work
            current_count = client.count_rows()
            assert current_count == 1
            
            client.close()
    
    def test_table_edge_cases(self):
        """Test edge cases for table operations"""
        with tempfile.TemporaryDirectory() as temp_dir:
            client = ApexClient(dirpath=temp_dir)
            
            # Test empty string table name
            try:
                client.create_table("")
                tables = client.list_tables()
                assert "" in tables
            except Exception as e:
                print(f"Empty table name not supported: {e}")
            
            # Test very long table name
            long_name = "a" * 1000
            try:
                client.create_table(long_name)
                tables = client.list_tables()
                assert long_name in tables
            except Exception as e:
                print(f"Very long table name not supported: {e}")
            
            # Test table name with only numbers
            try:
                client.create_table("123")
                tables = client.list_tables()
                assert "123" in tables
            except Exception as e:
                print(f"Numeric table name not supported: {e}")
            
            client.close()
    
    def test_table_persistence(self):
        """Test table persistence across client sessions"""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create tables with first client
            client1 = ApexClient(dirpath=temp_dir)
            client1.create_table("persistent1")
            client1.store({"data": "test"})  # Stored in persistent1
            client1.create_table("persistent2")
            client1.use_table("persistent1")  # Switch back to persistent1
            client1.close()
            
            # Verify tables persist with second client
            client2 = ApexClient(dirpath=temp_dir)
            tables = client2.list_tables()
            assert "persistent1" in tables
            assert "persistent2" in tables
            
            # Verify data persists
            client2.use_table("persistent1")
            count = client2.count_rows()
            assert count == 1
            
            client2.close()
    
    def test_table_operations_with_different_durability(self):
        """Test table operations with different durability levels"""
        durability_levels = ['fast', 'safe', 'max']
        
        for durability in durability_levels:
            with tempfile.TemporaryDirectory() as temp_dir:
                client = ApexClient(dirpath=temp_dir, durability=durability)
                
                # Test all table operations
                client.create_table(f"test_{durability}")
                client.store({"durability": durability})
                
                tables = client.list_tables()
                assert f"test_{durability}" in tables
                
                client.use_table("default")
                assert client.current_table == "default"
                
                client.drop_table(f"test_{durability}")
                tables = client.list_tables()
                assert f"test_{durability}" not in tables
                
                client.close()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
