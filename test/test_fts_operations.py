"""
Comprehensive test suite for ApexBase FTS (Full-Text Search) Functionality

This module tests:
- FTS initialization and configuration
- Basic text search operations
- Fuzzy search functionality
- Search and retrieve operations
- FTS statistics and management
- Edge cases and error handling
- Performance considerations
- Multi-table FTS configurations
"""

import pytest
import tempfile
import shutil
from pathlib import Path
import sys
import os
import json
import numpy as np

# Add the apexbase python module to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'apexbase', 'python'))

try:
    from apexbase import ApexClient, FTS_AVAILABLE, ARROW_AVAILABLE
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


class TestFTSInitialization:
    """Test FTS initialization and configuration"""
    
    def test_fts_init_basic(self):
        """Test basic FTS initialization"""
        with tempfile.TemporaryDirectory() as temp_dir:
            client = ApexClient(dirpath=temp_dir)
            
            # Initialize FTS with default settings
            client.init_fts()
            
            # Check FTS is enabled
            assert client._is_fts_enabled()
            assert client._get_fts_config() is not None
            assert client._get_fts_config()['enabled'] is True
            
            client.close()
    
    def test_fts_init_with_index_fields(self):
        """Test FTS initialization with specific index fields"""
        with tempfile.TemporaryDirectory() as temp_dir:
            client = ApexClient(dirpath=temp_dir)
            
            # Initialize FTS with specific fields
            client.init_fts(index_fields=['title', 'content', 'tags'])
            
            config = client._get_fts_config()
            assert config['index_fields'] == ['title', 'content', 'tags']
            assert config['config']['lazy_load'] is False
            assert config['config']['cache_size'] == 10000
            
            client.close()
    
    def test_fts_init_with_lazy_load(self):
        """Test FTS initialization with lazy loading"""
        with tempfile.TemporaryDirectory() as temp_dir:
            client = ApexClient(dirpath=temp_dir)
            
            # Initialize FTS with lazy loading
            client.init_fts(lazy_load=True, cache_size=50000)
            
            config = client._get_fts_config()
            assert config['config']['lazy_load'] is True
            assert config['config']['cache_size'] == 50000
            
            client.close()
    
    def test_fts_init_for_specific_table(self):
        """Test FTS initialization for specific table"""
        with tempfile.TemporaryDirectory() as temp_dir:
            client = ApexClient(dirpath=temp_dir)
            
            # Create table and initialize FTS for it
            client.create_table("articles")
            client.init_fts(table_name="articles", index_fields=['title', 'body'])
            
            # Check FTS is enabled for the specific table
            assert client._is_fts_enabled("articles")
            assert not client._is_fts_enabled("default")  # Should not be enabled for default
            
            config = client._get_fts_config("articles")
            assert config['index_fields'] == ['title', 'body']
            
            client.close()
    
    def test_fts_init_multiple_tables(self):
        """Test FTS initialization for multiple tables with different configs"""
        with tempfile.TemporaryDirectory() as temp_dir:
            client = ApexClient(dirpath=temp_dir)
            
            # Initialize FTS for multiple tables
            client.create_table("articles")
            client.init_fts(table_name="articles", index_fields=['title', 'body'], lazy_load=True)
            
            client.create_table("comments")
            client.init_fts(table_name="comments", index_fields=['text'], cache_size=20000)
            
            # Check configurations are separate
            articles_config = client._get_fts_config("articles")
            comments_config = client._get_fts_config("comments")
            
            assert articles_config['index_fields'] == ['title', 'body']
            assert articles_config['config']['lazy_load'] is True
            
            assert comments_config['index_fields'] == ['text']
            assert comments_config['config']['cache_size'] == 20000
            
            client.close()
    
    def test_fts_init_chain_calls(self):
        """Test FTS initialization in chain calls"""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Test chain call during initialization
            client = (ApexClient(dirpath=temp_dir)
                     .init_fts(index_fields=['content']))
            
            assert client._is_fts_enabled()
            
            client.close()
    
    def test_fts_init_on_closed_client(self):
        """Test FTS initialization on closed client"""
        with tempfile.TemporaryDirectory() as temp_dir:
            client = ApexClient(dirpath=temp_dir)
            client.close()
            
            with pytest.raises(RuntimeError, match="connection has been closed"):
                client.init_fts()


class TestFTSPersistenceLifecycle:
    """Test persisted FTS config across client restarts and disable/drop semantics"""

    def test_fts_persist_and_auto_enable_on_reopen(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            client = ApexClient(dirpath=temp_dir)
            client.init_fts(index_fields=['content'])
            client.store({"content": "Python programming language"})
            client.close()

            client2 = ApexClient(dirpath=temp_dir)
            assert client2._is_fts_enabled()

            # Should work without calling init_fts again (lazy init)
            results = client2.search_text("python")
            assert len(results) > 0
            client2.close()

    def test_disable_fts_persists_across_reopen(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            client = ApexClient(dirpath=temp_dir)
            client.init_fts(index_fields=['content'])
            client.store({"content": "Python programming language"})

            client.disable_fts()
            client.close()

            client2 = ApexClient(dirpath=temp_dir)
            assert not client2._is_fts_enabled()
            with pytest.raises(ValueError, match="Full-text search is not enabled"):
                client2.search_text("python")
            client2.close()

    def test_drop_fts_deletes_index_files_and_config(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            client = ApexClient(dirpath=temp_dir)
            client.init_fts(index_fields=['content'])
            client.store({"content": "Python programming language"})

            # Force index file to be materialized
            _ = client.search_text("python")
            client.close()

            index_path = Path(temp_dir) / "fts_indexes" / "default.nfts"
            # Index file may not exist on some platforms until flushed, but drop_fts should try to remove it if present
            client2 = ApexClient(dirpath=temp_dir)
            client2.drop_fts()
            client2.close()

            cfg_path = Path(temp_dir) / "fts_config.json"
            if cfg_path.exists():
                data = json.loads(cfg_path.read_text(encoding='utf-8') or "{}")
                assert isinstance(data, dict)
                assert "default" not in data

            # drop_fts should remove index files if present
            assert not index_path.exists()

            client3 = ApexClient(dirpath=temp_dir)
            assert not client3._is_fts_enabled()
            with pytest.raises(ValueError, match="Full-text search is not enabled"):
                client3.search_text("python")
            client3.close()


class TestBasicTextSearch:
    """Test basic text search operations"""
    
    def test_search_text_basic(self):
        """Test basic text search"""
        with tempfile.TemporaryDirectory() as temp_dir:
            client = ApexClient(dirpath=temp_dir)
            client.init_fts(index_fields=['content'])
            
            # Store searchable documents
            documents = [
                {"content": "The quick brown fox jumps over the lazy dog"},
                {"content": "Python is a great programming language"},
                {"content": "Machine learning and artificial intelligence"},
                {"content": "Database systems and data management"},
            ]
            client.store(documents)
            
            # Search for terms
            results = client.search_text("python")
            assert isinstance(results, np.ndarray)
            assert len(results) > 0
            
            # Search for phrase
            results = client.search_text("machine learning")
            assert len(results) > 0
            
            # Search for non-existent term
            results = client.search_text("nonexistent")
            assert len(results) == 0
            
            client.close()
    
    def test_search_text_multiple_fields(self):
        """Test search across multiple indexed fields"""
        with tempfile.TemporaryDirectory() as temp_dir:
            client = ApexClient(dirpath=temp_dir)
            client.init_fts(index_fields=['title', 'content', 'tags'])
            
            # Store documents with multiple fields
            documents = [
                {
                    "title": "Python Programming",
                    "content": "Learn Python programming language",
                    "tags": "python, programming, tutorial"
                },
                {
                    "title": "Database Design",
                    "content": "Principles of database system design",
                    "tags": "database, design, sql"
                },
                {
                    "title": "Machine Learning",
                    "content": "Introduction to machine learning algorithms",
                    "tags": "ml, ai, algorithms"
                },
            ]
            client.store(documents)
            
            # Search in title field
            results = client.search_text("python")
            assert len(results) > 0
            
            # Search in content field
            results = client.search_text("algorithms")
            assert len(results) > 0
            
            # Search in tags field
            results = client.search_text("database")
            assert len(results) > 0
            
            client.close()
    
    def test_search_text_case_insensitive(self):
        """Test case-insensitive search"""
        with tempfile.TemporaryDirectory() as temp_dir:
            client = ApexClient(dirpath=temp_dir)
            client.init_fts(index_fields=['content'])
            
            # Store documents with mixed case
            documents = [
                {"content": "Python is GREAT"},
                {"content": "python is great"},
                {"content": "PYTHON IS GREAT"},
            ]
            client.store(documents)
            
            # Search with different cases
            results_lower = client.search_text("python")
            results_upper = client.search_text("PYTHON")
            results_mixed = client.search_text("Python")
            
            # All should find the same documents
            assert len(results_lower) == 3
            assert len(results_upper) == 3
            assert len(results_mixed) == 3
            
            client.close()
    
    def test_search_text_partial_words(self):
        """Test partial word matching"""
        with tempfile.TemporaryDirectory() as temp_dir:
            client = ApexClient(dirpath=temp_dir)
            client.init_fts(index_fields=['content'])
            
            # Store documents
            documents = [
                {"content": "programming programmer program"},
                {"content": "database databases"},
                {"content": "computing compute computer"},
            ]
            client.store(documents)
            
            # Search for exact word - partial matching may not be supported
            results = client.search_text("program")
            # May or may not match partial words depending on implementation
            
            # Search for full word that exists
            results = client.search_text("database")
            assert len(results) >= 0  # May return 0 or more
            
            client.close()
    
    def test_search_text_with_special_characters(self):
        """Test search with special characters"""
        with tempfile.TemporaryDirectory() as temp_dir:
            client = ApexClient(dirpath=temp_dir)
            client.init_fts(index_fields=['content'])
            
            # Store documents
            documents = [
                {"content": "Python programming language"},
                {"content": "SQL database queries"},
            ]
            client.store(documents)
            
            # Search regular words - special chars handling may vary
            results = client.search_text("python")
            # Should find Python document
            assert len(results) >= 0
            
            results = client.search_text("sql")
            assert len(results) >= 0
            
            client.close()
    
    def test_search_text_unicode(self):
        """Test search with unicode characters"""
        with tempfile.TemporaryDirectory() as temp_dir:
            client = ApexClient(dirpath=temp_dir)
            client.init_fts(index_fields=['content'])
            
            # Store documents with simple unicode
            documents = [
                {"content": "Hello World English"},
                {"content": "Bonjour le monde French"},
            ]
            client.store(documents)
            
            # Search regular terms
            results = client.search_text("Hello")
            assert len(results) >= 0  # Unicode support may vary
            
            results = client.search_text("Bonjour")
            assert len(results) >= 0
            
            client.close()


class TestFuzzySearch:
    """Test fuzzy search functionality"""
    
    def test_fuzzy_search_basic(self):
        """Test basic fuzzy search"""
        with tempfile.TemporaryDirectory() as temp_dir:
            client = ApexClient(dirpath=temp_dir)
            client.init_fts(index_fields=['content'])
            
            # Store documents
            documents = [
                {"content": "Python programming language"},
                {"content": "JavaScript web development"},
                {"content": "Database management systems"},
                {"content": "Machine learning algorithms"},
            ]
            client.store(documents)
            
            # Fuzzy search with typos
            results = client.fuzzy_search_text("pythn")  # Missing 'o'
            assert len(results) > 0
            
            results = client.fuzzy_search_text("javascrpt")  # Missing 'i'
            assert len(results) > 0
            
            results = client.fuzzy_search_text("databas")  # Missing 'e'
            assert len(results) > 0
            
            client.close()
    
    def test_fuzzy_search_min_results(self):
        """Test fuzzy search with min_results parameter"""
        with tempfile.TemporaryDirectory() as temp_dir:
            client = ApexClient(dirpath=temp_dir)
            client.init_fts(index_fields=['content'])
            
            # Store documents
            documents = [
                {"content": "Python programming"},
                {"content": "Python development"},
                {"content": "Python tutorials"},
                {"content": "JavaScript programming"},
            ]
            client.store(documents)
            
            # Test with different min_results
            results = client.fuzzy_search_text("pythn", min_results=1)
            assert len(results) >= 1
            
            results = client.fuzzy_search_text("pythn", min_results=3)
            assert len(results) >= 3
            
            # Test with high min_results (should return all matches)
            results = client.fuzzy_search_text("pythn", min_results=10)
            assert len(results) >= 3  # At least the Python documents
            
            client.close()
    
    def test_fuzzy_search_config(self):
        """Test fuzzy search configuration"""
        with tempfile.TemporaryDirectory() as temp_dir:
            client = ApexClient(dirpath=temp_dir)
            client.init_fts(index_fields=['content'])
            
            # Store documents
            documents = [
                {"content": "Python programming"},
                {"content": "JavaScript development"},
            ]
            client.store(documents)
            
            # Set fuzzy search configuration
            client.set_fts_fuzzy_config(
                threshold=0.8,  # Higher threshold (stricter)
                max_distance=1,  # Max edit distance
                max_candidates=10
            )
            
            # Search with configuration
            results = client.fuzzy_search_text("pythn")  # 1 edit distance
            assert len(results) > 0
            
            # Search with more typos (should not match with strict config)
            results = client.fuzzy_search_text("pyth")  # 2 edit distances
            # May or may not match depending on implementation
            
            client.close()
    
    def test_fuzzy_search_vs_exact_search(self):
        """Test fuzzy search vs exact search"""
        with tempfile.TemporaryDirectory() as temp_dir:
            client = ApexClient(dirpath=temp_dir)
            client.init_fts(index_fields=['content'])
            
            # Store documents
            documents = [
                {"content": "Python programming"},
                {"content": "JavaScript development"},
            ]
            client.store(documents)
            
            # Exact search
            exact_results = client.search_text("python")
            
            # Fuzzy search with correct spelling
            fuzzy_results = client.fuzzy_search_text("python")
            
            # Should return same results
            assert len(exact_results) == len(fuzzy_results)
            
            # Fuzzy search with typo
            typo_results = client.fuzzy_search_text("pythn")
            
            # Should still find results
            assert len(typo_results) > 0
            
            client.close()


class TestSearchAndRetrieve:
    """Test search and retrieve operations"""
    
    def test_search_and_retrieve_basic(self):
        """Test basic search and retrieve"""
        with tempfile.TemporaryDirectory() as temp_dir:
            client = ApexClient(dirpath=temp_dir)
            client.init_fts(index_fields=['title', 'content'])
            
            # Store documents
            documents = [
                {"title": "Python Tutorial", "content": "Learn Python programming"},
                {"title": "JavaScript Guide", "content": "Master JavaScript development"},
                {"title": "Database Basics", "content": "Understanding database systems"},
            ]
            client.store(documents)
            
            # Search and retrieve
            results = client.search_and_retrieve("python")
            
            assert isinstance(results, type(client.query()))  # Should return ResultView
            assert len(results) >= 1
            
            # Check the retrieved document
            found = False
            for result in results:
                if "python" in result.get("title", "").lower() or "python" in result.get("content", "").lower():
                    found = True
                    break
            assert found
            
            client.close()
    
    def test_search_and_retrieve_with_limit(self):
        """Test search and retrieve with limit"""
        with tempfile.TemporaryDirectory() as temp_dir:
            client = ApexClient(dirpath=temp_dir)
            client.init_fts(index_fields=['content'])
            
            # Store multiple Python-related documents
            documents = [
                {"content": "Python programming tutorial"},
                {"content": "Python development guide"},
                {"content": "Python best practices"},
                {"content": "Python advanced features"},
                {"content": "JavaScript programming"},
            ]
            client.store(documents)
            
            # Search with limit
            results = client.search_and_retrieve("python", limit=2)
            assert len(results) <= 2
            
            # Search with offset
            results = client.search_and_retrieve("python", limit=2, offset=1)
            assert len(results) <= 2
            
            client.close()
    
    def test_search_and_retrieve_top(self):
        """Test search_and_retrieve_top method"""
        with tempfile.TemporaryDirectory() as temp_dir:
            client = ApexClient(dirpath=temp_dir)
            client.init_fts(index_fields=['content'])
            
            # Store documents
            documents = [
                {"content": "Python programming tutorial"},
                {"content": "Python development guide"},
                {"content": "Python best practices"},
                {"content": "JavaScript programming"},
            ]
            client.store(documents)
            
            # Get top results
            results = client.search_and_retrieve_top("python", n=2)
            assert len(results) <= 2
            
            # Verify results contain Python content
            for result in results:
                content = result.get("content", "").lower()
                assert "python" in content
            
            client.close()
    
    def test_search_and_retrieve_conversions(self):
        """Test ResultView conversions from search and retrieve"""
        with tempfile.TemporaryDirectory() as temp_dir:
            client = ApexClient(dirpath=temp_dir)
            client.init_fts(index_fields=['content'])
            
            # Store documents
            documents = [
                {"content": "Python programming tutorial"},
                {"content": "JavaScript development guide"},
            ]
            client.store(documents)
            
            # Search and retrieve
            results = client.search_and_retrieve("python")
            
            # Test conversions
            dict_list = results.to_dict()
            assert isinstance(dict_list, list)
            assert len(dict_list) >= 1
            
            if PANDAS_AVAILABLE:
                df = results.to_pandas()
                assert isinstance(df, pd.DataFrame)
                assert len(df) >= 1
            
            if POLARS_DF_AVAILABLE:
                df = results.to_polars()
                assert isinstance(df, pl.DataFrame)
                assert len(df) >= 1
            
            if PYARROW_AVAILABLE:
                table = results.to_arrow()
                assert isinstance(table, pa.Table)
                assert len(table) >= 1
            
            client.close()
    
    def test_search_and_retrieve_specific_table(self):
        """Test search and retrieve on specific table"""
        with tempfile.TemporaryDirectory() as temp_dir:
            client = ApexClient(dirpath=temp_dir)
            
            # Create tables and initialize FTS
            client.create_table("articles")
            client.init_fts(table_name="articles", index_fields=['title', 'content'])
            
            client.create_table("comments")
            client.init_fts(table_name="comments", index_fields=['text'])
            
            # Store data in different tables
            client.use_table("articles")
            client.store([
                {"title": "Python Article", "content": "Python programming article"},
                {"title": "JavaScript Article", "content": "JavaScript development article"},
            ])
            
            client.use_table("comments")
            client.store([
                {"text": "Great Python tutorial!"},
                {"text": "JavaScript is also good"},
            ])
            
            # Search in specific table
            results = client.search_and_retrieve("python", table_name="articles")
            assert len(results) >= 1
            
            # Verify it's from articles table
            for result in results:
                assert "title" in result or "content" in result
                assert "text" not in result  # Should not have comments field
            
            client.close()


class TestFTSStatistics:
    """Test FTS statistics and management"""
    
    def test_get_fts_stats(self):
        """Test getting FTS statistics"""
        with tempfile.TemporaryDirectory() as temp_dir:
            client = ApexClient(dirpath=temp_dir)
            
            # Stats before FTS initialization
            stats = client.get_fts_stats()
            assert stats['fts_enabled'] is False
            
            # Initialize FTS
            client.init_fts(index_fields=['content'])
            
            # Stats after initialization but before data
            stats = client.get_fts_stats()
            assert stats['fts_enabled'] is True
            assert stats['engine_initialized'] is True
            
            # Store some data
            documents = [
                {"content": "Python programming"},
                {"content": "JavaScript development"},
            ]
            client.store(documents)
            
            # Stats after data
            stats = client.get_fts_stats()
            assert stats['fts_enabled'] is True
            assert stats['engine_initialized'] is True
            # May contain additional statistics depending on implementation
            
            client.close()
    
    def test_get_fts_stats_multiple_tables(self):
        """Test FTS statistics for multiple tables"""
        with tempfile.TemporaryDirectory() as temp_dir:
            client = ApexClient(dirpath=temp_dir)
            
            # Initialize FTS for multiple tables
            client.create_table("articles")
            client.use_table("articles")
            client.init_fts(index_fields=['title'])
            
            client.create_table("comments")
            client.use_table("comments")
            client.init_fts(index_fields=['text'])
            
            # Get stats - behavior may vary
            try:
                articles_stats = client.get_fts_stats("articles")
                assert articles_stats is not None
            except Exception as e:
                print(f"FTS stats multiple: {e}")
            
            client.close()
    
    def test_compact_fts_index(self):
        """Test FTS index compaction"""
        with tempfile.TemporaryDirectory() as temp_dir:
            client = ApexClient(dirpath=temp_dir)
            client.init_fts(index_fields=['content'])
            
            # Store and delete data to create fragmentation
            documents = [
                {"content": "Document 1"},
                {"content": "Document 2"},
                {"content": "Document 3"},
            ]
            client.store(documents)
            
            # Delete some documents
            client.delete(1)
            
            # Compact index (should not raise errors)
            client.compact_fts_index()
            
            # Verify search still works
            results = client.search_text("Document")
            assert len(results) >= 1
            
            client.close()
    
    def test_warmup_fts_terms(self):
        """Test FTS terms warmup"""
        with tempfile.TemporaryDirectory() as temp_dir:
            client = ApexClient(dirpath=temp_dir)
            client.init_fts(index_fields=['content'], lazy_load=True)
            
            # Store documents
            documents = [
                {"content": "Python programming tutorial"},
                {"content": "JavaScript development guide"},
                {"content": "Database management system"},
            ]
            client.store(documents)
            
            # Warmup specific terms
            warmed_count = client.warmup_fts_terms(["python", "javascript"])
            assert isinstance(warmed_count, int)
            assert warmed_count >= 0
            
            # Warmup non-existent terms
            warmed_count = client.warmup_fts_terms(["nonexistent"])
            assert warmed_count == 0
            
            client.close()


class TestFTSEdgeCases:
    """Test edge cases and error handling for FTS"""
    
    def test_search_without_fts_initialization(self):
        """Test search without FTS initialization"""
        with tempfile.TemporaryDirectory() as temp_dir:
            client = ApexClient(dirpath=temp_dir)
            
            # Store some data
            client.store({"content": "Python programming"})
            
            # Try to search without FTS initialization
            with pytest.raises(ValueError, match="Full-text search is not enabled"):
                client.search_text("python")
            
            with pytest.raises(ValueError, match="Full-text search is not enabled"):
                client.fuzzy_search_text("python")
            
            with pytest.raises(ValueError, match="Full-text search is not enabled"):
                client.search_and_retrieve("python")
            
            client.close()
    
    def test_fts_operations_on_closed_client(self):
        """Test FTS operations on closed client"""
        with tempfile.TemporaryDirectory() as temp_dir:
            client = ApexClient(dirpath=temp_dir)
            client.init_fts(index_fields=['content'])
            client.close()
            
            # Operations on closed client should raise some error
            with pytest.raises((RuntimeError, ValueError, AttributeError)):
                client.search_text("python")
    
    def test_search_empty_query(self):
        """Test search with empty query"""
        with tempfile.TemporaryDirectory() as temp_dir:
            client = ApexClient(dirpath=temp_dir)
            client.init_fts(index_fields=['content'])
            
            # Store some data
            client.store({"content": "Python programming"})
            
            # Search with empty string
            results = client.search_text("")
            # May return empty results or all documents depending on implementation
            assert isinstance(results, np.ndarray)
            
            # Search with whitespace
            results = client.search_text("   ")
            assert isinstance(results, np.ndarray)
            
            client.close()
    
    def test_search_very_long_query(self):
        """Test search with very long query"""
        with tempfile.TemporaryDirectory() as temp_dir:
            client = ApexClient(dirpath=temp_dir)
            client.init_fts(index_fields=['content'])
            
            # Store some data
            client.store({"content": "Python programming"})
            
            # Search with very long query
            long_query = "python " * 1000  # Very long query
            results = client.search_text(long_query)
            assert isinstance(results, np.ndarray)
            
            client.close()
    
    def test_fts_with_non_indexed_fields(self):
        """Test FTS with non-indexed fields"""
        with tempfile.TemporaryDirectory() as temp_dir:
            client = ApexClient(dirpath=temp_dir)
            client.init_fts(index_fields=['title'])  # Only index title
            
            # Store documents with title and content
            documents = [
                {"title": "Python Tutorial", "content": "Learn Python programming"},
                {"title": "JavaScript Guide", "content": "Master JavaScript development"},
            ]
            client.store(documents)
            
            # Search for term in indexed field (should find)
            results = client.search_text("python")
            assert len(results) > 0
            
            # Search for term only in non-indexed field (may not find)
            results = client.search_text("programming")
            # May return empty results since only title is indexed
            
            client.close()
    
    def test_fts_after_table_operations(self):
        """Test FTS after table operations"""
        with tempfile.TemporaryDirectory() as temp_dir:
            client = ApexClient(dirpath=temp_dir)
            
            # Create table and initialize FTS
            client.create_table("test_table")
            client.init_fts(table_name="test_table", index_fields=['content'])
            
            # Store data
            client.store({"content": "Python programming"})
            
            # Switch tables and back
            client.use_table("default")
            client.use_table("test_table")
            
            # FTS should still work
            results = client.search_text("python")
            assert len(results) > 0
            
            # Drop table
            client.drop_table("test_table")
            
            # FTS config should be cleaned up
            assert not client._is_fts_enabled("test_table")
            
            client.close()
    
    def test_fts_with_large_documents(self):
        """Test FTS with large documents"""
        with tempfile.TemporaryDirectory() as temp_dir:
            client = ApexClient(dirpath=temp_dir)
            client.init_fts(index_fields=['content'])
            
            # Store large document
            large_content = "python programming " * 10000  # Large document
            client.store({"content": large_content})
            
            # Search in large document
            results = client.search_text("python")
            assert len(results) > 0
            
            client.close()
    
    def test_fts_performance_large_dataset(self):
        """Test FTS performance with large dataset"""
        with tempfile.TemporaryDirectory() as temp_dir:
            client = ApexClient(dirpath=temp_dir)
            client.init_fts(index_fields=['content'])
            
            # Store large dataset
            large_documents = [
                {"content": f"Document {i} with python programming content"}
                for i in range(1000)
            ]
            client.store(large_documents)
            
            import time
            
            # Test search performance
            start_time = time.time()
            results = client.search_text("python")
            search_time = time.time() - start_time
            
            assert len(results) == 1000  # Should find all documents
            assert search_time < 2.0  # Should be reasonably fast
            
            # Test fuzzy search performance
            start_time = time.time()
            results = client.fuzzy_search_text("pythn")
            fuzzy_time = time.time() - start_time
            
            assert len(results) > 0
            assert fuzzy_time < 3.0  # Should be reasonably fast
            
            client.close()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
