"""
Test cases for column encoding compatibility.
Tests various string encodings that may be encountered from different systems.
"""
import apexbase as ab
from apexbase import ApexClient
import tempfile
import pytest


class TestColumnEncoding:
    """Test various column encoding scenarios"""

    def test_create_index_on_string_column(self):
        """Test creating index on regular string column"""
        with tempfile.TemporaryDirectory() as tmpdir:
            client = ab.ApexClient(dirpath=tmpdir)
            client.create_table('test')

            # Insert test data
            data = [{'col': f'value_{i}'} for i in range(100)]
            client.store(data)

            # Create index
            client.execute('CREATE INDEX idx_col ON test (col)')
            print('Index created successfully')

            # Query using index
            result = client.execute("SELECT * FROM test WHERE col = 'value_50'")
            assert len(result) == 1
            assert result[0]['col'] == 'value_50'

            client.close()

    def test_create_index_on_dict_encoded_column(self):
        """Test creating index on dictionary-encoded column (low cardinality)"""
        with tempfile.TemporaryDirectory() as tmpdir:
            client = ab.ApexClient(dirpath=tmpdir)
            client.create_table('test')

            # Insert test data with low cardinality (good for dict encoding)
            data = [{'category': f'cat_{i % 5}'} for i in range(1000)]
            client.store(data)

            # Create index on low-cardinality column
            client.execute('CREATE INDEX idx_category ON test (category)')
            print('Index on dict-encoded column created successfully')

            # Query using index
            result = client.execute("SELECT * FROM test WHERE category = 'cat_2'")
            assert len(result) == 200

            client.close()

    def test_unicode_string_column(self):
        """Test Unicode string handling"""
        with tempfile.TemporaryDirectory() as tmpdir:
            client = ab.ApexClient(dirpath=tmpdir)
            client.create_table('test')

            # Insert Unicode data
            data = [
                {'text': '你好世界'},
                {'text': 'Hello World'},
                {'text': '日本語テスト'},
                {'text': '🔒 Emoji Test'},
            ]
            client.store(data)

            # Create index
            client.execute('CREATE INDEX idx_text ON test (text)')
            print('Index on Unicode column created successfully')

            # Query Unicode
            result = client.execute("SELECT * FROM test WHERE text = '你好世界'")
            assert len(result) == 1

            client.close()

    def test_special_characters_column(self):
        """Test special characters in strings"""
        with tempfile.TemporaryDirectory() as tmpdir:
            client = ab.ApexClient(dirpath=tmpdir)
            client.create_table('test')

            # Insert data with special characters
            data = [
                {'text': 'line1\nline2'},
                {'text': 'tab\there'},
                {'text': 'quote"here'},
                {'text': 'backslash\\here'},
            ]
            client.store(data)

            # Create index
            client.execute('CREATE INDEX idx_text ON test (text)')
            print('Index on special chars column created successfully')

            client.close()

    def test_large_string_column(self):
        """Test large string values"""
        with tempfile.TemporaryDirectory() as tmpdir:
            client = ab.ApexClient(dirpath=tmpdir)
            client.create_table('test')

            # Insert large string data
            large_text = 'x' * 10000
            data = [
                {'text': large_text},
                {'text': 'short'},
                {'text': 'medium' * 100},
            ]
            client.store(data)

            # Create index
            client.execute('CREATE INDEX idx_text ON test (text)')
            print('Index on large string column created successfully')

            client.close()

    def test_multiple_string_columns(self):
        """Test multiple string columns with different characteristics"""
        with tempfile.TemporaryDirectory() as tmpdir:
            client = ab.ApexClient(dirpath=tmpdir)
            client.create_table('test')

            # Insert data with multiple string columns
            data = []
            for i in range(500):
                data.append({
                    'low_card': f'type_{i % 3}',  # Very low cardinality
                    'med_card': f'group_{i % 50}',  # Medium cardinality
                    'high_card': f'unique_{i}',  # High cardinality
                })
            client.store(data)

            # Create indexes on all string columns
            client.execute('CREATE INDEX idx_low ON test (low_card)')
            client.execute('CREATE INDEX idx_med ON test (med_card)')
            client.execute('CREATE INDEX idx_high ON test (high_card)')
            print('Indexes on multiple string columns created successfully')

            # Query using different indexes
            result1 = client.execute("SELECT * FROM test WHERE low_card = 'type_1'")
            assert len(result1) > 0

            result2 = client.execute("SELECT * FROM test WHERE med_card = 'group_25'")
            assert len(result2) > 0

            client.close()

    def test_mixed_type_columns_with_string(self):
        """Test table with mixed types including string"""
        with tempfile.TemporaryDirectory() as tmpdir:
            client = ab.ApexClient(dirpath=tmpdir)
            client.create_table('test')

            # Insert mixed type data
            data = []
            for i in range(200):
                data.append({
                    'id': i,
                    'name': f'name_{i}',
                    'value': float(i) * 1.5,
                    'active': i % 2 == 0,
                })
            client.store(data)

            # Create index on string column
            client.execute('CREATE INDEX idx_name ON test (name)')
            print('Index on mixed-type table string column created successfully')

            # Query
            result = client.execute("SELECT * FROM test WHERE name = 'name_100'")
            assert len(result) == 1
            assert result[0]['id'] == 100

            client.close()


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
