"""
Tests for MVCC Transaction support (Phase 4)
BEGIN / COMMIT / ROLLBACK with OCC conflict detection
"""

import pytest
import tempfile
import os
from apexbase import ApexClient


@pytest.fixture
def client():
    d = tempfile.mkdtemp()
    c = ApexClient(d)
    c.create_table('txn_test', {'name': 'string', 'age': 'int', 'city': 'string'})
    c.use_table('txn_test')
    yield c
    c.close()


class TestTransactionBasics:
    """Basic BEGIN / COMMIT / ROLLBACK lifecycle."""

    def test_begin_commit_insert(self, client):
        """INSERT within a transaction is only visible after COMMIT."""
        client.store([{'name': 'Alice', 'age': 25, 'city': 'NYC'}])
        assert client.count_rows() == 1

        client.execute('BEGIN')
        client.execute("INSERT INTO txn_test (name, age, city) VALUES ('Bob', 30, 'LA')")
        client.execute('COMMIT')

        assert client.count_rows() == 2
        df = client.execute("SELECT name FROM txn_test ORDER BY name").to_pandas()
        assert list(df['name']) == ['Alice', 'Bob']

    def test_rollback_discards_insert(self, client):
        """INSERT within a rolled-back transaction is discarded."""
        client.store([{'name': 'Alice', 'age': 25, 'city': 'NYC'}])
        assert client.count_rows() == 1

        client.execute('BEGIN')
        client.execute("INSERT INTO txn_test (name, age, city) VALUES ('Eve', 99, 'Berlin')")
        client.execute('ROLLBACK')

        assert client.count_rows() == 1

    def test_begin_commit_delete(self, client):
        """DELETE within a transaction is applied on COMMIT."""
        client.store([
            {'name': 'Alice', 'age': 25, 'city': 'NYC'},
            {'name': 'Bob', 'age': 30, 'city': 'LA'},
        ])
        assert client.count_rows() == 2

        client.execute('BEGIN')
        client.execute("DELETE FROM txn_test WHERE name = 'Bob'")
        client.execute('COMMIT')

        assert client.count_rows() == 1
        df = client.execute("SELECT name FROM txn_test").to_pandas()
        assert list(df['name']) == ['Alice']

    def test_rollback_discards_delete(self, client):
        """DELETE within a rolled-back transaction is discarded."""
        client.store([
            {'name': 'Alice', 'age': 25, 'city': 'NYC'},
            {'name': 'Bob', 'age': 30, 'city': 'LA'},
        ])

        client.execute('BEGIN')
        client.execute("DELETE FROM txn_test WHERE name = 'Alice'")
        client.execute('ROLLBACK')

        assert client.count_rows() == 2

    def test_begin_commit_update(self, client):
        """UPDATE within a transaction is applied on COMMIT."""
        client.store([
            {'name': 'Alice', 'age': 25, 'city': 'NYC'},
            {'name': 'Bob', 'age': 30, 'city': 'LA'},
        ])

        client.execute('BEGIN')
        client.execute("UPDATE txn_test SET age = 99 WHERE name = 'Alice'")
        client.execute('COMMIT')

        df = client.execute("SELECT age FROM txn_test WHERE name = 'Alice'").to_pandas()
        assert df['age'].iloc[0] == 99

    def test_rollback_discards_update(self, client):
        """UPDATE within a rolled-back transaction is discarded."""
        client.store([{'name': 'Alice', 'age': 25, 'city': 'NYC'}])

        client.execute('BEGIN')
        client.execute("UPDATE txn_test SET age = 99 WHERE name = 'Alice'")
        client.execute('ROLLBACK')

        df = client.execute("SELECT age FROM txn_test WHERE name = 'Alice'").to_pandas()
        assert df['age'].iloc[0] == 25


class TestTransactionSyntax:
    """Various SQL syntax forms for transaction control."""

    def test_begin_transaction_keyword(self, client):
        """BEGIN TRANSACTION is accepted."""
        client.store([{'name': 'Alice', 'age': 25, 'city': 'NYC'}])
        client.execute('BEGIN TRANSACTION')
        client.execute("INSERT INTO txn_test (name, age, city) VALUES ('X', 1, 'Z')")
        client.execute('COMMIT')
        assert client.count_rows() == 2

    def test_begin_read_only(self, client):
        """BEGIN TRANSACTION READ ONLY allows reads, commits cleanly."""
        client.store([{'name': 'Alice', 'age': 25, 'city': 'NYC'}])
        client.execute('BEGIN TRANSACTION READ ONLY')
        df = client.execute("SELECT * FROM txn_test").to_pandas()
        assert len(df) == 1
        client.execute('COMMIT')

    def test_commit_without_changes(self, client):
        """COMMIT with no buffered writes is a no-op."""
        client.store([{'name': 'Alice', 'age': 25, 'city': 'NYC'}])
        client.execute('BEGIN')
        client.execute('COMMIT')
        assert client.count_rows() == 1

    def test_rollback_without_changes(self, client):
        """ROLLBACK with no buffered writes is a no-op."""
        client.store([{'name': 'Alice', 'age': 25, 'city': 'NYC'}])
        client.execute('BEGIN')
        client.execute('ROLLBACK')
        assert client.count_rows() == 1


class TestTransactionMultiDML:
    """Multiple DML operations within a single transaction."""

    def test_multiple_inserts_in_txn(self, client):
        """Multiple INSERTs in one transaction all applied on COMMIT."""
        client.execute('BEGIN')
        client.execute("INSERT INTO txn_test (name, age, city) VALUES ('A', 1, 'X')")
        client.execute("INSERT INTO txn_test (name, age, city) VALUES ('B', 2, 'Y')")
        client.execute("INSERT INTO txn_test (name, age, city) VALUES ('C', 3, 'Z')")
        client.execute('COMMIT')
        assert client.count_rows() == 3

    def test_insert_then_delete_in_txn(self, client):
        """INSERT followed by DELETE in same transaction."""
        client.store([{'name': 'Alice', 'age': 25, 'city': 'NYC'}])
        client.execute('BEGIN')
        client.execute("INSERT INTO txn_test (name, age, city) VALUES ('Bob', 30, 'LA')")
        client.execute("DELETE FROM txn_test WHERE name = 'Alice'")
        client.execute('COMMIT')
        # Alice deleted, Bob inserted
        assert client.count_rows() == 1
        df = client.execute("SELECT name FROM txn_test").to_pandas()
        assert list(df['name']) == ['Bob']

    def test_multiple_rollback(self, client):
        """Multiple DML ops all discarded by ROLLBACK."""
        client.store([{'name': 'Alice', 'age': 25, 'city': 'NYC'}])
        client.execute('BEGIN')
        client.execute("INSERT INTO txn_test (name, age, city) VALUES ('Bob', 30, 'LA')")
        client.execute("UPDATE txn_test SET age = 99 WHERE name = 'Alice'")
        client.execute('ROLLBACK')
        assert client.count_rows() == 1
        df = client.execute("SELECT age FROM txn_test WHERE name = 'Alice'").to_pandas()
        assert df['age'].iloc[0] == 25


class TestTransactionSelect:
    """SELECT operations within transactions."""

    def test_select_in_txn(self, client):
        """SELECT works normally inside a transaction."""
        client.store([
            {'name': 'Alice', 'age': 25, 'city': 'NYC'},
            {'name': 'Bob', 'age': 30, 'city': 'LA'},
        ])
        client.execute('BEGIN')
        df = client.execute("SELECT * FROM txn_test ORDER BY name").to_pandas()
        assert len(df) == 2
        assert list(df['name']) == ['Alice', 'Bob']
        client.execute('COMMIT')

    def test_select_after_non_txn_operations(self, client):
        """Normal operations work after a committed transaction."""
        client.execute('BEGIN')
        client.execute("INSERT INTO txn_test (name, age, city) VALUES ('Alice', 25, 'NYC')")
        client.execute('COMMIT')

        # Non-transactional query
        df = client.execute("SELECT * FROM txn_test").to_pandas()
        assert len(df) == 1

        # Non-transactional insert
        client.execute("INSERT INTO txn_test (name, age, city) VALUES ('Bob', 30, 'LA')")
        assert client.count_rows() == 2
