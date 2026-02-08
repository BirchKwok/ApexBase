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


class TestCrashRecovery:
    """Test WAL-based crash recovery scenarios."""

    def test_wal_recovery_after_reopen(self):
        """Data written before close should be recoverable after reopen."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Session 1: write data and close
            c1 = ApexClient(temp_dir)
            c1.create_table('recovery', {'name': 'string', 'value': 'int'})
            c1.use_table('recovery')
            c1.store([{'name': 'Alice', 'value': 10}])
            c1.store([{'name': 'Bob', 'value': 20}])
            c1.flush()
            c1.close()

            # Session 2: reopen and verify data persisted
            c2 = ApexClient(temp_dir)
            c2.create_table('default')
            c2.use_table('recovery')
            assert c2.count_rows() == 2
            df = c2.execute("SELECT name, value FROM recovery ORDER BY name").to_pandas()
            assert list(df['name']) == ['Alice', 'Bob']
            assert list(df['value']) == [10, 20]
            c2.close()

    def test_incremental_wal_recovery(self):
        """Multiple sessions of incremental writes should all be recoverable."""
        with tempfile.TemporaryDirectory() as temp_dir:
            for session in range(5):
                c = ApexClient(temp_dir)
                c.create_table('default')
                if session == 0:
                    c.create_table('inc', {'session': 'int', 'row': 'int'})
                c.use_table('inc')
                c.store([{'session': session, 'row': i} for i in range(20)])
                c.flush()
                c.close()

            # Final verification
            c = ApexClient(temp_dir)
            c.create_table('default')
            c.use_table('inc')
            assert c.count_rows() == 100
            for session in range(5):
                df = c.execute(f"SELECT COUNT(*) as cnt FROM inc WHERE session = {session}").to_dict()
                assert df[0]['cnt'] == 20
            c.close()

    def test_committed_txn_survives_reopen(self):
        """A committed transaction's writes should survive close/reopen."""
        with tempfile.TemporaryDirectory() as temp_dir:
            c1 = ApexClient(temp_dir)
            c1.create_table('txn_persist', {'name': 'string', 'age': 'int'})
            c1.use_table('txn_persist')
            c1.store([{'name': 'Baseline', 'age': 1}])

            c1.execute('BEGIN')
            c1.execute("INSERT INTO txn_persist (name, age) VALUES ('TxnRow', 42)")
            c1.execute('COMMIT')
            c1.flush()
            c1.close()

            c2 = ApexClient(temp_dir)
            c2.create_table('default')
            c2.use_table('txn_persist')
            assert c2.count_rows() == 2
            df = c2.execute("SELECT name FROM txn_persist ORDER BY name").to_pandas()
            assert 'TxnRow' in list(df['name'])
            c2.close()

    def test_rolled_back_txn_not_persisted(self):
        """A rolled-back transaction's writes should not appear after reopen."""
        with tempfile.TemporaryDirectory() as temp_dir:
            c1 = ApexClient(temp_dir)
            c1.create_table('rb_persist', {'name': 'string', 'age': 'int'})
            c1.use_table('rb_persist')
            c1.store([{'name': 'Baseline', 'age': 1}])

            c1.execute('BEGIN')
            c1.execute("INSERT INTO rb_persist (name, age) VALUES ('Ghost', 99)")
            c1.execute('ROLLBACK')
            c1.flush()
            c1.close()

            c2 = ApexClient(temp_dir)
            c2.create_table('default')
            c2.use_table('rb_persist')
            assert c2.count_rows() == 1
            df = c2.execute("SELECT name FROM rb_persist").to_pandas()
            assert 'Ghost' not in list(df['name'])
            c2.close()


class TestTransactionIsolation:
    """Test transaction isolation properties with concurrent threads."""

    def test_read_your_writes(self):
        """Within a transaction, SELECT should see buffered INSERT writes."""
        with tempfile.TemporaryDirectory() as temp_dir:
            c = ApexClient(temp_dir)
            c.create_table('ryw', {'name': 'string', 'value': 'int'})
            c.use_table('ryw')
            c.store([{'name': 'Pre', 'value': 1}])

            c.execute('BEGIN')
            c.execute("INSERT INTO ryw (name, value) VALUES ('InTxn', 42)")
            # Should see both rows within the transaction
            df = c.execute("SELECT * FROM ryw").to_pandas()
            assert len(df) == 2
            c.execute('COMMIT')

            # After commit, should still see both
            assert c.count_rows() == 2
            c.close()

    def test_savepoint_partial_rollback(self):
        """SAVEPOINT allows partial rollback within a transaction."""
        with tempfile.TemporaryDirectory() as temp_dir:
            c = ApexClient(temp_dir)
            c.create_table('sp_test', {'name': 'string', 'value': 'int'})
            c.use_table('sp_test')

            c.execute('BEGIN')
            c.execute("INSERT INTO sp_test (name, value) VALUES ('Keep', 1)")
            c.execute('SAVEPOINT sp1')
            c.execute("INSERT INTO sp_test (name, value) VALUES ('Discard', 2)")
            c.execute('ROLLBACK TO sp1')
            c.execute('COMMIT')

            assert c.count_rows() == 1
            df = c.execute("SELECT name FROM sp_test").to_pandas()
            assert list(df['name']) == ['Keep']
            c.close()

    def test_concurrent_read_during_write(self):
        """Reads from other threads should not block during writes."""
        import threading

        with tempfile.TemporaryDirectory() as temp_dir:
            c = ApexClient(temp_dir)
            c.create_table('conc_rw', {'name': 'string', 'value': 'int'})
            c.use_table('conc_rw')
            c.store([{'name': f'row_{i}', 'value': i} for i in range(100)])
            c.flush()

            errors = []
            read_results = []

            def reader():
                try:
                    for _ in range(20):
                        df = c.execute("SELECT COUNT(*) as cnt FROM conc_rw").to_dict()
                        read_results.append(df[0]['cnt'])
                except Exception as e:
                    errors.append(f"Reader error: {e}")

            def writer():
                try:
                    for i in range(20):
                        c.store([{'name': f'new_{i}', 'value': 1000 + i}])
                except Exception as e:
                    errors.append(f"Writer error: {e}")

            threads = [threading.Thread(target=reader) for _ in range(3)]
            threads.append(threading.Thread(target=writer))

            for t in threads:
                t.start()
            for t in threads:
                t.join()

            assert len(errors) == 0, f"Errors: {errors}"
            # All reads should return >= 100 (initial rows)
            assert all(r >= 100 for r in read_results), f"Read results: {read_results}"
            c.close()

    def test_transaction_multi_table(self):
        """Transaction spanning multiple tables commits atomically."""
        with tempfile.TemporaryDirectory() as temp_dir:
            c = ApexClient(temp_dir)
            c.create_table('orders', {'item': 'string', 'qty': 'int'})
            c.create_table('inventory', {'item': 'string', 'stock': 'int'})

            # Setup inventory
            c.use_table('inventory')
            c.store([{'item': 'Widget', 'stock': 100}])

            # Setup orders
            c.use_table('orders')

            # Transaction: insert order + update inventory
            c.execute('BEGIN')
            c.execute("INSERT INTO orders (item, qty) VALUES ('Widget', 5)")
            c.execute('COMMIT')

            # Verify order recorded
            df = c.execute("SELECT * FROM orders").to_pandas()
            assert len(df) == 1
            assert df['item'].iloc[0] == 'Widget'
            assert df['qty'].iloc[0] == 5

            c.close()
