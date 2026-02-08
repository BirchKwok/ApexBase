"""
Concurrency stress tests for ApexBase.

Tests multi-thread read/write contention, lock retry behavior,
and data integrity under concurrent load.
"""

import pytest
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from apexbase import ApexClient


class TestConcurrentReadWrite:
    """Stress test concurrent reads and writes on a single client."""

    def test_heavy_concurrent_reads(self):
        """10 threads × 50 queries each — no errors, correct row count."""
        with tempfile.TemporaryDirectory() as d:
            c = ApexClient(d)
            c.create_table('stress', {'name': 'string', 'value': 'int'})
            c.use_table('stress')
            c.store([{'name': f'row_{i}', 'value': i} for i in range(500)])
            c.flush()

            errors = []
            results = []

            def reader():
                try:
                    for _ in range(50):
                        r = c.execute("SELECT COUNT(*) as cnt FROM stress").to_dict()
                        results.append(r[0]['cnt'])
                except Exception as e:
                    errors.append(str(e))

            threads = [threading.Thread(target=reader) for _ in range(10)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()

            assert len(errors) == 0, f"Errors: {errors}"
            assert all(r >= 500 for r in results)
            c.close()

    def test_concurrent_write_then_read(self):
        """5 writer threads + 5 reader threads simultaneously."""
        with tempfile.TemporaryDirectory() as d:
            c = ApexClient(d)
            c.create_table('wr', {'key': 'string', 'val': 'int'})
            c.use_table('wr')
            c.store([{'key': 'init', 'val': 0}])
            c.flush()

            errors = []
            read_counts = []

            def writer(thread_id):
                try:
                    for i in range(30):
                        c.store([{'key': f't{thread_id}_r{i}', 'val': thread_id * 100 + i}])
                except Exception as e:
                    errors.append(f"Writer {thread_id}: {e}")

            def reader(thread_id):
                try:
                    for _ in range(30):
                        r = c.execute("SELECT COUNT(*) as cnt FROM wr").to_dict()
                        read_counts.append(r[0]['cnt'])
                except Exception as e:
                    errors.append(f"Reader {thread_id}: {e}")

            threads = []
            for i in range(5):
                threads.append(threading.Thread(target=writer, args=(i,)))
                threads.append(threading.Thread(target=reader, args=(i,)))

            for t in threads:
                t.start()
            for t in threads:
                t.join()

            assert len(errors) == 0, f"Errors: {errors}"
            # All reads should see at least the initial row
            assert all(r >= 1 for r in read_counts), f"Read counts: {min(read_counts)}"
            # Final count should include all writes
            final = c.count_rows()
            assert final == 1 + 5 * 30, f"Expected {1 + 5 * 30}, got {final}"
            c.close()

    def test_concurrent_sql_queries(self):
        """Multiple threads executing different SQL queries concurrently."""
        with tempfile.TemporaryDirectory() as d:
            c = ApexClient(d)
            c.create_table('sql_stress', {'name': 'string', 'age': 'int', 'city': 'string'})
            c.use_table('sql_stress')
            cities = ['NYC', 'LA', 'Chicago', 'Houston', 'Phoenix']
            rows = [{'name': f'user_{i}', 'age': 20 + (i % 50), 'city': cities[i % 5]}
                    for i in range(1000)]
            c.store(rows)
            c.flush()

            errors = []

            queries = [
                "SELECT COUNT(*) as cnt FROM sql_stress",
                "SELECT * FROM sql_stress WHERE city = 'NYC' LIMIT 10",
                "SELECT city, COUNT(*) as cnt FROM sql_stress GROUP BY city",
                "SELECT * FROM sql_stress ORDER BY age LIMIT 5",
                "SELECT AVG(age) as avg_age FROM sql_stress",
                "SELECT * FROM sql_stress WHERE age BETWEEN 25 AND 35 LIMIT 20",
            ]

            def query_runner(query_idx):
                try:
                    q = queries[query_idx % len(queries)]
                    for _ in range(20):
                        result = c.execute(q)
                        _ = result.to_dict()
                except Exception as e:
                    errors.append(f"Query {query_idx}: {e}")

            with ThreadPoolExecutor(max_workers=8) as pool:
                futures = [pool.submit(query_runner, i) for i in range(24)]
                for f in as_completed(futures):
                    f.result()

            assert len(errors) == 0, f"Errors: {errors}"
            c.close()

    def test_concurrent_insert_sql(self):
        """Multiple threads doing SQL INSERTs concurrently."""
        with tempfile.TemporaryDirectory() as d:
            c = ApexClient(d)
            c.create_table('ins', {'thread_id': 'int', 'seq': 'int'})
            c.use_table('ins')

            errors = []

            def inserter(tid):
                try:
                    for s in range(20):
                        c.execute(f"INSERT INTO ins (thread_id, seq) VALUES ({tid}, {s})")
                except Exception as e:
                    errors.append(f"Inserter {tid}: {e}")

            threads = [threading.Thread(target=inserter, args=(i,)) for i in range(4)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()

            assert len(errors) == 0, f"Errors: {errors}"
            assert c.count_rows() == 4 * 20
            c.close()


class TestConcurrentMultiTable:
    """Test concurrent operations on different tables."""

    def test_concurrent_different_tables(self):
        """Each thread operates on a separate database directory — should not interfere."""
        import os
        with tempfile.TemporaryDirectory() as d:
            n_tables = 4
            errors = []
            clients = []

            # Each thread gets its own subdirectory to avoid registry conflicts
            # (ApexClient registers by db_path; same-path clients force-close previous ones)
            for i in range(n_tables):
                subdir = os.path.join(d, f'db_{i}')
                os.makedirs(subdir)
                cl = ApexClient(subdir)
                cl.create_table('data', {'value': 'int'})
                cl.use_table('data')
                clients.append(cl)

            def table_worker(table_idx, client):
                try:
                    for j in range(50):
                        client.store([{'value': table_idx * 1000 + j}])
                    client.flush()
                except Exception as e:
                    errors.append(f"Table {table_idx}: {e}")

            threads = [threading.Thread(target=table_worker, args=(i, clients[i])) for i in range(n_tables)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()

            assert len(errors) == 0, f"Errors: {errors}"
            for i in range(n_tables):
                assert clients[i].count_rows() == 50, f"DB {i}: expected 50, got {clients[i].count_rows()}"
            for cl in clients:
                cl.close()


class TestConcurrentTransactions:
    """Test concurrent transaction operations."""

    def test_sequential_transactions(self):
        """Multiple transactions executed one after another — correctness check."""
        with tempfile.TemporaryDirectory() as d:
            c = ApexClient(d)
            c.create_table('txn_seq', {'counter': 'int'})
            c.use_table('txn_seq')

            for i in range(10):
                c.execute('BEGIN')
                c.execute(f"INSERT INTO txn_seq (counter) VALUES ({i})")
                if i % 3 == 0:
                    c.execute('ROLLBACK')
                else:
                    c.execute('COMMIT')

            # Should have 10 - 4 (rolled back: 0, 3, 6, 9) = 6 rows
            assert c.count_rows() == 6
            c.close()

    def test_concurrent_reads_during_transaction(self):
        """Reads from other threads while a transaction is in progress."""
        with tempfile.TemporaryDirectory() as d:
            c = ApexClient(d)
            c.create_table('txn_conc', {'value': 'int'})
            c.use_table('txn_conc')
            c.store([{'value': i} for i in range(100)])
            c.flush()

            errors = []
            read_results = []

            def do_transaction():
                try:
                    c.execute('BEGIN')
                    for i in range(10):
                        c.execute(f"INSERT INTO txn_conc (value) VALUES ({1000 + i})")
                    time.sleep(0.01)  # Hold transaction briefly
                    c.execute('COMMIT')
                except Exception as e:
                    errors.append(f"Txn: {e}")

            def do_reads():
                try:
                    for _ in range(30):
                        r = c.execute("SELECT COUNT(*) as cnt FROM txn_conc").to_dict()
                        read_results.append(r[0]['cnt'])
                        time.sleep(0.001)
                except Exception as e:
                    errors.append(f"Read: {e}")

            t_txn = threading.Thread(target=do_transaction)
            t_reads = [threading.Thread(target=do_reads) for _ in range(3)]

            for t in t_reads:
                t.start()
            t_txn.start()
            t_txn.join()
            for t in t_reads:
                t.join()

            assert len(errors) == 0, f"Errors: {errors}"
            # All reads should see >= 100 (initial rows)
            assert all(r >= 100 for r in read_results)
            c.close()


class TestDataIntegrity:
    """Verify data integrity under concurrent load."""

    def test_no_lost_writes(self):
        """All writes from all threads should be present after completion."""
        with tempfile.TemporaryDirectory() as d:
            c = ApexClient(d)
            c.create_table('integrity', {'thread': 'int', 'seq': 'int'})
            c.use_table('integrity')

            n_threads = 6
            n_writes = 50
            errors = []

            def writer(tid):
                try:
                    for s in range(n_writes):
                        c.store([{'thread': tid, 'seq': s}])
                except Exception as e:
                    errors.append(f"Writer {tid}: {e}")

            threads = [threading.Thread(target=writer, args=(i,)) for i in range(n_threads)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()

            assert len(errors) == 0, f"Errors: {errors}"
            assert c.count_rows() == n_threads * n_writes

            # Verify each thread's data is complete
            for tid in range(n_threads):
                df = c.execute(f"SELECT COUNT(*) as cnt FROM integrity WHERE thread = {tid}").to_dict()
                assert df[0]['cnt'] == n_writes, \
                    f"Thread {tid}: expected {n_writes}, got {df[0]['cnt']}"
            c.close()

    def test_concurrent_read_consistency(self):
        """Reads should always return consistent snapshots."""
        with tempfile.TemporaryDirectory() as d:
            c = ApexClient(d)
            c.create_table('snap', {'a': 'int', 'b': 'int'})
            c.use_table('snap')
            # Insert rows where a + b always equals 100
            c.store([{'a': i, 'b': 100 - i} for i in range(200)])
            c.flush()

            errors = []

            def checker():
                try:
                    for _ in range(30):
                        df = c.execute("SELECT a, b FROM snap LIMIT 50").to_pandas()
                        sums = df['a'] + df['b']
                        if not (sums == 100).all():
                            errors.append(f"Inconsistent snapshot: sums={sums.unique()}")
                except Exception as e:
                    errors.append(f"Checker: {e}")

            threads = [threading.Thread(target=checker) for _ in range(5)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()

            assert len(errors) == 0, f"Errors: {errors}"
            c.close()

    def test_throughput_measurement(self):
        """Measure concurrent read throughput (informational)."""
        with tempfile.TemporaryDirectory() as d:
            c = ApexClient(d)
            c.create_table('perf', {'x': 'int', 'y': 'float', 'z': 'string'})
            c.use_table('perf')
            c.store([{'x': i, 'y': float(i) * 1.5, 'z': f'str_{i}'} for i in range(10000)])
            c.flush()

            n_threads = 4
            n_queries = 50
            results = []

            def bench_reader():
                local_count = 0
                for _ in range(n_queries):
                    r = c.execute("SELECT COUNT(*) as cnt FROM perf").to_dict()
                    local_count += 1
                results.append(local_count)

            start = time.time()
            threads = [threading.Thread(target=bench_reader) for _ in range(n_threads)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()
            elapsed = time.time() - start

            total_queries = sum(results)
            qps = total_queries / elapsed if elapsed > 0 else 0

            assert total_queries == n_threads * n_queries
            # Should complete reasonably fast (> 100 queries/sec)
            assert qps > 100, f"Throughput too low: {qps:.0f} queries/sec"
            c.close()
