#!/usr/bin/env python3
"""
ApexBase vs DuckDB Performance Comparison Benchmark

Runs identical queries on both databases to identify optimization opportunities.
"""

import time
import tempfile
import os

# Check if duckdb is installed
try:
    import duckdb
    HAS_DUCKDB = True
except ImportError:
    HAS_DUCKDB = False
    print("Warning: duckdb not installed. Run: pip install duckdb")

from apexbase import ApexClient


def create_test_data_apex(client, num_rows=1_000_000):
    """Create test data for ApexBase."""
    print(f"Creating {num_rows:,} rows in ApexBase...")
    batch_size = 100_000
    
    for batch in range(num_rows // batch_size):
        docs = []
        for i in range(batch_size):
            row_id = batch * batch_size + i
            docs.append({
                "_id": f"doc_{row_id}",
                "category": f"cat_{row_id % 100}",
                "value": float(row_id % 1000),
                "status": "active" if row_id % 2 == 0 else "inactive",
                "region": f"region_{row_id % 10}",
            })
        client.store(docs)
    
    # Force flush to disk to trigger dictionary encoding for low-cardinality columns
    client.flush()
    print(f"Created {num_rows:,} rows in ApexBase (with dictionary encoding)")


def create_test_data_duckdb(conn, num_rows=1_000_000):
    """Create test data for DuckDB."""
    print(f"Creating {num_rows:,} rows in DuckDB...")
    
    # Create table
    conn.execute("""
        CREATE TABLE IF NOT EXISTS test_data (
            _id VARCHAR,
            category VARCHAR,
            value DOUBLE,
            status VARCHAR,
            region VARCHAR
        )
    """)
    
    # Insert data in batches using SQL generation
    batch_size = 100_000
    for batch in range(num_rows // batch_size):
        conn.execute(f"""
            INSERT INTO test_data
            SELECT 
                'doc_' || (i + {batch * batch_size}) as _id,
                'cat_' || ((i + {batch * batch_size}) % 100) as category,
                CAST(((i + {batch * batch_size}) % 1000) AS DOUBLE) as value,
                CASE WHEN ((i + {batch * batch_size}) % 2) = 0 THEN 'active' ELSE 'inactive' END as status,
                'region_' || ((i + {batch * batch_size}) % 10) as region
            FROM range(0, {batch_size}) t(i)
        """)
    
    print(f"Created {num_rows:,} rows in DuckDB")


def benchmark_query(name, apex_func, duckdb_func, iterations=3, duckdb_path=None, duckdb_query=None):
    """Benchmark a query on both databases.
    
    If duckdb_path and duckdb_query are provided, reopen connection for each iteration
    to simulate cold-start disk I/O (no caching).
    """
    # ApexBase
    apex_times = []
    for _ in range(iterations):
        start = time.perf_counter()
        apex_func()
        apex_times.append(time.perf_counter() - start)
    apex_avg = sum(apex_times) / len(apex_times) * 1000
    apex_min = min(apex_times) * 1000
    
    # DuckDB
    duck_times = []
    if HAS_DUCKDB:
        if duckdb_path and duckdb_query:
            # Cold-start mode: reopen connection for each iteration
            for _ in range(iterations):
                conn = duckdb.connect(duckdb_path)
                start = time.perf_counter()
                conn.execute(duckdb_query).fetchall()
                duck_times.append(time.perf_counter() - start)
                conn.close()
        elif duckdb_func:
            for _ in range(iterations):
                start = time.perf_counter()
                duckdb_func()
                duck_times.append(time.perf_counter() - start)
        
        if duck_times:
            duck_avg = sum(duck_times) / len(duck_times) * 1000
            duck_min = min(duck_times) * 1000
            ratio = apex_avg / duck_avg if duck_avg > 0 else float('inf')
        else:
            duck_avg = duck_min = 0
            ratio = 0
    else:
        duck_avg = duck_min = 0
        ratio = 0
    
    return {
        "name": name,
        "apex_avg": apex_avg,
        "apex_min": apex_min,
        "duck_avg": duck_avg,
        "duck_min": duck_min,
        "ratio": ratio,
    }


def print_results(results):
    """Print benchmark results in a formatted table."""
    print("\n" + "=" * 90)
    print(f"{'Query':<35} {'ApexBase':>12} {'DuckDB':>12} {'Ratio':>10} {'Gap':>15}")
    print(f"{'':35} {'(ms)':>12} {'(ms)':>12} {'(x slower)':>10} {'':>15}")
    print("=" * 90)
    
    for r in results:
        gap = ""
        if r["ratio"] > 5:
            gap = "⚠️ LARGE GAP"
        elif r["ratio"] > 2:
            gap = "⚡ Medium gap"
        elif r["ratio"] < 1.5:
            gap = "✅ Good"
        
        print(f"{r['name']:<35} {r['apex_avg']:>12.2f} {r['duck_avg']:>12.2f} {r['ratio']:>10.1f}x {gap:>15}")
    
    print("=" * 90)
    
    # Summary
    print("\n--- Gap Analysis ---")
    sorted_results = sorted(results, key=lambda x: x["ratio"], reverse=True)
    print("\nLargest gaps (optimization priorities):")
    for i, r in enumerate(sorted_results[:5], 1):
        print(f"  {i}. {r['name']}: {r['ratio']:.1f}x slower than DuckDB")


def main():
    if not HAS_DUCKDB:
        print("DuckDB not available. Please install: pip install duckdb")
        return
    
    print("=" * 70)
    print("ApexBase vs DuckDB Performance Comparison")
    print("=" * 70)
    
    num_rows = 1_000_000
    
    # Setup ApexBase
    with tempfile.TemporaryDirectory() as tmpdir:
        apex_path = os.path.join(tmpdir, "apex_bench")
        client = ApexClient(apex_path)
        create_test_data_apex(client, num_rows)
        
        # Close and reopen to ensure data is loaded from disk (with dictionary encoding)
        del client
        client = ApexClient(apex_path)
        print("Reopened ApexBase from disk (dictionary-encoded columns)")
        
        # Setup DuckDB - use FILE-BASED storage for fair disk I/O comparison
        duckdb_path = os.path.join(tmpdir, "duckdb_bench.db")
        conn = duckdb.connect(duckdb_path)
        create_test_data_duckdb(conn, num_rows)
        
        # Close and reopen to ensure data is flushed to disk
        conn.close()
        conn = duckdb.connect(duckdb_path)
        
        print("\n--- Running Benchmarks (DuckDB cold-start: reopen connection each query) ---\n")
        
        results = []
        
        # 1. Simple string filter
        results.append(benchmark_query(
            "1. String filter (status='active')",
            lambda: client.execute("SELECT * FROM default WHERE status = 'active' LIMIT 100"),
            None, duckdb_path=duckdb_path,
            duckdb_query="SELECT * FROM test_data WHERE status = 'active' LIMIT 100"
        ))
        
        # 2. Range filter
        results.append(benchmark_query(
            "2. Range filter (BETWEEN)",
            lambda: client.execute("SELECT * FROM default WHERE value BETWEEN 100 AND 200 LIMIT 100"),
            None, duckdb_path=duckdb_path,
            duckdb_query="SELECT * FROM test_data WHERE value BETWEEN 100 AND 200 LIMIT 100"
        ))
        
        # 3. COUNT(*)
        results.append(benchmark_query(
            "3. COUNT(*)",
            lambda: client.execute("SELECT COUNT(*) FROM default"),
            None, duckdb_path=duckdb_path,
            duckdb_query="SELECT COUNT(*) FROM test_data"
        ))
        
        # 4. GROUP BY (100 groups)
        results.append(benchmark_query(
            "4. GROUP BY (100 groups)",
            lambda: client.execute("SELECT category, COUNT(*) FROM default GROUP BY category"),
            None, duckdb_path=duckdb_path,
            duckdb_query="SELECT category, COUNT(*) FROM test_data GROUP BY category"
        ))
        
        # 5. ORDER BY + LIMIT
        results.append(benchmark_query(
            "5. ORDER BY + LIMIT 10",
            lambda: client.execute("SELECT * FROM default ORDER BY value DESC LIMIT 10"),
            None, duckdb_path=duckdb_path,
            duckdb_query="SELECT * FROM test_data ORDER BY value DESC LIMIT 10"
        ))
        
        # 6. Complex query
        results.append(benchmark_query(
            "6. Complex (Filter+Group+Order)",
            lambda: client.execute("SELECT region, SUM(value) as total FROM default WHERE status = 'active' GROUP BY region ORDER BY total DESC LIMIT 5"),
            None, duckdb_path=duckdb_path,
            duckdb_query="SELECT region, SUM(value) as total FROM test_data WHERE status = 'active' GROUP BY region ORDER BY total DESC LIMIT 5"
        ))
        
        # 7. Multi-column GROUP BY
        results.append(benchmark_query(
            "7. GROUP BY (2 cols, 1000 groups)",
            lambda: client.execute("SELECT category, region, COUNT(*) FROM default GROUP BY category, region"),
            None, duckdb_path=duckdb_path,
            duckdb_query="SELECT category, region, COUNT(*) FROM test_data GROUP BY category, region"
        ))
        
        # 8. AVG aggregate
        results.append(benchmark_query(
            "8. AVG aggregate",
            lambda: client.execute("SELECT AVG(value) FROM default"),
            None, duckdb_path=duckdb_path,
            duckdb_query="SELECT AVG(value) FROM test_data"
        ))
        
        # 9. Multi-condition WHERE
        results.append(benchmark_query(
            "9. Multi-condition WHERE",
            lambda: client.execute("SELECT * FROM default WHERE status = 'active' AND value > 500 LIMIT 100"),
            None, duckdb_path=duckdb_path,
            duckdb_query="SELECT * FROM test_data WHERE status = 'active' AND value > 500 LIMIT 100"
        ))
        
        # 10. Large LIMIT
        results.append(benchmark_query(
            "10. Large LIMIT (10000 rows)",
            lambda: client.execute("SELECT * FROM default LIMIT 10000"),
            None, duckdb_path=duckdb_path,
            duckdb_query="SELECT * FROM test_data LIMIT 10000"
        ))
        
        # 11. Column projection
        results.append(benchmark_query(
            "11. Column projection (3 cols)",
            lambda: client.execute("SELECT _id, category, value FROM default LIMIT 1000"),
            None, duckdb_path=duckdb_path,
            duckdb_query="SELECT _id, category, value FROM test_data LIMIT 1000"
        ))
        
        # 12. SUM aggregate
        results.append(benchmark_query(
            "12. SUM aggregate",
            lambda: client.execute("SELECT SUM(value) FROM default"),
            None, duckdb_path=duckdb_path,
            duckdb_query="SELECT SUM(value) FROM test_data"
        ))
        
        # 13. MIN/MAX
        results.append(benchmark_query(
            "13. MIN/MAX aggregates",
            lambda: client.execute("SELECT MIN(value), MAX(value) FROM default"),
            None, duckdb_path=duckdb_path,
            duckdb_query="SELECT MIN(value), MAX(value) FROM test_data"
        ))
        
        # 14. DISTINCT
        results.append(benchmark_query(
            "14. DISTINCT categories",
            lambda: client.execute("SELECT DISTINCT category FROM default"),
            None, duckdb_path=duckdb_path,
            duckdb_query="SELECT DISTINCT category FROM test_data"
        ))
        
        # 15. GROUP BY with HAVING
        results.append(benchmark_query(
            "15. GROUP BY + HAVING",
            lambda: client.execute("SELECT category, COUNT(*) as cnt FROM default GROUP BY category HAVING cnt > 5000"),
            None, duckdb_path=duckdb_path,
            duckdb_query="SELECT category, COUNT(*) as cnt FROM test_data GROUP BY category HAVING cnt > 5000"
        ))
        
        print_results(results)
        
        client.close()
        conn.close()


if __name__ == "__main__":
    main()
