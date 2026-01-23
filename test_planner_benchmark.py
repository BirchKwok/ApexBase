#!/usr/bin/env python3
"""
Performance benchmark for DataFusion-inspired query planner.
Tests: Filter, Aggregate, JOIN, ORDER BY, LIMIT operations on 1M rows.
"""

import time
import tempfile
import os
import shutil
from apexbase import ApexClient

def create_test_data(client, num_rows=1_000_000):
    """Create test data with 1M rows."""
    print(f"Creating {num_rows:,} rows...")
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
    client.flush()
    
    print(f"Created {num_rows:,} rows")

def benchmark_query(client, name, sql, iterations=3):
    """Benchmark a query and return average time."""
    times = []
    for i in range(iterations):
        start = time.perf_counter()
        result = client.execute(sql)
        elapsed = time.perf_counter() - start
        times.append(elapsed)
    
    avg_time = sum(times) / len(times)
    min_time = min(times)
    print(f"  {name}: avg={avg_time*1000:.2f}ms, min={min_time*1000:.2f}ms")
    return avg_time

def main():
    # Create temp directory
    temp_dir = tempfile.mkdtemp(prefix="planner_bench_")
    
    try:
        print("=" * 60)
        print("DataFusion-Inspired Query Planner Benchmark")
        print("=" * 60)
        
        # Create client and data using dirpath approach (like test_benchmark_1m.py)
        client = ApexClient(dirpath=temp_dir, drop_if_exists=True)
        # Store data first (auto-creates default table)
        create_test_data(client, num_rows=1_000_000)
        
        print("\n--- Query Performance (1M rows) ---\n")
        
        results = {}
        
        # 1. Simple filter
        results["filter_simple"] = benchmark_query(
            client, "Simple Filter (status='active')",
            "SELECT * FROM default WHERE status = 'active' LIMIT 100"
        )
        
        # 2. Range filter
        results["filter_range"] = benchmark_query(
            client, "Range Filter (value BETWEEN)",
            "SELECT * FROM default WHERE value >= 100 AND value < 200 LIMIT 100"
        )
        
        # 3. COUNT aggregate
        results["count_all"] = benchmark_query(
            client, "COUNT(*)",
            "SELECT COUNT(*) FROM default"
        )
        
        # 4. GROUP BY aggregate
        results["group_by"] = benchmark_query(
            client, "GROUP BY (100 groups)",
            "SELECT category, COUNT(*), AVG(value) FROM default GROUP BY category"
        )
        
        # 5. ORDER BY with LIMIT
        results["order_limit"] = benchmark_query(
            client, "ORDER BY + LIMIT 10",
            "SELECT * FROM default ORDER BY value DESC LIMIT 10"
        )
        
        # 6. Complex query: Filter + Group + Order
        results["complex"] = benchmark_query(
            client, "Complex: Filter + Group + Order",
            "SELECT region, SUM(value) as total FROM default WHERE status = 'active' GROUP BY region ORDER BY total DESC LIMIT 5"
        )
        
        # Additional benchmark cases
        print("\n--- Extended Benchmarks ---\n")
        
        # 7. Multiple column GROUP BY
        results["group_by_multi"] = benchmark_query(
            client, "GROUP BY (2 cols, 1000 groups)",
            "SELECT category, region, COUNT(*) FROM default GROUP BY category, region"
        )
        
        # 8. String equality filter (common pattern)
        results["string_eq"] = benchmark_query(
            client, "String = 'value' filter",
            "SELECT _id, category FROM default WHERE category = 'cat_50' LIMIT 100"
        )
        
        # 9. AVG aggregate
        results["avg_agg"] = benchmark_query(
            client, "AVG aggregate",
            "SELECT AVG(value) FROM default"
        )
        
        # 10. Multi-condition WHERE
        results["multi_where"] = benchmark_query(
            client, "Multi-condition WHERE",
            "SELECT * FROM default WHERE status = 'active' AND value > 500 LIMIT 100"
        )
        
        # 11. Large LIMIT (test projection overhead)
        results["large_limit"] = benchmark_query(
            client, "Large LIMIT (10000 rows)",
            "SELECT * FROM default LIMIT 10000"
        )
        
        # 12. Specific column projection
        results["projection"] = benchmark_query(
            client, "Column projection (3 cols)",
            "SELECT _id, category, value FROM default LIMIT 1000"
        )
        
        # Summary
        print("\n--- Summary ---")
        print(f"Total rows: 1,000,000")
        print(f"Simple filter:    {results['filter_simple']*1000:.2f}ms")
        print(f"Range filter:     {results['filter_range']*1000:.2f}ms")
        print(f"COUNT(*):         {results['count_all']*1000:.2f}ms")
        print(f"GROUP BY (100):   {results['group_by']*1000:.2f}ms")
        print(f"ORDER + LIMIT:    {results['order_limit']*1000:.2f}ms")
        print(f"Complex query:    {results['complex']*1000:.2f}ms")
        
        return results
        
    finally:
        # Cleanup
        shutil.rmtree(temp_dir, ignore_errors=True)

if __name__ == "__main__":
    main()
