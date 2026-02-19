#!/usr/bin/env python3
"""
ApexBase Performance Benchmark
Tests the performance improvements from optimization work:
1. String filter optimization (fast rejection + pointer scanning)
2. GROUP BY integer key optimization (direct counting arrays)
3. Storage cache LRU
4. Vec preallocation
"""

import time
import random
import string
import statistics
import tempfile
import shutil
from pathlib import Path

from apexbase.client import ApexClient

# Configuration
NUM_ROWS = 1_000_000  # 1M rows
NUM_WARMUP = 2
NUM_ITERATIONS = 5
CATEGORIES = ["cat_A", "cat_B", "cat_C", "cat_D", "cat_E", 
              "cat_F", "cat_G", "cat_H", "cat_I", "cat_J"]  # 10 categories


def generate_test_data(num_rows: int) -> list:
    """Generate test data with various column types for benchmarking."""
    print(f"Generating {num_rows:,} rows of test data...")
    start = time.perf_counter()
    
    data = []
    for i in range(num_rows):
        data.append({
            "id": i,
            "category": random.choice(CATEGORIES),
            "group_id": random.randint(0, 99),  # 100 distinct values for GROUP BY
            "value": random.randint(1, 1000),
            "price": round(random.uniform(10.0, 1000.0), 2),
            "name": f"item_{i % 10000}",  # 10K distinct names
            "status": random.choice(["active", "inactive", "pending"]),
        })
    
    elapsed = time.perf_counter() - start
    print(f"  Generated in {elapsed:.2f}s")
    return data


def benchmark_query(client, sql: str, name: str, warmup: int = NUM_WARMUP, iterations: int = NUM_ITERATIONS):
    """Run a benchmark for a single query."""
    # Warmup runs
    for _ in range(warmup):
        client.execute(sql)
    
    # Timed runs
    times = []
    for _ in range(iterations):
        start = time.perf_counter()
        result = client.execute(sql)
        elapsed = time.perf_counter() - start
        times.append(elapsed * 1000)  # Convert to ms
        _ = len(result)  # Ensure result is materialized
    
    avg_ms = statistics.mean(times)
    std_ms = statistics.stdev(times) if len(times) > 1 else 0
    min_ms = min(times)
    max_ms = max(times)
    
    return {
        "name": name,
        "avg_ms": avg_ms,
        "std_ms": std_ms,
        "min_ms": min_ms,
        "max_ms": max_ms,
        "iterations": iterations,
    }


def run_benchmarks(client) -> list:
    """Run all benchmark queries."""
    results = []
    
    print("\n" + "="*70)
    print("Running Benchmarks...")
    print("="*70)
    
    # =========================================================================
    # 1. String Filter Benchmarks (P0 optimization)
    # =========================================================================
    print("\n[1] String Filter Benchmarks")
    print("-"*50)
    
    # Simple string equality with LIMIT
    sql = "SELECT * FROM test WHERE category = 'cat_A' LIMIT 100"
    r = benchmark_query(client, sql, "String filter + LIMIT")
    print(f"  {r['name']}: {r['avg_ms']:.2f}ms (±{r['std_ms']:.2f})")
    results.append(r)
    
    # String equality without LIMIT (full scan)
    sql = "SELECT COUNT(*) FROM test WHERE category = 'cat_A'"
    r = benchmark_query(client, sql, "String filter COUNT(*)")
    print(f"  {r['name']}: {r['avg_ms']:.2f}ms (±{r['std_ms']:.2f})")
    results.append(r)
    
    # String NOT EQUAL
    sql = "SELECT COUNT(*) FROM test WHERE category != 'cat_A'"
    r = benchmark_query(client, sql, "String != filter")
    print(f"  {r['name']}: {r['avg_ms']:.2f}ms (±{r['std_ms']:.2f})")
    results.append(r)
    
    # =========================================================================
    # 2. GROUP BY Benchmarks (P0 optimization)
    # =========================================================================
    print("\n[2] GROUP BY Benchmarks")
    print("-"*50)
    
    # Integer GROUP BY (should use direct counting)
    sql = "SELECT group_id, COUNT(*) FROM test GROUP BY group_id"
    r = benchmark_query(client, sql, "GROUP BY int (100 groups)")
    print(f"  {r['name']}: {r['avg_ms']:.2f}ms (±{r['std_ms']:.2f})")
    results.append(r)
    
    # Integer GROUP BY with SUM
    sql = "SELECT group_id, SUM(value) FROM test GROUP BY group_id"
    r = benchmark_query(client, sql, "GROUP BY int + SUM")
    print(f"  {r['name']}: {r['avg_ms']:.2f}ms (±{r['std_ms']:.2f})")
    results.append(r)
    
    # Integer GROUP BY with AVG
    sql = "SELECT group_id, AVG(price) FROM test GROUP BY group_id"
    r = benchmark_query(client, sql, "GROUP BY int + AVG")
    print(f"  {r['name']}: {r['avg_ms']:.2f}ms (±{r['std_ms']:.2f})")
    results.append(r)
    
    # String GROUP BY (uses hash aggregation)
    sql = "SELECT category, COUNT(*) FROM test GROUP BY category"
    r = benchmark_query(client, sql, "GROUP BY string (10 groups)")
    print(f"  {r['name']}: {r['avg_ms']:.2f}ms (±{r['std_ms']:.2f})")
    results.append(r)
    
    # String GROUP BY with SUM
    sql = "SELECT category, SUM(value) FROM test GROUP BY category"
    r = benchmark_query(client, sql, "GROUP BY string + SUM")
    print(f"  {r['name']}: {r['avg_ms']:.2f}ms (±{r['std_ms']:.2f})")
    results.append(r)
    
    # GROUP BY with HAVING
    sql = "SELECT group_id, COUNT(*) as cnt FROM test GROUP BY group_id HAVING cnt > 5000"
    r = benchmark_query(client, sql, "GROUP BY + HAVING")
    print(f"  {r['name']}: {r['avg_ms']:.2f}ms (±{r['std_ms']:.2f})")
    results.append(r)
    
    # =========================================================================
    # 3. Aggregation Benchmarks
    # =========================================================================
    print("\n[3] Aggregation Benchmarks")
    print("-"*50)
    
    # Simple COUNT(*)
    sql = "SELECT COUNT(*) FROM test"
    r = benchmark_query(client, sql, "COUNT(*)")
    print(f"  {r['name']}: {r['avg_ms']:.2f}ms (±{r['std_ms']:.2f})")
    results.append(r)
    
    # SUM
    sql = "SELECT SUM(value) FROM test"
    r = benchmark_query(client, sql, "SUM(value)")
    print(f"  {r['name']}: {r['avg_ms']:.2f}ms (±{r['std_ms']:.2f})")
    results.append(r)
    
    # AVG
    sql = "SELECT AVG(price) FROM test"
    r = benchmark_query(client, sql, "AVG(price)")
    print(f"  {r['name']}: {r['avg_ms']:.2f}ms (±{r['std_ms']:.2f})")
    results.append(r)
    
    # MIN/MAX
    sql = "SELECT MIN(value), MAX(value) FROM test"
    r = benchmark_query(client, sql, "MIN/MAX")
    print(f"  {r['name']}: {r['avg_ms']:.2f}ms (±{r['std_ms']:.2f})")
    results.append(r)
    
    # =========================================================================
    # 4. Range Filter Benchmarks
    # =========================================================================
    print("\n[4] Range Filter Benchmarks")
    print("-"*50)
    
    # BETWEEN
    sql = "SELECT COUNT(*) FROM test WHERE value BETWEEN 100 AND 200"
    r = benchmark_query(client, sql, "BETWEEN filter")
    print(f"  {r['name']}: {r['avg_ms']:.2f}ms (±{r['std_ms']:.2f})")
    results.append(r)
    
    # Comparison
    sql = "SELECT COUNT(*) FROM test WHERE price > 500"
    r = benchmark_query(client, sql, "Numeric > filter")
    print(f"  {r['name']}: {r['avg_ms']:.2f}ms (±{r['std_ms']:.2f})")
    results.append(r)
    
    # =========================================================================
    # 5. Complex Query Benchmarks
    # =========================================================================
    print("\n[5] Complex Query Benchmarks")
    print("-"*50)
    
    # Filter + Group + Order
    sql = """
        SELECT category, SUM(value) as total 
        FROM test 
        WHERE status = 'active' 
        GROUP BY category 
        ORDER BY total DESC 
        LIMIT 5
    """
    r = benchmark_query(client, sql, "Filter+Group+Order+Limit")
    print(f"  {r['name']}: {r['avg_ms']:.2f}ms (±{r['std_ms']:.2f})")
    results.append(r)
    
    # Multiple conditions
    sql = """
        SELECT COUNT(*) 
        FROM test 
        WHERE category = 'cat_A' AND value > 500
    """
    r = benchmark_query(client, sql, "Multi-condition filter")
    print(f"  {r['name']}: {r['avg_ms']:.2f}ms (±{r['std_ms']:.2f})")
    results.append(r)
    
    # ORDER BY + LIMIT (top-k)
    sql = "SELECT * FROM test ORDER BY price DESC LIMIT 100"
    r = benchmark_query(client, sql, "ORDER BY + LIMIT 100")
    print(f"  {r['name']}: {r['avg_ms']:.2f}ms (±{r['std_ms']:.2f})")
    results.append(r)
    
    # DISTINCT
    sql = "SELECT DISTINCT category FROM test"
    r = benchmark_query(client, sql, "DISTINCT")
    print(f"  {r['name']}: {r['avg_ms']:.2f}ms (±{r['std_ms']:.2f})")
    results.append(r)
    
    # =========================================================================
    # 6. Cache Performance (repeated queries)
    # =========================================================================
    print("\n[6] Cache Performance (repeated queries)")
    print("-"*50)
    
    # First query (cold)
    sql = "SELECT * FROM test WHERE group_id = 50 LIMIT 100"
    start = time.perf_counter()
    client.execute(sql)
    cold_ms = (time.perf_counter() - start) * 1000
    
    # Subsequent queries (warm)
    warm_times = []
    for _ in range(5):
        start = time.perf_counter()
        client.execute(sql)
        warm_times.append((time.perf_counter() - start) * 1000)
    
    warm_avg = statistics.mean(warm_times)
    speedup = cold_ms / warm_avg if warm_avg > 0 else 1
    
    print(f"  Cold query: {cold_ms:.2f}ms")
    print(f"  Warm query (avg): {warm_avg:.2f}ms")
    print(f"  Cache speedup: {speedup:.1f}x")
    
    results.append({
        "name": "Cache cold",
        "avg_ms": cold_ms,
        "std_ms": 0,
        "min_ms": cold_ms,
        "max_ms": cold_ms,
        "iterations": 1,
    })
    results.append({
        "name": "Cache warm",
        "avg_ms": warm_avg,
        "std_ms": statistics.stdev(warm_times),
        "min_ms": min(warm_times),
        "max_ms": max(warm_times),
        "iterations": 5,
    })
    
    return results


def print_summary(results: list, num_rows: int):
    """Print a summary report of all benchmarks."""
    print("\n" + "="*70)
    print("BENCHMARK SUMMARY")
    print(f"Dataset: {num_rows:,} rows")
    print("="*70)
    
    print(f"\n{'Query':<35} {'Avg (ms)':<12} {'Min (ms)':<12} {'Throughput':<15}")
    print("-"*70)
    
    for r in results:
        rows_per_sec = num_rows / (r['avg_ms'] / 1000) if r['avg_ms'] > 0 else 0
        throughput = f"{rows_per_sec/1e6:.2f}M rows/s"
        print(f"{r['name']:<35} {r['avg_ms']:<12.2f} {r['min_ms']:<12.2f} {throughput:<15}")
    
    print("\n" + "="*70)
    
    # Highlight key metrics
    string_filter = next((r for r in results if r['name'] == "String filter + LIMIT"), None)
    group_by_int = next((r for r in results if r['name'] == "GROUP BY int (100 groups)"), None)
    count_star = next((r for r in results if r['name'] == "COUNT(*)"), None)
    
    print("\nKey Performance Metrics:")
    if string_filter:
        print(f"  • String filter + LIMIT: {string_filter['avg_ms']:.2f}ms")
    if group_by_int:
        print(f"  • GROUP BY (100 groups): {group_by_int['avg_ms']:.2f}ms")
    if count_star:
        print(f"  • COUNT(*): {count_star['avg_ms']:.2f}ms ({num_rows/(count_star['avg_ms']/1000)/1e6:.1f}M rows/s)")


def main():
    print("="*70)
    print("ApexBase Performance Benchmark")
    print("="*70)
    
    # Create temporary directory for test database
    tmp_dir = tempfile.mkdtemp(prefix="apexbase_bench_")
    print(f"Using temp directory: {tmp_dir}")
    
    try:
        # Initialize client
        client = ApexClient(tmp_dir)
        client.create_table("test")
        
        # Generate and store test data
        data = generate_test_data(NUM_ROWS)
        
        print(f"Storing {NUM_ROWS:,} rows...")
        start = time.perf_counter()
        client.store(data)
        store_time = time.perf_counter() - start
        print(f"  Stored in {store_time:.2f}s ({NUM_ROWS/store_time:,.0f} rows/s)")
        
        # Run benchmarks
        results = run_benchmarks(client)
        
        # Print summary
        print_summary(results, NUM_ROWS)
        
        # Close client
        client.close()
        
    finally:
        # Cleanup
        shutil.rmtree(tmp_dir, ignore_errors=True)
        print(f"\nCleaned up temp directory: {tmp_dir}")


if __name__ == "__main__":
    main()
