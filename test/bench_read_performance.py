#!/usr/bin/env python3
"""
Benchmark script for ApexBase read performance.
Tests both sequential and random read patterns.
"""

import time
import random
import tempfile
import shutil
import sys
import os
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'apexbase', 'python'))
from apexbase import ApexClient

def generate_test_data(n_rows: int):
    """Generate test data with mixed column types."""
    return [
        {
            "int_col": i,
            "float_col": float(i) * 1.5,
            "str_col": f"value_{i % 1000}",  # Low cardinality for dict encoding
            "str_col2": f"unique_{i}",       # High cardinality
        }
        for i in range(n_rows)
    ]

def benchmark_sequential_read(client, table_name: str, n_iterations: int = 10):
    """Benchmark sequential full table scan."""
    times = []
    for _ in range(n_iterations):
        start = time.perf_counter()
        result = client.execute(f"SELECT * FROM {table_name}")
        _ = result.to_pandas()
        elapsed = time.perf_counter() - start
        times.append(elapsed)
    
    avg_time = sum(times) / len(times)
    min_time = min(times)
    max_time = max(times)
    return {"avg": avg_time, "min": min_time, "max": max_time, "times": times}

def benchmark_column_projection(client, table_name: str, n_iterations: int = 10):
    """Benchmark reading specific columns only."""
    times = []
    for _ in range(n_iterations):
        start = time.perf_counter()
        result = client.execute(f"SELECT int_col, str_col FROM {table_name}")
        _ = result.to_pandas()
        elapsed = time.perf_counter() - start
        times.append(elapsed)
    
    avg_time = sum(times) / len(times)
    min_time = min(times)
    return {"avg": avg_time, "min": min_time}

def benchmark_filter_int(client, table_name: str, n_rows: int, n_iterations: int = 10):
    """Benchmark integer filter (range scan)."""
    mid = n_rows // 2
    times = []
    for _ in range(n_iterations):
        start = time.perf_counter()
        result = client.execute(f"SELECT * FROM {table_name} WHERE int_col >= {mid} AND int_col < {mid + 1000}")
        _ = result.to_pandas()
        elapsed = time.perf_counter() - start
        times.append(elapsed)
    
    avg_time = sum(times) / len(times)
    min_time = min(times)
    return {"avg": avg_time, "min": min_time}

def benchmark_filter_string(client, table_name: str, n_iterations: int = 10):
    """Benchmark string equality filter (dict-encoded)."""
    times = []
    for _ in range(n_iterations):
        start = time.perf_counter()
        result = client.execute(f"SELECT * FROM {table_name} WHERE str_col = 'value_500'")
        _ = result.to_pandas()
        elapsed = time.perf_counter() - start
        times.append(elapsed)
    
    avg_time = sum(times) / len(times)
    min_time = min(times)
    return {"avg": avg_time, "min": min_time}

def benchmark_random_access(client, table_name: str, n_rows: int, n_samples: int = 100):
    """Benchmark random row access via _id."""
    # Get all IDs first
    ids_result = client.execute(f"SELECT _id FROM {table_name}")
    all_ids = [row["_id"] for row in ids_result]
    
    # Random sample of IDs
    sample_ids = random.sample(all_ids, min(n_samples, len(all_ids)))
    
    times = []
    for id_val in sample_ids:
        start = time.perf_counter()
        result = client.execute(f"SELECT * FROM {table_name} WHERE _id = {id_val}")
        _ = list(result)
        elapsed = time.perf_counter() - start
        times.append(elapsed)
    
    avg_time = sum(times) / len(times)
    min_time = min(times)
    max_time = max(times)
    return {"avg": avg_time, "min": min_time, "max": max_time, "total": sum(times)}

def benchmark_limit_query(client, table_name: str, n_iterations: int = 10):
    """Benchmark LIMIT query with filter."""
    times = []
    for _ in range(n_iterations):
        start = time.perf_counter()
        result = client.execute(f"SELECT * FROM {table_name} WHERE str_col = 'value_100' LIMIT 10")
        _ = list(result)
        elapsed = time.perf_counter() - start
        times.append(elapsed)
    
    avg_time = sum(times) / len(times)
    min_time = min(times)
    return {"avg": avg_time, "min": min_time}

def benchmark_aggregation(client, table_name: str, n_iterations: int = 10):
    """Benchmark aggregation query."""
    times = []
    for _ in range(n_iterations):
        start = time.perf_counter()
        result = client.execute(f"SELECT str_col, COUNT(*), SUM(int_col), AVG(float_col) FROM {table_name} GROUP BY str_col")
        _ = result.to_pandas()
        elapsed = time.perf_counter() - start
        times.append(elapsed)
    
    avg_time = sum(times) / len(times)
    min_time = min(times)
    return {"avg": avg_time, "min": min_time}

def run_benchmarks(n_rows: int = 100_000):
    """Run all benchmarks."""
    print(f"\n{'='*60}")
    print(f"ApexBase Read Performance Benchmark")
    print(f"Rows: {n_rows:,}")
    print(f"{'='*60}\n")
    
    # Create temp directory
    temp_dir = tempfile.mkdtemp(prefix="apexbase_bench_")
    try:
        # Initialize client and create data
        client = ApexClient(temp_dir)
        table_name = "bench_table"
        client.create_table(table_name)
        
        print("Generating test data...")
        start = time.perf_counter()
        data = generate_test_data(n_rows)
        gen_time = time.perf_counter() - start
        print(f"  Data generation: {gen_time:.2f}s")
        
        print("Inserting data...")
        start = time.perf_counter()
        client.store(data)
        insert_time = time.perf_counter() - start
        print(f"  Insert time: {insert_time:.2f}s ({n_rows/insert_time:,.0f} rows/sec)")
        
        print("\n--- Sequential Read Benchmarks ---")
        
        # Full table scan
        result = benchmark_sequential_read(client, table_name)
        rows_per_sec = n_rows / result["avg"]
        print(f"Full table scan (SELECT *):     avg={result['avg']*1000:.2f}ms  min={result['min']*1000:.2f}ms  ({rows_per_sec:,.0f} rows/sec)")
        
        # Column projection
        result = benchmark_column_projection(client, table_name)
        rows_per_sec = n_rows / result["avg"]
        print(f"Column projection (2 cols):     avg={result['avg']*1000:.2f}ms  min={result['min']*1000:.2f}ms  ({rows_per_sec:,.0f} rows/sec)")
        
        # Integer filter
        result = benchmark_filter_int(client, table_name, n_rows)
        print(f"Integer range filter (1000 rows): avg={result['avg']*1000:.2f}ms  min={result['min']*1000:.2f}ms")
        
        # String filter
        result = benchmark_filter_string(client, table_name)
        print(f"String equality filter:         avg={result['avg']*1000:.2f}ms  min={result['min']*1000:.2f}ms")
        
        # LIMIT query
        result = benchmark_limit_query(client, table_name)
        print(f"String filter + LIMIT 10:       avg={result['avg']*1000:.2f}ms  min={result['min']*1000:.2f}ms")
        
        # Aggregation
        result = benchmark_aggregation(client, table_name)
        print(f"GROUP BY aggregation:           avg={result['avg']*1000:.2f}ms  min={result['min']*1000:.2f}ms")
        
        print("\n--- Random Access Benchmarks ---")
        
        # Random single row access
        result = benchmark_random_access(client, table_name, n_rows, n_samples=100)
        print(f"Random single row by _id (100x): avg={result['avg']*1000:.3f}ms  total={result['total']*1000:.1f}ms")
        
        client.close()
        
        print(f"\n{'='*60}")
        print("Benchmark complete!")
        print(f"{'='*60}\n")
        
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

if __name__ == "__main__":
    # Run with 100K rows for quick baseline
    run_benchmarks(n_rows=100_000)
    
    # Run with 1M rows for more realistic test
    print("\n" + "="*60)
    print("Running larger benchmark (1M rows)...")
    print("="*60)
    run_benchmarks(n_rows=1_000_000)
