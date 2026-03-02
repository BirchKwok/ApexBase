#!/usr/bin/env python3
"""
ApexBase Comprehensive Performance Comparison Benchmark

This script performs a complete performance comparison across:
1. Standard 28 scenario benchmark tests
2. High concurrency stress tests (10/20/50 threads)
3. Large dataset tests (100K/1M rows)
4. Mixed read/write workloads (80% read / 20% write)

Results are output in detailed JSON format with latency distributions.

Usage:
    python benchmarks/full_performance_comparison.py [--rows N] [--output FILE]
    
Examples:
    python benchmarks/full_performance_comparison.py
    python benchmarks/full_performance_comparison.py --rows 100000 --output results.json
"""

import argparse
import gc
import json
import os
import platform
import random
import shutil
import sqlite3
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple
from pathlib import Path

import numpy as np

# Add apexbase to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'apexbase', 'python'))

# Optional imports
try:
    import duckdb
    HAS_DUCKDB = True
except ImportError:
    HAS_DUCKDB = False
    print("WARNING: DuckDB not available")

try:
    from apexbase import ApexClient
    HAS_APEXBASE = True
except ImportError:
    HAS_APEXBASE = False
    print("WARNING: ApexBase not available")

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class BenchmarkConfig:
    """Configuration for benchmark tests."""
    # Data settings
    default_rows: int = 100_000  # Default test dataset size
    large_dataset_rows: int = 1_000_000
    medium_dataset_rows: int = 100_000
    
    # Test iteration settings
    warmup_iterations: int = 2
    timed_iterations: int = 5
    
    # Concurrency test settings
    concurrency_levels: List[int] = field(default_factory=lambda: [10, 20, 50])
    operations_per_thread: int = 200
    
    # Data generation
    n_cities: int = 10
    n_categories: int = 20
    
    # Output
    output_file: Optional[str] = None


# =============================================================================
# Data Generation
# =============================================================================

CITIES = ["Beijing", "Shanghai", "Guangzhou", "Shenzhen", "Hangzhou",
          "Nanjing", "Chengdu", "Wuhan", "Xian", "Qingdao"]
CATEGORIES = ["Electronics", "Clothing", "Food", "Sports", "Books",
              "Home", "Auto", "Health", "Travel", "Gaming"]


def generate_benchmark_data(n: int, seed: int = 42) -> Dict[str, List]:
    """Generate test data for benchmark scenarios."""
    rng = random.Random(seed)
    return {
        "name": [f"user_{i}" for i in range(n)],
        "age": [rng.randint(18, 80) for _ in range(n)],
        "score": [round(rng.uniform(0, 100), 2) for _ in range(n)],
        "city": [rng.choice(CITIES) for _ in range(n)],
        "category": [rng.choice(CATEGORIES) for _ in range(n)],
    }


def generate_concurrent_data(n: int, seed: int = 42) -> List[Dict]:
    """Generate test data for concurrent tests."""
    rng = np.random.default_rng(seed)
    cities = [f"City_{i}" for i in range(10)]
    categories = [f"Category_{i}" for i in range(20)]
    
    data = []
    for i in range(n):
        record = {
            '_id': i + 1,
            'user_id': rng.integers(1, 50_000),
            'category': rng.choice(categories),
            'city': rng.choice(cities),
            'price': rng.uniform(10.0, 1000.0),
            'quantity': rng.integers(1, 100),
            'timestamp': int(time.time() - rng.integers(0, 86400 * 30)),
            'is_active': rng.choice([True, False], p=[0.8, 0.2]),
            'score': rng.uniform(0.0, 1.0)
        }
        data.append(record)
    return data


# =============================================================================
# Timing Utilities
# =============================================================================

@contextmanager
def timer():
    """Context manager that yields a dict; sets 'elapsed_ms' on exit."""
    result = {}
    gc.collect()
    t0 = time.perf_counter()
    yield result
    result["elapsed_ms"] = (time.perf_counter() - t0) * 1000


def run_benchmark(fn: Callable, warmup: int = 2, iterations: int = 5) -> Dict[str, float]:
    """Run fn() with warmup, return timing statistics."""
    for _ in range(warmup):
        fn()
    
    times = []
    for _ in range(iterations):
        with timer() as t:
            fn()
        times.append(t["elapsed_ms"])
    
    return {
        "mean_ms": np.mean(times),
        "median_ms": np.median(times),
        "min_ms": np.min(times),
        "max_ms": np.max(times),
        "std_ms": np.std(times),
        "p50_ms": np.percentile(times, 50),
        "p90_ms": np.percentile(times, 90),
        "p99_ms": np.percentile(times, 99),
        "all_times_ms": times
    }


def run_benchmark_with_timing(fn: Callable, warmup: int = 2, iterations: int = 5) -> Tuple[float, Dict[str, float]]:
    """Run benchmark and return average time and statistics."""
    stats = run_benchmark(fn, warmup, iterations)
    return stats["mean_ms"], stats


def fmt_ms(ms: float) -> str:
    """Format milliseconds to human readable string."""
    if ms < 0.01:
        return f"{ms * 1000:.2f}us"
    if ms < 1:
        return f"{ms:.3f}ms"
    if ms < 1000:
        return f"{ms:.2f}ms"
    return f"{ms / 1000:.2f}s"


# =============================================================================
# Database Benchmarks - SQLite
# =============================================================================

class SQLiteBenchmark:
    """SQLite benchmark implementation."""
    
    def __init__(self, db_path: str, data: Dict[str, List] = None):
        self.db_path = db_path
        self.data = data
        self.conn = None
        self.n = len(data["name"]) if data else 0
    
    def setup(self):
        """Initialize database and create tables."""
        if os.path.exists(self.db_path):
            os.remove(self.db_path)
        self.conn = sqlite3.connect(self.db_path)
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA synchronous=OFF")
        self.conn.execute("""
            CREATE TABLE bench (
                _id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT,
                age INTEGER,
                score REAL,
                city TEXT,
                category TEXT
            )
        """)
    
    # ==================== Standard Benchmarks ====================
    
    def bench_insert(self):
        """Bulk insert N rows."""
        rows = list(zip(
            self.data["name"], self.data["age"], self.data["score"],
            self.data["city"], self.data["category"]
        ))
        self.conn.executemany(
            "INSERT INTO bench (name, age, score, city, category) VALUES (?,?,?,?,?)",
            rows,
        )
        self.conn.commit()
    
    def bench_count(self):
        return self.conn.execute("SELECT COUNT(*) FROM bench").fetchone()[0]
    
    def bench_select_limit(self, limit: int = 100):
        return self.conn.execute(f"SELECT * FROM bench LIMIT {limit}").fetchall()
    
    def bench_select_limit_10k(self):
        return self.conn.execute("SELECT * FROM bench LIMIT 10000").fetchall()
    
    def bench_filter_string(self):
        return self.conn.execute(
            "SELECT * FROM bench WHERE name = 'user_5000'"
        ).fetchall()
    
    def bench_filter_range(self):
        return self.conn.execute(
            "SELECT * FROM bench WHERE age BETWEEN 25 AND 35"
        ).fetchall()
    
    def bench_filter_like(self):
        return self.conn.execute(
            "SELECT * FROM bench WHERE name LIKE 'user_1%'"
        ).fetchall()
    
    def bench_filter_multi_cond(self):
        return self.conn.execute(
            "SELECT * FROM bench WHERE age > 30 AND score > 50.0"
        ).fetchall()
    
    def bench_filter_in(self):
        return self.conn.execute(
            "SELECT * FROM bench WHERE city IN ('Beijing', 'Shanghai', 'Guangzhou')"
        ).fetchall()
    
    def bench_group_by(self):
        return self.conn.execute(
            "SELECT city, COUNT(*), AVG(score) FROM bench GROUP BY city"
        ).fetchall()
    
    def bench_group_by_having(self):
        return self.conn.execute(
            "SELECT city, COUNT(*) as cnt, AVG(score) FROM bench GROUP BY city HAVING cnt > 1000"
        ).fetchall()
    
    def bench_group_by_2cols(self):
        return self.conn.execute(
            "SELECT city, category, COUNT(*), AVG(score) FROM bench GROUP BY city, category"
        ).fetchall()
    
    def bench_aggregation(self):
        return self.conn.execute(
            "SELECT COUNT(*), AVG(age), SUM(score), MIN(age), MAX(age) FROM bench"
        ).fetchone()
    
    def bench_complex(self):
        return self.conn.execute(
            "SELECT city, AVG(score) as avg_s FROM bench WHERE age BETWEEN 25 AND 50 GROUP BY city ORDER BY avg_s DESC LIMIT 5"
        ).fetchall()
    
    def bench_order_limit(self):
        return self.conn.execute(
            "SELECT * FROM bench ORDER BY score DESC LIMIT 100"
        ).fetchall()
    
    def bench_order_by_multi(self):
        return self.conn.execute(
            "SELECT * FROM bench ORDER BY city ASC, score DESC LIMIT 100"
        ).fetchall()
    
    def bench_point_lookup(self):
        return self.conn.execute(
            "SELECT * FROM bench WHERE _id = 5000"
        ).fetchone()
    
    def bench_count_distinct(self):
        return self.conn.execute(
            "SELECT COUNT(DISTINCT city) FROM bench"
        ).fetchone()[0]
    
    def bench_insert_1k(self):
        rows = [(f"new_{i}", 25, 50.0, "Beijing", "Books") for i in range(1000)]
        self.conn.executemany(
            "INSERT INTO bench (name, age, score, city, category) VALUES (?,?,?,?,?)",
            rows,
        )
        self.conn.commit()
    
    def bench_update_1k(self):
        self.conn.execute("UPDATE bench SET score = 50.0 WHERE age = 25")
        self.conn.commit()
    
    def bench_delete_1k(self):
        rows = [(f"del_{i}", 99, 99.0, "Beijing", "Books") for i in range(1000)]
        self.conn.executemany(
            "INSERT INTO bench (name, age, score, city, category) VALUES (?,?,?,?,?)",
            rows,
        )
        self.conn.commit()
        self.conn.execute("DELETE FROM bench WHERE age = 99")
        self.conn.commit()
    
    def bench_window_row_number(self):
        return self.conn.execute(
            "SELECT name, city, score, "
            "ROW_NUMBER() OVER (PARTITION BY city ORDER BY score DESC) as rn "
            "FROM bench LIMIT 1000"
        ).fetchall()
    
    def close(self):
        if self.conn:
            self.conn.close()


# =============================================================================
# Database Benchmarks - DuckDB
# =============================================================================

class DuckDBBenchmark:
    """DuckDB benchmark implementation."""
    
    def __init__(self, db_path: str, data: Dict[str, List] = None):
        self.db_path = db_path
        self.data = data
        self.conn = None
        self.n = len(data["name"]) if data else 0
    
    def setup(self):
        if os.path.exists(self.db_path):
            os.remove(self.db_path)
        self.conn = duckdb.connect(self.db_path)
        self.conn.execute("""
            CREATE TABLE bench (
                name VARCHAR,
                age INTEGER,
                score DOUBLE,
                city VARCHAR,
                category VARCHAR
            )
        """)
    
    # ==================== Standard Benchmarks ====================
    
    def bench_insert(self):
        if HAS_PANDAS:
            df = pd.DataFrame(self.data)
            self.conn.execute("INSERT INTO bench SELECT * FROM df")
        else:
            rows = list(zip(
                self.data["name"], self.data["age"], self.data["score"],
                self.data["city"], self.data["category"]
            ))
            self.conn.executemany(
                "INSERT INTO bench VALUES (?,?,?,?,?)", rows
            )
    
    def bench_count(self):
        return self.conn.execute("SELECT COUNT(*) FROM bench").fetchone()[0]
    
    def bench_select_limit(self, limit: int = 100):
        return self.conn.execute(f"SELECT * FROM bench LIMIT {limit}").fetchall()
    
    def bench_select_limit_10k(self):
        return self.conn.execute("SELECT * FROM bench LIMIT 10000").fetchall()
    
    def bench_filter_string(self):
        return self.conn.execute(
            "SELECT * FROM bench WHERE name = 'user_5000'"
        ).fetchall()
    
    def bench_filter_range(self):
        return self.conn.execute(
            "SELECT * FROM bench WHERE age BETWEEN 25 AND 35"
        ).fetchall()
    
    def bench_filter_like(self):
        return self.conn.execute(
            "SELECT * FROM bench WHERE name LIKE 'user_1%'"
        ).fetchall()
    
    def bench_filter_multi_cond(self):
        return self.conn.execute(
            "SELECT * FROM bench WHERE age > 30 AND score > 50.0"
        ).fetchall()
    
    def bench_filter_in(self):
        return self.conn.execute(
            "SELECT * FROM bench WHERE city IN ('Beijing', 'Shanghai', 'Guangzhou')"
        ).fetchall()
    
    def bench_group_by(self):
        return self.conn.execute(
            "SELECT city, COUNT(*), AVG(score) FROM bench GROUP BY city"
        ).fetchall()
    
    def bench_group_by_having(self):
        return self.conn.execute(
            "SELECT city, COUNT(*) as cnt, AVG(score) FROM bench GROUP BY city HAVING cnt > 1000"
        ).fetchall()
    
    def bench_group_by_2cols(self):
        return self.conn.execute(
            "SELECT city, category, COUNT(*), AVG(score) FROM bench GROUP BY city, category"
        ).fetchall()
    
    def bench_aggregation(self):
        return self.conn.execute(
            "SELECT COUNT(*), AVG(age), SUM(score), MIN(age), MAX(age) FROM bench"
        ).fetchone()
    
    def bench_complex(self):
        return self.conn.execute(
            "SELECT city, AVG(score) as avg_s FROM bench WHERE age BETWEEN 25 AND 50 GROUP BY city ORDER BY avg_s DESC LIMIT 5"
        ).fetchall()
    
    def bench_order_limit(self):
        return self.conn.execute(
            "SELECT * FROM bench ORDER BY score DESC LIMIT 100"
        ).fetchall()
    
    def bench_order_by_multi(self):
        return self.conn.execute(
            "SELECT * FROM bench ORDER BY city ASC, score DESC LIMIT 100"
        ).fetchall()
    
    def bench_point_lookup(self):
        return self.conn.execute(
            "SELECT * FROM bench WHERE rowid = 5000"
        ).fetchall()
    
    def bench_count_distinct(self):
        return self.conn.execute(
            "SELECT COUNT(DISTINCT city) FROM bench"
        ).fetchone()[0]
    
    def bench_insert_1k(self):
        if HAS_PANDAS:
            df = pd.DataFrame({
                "name": [f"new_{i}" for i in range(1000)],
                "age": [25] * 1000,
                "score": [50.0] * 1000,
                "city": ["Beijing"] * 1000,
                "category": ["Books"] * 1000,
            })
            self.conn.execute("INSERT INTO bench SELECT * FROM df")
        else:
            rows = [(f"new_{i}", 25, 50.0, "Beijing", "Books") for i in range(1000)]
            self.conn.executemany("INSERT INTO bench VALUES (?,?,?,?,?)", rows)
    
    def bench_update_1k(self):
        self.conn.execute("UPDATE bench SET score = 50.0 WHERE age = 25")
    
    def bench_delete_1k(self):
        if HAS_PANDAS:
            df = pd.DataFrame({
                "name": [f"del_{i}" for i in range(1000)],
                "age": [99] * 1000,
                "score": [99.0] * 1000,
                "city": ["Beijing"] * 1000,
                "category": ["Books"] * 1000,
            })
            self.conn.execute("INSERT INTO bench SELECT * FROM df")
        else:
            rows = [(f"del_{i}", 99, 99.0, "Beijing", "Books") for i in range(1000)]
            self.conn.executemany("INSERT INTO bench VALUES (?,?,?,?,?)", rows)
        self.conn.execute("DELETE FROM bench WHERE age = 99")
    
    def bench_window_row_number(self):
        return self.conn.execute(
            "SELECT name, city, score, "
            "ROW_NUMBER() OVER (PARTITION BY city ORDER BY score DESC) as rn "
            "FROM bench LIMIT 1000"
        ).fetchall()
    
    def close(self):
        if self.conn:
            self.conn.close()


# =============================================================================
# Database Benchmarks - ApexBase
# =============================================================================

class ApexBaseBenchmark:
    """ApexBase benchmark implementation."""
    
    def __init__(self, db_dir: str, data: Dict[str, List] = None):
        self.db_dir = db_dir
        self.data = data
        self.client = None
        self.n = len(data["name"]) if data else 0
    
    def setup(self):
        if os.path.exists(self.db_dir):
            shutil.rmtree(self.db_dir)
        self.client = ApexClient(self.db_dir, drop_if_exists=True)
        self.client.create_table('default')
    
    # ==================== Standard Benchmarks ====================
    
    def bench_insert(self):
        self.client.store(self.data)
    
    def bench_count(self):
        return self.client.execute("SELECT COUNT(*) FROM default").scalar()
    
    def bench_select_limit(self, limit: int = 100):
        return self.client.execute(f"SELECT * FROM default LIMIT {limit}")
    
    def bench_select_limit_10k(self):
        return self.client.execute("SELECT * FROM default LIMIT 10000")
    
    def bench_filter_string(self):
        return self.client.execute(
            "SELECT * FROM default WHERE name = 'user_5000'"
        )
    
    def bench_filter_range(self):
        return self.client.execute(
            "SELECT * FROM default WHERE age BETWEEN 25 AND 35"
        )
    
    def bench_filter_like(self):
        return self.client.execute(
            "SELECT * FROM default WHERE name LIKE 'user_1%'"
        )
    
    def bench_filter_multi_cond(self):
        return self.client.execute(
            "SELECT * FROM default WHERE age > 30 AND score > 50.0"
        )
    
    def bench_filter_in(self):
        return self.client.execute(
            "SELECT * FROM default WHERE city IN ('Beijing', 'Shanghai', 'Guangzhou')"
        )
    
    def bench_group_by(self):
        return self.client.execute(
            "SELECT city, COUNT(*), AVG(score) FROM default GROUP BY city"
        )
    
    def bench_group_by_having(self):
        return self.client.execute(
            "SELECT city, COUNT(*) as cnt, AVG(score) FROM default GROUP BY city HAVING cnt > 1000"
        )
    
    def bench_group_by_2cols(self):
        return self.client.execute(
            "SELECT city, category, COUNT(*), AVG(score) FROM default GROUP BY city, category"
        )
    
    def bench_aggregation(self):
        return self.client.execute(
            "SELECT COUNT(*), AVG(age), SUM(score), MIN(age), MAX(age) FROM default"
        )
    
    def bench_complex(self):
        return self.client.execute(
            "SELECT city, AVG(score) as avg_s FROM default WHERE age BETWEEN 25 AND 50 GROUP BY city ORDER BY avg_s DESC LIMIT 5"
        )
    
    def bench_order_limit(self):
        return self.client.execute(
            "SELECT * FROM default ORDER BY score DESC LIMIT 100"
        )
    
    def bench_order_by_multi(self):
        return self.client.execute(
            "SELECT * FROM default ORDER BY city ASC, score DESC LIMIT 100"
        )
    
    def bench_point_lookup(self):
        return self.client.execute(
            "SELECT * FROM default WHERE _id = 5000"
        )
    
    def bench_count_distinct(self):
        return self.client.execute(
            "SELECT COUNT(DISTINCT city) FROM default"
        )
    
    def bench_insert_1k(self):
        data_1k = {
            "name": [f"new_{i}" for i in range(1000)],
            "age": [25] * 1000,
            "score": [50.0] * 1000,
            "city": ["Beijing"] * 1000,
            "category": ["Books"] * 1000,
        }
        self.client.store(data_1k)
    
    def bench_update_1k(self):
        return self.client.execute(
            "UPDATE default SET score = 50.0 WHERE age = 25"
        )
    
    def bench_delete_1k(self):
        data = {
            "name": [f"del_{i}" for i in range(1000)],
            "age": [99] * 1000,
            "score": [99.0] * 1000,
            "city": ["Beijing"] * 1000,
            "category": ["Books"] * 1000,
        }
        self.client.store(data)
        self.client.execute("DELETE FROM default WHERE age = 99")
    
    def bench_window_row_number(self):
        return self.client.execute(
            "SELECT name, city, score, "
            "ROW_NUMBER() OVER (PARTITION BY city ORDER BY score DESC) as rn "
            "FROM default LIMIT 1000"
        )
    
    def close(self):
        if self.client:
            self.client.close()


# =============================================================================
# Benchmark Definitions
# =============================================================================

BENCHMARK_SCENARIOS = [
    # (display_name, method_name, category)
    ("Bulk Insert", "bench_insert", "data_operations"),
    ("COUNT(*)", "bench_count", "query"),
    ("SELECT * LIMIT 100", "bench_select_limit", "query"),
    ("SELECT * LIMIT 10K", "bench_select_limit_10k", "query"),
    ("Filter (name = 'user_5000')", "bench_filter_string", "filter"),
    ("Filter (age BETWEEN 25 AND 35)", "bench_filter_range", "filter"),
    ("LIKE filter (name LIKE 'user_1%')", "bench_filter_like", "filter"),
    ("Multi-cond (age>30 AND score>50)", "bench_filter_multi_cond", "filter"),
    ("IN filter (city IN 3 cities)", "bench_filter_in", "filter"),
    ("GROUP BY city", "bench_group_by", "aggregation"),
    ("GROUP BY + HAVING", "bench_group_by_having", "aggregation"),
    ("GROUP BY city,category", "bench_group_by_2cols", "aggregation"),
    ("Aggregation (5 funcs)", "bench_aggregation", "aggregation"),
    ("Complex (Filter+Group+Order)", "bench_complex", "aggregation"),
    ("ORDER BY score LIMIT 100", "bench_order_limit", "sort"),
    ("ORDER BY city,score LIMIT 100", "bench_order_by_multi", "sort"),
    ("Point Lookup (by ID)", "bench_point_lookup", "lookup"),
    ("COUNT(DISTINCT city)", "bench_count_distinct", "aggregation"),
    ("Insert 1K rows", "bench_insert_1k", "data_operations"),
    ("UPDATE 1K rows", "bench_update_1k", "data_operations"),
    ("DELETE 1K rows", "bench_delete_1k", "data_operations"),
    ("Window ROW_NUMBER", "bench_window_row_number", "window"),
]


# =============================================================================
# Concurrency Tests
# =============================================================================

def apex_worker(thread_id: int, db_path: str, ops_count: int, read_ratio: float = 0.8) -> Dict:
    """ApexBase concurrent worker."""
    client = ApexClient(db_path)
    results = {
        'thread_id': thread_id,
        'operations': 0,
        'errors': 0,
        'read_times': [],
        'write_times': [],
        'total_time': 0
    }
    
    start_time = time.perf_counter()
    
    try:
        client.use_table('default')
        
        for i in range(ops_count):
            op_start = time.perf_counter()
            
            if random.random() < read_ratio:  # Read operations
                if random.random() < 0.6:
                    result = client.execute(f"SELECT * FROM default WHERE _id = {random.randint(1, 100000)}")
                else:
                    result = client.execute("SELECT COUNT(*) FROM default WHERE city = 'Beijing'")
                
                results['read_times'].append(time.perf_counter() - op_start)
            else:  # Write operations
                new_record = {
                    'name': f"user_{random.randint(1, 1000000)}",
                    'age': random.randint(18, 80),
                    'score': random.uniform(0, 100),
                    'city': random.choice(CITIES),
                    'category': random.choice(CATEGORIES),
                }
                
                client.store(new_record)
                results['write_times'].append(time.perf_counter() - op_start)
            
            results['operations'] += 1
            
    except Exception as e:
        results['errors'] += 1
    finally:
        try:
            client.close()
        except:
            pass
        results['total_time'] = time.perf_counter() - start_time
    
    return results


def sqlite_worker(thread_id: int, db_path: str, ops_count: int, read_ratio: float = 0.8) -> Dict:
    """SQLite concurrent worker."""
    conn = sqlite3.connect(db_path, timeout=30.0)
    results = {
        'thread_id': thread_id,
        'operations': 0,
        'errors': 0,
        'read_times': [],
        'write_times': [],
        'total_time': 0
    }
    
    start_time = time.perf_counter()
    
    try:
        for i in range(ops_count):
            op_start = time.perf_counter()
            
            if random.random() < read_ratio:  # Read operations
                if random.random() < 0.6:
                    result = conn.execute(f"SELECT * FROM bench WHERE _id = {random.randint(1, 100000)}").fetchall()
                else:
                    result = conn.execute("SELECT COUNT(*) FROM bench WHERE city = 'Beijing'").fetchall()
                
                results['read_times'].append(time.perf_counter() - op_start)
            else:  # Write operations
                new_record = (
                    f"user_{random.randint(1, 1000000)}",
                    random.randint(18, 80),
                    random.uniform(0, 100),
                    random.choice(CITIES),
                    random.choice(CATEGORIES),
                )
                
                conn.execute(
                    "INSERT INTO bench (name, age, score, city, category) VALUES (?, ?, ?, ?, ?)",
                    new_record
                )
                conn.commit()
                results['write_times'].append(time.perf_counter() - op_start)
            
            results['operations'] += 1
            
    except Exception as e:
        results['errors'] += 1
    finally:
        conn.close()
        results['total_time'] = time.perf_counter() - start_time
    
    return results


def duckdb_worker(thread_id: int, db_path: str, ops_count: int, read_ratio: float = 0.8) -> Dict:
    """DuckDB concurrent worker."""
    conn = duckdb.connect(db_path)
    results = {
        'thread_id': thread_id,
        'operations': 0,
        'errors': 0,
        'read_times': [],
        'write_times': [],
        'total_time': 0
    }
    
    start_time = time.perf_counter()
    
    try:
        for i in range(ops_count):
            op_start = time.perf_counter()
            
            if random.random() < read_ratio:  # Read operations
                if random.random() < 0.6:
                    result = conn.execute(f"SELECT * FROM bench WHERE _id = {random.randint(1, 100000)}").fetchall()
                else:
                    result = conn.execute("SELECT COUNT(*) FROM bench WHERE city = 'Beijing'").fetchall()
                
                results['read_times'].append(time.perf_counter() - op_start)
            else:  # Write operations
                new_record = {
                    'name': f"user_{random.randint(1, 1000000)}",
                    'age': random.randint(18, 80),
                    'score': random.uniform(0, 100),
                    'city': random.choice(CITIES),
                    'category': random.choice(CATEGORIES),
                }
                
                conn.execute('''
                    INSERT INTO bench VALUES 
                    (?, ?, ?, ?, ?)
                ''', [
                    new_record['name'], new_record['age'], new_record['score'],
                    new_record['city'], new_record['category']
                ])
                results['write_times'].append(time.perf_counter() - op_start)
            
            results['operations'] += 1
            
    except Exception as e:
        results['errors'] += 1
    finally:
        conn.close()
        results['total_time'] = time.perf_counter() - start_time
    
    return results


def run_concurrent_test(db_path: str, worker_func, db_name: str, threads: int, ops_per_thread: int) -> Dict:
    """Run concurrent stress test."""
    print(f"    Running {db_name} with {threads} threads...")
    
    with ThreadPoolExecutor(max_workers=threads) as executor:
        futures = []
        for i in range(threads):
            future = executor.submit(worker_func, i, db_path, ops_per_thread)
            futures.append(future)
        
        all_results = []
        for future in as_completed(futures):
            result = future.result()
            all_results.append(result)
    
    total_ops = sum(r['operations'] for r in all_results)
    total_errors = sum(r['errors'] for r in all_results)
    all_read_times = []
    all_write_times = []
    
    for r in all_results:
        all_read_times.extend(r['read_times'])
        all_write_times.extend(r['write_times'])
    
    total_time = sum(r['total_time'] for r in all_results)
    
    return {
        'db_name': db_name,
        'threads': threads,
        'total_operations': total_ops,
        'total_errors': total_errors,
        'error_rate': total_errors / total_ops if total_ops > 0 else 0,
        'ops_per_sec': total_ops / total_time if total_time > 0 else 0,
        'avg_read_time_ms': np.mean(all_read_times) * 1000 if all_read_times else 0,
        'avg_write_time_ms': np.mean(all_write_times) * 1000 if all_write_times else 0,
        'p50_read_time_ms': np.percentile(all_read_times, 50) * 1000 if all_read_times else 0,
        'p90_read_time_ms': np.percentile(all_read_times, 90) * 1000 if all_read_times else 0,
        'p99_read_time_ms': np.percentile(all_read_times, 99) * 1000 if all_read_times else 0,
        'p50_write_time_ms': np.percentile(all_write_times, 50) * 1000 if all_write_times else 0,
        'p90_write_time_ms': np.percentile(all_write_times, 90) * 1000 if all_write_times else 0,
        'p99_write_time_ms': np.percentile(all_write_times, 99) * 1000 if all_write_times else 0,
    }


# =============================================================================
# System Information
# =============================================================================

def get_system_info() -> Dict:
    """Get system information."""
    info = {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "python": platform.python_version(),
    }
    
    try:
        import psutil
        info["cpu_count"] = psutil.cpu_count(logical=True)
        info["memory_gb"] = round(psutil.virtual_memory().total / (1024**3), 1)
    except ImportError:
        info["cpu_count"] = os.cpu_count()
        info["memory_gb"] = "N/A"
    
    if HAS_APEXBASE:
        try:
            from apexbase._core import __version__
            info["apexbase"] = __version__
        except Exception:
            info["apexbase"] = "unknown"
    
    if HAS_DUCKDB:
        info["duckdb"] = duckdb.__version__
    
    info["sqlite"] = sqlite3.sqlite_version
    
    return info


# =============================================================================
# Main Benchmark Runner
# =============================================================================

def run_standard_benchmarks(data: Dict[str, List], engines: Dict, warmup: int, iterations: int) -> Dict:
    """Run standard benchmark scenarios."""
    results = {}
    
    for bench_name, method_name, category in BENCHMARK_SCENARIOS:
        results[bench_name] = {"category": category, "engines": {}}
        
        for eng_name, engine in engines.items():
            fn = getattr(engine, method_name, None)
            if fn is None:
                results[bench_name]["engines"][eng_name] = None
                continue
            
            try:
                if method_name == "bench_insert":
                    # For insert, we need to re-setup
                    engine.setup()
                
                avg_ms, stats = run_benchmark_with_timing(fn, warmup, iterations)
                
                # Calculate throughput
                if "Insert" in bench_name:
                    throughput = engine.n / (avg_ms / 1000) if avg_ms > 0 else 0
                elif "LIMIT" in bench_name:
                    limit = int(bench_name.split("LIMIT ")[1].split(" ")[0]) if "LIMIT" in bench_name else 100
                    throughput = limit / (avg_ms / 1000) if avg_ms > 0 else 0
                else:
                    throughput = 1000 / avg_ms if avg_ms > 0 else 0  # queries per second
                
                results[bench_name]["engines"][eng_name] = {
                    "mean_ms": round(avg_ms, 3),
                    "median_ms": round(stats["median_ms"], 3),
                    "min_ms": round(stats["min_ms"], 3),
                    "max_ms": round(stats["max_ms"], 3),
                    "std_ms": round(stats["std_ms"], 3),
                    "p50_ms": round(stats["p50_ms"], 3),
                    "p90_ms": round(stats["p90_ms"], 3),
                    "p99_ms": round(stats["p99_ms"], 3),
                    "throughput": round(throughput, 2)
                }
            except Exception as e:
                results[bench_name]["engines"][eng_name] = {"error": str(e)}
    
    return results


def run_concurrency_tests(data: Dict[str, List], config: BenchmarkConfig) -> Dict:
    """Run concurrency stress tests."""
    results = {"concurrent_tests": {}}
    
    # Prepare database paths
    tmpdir = tempfile.mkdtemp(prefix="concurrent_bench_")
    db_paths = {
        'ApexBase': os.path.join(tmpdir, "apexbase.apex"),
        'SQLite': os.path.join(tmpdir, "sqlite.db"),
        'DuckDB': os.path.join(tmpdir, "duckdb.duckdb")
    }
    
    # Initialize and populate databases
    engines = {}
    
    if HAS_APEXBASE:
        apex = ApexBaseBenchmark(db_paths['ApexBase'], data)
        apex.setup()
        apex.bench_insert()
        engines['ApexBase'] = (apex, apex_worker, db_paths['ApexBase'])
    
    sqlite = SQLiteBenchmark(db_paths['SQLite'], data)
    sqlite.setup()
    sqlite.bench_insert()
    engines['SQLite'] = (sqlite, sqlite_worker, db_paths['SQLite'])
    
    if HAS_DUCKDB:
        duck = DuckDBBenchmark(db_paths['DuckDB'], data)
        duck.setup()
        duck.bench_insert()
        engines['DuckDB'] = (duck, duckdb_worker, db_paths['DuckDB'])
    
    # Run tests for each concurrency level
    for threads in config.concurrency_levels:
        results["concurrent_tests"][f"{threads}_threads"] = {}
        
        for eng_name, (engine, worker_func, db_path) in engines.items():
            try:
                test_result = run_concurrent_test(
                    db_path, worker_func, eng_name, 
                    threads, config.operations_per_thread
                )
                results["concurrent_tests"][f"{threads}_threads"][eng_name] = test_result
            except Exception as e:
                results["concurrent_tests"][f"{threads}_threads"][eng_name] = {"error": str(e)}
    
    # Cleanup
    for engine, _, _ in engines.values():
        engine.close()
    
    shutil.rmtree(tmpdir)
    
    return results


def run_large_dataset_tests(config: BenchmarkConfig) -> Dict:
    """Run tests with large datasets (100K and 1M rows)."""
    results = {"large_dataset_tests": {}}
    
    tmpdir = tempfile.mkdtemp(prefix="large_dataset_")
    
    for dataset_name, row_count in [("100K", 100000), ("1M", 1000000)]:
        print(f"\n  Testing with {dataset_name} rows ({row_count:,})...")
        
        results["large_dataset_tests"][dataset_name] = {
            "row_count": row_count,
            "engines": {}
        }
        
        data = generate_benchmark_data(row_count)
        
        # Test with different engines
        if HAS_APEXBASE:
            db_path = os.path.join(tmpdir, f"apex_{dataset_name}.apex")
            engine = ApexBaseBenchmark(db_path, data)
            engine.setup()
            
            start = time.perf_counter()
            engine.bench_insert()
            insert_time = (time.perf_counter() - start) * 1000
            
            # Query test
            start = time.perf_counter()
            engine.bench_count()
            query_time = (time.perf_counter() - start) * 1000
            
            results["large_dataset_tests"][dataset_name]["engines"]["ApexBase"] = {
                "insert_ms": round(insert_time, 2),
                "query_ms": round(query_time, 2),
                "insert_rows_per_sec": round(row_count / (insert_time / 1000), 0),
                "query_rows_per_sec": round(row_count / (query_time / 1000), 0)
            }
            
            engine.close()
        
        # SQLite
        db_path = os.path.join(tmpdir, f"sqlite_{dataset_name}.db")
        engine = SQLiteBenchmark(db_path, data)
        engine.setup()
        
        start = time.perf_counter()
        engine.bench_insert()
        insert_time = (time.perf_counter() - start) * 1000
        
        start = time.perf_counter()
        engine.bench_count()
        query_time = (time.perf_counter() - start) * 1000
        
        results["large_dataset_tests"][dataset_name]["engines"]["SQLite"] = {
            "insert_ms": round(insert_time, 2),
            "query_ms": round(query_time, 2),
            "insert_rows_per_sec": round(row_count / (insert_time / 1000), 0),
            "query_rows_per_sec": round(row_count / (query_time / 1000), 0)
        }
        
        engine.close()
        
        # DuckDB
        if HAS_DUCKDB:
            db_path = os.path.join(tmpdir, f"duckdb_{dataset_name}.duckdb")
            engine = DuckDBBenchmark(db_path, data)
            engine.setup()
            
            start = time.perf_counter()
            engine.bench_insert()
            insert_time = (time.perf_counter() - start) * 1000
            
            start = time.perf_counter()
            engine.bench_count()
            query_time = (time.perf_counter() - start) * 1000
            
            results["large_dataset_tests"][dataset_name]["engines"]["DuckDB"] = {
                "insert_ms": round(insert_time, 2),
                "query_ms": round(query_time, 2),
                "insert_rows_per_sec": round(row_count / (insert_time / 1000), 0),
                "query_rows_per_sec": round(row_count / (query_time / 1000), 0)
            }
            
            engine.close()
    
    shutil.rmtree(tmpdir)
    
    return results


def analyze_results(results: Dict) -> Dict:
    """Analyze benchmark results and identify落后 (lagging) scenarios."""
    analysis = {
        "lagging_scenarios": [],
        "winning_scenarios": [],
        "summary": {}
    }
    
    # Analyze standard benchmarks
    std_results = results.get("standard_benchmarks", {})
    
    for bench_name, bench_data in std_results.items():
        engines = bench_data.get("engines", {})
        
        if len(engines) < 2:
            continue
        
        # Find the fastest engine
        valid_engines = {k: v for k, v in engines.items() if v and "mean_ms" in v}
        
        if not valid_engines:
            continue
        
        fastest = min(valid_engines.items(), key=lambda x: x[1]["mean_ms"])
        fastest_name = fastest[0]
        fastest_time = fastest[1]["mean_ms"]
        
        for eng_name, eng_data in valid_engines.items():
            if eng_name == fastest_name:
                continue
            
            ratio = eng_data["mean_ms"] / fastest_time if fastest_time > 0 else float('inf')
            
            if ratio > 1.5:  # More than 50% slower
                analysis["lagging_scenarios"].append({
                    "scenario": bench_name,
                    "engine": eng_name,
                    "time_ms": round(eng_data["mean_ms"], 2),
                    "fastest_engine": fastest_name,
                    "fastest_time_ms": round(fastest_time, 2),
                    "ratio": round(ratio, 2)
                })
    
    return analysis


def print_summary(results: Dict):
    """Print a summary of the benchmark results."""
    print("\n" + "=" * 80)
    print("BENCHMARK RESULTS SUMMARY")
    print("=" * 80)
    
    # Standard benchmarks summary
    std_results = results.get("standard_benchmarks", {})
    
    print("\n--- Standard Benchmark Results ---")
    print(f"{'Scenario':<40} {'ApexBase':>12} {'SQLite':>12} {'DuckDB':>12}")
    print("-" * 80)
    
    for bench_name, bench_data in std_results.items():
        engines = bench_data.get("engines", {})
        
        apex_ms = engines.get("ApexBase", {}).get("mean_ms", "N/A")
        sqlite_ms = engines.get("SQLite", {}).get("mean_ms", "N/A")
        duck_ms = engines.get("DuckDB", {}).get("mean_ms", "N/A")
        
        apex_str = f"{apex_ms:.2f}ms" if isinstance(apex_ms, float) else str(apex_ms)
        sqlite_str = f"{sqlite_ms:.2f}ms" if isinstance(sqlite_ms, float) else str(sqlite_ms)
        duck_str = f"{duck_ms:.2f}ms" if isinstance(duck_ms, float) else str(duck_ms)
        
        print(f"{bench_name:<40} {apex_str:>12} {sqlite_str:>12} {duck_str:>12}")
    
    # Concurrency tests summary
    conc_results = results.get("concurrency_tests", {}).get("concurrent_tests", {})
    
    print("\n--- Concurrency Tests (ops/sec) ---")
    print(f"{'Threads':<15} {'ApexBase':>15} {'SQLite':>15} {'DuckDB':>15}")
    print("-" * 60)
    
    for threads_key, thread_data in conc_results.items():
        apex_ops = thread_data.get("ApexBase", {}).get("ops_per_sec", "N/A")
        sqlite_ops = thread_data.get("SQLite", {}).get("ops_per_sec", "N/A")
        duck_ops = thread_data.get("DuckDB", {}).get("ops_per_sec", "N/A")
        
        apex_str = f"{apex_ops:,.0f}" if isinstance(apex_ops, (int, float)) else str(apex_ops)
        sqlite_str = f"{sqlite_ops:,.0f}" if isinstance(sqlite_ops, (int, float)) else str(sqlite_ops)
        duck_str = f"{duck_ops:,.0f}" if isinstance(duck_ops, (int, float)) else str(duck_ops)
        
        print(f"{threads_key:<15} {apex_str:>15} {sqlite_str:>15} {duck_str:>15}")
    
    # Large dataset tests
    large_results = results.get("large_dataset_tests", {})
    
    print("\n--- Large Dataset Tests ---")
    for dataset_name, dataset_data in large_results.items():
        print(f"\n{dataset_name} rows:")
        for eng_name, eng_data in dataset_data.get("engines", {}).items():
            print(f"  {eng_name}: Insert={eng_data.get('insert_ms', 'N/A')}ms, "
                  f"Query={eng_data.get('query_ms', 'N/A')}ms")
    
    # Lagging scenarios
    analysis = results.get("analysis", {})
    lagging = analysis.get("lagging_scenarios", [])
    
    if lagging:
        print("\n--- Lagging Scenarios (>50% slower than fastest) ---")
        for item in lagging:
            print(f"  {item['scenario']}: {item['engine']} is {item['ratio']}x slower "
                  f"({item['time_ms']}ms vs {item['fastest_time_ms']}ms)")


# =============================================================================
# Main Entry Point
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="ApexBase Comprehensive Performance Comparison Benchmark"
    )
    parser.add_argument(
        "--rows", type=int, default=100_000,
        help="Number of rows for standard benchmarks (default: 100,000)"
    )
    parser.add_argument(
        "--warmup", type=int, default=2,
        help="Warmup iterations (default: 2)"
    )
    parser.add_argument(
        "--iterations", type=int, default=5,
        help="Timed iterations (default: 5)"
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="JSON output file (optional)"
    )
    parser.add_argument(
        "--skip-concurrency", action="store_true",
        help="Skip concurrency tests"
    )
    parser.add_argument(
        "--skip-large", action="store_true",
        help="Skip large dataset tests"
    )
    
    args = parser.parse_args()
    
    config = BenchmarkConfig(
        default_rows=args.rows,
        warmup_iterations=args.warmup,
        timed_iterations=args.iterations,
        output_file=args.output
    )
    
    print("=" * 80)
    print("ApexBase Comprehensive Performance Comparison Benchmark")
    print("=" * 80)
    
    # Get system info
    sys_info = get_system_info()
    print(f"\nSystem: {sys_info['platform']} ({sys_info['machine']})")
    print(f"CPU: {sys_info.get('processor', 'N/A')} ({sys_info['cpu_count']} cores)")
    print(f"Memory: {sys_info['memory_gb']} GB")
    print(f"Python: {sys_info['python']}")
    
    if "apexbase" in sys_info:
        print(f"ApexBase: v{sys_info['apexbase']}")
    print(f"SQLite: v{sys_info['sqlite']}")
    if HAS_DUCKDB:
        print(f"DuckDB: v{sys_info['duckdb']}")
    
    print(f"\nDataset: {config.default_rows:,} rows × 5 columns")
    print(f"Warmup: {config.warmup_iterations} iterations, Timed: {config.timed_iterations} iterations")
    
    all_results = {
        "system_info": sys_info,
        "config": {
            "rows": config.default_rows,
            "warmup": config.warmup_iterations,
            "iterations": config.timed_iterations,
            "concurrency_levels": config.concurrency_levels
        }
    }
    
    # Generate test data
    print("\nGenerating test data...")
    data = generate_benchmark_data(config.default_rows)
    print(f"  Generated {len(data['name']):,} rows")
    
    # Setup temp directory
    tmpdir = tempfile.mkdtemp(prefix="full_perf_bench_")
    
    # Initialize engines
    engines = {}
    db_paths = {
        'ApexBase': os.path.join(tmpdir, "apexbase.apex"),
        'SQLite': os.path.join(tmpdir, "sqlite.db"),
        'DuckDB': os.path.join(tmpdir, "duckdb.duckdb")
    }
    
    print("\nSetting up databases...")
    
    if HAS_APEXBASE:
        engines['ApexBase'] = ApexBaseBenchmark(db_paths['ApexBase'], data)
        engines['ApexBase'].setup()
        print("  ApexBase: Ready")
    
    engines['SQLite'] = SQLiteBenchmark(db_paths['SQLite'], data)
    engines['SQLite'].setup()
    print("  SQLite: Ready")
    
    if HAS_DUCKDB:
        engines['DuckDB'] = DuckDBBenchmark(db_paths['DuckDB'], data)
        engines['DuckDB'].setup()
        print("  DuckDB: Ready")
    
    # Run standard benchmarks
    print("\n" + "-" * 60)
    print("Running Standard Benchmarks...")
    print("-" * 60)
    
    std_results = run_standard_benchmarks(
        data, engines, 
        config.warmup_iterations, config.timed_iterations
    )
    all_results["standard_benchmarks"] = std_results
    
    # Run concurrency tests
    if not args.skip_concurrency:
        print("\n" + "-" * 60)
        print("Running Concurrency Tests...")
        print("-" * 60)
        
        conc_results = run_concurrency_tests(data, config)
        all_results["concurrency_tests"] = conc_results
    
    # Run large dataset tests
    if not args.skip_large:
        print("\n" + "-" * 60)
        print("Running Large Dataset Tests...")
        print("-" * 60)
        
        large_results = run_large_dataset_tests(config)
        all_results["large_dataset_tests"] = large_results
    
    # Analyze results
    print("\n" + "-" * 60)
    print("Analyzing Results...")
    print("-" * 60)
    
    analysis = analyze_results(all_results)
    all_results["analysis"] = analysis
    
    # Print summary
    print_summary(all_results)
    
    # Cleanup
    for engine in engines.values():
        engine.close()
    
    shutil.rmtree(tmpdir)
    
    # Save results to JSON
    if config.output_file:
        with open(config.output_file, 'w') as f:
            json.dump(all_results, f, indent=2)
        print(f"\nResults saved to: {config.output_file}")
    
    print("\n" + "=" * 80)
    print("Benchmark Complete!")
    print("=" * 80)
    
    return all_results


if __name__ == "__main__":
    main()
