"""
ApexBase Performance Benchmark: ApexBase vs SQLite vs DuckDB

Measures key HTAP operations across all three engines on the same dataset.
Results are printed as a formatted table and optionally saved to JSON.

Usage:
    python benchmarks/bench_vs_sqlite_duckdb.py [--rows N] [--warmup N] [--iterations N] [--output FILE]
"""

import argparse
import gc
import json
import os
import platform
import random
import shutil
import sqlite3
import string
import sys
import tempfile
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Optional imports
# ---------------------------------------------------------------------------
try:
    import duckdb
    HAS_DUCKDB = True
except ImportError:
    HAS_DUCKDB = False

try:
    from apexbase import ApexClient
    HAS_APEXBASE = True
except ImportError:
    HAS_APEXBASE = False

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False

try:
    import pyarrow as pa
    HAS_PYARROW = True
except ImportError:
    HAS_PYARROW = False

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

CITIES = ["Beijing", "Shanghai", "Guangzhou", "Shenzhen", "Hangzhou",
          "Nanjing", "Chengdu", "Wuhan", "Xian", "Qingdao"]
CATEGORIES = ["Electronics", "Clothing", "Food", "Sports", "Books",
              "Home", "Auto", "Health", "Travel", "Gaming"]


def generate_data(n: int):
    """Generate test data as columnar dict."""
    rng = random.Random(42)
    names = [f"user_{i}" for i in range(n)]
    ages = [rng.randint(18, 80) for _ in range(n)]
    scores = [round(rng.uniform(0, 100), 2) for _ in range(n)]
    cities = [rng.choice(CITIES) for _ in range(n)]
    categories = [rng.choice(CATEGORIES) for _ in range(n)]
    return {
        "name": names,
        "age": ages,
        "score": scores,
        "city": cities,
        "category": categories,
    }


@contextmanager
def timer():
    """Context manager that yields a dict; sets 'elapsed_ms' on exit."""
    result = {}
    gc.collect()
    t0 = time.perf_counter()
    yield result
    result["elapsed_ms"] = (time.perf_counter() - t0) * 1000


def fmt_ms(ms):
    if ms < 0.01:
        return f"{ms * 1000:.2f}us"
    if ms < 1:
        return f"{ms:.3f}ms"
    if ms < 1000:
        return f"{ms:.2f}ms"
    return f"{ms / 1000:.2f}s"


def run_bench(fn, warmup=2, iterations=5):
    """Run fn() with warmup, return average ms."""
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(iterations):
        with timer() as t:
            fn()
        times.append(t["elapsed_ms"])
    return sum(times) / len(times)


def run_bench_cold(setup_fn, bench_fn, warmup=2, iterations=5):
    """Cold-start benchmark: re-run setup_fn() before EVERY iteration.
    Measures true from-scratch query latency (no warm caches)."""
    for _ in range(warmup):
        setup_fn()
        bench_fn()
    times = []
    for _ in range(iterations):
        setup_fn()   # cold restart — invalidates all in-process caches
        gc.collect()
        with timer() as t:
            bench_fn()
        times.append(t["elapsed_ms"])
    return sum(times) / len(times)


def run_bench_cold_nogc(setup_fn, bench_fn, warmup=2, iterations=5):
    """Cold-start benchmark WITHOUT gc.collect() before timing.
    Measures file-open + query latency without CPU-cache eviction noise."""
    for _ in range(warmup):
        setup_fn()
        bench_fn()
    times = []
    for _ in range(iterations):
        setup_fn()
        t0 = time.perf_counter()
        bench_fn()
        times.append((time.perf_counter() - t0) * 1000)
    return sum(times) / len(times)


def run_bench_nogc(fn, warmup=2, iterations=5):
    """Warm benchmark WITHOUT gc.collect() before timing."""
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        fn()
        times.append((time.perf_counter() - t0) * 1000)
    return sum(times) / len(times)


def run_bench_with_setup(setup_fn, bench_fn, warmup=2, iterations=5):
    """Per-iteration setup WITHOUT cold-start (no DB reopen). Times only bench_fn."""
    for _ in range(warmup):
        setup_fn()
        bench_fn()
    times = []
    for _ in range(iterations):
        setup_fn()
        t0 = time.perf_counter()
        bench_fn()
        times.append((time.perf_counter() - t0) * 1000)
    return sum(times) / len(times)


def measure_rss_mb():
    """Return current process RSS in MB (requires psutil)."""
    try:
        import psutil
        return psutil.Process().memory_info().rss / (1024 * 1024)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# SQLite benchmark
# ---------------------------------------------------------------------------

class SQLiteBench:
    def __init__(self, tmpdir, data):
        self.db_path = os.path.join(tmpdir, "bench.db")
        self.data = data
        self.n = len(data["name"])
        self.conn = None

    def setup(self):
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

    def bench_insert(self):
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

    def bench_select_limit(self, limit=100):
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

    def bench_group_by(self):
        return self.conn.execute(
            "SELECT city, COUNT(*), AVG(score) FROM bench GROUP BY city"
        ).fetchall()

    def bench_group_by_having(self):
        return self.conn.execute(
            "SELECT city, COUNT(*) as cnt, AVG(score) FROM bench GROUP BY city HAVING cnt > 1000"
        ).fetchall()

    def bench_order_limit(self):
        return self.conn.execute(
            "SELECT * FROM bench ORDER BY score DESC LIMIT 100"
        ).fetchall()

    def bench_aggregation(self):
        return self.conn.execute(
            "SELECT COUNT(*), AVG(age), SUM(score), MIN(age), MAX(age) FROM bench"
        ).fetchone()

    def bench_complex(self):
        return self.conn.execute(
            "SELECT city, AVG(score) as avg_s FROM bench WHERE age BETWEEN 25 AND 50 GROUP BY city ORDER BY avg_s DESC LIMIT 5"
        ).fetchall()

    def bench_point_lookup(self):
        return self.conn.execute(
            "SELECT * FROM bench WHERE _id = 5000"
        ).fetchone()

    def bench_retrieve_many(self):
        # Retrieve 100 random IDs
        ids = random.sample(range(1, self.n + 1), 100)
        placeholders = ",".join(["?"] * 100)
        return self.conn.execute(
            f"SELECT * FROM bench WHERE _id IN ({placeholders})", ids
        ).fetchall()

    def bench_insert_1k(self):
        rows = [(f"new_{i}", 25, 50.0, "Beijing", "Books") for i in range(1000)]
        self.conn.executemany(
            "INSERT INTO bench (name, age, score, city, category) VALUES (?,?,?,?,?)",
            rows,
        )
        self.conn.commit()

    def bench_full_scan_pandas(self):
        if HAS_PANDAS:
            return pd.read_sql("SELECT * FROM bench", self.conn)
        return self.conn.execute("SELECT * FROM bench").fetchall()

    def bench_group_by_2cols(self):
        return self.conn.execute(
            "SELECT city, category, COUNT(*), AVG(score) FROM bench GROUP BY city, category"
        ).fetchall()

    def bench_filter_like(self):
        return self.conn.execute(
            "SELECT * FROM bench WHERE name LIKE 'user_1%'"
        ).fetchall()

    def bench_filter_multi_cond(self):
        return self.conn.execute(
            "SELECT * FROM bench WHERE age > 30 AND score > 50.0"
        ).fetchall()

    def bench_order_by_multi(self):
        return self.conn.execute(
            "SELECT * FROM bench ORDER BY city ASC, score DESC LIMIT 100"
        ).fetchall()

    def bench_count_distinct(self):
        return self.conn.execute(
            "SELECT COUNT(DISTINCT city) FROM bench"
        ).fetchone()[0]

    def bench_filter_in(self):
        return self.conn.execute(
            "SELECT * FROM bench WHERE city IN ('Beijing', 'Shanghai', 'Guangzhou')"
        ).fetchall()

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

    def bench_delete_1k_setup(self):
        rows = [(f"del_{i}", 99, 99.0, "Beijing", "Books") for i in range(1000)]
        self.conn.executemany(
            "INSERT INTO bench (name, age, score, city, category) VALUES (?,?,?,?,?)",
            rows,
        )
        self.conn.commit()

    def bench_delete_1k_only(self):
        self.conn.execute("DELETE FROM bench WHERE age = 99")
        self.conn.commit()

    def bench_window_row_number(self):
        return self.conn.execute(
            "SELECT name, city, score, "
            "ROW_NUMBER() OVER (PARTITION BY city ORDER BY score DESC) as rn "
            "FROM bench LIMIT 1000"
        ).fetchall()

    def bench_fts_build(self):
        try:
            self.conn.execute("DROP TABLE IF EXISTS bench_fts")
            self.conn.execute("""
                CREATE VIRTUAL TABLE bench_fts
                USING fts5(name, city, category, content='bench', content_rowid='_id')
            """)
            self.conn.execute("INSERT INTO bench_fts(bench_fts) VALUES('rebuild')")
            self.conn.commit()
            self._fts_ready = True
        except Exception:
            self._fts_ready = False

    def bench_fts_search(self):
        if not getattr(self, '_fts_ready', False):
            return None
        return self.conn.execute(
            "SELECT rowid FROM bench_fts WHERE bench_fts MATCH 'Electronics'"
        ).fetchall()

    def close(self):
        if self.conn:
            self.conn.close()


# ---------------------------------------------------------------------------
# DuckDB benchmark
# ---------------------------------------------------------------------------

class DuckDBBench:
    def __init__(self, tmpdir, data):
        self.db_path = os.path.join(tmpdir, "bench.duckdb")
        self.data = data
        self.n = len(data["name"])
        self.conn = None

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

    def bench_insert(self):
        # Use executemany for reliable cross-version compatibility
        rows = list(zip(
            self.data["name"], self.data["age"], self.data["score"],
            self.data["city"], self.data["category"]
        ))
        self.conn.executemany(
            "INSERT INTO bench VALUES (?,?,?,?,?)", rows
        )

    def bench_count(self):
        return self.conn.execute("SELECT COUNT(*) FROM bench").fetchone()[0]

    def bench_select_limit(self, limit=100):
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

    def bench_group_by(self):
        return self.conn.execute(
            "SELECT city, COUNT(*), AVG(score) FROM bench GROUP BY city"
        ).fetchall()

    def bench_group_by_having(self):
        return self.conn.execute(
            "SELECT city, COUNT(*) as cnt, AVG(score) FROM bench GROUP BY city HAVING cnt > 1000"
        ).fetchall()

    def bench_order_limit(self):
        return self.conn.execute(
            "SELECT * FROM bench ORDER BY score DESC LIMIT 100"
        ).fetchall()

    def bench_aggregation(self):
        return self.conn.execute(
            "SELECT COUNT(*), AVG(age), SUM(score), MIN(age), MAX(age) FROM bench"
        ).fetchone()

    def bench_complex(self):
        return self.conn.execute(
            "SELECT city, AVG(score) as avg_s FROM bench WHERE age BETWEEN 25 AND 50 GROUP BY city ORDER BY avg_s DESC LIMIT 5"
        ).fetchall()

    def bench_point_lookup(self):
        return self.conn.execute(
            "SELECT * FROM bench WHERE rowid = 5000"
        ).fetchall()

    def bench_retrieve_many(self):
        # Retrieve 100 random IDs using rowid
        ids = random.sample(range(1, self.n + 1), 100)
        placeholders = ",".join(["?"] * 100)
        return self.conn.execute(
            f"SELECT * FROM bench WHERE rowid IN ({placeholders})", ids
        ).fetchall()

    def bench_insert_1k(self):
        # Use executemany for reliable cross-version compatibility
        rows = [(f"new_{i}", 25, 50.0, "Beijing", "Books") for i in range(1000)]
        self.conn.executemany("INSERT INTO bench VALUES (?,?,?,?,?)", rows)

    def bench_full_scan_pandas(self):
        if HAS_PANDAS:
            return self.conn.execute("SELECT * FROM bench").df()
        return self.conn.execute("SELECT * FROM bench").fetchall()

    def bench_group_by_2cols(self):
        return self.conn.execute(
            "SELECT city, category, COUNT(*), AVG(score) FROM bench GROUP BY city, category"
        ).fetchall()

    def bench_filter_like(self):
        return self.conn.execute(
            "SELECT * FROM bench WHERE name LIKE 'user_1%'"
        ).fetchall()

    def bench_filter_multi_cond(self):
        return self.conn.execute(
            "SELECT * FROM bench WHERE age > 30 AND score > 50.0"
        ).fetchall()

    def bench_order_by_multi(self):
        return self.conn.execute(
            "SELECT * FROM bench ORDER BY city ASC, score DESC LIMIT 100"
        ).fetchall()

    def bench_count_distinct(self):
        return self.conn.execute(
            "SELECT COUNT(DISTINCT city) FROM bench"
        ).fetchone()[0]

    def bench_filter_in(self):
        return self.conn.execute(
            "SELECT * FROM bench WHERE city IN ('Beijing', 'Shanghai', 'Guangzhou')"
        ).fetchall()

    def bench_update_1k(self):
        self.conn.execute("UPDATE bench SET score = 50.0 WHERE age = 25")

    def bench_delete_1k(self):
        if HAS_PANDAS:
            rows = [(f"del_{i}", 99, 99.0, "Beijing", "Books") for i in range(1000)]
            self.conn.executemany("INSERT INTO bench VALUES (?,?,?,?,?)", rows)
        self.conn.execute("DELETE FROM bench WHERE age = 99")

    def bench_delete_1k_setup(self):
        rows = [(f"del_{i}", 99, 99.0, "Beijing", "Books") for i in range(1000)]
        self.conn.executemany("INSERT INTO bench VALUES (?,?,?,?,?)", rows)

    def bench_delete_1k_only(self):
        self.conn.execute("DELETE FROM bench WHERE age = 99")

    def bench_window_row_number(self):
        return self.conn.execute(
            "SELECT name, city, score, "
            "ROW_NUMBER() OVER (PARTITION BY city ORDER BY score DESC) as rn "
            "FROM bench LIMIT 1000"
        ).fetchall()

    def bench_fts_build(self):
        try:
            self.conn.execute("INSTALL fts")
            self.conn.execute("LOAD fts")
            self.conn.execute(
                "PRAGMA create_fts_index('bench', 'rowid', 'name', 'city', 'category')"
            )
            self._fts_ready = True
        except Exception:
            self._fts_ready = False

    def bench_fts_search(self):
        if not getattr(self, '_fts_ready', False):
            return None
        try:
            return self.conn.execute(
                "SELECT * FROM bench "
                "WHERE fts_main_bench.match_bm25(rowid, 'Electronics') IS NOT NULL"
            ).fetchall()
        except Exception:
            return None

    def close(self):
        if self.conn:
            self.conn.close()


# ---------------------------------------------------------------------------
# ApexBase benchmark
# ---------------------------------------------------------------------------

class ApexBaseBench:
    def __init__(self, tmpdir, data, low_memory=False):
        self.db_dir = os.path.join(tmpdir, "apex_bench")
        self.data = data
        self.n = len(data["name"])
        self.client = None
        self.low_memory = low_memory

    def setup(self):
        if os.path.exists(self.db_dir):
            shutil.rmtree(self.db_dir)
        self.client = ApexClient(self.db_dir, drop_if_exists=True)
        self.client.create_table('default')

    def cold_start_setup(self):
        """Close and reopen client — clears all Python/Rust-side caches (arrow_batch_cache etc.)."""
        if self.client:
            self.client.close()
        self.client = ApexClient(self.db_dir)
        self.client.use_table('default')

    def bench_insert(self):
        self.client.store(self.data)

    def bench_count(self):
        return self.client.execute("SELECT COUNT(*) FROM default").scalar()

    def bench_select_limit(self, limit=100):
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

    def bench_group_by(self):
        return self.client.execute(
            "SELECT city, COUNT(*), AVG(score) FROM default GROUP BY city"
        )

    def bench_group_by_having(self):
        return self.client.execute(
            "SELECT city, COUNT(*) as cnt, AVG(score) FROM default GROUP BY city HAVING cnt > 1000"
        )

    def bench_order_limit(self):
        return self.client.execute(
            "SELECT * FROM default ORDER BY score DESC LIMIT 100"
        )

    def bench_aggregation(self):
        return self.client.execute(
            "SELECT COUNT(*), AVG(age), SUM(score), MIN(age), MAX(age) FROM default"
        )

    def bench_complex(self):
        return self.client.execute(
            "SELECT city, AVG(score) as avg_s FROM default WHERE age BETWEEN 25 AND 50 GROUP BY city ORDER BY avg_s DESC LIMIT 5"
        )

    def bench_point_lookup(self):
        # Use optimized retrieve API for point lookup by ID
        return self.client.retrieve(5000)

    def bench_retrieve_many(self):
        # Retrieve 100 random IDs using retrieve_many
        ids = random.sample(range(1, self.n + 1), 100)
        return self.client.retrieve_many(ids)

    def bench_insert_1k(self):
        data_1k = {
            "name": [f"new_{i}" for i in range(1000)],
            "age": [25] * 1000,
            "score": [50.0] * 1000,
            "city": ["Beijing"] * 1000,
            "category": ["Books"] * 1000,
        }
        self.client.store(data_1k)

    def bench_full_scan_pandas(self):
        result = self.client.execute("SELECT * FROM default")
        if HAS_PANDAS:
            return result.to_pandas()
        return result

    def bench_group_by_2cols(self):
        return self.client.execute(
            "SELECT city, category, COUNT(*), AVG(score) FROM default GROUP BY city, category"
        )

    def bench_filter_like(self):
        return self.client.execute(
            "SELECT * FROM default WHERE name LIKE 'user_1%'"
        )

    def bench_filter_multi_cond(self):
        return self.client.execute(
            "SELECT * FROM default WHERE age > 30 AND score > 50.0"
        )

    def bench_order_by_multi(self):
        return self.client.execute(
            "SELECT * FROM default ORDER BY city ASC, score DESC LIMIT 100"
        )

    def bench_count_distinct(self):
        return self.client.execute(
            "SELECT COUNT(DISTINCT city) FROM default"
        )

    def bench_filter_in(self):
        return self.client.execute(
            "SELECT * FROM default WHERE city IN ('Beijing', 'Shanghai', 'Guangzhou')"
        )

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

    def bench_delete_1k_setup(self):
        data = {
            "name": [f"del_{i}" for i in range(1000)],
            "age": [99] * 1000,
            "score": [99.0] * 1000,
            "city": ["Beijing"] * 1000,
            "category": ["Books"] * 1000,
        }
        self.client.store(data)

    def bench_delete_1k_only(self):
        self.client.execute("DELETE FROM default WHERE age = 99")

    def bench_window_row_number(self):
        return self.client.execute(
            "SELECT name, city, score, "
            "ROW_NUMBER() OVER (PARTITION BY city ORDER BY score DESC) as rn "
            "FROM default LIMIT 1000"
        )

    def bench_fts_build(self):
        try:
            # Close + reopen to clear the executor STORAGE_CACHE before backfill,
            # ensuring CREATE FTS INDEX reads fresh row data from disk.
            self.client.close()
            self.client = ApexClient(self.db_dir)
            self.client.use_table('default')
            self.client.execute(
                "CREATE FTS INDEX ON default (name, city, category)"
            )
            self.client._fts_tables['default'] = {
                'enabled': True,
                'index_fields': ['name', 'city', 'category'],
                'config': {'lazy_load': False, 'cache_size': 10000},
            }
            self._fts_ready = True
        except Exception:
            self._fts_ready = False

    def bench_fts_search(self):
        if not getattr(self, '_fts_ready', False):
            return None
        return self.client.search_text('Electronics')

    def close(self):
        if self.client:
            self.client.close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

# (display_name, method_name, is_write, is_cold, is_warm_nogc, setup_method)
# is_cold=True      -> cold_start_setup() per iter (DB reopen), no gc
# is_warm_nogc=True -> warm cached backend, no gc
# setup_method      -> if not None, call bench.{setup_method}() before each timed iter
BENCHMARKS = [
    ("Bulk Insert (N rows)",             "bench_insert",           True,  False, False, None),
    ("COUNT(*)",                         "bench_count",            False, False, False, None),
    ("SELECT * LIMIT 100 [cold]",        "bench_select_limit",     False, True,  False, None),
    ("SELECT * LIMIT 100 [warm]",        "bench_select_limit",     False, False, True,  None),
    ("SELECT * LIMIT 10K [cold]",        "bench_select_limit_10k", False, True,  False, None),
    ("SELECT * LIMIT 10K [warm]",        "bench_select_limit_10k", False, False, True,  None),
    ("Filter (name = 'user_5000')",      "bench_filter_string",    False, False, False, None),
    ("Filter (age BETWEEN 25 AND 35)",   "bench_filter_range",     False, False, False, None),
    ("GROUP BY city (10 groups)",        "bench_group_by",         False, False, False, None),
    ("GROUP BY + HAVING",                "bench_group_by_having",  False, False, False, None),
    ("ORDER BY score LIMIT 100",         "bench_order_limit",      False, False, False, None),
    ("Aggregation (5 funcs)",            "bench_aggregation",      False, False, False, None),
    ("Complex (Filter+Group+Order)",     "bench_complex",          False, False, False, None),
    ("Point Lookup (by ID)",             "bench_point_lookup",     False, False, False, None),
    ("Retrieve Many (100 IDs)",          "bench_retrieve_many",    False, False, False, None),
    ("Insert 1K rows",                   "bench_insert_1k",        False, False, False, None),
    # --- New cases ---
    ("SELECT * -> pandas (full scan)",   "bench_full_scan_pandas", False, False, False, None),
    ("GROUP BY city,category (100 grp)","bench_group_by_2cols",   False, False, False, None),
    ("LIKE filter (name LIKE user_1%)",  "bench_filter_like",      False, False, False, None),
    ("Multi-cond (age>30 AND score>50)", "bench_filter_multi_cond",False, False, False, None),
    ("ORDER BY city,score DESC LIMIT100","bench_order_by_multi",   False, False, False, None),
    ("COUNT(DISTINCT city)",             "bench_count_distinct",   False, False, False, None),
    ("IN filter (city IN 3 cities)",     "bench_filter_in",        False, False, False, None),
    ("UPDATE rows (age=25, idempotent)", "bench_update_1k",        False, False, False, None),
    # --- Delete / Window / FTS ---
    ("Store+DELETE 1K (combined)",           "bench_delete_1k",         False, False, False, None),
    ("DELETE 1K [pure delete only]",         "bench_delete_1k_only",    False, False, False, "bench_delete_1k_setup"),
    ("Window ROW_NUMBER PARTITION BY city",  "bench_window_row_number", False, False, False, None),
    ("FTS Index Build (name,city,category)", "bench_fts_build",         True,  False, False, None),
    ("FTS Search ('Electronics')",           "bench_fts_search",        False, False, False, None),
]


def get_system_info():
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
    if HAS_PYARROW:
        info["pyarrow"] = pa.__version__
    return info


def main():
    parser = argparse.ArgumentParser(description="ApexBase vs SQLite vs DuckDB benchmark")
    parser.add_argument("--rows", type=int, default=1_000_000, help="Number of rows (default: 1M)")
    parser.add_argument("--warmup", type=int, default=2, help="Warmup iterations (default: 2)")
    parser.add_argument("--iterations", type=int, default=5, help="Timed iterations (default: 5)")
    parser.add_argument("--output", type=str, default=None, help="JSON output file")
    parser.add_argument("--memory", action="store_true", help="Track RSS memory delta per query")
    parser.add_argument("--low-memory", action="store_true",
                        help="Disable ApexBase arrow_batch_cache (simulate low-memory mode like SQLite/DuckDB)")
    args = parser.parse_args()

    N = args.rows
    WARMUP = args.warmup
    ITERS = args.iterations

    print("=" * 80)
    print(f" ApexBase vs SQLite vs DuckDB — Performance Benchmark")
    print("=" * 80)

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
    if HAS_PYARROW:
        print(f"PyArrow: v{sys_info['pyarrow']}")
    print(f"\nDataset: {N:,} rows × 5 columns (name, age, score, city, category)")
    print(f"Warmup: {WARMUP} iterations, Timed: {ITERS} iterations (average)")
    if args.low_memory:
        print("Mode: LOW-MEMORY (ApexBase arrow_batch_cache disabled)")
    print()

    # Generate data
    print("Generating test data...", end=" ", flush=True)
    data = generate_data(N)
    print("done.")

    tmpdir = tempfile.mkdtemp(prefix="apexbase_bench_")
    results = {}

    engines = []
    if HAS_APEXBASE:
        engines.append(("ApexBase", ApexBaseBench(tmpdir, data, low_memory=args.low_memory)))
    engines.append(("SQLite", SQLiteBench(tmpdir, data)))
    if HAS_DUCKDB:
        engines.append(("DuckDB", DuckDBBench(tmpdir, data)))

    if not engines:
        print("ERROR: No database engines available!")
        return

    # Setup all engines
    for name, bench in engines:
        bench.setup()

    mem_results = {}  # bench_name -> {eng_name: rss_delta_mb}

    # Run benchmarks
    for bench_name, method_name, is_insert, is_cold, is_warm_nogc, setup_method in BENCHMARKS:
        results[bench_name] = {}
        mem_results.setdefault(bench_name, {})
        for eng_name, bench in engines:
            fn = getattr(bench, method_name, None)
            if fn is None:
                results[bench_name][eng_name] = None
                continue

            if is_insert:
                rss_before = measure_rss_mb()
                with timer() as t:
                    fn()
                ms = t["elapsed_ms"]
                rss_after = measure_rss_mb()
                results[bench_name][eng_name] = ms
                if rss_before and rss_after:
                    mem_results[bench_name][eng_name] = rss_after - rss_before
            elif is_cold:
                # Cold-start (no gc): reopen DB before each iteration
                cold_setup_fn = getattr(bench, "cold_start_setup", None)
                if cold_setup_fn is None:
                    # Engine has no cold_start_setup — run warm no-gc
                    ms = run_bench_nogc(fn, warmup=WARMUP, iterations=ITERS)
                else:
                    rss_before = measure_rss_mb()
                    ms = run_bench_cold_nogc(cold_setup_fn, fn, warmup=WARMUP, iterations=ITERS)
                    rss_after = measure_rss_mb()
                    if rss_before and rss_after:
                        mem_results[bench_name][eng_name] = rss_after - rss_before
                results[bench_name][eng_name] = ms
            elif is_warm_nogc:
                # Warm cached backend, no gc
                rss_before = measure_rss_mb()
                ms = run_bench_nogc(fn, warmup=WARMUP, iterations=ITERS)
                rss_after = measure_rss_mb()
                results[bench_name][eng_name] = ms
                if rss_before and rss_after:
                    mem_results[bench_name][eng_name] = rss_after - rss_before
            elif setup_method is not None:
                # Per-iteration setup (e.g. pre-insert rows before timing delete)
                per_setup_fn = getattr(bench, setup_method, None)
                if per_setup_fn is None:
                    results[bench_name][eng_name] = None
                    continue
                rss_before = measure_rss_mb()
                ms = run_bench_with_setup(per_setup_fn, fn, warmup=WARMUP, iterations=ITERS)
                rss_after = measure_rss_mb()
                results[bench_name][eng_name] = ms
                if rss_before and rss_after:
                    mem_results[bench_name][eng_name] = rss_after - rss_before
            else:
                rss_before = measure_rss_mb()
                # low_memory mode: disable arrow_batch_cache for ApexBase by reopening each iter
                cold_setup_fn = getattr(bench, "cold_start_setup", None)
                use_cold = (HAS_APEXBASE and isinstance(bench, ApexBaseBench)
                            and bench.low_memory and cold_setup_fn is not None)
                if use_cold:
                    ms = run_bench_cold_nogc(cold_setup_fn, fn, warmup=WARMUP, iterations=ITERS)
                else:
                    ms = run_bench(fn, warmup=WARMUP, iterations=ITERS)
                rss_after = measure_rss_mb()
                results[bench_name][eng_name] = ms
                if rss_before and rss_after:
                    mem_results[bench_name][eng_name] = rss_after - rss_before

    # Cleanup
    for name, bench in engines:
        bench.close()

    # Print results table
    eng_names = [name for name, _ in engines]
    col_width = 16

    print()
    header = f"{'Query':<42}"
    for name in eng_names:
        header += f" | {name:>{col_width}}"
    if len(eng_names) >= 2:
        header += f" | {'Ratio (Apex/Best)':>{col_width}}"
    print(header)
    print("-" * len(header))

    json_results = []
    for bench_name, method_name, is_insert, is_cold, is_warm_nogc, setup_method in BENCHMARKS:
        row = f"{bench_name:<42}"
        values = {}
        for eng_name in eng_names:
            ms = results.get(bench_name, {}).get(eng_name)
            if ms is not None:
                row += f" | {fmt_ms(ms):>{col_width}}"
                values[eng_name] = ms
            else:
                row += f" | {'N/A':>{col_width}}"

        if len(eng_names) >= 2 and "ApexBase" in values:
            others = {k: v for k, v in values.items() if k != "ApexBase"}
            if others:
                best_other = min(others.values())
                ratio = values["ApexBase"] / best_other if best_other > 0 else float("inf")
                if ratio < 1:
                    label = f"{ratio:.2f}x (faster)"
                elif ratio < 1.05:
                    label = f"~1.0x (tied)"
                else:
                    label = f"{ratio:.1f}x (slower)"
                row += f" | {label:>{col_width}}"

        print(row)
        json_results.append({
            "query": bench_name,
            **{k: round(v, 3) for k, v in values.items()},
        })

    print()

    # Memory summary (if tracking enabled)
    if args.memory:
        print()
        print("Memory delta per query (RSS change, MB):")
        mem_header = f"  {'Query':<42}"
        for name in eng_names:
            mem_header += f" | {name:>{col_width}}"
        print(mem_header)
        print("  " + "-" * (len(mem_header) - 2))
        for bench_name, _, _, _, _, _ in BENCHMARKS:
            row = f"  {bench_name:<42}"
            for eng_name in eng_names:
                delta = mem_results.get(bench_name, {}).get(eng_name)
                if delta is not None:
                    row += f" | {delta:>+{col_width}.1f}"
                else:
                    row += f" | {'N/A':>{col_width}}"
            print(row)

    # Summary
    if "ApexBase" in [n for n, _ in engines]:
        wins = 0
        ties = 0
        total = 0
        for bench_name, _, _, _, _, _ in BENCHMARKS:
            vals = results.get(bench_name, {})
            apex_ms = vals.get("ApexBase")
            if apex_ms is None:
                continue
            others = {k: v for k, v in vals.items() if k != "ApexBase" and v is not None}
            if not others:
                continue
            total += 1
            best_other = min(others.values())
            ratio = apex_ms / best_other if best_other > 0 else float("inf")
            if ratio < 0.95:
                wins += 1
            elif ratio <= 1.05:
                ties += 1
        losses = total - wins - ties
        print(f"Summary: ApexBase wins {wins}/{total}, ties {ties}/{total}, slower {losses}/{total}")

    # ========================================================================
    # Q/s Throughput Tests (Single & Concurrent)
    # ========================================================================
    from concurrent.futures import ThreadPoolExecutor
    import threading

    # Q/s test queries - mix of simple and complex
    QPS_QUERIES_APEX = [
        "SELECT COUNT(*) FROM default",
        "SELECT city, COUNT(*) FROM default GROUP BY city",
        "SELECT category, AVG(score) FROM default GROUP BY category",
        "SELECT * FROM default WHERE age > 30 LIMIT 100",
    ]
    
    QPS_QUERIES_OTHER = [
        "SELECT COUNT(*) FROM bench",
        "SELECT city, COUNT(*) FROM bench GROUP BY city",
        "SELECT category, AVG(score) FROM bench GROUP BY category",
        "SELECT * FROM bench WHERE age > 30 LIMIT 100",
    ]

    def run_qps_benchmark(tmpdir, data, n_threads=4, min_duration=1.0, min_iterations=100):
        """Measure Q/s (queries per second) for single-threaded and concurrent scenarios.
        
        Args:
            min_duration: Minimum test duration in seconds for accurate timing
            min_iterations: Minimum number of query batches to run
        """
        results = {}

        # Create fresh engines for Q/s tests
        qps_engines = []
        if HAS_APEXBASE:
            qps_engines.append(("ApexBase", ApexBaseBench(tmpdir, data), QPS_QUERIES_APEX))
        qps_engines.append(("SQLite", SQLiteBench(tmpdir, data), QPS_QUERIES_OTHER))
        if HAS_DUCKDB:
            qps_engines.append(("DuckDB", DuckDBBench(tmpdir, data), QPS_QUERIES_OTHER))

        # Setup fresh engines and insert data
        for name, bench, _ in qps_engines:
            bench.setup()
            # Insert data for Q/S tests
            if hasattr(bench, 'bench_insert'):
                bench.bench_insert()

        # Pre-warm all engines - run each query once to ensure caching is comparable
        for name, bench, queries in qps_engines:
            try:
                if hasattr(bench, 'client'):
                    # ApexBase: execute and materialize results
                    for q in queries:
                        _ = bench.client.execute(q).to_pandas()
                elif hasattr(bench, 'conn'):
                    # DuckDB/SQLite: execute and materialize results
                    for q in queries:
                        _ = bench.conn.execute(q).fetchall()
            except Exception:
                pass

        # 1. Single-threaded Q/s
        print("\n--- Single-threaded Q/s ---")
        for name, bench, queries in qps_engines:
            try:
                # Determine execute method - make it consistent across engines
                # All engines will use Arrow for fair comparison (ApexBase uses Arrow internally)
                is_duckdb = hasattr(bench, 'conn') and 'duckdb' in str(type(bench.conn)).lower()

                if hasattr(bench, 'client'):
                    # ApexBase: use execute() for single queries (more efficient than scheduler for n=1)
                    # The scheduler is optimized for batch workloads with multiple queries
                    exec_fn = lambda q, b=bench: list(bench.client.execute(q))
                elif hasattr(bench, 'conn'):
                    if is_duckdb:
                        # DuckDB: use Arrow for fair comparison with ApexBase
                        exec_fn = lambda q, b=bench: b.conn.execute(q).arrow().to_pydict()
                    else:
                        # SQLite: use fetchall() as baseline
                        exec_fn = lambda q, b=bench: b.conn.execute(q).fetchall()
                else:
                    continue

                # Run sequential queries with proper timing
                gc.collect()
                
                # First, determine appropriate number of iterations based on time
                # Run a small batch first to estimate time per query
                batch_size = len(queries)
                trial_iterations = 5
                
                t0 = time.perf_counter()
                for _ in range(trial_iterations):
                    for q in queries:
                        exec_fn(q)
                trial_time = time.perf_counter() - t0
                
                # Calculate iterations needed for min_duration seconds
                # At least min_iterations batches, but also enough to run for min_duration
                time_per_batch = trial_time / trial_iterations
                iterations = max(min_iterations, int(min_duration / time_per_batch))
                
                # Run the actual benchmark
                t0 = time.perf_counter()
                for _ in range(iterations):
                    for q in queries:
                        exec_fn(q)
                elapsed = time.perf_counter() - t0

                total_queries = iterations * len(queries)
                qps = total_queries / elapsed if elapsed > 0 else 0
                results[f'{name}_single'] = qps
                print(f"  {name}: {qps:.1f} Q/s ({total_queries} queries in {elapsed:.3f}s)")
            except Exception as e:
                print(f"  {name}: Error - {e}")
            finally:
                bench.close()

        # 2. Concurrent Q/s (multiple threads)
        print(f"\n--- Concurrent Q/s ({n_threads} threads) ---")
        
        # Create engines for concurrent test - use fresh instances
        concurrent_engines = []
        if HAS_APEXBASE:
            concurrent_engines.append(("ApexBase", ApexBaseBench(tmpdir, data), QPS_QUERIES_APEX))
        concurrent_engines.append(("SQLite", SQLiteBench(tmpdir, data), QPS_QUERIES_OTHER))
        if HAS_DUCKDB:
            concurrent_engines.append(("DuckDB", DuckDBBench(tmpdir, data), QPS_QUERIES_OTHER))

        for name, bench, _ in concurrent_engines:
            bench.setup()
            # Insert data for Q/S tests
            if hasattr(bench, 'bench_insert'):
                bench.bench_insert()

        for name, bench, queries in concurrent_engines:
            try:
                # For each thread, we need a separate connection for SQLite/DuckDB
                # ApexBase handles concurrency internally
                if hasattr(bench, 'client'):
                    # ApexBase: shared client
                    def concurrent_worker_apex():
                        for _ in range(iterations):
                            for q in queries:
                                list(bench.client.execute(q))
                    
                    gc.collect()
                    t0 = time.perf_counter()
                    with ThreadPoolExecutor(max_workers=n_threads) as executor:
                        list(executor.map(lambda _: concurrent_worker_apex(), range(n_threads)))
                    elapsed = time.perf_counter() - t0
                    total_queries = n_threads * iterations * len(queries)
                    qps = total_queries / elapsed if elapsed > 0.001 else 0
                    
                elif hasattr(bench, 'conn'):
                    # SQLite/DuckDB: create connection per thread
                    db_path = bench.db_path
                    is_duckdb = 'duckdb' in str(type(bench.conn)).lower()

                    def concurrent_worker_sql(db_path, is_duckdb, queries, iterations):
                        # Each thread creates its own connection
                        try:
                            if is_duckdb:
                                import duckdb
                                conn = duckdb.connect(db_path)
                            else:
                                import sqlite3
                                conn = sqlite3.connect(db_path, timeout=30)
                            for _ in range(iterations):
                                for q in queries:
                                    # Use Arrow for DuckDB (fair comparison with ApexBase)
                                    # For SQLite, use fetchall() as baseline
                                    if is_duckdb:
                                        result = conn.execute(q)
                                        # Use Arrow for DuckDB - most efficient path
                                        arrow_result = result.arrow()
                                        # Materialize to match ApexBase behavior
                                        _ = arrow_result.to_pydict()
                                    else:
                                        conn.execute(q).fetchall()
                            conn.close()
                        except Exception:
                            pass
                    
                    gc.collect()
                    t0 = time.perf_counter()
                    with ThreadPoolExecutor(max_workers=n_threads) as executor:
                        list(executor.map(
                            lambda _: concurrent_worker_sql(db_path, is_duckdb, queries, iterations),
                            range(n_threads)
                        ))
                    elapsed = time.perf_counter() - t0
                    total_queries = n_threads * iterations * len(queries)
                    qps = total_queries / elapsed if elapsed > 0.001 else 0
                else:
                    qps = 0
                    total_queries = 0
                    elapsed = 0

                results[f'{name}_concurrent_{n_threads}'] = qps
                if elapsed > 0.001:
                    print(f"  {name}: {qps:.1f} Q/s ({total_queries} queries in {elapsed:.3f}s)")
                else:
                    print(f"  {name}: Error - test time too short ({elapsed:.6f}s)")
            except Exception as e:
                print(f"  {name}: Error - {e}")
            finally:
                bench.close()

        # Print Q/s Summary
        print("\n--- Q/s Summary ---")
        apex_single = results.get("ApexBase_single", 0)
        sqlite_single = results.get("SQLite_single", 0)
        duckdb_single = results.get("DuckDB_single", 0)
        apex_concurrent = results.get("ApexBase_concurrent_4", 0)
        sqlite_concurrent = results.get("SQLite_concurrent_4", 0)
        duckdb_concurrent = results.get("DuckDB_concurrent_4", 0)
        print(f"  ApexBase (single-threaded): {apex_single:.1f} Q/s")
        print(f"  SQLite (single-threaded): {sqlite_single:.1f} Q/s")
        print(f"  DuckDB (single-threaded): {duckdb_single:.1f} Q/s")
        print(f"  ApexBase (4-thread concurrent): {apex_concurrent:.1f} Q/s")
        print(f"  SQLite (4-thread concurrent): {sqlite_concurrent:.1f} Q/s")
        print(f"  DuckDB (4-thread concurrent): {duckdb_concurrent:.1f} Q/s")

        return results

    # Run Q/s tests with improved timing
    qps_results = run_qps_benchmark(tmpdir, data, n_threads=4, min_duration=3.0, min_iterations=500)

    # Save JSON if requested
    if args.output:
        output = {
            "system": sys_info,
            "config": {"rows": N, "warmup": WARMUP, "iterations": ITERS},
            "results": json_results,
        }
        with open(args.output, "w") as f:
            json.dump(output, f, indent=2)
        print(f"Results saved to {args.output}")

    # Cleanup tmpdir
    try:
        shutil.rmtree(tmpdir)
    except Exception:
        pass


if __name__ == "__main__":
    main()
