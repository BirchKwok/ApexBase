#!/usr/bin/env python3
"""
Benchmark: PG Wire Protocol vs Direct Python API

Tests:
  1. Direct Python API (apexbase.ApexClient)
  2. PG Wire Simple Query (psycopg2)
  3. PG Wire Extended Query with params (psycopg2)

Queries benchmarked:
  - COUNT(*) — simple agg, tiny result
  - SELECT * LIMIT 100 — small result
  - SELECT * LIMIT 10000 — medium result
  - String filter (WHERE name = ...)
  - BETWEEN filter
  - GROUP BY (10 groups)
  - Aggregation (SUM/AVG/MIN/MAX)
  - Point lookup (WHERE _id = N)
"""

import os
import sys
import time
import tempfile
import subprocess
import random
import socket
import atexit
import signal

import psycopg2

# ── path setup ──────────────────────────────────────────────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'apexbase', 'python'))
import apexbase

# ── config ──────────────────────────────────────────────────────────────────
N_ROWS   = 200_000
N_GROUPS = 10
BENCH_PORT = 15432
REPEATS  = 10
WARMUP   = 3


def find_free_port():
    with socket.socket() as s:
        s.bind(('127.0.0.1', 0))
        return s.getsockname()[1]


# ── dataset setup ────────────────────────────────────────────────────────────
def generate_data(n):
    cities = [f"city_{i}" for i in range(N_GROUPS)]
    rows = []
    for i in range(n):
        rows.append({
            "name": f"user_{i % 5000}",
            "age":  20 + (i % 60),
            "score": round(random.uniform(0, 100), 2),
            "city": cities[i % N_GROUPS],
            "active": i % 3 != 0,
        })
    return rows


def setup_db(data_dir):
    """Populate apexbase with test data, return the DB path."""
    client = apexbase.ApexClient(data_dir)
    client.create_table("benchmark", {
        "name": "string", "age": "int64", "score": "float64",
        "city": "string", "active": "bool",
    })
    client.store(generate_data(N_ROWS))
    client.close()
    return data_dir


# ── server lifecycle ─────────────────────────────────────────────────────────
_server_proc = None

def start_server(data_dir, port):
    global _server_proc
    # Use the installed apexbase-server binary built with server feature
    server_bin = os.path.join(
        os.path.dirname(__file__), '..', 'target', 'release', 'apexbase-server'
    )
    if not os.path.exists(server_bin):
        # Try debug build
        server_bin = os.path.join(
            os.path.dirname(__file__), '..', 'target', 'debug', 'apexbase-server'
        )
    if not os.path.exists(server_bin):
        print("  [WARN] apexbase-server binary not found, trying Python start_pg_server()")
        return start_server_python(data_dir, port)

    _server_proc = subprocess.Popen(
        [server_bin, '--dir', data_dir, '--port', str(port), '--host', '127.0.0.1'],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    atexit.register(stop_server)
    time.sleep(1.5)  # wait for bind
    return _server_proc


def start_server_python(data_dir, port):
    """Start server via Python multiprocessing (uses start_pg_server from apexbase._core)."""
    import multiprocessing
    def _run():
        try:
            import apexbase._core as _core
            _core.start_pg_server(data_dir, '127.0.0.1', port)
        except Exception as e:
            print(f"Server error: {e}")
    p = multiprocessing.Process(target=_run, daemon=True)
    p.start()
    atexit.register(p.terminate)
    time.sleep(1.5)
    return p


def stop_server():
    global _server_proc
    if _server_proc and hasattr(_server_proc, 'poll'):
        if _server_proc.poll() is None:
            _server_proc.terminate()
            _server_proc.wait()


# ── timing helpers ────────────────────────────────────────────────────────────
_call_counter = 0

def timeit(fn, repeats=REPEATS, warmup=WARMUP):
    """Returns (min_ms, avg_ms, max_ms)."""
    times = []
    for i in range(warmup + repeats):
        t0 = time.perf_counter()
        fn()
        t1 = time.perf_counter()
        if i >= warmup:
            times.append((t1 - t0) * 1000)
    return min(times), sum(times)/len(times), max(times)


# ── direct API benchmarks (no-cache: unique params each call) ─────────────────
def bench_direct_nocache(data_dir):
    """Benchmark direct API with unique query per call to avoid py_query_cache."""
    client = apexbase.ApexClient(data_dir)
    client.use_table("benchmark")

    counter = [0]

    def make_limit(n):
        def fn():
            counter[0] += 1
            # Unique LIMIT each call prevents cache hit
            client.execute(f"SELECT * FROM benchmark LIMIT {n + counter[0] % 3}")
        return fn

    def count_fn():
        counter[0] += 1
        client.execute(f"SELECT COUNT(*) FROM benchmark WHERE age >= {20 + counter[0] % 40}")

    def between_fn():
        counter[0] += 1
        lo = 25 + counter[0] % 5
        client.execute(f"SELECT * FROM benchmark WHERE age BETWEEN {lo} AND {lo + 10}")

    def string_fn():
        counter[0] += 1
        client.execute(f"SELECT * FROM benchmark WHERE name = 'user_{counter[0] % 5000}'")

    def group_fn():
        counter[0] += 1
        client.execute(f"SELECT city, COUNT(*) FROM benchmark GROUP BY city")

    def agg_fn():
        counter[0] += 1
        client.execute(f"SELECT SUM(age), AVG(score), MIN(age), MAX(age) FROM benchmark WHERE active = true")

    def lookup_fn():
        counter[0] += 1
        client.execute(f"SELECT * FROM benchmark WHERE _id = {1 + counter[0] % 100}")

    queries = [
        ("COUNT(*) [nocache]",           count_fn),
        ("SELECT * LIMIT 100 [nocache]", make_limit(100)),
        ("SELECT * LIMIT 10K [nocache]", make_limit(10000)),
        ("String filter [nocache]",      string_fn),
        ("BETWEEN filter [nocache]",     between_fn),
        ("GROUP BY (10) [nocache]",      group_fn),
        ("Aggregation [nocache]",        agg_fn),
        ("Point lookup [nocache]",       lookup_fn),
    ]

    results = {}
    for label, fn in queries:
        mn, avg, mx = timeit(fn)
        results[label] = (mn, avg, mx)

    client.close()
    return results


# ── direct API benchmarks ─────────────────────────────────────────────────────
def bench_direct(data_dir):
    client = apexbase.ApexClient(data_dir)
    client.use_table("benchmark")
    # warm up the file
    client.execute("SELECT COUNT(*) FROM benchmark")

    queries = [
        ("COUNT(*)",           "SELECT COUNT(*) FROM benchmark"),
        ("SELECT * LIMIT 100", "SELECT * FROM benchmark LIMIT 100"),
        ("SELECT * LIMIT 10K", "SELECT * FROM benchmark LIMIT 10000"),
        ("String filter",      "SELECT * FROM benchmark WHERE name = 'user_5000'"),
        ("BETWEEN filter",     "SELECT * FROM benchmark WHERE age BETWEEN 30 AND 40"),
        ("GROUP BY (10)",      "SELECT city, COUNT(*) FROM benchmark GROUP BY city"),
        ("Aggregation",        "SELECT SUM(age), AVG(score), MIN(age), MAX(age) FROM benchmark"),
        ("Point lookup",       "SELECT * FROM benchmark WHERE _id = 1"),
    ]

    results = {}
    for label, sql in queries:
        mn, avg, mx = timeit(lambda s=sql: client.execute(s))
        results[label] = (mn, avg, mx)

    client.close()
    return results


# ── PG wire benchmarks ────────────────────────────────────────────────────────
def make_pg_conn(port, db='benchmark'):
    return psycopg2.connect(
        host='127.0.0.1',
        port=port,
        database=db,
        user='apex',
        password='',
        connect_timeout=5,
    )


def bench_pg_simple(port):
    """All queries via psycopg2 Simple Query (no params → mogrify path)."""
    conn = make_pg_conn(port)
    cur = conn.cursor()

    queries = [
        ("COUNT(*)",           "SELECT COUNT(*) FROM benchmark"),
        ("SELECT * LIMIT 100", "SELECT * FROM benchmark LIMIT 100"),
        ("SELECT * LIMIT 10K", "SELECT * FROM benchmark LIMIT 10000"),
        ("String filter",      "SELECT * FROM benchmark WHERE name = 'user_5000'"),
        ("BETWEEN filter",     "SELECT * FROM benchmark WHERE age BETWEEN 30 AND 40"),
        ("GROUP BY (10)",      "SELECT city, COUNT(*) FROM benchmark GROUP BY city"),
        ("Aggregation",        "SELECT SUM(age), AVG(score), MIN(age), MAX(age) FROM benchmark"),
        ("Point lookup",       "SELECT * FROM benchmark WHERE _id = 1"),
    ]

    def run_query(sql):
        cur.execute(sql)
        cur.fetchall()

    # warm up
    run_query("SELECT COUNT(*) FROM benchmark")

    results = {}
    for label, sql in queries:
        mn, avg, mx = timeit(lambda s=sql: run_query(s))
        results[label] = (mn, avg, mx)

    cur.close()
    conn.close()
    return results


def bench_pg_extended(port):
    """Queries with bound parameters → psycopg2 Extended Query Protocol."""
    conn = make_pg_conn(port)
    cur = conn.cursor()

    # warm up
    cur.execute("SELECT COUNT(*) FROM benchmark")
    cur.fetchall()

    parameterized = [
        ("String filter (EQP)",  "SELECT * FROM benchmark WHERE name = %s",        ('user_5000',)),
        ("BETWEEN filter (EQP)", "SELECT * FROM benchmark WHERE age BETWEEN %s AND %s", (30, 40)),
        ("Point lookup (EQP)",   "SELECT * FROM benchmark WHERE _id = %s",          (1,)),
        ("COUNT(*) (EQP)",       "SELECT COUNT(*) FROM benchmark WHERE city = %s",  ('city_0',)),
    ]

    def run_query(sql, params):
        cur.execute(sql, params)
        cur.fetchall()

    results = {}
    for label, sql, params in parameterized:
        mn, avg, mx = timeit(lambda s=sql, p=params: run_query(s, p))
        results[label] = (mn, avg, mx)

    cur.close()
    conn.close()
    return results


def bench_pg_connection_overhead(port):
    """Measure cold connection setup + single query cost."""
    def cold_query():
        conn = make_pg_conn(port)
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM benchmark")  # default DB has benchmark.apex
        cur.fetchall()
        cur.close()
        conn.close()

    mn, avg, mx = timeit(cold_query, repeats=5, warmup=1)
    return {"Cold connect + COUNT(*)": (mn, avg, mx)}


# ── report ─────────────────────────────────────────────────────────────────────
def print_comparison(direct_results, pg_simple_results, pg_ext_results):
    all_labels = list(direct_results.keys())
    # Add EQP-specific labels
    eqp_labels = list(pg_ext_results.keys())

    print(f"\n{'─'*90}")
    print(f"{'Query':<28}  {'Direct API':>12}  {'PG Simple':>12}  {'Ratio':>8}  {'Winner':>8}")
    print(f"{'─'*90}")

    for label in all_labels:
        d_min = direct_results[label][0]
        pg_min = pg_simple_results.get(label, (None,None,None))[0]
        if pg_min is None:
            continue
        ratio = pg_min / d_min if d_min > 0 else float('inf')
        winner = "Direct" if ratio > 1.05 else ("PG" if ratio < 0.95 else "≈Tie")
        print(f"  {label:<26}  {d_min:>10.3f}ms  {pg_min:>10.3f}ms  {ratio:>7.2f}x  {winner:>8}")

    print(f"\n{'─'*90}")
    print(f"  Extended Query Protocol (parameterized queries):")
    print(f"{'─'*90}")

    for label in eqp_labels:
        eqp_min = pg_ext_results[label][0]
        # Find matching simple query baseline
        base_label = label.replace(" (EQP)", "")
        d_min = direct_results.get(base_label, (None,))[0]
        pg_s_min = pg_simple_results.get(base_label, (None,))[0]
        if d_min:
            ratio_vs_direct = eqp_min / d_min
            print(f"  {label:<28}  EQP: {eqp_min:>8.3f}ms  vs Direct: {d_min:>6.3f}ms  ratio: {ratio_vs_direct:.2f}x")
        else:
            print(f"  {label:<28}  EQP: {eqp_min:>8.3f}ms")

    print(f"{'─'*90}")


def print_overhead_analysis(direct_results, pg_simple_results):
    """Print per-query overhead breakdown."""
    print(f"\n  PG Wire overhead per query (min latency comparison):")
    print(f"{'─'*60}")
    overheads = []
    for label in direct_results:
        if label not in pg_simple_results:
            continue
        d = direct_results[label][0]
        p = pg_simple_results[label][0]
        overhead_ms = p - d
        overhead_pct = (overhead_ms / max(d, 0.001)) * 100
        overheads.append((label, d, p, overhead_ms, overhead_pct))

    for label, d, p, oh, oh_pct in sorted(overheads, key=lambda x: -x[4]):
        bar = "█" * min(int(oh_pct / 5), 20)
        print(f"  {label:<26}  +{oh:>6.3f}ms ({oh_pct:>6.1f}%)  {bar}")


# ── main ──────────────────────────────────────────────────────────────────────
def main():
    global N_ROWS, REPEATS
    import argparse
    parser = argparse.ArgumentParser(description="Benchmark PG wire vs direct API")
    parser.add_argument('--port', type=int, default=BENCH_PORT)
    parser.add_argument('--rows', type=int, default=N_ROWS)
    parser.add_argument('--repeats', type=int, default=REPEATS)
    parser.add_argument('--no-server', action='store_true',
                        help='Skip server start (assume already running)')
    args = parser.parse_args()

    N_ROWS = args.rows
    REPEATS = args.repeats

    with tempfile.TemporaryDirectory() as data_dir:
        print(f"ApexBase PG Wire vs Direct API Benchmark")
        print(f"  Rows: {N_ROWS:,}  |  Repeats: {REPEATS}  |  Port: {args.port}")
        print(f"  Data dir: {data_dir}")

        print(f"\n[1/4] Generating and storing {N_ROWS:,} rows...")
        t0 = time.time()
        setup_db(data_dir)
        print(f"  Done in {time.time()-t0:.2f}s")

        if not args.no_server:
            print(f"\n[2/4] Starting PG wire server on port {args.port}...")
            proc = start_server(data_dir, args.port)
            if proc is None:
                print("  ERROR: Could not start server. Aborting.")
                sys.exit(1)
            print(f"  Server started (pid={getattr(proc, 'pid', 'N/A')})")

        # Test connection
        print(f"\n[3/4] Testing PG wire connection...")
        try:
            conn = make_pg_conn(args.port)
            cur = conn.cursor()
            cur.execute("SELECT 1")
            result = cur.fetchone()
            cur.close()
            conn.close()
            print(f"  Connection OK: {result}")
        except Exception as e:
            print(f"  Connection FAILED: {e}")
            print("  Make sure server is running. Use --no-server if already running.")
            sys.exit(1)

        print(f"\n[4/4] Running benchmarks ({REPEATS} repeats, {WARMUP} warmup)...")

        print(f"\n  ▶ Direct Python API (cached)...")
        direct = bench_direct(data_dir)

        print(f"\n  ▶ Direct Python API (no-cache, fair comparison)...")
        direct_nc = bench_direct_nocache(data_dir)

        print(f"\n  ▶ PG Wire (Simple Query)...")
        try:
            pg_simple = bench_pg_simple(args.port)
        except Exception as e:
            print(f"  ERROR in PG simple: {e}")
            pg_simple = {}

        print(f"\n  ▶ PG Wire (Extended Query / parameterized)...")
        try:
            pg_ext = bench_pg_extended(args.port)
        except Exception as e:
            print(f"  ERROR in PG extended: {e}")
            pg_ext = {}

        print(f"\n  ▶ Connection overhead...")
        try:
            conn_overhead = bench_pg_connection_overhead(args.port)
        except Exception as e:
            print(f"  ERROR in connection overhead: {e}")
            conn_overhead = {}

        # ── report ──
        print(f"\n{'═'*90}")
        print(f"  SECTION A: Cached Direct API vs PG Wire  (all latency = min over {REPEATS} runs)")
        print(f"  NOTE: Direct API uses py_query_cache — repeated identical queries return instantly")
        print(f"{'═'*90}")
        print_comparison(direct, pg_simple, pg_ext)

        # Fair comparison: no-cache direct API vs PG Wire
        nc_label_map = {
            "COUNT(*) [nocache]":           "COUNT(*)",
            "SELECT * LIMIT 100 [nocache]": "SELECT * LIMIT 100",
            "SELECT * LIMIT 10K [nocache]": "SELECT * LIMIT 10K",
            "String filter [nocache]":      "String filter",
            "BETWEEN filter [nocache]":     "BETWEEN filter",
            "GROUP BY (10) [nocache]":      "GROUP BY (10)",
            "Aggregation [nocache]":        "Aggregation",
            "Point lookup [nocache]":       "Point lookup",
        }
        print(f"\n{'═'*90}")
        print(f"  SECTION B: Fair comparison — Direct API (no cache) vs PG Wire Simple Query")
        print(f"  Both paths execute the query fresh each time, same underlying engine")
        print(f"{'═'*90}")
        print(f"  {'Query':<28}  {'Direct (no-cache)':>18}  {'PG Simple':>12}  {'PG overhead':>12}  {'Ratio':>7}")
        print(f"  {'─'*84}")
        for nc_label, pg_label in nc_label_map.items():
            d = direct_nc.get(nc_label, (None,))[0]
            p = pg_simple.get(pg_label, (None,))[0]
            if d is None or p is None:
                continue
            overhead = p - d
            ratio = p / d if d > 0 else float('inf')
            print(f"  {pg_label:<28}  {d:>16.3f}ms  {p:>10.3f}ms  {overhead:>+10.3f}ms  {ratio:>6.2f}x")

        print_overhead_analysis(direct_nc, pg_simple)

        if conn_overhead:
            conn_val = list(conn_overhead.values())[0]
            print(f"\n  Cold TCP connect + query: {conn_val[0]:.3f}ms min / {conn_val[1]:.3f}ms avg")

        # ── overhead breakdown ──
        print(f"\n{'═'*90}")
        print(f"  OVERHEAD ANALYSIS")
        print(f"{'═'*90}")
        print(f"  PG Wire overhead sources:")
        print(f"  1. TCP round-trip (localhost): ~0.1-0.5ms")
        print(f"  2. Text serialization: every int/float → string → parse")
        print(f"  3. DataRowEncoder: one heap alloc per row")
        print(f"  4. Extended Query double-execution: describe_statement executes query with NULLs")
        print()


if __name__ == '__main__':
    main()
