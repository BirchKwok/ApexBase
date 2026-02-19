#!/usr/bin/env python3
"""
Benchmark: Arrow Flight vs PG Wire Protocol vs Direct Python API

Arrow Flight sends Arrow IPC RecordBatch directly over gRPC (HTTP/2) —
zero serialization cost since ApexBase already stores data as Arrow internally.

Usage:
    python benchmarks/bench_flight.py --rows 200000 --repeats 10
"""

import sys
import os
import time
import tempfile
import subprocess
import argparse
import random

# ── config ────────────────────────────────────────────────────────────────────
FLIGHT_PORT = 50051
PG_PORT     = 15435
N_ROWS      = 200_000
N_GROUPS    = 10
REPEATS     = 10
WARMUP      = 3

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FLIGHT_BIN = os.path.join(REPO_ROOT, "target", "release", "apexbase-flight")
PG_BIN     = os.path.join(REPO_ROOT, "target", "release", "apexbase-server")

sys.path.insert(0, os.path.join(REPO_ROOT, "apexbase", "python"))
import apexbase
import pyarrow.flight as fl
import psycopg2

# ── data generation ───────────────────────────────────────────────────────────
def generate_data(n):
    names  = [f"user_{i % 5000}" for i in range(n)]
    ages   = [20 + (i % 60) for i in range(n)]
    scores = [round(50.0 + (i % 50) * 0.5, 2) for i in range(n)]
    cities = [f"city_{i % N_GROUPS}" for i in range(n)]
    active = [bool(i % 2) for i in range(n)]
    return {"name": names, "age": ages, "score": scores, "city": cities, "active": active}


def setup_db(data_dir):
    client = apexbase.ApexClient(data_dir)
    client.create_table("benchmark", {
        "name": "string", "age": "int64", "score": "float64",
        "city": "string", "active": "bool",
    })
    client.store(generate_data(N_ROWS))
    client.close()


# ── server lifecycle ──────────────────────────────────────────────────────────
_procs = {}

def start_server(binary, data_dir, port, name):
    if not os.path.exists(binary):
        print(f"  ERROR: {name} binary not found at {binary}")
        print(f"  Build with: cargo build --release --bin {os.path.basename(binary)} --no-default-features --features {name.lower()}")
        return None
    proc = subprocess.Popen(
        [binary, "--dir", data_dir, "--port", str(port)],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    _procs[name] = proc
    time.sleep(1.0)
    if proc.poll() is not None:
        print(f"  ERROR: {name} server exited immediately")
        return None
    return proc


def stop_all():
    for name, proc in _procs.items():
        if proc and proc.poll() is None:
            proc.terminate()
            proc.wait()


# ── timing helper ──────────────────────────────────────────────────────────────
def timeit(fn, repeats=REPEATS, warmup=WARMUP):
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        times.append((time.perf_counter() - t0) * 1000)
    return min(times), sum(times) / len(times), max(times)


# ── Arrow Flight benchmarks ───────────────────────────────────────────────────
def bench_flight(port):
    client = fl.connect(f"grpc://127.0.0.1:{port}")

    queries = [
        ("COUNT(*)",           "SELECT COUNT(*) FROM benchmark"),
        ("SELECT * LIMIT 100", "SELECT * FROM benchmark LIMIT 100"),
        ("SELECT * LIMIT 10K", "SELECT * FROM benchmark LIMIT 10000"),
        ("String filter",      "SELECT * FROM benchmark WHERE name = 'user_42'"),
        ("BETWEEN filter",     "SELECT * FROM benchmark WHERE age BETWEEN 25 AND 35"),
        ("GROUP BY (10)",      "SELECT city, COUNT(*) FROM benchmark GROUP BY city"),
        ("Aggregation",        "SELECT SUM(age), AVG(score), MIN(age), MAX(age) FROM benchmark"),
        ("Point lookup",       "SELECT * FROM benchmark WHERE _id = 1"),
    ]

    results = {}
    for label, sql in queries:
        def fn(s=sql):
            reader = client.do_get(fl.Ticket(s.encode()))
            return reader.read_all()

        mn, avg, mx = timeit(fn)
        results[label] = (mn, avg, mx)

    return results


# ── PG Wire benchmarks ────────────────────────────────────────────────────────
def make_pg_conn(port):
    for attempt in range(10):
        try:
            return psycopg2.connect(
                host="127.0.0.1", port=port, dbname="postgres",
                user="postgres", password="postgres",
                connect_timeout=3,
            )
        except Exception:
            time.sleep(0.5)
    raise RuntimeError(f"Cannot connect to PG server on port {port}")


def bench_pg(port):
    conn = make_pg_conn(port)
    conn.autocommit = True
    cur = conn.cursor()

    queries = [
        ("COUNT(*)",           "SELECT COUNT(*) FROM benchmark"),
        ("SELECT * LIMIT 100", "SELECT * FROM benchmark LIMIT 100"),
        ("SELECT * LIMIT 10K", "SELECT * FROM benchmark LIMIT 10000"),
        ("String filter",      "SELECT * FROM benchmark WHERE name = 'user_42'"),
        ("BETWEEN filter",     "SELECT * FROM benchmark WHERE age BETWEEN 25 AND 35"),
        ("GROUP BY (10)",      "SELECT city, COUNT(*) FROM benchmark GROUP BY city"),
        ("Aggregation",        "SELECT SUM(age), AVG(score), MIN(age), MAX(age) FROM benchmark"),
        ("Point lookup",       "SELECT * FROM benchmark WHERE _id = 1"),
    ]

    results = {}
    for label, sql in queries:
        def fn(s=sql):
            cur.execute(s)
            cur.fetchall()

        mn, avg, mx = timeit(fn)
        results[label] = (mn, avg, mx)

    cur.close()
    conn.close()
    return results


# ── Direct API benchmark (no-cache) ──────────────────────────────────────────
def bench_direct(data_dir):
    client = apexbase.ApexClient(data_dir)
    client.use_table("benchmark")

    counter = [0]

    def count_fn():
        counter[0] += 1
        client.execute(f"SELECT COUNT(*) FROM benchmark WHERE age >= {20 + counter[0] % 40}")

    def limit100_fn():
        counter[0] += 1
        client.execute(f"SELECT * FROM benchmark LIMIT {100 + counter[0] % 3}")

    def limit10k_fn():
        counter[0] += 1
        client.execute(f"SELECT * FROM benchmark LIMIT {10000 + counter[0] % 3}")

    def string_fn():
        counter[0] += 1
        client.execute(f"SELECT * FROM benchmark WHERE name = 'user_{counter[0] % 5000}'")

    def between_fn():
        counter[0] += 1
        lo = 25 + counter[0] % 5
        client.execute(f"SELECT * FROM benchmark WHERE age BETWEEN {lo} AND {lo + 10}")

    def group_fn():
        counter[0] += 1
        client.execute("SELECT city, COUNT(*) FROM benchmark GROUP BY city")

    def agg_fn():
        counter[0] += 1
        client.execute(f"SELECT SUM(age), AVG(score), MIN(age), MAX(age) FROM benchmark WHERE active = true")

    def lookup_fn():
        counter[0] += 1
        client.execute(f"SELECT * FROM benchmark WHERE _id = {1 + counter[0] % 100}")

    queries = [
        ("COUNT(*)",           count_fn),
        ("SELECT * LIMIT 100", limit100_fn),
        ("SELECT * LIMIT 10K", limit10k_fn),
        ("String filter",      string_fn),
        ("BETWEEN filter",     between_fn),
        ("GROUP BY (10)",      group_fn),
        ("Aggregation",        agg_fn),
        ("Point lookup",       lookup_fn),
    ]

    results = {}
    for label, fn in queries:
        mn, avg, mx = timeit(fn)
        results[label] = (mn, avg, mx)

    client.close()
    return results


# ── report ─────────────────────────────────────────────────────────────────────
def print_report(direct, flight, pg):
    W = 100
    print(f"\n{'═'*W}")
    print(f"  Arrow Flight vs PG Wire vs Direct API  (min latency / {REPEATS} runs, ms)")
    print(f"{'═'*W}")
    hdr = f"  {'Query':<26}  {'Direct':>10}  {'Flight':>10}  {'PG Wire':>10}  {'Flight/Direct':>14}  {'PG/Direct':>10}  {'PG/Flight':>10}"
    print(hdr)
    print(f"  {'─'*96}")

    queries = [
        "COUNT(*)", "SELECT * LIMIT 100", "SELECT * LIMIT 10K",
        "String filter", "BETWEEN filter", "GROUP BY (10)", "Aggregation", "Point lookup",
    ]

    for q in queries:
        d = direct.get(q, (None,))[0]
        f = flight.get(q, (None,))[0]
        p = pg.get(q, (None,))[0]

        d_s  = f"{d:.3f}" if d is not None else "  N/A "
        f_s  = f"{f:.3f}" if f is not None else "  N/A "
        p_s  = f"{p:.3f}" if p is not None else "  N/A "
        fd_s = f"{f/d:.2f}x" if (f and d) else "  N/A "
        pd_s = f"{p/d:.2f}x" if (p and d) else "  N/A "
        pf_s = f"{p/f:.2f}x" if (p and f) else "  N/A "

        print(f"  {q:<26}  {d_s:>10}  {f_s:>10}  {p_s:>10}  {fd_s:>14}  {pd_s:>10}  {pf_s:>10}")

    print(f"\n  Notes:")
    print(f"  - Direct API: no-cache (unique params per call), in-process, no network")
    print(f"  - Arrow Flight: Arrow IPC over gRPC/HTTP2, binary columnar, zero serialization")
    print(f"  - PG Wire: text DataRow messages over TCP, per-row per-field to_string()")
    print(f"  - Flight/Direct ratio: pure protocol overhead (Arrow IPC framing + gRPC)")
    print(f"  - PG/Flight ratio: serialization cost of PG text vs Arrow binary format")


# ── main ─────────────────────────────────────────────────────────────────────
def main():
    global N_ROWS, REPEATS

    parser = argparse.ArgumentParser(description="Benchmark Arrow Flight vs PG Wire")
    parser.add_argument("--rows",    type=int, default=N_ROWS)
    parser.add_argument("--repeats", type=int, default=REPEATS)
    parser.add_argument("--flight-port", type=int, default=FLIGHT_PORT)
    parser.add_argument("--pg-port",     type=int, default=PG_PORT)
    parser.add_argument("--no-server",   action="store_true")
    args = parser.parse_args()

    N_ROWS  = args.rows
    REPEATS = args.repeats

    try:
        with tempfile.TemporaryDirectory() as data_dir:
            print(f"Arrow Flight vs PG Wire vs Direct API Benchmark")
            print(f"  Rows: {N_ROWS:,}  |  Repeats: {REPEATS}  |  Data: {data_dir}")

            print(f"\n[1/4] Storing {N_ROWS:,} rows...")
            t0 = time.time()
            setup_db(data_dir)
            print(f"  Done in {time.time()-t0:.2f}s")

            if not args.no_server:
                print(f"\n[2/4] Starting servers...")
                fp = start_server(FLIGHT_BIN, data_dir, args.flight_port, "flight")
                pp = start_server(PG_BIN,     data_dir, args.pg_port,     "server")
                if fp is None or pp is None:
                    print("  One or more servers failed to start. Aborting.")
                    stop_all()
                    sys.exit(1)
                print(f"  Flight pid={fp.pid}, PG pid={pp.pid}")

            print(f"\n[3/4] Verifying connections...")
            try:
                fc = fl.connect(f"grpc://127.0.0.1:{args.flight_port}")
                fc.do_get(fl.Ticket(b"SELECT 1")).read_all()
                print(f"  Flight OK")
            except Exception as e:
                print(f"  Flight FAILED: {e}")
                stop_all(); sys.exit(1)

            try:
                conn = make_pg_conn(args.pg_port)
                conn.close()
                print(f"  PG Wire OK")
            except Exception as e:
                print(f"  PG Wire FAILED: {e}")
                stop_all(); sys.exit(1)

            print(f"\n[4/4] Benchmarking ({REPEATS} repeats, {WARMUP} warmup)...")
            print("  ▶ Direct API (no-cache)...")
            direct = bench_direct(data_dir)
            print("  ▶ Arrow Flight...")
            flight = bench_flight(args.flight_port)
            print("  ▶ PG Wire...")
            pg = bench_pg(args.pg_port)

            print_report(direct, flight, pg)

    finally:
        stop_all()


if __name__ == "__main__":
    main()
