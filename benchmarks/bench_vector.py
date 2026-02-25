#!/usr/bin/env python3
"""
Vector distance TopK benchmark: ApexBase vs DuckDB.

Usage:
    python benchmarks/bench_vector.py [--rows N] [--dim D] [--k K] [--warmup W] [--iters I]

Requires: duckdb  (pip install duckdb)

DuckDB fetch strategy:
    fetch_arrow_table() — zero-copy Arrow IPC, fastest for columnar results.
    fetchall() is 2-5x slower for large k due to Python tuple boxing.
    DuckDB has no native L1 / L∞ distance functions, so those metrics are
    benchmarked for ApexBase only.
"""

import argparse
import gc
import tempfile
import time

import numpy as np

# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--rows",   type=int, default=1_000_000)
    p.add_argument("--dim",    type=int, default=128)
    p.add_argument("--k",      type=int, default=10)
    p.add_argument("--warmup", type=int, default=0)
    p.add_argument("--iters",  type=int, default=1)
    return p.parse_args()


def fmt(ms: float) -> str:
    return f"{ms:.3f} ms"


def median(vals):
    s = sorted(vals)
    n = len(s)
    return s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2


# ─────────────────────────────────────────────────────────────────────────────
# Data generation
# ─────────────────────────────────────────────────────────────────────────────

def generate_data(n: int, dim: int, seed: int = 42):
    rng = np.random.default_rng(seed)
    vecs = rng.random((n, dim), dtype=np.float32)
    query = rng.random(dim, dtype=np.float32)
    return vecs, query


# ─────────────────────────────────────────────────────────────────────────────
# ApexBase setup
# ─────────────────────────────────────────────────────────────────────────────

def setup_apex(vecs: np.ndarray, tmp_dir: str):
    from apexbase.client import ApexClient

    client = ApexClient(dirpath=tmp_dir, drop_if_exists=True)
    client.create_table("vecs")

    batch_size = 10_000
    n = len(vecs)
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        rows = [{"id": i, "vec": vecs[i]} for i in range(start, end)]
        client.store(rows)
    return client


def bench_apex(client, query: np.ndarray, k: int, metric: str, warmup: int, iters: int):
    """Benchmark ApexBase using topk_distance (O(n log k) fused heap — optimal path)."""
    # topk_distance metric names
    _metric_map = {
        "l2":         "l2",
        "cosine":     "cosine_distance",
        "dot":        "dot",
        "l2_squared": "l2_squared",
        "l1":         "l1",
        "linf":       "linf",
    }
    apex_metric = _metric_map[metric]
    q_str = ",".join(f"{v:.6f}" for v in query)
    sql = (
        f"SELECT explode_rename(topk_distance(vec, [{q_str}], {k}, '{apex_metric}'), '_id', 'dist') "
        f"FROM vecs"
    )

    # Warmup: populate OS page cache (not timed)
    for _ in range(max(1, warmup)):
        client.execute(sql).to_arrow()

    times = []
    for _ in range(iters):
        gc.collect()
        t0 = time.perf_counter()
        client.execute(sql).to_arrow()
        times.append((time.perf_counter() - t0) * 1000)
    return times


# ─────────────────────────────────────────────────────────────────────────────
# DuckDB setup
# ─────────────────────────────────────────────────────────────────────────────

def setup_duckdb(vecs: np.ndarray):
    try:
        import duckdb
        import pandas as pd
    except ImportError:
        return None

    con = duckdb.connect(":memory:")
    dim = vecs.shape[1]
    n = len(vecs)

    # Bulk-load via pandas DataFrame registration (much faster than executemany)
    df = pd.DataFrame({"id": np.arange(n, dtype=np.int32), "vec": list(vecs)})
    con.register("_vecs_src", df)
    con.execute(f"CREATE TABLE vecs AS SELECT id, vec::FLOAT[{dim}] AS vec FROM _vecs_src")
    con.execute("DROP VIEW IF EXISTS _vecs_src")

    return con


def bench_duckdb(con, query: np.ndarray, k: int, metric: str, warmup: int, iters: int):
    """Benchmark DuckDB using fetch_arrow_table() — zero-copy, fastest path.

    Shared metrics only (DuckDB has no native L1/L∞ functions):
      l2    → array_distance
      cosine → array_cosine_distance
      dot   → array_negative_inner_product
    """
    q_list = query.tolist()
    q_str = ",".join(f"{v:.6f}" for v in q_list)
    q_cast = f"[{q_str}]::FLOAT[{len(query)}]"

    if metric == "l2":
        sql = f"SELECT id, array_distance(vec, {q_cast}) AS dist FROM vecs ORDER BY dist LIMIT {k}"
    elif metric == "cosine":
        sql = f"SELECT id, array_cosine_distance(vec, {q_cast}) AS dist FROM vecs ORDER BY dist LIMIT {k}"
    elif metric == "dot":
        sql = f"SELECT id, array_negative_inner_product(vec, {q_cast}) AS dist FROM vecs ORDER BY dist LIMIT {k}"
    else:
        raise ValueError(metric)

    # Warmup with fetch_arrow_table (same code path as timed runs)
    for _ in range(warmup):
        con.execute(sql).fetch_arrow_table()

    times = []
    for _ in range(iters):
        gc.collect()
        t0 = time.perf_counter()
        con.execute(sql).fetch_arrow_table()  # zero-copy Arrow — fastest DuckDB fetch
        times.append((time.perf_counter() - t0) * 1000)
    return times


def bench_duckdb_batch_sequential(con, queries: np.ndarray, k: int, metric: str):
    """DuckDB batch: N sequential queries — fair comparison with ApexBase.

    Returns total elapsed ms for all queries.
    """
    dim = queries.shape[1]
    q_cast = f"FLOAT[{dim}]"

    if metric == "l2":
        sql_tmpl = "SELECT id, array_distance(vec, {q}::{cast}) AS dist FROM vecs ORDER BY dist LIMIT {k}"
    elif metric == "cosine":
        sql_tmpl = "SELECT id, array_cosine_distance(vec, {q}::{cast}) AS dist FROM vecs ORDER BY dist LIMIT {k}"
    elif metric == "dot":
        sql_tmpl = "SELECT id, array_negative_inner_product(vec, {q}::{cast}) AS dist FROM vecs ORDER BY dist LIMIT {k}"
    else:
        raise ValueError(metric)

    gc.collect()
    t0 = time.perf_counter()
    for qvec in queries:
        q_str = ",".join(f"{v:.6f}" for v in qvec)
        sql = sql_tmpl.format(q=f"[{q_str}]", cast=q_cast, k=k)
        con.execute(sql).fetch_arrow_table()
    return (time.perf_counter() - t0) * 1000

# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    print(f"\n{'='*70}")
    print(f"Vector TopK Benchmark")
    print(f"  rows={args.rows:,}  dim={args.dim}  k={args.k}  warmup={args.warmup}  iters={args.iters}")
    print(f"  DuckDB fetch: fetch_arrow_table() [zero-copy Arrow]")
    print(f"{'='*70}\n")

    rng = np.random.default_rng(42)
    vecs, query = generate_data(args.rows, args.dim)
    # Multiple query vectors for batch section
    batch_queries = rng.random((10, args.dim), dtype=np.float32)

    with tempfile.TemporaryDirectory() as apex_dir:
        print("Setting up ApexBase...", end="", flush=True)
        client = setup_apex(vecs, apex_dir)
        print(" done.")

        print("Setting up DuckDB...", end="", flush=True)
        duck_con = setup_duckdb(vecs)
        print(f" {'done' if duck_con else 'SKIPPED (pip install duckdb)'}.")

        # ── Section 1: Shared metrics (ApexBase vs DuckDB) ────────────────────
        print()
        print("── Section 1: Head-to-head (shared metrics) ──")
        header = f"{'Metric':<18} {'ApexBase':>12} {'DuckDB':>12} {'Ratio (Apex/Duck)':>18}"
        print(header)
        print("-" * len(header))

        shared_metrics = ["l2", "cosine", "dot"]
        for metric in shared_metrics:
            apex_times = bench_apex(client, query, args.k, metric, args.warmup, args.iters)
            apex_med = median(apex_times)

            if duck_con is not None:
                duck_times = bench_duckdb(duck_con, query, args.k, metric, args.warmup, args.iters)
                duck_med = median(duck_times)
                ratio = apex_med / duck_med
                faster = "✅" if ratio < 1.0 else "❌"
                duck_str = fmt(duck_med)
                ratio_str = f"{ratio:.2f}x {faster}"
            else:
                duck_str = "N/A"
                ratio_str = "N/A"

            print(f"{metric:<18} {fmt(apex_med):>12} {duck_str:>12} {ratio_str:>18}")

        # ── Section 2: ApexBase-only metrics (no DuckDB native equivalent) ────
        print()
        print("── Section 2: ApexBase-only metrics (DuckDB has no native equiv.) ──")
        apex_only_header = f"{'Metric':<18} {'ApexBase':>12}"
        print(apex_only_header)
        print("-" * len(apex_only_header))

        for metric in ["l2_squared", "l1", "linf"]:
            apex_times = bench_apex(client, query, args.k, metric, args.warmup, args.iters)
            apex_med = median(apex_times)
            print(f"{metric:<18} {fmt(apex_med):>12}")

        # ── Section 3: Batch query (10 query vectors, single SQL round-trip) ──
        print()
        print("── Section 3: Batch TopK (10 query vectors, single round-trip) ──")
        batch_header = f"{'Metric':<18} {'ApexBase×10':>14} {'DuckDB×10':>14} {'per-query Apex':>16} {'per-query Duck':>16}"
        print(batch_header)
        print("-" * len(batch_header))

        nq = len(batch_queries)
        for metric in ["l2", "cosine", "dot"]:
            # ApexBase: N sequential topk_distance calls
            apex_batch_times = []
            for _ in range(max(1, args.iters)):
                gc.collect()
                t0 = time.perf_counter()
                for qvec in batch_queries:
                    _run_single_apex(client, qvec, args.k, metric)
                apex_batch_times.append((time.perf_counter() - t0) * 1000)
            apex_batch_med = median(apex_batch_times)

            if duck_con is not None:
                # DuckDB: N sequential queries — FAIR apples-to-apples comparison
                duck_batch_ms = bench_duckdb_batch_sequential(duck_con, batch_queries, args.k, metric)
                duck_str = fmt(duck_batch_ms)
                ratio = apex_batch_med / duck_batch_ms
                faster = "✅" if ratio < 1.0 else "❌"
                per_duck = fmt(duck_batch_ms / nq)
            else:
                duck_str = "N/A"
                per_duck = "N/A"
                faster = ""

            per_apex = fmt(apex_batch_med / nq)
            print(f"{metric:<18} {fmt(apex_batch_med):>14} {duck_str:>14} {per_apex:>16} {per_duck:>16}")

        # ── Section 4: ApexBase detailed stats ────────────────────────────────
        print()
        print("── Section 4: ApexBase min/median/max (all metrics) ──")
        for metric in shared_metrics + ["l2_squared", "l1", "linf"]:
            t = bench_apex(client, query, args.k, metric, 1, args.iters)
            print(f"  {metric:<12}: min={fmt(min(t))}  med={fmt(median(t))}  max={fmt(max(t))}")

        client.close()


def _run_single_apex(client, query: np.ndarray, k: int, metric: str):
    """Single TopK call via the expression-based topk_distance API."""
    _metric_map = {"l2": "l2", "cosine": "cosine_distance", "dot": "dot"}
    q_str = ",".join(f"{v:.6f}" for v in query)
    sql = (
        f"SELECT explode_rename(topk_distance(vec, [{q_str}], {k}, '{_metric_map[metric]}'), '_id', 'dist') "
        f"FROM vecs"
    )
    client.execute(sql).to_arrow()


if __name__ == "__main__":
    main()
