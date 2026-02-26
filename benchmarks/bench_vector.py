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
import os
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
    p.add_argument("--no-f16", action="store_true", dest="no_f16",
                   help="Skip Section 5 (f32 vs f16 comparison)")
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


def setup_apex_f16(vecs: np.ndarray, tmp_dir: str):
    """Set up ApexBase with FLOAT16_VECTOR column (2 bytes/elem vs f32's 4 bytes)."""
    from apexbase.client import ApexClient

    client = ApexClient(dirpath=tmp_dir, drop_if_exists=True)
    client.execute("CREATE TABLE vecs (id INT, vec FLOAT16_VECTOR)")
    client.use_table("vecs")

    batch_size = 10_000
    n = len(vecs)
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        rows = [{"id": i, "vec": vecs[i]} for i in range(start, end)]
        client.store(rows)
    return client


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


def bench_apex_batch(client, queries: np.ndarray, k: int, metric: str, iters: int):
    """Benchmark ApexBase batch_topk_distance — all N queries in a single Rust call."""
    _metric_map = {
        "l2":     "l2",
        "cosine": "cosine_distance",
        "dot":    "dot",
    }
    apex_metric = _metric_map[metric]
    # warmup
    client.batch_topk_distance("vec", queries, k, apex_metric)

    times = []
    for _ in range(max(1, iters)):
        gc.collect()
        t0 = time.perf_counter()
        client.batch_topk_distance("vec", queries, k, apex_metric)
        times.append((time.perf_counter() - t0) * 1000)
    return times


def bench_apex_sequential(client, queries: np.ndarray, k: int, metric: str, iters: int):
    """ApexBase: N sequential _topk_distance_ffi calls — baseline for batch comparison."""
    _metric_map = {
        "l2":     "l2",
        "cosine": "cosine_distance",
        "dot":    "dot",
    }
    apex_metric = _metric_map[metric]

    times = []
    for _ in range(max(1, iters)):
        gc.collect()
        t0 = time.perf_counter()
        for qvec in queries:
            client._storage._topk_distance_ffi("vec", qvec.tobytes(), k, apex_metric)
        times.append((time.perf_counter() - t0) * 1000)
    return times

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

    with tempfile.TemporaryDirectory() as apex_dir, \
         tempfile.TemporaryDirectory() as apex_f16_dir:
        print("Setting up ApexBase (f32)...", end="", flush=True)
        client = setup_apex(vecs, apex_dir)
        print(" done.")

        if not args.no_f16:
            print("Setting up ApexBase (f16)...", end="", flush=True)
            client_f16 = setup_apex_f16(vecs, apex_f16_dir)
            print(" done.")
        else:
            client_f16 = None

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

        # ── Section 3: Batch TopK — new batch_topk_distance API ──────────────
        print()
        nq = len(batch_queries)
        print(f"── Section 3: Batch TopK ({nq} queries) — batch API vs N×single ──")
        batch_header = (
            f"{'Metric':<18} {'Apex-batch(1call)':>18} {'Apex-seq(N calls)':>18}"
            f" {'DuckDB-seq(N calls)':>20} {'speedup(seq/batch)':>20}"
        )
        print(batch_header)
        print("-" * len(batch_header))

        for metric in ["l2", "cosine", "dot"]:
            # ApexBase batch: single batch_topk_distance call
            apex_batch_times = bench_apex_batch(client, batch_queries, args.k, metric, args.iters)
            apex_batch_med = median(apex_batch_times)

            # ApexBase sequential: N individual _topk_distance_ffi calls
            apex_seq_times = bench_apex_sequential(client, batch_queries, args.k, metric, args.iters)
            apex_seq_med = median(apex_seq_times)

            speedup = apex_seq_med / apex_batch_med
            speedup_str = f"{speedup:.2f}x {'✅' if speedup > 1.0 else '❌'}"

            if duck_con is not None:
                duck_ms = bench_duckdb_batch_sequential(duck_con, batch_queries, args.k, metric)
                duck_str = fmt(duck_ms)
            else:
                duck_str = "N/A"

            print(
                f"{metric:<18} {fmt(apex_batch_med):>18} {fmt(apex_seq_med):>18}"
                f" {duck_str:>20} {speedup_str:>20}"
            )

        # ── Section 4: ApexBase detailed stats ────────────────────────────────
        print()
        print("── Section 4: ApexBase min/median/max (all metrics) ──")
        for metric in shared_metrics + ["l2_squared", "l1", "linf"]:
            t = bench_apex(client, query, args.k, metric, 1, args.iters)
            print(f"  {metric:<12}: min={fmt(min(t))}  med={fmt(median(t))}  max={fmt(max(t))}")

        # ── Section 5: Float32 vs Float16 — storage size & query speed ────────
        if client_f16 is not None:
            print()
            print("── Section 5: Float32 vs Float16 — storage size & query latency ──")

            # Storage size comparison
            def apex_file_size(dirpath: str) -> int:
                return sum(
                    os.path.getsize(os.path.join(dirpath, f))
                    for f in os.listdir(dirpath)
                    if os.path.isfile(os.path.join(dirpath, f))
                        and f.endswith(".apex")
                )

            f32_bytes = apex_file_size(apex_dir)
            f16_bytes = apex_file_size(apex_f16_dir)
            savings_pct = (1 - f16_bytes / f32_bytes) * 100 if f32_bytes > 0 else 0
            print(f"  File size  f32: {f32_bytes / 1024 / 1024:.2f} MB")
            print(f"  File size  f16: {f16_bytes / 1024 / 1024:.2f} MB  "
                  f"({savings_pct:.1f}% smaller)")
            print()

            # Query speed comparison
            f16_metrics = ["l2", "cosine", "l1"]
            sec5_header = (
                f"{'Metric':<12} {'f32 (FixedList)':>16} {'f16 (Float16List)':>18}"
                f" {'f16/f32 ratio':>14}"
            )
            print(sec5_header)
            print("-" * len(sec5_header))

            for metric in f16_metrics:
                f32_times = bench_apex(client,     query, args.k, metric, args.warmup, args.iters)
                f16_times = bench_apex(client_f16, query, args.k, metric, args.warmup, args.iters)
                f32_med = median(f32_times)
                f16_med = median(f16_times)
                ratio = f16_med / f32_med
                label = "✅ faster" if ratio < 1.0 else "  slower"
                print(
                    f"{metric:<12} {fmt(f32_med):>16} {fmt(f16_med):>18}"
                    f" {ratio:>8.2f}x {label}"
                )

            client_f16.close()

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
