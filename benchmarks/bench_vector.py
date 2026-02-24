#!/usr/bin/env python3
"""
Vector distance TopK benchmark: ApexBase vs DuckDB.

Usage:
    python benchmarks/bench_vector.py [--rows N] [--dim D] [--k K] [--warmup W] [--iters I]

Requires: duckdb  (pip install duckdb)
"""

import argparse
import gc
import math
import os
import struct
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
    p.add_argument("--iters",  type=int, default=10)
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
    from apexbase.client import ApexClient, encode_vector

    client = ApexClient(dirpath=tmp_dir, drop_if_exists=True)
    client.create_table("vecs")

    batch_size = 10_000
    n = len(vecs)
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        rows = [{"id": i, "vec": encode_vector(vecs[i])} for i in range(start, end)]
        client.store(rows)
    return client


def bench_apex(client, query: np.ndarray, k: int, metric: str, warmup: int, iters: int):
    from apexbase.client import encode_vector
    q_str = ",".join(f"{v:.6f}" for v in query)

    if metric == "l2":
        sql = f"SELECT id, array_distance(vec, [{q_str}]) AS dist FROM vecs ORDER BY dist LIMIT {k}"
    elif metric == "cosine":
        sql = f"SELECT id, cosine_distance(vec, [{q_str}]) AS dist FROM vecs ORDER BY dist LIMIT {k}"
    elif metric == "dot":
        sql = f"SELECT id, negative_inner_product(vec, [{q_str}]) AS dist FROM vecs ORDER BY dist LIMIT {k}"
    else:
        raise ValueError(metric)

    for _ in range(warmup):
        client.execute(sql).to_pandas()

    times = []
    for _ in range(iters):
        gc.collect()
        t0 = time.perf_counter()
        client.execute(sql).to_pandas()
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

    for _ in range(warmup):
        con.execute(sql).fetchall()

    times = []
    for _ in range(iters):
        gc.collect()
        t0 = time.perf_counter()
        con.execute(sql).fetchall()
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
    print(f"{'='*70}\n")

    vecs, query = generate_data(args.rows, args.dim)

    with tempfile.TemporaryDirectory() as apex_dir:
        print("Setting up ApexBase...", end="", flush=True)
        client = setup_apex(vecs, apex_dir)
        print(" done.")

        print("Setting up DuckDB...", end="", flush=True)
        duck_con = setup_duckdb(vecs)
        print(f" {'done' if duck_con else 'SKIPPED (pip install duckdb)'}.")

        print()
        header = f"{'Metric':<18} {'ApexBase':>12} {'DuckDB':>12} {'Ratio (Apex/Duck)':>18}"
        print(header)
        print("-" * len(header))

        for metric in ["l2", "cosine", "dot"]:
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

        print()
        print("ApexBase min/median/max per metric:")
        for metric in ["l2", "cosine", "dot"]:
            t = bench_apex(client, query, args.k, metric, 1, args.iters)
            print(f"  {metric}: min={fmt(min(t))}  med={fmt(median(t))}  max={fmt(max(t))}")

        client.close()


if __name__ == "__main__":
    main()
