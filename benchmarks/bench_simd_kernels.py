#!/usr/bin/env python3
"""
Micro-benchmark: isolate pure SIMD kernel cost from I/O and heap overhead.

Sections
--------
A  Pure distance kernel  — compute metric for 1M vectors, discard results (no TopK heap)
B  TopK heap overhead    — use pre-computed random distances, just run heap
C  End-to-end TopK       — full topk_distance SQL (as in bench_numpack_vs_apexbase.py)
D  Before/After compare  — run after SIMD optimisation to show speedup

Usage
-----
    python benchmarks/bench_simd_kernels.py [--rows N] [--dim D] [--k K] [--iters I]
"""

import argparse, gc, os, sys, tempfile, time
from typing import List
import numpy as np

# ── CLI ───────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--rows",  type=int, default=1_000_000)
    p.add_argument("--dim",   type=int, default=128)
    p.add_argument("--k",     type=int, default=10)
    p.add_argument("--iters", type=int, default=5)
    return p.parse_args()

def fmt(ms): return f"{ms:>9.3f} ms"
def stats(ts):
    s = sorted(ts); n = len(s)
    return min(s), (s[n//2] if n%2 else (s[n//2-1]+s[n//2])/2), max(s)
def hdr(t): print(f"\n{'─'*68}\n  {t}\n{'─'*68}")

# ── ApexBase setup ────────────────────────────────────────────────────────────
def setup_apex(vecs, apex_dir):
    from apexbase import ApexClient
    client = ApexClient(dirpath=apex_dir, drop_if_exists=True)
    client.create_table("vecs")
    bs = 10_000
    for s in range(0, len(vecs), bs):
        client.store([{"vec": vecs[i]} for i in range(s, min(s+bs, len(vecs)))])
    return client

# ── Section A: pure kernel (batch_distance API → no TopK heap) ───────────────
def bench_kernel_apex(client, query, metric, iters):
    """Call _topk_distance_ffi with k=n_rows to force scanning everything (no pruning)."""
    # Use batch_topk_distance with k=1 after warmup to get a pure scan measure.
    # Actually: use topk with k=rows to disable heap pruning? Too slow.
    # Better: use the SQL path and measure raw throughput via array_distance (no LIMIT).
    # Simplest proxy: time topk with k=1 (least heap overhead, max kernel time fraction).
    apex_metric = {"l2":"l2","cosine":"cosine_distance","dot":"dot"}[metric]
    q_str = ",".join(f"{v:.6f}" for v in query)
    sql_k1   = (f"SELECT explode_rename(topk_distance(vec,[{q_str}],1,'{apex_metric}'),'_id','dist') FROM vecs")
    sql_k1000= (f"SELECT explode_rename(topk_distance(vec,[{q_str}],1000,'{apex_metric}'),'_id','dist') FROM vecs")

    # warmup
    for _ in range(2): client.execute(sql_k1).to_arrow()

    t_k1 = []
    for _ in range(iters):
        gc.collect(); t0 = time.perf_counter()
        client.execute(sql_k1).to_arrow()
        t_k1.append((time.perf_counter()-t0)*1000)

    t_k1000 = []
    for _ in range(iters):
        gc.collect(); t0 = time.perf_counter()
        client.execute(sql_k1000).to_arrow()
        t_k1000.append((time.perf_counter()-t0)*1000)

    return t_k1, t_k1000

# ── Section B: NumPack reference kernel (no heap) ────────────────────────────
def bench_kernel_numpack(vecs, query, metric, iters):
    from numpack.vector_engine import VectorEngine
    engine = VectorEngine()
    np_metric = {"l2":"l2","cosine":"cosine","dot":"dot"}[metric]

    # warmup
    for _ in range(2): engine.top_k_search(query, vecs, np_metric, k=1)

    t_k1 = []
    for _ in range(iters):
        gc.collect(); t0 = time.perf_counter()
        engine.top_k_search(query, vecs, np_metric, k=1)
        t_k1.append((time.perf_counter()-t0)*1000)

    t_k1000 = []
    for _ in range(iters):
        gc.collect(); t0 = time.perf_counter()
        engine.top_k_search(query, vecs, np_metric, k=1000)
        t_k1000.append((time.perf_counter()-t0)*1000)

    return t_k1, t_k1000

# ── Section C: heap overhead estimate ────────────────────────────────────────
def estimate_heap_overhead(t_k1, t_k1000):
    """
    Both k=1 and k=1000 scan all rows; only heap maintenance differs.
    k=1:    heap size=1 → nearly no push (threshold filter kicks in early)
    k=1000: heap size=1000 → more pushes
    Difference ≈ heap overhead for k=1000 vs k=1.
    """
    _, med1,    _ = stats(t_k1)
    _, med1000, _ = stats(t_k1000)
    return med1000 - med1

# ── main ──────────────────────────────────────────────────────────────────────
def main():
    args = parse_args()
    rows, dim, k = args.rows, args.dim, args.k

    print(f"\n{'='*68}")
    print(f"  SIMD Kernel Micro-Benchmark")
    print(f"  rows={rows:,}  dim={dim}  k={k}  iters={args.iters}")
    print(f"{'='*68}")

    rng = np.random.default_rng(42)
    vecs  = rng.random((rows, dim), dtype=np.float32)
    query = rng.random(dim, dtype=np.float32)

    with tempfile.TemporaryDirectory() as tmp:
        apex_dir = os.path.join(tmp, "apex")
        print("Setting up ApexBase...", end="", flush=True)
        client = setup_apex(vecs, apex_dir)
        print(" done.")

        hdr("Section A — Pure Kernel Cost (k=1: minimal heap overhead)")
        print(f"  {'System':<30} {'Metric':<10} {'k=1 med':>10}  {'k=1000 med':>12}  {'heap Δ':>10}")
        print(f"  {'-'*72}")

        metrics = ["l2", "cosine", "dot"]
        apex_k1 = {}; apex_k1000 = {}
        np_k1   = {}; np_k1000   = {}

        for metric in metrics:
            ax1, ax1000 = bench_kernel_apex(client, query, metric, args.iters)
            nm1, nm1000 = bench_kernel_numpack(vecs, query, metric, args.iters)
            apex_k1[metric] = ax1; apex_k1000[metric] = ax1000
            np_k1[metric]   = nm1; np_k1000[metric]   = nm1000

            _, ax1_med,    _ = stats(ax1)
            _, ax1000_med, _ = stats(ax1000)
            _, nm1_med,    _ = stats(nm1)
            _, nm1000_med, _ = stats(nm1000)

            print(f"  {'ApexBase':<30} {metric:<10} {fmt(ax1_med)} {fmt(ax1000_med)}  {fmt(ax1000_med-ax1_med)}")
            print(f"  {'NumPack (eager)':<30} {metric:<10} {fmt(nm1_med)} {fmt(nm1000_med)}  {fmt(nm1000_med-nm1_med)}")
            print()

        hdr("Section B — Kernel-only speed ratio (k=1, min heap pressure)")
        print(f"  {'Metric':<10} {'ApexBase k=1':>14} {'NumPack k=1':>14} {'Ratio':>10}")
        print(f"  {'-'*52}")
        for metric in metrics:
            _, ax_med, _ = stats(apex_k1[metric])
            _, nm_med, _ = stats(np_k1[metric])
            r = ax_med / nm_med
            mark = "✅" if r < 1.05 else f"⚡ NP {r:.2f}×"
            print(f"  {metric:<10} {fmt(ax_med)} {fmt(nm_med)} {mark}")

        hdr("Section C — Heap overhead estimate")
        print(f"  {'System':<30} {'Metric':<10} {'k=1':>10} {'k=1000':>10} {'heap Δ':>10}")
        print(f"  {'-'*64}")
        for metric in metrics:
            _, ax1_med,    _ = stats(apex_k1[metric])
            _, ax1000_med, _ = stats(apex_k1000[metric])
            _, nm1_med,    _ = stats(np_k1[metric])
            _, nm1000_med, _ = stats(np_k1000[metric])
            print(f"  {'ApexBase':<30} {metric:<10} {fmt(ax1_med)} {fmt(ax1000_med)} {fmt(ax1000_med-ax1_med)}")
            print(f"  {'NumPack':<30} {metric:<10} {fmt(nm1_med)} {fmt(nm1000_med)} {fmt(nm1000_med-nm1_med)}")
            print()

        hdr("Section D — Bottleneck diagnosis")
        for metric in metrics:
            _, ax_k1_med, _ = stats(apex_k1[metric])
            _, nm_k1_med, _ = stats(np_k1[metric])
            kernel_gap = ax_k1_med - nm_k1_med
            print(f"  {metric}: kernel gap = {kernel_gap:+.2f}ms  "
                  f"({'kernel bottleneck' if kernel_gap > 1.0 else 'within noise'})")

        client.close()
    print(f"\n{'='*68}\n")

if __name__ == "__main__":
    main()
