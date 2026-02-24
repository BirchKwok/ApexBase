#!/usr/bin/env python3
"""
TopK distance benchmark: ApexBase vs SimSIMD (in-memory baseline).

SimSIMD is the fastest SIMD-accelerated distance library on CPU.
ApexBase reads from on-disk storage (binary-encoded float32 columns).

The purpose is to measure how close ApexBase's fused O(n log k) heap
gets to the theoretical in-memory SIMD ceiling.

Usage:
    python benchmarks/bench_simsimd.py [--rows N] [--dim D] [--k K] [--iters I]

Requires: simsimd  (pip install simsimd)
"""

import argparse
import gc
import tempfile
import time

import numpy as np
import simsimd


# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--rows",  type=int, default=1_000_000)
    p.add_argument("--dim",   type=int, default=128)
    p.add_argument("--k",     type=int, default=10)
    p.add_argument("--iters", type=int, default=10)
    return p.parse_args()


def fmt(ms: float) -> str:
    return f"{ms:.3f} ms"


def median(vals):
    s = sorted(vals)
    n = len(s)
    return s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2


# ─────────────────────────────────────────────────────────────────────────────
# SimSIMD topk helper
# ─────────────────────────────────────────────────────────────────────────────

def simsimd_topk(matrix: np.ndarray, query: np.ndarray, k: int, metric: str) -> np.ndarray:
    """Compute topk row indices by distance using SimSIMD + np.argpartition."""
    if metric == "l2":
        dists = np.asarray(simsimd.euclidean(query, matrix))
    elif metric == "l2_squared":
        dists = np.asarray(simsimd.sqeuclidean(query, matrix))
    elif metric == "cosine":
        dists = np.asarray(simsimd.cosine(query, matrix))
    elif metric == "dot":
        # inner product: larger = better, so negate for topk ascending
        dists = -np.asarray(simsimd.inner(query, matrix))
    else:
        raise ValueError(metric)

    k = min(k, len(dists))
    idx = np.argpartition(dists, k)[:k]
    idx = idx[np.argsort(dists[idx])]   # sort the k candidates
    return idx, dists[idx]


# ─────────────────────────────────────────────────────────────────────────────
# ApexBase setup & bench
# ─────────────────────────────────────────────────────────────────────────────

def setup_apex(vecs: np.ndarray, tmp_dir: str):
    from apexbase.client import ApexClient, encode_vector
    client = ApexClient(dirpath=tmp_dir, drop_if_exists=True)
    client.create_table("vecs")
    batch = 10_000
    n = len(vecs)
    for s in range(0, n, batch):
        client.store([{"vec": encode_vector(vecs[i])} for i in range(s, min(s + batch, n))])
    return client


def bench_apex(client, query: np.ndarray, k: int, metric: str, iters: int):
    # Apex metric names
    _m = {"l2": "l2", "l2_squared": "l2_squared", "cosine": "cosine_distance", "dot": "dot"}
    apex_metric = _m[metric]
    # One warmup to populate OS page cache (NOT timed)
    client.topk_distance("vec", query, k=k, metric=apex_metric)
    times = []
    for _ in range(iters):
        gc.collect()
        t0 = time.perf_counter()
        client.topk_distance("vec", query, k=k, metric=apex_metric)
        times.append((time.perf_counter() - t0) * 1000)
    return times


def bench_simsimd(matrix: np.ndarray, query: np.ndarray, k: int, metric: str, iters: int):
    times = []
    for _ in range(iters):
        gc.collect()
        t0 = time.perf_counter()
        simsimd_topk(matrix, query, k, metric)
        times.append((time.perf_counter() - t0) * 1000)
    return times


# ─────────────────────────────────────────────────────────────────────────────
# Correctness check
# ─────────────────────────────────────────────────────────────────────────────

def verify(client, matrix: np.ndarray, query: np.ndarray, k: int):
    """Check that ApexBase and SimSIMD return the same top-k set for L2."""
    from apexbase.client import encode_vector

    # ApexBase result (_id values)
    apex_rows = client.topk_distance("vec", query, k=k, metric="l2").to_dict()
    apex_ids = {r["_id"] for r in apex_rows}

    # SimSIMD result (row indices = _id since we inserted in order 0..n-1)
    ss_idx, _ = simsimd_topk(matrix, query, k, "l2")
    ss_ids = set(int(i) for i in ss_idx)

    if apex_ids == ss_ids:
        return True, None
    return False, (apex_ids, ss_ids)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    rng  = np.random.default_rng(42)
    vecs = rng.random((args.rows, args.dim), dtype=np.float32)
    query = rng.random(args.dim, dtype=np.float32)

    caps = simsimd.get_capabilities()
    active = [k for k, v in caps.items() if v]

    print(f"\n{'='*72}")
    print(f"TopK Distance Benchmark: ApexBase (on-disk) vs SimSIMD (in-RAM)")
    print(f"  rows={args.rows:,}  dim={args.dim}  k={args.k}  iters={args.iters}")
    print(f"  SimSIMD capabilities: {', '.join(active)}")
    print(f"{'='*72}\n")

    with tempfile.TemporaryDirectory() as apex_dir:
        print("Setting up ApexBase (storing vectors to disk)...", end="", flush=True)
        client = setup_apex(vecs, apex_dir)
        print(" done.")

        print("SimSIMD uses the in-memory float32 matrix directly.\n")

        # OS page-cache warmup: read through all data once (not timed)
        print("Warming up OS page cache...", end="", flush=True)
        client.topk_distance("vec", query, k=args.k, metric="l2")
        print(" done.\n")

        # ── Correctness ──────────────────────────────────────────────────────
        ok, diff = verify(client, vecs, query, args.k)
        status = "✅ match" if ok else f"❌ MISMATCH apex={diff[0]} simsimd={diff[1]}"
        print(f"Correctness check (L2, k={args.k}): {status}\n")

        # ── Head-to-head ─────────────────────────────────────────────────────
        metrics = [
            ("l2",         "L2 / Euclidean",     "euclidean"),
            ("l2_squared", "L2 Squared",          "sqeuclidean"),
            ("cosine",     "Cosine distance",     "cosine"),
            ("dot",        "Dot / Inner product", "inner (neg)"),
        ]

        w = 22
        hdr = f"{'Metric':<{w}} {'ApexBase':>12} {'SimSIMD':>12} {'Ratio A/S':>12}  Note"
        print(hdr)
        print("─" * len(hdr))

        for metric, label, ss_label in metrics:
            apex_t  = bench_apex(client, query, args.k, metric, args.iters)
            simsimd_t = bench_simsimd(vecs, query, args.k, metric, args.iters)

            a_med = median(apex_t)
            s_med = median(simsimd_t)
            ratio = a_med / s_med
            flag  = "✅" if ratio < 2.0 else ("⚠️" if ratio < 5.0 else "❌")

            note = f"(SimSIMD: {ss_label})"
            print(f"{label:<{w}} {fmt(a_med):>12} {fmt(s_med):>12} {ratio:>10.2f}x {flag}  {note}")

        print()

        # ── Pipeline breakdown (L2) ──────────────────────────────────────────
        print("── Pipeline breakdown: what each step costs (L2) ──")

        def timed(fn, iters):
            results = []
            for _ in range(iters):
                gc.collect()
                t0 = time.perf_counter()
                fn()
                results.append((time.perf_counter() - t0) * 1000)
            return median(results)

        k = args.k
        n = args.rows

        # 1. SimSIMD: distance-only (no topk)
        t_simd_dist = timed(lambda: simsimd.euclidean(query, vecs), args.iters)

        # 2. SimSIMD: distance + argpartition topk
        t_simd_topk = timed(lambda: simsimd_topk(vecs, query, k, "l2"), args.iters)

        # 3. Numpy baseline: L2 norm via numpy + argpartition
        t_np_topk = timed(
            lambda: (lambda d: d[np.argpartition(d, k)[:k]])(
                np.linalg.norm(vecs - query, axis=1).astype(np.float32)
            ),
            args.iters,
        )

        # 4. ApexBase full pipeline (storage decode + heap)
        t_apex = timed(lambda: client.topk_distance("vec", query, k=k, metric="l2"), args.iters)

        print(f"  {'Step':<42} {'Time':>10}  {'M vec/s':>10}")
        print(f"  {'-'*64}")
        print(f"  {'SimSIMD euclidean() — pure distance, no topk':<42} {fmt(t_simd_dist):>10}  {n/(t_simd_dist/1000)/1e6:>9.1f}")
        print(f"  {'SimSIMD + argpartition topk':<42} {fmt(t_simd_topk):>10}  {n/(t_simd_topk/1000)/1e6:>9.1f}")
        print(f"  {'NumPy linalg.norm + argpartition topk':<42} {fmt(t_np_topk):>10}  {n/(t_np_topk/1000)/1e6:>9.1f}")
        print(f"  {'ApexBase topk_distance (storage+decode+heap)':<42} {fmt(t_apex):>10}  {n/(t_apex/1000)/1e6:>9.1f}")
        print()

        decode_overhead = max(t_apex - t_simd_topk, 0.0)
        print(f"  Storage/decode overhead vs SimSIMD+topk: {fmt(decode_overhead)} ({decode_overhead/t_apex*100:.0f}% of ApexBase time)")
        print()

        # ── min/med/max detail ────────────────────────────────────────────────
        print("── Per-run detail (min / med / max) ──")
        for metric, label, _ in metrics:
            at = bench_apex(client, query, args.k, metric, args.iters)
            st = bench_simsimd(vecs, query, args.k, metric, args.iters)
            print(f"  {label:<22}  ApexBase {fmt(min(at))}/{fmt(median(at))}/{fmt(max(at))}"
                  f"   SimSIMD {fmt(min(st))}/{fmt(median(st))}/{fmt(max(st))}")

        client.close()


if __name__ == "__main__":
    main()
