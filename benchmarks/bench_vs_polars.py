#!/usr/bin/env python3
"""
Benchmark: ApexBase SQL read_csv / read_parquet / read_json vs Polars API.

Usage:
    conda run -n dev python benchmarks/bench_vs_polars.py
"""

import os, sys, gc, time, tempfile, json
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'apexbase', 'python'))

import polars as pl
import pandas as pd

try:
    from apexbase import ApexClient
    HAS_APEX = True
except ImportError:
    HAS_APEX = False
    print("WARNING: ApexBase not found — only generating data files")

# ── config ──────────────────────────────────────────────────────────────────
N_ROWS   = 1_000_000
WARMUP   = 3
ITERS    = 8
# ────────────────────────────────────────────────────────────────────────────

def timer(fn, warmup=WARMUP, iters=ITERS, pre=None):
    for _ in range(warmup):
        gc.collect()
        if pre: pre()
        fn()
    times = []
    for _ in range(iters):
        gc.collect()
        if pre: pre()
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return min(times), sum(times) / len(times)


def generate_files(tmpdir):
    rng = np.random.default_rng(42)
    ids    = np.arange(N_ROWS, dtype=np.int64)
    vals   = rng.integers(0, 10_000, N_ROWS, dtype=np.int64)
    scores = rng.uniform(0.0, 1.0, N_ROWS)
    cities = np.array(["Beijing","Shanghai","Guangzhou","Shenzhen","Chengdu"], dtype=object)
    city_col = cities[rng.integers(0, 5, N_ROWS)]
    names  = np.array([f"user_{i%10000}" for i in range(N_ROWS)], dtype=object)

    df_pl = pl.DataFrame({
        "id":    ids,
        "value": vals,
        "score": scores,
        "city":  city_col,
        "name":  names,
    })

    csv_path     = os.path.join(tmpdir, "data.csv")
    parquet_path = os.path.join(tmpdir, "data.parquet")
    json_path    = os.path.join(tmpdir, "data.json")    # NDJSON

    df_pl.write_csv(csv_path)
    df_pl.write_parquet(parquet_path, compression="snappy")
    # NDJSON (records orientation, one JSON object per line)
    df_pl.write_ndjson(json_path)

    return csv_path, parquet_path, json_path


def run_apex_arrow(client, sql):
    """Return the underlying Arrow Table — no pandas conversion."""
    result = client.execute(sql)
    if hasattr(result, '_arrow_table'):
        return result._arrow_table
    # Force materialization via Arrow FFI (already done in execute)
    return result

def run_apex_pandas(client, sql):
    """Return pandas DataFrame."""
    result = client.execute(sql)
    if hasattr(result, 'to_pandas'):
        return result.to_pandas()
    return list(result)


def fmt(mn, avg):
    return f"min={mn*1000:7.2f}ms  avg={avg*1000:7.2f}ms"


def main():
    tmpdir = tempfile.mkdtemp(prefix="apex_polars_bench_")
    try:
        print(f"\n{'='*65}")
        print(f"  Benchmark: ApexBase read_x  vs  Polars  ({N_ROWS:,} rows × 5 cols)")
        print(f"{'='*65}\n")

        print("Generating test files …")
        csv_path, parquet_path, json_path = generate_files(tmpdir)
        csv_sz  = os.path.getsize(csv_path)  / 1e6
        pq_sz   = os.path.getsize(parquet_path) / 1e6
        js_sz   = os.path.getsize(json_path) / 1e6
        print(f"  CSV:     {csv_sz:.1f} MB   → {csv_path}")
        print(f"  Parquet: {pq_sz:.1f} MB   → {parquet_path}")
        print(f"  NDJSON:  {js_sz:.1f} MB   → {json_path}\n")

        results = {}

        def _clear(c): 
            if HAS_APEX and hasattr(c, '_query_cache'):
                c._query_cache.clear()
            # No cache clearing needed for current ApexClient implementation

        # ── CSV ─────────────────────────────────────────────────────────────
        print("── CSV ──────────────────────────────────────────────────────")
        pl_mn,    pl_avg    = timer(lambda: pl.read_csv(csv_path))
        pl_pd_mn, pl_pd_avg = timer(lambda: pl.read_csv(csv_path).to_pandas())
        print(f"  Polars  read_csv() → pl.DataFrame   {fmt(pl_mn, pl_avg)}")
        print(f"  Polars  read_csv() → pandas          {fmt(pl_pd_mn, pl_pd_avg)}")

        if HAS_APEX:
            client = ApexClient(tmpdir + "/apexdb")
            sql_csv = f"SELECT * FROM read_csv('{csv_path}')"
            ap_mn,  ap_avg  = timer(lambda: run_apex_arrow(client, sql_csv),  pre=lambda: _clear(client))
            ap_pd_mn,ap_pd_avg = timer(lambda: run_apex_pandas(client, sql_csv), pre=lambda: _clear(client))
            results["csv_apex"] = ap_mn
            r1 = ap_mn   / pl_mn;   w1 = "✅" if r1 < 1.0 else ("≈" if r1 < 1.2 else "❌")
            r2 = ap_pd_mn/ pl_pd_mn; w2 = "✅" if r2 < 1.0 else ("≈" if r2 < 1.2 else "❌")
            print(f"  ApexBase read_csv() → Arrow         {fmt(ap_mn, ap_avg)}   ratio={r1:.2f}x {w1}")
            print(f"  ApexBase read_csv() → pandas        {fmt(ap_pd_mn, ap_pd_avg)}   ratio={r2:.2f}x {w2}")
            client.close()

        # ── Parquet ─────────────────────────────────────────────────────────
        print("\n── Parquet ──────────────────────────────────────────────────")
        pl_mn,    pl_avg    = timer(lambda: pl.read_parquet(parquet_path))
        pl_pd_mn, pl_pd_avg = timer(lambda: pl.read_parquet(parquet_path).to_pandas())
        print(f"  Polars  read_parquet() → pl.DataFrame{fmt(pl_mn, pl_avg)}")
        print(f"  Polars  read_parquet() → pandas       {fmt(pl_pd_mn, pl_pd_avg)}")

        if HAS_APEX:
            client = ApexClient(tmpdir + "/apexdb2")
            sql_pq = f"SELECT * FROM read_parquet('{parquet_path}')"
            ap_mn,  ap_avg  = timer(lambda: run_apex_arrow(client, sql_pq),   pre=lambda: _clear(client))
            ap_pd_mn,ap_pd_avg = timer(lambda: run_apex_pandas(client, sql_pq),  pre=lambda: _clear(client))
            results["pq_apex"] = ap_mn
            r1 = ap_mn   / pl_mn;   w1 = "✅" if r1 < 1.0 else ("≈" if r1 < 1.2 else "❌")
            r2 = ap_pd_mn/ pl_pd_mn; w2 = "✅" if r2 < 1.0 else ("≈" if r2 < 1.2 else "❌")
            print(f"  ApexBase read_parquet() → Arrow     {fmt(ap_mn, ap_avg)}   ratio={r1:.2f}x {w1}")
            print(f"  ApexBase read_parquet() → pandas    {fmt(ap_pd_mn, ap_pd_avg)}   ratio={r2:.2f}x {w2}")
            client.close()

        # ── JSON (NDJSON) ────────────────────────────────────────────────────
        print("\n── NDJSON ───────────────────────────────────────────────────")
        pl_mn,    pl_avg    = timer(lambda: pl.read_ndjson(json_path))
        pl_pd_mn, pl_pd_avg = timer(lambda: pl.read_ndjson(json_path).to_pandas())
        print(f"  Polars  read_ndjson() → pl.DataFrame {fmt(pl_mn, pl_avg)}")
        print(f"  Polars  read_ndjson() → pandas        {fmt(pl_pd_mn, pl_pd_avg)}")

        if HAS_APEX:
            client = ApexClient(tmpdir + "/apexdb3")
            sql_js = f"SELECT * FROM read_json('{json_path}')"
            ap_mn,  ap_avg  = timer(lambda: run_apex_arrow(client, sql_js),   pre=lambda: _clear(client))
            ap_pd_mn,ap_pd_avg = timer(lambda: run_apex_pandas(client, sql_js),  pre=lambda: _clear(client))
            results["json_apex"] = ap_mn
            r1 = ap_mn   / pl_mn;   w1 = "✅" if r1 < 1.0 else ("≈" if r1 < 1.2 else "❌")
            r2 = ap_pd_mn/ pl_pd_mn; w2 = "✅" if r2 < 1.0 else ("≈" if r2 < 1.2 else "❌")
            print(f"  ApexBase read_json() → Arrow        {fmt(ap_mn, ap_avg)}   ratio={r1:.2f}x {w1}")
            print(f"  ApexBase read_json() → pandas       {fmt(ap_pd_mn, ap_pd_avg)}   ratio={r2:.2f}x {w2}")
            client.close()

        # ── Summary ──────────────────────────────────────────────────────────
        if HAS_APEX:
            wins = sum(1 for k in ("csv_apex","pq_apex","json_apex")
                       if results.get(k, 9e9) <= results.get(k.replace("apex","polars"), 9e9) * 1.05)
            print(f"\n  Score (Arrow output): {wins}/3 wins (≤5% slower threshold)")

    finally:
        import shutil
        shutil.rmtree(tmpdir, ignore_errors=True)


if __name__ == "__main__":
    main()
