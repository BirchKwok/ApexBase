"""
Quick benchmark: ApexBase WITH vs WITHOUT Python-level query cache.
Run from the repo root after activating the dev environment.
"""
import gc
import shutil
import time
import tempfile
from contextlib import contextmanager

from apexbase import ApexClient

ROWS      = 100_000
WARMUP    = 3
ITERS     = 20
LOOKUP_ID = 50_000

@contextmanager
def timer():
    t = [0.0]
    start = time.perf_counter()
    yield t
    t[0] = time.perf_counter() - start

def insert_data(client):
    batch = {
        "name":     [f"user_{i}"   for i in range(ROWS)],
        "age":      list(range(ROWS)),
        "score":    [float(i) * 0.1 for i in range(ROWS)],
        "city":     [f"city_{i % 100}" for i in range(ROWS)],
        "category": [f"cat_{i % 10}"  for i in range(ROWS)],
    }
    client.store(batch)

def bench(label: str, fn, warmup=WARMUP, iters=ITERS):
    for _ in range(warmup):
        fn()
    gc.collect()
    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    avg_us = (sum(times) / len(times)) * 1e6
    print(f"  {label:<40s}  {avg_us:8.1f} µs")
    return avg_us


def main():
    tmpdir = tempfile.mkdtemp(prefix="apex_nocache_")
    try:
        client = ApexClient(tmpdir, drop_if_exists=True)
        client.create_table("default")
        print(f"Inserting {ROWS:,} rows…", flush=True)
        insert_data(client)
        print()

        sql = f"SELECT * FROM default WHERE _id = {LOOKUP_ID}"

        # ── Warm backend path ───────────────────────────────────────────────
        print("Point Lookup  (execute, warm backend):")
        def lookup_warm():
            return client.execute(sql)
        t_warm = bench("execute(_id=X)  warm backend", lookup_warm)

        # ── Cold backend metadata/data path: no query results are cached.
        print("\nPoint Lookup  (execute, flush read caches before each call):")
        def lookup_cold_read_path():
            client.flush_cache()
            return client.execute(sql)
        t_cold = bench("execute(_id=X)  cold read path", lookup_cold_read_path)

        # ── retrieve() for reference ────────────────────────────────────────
        print("\nPoint Lookup  (retrieve, no SQL path):")
        def lookup_retrieve():
            return client.retrieve(LOOKUP_ID)
        t_retrieve = bench("retrieve(_id)              ", lookup_retrieve)

        print()
        print(f"  warm speedup over cold read path : {t_cold/t_warm:.1f}×")
        print(f"  retrieve vs warm execute         : {t_warm/t_retrieve:.1f}× (execute is {'slower' if t_warm>t_retrieve else 'faster'})")
        print(f"  retrieve vs cold read execute    : {t_cold/t_retrieve:.1f}×")

    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

if __name__ == "__main__":
    main()
