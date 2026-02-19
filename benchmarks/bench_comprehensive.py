"""Comprehensive benchmark: store, query, GROUP BY, random read, insert throughput."""
import time, os, shutil
from apexbase import ApexClient

DB = "/tmp/apexbase_bench_v2"
N = 100_000
ROUNDS = 10

def gen(n, off=0):
    return [{"name": f"user_{off+i}", "age": 20+(i%60), "score": 50.0+(i%100)*0.5,
             "city": ["Beijing","Shanghai","Shenzhen","Guangzhou","Hangzhou"][i%5],
             "active": i%3!=0} for i in range(n)]

def bench():
    if os.path.exists(DB): shutil.rmtree(DB)
    c = ApexClient(DB)
    tbl = c._current_table

    # Store
    t0 = time.perf_counter()
    for r in range(ROUNDS):
        c.store(gen(N, r*N))
    t_store = time.perf_counter() - t0
    total = N * ROUNDS
    print(f"Store {total:,} rows: {t_store:.2f}s ({total/t_store:,.0f} rows/s)")

    # Cold reload
    del c
    t1 = time.perf_counter()
    c2 = ApexClient(DB)
    tbl = c2._current_table
    c2.execute(f"SELECT COUNT(*) FROM {tbl}")
    print(f"Cold open: {time.perf_counter()-t1:.3f}s")

    # SELECT * → pandas
    t2 = time.perf_counter()
    df = c2.execute(f"SELECT * FROM {tbl}").to_pandas()
    print(f"SELECT * → pandas ({len(df):,}): {time.perf_counter()-t2:.3f}s")

    # COUNT(*)
    times = []
    for _ in range(20):
        t = time.perf_counter()
        c2.execute(f"SELECT COUNT(*) FROM {tbl}")
        times.append(time.perf_counter()-t)
    print(f"COUNT(*) avg: {sum(times)/len(times)*1000:.2f}ms")

    # GROUP BY city (5 groups)
    times = []
    for _ in range(20):
        t = time.perf_counter()
        c2.execute(f"SELECT city, AVG(score), COUNT(*) FROM {tbl} GROUP BY city")
        times.append(time.perf_counter()-t)
    print(f"GROUP BY city (5 groups) avg: {sum(times)/len(times)*1000:.2f}ms")

    # GROUP BY age (60 groups)
    times = []
    for _ in range(20):
        t = time.perf_counter()
        c2.execute(f"SELECT age, COUNT(*), AVG(score) FROM {tbl} GROUP BY age")
        times.append(time.perf_counter()-t)
    print(f"GROUP BY age (60 groups) avg: {sum(times)/len(times)*1000:.2f}ms")

    # WHERE + ORDER BY + LIMIT
    times = []
    for _ in range(20):
        t = time.perf_counter()
        c2.execute(f"SELECT * FROM {tbl} WHERE age BETWEEN 30 AND 40 ORDER BY score DESC LIMIT 100")
        times.append(time.perf_counter()-t)
    print(f"Filter+Order+Limit avg: {sum(times)/len(times)*1000:.2f}ms")

    # WHERE string filter
    times = []
    for _ in range(20):
        t = time.perf_counter()
        c2.execute(f"SELECT * FROM {tbl} WHERE city = 'Beijing' LIMIT 100")
        times.append(time.perf_counter()-t)
    print(f"String filter avg: {sum(times)/len(times)*1000:.2f}ms")

    # Point lookup by _id
    times = []
    for _ in range(100):
        t = time.perf_counter()
        c2.execute(f"SELECT * FROM {tbl} WHERE _id = 500000")
        times.append(time.perf_counter()-t)
    print(f"Point lookup _id avg: {sum(times)/len(times)*1000:.3f}ms")

    # Incremental insert (after initial load)
    batch = gen(1000)
    times = []
    for _ in range(20):
        t = time.perf_counter()
        c2.store(batch)
        times.append(time.perf_counter()-t)
    print(f"Insert 1K rows avg: {sum(times)/len(times)*1000:.2f}ms")

    del c2
    shutil.rmtree(DB, ignore_errors=True)

if __name__ == "__main__":
    bench()
