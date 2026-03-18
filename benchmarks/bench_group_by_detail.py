"""Detailed GROUP BY benchmark to identify bottlenecks."""
import time, os, shutil, random
from apexbase import ApexClient

DB = "/tmp/apexbase_bench_gb"
N = 1_000_000

CITIES = ["Beijing", "Shanghai", "Guangzhou", "Shenzhen", "Hangzhou",
          "Nanjing", "Chengdu", "Wuhan", "Xian", "Qingdao"]
CATEGORIES = ["Electronics", "Clothing", "Food", "Sports", "Books",
              "Home", "Auto", "Health", "Travel", "Gaming"]

def gen_data(n):
    rng = random.Random(42)
    return {
        "name": [f"user_{i}" for i in range(n)],
        "age": [rng.randint(18, 80) for _ in range(n)],
        "score": [round(rng.uniform(0, 100), 2) for _ in range(n)],
        "city": [rng.choice(CITIES) for _ in range(n)],
        "category": [rng.choice(CATEGORIES) for _ in range(n)],
    }

def bench():
    if os.path.exists(DB): shutil.rmtree(DB)
    c = ApexClient(DB, drop_if_exists=True)
    c.create_table('default')
    c.store(gen_data(N))
    tbl = 'default'

    # Warm up caches
    for _ in range(3):
        c.execute(f"SELECT city, category, COUNT(*), AVG(score) FROM {tbl} GROUP BY city, category")

    queries = {
        "GROUP BY 1-col (city, 10 grp)": f"SELECT city, COUNT(*), AVG(score) FROM {tbl} GROUP BY city",
        "GROUP BY 1-col (category, 10 grp)": f"SELECT category, COUNT(*), AVG(score) FROM {tbl} GROUP BY category",
        "GROUP BY 2-col (city,cat, 100 grp)": f"SELECT city, category, COUNT(*), AVG(score) FROM {tbl} GROUP BY city, category",
        "GROUP BY 2-col COUNT(*) only": f"SELECT city, category, COUNT(*) FROM {tbl} GROUP BY city, category",
        "GROUP BY 1-col via FFI only": f"SELECT city, COUNT(*), AVG(score) FROM {tbl} GROUP BY city",
    }

    for name, sql in queries.items():
        # Warm
        for _ in range(3):
            c.execute(sql)
        times = []
        for _ in range(20):
            t0 = time.perf_counter()
            c.execute(sql)
            times.append((time.perf_counter() - t0) * 1000)
        avg = sum(times) / len(times)
        mn = min(times)
        print(f"{name:45s}  avg={avg:.3f}ms  min={mn:.3f}ms")

    # Also measure raw _execute_arrow_ffi
    sql = f"SELECT city, category, COUNT(*), AVG(score) FROM {tbl} GROUP BY city, category"
    for _ in range(3):
        ptrs = c._storage._execute_arrow_ffi(sql)
        c._storage._free_arrow_ffi(*ptrs)
    times = []
    for _ in range(20):
        t0 = time.perf_counter()
        ptrs = c._storage._execute_arrow_ffi(sql)
        c._storage._free_arrow_ffi(*ptrs)
        times.append((time.perf_counter() - t0) * 1000)
    avg = sum(times) / len(times)
    mn = min(times)
    print(f"{'Raw _execute_arrow_ffi (2-col GB)':45s}  avg={avg:.3f}ms  min={mn:.3f}ms")

    # Measure just the Rust execute() call (returns dict)
    for _ in range(3):
        c._storage.execute(sql)
    times = []
    for _ in range(20):
        t0 = time.perf_counter()
        c._storage.execute(sql)
        times.append((time.perf_counter() - t0) * 1000)
    avg = sum(times) / len(times)
    mn = min(times)
    print(f"{'Raw Rust execute() (2-col GB)':45s}  avg={avg:.3f}ms  min={mn:.3f}ms")

    # Now measure the Python client.execute overhead
    for _ in range(3):
        c.execute(sql)
    times = []
    for _ in range(20):
        t0 = time.perf_counter()
        c.execute(sql)
        times.append((time.perf_counter() - t0) * 1000)
    avg = sum(times) / len(times)
    mn = min(times)
    print(f"{'Python client.execute() (2-col GB)':45s}  avg={avg:.3f}ms  min={mn:.3f}ms")

    # UPDATE benchmark detail
    print("\n--- UPDATE Detail ---")
    sql_update = f"UPDATE {tbl} SET score = 50.0 WHERE age = 25"
    # First run to warm up
    c.execute(sql_update)
    times = []
    for _ in range(10):
        t0 = time.perf_counter()
        c.execute(sql_update)
        times.append((time.perf_counter() - t0) * 1000)
    avg = sum(times) / len(times)
    mn = min(times)
    print(f"{'UPDATE SET score WHERE age=25':45s}  avg={avg:.3f}ms  min={mn:.3f}ms")

    # Raw Rust execute for UPDATE
    times = []
    for _ in range(10):
        t0 = time.perf_counter()
        c._storage.execute(sql_update)
        times.append((time.perf_counter() - t0) * 1000)
    avg = sum(times) / len(times)
    mn = min(times)
    print(f"{'Raw Rust execute() UPDATE':45s}  avg={avg:.3f}ms  min={mn:.3f}ms")

    c.close()
    shutil.rmtree(DB, ignore_errors=True)

if __name__ == "__main__":
    bench()
