"""Performance benchmark after QuerySignature refactor — check for regressions."""
import time, os, shutil
from apexbase import ApexClient

DB = "/tmp/apexbase_bench_refactor"
if os.path.exists(DB):
    shutil.rmtree(DB)

c = ApexClient(DB)
c.create_table("bench", {
    "name": "string", "age": "int", "score": "float",
    "city": "string", "active": "bool"
})

# Store 200K rows
data = [{"name": f"user_{i}", "age": 20 + (i % 60), "score": 50.0 + (i % 100) * 0.5,
         "city": ["Beijing", "Shanghai", "Shenzhen", "Guangzhou", "Hangzhou"][i % 5],
         "active": i % 3 != 0} for i in range(200_000)]
t0 = time.perf_counter()
c.store(data)
print(f"Store 200K rows: {time.perf_counter() - t0:.3f}s")

# Warm up
c.execute("SELECT COUNT(*) FROM bench")
c.execute("SELECT * FROM bench LIMIT 1")

def bench(label, sql, iters=50):
    times = []
    for _ in range(iters):
        t = time.perf_counter()
        c.execute(sql)
        times.append(time.perf_counter() - t)
    avg = sum(times) / len(times) * 1000
    mn = min(times) * 1000
    print(f"{label:40s}  avg={avg:8.3f}ms  min={mn:8.3f}ms")

print("\n=== Query Performance (200K rows x 5 cols) ===\n")

bench("COUNT(*)", "SELECT COUNT(*) FROM bench")
bench("Point lookup _id=100000", "SELECT * FROM bench WHERE _id = 100000", iters=100)
bench("SELECT * LIMIT 100", "SELECT * FROM bench LIMIT 100")
bench("SELECT * LIMIT 10", "SELECT * FROM bench LIMIT 10")
bench("String filter city='Beijing'", "SELECT * FROM bench WHERE city = 'Beijing'")
bench("LIKE filter name LIKE 'user_1%'", "SELECT * FROM bench WHERE name LIKE 'user_1%'")
bench("BETWEEN age 30-40", "SELECT * FROM bench WHERE age BETWEEN 30 AND 40")
bench("GROUP BY city (5 groups)", "SELECT city, AVG(score), COUNT(*) FROM bench GROUP BY city", iters=20)
bench("GROUP BY age (60 groups)", "SELECT age, COUNT(*), AVG(score) FROM bench GROUP BY age", iters=20)
bench("Filter+Order+Limit", "SELECT * FROM bench WHERE age BETWEEN 30 AND 40 ORDER BY score DESC LIMIT 100", iters=20)

# Transaction test
print("\n=== Transaction Performance ===\n")
times = []
for i in range(20):
    t = time.perf_counter()
    c.execute("BEGIN")
    c.execute(f"INSERT INTO bench (name, age, score, city, active) VALUES ('txn_{i}', 25, 75.0, 'X', true)")
    c.execute("COMMIT")
    times.append(time.perf_counter() - t)
avg = sum(times) / len(times) * 1000
mn = min(times) * 1000
print(f"{'BEGIN+INSERT+COMMIT':40s}  avg={avg:8.3f}ms  min={mn:8.3f}ms")

# Multi-statement transaction
times = []
for i in range(20):
    t = time.perf_counter()
    c.execute(f"BEGIN; INSERT INTO bench (name, age, score, city, active) VALUES ('ms_{i}', 30, 80.0, 'Y', true); COMMIT;")
    times.append(time.perf_counter() - t)
avg = sum(times) / len(times) * 1000
mn = min(times) * 1000
print(f"{'Multi-stmt BEGIN+INSERT+COMMIT':40s}  avg={avg:8.3f}ms  min={mn:8.3f}ms")

# Insert throughput
print("\n=== Insert Throughput ===\n")
batch = [{"name": f"b_{i}", "age": 30, "score": 60.0, "city": "Test", "active": True} for i in range(1000)]
times = []
for _ in range(10):
    t = time.perf_counter()
    c.store(batch)
    times.append(time.perf_counter() - t)
avg = sum(times) / len(times) * 1000
print(f"{'Insert 1K rows':40s}  avg={avg:8.2f}ms")

del c
shutil.rmtree(DB, ignore_errors=True)
print("\nDone.")
