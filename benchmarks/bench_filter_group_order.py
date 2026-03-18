"""
Benchmark: execute_filter_group_order V4 fast path
Tests: SELECT region, SUM(value) FROM t WHERE status='active' GROUP BY region ORDER BY ... LIMIT

Compares the fast path (V4 single-pass mmap) against the generic query path.
"""
import time
import random
import shutil
import os
from apexbase import ApexClient

DB_PATH = "/tmp/bench_fgo_db"
TABLE = "orders"
NUM_ROWS = 500_000
REGIONS = [f"region_{i}" for i in range(50)]
STATUSES = ["active", "inactive", "pending", "archived"]
WARMUP = 3
ITERATIONS = 20


def setup():
    if os.path.exists(DB_PATH):
        shutil.rmtree(DB_PATH)
    db = ApexClient(DB_PATH)
    db.create_table(TABLE)

    # Bulk insert
    batch_size = 10000
    for start in range(0, NUM_ROWS, batch_size):
        rows = []
        for i in range(start, min(start + batch_size, NUM_ROWS)):
            rows.append({
                "status": random.choice(STATUSES),
                "region": random.choice(REGIONS),
                "value": random.randint(1, 1000),
                "amount": round(random.uniform(1.0, 500.0), 2),
            })
        db.store(rows)

    # Force flush
    db.close()
    db2 = ApexClient(DB_PATH)
    db2.use_table(TABLE)
    return db2


def bench_query(db, sql, label, warmup=WARMUP, iterations=ITERATIONS):
    # Warmup
    for _ in range(warmup):
        db.execute(sql)

    times = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        result = db.execute(sql)
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)

    times.sort()
    avg = sum(times) / len(times)
    med = times[len(times) // 2]
    mn = times[0]
    mx = times[-1]
    p95 = times[int(len(times) * 0.95)]
    rows = len(result) if result else 0
    print(f"  {label:50s}  avg={avg:8.3f}ms  med={med:8.3f}ms  min={mn:8.3f}ms  max={mx:8.3f}ms  p95={p95:8.3f}ms  rows={rows}")
    return avg


def main():
    print(f"Setting up {NUM_ROWS:,} rows × 4 cols, {len(REGIONS)} regions, {len(STATUSES)} statuses...")
    db = setup()
    print("Setup complete.\n")

    print("=" * 120)
    print("Benchmark: Filter + Group + Order (V4 fast path)")
    print("=" * 120)

    queries = [
        # SUM with ORDER BY DESC LIMIT
        (
            f"SELECT region, SUM(value) AS total FROM {TABLE} WHERE status = 'active' GROUP BY region ORDER BY total DESC LIMIT 10",
            "SUM + WHERE + GROUP BY + ORDER DESC LIMIT 10"
        ),
        # COUNT with ORDER BY DESC LIMIT
        (
            f"SELECT region, COUNT(*) AS cnt FROM {TABLE} WHERE status = 'active' GROUP BY region ORDER BY cnt DESC LIMIT 10",
            "COUNT + WHERE + GROUP BY + ORDER DESC LIMIT 10"
        ),
        # SUM all groups
        (
            f"SELECT region, SUM(value) AS total FROM {TABLE} WHERE status = 'active' GROUP BY region ORDER BY total DESC LIMIT 100",
            "SUM + WHERE + GROUP BY + ORDER DESC LIMIT 100 (all)"
        ),
        # Different filter value
        (
            f"SELECT region, SUM(value) AS total FROM {TABLE} WHERE status = 'pending' GROUP BY region ORDER BY total ASC LIMIT 5",
            "SUM + WHERE='pending' + ORDER ASC LIMIT 5"
        ),
        # Float column
        (
            f"SELECT region, SUM(amount) AS total FROM {TABLE} WHERE status = 'active' GROUP BY region ORDER BY total DESC LIMIT 10",
            "SUM(float) + WHERE + GROUP BY + ORDER DESC LIMIT 10"
        ),
    ]

    for sql, label in queries:
        bench_query(db, sql, label)

    print("\nDone.")
    db.close()

    # Cleanup
    shutil.rmtree(DB_PATH, ignore_errors=True)


if __name__ == "__main__":
    main()
