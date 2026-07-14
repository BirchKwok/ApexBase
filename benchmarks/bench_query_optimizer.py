"""Bounded P0/P1 query-optimizer micro-benchmark.

Reports median and P95 planning/execution latency separately and exercises the
index materialization crossover across hit count, projection width, covering,
cache reopen, mmap base data, and appended delta rows.
"""

from __future__ import annotations

import argparse
import shutil
import statistics
import tempfile
import time
from pathlib import Path

from apexbase import ApexClient


def measure(call, iterations: int) -> tuple[float, float]:
    samples = []
    for _ in range(iterations):
        started = time.perf_counter_ns()
        call()
        samples.append((time.perf_counter_ns() - started) / 1_000_000)
    samples.sort()
    p95 = samples[min(len(samples) - 1, int(len(samples) * 0.95))]
    return statistics.median(samples), p95


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=100_000)
    parser.add_argument("--iterations", type=int, default=25)
    args = parser.parse_args()

    root = Path(tempfile.mkdtemp(prefix="apexbase_optimizer_bench_"))
    try:
        client = ApexClient(dirpath=str(root))
        client.create_table("fact")
        client.store(
            [
                {
                    "tenant": f"t{i % 100}",
                    "bucket": i % 1000,
                    "payload": f"payload_{i}",
                    "wide_a": i * 2,
                    "wide_b": i * 3,
                }
                for i in range(args.rows)
            ]
        )
        client.execute("CREATE INDEX idx_tenant ON fact (tenant)")
        client.execute("CREATE INDEX idx_bucket ON fact (bucket) USING BTREE")
        client.execute(
            "CREATE INDEX idx_tenant_bucket ON fact (tenant, bucket) USING BTREE"
        )
        client.execute("ANALYZE fact")

        # A post-ANALYZE append covers delta invalidation/materialization.
        client.store(
            [{"tenant": "delta", "bucket": i, "payload": f"d{i}", "wide_a": i, "wide_b": i}
             for i in range(64)]
        )

        planning = {
            "no_where": "EXPLAIN SELECT payload FROM fact",
            "single_where": "EXPLAIN SELECT payload FROM fact WHERE bucket = 42",
            "two_table_join": (
                "EXPLAIN SELECT f.payload FROM fact f "
                "INNER JOIN fact d ON f.bucket = d.bucket WHERE f.tenant = 't1'"
            ),
            "three_table_join": (
                "EXPLAIN SELECT f.payload FROM fact f "
                "INNER JOIN fact d ON f.bucket = d.bucket "
                "INNER JOIN fact e ON d.bucket = e.bucket WHERE f.tenant = 't1'"
            ),
            "explain_analyze": "EXPLAIN ANALYZE SELECT payload FROM fact WHERE bucket = -1",
        }
        execution = {
            "miss_narrow": "SELECT payload FROM fact WHERE bucket = -1",
            "one_row_covering": "SELECT bucket FROM fact WHERE bucket = 42 LIMIT 1",
            "one_row_wide": "SELECT payload, wide_a, wide_b FROM fact WHERE bucket = 42 LIMIT 1",
            "hundred_rows_narrow": "SELECT payload FROM fact WHERE bucket = 42",
            "prefix_equality": "SELECT payload FROM fact WHERE tenant = 't1'",
            "prefix_range": (
                "SELECT payload FROM fact WHERE tenant = 't1' "
                "AND bucket BETWEEN 100 AND 300"
            ),
            "and_intersection": "SELECT payload FROM fact WHERE tenant = 't1' AND bucket = 101",
            "or_union": "SELECT payload FROM fact WHERE bucket = 101 OR bucket = 102",
            "delta_hit": "SELECT payload FROM fact WHERE tenant = 'delta' AND bucket = 7",
        }

        print("metric,median_ms,p95_ms")
        for name, sql in planning.items():
            median, p95 = measure(lambda sql=sql: client.execute(sql), args.iterations)
            print(f"planning/{name},{median:.4f},{p95:.4f}")
        for name, sql in execution.items():
            median, p95 = measure(lambda sql=sql: client.execute(sql), args.iterations)
            print(f"execution/{name},{median:.4f},{p95:.4f}")

        client.close()
        cold_sql = "SELECT payload FROM fact WHERE bucket = 42"
        median, p95 = measure(
            lambda: _cold_query(root, cold_sql), max(5, args.iterations // 5)
        )
        print(f"execution/cold_reopen,{median:.4f},{p95:.4f}")
    finally:
        shutil.rmtree(root, ignore_errors=True)


def _cold_query(root: Path, sql: str) -> None:
    client = ApexClient(dirpath=str(root))
    try:
        client.use_table("fact")
        client.execute(sql)
    finally:
        client.close()


if __name__ == "__main__":
    main()
