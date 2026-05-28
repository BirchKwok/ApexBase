# Performance

This page tracks the latest verified local benchmark snapshot rather than an old best-case run. Benchmarks are meant to be reproducible, not magical; always rerun them on your own workload and hardware.

## Latest Verified Snapshot

- **System**: macOS 26.4.1, Apple arm (10 cores), 32 GB RAM
- **Stack**: Python 3.12.4, ApexBase 1.18.0, SQLite 3.45.3, DuckDB 1.1.3, PyArrow 23.0.1
- **Dataset**: 200,000 rows x 5 columns (`name`, `age`, `score`, `city`, `category`)
- **Vector dataset**: 200,000 vectors x dim=128, `k=10`, batch size 10 queries
- **Method**: 2 warmup iterations + 3 timed iterations
- **Layout**: 92 named metrics total (37 OLAP, 46 OLTP, 9 vector)
- **Fairness rule**: only the default fair OLAP/OLTP cross-engine tables count toward the `38/38` win/loss summary. Vector similarity uses a separate dataset and its own ApexBase-vs-DuckDB scoreboard.

## Scoreboard

| Scope | Metrics | Apex wins | Ties | Slower |
| --- | ---: | ---: | ---: | ---: |
| Default fair (OLAP + OLTP) | 38 | 38 | 0 | 0 |
| OLAP fair | 29 | 29 | 0 | 0 |
| OLTP fair | 9 | 9 | 0 | 0 |
| Vector similarity (ApexBase vs DuckDB) | 6 | 6 | 0 | 0 |

Stock SQLite is not ranked in the vector table because the built-in `sqlite3` used here has no native vector distance/top-k functions in this harness.

## Representative OLAP Gaps

| Metric | ApexBase | SQLite | DuckDB | Gap to best other |
| --- | ---: | ---: | ---: | --- |
| COUNT(*) | 0.106 ms | 1.775 ms | 0.397 ms | 3.7x faster vs DuckDB |
| SELECT * LIMIT 100 (warm cache) | 6 us | 0.107 ms | 0.236 ms | 17.8x faster vs SQLite |
| Filtered LIMIT 100 (age>30) | 0.050 ms | 0.173 ms | 0.603 ms | 3.5x faster vs SQLite |
| GROUP BY city (10 groups) | 0.060 ms | 60.108 ms | 2.399 ms | 40.0x faster vs DuckDB |
| Window ROW_NUMBER PARTITION BY city | 0.622 ms | 99.086 ms | 12.809 ms | 20.6x faster vs DuckDB |

## Representative OLTP Gaps

| Metric | ApexBase | SQLite | DuckDB | Gap to best other |
| --- | ---: | ---: | ---: | --- |
| Bulk Insert (N rows; default fair) | 53.948 ms | 197.464 ms | 35.84 s | 3.7x faster vs SQLite |
| Point Lookup (SQL by ID) | 0.035 ms | 0.067 ms | 2.198 ms | 1.9x faster vs SQLite |
| Retrieve Many (SQL, 100 IDs) | 0.175 ms | 0.317 ms | 3.942 ms | 1.8x faster vs SQLite |
| FTS Index Build (name,city,category) | 103.738 ms | 246.588 ms | 790.703 ms | 2.4x faster vs SQLite |
| FTS Search ('Electronics') | 0.160 ms | 5.700 ms | 14.644 ms | 35.6x faster vs SQLite |

## Representative Vector Gaps

SQLite is excluded here because stock `sqlite3` in this harness has no native vector distance/top-k support.

| Metric | ApexBase | DuckDB | Gap to DuckDB |
| --- | ---: | ---: | --- |
| TopK L2 | 3.58 ms | 26.46 ms | 7.4x faster |
| TopK Cosine | 3.80 ms | 31.89 ms | 8.4x faster |
| TopK Dot | 3.48 ms | 26.37 ms | 7.6x faster |
| Batch TopK L2 (10 queries) | 23.18 ms | 266.92 ms | 11.5x faster |
| Batch TopK Cosine (10 queries) | 23.24 ms | 322.07 ms | 13.9x faster |
| Batch TopK Dot (10 queries) | 21.12 ms | 268.44 ms | 12.7x faster |

## Throughput Snapshot

Q/s uses a mixed analytical profile: `COUNT(*)`, two `GROUP BY` scans, and `Filtered LIMIT 100`, all materialized to Python rows.

| Throughput metric | ApexBase | SQLite | DuckDB | Gap to best other |
| --- | ---: | ---: | ---: | --- |
| OLAP Q/s (single thread) | 123,700.3 | 34.8 | 942.2 | 131.3x higher vs DuckDB |
| OLAP Q/s (4 threads) | 125,196.3 | 126.6 | 2,776.8 | 45.1x higher vs DuckDB |

## Hot-Path Latency Snapshot

These tables are not part of the `38/38` fair scoreboard. They answer a different question: how fast is the already-loaded hot path, and what happens when durability or transaction semantics are made explicit?

### Default Microbenchmarks

| Metric | ApexBase | SQLite | DuckDB | Gap to best other |
| --- | ---: | ---: | ---: | --- |
| COUNT(*) (direct API) | 7.43 us | 1.243 ms | 0.132 ms | 17.8x faster vs DuckDB |
| Point lookup (projected SQL) | 2.12 us | 2.99 us | 1.722 ms | 1.4x faster vs SQLite |
| Retrieve 100 IDs (projected SQL) | 0.041 ms | 0.099 ms | 3.438 ms | 2.4x faster vs SQLite |
| Insert 1 row (default fair) | 0.010 ms | 0.014 ms | 0.297 ms | 1.4x faster vs SQLite |
| UPDATE by ID | 1.13 us | 4.23 us | 0.483 ms | 3.7x faster vs SQLite |
| DELETE missing ID | 2.72 us | 3.81 us | 1.160 ms | 1.4x faster vs SQLite |

### Durable Fair Microbenchmarks

| Metric | ApexBase | SQLite | DuckDB | Gap to best other |
| --- | ---: | ---: | ---: | --- |
| Insert 1 row (durable fair) | 0.101 ms | 0.126 ms | 31.106 ms | 1.2x faster vs SQLite |
| UPDATE by ID (durable fair) | 2.02 us | 6.53 us | 4.469 ms | 3.2x faster vs SQLite |

### Transaction Fair Microbenchmarks

| Metric | ApexBase | SQLite | DuckDB | Gap to best other |
| --- | ---: | ---: | ---: | --- |
| TXN empty (BEGIN+COMMIT; durable sync) | 3.29 us | 4.08 us | 0.148 ms | 1.2x faster vs SQLite |
| TXN read COUNT(*) (COMMIT; durable sync) | 0.018 ms | 1.295 ms | 0.294 ms | 16.3x faster vs DuckDB |
| TXN backlog string miss (COMMIT; 1500 preseed; durable sync) | 0.051 ms | 8.011 ms | 0.404 ms | 7.9x faster vs DuckDB |
| TXN backlog COUNT(*) (COMMIT; 1500 preseed; durable sync) | 0.030 ms | 3.793 ms | 0.305 ms | 10.2x faster vs DuckDB |
| TXN backlog INSERT+read-own-name (COMMIT; 1500 preseed; durable sync) | 0.262 ms | 8.073 ms | 33.522 ms | 30.8x faster vs SQLite |

## OLTP Write Visibility

ApexBase exposes two fast single-row append paths, and the benchmark keeps them out of the fair scoreboard because their visibility rules are Apex-specific:

- **Memtable OLTP** is the default fast single-row path for schema-stable `store({...})` calls with `durability="fast"`. The writing client can read the row immediately, managed clients in the same Python process share the storage instance, and `flush()` / `close()` persists pending rows.
- **Buffered OLTP** is explicit: call `begin_buffered_writes()`, issue many single-row `store({...})` calls, then call `flush_buffered_writes()` or `end_buffered_writes(flush=True)`. Buffered rows are not visible until flushed.

That separation is deliberate: the fair tables compare committed cross-engine behavior, while Apex-only write modes remain visible as diagnostics instead of being mixed into the competitive summary.

## Reproduce

Use the same command as the snapshot above:

```bash
python benchmarks/bench_vs_sqlite_duckdb.py --rows 200000 --warmup 2 --iterations 3
```

Add `--skip-vector` if you want a tabular-only rerun without the separate vector module.

For a larger stress run, increase `--rows` to `1000000`.
