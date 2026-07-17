# Performance

This page tracks the latest verified local benchmark snapshot rather than an old best-case run. Benchmarks are meant to be reproducible, not magical; always rerun them on your own workload and hardware.

## Latest Verified Snapshot

- **System**: macOS 26.5.2, Apple arm (10 cores), 32 GB RAM
- **Stack**: Python 3.12.2, ApexBase 1.23.0, SQLite 3.46.0, DuckDB 1.1.3, PyArrow 23.0.1
- **Dataset**: 1,000,000 rows x 5 columns (`name`, `age`, `score`, `city`, `category`)
- **Vector dataset**: 1,000,000 vectors x dim=128, `k=10`, batch size 10 queries
- **Method**: 2 warmup iterations + 5 timed iterations
- **Layout**: the default benchmark entrypoint tracks the README public scoreboard, including six competitive vector metrics; Apex-only vector metrics and other extended diagnostics live in `benchmarks/bench_vs_sqlite_duckdb_extended.py`.
- **Fairness rule**: only the default fair OLAP/OLTP cross-engine tables count toward the `72/72` win/loss summary. Vector similarity uses a separate dataset and its own ApexBase-vs-DuckDB scoreboard.

## Scoreboard

| Scope | Metrics | Apex wins | Ties | Slower |
| --- | ---: | ---: | ---: | ---: |
| Default fair (OLAP + OLTP) | 72 | 72 | 0 | 0 |
| OLAP fair | 45 | 45 | 0 | 0 |
| OLTP fair | 27 | 27 | 0 | 0 |
| Vector similarity (ApexBase vs DuckDB) | 6 | 6 | 0 | 0 |

Stock SQLite is not ranked in the vector table because the built-in `sqlite3` used here has no native vector distance/top-k functions in this harness.

## Representative OLAP Gaps

| Metric | ApexBase | SQLite | DuckDB | Gap to best other |
| --- | ---: | ---: | ---: | --- |
| COUNT(*) | 0.072 ms | 7.94 ms | 0.493 ms | 6.8x faster vs DuckDB |
| SELECT * LIMIT 100 (warm cache) | 0.075 ms | 0.124 ms | 0.244 ms | 1.7x faster vs SQLite |
| Filtered LIMIT 100 (age>30) | 0.014 ms | 0.127 ms | 0.298 ms | 9.1x faster vs SQLite |
| GROUP BY city (10 groups) | 1.51 ms | 357.54 ms | 3.89 ms | 2.6x faster vs DuckDB |
| Temp Table (CSV) Query (filter+agg) | 0.538 ms | N/A | 0.797 ms | 1.5x faster vs DuckDB |

## Representative OLTP Gaps

| Metric | ApexBase | SQLite | DuckDB | Gap to best other |
| --- | ---: | ---: | ---: | --- |
| Bulk Insert (N rows; default fair) | 225.05 ms | 1.02 s | 211.09 s | 4.5x faster vs SQLite |
| Point Lookup (SQL by ID) | 0.039 ms | 0.055 ms | 4.00 ms | 1.4x faster vs SQLite |
| Retrieve Many (SQL, 100 IDs) | 0.168 ms | 0.279 ms | 5.70 ms | 1.7x faster vs SQLite |
| FTS Index Build (name,city,category) | 1.38 ms | 1.54 s | 1.35 s | 978x faster vs DuckDB |
| FTS Search ('Electronics') | 5.45 ms | 29.91 ms | 24.34 ms | 4.5x faster vs DuckDB |

## Representative Vector Gaps

SQLite is excluded here because stock `sqlite3` in this harness has no native vector distance/top-k support.

Single-query rows compare one materialized TopK result from each engine. Batch rows compare ApexBase's `batch_topk_distance()` with ten DuckDB single-query SQL calls over the same deterministic query batch; every DuckDB result is materialized before the next query runs.

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
python benchmarks/bench_vs_sqlite_duckdb.py
```

Add `--skip-vector` if you want a tabular-only rerun without the separate vector module.
Run `python benchmarks/bench_vs_sqlite_duckdb_extended.py --rows 200000 --warmup 2 --iterations 3` for the file-format, materialization, Q/s, microbenchmark, durable, transaction, buffered/memtable, and full vector diagnostics.

### Out-of-core file comparison

Use the focused harness to compare ApexBase and DuckDB against the exact same
generated CSV or Parquet source:

```bash
python benchmarks/bench_out_of_core_import.py --rows 1000000 --format csv
python benchmarks/bench_out_of_core_import.py --rows 1000000 --format parquet
```

Each engine runs in an isolated process. The report separates direct file query
time, disk-backed table materialization time, repeated native-table query time,
incremental peak RSS, and storage size. Results and filtered row counts are
cross-checked before ratios are printed. DuckDB defaults to a `1GB` memory limit
and an explicit spill directory; change it with `--memory-limit 512MB`. Increase
`--rows` until the generated source exceeds physical memory for a true
out-of-core stress run. Ratios are reported as ApexBase divided by DuckDB, so a
value below `1.0x` is better for ApexBase.

On the same Apple Silicon development machine, the focused 1,000,000-row
Parquet run (21 measured queries after 5 warmups) produced this verification
snapshot:

| Metric | ApexBase | DuckDB | ApexBase / DuckDB |
| --- | ---: | ---: | ---: |
| Direct filtered Parquet count | 1.35 ms | 9.75 ms | 0.138x |
| Disk-backed materialization | 0.131 s | 0.278 s | 0.469x |
| Filter + GROUP BY + COUNT/AVG | 1.175 ms | 3.862 ms | 0.304x |
| Incremental peak RSS | 77.8 MB | 126.9 MB | 0.613x |
| Native storage size | 9.5 MiB | 6.5 MiB | 1.454x |

The native storage result first decreased from 33.9 MiB to 27.4 MiB by
selecting string dictionary encoding from its actual serialized size instead
of a periodic cardinality sample. One-, two-, or four-byte string dictionary
indices reduced it further to 22.6 MiB. Temp materializations now omit the
physical `_id` array when a row group's IDs are contiguous, reconstructing IDs
from `min_id` only in `.apex_tmp` files; this keeps mutable-table storage and
DML paths on their established format. Lossless low-cardinality `Float64`
dictionaries then reduce this workload to 9.5 MiB. The float encoding is used
only for at least 32K rows when sampling shows useful repetition and its
serialized size is below 70% of plain storage; high-cardinality data retains
the original encoding. The fused range-filter aggregation evaluates each
dictionary value once and scans compact row indices without materializing a
full float vector. Nullable inputs and more complex SQL shapes deliberately
fall back to the general executor.

Blob storage has a focused Lance comparison harness:

```bash
python benchmarks/bench_blob_lance.py --rows 200 --reads 200 --iterations 3
```

The script measures write throughput, non-blob projection scans, descriptor metadata reads, random full blob reads, random range reads, and projected blob materialization. It uses Lance Blob helpers when the installed Lance package exposes them, and reports unavailable or fallback modes explicitly.

For a larger stress run, increase `--rows` to `1000000`.
