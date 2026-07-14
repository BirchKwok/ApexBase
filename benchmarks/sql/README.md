# Hive benchmark SQL dialects

The `hive_user_*.sql` files are the source workloads. ApexBase executes
those files verbatim, with only `${biz_date}` bound by the benchmark.

`duckdb/` contains full-workload native DuckDB rewrites. They preserve the Hive
CTEs, row generators, aggregations, joins, windows, and target-table writes.

SQLite does not natively implement Hive arrays, `LATERAL VIEW`, approximate
percentiles, grouping sets, cube/rollup, or multi-table insert. `sqlite/`
therefore contains the explicitly labeled portable output kernels used for the
three-way ApexBase/DuckDB/SQLite comparison. They exercise the shared user,
behavior, trade, refund, marketing, aggregation, join, window, scoring, and
ordering semantics without UDFs. They are not presented as full rewrites of
the full Hive jobs.

Regenerate all dialect artifacts after changing a source or kernel:

```bash
python benchmarks/generate_hive_native_sql.py
```

Both benchmarks reject stale generated SQL.
