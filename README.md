# ApexBase

**High-performance HTAP embedded database with Rust core and Python API**

ApexBase is an embedded columnar database designed for **Hybrid Transactional/Analytical Processing (HTAP)** workloads. It combines a high-throughput columnar storage engine written in Rust with an ergonomic Python API, delivering analytical query performance that surpasses DuckDB and SQLite on most benchmarks — all in a single `.apex` file with zero external dependencies.

---

## Features

- **HTAP architecture** — V4 Row Group columnar storage with DeltaStore for cell-level updates; fast inserts and fast analytical scans in one engine
- **Single-file storage** — custom `.apex` format per table, no server process, no external dependencies
- **Comprehensive SQL** — DDL, DML, JOINs (INNER/LEFT/RIGHT/FULL/CROSS), subqueries (IN/EXISTS/scalar), CTEs (WITH ... AS), UNION/UNION ALL, window functions, EXPLAIN/ANALYZE, multi-statement execution
- **70+ built-in functions** — math (ABS, SQRT, POWER, LOG, trig), string (UPPER, LOWER, SUBSTR, REPLACE, CONCAT, REGEXP_REPLACE, ...), date (YEAR, MONTH, DAY, DATEDIFF, DATE_ADD, ...), conditional (COALESCE, IFNULL, NULLIF, CASE WHEN, GREATEST, LEAST)
- **Aggregation and analytics** — COUNT, SUM, AVG, MIN, MAX, COUNT(DISTINCT), GROUP BY, HAVING, ORDER BY with NULLS FIRST/LAST
- **Window functions** — ROW_NUMBER, RANK, DENSE_RANK, NTILE, PERCENT_RANK, CUME_DIST, LAG, LEAD, FIRST_VALUE, LAST_VALUE, NTH_VALUE, RUNNING_SUM, and windowed SUM/AVG/COUNT/MIN/MAX with PARTITION BY and ORDER BY
- **Transactions** — BEGIN / COMMIT / ROLLBACK with OCC (Optimistic Concurrency Control), SAVEPOINT / ROLLBACK TO / RELEASE, statement-level auto-rollback
- **MVCC** — multi-version concurrency control with snapshot isolation, version store, and garbage collection
- **Indexing** — B-Tree and Hash indexes with CREATE INDEX / DROP INDEX / REINDEX; automatic multi-index AND intersection for compound predicates
- **Full-text search** — built-in NanoFTS integration with fuzzy matching
- **JIT compilation** — Cranelift-based JIT for predicate evaluation and SIMD-vectorized aggregations
- **Zero-copy Python bridge** — Arrow IPC between Rust and Python; direct conversion to Pandas, Polars, and PyArrow
- **Durability levels** — configurable `fast` / `safe` / `max` with WAL support and crash recovery
- **Compact storage** — dictionary encoding for low-cardinality strings, LZ4 and Zstd compression
- **Parquet interop** — COPY TO / COPY FROM Parquet files
- **Cross-platform** — Linux, macOS, and Windows; x86_64 and ARM64; Python 3.9 -- 3.13

---

## Installation

```bash
pip install apexbase
```

Build from source (requires Rust toolchain):

```bash
maturin develop --release
```

---

## Quick Start

```python
from apexbase import ApexClient

# Open (or create) a database directory
client = ApexClient("./data")

# Create a table
client.create_table("users")

# Store records
client.store({"name": "Alice", "age": 30, "city": "Beijing"})
client.store([
    {"name": "Bob", "age": 25, "city": "Shanghai"},
    {"name": "Charlie", "age": 35, "city": "Beijing"},
])

# SQL query
results = client.execute("SELECT * FROM users WHERE age > 28 ORDER BY age DESC")

# Convert to DataFrame
df = results.to_pandas()

client.close()
```

---

## Usage Guide

### Table Management

Each table is stored as a separate `.apex` file. Tables must be created before use.

```python
# Create with optional schema
client.create_table("orders", schema={
    "order_id": "int64",
    "product": "string",
    "price": "float64",
})

# Switch tables
client.use_table("users")

# List / drop
tables = client.list_tables()
client.drop_table("orders")
```

### Data Ingestion

```python
import pandas as pd
import polars as pl
import pyarrow as pa

# Columnar dict (fastest for bulk data)
client.store({
    "name": ["D", "E", "F"],
    "age": [22, 32, 42],
})

# From pandas / polars / PyArrow (auto-creates table when table_name given)
client.from_pandas(pd.DataFrame({"name": ["G"], "age": [28]}), table_name="users")
client.from_polars(pl.DataFrame({"name": ["H"], "age": [38]}), table_name="users")
client.from_pyarrow(pa.table({"name": ["I"], "age": [48]}), table_name="users")
```

### SQL

ApexBase supports a broad SQL dialect. Examples:

```python
# DDL
client.execute("CREATE TABLE IF NOT EXISTS products")
client.execute("ALTER TABLE products ADD COLUMN name STRING")
client.execute("DROP TABLE IF EXISTS products")

# DML
client.execute("INSERT INTO users (name, age) VALUES ('Zoe', 29)")
client.execute("UPDATE users SET age = 31 WHERE name = 'Alice'")
client.execute("DELETE FROM users WHERE age < 20")

# SELECT with full clause support
client.execute("""
    SELECT city, COUNT(*) AS cnt, AVG(age) AS avg_age
    FROM users
    WHERE age BETWEEN 20 AND 40
    GROUP BY city
    HAVING cnt > 1
    ORDER BY avg_age DESC
    LIMIT 10
""")

# JOINs
client.execute("""
    SELECT u.name, o.product
    FROM users u
    INNER JOIN orders o ON u._id = o.user_id
""")

# Subqueries
client.execute("SELECT * FROM users WHERE age > (SELECT AVG(age) FROM users)")
client.execute("SELECT * FROM users WHERE city IN (SELECT city FROM cities WHERE pop > 1000000)")

# CTEs
client.execute("""
    WITH seniors AS (SELECT * FROM users WHERE age >= 30)
    SELECT city, COUNT(*) FROM seniors GROUP BY city
""")

# Window functions
client.execute("""
    SELECT name, age,
           ROW_NUMBER() OVER (ORDER BY age DESC) AS rank,
           AVG(age) OVER (PARTITION BY city) AS city_avg
    FROM users
""")

# UNION
client.execute("""
    SELECT name FROM users WHERE city = 'Beijing'
    UNION ALL
    SELECT name FROM users WHERE city = 'Shanghai'
""")

# Multi-statement
client.execute("""
    INSERT INTO users (name, age) VALUES ('New1', 20);
    INSERT INTO users (name, age) VALUES ('New2', 21);
    SELECT COUNT(*) FROM users
""")

# INSERT ... ON CONFLICT (upsert)
client.execute("""
    INSERT INTO users (name, age) VALUES ('Alice', 31)
    ON CONFLICT (name) DO UPDATE SET age = 31
""")

# CREATE TABLE AS
client.execute("CREATE TABLE seniors AS SELECT * FROM users WHERE age >= 30")

# EXPLAIN / EXPLAIN ANALYZE
client.execute("EXPLAIN SELECT * FROM users WHERE age > 25")

# Parquet interop
client.execute("COPY users TO '/tmp/users.parquet'")
client.execute("COPY users FROM '/tmp/users.parquet'")
```

### Transactions

```python
client.execute("BEGIN")
client.execute("INSERT INTO users (name, age) VALUES ('Tx1', 20)")
client.execute("SAVEPOINT sp1")
client.execute("INSERT INTO users (name, age) VALUES ('Tx2', 21)")
client.execute("ROLLBACK TO sp1")   # undo Tx2 only
client.execute("COMMIT")            # Tx1 persisted
```

Transactions use OCC validation — concurrent writes are detected at commit time.

### Indexes

```python
client.execute("CREATE INDEX idx_age ON users (age)")
client.execute("CREATE UNIQUE INDEX idx_name ON users (name)")

# Queries automatically use indexes when applicable
client.execute("SELECT * FROM users WHERE age = 30")  # index scan

client.execute("DROP INDEX idx_age ON users")
client.execute("REINDEX users")
```

### Full-Text Search

```python
client.init_fts(index_fields=["name", "city"])

ids = client.search_text("Alice")
records = client.search_and_retrieve("Beijing", limit=10)
fuzzy = client.fuzzy_search_text("Alic")  # tolerates typos

client.get_fts_stats()
client.drop_fts()
```

### Record-Level Operations

```python
record = client.retrieve(0)               # by internal _id
records = client.retrieve_many([0, 1, 2])
all_data = client.retrieve_all()

client.replace(0, {"name": "Alice2", "age": 31})
client.delete(0)
client.delete([1, 2, 3])
```

### Column Operations

```python
client.add_column("email", "String")
client.rename_column("email", "email_addr")
client.drop_column("email_addr")
client.get_column_dtype("age")    # "Int64"
client.list_fields()              # ["name", "age", "city"]
```

### ResultView

Query results are returned as `ResultView` objects with multiple output formats:

```python
results = client.execute("SELECT * FROM users")

df = results.to_pandas()       # pandas DataFrame (zero-copy by default)
pl_df = results.to_polars()    # polars DataFrame
arrow = results.to_arrow()     # PyArrow Table
dicts = results.to_dict()      # list of dicts

results.shape                  # (rows, columns)
results.columns                # column names
len(results)                   # row count
results.first()                # first row as dict
results.scalar()               # single value (for aggregates)
results.get_ids()              # numpy array of _id values
```

### Context Manager

```python
with ApexClient("./data") as client:
    client.create_table("tmp")
    client.store({"key": "value"})
    # Automatically closed on exit
```

---

## Performance

### ApexBase vs SQLite vs DuckDB (1M rows)

Three-way comparison on macOS 26.2, Apple M1 Pro (10 cores), 32 GB RAM.
Python 3.11.10, ApexBase v0.6.0, SQLite v3.45.3, DuckDB v1.1.3, PyArrow 19.0.0.

Dataset: 1,000,000 rows x 5 columns (name, age, score, city, category).
Average of 5 timed iterations after 2 warmup runs.

| Query | ApexBase | SQLite | DuckDB | vs Best Other |
|-------|----------|--------|--------|---------------|
| Bulk Insert (1M rows) | 357ms | 976ms | 927ms | 2.6x faster |
| COUNT(\*) | 0.068ms | 9.05ms | 0.49ms | 7.2x faster |
| SELECT \* LIMIT 100 | 0.13ms | 0.12ms | 0.50ms | ~tied |
| SELECT \* LIMIT 10K | 0.031ms | 7.46ms | 5.27ms | 170x faster |
| Filter (string =) | 0.020ms | 53.6ms | 1.73ms | 87x faster |
| Filter (BETWEEN) | 0.018ms | 191ms | 94.7ms | 5300x faster |
| GROUP BY (10 groups) | 0.026ms | 358ms | 3.70ms | 142x faster |
| GROUP BY + HAVING | 0.030ms | 439ms | 4.40ms | 147x faster |
| ORDER BY + LIMIT | 0.027ms | 67.4ms | 38.7ms | 1400x faster |
| Aggregation (5 funcs) | 0.48ms | 85.9ms | 1.59ms | 3.3x faster |
| Complex (Filter+Group+Order) | 0.029ms | 175ms | 3.59ms | 124x faster |
| Point Lookup (by ID) | 0.39ms | 0.050ms | 4.29ms | 7.9x slower |
| Insert 1K rows | 1.01ms | 1.45ms | 2.95ms | 1.4x faster |

**Summary**: wins 11 of 13 benchmarks, ties 1. No metric loses to both competitors simultaneously (Point Lookup only trails SQLite while beating DuckDB 11x).

Reproduce: `python benchmarks/bench_vs_sqlite_duckdb.py --rows 1000000`

---

## Architecture

```
Python (ApexClient)
  |
  |-- Arrow IPC / columnar dict --------> ResultView (Pandas / Polars / PyArrow)
  |
Rust Core (PyO3 bindings)
  |
  +-- SQL Parser -----> Query Planner -----> Query Executor
  |                                              |
  |   +-- JIT Compiler (Cranelift)               |
  |   +-- Expression Evaluator (70+ functions)   |
  |   +-- Window Function Engine                 |
  |                                              |
  +-- Storage Engine                             |
  |     +-- V4 Row Group Format (.apex)          |
  |     +-- DeltaStore (cell-level updates)      |
  |     +-- WAL (write-ahead log)                |
  |     +-- Mmap on-demand reads                 |
  |     +-- LZ4 / Zstd compression              |
  |     +-- Dictionary encoding                  |
  |                                              |
  +-- Index Manager (B-Tree, Hash)               |
  +-- TxnManager (OCC + MVCC)                    |
  +-- NanoFTS (full-text search)                  |
```

### Storage Format

ApexBase uses a custom V4 Row Group format:

- Each table is a single `.apex` file containing a header, row groups, and a footer
- Row groups store columns contiguously with per-column compression (LZ4 or Zstd)
- Low-cardinality string columns are dictionary-encoded on disk
- Null bitmaps are stored per column per row group
- A DeltaStore file (`.deltastore`) holds cell-level updates that are merged on read and compacted automatically
- WAL records provide crash recovery with idempotent replay

### Query Execution

- The SQL parser produces an AST that the query planner analyzes for optimization strategy
- Fast paths bypass the full executor for common patterns (COUNT(\*), SELECT \* LIMIT N, point lookups, single-column GROUP BY)
- Arrow RecordBatch is the internal data representation; results flow to Python via Arrow IPC with zero-copy when possible
- Repeated identical read queries are served from an in-process result cache

---

## API Reference

### ApexClient

**Constructor**

```python
ApexClient(
    dirpath="./data",           # data directory
    drop_if_exists=False,       # clear existing data on open
    batch_size=1000,            # batch size for operations
    enable_cache=True,          # enable query cache
    cache_size=10000,           # cache capacity
    prefer_arrow_format=True,   # prefer Arrow format for results
    durability="fast",          # "fast" | "safe" | "max"
)
```

**Table Management**

| Method | Description |
|--------|-------------|
| `create_table(name, schema=None)` | Create a new table, optionally with pre-defined schema |
| `drop_table(name)` | Drop a table |
| `use_table(name)` | Switch active table |
| `list_tables()` | List all tables |
| `current_table` | Property: current table name |

**Data Storage**

| Method | Description |
|--------|-------------|
| `store(data)` | Store data (dict, list, DataFrame, Arrow Table) |
| `from_pandas(df, table_name=None)` | Import from pandas DataFrame |
| `from_polars(df, table_name=None)` | Import from polars DataFrame |
| `from_pyarrow(table, table_name=None)` | Import from PyArrow Table |

**Data Retrieval**

| Method | Description |
|--------|-------------|
| `execute(sql)` | Execute SQL statement(s) |
| `query(where, limit)` | Query with WHERE expression |
| `retrieve(id)` | Get record by \_id |
| `retrieve_many(ids)` | Get multiple records by \_id |
| `retrieve_all()` | Get all records |
| `count_rows(table)` | Count rows in table |

**Data Modification**

| Method | Description |
|--------|-------------|
| `replace(id, data)` | Replace a record |
| `batch_replace({id: data})` | Batch replace records |
| `delete(id)` or `delete([ids])` | Delete record(s) |

**Column Operations**

| Method | Description |
|--------|-------------|
| `add_column(name, type)` | Add a column |
| `drop_column(name)` | Drop a column |
| `rename_column(old, new)` | Rename a column |
| `get_column_dtype(name)` | Get column data type |
| `list_fields()` | List all fields |

**Full-Text Search**

| Method | Description |
|--------|-------------|
| `init_fts(fields, lazy_load, cache_size)` | Initialize FTS |
| `search_text(query)` | Search documents |
| `fuzzy_search_text(query)` | Fuzzy search |
| `search_and_retrieve(query, limit, offset)` | Search and return records |
| `search_and_retrieve_top(query, n)` | Top N results |
| `get_fts_stats()` | FTS statistics |
| `disable_fts()` / `drop_fts()` | Disable or drop FTS |

**Utility**

| Method | Description |
|--------|-------------|
| `flush()` | Flush data to disk |
| `set_auto_flush(rows, bytes)` | Set auto-flush thresholds |
| `get_auto_flush()` | Get auto-flush config |
| `estimate_memory_bytes()` | Estimate memory usage |
| `close()` | Close the client |

### ResultView

| Method / Property | Description |
|-------------------|-------------|
| `to_pandas(zero_copy=True)` | Convert to pandas DataFrame |
| `to_polars()` | Convert to polars DataFrame |
| `to_arrow()` | Convert to PyArrow Table |
| `to_dict()` | Convert to list of dicts |
| `scalar()` | Get single scalar value |
| `first()` | Get first row as dict |
| `get_ids(return_list=False)` | Get record IDs |
| `shape` | (rows, columns) |
| `columns` | Column names |
| `__len__()` | Row count |
| `__iter__()` | Iterate over rows |
| `__getitem__(idx)` | Index access |

---

## Documentation

Additional documentation is available in the `docs/` directory.

## License

Apache-2.0
