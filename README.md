# ApexBase

**High-performance HTAP embedded database with Rust core and Python API**

ApexBase is an embedded columnar database designed for **Hybrid Transactional/Analytical Processing (HTAP)** workloads. It combines a high-throughput columnar storage engine written in Rust with an ergonomic Python API, delivering analytical query performance that surpasses DuckDB and SQLite on most benchmarks — all in a single `.apex` file with zero external dependencies.

---

## Features

- **HTAP architecture** — V4 Row Group columnar storage with DeltaStore for cell-level updates; fast inserts and fast analytical scans in one engine
- **Multi-database support** — multiple isolated databases in one directory; cross-database queries with standard `db.table` SQL syntax
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
- **PostgreSQL wire protocol** — built-in server for DBeaver, psql, DataGrip, pgAdmin, Navicat, and any PostgreSQL-compatible client; two distribution modes (Python CLI or standalone Rust binary)
- **Arrow Flight gRPC server** — high-performance columnar data transfer over HTTP/2; streams Arrow IPC RecordBatch directly, 4–7× faster than PG wire for large result sets; accessible via `pyarrow.flight`, Go arrow, Java arrow, and any Arrow Flight client
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

### Database Management

ApexBase supports multiple isolated databases within a single root directory. Each named database lives in its own subdirectory; the default database uses the root directory.

```python
# Switch to a named database (creates it if needed)
client.use_database("analytics")

# Combined: switch database + select/create a table in one call
client.use(database="analytics", table="events")

# List all databases
dbs = client.list_databases()  # ["analytics", "default", "hr"]

# Current database
print(client.current_database)  # "analytics"

# Cross-database SQL — standard db.table syntax
client.execute("SELECT * FROM default.users")
client.execute("SELECT u.name, e.event FROM default.users u JOIN analytics.events e ON u.id = e.user_id")
client.execute("INSERT INTO analytics.events (name) VALUES ('click')")
client.execute("UPDATE default.users SET age = 31 WHERE name = 'Alice'")
client.execute("DELETE FROM default.users WHERE age < 18")
```

All SQL operations (SELECT, INSERT, UPDATE, DELETE, JOIN, CREATE TABLE, DROP TABLE, ALTER TABLE) support `database.table` qualified names, allowing cross-database queries in a single statement.

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

ApexBase ships a native full-text search engine (NanoFTS) integrated directly into the SQL executor. FTS is available through **all interfaces** — Python API, PostgreSQL Wire, and Arrow Flight — without any Python-side middleware.

#### SQL interface (recommended)

```python
# 1. Create the FTS index via SQL DDL
client.execute("CREATE FTS INDEX ON articles (title, content)")

# Optional: specify lazy loading and cache size
client.execute("CREATE FTS INDEX ON logs WITH (lazy_load=true, cache_size=50000)")

# 2. Query using MATCH() / FUZZY_MATCH() in WHERE
results = client.execute("SELECT * FROM articles WHERE MATCH('rust programming')")
results = client.execute("SELECT title, content FROM articles WHERE FUZZY_MATCH('pytohn')")

# Combine with other predicates
results = client.execute("""
    SELECT * FROM articles
    WHERE MATCH('machine learning') AND published_at > '2024-01-01'
    ORDER BY _id DESC LIMIT 20
""")

# FTS also works in aggregations
count = client.execute("SELECT COUNT(*) FROM articles WHERE MATCH('deep learning')")

# Manage indexes
client.execute("SHOW FTS INDEXES")           # list all FTS-enabled tables
client.execute("ALTER FTS INDEX ON articles DISABLE")  # disable, keep files
client.execute("DROP FTS INDEX ON articles") # remove index + delete files
```

#### Python API (alternative)

```python
# Initialize FTS for current table
client.use_table("articles")
client.init_fts(index_fields=["title", "content"])

# Search
ids    = client.search_text("database")
fuzzy  = client.fuzzy_search_text("databse")   # tolerates typos
recs   = client.search_and_retrieve("python", limit=10)
top5   = client.search_and_retrieve_top("neural network", n=5)

# Lifecycle
client.get_fts_stats()
client.disable_fts()   # suspend without deleting files
client.drop_fts()      # remove index + delete files
```

> **Tip:** The SQL interface (`MATCH()` / `FUZZY_MATCH()`) works over PG Wire and Arrow Flight without any extra setup; the Python API methods are Python-process-only.

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

Three-way comparison on macOS 26.3, Apple arm (10 cores), 32 GB RAM.
Python 3.11.10, ApexBase v1.6.0, SQLite v3.45.3, DuckDB v1.1.3, PyArrow v19.0.0.

Dataset: 1,000,000 rows × 5 columns (name, age, score, city, category).
Average of 5 timed iterations after 2 warmup runs.

| Query | ApexBase | SQLite | DuckDB | vs Best Other |
|-------|----------|--------|--------|---------------|
| Bulk Insert (1M rows) | 340ms | 915ms | 898ms | **2.6x faster** |
| COUNT(\*) | 0.070ms | 9.29ms | 0.523ms | **7.5x faster** |
| SELECT \* LIMIT 100 [cold] | 0.042ms | 0.065ms | 0.259ms | **1.5x faster** |
| SELECT \* LIMIT 100 [warm] | 2.54µs | 0.064ms | 0.273ms | **25x faster** |
| SELECT \* LIMIT 10K [cold] | 0.801ms | 6.76ms | 4.80ms | **6x faster** |
| SELECT \* LIMIT 10K [warm] | 3.55µs | 6.96ms | 4.69ms | **>1000x faster** |
| Filter (name = 'user\_5000') | 0.046ms | 41.44ms | 1.64ms | **36x faster** |
| Filter (age BETWEEN 25 AND 35) | 0.041ms | 169ms | 96.75ms | **>2000x faster** |
| GROUP BY city (10 groups) | 0.032ms | 360ms | 3.87ms | **121x faster** |
| GROUP BY + HAVING | 0.032ms | 370ms | 4.23ms | **132x faster** |
| ORDER BY score LIMIT 100 | 0.032ms | 52.72ms | 8.52ms | **266x faster** |
| Aggregation (5 funcs) | 0.040ms | 85.26ms | 1.65ms | **41x faster** |
| Complex (Filter+Group+Order) | 0.032ms | 166ms | 3.18ms | **99x faster** |
| Point Lookup (by \_id) | 0.030ms | 0.053ms | 3.58ms | **1.8x faster** |
| Insert 1K rows | 0.640ms | 1.33ms | 2.86ms | **2.1x faster** |
| SELECT \* → pandas (full scan) | 0.744ms | 1210ms | 181ms | **243x faster** |
| GROUP BY city, category (100 grp) | 0.032ms | 722ms | 6.03ms | **188x faster** |
| LIKE filter (name LIKE 'user\_1%') | 33.70ms | 137ms | 60.23ms | **1.8x faster** |
| Multi-cond (age>30 AND score>50) | 0.037ms | 356ms | 212ms | **>5000x faster** |
| ORDER BY city, score DESC LIMIT 100 | 0.035ms | 71.03ms | 7.63ms | **218x faster** |
| COUNT(DISTINCT city) | 0.035ms | 92.58ms | 4.51ms | **129x faster** |
| IN filter (city IN 3 cities) | 0.038ms | 327ms | 161ms | **>4000x faster** |
| UPDATE rows (age = 25) | 8.51ms | 39.48ms | 17.17ms | **2.0x faster** |

**Summary**: wins all 23 of 23 benchmarks. "Cold" = fresh DB open per iteration; "warm" = cached backend.

Cold comparison is fair: all three engines measured without gc.collect() interference.

Reproduce: `python benchmarks/bench_vs_sqlite_duckdb.py --rows 1000000`

---

## Server Protocols

ApexBase ships two complementary server protocols for external access:

| Protocol | Port | Best for | Binary / CLI |
|----------|------|----------|--------------|
| **PG Wire** | 5432 | DBeaver, psql, DataGrip, BI tools | `apexbase-server` |
| **Arrow Flight** | 50051 | Python (pyarrow), Go, Java, Spark | `apexbase-flight` |

### Combined Launcher (Both Servers at Once)

```bash
# Start PG Wire + Arrow Flight simultaneously
apexbase-serve --dir /path/to/data

# Custom ports
apexbase-serve --dir /path/to/data --pg-port 5432 --flight-port 50051

# Disable one server
apexbase-serve --dir /path/to/data --no-flight   # PG Wire only
apexbase-serve --dir /path/to/data --no-pg       # Arrow Flight only
```

| Flag | Default | Description |
|------|---------|-------------|
| `--dir`, `-d` | `.` | Directory containing `.apex` database files |
| `--host` | `127.0.0.1` | Bind host for both servers |
| `--pg-port` | `5432` | PostgreSQL Wire port |
| `--flight-port` | `50051` | Arrow Flight gRPC port |
| `--no-pg` | — | Disable PG Wire server |
| `--no-flight` | — | Disable Arrow Flight server |

---

## PostgreSQL Wire Protocol Server

ApexBase includes a built-in PostgreSQL wire protocol server, allowing you to connect using **DBeaver**, **psql**, **DataGrip**, **pgAdmin**, **Navicat**, and any other tool that supports the PostgreSQL protocol.

### Starting the Server

**Method 1: Python CLI (after `pip install apexbase`)**

```bash
apexbase-server --dir /path/to/data --port 5432
```

Options:

| Flag | Default | Description |
|------|---------|-------------|
| `--dir`, `-d` | `.` | Directory containing `.apex` database files |
| `--host` | `127.0.0.1` | Host to bind to (use `0.0.0.0` for remote access) |
| `--port`, `-p` | `5432` | Port to listen on |

**Method 2: Standalone Rust binary (no Python required)**

```bash
# Build
cargo build --release --bin apexbase-server --no-default-features --features server

# Run
./target/release/apexbase-server --dir /path/to/data --port 5432
```

### Connecting with Database Tools

The server emulates PostgreSQL 15.0, reports a `pg_catalog` and `information_schema` compatible metadata layer, and supports `SimpleQuery` protocol. No username or password is required (authentication is disabled).

#### DBeaver

1. **New Database Connection** → choose **PostgreSQL**
2. Fill in connection details:
   - **Host**: `127.0.0.1` (or the `--host` you specified)
   - **Port**: `5432` (or the `--port` you specified)
   - **Database**: `apexbase` (any value accepted)
   - **Authentication**: select **No Authentication** or leave username/password empty
3. Click **Test Connection** → **Finish**
4. DBeaver will discover tables and columns automatically via `pg_catalog` / `information_schema`

#### psql

```bash
psql -h 127.0.0.1 -p 5432 -d apexbase
```

#### DataGrip / IntelliJ IDEA

1. **Database** tool window → **+** → **Data Source** → **PostgreSQL**
2. Set **Host**, **Port**, **Database** as above; leave **User** and **Password** empty
3. Click **Test Connection** → **OK**

#### pgAdmin

1. **Add New Server** → **General** tab: give it a name
2. **Connection** tab: set **Host** and **Port**; leave **Username** as `postgres` (ignored) and **Password** empty
3. **Save** — tables appear under **Databases > apexbase > Schemas > public > Tables**

#### Navicat for PostgreSQL

1. **Connection** → **PostgreSQL**
2. Set **Host**, **Port**; leave **User** and **Password** blank
3. **Test Connection** → **OK**

#### Other Compatible Tools

Any tool or library that speaks the PostgreSQL wire protocol (libpq) can connect, including:

- **TablePlus**, **Beekeeper Studio**, **Heidisql**
- **Python**: `psycopg2` / `asyncpg`
- **Node.js**: `pg` (`node-postgres`)
- **Go**: `pgx` / `lib/pq`
- **Rust**: `tokio-postgres` / `sqlx`
- **Java**: JDBC PostgreSQL driver

Example with `psycopg2`:

```python
import psycopg2

conn = psycopg2.connect(host="127.0.0.1", port=5432, dbname="apexbase")
cur = conn.cursor()
cur.execute("SELECT * FROM users LIMIT 10")
print(cur.fetchall())
conn.close()
```

### Supported SQL over Wire Protocol

The wire protocol server passes SQL directly to the ApexBase query engine. All SQL features listed in [Usage Guide](#usage-guide) are available, including JOINs, CTEs, window functions, transactions, and DDL.

### Metadata Compatibility

The server implements a `pg_catalog` compatibility layer that responds to common catalog queries:

| Catalog / View | Purpose |
|----------------|---------|
| `pg_catalog.pg_namespace` | Schema listing |
| `pg_catalog.pg_database` | Database listing |
| `pg_catalog.pg_class` | Table discovery |
| `pg_catalog.pg_attribute` | Column metadata |
| `pg_catalog.pg_type` | Type information |
| `pg_catalog.pg_settings` | Server settings |
| `information_schema.tables` | Standard table listing |
| `information_schema.columns` | Standard column listing |
| `SET` / `SHOW` statements | Client configuration probes |

This enables GUI tools to browse tables, inspect columns, and display data types without modification.

### Supported Protocol Features

| Feature | Status |
|---------|--------|
| Simple Query Protocol | ✅ Fully supported |
| Extended Query Protocol (prepared statements) | ✅ Supported — schema cached, binary format for psycopg3 |
| Cross-database SQL (`db.table`) | ✅ Supported — `USE dbname` / `\c dbname` to switch context |
| `pg_catalog` / `information_schema` | ✅ Compatible layer for GUI tools |
| All ApexBase SQL (JOINs, CTEs, window functions, DDL) | ✅ Full pass-through to query engine |

### Limitations

- **Authentication** is not implemented — the server accepts all connections regardless of username/password
- **SSL/TLS** is not supported — use an SSH tunnel (`ssh -L 5432:127.0.0.1:5432 user@host`) for remote access

---

## Arrow Flight gRPC Server

Arrow Flight sends Arrow IPC RecordBatch directly over gRPC (HTTP/2), bypassing per-row text serialization entirely. It is **4–7× faster than PG wire for large result sets** (10K+ rows).

| Query | PG Wire | Arrow Flight | Speedup |
|-------|---------|--------------|--------|
| SELECT 10K rows | 5.1ms | 0.7ms | **7× faster** |
| BETWEEN (~33K rows) | 22ms | 5.6ms | **4× faster** |
| Single row / point lookup | ~7.5ms | ~7.9ms | equal |

### Starting the Flight Server

**Python CLI:**

```bash
apexbase-flight --dir /path/to/data --port 50051
```

**Standalone Rust binary:**

```bash
cargo build --release --bin apexbase-flight --no-default-features --features flight
./target/release/apexbase-flight --dir /path/to/data --port 50051
```

### Python Client

```python
import pyarrow.flight as fl
import pandas as pd

client = fl.connect("grpc://127.0.0.1:50051")

# SELECT — returns Arrow Table
table = client.do_get(fl.Ticket(b"SELECT * FROM users LIMIT 10000")).read_all()
df = table.to_pandas()              # zero-copy to pandas
pl_df = pl.from_arrow(table)        # zero-copy to polars

# DML / DDL
client.do_action(fl.Action("sql", b"INSERT INTO users (name, age) VALUES ('Alice', 30)"))
client.do_action(fl.Action("sql", b"CREATE TABLE logs (event STRING, ts INT64)"))

# List available actions
for action in client.list_actions():
    print(action.type, "—", action.description)
```

### When to Use Arrow Flight vs PG Wire

| Scenario | Recommendation |
|----------|---------------|
| DBeaver / Tableau / BI tools | **PG Wire** (only option) |
| Python + small queries (<100 rows) | **Native API** (fastest, in-process) |
| Python + large queries (10K+ rows, remote) | **Arrow Flight** (4–7× faster than PG wire) |
| Go / Java / Spark workers | **Arrow Flight** (native Arrow support) |
| Local Python (same machine) | **Native API** (`ApexClient.execute()`) |

### PyO3 Python API

Both servers are also accessible as blocking Python functions (released GIL):

```python
import threading
from apexbase._core import start_pg_server, start_flight_server

t1 = threading.Thread(target=start_pg_server,     args=("/data", "0.0.0.0", 5432),  daemon=True)
t2 = threading.Thread(target=start_flight_server, args=("/data", "0.0.0.0", 50051), daemon=True)
t1.start()
t2.start()
```

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
  +-- PG Wire Protocol Server (pgwire)             |
  |   +-- DBeaver / psql / DataGrip / pgAdmin      |
  |   +-- pg_catalog & information_schema compat    |
  |                                                 |
  +-- Arrow Flight gRPC Server (tonic + HTTP/2)     |
      +-- pyarrow.flight / Go / Java / Spark        |
      +-- Arrow IPC — zero serialization overhead   |
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

**Database Management**

| Method | Description |
|--------|-------------|
| `use_database(database='default')` | Switch to a named database (creates it if needed) |
| `use(database='default', table=None)` | Switch database and optionally select/create a table |
| `list_databases()` | List all databases (`'default'` always included) |
| `current_database` | Property: current database name |

**Table Management**

| Method | Description |
|--------|-------------|
| `create_table(name, schema=None)` | Create a new table, optionally with pre-defined schema |
| `drop_table(name)` | Drop a table |
| `use_table(name)` | Switch active table |
| `list_tables()` | List all tables in the current database |
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
