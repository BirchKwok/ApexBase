# ApexBase Documentation

Complete documentation for ApexBase — a high-performance HTAP embedded database with Rust core and Python API.

## Quick Links

| Document | Description |
|----------|-------------|
| [Quick Start](QUICK_START.md) | Get started in 5 minutes |
| [API Reference](API_REFERENCE.md) | Complete API documentation (100% coverage) |
| [Examples](EXAMPLES.md) | Code examples and use cases |
| [Storage Architecture](STORAGE_ARCHITECTURE.md) | V4 Row Group format, engine internals |
| [HTAP Roadmap](HTAP_ROADMAP.md) | Roadmap and current status |
| [Root README](../README.md) | Project overview, benchmarks, installation |

### Server Quick Reference

| CLI | Default | Description |
|-----|---------|-------------|
| `apexbase-serve` | pg=5432, flight=50051 | **Start both servers simultaneously** |
| `apexbase-server` | 5432 | PostgreSQL Wire only |
| `apexbase-flight` | 50051 | Arrow Flight gRPC only |

## Installation

```bash
# From PyPI (Linux, macOS, Windows — Python 3.9–3.13)
pip install apexbase
```

Or build from source:

```bash
conda activate dev
maturin develop --release
```

## Usage Overview

```python
from apexbase import ApexClient

# Create client
client = ApexClient("./data")

# Create a table (required before any data operations)
client.create_table("users")

# Store data (columnar batch — fastest path)
client.store({
    "name": ["Alice", "Bob", "Charlie"],
    "age": [30, 25, 35],
    "city": ["Beijing", "Shanghai", "Beijing"],
})

# SQL query (use your table name in FROM clause)
results = client.execute("SELECT * FROM users WHERE age > 25")

# Convert to DataFrame (zero-copy Arrow IPC)
df = results.to_pandas()

# Close
client.close()
```

## Key Features

- **HTAP architecture** — columnar V4 Row Group storage + delta writes for fast inserts
- **Multi-database support** — multiple isolated databases; cross-database queries with `db.table` SQL syntax
- **Single-file storage** — custom `.apex` format, no server, no external dependencies
- **Full SQL support** — DDL, DML, aggregations, GROUP BY, HAVING, ORDER BY, JOINs, cross-db queries
- **DataFrame integration** — native Pandas / Polars / PyArrow support via zero-copy Arrow IPC
- **Full-text search** — built-in NanoFTS with fuzzy matching
- **JIT compilation** — Cranelift-based JIT for predicate evaluation
- **Durability** — configurable `fast` / `safe` / `max` with WAL support
- **PostgreSQL wire protocol** — connect DBeaver, psql, DataGrip, pgAdmin, Navicat, and any libpq client
- **Arrow Flight gRPC** — 4–7× faster than PG wire for large result sets; native pyarrow.flight / Go / Java support
- **Cross-platform** — Linux, macOS, Windows (x86_64 & ARM64)

## API Coverage

This documentation covers 100% of the public Python API:

- **ApexClient** — all 50+ public methods including `use_database()`, `use()`, `list_databases()`
- **ResultView** — all conversion and access methods
- **Constants** — module-level exports
- **SQL syntax** — supported SQL operations including cross-database `db.table` syntax

See [API_REFERENCE.md](API_REFERENCE.md) for complete details.

## Documentation Structure

```
docs/
├── README.md                 # This file — documentation index
├── QUICK_START.md            # 5-minute quick start guide
├── API_REFERENCE.md          # Complete API reference (100% coverage)
├── EXAMPLES.md               # Real-world usage examples
├── STORAGE_ARCHITECTURE.md   # V4 Row Group format, engine design
└── HTAP_ROADMAP.md           # Roadmap and status

benchmarks/
├── bench_vs_sqlite_duckdb.py # Engine vs SQLite vs DuckDB
├── bench_pg_wire.py          # PG Wire protocol performance
└── bench_flight.py           # Arrow Flight vs PG Wire vs Direct API
```

## Server Launch

```bash
# Both servers simultaneously (recommended)
apexbase-serve --dir /path/to/data
apexbase-serve --dir /path/to/data --pg-port 5432 --flight-port 50051

# Individual servers
apexbase-server --dir /path/to/data --port 5432
apexbase-flight --dir /path/to/data --port 50051

# Connect (PG Wire)
psql -h 127.0.0.1 -p 5432 -d apexbase
python -c "import psycopg2; conn = psycopg2.connect(host='127.0.0.1', port=5432)"

# Connect (Arrow Flight)
python -c "
import pyarrow.flight as fl
client = fl.connect('grpc://127.0.0.1:50051')
df = client.do_get(fl.Ticket(b'SELECT * FROM t LIMIT 100')).read_all().to_pandas()
print(df)
"
```

## Development

```bash
conda activate dev

# Build + install (includes both server and flight features)
maturin develop --release

# Run tests
pytest test/ -q

# Benchmark: engine vs SQLite vs DuckDB
python benchmarks/bench_vs_sqlite_duckdb.py --rows 1000000

# Benchmark: Arrow Flight vs PG Wire vs Direct API
python benchmarks/bench_flight.py --rows 200000

# Benchmark: PG Wire protocol deep-dive
python benchmarks/bench_pg_wire.py --rows 200000
```

## Version Info

| Component | Requirement |
|-----------|-------------|
| Python | 3.9+ |
| PyArrow | 10.0+ |
| pandas | 2.0+ |
| polars | 0.15+ |

## Notes

- Primary API entry: `apexbase.ApexClient`
- Multi-database: use `use_database(name)` or `use(database=name, table=name)` to switch context
- Cross-database SQL: `SELECT * FROM db.table`, `JOIN db.table ON ...`, `INSERT INTO db.table ...`
- Storage layout: `root_dir/<table>.apex` for default db; `root_dir/<db>/<table>.apex` for named dbs
- Data persistence: single `.apex` file per table per database directory
- Internal ID: records have auto-increment `_id` field
- Query preference: use `execute(sql)` for full SQL, `query(where)` for simple filters
- **Server choice**: use `apexbase-serve` for the simplest deployment (both protocols at once); use individual commands for single-protocol deployments
- **Arrow Flight** requires `pyarrow>=10.0.0` (already a package dependency); no extra install needed
- **PG Wire auth**: authentication is disabled — all connections are accepted without credentials

## License

Apache-2.0
