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

# Create client (single .apex file storage)
client = ApexClient("./data")

# Store data (columnar batch — fastest path)
client.store({
    "name": ["Alice", "Bob", "Charlie"],
    "age": [30, 25, 35],
    "city": ["Beijing", "Shanghai", "Beijing"],
})

# SQL query
results = client.execute("SELECT * FROM default WHERE age > 25")

# Convert to DataFrame (zero-copy Arrow IPC)
df = results.to_pandas()

# Close
client.close()
```

## Key Features

- **HTAP architecture** — columnar V4 Row Group storage + delta writes for fast inserts
- **Single-file storage** — custom `.apex` format, no server, no external dependencies
- **Full SQL support** — DDL, DML, aggregations, GROUP BY, HAVING, ORDER BY, JOINs
- **DataFrame integration** — native Pandas / Polars / PyArrow support via zero-copy Arrow IPC
- **Full-text search** — built-in NanoFTS with fuzzy matching
- **JIT compilation** — Cranelift-based JIT for predicate evaluation
- **Durability** — configurable `fast` / `safe` / `max` with WAL support
- **Cross-platform** — Linux, macOS, Windows (x86_64 & ARM64)

## API Coverage

This documentation covers 100% of the public Python API:

- **ApexClient** — all 50+ public methods
- **ResultView** — all conversion and access methods
- **Constants** — module-level exports
- **SQL syntax** — supported SQL operations

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
```

## Development

```bash
conda activate dev

# Build + install
maturin develop --release

# Run tests
pytest test/ -q

# Run benchmarks
python benchmarks/bench_vs_sqlite_duckdb.py --rows 1000000
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
- Data persistence: single `.apex` file per table per database directory
- Internal ID: records have auto-increment `_id` field
- Query preference: use `execute(sql)` for full SQL, `query(where)` for simple filters

## License

Apache-2.0
