# ApexBase Documentation

Complete documentation for ApexBase high-performance embedded database.

## Quick Links

| Document | Description |
|----------|-------------|
| [Quick Start](QUICK_START.md) | Get started in 5 minutes |
| [API Reference](API_REFERENCE.md) | Complete API documentation (100% coverage) |
| [Examples](EXAMPLES.md) | Code examples and use cases |
| [Root README](../README.md) | Project overview and installation |

## Installation

```bash
pip install apexbase
```

Or build from source:

```bash
# Using conda dev environment
conda activate dev
maturin develop --release
```

## Usage Overview

```python
from apexbase import ApexClient

# Create client (single .apex file storage)
client = ApexClient("./data")

# Store data
client.store({"name": "Alice", "age": 30})
client.store([{"name": "Bob", "age": 25}, {"name": "Charlie", "age": 35}])

# SQL query
results = client.execute("SELECT * FROM default WHERE age > 25")

# Convert to DataFrame
df = results.to_pandas()

# Close
client.close()
```

## Key Features

- **Single-file storage** - Custom `.apex` format, no external dependencies
- **SQL support** - Full SQL query with aggregations, GROUP BY, JOINs
- **DataFrame integration** - Native pandas, polars, PyArrow support
- **Full-text search** - Built-in FTS with fuzzy matching
- **High performance** - Rust core with zero-copy Python API

## API Coverage

This documentation covers 100% of the public Python API:

- **ApexClient** - All 50+ public methods
- **ResultView** - All conversion and access methods
- **Constants** - Module-level exports
- **SQL syntax** - Supported SQL operations

See [API_REFERENCE.md](API_REFERENCE.md) for complete details.

## Documentation Structure

```
docs/
├── README.md           # This file - documentation index
├── QUICK_START.md      # 5-minute quick start guide
├── API_REFERENCE.md    # Complete API reference (100% coverage)
└── EXAMPLES.md         # Real-world usage examples
```

## Development

```bash
# Activate environment
conda activate dev

# Run tests
python run_tests.py

# Or use pytest directly
pytest -q
```

## Version Info

| Component | Requirement |
|-----------|-------------|
| Python | 3.8+ |
| PyArrow | 14.0+ |
| pandas | 2.0+ (recommended) |
| polars | 0.20+ |

## Notes

- Primary API entry: `apexbase.ApexClient`
- Data persistence: Single `.apex` file per database directory
- Internal ID: Records have auto-increment `_id` field
- Query preference: Use `execute(sql)` for full SQL, `query(where)` for simple filters

## License

Apache-2.0
