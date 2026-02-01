# ApexBase

**High-performance embedded database with Rust core and Python API**

ApexBase is a high-performance embedded database powered by a Rust core, with a clean and ergonomic Python API.

## ✨ Features

- 🚀 **High performance** - Rust core with batch write throughput up to 970K+ ops/s
- 📦 **Single-file storage** - custom `.apex` file format with no external dependencies
- 🔍 **Full-text search** - NanoFTS integration with fuzzy search support
- 🐍 **Python-friendly** - clean API with Pandas/Polars/PyArrow integrations
- 💾 **Compact storage** - ~45% smaller on disk compared to traditional approaches

## 📦 Installation

```bash
# Install from PyPI
pip install apexbase

# Build from source (recommended in the conda dev environment)
# conda activate dev
maturin develop --release
```

## 🚀 Quick Start

```python
from apexbase import ApexClient

# Create a client
client = ApexClient("./data")

# Store data
client.store({"name": "Alice", "age": 30, "city": "Beijing"})
client.store([
    {"name": "Bob", "age": 25},
    {"name": "Charlie", "age": 35}
])

# SQL query (recommended)
results = client.execute("SELECT * FROM default WHERE age > 28")

# You can also pass a WHERE expression (compatibility mode)
results2 = client.query("age > 28", limit=100)

# Retrieve by _id (_id is an internal auto-increment ID)
record = client.retrieve(0)
all_data = client.retrieve_all()

# Full-text search
client.init_fts(index_fields=["name", "city"], lazy_load=True)
doc_ids = client.search_text("Alice")
records = client.search_and_retrieve("Beijing")

# Convert to DataFrame
df = results.to_pandas()
pl_df = results.to_polars()

# Close
client.close()
```

## 📊 Performance Comparison

| Operation | ApexBase (Rust) | Baseline | Speedup |
|------|-----------------|----------|------|
| Batch write (10K) | 17ms | 57ms | **3.3x** |
| Single read | 0.01ms | 0.4ms | **40x** |
| Batch read (100) | 0.08ms | 1.1ms | **14x** |
| Storage size | 2.1 MB | 3.9 MB | **1.8x smaller** |

## 📁 Project Structure

```
ApexBase/
├── apexbase/                    # main package
│   ├── src/                     # Rust source
│   │   ├── storage/             # storage engine
│   │   ├── table/               # table management
│   │   ├── query/               # query executor
│   │   ├── index/               # B-tree index
│   │   ├── cache/               # LRU cache
│   │   ├── data/                # data types
│   │   └── python/              # PyO3 bindings
│   ├── python/                  # Python wrapper
│   │   └── apexbase/
│   │       └── __init__.py      # Python API
│   ├── Cargo.toml
│   └── pyproject.toml
├── Cargo.toml                   # workspace config
└── pyproject.toml               # project config
```

## 🔧 API Reference

### ApexClient

```python
# Initialization
client = ApexClient(
    dirpath="./data",           # data directory
    drop_if_exists=False,       # whether to delete existing data
    batch_size=1000,
    enable_cache=True,
    cache_size=10000,
    prefer_arrow_format=True,
    durability="fast",         # fast | safe | max
)

# Table operations
client.create_table("users")
client.use_table("users")
client.drop_table("users")
tables = client.list_tables()

# CRUD operations
client.store({"key": "value"})
client.store([{...}, {...}])
record = client.retrieve(0)
records = client.retrieve_many([1, 2, 3])
client.replace(0, {"new": "data"})
client.delete(0)
client.delete([1, 2, 3])

# Query
results = client.query("age > 30")
results = client.query("name LIKE 'A%'")
results = client.execute("SELECT name, age FROM default ORDER BY age DESC LIMIT 10")
count = client.count_rows()

# Full-text search
client.init_fts(index_fields=["title", "content"], lazy_load=True)
ids = client.search_text("keyword")
ids = client.fuzzy_search_text("keywrd")  # fuzzy search
records = client.search_and_retrieve("keyword")

# DataFrame integrations
client.from_pandas(df)
client.from_polars(df)
results.to_pandas()
results.to_polars()
results.to_arrow()
```

## 🧪 Development & Testing

```bash
# Run tests (recommended in the conda dev environment)
# conda activate dev
python run_tests.py

# Or run pytest directly
pytest -q
```

## 📦 Release Process (GitHub Actions)

This repository provides a tag-based automated build and release workflow. When you push a `v*` tag, CI runs tests, builds wheels/sdist, and publishes to PyPI via `twine`.

- **Workflow**: `.github/workflows/build_release.yml`
- **Tag**: format like `v0.4.0`
- **Secret**: `PYPI_API_TOKEN`

## 📚 Documentation

Documentation entry point: `docs/README.md`

## 📄 License

Apache-2.0
