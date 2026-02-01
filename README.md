# ApexBase

**High-performance embedded database with Rust core and Python API**

ApexBase is a high-performance embedded database powered by a Rust core, with a clean and ergonomic Python API.

## ✨ Features

- 🚀 **High performance** - Rust core with batch write throughput up to 970K+ ops/s
- 📦 **Single-file storage** - custom `.apex` file format with no external dependencies
- �️ **SQL DDL support** - CREATE TABLE, ALTER TABLE, DROP TABLE via standard SQL
- � **Full-text search** - NanoFTS integration with fuzzy search support
- 🐍 **Python-friendly** - clean API with Pandas/Polars/PyArrow integrations
- 💾 **Compact storage** - ~45% smaller on disk compared to traditional approaches

## 📦 Installation

```bash
# Install from PyPI
pip install apexbase

# Build from source
maturin develop --release
```

## 🚀 Quick Start

### Installation

```bash
pip install apexbase
```

### Basic Usage

```python
from apexbase import ApexClient

# Create a client (data stored in single .apex file)
client = ApexClient("./data")

# Store single record
client.store({"name": "Alice", "age": 30, "city": "Beijing"})

# Store multiple records
client.store([
    {"name": "Bob", "age": 25, "city": "Shanghai"},
    {"name": "Charlie", "age": 35, "city": "Beijing"}
])

# SQL query (recommended)
results = client.execute("SELECT * FROM default WHERE age > 28")

# Convert to DataFrame
df = results.to_pandas()

# Close client
client.close()
```

### Table Management

```python
# Create and switch tables
client.create_table("users")
client.use_table("users")

# List all tables
tables = client.list_tables()

# Drop table
client.drop_table("old_table")
```

### Data Operations

```python
# Store from various formats
import pandas as pd
import polars as pl
import pyarrow as pa

# From pandas DataFrame
df = pd.DataFrame({"name": ["A", "B"], "age": [20, 30]})
client.from_pandas(df)

# From polars DataFrame
df_pl = pl.DataFrame({"name": ["C", "D"], "age": [25, 35]})
client.from_polars(df_pl)

# From PyArrow Table
table = pa.table({"name": ["E", "F"], "age": [28, 38]})
client.from_pyarrow(table)

# Columnar storage (fastest for bulk data)
client.store({
    "name": ["G", "H", "I"],
    "age": [22, 32, 42]
})
```

### Query Operations

```python
# Full SQL support
results = client.execute("SELECT name, age FROM default WHERE age > 25 ORDER BY age DESC LIMIT 10")

# WHERE expression (compatibility mode)
results = client.query("age > 28")
results = client.query("name LIKE 'A%'")
results = client.query(where_clause="city = 'Beijing'", limit=100)

# Aggregation
agg = client.execute("SELECT COUNT(*), AVG(age), MAX(age) FROM default")
count = agg.scalar()  # Get single value

# Retrieve by _id (internal auto-increment ID)
record = client.retrieve(0)
records = client.retrieve_many([0, 1, 2])
all_data = client.retrieve_all()
```

### Column Operations

```python
# Add column
client.add_column("email", "String")

# Rename column
client.rename_column("email", "email_address")

# Drop column
client.drop_column("email_address")

# Get column type
dtype = client.get_column_dtype("age")

# List all fields
fields = client.list_fields()
```

### SQL DDL (Data Definition Language)

ApexBase supports full SQL DDL operations:

```python
# Create table via SQL
client.execute("CREATE TABLE employees")
client.execute("CREATE TABLE IF NOT EXISTS departments")  # No error if exists

# Add columns via SQL
client.execute("ALTER TABLE employees ADD COLUMN name STRING")
client.execute("ALTER TABLE employees ADD COLUMN age INT")

# Insert data via SQL
client.execute("INSERT INTO employees (name, age) VALUES ('Alice', 30)")
client.execute("INSERT INTO employees (name, age) VALUES ('Bob', 25), ('Charlie', 35)")

# Query the data
results = client.execute("SELECT * FROM employees WHERE age > 28")

# Drop table via SQL
client.execute("DROP TABLE employees")
client.execute("DROP TABLE IF EXISTS departments")  # No error if not exists
```

#### Multi-Statement SQL

You can execute multiple SQL statements in a single call by separating them with semicolons:

```python
# Execute multiple DDL statements at once
client.execute("""
    CREATE TABLE IF NOT EXISTS products;
    ALTER TABLE products ADD COLUMN name STRING;
    ALTER TABLE products ADD COLUMN price FLOAT;
    INSERT INTO products (name, price) VALUES ('Laptop', 999.99)
""")

# Execute multiple INSERT statements
client.execute("""
    INSERT INTO products (name, price) VALUES ('Mouse', 29.99);
    INSERT INTO products (name, price) VALUES ('Keyboard', 79.99);
    INSERT INTO products (name, price) VALUES ('Monitor', 299.99)
""")

# Query results
results = client.execute("SELECT * FROM products ORDER BY price DESC")
print(results.to_pandas())
```

### Full-Text Search

```python
# Initialize FTS
client.init_fts(index_fields=["name", "city"], lazy_load=True)

# Search
ids = client.search_text("Alice")
records = client.search_and_retrieve("Beijing")
top_records = client.search_and_retrieve_top("keyword", n=10)

# Fuzzy search (tolerates typos)
ids = client.fuzzy_search_text("Alic")

# FTS stats
stats = client.get_fts_stats()

# Disable or drop FTS
client.disable_fts()
client.drop_fts()
```

### ResultView Operations

```python
results = client.execute("SELECT * FROM default")

# Convert to different formats
df = results.to_pandas()          # pandas DataFrame
pl_df = results.to_polars()       # polars DataFrame
arrow_table = results.to_arrow()  # PyArrow Table
dicts = results.to_dict()         # List of dicts

# Result properties
print(results.shape)       # (rows, columns)
print(results.columns)     # column names
print(len(results))        # row count

# Get single values
first_row = results.first()
ids = results.get_ids()    # numpy array
scalar = client.execute("SELECT COUNT(*) FROM default").scalar()
```

### Context Manager Support

```python
# Automatic cleanup with context manager
with ApexClient("./data") as client:
    client.store({"key": "value"})
    results = client.execute("SELECT * FROM default")
    # Client automatically closed on exit
```

## 📊 Performance Comparison

### ApexBase vs DuckDB

Comparison with [DuckDB](https://duckdb.org/) (v1.1.3), a popular embedded analytical database.

**Test Environment**

| Component | Specification |
|-----------|---------------|
| **Platform** | macOS 26.2 (arm64) |
| **CPU** | Apple M1 Pro |
| **Memory** | 32.0 GB |
| **Python** | 3.11.10 |
| **ApexBase** | v0.4.2 |
| **DuckDB** | v1.1.3 |
| **PyArrow** | 19.0.0 |

**Dataset**: 1,000,000 rows with columns: `name` (string), `age` (int), `score` (float), `category` (string)

**Query Performance** (average of 5 iterations, after 2 warmup runs)

| Query | ApexBase | DuckDB | Ratio |
|-------|----------|--------|-------|
| COUNT(*) | 0.08ms | 0.37ms | **0.22x** (4.6x faster) |
| SELECT * LIMIT 100 | 0.09ms | 0.25ms | **0.35x** (2.9x faster) |
| SELECT * LIMIT 10K | 0.26ms | 3.53ms | **0.07x** (13.6x faster) |
| Filter (name = 'user_5000') | 7.41ms | 6.35ms | **1.17x** |
| Insert 1M rows | 327.44ms | 197844.50ms | **0.00x** (604x faster) |

**Notes**: 
- Ratio < 1 means ApexBase is faster than DuckDB
- ApexBase excels at INSERT operations and large LIMIT queries due to Arrow IPC optimization
- DuckDB has better performance on complex GROUP BY and ORDER BY operations

## 🔧 API Reference

### ApexClient

#### Initialization

```python
client = ApexClient(
    dirpath="./data",           # Data directory (default: current dir)
    drop_if_exists=False,       # Delete existing data on open
    batch_size=1000,            # Batch size for operations
    enable_cache=True,          # Enable query cache
    cache_size=10000,           # Cache size
    prefer_arrow_format=True,   # Prefer Arrow format for results
    durability="fast",          # Durability level: "fast" | "safe" | "max"
)

# Create clean instance (drop existing data)
client = ApexClient.create_clean("./data")

# Context manager
with ApexClient("./data") as client:
    ...
```

#### Table Management

| Method | Description |
|--------|-------------|
| `create_table(name)` | Create a new table |
| `drop_table(name)` | Drop a table |
| `use_table(name)` | Switch to a table |
| `list_tables()` | List all tables |
| `current_table` | Property: get current table name |

#### Data Storage

| Method | Description |
|--------|-------------|
| `store(data)` | Store data (dict, list, DataFrame, Arrow Table) |
| `from_pandas(df)` | Import from pandas DataFrame |
| `from_polars(df)` | Import from polars DataFrame |
| `from_pyarrow(table)` | Import from PyArrow Table |

#### Data Retrieval

| Method | Description |
|--------|-------------|
| `retrieve(id)` | Get record by internal _id |
| `retrieve_many(ids)` | Get multiple records by _id |
| `retrieve_all()` | Get all records |
| `execute(sql)` | Execute SQL query |
| `query(where, limit)` | Query with WHERE expression |
| `count_rows(table)` | Count rows in table |

#### Data Modification

| Method | Description |
|--------|-------------|
| `replace(id, data)` | Replace a record |
| `batch_replace({id: data})` | Batch replace records |
| `delete(id)` or `delete([ids])` | Delete record(s) |

#### Column Operations

| Method | Description |
|--------|-------------|
| `add_column(name, type)` | Add a column |
| `drop_column(name)` | Drop a column |
| `rename_column(old, new)` | Rename a column |
| `get_column_dtype(name)` | Get column data type |
| `list_fields()` | List all fields/columns |

#### Full-Text Search

| Method | Description |
|--------|-------------|
| `init_fts(fields, lazy_load, cache_size)` | Initialize FTS |
| `search_text(query)` | Search documents |
| `fuzzy_search_text(query)` | Fuzzy search (tolerates typos) |
| `search_and_retrieve(query, limit, offset)` | Search and return records |
| `search_and_retrieve_top(query, n)` | Return top N results |
| `get_fts_stats()` | Get FTS statistics |
| `disable_fts()` | Disable FTS |
| `drop_fts()` | Drop FTS index |

#### Utility

| Method | Description |
|--------|-------------|
| `flush()` | Flush data to disk |
| `set_auto_flush(rows, bytes)` | Set auto-flush thresholds |
| `get_auto_flush()` | Get auto-flush configuration |
| `estimate_memory_bytes()` | Estimate memory usage |
| `close()` | Close the client |

### ResultView

Query results are returned as `ResultView` objects:

| Method/Property | Description |
|-----------------|-------------|
| `to_pandas(zero_copy=True)` | Convert to pandas DataFrame |
| `to_polars()` | Convert to polars DataFrame |
| `to_arrow()` | Convert to PyArrow Table |
| `to_dict()` | Convert to list of dicts |
| `scalar()` | Get single scalar value |
| `first()` | Get first row |
| `get_ids(return_list=False)` | Get record IDs |
| `shape` | Property: (rows, columns) |
| `columns` | Property: column names |
| `__len__()` | Row count |
| `__iter__()` | Iterate over rows |
| `__getitem__(idx)` | Index access |

## 📚 Documentation

Documentation entry point: `docs/README.md`

## 📄 License

Apache-2.0
