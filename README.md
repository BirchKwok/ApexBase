# ApexBase

🚀 A lightning-fast, feature-rich embedded database designed for modern Python applications.

## Features

✨ **High Performance**
- Built on DuckDB with optimized columnar storage
- Efficient batch operations support
- Configurable caching system for better write performance
- Concurrent query execution
- Optimized for both OLAP and OLTP workloads

🔍 **Advanced Query Capabilities**
- SQL-like query syntax with WHERE clauses
- Complex queries with multiple conditions and operators
- LIKE pattern matching support
- Lazy evaluation with ResultView
- Vectorized query execution
- Native JSON field support

🔤 **Full-Text Search (FTS)**
- Built-in full-text search capabilities
- Chinese text segmentation support
- Configurable indexing parameters
- Parallel index building
- Disk-based index storage with memory caching

📊 **Data Framework Integration**
- Seamless integration with Pandas
- Native support for PyArrow
- Built-in Polars compatibility
- Automatic type inference and conversion

🎯 **Advanced Table Management**
- Multiple table support with easy switching
- Dynamic schema management
- Add, drop, and rename columns at runtime
- Get column data types
- Table creation and deletion

🛡️ **Data Integrity & Performance**
- ACID compliance through DuckDB
- Transaction support
- Automatic error handling
- Data consistency guarantees
- Performance optimization utilities

🔧 **Developer Friendly**
- Simple and intuitive API
- Flexible configuration options
- Comprehensive documentation
- Extensive test coverage
- Performance monitoring and testing

## Installation

```bash
pip install apexbase
```

## Quick Start

```python
from apexbase import ApexClient

# Initialize the database with configuration options
client = ApexClient(
    dirpath="my_database",
    batch_size=1000,
    enable_cache=True,
    cache_size=10000,
    drop_if_exists=False
)

# Store single record
record = {"name": "John", "age": 30, "tags": ["python", "rust"]}
id_ = client.store(record)

# Store multiple records
records = [
    {"name": "Jane", "age": 25},
    {"name": "Bob", "age": 35}
]
ids = client.store(records)

# Query records with lazy evaluation
results = client.query("age > 25")
print(f"Found {results.shape[0]} records")

# Convert to different formats
pandas_df = results.to_pandas()
polars_df = results.to_polars()
pyarrow_table = results.to_arrow()

# Iterate over results
for record in results:
    print(record)

# Import from data frameworks
import pandas as pd
df = pd.DataFrame({"name": ["Alice", "Bob"], "age": [28, 32]})
client.from_pandas(df)
```

### DuckDB Advantages

ApexBase is built on DuckDB, providing significant advantages:

- **Columnar storage** for better analytics performance
- **Efficient compression** for reduced storage footprint
- **Vectorized query execution** for faster results
- **Parallel query processing** for multi-core utilization
- **Native support for complex aggregations**
- **Optimized for both OLAP and OLTP workloads**
- **Efficient memory management** for large datasets

## Advanced Usage

### Table Management

```python
# Create and switch tables
client.create_table("users")
client.create_table("orders") 
client.use_table("users")

# Check current table
print(f"Current table: {client.current_table}")

# List all tables
tables = client.list_tables()
print(f"Available tables: {tables}")

# Store user data
user = {"name": "John", "email": "john@example.com"}
user_id = client.store(user)

# Switch to orders table
client.use_table("orders")
order = {"user_id": user_id, "product": "Laptop", "price": 999.99}
client.store(order)

# Drop table when no longer needed
client.drop_table("orders")
```

### Schema Management

```python
# Add new columns dynamically
client.add_column("phone", "VARCHAR")
client.add_column("salary", "DOUBLE")
client.add_column("is_manager", "BOOLEAN")

# Get column information
fields = client.list_fields()
print(f"Available fields: {fields}")

# Check column data type
dtype = client.get_column_dtype("salary")
print(f"Salary column type: {dtype}")

# Rename columns
client.rename_column("phone", "phone_number")

# Drop columns (when no longer needed)
client.drop_column("phone_number")
```

### Advanced Querying

```python
# Multiple conditions with ResultView
results = client.query("age > 25 AND city = 'New York'")

# Get query metadata
print(f"Query returned {results.shape[0]} rows and {results.shape[1]} columns")
print(f"Column names: {results.columns}")

# Convert to different formats
pandas_df = results.to_pandas()
polars_df = results.to_polars() 
arrow_table = results.to_arrow()

# Collect all results
all_data = results.collect()

# Range queries
results = client.query("score >= 85.0 AND score <= 90.0")

# LIKE pattern matching
results = client.query("name LIKE 'J%'")

# Complex string queries
results = client.query("email LIKE '%@gmail.com'")

# Retrieve all records
all_results = client.retrieve_all()
```

### Full-Text Search

```python
from apexbase.fts import FullTextSearch

# Initialize FTS with configuration
fts = FullTextSearch(
    index_dir="fts_index",
    max_chinese_length=4,
    num_workers=4,
    shard_size=100000,
    min_term_length=2,
    auto_save=True,
    batch_size=1000
)

# Add documents for full-text search
documents = [
    {"text": "Python is a great programming language"},
    {"text": "Machine learning with Python and TensorFlow"}, 
    {"text": "Data science using pandas and numpy"},
    {"text": "深度学习和自然语言处理技术"},
]

# Index documents
for i, doc in enumerate(documents):
    fts.add_document(i, doc["text"])

# Search for terms
results = fts.search("Python programming")
print(f"Found documents: {list(results)}")

# Search Chinese text
results = fts.search("深度学习")
print(f"Found Chinese documents: {list(results)}")

# Save index to disk
fts.save()

# Close FTS
fts.close()
```

### Data Framework Integration

```python
import pandas as pd
import polars as pl
import pyarrow as pa

# Import from Pandas
pdf = pd.DataFrame({
    "name": ["Alice", "Bob", "Charlie"],
    "age": [25, 30, 35],
    "department": ["Engineering", "Sales", "Marketing"]
})
client.from_pandas(pdf)

# Import from Polars  
pldf = pl.DataFrame({
    "product": ["Laptop", "Phone", "Tablet"],
    "price": [999.99, 599.99, 399.99],
    "in_stock": [True, False, True]
})
client.from_polars(pldf)

# Import from PyArrow
table = pa.Table.from_pandas(pdf)
client.from_pyarrow(table)

# Query and export to different formats
results = client.query("age > 28")
export_df = results.to_pandas()
export_pl = results.to_polars()
export_arrow = results.to_arrow()
```

### Performance Optimization & Monitoring

```python
# Store large batch of records efficiently
large_batch = [
    {
        "id": i, 
        "value": i * 2,
        "category": f"cat_{i % 10}",
        "timestamp": f"2024-01-{(i % 30) + 1:02d}"
    } 
    for i in range(100000)
]

# Batch storage with performance monitoring
import time
start_time = time.time()
ids = client.store(large_batch)
storage_time = time.time() - start_time
print(f"Stored {len(ids)} records in {storage_time:.2f} seconds")

# Force cache flush for immediate persistence
client.flush_cache()

# Optimize database performance
client.optimize()

# Monitor row counts
total_rows = client.count_rows()
print(f"Total rows in current table: {total_rows}")

# Performance tips:
# 1. Use batch operations for large datasets
# 2. Enable caching for write-heavy workloads  
# 3. Adjust batch_size based on your memory constraints
# 4. Call optimize() after large data loads
# 5. Use appropriate cache_size for your working set
```

## Requirements

- Python >= 3.9
- Core Dependencies:
  - `duckdb` - High-performance columnar database engine
  - `orjson` - Fast JSON serialization
  - `pandas` - Data manipulation and analysis
  - `pyarrow` - In-memory columnar data format
  - `polars` - Fast DataFrames library
  - `numpy` - Numerical computing
  - `psutil` - System and process utilities
- Full-Text Search Dependencies:
  - `pyroaring` - Compressed bitmap data structure
  - `msgpack` - Efficient binary serialization

## Configuration Options

When initializing `ApexClient`, you can customize various settings:

```python
client = ApexClient(
    dirpath="path/to/database",     # Database directory path
    batch_size=1000,                # Batch processing size
    drop_if_exists=False,           # Drop existing database
    enable_cache=True,              # Enable write caching
    cache_size=10000               # Cache size for write operations
)
```

### Configuration Parameters

- **`dirpath`** (str): Directory for database storage. Defaults to current directory.
- **`batch_size`** (int): Number of records to process in each batch. Default: 1000.
- **`drop_if_exists`** (bool): Whether to delete existing database. Default: False.
- **`enable_cache`** (bool): Enable caching for better write performance. Default: True.
- **`cache_size`** (int): Maximum number of cached records. Default: 10000.

## Performance Benchmarks

ApexBase is optimized for high-performance operations:

- **Single record storage**: < 1ms
- **Batch operations**: 10,000+ records/second
- **Query performance**: Leverages DuckDB's vectorized execution
- **Memory efficiency**: Optimized for large datasets with controlled memory usage
- **Full-text search**: Efficient indexing and searching with configurable parameters

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
