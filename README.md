# ApexBase

🚀 A lightning-fast, feature-rich embedded database designed for modern Python applications.

## Features

✨ **High Performance**
- Built on SQLite/DuckDB with optimized configurations
- Efficient batch operations support
- Automatic performance optimization
- Concurrent access support
- Columnar storage support with DuckDB backend

🔍 **Powerful Query Capabilities**
- SQL-like query syntax
- Complex queries with multiple conditions
- JSON field support

📊 **Data Framework Integration**
- Seamless integration with Pandas
- Native support for PyArrow
- Built-in Polars compatibility

🎯 **Multi-table Support**
- Multiple table management
- Easy table switching
- Automatic table creation and deletion

🛡️ **Data Integrity**
- ACID compliance
- Transaction support
- Automatic error handling
- Data consistency guarantees

🔧 **Developer Friendly**
- Simple and intuitive API
- Minimal configuration required
- Comprehensive documentation
- Extensive test coverage

## Installation

```bash
pip install apexbase
```

## Quick Start

```bash
pip install apexbase
```

```python
from apexbase import ApexClient

# Initialize the database with SQLite backend (default)
client = ApexClient("my_database")

# Or use DuckDB backend for columnar storage and analytics
client = ApexClient("my_database", backend="duckdb")

# Store single record
record = {"name": "John", "age": 30, "tags": ["python", "rust"]}
id_ = client.store(record)

# Store multiple records
records = [
    {"name": "Jane", "age": 25},
    {"name": "Bob", "age": 35}
]
ids = client.store(records)

# Query records
results = client.query("age > 25")
for record in results:
    print(record)

# Import from Pandas
import pandas as pd
df = pd.DataFrame({"name": ["Alice", "Bob"], "age": [28, 32]})
client.from_pandas(df)
```

### Storage Backend Selection

```python
# Use SQLite backend (default) - optimized for OLTP workloads
client = ApexClient("my_database", backend="sqlite")

# Use DuckDB backend - optimized for OLAP and analytics workloads
client = ApexClient("my_database", backend="duckdb")

# DuckDB backend advantages:
# - Columnar storage for better analytics performance
# - Efficient compression
# - Vectorized query execution
# - Better performance for analytical queries
# - Native support for complex aggregations
```

## Advanced Usage

### Multi-table Operations

```python
# Create and switch tables
client.create_table("users")
client.create_table("orders")
client.use_table("users")

# Store user data
user = {"name": "John", "email": "john@example.com"}
user_id = client.store(user)

# Switch to orders table
client.use_table("orders")
order = {"user_id": user_id, "product": "Laptop"}
client.store(order)
```

### Complex Queries

```python
# Multiple conditions
results = client.query("age > 25 AND city = 'New York'")

# Range queries
results = client.query("score >= 85.0 AND score <= 90.0")

# LIKE queries
results = client.query("name LIKE 'J%'")
```

### Performance Optimization

```python
# Store large batch of records
records = [{"id": i, "value": i * 2} for i in range(10000)]
ids = client.store(records)
```

## Requirements

- Python >= 3.9
- Dependencies:
  - orjson
  - pandas
  - pyarrow
  - polars
  - numpy
  - psutil
  - duckdb (optional, for columnar storage backend)

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
