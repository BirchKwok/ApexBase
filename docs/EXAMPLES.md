# ApexBase Usage Examples

Comprehensive examples covering 100% of the ApexBase Python API.

## Table of Contents

1. [Basic Operations](#basic-operations)
2. [Table Management](#table-management)
3. [SQL DDL Operations](#sql-ddl-operations)
4. [Data Import](#data-import)
5. [Querying Data](#querying-data)
6. [Column Operations](#column-operations)
7. [Full-Text Search](#full-text-search)
8. [Data Modification](#data-modification)
9. [Utility Methods](#utility-methods)
10. [Advanced Usage](#advanced-usage)

---

## Basic Operations

### Initialization

```python
from apexbase import ApexClient

# Basic initialization
client = ApexClient("./data")

# With options
client = ApexClient(
    dirpath="./data",
    durability="safe",        # "fast" | "safe" | "max"
    batch_size=1000,
    cache_size=10000
)

# Create clean instance (deletes existing data)
client = ApexClient.create_clean("./fresh_data")
```

### Storing Data

```python
# Create a table first (required before any data operations)
client.create_table("users")

# Single record
client.store({"name": "Alice", "age": 30})

# Multiple records
client.store([
    {"name": "Bob", "age": 25},
    {"name": "Charlie", "age": 35}
])

# Columnar format (fastest for bulk)
client.store({
    "name": ["David", "Eve", "Frank"],
    "age": [28, 32, 40],
    "city": ["NYC", "LA", "Chicago"]
})
```

### Closing

```python
# Explicit close
client.close()

# Context manager (recommended)
with ApexClient("./data") as client:
    client.create_table("mydata")
    client.store({"key": "value"})
    # Auto-closes on exit
```

---

## Table Management

ApexBase requires explicit table creation before any data operations. Each table is stored as a separate `.apex` file.

```python
client = ApexClient("./data")

# Create tables (the last created table becomes the active table)
client.create_table("users")

# Create table with pre-defined schema
client.create_table("orders", schema={
    "order_id": "int64",
    "product": "string",
    "price": "float64"
})

# List tables
tables = client.list_tables()
print(tables)  # ['users', 'orders']

# Switch table
client.use_table("users")
print(client.current_table)  # 'users'

# Store in the active table
client.store({"username": "alice", "email": "alice@example.com"})

# Reopen an existing database
client2 = ApexClient("./data")
client2.use_table("users")  # Select an existing table

# Drop table (active table resets to None)
client.drop_table("orders")

client.close()
```

---

## SQL DDL Operations

ApexBase supports full SQL DDL (Data Definition Language) operations:

### CREATE TABLE

```python
client = ApexClient("./data")

# Create a new table (becomes the active table)
client.execute("CREATE TABLE employees")

# Create table only if it doesn't exist
client.execute("CREATE TABLE IF NOT EXISTS departments")

# Verify tables were created
print(client.list_tables())  # ['employees', 'departments']
```

### INSERT

```python
# Insert single row with column names
client.execute("INSERT INTO employees (name, age, department) VALUES ('Alice', 30, 'Engineering')")

# Insert multiple rows in one statement
client.execute("""
    INSERT INTO employees (name, age, department) 
    VALUES 
        ('Charlie', 35, 'Marketing'),
        ('David', 28, 'Engineering')
""")

# Query the data
results = client.execute("SELECT * FROM employees")
print(f"Inserted {len(results)} records")
```

### ALTER TABLE

```python
# Add a column with type
client.execute("ALTER TABLE employees ADD COLUMN email STRING")
client.execute("ALTER TABLE employees ADD COLUMN salary FLOAT")

# Rename a column
client.execute("ALTER TABLE employees RENAME COLUMN email TO email_address")

# Drop a column
client.execute("ALTER TABLE employees DROP COLUMN email_address")
```

### DROP TABLE

```python
# Drop a table
client.execute("DROP TABLE departments")

# Drop only if exists (avoids error if table doesn't exist)
client.execute("DROP TABLE IF EXISTS temp_table")

# Verify department table was dropped
print(client.list_tables())  # ['employees']
```

### Complete DDL Workflow

```python
client = ApexClient("./data")

# 1. Create tables via SQL
client.execute("CREATE TABLE IF NOT EXISTS products")
client.execute("CREATE TABLE IF NOT EXISTS categories")

# 2. Add columns via SQL
client.execute("ALTER TABLE categories ADD COLUMN name STRING")
client.execute("ALTER TABLE categories ADD COLUMN description STRING")
client.execute("ALTER TABLE products ADD COLUMN name STRING")
client.execute("ALTER TABLE products ADD COLUMN price FLOAT")
client.execute("ALTER TABLE products ADD COLUMN category_id INT")

# 3. Insert data via SQL
client.execute("INSERT INTO categories (name, description) VALUES ('Electronics', 'Gadgets')")
client.execute("""
    INSERT INTO products (name, price, category_id) 
    VALUES 
        ('Laptop', 999.99, 1),
        ('Smartphone', 699.99, 1)
""")

# 4. Query the data
products = client.execute("SELECT * FROM products")
print(products.to_pandas())

# 5. Clean up via SQL
client.execute("DROP TABLE IF EXISTS products")
client.execute("DROP TABLE IF EXISTS categories")

client.close()
```

### Supported DDL Syntax

| Statement | Description | Example |
|-----------|-------------|---------|
| `CREATE TABLE [IF NOT EXISTS] name` | Create new table | `CREATE TABLE users` |
| `INSERT INTO ... VALUES ...` | Insert single row | `INSERT INTO users (name) VALUES ('Alice')` |
| `INSERT INTO ... VALUES (...), (...)` | Insert multiple rows | `INSERT INTO users (name) VALUES ('A'), ('B')` |
| `ALTER TABLE ... ADD COLUMN ...` | Add column | `ALTER TABLE users ADD COLUMN age INT` |
| `ALTER TABLE ... RENAME COLUMN ...` | Rename column | `ALTER TABLE users RENAME COLUMN age TO years` |
| `ALTER TABLE ... DROP COLUMN ...` | Drop column | `ALTER TABLE users DROP COLUMN age` |
| `DROP TABLE [IF EXISTS] name` | Drop table | `DROP TABLE users` |

### Supported Data Types in DDL

| Type | Aliases | Description |
|------|---------|-------------|
| `STRING` | `VARCHAR`, `TEXT` | Variable-length text |
| `INT` | `INTEGER`, `INT32`, `INT64` | Integer numbers |
| `FLOAT` | `DOUBLE`, `FLOAT64` | Floating-point numbers |
| `BOOL` | `BOOLEAN` | True/false values |

### Multi-Statement SQL

Execute multiple SQL statements separated by semicolons:

```python
client = ApexClient("./data")

# Setup entire schema in one call
client.execute("""
    CREATE TABLE IF NOT EXISTS products;
    ALTER TABLE products ADD COLUMN name STRING;
    ALTER TABLE products ADD COLUMN price FLOAT;
    ALTER TABLE products ADD COLUMN category STRING;
    INSERT INTO products (name, price, category) VALUES ('Laptop', 999.99, 'Electronics')
""")

# Insert multiple rows efficiently
client.execute("""
    INSERT INTO products (name, price, category) VALUES ('Mouse', 29.99, 'Electronics');
    INSERT INTO products (name, price, category) VALUES ('Keyboard', 79.99, 'Electronics');
    INSERT INTO products (name, price, category) VALUES ('Monitor', 299.99, 'Electronics');
    INSERT INTO products (name, price, category) VALUES ('Desk', 199.99, 'Furniture')
""")

# Query and analyze
results = client.execute("""
    SELECT category, COUNT(*) as count, AVG(price) as avg_price 
    FROM products 
    GROUP BY category
""")
print(results.to_pandas())

# Clean up multiple tables
client.execute("""
    DROP TABLE IF EXISTS products;
    DROP TABLE IF EXISTS temp_table
""")

client.close()
```

**Notes on Multi-Statement SQL:**
- Statements are separated by semicolons (`;`)
- Semicolons inside string literals are handled correctly
- Statements execute sequentially
- The result of the last statement is returned
- Useful for schema setup and batch operations

---

## Data Import

### From Pandas

```python
import pandas as pd

client = ApexClient("./data")

df = pd.DataFrame({
    "product": ["A", "B", "C"],
    "price": [10.5, 20.0, 15.0],
    "quantity": [100, 200, 150]
})

# table_name auto-creates and selects the table
client.from_pandas(df, table_name="products")
```

### From Polars

```python
import polars as pl

client = ApexClient("./data")

df = pl.DataFrame({
    "name": ["Alice", "Bob", "Charlie"],
    "score": [85, 92, 78]
})

client.from_polars(df, table_name="scores")
```

### From PyArrow

```python
import pyarrow as pa

client = ApexClient("./data")

table = pa.table({
    "id": [1, 2, 3],
    "value": ["a", "b", "c"]
})

client.from_pyarrow(table, table_name="items")
```

---

## Querying Data

### SQL Queries

```python
client = ApexClient("./data")
client.create_table("metrics")

# Insert test data
for i in range(100):
    client.store({"id": i, "value": i * 10, "category": f"cat_{i % 5}"})

# Basic SELECT (use your table name in FROM clause)
results = client.execute("SELECT * FROM metrics")
print(f"Total rows: {len(results)}")

# WHERE clause
results = client.execute("SELECT * FROM metrics WHERE value > 500")

# ORDER BY with LIMIT
results = client.execute("""
    SELECT * FROM metrics
    WHERE category = 'cat_1'
    ORDER BY value DESC
    LIMIT 10
""")

# Aggregation
results = client.execute("""
    SELECT 
        COUNT(*) as total,
        AVG(value) as avg_value,
        MAX(value) as max_value,
        MIN(value) as min_value
    FROM metrics
""")
print(results.first())

# GROUP BY
results = client.execute("""
    SELECT category, COUNT(*), AVG(value)
    FROM metrics
    GROUP BY category
""")
for row in results:
    print(row)

# Get single scalar value
count = client.execute("SELECT COUNT(*) FROM metrics").scalar()
print(f"Count: {count}")

# Count rows shortcut
count = client.count_rows()
count = client.count_rows("users")  # Specific table

client.close()
```

### Using query() Method

```python
client = ApexClient("./data")
client.use_table("users")  # Select existing table

# Simple WHERE expression (uses the active table)
results = client.query("age > 25")
results = client.query("name LIKE 'A%'")

# With limit
results = client.query("age > 25", limit=100)

# Using where_clause parameter
results = client.query(where_clause="city = 'NYC'", limit=50)

client.close()
```

### Retrieve by ID

```python
client = ApexClient("./data")
client.use_table("users")

# Single record
record = client.retrieve(0)
print(record)  # {'_id': 0, 'name': 'Alice', ...}

# Multiple records
results = client.retrieve_many([0, 5, 10, 15])
df = results.to_pandas()

# All records
results = client.retrieve_all()
print(f"Shape: {results.shape}")

client.close()
```

### ResultView Operations

```python
results = client.execute("SELECT * FROM users WHERE age > 25")

# Convert formats
df = results.to_pandas()
pl_df = results.to_polars()
arrow_table = results.to_arrow()
dicts = results.to_dict()

# Properties
print(results.shape)      # (rows, columns)
print(results.columns)    # ['_id', 'name', 'age', ...]
print(len(results))       # row count

# Get IDs
ids = results.get_ids()           # numpy array
ids = results.get_ids(return_list=True)  # Python list

# Get single values
first = results.first()
scalar = client.execute("SELECT COUNT(*) FROM users").scalar()

# Iteration
for row in results:
    print(row["name"])

# Indexing
row = results[0]
```

---

## Column Operations

```python
client = ApexClient("./data")
client.create_table("people")
client.store({"name": "Alice", "age": 30})

# Add column
client.add_column("email", "String")
client.add_column("score", "Float64")

# Rename column
client.rename_column("email", "email_address")

# Get column type
dtype = client.get_column_dtype("age")
print(f"age column type: {dtype}")  # Int64

# List all fields
fields = client.list_fields()
print(fields)  # ['_id', 'name', 'age', 'email_address', 'score']

# Drop column
client.drop_column("score")

client.close()
```

---

## Full-Text Search

FTS is available via two interfaces. The SQL interface (recommended) works over Python, PG Wire, and Arrow Flight. The Python API provides direct programmatic access.

> See [FTS_GUIDE.md](FTS_GUIDE.md) for the complete reference.

### SQL Interface (Recommended)

```python
client = ApexClient("./data")
client.create_table("articles")

# 1. Create the FTS index via SQL DDL
client.execute("CREATE FTS INDEX ON articles (title, content)")

# 2. Insert data — rows are indexed automatically
client.store([
    {"title": "Python Tutorial",     "content": "Learn Python programming"},
    {"title": "Rust Guide",          "content": "Systems programming with Rust"},
    {"title": "Database Design",     "content": "Designing efficient databases"},
    {"title": "Machine Learning",    "content": "Deep learning with PyTorch"},
])

# 3. MATCH — all query terms must appear
results = client.execute("SELECT * FROM articles WHERE MATCH('python')")
print(results.to_pandas())
#    _id            title                    content
# 0    0  Python Tutorial  Learn Python programming

# 4. FUZZY_MATCH — tolerates typos
results = client.execute("SELECT * FROM articles WHERE FUZZY_MATCH('progaming')")

# 5. Combine FTS with other predicates
results = client.execute("""
    SELECT title FROM articles
    WHERE MATCH('programming') AND _id > 0
    ORDER BY _id DESC LIMIT 5
""")

# 6. FTS + aggregation
n = client.execute("SELECT COUNT(*) FROM articles WHERE MATCH('python')").scalar()
print(f"Python articles: {n}")

# 7. Manage indexes
client.execute("SHOW FTS INDEXES")                       # lists all databases
client.execute("ALTER FTS INDEX ON articles DISABLE")    # suspend, keep files
client.execute("ALTER FTS INDEX ON articles ENABLE")     # resume + back-fill missed rows
client.execute("DROP FTS INDEX ON articles")             # remove + delete files

client.close()
```

### FTS with Options

```python
# Large index: lazy loading + bigger cache
client.execute("""
    CREATE FTS INDEX ON logs (message, source)
    WITH (lazy_load=true, cache_size=100000)
""")

# Cross-interface: after init, PG Wire and Arrow Flight can use MATCH() too
# (No extra configuration needed — FTS registry is global in the Rust executor)
```

### Python API (Alternative)

```python
client = ApexClient("./data")
client.create_table("docs")

client.store([
    {"title": "Python Tutorial", "content": "Learn Python programming"},
    {"title": "Rust Guide",      "content": "Systems programming with Rust"},
    {"title": "Database Design", "content": "Designing efficient databases"},
])

# Initialize FTS (also registers with global SQL executor)
client.init_fts(index_fields=["title", "content"])

# Search — returns numpy array of _ids
ids = client.search_text("Python")
print(f"Found {len(ids)} documents")

# Search and retrieve full records
results = client.search_and_retrieve("programming")
for row in results:
    print(f"Title: {row['title']}")

# Top N results
top_results = client.search_and_retrieve_top("database", n=5)

# Fuzzy search (tolerates typos)
ids = client.fuzzy_search_text("progamming")  # Note typo

# Stats
stats = client.get_fts_stats()
print(f"Documents: {stats['doc_count']}, Terms: {stats['term_count']}")

client.disable_fts()   # suspend (keep files)
client.drop_fts()      # remove (delete files)

client.close()
```

### FTS with Lazy Loading

```python
client = ApexClient("./data")
client.use_table("docs")

# Initialize with lazy loading
client.init_fts(
    index_fields=["content"],
    lazy_load=True,      # Index loaded on first search
    cache_size=50000     # FTS cache size
)

# First search will load the index
results = client.search_and_retrieve("keyword")

client.close()
```

---

## Data Modification

### Replace Records

```python
client = ApexClient("./data")
client.create_table("people")

# Insert test data
client.store({"name": "Alice", "age": 30})

# Replace by ID
success = client.replace(0, {"name": "Alice Smith", "age": 31})

# Batch replace
updated_ids = client.batch_replace({
    0: {"name": "Alice Updated", "age": 32},
    1: {"name": "Bob Updated", "age": 26}
})

client.close()
```

### Delete Records

```python
client = ApexClient("./data")
client.use_table("people")

# Delete single record
client.delete(5)

# Delete multiple records
client.delete([1, 2, 3, 4, 5])

client.close()
```

---

## Utility Methods

```python
client = ApexClient("./data")
client.use_table("users")

# Flush data to disk
client.flush()

# Flush cache (alias for flush)
client.flush_cache()

# Set auto-flush thresholds
client.set_auto_flush(rows=1000)        # Flush every 1000 rows
client.set_auto_flush(bytes=1024*1024)  # Flush every 1MB
client.set_auto_flush(rows=500, bytes=512*1024)

# Get auto-flush config
rows, bytes = client.get_auto_flush()

# Estimate memory usage
mem_bytes = client.estimate_memory_bytes()
print(f"Memory: {mem_bytes / 1024 / 1024:.2f} MB")

# Optimize storage
client.optimize()  # Currently same as flush

client.close()
```

---

## Advanced Usage

### Multi-Table Operations

```python
client = ApexClient("./data")

# Create multiple tables
client.create_table("users")
client.create_table("products")
client.create_table("orders")

# Work with users
client.use_table("users")
client.store([
    {"user_id": 1, "name": "Alice"},
    {"user_id": 2, "name": "Bob"}
])

# Work with products
client.use_table("products")
client.store([
    {"product_id": 1, "name": "Laptop", "price": 999.99},
    {"product_id": 2, "name": "Mouse", "price": 29.99}
])

# Query across tables
users = client.execute("SELECT * FROM users")
client.use_table("products")
products = client.execute("SELECT * FROM products")

# Get row counts for all tables
for table in client.list_tables():
    count = client.count_rows(table)
    print(f"{table}: {count} rows")

client.close()
```

### Durability Options

```python
# Fast - async writes, best performance (default)
client = ApexClient("./data", durability="fast")

# Safe - sync writes, data safety
client = ApexClient("./data", durability="safe")

# Max - fsync every write, maximum durability
client = ApexClient("./data", durability="max")
```

### Working with Large Datasets

```python
# Use columnar storage for bulk inserts
client = ApexClient("./data")
client.create_table("benchmark")

# Generate large dataset
import numpy as np
n = 1_000_000

data = {
    "id": range(n),
    "value": np.random.randn(n),
    "category": np.random.choice(["A", "B", "C"], n)
}

# Columnar insert is much faster
client.store(data)

# Query with limit
results = client.execute("SELECT * FROM benchmark LIMIT 100")

client.close()
```

### Complete Workflow Example

```python
from apexbase import ApexClient
import pandas as pd

# Initialize with safe durability
with ApexClient("./analytics", durability="safe") as client:
    # Create table
    client.create_table("sales")
    
    # Import data from pandas
    df = pd.DataFrame({
        "date": pd.date_range("2024-01-01", periods=365),
        "product": np.random.choice(["A", "B", "C"], 365),
        "quantity": np.random.randint(1, 100, 365),
        "revenue": np.random.randn(365) * 100 + 500
    })
    client.from_pandas(df)
    
    # Add computed column
    client.add_column("region", "String")
    
    # Query analytics
    monthly = client.execute("""
        SELECT 
            strftime('%Y-%m', date) as month,
            SUM(revenue) as total_revenue,
            AVG(quantity) as avg_quantity
        FROM sales
        GROUP BY month
        ORDER BY month
    """)
    
    print(monthly.to_pandas())
    
    # Initialize FTS for product search
    client.init_fts(index_fields=["product"])
    
    # Search products
    results = client.search_and_retrieve("A")
    print(f"Found {len(results)} records")
    
    # Cleanup
    client.drop_fts()
    
# Client automatically closed
```

---

## API Summary

### ApexClient Methods

**Initialization:** `__init__`, `create_clean`, `close`

**Table Management:** `use_table`, `create_table`, `drop_table`, `list_tables`, `current_table`

**Data Storage:** `store`, `from_pandas(df, table_name=)`, `from_polars(df, table_name=)`, `from_pyarrow(table, table_name=)`

**Query:** `execute`, `query`, `retrieve`, `retrieve_many`, `retrieve_all`, `count_rows`

**Modification:** `replace`, `batch_replace`, `delete`

**Columns:** `add_column`, `drop_column`, `rename_column`, `get_column_dtype`, `list_fields`

**FTS:** `init_fts`, `search_text`, `fuzzy_search_text`, `search_and_retrieve`, `search_and_retrieve_top`, `get_fts_stats`, `disable_fts`, `drop_fts`

**Utility:** `flush`, `flush_cache`, `set_auto_flush`, `get_auto_flush`, `estimate_memory_bytes`, `optimize`

### ResultView Methods

**Conversion:** `to_pandas`, `to_polars`, `to_arrow`, `to_dict`

**Access:** `scalar`, `first`, `get_ids`, `__len__`, `__iter__`, `__getitem__`

**Properties:** `shape`, `columns`

---

*This documentation covers 100% of the public ApexBase Python API.*
