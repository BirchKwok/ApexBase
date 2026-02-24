# ApexBase API Reference

Complete API reference for ApexBase Python SDK.

## Table of Contents

1. [ApexClient](#apexclient) - Main client class
2. [ResultView](#resultview) - Query results
3. [Constants](#constants) - Module constants
4. [File Reading Table Functions](#file-reading-table-functions) - read_csv / read_parquet / read_json
5. [Set Operations](#set-operations) - UNION / INTERSECT / EXCEPT

---

## ApexClient

The main entry point for ApexBase operations.

### Constructor

```python
ApexClient(
    dirpath: str = None,
    batch_size: int = 1000,
    drop_if_exists: bool = False,
    enable_cache: bool = True,
    cache_size: int = 10000,
    prefer_arrow_format: bool = True,
    durability: Literal['fast', 'safe', 'max'] = 'fast',
    _auto_manage: bool = True
)
```

**Parameters:**
- `dirpath`: Data directory path (default: current directory)
- `batch_size`: Batch size for bulk operations
- `drop_if_exists`: If True, delete existing data on open
- `enable_cache`: Enable query result caching
- `cache_size`: Maximum cache entries
- `prefer_arrow_format`: Prefer Arrow format for internal transfers
- `durability`: Persistence level - 'fast' (async), 'safe' (sync), 'max' (fsync every write)

**Example:**
```python
from apexbase import ApexClient

# Basic usage
client = ApexClient("./data")

# With durability options
client = ApexClient("./data", durability="safe")

# Clean start (drop existing)
client = ApexClient.create_clean("./data")
```

---

### Database Management

ApexBase supports multiple isolated databases within a single root directory. Each named database is stored as a subdirectory; the `'default'` database maps to the root directory (backward-compatible).

#### use_database
```python
use_database(database: str = 'default') -> ApexClient
```
Switch to a named database. Creates the database subdirectory if it does not exist. Resets the current table to `None`.

**Parameters:**
- `database`: Database name. `'default'` (or `''`) maps to the root directory.

**Returns:** `self` (for method chaining)

**Examples:**
```python
# Switch to analytics database
client.use_database("analytics")

# Switch back to default (root-level tables)
client.use_database("default")

# Method chaining
client.use_database("hr").create_table("employees")
```

---

#### use
```python
use(database: str = 'default', table: str = None) -> ApexClient
```
Switch to a named database and optionally select or create a table in one call. If `table` is specified and does not exist it is created automatically.

**Parameters:**
- `database`: Database name (default = root-level).
- `table`: Table name to select. If `None`, only the database is switched.

**Returns:** `self` (for method chaining)

**Examples:**
```python
# Switch database only
client.use(database="analytics")

# Switch database and select an existing table
client.use(database="analytics", table="events")

# Switch database and auto-create table if missing
client.use(database="new_db", table="new_table")
client.store({"key": "value"})
```

---

#### list_databases
```python
list_databases() -> List[str]
```
Return a sorted list of all available databases. `'default'` is always included.

**Example:**
```python
dbs = client.list_databases()
print(dbs)  # ['analytics', 'default', 'hr']
```

---

#### current_database
```python
current_database: str  # Property
```
Return the name of the currently active database. Returns `'default'` when operating on root-level tables.

**Example:**
```python
client.use_database("analytics")
print(client.current_database)  # 'analytics'
```

---

### Cross-Database SQL

All SQL operations support the standard `database.table` qualified name syntax. The active database context only affects unqualified table references; qualified references always resolve to the correct database regardless of context.

**Supported operations:**
```python
# SELECT across databases
client.execute("SELECT * FROM default.users")
client.execute("SELECT * FROM analytics.events WHERE cnt > 10")

# JOIN across databases
client.execute("""
    SELECT u.name, e.event
    FROM default.users u
    JOIN analytics.events e ON u.id = e.user_id
""")

# INSERT into a different database
client.execute("INSERT INTO analytics.events (name, cnt) VALUES ('click', 1)")

# UPDATE in a different database
client.execute("UPDATE default.users SET age = 31 WHERE name = 'Alice'")

# DELETE from a different database
client.execute("DELETE FROM default.users WHERE age < 18")

# DDL across databases
client.execute("CREATE TABLE analytics.summary (total INT)")
client.execute("DROP TABLE IF EXISTS analytics.old_table")
```

---

### Table Management

#### create_table
```python
create_table(table_name: str, schema: dict = None) -> None
```
Create a new table, optionally with a pre-defined schema.

**Parameters:**
- `table_name`: Name of the table to create.
- `schema`: Optional dict mapping column names to type strings. Pre-defining schema avoids type inference on the first insert, providing a performance benefit for bulk loading.

**Supported types:** `int8`, `int16`, `int32`, `int64`, `uint8`, `uint16`, `uint32`, `uint64`, `float32`, `float64`, `bool`, `string`, `binary`

**Examples:**
```python
# Without schema
client.create_table("users")

# With pre-defined schema
client.create_table("orders", schema={
    "order_id": "int64",
    "product": "string",
    "price": "float64",
    "paid": "bool"
})
```

#### drop_table
```python
drop_table(table_name: str) -> None
```
Drop a table and all its data.

**Example:**
```python
client.drop_table("old_table")
```

#### use_table
```python
use_table(table_name: str) -> None
```
Switch to a different table for subsequent operations.

**Example:**
```python
client.use_table("users")
```

#### list_tables
```python
list_tables() -> List[str]
```
Return list of all table names.

**Example:**
```python
tables = client.list_tables()
print(tables)  # ['users', 'orders']
```

#### current_table
```python
current_table: Optional[str]  # Property
```
Get the name of the currently active table. Returns `None` if no table is selected.

**Example:**
```python
print(client.current_table)  # 'users'
```

---

### Data Storage

#### store
```python
store(data) -> None
```
Store data in the active table. Requires a table to be selected via `create_table()` or `use_table()` first. Accepts multiple formats:
- Single dict: `{"name": "Alice", "age": 30}`
- List of dicts: `[{"name": "A"}, {"name": "B"}]`
- Dict of columns: `{"name": ["A", "B"], "age": [20, 30]}`
- pandas DataFrame
- polars DataFrame  
- PyArrow Table

**Examples:**
```python
# Single record
client.store({"name": "Alice", "age": 30})

# Multiple records
client.store([
    {"name": "Bob", "age": 25},
    {"name": "Charlie", "age": 35}
])

# Columnar format (fastest for bulk)
client.store({
    "name": ["David", "Eve"],
    "age": [28, 32]
})
```

#### from_pandas
```python
from_pandas(df: pd.DataFrame, table_name: str = None) -> ApexClient
```
Import data from pandas DataFrame. Returns self for chaining.

**Parameters:**
- `df`: pandas DataFrame to import
- `table_name`: Optional. If provided, auto-creates/selects the table before importing.

**Example:**
```python
import pandas as pd
df = pd.DataFrame({"name": ["A", "B"], "age": [20, 30]})
client.from_pandas(df, table_name="users")
```

#### from_polars
```python
from_polars(df: pl.DataFrame, table_name: str = None) -> ApexClient
```
Import data from polars DataFrame. Returns self for chaining.

**Parameters:**
- `df`: polars DataFrame to import
- `table_name`: Optional. If provided, auto-creates/selects the table before importing.

**Example:**
```python
import polars as pl
df = pl.DataFrame({"name": ["A", "B"], "age": [20, 30]})
client.from_polars(df, table_name="users")
```

#### from_pyarrow
```python
from_pyarrow(table: pa.Table, table_name: str = None) -> ApexClient
```
Import data from PyArrow Table. Returns self for chaining.

**Parameters:**
- `table`: PyArrow Table to import
- `table_name`: Optional. If provided, auto-creates/selects the table before importing.

**Example:**
```python
import pyarrow as pa
table = pa.table({"name": ["A", "B"], "age": [20, 30]})
client.from_pyarrow(table, table_name="users")
```

---

### Data Retrieval

#### execute
```python
execute(sql: str, show_internal_id: bool = None) -> ResultView
```
Execute SQL query and return results.

**Parameters:**
- `sql`: SQL statement (SELECT, INSERT, etc.)
- `show_internal_id`: If True, include _id column in results

**Example:**
```python
# Basic query (use your table name in FROM clause)
results = client.execute("SELECT * FROM users WHERE age > 25")

# Aggregation
results = client.execute("SELECT COUNT(*), AVG(age) FROM users")
count = results.scalar()

# With ordering and limits
results = client.execute("SELECT name, age FROM users ORDER BY age DESC LIMIT 10")
```

#### query
```python
query(
    sql: str = None,
    where_clause: str = None,
    limit: int = None
) -> ResultView
```
Query with WHERE expression (backward compatibility).

**Example:**
```python
# WHERE expression only
results = client.query("age > 25")
results = client.query("name LIKE 'A%'")

# With limit
results = client.query(where_clause="city = 'NYC'", limit=100)
```

#### retrieve
```python
retrieve(id_: int) -> Optional[dict]
```
Get a single record by its internal _id.

**Example:**
```python
record = client.retrieve(0)
print(record)  # {'_id': 0, 'name': 'Alice', 'age': 30}
```

#### retrieve_many
```python
retrieve_many(ids: List[int]) -> ResultView
```
Get multiple records by their internal _ids.

**Example:**
```python
results = client.retrieve_many([0, 1, 2, 5])
df = results.to_pandas()
```

#### retrieve_all
```python
retrieve_all() -> ResultView
```
Get all records from the current table.

**Example:**
```python
results = client.retrieve_all()
print(len(results))  # Total row count
```

#### count_rows
```python
count_rows(table_name: str = None) -> int
```
Count rows in a table.

**Example:**
```python
count = client.count_rows()
count = client.count_rows("users")  # Specific table
```

---

### Data Modification

#### replace
```python
replace(id_: int, data: dict) -> bool
```
Replace a record by _id.

**Example:**
```python
success = client.replace(0, {"name": "Alice", "age": 31})
```

#### batch_replace
```python
batch_replace(data_dict: Dict[int, dict]) -> List[int]
```
Batch replace multiple records.

**Example:**
```python
updated = client.batch_replace({
    0: {"name": "Alice", "age": 31},
    1: {"name": "Bob", "age": 26}
})
```

#### delete
```python
delete(ids: Union[int, List[int]]) -> bool
```
Delete record(s) by _id.

**Example:**
```python
# Single delete
client.delete(5)

# Batch delete
client.delete([1, 2, 3])
```

---

### Column Operations

#### add_column
```python
add_column(column_name: str, column_type: str) -> None
```
Add a new column to the current table.

**Types:** `Int8`, `Int16`, `Int32`, `Int64`, `UInt8`, `UInt16`, `UInt32`, `UInt64`, `Float32`, `Float64`, `String`, `Bool`

**Example:**
```python
client.add_column("email", "String")
client.add_column("score", "Float64")
```

#### drop_column
```python
drop_column(column_name: str) -> None
```
Drop a column from the current table. Cannot drop _id column.

**Example:**
```python
client.drop_column("temp_field")
```

#### rename_column
```python
rename_column(old_column_name: str, new_column_name: str) -> None
```
Rename a column. Cannot rename _id column.

**Example:**
```python
client.rename_column("email", "email_address")
```

#### get_column_dtype
```python
get_column_dtype(column_name: str) -> str
```
Get the data type of a column.

**Example:**
```python
dtype = client.get_column_dtype("age")  # 'Int64'
```

#### list_fields
```python
list_fields() -> List[str]
```
List all column names in the current table.

**Example:**
```python
fields = client.list_fields()
print(fields)  # ['_id', 'name', 'age', 'city']
```

---

### Full-Text Search

FTS is implemented natively in Rust and available through all interfaces (Python API, PG Wire, Arrow Flight). The recommended way to manage and query FTS indexes is via SQL.

#### FTS SQL DDL and Query Reference

| Statement | Description |
|-----------|-------------|
| `CREATE FTS INDEX ON table (col1, col2)` | Create FTS index on specified columns |
| `CREATE FTS INDEX ON table` | Create FTS index on all string columns |
| `CREATE FTS INDEX ON table WITH (opt=val)` | Create with options |
| `DROP FTS INDEX ON table` | Drop index and delete files |
| `ALTER FTS INDEX ON table DISABLE` | Suspend indexing, keep files |
| `ALTER FTS INDEX ON table ENABLE` | Resume indexing and back-fill any missed rows |
| `SHOW FTS INDEXES` | List FTS-configured tables across all databases |
| `WHERE MATCH('query')` | Exact / ranked full-text search |
| `WHERE FUZZY_MATCH('query')` | Fuzzy / typo-tolerant search |

**`CREATE FTS INDEX`**
```sql
CREATE FTS INDEX ON table_name [(col1, col2, ...)] [WITH (lazy_load=bool, cache_size=N)]
```
- `(col1, col2)` — optional column list; omit to index all string columns
- `lazy_load` — defer loading index into RAM until first search (default `false`)
- `cache_size` — LRU cache entries for the index (default `10000`)

```python
client.execute("CREATE FTS INDEX ON articles (title, content)")
client.execute("CREATE FTS INDEX ON logs WITH (lazy_load=true, cache_size=50000)")
```

**`DROP FTS INDEX`**
```sql
DROP FTS INDEX ON table_name
```
Removes the index entry and deletes the `.nfts` index files from disk.

**`ALTER FTS INDEX ... DISABLE`**
```sql
ALTER FTS INDEX ON table_name DISABLE
```
Suspends FTS write-sync (INSERT / DELETE no longer update the index). Index files are kept on disk. Use `ALTER FTS INDEX ... ENABLE` to resume.

**`ALTER FTS INDEX ... ENABLE`**
```sql
ALTER FTS INDEX ON table_name ENABLE
```
Resumes FTS write-sync and **back-fills all rows** currently in the table, including any rows inserted while FTS was disabled.

**`SHOW FTS INDEXES`**
```sql
SHOW FTS INDEXES
```
Returns a result set with columns: `database`, `table`, `enabled`, `fields`, `lazy_load`, `cache_size`. Lists indexes across the root directory and all named sub-databases.

```python
result = client.execute("SHOW FTS INDEXES")
df = result.to_pandas()
#   database     table  enabled          fields  lazy_load  cache_size
#    default  articles     True  title, content      False       10000
```

**`MATCH('query')`**

Used in `WHERE` clauses to filter rows whose indexed text contains all query terms.

```python
# Simple search
client.execute("SELECT * FROM articles WHERE MATCH('python tutorial')")

# Combined with other conditions
client.execute("""
    SELECT title, content FROM articles
    WHERE MATCH('machine learning') AND year >= 2023
    ORDER BY _id DESC
    LIMIT 10
""")

# Aggregations
client.execute("SELECT COUNT(*) FROM articles WHERE MATCH('rust')")
```

**`FUZZY_MATCH('query')`**

Like `MATCH()` but tolerates typos and spelling variations.

```python
client.execute("SELECT * FROM articles WHERE FUZZY_MATCH('pytohn')")   # matches 'python'
client.execute("SELECT * FROM articles WHERE FUZZY_MATCH('databse')")  # matches 'database'
```

> **Note:** `MATCH()` / `FUZZY_MATCH()` require a FTS index to exist for the queried table. Use `CREATE FTS INDEX ON table` first. The SQL interface works over all transports (Python, PG Wire, Arrow Flight).

---

#### Python API

#### init_fts
```python
init_fts(
    table_name: str = None,
    index_fields: Optional[List[str]] = None,
    lazy_load: bool = False,
    cache_size: int = 10000
) -> ApexClient
```
Initialize full-text search for a table.

**Parameters:**
- `table_name`: Table to index (default: current table)
- `index_fields`: Fields to index (None = all string fields)
- `lazy_load`: Load index on first search
- `cache_size`: FTS cache size

**Example:**
```python
client.init_fts(index_fields=["title", "content"])
client.init_fts(index_fields=["name"], lazy_load=True)
```

#### search_text
```python
search_text(query: str, table_name: str = None) -> np.ndarray
```
Search for documents containing query terms. Returns array of _ids.

**Example:**
```python
ids = client.search_text("database")
print(ids)  # array([0, 5, 10])
```

#### fuzzy_search_text
```python
fuzzy_search_text(
    query: str,
    min_results: int = 1,
    table_name: str = None
) -> np.ndarray
```
Fuzzy search tolerating typos. Returns array of _ids.

**Example:**
```python
ids = client.fuzzy_search_text("databse")  # Matches "database"
```

#### search_and_retrieve
```python
search_and_retrieve(
    query: str,
    table_name: str = None,
    limit: Optional[int] = None,
    offset: int = 0
) -> ResultView
```
Search and return full records.

**Example:**
```python
results = client.search_and_retrieve("python")
results = client.search_and_retrieve("python", limit=10, offset=20)
```

#### search_and_retrieve_top
```python
search_and_retrieve_top(
    query: str,
    n: int = 100,
    table_name: str = None
) -> ResultView
```
Return top N search results.

**Example:**
```python
results = client.search_and_retrieve_top("important", n=5)
```

#### get_fts_stats
```python
get_fts_stats(table_name: str = None) -> Dict
```
Get FTS statistics.

**Example:**
```python
stats = client.get_fts_stats()
print(stats)  # {'fts_enabled': True, 'doc_count': 1000, 'term_count': 5000}
```

#### disable_fts
```python
disable_fts(table_name: str = None) -> ApexClient
```
Disable FTS (keeps index files).

**Example:**
```python
client.disable_fts()
```

#### drop_fts
```python
drop_fts(table_name: str = None) -> ApexClient
```
Disable FTS and delete index files.

**Example:**
```python
client.drop_fts()
```

---

### Utility Methods

#### flush
```python
flush() -> None
```
Flush all pending writes to disk.

**Example:**
```python
client.flush()
```

#### set_auto_flush
```python
set_auto_flush(rows: int = 0, bytes: int = 0) -> None
```
Set auto-flush thresholds.

**Example:**
```python
client.set_auto_flush(rows=1000)  # Flush every 1000 rows
client.set_auto_flush(bytes=1024*1024)  # Flush every 1MB
```

#### get_auto_flush
```python
get_auto_flush() -> tuple
```
Get current auto-flush configuration.

**Example:**
```python
rows, bytes = client.get_auto_flush()
```

#### estimate_memory_bytes
```python
estimate_memory_bytes() -> int
```
Estimate current memory usage in bytes.

**Example:**
```python
mem_bytes = client.estimate_memory_bytes()
print(f"Using {mem_bytes / 1024 / 1024:.1f} MB")
```

#### close
```python
close() -> None
```
Close the client and release resources.

**Example:**
```python
client.close()
```

---

## ResultView

Container for query results with multiple output formats.

### Conversion Methods

#### to_pandas
```python
to_pandas(zero_copy: bool = True) -> pd.DataFrame
```
Convert to pandas DataFrame.

**Parameters:**
- `zero_copy`: Use ArrowDtype for zero-copy (pandas 2.0+)

**Example:**
```python
df = results.to_pandas()
df = results.to_pandas(zero_copy=False)  # Traditional NumPy types
```

#### to_polars
```python
to_polars() -> pl.DataFrame
```
Convert to polars DataFrame.

**Example:**
```python
df = results.to_polars()
```

#### to_arrow
```python
to_arrow() -> pa.Table
```
Convert to PyArrow Table.

**Example:**
```python
table = results.to_arrow()
```

#### to_dict
```python
to_dict() -> List[dict]
```
Convert to list of dictionaries.

**Example:**
```python
records = results.to_dict()
for record in records:
    print(record["name"])
```

### Access Methods

#### scalar
```python
scalar() -> Any
```
Get single scalar value (for aggregate queries).

**Example:**
```python
count = client.execute("SELECT COUNT(*) FROM users").scalar()
```

#### first
```python
first() -> Optional[dict]
```
Get first row as dictionary.

**Example:**
```python
row = results.first()
```

#### get_ids
```python
get_ids(return_list: bool = False) -> Union[np.ndarray, List[int]]
```
Get internal _ids from results.

**Example:**
```python
ids = results.get_ids()  # numpy array (default)
ids = results.get_ids(return_list=True)  # Python list
```

### Properties

#### shape
```python
shape: tuple  # (rows, columns)
```

#### columns
```python
columns: List[str]
```

### Sequence Interface

```python
# Length
len(results)

# Iteration
for row in results:
    print(row)

# Indexing
first = results[0]
second = results[1]
```

---

## Constants

### Module Constants

```python
from apexbase import (
    __version__,      # Package version
    FTS_AVAILABLE,    # True (FTS always available)
    ARROW_AVAILABLE,  # True if pyarrow installed
    POLARS_AVAILABLE, # True if polars installed
    DurabilityLevel,  # Type hint: Literal['fast', 'safe', 'max']
)
```

---

## SQL Support

ApexBase supports standard SQL for querying:

### SELECT
```sql
SELECT * FROM table
SELECT col1, col2 FROM table
SELECT col1 AS alias FROM table
SELECT DISTINCT col1 FROM table
SELECT * FROM table WHERE condition
SELECT * FROM table ORDER BY col DESC
SELECT * FROM table LIMIT 100
SELECT * FROM table LIMIT 100 OFFSET 10
SELECT * FROM table ORDER BY col LIMIT 100
```

### Aggregate Functions
```sql
SELECT COUNT(*) FROM table
SELECT COUNT(DISTINCT col) FROM table
SELECT SUM(col), AVG(col), MAX(col), MIN(col) FROM table
```

### WHERE Clauses
```sql
WHERE col = value
WHERE col > value
WHERE col LIKE 'pattern%'
WHERE col IN (1, 2, 3)
WHERE col BETWEEN 10 AND 20
WHERE col IS NULL
WHERE col IS NOT NULL
WHERE condition1 AND condition2
WHERE condition1 OR condition2
```

### GROUP BY / HAVING
```sql
SELECT category, COUNT(*), AVG(price) 
FROM products 
GROUP BY category

SELECT category, COUNT(*) 
FROM products 
GROUP BY category 
HAVING COUNT(*) > 10
```

### JOINs
```sql
SELECT * FROM table1 JOIN table2 ON table1.id = table2.id
```

---

## File Reading Table Functions

ApexBase provides SQL table functions that read external files directly in a `FROM` clause without importing data into a table first. All three functions return a result compatible with the full SQL engine — you can apply `WHERE`, `GROUP BY`, `ORDER BY`, `LIMIT`, `JOIN`, and `UNION` on top of them.

### read_csv

```sql
SELECT * FROM read_csv('path/to/file.csv')
SELECT * FROM read_csv('path/to/file.csv', header=true, delimiter=',')
```

**Parameters:**

| Option | Default | Description |
|--------|---------|-------------|
| `header` | `true` | Whether the first row is a header. Set to `false` / `0` if the file has no header row. |
| `delimiter` / `delim` / `sep` | `,` | Field delimiter character. Use `'\t'` for TSV files. |

**Schema inference**: types are inferred automatically from the first 100 data rows (Int64, Float64, Bool, or String).

**Examples:**

```python
# Read a comma-delimited CSV with header row
result = client.execute("SELECT * FROM read_csv('/data/sales.csv')")

# Tab-separated values (TSV)
result = client.execute("SELECT * FROM read_csv('/data/data.tsv', delimiter='\t')")

# No header row
result = client.execute("SELECT * FROM read_csv('/data/raw.csv', header=false)")

# Full SQL on top of the file
result = client.execute("""
    SELECT city, COUNT(*) AS cnt, AVG(price)
    FROM read_csv('/data/orders.csv')
    WHERE price > 100
    GROUP BY city
    ORDER BY cnt DESC
    LIMIT 10
""")

# Convert directly to DataFrame
df = client.execute("SELECT * FROM read_csv('/data/large.csv')").to_pandas()
```

### read_parquet

```sql
SELECT * FROM read_parquet('path/to/file.parquet')
```

No options — schema is read directly from the Parquet file's metadata.

**Examples:**

```python
# Read a Parquet file
result = client.execute("SELECT * FROM read_parquet('/data/events.parquet')")

# Projection and filter
result = client.execute("""
    SELECT user_id, SUM(amount) AS total
    FROM read_parquet('/data/transactions.parquet')
    WHERE category = 'food'
    GROUP BY user_id
""")

# Zero-copy to Arrow
table = client.execute("SELECT * FROM read_parquet('/data/wide.parquet')").to_arrow()
```

### read_json

```sql
SELECT * FROM read_json('path/to/file.json')
```

Handles two formats automatically:
- **NDJSON / JSON Lines** — one JSON object per line (`.json`, `.jsonl`, `.ndjson`)
- **pandas column-oriented JSON** — output of `df.to_json(orient='columns')` or `orient='split'`

No options — format is detected automatically.

**Examples:**

```python
# Read NDJSON (one JSON object per line)
result = client.execute("SELECT * FROM read_json('/data/logs.ndjson')")

# Read pandas-exported JSON
result = client.execute("SELECT * FROM read_json('/data/export.json')")

# Apply aggregation
result = client.execute("""
    SELECT status, COUNT(*) AS cnt
    FROM read_json('/data/events.json')
    GROUP BY status
    ORDER BY cnt DESC
""")
```

### Joining file reads with tables

```python
# JOIN a file read with a stored table
result = client.execute("""
    SELECT u.name, f.score
    FROM users u
    JOIN read_csv('/data/scores.csv') f ON u.id = f.user_id
    WHERE f.score > 90
""")

# UNION a file with a table
result = client.execute("""
    SELECT name, city FROM users
    UNION ALL
    SELECT name, city FROM read_csv('/data/new_users.csv')
""")
```

### Performance notes

- All three functions use `mmap` for zero-copy file access.
- CSV and JSON files are parsed in parallel (one chunk per CPU core via Rayon).
- Parquet files use parallel column decoding with shared metadata (zero re-parse overhead).
- Benchmarked against Polars on 1M rows: CSV 0.95×, NDJSON 0.93×, Parquet 1.33× (Arrow output).

---

## Set Operations

SQL set operations combine result sets from two or more `SELECT` statements. Both sides must return the same number of columns.

### UNION

Combines rows from both sides and **removes duplicate rows**.

```sql
SELECT col FROM table_a
UNION
SELECT col FROM table_b
```

### UNION ALL

Combines rows from both sides and **keeps all rows**, including duplicates.

```sql
SELECT col FROM table_a
UNION ALL
SELECT col FROM table_b
```

### INTERSECT

Returns only rows that appear in **both** result sets (deduplicated).

```sql
SELECT col FROM table_a
INTERSECT
SELECT col FROM table_b
```

### EXCEPT

Returns rows from the **left** side that do **not** appear in the right side (deduplicated).

```sql
SELECT col FROM table_a
EXCEPT
SELECT col FROM table_b
```

### Ordering and limiting set operation results

Append `ORDER BY`, `LIMIT`, and `OFFSET` after the final set operation; they apply to the combined result:

```sql
SELECT val FROM a
UNION
SELECT val FROM b
ORDER BY val ASC
LIMIT 100
```

### Examples

```python
# UNION: unique names across two tables
client.execute("""
    SELECT name FROM customers
    UNION
    SELECT name FROM leads
    ORDER BY name
""")

# UNION ALL: all rows from both tables (duplicates kept)
client.execute("""
    SELECT name, city FROM domestic_users
    UNION ALL
    SELECT name, city FROM international_users
""")

# INTERSECT: users who appear in both the orders and wishlist tables
client.execute("""
    SELECT user_id FROM orders
    INTERSECT
    SELECT user_id FROM wishlist
""")

# EXCEPT: users who placed orders but have no open support tickets
client.execute("""
    SELECT user_id FROM orders
    EXCEPT
    SELECT user_id FROM support_tickets WHERE status = 'open'
""")

# Works with read_csv too
client.execute("""
    SELECT email FROM users
    EXCEPT
    SELECT email FROM read_csv('/data/unsubscribed.csv')
""")
```

### Set operation summary

| Operation | Duplicates | Rows returned |
|-----------|-----------|---------------|
| `UNION` | removed | left ∪ right (unique) |
| `UNION ALL` | kept | all rows from both sides |
| `INTERSECT` | removed | left ∩ right |
| `EXCEPT` | removed | left \ right |

---

### INSERT
```sql
INSERT INTO table (col1, col2) VALUES (1, 'a')
INSERT INTO table VALUES (1, 'a', 3.14)
INSERT INTO table (col1, col2) VALUES (1, 'a'), (2, 'b'), (3, 'c')
```

### DDL (Data Definition Language)

ApexBase supports full SQL DDL operations.

#### CREATE TABLE
```sql
CREATE TABLE table_name
CREATE TABLE IF NOT EXISTS table_name
```

#### ALTER TABLE
```sql
-- Add column
ALTER TABLE table_name ADD COLUMN column_name DATA_TYPE

-- Rename column  
ALTER TABLE table_name RENAME COLUMN old_name TO new_name

-- Drop column
ALTER TABLE table_name DROP COLUMN column_name
```

#### DROP TABLE
```sql
DROP TABLE table_name
DROP TABLE IF EXISTS table_name
```

#### Supported Data Types

| Type | Aliases | Description |
|------|---------|-------------|
| `STRING` | `VARCHAR`, `TEXT` | String/text data |
| `INT` | `INTEGER`, `INT32`, `INT64` | Integer numbers |
| `FLOAT` | `DOUBLE`, `FLOAT64` | Floating point numbers |
| `BOOL` | `BOOLEAN` | Boolean values |

### Examples

```python
# Create table via SQL
client.execute("CREATE TABLE IF NOT EXISTS users")

# Add columns via SQL
client.execute("ALTER TABLE users ADD COLUMN name STRING")
client.execute("ALTER TABLE users ADD COLUMN age INT")

# Insert data via SQL
client.execute("INSERT INTO users (name, age) VALUES ('Alice', 30)")
results = client.execute("SELECT * FROM users WHERE age > 25")

# Modify schema via SQL
client.execute("ALTER TABLE users RENAME COLUMN name TO full_name")
client.execute("ALTER TABLE users DROP COLUMN age")

# Drop table via SQL
client.execute("DROP TABLE IF EXISTS users")
```

#### Multi-Statement SQL

You can execute multiple SQL statements in a single call by separating them with semicolons:

```python
# Execute multiple statements at once
client.execute("""
    CREATE TABLE IF NOT EXISTS products;
    ALTER TABLE products ADD COLUMN name STRING;
    ALTER TABLE products ADD COLUMN price FLOAT;
    INSERT INTO products (name, price) VALUES ('Laptop', 999.99)
""")

# Multiple INSERT statements
client.execute("""
    INSERT INTO products (name, price) VALUES ('Mouse', 29.99);
    INSERT INTO products (name, price) VALUES ('Keyboard', 79.99);
    INSERT INTO products (name, price) VALUES ('Monitor', 299.99)
""")

# The result of the last statement is returned
results = client.execute("""
    CREATE TABLE IF NOT EXISTS temp;
    INSERT INTO temp (name) VALUES ('test');
    SELECT * FROM temp
""")
```

**Multi-Statement SQL Rules:**
- Statements are separated by semicolons (`;`)
- Semicolons inside string literals are handled correctly
- Statements execute sequentially in order
- The result of the last SELECT statement is returned
- DDL statements (CREATE, ALTER, DROP, INSERT) return empty results

---
