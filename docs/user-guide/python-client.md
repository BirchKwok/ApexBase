# Python Client Guide

The Python client is the primary ApexBase interface for embedded applications, scripts, and notebooks.

## Client Lifecycle

Use a context manager so files and temporary tables are cleaned up reliably:

```python
from apexbase import ApexClient

with ApexClient("./data", durability="safe") as client:
    client.create_table("events")
    client.store({"kind": "signup", "user_id": 1})
```

For repeatable examples or tests, start from a clean directory:

```python
with ApexClient.create_clean("./tmp_data") as client:
    client.create_table("scratch")
```

## Database And Table Selection

```python
client.use_database("analytics")
client.create_table("events")
client.use_table("events")

client.use(database="analytics", table="events")
```

`use(database=..., table=...)` switches database context and creates the table if it is missing.

## Writing Data

Single row:

```python
client.store({"name": "Alice", "age": 30})
```

Multiple rows:

```python
client.store([
    {"name": "Bob", "age": 25},
    {"name": "Charlie", "age": 35},
])
```

Columnar batch:

```python
client.store({
    "name": ["Diana", "Eve", "Frank"],
    "age": [29, 31, 44],
})
```

For large inserts, prefer columnar batches or DataFrame import.

## Querying

Use `execute()` for SQL:

```python
result = client.execute("""
    SELECT age, COUNT(*) AS users
    FROM users
    GROUP BY age
    ORDER BY users DESC
""")
```

Use `query()` for simple table-scoped filtering:

```python
result = client.query(where_clause="age >= 30", limit=100)
```

Use record helpers when you already know row ids:

```python
row = client.retrieve(1)
rows = client.retrieve_many([1, 2, 3])
all_rows = client.retrieve_all()
```

## Working With Results

```python
result = client.execute("SELECT * FROM users")

print(result.shape)
print(result.columns)

rows = result.to_dict()
df = result.to_pandas()
pl_df = result.to_polars()
arrow_table = result.to_arrow()
```

`ResultView` is intentionally lightweight: keep it while you need query results, then convert to the format your application already uses.

## Updating And Deleting

```python
client.replace(1, {"name": "Alice", "age": 31})
client.batch_replace({
    2: {"name": "Bob", "age": 26},
    3: {"name": "Charlie", "age": 36},
})

client.delete(where_clause="age < 18")
```

SQL DML is also supported:

```python
client.execute("UPDATE users SET age = 31 WHERE name = 'Alice'")
client.execute("DELETE FROM users WHERE age < 18")
```

## Schema Changes

```python
client.add_column("email", "String")
client.rename_column("email", "contact_email")
client.drop_column("contact_email")
print(client.list_fields())
```

For SQL-first workflows:

```python
client.execute("ALTER TABLE users ADD COLUMN email STRING")
```

## Durability And Flushing

```python
client = ApexClient("./data", durability="safe")
client.flush()
```

Use `fast` for maximum throughput, `safe` for balanced persistence, and `max` when every write must be fsynced.

## Where To Go Next

- [SQL Guide](sql.md) for supported query patterns.
- [Data Import](data-import.md) for DataFrame and file workflows.
- [Python API Reference](../API_REFERENCE.md) for complete method signatures.
