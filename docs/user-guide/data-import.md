# Data Import

ApexBase is designed to move data in and out of Python data tools without turning every workflow into a custom ETL job.

## Choose An Import Path

| Source | Best method |
| --- | --- |
| Python records | `client.store(dict)` or `client.store(list[dict])` |
| Large Python batches | `client.store(columnar_dict)` |
| Pandas | `client.from_pandas(df, table_name=...)` |
| Polars | `client.from_polars(df, table_name=...)` |
| PyArrow | `client.from_pyarrow(table, table_name=...)` |
| CSV / JSON / Parquet one-off query | SQL `read_csv()`, `read_json()`, `read_parquet()` |
| CSV / JSON / Parquet repeated queries | `register_temp_table()` |

## Columnar Batches

Columnar dictionaries are usually the fastest pure-Python ingest format:

```python
client.create_table("events")
client.store({
    "user_id": [1, 2, 3],
    "event": ["signup", "click", "purchase"],
    "value": [0.0, 0.0, 29.9],
})
```

## DataFrames

```python
import pandas as pd
import polars as pl
import pyarrow as pa

client.from_pandas(
    pd.DataFrame({"name": ["Alice"], "age": [30]}),
    table_name="users",
)

client.from_polars(
    pl.DataFrame({"name": ["Bob"], "age": [25]}),
    table_name="users",
)

client.from_pyarrow(
    pa.table({"name": ["Charlie"], "age": [35]}),
    table_name="users",
)
```

When `table_name` is provided, ApexBase selects or creates that table for the import.

## Query Files Directly

```python
result = client.execute("""
    SELECT city, COUNT(*) AS rows
    FROM read_csv('events.csv')
    GROUP BY city
    ORDER BY rows DESC
""")
```

```python
client.execute("SELECT * FROM read_parquet('orders.parquet') LIMIT 10")
client.execute("SELECT * FROM read_json('events.ndjson') WHERE kind = 'click'")
```

Direct file functions are a good fit for ad hoc analysis and one-time joins.

## Register Temporary Tables

For repeated queries against the same file, parse it once and use a temporary table:

```python
client.register_temp_table("events_file", "events.csv")

client.execute("""
    SELECT event, COUNT(*) AS rows
    FROM events_file
    GROUP BY event
""")

client.drop_temp_table("events_file")
```

Temporary tables are cleaned up when the client closes.

## Join Files With Stored Data

```python
client.execute("""
    SELECT u.name, f.event
    FROM users u
    JOIN read_csv('events.csv') f ON u.id = f.user_id
    WHERE f.event = 'purchase'
""")
```

## Parquet Interop

```python
client.execute("COPY users TO 'users.parquet' (FORMAT PARQUET)")
client.execute("COPY users FROM 'users.parquet' (FORMAT PARQUET)")
```

## Performance Tips

- Create tables with schemas for large repeated imports.
- Prefer columnar dictionaries or DataFrame import for bulk writes.
- Use `register_temp_table()` when querying the same file many times.
- Convert results to Arrow or Polars when downstream code is columnar.
- Use `durability="fast"` for scratch imports and `durability="safe"` or `durability="max"` for application data.
