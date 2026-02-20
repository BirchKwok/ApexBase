# Full-Text Search (FTS) Guide

ApexBase integrates a high-performance full-text search engine ([NanoFTS](https://crates.io/crates/nanofts)) directly into the Rust SQL executor. FTS is a **first-class SQL feature** — indexes are created with DDL statements and queried with standard `WHERE` predicates, making FTS available through every interface: Python API, PostgreSQL Wire, and Arrow Flight.

---

## Table of Contents

1. [Architecture](#architecture)
2. [Quick Start](#quick-start)
3. [DDL Reference](#ddl-reference)
   - [CREATE FTS INDEX](#create-fts-index)
   - [DROP FTS INDEX](#drop-fts-index)
   - [ALTER FTS INDEX DISABLE](#alter-fts-index-disable)
   - [SHOW FTS INDEXES](#show-fts-indexes)
4. [Query Reference](#query-reference)
   - [MATCH()](#match)
   - [FUZZY_MATCH()](#fuzzy_match)
5. [Combining FTS with SQL](#combining-fts-with-sql)
6. [Python API](#python-api)
7. [PG Wire and Arrow Flight](#pg-wire-and-arrow-flight)
8. [Lifecycle and Storage](#lifecycle-and-storage)
9. [Configuration Options](#configuration-options)
10. [Performance Tips](#performance-tips)
11. [Limitations](#limitations)

---

## Architecture

```
                    ┌─────────────────────────────────────┐
                    │         SQL Executor (Rust)          │
                    │                                      │
  Python API  ───►  │  parse SQL → detect MATCH() →       │
  PG Wire     ───►  │  look up FtsManager →               │
  Arrow Flight ───► │  search → _id IN (...) → filter     │
                    │                                      │
                    └────────────┬────────────────────────┘
                                 │
                    ┌────────────▼────────────────────────┐
                    │  Global FTS Registry                 │
                    │  key: base_dir → Arc<FtsManager>    │
                    │  FtsManager: table → FtsEngine      │
                    └────────────┬────────────────────────┘
                                 │
                    ┌────────────▼────────────────────────┐
                    │  Disk: {dir}/fts_indexes/{table}.nfts│
                    │         {dir}/fts_config.json        │
                    └─────────────────────────────────────┘
```

Key design points:
- FTS state is stored in a **global Rust registry** keyed by database directory, so PG Wire and Arrow Flight connections share the same FTS engines as Python API calls.
- `MATCH('query')` in a `WHERE` clause is resolved to `_id IN (matching_ids)` before the query planner runs — zero changes to the rest of the execution pipeline.
- The configuration is persisted in `fts_config.json` alongside the `.apex` data files, and is re-loaded automatically on process restart.

---

## Quick Start

```python
from apexbase import ApexClient

client = ApexClient("./data")
client.create_table("articles")

# 1. Create the FTS index (DDL)
client.execute("CREATE FTS INDEX ON articles (title, content)")

# 2. Insert data — rows are indexed automatically on each store()
client.store([
    {"title": "Rust programming language", "content": "Rust is fast and safe"},
    {"title": "Python tutorial",           "content": "Python is easy to learn"},
    {"title": "Machine learning basics",   "content": "Deep learning with PyTorch"},
])

# 3. Query using MATCH() in WHERE
results = client.execute("SELECT * FROM articles WHERE MATCH('rust')")
print(results.to_pandas())
#    _id                  title                    content
# 0    0  Rust programming language  Rust is fast and safe

# 4. Fuzzy search — tolerates typos
results = client.execute("SELECT * FROM articles WHERE FUZZY_MATCH('pytohn')")
print(results.to_pandas())
#    _id            title                    content
# 1    1  Python tutorial  Python is easy to learn

client.close()
```

---

## DDL Reference

### CREATE FTS INDEX

```sql
CREATE FTS INDEX ON table_name
    [(col1 [, col2, ...])]
    [WITH (option = value [, ...])]
```

**Effect:**
- Registers the table in `fts_config.json` with `enabled = true`.
- Creates (or opens) the NanoFTS engine for the table under `{dir}/fts_indexes/{table}.nfts`.
- New documents stored via `store()` / `INSERT` are indexed automatically from this point.
- Existing documents are **not** back-filled automatically — use `init_fts()` Python API or re-insert data if back-filling is needed.

**Column list (optional)**

If omitted, all string columns are indexed. Specify columns to reduce index size and improve precision:

```sql
-- Index specific columns
CREATE FTS INDEX ON articles (title, content)

-- Index all string columns (no column list)
CREATE FTS INDEX ON articles
```

**Options (WITH clause)**

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `lazy_load` | bool | `false` | Load index lazily on first search instead of at startup |
| `cache_size` | int | `10000` | Number of terms to keep in the LRU cache |

```sql
CREATE FTS INDEX ON logs WITH (lazy_load=true, cache_size=50000)
CREATE FTS INDEX ON articles (title) WITH (cache_size=100000)
```

**Examples:**

```python
# Index title + content
client.execute("CREATE FTS INDEX ON articles (title, content)")

# Index all string columns with a large cache
client.execute("CREATE FTS INDEX ON wiki WITH (cache_size=200000)")

# Large index with lazy loading
client.execute("CREATE FTS INDEX ON emails (subject, body) WITH (lazy_load=true)")
```

---

### DROP FTS INDEX

```sql
DROP FTS INDEX ON table_name
```

**Effect:**
- Removes the table entry from `fts_config.json`.
- Deletes the `.nfts` index file and its WAL from `{dir}/fts_indexes/`.
- Removes the in-memory engine from the global registry.

```python
client.execute("DROP FTS INDEX ON articles")
```

---

### ALTER FTS INDEX DISABLE

```sql
ALTER FTS INDEX ON table_name DISABLE
```

**Effect:**
- Sets `enabled = false` in `fts_config.json`.
- **Does not** delete index files — the index can be re-activated by running `CREATE FTS INDEX ON table_name` again.
- Useful when temporarily suspending FTS to avoid write overhead during a bulk load.

```python
# Disable while doing a large bulk import
client.execute("ALTER FTS INDEX ON articles DISABLE")

# ... bulk import ...

# Re-enable
client.execute("CREATE FTS INDEX ON articles (title, content)")
```

---

### SHOW FTS INDEXES

```sql
SHOW FTS INDEXES
```

Returns a result set describing all FTS-enabled tables in the current database.

**Result columns:**

| Column | Type | Description |
|--------|------|-------------|
| `table` | string | Table name |
| `enabled` | bool | Whether FTS is currently active |
| `fields` | string | Indexed columns (comma-separated) |
| `lazy_load` | bool | Lazy-load mode |
| `cache_size` | int | LRU cache size |

```python
df = client.execute("SHOW FTS INDEXES").to_pandas()
print(df)
#       table  enabled          fields  lazy_load  cache_size
# 0  articles     True  title, content      False       10000
# 1      wiki     True  (all string cols)    False      200000
```

---

## Query Reference

### MATCH()

```sql
WHERE MATCH('query text')
```

Performs an exact full-text search. Returns all rows where the indexed text contains **all** of the query terms. Multi-word queries require all words to appear (AND semantics).

```sql
-- Single term
SELECT * FROM articles WHERE MATCH('python')

-- Multi-term (all terms must appear)
SELECT * FROM articles WHERE MATCH('machine learning')

-- Chinese / CJK supported
SELECT * FROM articles WHERE MATCH('人工智能')
```

**Return behaviour:** Internally resolves to `_id IN (doc1, doc2, ...)` — zero rows returned if no matches.

---

### FUZZY_MATCH()

```sql
WHERE FUZZY_MATCH('query text')
```

Fuzzy full-text search that tolerates spelling errors and typos. Uses edit-distance based matching.

```sql
-- Typo: 'pytohn' → matches 'python'
SELECT * FROM articles WHERE FUZZY_MATCH('pytohn')

-- Typo: 'databse' → matches 'database'
SELECT * FROM articles WHERE FUZZY_MATCH('databse')
```

> **When to use:** Prefer `MATCH()` for known-correct queries (faster). Use `FUZZY_MATCH()` for user-typed search inputs where typos are expected.

---

## Combining FTS with SQL

`MATCH()` / `FUZZY_MATCH()` are standard `WHERE` predicates and compose with all other SQL features:

```python
# FTS + date filter + ORDER BY + LIMIT
client.execute("""
    SELECT title, author, published_at
    FROM articles
    WHERE MATCH('neural network') AND published_at >= '2023-01-01'
    ORDER BY published_at DESC
    LIMIT 20
""")

# FTS + aggregation
client.execute("""
    SELECT author, COUNT(*) AS article_count
    FROM articles
    WHERE MATCH('deep learning')
    GROUP BY author
    ORDER BY article_count DESC
""")

# FTS in a CTE
client.execute("""
    WITH matched AS (
        SELECT * FROM articles WHERE MATCH('rust async')
    )
    SELECT author, COUNT(*) FROM matched GROUP BY author
""")

# COUNT
n = client.execute("SELECT COUNT(*) FROM articles WHERE MATCH('python')").scalar()

# Combined fuzzy + exact
client.execute("""
    SELECT * FROM articles
    WHERE FUZZY_MATCH('machne lerning') AND category = 'AI'
""")
```

---

## Python API

The Python API provides direct FTS access without going through SQL. It is useful for programmatic control and when you need the raw ID arrays.

### init_fts

```python
client.init_fts(
    table_name: str = None,
    index_fields: Optional[List[str]] = None,
    lazy_load: bool = False,
    cache_size: int = 10000
) -> ApexClient
```

Initialize FTS for the current (or specified) table. Equivalent to `CREATE FTS INDEX ON table`.

> **Note:** `init_fts()` also registers the FTS engine with the global Rust registry, so subsequent SQL queries via PG Wire / Arrow Flight can use `MATCH()` on the same index.

```python
client.use_table("articles")
client.init_fts(index_fields=["title", "content"])
```

### search_text

```python
ids = client.search_text("query", table_name=None)  # → np.ndarray[int64]
```

### fuzzy_search_text

```python
ids = client.fuzzy_search_text("query", min_results=1, table_name=None)  # → np.ndarray[int64]
```

### search_and_retrieve / search_and_retrieve_top

```python
results = client.search_and_retrieve("query", limit=100, offset=0)     # → ResultView
top5    = client.search_and_retrieve_top("query", n=5)                  # → ResultView
```

### Lifecycle

```python
client.disable_fts()    # suspend (keep files)
client.drop_fts()       # drop (delete files)
stats = client.get_fts_stats()   # {'fts_enabled': True, 'doc_count': N, ...}
```

---

## PG Wire and Arrow Flight

Because FTS is implemented entirely in the Rust SQL executor, `MATCH()` and `FUZZY_MATCH()` work transparently over all server protocols:

```bash
# Start the combined server
apexbase-serve --dir /path/to/data
```

```python
# Python API — create the index
import apexbase
c = apexbase.ApexClient("/path/to/data")
c.create_table("articles")
c.execute("CREATE FTS INDEX ON articles (title, content)")
c.store([...])   # rows auto-indexed
c.close()
```

```python
# psycopg2 via PG Wire — query using MATCH()
import psycopg2
conn = psycopg2.connect(host="localhost", port=5432, dbname="default")
cur = conn.cursor()
cur.execute("SELECT * FROM articles WHERE MATCH('rust')")
rows = cur.fetchall()
```

```python
# Arrow Flight — query using MATCH()
import pyarrow.flight as flight
client = flight.connect("grpc://localhost:50051")
info = client.get_flight_info(flight.FlightDescriptor.for_command(
    b"SELECT title FROM articles WHERE MATCH('python')"
))
reader = client.do_get(info.endpoints[0].ticket)
table = reader.read_all()
```

No extra configuration required — the FTS registry is global within the server process.

---

## Lifecycle and Storage

```
{database_dir}/
├── articles.apex          ← table data
├── fts_config.json        ← FTS configuration (shared with Python API)
└── fts_indexes/
    ├── articles.nfts      ← NanoFTS index file
    └── articles.nfts.wal  ← Write-ahead log
```

**`fts_config.json` format:**

```json
{
  "articles": {
    "enabled": true,
    "index_fields": ["title", "content"],
    "config": { "lazy_load": false, "cache_size": 10000 }
  },
  "wiki": {
    "enabled": true,
    "index_fields": null,
    "config": { "lazy_load": true, "cache_size": 200000 }
  }
}
```

This file is written by both the Rust DDL handlers (`CREATE FTS INDEX`) and the Python `init_fts()` / `drop_fts()` methods, ensuring consistent state across interfaces.

---

## Configuration Options

| Option | Default | Description |
|--------|---------|-------------|
| `lazy_load` | `false` | Load index from disk only on first search. Reduces startup time for large indexes. |
| `cache_size` | `10000` | Number of terms held in the LRU cache. Larger = faster repeated queries, more RAM. |

Typical values:

| Use case | `cache_size` | `lazy_load` |
|----------|-------------|-------------|
| Small table (< 100K docs) | `10000` | `false` |
| Medium table (100K–1M docs) | `50000` | `false` |
| Large table (> 1M docs) | `100000`–`500000` | `true` |
| Log table (write-heavy) | `10000` | `true` |

---

## Performance Tips

1. **Index only the columns you search.** Specifying `(title, content)` instead of indexing all string columns reduces index size and improves write throughput.

2. **Use `MATCH()` for known-correct queries.** `FUZZY_MATCH()` is slower due to edit-distance computation — reserve it for user search boxes.

3. **Combine with secondary indexes.** FTS resolves to `_id IN (...)` which the executor filters efficiently. Add a B-Tree index on high-selectivity non-FTS columns to speed up compound predicates:
   ```sql
   CREATE INDEX idx_cat ON articles (category);
   SELECT * FROM articles WHERE MATCH('python') AND category = 'tutorial';
   ```

4. **Flush before searching.** After a bulk `store()`, call `client._storage._fts_flush()` (or just wait — the WAL is flushed automatically on close) to ensure all documents are searchable.

5. **Lazy load for large indexes.** Set `lazy_load=true` for indexes > 500 MB to avoid long startup times:
   ```sql
   CREATE FTS INDEX ON large_table WITH (lazy_load=true, cache_size=200000)
   ```

---

## Limitations

- **No back-fill on CREATE.** Running `CREATE FTS INDEX ON table` does not index existing rows. For existing data, use `client.init_fts()` via the Python API (which reads and re-indexes all existing rows).
- **Deletion sync.** Deleted rows are removed from the FTS index by the Python `_fts_remove()` path. SQL `DELETE` statements do not currently sync to the FTS index — re-create the index if stale results appear after heavy deletions.
- **Single-database scope.** `SHOW FTS INDEXES` lists indexes for the currently active database directory only.
- **No ranking scores.** `MATCH()` returns a boolean filter (matched / not matched). Document ranking scores are not exposed in SQL; use the Python `search_text()` API for ranked result ordering.
