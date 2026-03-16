# ApexBase Embedded Rust API

Use ApexBase as a high-performance embedded database directly from Rust — no Python, no FFI overhead.

## Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Core Types](#core-types)
- [Database Operations](#database-operations)
- [Table Operations](#table-operations)
- [SQL Queries](#sql-queries)
- [ResultSet](#resultset)
- [Schema Management](#schema-management)
- [Multi-Database Support](#multi-database-support)
- [Durability Levels](#durability-levels)
- [Error Handling](#error-handling)
- [Running the Example](#running-the-example)

---

## Installation

Add to your `Cargo.toml`:

```toml
[dependencies]
apexbase = { path = "path/to/ApexBase", default-features = false }
```

Disable the `python` feature (default) when using as a pure Rust library to avoid the PyO3 dependency:

```toml
apexbase = { path = "...", default-features = false }
```

---

## Quick Start

```rust
use apexbase::embedded::{ApexDB, Row};
use apexbase::data::Value;
use std::collections::HashMap;

fn main() -> apexbase::Result<()> {
    // Open a database directory
    let db = ApexDB::open("./my_database")?;

    // Create a table
    let table = db.create_table("users")?;

    // Insert a record
    let id = table.insert([
        ("name".to_string(), Value::String("Alice".to_string())),
        ("age".to_string(),  Value::Int64(30)),
        ("score".to_string(), Value::Float64(92.5)),
    ].into_iter().collect())?;

    println!("Inserted _id = {id}");

    // SQL query → Arrow RecordBatch
    let rs = table.execute("SELECT * FROM users WHERE age > 25")?;
    let batch = rs.to_record_batch()?;
    println!("{} rows", batch.num_rows());

    // Or convert to Vec<HashMap>
    let rs = table.execute("SELECT name, age FROM users ORDER BY age")?;
    let rows = rs.to_rows()?;
    for row in &rows {
        println!("{:?}", row.get("name"));
    }

    Ok(())
}
```

---

## Core Types

| Type | Description |
|------|-------------|
| `ApexDB` | Top-level database handle. `Clone`-able (`Arc` internally). |
| `ApexDBBuilder` | Builder for `ApexDB` with durability and drop-if-exists options. |
| `Table` | Table-scoped operations handle. `Clone`-able. |
| `ResultSet` | Result of a SQL query or DML statement. |
| `Row` | Type alias: `HashMap<String, Value>`. |
| `Value` | Enum for all supported column values (Int64, Float64, String, Bool, Binary, …). |
| `DurabilityLevel` | `Fast` / `Safe` / `Max` — controls fsync behavior. |
| `ColumnType` | Enum for schema column types (Int64, Float64, String, Bool, …). |

---

## Database Operations

### Opening a Database

```rust
use apexbase::embedded::ApexDB;

// Default: Fast durability, no drop
let db = ApexDB::open("./data")?;

// Builder pattern for more control
use apexbase::storage::DurabilityLevel;

let db = ApexDB::builder("./data")
    .durability(DurabilityLevel::Safe)
    .drop_if_exists(true)   // wipe existing .apex files first
    .build()?;
```

### Table DDL

```rust
// Create an empty table (schema inferred from first insert)
let table = db.create_table("events")?;

// Create with a predefined schema (avoids inference; guarantees column order)
use apexbase::storage::on_demand::ColumnType;

let table = db.create_table_with_schema("orders", &[
    ("order_id".to_string(), ColumnType::Int64),
    ("product".to_string(),  ColumnType::String),
    ("price".to_string(),    ColumnType::Float64),
    ("shipped".to_string(),  ColumnType::Bool),
])?;

// Open an existing table
let table = db.table("events")?;

// Drop a table
db.drop_table("events")?;

// List all tables
let names: Vec<String> = db.list_tables();
```

### Database-level SQL

```rust
// Execute SQL without a table context
let rs = db.execute("SELECT COUNT(*) FROM users")?;
let rs = db.execute("CREATE INDEX ON users (age)")?;
```

---

## Table Operations

### Insert

```rust
use apexbase::embedded::Row;
use apexbase::data::Value;

// Single record → returns assigned _id (u64)
let mut rec: Row = HashMap::new();
rec.insert("name".to_string(), Value::String("Bob".to_string()));
rec.insert("age".to_string(),  Value::Int64(25));
let id: u64 = table.insert(rec)?;

// Batch insert → returns Vec<u64> of _ids
let records: Vec<Row> = (0..1000).map(|i| {
    [
        ("n".to_string(), Value::Int64(i)),
        ("v".to_string(), Value::Float64(i as f64 * 1.5)),
    ].into_iter().collect()
}).collect();
let ids = table.insert_batch(&records)?;

// Insert from Arrow RecordBatch (zero-copy for V4 format)
use arrow::record_batch::RecordBatch;
let ids = table.insert_arrow(&batch)?;
```

### Delete

```rust
// Delete by _id → true if the row existed
let existed: bool = table.delete(id)?;

// Delete multiple rows → returns count deleted
let deleted: usize = table.delete_batch(&[1, 2, 3, 4, 5])?;

// SQL DELETE with WHERE clause
table.execute("DELETE FROM users WHERE age < 18")?;
```

### Replace (update whole row)

```rust
let mut updated: Row = HashMap::new();
updated.insert("name".to_string(),  Value::String("Bob 2.0".to_string()));
updated.insert("age".to_string(),   Value::Int64(26));
let existed: bool = table.replace(id, updated)?;
```

### Read

```rust
// Point lookup by _id → Option<Row>
let row: Option<Row> = table.retrieve(id)?;
if let Some(r) = row {
    println!("{:?}", r.get("name"));
}

// Batch lookup by _ids → Arrow RecordBatch
let batch = table.retrieve_many(&[1, 2, 3])?;

// Row count (O(1) for V4 format)
let n: u64 = table.count()?;

// Check existence
let exists: bool = table.exists(id)?;
```

---

## SQL Queries

The full SQL engine is available on `Table::execute`. The table is automatically set as the default context.

```rust
// SELECT with filter
let rs = table.execute("SELECT name, score FROM users WHERE score > 80")?;

// Aggregation
let rs = table.execute(
    "SELECT city, AVG(score) AS avg_score FROM users GROUP BY city HAVING COUNT(*) > 2"
)?;

// ORDER BY + LIMIT
let rs = table.execute(
    "SELECT * FROM users ORDER BY score DESC LIMIT 10"
)?;

// JOINs (use db.execute for cross-table queries)
let rs = db.execute(
    "SELECT u.name, o.product FROM users u JOIN orders o ON u.id = o.user_id"
)?;

// DML: UPDATE
table.execute("UPDATE users SET active = true WHERE last_login > '2024-01-01'")?;

// DML: DELETE with complex condition
table.execute("DELETE FROM users WHERE score < 50 AND age > 60")?;

// Window functions
let rs = table.execute(
    "SELECT name, ROW_NUMBER() OVER (PARTITION BY city ORDER BY score DESC) AS rn FROM users"
)?;

// Full-text search (requires FTS index)
let rs = table.execute("SELECT * FROM docs WHERE MATCH(body, 'embedded database')")?;
```

---

## ResultSet

```rust
let rs = table.execute("SELECT * FROM users WHERE age > 25")?;

// Number of rows
println!("{} rows", rs.num_rows());

// Column names
println!("{:?}", rs.columns());   // ["name", "age", "score", ...]

// Is there any data?
if rs.is_empty() { return Ok(()); }

// Convert to Arrow RecordBatch (zero-copy for Data variant)
let batch: RecordBatch = rs.to_record_batch()?;

// Convert to Vec<Row> (HashMap<String, Value>)
let rs = table.execute("SELECT * FROM users LIMIT 10")?;
let rows: Vec<Row> = rs.to_rows()?;

// Scalar result (COUNT, SUM, etc.)
let rs = table.execute("SELECT COUNT(*) FROM users")?;
if let Some(count) = rs.scalar() {
    println!("count = {count}");
}
```

---

## Schema Management

```rust
// Get schema as (column_name, DataType) pairs
let schema: Vec<(String, DataType)> = table.schema()?;

// Get column names only
let cols: Vec<String> = table.columns()?;

// Get type of a single column
use apexbase::data::DataType;
let dtype: Option<DataType> = table.column_type("age")?;

// Add a new column (all existing rows → NULL)
table.add_column("active", DataType::Bool)?;

// Drop a column
table.drop_column("temporary_col")?;

// Rename a column
table.rename_column("old_name", "new_name")?;
```

---

## Multi-Database Support

A single `ApexDB` can manage multiple named sub-databases (sub-directories):

```rust
// Switch to a named database (creates sub-dir if needed)
db.use_database("production")?;
let prod_users = db.create_table("users")?;

// Switch to another
db.use_database("staging")?;
let stage_users = db.table("users")?;   // separate file

// Revert to root
db.use_database("")?;
```

---

## Durability Levels

| Level | `fsync` | Use Case |
|-------|---------|----------|
| `Fast` (default) | Never | Bulk import, analytics, reconstructible data |
| `Safe` | On `flush()` | Most production write workloads |
| `Max` | Every write | Financial records, orders, critical data |

```rust
use apexbase::storage::DurabilityLevel;

let db = ApexDB::builder("./data")
    .durability(DurabilityLevel::Safe)
    .build()?;

// Manual flush for Safe durability
table.flush()?;
```

---

## Error Handling

All methods return `apexbase::Result<T>` which is `Result<T, ApexError>`.

```rust
use apexbase::ApexError;

match db.table("nonexistent") {
    Err(ApexError::TableNotFound(name)) => eprintln!("No table: {name}"),
    Err(ApexError::TableExists(name))   => eprintln!("Already exists: {name}"),
    Err(ApexError::Io(e))               => eprintln!("IO error: {e}"),
    Err(e)                              => eprintln!("Other: {e}"),
    Ok(table)                           => { /* use table */ }
}
```

---

## Running the Example

```bash
cargo run --example embedded --no-default-features
```

The example at `examples/embedded.rs` demonstrates:
- Opening a database with the builder
- Creating a table with predefined schema
- Inserting individual and batch records
- SQL queries (filter, aggregate, GROUP BY)
- Bulk insert (1 000 rows)
- Delete and replace by `_id`
- `retrieve_many` (Arrow batch read)
- Schema introspection
- Add / drop columns
- Multi-table operations
- Database-level SQL

---

## Performance Notes

- **Point lookups** (`retrieve`) use the V4 RCIX index — O(log n) per lookup, ~24µs.
- **Batch reads** (`retrieve_many`) use mmap for V4 files — single footer lock + one mmap slice per row-group.
- **Bulk insert** (`insert_batch`) routes through `engine.write()` which auto-selects delta write or full V4 append based on schema compatibility.
- **SQL queries** use the same Arrow-native query engine as the Python API — JIT compilation, vectorized filters, zone-map pruning.
- **Count** (`count()`) is O(1) for V4-format tables — reads the footer metadata only.
