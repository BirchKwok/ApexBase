//! ApexBase Embedded Rust API — comprehensive usage example
//!
//! Run with:
//!   cargo run --example embedded --no-default-features

use apexbase::data::Value;
use apexbase::embedded::{ApexDB, Row};
use apexbase::storage::DurabilityLevel;
use apexbase::storage::on_demand::ColumnType;
use std::collections::HashMap;

fn main() -> apexbase::Result<()> {
    let tmp = tempfile::tempdir().expect("tempdir");

    println!("=== ApexBase Embedded Rust API Demo ===\n");

    // ── 1. Open a database ────────────────────────────────────────────────────
    let db = ApexDB::builder(tmp.path())
        .durability(DurabilityLevel::Fast)
        .drop_if_exists(true)
        .build()?;

    println!("Database opened at: {}", db.base_dir().display());

    // ── 2. Create a table with a predefined schema ────────────────────────────
    let users = db.create_table_with_schema("users", &[
        ("name".to_string(),  ColumnType::String),
        ("age".to_string(),   ColumnType::Int64),
        ("score".to_string(), ColumnType::Float64),
        ("city".to_string(),  ColumnType::String),
    ])?;
    println!("Created table '{}' at {}", users.name, users.path().display());

    // ── 3. Insert individual rows ─────────────────────────────────────────────
    let sample: &[(&str, i64, f64, &str)] = &[
        ("Alice",  30, 92.5, "New York"),
        ("Bob",    25, 78.3, "London"),
        ("Carol",  35, 88.0, "New York"),
        ("Dave",   28, 95.1, "Tokyo"),
        ("Eve",    22, 61.7, "London"),
    ];

    let mut ids = Vec::new();
    for (name, age, score, city) in sample {
        let mut r = Row::new();
        r.insert("name".to_string(),  Value::String(name.to_string()));
        r.insert("age".to_string(),   Value::Int64(*age));
        r.insert("score".to_string(), Value::Float64(*score));
        r.insert("city".to_string(),  Value::String(city.to_string()));
        ids.push(users.insert(r)?);
    }
    println!("Inserted {} rows, _ids = {:?}", ids.len(), ids);

    // ── 4. Row count ──────────────────────────────────────────────────────────
    println!("Row count: {}", users.count()?);

    // ── 5. Retrieve a single row by ID ────────────────────────────────────────
    let row = users.retrieve(ids[0])?.expect("row must exist");
    println!("Retrieve _id={}: name={:?}", ids[0], row.get("name"));

    // ── 6. SQL query → RecordBatch ────────────────────────────────────────────
    let rs = users.execute("SELECT name, age, score FROM users WHERE age > 25 ORDER BY score DESC")?;
    let batch = rs.to_record_batch()?;
    println!("SQL result: {} rows × {} columns", batch.num_rows(), batch.num_columns());
    for i in 0..batch.num_columns() {
        print!("  col[{}]={}", i, batch.schema().field(i).name());
    }
    println!();

    // ── 7. Aggregate query ────────────────────────────────────────────────────
    let rs = users.execute("SELECT city, COUNT(*) AS n FROM users GROUP BY city ORDER BY n DESC")?;
    println!("GROUP BY result ({} rows):", rs.num_rows());
    let rows = rs.to_rows()?;
    for r in &rows {
        println!("  city={:?}  n={:?}", r.get("city"), r.get("n"));
    }

    // ── 8. Bulk insert ────────────────────────────────────────────────────────
    let bulk: Vec<Row> = (0..1_000i64).map(|i| {
        let mut r = HashMap::new();
        r.insert("name".to_string(),  Value::String(format!("user_{}", i)));
        r.insert("age".to_string(),   Value::Int64(20 + i % 50));
        r.insert("score".to_string(), Value::Float64(50.0 + (i % 50) as f64));
        r.insert("city".to_string(),  Value::String(if i % 2 == 0 { "A".to_string() } else { "B".to_string() }));
        r
    }).collect();
    let bulk_ids = users.insert_batch(&bulk)?;
    println!("Bulk-inserted {} rows", bulk_ids.len());
    println!("Total rows now: {}", users.count()?);

    // ── 9. Delete ─────────────────────────────────────────────────────────────
    let deleted = users.delete(ids[0])?;
    println!("Delete _id={}: {}", ids[0], deleted);
    println!("Rows after delete: {}", users.count()?);

    // ── 10. Replace (update a whole row) ─────────────────────────────────────
    let mut updated = Row::new();
    updated.insert("name".to_string(),  Value::String("Bob Updated".to_string()));
    updated.insert("age".to_string(),   Value::Int64(26));
    updated.insert("score".to_string(), Value::Float64(99.0));
    updated.insert("city".to_string(),  Value::String("Berlin".to_string()));
    let replaced = users.replace(ids[1], updated)?;
    println!("Replace _id={}: {}", ids[1], replaced);

    // ── 11. retrieve_many ─────────────────────────────────────────────────────
    let many = users.retrieve_many(&ids[1..=2])?;
    println!("retrieve_many: {} rows", many.num_rows());

    // ── 12. Schema introspection ──────────────────────────────────────────────
    println!("Schema: {:?}", users.columns()?);

    // ── 13. Add / drop column on a fresh table ────────────────────────────────
    let schema_demo = db.create_table("schema_demo")?;
    let mut sd = Row::new();
    sd.insert("x".to_string(), Value::Int64(1));
    schema_demo.insert(sd)?;
    schema_demo.add_column("y", apexbase::data::DataType::Float64)?;
    println!("After add_column: {:?}", schema_demo.columns()?);
    schema_demo.drop_column("y")?;
    println!("After drop_column: {:?}", schema_demo.columns()?);
    db.drop_table("schema_demo")?;

    // ── 14. Multi-table: create a second table ────────────────────────────────
    let orders = db.create_table("orders")?;
    let mut o = Row::new();
    o.insert("product".to_string(), Value::String("Widget".to_string()));
    o.insert("amount".to_string(),  Value::Float64(19.99));
    orders.insert(o)?;
    println!("Tables: {:?}", db.list_tables());

    // ── 15. db.execute (cross-table SQL) ─────────────────────────────────────
    let rs = db.execute("SELECT COUNT(*) FROM users")?;
    println!("db.execute COUNT(*) users = {:?}", rs.scalar());

    // ── 16. Drop table ────────────────────────────────────────────────────────
    db.drop_table("orders")?;
    println!("After drop_table: {:?}", db.list_tables());

    println!("\n=== All demo steps completed successfully ===");
    Ok(())
}
