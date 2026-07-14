//! Per-public-API latency and resident-memory benchmark for ApexBase's supported Rust API.
//!
//! Run with:
//!   cargo run --release --example bench_rust_public_api_memory --no-default-features -- --json report.json

use std::collections::HashMap;
use std::fs;
use std::hint::black_box;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

use apexbase::data::{DataType, Row as DataRow, Value};
use apexbase::embedded::{arrow_value_at, record_batch_to_rows, ApexDB, Row};
use apexbase::storage::{ColumnType, DurabilityLevel};
use arrow::array::{ArrayRef, Float64Array, Int64Array, StringArray};
use arrow::datatypes::{DataType as ArrowDataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use serde::Serialize;

extern "C" {
    // ApexBase already uses mimalloc through nanofts. This allocator-provided
    // API reports precise current/peak RSS on macOS without replacing the
    // process allocator (which would conflict with nanofts).
    fn mi_process_info(
        elapsed_msecs: *mut usize,
        user_msecs: *mut usize,
        system_msecs: *mut usize,
        current_rss: *mut usize,
        peak_rss: *mut usize,
        current_commit: *mut usize,
        peak_commit: *mut usize,
        page_faults: *mut usize,
    );
}

fn process_memory() -> (usize, usize) {
    let mut current_rss = 0;
    let mut peak_rss = 0;
    unsafe {
        mi_process_info(
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            &mut current_rss,
            &mut peak_rss,
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            std::ptr::null_mut(),
        );
    }
    (current_rss, peak_rss)
}

/// Supported Rust facade/core methods. The pytest coverage guard compares this
/// list with the public methods declared on these types.
pub const RUST_PUBLIC_API_CASES: &[&str] = &[
    "ApexDBBuilder.durability",
    "ApexDBBuilder.drop_if_exists",
    "ApexDBBuilder.build",
    "ApexDB.open",
    "ApexDB.builder",
    "ApexDB.create_table",
    "ApexDB.create_table_with_schema",
    "ApexDB.table",
    "ApexDB.drop_table",
    "ApexDB.list_tables",
    "ApexDB.register_temp_table",
    "ApexDB.drop_temp_table",
    "ApexDB.execute",
    "ApexDB.use_database",
    "ApexDB.invalidate_cache",
    "ApexDB.base_dir",
    "Table.insert",
    "Table.insert_batch",
    "Table.insert_arrow",
    "Table.replace",
    "Table.delete",
    "Table.delete_batch",
    "Table.retrieve",
    "Table.retrieve_many",
    "Table.count",
    "Table.exists",
    "Table.execute",
    "Table.schema",
    "Table.columns",
    "Table.column_type",
    "Table.add_column",
    "Table.rename_column",
    "Table.drop_column",
    "Table.flush",
    "Table.path",
    "ResultSet.to_record_batch",
    "ResultSet.to_rows",
    "ResultSet.num_rows",
    "ResultSet.scalar",
    "ResultSet.columns",
    "ResultSet.is_empty",
    "embedded.record_batch_to_rows",
    "embedded.arrow_value_at",
    "Row.new",
    "Row.from_fields",
    "Row.get",
    "Row.set",
    "Row.has_field",
    "Row.field_names",
    "Row.len",
    "Row.is_empty",
    "Row.to_bytes",
    "Row.from_bytes",
    "Row.to_json",
    "Row.from_json",
    "Row.merge",
    "Row.remove",
    "Row.iter",
    "Value.data_type",
    "Value.is_null",
    "Value.as_i64",
    "Value.as_f64",
    "Value.as_str",
    "Value.as_bool",
    "Value.to_string_value",
    "Value.to_bytes",
    "Value.from_bytes",
    "Value.infer_from_python_value",
    "Value.to_json_value",
    "DataType.fixed_size",
    "DataType.is_numeric",
    "DataType.is_variable_length",
    "DataType.is_decimal",
    "DataType.from_sql_type",
    "DataType.to_sql_type",
    "DurabilityLevel.from_str",
    "DurabilityLevel.as_str",
];

#[derive(Serialize)]
struct Measurement {
    api: &'static str,
    iterations: usize,
    latency_ns: u128,
    rss_before_bytes: usize,
    rss_after_bytes: usize,
    peak_rss_increment_bytes: usize,
    net_rss_bytes: isize,
}

fn measure<F>(api: &'static str, iterations: usize, mut call: F) -> Measurement
where
    F: FnMut(),
{
    let (before, peak_before) = process_memory();
    let start = Instant::now();
    for _ in 0..iterations {
        call();
    }
    let latency_ns = start.elapsed().as_nanos() / iterations as u128;
    let (after, peak_after) = process_memory();
    Measurement {
        api,
        iterations,
        latency_ns,
        rss_before_bytes: before,
        rss_after_bytes: after,
        peak_rss_increment_bytes: peak_after.saturating_sub(peak_before),
        net_rss_bytes: after as isize - before as isize,
    }
}

fn embedded_row(i: i64) -> Row {
    HashMap::from([
        ("value".to_string(), Value::Int64(i)),
        ("score".to_string(), Value::Float64(i as f64 * 0.5)),
        ("name".to_string(), Value::String(format!("row_{i}"))),
    ])
}

fn arrow_batch(rows: usize) -> RecordBatch {
    let schema = Arc::new(Schema::new(vec![
        Field::new("value", ArrowDataType::Int64, false),
        Field::new("score", ArrowDataType::Float64, false),
        Field::new("name", ArrowDataType::Utf8, false),
    ]));
    RecordBatch::try_new(
        schema,
        vec![
            Arc::new(Int64Array::from_iter_values(0..rows as i64)),
            Arc::new(Float64Array::from_iter_values(
                (0..rows).map(|i| i as f64 * 0.5),
            )),
            Arc::new(StringArray::from_iter_values(
                (0..rows).map(|i| format!("row_{i}")),
            )),
        ],
    )
    .unwrap()
}

fn push_case<F>(reports: &mut Vec<Measurement>, api: &'static str, iterations: usize, call: F)
where
    F: FnMut(),
{
    let report = measure(api, iterations, call);
    println!(
        "{:<38} {:>10.3} us  peak {:>9.1} KiB  net {:>9.1} KiB",
        report.api,
        report.latency_ns as f64 / 1_000.0,
        report.peak_rss_increment_bytes as f64 / 1024.0,
        report.net_rss_bytes as f64 / 1024.0,
    );
    reports.push(report);
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let json_path = std::env::args()
        .skip(1)
        .collect::<Vec<_>>()
        .windows(2)
        .find(|args| args[0] == "--json")
        .map(|args| PathBuf::from(&args[1]));

    let root = tempfile::tempdir()?;
    let db = ApexDB::open(root.path())?;
    let table = db.create_table_with_schema(
        "main",
        &[
            ("value".to_string(), ColumnType::Int64),
            ("score".to_string(), ColumnType::Float64),
            ("name".to_string(), ColumnType::String),
        ],
    )?;
    let rows: Vec<Row> = (0..2_000).map(embedded_row).collect();
    table.insert_batch(&rows)?;
    table.flush()?;
    let batch = arrow_batch(2_000);
    let csv = root.path().join("sample.csv");
    fs::write(&csv, "value,score,name\n1,0.5,a\n2,1.0,b\n")?;

    // Initialize lazy global state before any per-API allocation measurement.
    black_box(table.count()?);
    black_box(
        table
            .execute("SELECT * FROM main LIMIT 1")?
            .to_record_batch()?,
    );

    let mut reports = Vec::with_capacity(RUST_PUBLIC_API_CASES.len());

    push_case(&mut reports, "ApexDBBuilder.durability", 7, || {
        black_box(ApexDB::builder(root.path()).durability(DurabilityLevel::Safe));
    });
    push_case(&mut reports, "ApexDBBuilder.drop_if_exists", 7, || {
        black_box(ApexDB::builder(root.path()).drop_if_exists(false));
    });
    let build_dir = root.path().join("builder_db");
    push_case(&mut reports, "ApexDBBuilder.build", 1, || {
        black_box(ApexDB::builder(&build_dir).build().unwrap());
    });
    let open_dir = root.path().join("open_db");
    push_case(&mut reports, "ApexDB.open", 1, || {
        black_box(ApexDB::open(&open_dir).unwrap());
    });
    push_case(&mut reports, "ApexDB.builder", 7, || {
        black_box(ApexDB::builder(root.path()));
    });
    push_case(&mut reports, "ApexDB.create_table", 1, || {
        black_box(db.create_table("create_plain").unwrap());
    });
    push_case(&mut reports, "ApexDB.create_table_with_schema", 1, || {
        black_box(
            db.create_table_with_schema("create_schema", &[("x".into(), ColumnType::Int64)])
                .unwrap(),
        );
    });
    push_case(&mut reports, "ApexDB.table", 7, || {
        black_box(db.table("main").unwrap());
    });
    db.create_table("drop_me")?;
    push_case(&mut reports, "ApexDB.drop_table", 1, || {
        db.drop_table("drop_me").unwrap();
    });
    push_case(&mut reports, "ApexDB.list_tables", 7, || {
        black_box(db.list_tables());
    });
    push_case(&mut reports, "ApexDB.register_temp_table", 1, || {
        db.register_temp_table("csv_temp", csv.to_str().unwrap())
            .unwrap();
    });
    push_case(&mut reports, "ApexDB.drop_temp_table", 1, || {
        db.drop_temp_table("csv_temp").unwrap();
    });
    push_case(&mut reports, "ApexDB.execute", 7, || {
        black_box(db.execute("SELECT COUNT(*) FROM main").unwrap());
    });
    push_case(&mut reports, "ApexDB.use_database", 1, || {
        db.use_database("bench_subdb").unwrap();
    });
    db.use_database("")?;
    push_case(&mut reports, "ApexDB.invalidate_cache", 1, || {
        db.invalidate_cache();
    });
    push_case(&mut reports, "ApexDB.base_dir", 7, || {
        black_box(db.base_dir());
    });

    let insert_table = db.create_table("insert_one")?;
    push_case(&mut reports, "Table.insert", 1, || {
        black_box(insert_table.insert(embedded_row(1)).unwrap());
    });
    let insert_batch_table = db.create_table("insert_batch")?;
    push_case(&mut reports, "Table.insert_batch", 1, || {
        black_box(insert_batch_table.insert_batch(&rows).unwrap());
    });
    let insert_arrow_table = db.create_table("insert_arrow")?;
    push_case(&mut reports, "Table.insert_arrow", 1, || {
        black_box(insert_arrow_table.insert_arrow(&batch).unwrap());
    });
    let replace_table = db.create_table("replace")?;
    let replace_id = replace_table.insert(embedded_row(1))?;
    push_case(&mut reports, "Table.replace", 1, || {
        black_box(replace_table.replace(replace_id, embedded_row(2)).unwrap());
    });
    let delete_table = db.create_table("delete_one")?;
    let delete_id = delete_table.insert(embedded_row(1))?;
    push_case(&mut reports, "Table.delete", 1, || {
        black_box(delete_table.delete(delete_id).unwrap());
    });
    let delete_batch_table = db.create_table("delete_batch")?;
    let delete_ids = delete_batch_table.insert_batch(&rows[..100])?;
    push_case(&mut reports, "Table.delete_batch", 1, || {
        black_box(delete_batch_table.delete_batch(&delete_ids).unwrap());
    });
    push_case(&mut reports, "Table.retrieve", 7, || {
        black_box(table.retrieve(1).unwrap());
    });
    let ids: Vec<u64> = (1..=100).collect();
    push_case(&mut reports, "Table.retrieve_many", 7, || {
        black_box(table.retrieve_many(&ids).unwrap());
    });
    push_case(&mut reports, "Table.count", 7, || {
        black_box(table.count().unwrap());
    });
    push_case(&mut reports, "Table.exists", 7, || {
        black_box(table.exists(1).unwrap());
    });
    push_case(&mut reports, "Table.execute", 7, || {
        black_box(table.execute("SELECT * FROM main LIMIT 100").unwrap());
    });
    push_case(&mut reports, "Table.schema", 7, || {
        black_box(table.schema().unwrap());
    });
    push_case(&mut reports, "Table.columns", 7, || {
        black_box(table.columns().unwrap());
    });
    push_case(&mut reports, "Table.column_type", 7, || {
        black_box(table.column_type("value").unwrap());
    });
    let schema_table = db.create_table("schema_ops")?;
    schema_table.insert(HashMap::from([("x".into(), Value::Int64(1))]))?;
    push_case(&mut reports, "Table.add_column", 1, || {
        schema_table.add_column("added", DataType::Float64).unwrap();
    });
    push_case(&mut reports, "Table.rename_column", 1, || {
        schema_table.rename_column("added", "renamed").unwrap();
    });
    push_case(&mut reports, "Table.drop_column", 1, || {
        schema_table.drop_column("renamed").unwrap();
    });
    push_case(&mut reports, "Table.flush", 1, || {
        table.flush().unwrap();
    });
    push_case(&mut reports, "Table.path", 7, || {
        black_box(table.path());
    });

    let mut rs = Some(table.execute("SELECT * FROM main LIMIT 100")?);
    push_case(&mut reports, "ResultSet.to_record_batch", 1, || {
        black_box(rs.take().unwrap().to_record_batch().unwrap());
    });
    let mut rs = Some(table.execute("SELECT * FROM main LIMIT 100")?);
    push_case(&mut reports, "ResultSet.to_rows", 1, || {
        black_box(rs.take().unwrap().to_rows().unwrap());
    });
    let rs = table.execute("SELECT * FROM main LIMIT 100")?;
    push_case(&mut reports, "ResultSet.num_rows", 7, || {
        black_box(rs.num_rows());
    });
    let scalar_rs = table.execute("SELECT COUNT(*) FROM main")?;
    push_case(&mut reports, "ResultSet.scalar", 7, || {
        black_box(scalar_rs.scalar());
    });
    let rs = table.execute("SELECT * FROM main LIMIT 100")?;
    push_case(&mut reports, "ResultSet.columns", 7, || {
        black_box(rs.columns());
    });
    push_case(&mut reports, "ResultSet.is_empty", 7, || {
        black_box(rs.is_empty());
    });
    push_case(&mut reports, "embedded.record_batch_to_rows", 1, || {
        black_box(record_batch_to_rows(&batch).unwrap());
    });
    let array: ArrayRef = batch.column(2).clone();
    push_case(&mut reports, "embedded.arrow_value_at", 7, || {
        black_box(arrow_value_at(&array, 1_000));
    });

    let mut data_fields = HashMap::from([
        ("name".to_string(), Value::String("Alice".to_string())),
        ("age".to_string(), Value::Int64(30)),
    ]);
    push_case(&mut reports, "Row.new", 7, || {
        black_box(DataRow::new(1));
    });
    push_case(&mut reports, "Row.from_fields", 1, || {
        black_box(DataRow::from_fields(1, data_fields.clone()));
    });
    let mut data_row = DataRow::from_fields(1, data_fields.clone());
    push_case(&mut reports, "Row.get", 7, || {
        black_box(data_row.get("name"));
    });
    push_case(&mut reports, "Row.set", 1, || {
        data_row.set("city", "Shanghai");
    });
    push_case(&mut reports, "Row.has_field", 7, || {
        black_box(data_row.has_field("name"));
    });
    push_case(&mut reports, "Row.field_names", 7, || {
        black_box(data_row.field_names());
    });
    push_case(&mut reports, "Row.len", 7, || {
        black_box(data_row.len());
    });
    push_case(&mut reports, "Row.is_empty", 7, || {
        black_box(data_row.is_empty());
    });
    let row_bytes = data_row.to_bytes();
    push_case(&mut reports, "Row.to_bytes", 7, || {
        black_box(data_row.to_bytes());
    });
    push_case(&mut reports, "Row.from_bytes", 7, || {
        black_box(DataRow::from_bytes(&row_bytes));
    });
    let row_json = data_row.to_json();
    push_case(&mut reports, "Row.to_json", 7, || {
        black_box(data_row.to_json());
    });
    push_case(&mut reports, "Row.from_json", 7, || {
        black_box(DataRow::from_json(1, &row_json));
    });
    let merge_row = DataRow::from_fields(
        2,
        HashMap::from([("score".to_string(), Value::Float64(99.0))]),
    );
    push_case(&mut reports, "Row.merge", 1, || {
        data_row.merge(&merge_row);
    });
    data_fields.insert("remove".to_string(), Value::Bool(true));
    let mut remove_row = DataRow::from_fields(1, data_fields);
    push_case(&mut reports, "Row.remove", 1, || {
        black_box(remove_row.remove("remove"));
    });
    push_case(&mut reports, "Row.iter", 7, || {
        black_box(data_row.iter().count());
    });

    let value = Value::String("apexbase".repeat(64));
    push_case(&mut reports, "Value.data_type", 7, || {
        black_box(value.data_type());
    });
    push_case(&mut reports, "Value.is_null", 7, || {
        black_box(value.is_null());
    });
    push_case(&mut reports, "Value.as_i64", 7, || {
        black_box(Value::Int64(42).as_i64());
    });
    push_case(&mut reports, "Value.as_f64", 7, || {
        black_box(Value::Float64(42.5).as_f64());
    });
    push_case(&mut reports, "Value.as_str", 7, || {
        black_box(value.as_str());
    });
    push_case(&mut reports, "Value.as_bool", 7, || {
        black_box(Value::Bool(true).as_bool());
    });
    push_case(&mut reports, "Value.to_string_value", 7, || {
        black_box(value.to_string_value());
    });
    let value_bytes = value.to_bytes();
    push_case(&mut reports, "Value.to_bytes", 7, || {
        black_box(value.to_bytes());
    });
    push_case(&mut reports, "Value.from_bytes", 7, || {
        black_box(Value::from_bytes(&value_bytes));
    });
    let json_value = serde_json::json!({"name": "apex", "values": [1, 2, 3]});
    push_case(&mut reports, "Value.infer_from_python_value", 7, || {
        black_box(Value::infer_from_python_value(&json_value));
    });
    push_case(&mut reports, "Value.to_json_value", 7, || {
        black_box(value.to_json_value());
    });

    push_case(&mut reports, "DataType.fixed_size", 7, || {
        black_box(DataType::Int64.fixed_size());
    });
    push_case(&mut reports, "DataType.is_numeric", 7, || {
        black_box(DataType::Int64.is_numeric());
    });
    push_case(&mut reports, "DataType.is_variable_length", 7, || {
        black_box(DataType::String.is_variable_length());
    });
    push_case(&mut reports, "DataType.is_decimal", 7, || {
        black_box(DataType::Decimal.is_decimal());
    });
    push_case(&mut reports, "DataType.from_sql_type", 7, || {
        black_box(DataType::from_sql_type("BIGINT"));
    });
    push_case(&mut reports, "DataType.to_sql_type", 7, || {
        black_box(DataType::Int64.to_sql_type());
    });
    push_case(&mut reports, "DurabilityLevel.from_str", 7, || {
        black_box(DurabilityLevel::from_str("safe"));
    });
    push_case(&mut reports, "DurabilityLevel.as_str", 7, || {
        black_box(DurabilityLevel::Safe.as_str());
    });

    let measured: Vec<_> = reports.iter().map(|r| r.api).collect();
    assert_eq!(
        measured, RUST_PUBLIC_API_CASES,
        "manifest and execution order differ"
    );
    if let Some(path) = json_path {
        fs::write(path, serde_json::to_vec_pretty(&reports)?)?;
    }
    Ok(())
}
