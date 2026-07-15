//! Isolated latency and resident-memory benchmark for internal Rust hot paths.
//!
//! Each case must run in a fresh process so allocator high-water marks from one
//! case do not hide allocations in the next:
//!   cargo run --release --example bench_rust_internal_memory -- --case row_insert
//!   cargo run --release --example bench_rust_internal_memory -- --case delta_insert
//!   cargo run --release --example bench_rust_internal_memory -- --case typed_append

use std::collections::HashMap;
use std::hint::black_box;
use std::time::Instant;

use apexbase::data::Value;
use apexbase::storage::{ColumnType, DurabilityLevel, StorageEngine, TableStorageBackend};
use serde::Serialize;

extern "C" {
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

pub const RUST_INTERNAL_MEMORY_CASES: &[&str] = &["row_insert", "delta_insert", "typed_append"];

#[derive(Serialize)]
struct Measurement<'a> {
    case: &'a str,
    rows: usize,
    latency_ns: u128,
    rss_before_bytes: usize,
    rss_after_bytes: usize,
    peak_rss_increment_bytes: usize,
    net_rss_bytes: isize,
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

fn row_batch(rows: usize) -> Vec<HashMap<String, Value>> {
    (0..rows)
        .map(|i| {
            HashMap::from([
                ("value".to_string(), Value::Int64(i as i64)),
                ("score".to_string(), Value::Float64(i as f64 * 0.5)),
                (
                    "name".to_string(),
                    Value::String(format!("row_{i:08}_{}", "x".repeat(48))),
                ),
                ("payload".to_string(), Value::Binary(vec![i as u8; 64])),
            ])
        })
        .collect()
}

fn measure<F>(case: &str, rows: usize, call: F) -> Measurement<'_>
where
    F: FnOnce(),
{
    let (before, peak_before) = process_memory();
    let start = Instant::now();
    call();
    let latency_ns = start.elapsed().as_nanos();
    let (after, peak_after) = process_memory();
    Measurement {
        case,
        rows,
        latency_ns,
        rss_before_bytes: before,
        rss_after_bytes: after,
        peak_rss_increment_bytes: peak_after.saturating_sub(peak_before),
        net_rss_bytes: after as isize - before as isize,
    }
}

fn row_insert(rows: usize) -> Result<Measurement<'static>, Box<dyn std::error::Error>> {
    let dir = tempfile::tempdir()?;
    let backend = TableStorageBackend::create(&dir.path().join("rows.apex"))?;
    let input = row_batch(rows);
    Ok(measure("row_insert", rows, || {
        black_box(backend.insert_rows(&input).unwrap());
    }))
}

fn delta_insert(rows: usize) -> Result<Measurement<'static>, Box<dyn std::error::Error>> {
    let dir = tempfile::tempdir()?;
    let path = dir.path().join("delta.apex");
    let backend = TableStorageBackend::create_with_schema_and_durability(
        &path,
        DurabilityLevel::Fast,
        &[
            ("value".to_string(), ColumnType::Int64),
            ("score".to_string(), ColumnType::Float64),
            ("name".to_string(), ColumnType::String),
            ("payload".to_string(), ColumnType::Binary),
        ],
    )?;
    backend.insert_rows(&row_batch(1))?;
    backend.save()?;
    let input = row_batch(rows);
    Ok(measure("delta_insert", rows, || {
        black_box(backend.insert_rows_to_delta(&input).unwrap());
    }))
}

fn typed_append(rows: usize) -> Result<Measurement<'static>, Box<dyn std::error::Error>> {
    let dir = tempfile::tempdir()?;
    let path = dir.path().join("typed.apex");
    let engine = StorageEngine::global();
    engine.write_typed(
        &path,
        HashMap::from([("value".to_string(), vec![0])]),
        HashMap::from([("score".to_string(), vec![0.0])]),
        HashMap::from([("name".to_string(), vec!["seed".to_string()])]),
        HashMap::new(),
        HashMap::new(),
        HashMap::new(),
        HashMap::new(),
        DurabilityLevel::Fast,
    )?;
    let int_columns = HashMap::from([("value".to_string(), (0..rows as i64).collect())]);
    let float_columns = HashMap::from([(
        "score".to_string(),
        (0..rows).map(|i| i as f64 * 0.5).collect(),
    )]);
    let string_columns = HashMap::from([(
        "name".to_string(),
        (0..rows).map(|i| format!("row_{i:08}")).collect(),
    )]);
    Ok(measure("typed_append", rows, || {
        black_box(
            engine
                .write_typed(
                    &path,
                    int_columns,
                    float_columns,
                    string_columns,
                    HashMap::new(),
                    HashMap::new(),
                    HashMap::new(),
                    HashMap::new(),
                    DurabilityLevel::Fast,
                )
                .unwrap(),
        );
    }))
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    let case = args
        .windows(2)
        .find(|args| args[0] == "--case")
        .map(|args| args[1].as_str())
        .ok_or("pass --case row_insert|delta_insert|typed_append")?;
    let rows = args
        .windows(2)
        .find(|args| args[0] == "--rows")
        .and_then(|args| args[1].parse().ok())
        .unwrap_or(50_000);

    let report = match case {
        "row_insert" => row_insert(rows)?,
        "delta_insert" => delta_insert(rows)?,
        "typed_append" => typed_append(rows)?,
        _ => return Err(format!("unknown case {case}").into()),
    };
    println!("{}", serde_json::to_string_pretty(&report)?);
    Ok(())
}
