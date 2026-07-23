//! Top-level database façade for coordinating query and storage services.

use std::collections::HashMap;
use std::io;
use std::path::Path;

use arrow::record_batch::RecordBatch;

use crate::data::Value;
use crate::query::{ApexExecutor, ApexResult, QuerySignature, SqlStatement};
use crate::storage::{engine, DurabilityLevel};

pub struct Database;

impl Database {
    #[inline]
    pub fn execute(sql: &str, base_dir: &Path, table_path: &Path) -> io::Result<ApexResult> {
        ApexExecutor::execute_with_base_dir(sql, base_dir, table_path)
    }

    #[inline]
    pub fn query(sql: &str, base_dir: &Path, table_path: &Path) -> io::Result<RecordBatch> {
        Self::execute(sql, base_dir, table_path)?.to_record_batch()
    }

    #[inline]
    pub(crate) fn execute_classified(
        sql: &str,
        signature: &QuerySignature,
        base_dir: &Path,
        table_path: &Path,
    ) -> io::Result<ApexResult> {
        ApexExecutor::execute_classified_with_base_dir(sql, signature, base_dir, table_path)
    }

    #[inline]
    pub fn execute_in_txn(
        txn_id: u64,
        statement: SqlStatement,
        base_dir: &Path,
        table_path: &Path,
    ) -> io::Result<ApexResult> {
        ApexExecutor::execute_in_txn(txn_id, statement, base_dir, table_path)
    }

    #[inline]
    pub fn commit_txn(txn_id: u64, base_dir: &Path, table_path: &Path) -> io::Result<ApexResult> {
        ApexExecutor::execute_commit_txn(txn_id, base_dir, table_path)
    }

    #[inline]
    pub fn rollback_txn(txn_id: u64) -> io::Result<ApexResult> {
        ApexExecutor::execute_rollback_txn(txn_id)
    }

    #[inline]
    pub fn execute_multi_with_txn(
        statements: Vec<SqlStatement>,
        base_dir: &Path,
        table_path: &Path,
        initial_txn_id: Option<u64>,
    ) -> io::Result<(ApexResult, Option<u64>)> {
        ApexExecutor::execute_multi_with_txn(statements, base_dir, table_path, initial_txn_id)
    }

    #[inline]
    pub(crate) fn copy_import(
        table_path: &Path,
        table_name: &str,
        file_path: &str,
        format: &str,
        options: &[(String, String)],
        base_dir: &Path,
        default_table_path: &Path,
    ) -> io::Result<ApexResult> {
        ApexExecutor::execute_copy_import(
            table_path,
            table_name,
            file_path,
            format,
            options,
            base_dir,
            default_table_path,
        )
    }

    #[inline]
    pub(crate) fn fts_backfill(
        base_dir: &Path,
        table: &str,
        fields: Option<&[String]>,
        manager: std::sync::Arc<crate::fts::FtsManager>,
    ) -> io::Result<usize> {
        ApexExecutor::fts_backfill_table(base_dir, table, fields, manager)
    }

    pub fn write(
        table_path: &Path,
        rows: &[HashMap<String, Value>],
        durability: DurabilityLevel,
    ) -> io::Result<Vec<u64>> {
        let ids = engine().write(table_path, rows, durability)?;
        Self::notify_indexes_after_write(table_path, &ids);
        Ok(ids)
    }

    #[inline]
    pub fn notify_indexes_after_write(table_path: &Path, ids: &[u64]) {
        ApexExecutor::notify_indexes_after_write(table_path, ids);
    }
}
