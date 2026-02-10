//! PostgreSQL wire protocol handler for ApexBase
//!
//! Bridges pgwire's SimpleQueryHandler to ApexBase's ApexExecutor.

use std::fmt::Debug;
use std::path::PathBuf;
use std::sync::Arc;

use async_trait::async_trait;
use futures::stream;
use futures::Sink;

use arrow::array::*;
use arrow::datatypes::{DataType as ArrowDataType, Int32Type, UInt32Type};
use arrow::record_batch::RecordBatch;

use pgwire::api::auth::noop::NoopStartupHandler;
use pgwire::api::copy::CopyHandler;
use pgwire::api::portal::Portal;
use pgwire::api::query::{ExtendedQueryHandler, SimpleQueryHandler};
use pgwire::api::results::{
    DataRowEncoder, DescribePortalResponse, DescribeResponse, DescribeStatementResponse,
    FieldFormat, FieldInfo, QueryResponse, Response, Tag,
};
use pgwire::api::stmt::{NoopQueryParser, StoredStatement};
use pgwire::api::store::PortalStore;
use pgwire::api::{ClientInfo, ClientPortalStore};
use pgwire::error::{PgWireError, PgWireResult};
use pgwire::messages::{PgWireBackendMessage, PgWireFrontendMessage};

use crate::query::{ApexExecutor, ApexResult};
use super::pg_catalog;
use super::types::schema_to_field_info;

/// ApexBase handler for PostgreSQL wire protocol
pub struct ApexBaseHandler {
    /// Base directory containing .apex database files
    base_dir: PathBuf,
    /// Default table path (apexbase.apex in base_dir)
    default_table_path: PathBuf,
}

impl ApexBaseHandler {
    pub fn new(base_dir: PathBuf) -> Self {
        let default_table_path = base_dir.join("apexbase.apex");
        Self {
            base_dir,
            default_table_path,
        }
    }

    /// Execute a SQL query against ApexBase and return a RecordBatch
    fn execute_query(&self, sql: &str) -> Result<RecordBatch, String> {
        // First check if it's a catalog/metadata query
        if let Some(batch) = pg_catalog::try_handle_catalog_query(sql, &self.base_dir) {
            return Ok(batch);
        }

        // Execute via ApexExecutor
        let result = ApexExecutor::execute_with_base_dir(sql, &self.base_dir, &self.default_table_path)
            .map_err(|e| e.to_string())?;

        result.to_record_batch().map_err(|e| e.to_string())
    }

    /// Encode a RecordBatch row data into Vec of encoded rows
    fn encode_batch_rows(batch: &RecordBatch, field_infos: &Arc<Vec<FieldInfo>>) -> PgWireResult<Vec<PgWireResult<pgwire::messages::data::DataRow>>> {
        let mut rows = Vec::with_capacity(batch.num_rows());
        let num_cols = batch.num_columns();

        for row_idx in 0..batch.num_rows() {
            let mut encoder = DataRowEncoder::new(field_infos.clone());
            for col_idx in 0..num_cols {
                let col = batch.column(col_idx);
                encode_arrow_value(&mut encoder, col, row_idx)?;
            }
            rows.push(encoder.finish());
        }
        Ok(rows)
    }

    /// Check if SQL is a write operation
    fn is_write_op(sql: &str) -> bool {
        let upper = sql.trim().to_uppercase();
        upper.starts_with("INSERT")
            || upper.starts_with("UPDATE")
            || upper.starts_with("DELETE")
            || upper.starts_with("CREATE")
            || upper.starts_with("DROP")
            || upper.starts_with("ALTER")
            || upper.starts_with("TRUNCATE")
    }
}

#[async_trait]
impl NoopStartupHandler for ApexBaseHandler {
    async fn post_startup<C>(
        &self,
        client: &mut C,
        _message: PgWireFrontendMessage,
    ) -> PgWireResult<()>
    where
        C: ClientInfo + Sink<PgWireBackendMessage> + Unpin + Send,
        C::Error: Debug,
        PgWireError: From<<C as Sink<PgWireBackendMessage>>::Error>,
    {
        log::info!(
            "Client connected from {}",
            client.socket_addr(),
        );
        Ok(())
    }
}

#[async_trait]
impl SimpleQueryHandler for ApexBaseHandler {
    async fn do_query<'a, 'b: 'a, C>(
        &'b self,
        _client: &mut C,
        query: &'a str,
    ) -> PgWireResult<Vec<Response<'a>>>
    where
        C: ClientInfo + Sink<PgWireBackendMessage> + Unpin + Send + Sync,
        C::Error: Debug,
        PgWireError: From<<C as Sink<PgWireBackendMessage>>::Error>,
    {
        log::debug!("Simple query: {}", query);

        // Handle multiple statements separated by semicolons
        let statements: Vec<&str> = query
            .split(';')
            .map(|s| s.trim())
            .filter(|s| !s.is_empty())
            .collect();

        let mut responses = Vec::new();

        for sql in statements {
            match self.execute_query(sql) {
                Ok(batch) => {
                    if Self::is_write_op(sql) {
                        let rows = batch.num_rows();
                        let tag = if sql.trim().to_uppercase().starts_with("INSERT") {
                            Tag::new("INSERT").with_oid(0).with_rows(rows)
                        } else if sql.trim().to_uppercase().starts_with("DELETE") {
                            Tag::new("DELETE").with_rows(rows)
                        } else if sql.trim().to_uppercase().starts_with("UPDATE") {
                            Tag::new("UPDATE").with_rows(rows)
                        } else {
                            Tag::new("OK")
                        };
                        responses.push(Response::Execution(tag));
                    } else {
                        let schema = batch.schema();
                        let field_infos: Vec<FieldInfo> = schema_to_field_info(&schema);
                        let field_infos = Arc::new(field_infos);

                        let rows = Self::encode_batch_rows(&batch, &field_infos)?;
                        let data_row_stream = stream::iter(rows);
                        responses.push(Response::Query(QueryResponse::new(
                            field_infos,
                            data_row_stream,
                        )));
                    }
                }
                Err(msg) => {
                    return Err(PgWireError::UserError(Box::new(
                        pgwire::error::ErrorInfo::new(
                            "ERROR".to_string(),
                            "42000".to_string(),
                            msg,
                        ),
                    )));
                }
            }
        }

        if responses.is_empty() {
            responses.push(Response::EmptyQuery);
        }

        Ok(responses)
    }
}

// CopyHandler — use default no-op implementations
#[async_trait]
impl CopyHandler for ApexBaseHandler {}

// Extended Query Protocol support
// This allows clients like psql, DBeaver, etc. to use prepared statements.
#[async_trait]
impl ExtendedQueryHandler for ApexBaseHandler {
    type Statement = String;
    type QueryParser = NoopQueryParser;

    fn query_parser(&self) -> Arc<Self::QueryParser> {
        Arc::new(NoopQueryParser::new())
    }

    async fn do_describe_statement<C>(
        &self,
        _client: &mut C,
        target: &StoredStatement<Self::Statement>,
    ) -> PgWireResult<DescribeStatementResponse>
    where
        C: ClientInfo + ClientPortalStore + Sink<PgWireBackendMessage> + Unpin + Send + Sync,
        C::PortalStore: PortalStore<Statement = Self::Statement>,
        C::Error: Debug,
        PgWireError: From<<C as Sink<PgWireBackendMessage>>::Error>,
    {
        let sql = &target.statement;
        if sql.trim().is_empty() || Self::is_write_op(sql) {
            return Ok(DescribeStatementResponse::new(vec![], vec![]));
        }
        // Execute query to discover schema for the JDBC driver
        match self.execute_query(sql) {
            Ok(batch) => {
                let fields = schema_to_field_info(&batch.schema());
                Ok(DescribeStatementResponse::new(vec![], fields))
            }
            Err(_) => Ok(DescribeStatementResponse::new(vec![], vec![])),
        }
    }

    async fn do_describe_portal<C>(
        &self,
        _client: &mut C,
        target: &Portal<Self::Statement>,
    ) -> PgWireResult<DescribePortalResponse>
    where
        C: ClientInfo + ClientPortalStore + Sink<PgWireBackendMessage> + Unpin + Send + Sync,
        C::PortalStore: PortalStore<Statement = Self::Statement>,
        C::Error: Debug,
        PgWireError: From<<C as Sink<PgWireBackendMessage>>::Error>,
    {
        let sql = &target.statement.statement;
        if sql.trim().is_empty() || Self::is_write_op(sql) {
            return Ok(DescribePortalResponse::new(vec![]));
        }
        match self.execute_query(sql) {
            Ok(batch) => {
                let fields = schema_to_field_info(&batch.schema());
                Ok(DescribePortalResponse::new(fields))
            }
            Err(_) => Ok(DescribePortalResponse::new(vec![])),
        }
    }

    async fn do_query<'a, 'b: 'a, C>(
        &'b self,
        _client: &mut C,
        portal: &'a Portal<Self::Statement>,
        _max_rows: usize,
    ) -> PgWireResult<Response<'a>>
    where
        C: ClientInfo + ClientPortalStore + Sink<PgWireBackendMessage> + Unpin + Send + Sync,
        C::PortalStore: PortalStore<Statement = Self::Statement>,
        C::Error: Debug,
        PgWireError: From<<C as Sink<PgWireBackendMessage>>::Error>,
    {
        let sql = &portal.statement.statement;
        log::debug!("Extended query: {}", sql);

        if sql.trim().is_empty() {
            return Ok(Response::EmptyQuery);
        }

        match self.execute_query(sql) {
            Ok(batch) => {
                if Self::is_write_op(sql) {
                    let rows = batch.num_rows();
                    let tag = if sql.trim().to_uppercase().starts_with("INSERT") {
                        Tag::new("INSERT").with_oid(0).with_rows(rows)
                    } else if sql.trim().to_uppercase().starts_with("DELETE") {
                        Tag::new("DELETE").with_rows(rows)
                    } else if sql.trim().to_uppercase().starts_with("UPDATE") {
                        Tag::new("UPDATE").with_rows(rows)
                    } else {
                        Tag::new("OK")
                    };
                    Ok(Response::Execution(tag))
                } else {
                    let schema = batch.schema();
                    let field_infos: Vec<FieldInfo> = schema_to_field_info(&schema);
                    let field_infos = Arc::new(field_infos);

                    let rows = Self::encode_batch_rows(&batch, &field_infos)?;
                    let data_row_stream = stream::iter(rows);
                    Ok(Response::Query(QueryResponse::new(
                        field_infos,
                        data_row_stream,
                    )))
                }
            }
            Err(msg) => Err(PgWireError::UserError(Box::new(
                pgwire::error::ErrorInfo::new(
                    "ERROR".to_string(),
                    "42000".to_string(),
                    msg,
                ),
            ))),
        }
    }

}

// ============================================================================
// Arrow value encoding helpers
// ============================================================================

/// Encode a single Arrow array value at a given row index into the DataRowEncoder
fn encode_arrow_value(
    encoder: &mut DataRowEncoder,
    array: &ArrayRef,
    row: usize,
) -> PgWireResult<()> {
    if array.is_null(row) {
        encoder.encode_field(&None::<&str>)?;
        return Ok(());
    }

    match array.data_type() {
        ArrowDataType::Boolean => {
            let arr = array.as_any().downcast_ref::<BooleanArray>().unwrap();
            encoder.encode_field(&arr.value(row))?;
        }
        ArrowDataType::Int8 => {
            let arr = array.as_any().downcast_ref::<Int8Array>().unwrap();
            encoder.encode_field(&(arr.value(row) as i16))?;
        }
        ArrowDataType::Int16 => {
            let arr = array.as_any().downcast_ref::<Int16Array>().unwrap();
            encoder.encode_field(&arr.value(row))?;
        }
        ArrowDataType::Int32 => {
            let arr = array.as_any().downcast_ref::<Int32Array>().unwrap();
            encoder.encode_field(&arr.value(row))?;
        }
        ArrowDataType::Int64 => {
            let arr = array.as_any().downcast_ref::<Int64Array>().unwrap();
            encoder.encode_field(&arr.value(row))?;
        }
        ArrowDataType::UInt8 => {
            let arr = array.as_any().downcast_ref::<UInt8Array>().unwrap();
            encoder.encode_field(&(arr.value(row) as i16))?;
        }
        ArrowDataType::UInt16 => {
            let arr = array.as_any().downcast_ref::<UInt16Array>().unwrap();
            encoder.encode_field(&(arr.value(row) as i32))?;
        }
        ArrowDataType::UInt32 => {
            let arr = array.as_any().downcast_ref::<UInt32Array>().unwrap();
            encoder.encode_field(&(arr.value(row) as i64))?;
        }
        ArrowDataType::UInt64 => {
            let arr = array.as_any().downcast_ref::<UInt64Array>().unwrap();
            encoder.encode_field(&arr.value(row).to_string())?;
        }
        ArrowDataType::Float32 => {
            let arr = array.as_any().downcast_ref::<Float32Array>().unwrap();
            encoder.encode_field(&arr.value(row))?;
        }
        ArrowDataType::Float64 => {
            let arr = array.as_any().downcast_ref::<Float64Array>().unwrap();
            encoder.encode_field(&arr.value(row))?;
        }
        ArrowDataType::Utf8 => {
            let arr = array.as_any().downcast_ref::<StringArray>().unwrap();
            encoder.encode_field(&arr.value(row))?;
        }
        ArrowDataType::LargeUtf8 => {
            let arr = array.as_any().downcast_ref::<LargeStringArray>().unwrap();
            encoder.encode_field(&arr.value(row))?;
        }
        ArrowDataType::Binary => {
            let arr = array.as_any().downcast_ref::<BinaryArray>().unwrap();
            encoder.encode_field(&arr.value(row))?;
        }
        ArrowDataType::LargeBinary => {
            let arr = array.as_any().downcast_ref::<LargeBinaryArray>().unwrap();
            encoder.encode_field(&arr.value(row))?;
        }
        ArrowDataType::Dictionary(_, _) => {
            if let Some(dict) = array.as_any().downcast_ref::<DictionaryArray<UInt32Type>>() {
                let values: &StringArray = dict.values().as_any().downcast_ref::<StringArray>().unwrap();
                let key = dict.keys().value(row) as usize;
                encoder.encode_field(&values.value(key))?;
            } else if let Some(dict) = array.as_any().downcast_ref::<DictionaryArray<Int32Type>>() {
                let values: &StringArray = dict.values().as_any().downcast_ref::<StringArray>().unwrap();
                let key = dict.keys().value(row) as usize;
                encoder.encode_field(&values.value(key))?;
            } else {
                encoder.encode_field(&"<dict>")?;
            }
        }
        _ => {
            let formatted = arrow::util::display::array_value_to_string(array, row)
                .unwrap_or_else(|_| "NULL".to_string());
            encoder.encode_field(&formatted)?;
        }
    }

    Ok(())
}
