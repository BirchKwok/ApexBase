use super::*;

impl SqlExecutor {
    pub(crate) fn execute_union(union: crate::query::UnionStatement, table: &mut ColumnTable) -> Result<SqlResult, ApexError> {
        if table.has_pending_writes() {
            table.flush_write_buffer();
        }

        fn exec_select_legacy(stmt: SelectStatement, table: &mut ColumnTable) -> Result<SqlResult, ApexError> {
            use crate::data::Value;
            if table.has_pending_writes() {
                table.flush_write_buffer();
            }

            if !stmt.joins.is_empty() {
                return Err(ApexError::QueryParseError(
                    "JOIN requires multi-table execution".to_string(),
                ));
            }
            if !stmt.group_by.is_empty() {
                return Err(ApexError::QueryParseError(
                    "GROUP BY in UNION branch is not supported yet".to_string(),
                ));
            }
            if stmt.having.is_some() {
                return Err(ApexError::QueryParseError(
                    "HAVING in UNION branch is not supported yet".to_string(),
                ));
            }
            if stmt.distinct {
                return Err(ApexError::QueryParseError(
                    "DISTINCT in UNION branch is not supported yet".to_string(),
                ));
            }
            if !stmt.order_by.is_empty() || stmt.limit.is_some() || stmt.offset.is_some() {
                return Err(ApexError::QueryParseError(
                    "ORDER BY/LIMIT/OFFSET must be applied to UNION result".to_string(),
                ));
            }

            let matching_indices: Vec<usize> = if let Some(ref where_expr) = stmt.where_clause {
                SqlExecutor::evaluate_where(where_expr, table)?
            } else {
                let deleted = table.deleted_ref();
                let row_count = table.get_row_count();
                (0..row_count).filter(|&i| !deleted.get(i)).collect()
            };

            let (result_columns, column_indices) = SqlExecutor::resolve_columns(&stmt.columns, table)?;
            let mut rows: Vec<Vec<Value>> = Vec::with_capacity(matching_indices.len().min(10000));
            for row_idx in matching_indices.iter() {
                let mut row_values: Vec<Value> = Vec::with_capacity(result_columns.len());
                for (col_name, col_idx) in column_indices.iter() {
                    if col_name == "_id" {
                        row_values.push(Value::Int64(*row_idx as i64));
                    } else if let Some(idx) = col_idx {
                        row_values.push(table.columns_ref()[*idx].get(*row_idx).unwrap_or(Value::Null));
                    } else {
                        row_values.push(Value::Null);
                    }
                }
                rows.push(row_values);
            }
            Ok(SqlResult::new(result_columns, rows))
        }

        fn exec_one(stmt: crate::query::SqlStatement, table: &mut ColumnTable) -> Result<SqlResult, ApexError> {
            match stmt {
                crate::query::SqlStatement::Select(sel) => exec_select_legacy(sel, table),
                crate::query::SqlStatement::Union(u) => SqlExecutor::execute_union(u, table),
                crate::query::SqlStatement::CreateView { .. }
                | crate::query::SqlStatement::DropView { .. } => Err(ApexError::QueryParseError(
                    "CREATE/DROP VIEW are only supported in multi-statement execution".to_string(),
                )),
            }
        }

        let left = exec_one(*union.left, table)?;
        let right = exec_one(*union.right, table)?;
        SqlExecutor::merge_union_results(left, right, union.all, &union.order_by, union.limit, union.offset)
    }

    pub(crate) fn execute_union_with_tables(
        union: crate::query::UnionStatement,
        tables: &mut HashMap<String, ColumnTable>,
        default_table: &str,
    ) -> Result<SqlResult, ApexError> {
        fn exec_select_legacy(stmt: SelectStatement, table: &mut ColumnTable) -> Result<SqlResult, ApexError> {
            use crate::data::Value;
            if table.has_pending_writes() {
                table.flush_write_buffer();
            }

            if !stmt.joins.is_empty() {
                return Err(ApexError::QueryParseError(
                    "JOIN requires multi-table execution".to_string(),
                ));
            }
            if !stmt.group_by.is_empty() {
                return Err(ApexError::QueryParseError(
                    "GROUP BY in UNION branch is not supported yet".to_string(),
                ));
            }
            if stmt.having.is_some() {
                return Err(ApexError::QueryParseError(
                    "HAVING in UNION branch is not supported yet".to_string(),
                ));
            }
            if stmt.distinct {
                return Err(ApexError::QueryParseError(
                    "DISTINCT in UNION branch is not supported yet".to_string(),
                ));
            }
            if !stmt.order_by.is_empty() || stmt.limit.is_some() || stmt.offset.is_some() {
                return Err(ApexError::QueryParseError(
                    "ORDER BY/LIMIT/OFFSET must be applied to UNION result".to_string(),
                ));
            }

            let matching_indices: Vec<usize> = if let Some(ref where_expr) = stmt.where_clause {
                SqlExecutor::evaluate_where(where_expr, table)?
            } else {
                let deleted = table.deleted_ref();
                let row_count = table.get_row_count();
                (0..row_count).filter(|&i| !deleted.get(i)).collect()
            };

            let (result_columns, column_indices) = SqlExecutor::resolve_columns(&stmt.columns, table)?;
            let mut rows: Vec<Vec<Value>> = Vec::with_capacity(matching_indices.len().min(10000));
            for row_idx in matching_indices.iter() {
                let mut row_values: Vec<Value> = Vec::with_capacity(result_columns.len());
                for (col_name, col_idx) in column_indices.iter() {
                    if col_name == "_id" {
                        row_values.push(Value::Int64(*row_idx as i64));
                    } else if let Some(idx) = col_idx {
                        row_values.push(table.columns_ref()[*idx].get(*row_idx).unwrap_or(Value::Null));
                    } else {
                        row_values.push(Value::Null);
                    }
                }
                rows.push(row_values);
            }
            Ok(SqlResult::new(result_columns, rows))
        }

        fn exec_one(
            stmt: crate::query::SqlStatement,
            tables: &mut HashMap<String, ColumnTable>,
            default_table: &str,
        ) -> Result<SqlResult, ApexError> {
            match stmt {
                crate::query::SqlStatement::Select(sel) => {
                    if !sel.joins.is_empty() {
                        return Err(ApexError::QueryParseError(
                            "UNION over JOIN queries is not supported yet".to_string(),
                        ));
                    }
                    let target_table = sel
                        .from
                        .as_ref()
                        .map(|f| match f {
                            FromItem::Table { table, .. } => table.clone(),
                            FromItem::Subquery { alias, .. } => alias.clone(),
                        })
                        .unwrap_or_else(|| default_table.to_string());
                    let table = tables.get_mut(&target_table).ok_or_else(|| {
                        ApexError::QueryParseError(format!("Table '{}' not found.", target_table))
                    })?;
                    exec_select_legacy(sel, table)
                }
                crate::query::SqlStatement::Union(u) => SqlExecutor::execute_union_with_tables(u, tables, default_table),
                crate::query::SqlStatement::CreateView { .. }
                | crate::query::SqlStatement::DropView { .. } => Err(ApexError::QueryParseError(
                    "CREATE/DROP VIEW are only supported in multi-statement execution".to_string(),
                )),
            }
        }

        let left = exec_one(*union.left, tables, default_table)?;
        let right = exec_one(*union.right, tables, default_table)?;
        Self::merge_union_results(left, right, union.all, &union.order_by, union.limit, union.offset)
    }

    pub(crate) fn merge_union_results(
        left: SqlResult,
        right: SqlResult,
        all: bool,
        order_by: &[OrderByClause],
        limit: Option<usize>,
        offset: Option<usize>,
    ) -> Result<SqlResult, ApexError> {
        fn materialize_arrow_rows(batch: &arrow::record_batch::RecordBatch) -> Vec<Vec<Value>> {
            use arrow::array::Array;
            use arrow::datatypes::DataType as ArrowDataType;

            let num_rows = batch.num_rows();
            let num_cols = batch.num_columns();
            let mut rows: Vec<Vec<Value>> = (0..num_rows).map(|_| Vec::with_capacity(num_cols)).collect();

            for ci in 0..num_cols {
                let col = batch.column(ci);
                match col.data_type() {
                    ArrowDataType::Int64 => {
                        let a = col.as_any().downcast_ref::<Int64Array>().unwrap();
                        for ri in 0..num_rows {
                            rows[ri].push(if a.is_null(ri) { Value::Null } else { Value::Int64(a.value(ri)) });
                        }
                    }
                    ArrowDataType::UInt64 => {
                        let a = col.as_any().downcast_ref::<UInt64Array>().unwrap();
                        for ri in 0..num_rows {
                            rows[ri].push(if a.is_null(ri) { Value::Null } else { Value::UInt64(a.value(ri)) });
                        }
                    }
                    ArrowDataType::Float64 => {
                        let a = col.as_any().downcast_ref::<Float64Array>().unwrap();
                        for ri in 0..num_rows {
                            rows[ri].push(if a.is_null(ri) { Value::Null } else { Value::Float64(a.value(ri)) });
                        }
                    }
                    ArrowDataType::Boolean => {
                        let a = col.as_any().downcast_ref::<BooleanArray>().unwrap();
                        for ri in 0..num_rows {
                            rows[ri].push(if a.is_null(ri) { Value::Null } else { Value::Bool(a.value(ri)) });
                        }
                    }
                    ArrowDataType::Utf8 => {
                        let a = col.as_any().downcast_ref::<StringArray>().unwrap();
                        for ri in 0..num_rows {
                            rows[ri].push(if a.is_null(ri) { Value::Null } else { Value::String(a.value(ri).to_string()) });
                        }
                    }
                    ArrowDataType::LargeUtf8 => {
                        let a = col.as_any().downcast_ref::<arrow::array::LargeStringArray>().unwrap();
                        for ri in 0..num_rows {
                            rows[ri].push(if a.is_null(ri) { Value::Null } else { Value::String(a.value(ri).to_string()) });
                        }
                    }
                    _ => {
                        for ri in 0..num_rows {
                            rows[ri].push(if col.is_null(ri) { Value::Null } else { Value::String(format!("{:?}", col)) });
                        }
                    }
                }
            }

            rows
        }

        fn ensure_rows(mut r: SqlResult) -> SqlResult {
            if r.rows.is_empty() {
                if let Some(batch) = r.arrow_batch.as_ref() {
                    r.rows = materialize_arrow_rows(batch);
                }
            }
            r
        }

        let left = ensure_rows(left);
        let right = ensure_rows(right);

        if left.columns != right.columns {
            return Err(ApexError::QueryParseError(
                "UNION requires both sides to have the same columns".to_string(),
            ));
        }

        let mut rows: Vec<Vec<Value>> = Vec::with_capacity(left.rows.len() + right.rows.len());
        rows.extend(left.rows);
        rows.extend(right.rows);

        if !all {
            let mut seen: std::collections::HashSet<String> = std::collections::HashSet::new();
            rows.retain(|r| {
                let k = r.iter().map(|v| v.to_string_value()).collect::<Vec<_>>().join("\u{1f}");
                seen.insert(k)
            });
        }

        if !order_by.is_empty() {
            // Map ORDER BY column name -> result column index
            let mut key_idx: Vec<(usize, bool, Option<bool>)> = Vec::new();
            for ob in order_by {
                let idx = left
                    .columns
                    .iter()
                    .position(|c| c == &ob.column)
                    .ok_or_else(|| {
                        ApexError::QueryParseError(format!(
                            "ORDER BY column '{}' must appear in UNION select list",
                            ob.column
                        ))
                    })?;
                key_idx.push((idx, ob.descending, ob.nulls_first));
            }

            rows.sort_by(|a, b| {
                for (idx, desc, nulls_first) in &key_idx {
                    let av = a.get(*idx);
                    let bv = b.get(*idx);
                    let cmp = SqlExecutor::compare_values(av, bv, *nulls_first);
                    let cmp = if *desc { cmp.reverse() } else { cmp };
                    if cmp != Ordering::Equal {
                        return cmp;
                    }
                }
                Ordering::Equal
            });
        }

        let off = offset.unwrap_or(0);
        let lim = limit.unwrap_or(usize::MAX);
        let rows = rows.into_iter().skip(off).take(lim).collect::<Vec<_>>();

        Ok(SqlResult::new(left.columns, rows))
    }

}
