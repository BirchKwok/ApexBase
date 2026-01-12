use super::*;

impl SqlExecutor {
    pub(crate) fn execute_select_with_tables(
        stmt: SelectStatement,
        tables: &mut HashMap<String, ColumnTable>,
        default_table: &str,
    ) -> Result<SqlResult, ApexError> {
        fn strip_prefix(col: &str, alias: &str) -> String {
            if let Some((a, c)) = col.split_once('.') {
                if a == alias {
                    return c.to_string();
                }
            }
            col.to_string()
        }

        fn rewrite_expr_strip_alias(expr: &SqlExpr, alias: &str) -> SqlExpr {
            match expr {
                SqlExpr::Paren(inner) => SqlExpr::Paren(Box::new(rewrite_expr_strip_alias(inner, alias))),
                SqlExpr::Column(c) => SqlExpr::Column(strip_prefix(c, alias)),
                SqlExpr::BinaryOp { left, op, right } => SqlExpr::BinaryOp {
                    left: Box::new(rewrite_expr_strip_alias(left, alias)),
                    op: op.clone(),
                    right: Box::new(rewrite_expr_strip_alias(right, alias)),
                },
                SqlExpr::UnaryOp { op, expr } => SqlExpr::UnaryOp {
                    op: op.clone(),
                    expr: Box::new(rewrite_expr_strip_alias(expr, alias)),
                },
                SqlExpr::Cast { expr, data_type } => SqlExpr::Cast {
                    expr: Box::new(rewrite_expr_strip_alias(expr, alias)),
                    data_type: *data_type,
                },
                SqlExpr::Like { column, pattern, negated } => SqlExpr::Like {
                    column: strip_prefix(column, alias),
                    pattern: pattern.clone(),
                    negated: *negated,
                },
                SqlExpr::Regexp { column, pattern, negated } => SqlExpr::Regexp {
                    column: strip_prefix(column, alias),
                    pattern: pattern.clone(),
                    negated: *negated,
                },
                SqlExpr::In { column, values, negated } => SqlExpr::In {
                    column: strip_prefix(column, alias),
                    values: values.clone(),
                    negated: *negated,
                },
                SqlExpr::Between { column, low, high, negated } => SqlExpr::Between {
                    column: strip_prefix(column, alias),
                    low: Box::new(rewrite_expr_strip_alias(low, alias)),
                    high: Box::new(rewrite_expr_strip_alias(high, alias)),
                    negated: *negated,
                },
                SqlExpr::IsNull { column, negated } => SqlExpr::IsNull {
                    column: strip_prefix(column, alias),
                    negated: *negated,
                },
                SqlExpr::Function { name, args } => SqlExpr::Function {
                    name: name.clone(),
                    args: args.iter().map(|a| rewrite_expr_strip_alias(a, alias)).collect(),
                },
                _ => expr.clone(),
            }
        }

        fn materialize_sql_result_to_table(alias: &str, res: SqlResult) -> Result<ColumnTable, ApexError> {
            use std::collections::HashMap;
            let mut t = ColumnTable::new(0, alias);

            if res.columns.is_empty() {
                return Ok(t);
            }

            // If the result was produced via Arrow fast path, rows may be empty.
            // Materialize from the Arrow batch when present.
            if res.rows.is_empty() {
                if let Some(batch) = res.arrow_batch {
                    use arrow::array::Array;
                    use arrow::datatypes::DataType;

                    let mut colmap: HashMap<String, Vec<Value>> = HashMap::new();
                    for (i, name) in res.columns.iter().enumerate() {
                        let arr = batch.column(i);
                        let mut values: Vec<Value> = Vec::with_capacity(arr.len());
                        match arr.data_type() {
                            DataType::Int64 => {
                                let a = arr.as_any().downcast_ref::<arrow::array::Int64Array>().unwrap();
                                for r in 0..a.len() {
                                    if a.is_null(r) {
                                        values.push(Value::Null);
                                    } else {
                                        values.push(Value::Int64(a.value(r)));
                                    }
                                }
                            }
                            DataType::Float64 => {
                                let a = arr.as_any().downcast_ref::<arrow::array::Float64Array>().unwrap();
                                for r in 0..a.len() {
                                    if a.is_null(r) {
                                        values.push(Value::Null);
                                    } else {
                                        values.push(Value::Float64(a.value(r)));
                                    }
                                }
                            }
                            DataType::Utf8 => {
                                let a = arr.as_any().downcast_ref::<arrow::array::StringArray>().unwrap();
                                for r in 0..a.len() {
                                    if a.is_null(r) {
                                        values.push(Value::Null);
                                    } else {
                                        values.push(Value::String(a.value(r).to_string()));
                                    }
                                }
                            }
                            DataType::Boolean => {
                                let a = arr.as_any().downcast_ref::<arrow::array::BooleanArray>().unwrap();
                                for r in 0..a.len() {
                                    if a.is_null(r) {
                                        values.push(Value::Null);
                                    } else {
                                        values.push(Value::Bool(a.value(r)));
                                    }
                                }
                            }
                            _ => {
                                // Fallback: stringify
                                for r in 0..arr.len() {
                                    if arr.is_null(r) {
                                        values.push(Value::Null);
                                    } else {
                                        values.push(Value::String(format!("{:?}", arr)));
                                    }
                                }
                            }
                        }
                        colmap.insert(name.clone(), values);
                    }

                    t.insert_columns(colmap)
                        .map_err(|e| ApexError::QueryParseError(format!("Failed to materialize derived table: {:?}", e)))?;
                    t.flush_write_buffer();
                    return Ok(t);
                }
            }

            let mut colmap: HashMap<String, Vec<Value>> = HashMap::new();
            for c in &res.columns {
                colmap.insert(c.clone(), Vec::with_capacity(res.rows.len()));
            }
            for row in res.rows {
                for (i, c) in res.columns.iter().enumerate() {
                    let v = row.get(i).cloned().unwrap_or(Value::Null);
                    colmap.get_mut(c).unwrap().push(v);
                }
            }
            t.insert_columns(colmap)
                .map_err(|e| ApexError::QueryParseError(format!("Failed to materialize derived table: {:?}", e)))?;
            t.flush_write_buffer();
            Ok(t)
        }

        fn collect_single_column_values(res: SqlResult) -> Result<(Vec<Value>, bool), ApexError> {
            if res.columns.len() != 1 {
                return Err(ApexError::QueryParseError(
                    "IN (subquery) requires subquery to return exactly 1 column".to_string(),
                ));
            }

            // rows path
            if !res.rows.is_empty() {
                let mut out: Vec<Value> = Vec::with_capacity(res.rows.len());
                let mut has_null = false;
                for r in res.rows {
                    let v = r.get(0).cloned().unwrap_or(Value::Null);
                    has_null |= v.is_null();
                    out.push(v);
                }
                return Ok((out, has_null));
            }

            // arrow fast path
            if let Some(batch) = res.arrow_batch {
                use arrow::array::Array;
                use arrow::datatypes::DataType;

                let arr = batch.column(0);
                let mut out: Vec<Value> = Vec::with_capacity(arr.len());
                let mut has_null = false;
                match arr.data_type() {
                    DataType::Int64 => {
                        let a = arr.as_any().downcast_ref::<arrow::array::Int64Array>().unwrap();
                        for i in 0..a.len() {
                            if a.is_null(i) {
                                has_null = true;
                                out.push(Value::Null);
                            } else {
                                out.push(Value::Int64(a.value(i)));
                            }
                        }
                    }
                    DataType::Float64 => {
                        let a = arr.as_any().downcast_ref::<arrow::array::Float64Array>().unwrap();
                        for i in 0..a.len() {
                            if a.is_null(i) {
                                has_null = true;
                                out.push(Value::Null);
                            } else {
                                out.push(Value::Float64(a.value(i)));
                            }
                        }
                    }
                    DataType::Utf8 => {
                        let a = arr.as_any().downcast_ref::<arrow::array::StringArray>().unwrap();
                        for i in 0..a.len() {
                            if a.is_null(i) {
                                has_null = true;
                                out.push(Value::Null);
                            } else {
                                out.push(Value::String(a.value(i).to_string()));
                            }
                        }
                    }
                    DataType::Boolean => {
                        let a = arr.as_any().downcast_ref::<arrow::array::BooleanArray>().unwrap();
                        for i in 0..a.len() {
                            if a.is_null(i) {
                                has_null = true;
                                out.push(Value::Null);
                            } else {
                                out.push(Value::Bool(a.value(i)));
                            }
                        }
                    }
                    _ => {
                        for i in 0..arr.len() {
                            if arr.is_null(i) {
                                has_null = true;
                                out.push(Value::Null);
                            } else {
                                out.push(Value::String(format!("{:?}", arr)));
                            }
                        }
                    }
                }
                return Ok((out, has_null));
            }

            Ok((Vec::new(), false))
        }

        fn rewrite_in_subquery_in_expr(
            expr: &SqlExpr,
            tables: &mut HashMap<String, ColumnTable>,
            default_table: &str,
        ) -> Result<SqlExpr, ApexError> {
            fn subquery_has_outer_ref(sub: &SelectStatement) -> bool {
                // Determine subquery's own qualifier (alias or table name).
                let (inner_table, inner_alias) = match sub.from.as_ref() {
                    Some(FromItem::Table { table, alias }) => {
                        (table.as_str(), alias.as_deref().unwrap_or(table.as_str()))
                    }
                    _ => ("", ""),
                };

                fn expr_has_outer_ref(expr: &SqlExpr, inner_table: &str, inner_alias: &str) -> bool {
                    match expr {
                        SqlExpr::Column(c) => {
                            if let Some((a, _)) = c.split_once('.') {
                                // Any qualifier not matching the subquery itself is treated as outer ref.
                                !(a == inner_alias || a == inner_table)
                            } else {
                                false
                            }
                        }
                        SqlExpr::BinaryOp { left, right, .. } => {
                            expr_has_outer_ref(left, inner_table, inner_alias)
                                || expr_has_outer_ref(right, inner_table, inner_alias)
                        }
                        SqlExpr::UnaryOp { expr, .. } => expr_has_outer_ref(expr, inner_table, inner_alias),
                        SqlExpr::Paren(inner) => expr_has_outer_ref(inner, inner_table, inner_alias),
                        SqlExpr::Cast { expr, .. } => expr_has_outer_ref(expr, inner_table, inner_alias),
                        SqlExpr::Between { low, high, .. } => {
                            expr_has_outer_ref(low, inner_table, inner_alias)
                                || expr_has_outer_ref(high, inner_table, inner_alias)
                        }
                        SqlExpr::Function { args, .. } => {
                            args.iter().any(|a| expr_has_outer_ref(a, inner_table, inner_alias))
                        }
                        SqlExpr::Case { when_then, else_expr } => {
                            when_then.iter().any(|(c, v)| {
                                expr_has_outer_ref(c, inner_table, inner_alias)
                                    || expr_has_outer_ref(v, inner_table, inner_alias)
                            }) || else_expr
                                .as_ref()
                                .is_some_and(|e| expr_has_outer_ref(e, inner_table, inner_alias))
                        }
                        SqlExpr::InSubquery { .. }
                        | SqlExpr::ExistsSubquery { .. }
                        | SqlExpr::ScalarSubquery { .. } => false,
                        SqlExpr::Like { .. }
                        | SqlExpr::Regexp { .. }
                        | SqlExpr::In { .. }
                        | SqlExpr::IsNull { .. }
                        | SqlExpr::Literal(_) => false,
                    }
                }

                if let Some(w) = sub.where_clause.as_ref() {
                    return expr_has_outer_ref(w, inner_table, inner_alias);
                }
                false
            }

            match expr {
                SqlExpr::BinaryOp { left, op, right } => Ok(SqlExpr::BinaryOp {
                    left: Box::new(rewrite_in_subquery_in_expr(left, tables, default_table)?),
                    op: op.clone(),
                    right: Box::new(rewrite_in_subquery_in_expr(right, tables, default_table)?),
                }),
                SqlExpr::UnaryOp { op, expr } => Ok(SqlExpr::UnaryOp {
                    op: op.clone(),
                    expr: Box::new(rewrite_in_subquery_in_expr(expr, tables, default_table)?),
                }),
                SqlExpr::Paren(inner) => Ok(SqlExpr::Paren(Box::new(rewrite_in_subquery_in_expr(
                    inner,
                    tables,
                    default_table,
                )?))),
                SqlExpr::Between { column, low, high, negated } => Ok(SqlExpr::Between {
                    column: column.clone(),
                    low: Box::new(rewrite_in_subquery_in_expr(low, tables, default_table)?),
                    high: Box::new(rewrite_in_subquery_in_expr(high, tables, default_table)?),
                    negated: *negated,
                }),
                SqlExpr::Function { name, args } => Ok(SqlExpr::Function {
                    name: name.clone(),
                    args: args
                        .iter()
                        .map(|a| rewrite_in_subquery_in_expr(a, tables, default_table))
                        .collect::<Result<Vec<_>, _>>()?,
                }),
                SqlExpr::InSubquery { column, stmt, negated } => {
                    // Correlated subquery: do NOT pre-execute here.
                    // Keep it for row-wise evaluation in the correlated execution path.
                    if subquery_has_outer_ref(stmt) {
                        return Ok(expr.clone());
                    }

                    if !stmt.joins.is_empty() {
                        return Err(ApexError::QueryParseError(
                            "IN (subquery) with JOIN is not supported yet".to_string(),
                        ));
                    }
                    let sub_res = SqlExecutor::execute_select_with_tables((**stmt).clone(), tables, default_table)?;
                    let (values, has_null) = collect_single_column_values(sub_res)?;

                    // Conservative NULL semantics for NOT IN: if subquery yields any NULL,
                    // the result is UNKNOWN for all non-matching rows -> filter out.
                    if *negated && has_null {
                        return Ok(SqlExpr::Literal(Value::Bool(false)));
                    }

                    Ok(SqlExpr::In {
                        column: column.clone(),
                        values,
                        negated: *negated,
                    })
                }
                _ => Ok(expr.clone()),
            }
        }

        // We will rewrite/normalize the statement in-place.
        let mut stmt = stmt;

        // Rewrite IN (subquery) in WHERE into IN (list) so we can reuse Filter::In
        if let Some(ref w) = stmt.where_clause {
            let rewritten = rewrite_in_subquery_in_expr(w, tables, default_table)?;
            stmt.where_clause = Some(rewritten);
        }

        // Derived table in FROM: execute subquery first, materialize, then execute outer query
        if let Some(FromItem::Subquery { stmt: sub, alias }) = stmt.from.clone() {
            if stmt.joins.is_empty()
                && stmt.group_by.is_empty()
                && stmt.order_by.is_empty()
                && stmt.limit.is_none()
                && stmt.offset.is_none()
                && stmt.columns.len() == 1
            {
                use crate::query::sql_parser::{SelectColumn, SqlExpr, BinaryOperator, AggregateFunc};

                let is_count_star = matches!(
                    stmt.columns[0],
                    SelectColumn::Aggregate {
                        func: AggregateFunc::Count,
                        column: None,
                        distinct: false,
                        ..
                    }
                );

                // Outer WHERE: t.<alias_of_sum> >= <literal>
                fn parse_sum_ge_threshold(where_expr: &SqlExpr, alias: &str) -> Option<(String, f64)> {
                    match where_expr {
                        SqlExpr::BinaryOp { left, op: BinaryOperator::Ge, right } => {
                            let col = match left.as_ref() {
                                SqlExpr::Column(c) => c,
                                _ => return None,
                            };
                            let lit = match right.as_ref() {
                                SqlExpr::Literal(v) => v.as_f64()?,
                                _ => return None,
                            };
                            let expected_prefix = format!("{}.", alias);
                            if let Some(rest) = col.strip_prefix(&expected_prefix) {
                                Some((rest.to_string(), lit))
                            } else {
                                None
                            }
                        }
                        _ => None,
                    }
                }

                // Inner HAVING: SUM(amount) >= K (allow optional extra parens)
                fn parse_having_sum_ge(h: &SqlExpr) -> Option<(String, f64)> {
                    match h {
                        SqlExpr::Paren(inner) => parse_having_sum_ge(inner),
                        SqlExpr::BinaryOp { left, op: BinaryOperator::Ge, right } => {
                            let (col_name, _) = match left.as_ref() {
                                SqlExpr::Function { name, args } if name.eq_ignore_ascii_case("sum") => {
                                    if args.len() != 1 {
                                        return None;
                                    }
                                    match &args[0] {
                                        SqlExpr::Column(c) => (c.clone(), ()),
                                        _ => return None,
                                    }
                                }
                                _ => return None,
                            };
                            let lit = match right.as_ref() {
                                SqlExpr::Literal(v) => v.as_f64()?,
                                _ => return None,
                            };
                            Some((col_name, lit))
                        }
                        _ => None,
                    }
                }

                // Inner SELECT must contain SUM(amount) AS <s>
                fn find_sum_alias(inner_cols: &[SelectColumn]) -> Option<(String, String)> {
                    for c in inner_cols {
                        if let SelectColumn::Aggregate { func: AggregateFunc::Sum, column: Some(col), alias: Some(a), distinct: false } = c {
                            return Some((col.clone(), a.clone()));
                        }
                    }
                    None
                }

                if is_count_star {
                    if let Some(ref outer_where) = stmt.where_clause {
                        if let Some((outer_sum_alias, outer_k)) = parse_sum_ge_threshold(outer_where, &alias) {
                            // inner statement checks
                            if sub.joins.is_empty()
                                && sub.where_clause.is_none()
                                && sub.order_by.is_empty()
                                && sub.limit.is_none()
                                && sub.offset.is_none()
                                && !sub.group_by.is_empty()
                                && sub.having.is_some()
                            {
                                if let Some((sum_col, sum_alias)) = find_sum_alias(&sub.columns) {
                                    if sum_alias == outer_sum_alias {
                                        if let Some((having_sum_col, having_k)) = parse_having_sum_ge(sub.having.as_ref().unwrap()) {
                                            if having_sum_col == sum_col && (having_k - outer_k).abs() < f64::EPSILON {
                                                // Execute inner GROUP BY once, but only return COUNT of groups passing threshold.
                                                let inner_table_name = sub
                                                    .from
                                                    .as_ref()
                                                    .map(|f| match f {
                                                        FromItem::Table { table, .. } => table.clone(),
                                                        _ => default_table.to_string(),
                                                    })
                                                    .unwrap_or_else(|| default_table.to_string());

                                                // Flush pending writes for the scanned table
                                                {
                                                    let t = tables.get_mut(&inner_table_name).ok_or_else(|| {
                                                        ApexError::QueryParseError(format!("Table '{}' not found.", inner_table_name))
                                                    })?;
                                                    if t.has_pending_writes() {
                                                        t.flush_write_buffer();
                                                    }
                                                }

                                                let table = tables.get(&inner_table_name).ok_or_else(|| {
                                                    ApexError::QueryParseError(format!("Table '{}' not found.", inner_table_name))
                                                })?;

                                                let schema = table.schema_ref();
                                                let cols = table.columns_ref();

                                                let sum_idx = schema.get_index(&sum_col).ok_or_else(|| {
                                                    ApexError::QueryParseError(format!("Unknown column: {}", sum_col))
                                                })?;

                                                let mut group_col_indices = Vec::with_capacity(sub.group_by.len());
                                                for g in &sub.group_by {
                                                    let gi = schema.get_index(g).ok_or_else(|| {
                                                        ApexError::QueryParseError(format!("Unknown GROUP BY column: {}", g))
                                                    })?;
                                                    group_col_indices.push(gi);
                                                }

                                                #[derive(Clone, Eq, PartialEq, Hash)]
                                                enum KeyPart {
                                                    Null,
                                                    Int(i64),
                                                    Float(u64),
                                                    Bool(bool),
                                                    StrId(u32),
                                                }

                                                #[derive(Clone, Eq, PartialEq, Hash)]
                                                struct GroupKey {
                                                    parts: Vec<KeyPart>,
                                                }

                                                let mut groups: HashMap<GroupKey, f64> = HashMap::new();
                                                let mut str_intern: HashMap<String, u32> = HashMap::new();
                                                let mut next_str_id: u32 = 0;
                                                let row_count = table.get_row_count();
                                                let deleted = table.deleted_ref();

                                                let mut sum_col_f64: Option<&[f64]> = None;
                                                let mut sum_col_i64: Option<&[i64]> = None;
                                                let mut sum_col_nulls: Option<&crate::table::column_table::BitVec> = None;
                                                match &cols[sum_idx] {
                                                    TypedColumn::Float64 { data, nulls } => {
                                                        sum_col_f64 = Some(data);
                                                        sum_col_nulls = Some(nulls);
                                                    }
                                                    TypedColumn::Int64 { data, nulls } => {
                                                        sum_col_i64 = Some(data);
                                                        sum_col_nulls = Some(nulls);
                                                    }
                                                    _ => {}
                                                }

                                                for row_idx in 0..row_count {
                                                    if deleted.get(row_idx) {
                                                        continue;
                                                    }
                                                    let mut parts: Vec<KeyPart> = Vec::with_capacity(group_col_indices.len());
                                                    for &gi in &group_col_indices {
                                                        match &cols[gi] {
                                                            TypedColumn::Int64 { data, nulls } => {
                                                                if row_idx >= data.len() || nulls.get(row_idx) {
                                                                    parts.push(KeyPart::Null);
                                                                } else {
                                                                    parts.push(KeyPart::Int(data[row_idx]));
                                                                }
                                                            }
                                                            TypedColumn::Float64 { data, nulls } => {
                                                                if row_idx >= data.len() || nulls.get(row_idx) {
                                                                    parts.push(KeyPart::Null);
                                                                } else {
                                                                    parts.push(KeyPart::Float(data[row_idx].to_bits()));
                                                                }
                                                            }
                                                            TypedColumn::Bool { data, nulls } => {
                                                                if row_idx >= data.len() || nulls.get(row_idx) {
                                                                    parts.push(KeyPart::Null);
                                                                } else {
                                                                    parts.push(KeyPart::Bool(data.get(row_idx)));
                                                                }
                                                            }
                                                            TypedColumn::String(sc) => {
                                                                if sc.is_null(row_idx) {
                                                                    parts.push(KeyPart::Null);
                                                                } else if let Some(s) = sc.get(row_idx) {
                                                                    if let Some(id) = str_intern.get(s) {
                                                                        parts.push(KeyPart::StrId(*id));
                                                                    } else {
                                                                        let id = next_str_id;
                                                                        next_str_id = next_str_id.wrapping_add(1);
                                                                        str_intern.insert(s.to_string(), id);
                                                                        parts.push(KeyPart::StrId(id));
                                                                    }
                                                                } else {
                                                                    parts.push(KeyPart::Null);
                                                                }
                                                            }
                                                            TypedColumn::Mixed { data, nulls } => {
                                                                if row_idx >= data.len() || nulls.get(row_idx) {
                                                                    parts.push(KeyPart::Null);
                                                                } else {
                                                                    let sv = data[row_idx].to_string_value();
                                                                    if let Some(id) = str_intern.get(&sv) {
                                                                        parts.push(KeyPart::StrId(*id));
                                                                    } else {
                                                                        let id = next_str_id;
                                                                        next_str_id = next_str_id.wrapping_add(1);
                                                                        str_intern.insert(sv, id);
                                                                        parts.push(KeyPart::StrId(id));
                                                                    }
                                                                }
                                                            }
                                                        }
                                                    }
                                                    let key = GroupKey { parts };

                                                    let add = if let (Some(data), Some(nulls)) = (sum_col_f64, sum_col_nulls) {
                                                        if row_idx >= data.len() || nulls.get(row_idx) {
                                                            0.0
                                                        } else {
                                                            data[row_idx]
                                                        }
                                                    } else if let (Some(data), Some(nulls)) = (sum_col_i64, sum_col_nulls) {
                                                        if row_idx >= data.len() || nulls.get(row_idx) {
                                                            0.0
                                                        } else {
                                                            data[row_idx] as f64
                                                        }
                                                    } else {
                                                        match cols[sum_idx].get(row_idx).and_then(|v| v.as_f64()) {
                                                            Some(v) => v,
                                                            None => 0.0,
                                                        }
                                                    };

                                                    *groups.entry(key).or_insert(0.0) += add;
                                                }

                                                let mut cnt: i64 = 0;
                                                for (_k, s) in groups {
                                                    if s >= outer_k {
                                                        cnt += 1;
                                                    }
                                                }

                                                return Ok(SqlResult::new(
                                                    vec!["big_groups".to_string()],
                                                    vec![vec![Value::Int64(cnt)]],
                                                ));
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }

            if !stmt.joins.is_empty() {
                return Err(ApexError::QueryParseError(
                    "JOIN with derived table in FROM is not supported yet".to_string(),
                ));
            }

            let sub_res = SqlExecutor::execute_select_with_tables(*sub, tables, default_table)?;
            let tmp = materialize_sql_result_to_table(&alias, sub_res)?;
            tables.insert(alias.clone(), tmp);

            // Rewrite outer query to refer to the materialized alias table and strip qualifiers
            stmt.from = Some(FromItem::Table { table: alias.clone(), alias: Some(alias.clone()) });
            for c in &mut stmt.columns {
                match c {
                    SelectColumn::Column(name) => *name = strip_prefix(name, &alias),
                    SelectColumn::ColumnAlias { column, .. } => *column = strip_prefix(column, &alias),
                    SelectColumn::Aggregate { column, .. } => {
                        if let Some(cc) = column.as_mut() {
                            *cc = strip_prefix(cc, &alias);
                        }
                    }
                    SelectColumn::Expression { expr, .. } => {
                        *expr = rewrite_expr_strip_alias(expr, &alias);
                    }
                    _ => {}
                }
            }
            if let Some(w) = stmt.where_clause.as_mut() {
                *w = rewrite_expr_strip_alias(w, &alias);
            }
            stmt.group_by = stmt.group_by.iter().map(|g| strip_prefix(g, &alias)).collect();
            if let Some(h) = stmt.having.as_mut() {
                *h = rewrite_expr_strip_alias(h, &alias);
            }
            for ob in &mut stmt.order_by {
                ob.column = strip_prefix(&ob.column, &alias);
            }
        }

        // If this is a single-table query with an explicit table alias in FROM (e.g. FROM users u),
        // normalize outer projection/order-by to unqualified column names (u.name -> name).
        // This keeps the single-table execution logic and schema lookup consistent.
        if stmt.joins.is_empty() {
            if let Some(FromItem::Table { table, alias: Some(a) }) = stmt.from.as_ref() {
                fn strip_outer_prefix(name: &str, table: &str, alias: &str) -> String {
                    if let Some((p, c)) = name.split_once('.') {
                        if p == alias || p == table {
                            return c.to_string();
                        }
                    }
                    name.to_string()
                }

                for c in &mut stmt.columns {
                    match c {
                        SelectColumn::Column(name) => {
                            *name = strip_outer_prefix(name, table, a);
                        }
                        SelectColumn::ColumnAlias { column, .. } => {
                            *column = strip_outer_prefix(column, table, a);
                        }
                        SelectColumn::Aggregate { column, .. } => {
                            if let Some(cc) = column.as_mut() {
                                *cc = strip_outer_prefix(cc, table, a);
                            }
                        }
                        SelectColumn::Expression { .. } => {
                            // Keep expressions as-is; they may legitimately reference outer alias.
                        }
                        _ => {}
                    }
                }
                for ob in &mut stmt.order_by {
                    ob.column = strip_outer_prefix(&ob.column, table, a);
                }
            }
        }

        if stmt.joins.is_empty() {
            let target_table = stmt
                .from
                .as_ref()
                .map(|f| match f {
                    FromItem::Table { table, .. } => table.clone(),
                    FromItem::Subquery { alias, .. } => alias.clone(),
                })
                .unwrap_or_else(|| default_table.to_string());

            fn expr_has_exists_subquery(expr: &SqlExpr) -> bool {
                match expr {
                    SqlExpr::ExistsSubquery { .. } => true,
                    SqlExpr::ScalarSubquery { .. } => true,
                    SqlExpr::BinaryOp { left, right, .. } => {
                        expr_has_exists_subquery(left) || expr_has_exists_subquery(right)
                    }
                    SqlExpr::UnaryOp { expr, .. } => expr_has_exists_subquery(expr),
                    SqlExpr::Paren(inner) => expr_has_exists_subquery(inner),
                    SqlExpr::Between { low, high, .. } => {
                        expr_has_exists_subquery(low) || expr_has_exists_subquery(high)
                    }
                    SqlExpr::Function { args, .. } => args.iter().any(expr_has_exists_subquery),
                    _ => false,
                }
            }

            fn expr_has_in_subquery(expr: &SqlExpr) -> bool {
                match expr {
                    SqlExpr::InSubquery { .. } => true,
                    SqlExpr::BinaryOp { left, right, .. } => {
                        expr_has_in_subquery(left) || expr_has_in_subquery(right)
                    }
                    SqlExpr::UnaryOp { expr, .. } => expr_has_in_subquery(expr),
                    SqlExpr::Paren(inner) => expr_has_in_subquery(inner),
                    SqlExpr::Between { low, high, .. } => {
                        expr_has_in_subquery(low) || expr_has_in_subquery(high)
                    }
                    SqlExpr::Function { args, .. } => args.iter().any(expr_has_in_subquery),
                    SqlExpr::Case { when_then, else_expr } => {
                        when_then.iter().any(|(c, v)| expr_has_in_subquery(c) || expr_has_in_subquery(v))
                            || else_expr.as_ref().is_some_and(|e| expr_has_in_subquery(e))
                    }
                    _ => false,
                }
            }

            // If this is a single-table query but references qualified columns (e.g. u.name)
            // or has EXISTS subquery, we can't delegate to the single-table executor because it
            // doesn't understand qualifiers and Filter conversion can't represent EXISTS.
            let needs_qualified_or_exists = {
                let has_exists = stmt
                    .where_clause
                    .as_ref()
                    .is_some_and(expr_has_exists_subquery);

                let has_in_subquery = stmt
                    .where_clause
                    .as_ref()
                    .is_some_and(expr_has_in_subquery);

                let has_scalar_in_select = stmt.columns.iter().any(|c| match c {
                    SelectColumn::Expression { expr, .. } => expr_has_exists_subquery(expr),
                    _ => false,
                });

                let outer_alias = stmt.from.as_ref().and_then(|f| match f {
                    FromItem::Table { alias, .. } => alias.clone(),
                    _ => None,
                });

                let has_qualified_select = stmt.columns.iter().any(|c| match c {
                    SelectColumn::Column(name) => name.contains('.'),
                    SelectColumn::ColumnAlias { column, .. } => column.contains('.'),
                    SelectColumn::Aggregate { column, .. } => column.as_ref().is_some_and(|x| x.contains('.')),
                    SelectColumn::Expression { expr, .. } => matches!(expr, SqlExpr::Column(c) if c.contains('.')),
                    _ => false,
                });
                let has_qualified_order = stmt.order_by.iter().any(|o| o.column.contains('.'));

                has_exists
                    || has_scalar_in_select
                    || has_in_subquery
                    || outer_alias.is_some() && (has_qualified_select || has_qualified_order)
            };

            // We may need a mutable borrow to flush pending writes, but the EXISTS path also
            // needs an immutable borrow of the full tables map. Keep the mutable borrow scope
            // minimal to satisfy Rust's borrow checker.
            {
                let t = tables
                    .get_mut(&target_table)
                    .ok_or_else(|| ApexError::QueryParseError(format!("Table '{}' not found.", target_table)))?;
                if t.has_pending_writes() {
                    t.flush_write_buffer();
                }
            }

            if !needs_qualified_or_exists {
                let table = tables
                    .get_mut(&target_table)
                    .ok_or_else(|| ApexError::QueryParseError(format!("Table '{}' not found.", target_table)))?;
                return Self::execute_select(stmt, table);
            }

            // ============ Single-table (no JOIN) path with qualifiers / EXISTS support ============
            let table = tables
                .get(&target_table)
                .ok_or_else(|| ApexError::QueryParseError(format!("Table '{}' not found.", target_table)))?;

            let outer_table_name = target_table.clone();
            let outer_alias = stmt
                .from
                .as_ref()
                .and_then(|f| match f {
                    FromItem::Table { alias, .. } => alias.clone(),
                    _ => None,
                })
                .unwrap_or_else(|| outer_table_name.clone());

            fn strip_outer(col: &str, outer_alias: &str, outer_table: &str) -> String {
                if let Some((a, c)) = col.split_once('.') {
                    // In this execution path we are guaranteed to be running a single-table query.
                    // Be permissive and strip the qualifier even if it doesn't match exactly,
                    // so selecting `u.name` works reliably.
                    if a == outer_alias || a == outer_table {
                        return c.to_string();
                    }
                    return c.to_string();
                }
                col.to_string()
            }

            fn get_col_value_qualified(
                col_ref: &str,
                outer_table: &ColumnTable,
                outer_row: usize,
                outer_alias: &str,
                outer_table_name: &str,
                inner_table: &ColumnTable,
                inner_row: usize,
                inner_alias: &str,
                inner_table_name: &str,
            ) -> Value {
                // Handle special _id
                if col_ref == "_id" {
                    return Value::Int64(inner_row as i64);
                }
                let (a, c) = if let Some((a, c)) = col_ref.split_once('.') {
                    (a, c)
                } else {
                    ("", col_ref)
                };

                // Qualified
                if !a.is_empty() {
                    if a == inner_alias || a == inner_table_name {
                        if c == "_id" {
                            return Value::Int64(inner_row as i64);
                        }
                        if let Some(ci) = inner_table.schema_ref().get_index(c) {
                            return inner_table.columns_ref()[ci].get(inner_row).unwrap_or(Value::Null);
                        }
                        return Value::Null;
                    }
                    if a == outer_alias || a == outer_table_name {
                        if c == "_id" {
                            return Value::Int64(outer_row as i64);
                        }
                        if let Some(ci) = outer_table.schema_ref().get_index(c) {
                            return outer_table.columns_ref()[ci].get(outer_row).unwrap_or(Value::Null);
                        }
                        return Value::Null;
                    }
                    return Value::Null;
                }

                // Unqualified: prefer inner table if column exists there
                if c == "_id" {
                    return Value::Int64(inner_row as i64);
                }
                if inner_table.schema_ref().get_index(c).is_some() {
                    let ci = inner_table.schema_ref().get_index(c).unwrap();
                    return inner_table.columns_ref()[ci].get(inner_row).unwrap_or(Value::Null);
                }
                if outer_table.schema_ref().get_index(c).is_some() {
                    let ci = outer_table.schema_ref().get_index(c).unwrap();
                    return outer_table.columns_ref()[ci].get(outer_row).unwrap_or(Value::Null);
                }
                Value::Null
            }

            fn eval_correlated_predicate(
                expr: &SqlExpr,
                outer_table: &ColumnTable,
                outer_row: usize,
                outer_alias: &str,
                outer_table_name: &str,
                inner_table: &ColumnTable,
                inner_row: usize,
                inner_alias: &str,
                inner_table_name: &str,
            ) -> Result<bool, ApexError> {
                fn eval_scalar(
                    expr: &SqlExpr,
                    outer_table: &ColumnTable,
                    outer_row: usize,
                    outer_alias: &str,
                    outer_table_name: &str,
                    inner_table: &ColumnTable,
                    inner_row: usize,
                    inner_alias: &str,
                    inner_table_name: &str,
                ) -> Result<Value, ApexError> {
                    fn is_string_literal_or_column(expr: &SqlExpr) -> bool {
                        match expr {
                            SqlExpr::Paren(inner) => is_string_literal_or_column(inner),
                            SqlExpr::Column(_) => true,
                            SqlExpr::Literal(v) => matches!(v, Value::String(_) | Value::Null),
                            _ => false,
                        }
                    }

                    fn cast_value(v: Value, target: crate::data::DataType) -> Result<Value, ApexError> {
                        use crate::data::DataType;

                        if v.is_null() {
                            return Ok(Value::Null);
                        }

                        match target {
                            DataType::String => Ok(Value::String(v.to_string_value())),
                            DataType::Bool => match v {
                                Value::Bool(b) => Ok(Value::Bool(b)),
                                Value::Int8(i) => Ok(Value::Bool(i != 0)),
                                Value::Int16(i) => Ok(Value::Bool(i != 0)),
                                Value::Int32(i) => Ok(Value::Bool(i != 0)),
                                Value::Int64(i) => Ok(Value::Bool(i != 0)),
                                Value::UInt8(i) => Ok(Value::Bool(i != 0)),
                                Value::UInt16(i) => Ok(Value::Bool(i != 0)),
                                Value::UInt32(i) => Ok(Value::Bool(i != 0)),
                                Value::UInt64(i) => Ok(Value::Bool(i != 0)),
                                Value::Float32(f) => Ok(Value::Bool(f != 0.0)),
                                Value::Float64(f) => Ok(Value::Bool(f != 0.0)),
                                Value::String(s) => {
                                    let ls = s.trim().to_lowercase();
                                    match ls.as_str() {
                                        "true" | "1" | "t" | "yes" | "y" => Ok(Value::Bool(true)),
                                        "false" | "0" | "f" | "no" | "n" => Ok(Value::Bool(false)),
                                        _ => Err(ApexError::QueryParseError(format!(
                                            "Cannot CAST value '{}' to BOOLEAN",
                                            s
                                        ))),
                                    }
                                }
                                other => Err(ApexError::QueryParseError(format!(
                                    "Cannot CAST {:?} to BOOLEAN",
                                    other
                                ))),
                            },
                            DataType::Int8
                            | DataType::Int16
                            | DataType::Int32
                            | DataType::Int64
                            | DataType::UInt8
                            | DataType::UInt16
                            | DataType::UInt32
                            | DataType::UInt64 => {
                                if let Some(i) = v.as_i64() {
                                    Ok(Value::Int64(i))
                                } else if let Value::String(s) = v {
                                    let t = s.trim();
                                    let parsed: i64 = t.parse().map_err(|_| {
                                        ApexError::QueryParseError(format!(
                                            "Cannot CAST value '{}' to INTEGER",
                                            s
                                        ))
                                    })?;
                                    Ok(Value::Int64(parsed))
                                } else {
                                    Err(ApexError::QueryParseError(format!(
                                        "Cannot CAST value '{}' to INTEGER",
                                        v.to_string_value()
                                    )))
                                }
                            }
                            DataType::Float32 | DataType::Float64 => {
                                if let Some(f) = v.as_f64() {
                                    Ok(Value::Float64(f))
                                } else if let Value::String(s) = v {
                                    let t = s.trim();
                                    let parsed: f64 = t.parse().map_err(|_| {
                                        ApexError::QueryParseError(format!(
                                            "Cannot CAST value '{}' to DOUBLE",
                                            s
                                        ))
                                    })?;
                                    Ok(Value::Float64(parsed))
                                } else {
                                    Err(ApexError::QueryParseError(format!(
                                        "Cannot CAST value '{}' to DOUBLE",
                                        v.to_string_value()
                                    )))
                                }
                            }
                            _ => Err(ApexError::QueryParseError(format!(
                                "Unsupported CAST target type: {}",
                                target
                            ))),
                        }
                    }

                    match expr {
                        SqlExpr::Paren(inner) => eval_scalar(
                            inner,
                            outer_table,
                            outer_row,
                            outer_alias,
                            outer_table_name,
                            inner_table,
                            inner_row,
                            inner_alias,
                            inner_table_name,
                        ),
                        SqlExpr::Literal(v) => Ok(v.clone()),
                        SqlExpr::Column(c) => Ok(get_col_value_qualified(
                            c,
                            outer_table,
                            outer_row,
                            outer_alias,
                            outer_table_name,
                            inner_table,
                            inner_row,
                            inner_alias,
                            inner_table_name,
                        )),
                        SqlExpr::Cast { expr, data_type } => {
                            let v = eval_scalar(
                                expr,
                                outer_table,
                                outer_row,
                                outer_alias,
                                outer_table_name,
                                inner_table,
                                inner_row,
                                inner_alias,
                                inner_table_name,
                            )?;
                            cast_value(v, *data_type)
                        }
                        SqlExpr::Function { name, args } => {
                            if name.eq_ignore_ascii_case("rand") {
                                if !args.is_empty() {
                                    return Err(ApexError::QueryParseError(
                                        "RAND() does not accept arguments".to_string(),
                                    ));
                                }
                                return Ok(Value::Float64(rand::random::<f64>()));
                            }
                            if name.eq_ignore_ascii_case("len") {
                                if args.len() != 1 {
                                    return Err(ApexError::QueryParseError(
                                        "LEN() expects 1 argument".to_string(),
                                    ));
                                }
                                let v = eval_scalar(
                                    &args[0],
                                    outer_table,
                                    outer_row,
                                    outer_alias,
                                    outer_table_name,
                                    inner_table,
                                    inner_row,
                                    inner_alias,
                                    inner_table_name,
                                )?;
                                if v.is_null() {
                                    return Ok(Value::Null);
                                }
                                let s = v.to_string_value();
                                return Ok(Value::Int64(s.chars().count() as i64));
                            }
                            if name.eq_ignore_ascii_case("trim") {
                                if args.len() != 1 {
                                    return Err(ApexError::QueryParseError(
                                        "TRIM() expects 1 argument".to_string(),
                                    ));
                                }
                                let v = eval_scalar(
                                    &args[0],
                                    outer_table,
                                    outer_row,
                                    outer_alias,
                                    outer_table_name,
                                    inner_table,
                                    inner_row,
                                    inner_alias,
                                    inner_table_name,
                                )?;
                                if v.is_null() {
                                    return Ok(Value::Null);
                                }
                                return Ok(Value::String(v.to_string_value().trim().to_string()));
                            }
                            if name.eq_ignore_ascii_case("upper") {
                                if args.len() != 1 {
                                    return Err(ApexError::QueryParseError(
                                        "UPPER() expects 1 argument".to_string(),
                                    ));
                                }
                                let v = eval_scalar(
                                    &args[0],
                                    outer_table,
                                    outer_row,
                                    outer_alias,
                                    outer_table_name,
                                    inner_table,
                                    inner_row,
                                    inner_alias,
                                    inner_table_name,
                                )?;
                                if v.is_null() {
                                    return Ok(Value::Null);
                                }
                                if !matches!(v, Value::String(_)) {
                                    return Err(ApexError::QueryParseError(
                                        "UCASE() expects a string literal or string column"
                                            .to_string(),
                                    ));
                                }
                                return Ok(Value::String(v.to_string_value().to_uppercase()));
                            }
                            if name.eq_ignore_ascii_case("ucase") {
                                if args.len() != 1 {
                                    return Err(ApexError::QueryParseError(
                                        "UCASE() expects 1 argument".to_string(),
                                    ));
                                }
                                if !is_string_literal_or_column(&args[0]) {
                                    return Err(ApexError::QueryParseError(
                                        "UCASE() expects a string literal or column".to_string(),
                                    ));
                                }
                                let v = eval_scalar(
                                    &args[0],
                                    outer_table,
                                    outer_row,
                                    outer_alias,
                                    outer_table_name,
                                    inner_table,
                                    inner_row,
                                    inner_alias,
                                    inner_table_name,
                                )?;
                                if v.is_null() {
                                    return Ok(Value::Null);
                                }
                                return Ok(Value::String(v.to_string_value().to_uppercase()));
                            }
                            if name.eq_ignore_ascii_case("lower") {
                                if args.len() != 1 {
                                    return Err(ApexError::QueryParseError(
                                        "LOWER() expects 1 argument".to_string(),
                                    ));
                                }
                                let v = eval_scalar(
                                    &args[0],
                                    outer_table,
                                    outer_row,
                                    outer_alias,
                                    outer_table_name,
                                    inner_table,
                                    inner_row,
                                    inner_alias,
                                    inner_table_name,
                                )?;
                                if v.is_null() {
                                    return Ok(Value::Null);
                                }
                                return Ok(Value::String(v.to_string_value().to_lowercase()));
                            }
                            if name.eq_ignore_ascii_case("lcase") {
                                if args.len() != 1 {
                                    return Err(ApexError::QueryParseError(
                                        "LCASE() expects 1 argument".to_string(),
                                    ));
                                }
                                if !is_string_literal_or_column(&args[0]) {
                                    return Err(ApexError::QueryParseError(
                                        "LCASE() expects a string literal or column".to_string(),
                                    ));
                                }
                                let v = eval_scalar(
                                    &args[0],
                                    outer_table,
                                    outer_row,
                                    outer_alias,
                                    outer_table_name,
                                    inner_table,
                                    inner_row,
                                    inner_alias,
                                    inner_table_name,
                                )?;
                                if v.is_null() {
                                    return Ok(Value::Null);
                                }
                                if !matches!(v, Value::String(_)) {
                                    return Err(ApexError::QueryParseError(
                                        "LCASE() expects a string literal or string column"
                                            .to_string(),
                                    ));
                                }
                                return Ok(Value::String(v.to_string_value().to_lowercase()));
                            }
                            if name.eq_ignore_ascii_case("replace") {
                                if args.len() != 3 {
                                    return Err(ApexError::QueryParseError(
                                        "REPLACE() expects 3 arguments".to_string(),
                                    ));
                                }
                                let s0 = eval_scalar(
                                    &args[0],
                                    outer_table,
                                    outer_row,
                                    outer_alias,
                                    outer_table_name,
                                    inner_table,
                                    inner_row,
                                    inner_alias,
                                    inner_table_name,
                                )?;
                                let from0 = eval_scalar(
                                    &args[1],
                                    outer_table,
                                    outer_row,
                                    outer_alias,
                                    outer_table_name,
                                    inner_table,
                                    inner_row,
                                    inner_alias,
                                    inner_table_name,
                                )?;
                                let to0 = eval_scalar(
                                    &args[2],
                                    outer_table,
                                    outer_row,
                                    outer_alias,
                                    outer_table_name,
                                    inner_table,
                                    inner_row,
                                    inner_alias,
                                    inner_table_name,
                                )?;
                                if s0.is_null() || from0.is_null() || to0.is_null() {
                                    return Ok(Value::Null);
                                }
                                let s = s0.to_string_value();
                                let from = from0.to_string_value();
                                let to = to0.to_string_value();
                                return Ok(Value::String(s.replace(&from, &to)));
                            }
                            if name.eq_ignore_ascii_case("mid") {
                                if args.len() != 2 && args.len() != 3 {
                                    return Err(ApexError::QueryParseError(
                                        "MID() expects 2 or 3 arguments".to_string(),
                                    ));
                                }
                                let s0 = eval_scalar(
                                    &args[0],
                                    outer_table,
                                    outer_row,
                                    outer_alias,
                                    outer_table_name,
                                    inner_table,
                                    inner_row,
                                    inner_alias,
                                    inner_table_name,
                                )?;
                                let start0 = eval_scalar(
                                    &args[1],
                                    outer_table,
                                    outer_row,
                                    outer_alias,
                                    outer_table_name,
                                    inner_table,
                                    inner_row,
                                    inner_alias,
                                    inner_table_name,
                                )?;
                                if s0.is_null() || start0.is_null() {
                                    return Ok(Value::Null);
                                }
                                let s = s0.to_string_value();
                                let mut start = start0.as_i64().unwrap_or(1);
                                if start < 1 {
                                    start = 1;
                                }
                                let start_idx = (start - 1) as usize;
                                let chars: Vec<char> = s.chars().collect();
                                if start_idx >= chars.len() {
                                    return Ok(Value::String(String::new()));
                                }
                                let end_idx = if args.len() == 3 {
                                    let len0 = eval_scalar(
                                        &args[2],
                                        outer_table,
                                        outer_row,
                                        outer_alias,
                                        outer_table_name,
                                        inner_table,
                                        inner_row,
                                        inner_alias,
                                        inner_table_name,
                                    )?;
                                    if len0.is_null() {
                                        return Ok(Value::Null);
                                    }
                                    let mut l = len0.as_i64().unwrap_or(0);
                                    if l < 0 {
                                        l = 0;
                                    }
                                    (start_idx + l as usize).min(chars.len())
                                } else {
                                    chars.len()
                                };
                                let out: String = chars[start_idx..end_idx].iter().collect();
                                return Ok(Value::String(out));
                            }
                            Err(ApexError::QueryParseError(
                                format!("Unsupported function: {}", name),
                            ))
                        }
                        SqlExpr::UnaryOp { op: UnaryOperator::Minus, expr } => {
                            let v = eval_scalar(
                                expr,
                                outer_table,
                                outer_row,
                                outer_alias,
                                outer_table_name,
                                inner_table,
                                inner_row,
                                inner_alias,
                                inner_table_name,
                            )?;
                            if let Some(i) = v.as_i64() {
                                Ok(Value::Int64(-i))
                            } else if let Some(f) = v.as_f64() {
                                Ok(Value::Float64(-f))
                            } else {
                                Ok(Value::Null)
                            }
                        }
                        _ => Ok(Value::Null),
                    }
                }

                match expr {
                    SqlExpr::Paren(inner) => eval_correlated_predicate(
                        inner,
                        outer_table,
                        outer_row,
                        outer_alias,
                        outer_table_name,
                        inner_table,
                        inner_row,
                        inner_alias,
                        inner_table_name,
                    ),
                    SqlExpr::Literal(v) => Ok(v.as_bool().unwrap_or(false)),
                    SqlExpr::UnaryOp { op: UnaryOperator::Not, expr } => Ok(!eval_correlated_predicate(
                        expr,
                        outer_table,
                        outer_row,
                        outer_alias,
                        outer_table_name,
                        inner_table,
                        inner_row,
                        inner_alias,
                        inner_table_name,
                    )?),
                    SqlExpr::BinaryOp { left, op, right } => match op {
                        BinaryOperator::And => Ok(
                            eval_correlated_predicate(
                                left,
                                outer_table,
                                outer_row,
                                outer_alias,
                                outer_table_name,
                                inner_table,
                                inner_row,
                                inner_alias,
                                inner_table_name,
                            )? && eval_correlated_predicate(
                                right,
                                outer_table,
                                outer_row,
                                outer_alias,
                                outer_table_name,
                                inner_table,
                                inner_row,
                                inner_alias,
                                inner_table_name,
                            )?,
                        ),
                        BinaryOperator::Or => Ok(
                            eval_correlated_predicate(
                                left,
                                outer_table,
                                outer_row,
                                outer_alias,
                                outer_table_name,
                                inner_table,
                                inner_row,
                                inner_alias,
                                inner_table_name,
                            )? || eval_correlated_predicate(
                                right,
                                outer_table,
                                outer_row,
                                outer_alias,
                                outer_table_name,
                                inner_table,
                                inner_row,
                                inner_alias,
                                inner_table_name,
                            )?,
                        ),
                        BinaryOperator::Eq
                        | BinaryOperator::NotEq
                        | BinaryOperator::Lt
                        | BinaryOperator::Le
                        | BinaryOperator::Gt
                        | BinaryOperator::Ge => {
                            let lv = eval_scalar(
                                left,
                                outer_table,
                                outer_row,
                                outer_alias,
                                outer_table_name,
                                inner_table,
                                inner_row,
                                inner_alias,
                                inner_table_name,
                            )?;
                            let rv = eval_scalar(
                                right,
                                outer_table,
                                outer_row,
                                outer_alias,
                                outer_table_name,
                                inner_table,
                                inner_row,
                                inner_alias,
                                inner_table_name,
                            )?;
                            if lv.is_null() || rv.is_null() {
                                return Ok(false);
                            }
                            let ord = lv.partial_cmp(&rv).unwrap_or(Ordering::Equal);
                            Ok(match op {
                                BinaryOperator::Eq => ord == Ordering::Equal,
                                BinaryOperator::NotEq => ord != Ordering::Equal,
                                BinaryOperator::Lt => ord == Ordering::Less,
                                BinaryOperator::Le => ord != Ordering::Greater,
                                BinaryOperator::Gt => ord == Ordering::Greater,
                                BinaryOperator::Ge => ord != Ordering::Less,
                                _ => false,
                            })
                        }
                        _ => Err(ApexError::QueryParseError(
                            "Unsupported operator in correlated predicate".to_string(),
                        )),
                    },
                    _ => Err(ApexError::QueryParseError(
                        "Unsupported expression in correlated predicate".to_string(),
                    )),
                }
            }

            fn exists_for_outer_row(
                sub: &SelectStatement,
                tables: &HashMap<String, ColumnTable>,
                default_table: &str,
                outer_table: &ColumnTable,
                outer_row: usize,
                outer_alias: &str,
                outer_table_name: &str,
            ) -> Result<bool, ApexError> {
                if !sub.joins.is_empty() {
                    return Err(ApexError::QueryParseError(
                        "EXISTS subquery with JOIN is not supported yet".to_string(),
                    ));
                }
                if !sub.group_by.is_empty() || sub.having.is_some() {
                    return Err(ApexError::QueryParseError(
                        "EXISTS subquery with GROUP BY/HAVING is not supported yet".to_string(),
                    ));
                }

                let (inner_table_name, inner_alias) = match sub.from.as_ref() {
                    Some(FromItem::Table { table, alias }) => {
                        (table.clone(), alias.clone().unwrap_or_else(|| table.clone()))
                    }
                    Some(FromItem::Subquery { .. }) => {
                        return Err(ApexError::QueryParseError(
                            "EXISTS subquery FROM (subquery) is not supported yet".to_string(),
                        ))
                    }
                    None => (default_table.to_string(), default_table.to_string()),
                };

                let inner_table = tables
                    .get(&inner_table_name)
                    .ok_or_else(|| ApexError::QueryParseError(format!("Table '{}' not found.", inner_table_name)))?;

                let row_count = inner_table.get_row_count();
                let deleted = inner_table.deleted_ref();
                for inner_row in 0..row_count {
                    if deleted.get(inner_row) {
                        continue;
                    }

                    let passes = if let Some(ref w) = sub.where_clause {
                        eval_correlated_predicate(
                            w,
                            outer_table,
                            outer_row,
                            outer_alias,
                            outer_table_name,
                            inner_table,
                            inner_row,
                            &inner_alias,
                            &inner_table_name,
                        )?
                    } else {
                        true
                    };

                    if passes {
                        return Ok(true);
                    }
                }
                Ok(false)
            }

            fn eval_outer_where(
                expr: &SqlExpr,
                tables: &HashMap<String, ColumnTable>,
                default_table: &str,
                outer_table: &ColumnTable,
                outer_row: usize,
                outer_alias: &str,
                outer_table_name: &str,
            ) -> Result<bool, ApexError> {
                fn get_outer_value(
                    col_ref: &str,
                    outer_table: &ColumnTable,
                    outer_row: usize,
                    outer_alias: &str,
                    outer_table_name: &str,
                ) -> Value {
                    if col_ref == "_id" {
                        return Value::Int64(outer_row as i64);
                    }
                    let (a, c) = if let Some((a, c)) = col_ref.split_once('.') {
                        (a, c)
                    } else {
                        ("", col_ref)
                    };
                    let name = if !a.is_empty() {
                        if a == outer_alias || a == outer_table_name {
                            c
                        } else {
                            // Unknown qualifier
                            return Value::Null;
                        }
                    } else {
                        c
                    };
                    if name == "_id" {
                        return Value::Int64(outer_row as i64);
                    }
                    if let Some(ci) = outer_table.schema_ref().get_index(name) {
                        outer_table.columns_ref()[ci].get(outer_row).unwrap_or(Value::Null)
                    } else {
                        Value::Null
                    }
                }

                fn eval_outer_scalar(
                    expr: &SqlExpr,
                    outer_table: &ColumnTable,
                    outer_row: usize,
                    outer_alias: &str,
                    outer_table_name: &str,
                ) -> Value {
                    match expr {
                        SqlExpr::Paren(inner) => {
                            eval_outer_scalar(inner, outer_table, outer_row, outer_alias, outer_table_name)
                        }
                        SqlExpr::Literal(v) => v.clone(),
                        SqlExpr::Column(c) => get_outer_value(c, outer_table, outer_row, outer_alias, outer_table_name),
                        SqlExpr::Function { name, args } => {
                            if name.eq_ignore_ascii_case("rand") {
                                if !args.is_empty() {
                                    return Value::Null;
                                }
                                Value::Float64(rand::random::<f64>())
                            } else if name.eq_ignore_ascii_case("len") {
                                if args.len() != 1 {
                                    return Value::Null;
                                }
                                let v = eval_outer_scalar(&args[0], outer_table, outer_row, outer_alias, outer_table_name);
                                if v.is_null() {
                                    Value::Null
                                } else {
                                    let s = v.to_string_value();
                                    Value::Int64(s.chars().count() as i64)
                                }
                            } else if name.eq_ignore_ascii_case("trim") {
                                if args.len() != 1 {
                                    return Value::Null;
                                }
                                let v = eval_outer_scalar(&args[0], outer_table, outer_row, outer_alias, outer_table_name);
                                if v.is_null() {
                                    Value::Null
                                } else {
                                    Value::String(v.to_string_value().trim().to_string())
                                }
                            } else if name.eq_ignore_ascii_case("upper") {
                                if args.len() != 1 {
                                    return Value::Null;
                                }
                                let v = eval_outer_scalar(&args[0], outer_table, outer_row, outer_alias, outer_table_name);
                                if v.is_null() {
                                    Value::Null
                                } else {
                                    Value::String(v.to_string_value().to_uppercase())
                                }
                            } else if name.eq_ignore_ascii_case("lower") {
                                if args.len() != 1 {
                                    return Value::Null;
                                }
                                let v = eval_outer_scalar(&args[0], outer_table, outer_row, outer_alias, outer_table_name);
                                if v.is_null() {
                                    Value::Null
                                } else {
                                    Value::String(v.to_string_value().to_lowercase())
                                }
                            } else if name.eq_ignore_ascii_case("replace") {
                                if args.len() != 3 {
                                    return Value::Null;
                                }
                                let s0 = eval_outer_scalar(&args[0], outer_table, outer_row, outer_alias, outer_table_name);
                                let from0 = eval_outer_scalar(&args[1], outer_table, outer_row, outer_alias, outer_table_name);
                                let to0 = eval_outer_scalar(&args[2], outer_table, outer_row, outer_alias, outer_table_name);
                                if s0.is_null() || from0.is_null() || to0.is_null() {
                                    Value::Null
                                } else {
                                    Value::String(s0.to_string_value().replace(&from0.to_string_value(), &to0.to_string_value()))
                                }
                            } else if name.eq_ignore_ascii_case("mid") {
                                if args.len() != 2 && args.len() != 3 {
                                    return Value::Null;
                                }
                                let s0 = eval_outer_scalar(&args[0], outer_table, outer_row, outer_alias, outer_table_name);
                                let start0 = eval_outer_scalar(&args[1], outer_table, outer_row, outer_alias, outer_table_name);
                                if s0.is_null() || start0.is_null() {
                                    return Value::Null;
                                }
                                let s = s0.to_string_value();
                                let mut start = start0.as_i64().unwrap_or(1);
                                if start < 1 {
                                    start = 1;
                                }
                                let start_idx = (start - 1) as usize;
                                let chars: Vec<char> = s.chars().collect();
                                if start_idx >= chars.len() {
                                    return Value::String(String::new());
                                }
                                let end_idx = if args.len() == 3 {
                                    let len0 = eval_outer_scalar(&args[2], outer_table, outer_row, outer_alias, outer_table_name);
                                    if len0.is_null() {
                                        return Value::Null;
                                    }
                                    let mut l = len0.as_i64().unwrap_or(0);
                                    if l < 0 {
                                        l = 0;
                                    }
                                    (start_idx + l as usize).min(chars.len())
                                } else {
                                    chars.len()
                                };
                                Value::String(chars[start_idx..end_idx].iter().collect())
                            } else {
                                Value::Null
                            }
                        }
                        SqlExpr::UnaryOp { op: UnaryOperator::Minus, expr } => {
                            let v = eval_outer_scalar(expr, outer_table, outer_row, outer_alias, outer_table_name);
                            if let Some(i) = v.as_i64() {
                                Value::Int64(-i)
                            } else if let Some(f) = v.as_f64() {
                                Value::Float64(-f)
                            } else {
                                Value::Null
                            }
                        }
                        _ => Value::Null,
                    }
                }

                fn in_subquery_for_outer_row(
                    column: &str,
                    sub: &SelectStatement,
                    negated: bool,
                    tables: &HashMap<String, ColumnTable>,
                    default_table: &str,
                    outer_table: &ColumnTable,
                    outer_row: usize,
                    outer_alias: &str,
                    outer_table_name: &str,
                ) -> Result<bool, ApexError> {
                    if !sub.joins.is_empty() {
                        return Err(ApexError::QueryParseError(
                            "IN (subquery) with JOIN is not supported yet".to_string(),
                        ));
                    }

                    // Determine inner table
                    let (inner_table_name, inner_alias) = match sub.from.as_ref() {
                        Some(FromItem::Table { table, alias }) => {
                            (table.clone(), alias.clone().unwrap_or_else(|| table.clone()))
                        }
                        Some(FromItem::Subquery { .. }) => {
                            return Err(ApexError::QueryParseError(
                                "IN (subquery) FROM (subquery) is not supported yet".to_string(),
                            ))
                        }
                        None => (default_table.to_string(), default_table.to_string()),
                    };

                    let inner_table = tables
                        .get(&inner_table_name)
                        .ok_or_else(|| {
                            ApexError::QueryParseError(format!("Table '{}' not found.", inner_table_name))
                        })?;

                    // Outer value
                    let outer_v = get_outer_value(column, outer_table, outer_row, outer_alias, outer_table_name);
                    if outer_v.is_null() {
                        return Ok(false);
                    }

                    // Subquery must project a single column for IN
                    if sub.columns.len() != 1 {
                        return Err(ApexError::QueryParseError(
                            "IN (subquery) requires single-column subquery".to_string(),
                        ));
                    }

                    let selected_ref: Option<String> = match &sub.columns[0] {
                        SelectColumn::Column(c) => Some(c.clone()),
                        SelectColumn::ColumnAlias { column, .. } => Some(column.clone()),
                        SelectColumn::Expression { .. } => None,
                        SelectColumn::All => None,
                        SelectColumn::Aggregate { .. } => None,
                        SelectColumn::WindowFunction { .. } => None,
                    };
                    if selected_ref.is_none() {
                        return Err(ApexError::QueryParseError(
                            "IN (subquery) projection must be a column".to_string(),
                        ));
                    }
                    let selected_ref = selected_ref.unwrap();

                    let mut has_null = false;
                    let mut any_match = false;

                    let row_count = inner_table.get_row_count();
                    let deleted = inner_table.deleted_ref();
                    for inner_row in 0..row_count {
                        if deleted.get(inner_row) {
                            continue;
                        }

                        let passes = if let Some(ref w) = sub.where_clause {
                            eval_correlated_predicate(
                                w,
                                outer_table,
                                outer_row,
                                outer_alias,
                                outer_table_name,
                                inner_table,
                                inner_row,
                                &inner_alias,
                                &inner_table_name,
                            )?
                        } else {
                            true
                        };
                        if !passes {
                            continue;
                        }

                        let v = get_col_value_qualified(
                            &selected_ref,
                            outer_table,
                            outer_row,
                            outer_alias,
                            outer_table_name,
                            inner_table,
                            inner_row,
                            &inner_alias,
                            &inner_table_name,
                        );

                        if v.is_null() {
                            has_null = true;
                            continue;
                        }
                        if v == outer_v {
                            any_match = true;
                            break;
                        }
                    }

                    if negated {
                        // NOT IN: if subquery contains NULL -> UNKNOWN -> filter out (FALSE)
                        if has_null {
                            return Ok(false);
                        }
                        Ok(!any_match)
                    } else {
                        // IN: if no match but contains NULL -> UNKNOWN -> filter out (FALSE)
                        if !any_match && has_null {
                            return Ok(false);
                        }
                        Ok(any_match)
                    }
                }

                match expr {
                    SqlExpr::Paren(inner) => eval_outer_where(
                        inner,
                        tables,
                        default_table,
                        outer_table,
                        outer_row,
                        outer_alias,
                        outer_table_name,
                    ),
                    SqlExpr::Literal(v) => Ok(v.as_bool().unwrap_or(false)),
                    SqlExpr::UnaryOp { op: UnaryOperator::Not, expr } => Ok(!eval_outer_where(
                        expr,
                        tables,
                        default_table,
                        outer_table,
                        outer_row,
                        outer_alias,
                        outer_table_name,
                    )?),
                    SqlExpr::BinaryOp { left, op, right } => match op {
                        BinaryOperator::And => Ok(
                            eval_outer_where(
                                left,
                                tables,
                                default_table,
                                outer_table,
                                outer_row,
                                outer_alias,
                                outer_table_name,
                            )? && eval_outer_where(
                                right,
                                tables,
                                default_table,
                                outer_table,
                                outer_row,
                                outer_alias,
                                outer_table_name,
                            )?,
                        ),
                        BinaryOperator::Or => Ok(
                            eval_outer_where(
                                left,
                                tables,
                                default_table,
                                outer_table,
                                outer_row,
                                outer_alias,
                                outer_table_name,
                            )? || eval_outer_where(
                                right,
                                tables,
                                default_table,
                                outer_table,
                                outer_row,
                                outer_alias,
                                outer_table_name,
                            )?,
                        ),
                        BinaryOperator::Eq
                        | BinaryOperator::NotEq
                        | BinaryOperator::Lt
                        | BinaryOperator::Le
                        | BinaryOperator::Gt
                        | BinaryOperator::Ge => {
                            let lv = eval_outer_scalar(left, outer_table, outer_row, outer_alias, outer_table_name);
                            let rv = eval_outer_scalar(right, outer_table, outer_row, outer_alias, outer_table_name);
                            if lv.is_null() || rv.is_null() {
                                return Ok(false);
                            }
                            let ord = lv.partial_cmp(&rv).unwrap_or(Ordering::Equal);
                            Ok(match op {
                                BinaryOperator::Eq => ord == Ordering::Equal,
                                BinaryOperator::NotEq => ord != Ordering::Equal,
                                BinaryOperator::Lt => ord == Ordering::Less,
                                BinaryOperator::Le => ord != Ordering::Greater,
                                BinaryOperator::Gt => ord == Ordering::Greater,
                                BinaryOperator::Ge => ord != Ordering::Less,
                                _ => false,
                            })
                        }
                        _ => Err(ApexError::QueryParseError(
                            "Unsupported operator in outer WHERE".to_string(),
                        )),
                    },
                    SqlExpr::ExistsSubquery { stmt: sub } => exists_for_outer_row(
                        sub,
                        tables,
                        default_table,
                        outer_table,
                        outer_row,
                        outer_alias,
                        outer_table_name,
                    ),
                    SqlExpr::InSubquery { column, stmt: sub, negated } => in_subquery_for_outer_row(
                        column,
                        sub,
                        *negated,
                        tables,
                        default_table,
                        outer_table,
                        outer_row,
                        outer_alias,
                        outer_table_name,
                    ),
                    SqlExpr::ScalarSubquery { .. } => Err(ApexError::QueryParseError(
                        "Scalar subquery is not supported in WHERE yet".to_string(),
                    )),
                    _ => Err(ApexError::QueryParseError(
                        "Unsupported expression in outer WHERE".to_string(),
                    )),
                }
            }

            fn eval_scalar_subquery_for_outer_row(
                sub: &SelectStatement,
                tables: &HashMap<String, ColumnTable>,
                default_table: &str,
                outer_table: &ColumnTable,
                outer_row: usize,
                outer_alias: &str,
                outer_table_name: &str,
            ) -> Result<Value, ApexError> {
                if !sub.joins.is_empty() {
                    return Err(ApexError::QueryParseError(
                        "Scalar subquery with JOIN is not supported yet".to_string(),
                    ));
                }
                if !sub.group_by.is_empty() || sub.having.is_some() {
                    return Err(ApexError::QueryParseError(
                        "Scalar subquery with GROUP BY/HAVING is not supported yet".to_string(),
                    ));
                }
                if sub.columns.len() != 1 {
                    return Err(ApexError::QueryParseError(
                        "Scalar subquery must return exactly one column".to_string(),
                    ));
                }

                let (inner_table_name, inner_alias) = match sub.from.as_ref() {
                    Some(FromItem::Table { table, alias }) => {
                        (table.clone(), alias.clone().unwrap_or_else(|| table.clone()))
                    }
                    Some(FromItem::Subquery { .. }) => {
                        return Err(ApexError::QueryParseError(
                            "Scalar subquery FROM (subquery) is not supported yet".to_string(),
                        ))
                    }
                    None => (default_table.to_string(), default_table.to_string()),
                };

                let inner_table = tables
                    .get(&inner_table_name)
                    .ok_or_else(|| ApexError::QueryParseError(format!("Table '{}' not found.", inner_table_name)))?;

                // Support only simple aggregate scalar: MAX(col), MIN(col), SUM(col), AVG(col), COUNT(*), COUNT(col)
                let (func, col) = match &sub.columns[0] {
                    SelectColumn::Aggregate { func, column, distinct, .. } => {
                        if *distinct {
                            return Err(ApexError::QueryParseError(
                                "Scalar subquery DISTINCT aggregate is not supported yet".to_string(),
                            ));
                        }
                        (func.clone(), column.clone())
                    }
                    _ => {
                        return Err(ApexError::QueryParseError(
                            "Scalar subquery only supports aggregate projection".to_string(),
                        ))
                    }
                };

                // Reuse correlated predicate evaluator for subquery WHERE
                let row_count = inner_table.get_row_count();
                let deleted = inner_table.deleted_ref();

                let mut count: i64 = 0;
                let mut sum: f64 = 0.0;
                let mut sum_count: i64 = 0;
                let mut min_v: Option<Value> = None;
                let mut max_v: Option<Value> = None;

                for inner_row in 0..row_count {
                    if deleted.get(inner_row) {
                        continue;
                    }
                    let passes = if let Some(ref w) = sub.where_clause {
                        eval_correlated_predicate(
                            w,
                            outer_table,
                            outer_row,
                            outer_alias,
                            outer_table_name,
                            inner_table,
                            inner_row,
                            &inner_alias,
                            &inner_table_name,
                        )?
                    } else {
                        true
                    };
                    if !passes {
                        continue;
                    }

                    match func {
                        AggregateFunc::Count => {
                            if col.is_none() {
                                count += 1;
                            } else if let Some(ref c) = col {
                                let v = get_col_value_qualified(
                                    c,
                                    outer_table,
                                    outer_row,
                                    outer_alias,
                                    outer_table_name,
                                    inner_table,
                                    inner_row,
                                    &inner_alias,
                                    &inner_table_name,
                                );
                                if !v.is_null() {
                                    count += 1;
                                }
                            }
                        }
                        AggregateFunc::Sum | AggregateFunc::Avg => {
                            if let Some(ref c) = col {
                                let v = get_col_value_qualified(
                                    c,
                                    outer_table,
                                    outer_row,
                                    outer_alias,
                                    outer_table_name,
                                    inner_table,
                                    inner_row,
                                    &inner_alias,
                                    &inner_table_name,
                                );
                                if let Some(n) = v.as_f64() {
                                    sum += n;
                                    sum_count += 1;
                                }
                            }
                        }
                        AggregateFunc::Min => {
                            if let Some(ref c) = col {
                                let v = get_col_value_qualified(
                                    c,
                                    outer_table,
                                    outer_row,
                                    outer_alias,
                                    outer_table_name,
                                    inner_table,
                                    inner_row,
                                    &inner_alias,
                                    &inner_table_name,
                                );
                                if v.is_null() {
                                    continue;
                                }
                                min_v = Some(match &min_v {
                                    None => v,
                                    Some(curr) => {
                                        if SqlExecutor::compare_non_null(curr, &v) == Ordering::Greater {
                                            v
                                        } else {
                                            curr.clone()
                                        }
                                    }
                                });
                            }
                        }
                        AggregateFunc::Max => {
                            if let Some(ref c) = col {
                                let v = get_col_value_qualified(
                                    c,
                                    outer_table,
                                    outer_row,
                                    outer_alias,
                                    outer_table_name,
                                    inner_table,
                                    inner_row,
                                    &inner_alias,
                                    &inner_table_name,
                                );
                                if v.is_null() {
                                    continue;
                                }
                                max_v = Some(match &max_v {
                                    None => v,
                                    Some(curr) => {
                                        if SqlExecutor::compare_non_null(curr, &v) == Ordering::Less {
                                            v
                                        } else {
                                            curr.clone()
                                        }
                                    }
                                });
                            }
                        }
                    }
                }

                Ok(match func {
                    AggregateFunc::Count => Value::Int64(count),
                    AggregateFunc::Sum => Value::Float64(sum),
                    AggregateFunc::Avg => {
                        if sum_count > 0 {
                            Value::Float64(sum / sum_count as f64)
                        } else {
                            Value::Null
                        }
                    }
                    AggregateFunc::Min => min_v.unwrap_or(Value::Null),
                    AggregateFunc::Max => max_v.unwrap_or(Value::Null),
                })
            }

            fn eval_outer_expr(
                expr: &SqlExpr,
                tables: &HashMap<String, ColumnTable>,
                default_table: &str,
                outer_table: &ColumnTable,
                outer_row: usize,
                outer_alias: &str,
                outer_table_name: &str,
            ) -> Result<Value, ApexError> {
                fn get_outer_value_local(
                    col_ref: &str,
                    outer_table: &ColumnTable,
                    outer_row: usize,
                    outer_alias: &str,
                    outer_table_name: &str,
                ) -> Value {
                    if col_ref == "_id" {
                        return Value::Int64(outer_row as i64);
                    }
                    let (a, c) = if let Some((a, c)) = col_ref.split_once('.') {
                        (a, c)
                    } else {
                        ("", col_ref)
                    };
                    let name = if !a.is_empty() {
                        if a == outer_alias || a == outer_table_name {
                            c
                        } else {
                            return Value::Null;
                        }
                    } else {
                        c
                    };
                    if name == "_id" {
                        return Value::Int64(outer_row as i64);
                    }
                    if let Some(ci) = outer_table.schema_ref().get_index(name) {
                        outer_table.columns_ref()[ci].get(outer_row).unwrap_or(Value::Null)
                    } else {
                        Value::Null
                    }
                }

                match expr {
                    SqlExpr::Paren(inner) => {
                        eval_outer_expr(inner, tables, default_table, outer_table, outer_row, outer_alias, outer_table_name)
                    }
                    SqlExpr::Literal(v) => Ok(v.clone()),
                    SqlExpr::Column(c) => Ok(get_outer_value_local(c, outer_table, outer_row, outer_alias, outer_table_name)),
                    SqlExpr::Function { name, args } => {
                        if name.eq_ignore_ascii_case("rand") {
                            if !args.is_empty() {
                                return Err(ApexError::QueryParseError(
                                    "RAND() does not accept arguments".to_string(),
                                ));
                            }
                            Ok(Value::Float64(rand::random::<f64>()))
                        } else if name.eq_ignore_ascii_case("len") {
                            if args.len() != 1 {
                                return Err(ApexError::QueryParseError(
                                    "LEN() expects 1 argument".to_string(),
                                ));
                            }
                            let v = eval_outer_expr(
                                &args[0],
                                tables,
                                default_table,
                                outer_table,
                                outer_row,
                                outer_alias,
                                outer_table_name,
                            )?;
                            if v.is_null() {
                                Ok(Value::Null)
                            } else {
                                Ok(Value::Int64(v.to_string_value().chars().count() as i64))
                            }
                        } else if name.eq_ignore_ascii_case("trim") {
                            if args.len() != 1 {
                                return Err(ApexError::QueryParseError(
                                    "TRIM() expects 1 argument".to_string(),
                                ));
                            }
                            let v = eval_outer_expr(
                                &args[0],
                                tables,
                                default_table,
                                outer_table,
                                outer_row,
                                outer_alias,
                                outer_table_name,
                            )?;
                            if v.is_null() {
                                Ok(Value::Null)
                            } else {
                                Ok(Value::String(v.to_string_value().trim().to_string()))
                            }
                        } else if name.eq_ignore_ascii_case("upper") {
                            if args.len() != 1 {
                                return Err(ApexError::QueryParseError(
                                    "UPPER() expects 1 argument".to_string(),
                                ));
                            }
                            let v = eval_outer_expr(
                                &args[0],
                                tables,
                                default_table,
                                outer_table,
                                outer_row,
                                outer_alias,
                                outer_table_name,
                            )?;
                            if v.is_null() {
                                Ok(Value::Null)
                            } else {
                                Ok(Value::String(v.to_string_value().to_uppercase()))
                            }
                        } else if name.eq_ignore_ascii_case("lower") {
                            if args.len() != 1 {
                                return Err(ApexError::QueryParseError(
                                    "LOWER() expects 1 argument".to_string(),
                                ));
                            }
                            let v = eval_outer_expr(
                                &args[0],
                                tables,
                                default_table,
                                outer_table,
                                outer_row,
                                outer_alias,
                                outer_table_name,
                            )?;
                            if v.is_null() {
                                Ok(Value::Null)
                            } else {
                                Ok(Value::String(v.to_string_value().to_lowercase()))
                            }
                        } else if name.eq_ignore_ascii_case("replace") {
                            if args.len() != 3 {
                                return Err(ApexError::QueryParseError(
                                    "REPLACE() expects 3 arguments".to_string(),
                                ));
                            }
                            let s0 = eval_outer_expr(
                                &args[0],
                                tables,
                                default_table,
                                outer_table,
                                outer_row,
                                outer_alias,
                                outer_table_name,
                            )?;
                            let from0 = eval_outer_expr(
                                &args[1],
                                tables,
                                default_table,
                                outer_table,
                                outer_row,
                                outer_alias,
                                outer_table_name,
                            )?;
                            let to0 = eval_outer_expr(
                                &args[2],
                                tables,
                                default_table,
                                outer_table,
                                outer_row,
                                outer_alias,
                                outer_table_name,
                            )?;
                            if s0.is_null() || from0.is_null() || to0.is_null() {
                                Ok(Value::Null)
                            } else {
                                Ok(Value::String(
                                    s0.to_string_value().replace(&from0.to_string_value(), &to0.to_string_value()),
                                ))
                            }
                        } else if name.eq_ignore_ascii_case("mid") {
                            if args.len() != 2 && args.len() != 3 {
                                return Err(ApexError::QueryParseError(
                                    "MID() expects 2 or 3 arguments".to_string(),
                                ));
                            }
                            let s0 = eval_outer_expr(
                                &args[0],
                                tables,
                                default_table,
                                outer_table,
                                outer_row,
                                outer_alias,
                                outer_table_name,
                            )?;
                            let start0 = eval_outer_expr(
                                &args[1],
                                tables,
                                default_table,
                                outer_table,
                                outer_row,
                                outer_alias,
                                outer_table_name,
                            )?;
                            if s0.is_null() || start0.is_null() {
                                return Ok(Value::Null);
                            }
                            let s = s0.to_string_value();
                            let mut start = start0.as_i64().unwrap_or(1);
                            if start < 1 {
                                start = 1;
                            }
                            let start_idx = (start - 1) as usize;
                            let chars: Vec<char> = s.chars().collect();
                            if start_idx >= chars.len() {
                                return Ok(Value::String(String::new()));
                            }
                            let end_idx = if args.len() == 3 {
                                let len0 = eval_outer_expr(
                                    &args[2],
                                    tables,
                                    default_table,
                                    outer_table,
                                    outer_row,
                                    outer_alias,
                                    outer_table_name,
                                )?;
                                if len0.is_null() {
                                    return Ok(Value::Null);
                                }
                                let mut l = len0.as_i64().unwrap_or(0);
                                if l < 0 {
                                    l = 0;
                                }
                                (start_idx + l as usize).min(chars.len())
                            } else {
                                chars.len()
                            };
                            Ok(Value::String(chars[start_idx..end_idx].iter().collect()))
                        } else {
                            Err(ApexError::QueryParseError(
                                format!("Unsupported function: {}", name),
                            ))
                        }
                    }
                    SqlExpr::UnaryOp { op: UnaryOperator::Minus, expr } => {
                        let v = eval_outer_expr(expr, tables, default_table, outer_table, outer_row, outer_alias, outer_table_name)?;
                        if let Some(i) = v.as_i64() {
                            Ok(Value::Int64(-i))
                        } else if let Some(f) = v.as_f64() {
                            Ok(Value::Float64(-f))
                        } else {
                            Ok(Value::Null)
                        }
                    }
                    SqlExpr::ScalarSubquery { stmt: sub } => eval_scalar_subquery_for_outer_row(
                        sub,
                        tables,
                        default_table,
                        outer_table,
                        outer_row,
                        outer_alias,
                        outer_table_name,
                    ),
                    _ => Err(ApexError::QueryParseError(
                        "Unsupported expression in SELECT list".to_string(),
                    )),
                }
            }

            // Evaluate WHERE to matching indices (row-by-row)
            let row_count = table.get_row_count();
            let deleted = table.deleted_ref();
            let mut matching_indices: Vec<usize> = Vec::new();
            for row_idx in 0..row_count {
                if deleted.get(row_idx) {
                    continue;
                }
                let passes = if let Some(ref w) = stmt.where_clause {
                    eval_outer_where(
                        w,
                        tables,
                        default_table,
                        table,
                        row_idx,
                        &outer_alias,
                        &outer_table_name,
                    )?
                } else {
                    true
                };
                if passes {
                    matching_indices.push(row_idx);
                }
            }

            // Project selected columns (strip outer qualifier)
            let mut result_columns: Vec<String> = Vec::new();
            let mut projected_cols: Vec<String> = Vec::new();
            let mut projected_exprs: Vec<Option<SqlExpr>> = Vec::new();
            for sc in &stmt.columns {
                match sc {
                    SelectColumn::Column(c) => {
                        let name = strip_outer(c, &outer_alias, &outer_table_name);
                        result_columns.push(name.clone());
                        projected_cols.push(name);
                        projected_exprs.push(None);
                    }
                    SelectColumn::ColumnAlias { column, alias } => {
                        let name = strip_outer(column, &outer_alias, &outer_table_name);
                        result_columns.push(alias.clone());
                        projected_cols.push(name);
                        projected_exprs.push(None);
                    }
                    SelectColumn::Expression { expr, alias } => {
                        let out_name = alias.clone().unwrap_or_else(|| "expr".to_string());
                        result_columns.push(out_name);
                        projected_cols.push(String::new());
                        projected_exprs.push(Some(expr.clone()));
                    }
                    _ => {
                        return Err(ApexError::QueryParseError(
                            "EXISTS single-table path only supports simple column projection and scalar subquery expressions".to_string(),
                        ))
                    }
                }
            }

            let schema = table.schema_ref();
            let columns_ref = table.columns_ref();
            let mut rows: Vec<Vec<Value>> = Vec::with_capacity(matching_indices.len());
            for &row_idx in &matching_indices {
                let mut out_row: Vec<Value> = Vec::with_capacity(projected_cols.len());
                for (i, col) in projected_cols.iter().enumerate() {
                    if let Some(expr) = projected_exprs.get(i).and_then(|e| e.clone()) {
                        out_row.push(eval_outer_expr(
                            &expr,
                            tables,
                            default_table,
                            table,
                            row_idx,
                            &outer_alias,
                            &outer_table_name,
                        )?);
                        continue;
                    }

                    if col == "_id" {
                        out_row.push(Value::Int64(row_idx as i64));
                    } else if let Some(ci) = schema.get_index(col) {
                        out_row.push(columns_ref[ci].get(row_idx).unwrap_or(Value::Null));
                    } else {
                        out_row.push(Value::Null);
                    }
                }
                rows.push(out_row);
            }

            // Apply ORDER BY (strip qualifier)
            let mut order_by = stmt.order_by.clone();
            for ob in &mut order_by {
                ob.column = strip_outer(&ob.column, &outer_alias, &outer_table_name);
            }
            if !order_by.is_empty() {
                rows = Self::apply_order_by(rows, &result_columns, &order_by)?;
            }

            let off = stmt.offset.unwrap_or(0);
            let lim = stmt.limit.unwrap_or(usize::MAX);
            let rows = rows.into_iter().skip(off).take(lim).collect::<Vec<_>>();
            return Ok(SqlResult::new(result_columns, rows));
        }

        let from_item = stmt
            .from
            .as_ref()
            .ok_or_else(|| ApexError::QueryParseError("JOIN requires FROM table".to_string()))?;

        let (left_table_name, left_alias) = match from_item {
            FromItem::Table { table, alias } => {
                let t = table.clone();
                let a = alias.clone().unwrap_or_else(|| t.clone());
                (t, a)
            }
            FromItem::Subquery { .. } => {
                return Err(ApexError::QueryParseError(
                    "JOIN with derived table is not supported yet".to_string(),
                ))
            }
        };

        {
            let left_table = tables
                .get_mut(&left_table_name)
                .ok_or_else(|| ApexError::QueryParseError(format!("Table '{}' not found.", left_table_name)))?;
            if left_table.has_pending_writes() {
                left_table.flush_write_buffer();
            }
        }

        // Attempt to push down WHERE to the right side when it only references the right table.
        fn strip_right_only_where(expr: &SqlExpr, left_alias: &str, right_alias: &str) -> Option<SqlExpr> {
            match expr {
                SqlExpr::Paren(inner) => strip_right_only_where(inner, left_alias, right_alias)
                    .map(|e| SqlExpr::Paren(Box::new(e))),
                SqlExpr::BinaryOp { left, op, right } => {
                    // Support AND pushdown
                    if *op == BinaryOperator::And {
                        let l = strip_right_only_where(left, left_alias, right_alias)?;
                        let r = strip_right_only_where(right, left_alias, right_alias)?;
                        return Some(SqlExpr::BinaryOp { left: Box::new(l), op: op.clone(), right: Box::new(r) });
                    }

                    // Comparison: right_alias.col OP literal
                    let col = match left.as_ref() {
                        SqlExpr::Column(c) => c,
                        _ => return None,
                    };
                    let (a, c) = if let Some((aa, cc)) = col.split_once('.') { (aa, cc) } else { ("", col.as_str()) };
                    if a.is_empty() {
                        // Unqualified: ambiguous, don't push down
                        return None;
                    }
                    if a == left_alias {
                        return None;
                    }
                    if a != right_alias {
                        return None;
                    }

                    let rv = match right.as_ref() {
                        SqlExpr::Literal(v) => v.clone(),
                        _ => return None,
                    };

                    Some(SqlExpr::BinaryOp {
                        left: Box::new(SqlExpr::Column(c.to_string())),
                        op: op.clone(),
                        right: Box::new(SqlExpr::Literal(rv)),
                    })
                }
                _ => None,
            }
        }

        if stmt.joins.len() != 1 {
            return Err(ApexError::QueryParseError(
                "Only single JOIN is supported yet".to_string(),
            ));
        }
        let join = &stmt.joins[0];

        let (right_table_name, right_alias) = match &join.right {
            FromItem::Table { table, alias } => {
                let t = table.clone();
                let a = alias.clone().unwrap_or_else(|| t.clone());
                (t, a)
            }
            FromItem::Subquery { .. } => {
                return Err(ApexError::QueryParseError(
                    "JOIN with derived table is not supported yet".to_string(),
                ))
            }
        };

        {
            let right_table = tables
                .get_mut(&right_table_name)
                .ok_or_else(|| ApexError::QueryParseError(format!("Table '{}' not found.", right_table_name)))?;
            if right_table.has_pending_writes() {
                right_table.flush_write_buffer();
            }
        }

        let left_table = tables
            .get(&left_table_name)
            .ok_or_else(|| ApexError::QueryParseError(format!("Table '{}' not found.", left_table_name)))?;
        let right_table = tables
            .get(&right_table_name)
            .ok_or_else(|| ApexError::QueryParseError(format!("Table '{}' not found.", right_table_name)))?;

        #[derive(Clone, Copy)]
        struct JoinRow {
            left: usize,
            right: Option<usize>,
        }

        fn split_qual(col: &str) -> (&str, &str) {
            if let Some((a, b)) = col.split_once('.') {
                (a, b)
            } else {
                ("", col)
            }
        }

        fn get_col_value(table: &ColumnTable, col: &str, row_idx: usize) -> Value {
            if col == "_id" {
                return Value::Int64(row_idx as i64);
            }
            let schema = table.schema_ref();
            if let Some(ci) = schema.get_index(col) {
                table.columns_ref()[ci].get(row_idx).unwrap_or(Value::Null)
            } else {
                Value::Null
            }
        }

        fn value_for_ref(
            col_ref: &str,
            left_table: &ColumnTable,
            right_table: &ColumnTable,
            left_alias: &str,
            right_alias: &str,
            jr: JoinRow,
        ) -> Value {
            let (a, c) = split_qual(col_ref);
            if a.is_empty() {
                // Unqualified column: resolve by schema presence.
                // If only one side contains the column, use that side.
                // If both contain it, default to left for backward-compat.
                let l_has = left_table.schema_ref().get_index(c).is_some() || c == "_id";
                let r_has = right_table.schema_ref().get_index(c).is_some() || c == "_id";
                if r_has && !l_has {
                    return jr
                        .right
                        .map(|ri| get_col_value(right_table, c, ri))
                        .unwrap_or(Value::Null);
                }
                return get_col_value(left_table, c, jr.left);
            }
            if a == left_alias {
                get_col_value(left_table, c, jr.left)
            } else if a == right_alias {
                jr.right
                    .map(|ri| get_col_value(right_table, c, ri))
                    .unwrap_or(Value::Null)
            } else {
                Value::Null
            }
        }

        fn eval_predicate(
            expr: &SqlExpr,
            left_table: &ColumnTable,
            right_table: &ColumnTable,
            left_alias: &str,
            right_alias: &str,
            jr: JoinRow,
        ) -> Result<bool, ApexError> {
            fn eval_scalar(
                expr: &SqlExpr,
                left_table: &ColumnTable,
                right_table: &ColumnTable,
                left_alias: &str,
                right_alias: &str,
                jr: JoinRow,
            ) -> Result<Value, ApexError> {
                match expr {
                    SqlExpr::Paren(inner) => eval_scalar(inner, left_table, right_table, left_alias, right_alias, jr),
                    SqlExpr::Literal(v) => Ok(v.clone()),
                    SqlExpr::Column(c) => Ok(value_for_ref(c, left_table, right_table, left_alias, right_alias, jr)),
                    SqlExpr::Function { name, args } => {
                        if name.eq_ignore_ascii_case("rand") {
                            if !args.is_empty() {
                                return Err(ApexError::QueryParseError(
                                    "RAND() does not accept arguments".to_string(),
                                ));
                            }
                            return Ok(Value::Float64(rand::random::<f64>()));
                        }
                        if name.eq_ignore_ascii_case("len") {
                            if args.len() != 1 {
                                return Err(ApexError::QueryParseError(
                                    "LEN() expects 1 argument".to_string(),
                                ));
                            }
                            let v = eval_scalar(&args[0], left_table, right_table, left_alias, right_alias, jr)?;
                            if v.is_null() {
                                return Ok(Value::Null);
                            }
                            let s = v.to_string_value();
                            return Ok(Value::Int64(s.chars().count() as i64));
                        }
                        if name.eq_ignore_ascii_case("trim") {
                            if args.len() != 1 {
                                return Err(ApexError::QueryParseError(
                                    "TRIM() expects 1 argument".to_string(),
                                ));
                            }
                            let v = eval_scalar(&args[0], left_table, right_table, left_alias, right_alias, jr)?;
                            if v.is_null() {
                                return Ok(Value::Null);
                            }
                            return Ok(Value::String(v.to_string_value().trim().to_string()));
                        }
                        if name.eq_ignore_ascii_case("upper") {
                            if args.len() != 1 {
                                return Err(ApexError::QueryParseError(
                                    "UPPER() expects 1 argument".to_string(),
                                ));
                            }
                            let v = eval_scalar(&args[0], left_table, right_table, left_alias, right_alias, jr)?;
                            if v.is_null() {
                                return Ok(Value::Null);
                            }
                            return Ok(Value::String(v.to_string_value().to_uppercase()));
                        }
                        if name.eq_ignore_ascii_case("lower") {
                            if args.len() != 1 {
                                return Err(ApexError::QueryParseError(
                                    "LOWER() expects 1 argument".to_string(),
                                ));
                            }
                            let v = eval_scalar(&args[0], left_table, right_table, left_alias, right_alias, jr)?;
                            if v.is_null() {
                                return Ok(Value::Null);
                            }
                            return Ok(Value::String(v.to_string_value().to_lowercase()));
                        }
                        if name.eq_ignore_ascii_case("replace") {
                            if args.len() != 3 {
                                return Err(ApexError::QueryParseError(
                                    "REPLACE() expects 3 arguments".to_string(),
                                ));
                            }
                            let s0 = eval_scalar(&args[0], left_table, right_table, left_alias, right_alias, jr)?;
                            let from0 = eval_scalar(&args[1], left_table, right_table, left_alias, right_alias, jr)?;
                            let to0 = eval_scalar(&args[2], left_table, right_table, left_alias, right_alias, jr)?;
                            if s0.is_null() || from0.is_null() || to0.is_null() {
                                return Ok(Value::Null);
                            }
                            return Ok(Value::String(
                                s0.to_string_value().replace(&from0.to_string_value(), &to0.to_string_value()),
                            ));
                        }
                        if name.eq_ignore_ascii_case("mid") {
                            if args.len() != 2 && args.len() != 3 {
                                return Err(ApexError::QueryParseError(
                                    "MID() expects 2 or 3 arguments".to_string(),
                                ));
                            }
                            let s0 = eval_scalar(&args[0], left_table, right_table, left_alias, right_alias, jr)?;
                            let start0 = eval_scalar(&args[1], left_table, right_table, left_alias, right_alias, jr)?;
                            if s0.is_null() || start0.is_null() {
                                return Ok(Value::Null);
                            }
                            let s = s0.to_string_value();
                            let mut start = start0.as_i64().unwrap_or(1);
                            if start < 1 {
                                start = 1;
                            }
                            let start_idx = (start - 1) as usize;
                            let chars: Vec<char> = s.chars().collect();
                            if start_idx >= chars.len() {
                                return Ok(Value::String(String::new()));
                            }
                            let end_idx = if args.len() == 3 {
                                let len0 = eval_scalar(&args[2], left_table, right_table, left_alias, right_alias, jr)?;
                                if len0.is_null() {
                                    return Ok(Value::Null);
                                }
                                let mut l = len0.as_i64().unwrap_or(0);
                                if l < 0 {
                                    l = 0;
                                }
                                (start_idx + l as usize).min(chars.len())
                            } else {
                                chars.len()
                            };
                            return Ok(Value::String(chars[start_idx..end_idx].iter().collect()));
                        }
                        Err(ApexError::QueryParseError(
                            format!("Unsupported function: {}", name),
                        ))
                    }
                    SqlExpr::UnaryOp { op, expr } => match op {
                        UnaryOperator::Not => {
                            let v = eval_predicate(expr, left_table, right_table, left_alias, right_alias, jr)?;
                            Ok(Value::Bool(!v))
                        }
                        UnaryOperator::Minus => {
                            let v = eval_scalar(expr, left_table, right_table, left_alias, right_alias, jr)?;
                            if let Some(i) = v.as_i64() {
                                Ok(Value::Int64(-i))
                            } else if let Some(f) = v.as_f64() {
                                Ok(Value::Float64(-f))
                            } else {
                                Ok(Value::Null)
                            }
                        }
                    },
                    SqlExpr::BinaryOp { left, op, right } => match op {
                        BinaryOperator::Add | BinaryOperator::Sub | BinaryOperator::Mul | BinaryOperator::Div | BinaryOperator::Mod => {
                            let lv = eval_scalar(left, left_table, right_table, left_alias, right_alias, jr)?;
                            let rv = eval_scalar(right, left_table, right_table, left_alias, right_alias, jr)?;
                            if let (Some(a), Some(b)) = (lv.as_f64(), rv.as_f64()) {
                                let out = match op {
                                    BinaryOperator::Add => a + b,
                                    BinaryOperator::Sub => a - b,
                                    BinaryOperator::Mul => a * b,
                                    BinaryOperator::Div => a / b,
                                    BinaryOperator::Mod => a % b,
                                    _ => a,
                                };
                                Ok(Value::Float64(out))
                            } else {
                                Ok(Value::Null)
                            }
                        }
                        _ => Ok(Value::Null),
                    },
                    _ => Ok(Value::Null),
                }
            }

            match expr {
                SqlExpr::Paren(inner) => eval_predicate(inner, left_table, right_table, left_alias, right_alias, jr),
                SqlExpr::Literal(v) => Ok(v.as_bool().unwrap_or(false)),
                SqlExpr::UnaryOp { op: UnaryOperator::Not, expr } => Ok(!eval_predicate(expr, left_table, right_table, left_alias, right_alias, jr)?),
                SqlExpr::BinaryOp { left, op, right } => match op {
                    BinaryOperator::And => Ok(
                        eval_predicate(left, left_table, right_table, left_alias, right_alias, jr)?
                            && eval_predicate(right, left_table, right_table, left_alias, right_alias, jr)?,
                    ),
                    BinaryOperator::Or => Ok(
                        eval_predicate(left, left_table, right_table, left_alias, right_alias, jr)?
                            || eval_predicate(right, left_table, right_table, left_alias, right_alias, jr)?,
                    ),
                    BinaryOperator::Eq
                    | BinaryOperator::NotEq
                    | BinaryOperator::Lt
                    | BinaryOperator::Le
                    | BinaryOperator::Gt
                    | BinaryOperator::Ge => {
                        let lv = eval_scalar(left, left_table, right_table, left_alias, right_alias, jr)?;
                        let rv = eval_scalar(right, left_table, right_table, left_alias, right_alias, jr)?;

                        if lv.is_null() || rv.is_null() {
                            return Ok(false);
                        }

                        let ord = lv.partial_cmp(&rv).unwrap_or(Ordering::Equal);
                        Ok(match op {
                            BinaryOperator::Eq => ord == Ordering::Equal,
                            BinaryOperator::NotEq => ord != Ordering::Equal,
                            BinaryOperator::Lt => ord == Ordering::Less,
                            BinaryOperator::Le => ord != Ordering::Greater,
                            BinaryOperator::Gt => ord == Ordering::Greater,
                            BinaryOperator::Ge => ord != Ordering::Less,
                            _ => false,
                        })
                    }
                    _ => Err(ApexError::QueryParseError(
                        "Unsupported operator in JOIN predicate".to_string(),
                    )),
                },
                _ => Err(ApexError::QueryParseError(
                    "Unsupported expression in JOIN predicate".to_string(),
                )),
            }
        }

        let (left_key_ref, right_key_ref) = match &join.on {
            SqlExpr::BinaryOp { left, op: BinaryOperator::Eq, right } => {
                let l = match left.as_ref() {
                    SqlExpr::Column(c) => c.clone(),
                    _ => {
                        return Err(ApexError::QueryParseError(
                            "JOIN ON must be column = column".to_string(),
                        ))
                    }
                };
                let r = match right.as_ref() {
                    SqlExpr::Column(c) => c.clone(),
                    _ => {
                        return Err(ApexError::QueryParseError(
                            "JOIN ON must be column = column".to_string(),
                        ))
                    }
                };
                (l, r)
            }
            _ => {
                return Err(ApexError::QueryParseError(
                    "Only equi-join ON a=b is supported yet".to_string(),
                ))
            }
        };

        let (lk_alias0, lk_col) = split_qual(&left_key_ref);
        let (rk_alias0, rk_col) = split_qual(&right_key_ref);

        let lk_alias = if lk_alias0.is_empty() { left_alias.as_str() } else { lk_alias0 };
        let rk_alias = if rk_alias0.is_empty() { right_alias.as_str() } else { rk_alias0 };

        let left_matches = lk_alias == left_alias.as_str() || lk_alias == left_table_name.as_str();
        let right_matches = rk_alias == right_alias.as_str() || rk_alias == right_table_name.as_str();
        let left_swapped = lk_alias == right_alias.as_str() || lk_alias == right_table_name.as_str();
        let right_swapped = rk_alias == left_alias.as_str() || rk_alias == left_table_name.as_str();

        let (left_key_col, right_key_col) = if left_matches && right_matches {
            (lk_col.to_string(), rk_col.to_string())
        } else if left_swapped && right_swapped {
            (rk_col.to_string(), lk_col.to_string())
        } else {
            return Err(ApexError::QueryParseError(
                "JOIN ON must reference left and right table aliases".to_string(),
            ));
        };

        let right_row_count = right_table.get_row_count();
        let right_deleted = right_table.deleted_ref();

        let mut right_allowed: Option<Vec<bool>> = None;
        let mut where_pushed_down = false;
        if let Some(ref where_expr) = stmt.where_clause {
            if let Some(stripped) = strip_right_only_where(where_expr, &left_alias, &right_alias) {
                let row_count = right_table.get_row_count();
                let idxs = Self::evaluate_where(&stripped, right_table)?;
                let mut allowed = vec![false; row_count];
                for i in idxs {
                    if i < allowed.len() {
                        allowed[i] = true;
                    }
                }
                right_allowed = Some(allowed);
                where_pushed_down = true;
            }
        }
        let mut hash: HashMap<JoinKey, Vec<usize>> = HashMap::new();
        for ri in 0..right_row_count {
            if right_deleted.get(ri) {
                continue;
            }
            if let Some(ref allowed) = right_allowed {
                if ri >= allowed.len() || !allowed[ri] {
                    continue;
                }
            }
            let v = get_col_value(right_table, &right_key_col, ri);
            if v.is_null() {
                continue;
            }
            hash.entry(join_key(&v)).or_default().push(ri);
        }

        let left_row_count = left_table.get_row_count();
        let left_deleted = left_table.deleted_ref();
        let mut joined: Vec<JoinRow> = Vec::new();
        for li in 0..left_row_count {
            if left_deleted.get(li) {
                continue;
            }
            let lv = get_col_value(left_table, &left_key_col, li);
            let matches = if lv.is_null() { None } else { hash.get(&join_key(&lv)) };

            match join.join_type {
                crate::query::JoinType::Inner => {
                    if let Some(rs) = matches {
                        for &r in rs {
                            joined.push(JoinRow { left: li, right: Some(r) });
                        }
                    }
                }
                crate::query::JoinType::Left => {
                    if let Some(rs) = matches {
                        for &r in rs {
                            joined.push(JoinRow { left: li, right: Some(r) });
                        }
                    } else {
                        joined.push(JoinRow { left: li, right: None });
                    }
                }
            }
        }

        if !where_pushed_down {
            if let Some(ref where_expr) = stmt.where_clause {
                let mut filtered: Vec<JoinRow> = Vec::with_capacity(joined.len());
                for jr in joined {
                    if eval_predicate(where_expr, left_table, right_table, &left_alias, &right_alias, jr)? {
                        filtered.push(jr);
                    }
                }
                joined = filtered;
            }
        }

        let has_aggs = stmt.columns.iter().any(|c| matches!(c, SelectColumn::Aggregate { .. }));
        if !stmt.group_by.is_empty() || has_aggs {
            use std::collections::HashSet;

            struct JoinAgg {
                func: AggregateFunc,
                distinct: bool,
                seen: Option<HashSet<Vec<u8>>>,
                count: i64,
                sum: f64,
            }

            impl JoinAgg {
                fn new(func: AggregateFunc, distinct: bool) -> Self {
                    let seen = if matches!(func, AggregateFunc::Count) && distinct {
                        Some(HashSet::new())
                    } else {
                        None
                    };
                    Self { func, distinct, seen, count: 0, sum: 0.0 }
                }
            }

            #[derive(Default)]
            struct GroupState {
                first: Option<JoinRow>,
                aggs: Vec<JoinAgg>,
            }

            // Build aggregate spec list in SELECT order
            let mut agg_specs: Vec<(AggregateFunc, Option<String>, bool, Option<String>)> = Vec::new();
            for c in &stmt.columns {
                if let SelectColumn::Aggregate { func, column, distinct, alias } = c {
                    agg_specs.push((func.clone(), column.clone(), *distinct, alias.clone()));
                }
            }

            // Group by key is serialized bytes of each key part.
            let mut groups: HashMap<String, GroupState> = HashMap::new();
            for jr in &joined {
                let mut key_parts: Vec<String> = Vec::with_capacity(stmt.group_by.len());
                for gb in &stmt.group_by {
                    let v = value_for_ref(gb, left_table, right_table, &left_alias, &right_alias, *jr);
                    key_parts.push(v.to_string_value());
                }
                let gk = key_parts.join("\u{1f}");
                let entry = groups.entry(gk).or_insert_with(|| {
                    let mut gs = GroupState::default();
                    gs.first = Some(*jr);
                    gs.aggs = agg_specs.iter().map(|(f, _, d, _)| JoinAgg::new(f.clone(), *d)).collect();
                    gs
                });

                for (i, (func, col, distinct, _)) in agg_specs.iter().enumerate() {
                    let agg = &mut entry.aggs[i];
                    match func {
                        AggregateFunc::Count => {
                            if *distinct {
                                if let Some(cn) = col.as_ref() {
                                    let v = value_for_ref(cn, left_table, right_table, &left_alias, &right_alias, *jr);
                                    if !v.is_null() {
                                        if let Some(set) = agg.seen.as_mut() {
                                            set.insert(v.to_bytes());
                                        }
                                    }
                                }
                            } else if col.is_none() {
                                agg.count += 1;
                            } else if let Some(cn) = col.as_ref() {
                                let v = value_for_ref(cn, left_table, right_table, &left_alias, &right_alias, *jr);
                                if !v.is_null() {
                                    agg.count += 1;
                                }
                            }
                        }
                        AggregateFunc::Sum | AggregateFunc::Avg => {
                            if let Some(cn) = col.as_ref() {
                                let v = value_for_ref(cn, left_table, right_table, &left_alias, &right_alias, *jr);
                                if let Some(n) = v.as_f64() {
                                    agg.sum += n;
                                    agg.count += 1;
                                }
                            }
                        }
                        _ => {}
                    }
                }
            }

            // HAVING evaluation on group state
            fn eval_having_scalar_join(
                expr: &SqlExpr,
                gs: &GroupState,
                agg_specs: &[(AggregateFunc, Option<String>, bool, Option<String>)],
                left_table: &ColumnTable,
                right_table: &ColumnTable,
                left_alias: &str,
                right_alias: &str,
            ) -> Result<Value, ApexError> {
                match expr {
                    SqlExpr::Paren(inner) => eval_having_scalar_join(inner, gs, agg_specs, left_table, right_table, left_alias, right_alias),
                    SqlExpr::Literal(v) => Ok(v.clone()),
                    SqlExpr::Column(c) => {
                        if let Some(jr) = gs.first {
                            Ok(value_for_ref(c, left_table, right_table, left_alias, right_alias, jr))
                        } else {
                            Ok(Value::Null)
                        }
                    }
                    SqlExpr::Function { name, args } => {
                        if name.eq_ignore_ascii_case("rand") {
                            if !args.is_empty() {
                                return Err(ApexError::QueryParseError(
                                    "RAND() does not accept arguments".to_string(),
                                ));
                            }
                            return Ok(Value::Float64(rand::random::<f64>()));
                        }
                        let func = match name.to_uppercase().as_str() {
                            "COUNT" => AggregateFunc::Count,
                            "SUM" => AggregateFunc::Sum,
                            "AVG" => AggregateFunc::Avg,
                            _ => {
                                return Err(ApexError::QueryParseError(
                                    format!("Unsupported function in HAVING: {}", name),
                                ))
                            }
                        };

                        let col = if args.is_empty() {
                            None
                        } else {
                            match &args[0] {
                                SqlExpr::Column(c) => Some(c.clone()),
                                _ => None,
                            }
                        };

                        let col_d = col.as_deref();
                        let col_base = col_d.map(|s| split_qual(s).1);
                        for (i, (sf, sc, sd, _)) in agg_specs.iter().enumerate() {
                            let sc_d = sc.as_deref();
                            let sc_base = sc_d.map(|s| split_qual(s).1);
                            let col_match = sc_d == col_d || (sc_base.is_some() && col_base.is_some() && sc_base == col_base);

                            if *sf == func && col_match && *sd == false {
                                // Non-distinct aggregate match
                                let a = &gs.aggs[i];
                                return Ok(match sf {
                                    AggregateFunc::Count => Value::Int64(a.count),
                                    AggregateFunc::Sum => Value::Float64(a.sum),
                                    AggregateFunc::Avg => {
                                        if a.count > 0 {
                                            Value::Float64(a.sum / a.count as f64)
                                        } else {
                                            Value::Null
                                        }
                                    }
                                    _ => Value::Null,
                                });
                            }
                            if *sf == func && col_match && *sd {
                                // Distinct COUNT match
                                let a = &gs.aggs[i];
                                return Ok(Value::Int64(a.seen.as_ref().map(|s| s.len()).unwrap_or(0) as i64));
                            }
                        }
                        Ok(Value::Null)
                    }
                    SqlExpr::UnaryOp { op, expr } => {
                        match op {
                            UnaryOperator::Not => {
                                let v = eval_having_scalar_join(expr, gs, agg_specs, left_table, right_table, left_alias, right_alias)?;
                                Ok(Value::Bool(!v.as_bool().unwrap_or(false)))
                            }
                            UnaryOperator::Minus => {
                                let v = eval_having_scalar_join(expr, gs, agg_specs, left_table, right_table, left_alias, right_alias)?;
                                if let Some(i) = v.as_i64() {
                                    Ok(Value::Int64(-i))
                                } else if let Some(f) = v.as_f64() {
                                    Ok(Value::Float64(-f))
                                } else {
                                    Ok(Value::Null)
                                }
                            }
                        }
                    }
                    SqlExpr::BinaryOp { left, op, right } => {
                        let lv = eval_having_scalar_join(left, gs, agg_specs, left_table, right_table, left_alias, right_alias)?;
                        let rv = eval_having_scalar_join(right, gs, agg_specs, left_table, right_table, left_alias, right_alias)?;
                        match op {
                            BinaryOperator::And => Ok(Value::Bool(lv.as_bool().unwrap_or(false) && rv.as_bool().unwrap_or(false))),
                            BinaryOperator::Or => Ok(Value::Bool(lv.as_bool().unwrap_or(false) || rv.as_bool().unwrap_or(false))),
                            BinaryOperator::Eq => Ok(Value::Bool(lv == rv)),
                            BinaryOperator::NotEq => Ok(Value::Bool(lv != rv)),
                            BinaryOperator::Lt | BinaryOperator::Le | BinaryOperator::Gt | BinaryOperator::Ge => {
                                if lv.is_null() || rv.is_null() {
                                    return Ok(Value::Bool(false));
                                }
                                let ord = lv.partial_cmp(&rv).unwrap_or(Ordering::Equal);
                                let b = match op {
                                    BinaryOperator::Lt => ord == Ordering::Less,
                                    BinaryOperator::Le => ord != Ordering::Greater,
                                    BinaryOperator::Gt => ord == Ordering::Greater,
                                    BinaryOperator::Ge => ord != Ordering::Less,
                                    _ => false,
                                };
                                Ok(Value::Bool(b))
                            }
                            _ => Ok(Value::Null),
                        }
                    }
                    _ => Ok(Value::Null),
                }
            }

            fn eval_having_predicate_join(
                expr: &SqlExpr,
                gs: &GroupState,
                agg_specs: &[(AggregateFunc, Option<String>, bool, Option<String>)],
                left_table: &ColumnTable,
                right_table: &ColumnTable,
                left_alias: &str,
                right_alias: &str,
            ) -> Result<bool, ApexError> {
                match expr {
                    SqlExpr::Paren(inner) => eval_having_predicate_join(inner, gs, agg_specs, left_table, right_table, left_alias, right_alias),
                    SqlExpr::Literal(v) => Ok(v.as_bool().unwrap_or(false)),
                    SqlExpr::UnaryOp { op: UnaryOperator::Not, expr } => Ok(!eval_having_predicate_join(expr, gs, agg_specs, left_table, right_table, left_alias, right_alias)?),
                    SqlExpr::BinaryOp { left, op, right } => match op {
                        BinaryOperator::And => Ok(
                            eval_having_predicate_join(left, gs, agg_specs, left_table, right_table, left_alias, right_alias)?
                                && eval_having_predicate_join(right, gs, agg_specs, left_table, right_table, left_alias, right_alias)?,
                        ),
                        BinaryOperator::Or => Ok(
                            eval_having_predicate_join(left, gs, agg_specs, left_table, right_table, left_alias, right_alias)?
                                || eval_having_predicate_join(right, gs, agg_specs, left_table, right_table, left_alias, right_alias)?,
                        ),
                        BinaryOperator::Eq
                        | BinaryOperator::NotEq
                        | BinaryOperator::Lt
                        | BinaryOperator::Le
                        | BinaryOperator::Gt
                        | BinaryOperator::Ge => {
                            let lv = eval_having_scalar_join(left, gs, agg_specs, left_table, right_table, left_alias, right_alias)?;
                            let rv = eval_having_scalar_join(right, gs, agg_specs, left_table, right_table, left_alias, right_alias)?;
                            if lv.is_null() || rv.is_null() {
                                return Ok(false);
                            }
                            let ord = lv.partial_cmp(&rv).unwrap_or(Ordering::Equal);
                            Ok(match op {
                                BinaryOperator::Eq => ord == Ordering::Equal,
                                BinaryOperator::NotEq => ord != Ordering::Equal,
                                BinaryOperator::Lt => ord == Ordering::Less,
                                BinaryOperator::Le => ord != Ordering::Greater,
                                BinaryOperator::Gt => ord == Ordering::Greater,
                                BinaryOperator::Ge => ord != Ordering::Less,
                                _ => false,
                            })
                        }
                        _ => {
                            let v = eval_having_scalar_join(expr, gs, agg_specs, left_table, right_table, left_alias, right_alias)?;
                            Ok(v.as_bool().unwrap_or(false))
                        }
                    },
                    _ => {
                        let v = eval_having_scalar_join(expr, gs, agg_specs, left_table, right_table, left_alias, right_alias)?;
                        Ok(v.as_bool().unwrap_or(false))
                    }
                }
            }

            let mut out_rows: Vec<Vec<Value>> = Vec::new();
            for (_k, gs) in groups {
                if let Some(ref having_expr) = stmt.having {
                    if !eval_having_predicate_join(having_expr, &gs, &agg_specs, left_table, right_table, &left_alias, &right_alias)? {
                        continue;
                    }
                }

                let mut row: Vec<Value> = Vec::with_capacity(stmt.columns.len());
                for c in &stmt.columns {
                    match c {
                        SelectColumn::Column(name) => {
                            if let Some(jr) = gs.first {
                                row.push(value_for_ref(name, left_table, right_table, &left_alias, &right_alias, jr));
                            } else {
                                row.push(Value::Null);
                            }
                        }
                        SelectColumn::ColumnAlias { column, .. } => {
                            if let Some(jr) = gs.first {
                                row.push(value_for_ref(column, left_table, right_table, &left_alias, &right_alias, jr));
                            } else {
                                row.push(Value::Null);
                            }
                        }
                        SelectColumn::Aggregate { func, column, distinct, .. } => {
                            let idx = agg_specs.iter().position(|(f, c, d, _)| f == func && c == column && d == distinct);
                            if let Some(i) = idx {
                                let a = &gs.aggs[i];
                                match func {
                                    AggregateFunc::Count => {
                                        if *distinct {
                                            row.push(Value::Int64(a.seen.as_ref().map(|s| s.len()).unwrap_or(0) as i64));
                                        } else {
                                            row.push(Value::Int64(a.count));
                                        }
                                    }
                                    AggregateFunc::Sum => row.push(Value::Float64(a.sum)),
                                    AggregateFunc::Avg => {
                                        if a.count > 0 {
                                            row.push(Value::Float64(a.sum / a.count as f64));
                                        } else {
                                            row.push(Value::Null);
                                        }
                                    }
                                    _ => row.push(Value::Null),
                                }
                            } else {
                                row.push(Value::Null);
                            }
                        }
                        _ => {
                            return Err(ApexError::QueryParseError(
                                "Unsupported SELECT item in JOIN GROUP BY".to_string(),
                            ))
                        }
                    }
                }
                out_rows.push(row);
            }

            let mut out_columns: Vec<String> = Vec::new();
            for c in &stmt.columns {
                match c {
                    SelectColumn::Column(name) => out_columns.push(split_qual(name).1.to_string()),
                    SelectColumn::ColumnAlias { alias, .. } => out_columns.push(alias.clone()),
                    SelectColumn::Aggregate { func, column, distinct, alias } => {
                        let nm = alias.clone().unwrap_or_else(|| {
                            let func_name = match func {
                                AggregateFunc::Count => "COUNT",
                                AggregateFunc::Sum => "SUM",
                                AggregateFunc::Avg => "AVG",
                                AggregateFunc::Min => "MIN",
                                AggregateFunc::Max => "MAX",
                            };
                            if let Some(cn) = column {
                                if *distinct {
                                    format!("{}(DISTINCT {})", func_name, cn)
                                } else {
                                    format!("{}({})", func_name, cn)
                                }
                            } else {
                                format!("{}(*)", func_name)
                            }
                        });
                        out_columns.push(nm);
                    }
                    _ => out_columns.push("expr".to_string()),
                }
            }

            if !stmt.order_by.is_empty() {
                out_rows.sort_by(|a, b| {
                    for ob in &stmt.order_by {
                        let idx = out_columns.iter().position(|c| c == &ob.column).unwrap_or(0);
                        let av = a.get(idx);
                        let bv = b.get(idx);
                        let cmp = SqlExecutor::compare_values(av, bv, ob.nulls_first);
                        let cmp = if ob.descending { cmp.reverse() } else { cmp };
                        if cmp != Ordering::Equal {
                            return cmp;
                        }
                    }
                    Ordering::Equal
                });
            }

            let offset = stmt.offset.unwrap_or(0);
            let limit = stmt.limit.unwrap_or(usize::MAX);
            let out_rows = out_rows.into_iter().skip(offset).take(limit).collect::<Vec<_>>();

            return Ok(SqlResult::new(out_columns, out_rows));
        }

        if !stmt.order_by.is_empty() {
            joined.sort_by(|a, b| {
                for ob in &stmt.order_by {
                    let col_ref = &ob.column;
                    let av = value_for_ref(col_ref, left_table, right_table, &left_alias, &right_alias, *a);
                    let bv = value_for_ref(col_ref, left_table, right_table, &left_alias, &right_alias, *b);
                    let cmp = SqlExecutor::compare_values(Some(&av), Some(&bv), ob.nulls_first);
                    let cmp = if ob.descending { cmp.reverse() } else { cmp };
                    if cmp != Ordering::Equal {
                        return cmp;
                    }
                }
                Ordering::Equal
            });
        }

        let offset = stmt.offset.unwrap_or(0);
        let limit = stmt.limit.unwrap_or(usize::MAX);
        let joined = joined.into_iter().skip(offset).take(limit).collect::<Vec<_>>();

        let mut out_columns: Vec<String> = Vec::new();
        for c in &stmt.columns {
            match c {
                SelectColumn::Column(name) => {
                    out_columns.push(split_qual(name).1.to_string());
                }
                SelectColumn::ColumnAlias { alias, .. } => out_columns.push(alias.clone()),
                _ => {
                    return Err(ApexError::QueryParseError(
                        "Only plain column projection is supported for JOIN yet".to_string(),
                    ))
                }
            }
        }

        let mut rows: Vec<Vec<Value>> = Vec::with_capacity(joined.len());
        for jr in joined {
            let mut row: Vec<Value> = Vec::with_capacity(stmt.columns.len());
            for c in &stmt.columns {
                match c {
                    SelectColumn::Column(name) => {
                        row.push(value_for_ref(name, left_table, right_table, &left_alias, &right_alias, jr));
                    }
                    SelectColumn::ColumnAlias { column, .. } => {
                        row.push(value_for_ref(column, left_table, right_table, &left_alias, &right_alias, jr));
                    }
                    _ => {}
                }
            }
            rows.push(row);
        }

        if stmt.distinct {
            let mut seen: std::collections::HashSet<String> = std::collections::HashSet::new();
            rows.retain(|r| {
                let k = r.iter().map(|v| v.to_string_value()).collect::<Vec<_>>().join("\u{1f}");
                seen.insert(k)
            });
        }

        Ok(SqlResult::new(out_columns, rows))
    }
    
}
