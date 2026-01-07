use crate::query::sql_expr_to_filter;
use crate::query::sql_parser::{BinaryOperator, FromItem, JoinType, SelectColumn, SelectStatement, SqlExpr, UnaryOperator};
use crate::query::{SqlExecutor, SqlResult};
use crate::table::ColumnTable;
use crate::ApexError;
use std::cmp::Ordering;
use std::collections::HashMap;

#[derive(Clone, Copy)]
pub(crate) struct JoinRow {
    left: usize,
    right: Option<usize>,
}

pub(crate) struct JoinContext {
    pub(crate) left_table_name: String,
    pub(crate) right_table_name: String,
    pub(crate) left_alias: String,
    pub(crate) right_alias: String,
}

#[inline]
fn split_qual(col: &str) -> (&str, &str) {
    if let Some((a, b)) = col.split_once('.') {
        (a, b)
    } else {
        ("", col)
    }
}

#[inline]
fn get_col_value(table: &ColumnTable, col: &str, row_idx: usize) -> crate::data::Value {
    use crate::data::Value;
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

#[inline]
fn value_for_ref(
    col_ref: &str,
    left_table: &ColumnTable,
    right_table: &ColumnTable,
    left_alias: &str,
    right_alias: &str,
    jr: JoinRow,
) -> crate::data::Value {
    use crate::data::Value;
    let (a, c) = split_qual(col_ref);
    if a.is_empty() {
        // Unqualified: prefer side where column exists; if both, default left.
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
    ) -> Result<crate::data::Value, ApexError> {
        use crate::data::Value;
        match expr {
            SqlExpr::Paren(inner) => eval_scalar(inner, left_table, right_table, left_alias, right_alias, jr),
            SqlExpr::Literal(v) => Ok(v.clone()),
            SqlExpr::Column(c) => Ok(value_for_ref(c, left_table, right_table, left_alias, right_alias, jr)),
            SqlExpr::UnaryOp { op, expr } => match op {
                UnaryOperator::Not => Ok(Value::Bool(!eval_predicate(
                    expr,
                    left_table,
                    right_table,
                    left_alias,
                    right_alias,
                    jr,
                )?)),
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
        SqlExpr::UnaryOp { op: UnaryOperator::Not, expr } => {
            Ok(!eval_predicate(expr, left_table, right_table, left_alias, right_alias, jr)?)
        }
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

fn strip_right_only_where(expr: &SqlExpr, left_alias: &str, right_alias: &str) -> Option<SqlExpr> {
    match expr {
        SqlExpr::Paren(inner) => strip_right_only_where(inner, left_alias, right_alias)
            .map(|e| SqlExpr::Paren(Box::new(e))),
        SqlExpr::BinaryOp { left, op, right } => {
            if *op == BinaryOperator::And {
                let l = strip_right_only_where(left, left_alias, right_alias)?;
                let r = strip_right_only_where(right, left_alias, right_alias)?;
                return Some(SqlExpr::BinaryOp {
                    left: Box::new(l),
                    op: op.clone(),
                    right: Box::new(r),
                });
            }

            let col = match left.as_ref() {
                SqlExpr::Column(c) => c,
                _ => return None,
            };
            let (a, c) = if let Some((aa, cc)) = col.split_once('.') {
                (aa, cc)
            } else {
                ("", col.as_str())
            };
            if a.is_empty() {
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

fn evaluate_where(expr: &SqlExpr, table: &ColumnTable) -> Result<Vec<usize>, ApexError> {
    let filter = sql_expr_to_filter(expr)?;
    let schema = table.schema_ref();
    let columns = table.columns_ref();
    let row_count = table.get_row_count();
    let deleted = table.deleted_ref();
    Ok(filter.filter_columns(schema, columns, row_count, deleted))
}

pub(crate) fn build_join_rows(
    stmt: &SelectStatement,
    tables: &mut HashMap<String, ColumnTable>,
) -> Result<(JoinContext, Vec<JoinRow>), ApexError> {
    if stmt.joins.is_empty() {
        return Err(ApexError::QueryParseError("not a join stmt".to_string()));
    }
    if stmt.joins.len() != 1 {
        return Err(ApexError::QueryParseError(
            "Only single JOIN is supported yet".to_string(),
        ));
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

    if let Some(t) = tables.get_mut(&left_table_name) {
        if t.has_pending_writes() {
            t.flush_write_buffer();
        }
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

    if let Some(t) = tables.get_mut(&right_table_name) {
        if t.has_pending_writes() {
            t.flush_write_buffer();
        }
    }

    let left_table = tables
        .get(&left_table_name)
        .ok_or_else(|| ApexError::QueryParseError(format!("Table '{}' not found.", left_table_name)))?;
    let right_table = tables
        .get(&right_table_name)
        .ok_or_else(|| ApexError::QueryParseError(format!("Table '{}' not found.", right_table_name)))?;

    let (left_key_ref, right_key_ref) = match &join.on {
        SqlExpr::BinaryOp {
            left,
            op: BinaryOperator::Eq,
            right,
        } => {
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

    let lk_alias = if lk_alias0.is_empty() {
        left_alias.as_str()
    } else {
        lk_alias0
    };
    let rk_alias = if rk_alias0.is_empty() {
        right_alias.as_str()
    } else {
        rk_alias0
    };

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

    let mut right_allowed: Option<Vec<bool>> = None;
    let mut where_pushed_down = false;
    if let Some(ref where_expr) = stmt.where_clause {
        if let Some(stripped) = strip_right_only_where(where_expr, &left_alias, &right_alias) {
            let row_count = right_table.get_row_count();
            let idxs = evaluate_where(&stripped, right_table)?;
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

    let right_row_count = right_table.get_row_count();
    let right_deleted = right_table.deleted_ref();
    let mut hash: HashMap<String, Vec<usize>> = HashMap::new();
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
        hash.entry(v.to_string_value()).or_default().push(ri);
    }

    let left_row_count = left_table.get_row_count();
    let left_deleted = left_table.deleted_ref();
    let mut joined: Vec<JoinRow> = Vec::new();
    for li in 0..left_row_count {
        if left_deleted.get(li) {
            continue;
        }
        let lv = get_col_value(left_table, &left_key_col, li);
        let matches = if lv.is_null() {
            None
        } else {
            hash.get(&lv.to_string_value()).cloned()
        };

        match join.join_type {
            JoinType::Inner => {
                if let Some(rs) = matches {
                    for r in rs {
                        joined.push(JoinRow {
                            left: li,
                            right: Some(r),
                        });
                    }
                }
            }
            JoinType::Left => {
                if let Some(rs) = matches {
                    for r in rs {
                        joined.push(JoinRow {
                            left: li,
                            right: Some(r),
                        });
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

    Ok((
        JoinContext {
            left_table_name,
            right_table_name,
            left_alias,
            right_alias,
        },
        joined,
    ))
}

pub(crate) fn sort_join_rows(
    ctx: &JoinContext,
    joined: &mut [JoinRow],
    order_by: &[crate::query::sql_parser::OrderByClause],
    tables: &HashMap<String, ColumnTable>,
) {
    if order_by.is_empty() || joined.len() <= 1 {
        return;
    }
    let left_table = match tables.get(&ctx.left_table_name) {
        Some(t) => t,
        None => return,
    };
    let right_table = match tables.get(&ctx.right_table_name) {
        Some(t) => t,
        None => return,
    };

    joined.sort_by(|a, b| {
        for ob in order_by {
            let col_ref = &ob.column;
            let av = value_for_ref(
                col_ref,
                left_table,
                right_table,
                &ctx.left_alias,
                &ctx.right_alias,
                *a,
            );
            let bv = value_for_ref(
                col_ref,
                left_table,
                right_table,
                &ctx.left_alias,
                &ctx.right_alias,
                *b,
            );
            let cmp = SqlExecutor::compare_values(Some(&av), Some(&bv), ob.nulls_first);
            let cmp = if ob.descending { cmp.reverse() } else { cmp };
            if cmp != Ordering::Equal {
                return cmp;
            }
        }
        Ordering::Equal
    });
}

pub(crate) fn project_join_rows_plain(
    stmt: &SelectStatement,
    ctx: &JoinContext,
    joined: &[JoinRow],
    tables: &HashMap<String, ColumnTable>,
) -> Result<SqlResult, ApexError> {
    let left_table = tables
        .get(&ctx.left_table_name)
        .ok_or_else(|| ApexError::QueryParseError(format!("Table '{}' not found.", ctx.left_table_name)))?;
    let right_table = tables
        .get(&ctx.right_table_name)
        .ok_or_else(|| ApexError::QueryParseError(format!("Table '{}' not found.", ctx.right_table_name)))?;

    let mut out_columns: Vec<String> = Vec::new();
    let mut col_refs: Vec<String> = Vec::new();
    for c in &stmt.columns {
        match c {
            SelectColumn::Column(name) => {
                out_columns.push(split_qual(name).1.to_string());
                col_refs.push(name.clone());
            }
            SelectColumn::ColumnAlias { column, alias } => {
                out_columns.push(alias.clone());
                col_refs.push(column.clone());
            }
            _ => {
                return Err(ApexError::QueryParseError(
                    "Only plain column projection is supported for JOIN yet".to_string(),
                ))
            }
        }
    }

    let mut rows: Vec<Vec<crate::data::Value>> = Vec::with_capacity(joined.len());
    for jr in joined {
        let mut row: Vec<crate::data::Value> = Vec::with_capacity(col_refs.len());
        for cref in &col_refs {
            row.push(value_for_ref(
                cref,
                left_table,
                right_table,
                &ctx.left_alias,
                &ctx.right_alias,
                *jr,
            ));
        }
        rows.push(row);
    }

    if stmt.distinct {
        let mut seen: std::collections::HashSet<String> = std::collections::HashSet::new();
        rows.retain(|r| {
            let k = r
                .iter()
                .map(|v| v.to_string_value())
                .collect::<Vec<_>>()
                .join("\u{1f}");
            seen.insert(k)
        });
    }
    Ok(SqlResult::new(out_columns, rows))
}

pub(crate) fn aggregate_join_rows(
    stmt: &SelectStatement,
    ctx: &JoinContext,
    joined: &[JoinRow],
    tables: &HashMap<String, ColumnTable>,
) -> Result<SqlResult, ApexError> {
    use crate::data::Value;
    use crate::query::sql_parser::AggregateFunc;
    use std::collections::HashSet;

    let left_table = tables
        .get(&ctx.left_table_name)
        .ok_or_else(|| ApexError::QueryParseError(format!("Table '{}' not found.", ctx.left_table_name)))?;
    let right_table = tables
        .get(&ctx.right_table_name)
        .ok_or_else(|| ApexError::QueryParseError(format!("Table '{}' not found.", ctx.right_table_name)))?;

    enum AggState {
        CountDistinct(HashSet<Vec<u8>>),
        Count(i64),
        Sum(f64),
        Min(Option<Value>),
        Max(Option<Value>),
    }

    struct AggSpec {
        func: AggregateFunc,
        column: Option<String>,
        distinct: bool,
        out_name: String,
    }

    fn out_name_for_agg(
        func: &AggregateFunc,
        column: &Option<String>,
        distinct: bool,
        alias: &Option<String>,
    ) -> String {
        if let Some(a) = alias {
            return a.clone();
        }
        let func_name = match func {
            AggregateFunc::Count => "COUNT",
            AggregateFunc::Sum => "SUM",
            AggregateFunc::Avg => "AVG",
            AggregateFunc::Min => "MIN",
            AggregateFunc::Max => "MAX",
        };
        if let Some(c) = column {
            if distinct {
                format!("{}(DISTINCT {})", func_name, c)
            } else {
                format!("{}({})", func_name, c)
            }
        } else {
            format!("{}(*)", func_name)
        }
    }

    fn agg_initial(func: &AggregateFunc, column: &Option<String>, distinct: bool) -> AggState {
        match func {
            AggregateFunc::Count => {
                if distinct {
                    AggState::CountDistinct(HashSet::new())
                } else {
                    let _ = column;
                    AggState::Count(0)
                }
            }
            AggregateFunc::Sum | AggregateFunc::Avg => {
                let _ = column;
                AggState::Sum(0.0)
            }
            AggregateFunc::Min => {
                let _ = column;
                AggState::Min(None)
            }
            AggregateFunc::Max => {
                let _ = column;
                AggState::Max(None)
            }
        }
    }

    fn agg_update(
        state: &mut AggState,
        spec: &AggSpec,
        left_table: &ColumnTable,
        right_table: &ColumnTable,
        left_alias: &str,
        right_alias: &str,
        jr: JoinRow,
    ) {
        let v = spec
            .column
            .as_ref()
            .map(|c| value_for_ref(c, left_table, right_table, left_alias, right_alias, jr));

        match (&spec.func, state) {
            (AggregateFunc::Count, AggState::CountDistinct(set)) => {
                if let Some(v) = v {
                    if !v.is_null() {
                        set.insert(v.to_bytes());
                    }
                }
            }
            (AggregateFunc::Count, AggState::Count(cn)) => match v {
                None => *cn += 1,
                Some(v) => {
                    if !v.is_null() {
                        *cn += 1;
                    }
                }
            },
            (AggregateFunc::Sum | AggregateFunc::Avg, AggState::Sum(sum)) => {
                if let Some(v) = v {
                    if let Some(n) = v.as_f64() {
                        *sum += n;
                    }
                }
            }
            (AggregateFunc::Min, AggState::Min(curr)) => {
                if let Some(v) = v {
                    if v.is_null() {
                        return;
                    }
                    match curr {
                        None => *curr = Some(v),
                        Some(c) => {
                            if SqlExecutor::compare_non_null(c, &v) == Ordering::Greater {
                                *curr = Some(v);
                            }
                        }
                    }
                }
            }
            (AggregateFunc::Max, AggState::Max(curr)) => {
                if let Some(v) = v {
                    if v.is_null() {
                        return;
                    }
                    match curr {
                        None => *curr = Some(v),
                        Some(c) => {
                            if SqlExecutor::compare_non_null(c, &v) == Ordering::Less {
                                *curr = Some(v);
                            }
                        }
                    }
                }
            }
            _ => {}
        }
    }

    fn agg_value(state: &AggState, func: &AggregateFunc) -> Value {
        match (func, state) {
            (AggregateFunc::Count, AggState::CountDistinct(set)) => Value::Int64(set.len() as i64),
            (AggregateFunc::Count, AggState::Count(cn)) => Value::Int64(*cn),
            (AggregateFunc::Sum, AggState::Sum(sum)) => Value::Float64(*sum),
            (AggregateFunc::Avg, AggState::Sum(sum)) => Value::Float64(*sum),
            (AggregateFunc::Min, AggState::Min(v)) => v.clone().unwrap_or(Value::Null),
            (AggregateFunc::Max, AggState::Max(v)) => v.clone().unwrap_or(Value::Null),
            _ => Value::Null,
        }
    }

    fn eval_having(expr: &SqlExpr, agg_out: &HashMap<String, Value>) -> Result<bool, ApexError> {
        match expr {
            SqlExpr::Paren(inner) => eval_having(inner, agg_out),
            SqlExpr::BinaryOp { left, op, right } => {
                let lv = match left.as_ref() {
                    SqlExpr::Function { name, args } => {
                        if args.len() != 1 {
                            return Err(ApexError::QueryParseError(
                                "HAVING only supports single-arg aggregates in JOIN yet".to_string(),
                            ));
                        }
                        let func = name.to_ascii_uppercase();
                        let col = match &args[0] {
                            SqlExpr::Column(c) => c.clone(),
                            _ => {
                                return Err(ApexError::QueryParseError(
                                    "HAVING aggregate arg must be a column".to_string(),
                                ))
                            }
                        };
                        let key = match func.as_str() {
                            "SUM" => format!("SUM({})", col),
                            "MIN" => format!("MIN({})", col),
                            "MAX" => format!("MAX({})", col),
                            "COUNT" => format!("COUNT({})", col),
                            _ => {
                                return Err(ApexError::QueryParseError(
                                    "Unsupported HAVING aggregate".to_string(),
                                ))
                            }
                        };
                        agg_out.get(&key).cloned().unwrap_or(Value::Null)
                    }
                    _ => {
                        return Err(ApexError::QueryParseError(
                            "Unsupported HAVING expression in JOIN yet".to_string(),
                        ))
                    }
                };

                let rv = match right.as_ref() {
                    SqlExpr::Literal(v) => v.clone(),
                    _ => {
                        return Err(ApexError::QueryParseError(
                            "HAVING comparison must be against a literal".to_string(),
                        ))
                    }
                };

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
                    _ => {
                        return Err(ApexError::QueryParseError(
                            "Unsupported HAVING operator".to_string(),
                        ))
                    }
                })
            }
            _ => Err(ApexError::QueryParseError(
                "Unsupported HAVING expression in JOIN yet".to_string(),
            )),
        }
    }

    let mut group_cols: Vec<String> = stmt.group_by.clone();
    if group_cols.is_empty() {
        group_cols = Vec::new();
    }

    let mut agg_specs: Vec<AggSpec> = Vec::new();
    for c in &stmt.columns {
        if let SelectColumn::Aggregate {
            func,
            column,
            distinct,
            alias,
        } = c
        {
            agg_specs.push(AggSpec {
                func: func.clone(),
                column: column.clone(),
                distinct: *distinct,
                out_name: out_name_for_agg(func, column, *distinct, alias),
            });
        }
    }

    let mut groups: HashMap<String, (Vec<Value>, Vec<AggState>)> = HashMap::new();

    for jr in joined {
        let gvals: Vec<Value> = group_cols
            .iter()
            .map(|c| value_for_ref(c, left_table, right_table, &ctx.left_alias, &ctx.right_alias, *jr))
            .collect();
        let gkey = gvals
            .iter()
            .map(|v| v.to_string_value())
            .collect::<Vec<_>>()
            .join("\u{1f}");

        let entry = groups.entry(gkey).or_insert_with(|| {
            let states = agg_specs
                .iter()
                .map(|s| agg_initial(&s.func, &s.column, s.distinct))
                .collect::<Vec<_>>();
            (gvals, states)
        });

        for (i, spec) in agg_specs.iter().enumerate() {
            agg_update(
                &mut entry.1[i],
                spec,
                left_table,
                right_table,
                &ctx.left_alias,
                &ctx.right_alias,
                *jr,
            );
        }
    }

    let mut out_columns: Vec<String> = Vec::new();
    for c in &stmt.columns {
        match c {
            SelectColumn::Column(name) => out_columns.push(split_qual(name).1.to_string()),
            SelectColumn::ColumnAlias { alias, .. } => out_columns.push(alias.clone()),
            SelectColumn::Aggregate {
                func,
                column,
                distinct,
                alias,
            } => out_columns.push(out_name_for_agg(func, column, *distinct, alias)),
            _ => {
                return Err(ApexError::QueryParseError(
                    "Unsupported SELECT item in JOIN aggregation yet".to_string(),
                ))
            }
        }
    }

    let mut rows: Vec<Vec<Value>> = Vec::with_capacity(groups.len());
    for (_k, (gvals, states)) in groups {
        let mut agg_out: HashMap<String, Value> = HashMap::new();
        for (i, spec) in agg_specs.iter().enumerate() {
            let val = agg_value(&states[i], &spec.func);

            let canon = match spec.func {
                AggregateFunc::Count => {
                    if let Some(c) = spec.column.clone() {
                        if spec.distinct {
                            format!("COUNT(DISTINCT {})", c)
                        } else {
                            format!("COUNT({})", c)
                        }
                    } else {
                        "COUNT(*)".to_string()
                    }
                }
                AggregateFunc::Sum => {
                    if let Some(c) = spec.column.clone() {
                        if spec.distinct {
                            format!("SUM(DISTINCT {})", c)
                        } else {
                            format!("SUM({})", c)
                        }
                    } else {
                        "SUM(*)".to_string()
                    }
                }
                AggregateFunc::Avg => {
                    if let Some(c) = spec.column.clone() {
                        if spec.distinct {
                            format!("AVG(DISTINCT {})", c)
                        } else {
                            format!("AVG({})", c)
                        }
                    } else {
                        "AVG(*)".to_string()
                    }
                }
                AggregateFunc::Min => spec
                    .column
                    .clone()
                    .map(|c| format!("MIN({})", c))
                    .unwrap_or_else(|| "MIN(*)".to_string()),
                AggregateFunc::Max => spec
                    .column
                    .clone()
                    .map(|c| format!("MAX({})", c))
                    .unwrap_or_else(|| "MAX(*)".to_string()),
            };

            agg_out.insert(canon, val.clone());
            agg_out.insert(spec.out_name.clone(), val);
        }

        if let Some(ref having) = stmt.having {
            if !eval_having(having, &agg_out)? {
                continue;
            }
        }

        let mut row: Vec<Value> = Vec::with_capacity(stmt.columns.len());
        for c in &stmt.columns {
            match c {
                SelectColumn::Column(name) => {
                    let (a, _c) = split_qual(name);
                    let v = if a.is_empty() {
                        group_cols
                            .iter()
                            .position(|gc| split_qual(gc).1 == split_qual(name).1)
                            .and_then(|idx| gvals.get(idx).cloned())
                            .unwrap_or(Value::Null)
                    } else {
                        group_cols
                            .iter()
                            .position(|gc| gc == name)
                            .and_then(|idx| gvals.get(idx).cloned())
                            .unwrap_or(Value::Null)
                    };
                    row.push(v);
                }
                SelectColumn::ColumnAlias { column, .. } => {
                    let (a, _c) = split_qual(column);
                    let v = if a.is_empty() {
                        group_cols
                            .iter()
                            .position(|gc| split_qual(gc).1 == split_qual(column).1)
                            .and_then(|idx| gvals.get(idx).cloned())
                            .unwrap_or(Value::Null)
                    } else {
                        group_cols
                            .iter()
                            .position(|gc| gc == column)
                            .and_then(|idx| gvals.get(idx).cloned())
                            .unwrap_or(Value::Null)
                    };
                    row.push(v);
                }
                SelectColumn::Aggregate {
                    func,
                    column,
                    distinct,
                    alias,
                } => {
                    let name = out_name_for_agg(func, column, *distinct, alias);
                    row.push(agg_out.get(&name).cloned().unwrap_or(Value::Null));
                }
                _ => {}
            }
        }
        rows.push(row);
    }

    if !stmt.order_by.is_empty() {
        rows.sort_by(|a, b| {
            for ob in &stmt.order_by {
                let (alias, col) = split_qual(&ob.column);
                let target = if alias.is_empty() { col } else { ob.column.as_str() };
                let idx = out_columns
                    .iter()
                    .position(|c| c == col || c == target)
                    .unwrap_or(0);
                let av = a.get(idx).cloned().unwrap_or(Value::Null);
                let bv = b.get(idx).cloned().unwrap_or(Value::Null);
                let cmp = SqlExecutor::compare_values(Some(&av), Some(&bv), ob.nulls_first);
                let cmp = if ob.descending { cmp.reverse() } else { cmp };
                if cmp != Ordering::Equal {
                    return cmp;
                }
            }
            Ordering::Equal
        });
    }

    if stmt.offset.is_some() || stmt.limit.is_some() {
        let offset = stmt.offset.unwrap_or(0);
        let limit = stmt.limit.unwrap_or(usize::MAX);
        rows = rows.into_iter().skip(offset).take(limit).collect::<Vec<_>>();
    }

    Ok(SqlResult::new(out_columns, rows))
}
