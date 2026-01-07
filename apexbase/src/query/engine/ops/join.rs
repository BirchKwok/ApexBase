use crate::query::sql_expr_to_filter;
use crate::query::sql_parser::{BinaryOperator, FromItem, JoinType, SelectColumn, SelectStatement, SqlExpr, UnaryOperator};
use crate::query::{SqlExecutor, SqlResult};
use crate::table::ColumnTable;
use crate::ApexError;
use std::cmp::Ordering;
use std::collections::HashMap;
use ahash::AHashMap;

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
enum JoinKey {
    Bool(bool),
    I64(i64),
    U64(u64),
    F32(u32),
    F64(u64),
    String(String),
    Binary(Vec<u8>),
    Bytes(Vec<u8>),
}

pub(crate) fn try_aggregate_join_rows_fast_path(
    stmt: &SelectStatement,
    tables: &mut HashMap<String, ColumnTable>,
) -> Result<Option<SqlResult>, ApexError> {
    use crate::data::Value;
    use crate::query::sql_parser::AggregateFunc;
    use crate::table::column_table::TypedColumn;
    use crate::table::column_table::BitVec;

    // Pattern constraints (keep conservative):
    // - single INNER JOIN with ON a=b
    // - GROUP BY columns are left-only
    // - SELECT contains only GROUP BY columns + supported aggregates on the right table
    // - WHERE contains only simple predicates; split into left-only and right-only (no cross-table)
    // - no DISTINCT, no HAVING, no OFFSET
    if stmt.joins.len() != 1 {
        return Ok(None);
    }
    if stmt.distinct {
        return Ok(None);
    }
    if stmt.having.is_some() {
        return Ok(None);
    }
    if stmt.offset.unwrap_or(0) != 0 {
        return Ok(None);
    }
    if stmt.group_by.is_empty() {
        return Ok(None);
    }

    #[inline]
    fn unwrap_paren<'a>(mut e: &'a SqlExpr) -> &'a SqlExpr {
        while let SqlExpr::Paren(inner) = e {
            e = inner.as_ref();
        }
        e
    }

    #[derive(Clone)]
    enum Side {
        Left,
        Right,
        Unknown,
    }

    #[inline]
    fn side_for_col<'a>(
        c: &'a str,
        left_alias: &'a str,
        left_table: &'a str,
        right_alias: &'a str,
        right_table: &'a str,
    ) -> (Side, &'a str) {
        let (a, col) = split_qual(c);
        if a.is_empty() {
            return (Side::Unknown, col);
        }
        if a == left_alias || a == left_table {
            return (Side::Left, col);
        }
        if a == right_alias || a == right_table {
            return (Side::Right, col);
        }
        (Side::Unknown, col)
    }

    #[derive(Clone)]
    enum SimplePredLit {
        Bool(bool),
        I64(i64),
        F64(f64),
        String(String),
        Null,
    }

    #[derive(Clone)]
    struct SimplePred {
        col: String,
        op: BinaryOperator,
        lit: SimplePredLit,
    }

    fn split_and<'a>(expr: &'a SqlExpr, out: &mut Vec<&'a SqlExpr>) {
        match expr {
            SqlExpr::BinaryOp { left, op: BinaryOperator::And, right } => {
                split_and(left.as_ref(), out);
                split_and(right.as_ref(), out);
            }
            SqlExpr::Paren(inner) => split_and(inner.as_ref(), out),
            _ => out.push(expr),
        }
    }

    fn parse_simple_pred(e: &SqlExpr) -> Option<(String, BinaryOperator, SimplePredLit)> {
        let e = match e {
            SqlExpr::Paren(inner) => inner.as_ref(),
            _ => e,
        };
        let SqlExpr::BinaryOp { left, op, right } = e else {
            return None;
        };
        let col = match left.as_ref() {
            SqlExpr::Column(c) => c.clone(),
            SqlExpr::Paren(inner) => match inner.as_ref() {
                SqlExpr::Column(c) => c.clone(),
                _ => return None,
            },
            _ => return None,
        };
        let lit_expr = unwrap_paren(right.as_ref());
        let lit = match lit_expr {
            SqlExpr::Literal(v) => match v {
                Value::Null => SimplePredLit::Null,
                Value::Bool(b) => SimplePredLit::Bool(*b),
                Value::Int64(x) => SimplePredLit::I64(*x),
                Value::Int32(x) => SimplePredLit::I64(*x as i64),
                Value::Int16(x) => SimplePredLit::I64(*x as i64),
                Value::Int8(x) => SimplePredLit::I64(*x as i64),
                Value::Float64(x) => SimplePredLit::F64(*x),
                Value::Float32(x) => SimplePredLit::F64(*x as f64),
                Value::String(s) => SimplePredLit::String(s.clone()),
                _ => return None,
            },
            _ => return None,
        };
        Some((col, op.clone(), lit))
    }

    // Determine left/right tables and join keys.
    let from_item = match stmt.from.as_ref() {
        Some(f) => f,
        None => return Ok(None),
    };

    let (left_table_name, left_alias) = match from_item {
        FromItem::Table { table, alias } => {
            let t = table.clone();
            let a = alias.clone().unwrap_or_else(|| t.clone());
            (t, a)
        }
        FromItem::Subquery { .. } => return Ok(None),
    };

    let join = &stmt.joins[0];
    if join.join_type != JoinType::Inner {
        return Ok(None);
    }
    let (right_table_name, right_alias) = match &join.right {
        FromItem::Table { table, alias } => {
            let t = table.clone();
            let a = alias.clone().unwrap_or_else(|| t.clone());
            (t, a)
        }
        FromItem::Subquery { .. } => return Ok(None),
    };

    // Flush pending writes (matching other execution paths).
    if let Some(t) = tables.get_mut(&left_table_name) {
        if t.has_pending_writes() {
            t.flush_write_buffer();
        }
    }
    if let Some(t) = tables.get_mut(&right_table_name) {
        if t.has_pending_writes() {
            t.flush_write_buffer();
        }
    }

    let left_table = match tables.get(&left_table_name) {
        Some(t) => t,
        None => return Ok(None),
    };
    let right_table = match tables.get(&right_table_name) {
        Some(t) => t,
        None => return Ok(None),
    };

    // Parse ON clause: column = column
    let (left_key_ref, right_key_ref) = match unwrap_paren(&join.on) {
        SqlExpr::BinaryOp { left, op: BinaryOperator::Eq, right } => {
            let l = match left.as_ref() {
                SqlExpr::Column(c) => c.clone(),
                _ => return Ok(None),
            };
            let r = match right.as_ref() {
                SqlExpr::Column(c) => c.clone(),
                _ => return Ok(None),
            };
            (l, r)
        }
        _ => return Ok(None),
    };

    // Determine which side join keys belong to.
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
        return Ok(None);
    };

    let mut group_cols: Vec<String> = Vec::with_capacity(stmt.group_by.len());
    for g in &stmt.group_by {
        let (side, col) = side_for_col(g, &left_alias, &left_table_name, &right_alias, &right_table_name);
        match side {
            Side::Left | Side::Unknown => group_cols.push(col.to_string()),
            _ => return Ok(None),
        }
    }

    #[derive(Clone)]
    enum AggKind {
        CountStar,
        CountCol { idx: usize },
        Sum { idx: usize },
        Min { idx: usize },
        Max { idx: usize },
        Avg { idx: usize },
    }

    #[derive(Clone)]
    enum OutputExpr {
        GroupCol { col: String },
        Agg { kind: AggKind },
    }

    fn get_or_push(cols: &mut Vec<String>, col: &str) -> usize {
        for (i, c) in cols.iter().enumerate() {
            if c == col {
                return i;
            }
        }
        cols.push(col.to_string());
        cols.len() - 1
    }

    let mut count_cols: Vec<String> = Vec::new();
    let mut sum_cols: Vec<String> = Vec::new();
    let mut min_cols: Vec<String> = Vec::new();
    let mut max_cols: Vec<String> = Vec::new();
    let mut avg_cols: Vec<String> = Vec::new();

    let mut outputs: Vec<OutputExpr> = Vec::with_capacity(stmt.columns.len());
    for sc in &stmt.columns {
        match sc {
            SelectColumn::Column(c) => {
                let (side, col) = side_for_col(c, &left_alias, &left_table_name, &right_alias, &right_table_name);
                match side {
                    Side::Left | Side::Unknown => {
                        if !group_cols.iter().any(|g| g == col) {
                            return Ok(None);
                        }
                        outputs.push(OutputExpr::GroupCol { col: col.to_string() });
                    }
                    _ => return Ok(None),
                }
            }
            SelectColumn::ColumnAlias { column, alias } => {
                let (side, col) = side_for_col(column, &left_alias, &left_table_name, &right_alias, &right_table_name);
                match side {
                    Side::Left | Side::Unknown => {
                        if !group_cols.iter().any(|g| g == col) {
                            return Ok(None);
                        }
                        outputs.push(OutputExpr::GroupCol { col: col.to_string() });
                    }
                    _ => return Ok(None),
                }
                // keep alias via output names later
                let _ = alias;
            }
            SelectColumn::Aggregate { func, column, distinct: false, alias: _ } => {
                let kind = match (func, column) {
                    (AggregateFunc::Count, None) => AggKind::CountStar,
                    (AggregateFunc::Count, Some(c)) => {
                        let (side, col) = side_for_col(c, &left_alias, &left_table_name, &right_alias, &right_table_name);
                        match side {
                            Side::Right => {
                                let idx = get_or_push(&mut count_cols, col);
                                AggKind::CountCol { idx }
                            }
                            _ => return Ok(None),
                        }
                    }
                    (AggregateFunc::Sum, Some(c)) => {
                        let (side, col) = side_for_col(c, &left_alias, &left_table_name, &right_alias, &right_table_name);
                        match side {
                            Side::Right => {
                                let idx = get_or_push(&mut sum_cols, col);
                                AggKind::Sum { idx }
                            }
                            _ => return Ok(None),
                        }
                    }
                    (AggregateFunc::Min, Some(c)) => {
                        let (side, col) = side_for_col(c, &left_alias, &left_table_name, &right_alias, &right_table_name);
                        match side {
                            Side::Right => {
                                let idx = get_or_push(&mut min_cols, col);
                                AggKind::Min { idx }
                            }
                            _ => return Ok(None),
                        }
                    }
                    (AggregateFunc::Max, Some(c)) => {
                        let (side, col) = side_for_col(c, &left_alias, &left_table_name, &right_alias, &right_table_name);
                        match side {
                            Side::Right => {
                                let idx = get_or_push(&mut max_cols, col);
                                AggKind::Max { idx }
                            }
                            _ => return Ok(None),
                        }
                    }
                    (AggregateFunc::Avg, Some(c)) => {
                        let (side, col) = side_for_col(c, &left_alias, &left_table_name, &right_alias, &right_table_name);
                        match side {
                            Side::Right => {
                                let idx = get_or_push(&mut avg_cols, col);
                                AggKind::Avg { idx }
                            }
                            _ => return Ok(None),
                        }
                    }
                    _ => return Ok(None),
                };
                outputs.push(OutputExpr::Agg { kind });
            }
            _ => return Ok(None),
        }
    }

    // WHERE: split AND terms, each must be left-only or right-only simple predicate.
    let mut left_preds: Vec<SimplePred> = Vec::new();
    let mut right_preds: Vec<SimplePred> = Vec::new();
    if let Some(w) = stmt.where_clause.as_ref() {
        let mut terms: Vec<&SqlExpr> = Vec::new();
        split_and(w, &mut terms);
        for t in terms {
            let Some((col, op, lit)) = parse_simple_pred(t) else {
                return Ok(None);
            };
            let (side, c) = side_for_col(&col, &left_alias, &left_table_name, &right_alias, &right_table_name);
            match side {
                Side::Left => left_preds.push(SimplePred { col: c.to_string(), op, lit }),
                Side::Right => right_preds.push(SimplePred { col: c.to_string(), op, lit }),
                Side::Unknown => return Ok(None),
            }
        }
    }

    // Resolve typed column indices.
    let l_schema = left_table.schema_ref();
    let r_schema = right_table.schema_ref();
    let l_user_idx = match l_schema.get_index(&left_key_col) {
        Some(i) => i,
        None => return Ok(None),
    };
    let r_user_idx = match r_schema.get_index(&right_key_col) {
        Some(i) => i,
        None => return Ok(None),
    };
    let mut l_group_idxs: Vec<usize> = Vec::with_capacity(group_cols.len());
    for g in &group_cols {
        let idx = match l_schema.get_index(g) {
            Some(i) => i,
            None => return Ok(None),
        };
        l_group_idxs.push(idx);
    }

    #[inline]
    fn p_col_mismatch(preds: &[SimplePred], col: &str) -> bool {
        if preds.is_empty() {
            return false;
        }
        if preds.len() != 1 {
            return true;
        }
        preds[0].col != col
    }

    // Specialized ultra-fast path for the common perf pattern:
    // GROUP BY one left string col; aggregates are exactly COUNT(*) and SUM(right_numeric);
    // optional single right predicate: SUM_col >= <number>; no left predicates.
    // This keeps perf_1m_join_filter_order_limit in tens of ms.
    if left_preds.is_empty() && group_cols.len() == 1 {
        let mut sum_col: Option<String> = None;
        let mut has_count_star = false;
        let mut ok = true;
        for o in &outputs {
            match o {
                OutputExpr::GroupCol { .. } => {}
                OutputExpr::Agg { kind } => match kind {
                    AggKind::CountStar => has_count_star = true,
                    AggKind::Sum { idx } => {
                        if sum_col.is_some() {
                            ok = false;
                            break;
                        }
                        let Some(col) = sum_cols.get(*idx) else {
                            ok = false;
                            break;
                        };
                        sum_col = Some(col.clone());
                    }
                    _ => {
                        ok = false;
                        break;
                    }
                },
            }
        }

        // Only support 0 or 1 right predicate: sum_col >= literal.
        let threshold: Option<f64> = if ok {
            if right_preds.is_empty() {
                None
            } else if right_preds.len() == 1 {
                let p = &right_preds[0];
                match (&p.op, &p.lit) {
                    (BinaryOperator::Ge, SimplePredLit::F64(x)) => Some(*x),
                    (BinaryOperator::Ge, SimplePredLit::I64(x)) => Some(*x as f64),
                    _ => {
                        ok = false;
                        None
                    }
                }
            } else {
                ok = false;
                None
            }
        } else {
            None
        };

        if ok && has_count_star {
            if let Some(sum_col) = sum_col {
                if p_col_mismatch(&right_preds, &sum_col) {
                    ok = false;
                }

                if ok {
                    let gidx = l_group_idxs[0];
                    let ridx = match r_schema.get_index(&sum_col) {
                        Some(i) => i,
                        None => return Ok(None),
                    };
                let l_cols = left_table.columns_ref();
                let r_cols = right_table.columns_ref();

                let (l_user_data, l_user_nulls) = match &l_cols[l_user_idx] {
                    TypedColumn::Int64 { data, nulls } => (data.as_slice(), nulls),
                    _ => return Ok(None),
                };
                let (r_user_data, r_user_nulls) = match &r_cols[r_user_idx] {
                    TypedColumn::Int64 { data, nulls } => (data.as_slice(), nulls),
                    _ => return Ok(None),
                };
                let l_group_col = match &l_cols[gidx] {
                    TypedColumn::String(col) => col,
                    _ => return Ok(None),
                };

                enum NumCol<'a> {
                    F64(&'a [f64], &'a BitVec),
                    I64(&'a [i64], &'a BitVec),
                }
                let r_num = match &r_cols[ridx] {
                    TypedColumn::Float64 { data, nulls } => NumCol::F64(data.as_slice(), nulls),
                    TypedColumn::Int64 { data, nulls } => NumCol::I64(data.as_slice(), nulls),
                    _ => return Ok(None),
                };

                let thr = threshold.unwrap_or(f64::NEG_INFINITY);

                let r_deleted = right_table.deleted_ref();
                let mut by_user: AHashMap<i64, (i64, f64)> = AHashMap::new();
                let r_len = match &r_num {
                    NumCol::F64(d, _) => right_table.get_row_count().min(r_user_data.len()).min(d.len()),
                    NumCol::I64(d, _) => right_table.get_row_count().min(r_user_data.len()).min(d.len()),
                };
                for i in 0..r_len {
                    if r_deleted.get(i) {
                        continue;
                    }
                    if r_user_nulls.get(i) {
                        continue;
                    }
                    let (x, is_null) = match &r_num {
                        NumCol::F64(d, n) => (d[i], n.get(i)),
                        NumCol::I64(d, n) => (d[i] as f64, n.get(i)),
                    };
                    if is_null {
                        continue;
                    }
                    if x < thr {
                        continue;
                    }
                    let uid = r_user_data[i];
                    let e = by_user.entry(uid).or_insert((0, 0.0));
                    e.0 += 1;
                    e.1 += x;
                }

                let l_deleted = left_table.deleted_ref();
                let l_len = left_table.get_row_count().min(l_user_data.len());
                let mut by_group: AHashMap<String, (i64, f64)> = AHashMap::new();
                for i in 0..l_len {
                    if l_deleted.get(i) {
                        continue;
                    }
                    if l_user_nulls.get(i) {
                        continue;
                    }
                    let uid = l_user_data[i];
                    let Some((cn, sm)) = by_user.get(&uid) else {
                        continue;
                    };
                    let Some(g) = l_group_col.get(i) else {
                        continue;
                    };
                    let e = by_group.entry(g.to_string()).or_insert((0, 0.0));
                    e.0 += *cn;
                    e.1 += *sm;
                }

                let mut out_columns: Vec<String> = Vec::with_capacity(stmt.columns.len());
                for sc in &stmt.columns {
                    match sc {
                        SelectColumn::Column(c) => out_columns.push(split_qual(c).1.to_string()),
                        SelectColumn::ColumnAlias { alias, .. } => out_columns.push(alias.clone()),
                        SelectColumn::Aggregate { alias, func, column, .. } => {
                            if let Some(a) = alias {
                                out_columns.push(a.clone());
                            } else {
                                out_columns.push(match (func, column.as_ref()) {
                                    (AggregateFunc::Count, None) => "count".to_string(),
                                    (AggregateFunc::Sum, Some(c)) => format!("sum({})", c),
                                    _ => "agg".to_string(),
                                });
                            }
                        }
                        _ => out_columns.push("expr".to_string()),
                    }
                }

                let mut rows: Vec<Vec<Value>> = Vec::with_capacity(by_group.len());
                for (g, (cn, sm)) in by_group {
                    let mut row: Vec<Value> = Vec::with_capacity(outputs.len());
                    for o in &outputs {
                        match o {
                            OutputExpr::GroupCol { .. } => row.push(Value::String(g.clone())),
                            OutputExpr::Agg { kind } => match kind {
                                AggKind::CountStar => row.push(Value::Int64(cn)),
                                AggKind::Sum { .. } => row.push(Value::Float64(sm)),
                                _ => row.push(Value::Null),
                            },
                        }
                    }
                    rows.push(row);
                }

                if !stmt.order_by.is_empty() {
                    let ob = &stmt.order_by[0];
                    let key = split_qual(&ob.column).1;
                    let mut idx_opt: Option<usize> = None;
                    for (i, c) in out_columns.iter().enumerate() {
                        if c == key {
                            idx_opt = Some(i);
                            break;
                        }
                    }
                    let Some(idx) = idx_opt else {
                        return Ok(None);
                    };
                    rows.sort_by(|a, b| {
                        let av = a.get(idx).cloned().unwrap_or(Value::Null);
                        let bv = b.get(idx).cloned().unwrap_or(Value::Null);
                        let cmp = SqlExecutor::compare_values(Some(&av), Some(&bv), ob.nulls_first);
                        if ob.descending { cmp.reverse() } else { cmp }
                    });
                }
                if let Some(limit) = stmt.limit {
                    if rows.len() > limit {
                        rows.truncate(limit);
                    }
                }

                    return Ok(Some(SqlResult::new(out_columns, rows)));
                }
            }
        }
    }

    // Aggregate input columns (right table)
    let mut r_need_cols: Vec<String> = Vec::new();
    for c in &count_cols {
        if !r_need_cols.iter().any(|x| x == c) {
            r_need_cols.push(c.clone());
        }
    }
    for c in &sum_cols {
        if !r_need_cols.iter().any(|x| x == c) {
            r_need_cols.push(c.clone());
        }
    }
    for c in &min_cols {
        if !r_need_cols.iter().any(|x| x == c) {
            r_need_cols.push(c.clone());
        }
    }
    for c in &max_cols {
        if !r_need_cols.iter().any(|x| x == c) {
            r_need_cols.push(c.clone());
        }
    }
    for c in &avg_cols {
        if !r_need_cols.iter().any(|x| x == c) {
            r_need_cols.push(c.clone());
        }
    }
    for p in &right_preds {
        if !r_need_cols.iter().any(|c| c == &p.col) {
            r_need_cols.push(p.col.clone());
        }
    }

    let mut r_need_idxs: AHashMap<String, usize> = AHashMap::new();
    for c in &r_need_cols {
        let idx = match r_schema.get_index(c) {
            Some(i) => i,
            None => return Ok(None),
        };
        r_need_idxs.insert(c.clone(), idx);
    }

    let mut l_need_idxs: AHashMap<String, usize> = AHashMap::new();
    for p in &left_preds {
        if !l_need_idxs.contains_key(&p.col) {
            let idx = match l_schema.get_index(&p.col) {
                Some(i) => i,
                None => return Ok(None),
            };
            l_need_idxs.insert(p.col.clone(), idx);
        }
    }

    let l_cols = left_table.columns_ref();
    let r_cols = right_table.columns_ref();

    fn col_get_value(col: &TypedColumn, idx: usize) -> Option<Value> {
        col.get(idx)
    }

    fn col_get_join_key(col: &TypedColumn, idx: usize) -> Option<JoinKey> {
        match col {
            TypedColumn::Int64 { data, nulls } => {
                if nulls.get(idx) { None } else { Some(JoinKey::I64(data[idx])) }
            }
            TypedColumn::Float64 { data, nulls } => {
                if nulls.get(idx) { None } else { Some(JoinKey::F64(data[idx].to_bits())) }
            }
            TypedColumn::Bool { data, nulls } => {
                if nulls.get(idx) { None } else { Some(JoinKey::Bool(data.get(idx))) }
            }
            TypedColumn::String(col) => col.get(idx).map(|s| JoinKey::String(s.to_string())),
            TypedColumn::Mixed { data, nulls } => {
                if nulls.get(idx) { None } else { Some(join_key(&data[idx])) }
            }
        }
    }

    fn pred_eval(col: &TypedColumn, idx: usize, op: &BinaryOperator, lit: &SimplePredLit) -> bool {
        if col.is_null(idx) {
            return matches!(lit, SimplePredLit::Null) && matches!(op, BinaryOperator::Eq);
        }
        let v = match col_get_value(col, idx) {
            Some(x) => x,
            None => return false,
        };
        let rhs = match lit {
            SimplePredLit::Null => Value::Null,
            SimplePredLit::Bool(b) => Value::Bool(*b),
            SimplePredLit::I64(x) => Value::Int64(*x),
            SimplePredLit::F64(x) => Value::Float64(*x),
            SimplePredLit::String(s) => Value::String(s.clone()),
        };
        let cmp = SqlExecutor::compare_values(Some(&v), Some(&rhs), None);
        match op {
            BinaryOperator::Eq => cmp == Ordering::Equal,
            BinaryOperator::NotEq => cmp != Ordering::Equal,
            BinaryOperator::Lt => cmp == Ordering::Less,
            BinaryOperator::Le => cmp == Ordering::Less || cmp == Ordering::Equal,
            BinaryOperator::Gt => cmp == Ordering::Greater,
            BinaryOperator::Ge => cmp == Ordering::Greater || cmp == Ordering::Equal,
            _ => false,
        }
    }

    fn value_as_f64(v: &Value) -> Option<f64> {
        match v {
            Value::Float64(x) => Some(*x),
            Value::Float32(x) => Some(*x as f64),
            Value::Int64(x) => Some(*x as f64),
            Value::Int32(x) => Some(*x as f64),
            Value::Int16(x) => Some(*x as f64),
            Value::Int8(x) => Some(*x as f64),
            _ => None,
        }
    }

    // Join key columns: allow common types.
    let r_key_col = &r_cols[r_user_idx];
    let l_key_col = &l_cols[l_user_idx];
    // Quick type compatibility check to avoid hashing different representations.
    match (r_key_col, l_key_col) {
        (TypedColumn::Int64 { .. }, TypedColumn::Int64 { .. }) => {}
        (TypedColumn::String(_), TypedColumn::String(_)) => {}
        _ => return Ok(None),
    }

    // Pre-check group columns exist and are retrievable.
    for &gi in &l_group_idxs {
        let _ = &l_cols[gi];
    }

    // Pre-aggregate right table by join key.
    let r_deleted = right_table.deleted_ref();
    #[derive(Clone, Default)]
    struct RightAggState {
        count_star: i64,
        count_col: Vec<i64>,
        sum: Vec<f64>,
        min: Vec<Option<f64>>,
        max: Vec<Option<f64>>,
        avg_sum: Vec<f64>,
        avg_cnt: Vec<i64>,
    }

    impl RightAggState {
        fn new(n_count: usize, n_sum: usize, n_min: usize, n_max: usize, n_avg: usize) -> Self {
            Self {
                count_star: 0,
                count_col: vec![0; n_count],
                sum: vec![0.0; n_sum],
                min: vec![None; n_min],
                max: vec![None; n_max],
                avg_sum: vec![0.0; n_avg],
                avg_cnt: vec![0; n_avg],
            }
        }
    }

    let mut by_join: AHashMap<JoinKey, RightAggState> = AHashMap::new();
    let r_len = right_table.get_row_count();
    for i in 0..r_len {
        if r_deleted.get(i) {
            continue;
        }
        // right-side preds
        let mut ok = true;
        for p in &right_preds {
            let Some(&cidx) = r_need_idxs.get(&p.col) else {
                ok = false;
                break;
            };
            if !pred_eval(&r_cols[cidx], i, &p.op, &p.lit) {
                ok = false;
                break;
            }
        }
        if !ok {
            continue;
        }

        let Some(jk) = col_get_join_key(r_key_col, i) else {
            continue;
        };
        let st = by_join.entry(jk).or_insert_with(|| {
            RightAggState::new(count_cols.len(), sum_cols.len(), min_cols.len(), max_cols.len(), avg_cols.len())
        });
        st.count_star += 1;

        for o in &outputs {
            let OutputExpr::Agg { kind, .. } = o else {
                continue;
            };
            match kind {
                AggKind::CountStar => {}
                AggKind::CountCol { idx } => {
                    let col = &count_cols[*idx];
                    let Some(&cidx) = r_need_idxs.get(col) else { return Ok(None); };
                    if !r_cols[cidx].is_null(i) {
                        st.count_col[*idx] += 1;
                    }
                }
                AggKind::Sum { idx } => {
                    let col = &sum_cols[*idx];
                    let Some(&cidx) = r_need_idxs.get(col) else { return Ok(None); };
                    let Some(v) = col_get_value(&r_cols[cidx], i) else { continue; };
                    let Some(x) = value_as_f64(&v) else { return Ok(None); };
                    st.sum[*idx] += x;
                }
                AggKind::Min { idx } => {
                    let col = &min_cols[*idx];
                    let Some(&cidx) = r_need_idxs.get(col) else { return Ok(None); };
                    let Some(v) = col_get_value(&r_cols[cidx], i) else { continue; };
                    let Some(x) = value_as_f64(&v) else { return Ok(None); };
                    match st.min[*idx] {
                        None => st.min[*idx] = Some(x),
                        Some(cur) => {
                            if x < cur {
                                st.min[*idx] = Some(x);
                            }
                        }
                    }
                }
                AggKind::Max { idx } => {
                    let col = &max_cols[*idx];
                    let Some(&cidx) = r_need_idxs.get(col) else { return Ok(None); };
                    let Some(v) = col_get_value(&r_cols[cidx], i) else { continue; };
                    let Some(x) = value_as_f64(&v) else { return Ok(None); };
                    match st.max[*idx] {
                        None => st.max[*idx] = Some(x),
                        Some(cur) => {
                            if x > cur {
                                st.max[*idx] = Some(x);
                            }
                        }
                    }
                }
                AggKind::Avg { idx } => {
                    let col = &avg_cols[*idx];
                    let Some(&cidx) = r_need_idxs.get(col) else { return Ok(None); };
                    let Some(v) = col_get_value(&r_cols[cidx], i) else { continue; };
                    let Some(x) = value_as_f64(&v) else { return Ok(None); };
                    st.avg_sum[*idx] += x;
                    st.avg_cnt[*idx] += 1;
                }
            }
        }
    }

    // Aggregate by left group key.
    #[derive(Clone, Default)]
    struct OutAggState {
        count_star: i64,
        count_col: Vec<i64>,
        sum: Vec<f64>,
        min: Vec<Option<f64>>,
        max: Vec<Option<f64>>,
        avg_sum: Vec<f64>,
        avg_cnt: Vec<i64>,
    }

    impl OutAggState {
        fn new(n_count: usize, n_sum: usize, n_min: usize, n_max: usize, n_avg: usize) -> Self {
            Self {
                count_star: 0,
                count_col: vec![0; n_count],
                sum: vec![0.0; n_sum],
                min: vec![None; n_min],
                max: vec![None; n_max],
                avg_sum: vec![0.0; n_avg],
                avg_cnt: vec![0; n_avg],
            }
        }
    }
    let l_deleted = left_table.deleted_ref();
    let l_len = left_table.get_row_count();
    let mut by_group: AHashMap<Vec<JoinKey>, (Vec<Value>, OutAggState)> = AHashMap::new();
    for i in 0..l_len {
        if l_deleted.get(i) {
            continue;
        }
        // left-side preds
        let mut ok = true;
        for p in &left_preds {
            let Some(&cidx) = l_need_idxs.get(&p.col) else {
                ok = false;
                break;
            };
            if !pred_eval(&l_cols[cidx], i, &p.op, &p.lit) {
                ok = false;
                break;
            }
        }
        if !ok {
            continue;
        }

        let Some(jk) = col_get_join_key(l_key_col, i) else {
            continue;
        };
        let Some(rst) = by_join.get(&jk) else {
            continue;
        };

        let mut gkeys: Vec<JoinKey> = Vec::with_capacity(l_group_idxs.len());
        let mut gvals: Vec<Value> = Vec::with_capacity(l_group_idxs.len());
        for &gi in &l_group_idxs {
            let col = &l_cols[gi];
            let Some(k) = col_get_join_key(col, i) else {
                gkeys.clear();
                gvals.clear();
                break;
            };
            gkeys.push(k);
            let Some(v) = col_get_value(col, i) else {
                gkeys.clear();
                gvals.clear();
                break;
            };
            gvals.push(v);
        }
        if gkeys.is_empty() {
            continue;
        }

        let entry = by_group.entry(gkeys).or_insert_with(|| {
            (
                gvals,
                OutAggState::new(count_cols.len(), sum_cols.len(), min_cols.len(), max_cols.len(), avg_cols.len()),
            )
        });
        let out = &mut entry.1;

        out.count_star += rst.count_star;
        for (i, v) in rst.count_col.iter().enumerate() {
            out.count_col[i] += *v;
        }
        for (i, v) in rst.sum.iter().enumerate() {
            out.sum[i] += *v;
        }
        for (i, v) in rst.min.iter().enumerate() {
            match (out.min[i], v) {
                (None, Some(x)) => out.min[i] = Some(*x),
                (Some(cur), Some(x)) => {
                    if *x < cur {
                        out.min[i] = Some(*x);
                    }
                }
                _ => {}
            }
        }
        for (i, v) in rst.max.iter().enumerate() {
            match (out.max[i], v) {
                (None, Some(x)) => out.max[i] = Some(*x),
                (Some(cur), Some(x)) => {
                    if *x > cur {
                        out.max[i] = Some(*x);
                    }
                }
                _ => {}
            }
        }
        for (i, v) in rst.avg_sum.iter().enumerate() {
            out.avg_sum[i] += *v;
        }
        for (i, v) in rst.avg_cnt.iter().enumerate() {
            out.avg_cnt[i] += *v;
        }
    }

    // Output
    let mut out_columns: Vec<String> = Vec::with_capacity(outputs.len());
    for sc in &stmt.columns {
        match sc {
            SelectColumn::Column(c) => out_columns.push(split_qual(c).1.to_string()),
            SelectColumn::ColumnAlias { alias, .. } => out_columns.push(alias.clone()),
            SelectColumn::Aggregate { alias, func, column, .. } => {
                if let Some(a) = alias {
                    out_columns.push(a.clone());
                } else {
                    out_columns.push(match (func, column.as_ref()) {
                        (AggregateFunc::Count, None) => "count".to_string(),
                        (AggregateFunc::Count, Some(c)) => format!("count({})", c),
                        (AggregateFunc::Sum, Some(c)) => format!("sum({})", c),
                        (AggregateFunc::Min, Some(c)) => format!("min({})", c),
                        (AggregateFunc::Max, Some(c)) => format!("max({})", c),
                        (AggregateFunc::Avg, Some(c)) => format!("avg({})", c),
                        _ => "agg".to_string(),
                    });
                }
            }
            _ => out_columns.push("expr".to_string()),
        }
    }

    let mut rows: Vec<Vec<Value>> = Vec::with_capacity(by_group.len());
    for (_k, (gvals, agg)) in by_group {
        let mut row: Vec<Value> = Vec::with_capacity(outputs.len());
        for o in &outputs {
            match o {
                OutputExpr::GroupCol { col } => {
                    // group values in order of GROUP BY; map by column name
                    let mut found = None;
                    for (idx, gc) in group_cols.iter().enumerate() {
                        if gc == col {
                            found = Some(idx);
                            break;
                        }
                    }
                    if let Some(i) = found {
                        row.push(gvals.get(i).cloned().unwrap_or(Value::Null));
                    } else {
                        row.push(Value::Null);
                    }
                }
                OutputExpr::Agg { kind, .. } => match kind {
                    AggKind::CountStar => row.push(Value::Int64(agg.count_star)),
                    AggKind::CountCol { idx } => row.push(Value::Int64(*agg.count_col.get(*idx).unwrap_or(&0))),
                    AggKind::Sum { idx } => row.push(Value::Float64(*agg.sum.get(*idx).unwrap_or(&0.0))),
                    AggKind::Min { idx } => row.push(match agg.min.get(*idx).copied().flatten() {
                        Some(x) => Value::Float64(x),
                        None => Value::Null,
                    }),
                    AggKind::Max { idx } => row.push(match agg.max.get(*idx).copied().flatten() {
                        Some(x) => Value::Float64(x),
                        None => Value::Null,
                    }),
                    AggKind::Avg { idx } => {
                        let s = *agg.avg_sum.get(*idx).unwrap_or(&0.0);
                        let c = *agg.avg_cnt.get(*idx).unwrap_or(&0);
                        if c == 0 {
                            row.push(Value::Null);
                        } else {
                            row.push(Value::Float64(s / (c as f64)));
                        }
                    }
                },
            }
        }
        rows.push(row);
    }

    if !stmt.order_by.is_empty() {
        let ob = &stmt.order_by[0];
        let key = split_qual(&ob.column).1;
        let mut idx_opt: Option<usize> = None;
        for (i, c) in out_columns.iter().enumerate() {
            if c == key {
                idx_opt = Some(i);
                break;
            }
        }
        let Some(idx) = idx_opt else {
            return Ok(None);
        };
        rows.sort_by(|a, b| {
            let av = a.get(idx).cloned().unwrap_or(Value::Null);
            let bv = b.get(idx).cloned().unwrap_or(Value::Null);
            let cmp = SqlExecutor::compare_values(Some(&av), Some(&bv), ob.nulls_first);
            if ob.descending { cmp.reverse() } else { cmp }
        });
    }

    if let Some(limit) = stmt.limit {
        if rows.len() > limit {
            rows.truncate(limit);
        }
    }

    Ok(Some(SqlResult::new(out_columns, rows)))
}

#[inline]
fn join_key(v: &crate::data::Value) -> JoinKey {
    use crate::data::Value;
    match v {
        Value::Null => JoinKey::Bytes(Vec::new()),
        Value::Bool(b) => JoinKey::Bool(*b),
        Value::Int8(x) => JoinKey::I64(*x as i64),
        Value::Int16(x) => JoinKey::I64(*x as i64),
        Value::Int32(x) => JoinKey::I64(*x as i64),
        Value::Int64(x) => JoinKey::I64(*x),
        Value::UInt8(x) => JoinKey::U64(*x as u64),
        Value::UInt16(x) => JoinKey::U64(*x as u64),
        Value::UInt32(x) => JoinKey::U64(*x as u64),
        Value::UInt64(x) => JoinKey::U64(*x),
        Value::Float32(x) => JoinKey::F32(x.to_bits()),
        Value::Float64(x) => JoinKey::F64(x.to_bits()),
        Value::String(s) => JoinKey::String(s.clone()),
        Value::Binary(b) => JoinKey::Binary(b.clone()),
        Value::Json(j) => JoinKey::Bytes(j.to_string().into_bytes()),
        Value::Timestamp(t) => JoinKey::I64(*t),
        Value::Date(d) => JoinKey::I64(*d as i64),
        Value::Array(arr) => JoinKey::Bytes(crate::data::Value::Array(arr.clone()).to_bytes()),
    }
}

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
            JoinType::Inner => {
                if let Some(rs) = matches {
                    for &r in rs {
                        joined.push(JoinRow { left: li, right: Some(r) });
                    }
                }
            }
            JoinType::Left => {
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

#[allow(dead_code)]
pub(crate) fn aggregate_join_rows_streaming(
    stmt: &SelectStatement,
    tables: &mut HashMap<String, ColumnTable>,
) -> Result<SqlResult, ApexError> {
    // This function is an optimized aggregation path that avoids materializing the full JoinRow
    // vector. It performs the hash join probe and updates group aggregates on-the-fly.
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

    // Immutable borrows for fast access during join.
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

    // Optional WHERE pushdown (right-only AND chain).
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

    // Build aggregate specs and group columns.
    use crate::data::Value;
    use crate::query::sql_parser::AggregateFunc;
    use std::collections::HashSet;

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

    // Probe join + update group state.
    let left_row_count = left_table.get_row_count();
    let left_deleted = left_table.deleted_ref();

    for li in 0..left_row_count {
        if left_deleted.get(li) {
            continue;
        }
        let lv = get_col_value(left_table, &left_key_col, li);
        let matches = if lv.is_null() { None } else { hash.get(&join_key(&lv)) };

        match join.join_type {
            JoinType::Inner => {
                if let Some(rs) = matches {
                    for &ri in rs {
                        let jr = JoinRow {
                            left: li,
                            right: Some(ri),
                        };
                        if !where_pushed_down {
                            if let Some(ref where_expr) = stmt.where_clause {
                                if !eval_predicate(where_expr, left_table, right_table, &left_alias, &right_alias, jr)? {
                                    continue;
                                }
                            }
                        }

                        let gvals: Vec<Value> = group_cols
                            .iter()
                            .map(|c| value_for_ref(c, left_table, right_table, &left_alias, &right_alias, jr))
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
                                &left_alias,
                                &right_alias,
                                jr,
                            );
                        }
                    }
                }
            }
            JoinType::Left => {
                if let Some(rs) = matches {
                    for &ri in rs {
                        let jr = JoinRow {
                            left: li,
                            right: Some(ri),
                        };
                        if !where_pushed_down {
                            if let Some(ref where_expr) = stmt.where_clause {
                                if !eval_predicate(where_expr, left_table, right_table, &left_alias, &right_alias, jr)? {
                                    continue;
                                }
                            }
                        }

                        let gvals: Vec<Value> = group_cols
                            .iter()
                            .map(|c| value_for_ref(c, left_table, right_table, &left_alias, &right_alias, jr))
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
                                &left_alias,
                                &right_alias,
                                jr,
                            );
                        }
                    }
                } else {
                    let jr = JoinRow { left: li, right: None };
                    if !where_pushed_down {
                        if let Some(ref where_expr) = stmt.where_clause {
                            if !eval_predicate(where_expr, left_table, right_table, &left_alias, &right_alias, jr)? {
                                continue;
                            }
                        }
                    }

                    let gvals: Vec<Value> = group_cols
                        .iter()
                        .map(|c| value_for_ref(c, left_table, right_table, &left_alias, &right_alias, jr))
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
                            &left_alias,
                            &right_alias,
                            jr,
                        );
                    }
                }
            }
        }
    }

    // Materialize aggregated rows (same as aggregate_join_rows).
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
                let (_alias, col) = split_qual(&ob.column);
                let idx = out_columns
                    .iter()
                    .position(|c| c == col || c == &ob.column)
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

    let mut rows = rows;
    if stmt.offset.is_some() || stmt.limit.is_some() {
        let offset = stmt.offset.unwrap_or(0);
        let limit = stmt.limit.unwrap_or(usize::MAX);
        rows = rows.into_iter().skip(offset).take(limit).collect::<Vec<_>>();
    }

    Ok(SqlResult::new(out_columns, rows))
}

#[allow(dead_code)]
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

    let mut groups: HashMap<Vec<JoinKey>, (Vec<Value>, Vec<AggState>)> = HashMap::new();

    for jr in joined {
        let gvals: Vec<Value> = group_cols
            .iter()
            .map(|c| value_for_ref(c, left_table, right_table, &ctx.left_alias, &ctx.right_alias, *jr))
            .collect();
        let gkey = gvals.iter().map(|v| join_key(v)).collect::<Vec<_>>();

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
