use crate::query::engine::ops::compare::compare_values;
use crate::query::sql_parser::OrderByClause;
use crate::table::ColumnTable;
use crate::ApexError;
use std::cmp::Ordering;

pub(crate) fn sort_indices_by_columns_topk(
    indices: &[usize],
    order_by: &[OrderByClause],
    table: &ColumnTable,
    k: usize,
) -> Result<Vec<usize>, ApexError> {
    if indices.is_empty() || k == 0 {
        return Ok(Vec::new());
    }

    if order_by.is_empty() {
        return Ok(indices.to_vec());
    }

    let mut out = indices.to_vec();
    let schema = table.schema_ref();
    let cols = table.columns_ref();

    out.sort_by(|&a, &b| {
        for ob in order_by {
            let idx = if ob.column == "_id" {
                None
            } else {
                schema.get_index(&ob.column)
            };

            let av = idx.and_then(|i| cols[i].get(a)).or_else(|| {
                if ob.column == "_id" {
                    Some(crate::data::Value::Int64(a as i64))
                } else {
                    None
                }
            });
            let bv = idx.and_then(|i| cols[i].get(b)).or_else(|| {
                if ob.column == "_id" {
                    Some(crate::data::Value::Int64(b as i64))
                } else {
                    None
                }
            });

            let cmp = compare_values(av.as_ref(), bv.as_ref(), ob.nulls_first);
            let cmp = if ob.descending { cmp.reverse() } else { cmp };
            if cmp != Ordering::Equal {
                return cmp;
            }
        }
        Ordering::Equal
    });

    out.truncate(k.min(out.len()));
    Ok(out)
}
