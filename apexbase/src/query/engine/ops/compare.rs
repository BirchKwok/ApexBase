use crate::data::Value;
use std::cmp::Ordering;

pub(crate) fn compare_values(a: Option<&Value>, b: Option<&Value>, nulls_first: Option<bool>) -> Ordering {
    match (a, b) {
        (None, None) => Ordering::Equal,
        (None, Some(_)) => {
            if nulls_first.unwrap_or(false) {
                Ordering::Less
            } else {
                Ordering::Greater
            }
        }
        (Some(_), None) => {
            if nulls_first.unwrap_or(false) {
                Ordering::Greater
            } else {
                Ordering::Less
            }
        }
        (Some(Value::Null), Some(Value::Null)) => Ordering::Equal,
        (Some(Value::Null), Some(_)) => {
            if nulls_first.unwrap_or(false) {
                Ordering::Less
            } else {
                Ordering::Greater
            }
        }
        (Some(_), Some(Value::Null)) => {
            if nulls_first.unwrap_or(false) {
                Ordering::Greater
            } else {
                Ordering::Less
            }
        }
        (Some(av), Some(bv)) => compare_non_null(av, bv),
    }
}

pub(crate) fn compare_non_null(a: &Value, b: &Value) -> Ordering {
    match (a, b) {
        (Value::Int64(x), Value::Int64(y)) => x.cmp(y),
        (Value::Float64(x), Value::Float64(y)) => x.partial_cmp(y).unwrap_or(Ordering::Equal),
        (Value::String(x), Value::String(y)) => x.cmp(y),
        (Value::Bool(x), Value::Bool(y)) => x.cmp(y),
        _ => Ordering::Equal,
    }
}
