use crate::data::Value;

pub(crate) fn apply_distinct(rows: Vec<Vec<Value>>) -> Vec<Vec<Value>> {
    let mut seen = std::collections::HashSet::new();
    let mut result = Vec::new();

    for row in rows {
        let key = format!("{:?}", row);
        if seen.insert(key) {
            result.push(row);
        }
    }

    result
}
