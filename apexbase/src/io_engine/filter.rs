//! Streaming filter evaluation for early termination
//!
//! This module provides optimized filter evaluation for streaming queries
//! with early termination support.

use crate::query::{Filter, LikeMatcher, CompareOp};
use crate::table::column_table::{TypedColumn, ColumnSchema};
use crate::data::Value;

/// Streaming filter evaluator for early termination
/// Compiles filter into optimized form for row-by-row evaluation
pub(crate) struct StreamingFilterEvaluator<'a> {
    filter: CompiledFilter<'a>,
}

/// Compiled filter for fast evaluation
pub(crate) enum CompiledFilter<'a> {
    True,
    False,
    CompareInt64 { data: &'a [i64], op: CompareOp, target: i64 },
    CompareFloat64 { data: &'a [f64], op: CompareOp, target: f64 },
    CompareString { data: &'a [String], op: CompareOp, target: String },
    Like { data: &'a [String], matcher: LikeMatcher, negated: bool },
    Range { data: &'a [i64], low: i64, high: i64 },
    And(Vec<CompiledFilter<'a>>),
    Or(Vec<CompiledFilter<'a>>),
    Not(Box<CompiledFilter<'a>>),
    Generic { filter: Filter, schema: &'a ColumnSchema, columns: &'a [TypedColumn] },
}

impl<'a> StreamingFilterEvaluator<'a> {
    pub(crate) fn new(
        filter: &Filter,
        schema: &'a ColumnSchema,
        columns: &'a [TypedColumn],
    ) -> Self {
        Self {
            filter: Self::compile(filter, schema, columns),
        }
    }
    
    fn compile(
        filter: &Filter,
        schema: &'a ColumnSchema,
        columns: &'a [TypedColumn],
    ) -> CompiledFilter<'a> {
        match filter {
            Filter::True => CompiledFilter::True,
            Filter::False => CompiledFilter::False,
            Filter::Compare { field, op, value } => {
                if let Some(col_idx) = schema.get_index(field) {
                    match (&columns[col_idx], value) {
                        (TypedColumn::Int64 { data, .. }, Value::Int64(target)) => {
                            CompiledFilter::CompareInt64 { data, op: op.clone(), target: *target }
                        }
                        (TypedColumn::Float64 { data, .. }, Value::Float64(target)) => {
                            CompiledFilter::CompareFloat64 { data, op: op.clone(), target: *target }
                        }
                        (TypedColumn::String { data, .. }, Value::String(target)) => {
                            CompiledFilter::CompareString { data, op: op.clone(), target: target.clone() }
                        }
                        _ => CompiledFilter::Generic { filter: filter.clone(), schema, columns }
                    }
                } else {
                    CompiledFilter::False
                }
            }
            Filter::Like { field, pattern } => {
                if let Some(col_idx) = schema.get_index(field) {
                    if let TypedColumn::String { data, .. } = &columns[col_idx] {
                        let matcher = LikeMatcher::new(pattern);
                        CompiledFilter::Like { data, matcher, negated: false }
                    } else {
                        CompiledFilter::False
                    }
                } else {
                    CompiledFilter::False
                }
            }
            Filter::Range { field, low, high, .. } => {
                if let Some(col_idx) = schema.get_index(field) {
                    if let (TypedColumn::Int64 { data, .. }, Value::Int64(l), Value::Int64(h)) = 
                        (&columns[col_idx], low, high) {
                        CompiledFilter::Range { data, low: *l, high: *h }
                    } else {
                        CompiledFilter::Generic { filter: filter.clone(), schema, columns }
                    }
                } else {
                    CompiledFilter::False
                }
            }
            Filter::And(filters) => {
                let compiled: Vec<_> = filters.iter()
                    .map(|f| Self::compile(f, schema, columns))
                    .collect();
                CompiledFilter::And(compiled)
            }
            Filter::Or(filters) => {
                let compiled: Vec<_> = filters.iter()
                    .map(|f| Self::compile(f, schema, columns))
                    .collect();
                CompiledFilter::Or(compiled)
            }
            Filter::Not(inner) => {
                let compiled = Self::compile(inner, schema, columns);
                CompiledFilter::Not(Box::new(compiled))
            }
            _ => CompiledFilter::Generic { filter: filter.clone(), schema, columns }
        }
    }
    
    #[inline(always)]
    pub(crate) fn matches(&self, row_idx: usize) -> bool {
        Self::eval(&self.filter, row_idx)
    }
    
    #[inline(always)]
    fn eval(filter: &CompiledFilter, row_idx: usize) -> bool {
        match filter {
            CompiledFilter::True => true,
            CompiledFilter::False => false,
            CompiledFilter::CompareInt64 { data, op, target } => {
                if row_idx >= data.len() { return false; }
                let val = data[row_idx];
                match op {
                    CompareOp::Equal => val == *target,
                    CompareOp::NotEqual => val != *target,
                    CompareOp::LessThan => val < *target,
                    CompareOp::LessEqual => val <= *target,
                    CompareOp::GreaterThan => val > *target,
                    CompareOp::GreaterEqual => val >= *target,
                    _ => false,
                }
            }
            CompiledFilter::CompareFloat64 { data, op, target } => {
                if row_idx >= data.len() { return false; }
                let val = data[row_idx];
                match op {
                    CompareOp::Equal => (val - target).abs() < f64::EPSILON,
                    CompareOp::NotEqual => (val - target).abs() >= f64::EPSILON,
                    CompareOp::LessThan => val < *target,
                    CompareOp::LessEqual => val <= *target,
                    CompareOp::GreaterThan => val > *target,
                    CompareOp::GreaterEqual => val >= *target,
                    _ => false,
                }
            }
            CompiledFilter::CompareString { data, op, target } => {
                if row_idx >= data.len() { return false; }
                let val = &data[row_idx];
                match op {
                    CompareOp::Equal => val == target,
                    CompareOp::NotEqual => val != target,
                    CompareOp::LessThan => val < target,
                    CompareOp::LessEqual => val <= target,
                    CompareOp::GreaterThan => val > target,
                    CompareOp::GreaterEqual => val >= target,
                    _ => false,
                }
            }
            CompiledFilter::Like { data, matcher, negated } => {
                if row_idx >= data.len() { return false; }
                let matches = matcher.matches(&data[row_idx]);
                if *negated { !matches } else { matches }
            }
            CompiledFilter::Range { data, low, high } => {
                if row_idx >= data.len() { return false; }
                let val = data[row_idx];
                val >= *low && val <= *high
            }
            CompiledFilter::And(filters) => filters.iter().all(|f| Self::eval(f, row_idx)),
            CompiledFilter::Or(filters) => filters.iter().any(|f| Self::eval(f, row_idx)),
            CompiledFilter::Not(inner) => !Self::eval(inner, row_idx),
            CompiledFilter::Generic { filter, schema, columns } => {
                // Fallback: build row and use filter.matches()
                let mut row = crate::data::Row::new(row_idx as u64);
                for (name, _) in &schema.columns {
                    if let Some(idx) = schema.get_index(name) {
                        if let Some(val) = columns[idx].get(row_idx) {
                            row.set(name.clone(), val);
                        }
                    }
                }
                filter.matches(&row)
            }
        }
    }
}
