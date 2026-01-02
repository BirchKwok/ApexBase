//! Table implementation

use super::Schema;
use crate::data::{DataType, Row};
use crate::query::{Filter, QueryExecutor};
use crate::Result;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// A table in the database
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Table {
    /// Table ID
    id: u32,
    /// Table name
    name: String,
    /// Schema
    schema: Schema,
    /// Rows (indexed by ID)
    rows: HashMap<u64, Row>,
    /// Next row ID
    next_row_id: u64,
    /// Deleted row IDs (for reuse)
    deleted_ids: Vec<u64>,
}

impl Table {
    /// Create a new table
    pub fn new(id: u32, name: &str) -> Self {
        Self {
            id,
            name: name.to_string(),
            schema: Schema::new(),
            rows: HashMap::new(),
            next_row_id: 1,
            deleted_ids: Vec::new(),
        }
    }

    /// Get table ID
    pub fn id(&self) -> u32 {
        self.id
    }

    /// Get table name
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Get schema
    pub fn schema(&self) -> &Schema {
        &self.schema
    }

    /// Get row count
    pub fn row_count(&self) -> u64 {
        self.rows.len() as u64
    }

    /// Get next row ID (without incrementing)
    fn allocate_row_id(&mut self) -> u64 {
        // Reuse deleted IDs if available
        if let Some(id) = self.deleted_ids.pop() {
            id
        } else {
            let id = self.next_row_id;
            self.next_row_id += 1;
            id
        }
    }

    /// Insert a row
    pub fn insert(&mut self, mut row: Row) -> Result<u64> {
        let id = self.allocate_row_id();
        row.id = id;

        // Update schema from data
        self.schema.update_from_data(&row.fields);

        self.rows.insert(id, row);
        Ok(id)
    }

    /// Insert multiple rows
    pub fn insert_batch(&mut self, rows: Vec<Row>) -> Result<Vec<u64>> {
        let mut ids = Vec::with_capacity(rows.len());

        for mut row in rows {
            let id = self.allocate_row_id();
            row.id = id;
            self.schema.update_from_data(&row.fields);
            self.rows.insert(id, row);
            ids.push(id);
        }

        Ok(ids)
    }

    /// Get a row by ID
    pub fn get(&self, id: u64) -> Option<Row> {
        self.rows.get(&id).cloned()
    }

    /// Get multiple rows by IDs
    pub fn get_many(&self, ids: &[u64]) -> Vec<Row> {
        ids.iter()
            .filter_map(|id| self.rows.get(id).cloned())
            .collect()
    }

    /// Get all rows
    pub fn get_all(&self) -> Vec<Row> {
        self.rows.values().cloned().collect()
    }

    /// Delete a row
    pub fn delete(&mut self, id: u64) -> bool {
        if self.rows.remove(&id).is_some() {
            self.deleted_ids.push(id);
            true
        } else {
            false
        }
    }

    /// Delete multiple rows
    pub fn delete_batch(&mut self, ids: &[u64]) -> bool {
        let mut all_deleted = true;
        for &id in ids {
            if self.rows.remove(&id).is_some() {
                self.deleted_ids.push(id);
            } else {
                all_deleted = false;
            }
        }
        all_deleted
    }

    /// Update a row
    pub fn update(&mut self, id: u64, mut row: Row) -> bool {
        if self.rows.contains_key(&id) {
            row.id = id;
            self.schema.update_from_data(&row.fields);
            self.rows.insert(id, row);
            true
        } else {
            false
        }
    }

    /// Update multiple rows
    pub fn update_batch(&mut self, updates: HashMap<u64, Row>) -> Vec<u64> {
        let mut success_ids = Vec::new();
        for (id, mut row) in updates {
            if self.rows.contains_key(&id) {
                row.id = id;
                self.schema.update_from_data(&row.fields);
                self.rows.insert(id, row);
                success_ids.push(id);
            }
        }
        success_ids
    }

    /// Query rows
    pub fn query(&self, where_clause: &str) -> Result<Vec<Row>> {
        let executor = QueryExecutor::new();
        let filter = executor.parse(where_clause)?;

        // Fast path for Filter::True (full table scan without filter)
        if matches!(filter, Filter::True) {
            // Direct collect without sorting - HashMap iteration is fast
            // For most practical sizes, sequential is faster due to allocation patterns
            Ok(self.rows.values().cloned().collect())
        } else {
            // Filter rows - sequential is often faster for moderate sizes
            // due to branch prediction and cache locality
            let len = self.rows.len();
            if len > 50000 {
                // Only use parallel for very large datasets
                Ok(self.rows
                    .values()
                    .par_bridge()
                    .filter(|row| filter.matches(row))
                    .cloned()
                    .collect())
            } else {
                Ok(self.rows
                    .values()
                    .filter(|row| filter.matches(row))
                    .cloned()
                    .collect())
            }
            }
    }
    
    /// Query rows with sorting (when order matters)
    pub fn query_sorted(&self, where_clause: &str) -> Result<Vec<Row>> {
        let mut results = self.query(where_clause)?;

        // Sort by ID (parallel sort for large results)
        if results.len() > 1000 {
            results.par_sort_by_key(|r| r.id);
        } else {
            results.sort_by_key(|r| r.id);
        }

        Ok(results)
    }

    /// Add a column
    pub fn add_column(&mut self, name: &str, data_type: DataType) -> Result<()> {
        self.schema.add_column(name, data_type)
    }

    /// Drop a column
    pub fn drop_column(&mut self, name: &str) -> Result<()> {
        self.schema.remove_column(name)?;

        // Remove the column from all rows
        for row in self.rows.values_mut() {
            row.remove(name);
        }

        Ok(())
    }

    /// Rename a column
    pub fn rename_column(&mut self, old_name: &str, new_name: &str) -> Result<()> {
        self.schema.rename_column(old_name, new_name)?;

        // Rename the column in all rows
        for row in self.rows.values_mut() {
            if let Some(value) = row.remove(old_name) {
                row.set(new_name, value);
            }
        }

        Ok(())
    }

    /// Compact the table (reclaim deleted IDs space)
    pub fn compact(&mut self) {
        self.deleted_ids.clear();
        // Optionally renumber rows if fragmentation is high
    }

    /// Check if table contains a row
    pub fn contains(&self, id: u64) -> bool {
        self.rows.contains_key(&id)
    }

    /// Iterate over rows
    pub fn iter(&self) -> impl Iterator<Item = (&u64, &Row)> {
        self.rows.iter()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::Value;

    #[test]
    fn test_table_operations() {
        let mut table = Table::new(1, "test");

        // Insert
        let mut row1 = Row::new(0);
        row1.set("name", "John");
        row1.set("age", Value::Int64(30));
        let id1 = table.insert(row1).unwrap();
        assert_eq!(id1, 1);

        let mut row2 = Row::new(0);
        row2.set("name", "Jane");
        row2.set("age", Value::Int64(25));
        let id2 = table.insert(row2).unwrap();
        assert_eq!(id2, 2);

        assert_eq!(table.row_count(), 2);

        // Get
        let retrieved = table.get(id1).unwrap();
        assert_eq!(retrieved.get("name"), Some(&Value::String("John".to_string())));

        // Update
        let mut updated = Row::new(0);
        updated.set("name", "John Doe");
        updated.set("age", Value::Int64(31));
        assert!(table.update(id1, updated));

        let retrieved = table.get(id1).unwrap();
        assert_eq!(retrieved.get("name"), Some(&Value::String("John Doe".to_string())));

        // Delete
        assert!(table.delete(id1));
        assert!(table.get(id1).is_none());
        assert_eq!(table.row_count(), 1);
    }

    #[test]
    fn test_table_query() {
        let mut table = Table::new(1, "test");

        for i in 0..10 {
            let mut row = Row::new(0);
            row.set("value", Value::Int64(i));
            table.insert(row).unwrap();
        }

        let results = table.query("value > 5").unwrap();
        assert_eq!(results.len(), 4); // 6, 7, 8, 9
    }

    #[test]
    fn test_table_schema_updates() {
        let mut table = Table::new(1, "test");

        let mut row = Row::new(0);
        row.set("name", "John");
        table.insert(row).unwrap();

        assert!(table.schema().has_column("name"));
        assert!(table.schema().has_column("_id"));

        // Add column via data
        let mut row2 = Row::new(0);
        row2.set("name", "Jane");
        row2.set("email", "jane@example.com");
        table.insert(row2).unwrap();

        assert!(table.schema().has_column("email"));
    }
}

