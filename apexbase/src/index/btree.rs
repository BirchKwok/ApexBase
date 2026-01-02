//! B+Tree index implementation

use super::RowId;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

/// B+Tree index for fast lookups
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BTreeIndex {
    /// Index name
    name: String,
    /// The actual B-tree (using std's BTreeMap for simplicity)
    tree: BTreeMap<Vec<u8>, Vec<RowId>>,
}

impl BTreeIndex {
    /// Create a new index
    pub fn new(name: &str) -> Self {
        Self {
            name: name.to_string(),
            tree: BTreeMap::new(),
        }
    }

    /// Get index name
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Insert a key-value pair
    pub fn insert(&mut self, key: &[u8], row_id: RowId) {
        self.tree
            .entry(key.to_vec())
            .or_insert_with(Vec::new)
            .push(row_id);
    }

    /// Delete a key-value pair
    pub fn delete(&mut self, key: &[u8], row_id: RowId) {
        if let Some(ids) = self.tree.get_mut(key) {
            ids.retain(|&id| id != row_id);
            if ids.is_empty() {
                self.tree.remove(key);
            }
        }
    }

    /// Search for an exact key
    pub fn search(&self, key: &[u8]) -> Vec<RowId> {
        self.tree.get(key).cloned().unwrap_or_default()
    }

    /// Range search (inclusive bounds)
    pub fn range(&self, start: &[u8], end: &[u8]) -> Vec<RowId> {
        self.tree
            .range(start.to_vec()..=end.to_vec())
            .flat_map(|(_, ids)| ids.iter().copied())
            .collect()
    }

    /// Get all row IDs
    pub fn all(&self) -> Vec<RowId> {
        self.tree.values().flat_map(|ids| ids.iter().copied()).collect()
    }

    /// Get the number of keys
    pub fn len(&self) -> usize {
        self.tree.len()
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.tree.is_empty()
    }

    /// Clear the index
    pub fn clear(&mut self) {
        self.tree.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_btree_operations() {
        let mut index = BTreeIndex::new("test_idx");

        // Insert
        index.insert(b"key1", 1);
        index.insert(b"key1", 2);
        index.insert(b"key2", 3);

        // Search
        let ids = index.search(b"key1");
        assert_eq!(ids, vec![1, 2]);

        let ids = index.search(b"key2");
        assert_eq!(ids, vec![3]);

        // Delete
        index.delete(b"key1", 1);
        let ids = index.search(b"key1");
        assert_eq!(ids, vec![2]);
    }

    #[test]
    fn test_range_search() {
        let mut index = BTreeIndex::new("test_idx");

        index.insert(b"a", 1);
        index.insert(b"b", 2);
        index.insert(b"c", 3);
        index.insert(b"d", 4);

        let ids = index.range(b"b", b"c");
        assert_eq!(ids, vec![2, 3]);
    }
}

