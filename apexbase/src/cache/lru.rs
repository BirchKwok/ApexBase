//! LRU cache implementation

use std::collections::HashMap;
use std::hash::Hash;

/// LRU cache entry
struct CacheEntry<V> {
    value: V,
    prev: Option<usize>,
    next: Option<usize>,
}

/// LRU cache
pub struct LRUCache<K, V> {
    capacity: usize,
    map: HashMap<K, usize>,
    entries: Vec<Option<CacheEntry<V>>>,
    free_list: Vec<usize>,
    head: Option<usize>,
    tail: Option<usize>,
}

impl<K: Eq + Hash + Clone, V> LRUCache<K, V> {
    /// Create a new LRU cache with the given capacity
    pub fn new(capacity: usize) -> Self {
        Self {
            capacity,
            map: HashMap::with_capacity(capacity),
            entries: Vec::with_capacity(capacity),
            free_list: Vec::new(),
            head: None,
            tail: None,
        }
    }

    /// Get a value from the cache
    pub fn get(&mut self, key: &K) -> Option<&V> {
        if let Some(&index) = self.map.get(key) {
            self.move_to_front(index);
            self.entries[index].as_ref().map(|e| &e.value)
        } else {
            None
        }
    }

    /// Put a value into the cache
    pub fn put(&mut self, key: K, value: V) {
        if let Some(&index) = self.map.get(&key) {
            // Update existing entry
            if let Some(entry) = &mut self.entries[index] {
                entry.value = value;
            }
            self.move_to_front(index);
        } else {
            // Insert new entry
            if self.map.len() >= self.capacity {
                self.evict();
            }

            let index = self.allocate_entry(value);
            self.map.insert(key, index);
            self.push_front(index);
        }
    }

    /// Remove a value from the cache
    pub fn remove(&mut self, key: &K) -> Option<V> {
        if let Some(index) = self.map.remove(key) {
            self.unlink(index);
            let entry = self.entries[index].take();
            self.free_list.push(index);
            entry.map(|e| e.value)
        } else {
            None
        }
    }

    /// Check if the cache contains a key
    pub fn contains(&self, key: &K) -> bool {
        self.map.contains_key(key)
    }

    /// Get the current size
    pub fn len(&self) -> usize {
        self.map.len()
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.map.is_empty()
    }

    /// Clear the cache
    pub fn clear(&mut self) {
        self.map.clear();
        self.entries.clear();
        self.free_list.clear();
        self.head = None;
        self.tail = None;
    }

    /// Allocate an entry
    fn allocate_entry(&mut self, value: V) -> usize {
        if let Some(index) = self.free_list.pop() {
            self.entries[index] = Some(CacheEntry {
                value,
                prev: None,
                next: None,
            });
            index
        } else {
            let index = self.entries.len();
            self.entries.push(Some(CacheEntry {
                value,
                prev: None,
                next: None,
            }));
            index
        }
    }

    /// Push an entry to the front of the list
    fn push_front(&mut self, index: usize) {
        if let Some(entry) = &mut self.entries[index] {
            entry.prev = None;
            entry.next = self.head;
        }

        if let Some(old_head) = self.head {
            if let Some(entry) = &mut self.entries[old_head] {
                entry.prev = Some(index);
            }
        }

        self.head = Some(index);

        if self.tail.is_none() {
            self.tail = Some(index);
        }
    }

    /// Unlink an entry from the list
    fn unlink(&mut self, index: usize) {
        let (prev, next) = if let Some(entry) = &self.entries[index] {
            (entry.prev, entry.next)
        } else {
            return;
        };

        if let Some(prev_index) = prev {
            if let Some(entry) = &mut self.entries[prev_index] {
                entry.next = next;
            }
        } else {
            self.head = next;
        }

        if let Some(next_index) = next {
            if let Some(entry) = &mut self.entries[next_index] {
                entry.prev = prev;
            }
        } else {
            self.tail = prev;
        }
    }

    /// Move an entry to the front
    fn move_to_front(&mut self, index: usize) {
        if self.head == Some(index) {
            return;
        }

        self.unlink(index);
        self.push_front(index);
    }

    /// Evict the least recently used entry
    fn evict(&mut self) {
        if let Some(tail_index) = self.tail {
            // Find and remove the key
            let key_to_remove = self.map.iter()
                .find(|(_, &v)| v == tail_index)
                .map(|(k, _)| k.clone());

            if let Some(key) = key_to_remove {
                self.remove(&key);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lru_basic() {
        let mut cache = LRUCache::new(3);

        cache.put("a", 1);
        cache.put("b", 2);
        cache.put("c", 3);

        assert_eq!(cache.get(&"a"), Some(&1));
        assert_eq!(cache.get(&"b"), Some(&2));
        assert_eq!(cache.get(&"c"), Some(&3));
    }

    #[test]
    fn test_lru_eviction() {
        let mut cache = LRUCache::new(2);

        cache.put("a", 1);
        cache.put("b", 2);

        // Access "a" to make it recently used
        cache.get(&"a");

        // Add "c", should evict "b"
        cache.put("c", 3);

        assert_eq!(cache.get(&"a"), Some(&1));
        assert_eq!(cache.get(&"b"), None);
        assert_eq!(cache.get(&"c"), Some(&3));
    }

    #[test]
    fn test_lru_update() {
        let mut cache = LRUCache::new(2);

        cache.put("a", 1);
        cache.put("a", 2);

        assert_eq!(cache.get(&"a"), Some(&2));
        assert_eq!(cache.len(), 1);
    }
}

