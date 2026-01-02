//! Core storage engine implementation

use super::{ApexFile, HEADER_SIZE};
use crate::data::{DataType, Row};
use crate::table::{TableCatalog, TableEntry};
use crate::table::table::Table;
use crate::{ApexError, Result};
use parking_lot::RwLock;
use std::collections::HashMap;
use std::path::Path;

/// The main storage engine
pub struct ApexStorageEngine {
    /// Underlying file
    file: RwLock<ApexFile>,
    /// Table catalog
    catalog: RwLock<TableCatalog>,
    /// Current table name
    current_table: RwLock<String>,
    /// In-memory table data (will be persisted)
    tables: RwLock<HashMap<String, Table>>,
    /// Write cache
    write_cache: RwLock<Vec<(String, Row)>>,
    /// Cache size limit
    cache_size: usize,
}

impl ApexStorageEngine {
    /// Create a new storage engine
    pub fn create(path: &Path) -> Result<Self> {
        let file = ApexFile::create(path)?;
        let catalog = TableCatalog::new();

        let engine = Self {
            file: RwLock::new(file),
            catalog: RwLock::new(catalog),
            current_table: RwLock::new("default".to_string()),
            tables: RwLock::new(HashMap::new()),
            write_cache: RwLock::new(Vec::new()),
            cache_size: 10000,
        };

        // Create default table
        engine.create_table_internal("default")?;

        Ok(engine)
    }

    /// Open an existing storage engine
    pub fn open(path: &Path) -> Result<Self> {
        let file = ApexFile::open(path)?;
        
        // Load catalog from file
        let catalog = Self::load_catalog(&file)?;
        
        // Load tables
        let tables = Self::load_tables(&file, &catalog)?;

        let engine = Self {
            file: RwLock::new(file),
            catalog: RwLock::new(catalog),
            current_table: RwLock::new("default".to_string()),
            tables: RwLock::new(tables),
            write_cache: RwLock::new(Vec::new()),
            cache_size: 10000,
        };

        Ok(engine)
    }

    /// Load catalog from file
    fn load_catalog(file: &ApexFile) -> Result<TableCatalog> {
        let offset = file.header().table_catalog_offset;
        if offset == 0 || offset == HEADER_SIZE as u64 {
            return Ok(TableCatalog::new());
        }

        // Read catalog data
        let len_bytes = file.read_bytes(offset, 8)?;
        let len = u64::from_le_bytes(len_bytes.try_into().unwrap()) as usize;
        
        if len == 0 {
            return Ok(TableCatalog::new());
        }

        let data = file.read_bytes(offset + 8, len)?;
        bincode::deserialize(&data).map_err(|e| ApexError::SerializationError(e.to_string()))
    }

    /// Load tables from file
    fn load_tables(file: &ApexFile, catalog: &TableCatalog) -> Result<HashMap<String, Table>> {
        let mut tables = HashMap::new();

        for entry in catalog.tables() {
            let table = Self::load_table(file, entry)?;
            tables.insert(entry.name.clone(), table);
        }

        Ok(tables)
    }

    /// Load a single table
    fn load_table(file: &ApexFile, entry: &TableEntry) -> Result<Table> {
        if entry.data_offset == 0 {
            return Ok(Table::new(entry.id, &entry.name));
        }

        // Read table data
        let len_bytes = file.read_bytes(entry.data_offset, 8)?;
        let len = u64::from_le_bytes(len_bytes.try_into().unwrap()) as usize;

        if len == 0 {
            return Ok(Table::new(entry.id, &entry.name));
        }

        let data = file.read_bytes(entry.data_offset + 8, len)?;
        bincode::deserialize(&data).map_err(|e| ApexError::SerializationError(e.to_string()))
    }

    /// Save catalog to file
    fn save_catalog(&self) -> Result<()> {
        let catalog = self.catalog.read();
        let data = bincode::serialize(&*catalog)
            .map_err(|e| ApexError::SerializationError(e.to_string()))?;

        let mut file = self.file.write();
        let offset = file.header().table_catalog_offset;

        // Write length prefix + data
        let len_bytes = (data.len() as u64).to_le_bytes();
        file.write_bytes(offset, &len_bytes)?;
        file.write_bytes(offset + 8, &data)?;

        Ok(())
    }

    /// Save a table to file
    fn save_table(&self, name: &str) -> Result<()> {
        let tables = self.tables.read();
        let table = tables.get(name).ok_or_else(|| ApexError::TableNotFound(name.to_string()))?;

        let data = bincode::serialize(table)
            .map_err(|e| ApexError::SerializationError(e.to_string()))?;

        let mut catalog = self.catalog.write();
        let entry = catalog.get_mut(name).ok_or_else(|| ApexError::TableNotFound(name.to_string()))?;

        let mut file = self.file.write();
        
        // Allocate space if needed
        if entry.data_offset == 0 {
            let page_id = file.allocate_page()?;
            entry.data_offset = HEADER_SIZE as u64 + (page_id as u64) * file.page_size() as u64;
        }

        // Write length prefix + data
        let len_bytes = (data.len() as u64).to_le_bytes();
        file.write_bytes(entry.data_offset, &len_bytes)?;
        file.write_bytes(entry.data_offset + 8, &data)?;

        entry.row_count = table.row_count();

        Ok(())
    }

    /// Internal create table
    fn create_table_internal(&self, name: &str) -> Result<()> {
        let mut catalog = self.catalog.write();
        if catalog.contains(name) {
            return Err(ApexError::TableExists(name.to_string()));
        }

        let table_id = catalog.next_id();
        let entry = TableEntry::new(table_id, name);
        catalog.add(entry);

        let table = Table::new(table_id, name);
        self.tables.write().insert(name.to_string(), table);

        Ok(())
    }

    // ============ Public API ============

    /// Use a table
    pub fn use_table(&self, name: &str) -> Result<()> {
        let catalog = self.catalog.read();
        if !catalog.contains(name) {
            return Err(ApexError::TableNotFound(name.to_string()));
        }
        *self.current_table.write() = name.to_string();
        Ok(())
    }

    /// Get current table name
    pub fn current_table(&self) -> String {
        self.current_table.read().clone()
    }

    /// Create a new table
    pub fn create_table(&self, name: &str) -> Result<()> {
        self.create_table_internal(name)?;
        *self.current_table.write() = name.to_string();
        Ok(())
    }

    /// Drop a table
    pub fn drop_table(&self, name: &str) -> Result<()> {
        if name == "default" {
            return Err(ApexError::CannotDropDefaultTable);
        }

        let mut catalog = self.catalog.write();
        if !catalog.contains(name) {
            return Ok(());
        }

        catalog.remove(name);
        self.tables.write().remove(name);

        // If current table was dropped, switch to default
        if *self.current_table.read() == name {
            *self.current_table.write() = "default".to_string();
        }

        Ok(())
    }

    /// List all tables
    pub fn list_tables(&self) -> Vec<String> {
        self.catalog.read().table_names()
    }

    /// Store a single row
    pub fn store(&self, row: Row) -> Result<u64> {
        let table_name = self.current_table();
        let mut tables = self.tables.write();
        let table = tables.get_mut(&table_name)
            .ok_or_else(|| ApexError::TableNotFound(table_name.clone()))?;

        let id = table.insert(row)?;
        
        // Update catalog
        if let Some(entry) = self.catalog.write().get_mut(&table_name) {
            entry.row_count = table.row_count();
        }

        Ok(id)
    }

    /// Store multiple rows
    pub fn store_batch(&self, rows: Vec<Row>) -> Result<Vec<u64>> {
        let table_name = self.current_table();
        let mut tables = self.tables.write();
        let table = tables.get_mut(&table_name)
            .ok_or_else(|| ApexError::TableNotFound(table_name.clone()))?;

        let ids = table.insert_batch(rows)?;

        // Update catalog
        if let Some(entry) = self.catalog.write().get_mut(&table_name) {
            entry.row_count = table.row_count();
        }

        Ok(ids)
    }

    /// Retrieve a row by ID
    pub fn retrieve(&self, id: u64) -> Result<Option<Row>> {
        let table_name = self.current_table();
        let tables = self.tables.read();
        let table = tables.get(&table_name)
            .ok_or_else(|| ApexError::TableNotFound(table_name))?;

        Ok(table.get(id))
    }

    /// Retrieve multiple rows
    pub fn retrieve_many(&self, ids: &[u64]) -> Result<Vec<Row>> {
        let table_name = self.current_table();
        let tables = self.tables.read();
        let table = tables.get(&table_name)
            .ok_or_else(|| ApexError::TableNotFound(table_name))?;

        Ok(table.get_many(ids))
    }

    /// Delete a row
    pub fn delete(&self, id: u64) -> Result<bool> {
        let table_name = self.current_table();
        let mut tables = self.tables.write();
        let table = tables.get_mut(&table_name)
            .ok_or_else(|| ApexError::TableNotFound(table_name.clone()))?;

        let result = table.delete(id);

        // Update catalog
        if let Some(entry) = self.catalog.write().get_mut(&table_name) {
            entry.row_count = table.row_count();
        }

        Ok(result)
    }

    /// Delete multiple rows
    pub fn delete_batch(&self, ids: &[u64]) -> Result<bool> {
        let table_name = self.current_table();
        let mut tables = self.tables.write();
        let table = tables.get_mut(&table_name)
            .ok_or_else(|| ApexError::TableNotFound(table_name.clone()))?;

        let result = table.delete_batch(ids);

        // Update catalog
        if let Some(entry) = self.catalog.write().get_mut(&table_name) {
            entry.row_count = table.row_count();
        }

        Ok(result)
    }

    /// Replace a row
    pub fn replace(&self, id: u64, row: Row) -> Result<bool> {
        let table_name = self.current_table();
        let mut tables = self.tables.write();
        let table = tables.get_mut(&table_name)
            .ok_or_else(|| ApexError::TableNotFound(table_name))?;

        Ok(table.update(id, row))
    }

    /// Query rows
    pub fn query(&self, where_clause: &str) -> Result<Vec<Row>> {
        let table_name = self.current_table();
        let tables = self.tables.read();
        let table = tables.get(&table_name)
            .ok_or_else(|| ApexError::TableNotFound(table_name))?;

        table.query(where_clause)
    }

    /// Count rows
    pub fn count(&self) -> u64 {
        let table_name = self.current_table();
        let tables = self.tables.read();
        tables.get(&table_name).map(|t| t.row_count()).unwrap_or(0)
    }

    /// List fields in current table
    pub fn list_fields(&self) -> Vec<String> {
        let table_name = self.current_table();
        let tables = self.tables.read();
        tables.get(&table_name)
            .map(|t| t.schema().column_names())
            .unwrap_or_default()
    }

    /// Add a column
    pub fn add_column(&self, name: &str, dtype: &str) -> Result<()> {
        let table_name = self.current_table();
        let mut tables = self.tables.write();
        let table = tables.get_mut(&table_name)
            .ok_or_else(|| ApexError::TableNotFound(table_name))?;

        let data_type = DataType::from_sql_type(dtype);
        table.add_column(name, data_type)
    }

    /// Drop a column
    pub fn drop_column(&self, name: &str) -> Result<()> {
        if name == "_id" {
            return Err(ApexError::CannotModifyIdColumn);
        }

        let table_name = self.current_table();
        let mut tables = self.tables.write();
        let table = tables.get_mut(&table_name)
            .ok_or_else(|| ApexError::TableNotFound(table_name))?;

        table.drop_column(name)
    }

    /// Rename a column
    pub fn rename_column(&self, old_name: &str, new_name: &str) -> Result<()> {
        if old_name == "_id" {
            return Err(ApexError::CannotModifyIdColumn);
        }

        let table_name = self.current_table();
        let mut tables = self.tables.write();
        let table = tables.get_mut(&table_name)
            .ok_or_else(|| ApexError::TableNotFound(table_name))?;

        table.rename_column(old_name, new_name)
    }

    /// Get column data type
    pub fn get_column_dtype(&self, name: &str) -> Result<String> {
        let table_name = self.current_table();
        let tables = self.tables.read();
        let table = tables.get(&table_name)
            .ok_or_else(|| ApexError::TableNotFound(table_name))?;

        table.schema().get_column(name)
            .map(|col| col.data_type.to_sql_type().to_string())
            .ok_or_else(|| ApexError::ColumnNotFound(name.to_string()))
    }

    /// Flush all changes to disk
    pub fn flush(&self) -> Result<()> {
        // Save all tables
        let table_names: Vec<String> = self.tables.read().keys().cloned().collect();
        for name in table_names {
            self.save_table(&name)?;
        }

        // Save catalog
        self.save_catalog()?;

        // Flush file
        self.file.write().flush()?;

        Ok(())
    }

    /// Optimize storage
    pub fn optimize(&self) -> Result<()> {
        // Compact tables
        let table_names: Vec<String> = self.tables.read().keys().cloned().collect();
        for name in table_names {
            let mut tables = self.tables.write();
            if let Some(table) = tables.get_mut(&name) {
                table.compact();
            }
        }

        // Flush
        self.flush()?;

        Ok(())
    }

    /// Close the storage engine
    pub fn close(&self) -> Result<()> {
        self.flush()?;
        self.file.write().sync()?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::Value;
    use tempfile::tempdir;

    #[test]
    fn test_create_and_open() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test.apex");

        // Create
        {
            let engine = ApexStorageEngine::create(&path).unwrap();
            assert_eq!(engine.list_tables(), vec!["default"]);
            engine.close().unwrap();
        }

        // Open
        {
            let engine = ApexStorageEngine::open(&path).unwrap();
            assert_eq!(engine.list_tables(), vec!["default"]);
        }
    }

    #[test]
    fn test_store_and_retrieve() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test.apex");

        let engine = ApexStorageEngine::create(&path).unwrap();

        let mut row = Row::new(0);
        row.set("name", "John");
        row.set("age", Value::Int64(30));

        let id = engine.store(row).unwrap();
        assert_eq!(id, 1);

        let retrieved = engine.retrieve(id).unwrap().unwrap();
        assert_eq!(retrieved.get("name"), Some(&Value::String("John".to_string())));
        assert_eq!(retrieved.get("age"), Some(&Value::Int64(30)));
    }

    #[test]
    fn test_table_operations() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test.apex");

        let engine = ApexStorageEngine::create(&path).unwrap();

        engine.create_table("users").unwrap();
        assert_eq!(engine.current_table(), "users");

        let tables = engine.list_tables();
        assert!(tables.contains(&"default".to_string()));
        assert!(tables.contains(&"users".to_string()));

        engine.drop_table("users").unwrap();
        assert_eq!(engine.current_table(), "default");
    }
}

