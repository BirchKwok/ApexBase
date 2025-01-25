import sqlite3
import orjson
from typing import Dict, List, Any, Optional
from pathlib import Path
from .limited_dict import LimitedDict
import json
import threading
import time


class Storage:
    """
    A multi-table storage class using SQLite as the backend.
    """
    def __init__(self, filepath=None, batch_size: int = 1000):
        """
        Initializes the Storage class.

        Parameters:
            filepath: str
                The file path for storage
            batch_size: int
                The size of batch operations
        """
        if filepath is None:
            raise ValueError("You must provide a file path.")

        self.filepath = Path(filepath)
        self.filepath.parent.mkdir(parents=True, exist_ok=True)

        self.batch_size = batch_size
        self._field_cache = LimitedDict(100)
        self._lock = threading.Lock()
        self.current_table = "default"
        self.auto_update_fts = False

        self.conn = sqlite3.connect(str(self.filepath), 
                                  isolation_level=None,
                                  check_same_thread=False)
        
        cursor = self.conn.cursor()
        cursor.execute("PRAGMA journal_mode=WAL")
        cursor.execute("PRAGMA wal_autocheckpoint=1000")
        cursor.execute("PRAGMA synchronous=NORMAL")
        cursor.execute("PRAGMA cache_size=-262144")
        cursor.execute("PRAGMA temp_store=MEMORY")
        cursor.execute("PRAGMA mmap_size=1099511627776")
        cursor.execute("PRAGMA page_size=32768")
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.execute("PRAGMA busy_timeout=300000")
        
        self._initialize_database()

    def _initialize_database(self):
        """
        Initializes the SQLite database, creating necessary system tables.
        """
        cursor = self.conn.cursor()
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS tables_meta (
                table_name TEXT PRIMARY KEY,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        if not self._table_exists("default"):
            self.create_table("default")

    def _get_table_name(self, table_name: str = None) -> str:
        """
        Gets the actual table name.

        Parameters:
            table_name: str
                The table name, or None to use the current table

        Returns:
            str: The actual table name
        """
        return table_name if table_name is not None else self.current_table

    def use_table(self, table_name: str):
        """
        Switches the current table.

        Parameters:
            table_name: str
                The table name to switch to
        """
        if not self._table_exists(table_name):
            raise ValueError(f"Table '{table_name}' does not exist")
        self.current_table = table_name
        self._invalidate_cache()

    def create_table(self, table_name: str):
        """
        Creates a new table.

        Parameters:
            table_name: str
                The table name to create
        """
        if self._table_exists(table_name):
            return

        cursor = self.conn.cursor()
        cursor.execute("BEGIN IMMEDIATE")
        try:
            cursor.execute(f"""
                CREATE TABLE {self._quote_identifier(table_name)} (
                    _id INTEGER PRIMARY KEY AUTOINCREMENT
                )
            """)
            
            cursor.execute(f"""
                CREATE TABLE {self._quote_identifier(table_name + '_fields_meta')} (
                    field_name TEXT PRIMARY KEY,
                    field_type TEXT NOT NULL,
                    is_searchable INTEGER DEFAULT 0,
                    is_indexed INTEGER DEFAULT 0
                )
            """)
            
            cursor.execute(f"""
                CREATE VIRTUAL TABLE {self._quote_identifier(table_name + '_fts')} USING fts5(
                    content,
                    field_name,
                    record_id UNINDEXED,
                    tokenize='porter unicode61'
                )
            """)
            
            cursor.execute(f"""
                CREATE TRIGGER {self._quote_identifier(table_name + '_fts_delete')} 
                AFTER DELETE ON {self._quote_identifier(table_name)} BEGIN
                    DELETE FROM {self._quote_identifier(table_name + '_fts')} 
                    WHERE record_id = old._id;
                END
            """)
            
            cursor.execute(
                "INSERT INTO tables_meta (table_name) VALUES (?)",
                [table_name]
            )
            
            cursor.execute("COMMIT")
            
        except Exception as e:
            cursor.execute("ROLLBACK")
            raise e

    def drop_table(self, table_name: str):
        """
        Drops a table.

        Parameters:
            table_name: str
                The table name to drop
        """
        if not self._table_exists(table_name):
            return

        if table_name == "default":
            raise ValueError("Cannot drop the default table")

        cursor = self.conn.cursor()
        try:
            cursor.execute("BEGIN IMMEDIATE")
            
            cursor.execute(f"DROP TABLE IF EXISTS {self._quote_identifier(table_name)}")
            cursor.execute(f"DROP TABLE IF EXISTS {self._quote_identifier(table_name + '_fields_meta')}")
            cursor.execute(f"DROP TABLE IF EXISTS {self._quote_identifier(table_name + '_fts')}")
            cursor.execute(f"DROP TRIGGER IF EXISTS {self._quote_identifier(table_name + '_fts_delete')}")
            
            cursor.execute("DELETE FROM tables_meta WHERE table_name = ?", [table_name])
            
            cursor.execute("COMMIT")
            
            if self.current_table == table_name:
                self.use_table("default")
            
            self._invalidate_cache()
            
        except Exception as e:
            cursor.execute("ROLLBACK")
            raise e

    def list_tables(self) -> List[str]:
        """
        Lists all tables.

        Returns:
            List[str]: The list of table names
        """
        cursor = self.conn.cursor()
        return [row[0] for row in cursor.execute("SELECT table_name FROM tables_meta ORDER BY table_name")]

    def _table_exists(self, table_name: str) -> bool:
        """
        Checks if a table exists.

        Parameters:
            table_name: str
                The table name to check

        Returns:
            bool: Whether the table exists
        """
        cursor = self.conn.cursor()
        return cursor.execute(
            "SELECT 1 FROM tables_meta WHERE table_name = ?",
            [table_name]
        ).fetchone() is not None

    def _quote_identifier(self, identifier: str) -> str:
        """
        Correctly escapes SQLite identifiers.

        Parameters:
            identifier: str
                The identifier to escape

        Returns:
            str: The escaped identifier
        """
        return f'"{identifier}"'

    def _ensure_field_exists(self, field_name: str, field_type: str, is_searchable: bool = True, table_name: str = None):
        """
        Ensures a field exists, creating it if it doesn't.

        Parameters:
            field_name: str
                The field name
            field_type: str
                The field type (TEXT, INTEGER, REAL, etc.)
            is_searchable: bool
                Whether the field is searchable
            table_name: str
                The table name, or None to use the current table
        """
        table_name = self._get_table_name(table_name)
        try:
            result = self.conn.execute(
                f"SELECT field_type FROM {self._quote_identifier(table_name + '_fields_meta')} WHERE field_name = ?",
                [field_name]
            ).fetchone()
            
            if not result:
                quoted_field_name = self._quote_identifier(field_name)
                self.conn.execute(f"ALTER TABLE {self._quote_identifier(table_name)} ADD COLUMN {quoted_field_name} {field_type}")
                self.conn.execute(
                    f"INSERT INTO {self._quote_identifier(table_name + '_fields_meta')} (field_name, field_type, is_searchable) VALUES (?, ?, ?)",
                    [field_name, field_type, 1 if is_searchable else 0]
                )
        except Exception as e:
            raise ValueError(f"Failed to ensure field exists: {str(e)}")

    def _infer_field_type(self, value: Any) -> str:
        """
        Infers the field type based on the value.

        Parameters:
            value: Any
                The field value

        Returns:
            str: The SQLite field type
        """
        if isinstance(value, bool):
            return "INTEGER"
        elif isinstance(value, int):
            return "INTEGER"
        elif isinstance(value, float):
            return "REAL"
        elif isinstance(value, (list, dict)):
            return "TEXT"
        else:
            return "TEXT"

    def _update_fts_index(self, record_id: int, data: dict, table_name: str = None):
        """
        Updates the FTS index.

        Parameters:
            record_id: int
                The record ID
            data: dict
                The record data
            table_name: str
                The table name, or None to use the current table
        """
        table_name = self._get_table_name(table_name)
        cursor = self.conn.cursor()
        
        searchable_fields = cursor.execute(
            f"SELECT field_name FROM {self._quote_identifier(table_name + '_fields_meta')} WHERE is_searchable = 1"
        ).fetchall()
        
        cursor.execute(f"DELETE FROM {self._quote_identifier(table_name + '_fts')} WHERE record_id = ?", [record_id])
        
        # Add index for each searchable field
        for (field_name,) in searchable_fields:
            if field_name in data:
                value = data[field_name]
                if value is not None:
                    if isinstance(value, (list, dict)):
                        content = json.dumps(value, ensure_ascii=False)
                    else:
                        content = str(value)
                    
                    content = content.replace(".", " ").replace("@", " ")
                    
                    cursor.execute(
                        f"INSERT INTO {self._quote_identifier(table_name + '_fts')} (content, field_name, record_id) VALUES (?, ?, ?)",
                        [content, field_name, record_id]
                    )

    def set_auto_update_fts(self, enabled: bool):
        """
        Sets whether to automatically update the FTS index.

        Parameters:
            enabled: bool
                Whether to enable automatic updates
        """
        self.auto_update_fts = enabled

    def store(self, data: dict, table_name: str = None) -> int:
        """
        Stores a record in the storage.

        Parameters:
            data: dict
                The record to store
            table_name: str
                The table name, or None to use the current table

        Returns:
            int: The record ID
        """
        if not isinstance(data, dict):
            raise ValueError("Only dict-type data is allowed.")

        table_name = self._get_table_name(table_name)
        max_retries = 3
        retry_count = 0
        
        while retry_count < max_retries:
            try:
                with self._lock:
                    cursor = self.conn.cursor()
                    cursor.execute("BEGIN IMMEDIATE")
                    
                    for field_name, value in data.items():
                        if field_name != '_id':
                            field_type = self._infer_field_type(value)
                            self._ensure_field_exists(field_name, field_type, table_name=table_name)
                    
                    fields = [field for field in data.keys() if field != '_id']
                    placeholders = ['?' for _ in fields]
                    values = [
                        json.dumps(data[field]) if isinstance(data[field], (dict, list)) else data[field]
                        for field in fields
                    ]
                    
                    if fields:
                        quoted_fields = [self._quote_identifier(field) for field in fields]
                        sql = f"INSERT INTO {self._quote_identifier(table_name)} ({', '.join(quoted_fields)}) VALUES ({', '.join(placeholders)})"
                    else:
                        sql = f"INSERT INTO {self._quote_identifier(table_name)} DEFAULT VALUES"
                    
                    cursor.execute(sql, values)
                    record_id = cursor.lastrowid
                    
                    if self.auto_update_fts:
                        self._update_fts_index(record_id, data, table_name)
                    
                    cursor.execute("COMMIT")
                    self._invalidate_cache()
                    return record_id
                    
            except sqlite3.OperationalError as e:
                cursor.execute("ROLLBACK")
                if "database is locked" in str(e) and retry_count < max_retries - 1:
                    retry_count += 1
                    time.sleep(0.1 * (2 ** retry_count))
                    continue
                raise e
            except Exception as e:
                cursor.execute("ROLLBACK")
                raise e

    def batch_store(self, data_list: List[dict], table_name: str = None) -> List[int]:
        """
        Batch stores records.

        Parameters:
            data_list: List[dict]
                The list of records to store
            table_name: str
                The table name, or None to use the current table

        Returns:
            List[int]: The list of record IDs
        """
        if not data_list:
            return []

        table_name = self._get_table_name(table_name)
        record_ids = []
        current_batch = []
        
        try:
            with self._lock:
                cursor = self.conn.cursor()
                cursor.execute("BEGIN TRANSACTION")
                
                all_fields = set()
                for data in data_list:
                    all_fields.update(data.keys())
                all_fields.discard('_id')
                
                for field_name in all_fields:
                    for data in data_list:
                        if field_name in data:
                            field_type = self._infer_field_type(data[field_name])
                            self._ensure_field_exists(field_name, field_type, table_name=table_name)
                            break
                
                for data in data_list:
                    current_batch.append(data)
                    
                    if len(current_batch) >= self.batch_size:
                        batch_ids = self._execute_batch_store(current_batch, table_name)
                        record_ids.extend(batch_ids)
                        current_batch = []
                
                if current_batch:
                    batch_ids = self._execute_batch_store(current_batch, table_name)
                    record_ids.extend(batch_ids)
                
                cursor.execute("COMMIT")
                self._invalidate_cache()
                return record_ids
                
        except Exception as e:
            cursor.execute("ROLLBACK")
            raise e

    def _execute_batch_store(self, batch: List[dict], table_name: str) -> List[int]:
        """
        Executes the batch store operation.

        Parameters:
            batch: List[dict]
                The batch of records to store
            table_name: str
                The table name

        Returns:
            List[int]: The list of record IDs
        """
        cursor = self.conn.cursor()
        record_ids = []
        
        for data in batch:
            fields = [field for field in data.keys() if field != '_id']
            placeholders = ['?' for _ in fields]
            values = [
                json.dumps(data[field]) if isinstance(data[field], (dict, list)) else data[field]
                for field in fields
            ]
            
            if fields:
                quoted_fields = [self._quote_identifier(field) for field in fields]
                sql = f"INSERT INTO {self._quote_identifier(table_name)} ({', '.join(quoted_fields)}) VALUES ({', '.join(placeholders)})"
            else:
                sql = f"INSERT INTO {self._quote_identifier(table_name)} DEFAULT VALUES"
            
            cursor.execute(sql, values)
            record_id = cursor.lastrowid
            record_ids.append(record_id)
            
            if self.auto_update_fts:
                self._update_fts_index(record_id, data, table_name)
        
        return record_ids

    def _parse_record(self, row: tuple, table_name: str = None) -> Dict[str, Any]:
        """
        Parses a record.

        Parameters:
            row: tuple
                The database row
            table_name: str
                The table name, or None to use the current table

        Returns:
            Dict[str, Any]: The parsed record
        """
        table_name = self._get_table_name(table_name)
        fields = self.list_fields(table_name=table_name)
        record = {}
        
        for i, (field_name, field_type) in enumerate(fields.items(), start=1):
            if i < len(row):
                value = row[i]
                if value is not None:
                    if field_type == 'TEXT':
                        try:
                            record[field_name] = json.loads(value)
                        except (json.JSONDecodeError, TypeError):
                            record[field_name] = value
                    else:
                        record[field_name] = value
        
        record['_id'] = row[0]
        return record

    def create_json_index(self, field_path: str):
        """
        Creates an index for a specified JSON field path.

        Parameters:
            field_path: str
                The JSON field path, e.g., "$.name" or "$.address.city"
        """
        try:
            safe_name = field_path.replace('$', '').replace('.', '_').replace('[', '_').replace(']', '_')
            index_name = f"idx_json_{safe_name.strip('_')}"
            
            self.conn.execute(f"""
                CREATE INDEX IF NOT EXISTS {index_name}
                ON records(json_extract(data, ?))
            """, (field_path,))
            
            self.conn.execute("ANALYZE")
        except Exception as e:
            raise ValueError(f"Failed to create JSON index: {str(e)}")

    def field_exists(self, field: str, use_cache: bool = True) -> bool:
        """
        Check if a field exists with caching.

        Parameters:
            field: str
                The field to check.
            use_cache: bool
                Whether to use cache.

        Returns:
            bool: True if the field exists, False otherwise.
        """
        field = field.strip(':')
        
        if use_cache:
            cache_key = self._get_cache_key("field_exists", field=field)
            cached_result = self._field_cache.get(cache_key)
            if cached_result is not None:
                return cached_result

        try:
            result = self.conn.execute(f"""
                SELECT COUNT(*) 
                FROM records 
                WHERE json_extract(data, '$.{field}') IS NOT NULL 
                LIMIT 1
            """).fetchone()
            exists = result[0] > 0
            
            if use_cache:
                self._field_cache[cache_key] = exists
            return exists
        except Exception:
            return False

    def list_fields(self, table_name: str = None, use_cache: bool = True) -> Dict[str, str]:
        """
        Lists the fields in a table.

        Parameters:
            table_name: str
                The table name, or None to use the current table
            use_cache: bool
                Whether to use cache

        Returns:
            Dict[str, str]: The mapping of field names to field types
        """
        table_name = self._get_table_name(table_name)
        cache_key = f"fields_{table_name}"
        
        if use_cache and cache_key in self._field_cache:
            return self._field_cache[cache_key]
        
        cursor = self.conn.cursor()
        fields = {}
        
        try:
            for row in cursor.execute(
                f"SELECT field_name, field_type FROM {self._quote_identifier(table_name + '_fields_meta')} ORDER BY field_name"
            ):
                fields[row[0]] = row[1]
            
            if use_cache:
                self._field_cache[cache_key] = fields.copy()
            
            return fields
            
        except Exception as e:
            raise ValueError(f"Failed to list fields: {str(e)}")

    def optimize(self, table_name: str = None):
        """
        Optimizes database performance.

        Parameters:
            table_name: str
                The table name, or None to use the current table
        """
        table_name = self._get_table_name(table_name)
        cursor = self.conn.cursor()
        
        try:
            cursor.execute("BEGIN TRANSACTION")
            
            cursor.execute(f"VACUUM")
            
            cursor.execute(f"ANALYZE {self._quote_identifier(table_name)}")
            cursor.execute(f"ANALYZE {self._quote_identifier(table_name + '_fields_meta')}")
            cursor.execute(f"ANALYZE {self._quote_identifier(table_name + '_fts')}")
            
            cursor.execute("COMMIT")
            
        except Exception as e:
            cursor.execute("ROLLBACK")
            raise ValueError(f"Failed to optimize database: {str(e)}")

    def _get_cache_key(self, operation: str, **kwargs) -> str:
        """Generate a cache key"""
        return f"{operation}:{orjson.dumps(kwargs).decode('utf-8')}"

    def _invalidate_cache(self):
        """Clear all caches"""
        self._field_cache.clear()

    def delete(self, id_: int) -> bool:
        """
        Deletes a record with the specified ID.

        Parameters:
            id_: int
                The ID of the record to delete

        Returns:
            bool: Whether the deletion was successful
        """
        try:
            table_name = self.current_table
            quoted_table = self._quote_identifier(table_name)
            
            cursor = self.conn.cursor()
            cursor.execute("BEGIN IMMEDIATE")
            try:
                exists = cursor.execute(
                    f"SELECT 1 FROM {quoted_table} WHERE _id = ?",
                    [id_]
                ).fetchone()
                if not exists:
                    cursor.execute("ROLLBACK")
                    return False
                
                cursor.execute(f"DELETE FROM {quoted_table} WHERE _id = ?", [id_])
                cursor.execute("COMMIT")
                self._invalidate_cache()
                return True
            except Exception as e:
                cursor.execute("ROLLBACK")
                raise e
        except Exception as e:
            raise ValueError(f"Failed to delete record: {str(e)}")

    def batch_delete(self, ids: List[int]) -> bool:
        """
        Batch deletes records.  

        Parameters:
            ids: List[int]
                The list of record IDs to delete

        Returns:
            List[int]: The list of record IDs that were successfully deleted
        """
        if not ids:
            return True

        try:
            table_name = self.current_table
            quoted_table = self._quote_identifier(table_name)
            quoted_fts = self._quote_identifier(table_name + '_fts')
            
            cursor = self.conn.cursor()
            cursor.execute("BEGIN IMMEDIATE")
            
            try:
                cursor.execute("DROP TRIGGER IF EXISTS " + self._quote_identifier(table_name + '_fts_delete'))
                
                batch_size = 1000
                for i in range(0, len(ids), batch_size):
                    batch_ids = ids[i:i + batch_size]
                    placeholders = ','.join('?' * len(batch_ids))
                    
                    cursor.execute(f"DELETE FROM {quoted_fts} WHERE record_id IN ({placeholders})", batch_ids)
                    cursor.execute(f"DELETE FROM {quoted_table} WHERE _id IN ({placeholders})", batch_ids)
                
                cursor.execute(f"""
                    CREATE TRIGGER {self._quote_identifier(table_name + '_fts_delete')} 
                    AFTER DELETE ON {quoted_table} BEGIN
                        DELETE FROM {quoted_fts} 
                        WHERE record_id = old._id;
                    END
                """)
                
                cursor.execute("COMMIT")
                self._invalidate_cache()
                return True
                
            except Exception as e:
                cursor.execute("ROLLBACK")
                raise e
        except Exception as e:
            raise ValueError(f"Batch deletion failed: {str(e)}")

    def replace(self, id_: int, data: dict) -> bool:
        """
        Replaces a record with the specified ID.

        Parameters:
            id_: int
                The ID of the record to replace
            data: dict
                The new record data

        Returns:
            bool: Whether the replacement was successful
        """
        if not isinstance(data, dict):
            raise ValueError("Only dict-type data is allowed.")

        try:
            table_name = self.current_table
            quoted_table = self._quote_identifier(table_name)
            
            cursor = self.conn.cursor()
            
            exists = cursor.execute(
                f"SELECT 1 FROM {quoted_table} WHERE _id = ?",
                [id_]
            ).fetchone()
            if not exists:
                return False

            cursor.execute("BEGIN IMMEDIATE")
            try:
                for field_name, value in data.items():
                    if field_name != '_id':
                        field_type = self._infer_field_type(value)
                        self._ensure_field_exists(field_name, field_type, table_name=table_name)
        
                field_updates = []
                params = []
                
                for field_name, value in data.items():
                    if field_name != '_id':
                        quoted_field_name = self._quote_identifier(field_name)
                        field_updates.append(f"{quoted_field_name} = ?")
                        # If it is a complex type, serialize to JSON
                        if isinstance(value, (list, dict)):
                            params.append(json.dumps(value))
                        else:
                            params.append(value)

                # If there are fields to update
                if field_updates:
                    update_sql = f"UPDATE {quoted_table} SET {', '.join(field_updates)} WHERE _id = ?"
                    params.append(id_)
                    cursor.execute(update_sql, params)

                # Update the FTS index
                self._update_fts_index(id_, data, table_name)

                cursor.execute("COMMIT")
                self._invalidate_cache()
                return True
            except Exception as e:
                cursor.execute("ROLLBACK")
                raise e
        except Exception as e:
            raise ValueError(f"Failed to replace record: {str(e)}")

    def batch_replace(self, data_dict: Dict[int, dict]) -> List[int]:
        """
        Batch replaces records.

        Parameters:
            data_dict: Dict[int, dict]
                The dictionary of records to replace, with keys as record IDs and values as new record data

        Returns:
            List[int]: The list of record IDs that were successfully replaced
        """
        if not data_dict:
            return []

        try:
            table_name = self.current_table
            quoted_table = self._quote_identifier(table_name)
            
            cursor = self.conn.cursor()
            cursor.execute("BEGIN IMMEDIATE")
            try:
                # First collect all unique fields and ensure they exist
                all_fields = set()
                for data in data_dict.values():
                    for field_name, value in data.items():
                        if field_name != '_id':
                            all_fields.add((field_name, self._infer_field_type(value)))

                # Batch create all required fields
                for field_name, field_type in all_fields:
                    self._ensure_field_exists(field_name, field_type, table_name=table_name)

                # Check if all IDs exist
                ids = list(data_dict.keys())
                placeholders = ','.join('?' * len(ids))
                existing_ids = cursor.execute(
                    f"SELECT _id FROM {quoted_table} WHERE _id IN ({placeholders})",
                    ids
                ).fetchall()
                existing_ids = {row[0] for row in existing_ids}

                # Only update existing records
                success_ids = []
                for id_, data in data_dict.items():
                    if id_ not in existing_ids:
                        continue

                    field_updates = []
                    params = []
                    for field_name, value in data.items():
                        if field_name != '_id':
                            quoted_field_name = self._quote_identifier(field_name)
                            field_updates.append(f"{quoted_field_name} = ?")
                            if isinstance(value, (list, dict)):
                                params.append(json.dumps(value))
                            else:
                                params.append(value)

                    if field_updates:
                        update_sql = f"UPDATE {quoted_table} SET {', '.join(field_updates)} WHERE _id = ?"
                        params.append(id_)
                        cursor.execute(update_sql, params)
                        success_ids.append(id_)

                cursor.execute("COMMIT")
                self._invalidate_cache()
                return success_ids
            except Exception as e:
                cursor.execute("ROLLBACK")
                raise e
        except Exception as e:
            raise ValueError(f"Batch replacement failed: {str(e)}")

    def search_text(self, query: str, fields: List[str] = None, table_name: str = None) -> List[int]:
        """
        Full-text search.

        Parameters:
            query: str
                The search query
            fields: List[str]
                The list of fields to search, or None to search all searchable fields
            table_name: str
                The table name, or None to use the current table

        Returns:
            List[int]: The list of record IDs that matched
        """
        table_name = self._get_table_name(table_name)
        cursor = self.conn.cursor()
        
        try:
            # Escape special characters
            escaped_query = query.replace(".", " ").replace("@", " ")
            
            if fields:
                # Validate fields are searchable
                searchable_fields = cursor.execute(
                    f"SELECT field_name FROM {self._quote_identifier(table_name + '_fields_meta')} WHERE is_searchable = 1"
                ).fetchall()
                searchable_fields = {row[0] for row in searchable_fields}
                
                invalid_fields = set(fields) - searchable_fields
                if invalid_fields:
                    raise ValueError(f"Fields {invalid_fields} are not searchable")
                
                # Build the query
                field_conditions = " OR ".join(f"field_name = ?" for _ in fields)
                sql = f"""
                    SELECT DISTINCT record_id 
                    FROM {self._quote_identifier(table_name + '_fts')}
                    WHERE ({field_conditions})
                    AND content MATCH ?
                    ORDER BY rank
                """
                params = fields + [escaped_query]
            else:
                sql = f"""
                    SELECT DISTINCT record_id 
                    FROM {self._quote_identifier(table_name + '_fts')}
                    WHERE content MATCH ?
                    ORDER BY rank
                """
                params = [escaped_query]
            
            return [row[0] for row in cursor.execute(sql, params)]
            
        except Exception as e:
            raise ValueError(f"Text search failed: {str(e)}")

    def set_searchable(self, field_name: str, is_searchable: bool = True, table_name: str = None):
        """
        Set whether a field is searchable.

        Parameters:
            field_name: str
                The field name
            is_searchable: bool
                Whether the field is searchable
            table_name: str
                The table name, or None to use the current table
        """
        table_name = self._get_table_name(table_name)
        cursor = self.conn.cursor()
        
        try:
            cursor.execute("BEGIN TRANSACTION")
            
            # Update field metadata
            cursor.execute(
                f"UPDATE {self._quote_identifier(table_name + '_fields_meta')} SET is_searchable = ? WHERE field_name = ?",
                [1 if is_searchable else 0, field_name]
            )
            
            if cursor.rowcount == 0:
                raise ValueError(f"Field {field_name} does not exist")
            
            # If set to searchable, add existing data to the FTS index
            if is_searchable:
                cursor.execute(f"DELETE FROM {self._quote_identifier(table_name + '_fts')} WHERE field_name = ?", [field_name])
                
                cursor.execute(
                    f"SELECT _id, {self._quote_identifier(field_name)} FROM {self._quote_identifier(table_name)} WHERE {self._quote_identifier(field_name)} IS NOT NULL"
                )
                
                for record_id, value in cursor.fetchall():
                    if isinstance(value, (list, dict)):
                        content = json.dumps(value, ensure_ascii=False)
                    else:
                        content = str(value)
                    
                    cursor.execute(
                        f"INSERT INTO {self._quote_identifier(table_name + '_fts')} (content, field_name, record_id) VALUES (?, ?, ?)",
                        [content, field_name, record_id]
                    )
            else:
                # If set to not searchable, delete from the FTS index
                cursor.execute(
                    f"DELETE FROM {self._quote_identifier(table_name + '_fts')} WHERE field_name = ?",
                    [field_name]
                )
            
            cursor.execute("COMMIT")
            self._invalidate_cache()
            
        except Exception as e:
            cursor.execute("ROLLBACK")
            raise ValueError(f"Failed to set searchable: {str(e)}")

    def rebuild_fts_index(self, table_name: str = None):
        """
        Rebuilds the full-text search index.

        Parameters:
            table_name: str
                The table name, or None to use the current table
        """
        table_name = self._get_table_name(table_name)
        cursor = self.conn.cursor()
        
        try:
            cursor.execute("BEGIN TRANSACTION")
            
            # Clear the FTS index
            cursor.execute(f"DELETE FROM {self._quote_identifier(table_name + '_fts')}")
            
            # Get searchable fields
            searchable_fields = cursor.execute(
                f"SELECT field_name FROM {self._quote_identifier(table_name + '_fields_meta')} WHERE is_searchable = 1"
            ).fetchall()
            
            # Rebuild the index
            for field_name, in searchable_fields:
                cursor.execute(
                    f"SELECT _id, {self._quote_identifier(field_name)} FROM {self._quote_identifier(table_name)} WHERE {self._quote_identifier(field_name)} IS NOT NULL"
                )
                
                for record_id, value in cursor.fetchall():
                    if value is not None:
                        if isinstance(value, (list, dict)):
                            content = json.dumps(value, ensure_ascii=False)
                        else:
                            content = str(value)
                        
                        cursor.execute(
                            f"INSERT INTO {self._quote_identifier(table_name + '_fts')} (content, field_name, record_id) VALUES (?, ?, ?)",
                            [content, field_name, record_id]
                        )
            
            cursor.execute("COMMIT")
            
        except Exception as e:
            cursor.execute("ROLLBACK")
            raise ValueError(f"Failed to rebuild FTS index: {str(e)}")

    def __del__(self):
        """
        Close all connections when the object is deleted.
        """
        if hasattr(self, 'conn'):
            self.conn.close()

    def _create_auto_indexes(self):
        """
        Automatically create indexes based on query patterns
        """
        try:
            cursor = self.conn.cursor()
            
            # Get all numeric fields
            numeric_fields = cursor.execute("""
                SELECT field_name 
                FROM fields_meta 
                WHERE field_type IN ('INTEGER', 'REAL')
            """).fetchall()
            
            # Create indexes for numeric fields (often used for range queries)
            for (field_name,) in numeric_fields:
                index_name = f"idx_{field_name}"
                quoted_field_name = self._quote_identifier(field_name)
                cursor.execute(f"""
                    CREATE INDEX IF NOT EXISTS {index_name}
                    ON records({quoted_field_name})
                """)
            
            # Get all TEXT fields
            text_fields = cursor.execute("""
                SELECT field_name 
                FROM fields_meta 
                WHERE field_type = 'TEXT'
            """).fetchall()
            
            # Create indexes for TEXT fields (used for LIKE queries)
            for (field_name,) in text_fields:
                index_name = f"idx_{field_name}_like"
                quoted_field_name = self._quote_identifier(field_name)
                cursor.execute(f"""
                    CREATE INDEX IF NOT EXISTS {index_name}
                    ON records({quoted_field_name} COLLATE NOCASE)
                """)
            
            # Analyze new indexes
            cursor.execute("ANALYZE")
            
        except Exception as e:
            print(f"Warning: Failed to create automatic indexes: {str(e)}")

    def analyze_query_performance(self, query: str) -> dict:
        """
        Analyze query performance
        
        Parameters:
            query: str
                The query to analyze
                
        Returns:
            dict: A dictionary containing the query plan and performance metrics
        """
        try:
            cursor = self.conn.cursor()
            
            # Enable query plan analysis
            cursor.execute("EXPLAIN QUERY PLAN " + query)
            query_plan = cursor.fetchall()
            
            # Collect performance metrics
            metrics = {
                'tables_used': set(),
                'indexes_used': set(),
                'scan_type': [],
                'estimated_rows': 0
            }
            
            for step in query_plan:
                detail = step[3]  # Query plan details
                
                if 'TABLE' in detail:
                    table = detail.split('TABLE')[1].split()[0]
                    metrics['tables_used'].add(table)
                
                if 'USING INDEX' in detail:
                    index = detail.split('USING INDEX')[1].split()[0]
                    metrics['indexes_used'].add(index)
                
                if 'SCAN' in detail:
                    scan_type = detail.split('SCAN')[0].strip()
                    metrics['scan_type'].append(scan_type)
                
                if 'rows=' in detail:
                    rows = int(detail.split('rows=')[1].split()[0])
                    metrics['estimated_rows'] = max(metrics['estimated_rows'], rows)
            
            return metrics
            
        except Exception as e:
            print(f"Warning: Failed to analyze query performance: {str(e)}")
            return {}

    def count_rows(self, table_name: str = None) -> int:
        """
        Returns the number of rows in a specified table or the current table.

        Parameters:
            table_name: str
                The table name, or None to use the current table

        Returns:
            int: The number of rows in the table

        Raises:
            ValueError: When the table does not exist
        """
        table_name = self._get_table_name(table_name)
        
        if not self._table_exists(table_name):
            raise ValueError(f"Table {table_name} does not exist")
            
        try:
            cursor = self.conn.cursor()
            result = cursor.execute(
                f"SELECT COUNT(*) FROM {self._quote_identifier(table_name)}"
            ).fetchone()
            return result[0] if result else 0
        except Exception as e:
            raise ValueError(f"Failed to count rows: {str(e)}")

    def retrieve(self, id_: int) -> Optional[dict]:
        """
        Retrieve a single record.

        Parameters:
            id_: int
                The record ID

        Returns:
            Optional[dict]: The record data, or None if it does not exist
        """
        table_name = self.current_table
        quoted_table = self._quote_identifier(table_name)
        cursor = self.conn.cursor()
        
        try:
            # Get all fields
            fields = self.list_fields()
            if not fields:
                return None
            
            # Build the query
            field_selects = [f"{self._quote_identifier(field)}" for field in fields]
            sql = f"SELECT _id, {', '.join(field_selects)} FROM {quoted_table} WHERE _id = ?"
            
            # Execute the query
            result = cursor.execute(sql, [id_]).fetchone()
            if not result:
                return None
            
            # Build the record dictionary
            record = {"_id": result[0]}
            for i, field in enumerate(fields, 1):
                value = result[i]
                if value is not None:
                    # Try to parse JSON strings
                    try:
                        record[field] = json.loads(value)
                    except (json.JSONDecodeError, TypeError):
                        record[field] = value
                else:
                    record[field] = None
            
            return record
            
        except Exception as e:
            raise ValueError(f"Failed to retrieve record: {str(e)}")

    def retrieve_many(self, ids: List[int]) -> List[dict]:
        """
        Retrieve multiple records.

        Parameters:
            ids: List[int]
                The list of record IDs

        Returns:
            List[dict]: The list of record data
        """
        if not ids:
            return []

        table_name = self.current_table
        quoted_table = self._quote_identifier(table_name)
        cursor = self.conn.cursor()
        
        try:
            # Get all fields
            fields = self.list_fields()
            if not fields:
                return []
            
            # Build the query
            field_selects = [f"{self._quote_identifier(field)}" for field in fields]
            placeholders = ','.join('?' * len(ids))
            sql = f"SELECT _id, {', '.join(field_selects)} FROM {quoted_table} WHERE _id IN ({placeholders})"
            
            # Execute the query
            results = cursor.execute(sql, ids).fetchall()
            
            # Build the record list
            records = []
            for result in results:
                record = {"_id": result[0]}
                for i, field in enumerate(fields, 1):
                    value = result[i]
                    if value is not None:
                        # Try to parse JSON strings
                        try:
                            record[field] = json.loads(value)
                        except (json.JSONDecodeError, TypeError):
                            record[field] = value
                    else:
                        record[field] = None
                records.append(record)
            
            return records
            
        except Exception as e:
            raise ValueError(f"Failed to retrieve records: {str(e)}")

    def close(self):
        """
        Close the database connection.
        """
        self.conn.close()
