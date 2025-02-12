import sqlite3
import time
import orjson
from typing import Dict, List, Any, Optional
import json
import threading
import os
import re
import pandas as pd

from apexbase.storage.id_manager import IDManager

from .base import BaseStorage, BaseSchema


class SQLiteSchema(BaseSchema):
    """SQLite schema."""

    def __init__(self, schema: dict = None):
        """Initialize the schema
        
        Args:
            schema: Optional schema dictionary, format: {'columns': {'column_name': 'column_type'}}
        """
        self.schema = schema or {'columns': {'_id': 'INTEGER'}}
        self._validate_schema()

    def _validate_schema(self):
        """Validate the schema format"""
        if not isinstance(self.schema, dict):
            raise ValueError("Schema must be a dictionary")
        if 'columns' not in self.schema:
            raise ValueError("Schema must have a 'columns' key")
        if not isinstance(self.schema['columns'], dict):
            raise ValueError("Schema columns must be a dictionary")
        if '_id' not in self.schema['columns']:
            self.schema['columns']['_id'] = 'INTEGER'

    def to_dict(self):
        """Convert to dictionary format"""
        return self.schema
    
    def drop_column(self, column_name: str):
        """Drop a column
        
        Args:
            column_name: Column name
        """
        if column_name == '_id':
            raise ValueError("Cannot drop _id column")
        if column_name in self.schema['columns']:
            del self.schema['columns'][column_name]

    def add_column(self, column_name: str, column_type: str):
        """Add a column
        
        Args:
            column_name: Column name
            column_type: Column type
        """
        if column_name in self.schema['columns']:
            raise ValueError(f"Column {column_name} already exists")
        self.schema['columns'][column_name] = column_type

    def rename_column(self, old_column_name: str, new_column_name: str):
        """Rename a column
        
        Args:
            old_column_name: Old column name
            new_column_name: New column name
        """
        if old_column_name == '_id':
            raise ValueError("Cannot rename _id column")
        if old_column_name not in self.schema['columns']:
            raise ValueError(f"Column {old_column_name} does not exist")
        if new_column_name in self.schema['columns']:
            raise ValueError(f"Column {new_column_name} already exists")
        self.schema['columns'][new_column_name] = self.schema['columns'].pop(old_column_name)

    def modify_column(self, column_name: str, column_type: str):
        """Modify the column type
        
        Args:
            column_name: Column name
            column_type: Column type
        """
        if column_name == '_id':
            raise ValueError("Cannot modify _id column type")
        if column_name not in self.schema['columns']:
            raise ValueError(f"Column {column_name} does not exist")
        self.schema['columns'][column_name] = column_type

    def get_column_type(self, column_name: str) -> str:
        """Get the column type
        
        Args:
            column_name: Column name
        """
        if column_name not in self.schema['columns']:
            raise ValueError(f"Column {column_name} does not exist")
        return self.schema['columns'][column_name]

    def has_column(self, column_name: str) -> bool:
        """Check if the column exists
        
        Args:
            column_name: Column name
        """
        return column_name in self.schema['columns']

    def get_columns(self) -> List[str]:
        """Get all column names
        
        Returns:
            Column name list
        """
        return list(self.schema['columns'].keys())

    def update_from_data(self, data: dict):
        """Update the schema from data
        
        Args:
            data: Data dictionary
        """
        for column_name, value in data.items():
            if column_name != '_id' and column_name not in self.schema['columns']:
                column_type = self._infer_column_type(value)
                self.add_column(column_name, column_type)

    def _infer_column_type(self, value) -> str:
        """Infer the column type
        
        Args:
            value: Value
            
        Returns:
            Column type
        """
        if isinstance(value, bool):
            return "INTEGER"  # SQLite没有布尔类型
        elif isinstance(value, int):
            return "INTEGER"
        elif isinstance(value, float):
            return "REAL"
        elif isinstance(value, (str, dict, list)):
            return "TEXT"
        elif pd.isna(value):
            return "TEXT"  # For empty values, default to TEXT
        else:
            return "TEXT"  # For unknown types, default to TEXT


class SQLiteStorage(BaseStorage):
    """SQLite implementation of the storage backend."""
    
    def __init__(self, db_path: str, batch_size: int = 1000, enable_cache: bool = True, cache_size: int = 10000):
        """Initialize the SQLite storage
        
        Args:
            db_path: Database file path
            batch_size: Batch size
        """
        self.db_path = db_path
        self.batch_size = batch_size

        self._lock = threading.Lock()
        self.current_table = "default"
        self.enable_cache = enable_cache
        self.cache_size = cache_size
        self._cache = []
        self.id_manager = IDManager(self)
        
        self._last_modified_time = None
        
        # 使用 SQLiteSchema 管理所有表的 schema
        self._schema_manager = SQLiteSchema({
            'columns': {
                '_id': 'INTEGER'
            },
            'tables': {}  # table_name -> {'columns': {'column_name': 'column_type'}}
        })

        # Create the database directory
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        
        # Initialize the database
        self._initialize_database()

    def _get_connection(self):
        """Get the database connection"""
        if not hasattr(self, '_conn'):
            self._conn = sqlite3.connect(self.db_path, isolation_level=None)
            self._conn.row_factory = sqlite3.Row
            
        return self._conn

    def _initialize_database(self):
        """Initialize the database, create necessary tables"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            # Set database parameters
            cursor.execute('PRAGMA journal_mode=WAL')
            cursor.execute('PRAGMA synchronous=NORMAL')
            cursor.execute('PRAGMA cache_size=2000000')  # Further increase cache
            cursor.execute('PRAGMA temp_store=MEMORY')
            cursor.execute('PRAGMA mmap_size=30000000000')
            cursor.execute('PRAGMA page_size=32768')
            cursor.execute('PRAGMA busy_timeout=5000')
            cursor.execute('PRAGMA locking_mode=EXCLUSIVE')
            cursor.execute('PRAGMA foreign_keys=OFF')
            cursor.execute('PRAGMA read_uncommitted=1')
            cursor.execute('PRAGMA recursive_triggers=0')
            cursor.execute('PRAGMA auto_vacuum=INCREMENTAL')
            cursor.execute('PRAGMA secure_delete=OFF')
            
            # Create reverse string function
            conn.create_function("reverse", 1, lambda s: s[::-1] if s else None)
            
            # Create the main table - only contains the _id field, other fields are dynamically added
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS "default" (
                _id INTEGER PRIMARY KEY
            )
            ''')
            
            # Create the metadata table
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS tables_meta (
                table_name TEXT PRIMARY KEY,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                schema TEXT  -- Store the field definition of the table
            )
            ''')
            
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS fields_meta (
                table_name TEXT,
                field_name TEXT,
                field_type TEXT,
                is_json BOOLEAN DEFAULT FALSE,
                is_indexed BOOLEAN DEFAULT FALSE,
                ordinal_position INTEGER,
                PRIMARY KEY (table_name, field_name)
            )
            ''')
            
            # Initialize metadata
            default_schema = SQLiteSchema().to_dict()
            cursor.execute(
                'INSERT OR IGNORE INTO tables_meta (table_name, schema) VALUES (?, ?)',
                ('default', json.dumps(default_schema))
            )
            
            # Initialize the _id field of the default table
            cursor.execute('''
            INSERT OR IGNORE INTO fields_meta (table_name, field_name, field_type, ordinal_position) 
            VALUES (?, ?, ?, ?)
            ''', ('default', '_id', 'INTEGER', 1))
            
            conn.commit()

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
        with self._lock:
            if not self._table_exists(table_name):
                raise ValueError(f"Table '{table_name}' does not exist")
            self.current_table = table_name

    def _get_table_schema(self, table_name: str) -> SQLiteSchema:
        """获取表的 schema
        
        Args:
            table_name: 表名
            
        Returns:
            表的 schema
        """
        if not self._schema_manager.has_column('tables'):
            self._schema_manager.add_column('tables', 'OBJECT')
            
        tables = self._schema_manager.to_dict().get('tables', {})
        if table_name not in tables:
            # 如果表不存在，创建默认 schema
            if not self._table_exists(table_name):
                tables[table_name] = {'columns': {'_id': 'INTEGER'}}
            else:
                # 从数据库中读取字段信息
                cursor = self._get_connection().cursor()
                fields = {'_id': 'INTEGER'}
                for row in cursor.execute("""
                    SELECT field_name, field_type 
                    FROM fields_meta 
                    WHERE table_name = ?
                    ORDER BY ordinal_position
                """, [table_name]):
                    fields[row[0]] = row[1]
                tables[table_name] = {'columns': fields}
            
            # 更新 schema_manager
            self._schema_manager.modify_column('tables', 'OBJECT')
        
        return SQLiteSchema(tables[table_name])
        
    def _update_table_schema(self, table_name: str, schema: SQLiteSchema):
        """更新表的 schema
        
        Args:
            table_name: 表名
            schema: 新的 schema
        """
        if not self._schema_manager.has_column('tables'):
            self._schema_manager.add_column('tables', 'OBJECT')
            
        tables = self._schema_manager.to_dict().get('tables', {})
        tables[table_name] = schema.to_dict()
        self._schema_manager.modify_column('tables', 'OBJECT')
        
    def _remove_table_schema(self, table_name: str):
        """删除表的 schema
        
        Args:
            table_name: 表名
        """
        if self._schema_manager.has_column('tables'):
            tables = self._schema_manager.to_dict().get('tables', {})
            if table_name in tables:
                del tables[table_name]
                self._schema_manager.modify_column('tables', 'OBJECT')

    def create_schema(self, table_name: str, schema: SQLiteSchema):
        """Create the schema of the table
        
        Args:
            table_name: Table name
            schema: Schema object
        """
        if self._table_exists(table_name):
            raise ValueError(f"Table '{table_name}' already exists")

        cursor = self._get_connection().cursor()
        try:
            cursor.execute("BEGIN TRANSACTION")
            
            # Create the table
            columns = []
            for col_name, col_type in schema.to_dict()['columns'].items():
                if col_name == '_id':
                    columns.append(f"{self._quote_identifier(col_name)} {col_type} PRIMARY KEY")
                else:
                    columns.append(f"{self._quote_identifier(col_name)} {col_type}")
            
            create_sql = f"""
                CREATE TABLE IF NOT EXISTS {self._quote_identifier(table_name)} (
                    {', '.join(columns)}
                )
            """
            cursor.execute(create_sql)
            
            # Initialize fields_meta table
            for position, (field_name, field_type) in enumerate(schema.to_dict()['columns'].items(), 1):
                cursor.execute("""
                    INSERT INTO fields_meta (table_name, field_name, field_type, ordinal_position)
                    VALUES (?, ?, ?, ?)
                    ON CONFLICT (table_name, field_name) DO UPDATE SET 
                        field_type = EXCLUDED.field_type,
                        ordinal_position = EXCLUDED.ordinal_position
                """, [table_name, field_name, field_type, position])
            
            cursor.execute("COMMIT")
            
            # 更新 schema
            self._update_table_schema(table_name, schema)
            
        except Exception as e:
            cursor.execute("ROLLBACK")
            raise e

    def create_table(self, table_name: str):
        """Create a new table, using the default schema
        
        Args:
            table_name: Table name
        """
        schema = SQLiteSchema()  # Use the default schema
        self.create_schema(table_name, schema)
        
        # 更新元数据
        cursor = self._get_connection().cursor()
        cursor.execute(
            "INSERT OR REPLACE INTO tables_meta (table_name, schema) VALUES (?, ?)",
            [table_name, json.dumps(schema.to_dict())]
        )

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

        cursor = self._get_connection().cursor()
        try:
            cursor.execute("BEGIN IMMEDIATE")
            
            cursor.execute(f"DROP TABLE IF EXISTS {self._quote_identifier(table_name)}")
            cursor.execute("DELETE FROM fields_meta WHERE table_name = ?", [table_name])
            cursor.execute("DELETE FROM tables_meta WHERE table_name = ?", [table_name])
            
            cursor.execute("COMMIT")
            
            # 删除 schema
            self._remove_table_schema(table_name)
            
            if self.current_table == table_name:
                self.use_table("default")
            
        except Exception as e:
            cursor.execute("ROLLBACK")
            raise e
        
        finally:
            self._last_modified_time = time.time()

    def list_tables(self) -> List[str]:
        """
        Lists all tables.

        Returns:
            List[str]: The list of table names
        """
        cursor = self._get_connection().cursor()
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
        cursor = self._get_connection().cursor()
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

    def _infer_field_type(self, value: Any) -> str:
        """Infer the field type
        
        Args:
            value: Field value
            
        Returns:
            Field type
        """
        if isinstance(value, bool):
            return "INTEGER"  # SQLite does not have a boolean type, use INTEGER
        elif isinstance(value, int):
            return "INTEGER"
        elif isinstance(value, float):
            return "REAL"
        elif isinstance(value, (list, dict)):
            return "TEXT"  # Complex types are serialized to JSON storage
        else:
            return "TEXT"

    def _ensure_fields_exist(self, data: dict, table_name: str = None):
        """Ensure all fields exist
        
        Args:
            data: Data dictionary
            table_name: Table name
        """
        table_name = self._get_table_name(table_name)
        cursor = self._get_connection().cursor()
        
        try:
            # If the table does not exist, create the table
            if not self._table_exists(table_name):
                schema = SQLiteSchema()
                for field_name, value in data.items():
                    if field_name != '_id':
                        field_type = self._infer_field_type(value)
                        schema.add_column(field_name, field_type)
                self.create_schema(table_name, schema)
                return
            
            # Get current schema
            current_schema = self._get_table_schema(table_name)
            
            # Check and add new fields
            new_fields = []
            for field_name, value in data.items():
                if field_name != '_id' and not current_schema.has_column(field_name):
                    field_type = self._infer_field_type(value)
                    new_fields.append((field_name, field_type))
                    current_schema.add_column(field_name, field_type)
            
            if new_fields:
                # Get the current maximum ordinal_position
                result = cursor.execute("""
                    SELECT COALESCE(MAX(ordinal_position), 0) + 1
                    FROM fields_meta
                    WHERE table_name = ?
                """, [table_name]).fetchone()
                next_position = result[0] if result else 2  # _id是1，所以从2开始
                
                # Add new fields in a single transaction
                cursor.execute("BEGIN TRANSACTION")
                try:
                    for field_name, field_type in new_fields:
                        quoted_field = self._quote_identifier(field_name)
                        
                        # Add the field to the table
                        cursor.execute(
                            f"ALTER TABLE {self._quote_identifier(table_name)} ADD COLUMN {quoted_field} {field_type}"
                        )
                        
                        # Update metadata
                        cursor.execute("""
                            INSERT INTO fields_meta (table_name, field_name, field_type, ordinal_position)
                            VALUES (?, ?, ?, ?)
                            ON CONFLICT (table_name, field_name) DO UPDATE SET 
                                field_type = EXCLUDED.field_type,
                                ordinal_position = EXCLUDED.ordinal_position
                        """, [table_name, field_name, field_type, next_position])
                        next_position += 1
                    
                    cursor.execute("COMMIT")
                    
                    # 更新 schema
                    self._update_table_schema(table_name, current_schema)
                    
                except Exception as e:
                    cursor.execute("ROLLBACK")
                    raise e
            
        except Exception as e:
            raise e

    def store(self, data: dict, table_name: str = None) -> int:
        """Store a single record
        
        Args:
            data: Data to store
            table_name: Table name
            
        Returns:
            The ID of the stored record
        """
        table_name = self._get_table_name(table_name)
        
        # Preprocess data
        processed_data = {}
        for k, v in data.items():
            if isinstance(v, (dict, list)):
                processed_data[k] = orjson.dumps(v).decode('utf-8')
            else:
                processed_data[k] = v
        
        # 只在第一次存储时检查字段
        if self.id_manager.get_next_id(table_name) == 1:
            self._ensure_fields_exist(processed_data, table_name)
        
        if self.enable_cache:
            with self._lock:
                self._cache.append(processed_data)
                self.id_manager.auto_increment(table_name)
                
                # 当缓存达到一定大小时，分批处理
                if len(self._cache) >= self.cache_size:
                    # 分批处理缓存数据
                    batch_size = 1000
                    for i in range(0, len(self._cache), batch_size):
                        batch = self._cache[i:i + batch_size]
                        self.batch_store(batch, table_name)
                    self._cache = []
                    
            return self.id_manager.current_id(table_name)
        
        with self._get_connection() as conn:
            cursor = conn.cursor()
            try:
                fields = []
                values = []
                params = []
                
                for field_name, value in processed_data.items():
                    if field_name != '_id':
                        fields.append(self._quote_identifier(field_name))
                        values.append('?')
                        params.append(value)
                
                sql = f"""
                    INSERT INTO {self._quote_identifier(table_name)} 
                    ({', '.join(fields)}) 
                    VALUES ({', '.join(values)})
                """
                cursor.execute(sql, params)
                
                record_id = cursor.lastrowid
                conn.commit()
                return record_id
                
            except Exception as e:
                conn.rollback()
                raise e
            
            finally:
                self.id_manager.reset_last_id(table_name)
                self._last_modified_time = time.time()

    def flush_cache(self):
        """Flush the cache"""
        if self._cache:
            with self._lock:
                self.batch_store(self._cache)
                self._cache = []

    def batch_store(self, data_list: List[dict], table_name: str = None) -> List[int]:
        """Batch store records
        
        Args:
            data_list: List of records to store
            table_name: Table name
            
        Returns:
            List of record IDs
        """
        if not data_list:
            return []
        
        table_name = self._get_table_name(table_name)
        
        # 先检查并创建表（如果需要）
        if not self._table_exists(table_name):
            self.create_table(table_name)
        
        # Preprocess: Get all fields and types, keep field order
        all_fields = {'_id': 'INTEGER'}  # Ensure _id field is included
        field_order = []  # Keep field addition order
        for data in data_list:
            for key, value in data.items():
                if key not in all_fields and key != '_id':  # Skip _id field
                    all_fields[key] = self._infer_field_type(value)
                    if key not in field_order:
                        field_order.append(key)
        
        # 使用更小的批次大小来处理数据
        batch_size = min(1000, len(data_list))
        all_ids = []
        
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                
                # 首先获取起始ID
                first_id = cursor.execute(
                    f"SELECT COALESCE(MAX(_id), 0) + 1 FROM {self._quote_identifier(table_name)}"
                ).fetchone()[0]
                current_id = first_id
                
                # 分批处理数据
                for i in range(0, len(data_list), batch_size):
                    batch = data_list[i:i + batch_size]
                    
                    try:
                        cursor.execute("BEGIN TRANSACTION")
                        
                        # Get existing fields
                        existing_fields = set()
                        for row in cursor.execute(
                            "SELECT field_name FROM fields_meta WHERE table_name = ?",
                            [table_name]
                        ):
                            existing_fields.add(row[0])
                        
                        # Get the current maximum ordinal_position
                        result = cursor.execute("""
                            SELECT COALESCE(MAX(ordinal_position), 0) + 1
                            FROM fields_meta
                            WHERE table_name = ?
                        """, [table_name]).fetchone()
                        next_position = result[0] if result else 2  # _id是1，所以从2开始
                        
                        # Add new fields in the order of field_order
                        for field_name in field_order:
                            if field_name not in existing_fields:
                                field_type = all_fields[field_name]
                                quoted_field = self._quote_identifier(field_name)
                                
                                # Add field to table
                                cursor.execute(
                                    f"ALTER TABLE {self._quote_identifier(table_name)} ADD COLUMN {quoted_field} {field_type}"
                                )
                                
                                # Update metadata
                                cursor.execute("""
                                    INSERT INTO fields_meta (table_name, field_name, field_type, ordinal_position)
                                    VALUES (?, ?, ?, ?)
                                    ON CONFLICT (table_name, field_name) DO UPDATE SET 
                                        field_type = EXCLUDED.field_type,
                                        ordinal_position = EXCLUDED.ordinal_position
                                """, [table_name, field_name, field_type, next_position])
                                next_position += 1
                        
                        # Prepare batch insertion
                        fields = ['_id'] + field_order
                        placeholders = ','.join(['?' for _ in fields])
                        sql = f"""
                            INSERT INTO {self._quote_identifier(table_name)}
                            ({','.join(self._quote_identifier(f) for f in fields)})
                            VALUES ({placeholders})
                        """
                        
                        # Prepare data
                        values = []
                        for data in batch:
                            row = [current_id]  # _id
                            current_id += 1
                            for field in field_order:
                                value = data.get(field)
                                if isinstance(value, (list, dict)):
                                    row.append(json.dumps(value))
                                else:
                                    row.append(value)
                            values.append(tuple(row))
                        
                        # Execute batch insertion
                        cursor.executemany(sql, values)
                        
                        cursor.execute("COMMIT")
                        all_ids.extend(range(current_id - len(batch), current_id))
                        
                    except Exception as e:
                        cursor.execute("ROLLBACK")
                        raise e
                
                return all_ids
                
        finally:
            self.id_manager.reset_last_id(table_name)
            self._last_modified_time = time.time()

    def list_fields(self, table_name: str = None) -> List[str]:
        """Get all fields of the table
        
        Args:
            table_name: Table name
            
        Returns:
            Field list
        """
        table_name = self._get_table_name(table_name)
        return self._get_table_schema(table_name).get_columns()

    def optimize(self, table_name: str = None):
        """
        Optimizes database performance.

        Parameters:
            table_name: str
                The table name, or None to use the current table
        """
        table_name = self._get_table_name(table_name)
        cursor = self._get_connection().cursor()
        
        try:
            # VACUUM cannot be executed in a transaction
            cursor.execute("VACUUM")
            
            # Other optimization operations are placed in a transaction
            cursor.execute("BEGIN TRANSACTION")
            cursor.execute(f"ANALYZE {self._quote_identifier(table_name)}")
            cursor.execute("COMMIT")
            
        except Exception as e:
            cursor.execute("ROLLBACK")
            raise ValueError(f"Failed to optimize database: {str(e)}")

    def _get_cache_key(self, operation: str, **kwargs) -> str:
        """Generate a cache key"""
        return f"{operation}:{orjson.dumps(kwargs).decode('utf-8')}"

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
            
            cursor = self._get_connection().cursor()
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
                return True
            except Exception as e:
                cursor.execute("ROLLBACK")
                raise e
        except Exception as e:
            raise ValueError(f"Failed to delete record: {str(e)}")
        
        finally:
            self.id_manager.reset_last_id(table_name)
            self._last_modified_time = time.time()

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
            
            cursor = self._get_connection().cursor()
            cursor.execute("BEGIN IMMEDIATE")
            
            try:
                batch_size = 1000
                for i in range(0, len(ids), batch_size):
                    batch_ids = ids[i:i + batch_size]
                    placeholders = ','.join('?' * len(batch_ids))
                    
                    cursor.execute(f"DELETE FROM {quoted_table} WHERE _id IN ({placeholders})", batch_ids)
                
                cursor.execute("COMMIT")

                return True
                
            except Exception as e:
                cursor.execute("ROLLBACK")
                raise e
        except Exception as e:
            raise ValueError(f"Batch deletion failed: {str(e)}")
        
        finally:
            self.id_manager.reset_last_id(table_name)
            self._last_modified_time = time.time()

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
            
            cursor = self._get_connection().cursor()
            
            exists = cursor.execute(
                f"SELECT 1 FROM {quoted_table} WHERE _id = ?",
                [id_]
            ).fetchone()
            if not exists:
                return False

            cursor.execute("BEGIN IMMEDIATE")
            try:
                # Ensure all fields exist
                self._ensure_fields_exist(data, table_name)
                
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

                cursor.execute("COMMIT")

                return True
            except Exception as e:
                cursor.execute("ROLLBACK")
                raise e
        except Exception as e:
            raise ValueError(f"Failed to replace record: {str(e)}")

        finally:
            self._last_modified_time = time.time()

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

            cursor = self._get_connection().cursor()
            cursor.execute("BEGIN IMMEDIATE")
            try:
                # Ensure all fields exist
                for data in data_dict.values():
                    self._ensure_fields_exist(data, table_name)

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

                return success_ids
            except Exception as e:
                cursor.execute("ROLLBACK")
                raise e
        except Exception as e:
            raise ValueError(f"Batch replacement failed: {str(e)}")

        finally:
            self._last_modified_time = time.time()  

    def _get_next_id(self, table_name: str) -> int:
        """Get the next ID"""
        cursor = self._get_connection().cursor()
        result = cursor.execute(f"""
            SELECT COALESCE(MAX(_id), 0) + 1 
            FROM {self._quote_identifier(table_name)}
        """).fetchone()
        return result[0] if result else 1
    
    def close(self):
        """Close the database connection"""
        if self.enable_cache:
            self.flush_cache()

        if hasattr(self, '_conn'):
            self._conn.close()
            del self._conn

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
            cursor = self._get_connection().cursor()
            
            # 如果有缓存中的数据，需要包含在计数中
            cache_count = len(self._cache) if self.enable_cache else 0
            
            result = cursor.execute(
                f"SELECT COUNT(*) FROM {self._quote_identifier(table_name)}"
            ).fetchone()
            return result[0] + cache_count if result else cache_count
            
        except Exception as e:
            raise ValueError(f"Failed to count rows: {str(e)}")

    def retrieve(self, id_: int) -> Optional[dict]:
        """Get a single record
        
        Args:
            id_: Record ID
            
        Returns:
            Record data dictionary
        """
        table_name = self._get_table_name()
        cursor = self._get_connection().cursor()
        
        try:
            # Get all fields
            fields = self.list_fields(table_name)
            if not fields:
                return None
            
            # Build query
            field_list = ','.join(self._quote_identifier(f) for f in fields)
            sql = f"""
                SELECT {field_list}
                FROM {self._quote_identifier(table_name)}
                WHERE _id = ?
            """
            
            result = cursor.execute(sql, [id_]).fetchone()
            if not result:
                return None
            
            # Build return data
            data = {}
            for i, field in enumerate(fields):
                value = result[i]
                if value is not None:
                    try:
                        # Try to parse JSON
                        data[field] = json.loads(value)
                    except (json.JSONDecodeError, TypeError):
                        data[field] = value
                else:
                    data[field] = None
            
            return data
            
        except Exception as e:
            raise ValueError(f"Failed to retrieve record: {str(e)}")

    def retrieve_many(self, ids: List[int]) -> List[dict]:
        """Get multiple records
        
        Args:
            ids: Record ID list
            
        Returns:
            Record data dictionary list
        """
        if not ids:
            return []
            
        table_name = self._get_table_name()
        cursor = self._get_connection().cursor()
        
        try:
            # Get all fields
            fields = self.list_fields(table_name)
            if not fields:
                return []
            
            # Build query
            field_list = ','.join(self._quote_identifier(f) for f in fields)
            placeholders = ','.join('?' for _ in ids)
            sql = f"""
                SELECT {field_list}
                FROM {self._quote_identifier(table_name)}
                WHERE _id IN ({placeholders})
                ORDER BY _id
            """
            
            results = cursor.execute(sql, ids).fetchall()
            
            # Build return data
            records = []
            for row in results:
                data = {}
                for i, field in enumerate(fields):
                    value = row[i]
                    if value is not None:
                        try:
                            # Try to parse JSON
                            data[field] = json.loads(value)
                        except (json.JSONDecodeError, TypeError):
                            data[field] = value
                    else:
                        data[field] = None
                records.append(data)
            
            return records
            
        except Exception as e:
            raise ValueError(f"Failed to retrieve records: {str(e)}")

    def query(self, sql: str, params: tuple = None) -> List[tuple]:
        """Execute a custom SQL query
        
        Args:
            sql: SQL statement
            params: Query parameters
            
        Returns:
            Query results
        """
        cursor = self._get_connection().cursor()
        
        # Global query optimization parameters
        cursor.execute('PRAGMA temp_store=MEMORY')
        cursor.execute('PRAGMA cache_size=2000000')
        cursor.execute('PRAGMA mmap_size=30000000000')
        cursor.execute('PRAGMA read_uncommitted=1')
        cursor.execute('PRAGMA synchronous=OFF')
        cursor.execute('PRAGMA journal_mode=MEMORY')
        cursor.execute('PRAGMA page_size=4096')
        
        # Optimize query
        if 'LIKE' in sql.upper():
            # Add index hint
            sql = f"/* USING INDEX */ {sql}"
            
            # Optimize LIKE query
            if 'LIKE' in sql and '%' in sql:
                # If it is a prefix match, use the index
                if sql.count('%') == 1 and sql.endswith("%'"):
                    sql = sql.replace('LIKE', 'GLOB')
                    sql = sql.replace('%', '*')
                # If it is a suffix match, reverse the string and use prefix matching
                elif sql.count('%') == 1 and sql.startswith("'%"):
                    field = re.search(r'(\w+)\s+LIKE', sql).group(1)
                    pattern = sql.split("'%")[1].rstrip("'")
                    sql = f"""
                        SELECT * FROM (
                            SELECT *, reverse({field}) as rev_{field}
                            FROM ({sql.split('WHERE')[0]})
                        ) WHERE rev_{field} GLOB '{pattern[::-1]}*'
                    """
        
        # Use iterator mode to execute the query
        cursor.execute(sql, params)
        chunk_size = 1000  # Number of records to fetch at a time
        results = []
        while True:
            chunk = cursor.fetchmany(chunk_size)
            if not chunk:
                break
            results.extend(chunk)
        return results

    def _create_indexes(self, table_name: str):
        """Create necessary indexes for the table"""
        cursor = self._get_connection().cursor()
        
        try:
            cursor.execute("BEGIN IMMEDIATE")
            
            # Get fields that need to be indexed
            fields = cursor.execute("""
                SELECT field_name, field_type 
                FROM fields_meta 
                WHERE table_name = ? AND is_indexed = FALSE
            """, [table_name]).fetchall()
            
            for field_name, field_type in fields:
                # Create index for TEXT/VARCHAR fields
                if field_type in ('TEXT', 'VARCHAR'):
                    index_name = f"idx_{table_name}_{field_name}"
                    # Use partial index and prefix index to optimize LIKE queries
                    cursor.execute(f"""
                        CREATE INDEX IF NOT EXISTS {index_name}
                        ON {self._quote_identifier(table_name)} ({self._quote_identifier(field_name)})
                        WHERE {self._quote_identifier(field_name)} IS NOT NULL
                    """)
                    
                    # Create prefix index
                    cursor.execute(f"""
                        CREATE INDEX IF NOT EXISTS {index_name}_prefix
                        ON {self._quote_identifier(table_name)} (
                            substr({self._quote_identifier(field_name)}, 1, 10)
                        )
                        WHERE {self._quote_identifier(field_name)} IS NOT NULL
                    """)
                    
                    # Create reverse index, for suffix matching optimization
                    cursor.execute(f"""
                        CREATE INDEX IF NOT EXISTS {index_name}_reverse
                        ON {self._quote_identifier(table_name)} (
                            reverse({self._quote_identifier(field_name)})
                        )
                        WHERE {self._quote_identifier(field_name)} IS NOT NULL
                    """)
                    
                    # Update index status
                    cursor.execute("""
                        UPDATE fields_meta 
                        SET is_indexed = TRUE 
                        WHERE table_name = ? AND field_name = ?
                    """, [table_name, field_name])
                    
            # Analyze table to optimize query plan
            cursor.execute(f"ANALYZE {self._quote_identifier(table_name)}")
            
            cursor.execute("COMMIT")
            
        except Exception as e:
            cursor.execute("ROLLBACK")
            raise e

    def to_pandas(self, sql: str, params: tuple = None) -> "pd.DataFrame":
        """Convert query results to Pandas DataFrame
        
        Args:
            sql: SQL statement
            params: Query parameters
            
        Returns:
            pd.DataFrame: Query results
        """
        cursor = self._get_connection().cursor()
        cursor.execute(sql, params)
        columns = [description[0] for description in cursor.description]
        
        chunk_size = 10000
        chunks = []
        while True:
            chunk = cursor.fetchmany(chunk_size)
            if not chunk:
                break
            chunks.append(pd.DataFrame(chunk, columns=columns))
        
        if not chunks:
            return pd.DataFrame(columns=columns)
        elif len(chunks) == 1:
            return chunks[0]
        
        df = pd.concat(chunks, ignore_index=True)
            
        return df

    def drop_column(self, column_name: str):
        """删除指定的列
        
        Args:
            column_name: 要删除的列名
        """
        if column_name == '_id':
            raise ValueError("Cannot drop _id column")
            
        table_name = self.current_table
        cursor = self._get_connection().cursor()
        
        try:
            cursor.execute("BEGIN TRANSACTION")
            
            # 获取当前表的所有列
            columns = self._get_table_columns(table_name)
            if column_name not in columns:
                raise ValueError(f"Column {column_name} does not exist")
                
            # 创建新表（不包含要删除的列）
            remaining_columns = [col for col in columns if col != column_name]
            columns_def = []
            for col in remaining_columns:
                col_type = cursor.execute("""
                    SELECT field_type FROM fields_meta 
                    WHERE table_name = ? AND field_name = ?
                """, [table_name, col]).fetchone()[0]
                if col == '_id':
                    columns_def.append(f"{self._quote_identifier(col)} {col_type} PRIMARY KEY")
                else:
                    columns_def.append(f"{self._quote_identifier(col)} {col_type}")
            
            temp_table = f"temp_{table_name}"
            cursor.execute(f"""
                CREATE TABLE {self._quote_identifier(temp_table)} (
                    {', '.join(columns_def)}
                )
            """)
            
            # 复制数据到新表
            cursor.execute(f"""
                INSERT INTO {self._quote_identifier(temp_table)} 
                SELECT {', '.join(self._quote_identifier(col) for col in remaining_columns)}
                FROM {self._quote_identifier(table_name)}
            """)
            
            # 删除旧表并重命名新表
            cursor.execute(f"DROP TABLE {self._quote_identifier(table_name)}")
            cursor.execute(f"""
                ALTER TABLE {self._quote_identifier(temp_table)} 
                RENAME TO {self._quote_identifier(table_name)}
            """)
            
            # 更新元数据
            cursor.execute("""
                DELETE FROM fields_meta 
                WHERE table_name = ? AND field_name = ?
            """, [table_name, column_name])
            
            cursor.execute("COMMIT")
            
        except Exception as e:
            cursor.execute("ROLLBACK")
            raise e

    def add_column(self, column_name: str, column_type: str):
        """添加新列
        
        Args:
            column_name: 新列的名称
            column_type: 新列的数据类型
        """
        if column_name == '_id':
            raise ValueError("Cannot add _id column")
            
        table_name = self.current_table
        cursor = self._get_connection().cursor()
        
        try:
            cursor.execute("BEGIN TRANSACTION")
            
            # 检查列是否已存在
            if self._column_exists(table_name, column_name):
                raise ValueError(f"Column {column_name} already exists")
            
            # 添加新列
            cursor.execute(f"""
                ALTER TABLE {self._quote_identifier(table_name)}
                ADD COLUMN {self._quote_identifier(column_name)} {column_type}
            """)
            
            # 更新元数据
            next_position = cursor.execute("""
                SELECT COALESCE(MAX(ordinal_position), 0) + 1
                FROM fields_meta
                WHERE table_name = ?
            """, [table_name]).fetchone()[0]
            
            cursor.execute("""
                INSERT INTO fields_meta (table_name, field_name, field_type, ordinal_position)
                VALUES (?, ?, ?, ?)
            """, [table_name, column_name, column_type, next_position])
            
            cursor.execute("COMMIT")
            
        except Exception as e:
            cursor.execute("ROLLBACK")
            raise e

    def rename_column(self, old_column_name: str, new_column_name: str):
        """重命名列
        
        Args:
            old_column_name: 原列名
            new_column_name: 新列名
        """
        if old_column_name == '_id':
            raise ValueError("Cannot rename _id column")
            
        table_name = self.current_table
        cursor = self._get_connection().cursor()
        
        try:
            cursor.execute("BEGIN TRANSACTION")
            
            # 检查原列是否存在
            if not self._column_exists(table_name, old_column_name):
                raise ValueError(f"Column {old_column_name} does not exist")
                
            # 检查新列名是否已存在
            if self._column_exists(table_name, new_column_name):
                raise ValueError(f"Column {new_column_name} already exists")
            
            # 获取列类型
            col_type = cursor.execute("""
                SELECT field_type FROM fields_meta 
                WHERE table_name = ? AND field_name = ?
            """, [table_name, old_column_name]).fetchone()[0]
            
            # 创建新表
            columns = self._get_table_columns(table_name)
            columns_def = []
            for col in columns:
                if col == old_column_name:
                    if col == '_id':
                        columns_def.append(f"{self._quote_identifier(new_column_name)} {col_type} PRIMARY KEY")
                    else:
                        columns_def.append(f"{self._quote_identifier(new_column_name)} {col_type}")
                else:
                    col_type = cursor.execute("""
                        SELECT field_type FROM fields_meta 
                        WHERE table_name = ? AND field_name = ?
                    """, [table_name, col]).fetchone()[0]
                    if col == '_id':
                        columns_def.append(f"{self._quote_identifier(col)} {col_type} PRIMARY KEY")
                    else:
                        columns_def.append(f"{self._quote_identifier(col)} {col_type}")
            
            temp_table = f"temp_{table_name}"
            cursor.execute(f"""
                CREATE TABLE {self._quote_identifier(temp_table)} (
                    {', '.join(columns_def)}
                )
            """)
            
            # 复制数据到新表
            select_columns = [
                f"{self._quote_identifier(col)} AS {self._quote_identifier(new_column_name)}" 
                if col == old_column_name else self._quote_identifier(col)
                for col in columns
            ]
            cursor.execute(f"""
                INSERT INTO {self._quote_identifier(temp_table)}
                SELECT {', '.join(select_columns)}
                FROM {self._quote_identifier(table_name)}
            """)
            
            # 删除旧表并重命名新表
            cursor.execute(f"DROP TABLE {self._quote_identifier(table_name)}")
            cursor.execute(f"""
                ALTER TABLE {self._quote_identifier(temp_table)}
                RENAME TO {self._quote_identifier(table_name)}
            """)
            
            # 更新元数据
            cursor.execute("""
                UPDATE fields_meta 
                SET field_name = ?
                WHERE table_name = ? AND field_name = ?
            """, [new_column_name, table_name, old_column_name])
            
            cursor.execute("COMMIT")
            
        except Exception as e:
            cursor.execute("ROLLBACK")
            raise e

    def get_column_dtype(self, column_name: str) -> str:
        """获取列的数据类型
        
        Args:
            column_name: 列名
            
        Returns:
            列的数据类型
        """
        table_name = self.current_table
        cursor = self._get_connection().cursor()
        
        result = cursor.execute("""
            SELECT field_type 
            FROM fields_meta 
            WHERE table_name = ? AND field_name = ?
        """, [table_name, column_name]).fetchone()
        
        if result is None:
            raise ValueError(f"Column {column_name} does not exist")
            
        return result[0]

    def _column_exists(self, table_name: str, column_name: str) -> bool:
        """检查列是否存在
        
        Args:
            table_name: 表名
            column_name: 列名
            
        Returns:
            列是否存在
        """
        cursor = self._get_connection().cursor()
        result = cursor.execute("""
            SELECT 1 
            FROM fields_meta 
            WHERE table_name = ? AND field_name = ?
        """, [table_name, column_name]).fetchone()
        return result is not None

    def _get_table_columns(self, table_name: str) -> List[str]:
        """获取表的所有列名
        
        Args:
            table_name: 表名
            
        Returns:
            列名列表
        """
        cursor = self._get_connection().cursor()
        result = cursor.execute("""
            SELECT field_name 
            FROM fields_meta 
            WHERE table_name = ?
            ORDER BY ordinal_position
        """, [table_name]).fetchall()
        return [row[0] for row in result]
