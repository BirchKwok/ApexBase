import time
import duckdb
import orjson
from typing import Dict, List, Optional, Union
from pathlib import Path
import threading
import json
import pandas as pd

from .id_manager import IDManager
from .base import BaseStorage, BaseSchema


class DuckDBSchema(BaseSchema):
    """DuckDB schema."""

    def __init__(self, schema: dict = None):
        """Initialize the Schema
        
        Args:
            schema: The optional schema dictionary, format is {'columns': {'column_name': 'column_type'}}
        """
        self.schema = schema or {'columns': {'_id': 'BIGINT'}}
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
            self.schema['columns']['_id'] = 'BIGINT'

    def to_dict(self):
        """Convert to a dictionary format"""
        return self.schema
    
    def drop_column(self, column_name: str):
        """Drop a column
        
        Args:
            column_name: The name of the column to drop
        """
        if column_name == '_id':
            raise ValueError("Cannot drop _id column")
        if column_name in self.schema['columns']:
            del self.schema['columns'][column_name]

    def add_column(self, column_name: str, column_type: str):
        """Add a column
        
        Args:
            column_name: The name of the column to add
            column_type: The type of the column to add
        """
        if column_name in self.schema['columns']:
            raise ValueError(f"Column {column_name} already exists")
        self.schema['columns'][column_name] = column_type

    def rename_column(self, old_column_name: str, new_column_name: str):
        """Rename a column
        
        Args:
            old_column_name: The old name of the column
            new_column_name: The new name of the column
        """
        if old_column_name == '_id':
            raise ValueError("Cannot rename _id column")
        if old_column_name not in self.schema['columns']:
            raise ValueError(f"Column {old_column_name} does not exist")
        if new_column_name in self.schema['columns']:
            raise ValueError(f"Column {new_column_name} already exists")
        self.schema['columns'][new_column_name] = self.schema['columns'].pop(old_column_name)

    def modify_column(self, column_name: str, column_type: str):
        """Modify the type of a column
        
        Args:
            column_name: The name of the column to modify
            column_type: The type of the column to modify
        """
        if column_name == '_id':
            raise ValueError("Cannot modify _id column type")
        if column_name not in self.schema['columns']:
            raise ValueError(f"Column {column_name} does not exist")
        self.schema['columns'][column_name] = column_type

    def get_column_type(self, column_name: str) -> str:
        """Get the type of a column
        
        Args:
            column_name: The name of the column
        """
        if column_name not in self.schema['columns']:
            raise ValueError(f"Column {column_name} does not exist")
        return self.schema['columns'][column_name]

    def has_column(self, column_name: str) -> bool:
        """Check if a column exists
        
        Args:
            column_name: The name of the column
        """
        return column_name in self.schema['columns']

    def get_columns(self) -> List[str]:
        """Get all column names
        
        Returns:
            The list of column names
        """
        return list(self.schema['columns'].keys())

    def update_from_data(self, data: dict):
        """Update the schema from data
        
        Args:
            data: The data dictionary
        """
        for column_name, value in data.items():
            if column_name != '_id' and column_name not in self.schema['columns']:
                column_type = self._infer_column_type(value)
                self.add_column(column_name, column_type)

    def _infer_column_type(self, value) -> str:
        """Infer the type of a column
        
        Args:
            value: The value
            
        Returns:
            The type of the column
        """
        if isinstance(value, bool):
            return "BOOLEAN"
        elif isinstance(value, int):
            return "BIGINT"
        elif isinstance(value, float):
            return "DOUBLE"
        elif isinstance(value, (str, dict, list)):
            return "VARCHAR"
        elif pd.isna(value):
            return "VARCHAR"  # 对于空值，默认使用VARCHAR
        else:
            return "VARCHAR"  # 对于未知类型，默认使用VARCHAR


class DuckDBStorage(BaseStorage):
    """DuckDB implementation of the storage backend with columnar storage."""
    
    def __init__(self, filepath=None, batch_size: int = 1000, 
                 enable_cache: bool = True, cache_size: int = 10000):
        """Initialize the DuckDB storage
        
        Args:
            filepath: The path to the database file
            batch_size: The size of the batch
        """
        if filepath is None:
            raise ValueError("You must provide a file path.")

        self.filepath = Path(filepath)
        self.filepath.parent.mkdir(parents=True, exist_ok=True)

        self.batch_size = batch_size
        self._lock = threading.Lock()
        self.current_table = "default"
        self.enable_cache = enable_cache
        self.cache_size = cache_size
        self._cache = []

        self.conn = duckdb.connect(str(self.filepath))
        self._initialize_database()

        self.id_manager = IDManager(self)

        self._last_modified_time = None

    def _initialize_database(self):
        """Initialize the database, create necessary system tables"""
        cursor = self.conn.cursor()
        
        # Set optimization parameters
        cursor.execute("PRAGMA memory_limit='4GB'")
        cursor.execute("PRAGMA threads=4")
        
        # Create metadata table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS tables_meta (
                table_name VARCHAR PRIMARY KEY,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                schema JSON  -- Store the field definitions of the table
            )
        """)
        
        # Create fields metadata table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS fields_meta (
                table_name VARCHAR,
                field_name VARCHAR,
                field_type VARCHAR,
                is_indexed BOOLEAN DEFAULT FALSE,
                ordinal_position INTEGER,
                PRIMARY KEY (table_name, field_name)
            )
        """)
        
        # If the default table does not exist, create it
        if not self._table_exists("default"):
            self.create_table("default")

    def _get_table_name(self, table_name: str = None) -> str:
        """Gets the actual table name."""
        return table_name if table_name is not None else self.current_table

    def use_table(self, table_name: str):
        """Switches the current table."""
        with self._lock:
            if not self._table_exists(table_name):
                raise ValueError(f"Table '{table_name}' does not exist")
            self.current_table = table_name

    def create_schema(self, table_name: str, schema: DuckDBSchema):
        """Create the schema of a table
        
        Args:
            table_name: The name of the table
            schema: The schema object
        """
        with self._lock:
            if self._table_exists(table_name):
                raise ValueError(f"Table '{table_name}' already exists")

            cursor = self.conn.cursor()
            cursor.execute("BEGIN TRANSACTION")
            try:
                # Create table
                columns = []
                for col_name, col_type in schema.to_dict()['columns'].items():
                    if col_name == '_id':
                        columns.append(f"{self._quote_identifier(col_name)} {col_type} PRIMARY KEY")
                    else:
                        columns.append(f"{self._quote_identifier(col_name)} {col_type}")
                
                create_sql = f"""
                    CREATE TABLE {self._quote_identifier(table_name)} (
                        {', '.join(columns)}
                    )
                """
                cursor.execute(create_sql)
                
                # Update metadata
                cursor.execute(
                    "INSERT INTO tables_meta (table_name, schema) VALUES (?, ?)",
                    [table_name, orjson.dumps(schema.to_dict()).decode('utf-8')]
                )
                
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
                
            except Exception as e:
                cursor.execute("ROLLBACK")
                raise e

    def create_table(self, table_name: str):
        """Create a new table, using the default schema
        
        Args:
            table_name: The name of the table
        """
        schema = DuckDBSchema()  # Use the default schema
        self.create_schema(table_name, schema)

    def drop_table(self, table_name: str):
        """Drops a table."""
        if not self._table_exists(table_name):
            return

        if table_name == "default":
            raise ValueError("Cannot drop the default table")

        cursor = self.conn.cursor()
        try:
            cursor.execute("BEGIN TRANSACTION")
            
            cursor.execute(f"DROP TABLE IF EXISTS {self._quote_identifier(table_name)}")
            cursor.execute("DELETE FROM tables_meta WHERE table_name = ?", [table_name])
            
            cursor.execute("COMMIT")
            
            if self.current_table == table_name:
                self.use_table("default")
            
        except Exception as e:
            cursor.execute("ROLLBACK")
            raise e
        
        finally:
            self._last_modified_time = time.time()

    def list_tables(self) -> List[str]:
        """Lists all tables."""
        cursor = self.conn.cursor()
        result = cursor.execute("SELECT table_name FROM tables_meta ORDER BY table_name")
        return [row[0] for row in result.fetchall()]

    def _table_exists(self, table_name: str) -> bool:
        """Checks if a table exists."""
        cursor = self.conn.cursor()
        try:
            cursor.execute(f"SELECT 1 FROM {self._quote_identifier(table_name)} LIMIT 1")
            return True
        except:
            return False

    def _quote_identifier(self, identifier: str) -> str:
        """Correctly escapes DuckDB identifiers."""
        return f'"{identifier}"'

    def _get_column_type(self, value) -> str:
        """Infer the DuckDB column type from the value."""
        if isinstance(value, bool):
            return "BOOLEAN"
        elif isinstance(value, int):
            return "BIGINT"
        elif isinstance(value, float):
            return "DOUBLE"
        elif isinstance(value, (str, dict, list)):
            return "VARCHAR"
        elif pd.isna(value):
            return "VARCHAR"  # For empty values, default to VARCHAR
        else:
            return "VARCHAR"  # For unknown types, default to VARCHAR

    def _create_table_if_not_exists(self, table_name: str, data: Union[dict, pd.DataFrame]):
        """Create or update a table based on the data, supports dynamic fields
        
        Args:
            table_name: The name of the table
            data: The data (dictionary or DataFrame)
        """
        if isinstance(data, dict):
            df = pd.DataFrame([data])
        else:
            df = data.copy()
            
        # If the table does not exist, create it
        if not self._table_exists(table_name):
            schema = DuckDBSchema()
            for col in df.columns:
                if col != '_id':
                    schema.add_column(col, self._get_duckdb_type(df[col].dtype))
            self.create_schema(table_name, schema)
            return
        
        # Get existing columns
        existing_columns = set(self._get_table_columns(table_name))
        # Keep the original order of new columns list
        columns = df.columns

        new_columns = [col for col in columns if col != '_id' and col not in existing_columns]
        
        if not new_columns:
            return
            
        # Add new columns
        with self._lock:
            cursor = self.conn.cursor()
            cursor.execute("BEGIN TRANSACTION")
            try:
                # Get the current schema
                result = cursor.execute(
                    "SELECT schema FROM tables_meta WHERE table_name = ?",
                    [table_name]
                ).fetchone()
                current_schema = DuckDBSchema(orjson.loads(result[0]))
                
                next_position = cursor.execute("""
                    SELECT COALESCE(MAX(ordinal_position), 0) + 1
                    FROM fields_meta
                    WHERE table_name = ?
                """, [table_name]).fetchone()[0]
                
                for col in new_columns:
                    sql_type = self._get_duckdb_type(df[col].dtype)
                    cursor.execute(f"""
                        ALTER TABLE {self._quote_identifier(table_name)}
                        ADD COLUMN {self._quote_identifier(col)} {sql_type}
                    """)
                    
                    # Update schema
                    current_schema.add_column(col, sql_type)
                    
                    # Add field to fields_meta table
                    cursor.execute("""
                        INSERT INTO fields_meta (table_name, field_name, field_type, ordinal_position)
                        VALUES (?, ?, ?, ?)
                        ON CONFLICT (table_name, field_name) DO UPDATE SET 
                            field_type = EXCLUDED.field_type,
                            ordinal_position = EXCLUDED.ordinal_position
                    """, [table_name, col, sql_type, next_position])
                    next_position += 1
                
                # Update the schema in tables_meta
                cursor.execute(
                    "UPDATE tables_meta SET schema = ? WHERE table_name = ?",
                    [orjson.dumps(current_schema.to_dict()).decode('utf-8'), table_name]
                )
                
                cursor.execute("COMMIT")
            except Exception as e:
                cursor.execute("ROLLBACK")
                raise e

    def _get_duckdb_type(self, pandas_type) -> str:
        """Convert the Pandas data type to the DuckDB data type"""
        type_str = str(pandas_type)
        if 'int' in type_str:
            return 'BIGINT'
        elif 'float' in type_str:
            return 'DOUBLE'
        elif 'bool' in type_str:
            return 'BOOLEAN'
        elif 'datetime' in type_str:
            return 'TIMESTAMP'
        else:
            return 'VARCHAR'

    def store(self, data: Union[dict, pd.DataFrame], table_name: str = None) -> Union[int, List[int]]:
        """Store data
        
        Args:
            data: The data to store, can be a dictionary or DataFrame
            table_name: The name of the table
            
        Returns:
            The record ID or ID list
        """
        table_name = self._get_table_name(table_name)
        
        if isinstance(data, dict):
            # Preprocess JSON fields
            processed_data = {}
            for k, v in data.items():
                if isinstance(v, (dict, list)):
                    processed_data[k] = json.dumps(v)
                else:
                    processed_data[k] = v
            df = [processed_data]
        elif isinstance(data, pd.DataFrame):
            df = data.copy()
            # Preprocess JSON fields in the DataFrame
            for col in df.columns:
                if df[col].apply(lambda x: isinstance(x, (dict, list))).any():
                    df[col] = df[col].apply(lambda x: json.dumps(x) if isinstance(x, (dict, list)) else x)
        
        # If the data is multiple rows, use batch_store
        if len(df) > 1:
            return self.batch_store(df, table_name)
        elif self.enable_cache and self.id_manager.get_next_id(table_name) != 1:
            with self._lock:
                if not isinstance(df, pd.DataFrame):
                    self._cache.append(df[0])
                else:
                    self._cache.append(df.to_dict(orient='records')[0])

                self.id_manager.auto_increment(table_name)
            if len(self._cache) >= self.cache_size:
                self.flush_cache()
            
            with self._lock:
                return self.id_manager.current_id(table_name)
        
        # Ensure the table exists and update the schema
        self._create_table_if_not_exists(table_name, df[0])
        
        if not isinstance(df, pd.DataFrame):
            df = pd.DataFrame(df)
            
        with self._lock:
            cursor = self.conn.cursor()
            cursor.execute("BEGIN TRANSACTION")
            try:
                # Get the next ID
                result = cursor.execute(f"""
                    SELECT COALESCE(MAX(_id), 0) + 1 
                    FROM {self._quote_identifier(table_name)}
                """).fetchone()
                next_id = result[0] if result else 1
                
                # Add ID column
                if '_id' in df.columns:
                    df = df.drop('_id', axis=1)
                df.insert(0, '_id', next_id)
                
                # Get the column names
                columns = [f'"{str(col)}"' for col in df.columns]
                
                # Insert data
                cursor.register('df_view', df)
                insert_sql = f"""
                    INSERT INTO {self._quote_identifier(table_name)} ({', '.join(columns)})
                    SELECT {', '.join(columns)} FROM df_view
                """
                cursor.execute(insert_sql)
                cursor.unregister('df_view')
                
                # Create indexes
                self._create_indexes(table_name)
                
                cursor.execute("COMMIT")
                return next_id
                
            except Exception as e:
                cursor.execute("ROLLBACK")
                raise e
            
            finally:
                self.id_manager.reset_last_id(table_name)
                self._last_modified_time = time.time()
    
    def _get_next_id(self, table_name: str) -> int:
        """Get the next ID"""
        cursor = self.conn.cursor()
        result = cursor.execute(f"""
            SELECT COALESCE(MAX(_id), 0) + 1 
            FROM {self._quote_identifier(table_name)}
        """).fetchone()
        return result[0] if result else 1
    
    def flush_cache(self):
        """Flush the cache"""
        if self._cache is not None and len(self._cache) > 0:
            self.batch_store(self._cache)
            self._cache = []

    def batch_store(self, data_list: List[dict], table_name: str = None) -> List[int]:
        """Batch store records
        
        Args:
            data_list: The list of records to store
            table_name: The name of the table
            
        Returns:
            The list of record IDs
        """
        if not data_list:
            return []
            
        table_name = self._get_table_name(table_name)
        
        # Preprocess: Get all fields and types
        all_fields = {'_id': 'BIGINT'}  # Ensure _id field is included
        field_order = []  # Maintain field addition order
        for data in data_list:
            for key, value in data.items():
                if key not in all_fields and key != '_id':  # Skip _id field
                    all_fields[key] = self._infer_field_type(value)
                    if key not in field_order:
                        field_order.append(key)
        
        # Preprocess data: Serialize complex types
        processed_data = []
        for data in data_list:
            processed_record = {}
            for key, value in data.items():
                if key != '_id':  # Skip _id field
                    if isinstance(value, (list, dict)):
                        processed_record[key] = json.dumps(value)
                    else:
                        processed_record[key] = value
            processed_data.append(processed_record)
        
        with self._lock:
            cursor = self.conn.cursor()
            try:
                cursor.execute("BEGIN TRANSACTION")
                
                # 1. Create all necessary columns at once
                if not self._table_exists(table_name):
                    self.create_table(table_name)
                
                existing_columns = set(self._get_table_columns(table_name))
                new_columns = [col for col in field_order if col not in existing_columns]
                
                if new_columns:
                    next_position = cursor.execute("""
                        SELECT COALESCE(MAX(ordinal_position), 0) + 1
                        FROM fields_meta
                        WHERE table_name = ?
                    """, [table_name]).fetchone()[0]
                    
                    # Batch add new columns
                    for col in new_columns:
                        field_type = all_fields[col]
                        cursor.execute(f"""
                            ALTER TABLE {self._quote_identifier(table_name)}
                            ADD COLUMN IF NOT EXISTS {self._quote_identifier(col)} {field_type}
                        """)
                        
                        # Add field to fields_meta table
                        cursor.execute("""
                            INSERT INTO fields_meta (table_name, field_name, field_type, ordinal_position)
                            VALUES (?, ?, ?, ?)
                            ON CONFLICT (table_name, field_name) DO UPDATE SET 
                                field_type = EXCLUDED.field_type,
                                ordinal_position = EXCLUDED.ordinal_position
                        """, [table_name, col, field_type, next_position])
                    
                    # Update schema information
                    schema = {
                        'columns': all_fields  # Use all_fields directly, it already contains all field types
                    }
                    cursor.execute(
                        "UPDATE tables_meta SET schema = ? WHERE table_name = ?",
                        [orjson.dumps(schema).decode('utf-8'), table_name]
                    )
                
                # 2. Get the starting ID
                result = cursor.execute(f"""
                    SELECT COALESCE(MAX(_id), 0) + 1 
                    FROM {self._quote_identifier(table_name)}
                """).fetchone()
                next_id = result[0] if result else 1
                
                # 3. Process in batches
                batch_size = self.batch_size
                all_ids = []
                
                for i in range(0, len(processed_data), batch_size):
                    batch = processed_data[i:i + batch_size]
                    
                    # Create the DataFrame for the current batch
                    df = pd.DataFrame(batch)
                    if '_id' in df.columns:
                        df = df.drop('_id', axis=1)
                    
                    # Add ID column
                    current_ids = range(next_id + i, next_id + i + len(batch))
                    df.insert(0, '_id', current_ids)
                    all_ids.extend(current_ids)
                    
                    # Get the column names
                    columns = [f'"{str(col)}"' for col in df.columns]
                    
                    # Use the DuckDB DataFrame interface to batch insert
                    cursor.register('df_view', df)
                    insert_sql = f"""
                        INSERT INTO {self._quote_identifier(table_name)} ({', '.join(columns)})
                        SELECT {', '.join(columns)} FROM df_view
                    """
                    cursor.execute(insert_sql)
                    cursor.unregister('df_view')
                
                cursor.execute("COMMIT")
                return all_ids
                
            except Exception as e:
                cursor.execute("ROLLBACK")
                raise e
            
            finally:
                self.id_manager.reset_last_id(table_name)
                self._last_modified_time = time.time()

    def _get_table_columns(self, table_name: str) -> List[str]:
        """Get the column names of the table."""
        cursor = self.conn.cursor()
        cursor.execute(f"DESCRIBE {self._quote_identifier(table_name)}")
        columns = cursor.fetchall()
        return [col[0] for col in columns]

    def retrieve(self, id_: int) -> Optional[dict]:
        """Get a single record
        
        Args:
            id_: The record ID
            
        Returns:
            The record data dictionary
        """
        table_name = self._get_table_name()
        cursor = self.conn.cursor()
        
        # Get all column names
        columns = self._get_table_columns(table_name)
        quoted_columns = [f'"{col}"' for col in columns]
        
        result = cursor.execute(f"""
            SELECT {', '.join(quoted_columns)}
            FROM {self._quote_identifier(table_name)}
            WHERE _id = ?
        """, [id_]).fetchone()
        
        if result:
            data = {}
            for i, col in enumerate(columns):
                value = result[i]
                if value is not None:
                    if col != '_id' and isinstance(value, str):
                        try:
                            # Try to parse the JSON string
                            data[col] = json.loads(value)
                        except json.JSONDecodeError:
                            data[col] = value
                    else:
                        data[col] = value
            return data
        return None

    def retrieve_many(self, ids: List[int]) -> List[dict]:
        """Get multiple records
        
        Args:
            ids: The list of record IDs
            
        Returns:
            The list of record data dictionaries
        """
        if not ids:
            return []
            
        table_name = self._get_table_name()
        cursor = self.conn.cursor()
        
        # Get all column names
        columns = self._get_table_columns(table_name)
        quoted_columns = [f'"{col}"' for col in columns]
        
        placeholders = ','.join(['?' for _ in ids])
        results = cursor.execute(f"""
            SELECT {', '.join(quoted_columns)}
            FROM {self._quote_identifier(table_name)}
            WHERE _id IN ({placeholders})
            ORDER BY _id
        """, ids).fetchall()
        
        data_list = []
        for row in results:
            data = {}
            for i, col in enumerate(columns):
                value = row[i]
                if value is not None:
                    if col != '_id' and isinstance(value, str):
                        try:
                            # Try to parse the JSON string
                            data[col] = json.loads(value)
                        except json.JSONDecodeError:
                            data[col] = value
                    else:
                        data[col] = value
            data_list.append(data)
        
        return data_list

    def delete(self, id_: int) -> bool:
        """Delete a record
        
        Args:
            id_: The record ID
            
        Returns:
            bool: Whether the deletion is successful
        """
        table_name = self._get_table_name()
        
        with self._lock:
            cursor = self.conn.cursor()
            cursor.execute("BEGIN TRANSACTION")
            try:
                # Check if the record exists
                exists = cursor.execute(
                    f"SELECT 1 FROM {self._quote_identifier(table_name)} WHERE _id = ?",
                    [id_]
                ).fetchone()
                if not exists:
                    cursor.execute("ROLLBACK")
                    return False
                
                # Execute the deletion
                cursor.execute(f"""
                    DELETE FROM {self._quote_identifier(table_name)}
                    WHERE _id = ?
                """, [id_])
                
                cursor.execute("COMMIT")
                return True
                
            except Exception as e:
                cursor.execute("ROLLBACK")
                raise e
            
            finally:
                self.id_manager.reset_last_id(table_name)
                self._last_modified_time = time.time()

    def batch_delete(self, ids: List[int]) -> bool:
        """Batch delete records
        
        Args:
            ids: The list of record IDs
            
        Returns:
            bool: Whether the deletion is successful
        """
        if not ids:
            return True
            
        table_name = self._get_table_name()
        
        with self._lock:
            cursor = self.conn.cursor()
            cursor.execute("BEGIN TRANSACTION")
            try:
                # Check if the records exist
                placeholders = ','.join(['?' for _ in ids])
                exists = cursor.execute(f"""
                    SELECT COUNT(*) 
                    FROM {self._quote_identifier(table_name)} 
                    WHERE _id IN ({placeholders})
                """, ids).fetchone()[0]
                
                print(f"DuckDB: Checking existence for IDs {ids}, found {exists} records")
                if exists != len(ids):
                    cursor.execute("ROLLBACK")
                    return False
                
                # Execute the deletion
                cursor.execute(f"""
                    DELETE FROM {self._quote_identifier(table_name)}
                    WHERE _id IN ({placeholders})
                """, ids)
                
                cursor.execute("COMMIT")
                return True
                
            except Exception as e:
                cursor.execute("ROLLBACK")
                raise e
            
            finally:
                self.id_manager.reset_last_id(table_name)
                self._last_modified_time = time.time()

    def replace(self, id_: int, data: dict) -> bool:
        """Replace a single record
    
        Args:
            id_: The record ID
            data: The new record data
            
        Returns:
            bool: Whether the replacement is successful
        """
        table_name = self._get_table_name()
        
        with self._lock:
            cursor = self.conn.cursor()
            
            # Check if the record exists
            exists = cursor.execute(
                f"SELECT 1 FROM {self._quote_identifier(table_name)} WHERE _id = ?",
                [id_]
            ).fetchone()
            
            if not exists:
                return False

            # Ensure all fields exist
            update_data = {k: v for k, v in data.items() if k != '_id'}  # Explicitly exclude _id field
            self._ensure_fields_exist(update_data, table_name, cursor)
            
            # Prepare update data
            set_clauses = []
            params = []
            
            for field, value in update_data.items():
                quoted_field = self._quote_identifier(field)
                if isinstance(value, (dict, list)):
                    set_clauses.append(f"{quoted_field} = ?")
                    params.append(json.dumps(value))
                else:
                    set_clauses.append(f"{quoted_field} = ?")
                    params.append(value)

            if set_clauses:
                params.append(id_)  # Add the WHERE condition parameter
                update_sql = f"""
                    UPDATE {self._quote_identifier(table_name)}
                    SET {', '.join(set_clauses)}
                    WHERE _id = ?
                """
                try:
                    cursor.execute(update_sql, params)
                except Exception as e:
                    # If the UPDATE fails, try DELETE + INSERT
                    cursor.execute(f"""
                        DELETE FROM {self._quote_identifier(table_name)}
                        WHERE _id = ?
                    """, [id_])
                    
                    # Prepare all fields
                    all_fields = self._get_table_columns(table_name)
                    current_data = cursor.execute(f"""
                        SELECT * FROM {self._quote_identifier(table_name)}
                        WHERE _id = ?
                    """, [id_]).fetchone()
                    
                    # Build the complete field value list
                    columns = []
                    values = []
                    for field in all_fields:
                        columns.append(self._quote_identifier(field))
                        if field == '_id':
                            values.append(id_)
                        elif field in update_data:
                            value = update_data[field]
                            if isinstance(value, (dict, list)):
                                values.append(json.dumps(value))
                            else:
                                values.append(value)
                        else:
                            idx = all_fields.index(field)
                            values.append(current_data[idx] if current_data else None)
                    
                    # Insert a new record
                    placeholders = ['?' for _ in columns]
                    insert_sql = f"""
                        INSERT INTO {self._quote_identifier(table_name)}
                        ({', '.join(columns)})
                        VALUES ({', '.join(placeholders)})
                    """
                    cursor.execute(insert_sql, values)

            self._last_modified_time = time.time()
            
            return True

    def batch_replace(self, data_dict: Dict[int, dict]) -> List[int]:
        """Replaces multiple records by IDs."""
        if not data_dict:
            return []
            
        table_name = self._get_table_name()
        success_ids = []
        
        with self._lock:
            cursor = self.conn.cursor()
            
            # Check if the records exist
            ids = list(data_dict.keys())
            placeholders = ','.join(['?' for _ in ids])
            existing_ids = cursor.execute(f"""
                SELECT _id FROM {self._quote_identifier(table_name)}
                WHERE _id IN ({placeholders})
            """, ids).fetchall()
            existing_ids = {row[0] for row in existing_ids}
            
            # Only update existing records
            for id_ in existing_ids:
                data = data_dict[id_]
                # Ensure all fields exist
                update_data = {k: v for k, v in data.items() if k != '_id'}
                self._ensure_fields_exist(update_data, table_name, cursor)
                
                # Prepare update data
                set_clauses = []
                params = []
                
                for field, value in update_data.items():
                    quoted_field = self._quote_identifier(field)
                    if isinstance(value, (dict, list)):
                        set_clauses.append(f"{quoted_field} = ?")
                        params.append(json.dumps(value))
                    else:
                        set_clauses.append(f"{quoted_field} = ?")
                        params.append(value)

                if set_clauses:
                    params.append(id_)  # Add the WHERE condition parameter
                    update_sql = f"""
                        UPDATE {self._quote_identifier(table_name)}
                        SET {', '.join(set_clauses)}
                        WHERE _id = ?
                    """
                    try:
                        cursor.execute(update_sql, params)
                    except Exception as e:
                        # If the UPDATE fails, try DELETE + INSERT
                        cursor.execute(f"""
                            DELETE FROM {self._quote_identifier(table_name)}
                            WHERE _id = ?
                        """, [id_])
                        
                        # Prepare all fields
                        all_fields = self._get_table_columns(table_name)
                        current_data = cursor.execute(f"""
                            SELECT * FROM {self._quote_identifier(table_name)}
                            WHERE _id = ?
                        """, [id_]).fetchone()
                        
                        # Build the complete field value list
                        columns = []
                        values = []
                        for field in all_fields:
                            columns.append(self._quote_identifier(field))
                            if field == '_id':
                                values.append(id_)
                            elif field in update_data:
                                value = update_data[field]
                                if isinstance(value, (dict, list)):
                                    values.append(json.dumps(value))
                                else:
                                    values.append(value)
                            else:
                                idx = all_fields.index(field)
                                values.append(current_data[idx] if current_data else None)
                        
                        # Insert a new record
                        placeholders = ['?' for _ in columns]
                        insert_sql = f"""
                            INSERT INTO {self._quote_identifier(table_name)}
                            ({', '.join(columns)})
                            VALUES ({', '.join(placeholders)})
                        """
                        cursor.execute(insert_sql, values)
                    
                    success_ids.append(id_)

            self._last_modified_time = time.time()
            return success_ids

    def query(self, sql: str, params: tuple = None) -> List[tuple]:
        """Execute a custom SQL query, supports parallel execution
        
        Args:
            sql: SQL statement
            params: Query parameters
            
        Returns:
            Query results
        """
        cursor = self.conn.cursor()
        
        # Add parallel query support
        cursor.execute("PRAGMA threads=4")
        
        # If it is a LIKE query, add an index hint
        if 'LIKE' in sql.upper():
            sql = f"/* use_index */ {sql}"
        
        return cursor.execute(sql, params).fetchall()

    def close(self):
        """Closes the database connection."""
        if self.enable_cache:
            self.flush_cache()

        if hasattr(self, 'conn'):
            self.conn.close()

    def _infer_field_type(self, value) -> str:
        """Infer the field type
        
        Args:
            value: Field value
            
        Returns:
            Field type
        """
        if value is None:
            return "VARCHAR"
        elif isinstance(value, bool):
            return "BOOLEAN"
        elif isinstance(value, int):
            return "BIGINT"
        elif isinstance(value, float):
            return "DOUBLE"
        elif isinstance(value, (list, dict)):
            return "JSON"
        elif isinstance(value, str):
            if len(value) > 255:
                return "TEXT"
            return "VARCHAR"
        else:
            return "VARCHAR"

    def list_fields(self, table_name: str = None) -> List[str]:
        """Get all fields of the table
        
        Args:
            table_name: Table name
            
        Returns:
            Field list
        """
        table_name = self._get_table_name(table_name)
        cursor = self.conn.cursor()
        
        # Get all fields sorted by ordinal_position
        result = cursor.execute("""
            SELECT field_name 
            FROM fields_meta 
            WHERE table_name = ? 
            ORDER BY ordinal_position
        """, [table_name]).fetchall()
        
        return [row[0] for row in result]

    def _create_indexes(self, table_name: str):
        """Create necessary indexes for the table"""
        cursor = self.conn.cursor()
        
        # Get the fields that need to be indexed
        fields = cursor.execute("""
            SELECT field_name, field_type 
            FROM fields_meta 
            WHERE table_name = ? AND is_indexed = FALSE
        """, [table_name]).fetchall()
        
        for field_name, field_type in fields:
            # Create an index for VARCHAR fields
            if field_type == 'VARCHAR':
                index_name = f"idx_{table_name}_{field_name}"
                cursor.execute(f"""
                    CREATE INDEX IF NOT EXISTS {index_name}
                    ON {self._quote_identifier(table_name)} ({self._quote_identifier(field_name)})
                """)
                
                # Update the index status
                cursor.execute("""
                    UPDATE fields_meta 
                    SET is_indexed = TRUE 
                    WHERE table_name = ? AND field_name = ?
                """, [table_name, field_name])

    def to_pandas(self, sql: str, params: tuple = None) -> "pd.DataFrame":
        """Convert the query result directly to a DataFrame
        
        Args:
            sql: SQL statement
            params: Query parameters
            
        Returns:
            DataFrame object
        """
        cursor = self.conn.cursor()
        
        # Get the field names
        fields = self.list_fields()
        field_list = ','.join(
            f'CAST({self._quote_identifier(f)} AS TEXT) AS {self._quote_identifier(f)}'
            for f in fields
        )
        
        # Build an optimized query
        optimized_sql = f"""
            WITH result AS (
                {sql}
            )
            SELECT {field_list}
            FROM result
        """
        
        # Use the native DuckDB DataFrame conversion
        return cursor.execute(optimized_sql, params).df()

    def _create_temp_table(self, table_name: str, suffix: str = None) -> str:
        """Create a temporary table and return the table name"""
        temp_name = f"temp_{table_name}"
        if suffix:
            temp_name = f"{temp_name}_{suffix}"
        temp_name = self._quote_identifier(temp_name)
        return temp_name

    def count_rows(self, table_name: str = None) -> int:
        """Get the number of records in the table
        
        Args:
            table_name: Table name
            
        Returns:
            Number of records
        """
        table_name = self._get_table_name(table_name)
        cursor = self.conn.cursor()
        
        # If there is data in the cache, it needs to be included in the count
        cache_count = len(self._cache) if self.enable_cache else 0
        
        result = cursor.execute(f"""
            SELECT COUNT(*) 
            FROM {self._quote_identifier(table_name)}
        """).fetchone()
        return result[0] + cache_count if result else cache_count

    def optimize(self):
        """Optimize the database performance"""
        table_name = self._get_table_name()
        cursor = self.conn.cursor()
        
        try:
            # DuckDB optimization operations
            cursor.execute("PRAGMA memory_limit='4GB'")
            cursor.execute("PRAGMA threads=4")
            cursor.execute("PRAGMA force_compression='none'")
            cursor.execute("PRAGMA checkpoint_threshold='1GB'")
            
            # Analyze the table to optimize the query plan
            cursor.execute(f"ANALYZE {self._quote_identifier(table_name)}")
            
        except Exception as e:
            raise ValueError(f"Failed to optimize database: {str(e)}")

    def _ensure_fields_exist(self, data: dict, table_name: str, cursor):
        """Ensure all fields exist (remove transaction management)"""
        # Get existing field metadata
        existing_fields = cursor.execute(
            "SELECT field_name FROM fields_meta WHERE table_name = ?",
            [table_name]
        ).fetchall()
        existing_fields = {row[0] for row in existing_fields}
        
        # Add new fields to the table and metadata
        for field in data.keys():
            if field == '_id':
                continue
            if field not in existing_fields:
                field_type = self._infer_field_type(data[field])
                # Add fields to the table
                cursor.execute(
                    f"ALTER TABLE {self._quote_identifier(table_name)} "
                    f"ADD COLUMN {self._quote_identifier(field)} {field_type}"
                )
                # Update the metadata table
                cursor.execute(
                    "INSERT INTO fields_meta (table_name, field_name, field_type) "
                    "VALUES (?, ?, ?) "
                    "ON CONFLICT (table_name, field_name) DO UPDATE SET "
                    "field_type = EXCLUDED.field_type",
                    [table_name, field, field_type]
                )
