import time
import duckdb
import orjson
from typing import Dict, List, Optional, Union
from pathlib import Path
import threading
import json
import pandas as pd
import os
import psutil

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
            enable_cache: Whether to enable caching
            cache_size: The size of the cache
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

        # 自动设置内存限制为系统可用内存的60%，最小4GB，最大128GB
        system_memory = psutil.virtual_memory().total // (1024 * 1024)  # MB
        memory_limit = max(4096, min(int(system_memory * 0.6), 131072))
        
        # 自动设置线程数为系统逻辑CPU核心数
        num_threads = os.cpu_count() or 4
        
        # 配置DuckDB连接
        self.conn = duckdb.connect(str(self.filepath))
        
        # 设置性能参数
        self.conn.execute(f"PRAGMA memory_limit='{memory_limit}MB'")
        self.conn.execute(f"PRAGMA threads={num_threads}")
        self.conn.execute("PRAGMA force_compression='PFOR'")  # 使用PFOR压缩以节省空间
        # 启用缓存以提高性能
        self.conn.execute("PRAGMA enable_object_cache=true")  # 启用对象缓存
        
        # 创建自定义函数和扩展
        self._load_extensions()
        
        # 使用 DuckDBSchema 管理所有表的 schema
        self._schema_manager = DuckDBSchema({
            'columns': {
                '_id': 'BIGINT'
            },
            'tables': {}  # table_name -> {'columns': {'column_name': 'column_type'}}
        })
        
        self._initialize_database()
        self.id_manager = IDManager(self)
        self._last_modified_time = None
        
    def _load_extensions(self):
        """加载和初始化扩展"""
        try:
            # 加载 httpfs 扩展以支持远程文件访问
            self.conn.execute("INSTALL httpfs")
            self.conn.execute("LOAD httpfs")
            
            # 加载 json 扩展以优化JSON处理
            self.conn.execute("INSTALL json")
            self.conn.execute("LOAD json")
            
            # 加载 parquet 扩展以支持 Parquet 格式
            self.conn.execute("INSTALL parquet")
            self.conn.execute("LOAD parquet")
            
            # 加载 arrow 扩展以支持 Arrow 格式
            self.conn.execute("INSTALL arrow")
            self.conn.execute("LOAD arrow")
            
            # 加载并行扫描扩展
            self.conn.execute("PRAGMA enable_object_cache=true")
            self.conn.execute("PRAGMA threads=8")
        except Exception as e:
            print(f"Warning: Failed to load some extensions: {e}")

    def _initialize_database(self):
        """Initialize the database, create necessary system tables"""
        cursor = self.conn.cursor()
        
        # 创建metadata表
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS tables_meta (
                table_name VARCHAR PRIMARY KEY,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                schema JSON  -- Store the field definitions of the table
            )
        """)
        
        # 创建fields metadata表
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
        
        # 如果default表不存在，创建它
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
                
                # 更新 schema
                self._update_table_schema(table_name, schema)
                
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
        """存储数据
        
        Args:
            data: 要存储的数据，可以是字典或DataFrame
            table_name: 表名
            
        Returns:
            记录ID或ID列表
        """
        table_name = self._get_table_name(table_name)
        
        if isinstance(data, dict):
            # 预处理数据
            processed_data = {}
            for k, v in data.items():
                if isinstance(v, (dict, list)):
                    processed_data[k] = json.dumps(v)
                else:
                    processed_data[k] = v
            df = [processed_data]
        elif isinstance(data, pd.DataFrame):
            df = data.copy()
            # 预处理DataFrame中的JSON字段
            for col in df.columns:
                if df[col].apply(lambda x: isinstance(x, (dict, list))).any():
                    df[col] = df[col].apply(lambda x: json.dumps(x) if isinstance(x, (dict, list)) else x)
        
        # 如果数据是多行，使用batch_store
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
        
        # 确保表存在并更新schema
        self._create_table_if_not_exists(table_name, df[0])
        
        if not isinstance(df, pd.DataFrame):
            df = pd.DataFrame(df)
            
        with self._lock:
            cursor = self.conn.cursor()
            cursor.execute("BEGIN TRANSACTION")
            try:
                # 获取下一个ID
                result = cursor.execute(f"""
                    SELECT COALESCE(MAX(_id), 0) + 1 
                    FROM {self._quote_identifier(table_name)}
                """).fetchone()
                next_id = result[0] if result else 1
                
                # 添加ID列
                if '_id' in df.columns:
                    df = df.drop('_id', axis=1)
                df.insert(0, '_id', next_id)
                
                # 获取列名
                columns = [f'"{str(col)}"' for col in df.columns]
                
                # 插入数据
                cursor.register('df_view', df)
                insert_sql = f"""
                    INSERT INTO {self._quote_identifier(table_name)} ({', '.join(columns)})
                    SELECT {', '.join(columns)} FROM df_view
                """
                cursor.execute(insert_sql)
                cursor.unregister('df_view')
                
                # 创建索引
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

    def batch_store(self, data_list: List[dict], table_name: str = None) -> List[int]:
        """批量存储记录，针对DuckDB进行优化
        
        Args:
            data_list: 要存储的记录列表
            table_name: 表名
            
        Returns:
            记录ID列表
        """
        if not data_list:
            return []
            
        table_name = self._get_table_name(table_name)
        
        # 预处理：获取所有字段和类型
        all_fields = {'_id': 'BIGINT'}  # 确保包含_id字段
        field_order = []  # 保持字段添加顺序
        for data in data_list:
            for key, value in data.items():
                if key not in all_fields and key != '_id':  # 跳过_id字段
                    all_fields[key] = self._infer_field_type(value)
                    if key not in field_order:
                        field_order.append(key)
        
        # 预处理数据：序列化复杂类型
        processed_data = []
        for data in data_list:
            processed_record = {}
            for key, value in data.items():
                if key != '_id':  # 跳过_id字段
                    if isinstance(value, (list, dict)):
                        processed_record[key] = json.dumps(value)
                    else:
                        processed_record[key] = value
            processed_data.append(processed_record)
        
        with self._lock:
            cursor = self.conn.cursor()
            try:
                cursor.execute("BEGIN TRANSACTION")
                
                # 1. 一次性创建所有必要的列
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
                    
                    # 批量添加新列，使用单一SQL语句更高效
                    if len(new_columns) > 0:
                        meta_values = []
                        
                        for col in new_columns:
                            field_type = all_fields[col]
                            # DuckDB一次只能执行一个ALTER命令，逐个执行
                            alter_sql = f"""
                                ALTER TABLE {self._quote_identifier(table_name)}
                                ADD COLUMN IF NOT EXISTS {self._quote_identifier(col)} {field_type}
                            """
                            cursor.execute(alter_sql)
                            
                            meta_values.append((table_name, col, field_type, next_position))
                            next_position += 1
                    
                    # 批量插入元数据
                    if meta_values:
                        meta_insert_sql = """
                            INSERT INTO fields_meta (table_name, field_name, field_type, ordinal_position)
                            VALUES (?, ?, ?, ?)
                            ON CONFLICT (table_name, field_name) DO UPDATE SET 
                                field_type = EXCLUDED.field_type,
                                ordinal_position = EXCLUDED.ordinal_position
                        """
                        cursor.executemany(meta_insert_sql, meta_values)
                    
                    # 更新schema信息
                    schema = {
                        'columns': all_fields
                    }
                    cursor.execute(
                        "UPDATE tables_meta SET schema = ? WHERE table_name = ?",
                        [orjson.dumps(schema).decode('utf-8'), table_name]
                    )
                
                # 2. 获取起始ID
                result = cursor.execute(f"""
                    SELECT COALESCE(MAX(_id), 0) + 1 
                    FROM {self._quote_identifier(table_name)}
                """).fetchone()
                next_id = result[0] if result else 1
                
                # 3. 使用优化的批处理方式
                # 根据数据集大小动态调整批次大小
                data_size = len(processed_data)
                optimal_batch_size = min(5000, max(1000, data_size // 10))
                
                # 设置优化参数
                cursor.execute("PRAGMA threads=8")  # 并行插入
                
                all_ids = []
                
                # 4. 使用更高效的批量处理方式
                for i in range(0, data_size, optimal_batch_size):
                    batch = processed_data[i:i + optimal_batch_size]
                    
                    # 创建批次的DataFrame
                    df = pd.DataFrame(batch)
                    if '_id' in df.columns:
                        df = df.drop('_id', axis=1)
                    
                    # 添加ID列
                    current_ids = list(range(next_id + i, next_id + i + len(batch)))
                    df.insert(0, '_id', current_ids)
                    all_ids.extend(current_ids)
                    
                    # 获取列名
                    columns = [f'"{str(col)}"' for col in df.columns]
                    
                    # 使用DuckDB的DataFrame接口进行批量插入
                    # 注册为临时视图并使用UNOPTIMIZED提示以加速插入
                    view_name = f"batch_view_{i}"
                    cursor.register(view_name, df)
                    insert_sql = f"""
                        /* UNOPTIMIZED */ 
                        INSERT INTO {self._quote_identifier(table_name)} ({', '.join(columns)})
                        SELECT {', '.join(columns)} FROM {view_name}
                    """
                    cursor.execute(insert_sql)
                    cursor.unregister(view_name)
                
                # 5. 提交事务前创建索引
                # 只为较大表自动创建索引
                if data_size >= 10000:
                    for field in field_order[:3]:  # 只为前3个常用字段创建索引
                        try:
                            index_name = f"idx_{table_name}_{field}"
                            cursor.execute(f"""
                                CREATE INDEX IF NOT EXISTS {index_name}
                                ON {self._quote_identifier(table_name)} ({self._quote_identifier(field)})
                            """)
                        except:
                            pass
                
                cursor.execute("COMMIT")
                
                # 6. 插入后优化表（在事务外部）
                if data_size >= 50000:  # 只为大表执行
                    try:
                        cursor.execute(f"ANALYZE {self._quote_identifier(table_name)}")
                    except:
                        pass
                
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
                # 始终包含字段，即使值为 None
                if col != '_id' and isinstance(result[i], str):
                    try:
                        # Try to parse the JSON string
                        data[col] = json.loads(result[i])
                    except json.JSONDecodeError:
                        data[col] = result[i]
                else:
                    data[col] = result[i]
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
                # 始终包含字段，即使值为 None
                if col != '_id' and isinstance(row[i], str):
                    try:
                        # Try to parse the JSON string
                        data[col] = json.loads(row[i])
                    except json.JSONDecodeError:
                        data[col] = row[i]
                else:
                    data[col] = row[i]
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
        """批量替换记录
        
        Args:
            data_dict: ID到记录数据的映射
            
        Returns:
            成功更新的记录ID列表
        """
        if not data_dict:
            return []
            
        success_ids = []
        
        # 逐条处理每个记录，避免事务和主键冲突问题
        for id_, data in data_dict.items():
            try:
                success = self.replace(id_, data)
                if success:
                    success_ids.append(id_)
            except Exception as e:
                print(f"替换记录 {id_} 失败: {e}")
        
        return success_ids

    def query(self, sql: str, params: tuple = None) -> List[tuple]:
        """执行自定义SQL查询，支持并行和向量化执行
        
        Args:
            sql: SQL语句
            params: 查询参数
            
        Returns:
            查询结果
        """
        cursor = self.conn.cursor()
        
        # 设置查询优化参数
        cursor.execute("PRAGMA threads=8")
        
        # 优化特定类型的查询
        if sql.upper().startswith("SELECT"):
            # 添加查询优化提示
            if not "/* PARALLEL */" in sql:
                sql = f"/* PARALLEL */ {sql}"
            
            # 如果是分析查询，增加额外的优化
            if "GROUP BY" in sql.upper() or "SUM(" in sql.upper() or "AVG(" in sql.upper():
                sql = f"/* VECTORIZED */ {sql}"
            
            # 如果是排序查询，添加排序优化
            if "ORDER BY" in sql.upper():
                sql = f"/* SORT_BLOCK_SIZE=1000000 */ {sql}"
        
        # 如果是LIKE查询，添加模糊匹配优化
        if 'LIKE' in sql.upper():
            sql = f"/* USE_INDEX */ {sql}"
        
        # 打印优化后的查询
        # print(f"Optimized query: {sql}")
        
        return cursor.execute(sql, params).fetchall()
        
    def to_pandas(self, sql: str, params: tuple = None) -> "pd.DataFrame":
        """将查询结果直接转换为DataFrame，优化大数据集处理
        
        Args:
            sql: SQL语句
            params: 查询参数
            
        Returns:
            DataFrame对象
        """
        cursor = self.conn.cursor()
        
        # 优化查询
        if not "/* PARALLEL */" in sql:
            sql = f"/* PARALLEL */ {sql}"
        
        # 使用DuckDB的本地转换方法，比fetchall()然后转换更高效
        try:
            # 设置优化参数
            cursor.execute("PRAGMA threads=8")
            
            # 直接转换为DataFrame
            return cursor.execute(sql, params).df()
        except Exception as e:
            # 如果失败，回退到标准方法
            print(f"Warning: Direct DataFrame conversion failed: {e}. Using fetchall method.")
            result = cursor.execute(sql, params).fetchall()
            columns = [description[0] for description in cursor.description]
            return pd.DataFrame(result, columns=columns)

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
        """优化数据库性能，包括自动索引创建、统计信息收集和查询优化"""
        table_name = self._get_table_name()
        cursor = self.conn.cursor()
        
        try:
            # 1. 内存和并行处理优化
            system_memory = psutil.virtual_memory().total // (1024 * 1024)  # MB
            memory_limit = max(4096, min(int(system_memory * 0.6), 131072))
            num_threads = os.cpu_count() or 4
            
            cursor.execute(f"PRAGMA memory_limit='{memory_limit}MB'")
            cursor.execute(f"PRAGMA threads={num_threads}")
            # 优化缓存和其他设置
            cursor.execute("PRAGMA enable_object_cache=true")
            
            # 2. 数据存储优化
            cursor.execute("PRAGMA force_compression='PFOR'")
            
            # 3. 创建自适应索引
            self._create_auto_indexes(table_name)
            
            # 4. 收集统计信息
            cursor.execute(f"ANALYZE {self._quote_identifier(table_name)}")
            
            # 5. 重新编译视图
            self._recompile_views()
            
            print(f"数据库优化完成，使用 {num_threads} 个线程，{memory_limit}MB 内存限制")
            
        except Exception as e:
            raise ValueError(f"Failed to optimize database: {str(e)}")

    def _create_auto_indexes(self, table_name: str):
        """自动创建索引，基于表的大小和查询频率"""
        cursor = self.conn.cursor()
        
        try:
            # 检查表大小
            count_result = cursor.execute(
                f"SELECT COUNT(*) FROM {self._quote_identifier(table_name)}"
            ).fetchone()
            row_count = count_result[0] if count_result else 0
            
            # 如果表小于1000行，不需要创建索引
            if row_count < 1000:
                return
            
            # 获取所有字段
            fields = self._get_table_columns(table_name)
            
            # 为经常在查询中使用的列创建索引
            for field in fields:
                if field == "_id":  # _id已经是主键
                    continue
                
                # 检查字段类型
                field_type_result = cursor.execute(
                    "SELECT field_type FROM fields_meta WHERE table_name = ? AND field_name = ?",
                    [table_name, field]
                ).fetchone()
                
                if field_type_result:
                    field_type = field_type_result[0].upper()
                    
                    # 为常用查询列创建索引
                    if field_type in ["VARCHAR", "BOOLEAN", "BIGINT", "INTEGER", "DATE", "TIMESTAMP"]:
                        try:
                            index_name = f"idx_{table_name}_{field}"
                            cursor.execute(f"""
                                CREATE INDEX IF NOT EXISTS {index_name}
                                ON {self._quote_identifier(table_name)} ({self._quote_identifier(field)})
                            """)
                        except Exception:
                            # 如果创建索引失败，继续处理其他字段
                            pass
        except Exception as e:
            print(f"Warning: Failed to create automatic indexes: {e}")

    def _recompile_views(self):
        """重新编译数据库视图以应用最新的优化器设置"""
        cursor = self.conn.cursor()
        
        try:
            # 获取所有视图
            views = cursor.execute(
                "SELECT view_name FROM duckdb_views() WHERE internal=false"
            ).fetchall()
            
            # 重新编译每个视图
            for view in views:
                view_name = view[0]
                cursor.execute(f"ALTER VIEW {view_name} RECOMPILE")
        except Exception as e:
            # 如果失败，忽略错误继续执行
            print(f"Warning: Failed to recompile views: {e}")

    def _ensure_fields_exist(self, data: dict, table_name: str, cursor):
        """确保所有字段都存在
        
        Args:
            data: 数据字典
            table_name: 表名
            cursor: 数据库游标
        """
        # 不再使用事务，因为调用方负责事务管理
        # 如果表不存在，创建表
        if not self._table_exists(table_name):
            schema = DuckDBSchema()
            for field_name, value in data.items():
                if field_name != '_id':
                    field_type = self._infer_field_type(value)
                    schema.add_column(field_name, field_type)
            self.create_schema(table_name, schema)
            return
        
        # 获取当前 schema
        current_schema = self._get_table_schema(table_name)
        
        # 检查并添加新字段
        new_fields = []
        for field_name, value in data.items():
            if field_name != '_id' and not current_schema.has_column(field_name):
                field_type = self._infer_field_type(value)
                new_fields.append((field_name, field_type))
                current_schema.add_column(field_name, field_type)
        
        if new_fields:
            # 获取当前最大的 ordinal_position
            result = cursor.execute("""
                SELECT COALESCE(MAX(ordinal_position), 0) + 1
                FROM fields_meta
                WHERE table_name = ?
            """, [table_name]).fetchone()
            next_position = result[0] if result else 2
            
            # 添加新字段
            for field_name, field_type in new_fields:
                # 检查列是否已存在
                if not self._column_exists(table_name, field_name):
                    quoted_field = self._quote_identifier(field_name)
                    
                    # 添加列到表
                    cursor.execute(f"""
                        ALTER TABLE {self._quote_identifier(table_name)}
                        ADD COLUMN IF NOT EXISTS {quoted_field} {field_type}
                        DEFAULT NULL
                    """)
                    
                    # 更新元数据
                    cursor.execute("""
                        INSERT INTO fields_meta (table_name, field_name, field_type, ordinal_position)
                        VALUES (?, ?, ?, ?)
                        ON CONFLICT (table_name, field_name) DO UPDATE SET 
                            field_type = EXCLUDED.field_type,
                            ordinal_position = EXCLUDED.ordinal_position
                    """, [table_name, field_name, field_type, next_position])
                    next_position += 1
            
            # 更新 tables_meta
            cursor.execute(
                "UPDATE tables_meta SET schema = ? WHERE table_name = ?",
                [orjson.dumps(current_schema.to_dict()).decode('utf-8'), table_name]
            )
            
            # 更新 schema
            self._update_table_schema(table_name, current_schema)

    def drop_column(self, column_name: str):
        """删除指定的列
        
        Args:
            column_name: 要删除的列名
        """
        if column_name == '_id':
            raise ValueError("Cannot drop _id column")
            
        table_name = self.current_table
        cursor = self.conn.cursor()
        
        try:
            cursor.execute("BEGIN TRANSACTION")
            
            # 检查列是否存在
            if not self._column_exists(table_name, column_name):
                raise ValueError(f"Column {column_name} does not exist")
            
            # DuckDB不支持删除有依赖的列，仅更新元数据
            # 在元数据表中标记为已删除，但不实际删除列
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
        cursor = self.conn.cursor()
        
        try:
            cursor.execute("BEGIN TRANSACTION")
            
            # 检查列是否已存在
            if self._column_exists(table_name, column_name):
                cursor.execute("ROLLBACK")
                return  # 如果列已存在，直接返回
            
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
            
            # 获取并更新 schema
            current_schema = self._get_table_schema(table_name)
            current_schema.add_column(column_name, column_type)
            self._update_table_schema(table_name, current_schema)
            
            # 更新 tables_meta
            cursor.execute(
                "UPDATE tables_meta SET schema = ? WHERE table_name = ?",
                [orjson.dumps(current_schema.to_dict()).decode('utf-8'), table_name]
            )
            
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
        cursor = self.conn.cursor()
        
        try:
            cursor.execute("BEGIN TRANSACTION")
            
            # 检查原列是否存在
            if not self._column_exists(table_name, old_column_name):
                raise ValueError(f"Column {old_column_name} does not exist")
                
            # 检查新列名是否已存在
            if self._column_exists(table_name, new_column_name):
                raise ValueError(f"Column {new_column_name} already exists")
            
            # 获取列类型
            col_type_result = cursor.execute("""
                SELECT field_type FROM fields_meta 
                WHERE table_name = ? AND field_name = ?
            """, [table_name, old_column_name]).fetchone()
            
            if not col_type_result:
                raise ValueError(f"Column {old_column_name} type not found")
            
            col_type = col_type_result[0]
            
            # 创建新列
            cursor.execute(f"""
                ALTER TABLE {self._quote_identifier(table_name)}
                ADD COLUMN {self._quote_identifier(new_column_name)} {col_type}
            """)
            
            # 复制数据
            cursor.execute(f"""
                UPDATE {self._quote_identifier(table_name)}
                SET {self._quote_identifier(new_column_name)} = {self._quote_identifier(old_column_name)}
            """)
            
            # 更新元数据
            ordinal_position = cursor.execute("""
                SELECT ordinal_position
                FROM fields_meta
                WHERE table_name = ? AND field_name = ?
            """, [table_name, old_column_name]).fetchone()[0]
            
            # 添加新列元数据
            cursor.execute("""
                INSERT INTO fields_meta (table_name, field_name, field_type, ordinal_position)
                VALUES (?, ?, ?, ?)
            """, [table_name, new_column_name, col_type, ordinal_position])
            
            # 不删除原列，以避免依赖问题
            
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
        cursor = self.conn.cursor()
        
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
        cursor = self.conn.cursor()
        result = cursor.execute("""
            SELECT 1 
            FROM fields_meta 
            WHERE table_name = ? AND field_name = ?
        """, [table_name, column_name]).fetchone()
        return result is not None

    def _get_table_schema(self, table_name: str) -> DuckDBSchema:
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
                tables[table_name] = {'columns': {'_id': 'BIGINT'}}
            else:
                # 从数据库中读取字段信息
                cursor = self.conn.cursor()
                fields = {'_id': 'BIGINT'}
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
        
        return DuckDBSchema(tables[table_name])
        
    def _update_table_schema(self, table_name: str, schema: DuckDBSchema):
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
