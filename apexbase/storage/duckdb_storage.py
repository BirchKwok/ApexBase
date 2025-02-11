import duckdb
import orjson
from typing import Dict, List, Optional, Union
from pathlib import Path

import pandas as pd

from apexbase.storage.id_manager import IDManager
from .base import BaseStorage, BaseSchema
import threading
import json


class DuckDBSchema(BaseSchema):
    """DuckDB schema."""

    def __init__(self, schema: dict = None):
        """初始化Schema
        
        Args:
            schema: 可选的schema字典，格式为 {'columns': {'column_name': 'column_type'}}
        """
        self.schema = schema or {'columns': {'_id': 'BIGINT'}}
        self._validate_schema()

    def _validate_schema(self):
        """验证schema格式"""
        if not isinstance(self.schema, dict):
            raise ValueError("Schema must be a dictionary")
        if 'columns' not in self.schema:
            raise ValueError("Schema must have a 'columns' key")
        if not isinstance(self.schema['columns'], dict):
            raise ValueError("Schema columns must be a dictionary")
        if '_id' not in self.schema['columns']:
            self.schema['columns']['_id'] = 'BIGINT'

    def to_dict(self):
        """转换为字典格式"""
        return self.schema
    
    def drop_column(self, column_name: str):
        """删除列
        
        Args:
            column_name: 列名
        """
        if column_name == '_id':
            raise ValueError("Cannot drop _id column")
        if column_name in self.schema['columns']:
            del self.schema['columns'][column_name]

    def add_column(self, column_name: str, column_type: str):
        """添加列
        
        Args:
            column_name: 列名
            column_type: 列类型
        """
        if column_name in self.schema['columns']:
            raise ValueError(f"Column {column_name} already exists")
        self.schema['columns'][column_name] = column_type

    def rename_column(self, old_column_name: str, new_column_name: str):
        """重命名列
        
        Args:
            old_column_name: 旧列名
            new_column_name: 新列名
        """
        if old_column_name == '_id':
            raise ValueError("Cannot rename _id column")
        if old_column_name not in self.schema['columns']:
            raise ValueError(f"Column {old_column_name} does not exist")
        if new_column_name in self.schema['columns']:
            raise ValueError(f"Column {new_column_name} already exists")
        self.schema['columns'][new_column_name] = self.schema['columns'].pop(old_column_name)

    def modify_column(self, column_name: str, column_type: str):
        """修改列类型
        
        Args:
            column_name: 列名
            column_type: 列类型
        """
        if column_name == '_id':
            raise ValueError("Cannot modify _id column type")
        if column_name not in self.schema['columns']:
            raise ValueError(f"Column {column_name} does not exist")
        self.schema['columns'][column_name] = column_type

    def get_column_type(self, column_name: str) -> str:
        """获取列类型
        
        Args:
            column_name: 列名
        """
        if column_name not in self.schema['columns']:
            raise ValueError(f"Column {column_name} does not exist")
        return self.schema['columns'][column_name]

    def has_column(self, column_name: str) -> bool:
        """检查列是否存在
        
        Args:
            column_name: 列名
        """
        return column_name in self.schema['columns']

    def get_columns(self) -> List[str]:
        """获取所有列名
        
        Returns:
            列名列表
        """
        return list(self.schema['columns'].keys())

    def update_from_data(self, data: dict):
        """从数据更新schema
        
        Args:
            data: 数据字典
        """
        for column_name, value in data.items():
            if column_name != '_id' and column_name not in self.schema['columns']:
                column_type = self._infer_column_type(value)
                self.add_column(column_name, column_type)

    def _infer_column_type(self, value) -> str:
        """推断列类型
        
        Args:
            value: 值
            
        Returns:
            列类型
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
        """初始化DuckDB存储
        
        Args:
            filepath: 数据库文件路径
            batch_size: 批处理大小
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

    def _initialize_database(self):
        """初始化数据库，创建必要的系统表"""
        cursor = self.conn.cursor()
        
        # 设置优化参数
        cursor.execute("PRAGMA memory_limit='4GB'")
        cursor.execute("PRAGMA threads=4")
        
        # 创建元数据表
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS tables_meta (
                table_name VARCHAR PRIMARY KEY,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                schema JSON  -- 存储表的字段定义
            )
        """)
        
        # 创建字段元数据表
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
        
        # 如果默认表不存在，则创建
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
        """创建表的schema
        
        Args:
            table_name: 表名
            schema: schema对象
        """
        with self._lock:
            if self._table_exists(table_name):
                raise ValueError(f"Table '{table_name}' already exists")

            cursor = self.conn.cursor()
            cursor.execute("BEGIN TRANSACTION")
            try:
                # 创建表
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
                
                # 更新元数据
                cursor.execute(
                    "INSERT INTO tables_meta (table_name, schema) VALUES (?, ?)",
                    [table_name, orjson.dumps(schema.to_dict()).decode('utf-8')]
                )
                
                # 初始化fields_meta表
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
        """创建新表，使用默认schema
        
        Args:
            table_name: 表名
        """
        schema = DuckDBSchema()  # 使用默认schema
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
        """根据值推断DuckDB列类型。"""
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

    def _create_table_if_not_exists(self, table_name: str, data: Union[dict, pd.DataFrame]):
        """根据数据创建或更新表，支持动态字段
        
        Args:
            table_name: 表名
            data: 数据（字典或DataFrame）
        """
        if isinstance(data, dict):
            df = pd.DataFrame([data])
        else:
            df = data.copy()
            
        # 如果表不存在，创建表
        if not self._table_exists(table_name):
            schema = DuckDBSchema()
            for col in df.columns:
                if col != '_id':
                    schema.add_column(col, self._get_duckdb_type(df[col].dtype))
            self.create_schema(table_name, schema)
            return
        
        # 获取现有列
        existing_columns = set(self._get_table_columns(table_name))
        # 保持原始顺序的新列列表
        columns = df.columns

        new_columns = [col for col in columns if col != '_id' and col not in existing_columns]
        
        if not new_columns:
            return
            
        # 添加新列
        with self._lock:
            cursor = self.conn.cursor()
            cursor.execute("BEGIN TRANSACTION")
            try:
                # 获取当前schema
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
                    
                    # 更新schema
                    current_schema.add_column(col, sql_type)
                    
                    # 添加字段到fields_meta表
                    cursor.execute("""
                        INSERT INTO fields_meta (table_name, field_name, field_type, ordinal_position)
                        VALUES (?, ?, ?, ?)
                        ON CONFLICT (table_name, field_name) DO UPDATE SET 
                            field_type = EXCLUDED.field_type,
                            ordinal_position = EXCLUDED.ordinal_position
                    """, [table_name, col, sql_type, next_position])
                    next_position += 1
                
                # 更新tables_meta中的schema
                cursor.execute(
                    "UPDATE tables_meta SET schema = ? WHERE table_name = ?",
                    [orjson.dumps(current_schema.to_dict()).decode('utf-8'), table_name]
                )
                
                cursor.execute("COMMIT")
            except Exception as e:
                cursor.execute("ROLLBACK")
                raise e

    def _get_duckdb_type(self, pandas_type) -> str:
        """将Pandas数据类型转换为DuckDB数据类型"""
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
            存储的记录ID或ID列表
        """
        table_name = self._get_table_name(table_name)
        
        if isinstance(data, dict):
            # 预处理 JSON 字段
            processed_data = {}
            for k, v in data.items():
                if isinstance(v, (dict, list)):
                    processed_data[k] = json.dumps(v)
                else:
                    processed_data[k] = v
            df = [processed_data]
        elif isinstance(data, pd.DataFrame):
            df = data.copy()
            # 预处理 DataFrame 中的 JSON 字段
            for col in df.columns:
                if df[col].apply(lambda x: isinstance(x, (dict, list))).any():
                    df[col] = df[col].apply(lambda x: json.dumps(x) if isinstance(x, (dict, list)) else x)
        
        # 如果是多行数据，使用batch_store
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
    
    def _get_next_id(self, table_name: str) -> int:
        """获取下一个ID"""
        cursor = self.conn.cursor()
        result = cursor.execute(f"""
            SELECT COALESCE(MAX(_id), 0) + 1 
            FROM {self._quote_identifier(table_name)}
        """).fetchone()
        return result[0] if result else 1
    
    def flush_cache(self):
        """刷新缓存"""
        if self._cache is not None and len(self._cache) > 0:
            self.batch_store(self._cache)
            self._cache = []

    def batch_store(self, data_list: List[dict], table_name: str = None) -> List[int]:
        """批量存储记录
        
        Args:
            data_list: 要存储的记录列表
            table_name: 表名
            
        Returns:
            存储的记录ID列表
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
                
                # 1. 一次性创建所有需要的列
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
                    
                    # 批量添加新列
                    for col in new_columns:
                        field_type = all_fields[col]
                        cursor.execute(f"""
                            ALTER TABLE {self._quote_identifier(table_name)}
                            ADD COLUMN IF NOT EXISTS {self._quote_identifier(col)} {field_type}
                        """)
                        
                        # 添加字段到fields_meta表
                        cursor.execute("""
                            INSERT INTO fields_meta (table_name, field_name, field_type, ordinal_position)
                            VALUES (?, ?, ?, ?)
                            ON CONFLICT (table_name, field_name) DO UPDATE SET 
                                field_type = EXCLUDED.field_type,
                                ordinal_position = EXCLUDED.ordinal_position
                        """, [table_name, col, field_type, next_position])
                    
                    # 更新schema信息
                    schema = {
                        'columns': all_fields  # 直接使用all_fields，它已经包含了所有字段的类型
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
                
                # 3. 分批处理
                batch_size = self.batch_size
                all_ids = []
                
                for i in range(0, len(processed_data), batch_size):
                    batch = processed_data[i:i + batch_size]
                    
                    # 创建当前批次的DataFrame
                    df = pd.DataFrame(batch)
                    if '_id' in df.columns:
                        df = df.drop('_id', axis=1)
                    
                    # 添加ID列
                    current_ids = range(next_id + i, next_id + i + len(batch))
                    df.insert(0, '_id', current_ids)
                    all_ids.extend(current_ids)
                    
                    # 获取列名
                    columns = [f'"{str(col)}"' for col in df.columns]
                    
                    # 使用DuckDB的DataFrame接口批量插入
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

    def _get_table_columns(self, table_name: str) -> List[str]:
        """获取表的列名。"""
        cursor = self.conn.cursor()
        cursor.execute(f"DESCRIBE {self._quote_identifier(table_name)}")
        columns = cursor.fetchall()
        return [col[0] for col in columns]

    def retrieve(self, id_: int) -> Optional[dict]:
        """获取单条记录
        
        Args:
            id_: 记录ID
            
        Returns:
            记录数据字典
        """
        table_name = self._get_table_name()
        cursor = self.conn.cursor()
        
        # 获取所有列名
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
                            # 尝试解析JSON字符串
                            data[col] = json.loads(value)
                        except json.JSONDecodeError:
                            data[col] = value
                    else:
                        data[col] = value
            return data
        return None

    def retrieve_many(self, ids: List[int]) -> List[dict]:
        """获取多条记录
        
        Args:
            ids: 记录ID列表
            
        Returns:
            记录数据字典列表
        """
        if not ids:
            return []
            
        table_name = self._get_table_name()
        cursor = self.conn.cursor()
        
        # 获取所有列名
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
                            # 尝试解析JSON字符串
                            data[col] = json.loads(value)
                        except json.JSONDecodeError:
                            data[col] = value
                    else:
                        data[col] = value
            data_list.append(data)
        
        return data_list

    def delete(self, id_: int) -> bool:
        """删除记录
        
        Args:
            id_: 记录ID
            
        Returns:
            bool: 删除是否成功
        """
        table_name = self._get_table_name()
        
        with self._lock:
            cursor = self.conn.cursor()
            cursor.execute("BEGIN TRANSACTION")
            try:
                # 检查记录是否存在
                exists = cursor.execute(
                    f"SELECT 1 FROM {self._quote_identifier(table_name)} WHERE _id = ?",
                    [id_]
                ).fetchone()
                if not exists:
                    cursor.execute("ROLLBACK")
                    return False
                
                # 执行删除
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

    def batch_delete(self, ids: List[int]) -> bool:
        """批量删除记录
        
        Args:
            ids: 记录ID列表
            
        Returns:
            bool: 删除是否成功
        """
        if not ids:
            return True
            
        table_name = self._get_table_name()
        
        with self._lock:
            cursor = self.conn.cursor()
            cursor.execute("BEGIN TRANSACTION")
            try:
                # 检查记录是否存在
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
                
                # 执行删除
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

    def replace(self, id_: int, data: dict) -> bool:
        """替换单条记录
    
        Args:
            id_: 记录ID
            data: 新的记录数据
            
        Returns:
            bool: 替换是否成功
        """
        table_name = self._get_table_name()
        
        with self._lock:
            cursor = self.conn.cursor()
            
            # 检查记录是否存在
            exists = cursor.execute(
                f"SELECT 1 FROM {self._quote_identifier(table_name)} WHERE _id = ?",
                [id_]
            ).fetchone()
            
            if not exists:
                return False

            # 确保所有字段存在
            update_data = {k: v for k, v in data.items() if k != '_id'}  # 显式排除_id字段
            self._ensure_fields_exist(update_data, table_name, cursor)
            
            # 准备更新数据
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
                params.append(id_)  # 添加WHERE条件的参数
                update_sql = f"""
                    UPDATE {self._quote_identifier(table_name)}
                    SET {', '.join(set_clauses)}
                    WHERE _id = ?
                """
                try:
                    cursor.execute(update_sql, params)
                except Exception as e:
                    # 如果 UPDATE 失败，尝试 DELETE + INSERT
                    cursor.execute(f"""
                        DELETE FROM {self._quote_identifier(table_name)}
                        WHERE _id = ?
                    """, [id_])
                    
                    # 准备所有字段
                    all_fields = self._get_table_columns(table_name)
                    current_data = cursor.execute(f"""
                        SELECT * FROM {self._quote_identifier(table_name)}
                        WHERE _id = ?
                    """, [id_]).fetchone()
                    
                    # 构建完整的字段值列表
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
                    
                    # 插入新记录
                    placeholders = ['?' for _ in columns]
                    insert_sql = f"""
                        INSERT INTO {self._quote_identifier(table_name)}
                        ({', '.join(columns)})
                        VALUES ({', '.join(placeholders)})
                    """
                    cursor.execute(insert_sql, values)
            
            return True

    def batch_replace(self, data_dict: Dict[int, dict]) -> List[int]:
        """Replaces multiple records by IDs."""
        if not data_dict:
            return []
            
        table_name = self._get_table_name()
        success_ids = []
        
        with self._lock:
            cursor = self.conn.cursor()
            
            # 检查记录是否存在
            ids = list(data_dict.keys())
            placeholders = ','.join(['?' for _ in ids])
            existing_ids = cursor.execute(f"""
                SELECT _id FROM {self._quote_identifier(table_name)}
                WHERE _id IN ({placeholders})
            """, ids).fetchall()
            existing_ids = {row[0] for row in existing_ids}
            
            # 只更新存在的记录
            for id_ in existing_ids:
                data = data_dict[id_]
                # 确保所有字段存在
                update_data = {k: v for k, v in data.items() if k != '_id'}
                self._ensure_fields_exist(update_data, table_name, cursor)
                
                # 准备更新数据
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
                    params.append(id_)  # 添加WHERE条件的参数
                    update_sql = f"""
                        UPDATE {self._quote_identifier(table_name)}
                        SET {', '.join(set_clauses)}
                        WHERE _id = ?
                    """
                    try:
                        cursor.execute(update_sql, params)
                    except Exception as e:
                        # 如果 UPDATE 失败，尝试 DELETE + INSERT
                        cursor.execute(f"""
                            DELETE FROM {self._quote_identifier(table_name)}
                            WHERE _id = ?
                        """, [id_])
                        
                        # 准备所有字段
                        all_fields = self._get_table_columns(table_name)
                        current_data = cursor.execute(f"""
                            SELECT * FROM {self._quote_identifier(table_name)}
                            WHERE _id = ?
                        """, [id_]).fetchone()
                        
                        # 构建完整的字段值列表
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
                        
                        # 插入新记录
                        placeholders = ['?' for _ in columns]
                        insert_sql = f"""
                            INSERT INTO {self._quote_identifier(table_name)}
                            ({', '.join(columns)})
                            VALUES ({', '.join(placeholders)})
                        """
                        cursor.execute(insert_sql, values)
                    
                    success_ids.append(id_)
            
            return success_ids

    def query(self, sql: str, params: tuple = None) -> List[tuple]:
        """执行自定义SQL查询，支持并行执行
        
        Args:
            sql: SQL语句
            params: 查询参数
            
        Returns:
            查询结果
        """
        cursor = self.conn.cursor()
        
        # 添加并行查询支持
        cursor.execute("PRAGMA threads=4")
        
        # 如果是 LIKE 查询，添加索引提示
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
        """推断字段类型
        
        Args:
            value: 字段值
            
        Returns:
            字段类型
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
        """获取表的所有字段
        
        Args:
            table_name: 表名
            
        Returns:
            字段列表
        """
        table_name = self._get_table_name(table_name)
        cursor = self.conn.cursor()
        
        # 按ordinal_position排序获取所有字段
        result = cursor.execute("""
            SELECT field_name 
            FROM fields_meta 
            WHERE table_name = ? 
            ORDER BY ordinal_position
        """, [table_name]).fetchall()
        
        return [row[0] for row in result]

    def _create_indexes(self, table_name: str):
        """为表创建必要的索引"""
        cursor = self.conn.cursor()
        
        # 获取需要创建索引的字段
        fields = cursor.execute("""
            SELECT field_name, field_type 
            FROM fields_meta 
            WHERE table_name = ? AND is_indexed = FALSE
        """, [table_name]).fetchall()
        
        for field_name, field_type in fields:
            # 为 VARCHAR 类型的字段创建索引
            if field_type == 'VARCHAR':
                index_name = f"idx_{table_name}_{field_name}"
                cursor.execute(f"""
                    CREATE INDEX IF NOT EXISTS {index_name}
                    ON {self._quote_identifier(table_name)} ({self._quote_identifier(field_name)})
                """)
                
                # 更新索引状态
                cursor.execute("""
                    UPDATE fields_meta 
                    SET is_indexed = TRUE 
                    WHERE table_name = ? AND field_name = ?
                """, [table_name, field_name])

    def to_pandas(self, sql: str, params: tuple = None) -> "pd.DataFrame":
        """将查询结果直接转换为 DataFrame
        
        Args:
            sql: SQL 语句
            params: 查询参数
            
        Returns:
            DataFrame 对象
        """
        cursor = self.conn.cursor()
        
        # 获取字段名
        fields = self.list_fields()
        field_list = ','.join(
            f'CAST({self._quote_identifier(f)} AS TEXT) AS {self._quote_identifier(f)}'
            for f in fields
        )
        
        # 构建优化的查询
        optimized_sql = f"""
            WITH result AS (
                {sql}
            )
            SELECT {field_list}
            FROM result
        """
        
        # 使用 DuckDB 的原生 DataFrame 转换
        return cursor.execute(optimized_sql, params).df()

    def _create_temp_table(self, table_name: str, suffix: str = None) -> str:
        """创建临时表并返回表名"""
        temp_name = f"temp_{table_name}"
        if suffix:
            temp_name = f"{temp_name}_{suffix}"
        temp_name = self._quote_identifier(temp_name)
        return temp_name

    def count_rows(self, table_name: str = None) -> int:
        """获取表中的记录数
        
        Args:
            table_name: 表名
            
        Returns:
            记录数
        """
        table_name = self._get_table_name(table_name)
        cursor = self.conn.cursor()
        
        # 如果有缓存中的数据，需要包含在计数中
        cache_count = len(self._cache) if self.enable_cache else 0
        
        result = cursor.execute(f"""
            SELECT COUNT(*) 
            FROM {self._quote_identifier(table_name)}
        """).fetchone()
        return result[0] + cache_count if result else cache_count

    def optimize(self):
        """优化数据库性能"""
        table_name = self._get_table_name()
        cursor = self.conn.cursor()
        
        try:
            # DuckDB的优化操作
            cursor.execute("PRAGMA memory_limit='4GB'")
            cursor.execute("PRAGMA threads=4")
            cursor.execute("PRAGMA force_compression='none'")
            cursor.execute("PRAGMA checkpoint_threshold='1GB'")
            
            # 分析表以优化查询计划
            cursor.execute(f"ANALYZE {self._quote_identifier(table_name)}")
            
        except Exception as e:
            raise ValueError(f"Failed to optimize database: {str(e)}")

    def _ensure_fields_exist(self, data: dict, table_name: str, cursor):
        """确保所有字段都存在（移除事务管理）"""
        # 获取现有字段元数据
        existing_fields = cursor.execute(
            "SELECT field_name FROM fields_meta WHERE table_name = ?",
            [table_name]
        ).fetchall()
        existing_fields = {row[0] for row in existing_fields}
        
        # 添加新字段到表和元数据
        for field in data.keys():
            if field == '_id':
                continue
            if field not in existing_fields:
                field_type = self._infer_field_type(data[field])
                # 添加字段到表
                cursor.execute(
                    f"ALTER TABLE {self._quote_identifier(table_name)} "
                    f"ADD COLUMN {self._quote_identifier(field)} {field_type}"
                )
                # 更新元数据表
                cursor.execute(
                    "INSERT INTO fields_meta (table_name, field_name, field_type) "
                    "VALUES (?, ?, ?) "
                    "ON CONFLICT (table_name, field_name) DO UPDATE SET "
                    "field_type = EXCLUDED.field_type",
                    [table_name, field, field_type]
                )

    def drop_column(self, table_name: str, column_name: str):
        """删除列
        
        Args:
            table_name: 表名
            column_name: 列名
        """
        pass

    def add_column(self, table_name: str, column_name: str, column_type: str):
        """添加列
        
        Args:
            table_name: 表名
            column_name: 列名
            column_type: 列类型
        """
        pass

    def rename_column(self, table_name: str, old_column_name: str, new_column_name: str):
        """重命名列
        
        Args:
            table_name: 表名
            old_column_name: 旧列名
            new_column_name: 新列名
        """
        pass

    def modify_column(self, table_name: str, column_name: str, column_type: str):
        """修改列类型
        
        Args:
            table_name: 表名
            column_name: 列名
            column_type: 列类型
        """
        pass