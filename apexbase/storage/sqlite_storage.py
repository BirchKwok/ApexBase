import sqlite3
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
        """初始化Schema
        
        Args:
            schema: 可选的schema字典，格式为 {'columns': {'column_name': 'column_type'}}
        """
        self.schema = schema or {'columns': {'_id': 'INTEGER'}}
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
            self.schema['columns']['_id'] = 'INTEGER'

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
            return "INTEGER"  # SQLite没有布尔类型
        elif isinstance(value, int):
            return "INTEGER"
        elif isinstance(value, float):
            return "REAL"
        elif isinstance(value, (str, dict, list)):
            return "TEXT"
        elif pd.isna(value):
            return "TEXT"  # 对于空值，默认使用TEXT
        else:
            return "TEXT"  # 对于未知类型，默认使用TEXT


class SQLiteStorage(BaseStorage):
    """SQLite implementation of the storage backend."""
    
    def __init__(self, db_path: str, batch_size: int = 1000, enable_cache: bool = True, cache_size: int = 10000):
        """初始化SQLite存储
        
        Args:
            db_path: 数据库文件路径
            batch_size: 批处理大小
        """
        self.db_path = db_path
        self.batch_size = batch_size

        self._lock = threading.Lock()
        self.current_table = "default"
        self.enable_cache = enable_cache
        self.cache_size = cache_size
        self._cache = []
        self.id_manager = IDManager(self)
        
        # 创建数据库目录
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        
        # 初始化数据库
        self._initialize_database()

    def _get_connection(self):
        """获取数据库连接"""
        if not hasattr(self, '_conn'):
            self._conn = sqlite3.connect(self.db_path, isolation_level=None)  # 自动提交模式
            self._conn.row_factory = sqlite3.Row
            
        return self._conn

    def _initialize_database(self):
        """初始化数据库,创建必要的表"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            # 设置数据库参数
            cursor.execute('PRAGMA journal_mode=WAL')
            cursor.execute('PRAGMA synchronous=NORMAL')
            cursor.execute('PRAGMA cache_size=2000000')  # 进一步增加缓存
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
            
            # 创建反向字符串函数
            conn.create_function("reverse", 1, lambda s: s[::-1] if s else None)
            
            # 创建主表 - 只包含_id字段，其他字段动态添加
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS "default" (
                _id INTEGER PRIMARY KEY
            )
            ''')
            
            # 创建元数据表
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS tables_meta (
                table_name TEXT PRIMARY KEY,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                schema TEXT  -- 存储表的字段定义
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
            
            # 初始化元数据
            default_schema = SQLiteSchema().to_dict()
            cursor.execute(
                'INSERT OR IGNORE INTO tables_meta (table_name, schema) VALUES (?, ?)',
                ('default', json.dumps(default_schema))
            )
            
            # 初始化default表的_id字段
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

    def create_schema(self, table_name: str, schema: SQLiteSchema):
        """创建表的schema
        
        Args:
            table_name: 表名
            schema: schema对象
        """
        if self._table_exists(table_name):
            raise ValueError(f"Table '{table_name}' already exists")

        cursor = self._get_connection().cursor()
        try:
            cursor.execute("BEGIN TRANSACTION")
            
            # 创建表
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
        schema = SQLiteSchema()  # 使用默认schema
        self.create_schema(table_name, schema)

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
            cursor.execute(f"DROP TABLE IF EXISTS {self._quote_identifier(table_name + '_fields_meta')}")
            
            cursor.execute("DELETE FROM tables_meta WHERE table_name = ?", [table_name])
            
            cursor.execute("COMMIT")
            
            if self.current_table == table_name:
                self.use_table("default")
            
        except Exception as e:
            cursor.execute("ROLLBACK")
            raise e

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
        """推断字段类型
        
        Args:
            value: 字段值
            
        Returns:
            字段类型
        """
        if isinstance(value, bool):
            return "INTEGER"  # SQLite没有布尔类型，使用INTEGER
        elif isinstance(value, int):
            return "INTEGER"
        elif isinstance(value, float):
            return "REAL"
        elif isinstance(value, (list, dict)):
            return "TEXT"  # 复杂类型序列化为JSON存储
        else:
            return "TEXT"

    def _ensure_fields_exist(self, data: dict, table_name: str = None):
        """确保所有字段都存在
        
        Args:
            data: 数据字典
            table_name: 表名
        """
        table_name = self._get_table_name(table_name)
        cursor = self._get_connection().cursor()
        
        try:
            # 如果表不存在，创建表
            if not self._table_exists(table_name):
                schema = SQLiteSchema()
                for field_name, value in data.items():
                    if field_name != '_id':
                        field_type = self._infer_field_type(value)
                        schema.add_column(field_name, field_type)
                self.create_schema(table_name, schema)
                return
            
            # 获取现有字段
            existing_fields = set()
            for row in cursor.execute(
                "SELECT field_name FROM fields_meta WHERE table_name = ?",
                [table_name]
            ):
                existing_fields.add(row[0])
            
            # 获取当前schema
            result = cursor.execute(
                "SELECT schema FROM tables_meta WHERE table_name = ?",
                [table_name]
            ).fetchone()
            current_schema = SQLiteSchema(json.loads(result[0]))
            
            # 获取当前最大的ordinal_position
            result = cursor.execute("""
                SELECT COALESCE(MAX(ordinal_position), 0) + 1
                FROM fields_meta
                WHERE table_name = ?
            """, [table_name]).fetchone()
            next_position = result[0] if result else 2  # _id是1，所以从2开始
            
            # 按照数据中的字段顺序添加新字段
            for field_name, value in data.items():
                if field_name != '_id' and field_name not in existing_fields:
                    field_type = self._infer_field_type(value)
                    is_json = isinstance(value, str) and (value.startswith('{') or value.startswith('['))
                    quoted_field = self._quote_identifier(field_name)
                    
                    # 添加字段到表
                    cursor.execute(
                        f"ALTER TABLE {self._quote_identifier(table_name)} ADD COLUMN {quoted_field} {field_type}"
                    )
                    
                    # 更新schema
                    current_schema.add_column(field_name, field_type)
                    
                    # 更新元数据
                    cursor.execute("""
                        INSERT INTO fields_meta (table_name, field_name, field_type, is_json, ordinal_position)
                        VALUES (?, ?, ?, ?, ?)
                        ON CONFLICT (table_name, field_name) DO UPDATE SET 
                            field_type = EXCLUDED.field_type,
                            is_json = EXCLUDED.is_json,
                            ordinal_position = EXCLUDED.ordinal_position
                    """, [table_name, field_name, field_type, is_json, next_position])
                    next_position += 1
            
            # 更新tables_meta中的schema
            cursor.execute(
                "UPDATE tables_meta SET schema = ? WHERE table_name = ?",
                [orjson.dumps(current_schema.to_dict()).decode('utf-8'), table_name]
            )
            
        except Exception as e:
            raise e

    def store(self, data: dict, table_name: str = None) -> int:
        """存储单条记录
        
        Args:
            data: 要存储的数据
            table_name: 表名
            
        Returns:
            存储的记录ID
        """
        table_name = self._get_table_name(table_name)
        
        # 预处理数据
        processed_data = {}
        for k, v in data.items():
            if isinstance(v, (dict, list)):
                processed_data[k] = orjson.dumps(v).decode('utf-8')
            else:
                processed_data[k] = v
        
        # 确保所有字段存在
        self._ensure_fields_exist(processed_data, table_name)
        
        if self.enable_cache and self.id_manager.get_next_id(table_name) != 1:
            with self._lock:
                self._cache.append(processed_data)

                self.id_manager.auto_increment(table_name)
                if len(self._cache) >= self.cache_size:
                    self.flush_cache()
            return self.id_manager.current_id(table_name)
        
        with self._get_connection() as conn:
            cursor = conn.cursor()
            try:
                # 准备插入语句
                fields = []
                values = []
                params = []
                
                for field_name, value in processed_data.items():
                    if field_name != '_id':
                        fields.append(self._quote_identifier(field_name))
                        values.append('?')
                        params.append(value)
                
                # 构建并执行插入语句
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
            
    def flush_cache(self):
        """刷新缓存"""
        if self._cache:
            with self._lock:
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
        
        # 预处理：获取所有字段和类型，保持字段顺序
        all_fields = {'_id': 'INTEGER'}  # 确保包含_id字段
        field_order = []  # 保持字段添加顺序
        for data in data_list:
            for key, value in data.items():
                if key not in all_fields and key != '_id':  # 跳过_id字段
                    all_fields[key] = self._infer_field_type(value)
                    if key not in field_order:
                        field_order.append(key)
        
        with self._get_connection() as conn:
            cursor = conn.cursor()
            try:
                cursor.execute("BEGIN TRANSACTION")
                
                # 1. 一次性创建所有需要的列
                if not self._table_exists(table_name):
                    self.create_table(table_name)
                
                # 获取现有字段
                existing_fields = set()
                for row in cursor.execute(
                    "SELECT field_name FROM fields_meta WHERE table_name = ?",
                    [table_name]
                ):
                    existing_fields.add(row[0])
                
                # 获取当前最大的ordinal_position
                result = cursor.execute("""
                    SELECT COALESCE(MAX(ordinal_position), 0) + 1
                    FROM fields_meta
                    WHERE table_name = ?
                """, [table_name]).fetchone()
                next_position = result[0] if result else 2  # _id是1，所以从2开始
                
                # 按照field_order添加新字段
                for field_name in field_order:
                    if field_name not in existing_fields:
                        field_type = all_fields[field_name]
                        quoted_field = self._quote_identifier(field_name)
                        
                        # 添加字段到表
                        cursor.execute(
                            f"ALTER TABLE {self._quote_identifier(table_name)} ADD COLUMN {quoted_field} {field_type}"
                        )
                        
                        # 更新元数据
                        cursor.execute("""
                            INSERT INTO fields_meta (table_name, field_name, field_type, ordinal_position)
                            VALUES (?, ?, ?, ?)
                            ON CONFLICT (table_name, field_name) DO UPDATE SET 
                                field_type = EXCLUDED.field_type,
                                ordinal_position = EXCLUDED.ordinal_position
                        """, [table_name, field_name, field_type, next_position])
                        next_position += 1
                
                # 准备批量插入
                fields = ['_id'] + field_order
                placeholders = ','.join(['?' for _ in fields])
                sql = f"""
                    INSERT INTO {self._quote_identifier(table_name)}
                    ({','.join(self._quote_identifier(f) for f in fields)})
                    VALUES ({placeholders})
                """
                
                # 准备数据
                values = []
                first_id = cursor.execute(
                    f"SELECT COALESCE(MAX(_id), 0) + 1 FROM {self._quote_identifier(table_name)}"
                ).fetchone()[0]
                
                for i, data in enumerate(data_list):
                    row = [first_id + i]  # _id
                    for field in field_order:
                        value = data.get(field)
                        if isinstance(value, (list, dict)):
                            row.append(json.dumps(value))
                        else:
                            row.append(value)
                    values.append(tuple(row))
                
                # 执行批量插入
                cursor.executemany(sql, values)
                
                cursor.execute("COMMIT")
                return list(range(first_id, first_id + len(data_list)))
                
            except Exception as e:
                cursor.execute("ROLLBACK")
                raise e
            
            finally:
                self.id_manager.reset_last_id(table_name)

    def list_fields(self, table_name: str = None) -> List[str]:
        """获取表的所有字段
        
        Args:
            table_name: 表名
            
        Returns:
            字段列表
        """
        table_name = self._get_table_name(table_name)
        cursor = self._get_connection().cursor()
        
        # 按ordinal_position排序获取所有字段
        result = cursor.execute("""
            SELECT field_name 
            FROM fields_meta 
            WHERE table_name = ? 
            ORDER BY ordinal_position
        """, [table_name]).fetchall()
        
        return [row[0] for row in result]

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
            # VACUUM 不能在事务中执行
            cursor.execute("VACUUM")
            
            # 其他优化操作放在事务中
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
                # 确保所有字段存在
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
                # 确保所有字段存在
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

    def _get_next_id(self, table_name: str) -> int:
        """获取下一个ID"""
        cursor = self._get_connection().cursor()
        result = cursor.execute(f"""
            SELECT COALESCE(MAX(_id), 0) + 1 
            FROM {self._quote_identifier(table_name)}
        """).fetchone()
        return result[0] if result else 1
    
    def close(self):
        """关闭数据库连接"""
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
        """获取单条记录
        
        Args:
            id_: 记录ID
            
        Returns:
            记录数据字典
        """
        table_name = self._get_table_name()
        cursor = self._get_connection().cursor()
        
        try:
            # 获取所有字段
            fields = self.list_fields(table_name)
            if not fields:
                return None
            
            # 构建查询
            field_list = ','.join(self._quote_identifier(f) for f in fields)
            sql = f"""
                SELECT {field_list}
                FROM {self._quote_identifier(table_name)}
                WHERE _id = ?
            """
            
            result = cursor.execute(sql, [id_]).fetchone()
            if not result:
                return None
            
            # 构建返回数据
            data = {}
            for i, field in enumerate(fields):
                value = result[i]
                if value is not None:
                    try:
                        # 尝试解析JSON
                        data[field] = json.loads(value)
                    except (json.JSONDecodeError, TypeError):
                        data[field] = value
                else:
                    data[field] = None
            
            return data
            
        except Exception as e:
            raise ValueError(f"Failed to retrieve record: {str(e)}")

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
        cursor = self._get_connection().cursor()
        
        try:
            # 获取所有字段
            fields = self.list_fields(table_name)
            if not fields:
                return []
            
            # 构建查询
            field_list = ','.join(self._quote_identifier(f) for f in fields)
            placeholders = ','.join('?' for _ in ids)
            sql = f"""
                SELECT {field_list}
                FROM {self._quote_identifier(table_name)}
                WHERE _id IN ({placeholders})
                ORDER BY _id
            """
            
            results = cursor.execute(sql, ids).fetchall()
            
            # 构建返回数据
            records = []
            for row in results:
                data = {}
                for i, field in enumerate(fields):
                    value = row[i]
                    if value is not None:
                        try:
                            # 尝试解析JSON
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
        """执行自定义SQL查询
        
        Args:
            sql: SQL语句
            params: 查询参数
            
        Returns:
            查询结果
        """
        cursor = self._get_connection().cursor()
        
        # 全局查询优化参数
        cursor.execute('PRAGMA temp_store=MEMORY')
        cursor.execute('PRAGMA cache_size=2000000')
        cursor.execute('PRAGMA mmap_size=30000000000')
        cursor.execute('PRAGMA read_uncommitted=1')
        cursor.execute('PRAGMA synchronous=OFF')
        cursor.execute('PRAGMA journal_mode=MEMORY')
        cursor.execute('PRAGMA page_size=4096')
        
        # 优化查询
        if 'LIKE' in sql.upper():
            # 添加索引提示
            sql = f"/* USING INDEX */ {sql}"
            
            # 优化 LIKE 查询
            if 'LIKE' in sql and '%' in sql:
                # 如果是前缀匹配，使用索引
                if sql.count('%') == 1 and sql.endswith("%'"):
                    sql = sql.replace('LIKE', 'GLOB')
                    sql = sql.replace('%', '*')
                # 如果是后缀匹配，反转字符串并使用前缀匹配
                elif sql.count('%') == 1 and sql.startswith("'%"):
                    field = re.search(r'(\w+)\s+LIKE', sql).group(1)
                    pattern = sql.split("'%")[1].rstrip("'")
                    sql = f"""
                        SELECT * FROM (
                            SELECT *, reverse({field}) as rev_{field}
                            FROM ({sql.split('WHERE')[0]})
                        ) WHERE rev_{field} GLOB '{pattern[::-1]}*'
                    """
        
        # 使用迭代器模式执行查询
        cursor.execute(sql, params)
        chunk_size = 1000  # 每次获取的记录数
        results = []
        while True:
            chunk = cursor.fetchmany(chunk_size)
            if not chunk:
                break
            results.extend(chunk)
        return results

    def _create_indexes(self, table_name: str):
        """为表创建必要的索引"""
        cursor = self._get_connection().cursor()
        
        try:
            cursor.execute("BEGIN IMMEDIATE")
            
            # 获取需要创建索引的字段
            fields = cursor.execute("""
                SELECT field_name, field_type 
                FROM fields_meta 
                WHERE table_name = ? AND is_indexed = FALSE
            """, [table_name]).fetchall()
            
            for field_name, field_type in fields:
                # 为 TEXT/VARCHAR 类型的字段创建索引
                if field_type in ('TEXT', 'VARCHAR'):
                    index_name = f"idx_{table_name}_{field_name}"
                    # 使用部分索引和前缀索引优化 LIKE 查询
                    cursor.execute(f"""
                        CREATE INDEX IF NOT EXISTS {index_name}
                        ON {self._quote_identifier(table_name)} ({self._quote_identifier(field_name)})
                        WHERE {self._quote_identifier(field_name)} IS NOT NULL
                    """)
                    
                    # 创建前缀索引
                    cursor.execute(f"""
                        CREATE INDEX IF NOT EXISTS {index_name}_prefix
                        ON {self._quote_identifier(table_name)} (
                            substr({self._quote_identifier(field_name)}, 1, 10)
                        )
                        WHERE {self._quote_identifier(field_name)} IS NOT NULL
                    """)
                    
                    # 创建反向索引，用于优化后缀匹配
                    cursor.execute(f"""
                        CREATE INDEX IF NOT EXISTS {index_name}_reverse
                        ON {self._quote_identifier(table_name)} (
                            reverse({self._quote_identifier(field_name)})
                        )
                        WHERE {self._quote_identifier(field_name)} IS NOT NULL
                    """)
                    
                    # 更新索引状态
                    cursor.execute("""
                        UPDATE fields_meta 
                        SET is_indexed = TRUE 
                        WHERE table_name = ? AND field_name = ?
                    """, [table_name, field_name])
                    
            # 分析表以优化查询计划
            cursor.execute(f"ANALYZE {self._quote_identifier(table_name)}")
            
            cursor.execute("COMMIT")
            
        except Exception as e:
            cursor.execute("ROLLBACK")
            raise e

    def to_pandas(self, sql: str, params: tuple = None) -> "pd.DataFrame":
        """将查询结果转换为 Pandas DataFrame
        
        Args:
            sql: SQL语句
            params: 查询参数
            
        Returns:
            pd.DataFrame: 查询结果
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