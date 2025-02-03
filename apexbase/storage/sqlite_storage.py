import sqlite3
import orjson
from typing import Dict, List, Any, Optional
from pathlib import Path
import json
import threading
import time
import os

from .base import BaseStorage
from ..limited_dict import LimitedDict



class SQLiteStorage(BaseStorage):
    """SQLite implementation of the storage backend."""
    
    def __init__(self, db_path: str, batch_size: int = 1000):
        """初始化SQLite存储
        
        Args:
            db_path: 数据库文件路径
            batch_size: 批处理大小
        """
        self.db_path = db_path
        self.batch_size = batch_size
        self._field_cache = LimitedDict(100)
        self._lock = threading.Lock()
        self.current_table = "default"
        
        # 创建数据库目录
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        
        # 初始化数据库
        self._initialize_database()

    def _get_connection(self):
        """获取数据库连接"""
        if not hasattr(self, '_conn'):
            self._conn = sqlite3.connect(self.db_path)
            self._conn.row_factory = sqlite3.Row
            
            # 设置数据库参数
            cursor = self._conn.cursor()
            cursor.execute("PRAGMA journal_mode=WAL")
            cursor.execute("PRAGMA wal_autocheckpoint=1000")
            cursor.execute("PRAGMA synchronous=NORMAL")
            cursor.execute("PRAGMA cache_size=-262144")
            cursor.execute("PRAGMA temp_store=MEMORY")
            cursor.execute("PRAGMA mmap_size=1099511627776")
            cursor.execute("PRAGMA page_size=32768")
            cursor.execute("PRAGMA foreign_keys=ON")
            cursor.execute("PRAGMA busy_timeout=300000")
            
        return self._conn

    def _initialize_database(self):
        """初始化数据库,创建必要的表"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            # 设置数据库参数
            cursor.execute('PRAGMA journal_mode=WAL')
            cursor.execute('PRAGMA synchronous=NORMAL')
            cursor.execute('PRAGMA cache_size=10000')
            cursor.execute('PRAGMA temp_store=MEMORY')
            cursor.execute('PRAGMA mmap_size=30000000000')
            
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
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            ''')
            
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS fields_meta (
                table_name TEXT,
                field_name TEXT,
                field_type TEXT,
                PRIMARY KEY (table_name, field_name),
                FOREIGN KEY (table_name) REFERENCES tables_meta(table_name)
            )
            ''')
            
            # 初始化元数据
            cursor.execute('INSERT OR IGNORE INTO tables_meta (table_name) VALUES (?)', ('default',))
            
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

        cursor = self._get_connection().cursor()
        try:
            # Create main table
            cursor.execute(f"""
                CREATE TABLE IF NOT EXISTS {self._quote_identifier(table_name)} (
                    _id INTEGER PRIMARY KEY,
                    data TEXT
                )
            """)
            
            # Add table to tables_meta
            cursor.execute(
                "INSERT OR IGNORE INTO tables_meta (table_name) VALUES (?)",
                [table_name]
            )
            
            # Add data field to fields_meta
            cursor.execute(
                "INSERT OR IGNORE INTO fields_meta (table_name, field_name, field_type) VALUES (?, ?, ?)",
                [table_name, "data", "TEXT"]
            )
            
            self._get_connection().commit()
            
        except Exception as e:
            self._get_connection().rollback()
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

        cursor = self._get_connection().cursor()
        try:
            cursor.execute("BEGIN IMMEDIATE")
            
            cursor.execute(f"DROP TABLE IF EXISTS {self._quote_identifier(table_name)}")
            cursor.execute(f"DROP TABLE IF EXISTS {self._quote_identifier(table_name + '_fields_meta')}")
            
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
            cursor.execute("BEGIN TRANSACTION")
            
            # 获取现有字段
            existing_fields = set()
            for row in cursor.execute(
                "SELECT field_name FROM fields_meta WHERE table_name = ?",
                [table_name]
            ):
                existing_fields.add(row[0])
            
            # 添加新字段
            for field_name, value in data.items():
                if field_name != '_id' and field_name not in existing_fields:
                    field_type = self._infer_field_type(value)
                    quoted_field = self._quote_identifier(field_name)
                    
                    # 添加字段到表
                    cursor.execute(
                        f"ALTER TABLE {self._quote_identifier(table_name)} ADD COLUMN {quoted_field} {field_type}"
                    )
                    
                    # 更新元数据
                    cursor.execute(
                        "INSERT INTO fields_meta (table_name, field_name, field_type) VALUES (?, ?, ?)",
                        [table_name, field_name, field_type]
                    )
            
            cursor.execute("COMMIT")
            
        except Exception as e:
            cursor.execute("ROLLBACK")
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
        
        # 确保所有字段存在
        self._ensure_fields_exist(data, table_name)
        
        with self._get_connection() as conn:
            cursor = conn.cursor()
            try:
                # 准备插入语句
                fields = []
                values = []
                params = []
                
                for field_name, value in data.items():
                    if field_name != '_id':
                        fields.append(self._quote_identifier(field_name))
                        values.append('?')
                        if isinstance(value, (list, dict)):
                            params.append(json.dumps(value))
                        else:
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
        
        # 确保所有字段存在
        all_fields = set()
        for data in data_list:
            all_fields.update(data.keys())
        
        # 创建一个包含所有字段的示例数据
        example_data = {field: None for field in all_fields}
        self._ensure_fields_exist(example_data, table_name)
        
        with self._get_connection() as conn:
            cursor = conn.cursor()
            try:
                # 获取所有字段
                fields = []
                for row in cursor.execute(
                    "SELECT field_name FROM fields_meta WHERE table_name = ? AND field_name != '_id'",
                    [table_name]
                ):
                    fields.append(row[0])
                
                # 准备批量插入
                placeholders = ','.join(['?' for _ in fields])
                sql = f"""
                    INSERT INTO {self._quote_identifier(table_name)}
                    ({','.join(self._quote_identifier(f) for f in fields)})
                    VALUES ({placeholders})
                """
                
                # 准备数据
                values = []
                for data in data_list:
                    row = []
                    for field in fields:
                        value = data.get(field)
                        if isinstance(value, (list, dict)):
                            row.append(json.dumps(value))
                        else:
                            row.append(value)
                    values.append(tuple(row))
                
                # 执行批量插入
                cursor.executemany(sql, values)
                
                # 获取插入的ID范围
                first_id = cursor.lastrowid - len(data_list) + 1
                
                conn.commit()
                return list(range(first_id, first_id + len(data_list)))
                
            except Exception as e:
                conn.rollback()
                raise e

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
            
            self._get_connection().execute(f"""
                CREATE INDEX IF NOT EXISTS {index_name}
                ON records(json_extract(data, ?))
            """, (field_path,))
            
            self._get_connection().execute("ANALYZE")
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
            result = self._get_connection().execute(f"""
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

    def list_fields(self, table_name: str = None) -> List[str]:
        """获取表的所有字段
        
        Args:
            table_name: 表名
            
        Returns:
            字段列表
        """
        table_name = self._get_table_name(table_name)
        cursor = self._get_connection().cursor()
        
        # 获取所有字段，包括_id
        fields = ['_id']  # 确保_id总是第一个字段
        for row in cursor.execute(
            "SELECT field_name FROM fields_meta WHERE table_name = ? ORDER BY field_name",
            [table_name]
        ):
            fields.append(row[0])
            
        return fields

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
            cursor.execute("BEGIN TRANSACTION")
            
            cursor.execute(f"VACUUM")
            
            cursor.execute(f"ANALYZE {self._quote_identifier(table_name)}")
            cursor.execute(f"ANALYZE {self._quote_identifier(table_name + '_fields_meta')}")
            
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
            
            cursor = self._get_connection().cursor()
            cursor.execute("BEGIN IMMEDIATE")
            
            try:
                batch_size = 1000
                for i in range(0, len(ids), batch_size):
                    batch_ids = ids[i:i + batch_size]
                    placeholders = ','.join('?' * len(batch_ids))
                    
                    cursor.execute(f"DELETE FROM {quoted_table} WHERE _id IN ({placeholders})", batch_ids)
                
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
            
            cursor = self._get_connection().cursor()
            
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
            
            cursor = self._get_connection().cursor()
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

    def set_searchable(self, field_name: str, is_searchable: bool = True, table_name: str = 'default'):
        """设置字段是否可搜索
        
        Args:
            field_name: 字段名
            is_searchable: 是否可搜索
            table_name: 表名
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            # 更新字段元数据
            cursor.execute(
                'UPDATE fields_meta SET is_searchable = ? WHERE table_name = ? AND field_name = ?',
                (1 if is_searchable else 0, table_name, field_name)
            )
            
            # 如果字段不存在,则插入
            if cursor.rowcount == 0:
                cursor.execute(
                    'INSERT INTO fields_meta (table_name, field_name, field_type) VALUES (?, ?, ?)',
                    (table_name, field_name, 'TEXT')
                )
            
            conn.commit()

    def rebuild_fts_index(self, table_name: str = None):
        """
        Rebuilds the full-text search index.

        Parameters:
            table_name: str
                The table name, or None to use the current table
        """
        table_name = self._get_table_name(table_name)
        cursor = self._get_connection().cursor()
        
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

    def close(self):
        """关闭数据库连接"""
        if hasattr(self, '_conn'):
            self._conn.close()
            del self._conn

    def _create_auto_indexes(self):
        """
        Automatically create indexes based on query patterns
        """
        cursor = self._get_connection().cursor()
        
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
            cursor = self._get_connection().cursor()
            
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
            cursor = self._get_connection().cursor()
            result = cursor.execute(
                f"SELECT COUNT(*) FROM {self._quote_identifier(table_name)}"
            ).fetchone()
            return result[0] if result else 0
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

    def query(self, sql: str, params: tuple) -> List[tuple]:
        """执行自定义SQL查询
        
        Args:
            sql: SQL语句
            params: 查询参数
            
        Returns:
            查询结果
        """
        cursor = self._get_connection().cursor()
        return cursor.execute(sql, params).fetchall()