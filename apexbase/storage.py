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
    使用SQLite作为后端存储的多表存储类。
    """
    def __init__(self, filepath=None, cache_size: int = 1000, batch_size: int = 1000):
        """
        初始化Storage类。

        Parameters:
            filepath: str
                存储文件路径
            cache_size: int
                查询结果缓存的最大数量
            batch_size: int
                批量操作的大小
        """
        if filepath is None:
            raise ValueError("You must provide a file path.")

        self.filepath = Path(filepath)
        self.filepath.parent.mkdir(parents=True, exist_ok=True)

        # 配置参数
        self.batch_size = batch_size
        self._query_cache = LimitedDict(cache_size)
        self._field_cache = LimitedDict(100)  # 缓存字段信息
        self._lock = threading.Lock()  # 添加锁用于并发控制
        self.current_table = "default"  # 默认表名
        self.auto_update_fts = False  # 是否自动更新FTS索引

        # SQLite连接
        self.conn = sqlite3.connect(str(self.filepath), 
                                  isolation_level=None,  # 自动提交模式，手动控制事务
                                  check_same_thread=False)  # 允许跨线程访问
        
        # 优化SQLite配置
        cursor = self.conn.cursor()
        # 使用WAL模式提高写入性能
        cursor.execute("PRAGMA journal_mode=WAL")
        # 设置较大的WAL文件大小限制（64MB）
        cursor.execute("PRAGMA wal_autocheckpoint=1000")
        # 降低同步级别提高性能
        cursor.execute("PRAGMA synchronous=NORMAL")
        # 设置较大的页缓存（约256MB）
        cursor.execute("PRAGMA cache_size=-262144")
        # 临时表和临时文件使用内存
        cursor.execute("PRAGMA temp_store=MEMORY")
        # 启用内存映射，提高读取性能
        cursor.execute("PRAGMA mmap_size=1099511627776")  # 1TB
        # 设置较大的页大小，提高读写效率
        cursor.execute("PRAGMA page_size=32768")  # 32KB
        # 启用严格的外键约束
        cursor.execute("PRAGMA foreign_keys=ON")
        # 读写锁定超时设置（5分钟）
        cursor.execute("PRAGMA busy_timeout=300000")
        
        self._initialize_database()

    def _initialize_database(self):
        """
        初始化SQLite数据库，创建必要的系统表。
        """
        cursor = self.conn.cursor()
        
        # 创建表管理表
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS tables_meta (
                table_name TEXT PRIMARY KEY,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # 创建默认表
        if not self._table_exists("default"):
            self.create_table("default")

    def _get_table_name(self, table_name: str = None) -> str:
        """
        获取实际的表名。

        Parameters:
            table_name: str
                表名，如果为None则使用当前表

        Returns:
            str: 实际的表名
        """
        return table_name if table_name is not None else self.current_table

    def use_table(self, table_name: str):
        """
        切换当前表。

        Parameters:
            table_name: str
                要切换到的表名
        """
        if not self._table_exists(table_name):
            raise ValueError(f"Table '{table_name}' does not exist")
        self.current_table = table_name
        self._invalidate_cache()

    def create_table(self, table_name: str):
        """
        创建新表。

        Parameters:
            table_name: str
                要创建的表名
        """
        if self._table_exists(table_name):
            return

        cursor = self.conn.cursor()
        cursor.execute("BEGIN IMMEDIATE")
        try:
            # 创建记录表
            cursor.execute(f"""
                CREATE TABLE {self._quote_identifier(table_name)} (
                    _id INTEGER PRIMARY KEY AUTOINCREMENT
                )
            """)
            
            # 创建字段元数据表
            cursor.execute(f"""
                CREATE TABLE {self._quote_identifier(table_name + '_fields_meta')} (
                    field_name TEXT PRIMARY KEY,
                    field_type TEXT NOT NULL,
                    is_searchable INTEGER DEFAULT 0,
                    is_indexed INTEGER DEFAULT 0
                )
            """)
            
            # 创建FTS5虚拟表
            cursor.execute(f"""
                CREATE VIRTUAL TABLE {self._quote_identifier(table_name + '_fts')} USING fts5(
                    content,
                    field_name,
                    record_id UNINDEXED,
                    tokenize='porter unicode61'
                )
            """)
            
            # 创建触发器以保持FTS索引同步
            cursor.execute(f"""
                CREATE TRIGGER {self._quote_identifier(table_name + '_fts_delete')} 
                AFTER DELETE ON {self._quote_identifier(table_name)} BEGIN
                    DELETE FROM {self._quote_identifier(table_name + '_fts')} 
                    WHERE record_id = old._id;
                END
            """)
            
            # 记录表信息
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
        删除表。

        Parameters:
            table_name: str
                要删除的表名
        """
        if not self._table_exists(table_name):
            return

        if table_name == "default":
            raise ValueError("Cannot drop the default table")

        cursor = self.conn.cursor()
        try:
            cursor.execute("BEGIN IMMEDIATE")
            
            # 删除相关的所有表和触发器
            cursor.execute(f"DROP TABLE IF EXISTS {self._quote_identifier(table_name)}")
            cursor.execute(f"DROP TABLE IF EXISTS {self._quote_identifier(table_name + '_fields_meta')}")
            cursor.execute(f"DROP TABLE IF EXISTS {self._quote_identifier(table_name + '_fts')}")
            cursor.execute(f"DROP TRIGGER IF EXISTS {self._quote_identifier(table_name + '_fts_delete')}")
            
            # 从表管理表中删除记录
            cursor.execute("DELETE FROM tables_meta WHERE table_name = ?", [table_name])
            
            cursor.execute("COMMIT")
            
            # 如果删除的是当前表，切换到默认表
            if self.current_table == table_name:
                self.use_table("default")
            
            self._invalidate_cache()
            
        except Exception as e:
            cursor.execute("ROLLBACK")
            raise e

    def list_tables(self) -> List[str]:
        """
        列出所有表。

        Returns:
            List[str]: 表名列表
        """
        cursor = self.conn.cursor()
        return [row[0] for row in cursor.execute("SELECT table_name FROM tables_meta ORDER BY table_name")]

    def _table_exists(self, table_name: str) -> bool:
        """
        检查表是否存在。

        Parameters:
            table_name: str
                要检查的表名

        Returns:
            bool: 表是否存在
        """
        cursor = self.conn.cursor()
        return cursor.execute(
            "SELECT 1 FROM tables_meta WHERE table_name = ?",
            [table_name]
        ).fetchone() is not None

    def _quote_identifier(self, identifier: str) -> str:
        """
        正确转义 SQLite 标识符。

        Parameters:
            identifier: str
                需要转义的标识符

        Returns:
            str: 转义后的标识符
        """
        return f'"{identifier}"'

    def _ensure_field_exists(self, field_name: str, field_type: str, is_searchable: bool = True, table_name: str = None):
        """
        确保字段存在，如不存在则创建。

        Parameters:
            field_name: str
                字段名称
            field_type: str
                字段类型 (TEXT, INTEGER, REAL, etc.)
            is_searchable: bool
                是否为可搜索字段
            table_name: str
                表名，如果为None则使用当前表
        """
        table_name = self._get_table_name(table_name)
        try:
            # 检查字段是否已存在
            result = self.conn.execute(
                f"SELECT field_type FROM {self._quote_identifier(table_name + '_fields_meta')} WHERE field_name = ?",
                [field_name]
            ).fetchone()
            
            if not result:
                # 添加字段到表，使用引号包裹字段名
                quoted_field_name = self._quote_identifier(field_name)
                self.conn.execute(f"ALTER TABLE {self._quote_identifier(table_name)} ADD COLUMN {quoted_field_name} {field_type}")
                # 记录字段元数据
                self.conn.execute(
                    f"INSERT INTO {self._quote_identifier(table_name + '_fields_meta')} (field_name, field_type, is_searchable) VALUES (?, ?, ?)",
                    [field_name, field_type, 1 if is_searchable else 0]
                )
        except Exception as e:
            raise ValueError(f"Failed to ensure field exists: {str(e)}")

    def _infer_field_type(self, value: Any) -> str:
        """
        根据值推断字段类型。

        Parameters:
            value: Any
                字段值

        Returns:
            str: SQLite字段类型
        """
        if isinstance(value, bool):
            return "INTEGER"  # SQLite没有布尔类型，使用INTEGER
        elif isinstance(value, int):
            return "INTEGER"
        elif isinstance(value, float):
            return "REAL"
        elif isinstance(value, (list, dict)):
            return "TEXT"  # 复杂类型序列化为JSON字符串
        else:
            return "TEXT"

    def _update_fts_index(self, record_id: int, data: dict, table_name: str = None):
        """
        更新FTS索引。

        Parameters:
            record_id: int
                记录ID
            data: dict
                记录数据
            table_name: str
                表名，如果为None则使用当前表
        """
        table_name = self._get_table_name(table_name)
        cursor = self.conn.cursor()
        
        # 获取可搜索字段
        searchable_fields = cursor.execute(
            f"SELECT field_name FROM {self._quote_identifier(table_name + '_fields_meta')} WHERE is_searchable = 1"
        ).fetchall()
        
        # 删除旧的索引内容
        cursor.execute(f"DELETE FROM {self._quote_identifier(table_name + '_fts')} WHERE record_id = ?", [record_id])
        
        # 为每个可搜索字段添加索引
        for (field_name,) in searchable_fields:
            if field_name in data:
                value = data[field_name]
                if value is not None:
                    # 如果是复杂类型，转换为字符串
                    if isinstance(value, (list, dict)):
                        content = json.dumps(value, ensure_ascii=False)
                    else:
                        content = str(value)
                    
                    # 转义特殊字符
                    content = content.replace(".", " ").replace("@", " ")
                    
                    cursor.execute(
                        f"INSERT INTO {self._quote_identifier(table_name + '_fts')} (content, field_name, record_id) VALUES (?, ?, ?)",
                        [content, field_name, record_id]
                    )

    def set_auto_update_fts(self, enabled: bool):
        """
        设置是否自动更新FTS索引。

        Parameters:
            enabled: bool
                是否启用自动更新
        """
        self.auto_update_fts = enabled

    def store(self, data: dict, table_name: str = None) -> int:
        """
        在存储中存储一条记录。

        Parameters:
            data: dict
                要存储的记录
            table_name: str
                表名，如果为None则使用当前表

        Returns:
            int: 记录的ID
        """
        if not isinstance(data, dict):
            raise ValueError("Only dict-type data is allowed.")

        table_name = self._get_table_name(table_name)
        max_retries = 3
        retry_count = 0
        
        while retry_count < max_retries:
            try:
                with self._lock:  # 使用锁确保并发安全
                    cursor = self.conn.cursor()
                    cursor.execute("BEGIN IMMEDIATE")
                    
                    # 为每个字段创建列
                    for field_name, value in data.items():
                        if field_name != '_id':  # 跳过ID字段
                            field_type = self._infer_field_type(value)
                            self._ensure_field_exists(field_name, field_type, table_name=table_name)
                    
                    # 构建插入语句
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
                    
                    # 如果启用了自动更新FTS索引，则更新
                    if self.auto_update_fts:
                        self._update_fts_index(record_id, data, table_name)
                    
                    cursor.execute("COMMIT")
                    self._invalidate_cache()
                    return record_id
                    
            except sqlite3.OperationalError as e:
                cursor.execute("ROLLBACK")
                if "database is locked" in str(e) and retry_count < max_retries - 1:
                    retry_count += 1
                    time.sleep(0.1 * (2 ** retry_count))  # 指数退避
                    continue
                raise e
            except Exception as e:
                cursor.execute("ROLLBACK")
                raise e

    def batch_store(self, data_list: List[dict], table_name: str = None) -> List[int]:
        """
        批量存储记录。

        Parameters:
            data_list: List[dict]
                要存储的记录列表
            table_name: str
                表名，如果为None则使用当前表

        Returns:
            List[int]: 记录ID列表
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
                
                # 确保所有字段都存在
                all_fields = set()
                for data in data_list:
                    all_fields.update(data.keys())
                all_fields.discard('_id')
                
                for field_name in all_fields:
                    # 使用第一个包含该字段的记录来推断类型
                    for data in data_list:
                        if field_name in data:
                            field_type = self._infer_field_type(data[field_name])
                            self._ensure_field_exists(field_name, field_type, table_name=table_name)
                            break
                
                # 批量插入记录
                for data in data_list:
                    current_batch.append(data)
                    
                    if len(current_batch) >= self.batch_size:
                        batch_ids = self._execute_batch_store(current_batch, table_name)
                        record_ids.extend(batch_ids)
                        current_batch = []
                
                # 处理剩余的记录
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
        执行批量存储操作。

        Parameters:
            batch: List[dict]
                要存储的记录批次
            table_name: str
                表名

        Returns:
            List[int]: 记录ID列表
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
            
            # 如果启用了自动更新FTS索引，则更新
            if self.auto_update_fts:
                self._update_fts_index(record_id, data, table_name)
        
        return record_ids

    def _parse_record(self, row: tuple, table_name: str = None) -> Dict[str, Any]:
        """
        解析记录。

        Parameters:
            row: tuple
                数据库行
            table_name: str
                表名，如果为None则使用当前表

        Returns:
            Dict[str, Any]: 解析后的记录
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
                            # 尝试解析JSON
                            record[field_name] = json.loads(value)
                        except (json.JSONDecodeError, TypeError):
                            record[field_name] = value
                    else:
                        record[field_name] = value
        
        record['_id'] = row[0]  # ID总是第一个字段
        return record

    def create_json_index(self, field_path: str):
        """
        为指定的JSON字段路径创建索引。

        Parameters:
            field_path: str
                JSON字段路径，例如 "$.name" 或 "$.address.city"
        """
        try:
            # 生成安全的索引名
            safe_name = field_path.replace('$', '').replace('.', '_').replace('[', '_').replace(']', '_')
            index_name = f"idx_json_{safe_name.strip('_')}"
            
            self.conn.execute(f"""
                CREATE INDEX IF NOT EXISTS {index_name}
                ON records(json_extract(data, ?))
            """, (field_path,))
            
            # 分析新创建的索引
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
        列出表中的字段。

        Parameters:
            table_name: str
                表名，如果为None则使用当前表
            use_cache: bool
                是否使用缓存

        Returns:
            Dict[str, str]: 字段名到字段类型的映射
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
        优化数据库性能。

        Parameters:
            table_name: str
                表名，如果为None则使用当前表
        """
        table_name = self._get_table_name(table_name)
        cursor = self.conn.cursor()
        
        try:
            cursor.execute("BEGIN TRANSACTION")
            
            # 重建表以回收空间
            cursor.execute(f"VACUUM")
            
            # 更新统计信息
            cursor.execute(f"ANALYZE {self._quote_identifier(table_name)}")
            cursor.execute(f"ANALYZE {self._quote_identifier(table_name + '_fields_meta')}")
            cursor.execute(f"ANALYZE {self._quote_identifier(table_name + '_fts')}")
            
            cursor.execute("COMMIT")
            
        except Exception as e:
            cursor.execute("ROLLBACK")
            raise ValueError(f"Failed to optimize database: {str(e)}")

    def _get_cache_key(self, operation: str, **kwargs) -> str:
        """生成缓存键"""
        return f"{operation}:{orjson.dumps(kwargs).decode('utf-8')}"

    def _invalidate_cache(self):
        """清除所有缓存"""
        self._query_cache.clear()
        self._field_cache.clear()

    def delete(self, id_: int) -> bool:
        """
        删除指定ID的记录。

        Parameters:
            id_: int
                要删除的记录ID

        Returns:
            bool: 删除是否成功
        """
        try:
            table_name = self.current_table
            quoted_table = self._quote_identifier(table_name)
            
            cursor = self.conn.cursor()
            cursor.execute("BEGIN IMMEDIATE")
            try:
                # 检查记录是否存在
                exists = cursor.execute(
                    f"SELECT 1 FROM {quoted_table} WHERE _id = ?",
                    [id_]
                ).fetchone()
                if not exists:
                    cursor.execute("ROLLBACK")
                    return False
                
                # 删除记录
                cursor.execute(f"DELETE FROM {quoted_table} WHERE _id = ?", [id_])
                cursor.execute("COMMIT")
                self._invalidate_cache()
                return True
            except Exception as e:
                cursor.execute("ROLLBACK")
                raise e
        except Exception as e:
            raise ValueError(f"Failed to delete record: {str(e)}")

    def batch_delete(self, ids: List[int]) -> List[int]:
        """
        批量删除记录。

        Parameters:
            ids: List[int]
                要删除的记录ID列表

        Returns:
            List[int]: 成功删除的记录ID列表
        """
        if not ids:
            return []

        try:
            table_name = self.current_table
            quoted_table = self._quote_identifier(table_name)
            quoted_fts = self._quote_identifier(table_name + '_fts')
            
            cursor = self.conn.cursor()
            cursor.execute("BEGIN IMMEDIATE")
            
            try:
                # 禁用触发器
                cursor.execute("DROP TRIGGER IF EXISTS " + self._quote_identifier(table_name + '_fts_delete'))
                
                # 分批处理删除，每批1000条
                batch_size = 1000
                for i in range(0, len(ids), batch_size):
                    batch_ids = ids[i:i + batch_size]
                    placeholders = ','.join('?' * len(batch_ids))
                    
                    # 直接删除FTS索引记录
                    cursor.execute(f"DELETE FROM {quoted_fts} WHERE record_id IN ({placeholders})", batch_ids)
                    # 删除主表记录
                    cursor.execute(f"DELETE FROM {quoted_table} WHERE _id IN ({placeholders})", batch_ids)
                
                # 重新创建触发器
                cursor.execute(f"""
                    CREATE TRIGGER {self._quote_identifier(table_name + '_fts_delete')} 
                    AFTER DELETE ON {quoted_table} BEGIN
                        DELETE FROM {quoted_fts} 
                        WHERE record_id = old._id;
                    END
                """)
                
                cursor.execute("COMMIT")
                self._invalidate_cache()
                return ids
                
            except Exception as e:
                cursor.execute("ROLLBACK")
                raise e
                
        except Exception as e:
            raise ValueError(f"Batch deletion failed: {str(e)}")

    def replace(self, id_: int, data: dict) -> bool:
        """
        替换指定ID的记录。

        Parameters:
            id_: int
                要替换的记录ID
            data: dict
                新的记录数据

        Returns:
            bool: 替换是否成功
        """
        if not isinstance(data, dict):
            raise ValueError("Only dict-type data is allowed.")

        try:
            table_name = self.current_table
            quoted_table = self._quote_identifier(table_name)
            
            cursor = self.conn.cursor()
            
            # 检查记录是否存在
            exists = cursor.execute(
                f"SELECT 1 FROM {quoted_table} WHERE _id = ?",
                [id_]
            ).fetchone()
            if not exists:
                return False

            cursor.execute("BEGIN IMMEDIATE")
            try:
                # 确保所有字段存在
                for field_name, value in data.items():
                    if field_name != '_id':
                        field_type = self._infer_field_type(value)
                        self._ensure_field_exists(field_name, field_type, table_name=table_name)

                # 处理每个字段
                field_updates = []
                params = []
                
                for field_name, value in data.items():
                    if field_name != '_id':
                        quoted_field_name = self._quote_identifier(field_name)
                        field_updates.append(f"{quoted_field_name} = ?")
                        # 如果是复杂类型，序列化为JSON
                        if isinstance(value, (list, dict)):
                            params.append(json.dumps(value))
                        else:
                            params.append(value)

                # 如果有字段需要更新
                if field_updates:
                    update_sql = f"UPDATE {quoted_table} SET {', '.join(field_updates)} WHERE _id = ?"
                    params.append(id_)
                    cursor.execute(update_sql, params)

                # 更新FTS索引
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
        批量替换记录。

        Parameters:
            data_dict: Dict[int, dict]
                要替换的记录字典，key为记录ID，value为新的记录数据

        Returns:
            List[int]: 成功替换的记录ID列表
        """
        if not data_dict:
            return []

        try:
            table_name = self.current_table
            quoted_table = self._quote_identifier(table_name)
            
            cursor = self.conn.cursor()
            cursor.execute("BEGIN IMMEDIATE")
            try:
                # 首先收集所有唯一字段并确保它们存在
                all_fields = set()
                for data in data_dict.values():
                    for field_name, value in data.items():
                        if field_name != '_id':
                            all_fields.add((field_name, self._infer_field_type(value)))

                # 批量创建所有需要的字段
                for field_name, field_type in all_fields:
                    self._ensure_field_exists(field_name, field_type, table_name=table_name)

                # 检查所有ID是否存在
                ids = list(data_dict.keys())
                placeholders = ','.join('?' * len(ids))
                existing_ids = cursor.execute(
                    f"SELECT _id FROM {quoted_table} WHERE _id IN ({placeholders})",
                    ids
                ).fetchall()
                existing_ids = {row[0] for row in existing_ids}

                # 只更新存在的记录
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
        全文搜索。

        Parameters:
            query: str
                搜索查询
            fields: List[str]
                要搜索的字段列表，如果为None则搜索所有可搜索字段
            table_name: str
                表名，如果为None则使用当前表

        Returns:
            List[int]: 匹配记录的ID列表
        """
        table_name = self._get_table_name(table_name)
        cursor = self.conn.cursor()
        
        try:
            # 转义特殊字符
            escaped_query = query.replace(".", " ").replace("@", " ")
            
            if fields:
                # 验证字段是否可搜索
                searchable_fields = cursor.execute(
                    f"SELECT field_name FROM {self._quote_identifier(table_name + '_fields_meta')} WHERE is_searchable = 1"
                ).fetchall()
                searchable_fields = {row[0] for row in searchable_fields}
                
                invalid_fields = set(fields) - searchable_fields
                if invalid_fields:
                    raise ValueError(f"Fields {invalid_fields} are not searchable")
                
                # 构建查询
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
        设置字段是否可搜索。

        Parameters:
            field_name: str
                字段名称
            is_searchable: bool
                是否可搜索
            table_name: str
                表名，如果为None则使用当前表
        """
        table_name = self._get_table_name(table_name)
        cursor = self.conn.cursor()
        
        try:
            cursor.execute("BEGIN TRANSACTION")
            
            # 更新字段元数据
            cursor.execute(
                f"UPDATE {self._quote_identifier(table_name + '_fields_meta')} SET is_searchable = ? WHERE field_name = ?",
                [1 if is_searchable else 0, field_name]
            )
            
            if cursor.rowcount == 0:
                raise ValueError(f"Field {field_name} does not exist")
            
            # 如果设置为可搜索，添加现有数据到FTS索引
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
                # 如果设置为不可搜索，从FTS索引中删除
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
        重建全文搜索索引。

        Parameters:
            table_name: str
                表名，如果为None则使用当前表
        """
        table_name = self._get_table_name(table_name)
        cursor = self.conn.cursor()
        
        try:
            cursor.execute("BEGIN TRANSACTION")
            
            # 清空FTS索引
            cursor.execute(f"DELETE FROM {self._quote_identifier(table_name + '_fts')}")
            
            # 获取可搜索字段
            searchable_fields = cursor.execute(
                f"SELECT field_name FROM {self._quote_identifier(table_name + '_fields_meta')} WHERE is_searchable = 1"
            ).fetchall()
            
            # 重建索引
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
        根据查询模式自动创建索引
        """
        try:
            cursor = self.conn.cursor()
            
            # 获取所有数值类型字段
            numeric_fields = cursor.execute("""
                SELECT field_name 
                FROM fields_meta 
                WHERE field_type IN ('INTEGER', 'REAL')
            """).fetchall()
            
            # 为数值类型字段创建索引（这些字段经常用于范围查询）
            for (field_name,) in numeric_fields:
                index_name = f"idx_{field_name}"
                quoted_field_name = self._quote_identifier(field_name)
                cursor.execute(f"""
                    CREATE INDEX IF NOT EXISTS {index_name}
                    ON records({quoted_field_name})
                """)
            
            # 获取所有TEXT类型字段
            text_fields = cursor.execute("""
                SELECT field_name 
                FROM fields_meta 
                WHERE field_type = 'TEXT'
            """).fetchall()
            
            # 为TEXT类型字段创建LIKE查询优化索引
            for (field_name,) in text_fields:
                index_name = f"idx_{field_name}_like"
                quoted_field_name = self._quote_identifier(field_name)
                cursor.execute(f"""
                    CREATE INDEX IF NOT EXISTS {index_name}
                    ON records({quoted_field_name} COLLATE NOCASE)
                """)
            
            # 分析新创建的索引
            cursor.execute("ANALYZE")
            
        except Exception as e:
            print(f"Warning: Failed to create automatic indexes: {str(e)}")

    def analyze_query_performance(self, query: str) -> dict:
        """
        分析查询性能
        
        Parameters:
            query: str
                要分析的查询语句
                
        Returns:
            dict: 包含查询计划和性能指标的字典
        """
        try:
            cursor = self.conn.cursor()
            
            # 启用查询计划分析
            cursor.execute("EXPLAIN QUERY PLAN " + query)
            query_plan = cursor.fetchall()
            
            # 收集性能指标
            metrics = {
                'tables_used': set(),
                'indexes_used': set(),
                'scan_type': [],
                'estimated_rows': 0
            }
            
            for step in query_plan:
                detail = step[3]  # 查询计划详情
                
                # 分析使用的表
                if 'TABLE' in detail:
                    table = detail.split('TABLE')[1].split()[0]
                    metrics['tables_used'].add(table)
                
                # 分析使用的索引
                if 'USING INDEX' in detail:
                    index = detail.split('USING INDEX')[1].split()[0]
                    metrics['indexes_used'].add(index)
                
                # 分析扫描类型
                if 'SCAN' in detail:
                    scan_type = detail.split('SCAN')[0].strip()
                    metrics['scan_type'].append(scan_type)
                
                # 估算处理的行数
                if 'rows=' in detail:
                    rows = int(detail.split('rows=')[1].split()[0])
                    metrics['estimated_rows'] = max(metrics['estimated_rows'], rows)
            
            return metrics
            
        except Exception as e:
            print(f"Warning: Failed to analyze query performance: {str(e)}")
            return {}

    def count_rows(self, table_name: str = None) -> int:
        """
        返回指定表的行数。

        Parameters:
            table_name: str
                表名，如果为None则使用当前表

        Returns:
            int: 表中的记录数

        Raises:
            ValueError: 当表不存在时抛出
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
        检索单条记录。

        Parameters:
            id_: int
                记录ID

        Returns:
            Optional[dict]: 记录数据，如果不存在则返回None
        """
        table_name = self.current_table
        quoted_table = self._quote_identifier(table_name)
        cursor = self.conn.cursor()
        
        try:
            # 获取所有字段
            fields = self.list_fields()
            if not fields:
                return None
            
            # 构建查询
            field_selects = [f"{self._quote_identifier(field)}" for field in fields]
            sql = f"SELECT _id, {', '.join(field_selects)} FROM {quoted_table} WHERE _id = ?"
            
            # 执行查询
            result = cursor.execute(sql, [id_]).fetchone()
            if not result:
                return None
            
            # 构建记录字典
            record = {"_id": result[0]}
            for i, field in enumerate(fields, 1):
                value = result[i]
                if value is not None:
                    # 尝试解析JSON字符串
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
        批量检索记录。

        Parameters:
            ids: List[int]
                记录ID列表

        Returns:
            List[dict]: 记录数据列表
        """
        if not ids:
            return []

        table_name = self.current_table
        quoted_table = self._quote_identifier(table_name)
        cursor = self.conn.cursor()
        
        try:
            # 获取所有字段
            fields = self.list_fields()
            if not fields:
                return []
            
            # 构建查询
            field_selects = [f"{self._quote_identifier(field)}" for field in fields]
            placeholders = ','.join('?' * len(ids))
            sql = f"SELECT _id, {', '.join(field_selects)} FROM {quoted_table} WHERE _id IN ({placeholders})"
            
            # 执行查询
            results = cursor.execute(sql, ids).fetchall()
            
            # 构建记录列表
            records = []
            for result in results:
                record = {"_id": result[0]}
                for i, field in enumerate(fields, 1):
                    value = result[i]
                    if value is not None:
                        # 尝试解析JSON字符串
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
