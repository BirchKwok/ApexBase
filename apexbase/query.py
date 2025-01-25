from typing import List, Optional, Tuple
import re
import sqlite3
from collections import OrderedDict

from .limited_dict import LimitedDict
from .sql_parser import SQLParser, SQLGenerator
import pandas as pd
import pyarrow as pa


class LRUCache(OrderedDict):
    """LRU缓存实现"""
    def __init__(self, maxsize=1000):
        super().__init__()
        self.maxsize = maxsize

    def get(self, key, default=None):
        try:
            value = self[key]
            self.move_to_end(key)
            return value
        except KeyError:
            return default

    def put(self, key, value):
        if key in self:
            self.move_to_end(key)
        self[key] = value
        if len(self) > self.maxsize:
            self.popitem(last=False)


class ResultView:
    """查询结果视图，支持延迟执行和LRU缓存"""
    _global_cache = LRUCache(maxsize=1000)  # 全局LRU缓存

    def __init__(self, storage, query_sql: str, params: tuple = None):
        self.storage = storage
        self.query_sql = query_sql
        self.params = params if params is not None else ()
        self._executed = False  # 标记是否已执行查询
        self._cache_key = f"{query_sql}:{params}"  # 缓存键

    def _execute_query(self):
        """执行查询并缓存结果"""
        if not self._executed:
            # 尝试从缓存获取结果
            cached_result = self._global_cache.get(self._cache_key)
            if cached_result is not None:
                self._ids, self._results = cached_result
            else:
                # 执行查询并缓存结果
                try:
                    cursor = self.storage.conn.cursor()
                    self._ids = [row[0] for row in cursor.execute(self.query_sql, self.params)]
                    self._results = None  # 延迟加载记录
                    self._global_cache.put(self._cache_key, (self._ids, self._results))
                except sqlite3.OperationalError as e:
                    raise ValueError(f"Invalid query syntax: {str(e)}")
            self._executed = True

    @property
    def ids(self) -> List[int]:
        """获取结果的ID列表，使用缓存"""
        if not self._executed:
            self._execute_query()
        return self._ids

    def to_dict(self) -> List[dict]:
        """将结果转换为字典列表，使用缓存"""
        if not self._executed:
            self._execute_query()
        
        # 检查缓存中是否有完整结果
        cached_result = self._global_cache.get(self._cache_key)
        if cached_result and cached_result[1] is not None:
            return cached_result[1]
        
        # 获取完整记录并更新缓存
        self._results = self.storage.retrieve_many(self._ids)
        self._global_cache.put(self._cache_key, (self._ids, self._results))
        return self._results

    def __len__(self):
        """返回结果数量，触发查询执行"""
        return len(self.ids)

    def __getitem__(self, idx):
        """通过索引访问结果，使用缓存"""
        return self.to_dict()[idx]

    def __iter__(self):
        """迭代结果，使用缓存"""
        return iter(self.to_dict())

    def to_pandas(self) -> "pd.DataFrame":
        """将结果转换为Pandas DataFrame，使用缓存，并将_id设置为无名称索引"""
        data = self.to_dict()
        df = pd.DataFrame(data)
        if '_id' in df.columns:
            df.set_index('_id', inplace=True)
            df.index.name = None
        return df

    def to_arrow(self) -> "pa.Table":
        """将结果转换为PyArrow Table，使用缓存，并将_id设置为索引"""
        df = self.to_pandas()  # 已经设置了_id为索引
        return pa.Table.from_pandas(df)


class Query:
    """
    The FieldsQuery class is used to query data in the fields_cache.
    Supports direct SQL-like query syntax for filtering records.
    
    Examples:
        - Basic comparison: "age > 18"
        - Range query: "10 < days <= 300"
        - Text search: "name LIKE '%John%'"
        - Multiple conditions: "age > 18 AND city = 'New York'"
        - JSON field access: "json_extract(data, '$.address.city') = 'Beijing'"
        - Numeric operations: "CAST(json_extract(data, '$.price') AS REAL) * CAST(json_extract(data, '$.quantity') AS REAL) > 1000"
    """
    def __init__(self, storage):
        """
        Initialize the FieldsQuery class.

        Parameters:
            storage: Storage
                The storage object.
        """
        self.storage = storage
        self.parser = SQLParser()
        self.generator = SQLGenerator()
        self._query_cache = LimitedDict(1000)  # 缓存查询结果

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

    def _build_query_sql(self, query_filter: str = None) -> Tuple[str, tuple]:
        """构建查询SQL语句"""
        table_name = self.storage._get_table_name(None)
        quoted_table = self.storage._quote_identifier(table_name)
        
        if not query_filter or query_filter.strip() == "1=1":
            sql = f"SELECT _id FROM {quoted_table}"
            return sql, ()
        
        try:
            # 使用 SQLParser 解析查询
            ast = self.parser.parse(query_filter)
            
            # 使用 SQLGenerator 生成 SQL 和参数
            self.generator.reset()
            where_clause = self.generator.generate(ast)
            params = self.generator.get_parameters()
            
            sql = f"SELECT _id FROM {quoted_table} WHERE {where_clause}"
            return sql, tuple(params)
            
        except Exception as e:
            raise ValueError(f"Invalid query syntax: {str(e)}")

    def query(self, query_filter: str = None) -> ResultView:
        """
        使用SQL语法查询记录。

        Parameters:
            query_filter: str
                SQL过滤条件。例如：
                - age > 30
                - name LIKE 'John%'
                - age > 30 AND city = 'New York'
                - field IN (1, 2, 3)
                不支持 ORDER BY, GROUP BY, HAVING 等语句

        Returns:
            ResultView: 查询结果视图
        """
        if not isinstance(query_filter, str) or not query_filter.strip():
            raise ValueError("Invalid query syntax")
            
        try:
            sql, params = self._build_query_sql(query_filter)
            return ResultView(self.storage, sql, params)
        except (ValueError, sqlite3.OperationalError) as e:
            raise ValueError(f"Invalid query syntax: {str(e)}")

    def search_text(self, text: str, fields: List[str] = None, table_name: str = None) -> ResultView:
        """
        全文搜索。

        Parameters:
            text: str
                搜索文本
            fields: List[str]
                要搜索的字段列表，如果为None则搜索所有可搜索字段
            table_name: str
                表名，如果为None则使用当前表

        Returns:
            ResultView: 搜索结果视图
        """
        table_name = self.storage._get_table_name(table_name)
        quoted_fts = self.storage._quote_identifier(table_name + '_fts')
        
        # 获取可搜索字段
        if fields:
            field_list = [f"'{field}'" for field in fields]
            field_filter = f"AND field_name IN ({','.join(field_list)})"
        else:
            field_filter = ""
        
        # 构建FTS查询
        sql = f"""
            SELECT DISTINCT record_id as _id
            FROM {quoted_fts}
            WHERE content MATCH ?
            {field_filter}
        """
        
        return ResultView(self.storage, sql, (text,))

    def retrieve(self, id_: int) -> Optional[dict]:
        """
        检索单条记录。

        Parameters:
            id_: int
                记录ID

        Returns:
            Optional[dict]: 记录数据，如果不存在则返回None
        """
        return self.storage.retrieve(id_)

    def retrieve_many(self, ids: List[int]) -> List[dict]:
        """
        批量检索记录。

        Parameters:
            ids: List[int]
                记录ID列表

        Returns:
            List[dict]: 记录数据列表
        """
        return self.storage.retrieve_many(ids)

    def list_fields(self) -> List[str]:
        """
        获取当前表的所有可用字段列表。

        Returns:
            List[str]: 字段名列表
        """
        try:
            cursor = self.storage.conn.cursor()
            table_name = self.storage.current_table
            results = cursor.execute(
                f"SELECT field_name FROM {self.storage._quote_identifier(table_name + '_fields_meta')}"
            ).fetchall()
            return [row[0] for row in results]
        except Exception as e:
            raise ValueError(f"Failed to list fields: {str(e)}")

    def get_field_type(self, field_name: str) -> Optional[str]:
        """
        获取字段类型。

        Parameters:
            field_name: str
                字段名称

        Returns:
            Optional[str]: 字段类型，如果字段不存在则返回None
        """
        try:
            table_name = self.storage.current_table
            cursor = self.storage.conn.cursor()
            result = cursor.execute(
                f"SELECT field_type FROM {self.storage._quote_identifier(table_name + '_fields_meta')} WHERE field_name = ?",
                [field_name]
            ).fetchone()
            return result[0] if result else None
        except Exception as e:
            raise ValueError(f"Failed to get field type: {str(e)}")

    def create_field_index(self, field_name: str):
        """
        为指定字段创建索引。

        Parameters:
            field_name: str
                字段名称
        """
        try:
            table_name = self.storage.current_table
            cursor = self.storage.conn.cursor()
            
            # 检查字段是否存在
            field_exists = cursor.execute(
                f"SELECT 1 FROM {self.storage._quote_identifier(table_name + '_fields_meta')} WHERE field_name = ?",
                [field_name]
            ).fetchone()
            
            if field_exists:
                # 创建索引，使用引号包裹字段名
                index_name = f"idx_{table_name}_{field_name}"
                quoted_field_name = self._quote_identifier(field_name)
                quoted_table = self.storage._quote_identifier(table_name)
                cursor.execute(f"""
                    CREATE INDEX IF NOT EXISTS {index_name}
                    ON {quoted_table}({quoted_field_name})
                """)
                cursor.execute("ANALYZE")
            else:
                raise ValueError(f"Field {field_name} does not exist")
        except Exception as e:
            raise ValueError(f"Failed to create index: {str(e)}")

    def _create_temp_indexes(self, field: str, json_path: str):
        """
        为JSON路径创建临时索引。

        Parameters:
            field: str
                字段名
            json_path: str
                JSON路径
        """
        try:
            table_name = self.storage.current_table
            cursor = self.storage.conn.cursor()
            
            # 生成安全的索引名
            safe_path = json_path.replace('$', '').replace('.', '_').replace('[', '_').replace(']', '_')
            index_name = f"idx_json_{table_name}_{field}_{safe_path.strip('_')}"
            quoted_table = self.storage._quote_identifier(table_name)
            
            # 创建索引
            cursor.execute(f"""
                CREATE INDEX IF NOT EXISTS {index_name}
                ON {quoted_table}(json_extract({field}, ?))
            """, (json_path,))
            
            # 分析新创建的索引
            cursor.execute("ANALYZE")
        except Exception as e:
            # 如果创建索引失败，记录错误但继续执行
            print(f"Warning: Failed to create JSON index: {str(e)}")

    def _validate_json_path(self, json_path: str) -> bool:
        """
        验证 JSON 路径语法。

        Parameters:
            json_path: str
                JSON 路径表达式，例如 '$.name' 或 '$.address.city'

        Returns:
            bool: 路径语法是否有效
        """
        if not json_path:
            return False

        # 基本语法检查
        if not json_path.startswith('$'):
            return False

        # 检查路径组件
        parts = json_path[2:].split('.')
        for part in parts:
            if not part:
                return False
            # 检查数组访问语法
            if '[' in part:
                if not part.endswith(']'):
                    return False
                array_part = part.split('[')[1][:-1]
                if not array_part.isdigit():
                    return False
            # 检查普通字段名
            else:
                if not re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*$', part):
                    return False

        return True
    
