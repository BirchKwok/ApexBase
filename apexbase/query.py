from typing import List, Optional, Tuple, Dict
import re
from collections import OrderedDict

import polars as pl
import orjson
import pandas as pd
import pyarrow as pa


class QueryError(Exception):
    """Exception raised for query errors."""
    pass


class LRUCache(OrderedDict):
    """LRU cache implementation"""
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


class LazyJSONField:
    """延迟解析的 JSON 字段"""
    def __init__(self, value):
        self._value = value
        self._parsed = None

    def __str__(self):
        return self._value

    def get(self):
        """获取解析后的值"""
        if self._parsed is None:
            try:
                self._parsed = orjson.loads(self._value)
            except:
                self._parsed = self._value
        return self._parsed


class ResultView:
    """Query result view, supports lazy execution and LRU cache"""
    _global_cache = LRUCache(maxsize=1000)

    def __init__(self, storage, query_sql: str, table_name: str = None, fields: str = None):
        self.storage = storage
        self.query_sql = query_sql
        self.fields = fields
        self.table_name = table_name
        
        self._executed = False
        self._cache_key = f"{query_sql}:{table_name if table_name else ''}"
        self._raw_results = None
        self._processed_results = None
        self._json_fields = None
        self._field_names = None

    def _build_query_sql(self, where: str = None) -> Tuple[str, tuple]:
        """构建查询SQL语句
        
        Args:
            where: 查询条件
            
        Returns:
            SQL语句和参数元组
        """
        table_name = self.storage._get_table_name(self.table_name)
        quoted_table = self.storage._quote_identifier(table_name)
        # 获取所有字段
        fields = self.storage.list_fields(table_name)
        field_list = ','.join(self.storage._quote_identifier(f) for f in fields)
        
        if not where or where.strip() == "1=1":
            sql = f"SELECT {field_list} FROM {quoted_table}"
            return sql, ()
        
        try:
            # 处理LIKE查询
            if 'LIKE' in where.upper():
                # 使用正则表达式匹配字段名和模式
                match = re.match(r'\s*(\w+)\s+LIKE\s+[\'"](.+?)[\'"]\s*', where, re.IGNORECASE)
                if match:
                    field, pattern = match.groups()
                    sql = f"""
                        SELECT {field_list} 
                        FROM {quoted_table}
                        WHERE {self.storage._quote_identifier(field)} LIKE ?
                    """
                    return sql, (pattern,)
                else:
                    raise ValueError(f"Invalid LIKE syntax in: {where}")
            
            sql = f"SELECT {field_list} FROM {quoted_table} WHERE {where}"
            return sql, ()
            
        except Exception as e:
            raise ValueError(f"Invalid query syntax: {str(e)}")
        
    def _execute_query(self):
        """执行查询并缓存结果"""
        try:
            query_sql, params = self._build_query_sql(self.query_sql)
            
            # 获取字段名
            self._field_names = self.storage.list_fields()
            if not self._field_names:
                self._ids = []
                self._raw_results = []
                return
            
            # 执行查询
            result = self.storage.query(query_sql, params)
            
            # 处理结果
            self._raw_results = result
            self._ids = []
            id_index = self._field_names.index('_id')
            self._ids = [row[id_index] for row in result]
            
            self._global_cache.put(self._cache_key, (self._ids, self._raw_results))
            self._executed = True
            
        except Exception as e:
            raise QueryError(f"Query execution failed: {str(e)}")

    def _process_row(self, row):
        """处理单行数据"""
        if not self._field_names:
            return {}
            
        data = {}
        for i, field in enumerate(self._field_names):
            value = row[i]
            if value is not None:
                if field in self._json_fields and isinstance(value, str):
                    data[field] = LazyJSONField(value)
                else:
                    data[field] = value
            else:
                data[field] = None
        return data

    def to_dict(self) -> List[Dict]:
        """Convert the result to a list of dictionaries, using cache"""
        return self.to_pandas().to_dict(orient='records')

    def __len__(self):
        """Return the number of results, triggering query execution"""
        return len(self.ids)

    def __getitem__(self, idx):
        """Access the result by index, using cache"""
        return self.to_pandas().iloc[idx]

    def __iter__(self):
        """Iterate over the result, using cache"""
        return iter(self.to_pandas())

    def to_pandas(self) -> "pd.DataFrame":
        """Convert the result to a Pandas DataFrame"""
        query_sql, params = self._build_query_sql(self.query_sql)
        df =  self.storage.to_pandas(query_sql, params)
        if '_id' in df.columns:
            df.set_index('_id', inplace=True)
            df.index.name = None
        return df

    def to_arrow(self) -> "pa.Table":
        """Convert the result to a PyArrow Table, using cache, and set _id as an index"""
        df = self.to_pandas()  # _id is already set as an index
        return pa.Table.from_pandas(df)
    
    def to_polars(self) -> "pl.DataFrame":
        """Convert the result to a Polars DataFrame"""
        return pl.from_pandas(self.to_pandas())

    @property
    def ids(self) -> List[int]:
        """Get the list of IDs for the result, using cache"""
        if not self._executed:
            self._execute_query()
        return self._ids
    
    @property
    def shape(self) -> Tuple[int, int]:
        """Get the shape of the result"""
        return self.to_pandas().shape
    
    @property
    def columns(self) -> List[str]:
        """Get the list of columns for the result"""
        return self.to_pandas().columns
    

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

    def _quote_identifier(self, identifier: str) -> str:
        """
        Correctly escape SQLite identifiers.

        Parameters:
            identifier: str
                The identifier to escape

        Returns:
            str: The escaped identifier
        """
        return f'"{identifier}"'

    def query(self, where: str = None, table_name: str = None) -> ResultView:
        """
        Query records using SQL syntax.

        Parameters:
            where: str
                SQL filter conditions. For example:
                - age > 30
                - name LIKE 'John%'
                - age > 30 AND city = 'New York'
                - field IN (1, 2, 3)

        Returns:
            ResultView: The query result view
        """
        if not isinstance(where, str) or not where.strip():
            raise ValueError("Invalid query syntax")
        
        return ResultView(self.storage, where, table_name)

    def retrieve(self, id_: int) -> Optional[dict]:
        """
        Retrieve a single record.

        Parameters:
            id_: int
                The record ID

        Returns:
            Optional[dict]: The record data, or None if it doesn't exist
        """
        return self.storage.retrieve(id_)

    def retrieve_many(self, ids: List[int]) -> List[dict]:
        """
        Retrieve multiple records.

        Parameters:
            ids: List[int]
                The list of record IDs

        Returns:
            List[dict]: The list of record data
        """
        return self.storage.retrieve_many(ids)

    def list_fields(self) -> List[str]:
        """
        Get the list of all available fields for the current table.

        Returns:
            List[str]: The list of field names
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
        Get the field type.

        Parameters:
            field_name: str
                The field name

        Returns:
            Optional[str]: The field type, or None if the field doesn't exist
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
        Create an index for a specified field.

        Parameters:
            field_name: str
                The field name
        """
        try:
            table_name = self.storage.current_table
            cursor = self.storage.conn.cursor()
            
            # Check if the field exists
            field_exists = cursor.execute(
                f"SELECT 1 FROM {self.storage._quote_identifier(table_name + '_fields_meta')} WHERE field_name = ?",
                [field_name]
            ).fetchone()
            
            if field_exists:
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
        Create temporary indexes for JSON paths.

        Parameters:
            field: str
                The field name
            json_path: str
                The JSON path
        """
        table_name = self.storage.current_table
        cursor = self.storage.conn.cursor()
        
        # Generate a safe index name
        safe_path = json_path.replace('$', '').replace('.', '_').replace('[', '_').replace(']', '_')
        index_name = f"idx_json_{table_name}_{field}_{safe_path.strip('_')}"
        quoted_table = self.storage._quote_identifier(table_name)
        
        # Create index
        cursor.execute(f"""
            CREATE INDEX IF NOT EXISTS {index_name}
            ON {quoted_table}(json_extract({field}, ?))
        """, (json_path,))
        
        # Analyze the newly created index
        cursor.execute("ANALYZE")

    def _validate_json_path(self, json_path: str) -> bool:
        """
        Validate the JSON path syntax.

        Parameters:
            json_path: str
                The JSON path expression, e.g. '$.name' or '$.address.city'

        Returns:
            bool: Whether the path syntax is valid
        """
        if not json_path:
            return False

        # Basic syntax check
        if not json_path.startswith('$'):
            return False

        # Check path components
        parts = json_path[2:].split('.')
        for part in parts:
            if not part:
                return False
            # Check array access syntax
            if '[' in part:
                if not part.endswith(']'):
                    return False
                array_part = part.split('[')[1][:-1]
                if not array_part.isdigit():
                    return False
            # Check normal field names
            else:
                if not re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*$', part):
                    return False

        return True
    
    def retrieve_all(self) -> ResultView:
        """
        Retrieve all records.

        Returns:
            ResultView: The result view
        """
        return self.query("1=1")
    