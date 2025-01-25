from typing import List, Optional, Tuple
import re
import sqlite3
from collections import OrderedDict

from .sql_parser import SQLParser, SQLGenerator
import pandas as pd
import pyarrow as pa


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


class ResultView:
    """Query result view, supports lazy execution and LRU cache"""
    _global_cache = LRUCache(maxsize=1000)

    def __init__(self, storage, query_sql: str, params: tuple = None):
        self.storage = storage
        self.query_sql = query_sql
        self.params = params if params is not None else ()
        self._executed = False
        self._cache_key = f"{query_sql}:{params}"

    def _execute_query(self):
        """Execute the query and cache the result"""
        if not self._executed:
            # Try to get the result from the cache
            cached_result = self._global_cache.get(self._cache_key)
            if cached_result is not None:
                self._ids, self._results = cached_result
            else:
                try:
                    cursor = self.storage.conn.cursor()
                    self._ids = [row[0] for row in cursor.execute(self.query_sql, self.params)]
                    self._results = None
                    self._global_cache.put(self._cache_key, (self._ids, self._results))
                except sqlite3.OperationalError as e:
                    raise ValueError(f"Invalid query syntax: {str(e)}")
            self._executed = True

    @property
    def ids(self) -> List[int]:
        """Get the list of IDs for the result, using cache"""
        if not self._executed:
            self._execute_query()
        return self._ids

    def to_dict(self) -> List[dict]:
        """Convert the result to a list of dictionaries, using cache"""
        if not self._executed:
            self._execute_query()
        
        # Check if the full result is in the cache
        cached_result = self._global_cache.get(self._cache_key)
        if cached_result and cached_result[1] is not None:
            return cached_result[1]
        
        # Get the full records and update the cache
        self._results = self.storage.retrieve_many(self._ids)
        self._global_cache.put(self._cache_key, (self._ids, self._results))
        return self._results

    def __len__(self):
        """Return the number of results, triggering query execution"""
        return len(self.ids)

    def __getitem__(self, idx):
        """Access the result by index, using cache"""
        return self.to_dict()[idx]

    def __iter__(self):
        """Iterate over the result, using cache"""
        return iter(self.to_dict())

    def to_pandas(self) -> "pd.DataFrame":
        """Convert the result to a Pandas DataFrame, using cache, and set _id as an unnamed index"""
        data = self.to_dict()
        df = pd.DataFrame(data)
        if '_id' in df.columns:
            df.set_index('_id', inplace=True)
            df.index.name = None
        return df

    def to_arrow(self) -> "pa.Table":
        """Convert the result to a PyArrow Table, using cache, and set _id as an index"""
        df = self.to_pandas()  # _id is already set as an index
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

    def _build_query_sql(self, query_filter: str = None) -> Tuple[str, tuple]:
        """Build the query SQL statement"""
        table_name = self.storage._get_table_name(None)
        quoted_table = self.storage._quote_identifier(table_name)
        
        if not query_filter or query_filter.strip() == "1=1":
            sql = f"SELECT _id FROM {quoted_table}"
            return sql, ()
        
        try:
            ast = self.parser.parse(query_filter)
            
            self.generator.reset()
            where_clause = self.generator.generate(ast)
            params = self.generator.get_parameters()
            
            sql = f"SELECT _id FROM {quoted_table} WHERE {where_clause}"
            return sql, tuple(params)
            
        except Exception as e:
            raise ValueError(f"Invalid query syntax: {str(e)}")

    def query(self, query_filter: str = None) -> ResultView:
        """
        Query records using SQL syntax.

        Parameters:
            query_filter: str
                SQL filter conditions. For example:
                - age > 30
                - name LIKE 'John%'
                - age > 30 AND city = 'New York'
                - field IN (1, 2, 3)

        Returns:
            ResultView: The query result view
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
        Full-text search.

        Parameters:
            text: str
                The search text
            fields: List[str]
                The list of fields to search, if None, search all searchable fields
            table_name: str
                The table name, if None, use the current table

        Returns:
            ResultView: The search result view
        """
        table_name = self.storage._get_table_name(table_name)
        quoted_fts = self.storage._quote_identifier(table_name + '_fts')
        
        # Get searchable fields
        if fields:
            field_list = [f"'{field}'" for field in fields]
            field_filter = f"AND field_name IN ({','.join(field_list)})"
        else:
            field_filter = ""
        
        # Build FTS query
        sql = f"""
            SELECT DISTINCT record_id as _id
            FROM {quoted_fts}
            WHERE content MATCH ?
            {field_filter}
        """
        
        return ResultView(self.storage, sql, (text,))

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
        try:
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
        except Exception as e:
            # If index creation fails, record the error but continue execution
            print(f"Warning: Failed to create JSON index: {str(e)}")

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
    