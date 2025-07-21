import shutil
from typing import List, Dict, Union, Optional
from pathlib import Path

from .storage import create_storage
from .query import Query, ResultView


version = "0.1.0"


class ApexClient:
    def __init__(
        self, 
        dirpath=None, 
        batch_size: int = 1000, 
        drop_if_exists: bool = False,
        enable_cache: bool = True,
        cache_size: int = 10000
    ):
        """
        Initializes a new instance of the ApexClient class.

        Parameters:
            dirpath: str
                The directory path for storing data. If None, the current directory is used.
            batch_size: int
                The size of batch operations.
            drop_if_exists: bool
                If True, the database file will be deleted if it already exists.
            enable_cache: bool
                Whether to enable caching for better write performance.
            cache_size: int
                The size of the cache for batch operations.
        """
        if dirpath is None:
            dirpath = "."
        
        self._dirpath = Path(dirpath)
        if drop_if_exists and self._dirpath.exists():
            shutil.rmtree(self._dirpath)
            
        self._dirpath.mkdir(parents=True, exist_ok=True)
        
        self._db_path = self._dirpath / "apexbase.db"
        
        self._storage = create_storage(str(self._db_path), 
                                      batch_size=batch_size, 
                                      enable_cache=enable_cache, cache_size=cache_size)
        self._query_handler = Query(self._storage)
        self._current_table = "default"  # Default table name

    def use_table(self, table_name: str):
        """
        Switches the current table for operations.

        Parameters:
            table_name: str
                The name of the table to switch to.
        """
        self._current_table = table_name
        self._storage.use_table(table_name)

    @property
    def current_table(self):
        """
        Returns the current table name.
        """
        return self._current_table

    def create_table(self, table_name: str):
        """
        Creates a new table and switches to it.

        Parameters:
            table_name: str
                The name of the table to create.
        """
        self._storage.create_table(table_name)
        self.use_table(table_name)  # 创建表后立即切换到该表

    def drop_table(self, table_name: str):
        """
        Drops a table.

        Parameters:
            table_name: str
                The name of the table to drop.
        """
        self._storage.drop_table(table_name)
        # If the table being dropped is the current table, switch to the default table
        if self._current_table == table_name:
            self._current_table = "default"

    def list_tables(self) -> List[str]:
        """
        Lists all tables.

        Returns:
            List[str]: A list of table names
        """
        return self._storage.list_tables()

    def store(self, data: Union[dict, List[dict]]) -> Union[int, List[int]]:
        """
        Stores one or more records.

        Parameters:
            data: Union[dict, List[dict]]
                The records to store, either as a single dictionary or a list of dictionaries.

        Returns:
            Union[int, List[int]]: The record ID or ID list
        """
        if isinstance(data, dict):
            # Single record
            return self._storage.store(data)
        elif isinstance(data, list):
            # Multiple records
            return self._storage.batch_store(data)
        else:
            raise ValueError("Data must be a dict or a list of dicts")

    def query(self, where: str = None) -> ResultView:
        """
        Queries records using SQL syntax.

        Parameters:
            where: str
                SQL filter conditions. For example:
                - age > 30
                - name LIKE 'John%'
                - age > 30 AND city = 'New York'
                - field IN (1, 2, 3)
                - ORDER BY, GROUP BY, HAVING are not supported

        Returns:
            ResultView: A view of query results, supporting deferred execution
        """
        return self._query_handler.query(where)

    def retrieve(self, id_: int) -> Optional[dict]:
        """
        Retrieves a single record.

        Parameters:
            id_: int
                The record ID

        Returns:
            Optional[dict]: The record data, or None if it doesn't exist
        """
        return self._query_handler.retrieve(id_)

    def retrieve_many(self, ids: List[int]) -> List[dict]:
        """
        Retrieves multiple records.

        Parameters:
            ids: List[int]
                The list of record IDs

        Returns:
            List[dict]: The list of record data
        """
        return self._query_handler.retrieve_many(ids)
    
    def retrieve_all(self) -> ResultView:
        return self._query_handler.retrieve_all()

    def list_fields(self):
        """
        List the fields in the cache.

        Returns:
            List[str]: List of fields.
        """
        return self._storage.list_fields()

    def delete(self, ids: Union[int, List[int]]) -> bool:
        """
        Deletes a single record.

        Parameters:
            ids: Union[int, List[int]]
                The record ID or list of record IDs to delete

        Returns:
            bool: Whether the deletion was successful
        """
        if isinstance(ids, int):
            return self._storage.delete(ids)
        elif isinstance(ids, list):
            return self._storage.batch_delete(ids)
        else:
            raise ValueError("ids must be an int or a list of ints")

    def replace(self, id_: int, data: dict) -> bool:
        """
        Replaces a single record.

        Parameters:
            id_: int
                The record ID to replace
            data: dict
                The new record data

        Returns:
            bool: Whether the replacement was successful
        """
        return self._storage.replace(id_, data)

    def batch_replace(self, data_dict: Dict[int, dict]) -> List[int]:
        """
        Replaces multiple records.

        Parameters:
            data_dict: Dict[int, dict]
                The dictionary of records to replace, with keys as record IDs and values as new record data

        Returns:
            List[int]: The list of successfully replaced record IDs
        """
        return self._storage.batch_replace(data_dict)

    def from_pandas(self, df) -> 'ApexClient':
        """
        Imports data from a Pandas DataFrame.

        Parameters:
            df: pandas.DataFrame
                The input DataFrame

        Returns:
            ApexClient: self, for chaining
        """
        records = df.to_dict('records')
        self.store(records)
        return self

    def from_pyarrow(self, table) -> 'ApexClient':
        """
        Imports data from a PyArrow Table.

        Parameters:
            table: pyarrow.Table
                The input PyArrow Table

        Returns:
            ApexClient: self
        """
        records = table.to_pylist()
        self.store(records)
        return self

    def from_polars(self, df) -> 'ApexClient':
        """
        Imports data from a Polars DataFrame.

        Parameters:
            df: polars.DataFrame
                The input Polars DataFrame

        Returns:
            ApexClient: self
        """
        records = df.to_dicts()
        self.store(records)
        return self

    def optimize(self):
        """
        Optimizes the database performance.
        """
        self._storage.optimize()

    def count_rows(self, table_name: str = None):
        """
        Returns the number of rows in a specified table or the current table.

        Parameters:
            table_name: str
                The table name, or None to use the current table

        Returns:
            int: The number of rows in the table
        """
        return self._storage.count_rows(table_name)

    def flush_cache(self):
        """
        Flushes the storage cache.
        """
        self._storage.flush_cache()

    def close(self):
        """
        Closes the database connection and flushes any remaining cache.
        """
        if hasattr(self, '_storage'):
            self.flush_cache()
            self._storage.close()

    def __del__(self):
        """
        Destructor to ensure the database connection is closed.
        """
        self.close()

    def drop_column(self, column_name: str):
        """
        删除指定的列。

        Parameters:
            column_name: str
                要删除的列名
        """
        if column_name == '_id':
            raise ValueError("Cannot drop _id column")
        self._storage.drop_column(column_name)

    def add_column(self, column_name: str, column_type: str):
        """
        添加新列。

        Parameters:
            column_name: str
                新列的名称
            column_type: str
                新列的数据类型
        """
        self._storage.add_column(column_name, column_type)

    def rename_column(self, old_column_name: str, new_column_name: str):
        """
        重命名列。

        Parameters:
            old_column_name: str
                原列名
            new_column_name: str
                新列名
        """
        if old_column_name == '_id':
            raise ValueError("Cannot rename _id column")
        self._storage.rename_column(old_column_name, new_column_name)

    def get_column_dtype(self, column_name: str) -> str:
        """
        获取列的数据类型。

        Parameters:
            column_name: str
                列名

        Returns:
            str: 列的数据类型
        """
        return self._storage.get_column_dtype(column_name)
