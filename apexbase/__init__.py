from typing import List, Dict, Union, Optional
from pathlib import Path

from .storage import Storage
from .query import Query, ResultView


version = "0.0.1"


class ApexClient:
    def __init__(self, dirpath=None, batch_size: int = 1000, drop_if_exists: bool = False):
        """
        Initializes a new instance of the ApexClient class.

        Parameters:
            dirpath: str
                The directory path for storing data. If None, the current directory is used.
            batch_size: int
                The size of batch operations.
            drop_if_exists: bool
                If True, the database file will be deleted if it already exists.
        """
        if dirpath is None:
            dirpath = "."
        
        self.dirpath = Path(dirpath)
        self.dirpath.mkdir(parents=True, exist_ok=True)
        
        self.db_path = self.dirpath / "apexbase.db"
        
        if drop_if_exists and self.db_path.exists():
            self.db_path.unlink()
        
        self.storage = Storage(str(self.db_path), batch_size=batch_size)
        self.query_handler = Query(self.storage)
        self.current_table = "default"  # Default table name

    def use_table(self, table_name: str):
        """
        Switches the current table for operations.

        Parameters:
            table_name: str
                The name of the table to switch to.
        """
        self.current_table = table_name
        self.storage.use_table(table_name)

    def create_table(self, table_name: str):
        """
        Creates a new table.

        Parameters:
            table_name: str
                The name of the table to create.
        """
        self.storage.create_table(table_name)

    def drop_table(self, table_name: str):
        """
        Drops a table.

        Parameters:
            table_name: str
                The name of the table to drop.
        """
        self.storage.drop_table(table_name)
        # If the table being dropped is the current table, switch to the default table
        if self.current_table == table_name:
            self.current_table = "default"

    def list_tables(self) -> List[str]:
        """
        Lists all tables.

        Returns:
            List[str]: A list of table names
        """
        return self.storage.list_tables()

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
            return self.storage.store(data)
        elif isinstance(data, list):
            # Multiple records
            return self.storage.batch_store(data)
        else:
            raise ValueError("Data must be a dict or a list of dicts")

    def query(self, query_filter: str = None) -> ResultView:
        """
        Queries records using SQL syntax.

        Parameters:
            query_filter: str
                SQL filter conditions. For example:
                - age > 30
                - name LIKE 'John%'
                - age > 30 AND city = 'New York'
                - field IN (1, 2, 3)
                - ORDER BY, GROUP BY, HAVING are not supported

        Returns:
            ResultView: A view of query results, supporting deferred execution
        """
        return self.query_handler.query(query_filter)

    def search_text(self, text: str, fields: List[str] = None) -> ResultView:
        """
        Full-text search.

        Parameters:
            text: str
                The text to search
            fields: List[str]
                The fields to search, if None, all searchable fields are searched

        Returns:
            ResultView: A view of search results, supporting deferred execution
        """
        return self.query_handler.search_text(text, fields)

    def retrieve(self, id_: int) -> Optional[dict]:
        """
        Retrieves a single record.

        Parameters:
            id_: int
                The record ID

        Returns:
            Optional[dict]: The record data, or None if it doesn't exist
        """
        return self.query_handler.retrieve(id_)

    def retrieve_many(self, ids: List[int]) -> List[dict]:
        """
        Retrieves multiple records.

        Parameters:
            ids: List[int]
                The list of record IDs

        Returns:
            List[dict]: The list of record data
        """
        return self.query_handler.retrieve_many(ids)
    
    def retrieve_all(self) -> ResultView:
        return self.query_handler.retrieve_all()

    def list_fields(self):
        """
        List the fields in the cache.

        Returns:
            List[str]: List of fields.
        """
        return list(self.storage.list_fields().keys())

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
            return self.storage.delete(ids)
        elif isinstance(ids, list):
            return self.storage.batch_delete(ids)
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
        return self.storage.replace(id_, data)

    def batch_replace(self, data_dict: Dict[int, dict]) -> List[int]:
        """
        Replaces multiple records.

        Parameters:
            data_dict: Dict[int, dict]
                The dictionary of records to replace, with keys as record IDs and values as new record data

        Returns:
            List[int]: The list of successfully replaced record IDs
        """
        return self.storage.batch_replace(data_dict)

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

    def set_searchable(self, field_name: str, is_searchable: bool = True):
        """
        Sets whether a field is searchable.

        Parameters:
            field_name: str
                The field name
            is_searchable: bool
                Whether the field is searchable
        """
        self.storage.set_searchable(field_name, is_searchable)

    def rebuild_search_index(self):
        """
        Rebuilds the full-text search index.
        """
        self.storage.rebuild_fts_index()

    def optimize(self):
        """
        Optimizes the database performance.
        """
        self.storage.optimize()

    def set_auto_update_fts(self, enabled: bool):
        """
        Sets whether to automatically update the full-text search index.
        Defaults to False, to improve batch write performance.
        If auto-update is disabled, you need to manually call rebuild_fts_index to update the index.

        Parameters:
            enabled: bool
                Whether to enable auto-update
        """
        self.storage.set_auto_update_fts(enabled)

    def rebuild_fts_index(self):
        """
        Rebuilds the full-text search index for the current table.
        Call this method after batch writes to update the index.
        """
        self.storage.rebuild_fts_index()

    def count_rows(self, table_name: str = None):
        """
        Returns the number of rows in a specified table or the current table.

        Parameters:
            table_name: str
                The table name, or None to use the current table

        Returns:
            int: The number of rows in the table
        """
        return self.storage.count_rows(table_name)

    def close(self):
        """
        Close the database connection.
        """
        self.storage.close()

    def __del__(self):
        """
        Destructor to ensure the database connection is closed.
        """
        self.close()
