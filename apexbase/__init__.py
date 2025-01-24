from typing import List, Dict, Union

from .storage import Storage
from .query import Query


class ApexClient:
    def __init__(self, filepath=None, cache_size: int = 1000, batch_size: int = 1000):
        """
        Create a ApexClient instance.

        Parameters:
            filepath: str
                The storage path for the cache file.
            cache_size: int
                Maximum number of query results to cache.
            batch_size: int
                Size of batches for bulk operations.
        """
        self.storage = Storage(filepath, cache_size=cache_size, batch_size=batch_size)
        self.query_handler = Query(self.storage)
        self.filepath = filepath

    def store(self, data: Union[dict, List[dict]]) -> Union[int, List[int]]:
        """
        存储一条或多条记录。

        Parameters:
            data: Union[dict, List[dict]]
                要存储的记录。可以是单个字典或字典列表。

        Returns:
            Union[int, List[int]]: 
                如果输入是单个字典，返回记录ID；
                如果输入是字典列表，返回记录ID列表。
        """
        if isinstance(data, dict):
            return self.storage.store(data)
        elif isinstance(data, list):
            if not data:
                return []
            if not all(isinstance(d, dict) for d in data):
                raise ValueError("All elements must be dictionaries when storing multiple records.")
            return self.storage.batch_store(data)
        else:
            raise ValueError("Data must be either a dictionary or a list of dictionaries.")

    def query(self, query_filter: str, return_ids_only: bool = True):
        """
        Query the fields cache.

        Parameters:
            query_filter: str
                The query filter.
            return_ids_only: bool
                If True, only return external IDs.

        Returns:
            List[dict]: Records. If not return_ids_only, returns records.
            List[int]: External IDs. If return_ids_only, returns external IDs.
        """
        return self.query_handler.query(query_filter, return_ids_only)

    def retrieve(self, id_):
        """
        Retrieve a record from the cache.

        Returns:
            dict: The record.
        """
        return self.query_handler.retrieve(id_)

    def concat(self, other: 'ApexClient') -> 'ApexClient':
        """
        Concatenate two caches.

        Parameters:
            other: ApexClient
                Another cache to concatenate.

        Returns:
            ApexClient: The concatenated cache.
        """
        if not isinstance(other, ApexClient):
            raise ValueError("The other cache must be an instance of ApexClient.")
        records = list(other.storage.retrieve_all())
        if records:
            self.store(records)
        return self

    def list_fields(self):
        """
        List the fields in the cache.

        Returns:
            List[str]: List of fields.
        """
        return list(self.storage.list_fields().keys())

    def retrieve_many(self, ids: List[int]):
        """
        Retrieve multiple records from the cache.

        Parameters:
            ids: List[int]
                List of external IDs.

        Returns:
            List[dict]: List of records.
        """
        return self.query_handler.retrieve_many(ids)

    def delete(self, id_: int) -> bool:
        """
        删除一条记录。

        Parameters:
            id_: int
                要删除的记录ID

        Returns:
            bool: 删除是否成功
        """
        return self.storage.delete(id_)

    def batch_delete(self, ids: List[int]) -> List[int]:
        """
        批量删除记录。

        Parameters:
            ids: List[int]
                要删除的记录ID列表

        Returns:
            List[int]: 成功删除的记录ID列表
        """
        return self.storage.batch_delete(ids)

    def replace(self, id_: int, data: dict) -> bool:
        """
        替换一条记录。

        Parameters:
            id_: int
                要替换的记录ID
            data: dict
                新的记录数据

        Returns:
            bool: 替换是否成功
        """
        return self.storage.replace(id_, data)

    def batch_replace(self, data_dict: Dict[int, dict]) -> List[int]:
        """
        批量替换记录。

        Parameters:
            data_dict: Dict[int, dict]
                要替换的记录字典，key为记录ID，value为新的记录数据

        Returns:
            List[int]: 成功替换的记录ID列表
        """
        return self.storage.batch_replace(data_dict)

    def from_pandas(self, df) -> 'ApexClient':
        """
        从Pandas DataFrame导入数据。

        Parameters:
            df: pandas.DataFrame
                输入的DataFrame

        Returns:
            ApexClient: self，用于链式调用
        """
        records = df.to_dict('records')
        self.store(records)
        return self

    def from_pyarrow(self, table) -> 'ApexClient':
        """
        从PyArrow Table导入数据。

        Parameters:
            table: pyarrow.Table
                输入的PyArrow Table

        Returns:
            ApexClient: self，用于链式调用
        """
        records = table.to_pylist()
        self.store(records)
        return self

    def from_polars(self, df) -> 'ApexClient':
        """
        从Polars DataFrame导入数据。

        Parameters:
            df: polars.DataFrame
                输入的Polars DataFrame

        Returns:
            ApexClient: self，用于链式调用
        """
        records = df.to_dicts()
        self.store(records)
        return self

    def search_text(self, query: str, fields: List[str] = None) -> List[int]:
        """
        全文搜索。

        Parameters:
            query: str
                搜索查询
            fields: List[str]
                要搜索的字段列表，如果为None则搜索所有可搜索字段

        Returns:
            List[int]: 匹配记录的ID列表
        """
        return self.storage.search_text(query, fields)

    def set_searchable(self, field_name: str, is_searchable: bool = True):
        """
        设置字段是否可搜索。

        Parameters:
            field_name: str
                字段名称
            is_searchable: bool
                是否可搜索
        """
        self.storage.set_searchable(field_name, is_searchable)

    def rebuild_search_index(self):
        """
        重建全文搜索索引。
        """
        self.storage.rebuild_fts_index()

    def optimize(self):
        """
        优化数据库性能。
        """
        self.storage.optimize()