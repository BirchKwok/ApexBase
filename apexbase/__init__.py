from typing import List, Dict, Union, Optional
import os
from pathlib import Path

from .storage import Storage
from .query import Query, ResultView


class ApexClient:
    def __init__(self, dirpath=None, cache_size: int = 1000, batch_size: int = 1000, drop_if_exists: bool = False):
        """
        创建 ApexClient 实例。

        Parameters:
            dirpath: str
                数据存储目录的路径。如果为None，则使用当前目录。
            cache_size: int
                查询结果缓存的最大数量。
            batch_size: int
                批量操作的大小。
            drop_if_exists: bool
                如果为True，则在数据库文件已存在时删除它。
        """
        if dirpath is None:
            dirpath = "."
        
        # 确保目录存在
        self.dirpath = Path(dirpath)
        self.dirpath.mkdir(parents=True, exist_ok=True)
        
        # 构建数据库文件路径
        self.db_path = self.dirpath / "apexbase.db"
        
        # 如果设置了drop_if_exists且文件存在，则删除文件
        if drop_if_exists and self.db_path.exists():
            self.db_path.unlink()
        
        self.storage = Storage(str(self.db_path), cache_size=cache_size, batch_size=batch_size)
        self.query_handler = Query(self.storage)
        self.current_table = "default"  # 默认表名

    def use_table(self, table_name: str):
        """
        切换当前操作的表。

        Parameters:
            table_name: str
                要切换到的表名
        """
        self.current_table = table_name
        self.storage.use_table(table_name)

    def create_table(self, table_name: str):
        """
        创建新表。

        Parameters:
            table_name: str
                要创建的表名
        """
        self.storage.create_table(table_name)

    def drop_table(self, table_name: str):
        """
        删除表。

        Parameters:
            table_name: str
                要删除的表名
        """
        self.storage.drop_table(table_name)
        # 如果删除的是当前表，切换到默认表
        if self.current_table == table_name:
            self.current_table = "default"

    def list_tables(self) -> List[str]:
        """
        列出所有表。

        Returns:
            List[str]: 表名列表
        """
        return self.storage.list_tables()

    def store(self, data: Union[dict, List[dict]]) -> Union[int, List[int]]:
        """
        存储一条或多条记录。

        Parameters:
            data: Union[dict, List[dict]]
                要存储的记录或记录列表

        Returns:
            Union[int, List[int]]: 记录ID或ID列表
        """
        if isinstance(data, dict):
            # 单条记录
            return self.storage.store(data)
        elif isinstance(data, list):
            # 多条记录
            return self.storage.batch_store(data)
        else:
            raise ValueError("Data must be a dict or a list of dicts")

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
            ResultView: 查询结果视图，支持延迟执行
        """
        return self.query_handler.query(query_filter)

    def search_text(self, text: str, fields: List[str] = None) -> ResultView:
        """
        全文搜索。

        Parameters:
            text: str
                搜索文本
            fields: List[str]
                要搜索的字段列表，如果为None则搜索所有可搜索字段

        Returns:
            ResultView: 搜索结果视图，支持延迟执行
        """
        return self.query_handler.search_text(text, fields)

    def retrieve(self, id_: int) -> Optional[dict]:
        """
        检索单条记录。

        Parameters:
            id_: int
                记录ID

        Returns:
            Optional[dict]: 记录数据，如果不存在则返回None
        """
        return self.query_handler.retrieve(id_)

    def retrieve_many(self, ids: List[int]) -> List[dict]:
        """
        批量检索记录。

        Parameters:
            ids: List[int]
                记录ID列表

        Returns:
            List[dict]: 记录数据列表
        """
        return self.query_handler.retrieve_many(ids)

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

    def set_auto_update_fts(self, enabled: bool):
        """
        设置是否自动更新全文搜索索引。
        默认为False，以提高批量写入性能。
        如果禁用自动更新，需要手动调用rebuild_fts_index来更新索引。

        Parameters:
            enabled: bool
                是否启用自动更新
        """
        self.storage.set_auto_update_fts(enabled)

    def rebuild_fts_index(self):
        """
        重建当前表的全文搜索索引。
        在批量写入后调用此方法来更新索引。
        """
        self.storage.rebuild_fts_index()

    def count_rows(self, table_name: str = None):
        """
        返回当前表或指定表的行数。
        """
        if table_name is None:
            table_name = self.current_table
        return self.storage.count_rows(table_name)
