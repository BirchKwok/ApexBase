from abc import ABC, abstractmethod
from typing import Dict, List, Optional


class BaseStorage(ABC):
    """Abstract base class for storage backends."""
    
    @abstractmethod
    def __init__(self, filepath=None, batch_size: int = 1000):
        pass
    
    @abstractmethod
    def use_table(self, table_name: str):
        pass
    
    @abstractmethod
    def create_table(self, table_name: str):
        pass
    
    @abstractmethod
    def drop_table(self, table_name: str):
        pass
    
    @abstractmethod
    def list_tables(self) -> List[str]:
        pass
    
    @abstractmethod
    def store(self, data: dict, table_name: str = None) -> int:
        pass
    
    @abstractmethod
    def batch_store(self, data_list: List[dict], table_name: str = None) -> List[int]:
        pass
    
    @abstractmethod
    def retrieve(self, id_: int) -> Optional[dict]:
        pass
    
    @abstractmethod
    def retrieve_many(self, ids: List[int]) -> List[dict]:
        pass
    
    @abstractmethod
    def delete(self, id_: int) -> bool:
        pass
    
    @abstractmethod
    def batch_delete(self, ids: List[int]) -> bool:
        pass
    
    @abstractmethod
    def replace(self, id_: int, data: dict) -> bool:
        pass
    
    @abstractmethod
    def batch_replace(self, data_dict: Dict[int, dict]) -> List[int]:
        pass
    
    @abstractmethod
    def query(self, query: str, table_name: str = None):
        pass
    
    @abstractmethod
    def close(self):
        pass


class BaseSchema(ABC):
    """Abstract base class for schema."""
    
    @abstractmethod
    def to_dict(self) -> dict:
        """转换为字典格式"""
        pass
    
    @abstractmethod
    def drop_column(self, column_name: str):
        """删除列
        
        Args:
            column_name: 列名
        """
        pass
    
    @abstractmethod
    def add_column(self, column_name: str, column_type: str):
        """添加列
        
        Args:
            column_name: 列名
            column_type: 列类型
        """
        pass

    @abstractmethod
    def rename_column(self, old_column_name: str, new_column_name: str):
        """重命名列
        
        Args:
            old_column_name: 旧列名
            new_column_name: 新列名
        """
        pass

    @abstractmethod
    def modify_column(self, column_name: str, column_type: str):
        """修改列类型
        
        Args:
            column_name: 列名
            column_type: 列类型
        """
        pass

    @abstractmethod
    def get_column_type(self, column_name: str) -> str:
        """获取列类型
        
        Args:
            column_name: 列名
        """
        pass

    @abstractmethod
    def has_column(self, column_name: str) -> bool:
        """检查列是否存在
        
        Args:
            column_name: 列名
        """
        pass

    @abstractmethod
    def get_columns(self) -> List[str]:
        """获取所有列名
        
        Returns:
            列名列表
        """
        pass

    @abstractmethod
    def update_from_data(self, data: dict):
        """从数据更新schema
        
        Args:
            data: 数据字典
        """
        pass
