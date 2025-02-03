

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
