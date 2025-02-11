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
        """Convert to dictionary format"""
        pass
    
    @abstractmethod
    def drop_column(self, column_name: str):
        """Drop a column
        
        Args:
            column_name: The name of the column to drop
        """
        pass
    
    @abstractmethod
    def add_column(self, column_name: str, column_type: str):
        """Add a column
        
        Args:
            column_name: The name of the column to add
            column_type: The type of the column to add
        """
        pass

    @abstractmethod
    def rename_column(self, old_column_name: str, new_column_name: str):
        """Rename a column
        
        Args:
            old_column_name: The old name of the column
            new_column_name: The new name of the column
        """
        pass

    @abstractmethod
    def modify_column(self, column_name: str, column_type: str):
        """Modify the type of a column
        
        Args:
            column_name: The name of the column to modify
            column_type: The type of the column to modify
        """
        pass

    @abstractmethod
    def get_column_type(self, column_name: str) -> str:
        """Get the type of a column
        
        Args:
            column_name: The name of the column
        """
        pass

    @abstractmethod
    def has_column(self, column_name: str) -> bool:
        """Check if a column exists
        
        Args:
            column_name: The name of the column
        """
        pass

    @abstractmethod
    def get_columns(self) -> List[str]:
        """Get all column names
        
        Returns:
            The list of column names
        """
        pass

    @abstractmethod
    def update_from_data(self, data: dict):
        """Update the schema from data
        
        Args:
            data: The data dictionary
        """
        pass
