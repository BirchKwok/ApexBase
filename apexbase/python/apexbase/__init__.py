"""
ApexBase - High-performance embedded database based on Rust core

Uses custom single-file storage format (.apex) to provide efficient data storage and query functionality.
"""


import weakref
import atexit
from typing import List, Optional, Literal

import numpy as np

# Import Rust core
from apexbase._core import ApexStorage, __version__

# FTS is now directly implemented in Rust layer, no need for Python nanofts package
# But keep compatibility flag
FTS_AVAILABLE = True  # Always available since integrated into Rust core

# Optional data framework support
import pyarrow as pa
import pandas as pd
ARROW_AVAILABLE = True

import polars as pl
POLARS_AVAILABLE = True

__version__ = "0.4.0"


class _InstanceRegistry:
    """Global instance registry"""
    
    def __init__(self):
        self._instances = {}
        self._lock = None
    
    def _get_lock(self):
        if self._lock is None:
            import threading
            self._lock = threading.Lock()
        return self._lock
    
    def register(self, instance, db_path: str):
        lock = self._get_lock()
        with lock:
            if db_path in self._instances:
                old_ref = self._instances[db_path]
                old_instance = old_ref() if old_ref else None
                if old_instance is not None:
                    try:
                        old_instance._force_close()
                    except Exception:
                        pass
            
            self._instances[db_path] = weakref.ref(instance, 
                                                   lambda ref: self._cleanup_ref(db_path, ref))
    
    def _cleanup_ref(self, db_path: str, ref):
        lock = self._get_lock()
        with lock:
            if self._instances.get(db_path) == ref:
                del self._instances[db_path]
    
    def unregister(self, db_path: str):
        lock = self._get_lock()
        with lock:
            self._instances.pop(db_path, None)
    
    def close_all(self):
        lock = self._get_lock()
        with lock:
            for ref in list(self._instances.values()):
                instance = ref() if ref else None
                if instance is not None:
                    try:
                        instance._force_close()
                    except Exception:
                        pass
            self._instances.clear()


_registry = _InstanceRegistry()
atexit.register(_registry.close_all)


class ResultView:
    """Query result view - Arrow-first high-performance implementation"""
    
    def __init__(self, arrow_table=None, data=None):
        """
        Initialize ResultView (Arrow-first mode)
        
        Args:
            arrow_table: PyArrow Table (primary data source, fastest)
            data: List[dict] data (optional, for fallback)
        """
        self._arrow_table = arrow_table
        self._data = data  # Lazy loading, convert from Arrow
        self._num_rows = arrow_table.num_rows if arrow_table is not None else (len(data) if data else 0)
    
    @classmethod
    def from_arrow_bytes(cls, arrow_bytes: bytes) -> 'ResultView':
        raise RuntimeError("Arrow IPC bytes path has been removed. Use Arrow FFI results only.")
    
    @classmethod
    def from_dicts(cls, data: List[dict]) -> 'ResultView':
        raise RuntimeError("Non-Arrow query path has been removed. Use Arrow FFI results only.")
    
    def _ensure_data(self):
        """Ensure _data is available (lazy load from Arrow conversion, optionally hide _id)"""
        if self._data is None and self._arrow_table is not None:
            show_id = bool(getattr(self, "_show_internal_id", False))
            if show_id:
                self._data = [dict(row) for row in self._arrow_table.to_pylist()]
            else:
                self._data = [{k: v for k, v in row.items() if k != '_id'} 
                              for row in self._arrow_table.to_pylist()]
        return self._data if self._data is not None else []
    
    def to_dict(self) -> List[dict]:
        """Convert results to a list of dictionaries.
        
        Returns:
            List[dict]: List of records as dictionaries, excluding the internal '_id' field.
        """
        return self._ensure_data()
    
    def to_pandas(self, zero_copy: bool = True):
        """Convert results to a pandas DataFrame.
        
        Args:
            zero_copy: If True, use ArrowDtype for zero-copy conversion (pandas 2.0+).
                If False, use traditional conversion copying data to NumPy.
                Defaults to True.
        
        Returns:
            pandas.DataFrame: DataFrame containing the query results.
        
        Raises:
            ImportError: If pandas is not available.
        
        Note:
            In zero-copy mode, DataFrame columns use Arrow native types (like string[pyarrow]).
            This performs better in most scenarios, but some NumPy operations may need
            type conversion first.
        """
        if not ARROW_AVAILABLE:
            raise ImportError("pandas not available. Install with: pip install pandas")
        
        if self._arrow_table is not None:
            show_id = bool(getattr(self, "_show_internal_id", False))
            if zero_copy:
                # Zero-copy mode: use ArrowDtype (pandas 2.0+)
                try:
                    df = self._arrow_table.to_pandas(types_mapper=pd.ArrowDtype)
                except (TypeError, AttributeError):
                    # Fallback: pandas < 2.0 doesn't support ArrowDtype
                    df = self._arrow_table.to_pandas()
            else:
                # Traditional mode: copy data to NumPy types
                df = self._arrow_table.to_pandas()

            if not show_id and '_id' in df.columns:
                df.set_index('_id', inplace=True)
                df.index.name = None
            return df
        
        # Fallback
        df = pd.DataFrame(self._ensure_data())
        if '_id' in df.columns:
            df.set_index('_id', inplace=True)
            df.index.name = None
        return df
    
    def to_polars(self):
        """Convert results to a polars DataFrame.
        
        Returns:
            polars.DataFrame: DataFrame containing the query results.
            
        Raises:
            ImportError: If polars is not available.
        """
        if not POLARS_AVAILABLE:
            raise ImportError("polars not available. Install with: pip install polars")
        
        if self._arrow_table is not None:
            df = pl.from_arrow(self._arrow_table)
            show_id = bool(getattr(self, "_show_internal_id", False))
            if not show_id and '_id' in df.columns:
                df = df.drop('_id')
            return df
        return pl.DataFrame(self._ensure_data())
    
    def to_arrow(self):
        """Convert results to a PyArrow Table.
        
        Returns:
            pyarrow.Table: Arrow Table containing the query results.
            
        Raises:
            ImportError: If pyarrow is not available.
        """
        if not ARROW_AVAILABLE:
            raise ImportError("pyarrow not available. Install with: pip install pyarrow")
        
        if self._arrow_table is not None:
            show_id = bool(getattr(self, "_show_internal_id", False))
            if not show_id:
                # Remove _id column
                if '_id' in self._arrow_table.column_names:
                    return self._arrow_table.drop(['_id'])
            return self._arrow_table
        return pa.Table.from_pylist(self._ensure_data())
    
    @property
    def shape(self):
        if self._arrow_table is not None:
            return (self._arrow_table.num_rows, self._arrow_table.num_columns)
        # When arrow_table is None (empty result), return (0, 0)
        return (0, 0)
    
    @property
    def columns(self):
        if self._arrow_table is not None:
            cols = self._arrow_table.column_names
            show_id = bool(getattr(self, "_show_internal_id", False))
            if show_id:
                return list(cols)
            return [c for c in cols if c != '_id']
        data = self._ensure_data()
        if not data:
            return []
        cols = list(data[0].keys())
        if '_id' in cols:
            cols.remove('_id')
        return cols
    
    @property
    def ids(self):
        """[Deprecated] Please use get_ids() method"""
        return self.get_ids(return_list=True)
    
    def get_ids(self, return_list: bool = False):
        """Get the internal IDs of the result records.
        
        Args:
            return_list: If True, return as Python list.
                If False, return as numpy.ndarray (default, zero-copy, fastest).
                Defaults to False.
        
        Returns:
            numpy.ndarray or list: Array of record IDs.
        """
        if self._arrow_table is not None and '_id' in self._arrow_table.column_names:
            # Zero-copy path: directly convert from Arrow to numpy, bypassing Python objects
            id_array = self._arrow_table.column('_id').to_numpy()
            if return_list:
                return id_array.tolist()
            return id_array
        else:
            # Fallback: generate sequential IDs
            ids = np.arange(self._num_rows, dtype=np.uint64)
            if return_list:
                return ids.tolist()
            return ids

    def scalar(self):
        """Get single scalar value (for aggregate queries like COUNT(*))"""
        if self._arrow_table is not None and self._arrow_table.num_rows > 0:
            # Skip _id if present
            col_names = self._arrow_table.column_names
            col_idx = 0
            if col_names and col_names[0] == '_id' and len(col_names) > 1:
                col_idx = 1
            return self._arrow_table.column(col_idx)[0].as_py()

        data = self._ensure_data()
        if data:
            first_row = data[0]
            if first_row:
                first_key = next(iter(first_row.keys()))
                return first_row.get(first_key)
        return None

    def first(self) -> Optional[dict]:
        """Get first row as dictionary (hide _id)"""
        data = self._ensure_data()
        if data:
            return data[0]
        return None
    
    def __len__(self):
        return self._num_rows
    
    def __iter__(self):
        return iter(self._ensure_data())
    
    def __getitem__(self, idx):
        return self._ensure_data()[idx]
    
    def __repr__(self):
        return f"ResultView(rows={self._num_rows})"


def _empty_result_view() -> ResultView:
    # Create empty ResultView with no columns
    # Use a special marker to indicate truly empty result
    rv = ResultView(arrow_table=None, data=[])
    return rv


# Durability level type
DurabilityLevel = Literal['fast', 'safe', 'max']

# Import ApexClient from client module
from .client import ApexClient

# Exports
__all__ = ['ApexClient', 'ApexStorage', 'ResultView', 'DurabilityLevel', '__version__', 'FTS_AVAILABLE', 'ARROW_AVAILABLE', 'POLARS_AVAILABLE']

