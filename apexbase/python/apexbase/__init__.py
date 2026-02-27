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

__version__ = "1.9.0"


class _InstanceRegistry:
    """Global instance registry with multi-client support
    
    This registry now supports multiple clients per database path using reference counting.
    The underlying storage is shared among all clients for the same database.
    """
    
    def __init__(self):
        # db_path -> {'storage': ApexStorage, 'clients': {client_id: weakref}, 'count': int}
        self._instances = {}
        self._lock = None
        self._client_id_counter = 0
        self._client_id_lock = None
    
    def _get_lock(self):
        if self._lock is None:
            import threading
            self._lock = threading.Lock()
        return self._lock
    
    def _get_client_id_lock(self):
        if self._client_id_lock is None:
            import threading
            self._client_id_lock = threading.Lock()
        return self._client_id_lock
    
    def _generate_client_id(self):
        """Generate a unique client ID"""
        with self._get_client_id_lock():
            self._client_id_counter += 1
            return self._client_id_counter
    
    def register(self, instance, db_path: str):
        """Register a new client for the given database path.
        
        If other active clients exist for this path, share the underlying storage.
        If all existing clients are closed, create new storage.
        """
        lock = self._get_lock()
        with lock:
            if db_path in self._instances:
                entry = self._instances[db_path]
                
                # Check if any existing clients are still active
                active_clients = []
                for cid, client_ref in list(entry['clients'].items()):
                    client = client_ref()
                    if client is not None and not getattr(client, '_is_closed', False):
                        active_clients.append(cid)
                
                if active_clients:
                    # Another active client exists - share the storage
                    client_id = self._generate_client_id()
                    entry['clients'][client_id] = weakref.ref(instance, 
                        lambda ref: self._cleanup_ref(db_path, client_id, ref))
                    entry['count'] += 1
                    # Attach shared storage to the new instance
                    instance._shared_storage = entry['storage']
                    instance._client_id = client_id
                    instance._is_shared_client = True
                else:
                    # All previous clients are closed - create new storage
                    client_id = self._generate_client_id()
                    self._instances[db_path] = {
                        'storage': None,  # Will be set by the client
                        'clients': {client_id: weakref.ref(instance)},
                        'count': 1
                    }
                    instance._client_id = client_id
                    instance._is_shared_client = False
            else:
                # First client for this database - create new storage
                client_id = self._generate_client_id()
                self._instances[db_path] = {
                    'storage': None,  # Will be set by the client
                    'clients': {client_id: weakref.ref(instance)},
                    'count': 1
                }
                instance._client_id = client_id
                instance._is_shared_client = False
    
    def set_storage(self, db_path: str, storage):
        """Set the storage instance for a database (called by client after creating storage)"""
        lock = self._get_lock()
        with lock:
            if db_path in self._instances:
                self._instances[db_path]['storage'] = storage
    
    def _cleanup_ref(self, db_path: str, client_id: int, ref):
        """Clean up when a client is garbage collected"""
        lock = self._get_lock()
        with lock:
            if db_path in self._instances:
                entry = self._instances[db_path]
                if client_id in entry['clients']:
                    del entry['clients'][client_id]
                    entry['count'] -= 1
                    
                    # If no more clients, clean up
                    if entry['count'] <= 0:
                        self._instances.pop(db_path, None)
    
    def unregister(self, db_path: str, client_id: int = None):
        """Unregister a client"""
        lock = self._get_lock()
        with lock:
            if db_path in self._instances:
                entry = self._instances[db_path]
                if client_id and client_id in entry['clients']:
                    del entry['clients'][client_id]
                    entry['count'] -= 1
                    
                    # If no more clients, the storage will be closed by the client's close() method
                    if entry['count'] <= 0:
                        self._instances.pop(db_path, None)
                elif not client_id:
                    # Remove all clients for this path
                    self._instances.pop(db_path, None)
    
    def get_storage(self, db_path: str):
        """Get the shared storage for a database path"""
        lock = self._get_lock()
        with lock:
            if db_path in self._instances:
                return self._instances[db_path]['storage']
            return None
    
    def close_all(self):
        """Close all instances (called at program exit)"""
        lock = self._get_lock()
        with lock:
            # First, mark all clients as closed
            for db_path, entry in list(self._instances.items()):
                for client_ref in list(entry['clients'].values()):
                    client = client_ref()
                    if client is not None:
                        try:
                            client._is_closed = True
                            client._storage = None
                        except Exception:
                            pass
            # Then close the storage
            for db_path, entry in list(self._instances.items()):
                storage = entry.get('storage')
                if storage is not None:
                    try:
                        storage.close()
                    except Exception:
                        pass
            self._instances.clear()


_registry = _InstanceRegistry()
atexit.register(_registry.close_all)


class ResultView:
    """Query result view - Arrow-first high-performance implementation"""
    
    def __init__(self, arrow_table=None, data=None, lazy_pydict=None):
        """
        Initialize ResultView (Arrow-first mode)
        
        Args:
            arrow_table: PyArrow Table (primary data source, fastest)
            data: List[dict] data (optional, for fallback)
            lazy_pydict: dict of column_name -> list (deferred Arrow creation)
        """
        self._arrow_table = arrow_table
        self._data = data  # Lazy loading, convert from Arrow
        self._lazy_pydict = lazy_pydict  # Deferred: convert to Arrow on demand
        if arrow_table is not None:
            self._num_rows = arrow_table.num_rows
        elif lazy_pydict is not None:
            first_col = next(iter(lazy_pydict.values()), [])
            self._num_rows = len(first_col)
        elif data is not None:
            self._num_rows = len(data)
        else:
            self._num_rows = 0
    
    @classmethod
    def from_arrow_bytes(cls, arrow_bytes: bytes) -> 'ResultView':
        raise RuntimeError("Arrow IPC bytes path has been removed. Use Arrow FFI results only.")
    
    @classmethod
    def from_dicts(cls, data: List[dict]) -> 'ResultView':
        raise RuntimeError("Non-Arrow query path has been removed. Use Arrow FFI results only.")
    
    def _ensure_arrow(self):
        """Materialize Arrow table from lazy_pydict if needed"""
        if self._arrow_table is None and self._lazy_pydict is not None:
            self._arrow_table = pa.Table.from_pydict(self._lazy_pydict)
            self._lazy_pydict = None
    
    def _ensure_data(self):
        """Ensure _data is available (lazy load from Arrow conversion, optionally hide _id)"""
        if self._data is None:
            # Try lazy pydict first (avoids Arrow round-trip for to_dict)
            if self._lazy_pydict is not None:
                show_id = bool(getattr(self, "_show_internal_id", False))
                d = self._lazy_pydict
                keys = [k for k in d if show_id or k != '_id']
                n = len(next(iter(d.values()), []))
                self._data = [{k: d[k][i] for k in keys} for i in range(n)]
                return self._data
            self._ensure_arrow()
            if self._arrow_table is not None:
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

    def tolist(self) -> List[dict]:
        """Convert results to a list of dictionaries.

        Alias for :meth:`to_dict`. Returns one dictionary per row, excluding
        the internal ``_id`` field.

        Returns:
            List[dict]: List of records as dictionaries.
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
        
        self._ensure_arrow()
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
        
        self._ensure_arrow()
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
        
        self._ensure_arrow()
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
        self._ensure_arrow()
        if self._arrow_table is not None:
            return (self._arrow_table.num_rows, self._arrow_table.num_columns)
        # When arrow_table is None (empty result), return (0, 0)
        return (0, 0)
    
    @property
    def columns(self):
        if self._lazy_pydict is not None:
            return [c for c in self._lazy_pydict if c != '_id']
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
        self._ensure_arrow()
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
        if self._lazy_pydict is not None:
            first_col = next((k for k in self._lazy_pydict if k != '_id'), None)
            if first_col and self._lazy_pydict[first_col]:
                return self._lazy_pydict[first_col][0]
            return None
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

