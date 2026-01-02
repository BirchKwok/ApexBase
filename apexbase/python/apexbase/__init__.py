"""
ApexBase - High-performance embedded database based on Rust core

Uses custom single-file storage format (.apex) to provide efficient data storage and query functionality.
"""

import shutil
import weakref
import atexit
from typing import List, Dict, Union, Optional, Literal
from pathlib import Path
import numpy as np

# Import Rust core
from apexbase._core import ApexStorage as RustStorage, __version__ as _core_version

# FTS is now directly implemented in Rust layer, no need for Python nanofts package
# But keep compatibility flag
FTS_AVAILABLE = True  # Always available since integrated into Rust core

# Optional data framework support
try:
    import pyarrow as pa
    import pandas as pd
    ARROW_AVAILABLE = True
except ImportError:
    ARROW_AVAILABLE = False
    pa = None
    pd = None

try:
    import polars as pl
    POLARS_AVAILABLE = True
except ImportError:
    POLARS_AVAILABLE = False
    pl = None

__version__ = "0.2.0"


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


class SqlResultArrow:
    """SQL execution result - Arrow-backed high-performance implementation"""
    
    def __init__(self, batch):
        """Initialize from Arrow RecordBatch"""
        self._batch = batch
        # Hide _id column
        self.columns = [c for c in batch.schema.names if c != '_id']
        self.rows_affected = batch.num_rows
        self._rows = None  # Lazy loading
    
    @property
    def rows(self):
        """Lazy load rows (convert only when needed, hide _id)"""
        if self._rows is None:
            self._rows = [
                [v for k, v in row.items() if k != '_id']
                for row in self._batch.to_pylist()
            ]
        return self._rows
    
    def __len__(self) -> int:
        return self._batch.num_rows
    
    def __iter__(self):
        for row in self._batch.to_pylist():
            yield {k: v for k, v in row.items() if k != '_id'}
    
    def to_dicts(self) -> list:
        """Convert to dictionary list (hide _id)"""
        return [{k: v for k, v in row.items() if k != '_id'} for row in self._batch.to_pylist()]
    
    def to_pandas(self):
        """Convert directly from Arrow to pandas (zero-copy, hide _id)"""
        df = self._batch.to_pandas()
        if '_id' in df.columns:
            df = df.drop(columns=['_id'])
        return df
    
    def to_polars(self):
        """Convert directly from Arrow to polars (hide _id)"""
        try:
            import polars as pl
            df = pl.from_arrow(self._batch)
            if '_id' in df.columns:
                df = df.drop('_id')
            return df
        except ImportError:
            raise ImportError("polars is required for to_polars()")
    
    def get_ids(self, return_list: bool = False):
        """
        Get ID list of results (high-performance zero-copy implementation)
        
        Parameters:
            return_list: bool, default False
                If True, return Python list
                If False, return numpy.ndarray (default, zero-copy, fastest)
        
        Returns:
            numpy.ndarray or list: ID array
        """
        if '_id' in self._batch.schema.names:
            # Zero-copy path: directly convert from Arrow to numpy, bypassing Python objects
            id_array = self._batch.column('_id').to_numpy()
            if return_list:
                return id_array.tolist()
            return id_array
        else:
            # Fallback: generate sequential IDs
            ids = np.arange(self._batch.num_rows, dtype=np.uint64)
            if return_list:
                return ids.tolist()
            return ids
    
    def scalar(self):
        if self._batch.num_rows > 0 and self._batch.num_columns > 0:
            col = self._batch.column(0)
            # Skip _id column
            if self._batch.schema.names[0] == '_id' and self._batch.num_columns > 1:
                col = self._batch.column(1)
            return col[0].as_py()
        return None
    
    def first(self) -> Optional[dict]:
        if self._batch.num_rows > 0:
            row = self._batch.to_pylist()[0]
            return {k: v for k, v in row.items() if k != '_id'}
        return None
    
    def __repr__(self) -> str:
        return f"SqlResultArrow(columns={self.columns}, rows={self._batch.num_rows})"


class SqlResult:
    """SQL execution result - SQL:2023 standard compatible"""
    
    def __init__(self, columns: list, rows: list, rows_affected: int = 0, arrow_batch=None):
        """
        Initialize SQL result
        
        Parameters:
            columns: List of column names
            rows: List of data rows, each row is a list of values
            rows_affected: Number of affected rows
            arrow_batch: Optional Arrow RecordBatch (for fast path with large result sets)
        """
        # Hide _id column
        self._all_columns = columns
        self.columns = [c for c in columns if c != '_id']
        self._id_col_idx = columns.index('_id') if '_id' in columns else None
        self.rows = rows
        self.rows_affected = rows_affected
        self._arrow_batch = arrow_batch  # Fast path: Arrow data
    
    def __len__(self) -> int:
        """Return number of result rows"""
        if self._arrow_batch is not None:
            return self._arrow_batch.num_rows
        return len(self.rows)
    
    def __iter__(self):
        """Iterate result rows, each row returned as dictionary (hide _id)"""
        if self._arrow_batch is not None:
            # Arrow fast path
            for row in self._arrow_batch.to_pylist():
                yield {k: v for k, v in row.items() if k != '_id'}
        else:
            for row in self.rows:
                yield {k: v for k, v in zip(self._all_columns, row) if k != '_id'}
    
    def to_dicts(self) -> list:
        """Convert to dictionary list (hide _id)"""
        if self._arrow_batch is not None:
            return [{k: v for k, v in row.items() if k != '_id'} for row in self._arrow_batch.to_pylist()]
        return [{k: v for k, v in zip(self._all_columns, row) if k != '_id'} for row in self.rows]
    
    def to_pandas(self):
        """Convert to pandas DataFrame (Arrow zero-copy fast path, hide _id)"""
        try:
            import pandas as pd
            if self._arrow_batch is not None:
                # Zero-copy fast path: use ArrowDtype (pandas 2.0+)
                try:
                    df = self._arrow_batch.to_pandas(types_mapper=pd.ArrowDtype)
                except (TypeError, AttributeError):
                    # Fallback: pandas < 2.0
                    df = self._arrow_batch.to_pandas()
            else:
                df = pd.DataFrame(self.rows, columns=self._all_columns)
            # Hide _id
            if '_id' in df.columns:
                df = df.drop(columns=['_id'])
            return df
        except ImportError:
            raise ImportError("pandas is required for to_pandas()")
    
    def to_polars(self):
        """Convert to polars DataFrame (hide _id)"""
        try:
            import polars as pl
            if self._arrow_batch is not None:
                # Fast path: directly convert from Arrow
                df = pl.from_arrow(self._arrow_batch)
            else:
                df = pl.DataFrame(dict(zip(self._all_columns, zip(*self.rows)))) if self.rows else pl.DataFrame()
            # Hide _id
            if '_id' in df.columns:
                df = df.drop('_id')
            return df
        except ImportError:
            raise ImportError("polars is required for to_polars()")
    
    def get_ids(self, return_list: bool = False):
        """
        Get ID list of results (high-performance zero-copy implementation)
        
        Parameters:
            return_list: bool, default False
                If True, return Python list
                If False, return numpy.ndarray (default, zero-copy, fastest)
        
        Returns:
            numpy.ndarray or list: ID array
        """
        if self._arrow_batch is not None and '_id' in self._arrow_batch.schema.names:
            # Zero-copy path: directly convert from Arrow to numpy, bypassing Python objects
            id_array = self._arrow_batch.column('_id').to_numpy()
            if return_list:
                return id_array.tolist()
            return id_array
        elif self._id_col_idx is not None:
            # Extract from rows (slower path)
            ids = np.array([row[self._id_col_idx] for row in self.rows], dtype=np.uint64)
            if return_list:
                return ids.tolist()
            return ids
        else:
            # Fallback: generate sequential IDs
            ids = np.arange(len(self), dtype=np.uint64)
            if return_list:
                return ids.tolist()
            return ids
    
    def scalar(self):
        """Get single scalar value (for aggregate queries like COUNT(*))"""
        if self._arrow_batch is not None and self._arrow_batch.num_rows > 0:
            # Skip _id column
            col_idx = 0
            if self._arrow_batch.schema.names[0] == '_id' and self._arrow_batch.num_columns > 1:
                col_idx = 1
            return self._arrow_batch.column(col_idx)[0].as_py()
        if self.rows and self.rows[0]:
            # Skip _id column
            idx = 0
            if self._id_col_idx == 0 and len(self.rows[0]) > 1:
                idx = 1
            return self.rows[0][idx]
        return None
    
    def first(self) -> Optional[dict]:
        """Get first row as dictionary (hide _id)"""
        if self._arrow_batch is not None and self._arrow_batch.num_rows > 0:
            return {col: self._arrow_batch.column(i)[0].as_py() 
                    for i, col in enumerate(self._arrow_batch.schema.names) if col != '_id'}
        if self.rows:
            return {k: v for k, v in zip(self._all_columns, self.rows[0]) if k != '_id'}
        return None
    
    def __repr__(self) -> str:
        row_count = len(self)
        return f"SqlResult(columns={self.columns}, rows={row_count}, rows_affected={self.rows_affected})"


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
        """Create from Arrow IPC bytes (fastest path)"""
        if not arrow_bytes or not ARROW_AVAILABLE:
            return cls(data=[])
        reader = pa.ipc.open_stream(arrow_bytes)
        table = reader.read_all()
        return cls(arrow_table=table)
    
    @classmethod
    def from_dicts(cls, data: List[dict]) -> 'ResultView':
        """Create from dictionary list (fallback path)"""
        return cls(data=data)
    
    def _ensure_data(self):
        """Ensure _data is available (lazy load from Arrow conversion, hide _id)"""
        if self._data is None and self._arrow_table is not None:
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
            
            if '_id' in df.columns:
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
            if '_id' in df.columns:
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
            # Remove _id column
            if '_id' in self._arrow_table.column_names:
                return self._arrow_table.drop(['_id'])
            return self._arrow_table
        return pa.Table.from_pylist(self._ensure_data())
    
    @property
    def shape(self):
        if self._arrow_table is not None:
            return (self._arrow_table.num_rows, self._arrow_table.num_columns)
        data = self._ensure_data()
        if not data:
            return (0, 0)
        return (len(data), len(data[0]) if data else 0)
    
    @property
    def columns(self):
        if self._arrow_table is not None:
            cols = self._arrow_table.column_names
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
            # 零拷贝路径：直接从 Arrow 转 numpy，不经过 Python 对象
            id_array = self._arrow_table.column('_id').to_numpy()
            if return_list:
                return id_array.tolist()
            return id_array
        else:
            # 回退：生成序列 ID
            ids = np.arange(self._num_rows, dtype=np.uint64)
            if return_list:
                return ids.tolist()
            return ids
    
    def __len__(self):
        return self._num_rows
    
    def __iter__(self):
        return iter(self._ensure_data())
    
    def __getitem__(self, idx):
        return self._ensure_data()[idx]
    
    def __repr__(self):
        return f"ResultView(rows={self._num_rows})"


# Durability level type
DurabilityLevel = Literal['fast', 'safe', 'max']


class ApexClient:
    """
    ApexBase client - High-performance embedded database based on Rust core
    
    Features:
    - Custom single-file storage format (.apex)
    - High-performance batch writes
    - Full-text search support (NanoFTS)
    - Integration with Pandas, Polars, PyArrow
    - Configurable durability levels
    """
    
    def __init__(
        self, 
        dirpath=None, 
        batch_size: int = 1000, 
        drop_if_exists: bool = False,
        enable_cache: bool = True,
        cache_size: int = 10000,
        prefer_arrow_format: bool = True,
        durability: DurabilityLevel = 'fast',
        _auto_manage: bool = True
    ):
        """
        Initialize ApexClient
        
        Parameters:
            dirpath: str
                Data storage directory path, if None, use current directory
            batch_size: int
                Size of batch operations
            drop_if_exists: bool
                If True, delete existing database files
            enable_cache: bool
                Whether to enable write cache
            cache_size: int
                Cache size
            prefer_arrow_format: bool
                Whether to prefer Arrow format
            durability: Literal['fast', 'safe', 'max']
                Durability level:
                - 'fast': Highest performance, data first written to memory buffer, persisted when flush()
                          Suitable for batch import, reconstructible data, extremely performance-sensitive scenarios
                - 'safe': Balanced mode, ensures data fully written to disk on each flush() (fsync)
                          Suitable for most production environments
                - 'max': Strongest ACID guarantee, immediate fsync on each write
                         Suitable for financial, orders, and other critical data scenarios
        
        Note:
            FTS (full-text search) functionality needs to be initialized separately through init_fts() method after connection.
            This allows more flexible configuration of FTS settings for each table.
            
            ApexClient supports context manager, recommended to use with statement for automatic resource management:
            
            >>> # Basic usage
            >>> with ApexClient("./my_db") as client:
            ...     client.store({"name": "Alice", "age": 25})
            ...     # Auto commit and close connection
            ... 
            >>> # Chain calls
            >>> with ApexClient("./my_db").init_fts(index_fields=['name']) as client:
            ...     client.store({"name": "Bob"})
            ...     # Auto close FTS index and database connection
        """
        if dirpath is None:
            dirpath = "."
        
        self._dirpath = Path(dirpath)
        self._dirpath.mkdir(parents=True, exist_ok=True)
        
        # Use .apex file format
        self._db_path = self._dirpath / "apexbase.apex"
        self._auto_manage = _auto_manage
        self._is_closed = False
        
        # Register to global registry
        if self._auto_manage:
            _registry.register(self, str(self._db_path))
        
        # Handle drop_if_exists
        if drop_if_exists and self._db_path.exists():
            self._db_path.unlink()
            # Also clean up FTS indexes
            fts_dir = self._dirpath / "fts_indexes"
            if fts_dir.exists():
                shutil.rmtree(fts_dir)
        
        # Validate durability parameter
        if durability not in ('fast', 'safe', 'max'):
            raise ValueError(f"durability must be 'fast', 'safe', or 'max', got '{durability}'")
        self._durability = durability
        
        # Initialize Rust storage engine, pass durability configuration
        self._storage = RustStorage(str(self._db_path), durability=durability)
        
        self._current_table = "default"
        self._batch_size = batch_size
        self._enable_cache = enable_cache
        self._cache_size = cache_size
        
        # FTS configuration - each table managed independently
        # key: table_name, value: {'enabled': bool, 'index_fields': List[str], 'config': Dict}
        self._fts_tables: Dict[str, Dict] = {}
        
        self._prefer_arrow_format = prefer_arrow_format and ARROW_AVAILABLE

    def _is_fts_enabled(self, table_name: str = None) -> bool:
        """Check if FTS is enabled for specified table"""
        table = table_name or self._current_table
        return table in self._fts_tables and self._fts_tables[table].get('enabled', False)
    
    def _get_fts_config(self, table_name: str = None) -> Optional[Dict]:
        """Get FTS configuration for specified table"""
        table = table_name or self._current_table
        return self._fts_tables.get(table)
    
    def _ensure_fts_initialized(self, table_name: str = None) -> bool:
        """Ensure FTS is initialized for specified table"""
        table = table_name or self._current_table
        
        if not self._is_fts_enabled(table):
            return False
        
        fts_config = self._fts_tables[table]
        
        if not self._storage._fts_is_initialized():
            self._storage._init_fts(
                index_fields=fts_config.get('index_fields'),
                lazy_load=fts_config.get('config', {}).get('lazy_load', False),
                cache_size=fts_config.get('config', {}).get('cache_size', 10000)
            )
        
        return True
    
    def init_fts(
        self,
        table_name: str = None,
        index_fields: Optional[List[str]] = None,
        lazy_load: bool = False,
        cache_size: int = 10000
    ) -> 'ApexClient':
        """
        Initialize full-text search (FTS) functionality
        
        This method must be called after ApexClient is properly connected. Different FTS settings can be configured for different tables.
        
        Parameters:
            table_name: str, optional
                Table name to enable FTS for. If None, use current table.
            index_fields: List[str], optional
                List of fields to index. If None, index all string fields.
            lazy_load: bool, default False
                Whether to enable lazy loading mode. In lazy loading mode, indexes are fully loaded to memory only on first query.
            cache_size: int, default 10000
                FTS cache size.
        
        Returns:
            ApexClient: Returns self, supports chain calls.
        
        Raises:
            RuntimeError: If ApexClient is not properly connected.
        
        Example:
            >>> # Basic usage - enable FTS for current table
            >>> client = ApexClient("./my_db")
            >>> client.init_fts(index_fields=['title', 'content'])
            
            >>> # Enable FTS for specific table
            >>> client.init_fts(table_name='articles', index_fields=['title', 'body'])
            
            >>> # Chain calls
            >>> client = ApexClient("./my_db").init_fts(index_fields=['name', 'description'])
            
            >>> # Advanced configuration
            >>> client.init_fts(
            ...     table_name='documents',
            ...     index_fields=['content'],
            ...     lazy_load=True,
            ...     cache_size=50000
            ... )
        """
        self._check_connection()
        
        table = table_name or self._current_table
        
        # If need to switch table
        need_switch = table != self._current_table
        original_table = self._current_table if need_switch else None
        
        try:
            if need_switch:
                self.use_table(table)
            
            # Save FTS configuration
            self._fts_tables[table] = {
                'enabled': True,
                'index_fields': index_fields,
                'config': {
                    'lazy_load': lazy_load,
                    'cache_size': cache_size,
                }
            }
            
            # Initialize Rust native FTS
            self._storage._init_fts(
                index_fields=index_fields,
                lazy_load=lazy_load,
                cache_size=cache_size
            )
            
        finally:
            if need_switch and original_table is not None:
                self.use_table(original_table)
        
        return self

    def _should_index_field(self, field_name: str, field_value, table_name: str = None) -> bool:
        """Determine if field should be indexed"""
        table = table_name or self._current_table
        
        if not self._is_fts_enabled(table):
            return False
        
        if field_name == '_id':
            return False
        
        fts_config = self._fts_tables.get(table, {})
        index_fields = fts_config.get('index_fields')
        
        if index_fields:
            return field_name in index_fields
        
        return isinstance(field_value, str)

    def _extract_indexable_content(self, data: dict, table_name: str = None) -> dict:
        """Extract indexable content"""
        table = table_name or self._current_table
        
        if not self._is_fts_enabled(table):
            return {}
        
        indexable = {}
        for key, value in data.items():
            if self._should_index_field(key, value, table):
                indexable[key] = str(value)
        return indexable

    def _check_connection(self):
        """Check connection status"""
        if self._is_closed:
            raise RuntimeError("ApexClient connection has been closed, cannot perform operations. Please create a new instance.")

    # ============ Public API ============

    def use_table(self, table_name: str):
        """Switch to the specified table.
        
        Args:
            table_name: Name of the table to switch to.
            
        Raises:
            RuntimeError: If the client connection is closed.
        """
        self._check_connection()
        self._storage.use_table(table_name)
        self._current_table = table_name
        # FTS engine is created on-demand in Rust layer, no need to manage in Python layer

    @property
    def current_table(self) -> str:
        """Get the name of the current table.
        
        Returns:
            str: Name of the current table.
        """
        return self._current_table

    def create_table(self, table_name: str):
        """Create a new table and switch to it.
        
        Args:
            table_name: Name of the table to create.
            
        Raises:
            RuntimeError: If the client connection is closed.
        """
        self._check_connection()
        self._storage.create_table(table_name)
        self._current_table = table_name
        # FTS engine is created on-demand in Rust layer

    def drop_table(self, table_name: str):
        """Delete the specified table.
        
        Args:
            table_name: Name of the table to delete.
            
        Raises:
            RuntimeError: If the client connection is closed.
        """
        self._check_connection()
        self._storage.drop_table(table_name)
        
        # FTS index files will be cleaned up in Rust layer (if needed)
        # Can also be manually cleaned up
        if self._is_fts_enabled(table_name):
            fts_index_file = self._dirpath / "fts_indexes" / f"{table_name}.nfts"
            fts_wal_file = self._dirpath / "fts_indexes" / f"{table_name}.nfts.wal"
            if fts_index_file.exists():
                fts_index_file.unlink()
            if fts_wal_file.exists():
                fts_wal_file.unlink()
            # Remove FTS configuration
            self._fts_tables.pop(table_name, None)
        
        if self._current_table == table_name:
            self._current_table = "default"

    def list_tables(self) -> List[str]:
        """List all tables in the database.
        
        Returns:
            List[str]: List of table names.
            
        Raises:
            RuntimeError: If the client connection is closed.
        """
        self._check_connection()
        return self._storage.list_tables()

    def store(self, data) -> None:
        """Store data using automatically selected optimal strategy for ultra-fast writes.
        
        Supports multiple input formats:
        - dict: Single record
        - List[dict]: Multiple records (automatically converted to columnar high-speed path)
        - Dict[str, list]: Columnar data (fastest path)
        - Dict[str, np.ndarray]: numpy columnar data (zero-copy, fastest)
        - pandas.DataFrame: Batch storage
        - polars.DataFrame: Batch storage
        - pyarrow.Table: Batch storage
        
        Args:
            data: Data to store in any supported format.
            
        Raises:
            RuntimeError: If the client connection is closed.
            ValueError: If data format is not supported.
            
        Note:
            Performance benchmarks (10,000 rows):
            - Dict[str, np.ndarray] pure numeric: ~0.1ms (90M rows/s)
            - Dict[str, list] mixed types: ~0.7ms (14M rows/s)
            - List[dict]: ~4.8ms (2M rows/s)
        
        Examples:
            Fastest numpy columnar:
            >>> client.store({
            ...     'id': np.arange(10000, dtype=np.int64),
            ...     'score': np.random.random(10000),
            ... })
            
            Fast list columnar:
            >>> client.store({
            ...     'name': ['Alice', 'Bob', 'Charlie'],
            ...     'age': [25, 30, 35],
            ... })
            
            Single record:
            >>> client.store({'name': 'Alice', 'age': 25})
        """
        self._check_connection()
        
        # 1. Detect columnar data Dict[str, list/ndarray] - fastest path
        if isinstance(data, dict):
            first_value = next(iter(data.values()), None) if data else None
            # Detect list, tuple, or numpy array
            if first_value is not None and (
                isinstance(first_value, (list, tuple)) or 
                hasattr(first_value, '__len__') and hasattr(first_value, 'dtype')
            ):
                self._store_columnar_fast(data)
                return
        
        # 2. PyArrow Table
        if ARROW_AVAILABLE and hasattr(data, 'schema'):
            self._store_via_arrow_fast(data)
            return
        
        # 3. Pandas DataFrame
        if ARROW_AVAILABLE and pd is not None and isinstance(data, pd.DataFrame):
            table = pa.Table.from_pandas(data)
            self._store_via_arrow_fast(table)
            return
        
        # 4. Polars DataFrame
        if POLARS_AVAILABLE and pl is not None and hasattr(data, 'to_arrow'):
            table = data.to_arrow()
            if ARROW_AVAILABLE:
                self._store_via_arrow_fast(table)
                return
        
        # 5. Single record dict
        if isinstance(data, dict):
            # Get row count before insertion as ID for new record
            doc_id = self._storage.count_rows()
            self._storage._store_single_no_return(data)
            
            # Update FTS index (using Rust native implementation)
            if self._is_fts_enabled() and self._ensure_fts_initialized():
                indexable = self._extract_indexable_content(data)
                if indexable:
                    self._storage._fts_add_document(doc_id, indexable)
                    self._storage._fts_flush()
            return
            
        # 6. List[dict] - automatically convert to columnar storage
        elif isinstance(data, list):
            if not data:
                return
            self._store_list_fast(data)
            return
        else:
            raise ValueError("Data must be dict, list of dicts, Dict[str, list], pandas.DataFrame, polars.DataFrame, or pyarrow.Table")

    def _store_list_fast(self, data: List[dict]) -> None:
        """Internal method: high-speed list storage - automatically convert to columnar, no return value"""
        if not data:
            return
        
        # Get row count before insertion, to calculate ID range after insertion
        start_id = self._storage.count_rows()
        
        # Convert to columnar format
        int_cols = {}
        float_cols = {}
        str_cols = {}
        bool_cols = {}
        bin_cols = {}
        
        # Determine column types from first row
        first_row = data[0]
        col_types = {}  # name -> type
        
        for name, value in first_row.items():
            if name == '_id':
                continue
            if isinstance(value, bool):  # bool must be checked before int
                col_types[name] = 'bool'
                bool_cols[name] = []
            elif isinstance(value, int):
                col_types[name] = 'int'
                int_cols[name] = []
            elif isinstance(value, float):
                col_types[name] = 'float'
                float_cols[name] = []
            elif isinstance(value, bytes):
                col_types[name] = 'bytes'
                bin_cols[name] = []
            elif isinstance(value, str):
                col_types[name] = 'str'
                str_cols[name] = []
            else:
                col_types[name] = 'str'  # default to string
                str_cols[name] = []
        
        # Collect all data
        for row in data:
            for name, col_type in col_types.items():
                value = row.get(name)
                if col_type == 'int':
                    int_cols[name].append(value if isinstance(value, int) else 0)
                elif col_type == 'float':
                    float_cols[name].append(float(value) if value is not None else 0.0)
                elif col_type == 'bool':
                    bool_cols[name].append(bool(value) if value is not None else False)
                elif col_type == 'bytes':
                    bin_cols[name].append(value if isinstance(value, bytes) else b'')
                else:  # str
                    str_cols[name].append(str(value) if value is not None else '')
        
        # Use high-speed API that doesn't return IDs
        # If FTS is enabled, pass index field names to let Rust directly build FTS documents (zero boundary crossing!)
        fts_config = self._fts_tables.get(self._current_table, {})
        fts_fields = fts_config.get('index_fields') if (self._is_fts_enabled() and self._ensure_fts_initialized()) else None
        self._storage._insert_typed_columns_fast(
            int_cols, float_cols, str_cols, bool_cols, bin_cols, fts_fields
        )

    def _store_columnar_fast(self, columns: Dict[str, list]) -> None:
        """内部方法：高速列式存储 - 无返回值"""
        if not columns:
            return
        
        # 获取插入前的行数，用于计算插入后的 ID 范围
        start_id = self._storage.count_rows()
        
        # 计算批次大小
        first_col = next(iter(columns.values()))
        batch_size = len(first_col) if hasattr(first_col, '__len__') else 0
        
        # 检查是否全是 numpy 数值类型 - 使用零拷贝高速路径
        all_numpy_numeric = True
        
        for name, values in columns.items():
            if hasattr(values, 'dtype'):
                dtype_str = str(values.dtype)
                if 'int' not in dtype_str and 'float' not in dtype_str and 'bool' not in dtype_str:
                    all_numpy_numeric = False
                    break
            else:
                all_numpy_numeric = False
                break
        
        # 纯 numpy 数值：使用 UNSAFE 零拷贝路径 - 最高性能
        if all_numpy_numeric:
            col_names = []
            int_arrays = []
            float_arrays = []
            bool_lists = []
            
            for name, arr in columns.items():
                col_names.append(name)
                dtype_str = str(arr.dtype)
                if 'int' in dtype_str:
                    int_arrays.append(np.ascontiguousarray(arr, dtype=np.int64))
                elif 'float' in dtype_str:
                    float_arrays.append(np.ascontiguousarray(arr, dtype=np.float64))
                elif 'bool' in dtype_str:
                    bool_lists.append(arr.tolist())
            
            self._storage._insert_numpy_unsafe(col_names, int_arrays, float_arrays, bool_lists)
            # 纯数值数据通常不需要 FTS 索引
            return
        
        # 混合类型：走通用路径
        int_cols = {}
        float_cols = {}
        str_cols = {}
        bool_cols = {}
        bin_cols = {}
        
        for name, values in columns.items():
            if hasattr(values, '__len__') and len(values) == 0:
                continue
            
            # 处理 numpy arrays
            if hasattr(values, 'dtype'):
                dtype_str = str(values.dtype)
                if 'int' in dtype_str:
                    int_cols[name] = values.tolist()
                elif 'float' in dtype_str:
                    float_cols[name] = values.tolist()
                elif 'bool' in dtype_str:
                    bool_cols[name] = values.tolist()
                else:
                    str_cols[name] = [str(v) for v in values]
                continue
            
            sample = values[0]
            if isinstance(sample, bool):
                bool_cols[name] = list(values) if not isinstance(values, list) else values
            elif isinstance(sample, int):
                int_cols[name] = list(values) if not isinstance(values, list) else values
            elif isinstance(sample, float):
                float_cols[name] = list(values) if not isinstance(values, list) else values
            elif isinstance(sample, bytes):
                bin_cols[name] = list(values) if not isinstance(values, list) else values
            elif isinstance(sample, str):
                str_cols[name] = list(values) if not isinstance(values, list) else values
            else:
                str_cols[name] = [str(v) for v in values]
        
        # 如果启用 FTS，传入索引字段名让 Rust 直接构建 FTS 文档 (零边界跨越!)
        fts_config = self._fts_tables.get(self._current_table, {})
        fts_fields = fts_config.get('index_fields') if (self._is_fts_enabled() and self._ensure_fts_initialized()) else None
        self._storage._insert_typed_columns_fast(
            int_cols, float_cols, str_cols, bool_cols, bin_cols, fts_fields
        )

    def _store_via_arrow_fast(self, table) -> None:
        """内部方法：通过 Arrow 存储 - 使用最快路径"""
        # 注意: 测试表明 Arrow IPC 序列化/反序列化开销较大
        # 直接转换为 Python 列表 + _insert_typed_columns_fast 更快
        columns = {}
        for col_name in table.column_names:
            col = table.column(col_name)
            columns[col_name] = col.to_pylist()
        
        self._store_columnar_fast(columns)

    def query(self, where: str = None, limit: int = None) -> ResultView:
        """Query records using SQL syntax with optional optimization.
        
        Args:
            where: SQL WHERE clause for filtering records (e.g., "age > 25 AND city = 'NYC'").
                If None or "1=1", returns all records.
            limit: Optional maximum number of records to return.
                When specified, enables streaming early-stop optimization for faster queries.
        
        Returns:
            ResultView: Query result view supporting multiple output formats:
                to_dict(), to_pandas(), to_polars(), to_arrow()
        
        Raises:
            RuntimeError: If the client connection is closed.
        
        Examples:
            Basic query:
            >>> results = client.query("age > 25")
            
            Limited query with optimization:
            >>> results = client.query("city = 'NYC'", limit=100)
            
            Convert to pandas:
            >>> df = client.query("score > 0.5").to_pandas()
        """
        self._check_connection()
        
        where_clause = where if where and where.strip() != "1=1" else "1=1"
        
        # If limit is specified, use streaming early-stop optimization (fastest path)
        if limit is not None:
            results = self._storage.query(where_clause, limit)
            return ResultView.from_dicts(results)
        
        if not ARROW_AVAILABLE:
            results = self._storage.query(where_clause)
            return ResultView.from_dicts(results)
        
        # Prioritize FFI zero-copy method (fastest)
        try:
            import pyarrow as pa
            
            schema_ptr, array_ptr = self._storage._query_arrow_ffi(where_clause)
            
            if schema_ptr == 0 and array_ptr == 0:
                return ResultView.from_arrow_bytes(b'')
            
            try:
                struct_array = pa.Array._import_from_c(array_ptr, schema_ptr)
                
                if isinstance(struct_array, pa.StructArray):
                    batch = pa.RecordBatch.from_struct_array(struct_array)
                    table = pa.Table.from_batches([batch])
                    return ResultView(table)
            finally:
                self._storage._free_arrow_ffi(schema_ptr, array_ptr)
                
        except Exception:
            pass  # FFI 失败，回退到 IPC 方式
        
        # 回退到 Arrow IPC 方式
        try:
            arrow_bytes = self._storage._query_arrow(where_clause)
            return ResultView.from_arrow_bytes(arrow_bytes)
        except Exception:
            pass
        
        # 最终回退到传统方式
        results = self._storage.query(where_clause)
        return ResultView.from_dicts(results)

    def execute(self, sql: str) -> 'SqlResult':
        """
        执行完整的 SQL 语句 (SQL:2023 标准)
        
        支持的 SQL 语法:
        - SELECT columns FROM table WHERE conditions
        - ORDER BY column [ASC|DESC] [NULLS FIRST|LAST]
        - LIMIT n OFFSET m
        - DISTINCT
        - 聚合函数: COUNT, SUM, AVG, MIN, MAX
        - GROUP BY / HAVING
        - 运算符: LIKE, IN, BETWEEN, IS NULL, AND, OR, NOT
        - 比较运算符: =, !=, <>, <, <=, >, >=
        
        示例:
            # 基本查询
            result = client.execute("SELECT * FROM data WHERE age > 18")
            
            # 带排序和限制
            result = client.execute("SELECT name, age FROM data ORDER BY age DESC LIMIT 10")
            
            # 聚合查询
            result = client.execute("SELECT COUNT(*), AVG(age) FROM data WHERE status = 'active'")
            
            # LIKE 模式匹配
            result = client.execute("SELECT * FROM data WHERE name LIKE 'John%'")
            
            # GROUP BY 分组
            result = client.execute("SELECT city, COUNT(*) FROM data GROUP BY city")
        
        Parameters:
            sql: 完整的 SQL SELECT 语句
        
        Returns:
            SqlResult: 包含 columns (列名列表) 和 rows (数据行列表) 的结果对象
        """
        self._check_connection()
        
        # 尝试使用 Arrow FFI 快速路径 (用于大结果集)
        if ARROW_AVAILABLE:
            try:
                import pyarrow as pa
                schema_ptr, array_ptr = self._storage._execute_arrow_ffi(sql)
                
                if schema_ptr != 0 and array_ptr != 0:
                    # 使用 Arrow C Data Interface 零拷贝导入
                    struct_array = pa.Array._import_from_c(array_ptr, schema_ptr)
                    if isinstance(struct_array, pa.StructArray):
                        batch = pa.RecordBatch.from_struct_array(struct_array)
                        columns = batch.schema.names
                        return SqlResult(columns, [], batch.num_rows, arrow_batch=batch)
            except Exception:
                pass  # 回退到标准路径
        
        # 标准路径: SqlExecutor 处理
        result = self._storage.execute(sql)
        return SqlResult(result['columns'], result['rows'], result.get('rows_affected', 0))

    def retrieve(self, id_: int) -> Optional[dict]:
        """
        获取单个记录 - 使用 Arrow C Data Interface 零拷贝传输
        
        性能优化 (按优先级):
        1. FFI 零拷贝 - 最快，无序列化开销
        2. Arrow IPC - 次快，有序列化开销
        3. 传统回退 - 最慢
        
        Parameters:
            id_: 记录 ID
        
        Returns:
            Optional[dict]: 记录字典，如果不存在则返回 None
        """
        self._check_connection()
        
        if not ARROW_AVAILABLE:
            return self._storage.retrieve(id_)
        
        # 优先使用 FFI 零拷贝方式 (最快)
        try:
            import pyarrow as pa
            
            schema_ptr, array_ptr = self._storage._retrieve_many_arrow_ffi([id_])
            
            if schema_ptr == 0 and array_ptr == 0:
                return None
            
            try:
                struct_array = pa.Array._import_from_c(array_ptr, schema_ptr)
                
                if isinstance(struct_array, pa.StructArray):
                    batch = pa.RecordBatch.from_struct_array(struct_array)
                    table = pa.Table.from_batches([batch])
                    results = table.to_pylist()
                    return results[0] if results else None
            finally:
                self._storage._free_arrow_ffi(schema_ptr, array_ptr)
                
        except Exception:
            pass  # FFI 失败，回退到 IPC 方式
        
        # 回退到 Arrow IPC 方式
        try:
            arrow_bytes = self._storage._retrieve_many_arrow([id_])
            if arrow_bytes:
                reader = pa.ipc.open_stream(arrow_bytes)
                table = reader.read_all()
                results = table.to_pylist()
                return results[0] if results else None
        except Exception:
            pass
        
        # 最终回退到传统方式
        return self._storage.retrieve(id_)

    def retrieve_many(self, ids: List[int]) -> 'ResultView':
        """
        获取多个记录 - 使用 Arrow C Data Interface 零拷贝传输
        
        性能优化 (按优先级):
        1. FFI 零拷贝 - 最快，无序列化开销
        2. Arrow IPC - 次快，有序列化开销
        3. 传统回退 - 最慢
        
        Parameters:
            ids: 要检索的记录 ID 列表
        
        Returns:
            ResultView: 记录视图，支持多种输出格式：
                - .to_arrow() -> pyarrow.Table （零拷贝，最快）
                - .to_pandas() -> pandas.DataFrame
                - .to_polars() -> polars.DataFrame
                - .to_dict() -> List[dict]
        """
        self._check_connection()
        if not ids:
            return ResultView.from_dicts([])
        
        if not ARROW_AVAILABLE:
            return ResultView.from_dicts(self._storage.retrieve_many(ids))
        
        # 优先使用 FFI 零拷贝方式 (最快)
        try:
            import pyarrow as pa
            
            schema_ptr, array_ptr = self._storage._retrieve_many_arrow_ffi(ids)
            
            if schema_ptr == 0 and array_ptr == 0:
                return ResultView.from_dicts([])
            
            try:
                struct_array = pa.Array._import_from_c(array_ptr, schema_ptr)
                
                if isinstance(struct_array, pa.StructArray):
                    batch = pa.RecordBatch.from_struct_array(struct_array)
                    table = pa.Table.from_batches([batch])
                    return ResultView(table)
            finally:
                self._storage._free_arrow_ffi(schema_ptr, array_ptr)
                
        except Exception:
            pass  # FFI 失败，回退到 IPC 方式
        
        # 回退到 Arrow IPC 方式
        try:
            arrow_bytes = self._storage._retrieve_many_arrow(ids)
            if arrow_bytes:
                return ResultView.from_arrow_bytes(arrow_bytes)
        except Exception:
            pass
        
        # 最终回退到传统方式
        return ResultView.from_dicts(self._storage.retrieve_many(ids))

    def retrieve_all(self) -> ResultView:
        """
        获取所有记录 - 使用 Arrow C Data Interface 零拷贝传输
        
        性能优化 (按优先级):
        1. FFI 零拷贝 - 最快，无序列化开销
        2. Arrow IPC - 次快，有序列化开销
        3. 查询回退 - 最慢
        
        Returns:
            ResultView: 所有记录的视图
        """
        self._check_connection()
        
        if not ARROW_AVAILABLE:
            return self.query("1=1")
        
        # 优先使用 FFI 零拷贝方式 (最快)
        try:
            import pyarrow as pa
            
            schema_ptr, array_ptr = self._storage._retrieve_all_arrow_ffi()
            
            if schema_ptr == 0 and array_ptr == 0:
                return ResultView.from_arrow_bytes(b'')
            
            try:
                struct_array = pa.Array._import_from_c(array_ptr, schema_ptr)
                
                if isinstance(struct_array, pa.StructArray):
                    batch = pa.RecordBatch.from_struct_array(struct_array)
                    table = pa.Table.from_batches([batch])
                    return ResultView(table)
            finally:
                self._storage._free_arrow_ffi(schema_ptr, array_ptr)
                
        except Exception:
            pass  # FFI 失败，回退到 IPC 方式
        
        # 回退到 Arrow IPC 方式
        try:
            arrow_bytes = self._storage._retrieve_all_arrow_direct()
            return ResultView.from_arrow_bytes(arrow_bytes)
        except Exception:
            pass
        
        # 最终回退到查询方式
        return self.query("1=1")

    def list_fields(self) -> List[str]:
        """列出当前表的字段"""
        self._check_connection()
        return self._storage.list_fields()

    def delete(self, ids: Union[int, List[int]]) -> bool:
        """删除记录"""
        self._check_connection()
        
        if isinstance(ids, int):
            result = self._storage.delete(ids)
            
            if result and self._is_fts_enabled() and self._storage._fts_is_initialized():
                self._storage._fts_remove_document(ids)
            
            return result
            
        elif isinstance(ids, list):
            result = self._storage.delete_batch(ids)
            
            if result and self._is_fts_enabled() and self._storage._fts_is_initialized():
                self._storage._fts_remove_documents(ids)
            
            return result
        else:
            raise ValueError("ids must be an int or a list of ints")

    def replace(self, id_: int, data: dict) -> bool:
        """替换记录"""
        self._check_connection()
        result = self._storage.replace(id_, data)
        
        if result and self._is_fts_enabled() and self._storage._fts_is_initialized():
            indexable = self._extract_indexable_content(data)
            if indexable:
                self._storage._fts_update_document(id_, indexable)
                self._storage._fts_flush()
            else:
                self._storage._fts_remove_document(id_)
        
        return result

    def batch_replace(self, data_dict: Dict[int, dict]) -> List[int]:
        """批量替换记录"""
        self._check_connection()
        success_ids = []
        
        for id_, data in data_dict.items():
            if self.replace(id_, data):
                success_ids.append(id_)
        
        return success_ids

    def from_pandas(self, df) -> 'ApexClient':
        """从 Pandas DataFrame 导入数据"""
        records = df.to_dict('records')
        self.store(records)
        return self

    def from_pyarrow(self, table) -> 'ApexClient':
        """从 PyArrow Table 导入数据"""
        records = table.to_pylist()
        self.store(records)
        return self

    def from_polars(self, df) -> 'ApexClient':
        """从 Polars DataFrame 导入数据"""
        records = df.to_dicts()
        self.store(records)
        return self

    def optimize(self):
        """优化数据库性能"""
        self._check_connection()
        self._storage.optimize()

    def count_rows(self, table_name: str = None) -> int:
        """获取行数"""
        self._check_connection()
        if table_name and table_name != self._current_table:
            original = self._current_table
            self.use_table(table_name)
            count = self._storage.count_rows()
            self.use_table(original)
            return count
        return self._storage.count_rows()

    def flush(self):
        """
        将所有数据持久化到磁盘
        
        包括：
        - 表数据（.apex 文件）
        - FTS 索引（.nfts 文件）
        
        使用场景：
        - 批量写入后确保数据安全
        - 不使用 with 语句时手动持久化
        - 程序意外退出前保护数据
        
        Example:
            client = ApexClient("./my_db")
            client.init_fts(index_fields=['title', 'content'])
            client.store(data)
            client.flush()  # 数据现在已安全持久化到磁盘
        """
        self._check_connection()
        # Flush FTS 索引（所有已启用 FTS 的表）
        if self._fts_tables:
            try:
                self._storage._fts_flush()
            except Exception:
                pass
        # Flush 表数据
        self._storage.flush()
    
    def flush_cache(self):
        """刷新缓存（已弃用，请使用 flush()）"""
        self.flush()

    def drop_column(self, column_name: str):
        """删除列"""
        self._check_connection()
        if column_name == '_id':
            raise ValueError("Cannot drop _id column")
        self._storage.drop_column(column_name)

    def add_column(self, column_name: str, column_type: str):
        """添加列"""
        self._check_connection()
        self._storage.add_column(column_name, column_type)

    def rename_column(self, old_column_name: str, new_column_name: str):
        """重命名列"""
        self._check_connection()
        if old_column_name == '_id':
            raise ValueError("Cannot rename _id column")
        self._storage.rename_column(old_column_name, new_column_name)

    def get_column_dtype(self, column_name: str) -> str:
        """获取列数据类型"""
        self._check_connection()
        return self._storage.get_column_dtype(column_name)

    # ============ FTS 方法 ============

    def search_text(self, query: str, table_name: str = None) -> Optional[np.ndarray]:
        """
        执行全文搜索（Rust 原生实现，零 Python 边界开销）
        
        Parameters:
            query: 搜索查询字符串
            table_name: 表名（可选，默认使用当前表）
        
        Returns:
            np.ndarray: 匹配的文档 ID 数组
        """
        table = table_name or self._current_table
        
        if not self._is_fts_enabled(table):
            raise ValueError(f"Full-text search is not enabled for table '{table}'. Call init_fts() first.")
        
        if not self._ensure_fts_initialized(table):
            return None
        
        # 切换表（如果需要）
        if table_name and table_name != self._current_table:
            original = self._current_table
            self.use_table(table_name)
            result = self._storage._fts_search(query)
            self.use_table(original)
            return result
        
        return self._storage._fts_search(query)

    def fuzzy_search_text(self, query: str, min_results: int = 1, table_name: str = None) -> Optional[np.ndarray]:
        """
        执行模糊全文搜索（Rust 原生实现，零 Python 边界开销）
        
        Parameters:
            query: 搜索查询字符串（支持拼写错误）
            min_results: 触发模糊搜索的最小结果数
            table_name: 表名（可选，默认使用当前表）
        
        Returns:
            np.ndarray: 匹配的文档 ID 数组
        """
        table = table_name or self._current_table
        
        if not self._is_fts_enabled(table):
            raise ValueError(f"Full-text search is not enabled for table '{table}'. Call init_fts() first.")
        
        if not self._ensure_fts_initialized(table):
            return None
        
        # 切换表（如果需要）
        if table_name and table_name != self._current_table:
            original = self._current_table
            self.use_table(table_name)
            result = self._storage._fts_fuzzy_search(query, min_results)
            self.use_table(original)
            return result
        
        return self._storage._fts_fuzzy_search(query, min_results)

    def search_and_retrieve(self, query: str, table_name: str = None, 
                           limit: Optional[int] = None, offset: int = 0) -> 'ResultView':
        """
        执行全文搜索并返回完整记录（Rust 原生实现，零 Python 边界开销）
        
        这是最快的搜索+检索路径，因为：
        1. 搜索在 Rust 层完成（无 Python 边界开销）
        2. 检索在 Rust 层完成（无 Python 边界开销）
        3. 直接返回 Arrow IPC 字节
        
        Parameters:
            query: 搜索查询字符串
            table_name: 表名（可选，默认使用当前表）
            limit: 返回结果数量限制
            offset: 结果偏移量
        
        Returns:
            ResultView: 查询结果视图，支持多种输出格式：
                - .to_arrow() -> pyarrow.Table （零拷贝，最快）
                - .to_pandas() -> pandas.DataFrame
                - .to_polars() -> polars.DataFrame
                - .to_dict() -> List[dict]
        
        Example:
            >>> results = client.search_and_retrieve("Python")
            >>> arrow_table = results.to_arrow()  # 最快
            >>> df = results.to_pandas()
        """
        table = table_name or self._current_table
        
        if not self._is_fts_enabled(table):
            raise ValueError(f"Full-text search is not enabled for table '{table}'. Call init_fts() first.")
        
        if not self._ensure_fts_initialized(table):
            return ResultView.from_dicts([])
        
        # 快速路径：使用默认表名
        if table_name is None:
            table_name = self._current_table
        
        # 切换表（如果需要）
        need_switch = table_name != self._current_table
        original_table = self._current_table if need_switch else None
        
        try:
            if need_switch:
                self.use_table(table_name)
            
            if not ARROW_AVAILABLE:
                return ResultView.from_dicts([])
            
            # 优先使用 FFI 零拷贝方式 (最快)
            try:
                import pyarrow as pa
                
                schema_ptr, array_ptr = self._storage._fts_search_and_retrieve_ffi(query, limit, offset)
                
                if schema_ptr == 0 and array_ptr == 0:
                    return ResultView.from_dicts([])
                
                try:
                    struct_array = pa.Array._import_from_c(array_ptr, schema_ptr)
                    
                    if isinstance(struct_array, pa.StructArray):
                        batch = pa.RecordBatch.from_struct_array(struct_array)
                        table = pa.Table.from_batches([batch])
                        return ResultView(table)
                finally:
                    self._storage._free_arrow_ffi(schema_ptr, array_ptr)
                    
            except Exception:
                pass  # FFI 失败，回退到 IPC 方式
            
            # 回退到 Arrow IPC 方式
            try:
                arrow_bytes = self._storage._fts_search_and_retrieve(query, limit, offset)
                
                if arrow_bytes and len(arrow_bytes) > 0:
                    return ResultView.from_arrow_bytes(arrow_bytes)
            except Exception:
                pass
            
            return ResultView.from_dicts([])
                
        finally:
            if need_switch and original_table is not None:
                self.use_table(original_table)

    def search_and_retrieve_top(self, query: str, n: int = 100, table_name: str = None) -> 'ResultView':
        """
        执行全文搜索并返回前 N 条完整记录（Rust 原生实现）
        
        这是 search_and_retrieve 的简化版本，专门用于获取前 N 条结果。
        
        Parameters:
            query: 搜索查询字符串
            n: 返回的最大结果数
            table_name: 表名（可选，默认使用当前表）
        
        Returns:
            ResultView: 查询结果视图
        """
        return self.search_and_retrieve(query, table_name=table_name, limit=n, offset=0)

    def set_fts_fuzzy_config(self, threshold: float = 0.7, max_distance: int = 2, 
                             max_candidates: int = 20, table_name: str = None):
        """
        设置模糊搜索配置
        
        Parameters:
            threshold: 相似度阈值（0.0-1.0），越高越严格
            max_distance: 最大编辑距离
            max_candidates: 最大候选词数量
            table_name: 表名（可选，默认使用当前表）
        """
        table = table_name or self._current_table
        
        if not self._is_fts_enabled(table):
            raise ValueError(f"Full-text search is not enabled for table '{table}'. Call init_fts() first.")
        
        if not self._ensure_fts_initialized(table):
            return
        
        # 切换表（如果需要）
        if table_name and table_name != self._current_table:
            original = self._current_table
            self.use_table(table_name)
            self._storage._fts_set_fuzzy_config(threshold, max_distance, max_candidates)
            self.use_table(original)
        else:
            self._storage._fts_set_fuzzy_config(threshold, max_distance, max_candidates)

    def get_fts_stats(self, table_name: str = None) -> Dict:
        """
        获取 FTS 引擎统计信息
        
        Parameters:
            table_name: 表名（可选，默认使用当前表）
        
        Returns:
            Dict: FTS 引擎统计信息
        """
        table = table_name or self._current_table
        
        if not self._is_fts_enabled(table):
            return {'fts_enabled': False, 'table': table}
        
        if not self._storage._fts_is_initialized():
            return {'fts_enabled': True, 'engine_initialized': False, 'table': table}
        
        # 切换表（如果需要）
        if table_name and table_name != self._current_table:
            original = self._current_table
            self.use_table(table_name)
            stats = self._storage._fts_stats()
            self.use_table(original)
        else:
            stats = self._storage._fts_stats()
        
        stats['fts_enabled'] = True
        stats['engine_initialized'] = True
        return stats

    def compact_fts_index(self, table_name: str = None):
        """
        压缩 FTS 索引，应用删除操作并优化存储
        
        Parameters:
            table_name: 表名（可选，默认使用当前表）
        """
        table = table_name or self._current_table
        
        if not self._is_fts_enabled(table) or not self._storage._fts_is_initialized():
            return
        
        # 切换表（如果需要）
        if table_name and table_name != self._current_table:
            original = self._current_table
            self.use_table(table_name)
            self._storage._fts_compact()
            self.use_table(original)
        else:
            self._storage._fts_compact()

    def warmup_fts_terms(self, terms: List[str], table_name: str = None) -> int:
        """
        预热 FTS 缓存（懒加载模式下有效）
        
        Parameters:
            terms: 要预热的词项列表
            table_name: 表名（可选，默认使用当前表）
        
        Returns:
            int: 成功加载的词项数量
        """
        table = table_name or self._current_table
        
        if not self._is_fts_enabled(table) or not self._storage._fts_is_initialized():
            return 0
        
        # 切换表（如果需要）
        if table_name and table_name != self._current_table:
            original = self._current_table
            self.use_table(table_name)
            result = self._storage._fts_warmup_terms(terms)
            self.use_table(original)
            return result
        
        return self._storage._fts_warmup_terms(terms)

    # ============ 生命周期管理 ============

    def _force_close(self):
        """强制关闭 - 确保数据持久化"""
        try:
            if hasattr(self, '_storage') and self._storage is not None:
                # 先 flush FTS 索引
                if hasattr(self, '_fts_tables') and self._fts_tables:
                    try:
                        self._storage._fts_flush()
                    except Exception:
                        pass
                # 必须先 flush 确保数据持久化！
                try:
                    self._storage.flush()
                except Exception:
                    pass
                # 然后关闭
                self._storage.close()
                self._storage = None
        except Exception:
            pass
        self._is_closed = True

    def close(self):
        """关闭数据库连接"""
        if self._is_closed:
            return
        
        try:
            if hasattr(self, '_storage') and self._storage is not None:
                # 先 flush FTS 索引
                if self._fts_tables:
                    try:
                        self._storage._fts_flush()
                    except Exception:
                        pass
                # 再 flush 数据
                self._storage.flush()
                self._storage.close()
                self._storage = None
        finally:
            self._is_closed = True
            if self._auto_manage:
                _registry.unregister(str(self._db_path))

    @classmethod
    def create_clean(cls, dirpath=None, **kwargs):
        """创建全新实例，强制清理之前的数据"""
        kwargs['drop_if_exists'] = True
        return cls(dirpath=dirpath, **kwargs)

    def __enter__(self):
        """
        上下文管理器入口 - 支持with语句
        
        Returns:
            ApexClient: 返回自身实例，支持链式调用
            
        Example:
            >>> with ApexClient("./my_db") as client:
            ...     client.store({"name": "Alice"})
            ...     # 自动调用 close()
        """
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        上下文管理器出口 - 自动清理资源
        
        无论是否发生异常，都会确保：
        - FTS 索引被正确刷新
        - 数据被持久化到磁盘（通过 flush()）
        - 数据库连接被关闭
        - 从全局注册表中注销（如果启用了auto_manage）
        
        注意：ApexBase 不实现传统的事务回滚机制。如果在上下文中
        发生异常，异常发生前的数据操作通常会被持久化，但具体行为
        取决于异常发生时数据是否已被刷新到磁盘。
        
        Args:
            exc_type: 异常类型（如果有）
            exc_val: 异常值（如果有）
            exc_tb: 异常跟踪（如果有）
            
        Returns:
            bool: 返回 False，不抑制异常，让调用者处理
        """
        self.close()
        return False

    def __del__(self):
        if hasattr(self, '_is_closed') and not self._is_closed:
            self._force_close()

    def __repr__(self):
        return f"ApexClient(path='{self._dirpath}', table='{self._current_table}')"


# 导出
__all__ = ['ApexClient', 'ResultView', 'DurabilityLevel', '__version__', 'FTS_AVAILABLE', 'ARROW_AVAILABLE', 'POLARS_AVAILABLE']

