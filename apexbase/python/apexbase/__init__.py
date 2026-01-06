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
import pyarrow as pa
import pandas as pd
ARROW_AVAILABLE = True

import polars as pl
POLARS_AVAILABLE = True

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
    return ResultView(arrow_table=pa.table({}), data=[])


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

        self._fts_dirty: bool = False
        
        self._prefer_arrow_format = prefer_arrow_format and ARROW_AVAILABLE
        
        # Reference to global registry for testing purposes
        self._registry = _registry

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
        if self._is_closed or self._storage is None:
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
        self._check_connection()
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
        try:
            self._storage.drop_table(table_name)
        except ValueError:
            # Graceful handling - table doesn't exist, nothing to drop
            pass
        
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
                    self._fts_dirty = True
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
        """Internal method: high-speed columnar storage - no return value"""
        if not columns:
            return
        
        # Validate all columns have same length
        lengths = {}
        for name, values in columns.items():
            length = len(values) if hasattr(values, '__len__') else 0
            lengths[name] = length
        
        unique_lengths = set(lengths.values())
        if len(unique_lengths) > 1:
            length_details = ", ".join(f"'{k}': {v}" for k, v in lengths.items())
            raise ValueError(
                f"All columns must have the same length for columnar storage. "
                f"Got different lengths: {{{length_details}}}"
            )
        
        # Get row count before insertion, to calculate ID range after insertion
        start_id = self._storage.count_rows()
        
        # Calculate batch size
        first_col = next(iter(columns.values()))
        batch_size = len(first_col) if hasattr(first_col, '__len__') else 0
        
        # Check if all are numpy numeric types - use zero-copy high-speed path
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
        
        # Pure numpy numeric: use UNSAFE zero-copy path - highest performance
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
            # Pure numeric data usually doesn't need FTS indexing
            return
        
        # Mixed types: use general path
        int_cols = {}
        float_cols = {}
        str_cols = {}
        bool_cols = {}
        bin_cols = {}
        
        for name, values in columns.items():
            if hasattr(values, '__len__') and len(values) == 0:
                continue
            
            # Handle numpy arrays
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
        
        # If FTS is enabled, pass index field names to let Rust directly build FTS documents (zero boundary crossing!)
        fts_config = self._fts_tables.get(self._current_table, {})
        fts_fields = fts_config.get('index_fields') if (self._is_fts_enabled() and self._ensure_fts_initialized()) else None
        self._storage._insert_typed_columns_fast(
            int_cols, float_cols, str_cols, bool_cols, bin_cols, fts_fields
        )

    def _store_via_arrow_fast(self, table) -> None:
        """Internal method: store via Arrow - use fastest path"""
        # Note: Testing shows Arrow IPC serialization/deserialization overhead is large
        # Direct conversion to Python lists + _insert_typed_columns_fast is faster
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
        
        schema_ptr, array_ptr = self._storage._query_arrow_ffi(where_clause, limit)

        if schema_ptr == 0 and array_ptr == 0:
            return ResultView(arrow_table=pa.table({}), data=[])

        try:
            struct_array = pa.Array._import_from_c(array_ptr, schema_ptr)
            if isinstance(struct_array, pa.StructArray):
                batch = pa.RecordBatch.from_struct_array(struct_array)
                table = pa.Table.from_batches([batch])
                return ResultView(table)
            raise RuntimeError("FFI import did not return StructArray")
        finally:
            self._storage._free_arrow_ffi(schema_ptr, array_ptr)

    def execute(self, sql: str) -> ResultView:
        """
        Execute complete SQL statement (SQL:2023 standard)
        
        Supported SQL syntax:
        - SELECT columns FROM table WHERE conditions
        - ORDER BY column [ASC|DESC] [NULLS FIRST|LAST]
        - LIMIT n OFFSET m
        - DISTINCT
        - Aggregate functions: COUNT, SUM, AVG, MIN, MAX
        - GROUP BY / HAVING
        - Operators: LIKE, IN, BETWEEN, IS NULL, AND, OR, NOT
        - Comparison operators: =, !=, <>, <, <=, >, >=
        
        Examples:
            Basic query:
            >>> result = client.execute("SELECT * FROM default WHERE age > 18")
            
            Query with ordering and limit:
            >>> result = client.execute("SELECT name, age FROM default ORDER BY age DESC LIMIT 10")
            
            Aggregate query:
            >>> result = client.execute("SELECT COUNT(*), AVG(age) FROM default WHERE status = 'active'")
            
            LIKE pattern matching:
            >>> result = client.execute("SELECT * FROM default WHERE name LIKE 'John%'")
            
            GROUP BY grouping:
            >>> result = client.execute("SELECT city, COUNT(*) FROM default GROUP BY city")
        
        Args:
            sql: Complete SQL SELECT statement
        
        Returns:
            ResultView: SQL execution result view (Arrow-first)
        """
        self._check_connection()
        
        # Prefer Arrow FFI fast path
        if ARROW_AVAILABLE:
            schema_ptr, array_ptr = self._storage._execute_arrow_ffi(sql)
            if schema_ptr == 0 and array_ptr == 0:
                return ResultView(arrow_table=pa.table({}), data=[])
            try:
                struct_array = pa.Array._import_from_c(array_ptr, schema_ptr)
                if isinstance(struct_array, pa.StructArray):
                    batch = pa.RecordBatch.from_struct_array(struct_array)
                    table = pa.Table.from_batches([batch])
                    return ResultView(table)
                raise RuntimeError("FFI import did not return StructArray")
            except Exception:
                pass
            finally:
                self._storage._free_arrow_ffi(schema_ptr, array_ptr)

        # Fallback path (no Arrow): execute via row-based API and wrap as ResultView
        result = self._storage.execute(sql)
        columns = result.get('columns', [])
        rows = result.get('rows', [])
        data = [{k: v for k, v in zip(columns, row) if k != '_id'} for row in rows]
        return ResultView(arrow_table=None, data=data)

    def retrieve(self, id_: int) -> Optional[dict]:
        """
        Retrieve single record - using traditional method for type fidelity
        
        Performance optimization (by priority):
        1. Traditional method - highest type fidelity (preserves binary data)
        2. FFI zero-copy - fastest, but may convert some types
        3. Arrow IPC - second fastest, has serialization overhead
        
        Args:
            id_: Record ID
        
        Returns:
            Optional[dict]: Record dictionary, returns None if not found
        """
        self._check_connection()

        # Highest type fidelity (preserves binary data)
        try:
            result = self._storage.retrieve(id_)
            if result is not None:
                return result
        except Exception:
            # Fallback to Arrow FFI path
            pass
        
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
            raise RuntimeError("FFI import did not return StructArray")
        finally:
            self._storage._free_arrow_ffi(schema_ptr, array_ptr)

    def retrieve_many(self, ids: List[int]) -> 'ResultView':
        """
        Retrieve multiple records - using Arrow C Data Interface zero-copy transfer
        
        Performance optimization (by priority):
        1. FFI zero-copy - fastest, no serialization overhead
        2. Arrow IPC - second fastest, has serialization overhead
        3. Traditional fallback - slowest
        
        Args:
            ids: List of record IDs to retrieve
        
        Returns:
            ResultView: Record view supporting multiple output formats:
                - .to_arrow() -> pyarrow.Table (zero-copy, fastest)
                - .to_pandas() -> pandas.DataFrame
                - .to_polars() -> polars.DataFrame
                - .to_dict() -> List[dict]
        """
        self._check_connection()
        if not ids:
            return _empty_result_view()

        schema_ptr, array_ptr = self._storage._retrieve_many_arrow_ffi(ids)
        if schema_ptr == 0 and array_ptr == 0:
            return _empty_result_view()
        try:
            struct_array = pa.Array._import_from_c(array_ptr, schema_ptr)
            if isinstance(struct_array, pa.StructArray):
                batch = pa.RecordBatch.from_struct_array(struct_array)
                table = pa.Table.from_batches([batch])
                return ResultView(table)
            raise RuntimeError("FFI import did not return StructArray")
        finally:
            self._storage._free_arrow_ffi(schema_ptr, array_ptr)

    def retrieve_all(self) -> ResultView:
        """
        Retrieve all records - using Arrow C Data Interface zero-copy transfer
        
        Performance optimization (by priority):
        1. FFI zero-copy - fastest, no serialization overhead
        2. Arrow IPC - second fastest, has serialization overhead
        3. Query fallback - slowest
        
        Returns:
            ResultView: View of all records
        """
        self._check_connection()
        
        schema_ptr, array_ptr = self._storage._retrieve_all_arrow_ffi()

        if schema_ptr == 0 and array_ptr == 0:
            return ResultView(arrow_table=pa.table({}), data=[])

        try:
            struct_array = pa.Array._import_from_c(array_ptr, schema_ptr)
            if isinstance(struct_array, pa.StructArray):
                batch = pa.RecordBatch.from_struct_array(struct_array)
                table = pa.Table.from_batches([batch])
                return ResultView(table)
            raise RuntimeError("FFI import did not return StructArray")
        finally:
            self._storage._free_arrow_ffi(schema_ptr, array_ptr)

    def list_fields(self) -> List[str]:
        """List fields of current table"""
        self._check_connection()
        return self._storage.list_fields()

    def delete(self, ids: Union[int, List[int]]) -> bool:
        """Delete records"""
        self._check_connection()
        
        if isinstance(ids, int):
            result = self._storage.delete(ids)
            
            if result and self._is_fts_enabled() and self._storage._fts_is_initialized():
                self._storage._fts_remove_document(ids)
                self._fts_dirty = True
            
            return result
            
        elif isinstance(ids, list):
            result = self._storage.delete_batch(ids)
            
            if result and self._is_fts_enabled() and self._storage._fts_is_initialized():
                self._storage._fts_remove_documents(ids)
                self._fts_dirty = True
            
            return result
        else:
            raise ValueError("ids must be an int or a list of ints")

    def replace(self, id_: int, data: dict) -> bool:
        """Replace record"""
        self._check_connection()
        result = self._storage.replace(id_, data)
        
        if result and self._is_fts_enabled() and self._storage._fts_is_initialized():
            indexable = self._extract_indexable_content(data)
            if indexable:
                self._storage._fts_update_document(id_, indexable)
                self._fts_dirty = True
            else:
                self._storage._fts_remove_document(id_)
                self._fts_dirty = True
        
        return result

    def batch_replace(self, data_dict: Dict[int, dict]) -> List[int]:
        """Batch replace records"""
        self._check_connection()
        success_ids = []
        
        for id_, data in data_dict.items():
            if self.replace(id_, data):
                success_ids.append(id_)
        
        return success_ids

    def from_pandas(self, df) -> 'ApexClient':
        """Import data from Pandas DataFrame"""
        records = df.to_dict('records')
        self.store(records)
        return self

    def from_pyarrow(self, table) -> 'ApexClient':
        """Import data from PyArrow Table"""
        records = table.to_pylist()
        self.store(records)
        return self

    def from_polars(self, df) -> 'ApexClient':
        """Import data from Polars DataFrame"""
        records = df.to_dicts()
        self.store(records)
        return self

    def optimize(self):
        """Optimize database performance"""
        self._check_connection()
        self._storage.optimize()

    def count_rows(self, table_name: str = None) -> int:
        """Get row count"""
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
        Persist all data to disk
        
        Includes:
        - Table data (.apex files)
        - FTS index (.nfts files)
        
        Use cases:
        - Ensure data safety after batch writes
        - Manual persistence when not using with statements
        - Protect data before unexpected program exit
        
        Example:
            client = ApexClient("./my_db")
            client.init_fts(index_fields=['title', 'content'])
            client.store(data)
            client.flush()  # Data is now safely persisted to disk
        """
        self._check_connection()
        # Flush FTS index (all tables with FTS enabled)
        if self._fts_tables and self._fts_dirty:
            try:
                self._storage._fts_flush()
            except Exception:
                pass
            self._fts_dirty = False

        # Flush table data
        self._storage.flush()
    
    def flush_cache(self):
        """Flush cache (deprecated, please use flush())"""
        self.flush()

    def drop_column(self, column_name: str):
        """Drop column"""
        self._check_connection()
        if column_name == '_id':
            raise ValueError("Cannot drop _id column")
        self._storage.drop_column(column_name)

    def add_column(self, column_name: str, column_type: str):
        """Add column"""
        self._check_connection()
        self._storage.add_column(column_name, column_type)

    def rename_column(self, old_column_name: str, new_column_name: str):
        """Rename column"""
        self._check_connection()
        if old_column_name == '_id':
            raise ValueError("Cannot rename _id column")
        self._storage.rename_column(old_column_name, new_column_name)

    def get_column_dtype(self, column_name: str) -> str:
        """Get column data type"""
        self._check_connection()
        return self._storage.get_column_dtype(column_name)

    # ============ FTS Methods ============

    def search_text(self, query: str, table_name: str = None) -> Optional[np.ndarray]:
        """
        Execute full-text search (Rust native implementation, zero Python boundary overhead)
        
        Args:
            query: Search query string
            table_name: Table name (optional, defaults to current table)
        
        Returns:
            np.ndarray: Array of matching document IDs
        """
        table = table_name or self._current_table
        
        if not self._is_fts_enabled(table):
            raise ValueError(f"Full-text search is not enabled for table '{table}'. Call init_fts() first.")
        
        if not self._ensure_fts_initialized(table):
            return None
        
        # Switch table if needed
        if table_name and table_name != self._current_table:
            original = self._current_table
            self.use_table(table_name)
            result = self._storage._fts_search(query)
            self.use_table(original)
            return result
        
        return self._storage._fts_search(query)

    def fuzzy_search_text(self, query: str, min_results: int = 1, table_name: str = None) -> Optional[np.ndarray]:
        """
        Execute fuzzy full-text search (Rust native implementation, zero Python boundary overhead)
        
        Args:
            query: Search query string (supports spelling errors)
            min_results: Minimum result count to trigger fuzzy search
            table_name: Table name (optional, defaults to current table)
        
        Returns:
            np.ndarray: Array of matching document IDs
        """
        table = table_name or self._current_table
        
        if not self._is_fts_enabled(table):
            raise ValueError(f"Full-text search is not enabled for table '{table}'. Call init_fts() first.")
        
        if not self._ensure_fts_initialized(table):
            return None
        
        # Switch table if needed
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
        Execute full-text search and return complete records (Rust native implementation, zero Python boundary overhead)
        
        This is the fastest search + retrieve path because:
        1. Search is completed at Rust layer (no Python boundary overhead)
        2. Retrieval is completed at Rust layer (no Python boundary overhead)
        3. Directly returns Arrow IPC bytes
        
        Args:
            query: Search query string
            table_name: Table name (optional, defaults to current table)
            limit: Result count limit
            offset: Result offset
        
        Returns:
            ResultView: Query result view supporting multiple output formats:
                - .to_arrow() -> pyarrow.Table (zero-copy, fastest)
                - .to_pandas() -> pandas.DataFrame
                - .to_polars() -> polars.DataFrame
                - .to_dict() -> List[dict]
        
        Example:
            >>> results = client.search_and_retrieve("Python")
            >>> arrow_table = results.to_arrow()  # Fastest
            >>> df = results.to_pandas()
        """
        table = table_name or self._current_table
        
        if not self._is_fts_enabled(table):
            raise ValueError(f"Full-text search is not enabled for table '{table}'. Call init_fts() first.")
        
        if not self._ensure_fts_initialized(table):
            return _empty_result_view()
        
        # Fast path: use default table name
        if table_name is None:
            table_name = self._current_table
        
        # Switch table if needed
        need_switch = table_name != self._current_table
        original_table = self._current_table if need_switch else None
        
        try:
            if need_switch:
                self.use_table(table_name)
            
            schema_ptr, array_ptr = self._storage._fts_search_and_retrieve_ffi(query, limit, offset)

            if schema_ptr == 0 and array_ptr == 0:
                return _empty_result_view()

            try:
                struct_array = pa.Array._import_from_c(array_ptr, schema_ptr)
                if isinstance(struct_array, pa.StructArray):
                    batch = pa.RecordBatch.from_struct_array(struct_array)
                    table = pa.Table.from_batches([batch])
                    return ResultView(table)
                raise RuntimeError("FFI import did not return StructArray")
            finally:
                self._storage._free_arrow_ffi(schema_ptr, array_ptr)
                
        finally:
            if need_switch and original_table is not None:
                self.use_table(original_table)

    def search_and_retrieve_top(self, query: str, n: int = 100, table_name: str = None) -> 'ResultView':
        """
        Execute full-text search and return top N complete records (Rust native implementation)
        
        This is a simplified version of search_and_retrieve, specifically for getting the top N results.
        
        Args:
            query: Search query string
            n: Maximum number of results to return
            table_name: Table name (optional, defaults to current table)
        
        Returns:
            ResultView: Query result view
        """
        return self.search_and_retrieve(query, table_name=table_name, limit=n, offset=0)

    def set_fts_fuzzy_config(self, threshold: float = 0.7, max_distance: int = 2, 
                             max_candidates: int = 20, table_name: str = None):
        """
        Set fuzzy search configuration
        
        Args:
            threshold: Similarity threshold (0.0-1.0), higher values are stricter
            max_distance: Maximum edit distance
            max_candidates: Maximum number of candidates
            table_name: Table name (optional, defaults to current table)
        """
        table = table_name or self._current_table
        
        if not self._is_fts_enabled(table):
            raise ValueError(f"Full-text search is not enabled for table '{table}'. Call init_fts() first.")
        
        if not self._ensure_fts_initialized(table):
            return
        
        if table_name and table_name != self._current_table:
            original = self._current_table
            self.use_table(table_name)
            self._storage._fts_set_fuzzy_config(threshold, max_distance, max_candidates)
            self.use_table(original)
        else:
            self._storage._fts_set_fuzzy_config(threshold, max_distance, max_candidates)

    def get_fts_stats(self, table_name: str = None) -> Dict:
        """
        Get FTS engine statistics
        
        Args:
            table_name: Table name (optional, defaults to current table)
        
        Returns:
            Dict: FTS engine statistics
        """
        table = table_name or self._current_table
        
        if not self._is_fts_enabled(table):
            return {'fts_enabled': False, 'table': table}
        
        if not self._storage._fts_is_initialized():
            return {'fts_enabled': True, 'engine_initialized': False, 'table': table}
        
        # Switch table if needed
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
        Compact FTS index, apply deletions and optimize storage
        
        Args:
            table_name: Table name (optional, defaults to current table)
        """
        table = table_name or self._current_table
        
        if not self._is_fts_enabled(table) or not self._storage._fts_is_initialized():
            return
        
        # Switch table if needed
        if table_name and table_name != self._current_table:
            original = self._current_table
            self.use_table(table_name)
            self._storage._fts_compact()
            self.use_table(original)
        else:
            self._storage._fts_compact()

    def warmup_fts_terms(self, terms: List[str], table_name: str = None) -> int:
        """
        Warm up FTS cache (effective in lazy loading mode)
        
        Args:
            terms: List of terms to warm up
            table_name: Table name (optional, defaults to current table)
        
        Returns:
            int: Number of successfully loaded terms
        """
        table = table_name or self._current_table
        
        if not self._is_fts_enabled(table) or not self._storage._fts_is_initialized():
            return 0
        
        # Switch table if needed
        if table_name and table_name != self._current_table:
            original = self._current_table
            self.use_table(table_name)
            result = self._storage._fts_warmup_terms(terms)
            self.use_table(original)
            return result
        
        return self._storage._fts_warmup_terms(terms)

    # ============ Lifecycle Management ============

    def _force_close(self):
        """Force close - ensure data persistence"""
        try:
            if hasattr(self, '_storage') and self._storage is not None:
                # First flush FTS index (only if dirty)
                if hasattr(self, '_fts_tables') and self._fts_tables and getattr(self, '_fts_dirty', False):
                    try:
                        self._storage._fts_flush()
                    except Exception:
                        pass
                    self._fts_dirty = False

                # Close (Rust close persists)
                self._storage.close()
                self._storage = None
        except Exception:
            pass
        self._is_closed = True

    def close(self):
        """Close database connection"""
        if self._is_closed:
            return
        
        try:
            if hasattr(self, '_storage') and self._storage is not None:
                # First flush FTS index (only if dirty)
                if self._fts_tables and getattr(self, '_fts_dirty', False):
                    try:
                        self._storage._fts_flush()
                    except Exception:
                        pass
                    self._fts_dirty = False

                # Close (Rust close persists)
                self._storage.close()
                self._storage = None
        finally:
            self._is_closed = True
            if self._auto_manage:
                _registry.unregister(str(self._db_path))

    @classmethod
    def create_clean(cls, dirpath=None, **kwargs):
        """Create completely new instance, force cleanup of previous data"""
        kwargs['drop_if_exists'] = True
        return cls(dirpath=dirpath, **kwargs)

    def __enter__(self):
        """
        Context manager entry - supports with statement
        
        Returns:
            ApexClient: Returns self instance, supports chain calls
            
        Example:
            >>> with ApexClient("./my_db") as client:
            ...     client.store({"name": "Alice"})
            ...     # Automatically calls close()
        """
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Context manager exit - automatically cleanup resources
        
        Regardless of whether an exception occurs, ensures:
        - FTS index is properly flushed
        - Data is persisted to disk (via flush())
        - Database connection is closed
        - Unregistered from global registry (if auto_manage is enabled)
        
        Note: ApexBase does not implement traditional transaction rollback mechanisms. If in the context
        an exception occurs, data operations before the exception are usually persisted, but the specific behavior
        depends on whether data had been flushed to disk when the exception occurred.
        
        Args:
            exc_type: Exception type (if any)
            exc_val: Exception value (if any)
            exc_tb: Exception traceback (if any)
            
        Returns:
            bool: Returns False, does not suppress exception, lets caller handle it
        """
        self.close()
        return False

    def __del__(self):
        if hasattr(self, '_is_closed') and not self._is_closed:
            self._force_close()

    def __repr__(self):
        return f"ApexClient(path='{self._dirpath}', table='{self._current_table}')"


# Exports
__all__ = ['ApexClient', 'ResultView', 'DurabilityLevel', '__version__', 'FTS_AVAILABLE', 'ARROW_AVAILABLE', 'POLARS_AVAILABLE']

