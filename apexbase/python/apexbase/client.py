"""
ApexClient - High-performance embedded database client

This module provides the ApexClient class that wraps ApexStorage with on-demand storage engine.
"""

import os
import re
import threading
import time
import json
import shutil
import tempfile
import weakref
import atexit
from typing import List, Dict, Union, Optional, Literal
from pathlib import Path
import numpy as np

from apexbase._core import ApexStorage
from . import ResultView, _empty_result_view, _registry, DurabilityLevel

import pyarrow as pa
import pandas as pd
import polars as pl

ARROW_AVAILABLE = True
POLARS_AVAILABLE = True

# Pre-compiled regex for SQL validation (avoids re-compilation on every query)
_RE_CREATE_TABLE = re.compile(r"\bcreate\s+(table|view)\b", re.IGNORECASE)
_RE_FROM_TABLE = re.compile(r"\bfrom\s+([\w]+(?:\.[\w]+)?)", re.IGNORECASE)



class ApexClient:
    """
    ApexClient - High-performance embedded database client
    
    Uses on-demand storage format (.apex) for persistence.
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
        if dirpath is None:
            dirpath = "."
        
        self._dirpath = Path(dirpath)
        self._dirpath.mkdir(parents=True, exist_ok=True)
        
        # Use .apex file format for V3 storage
        self._db_path = self._dirpath / "apexbase.apex"
        self._auto_manage = _auto_manage
        self._is_closed = False
        
        # Register to global registry
        if self._auto_manage:
            _registry.register(self, str(self._db_path))
        
        # Validate durability parameter
        if durability not in ('fast', 'safe', 'max'):
            raise ValueError(f"durability must be 'fast', 'safe', or 'max', got '{durability}'")
        self._durability = durability
        
        # Initialize V3 storage engine with durability level (best-effort across bindings)
        try:
            self._storage = ApexStorage(str(self._db_path), drop_if_exists=drop_if_exists, durability=durability)
        except TypeError:
            self._storage = ApexStorage(str(self._db_path), drop_if_exists=drop_if_exists)
        self._connected = True
        self._lock = threading.RLock()
        
        self._current_table = None  # No default table - user must create/use a table explicitly
        self._current_database = 'default'  # Active database name ('default' = root dir)
        self._batch_size = batch_size
        self._enable_cache = enable_cache
        self._cache_size = cache_size
        
        # FTS configuration
        self._fts_tables: Dict[str, Dict] = {}
        self._fts_dirty: bool = False

        # Persisted FTS configuration path
        self._fts_config_path = self._dirpath / "fts_config.json"

        # If recreating DB, clear any persisted FTS config
        if drop_if_exists:
            try:
                if self._fts_config_path.exists():
                    self._fts_config_path.unlink()
            except Exception:
                pass

        # Load persisted FTS config (if any)
        self._load_fts_config()
        
        self._prefer_arrow_format = prefer_arrow_format and ARROW_AVAILABLE
        self._registry = _registry
        self._query_cache = {}  # SQL -> ResultView cache for repeated read queries
        self._has_writes = False  # True after any write; disables _storage.execute() fast paths

    def _load_fts_config(self) -> None:
        try:
            if not self._fts_config_path.exists():
                return
            with open(self._fts_config_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            if isinstance(data, dict):
                # Only accept dict[str, dict] shape
                self._fts_tables = {str(k): v for k, v in data.items() if isinstance(v, dict)}
        except Exception:
            # Best-effort: if config is corrupted, ignore it
            self._fts_tables = {}

    def _save_fts_config(self) -> None:
        try:
            with open(self._fts_config_path, 'w', encoding='utf-8') as f:
                json.dump(self._fts_tables, f, ensure_ascii=False)
        except Exception:
            pass

    def _is_fts_enabled(self, table_name: str = None) -> bool:
        table = table_name or self._current_table
        return table in self._fts_tables and self._fts_tables[table].get('enabled', False)
    
    def _get_fts_config(self, table_name: str = None) -> Optional[Dict]:
        table = table_name or self._current_table
        return self._fts_tables.get(table)
    
    def _ensure_fts_initialized(self, table_name: str = None) -> bool:
        table = table_name or self._current_table
        if not self._is_fts_enabled(table):
            return False

        # Lazily initialize Rust FTS engine on first use, using persisted config
        try:
            if not self._storage._is_fts_enabled():
                fts_config = self._fts_tables.get(table, {})
                cfg = fts_config.get('config', {}) if isinstance(fts_config, dict) else {}
                index_fields = fts_config.get('index_fields') if isinstance(fts_config, dict) else None
                self._storage._init_fts(
                    index_fields=index_fields,
                    lazy_load=bool(cfg.get('lazy_load', False)),
                    cache_size=int(cfg.get('cache_size', 10000)),
                )
        except Exception:
            # If initialization fails, report as not initialized
            return False

        return True
    
    def _check_connection(self):
        if self._is_closed or self._storage is None:
            raise RuntimeError("ApexClient connection has been closed, cannot perform operations.")
    
    def _ensure_table_selected(self):
        if self._current_table is None:
            raise RuntimeError("No table selected. Call create_table() or use_table() first.")

    # ============ Database Management ============

    def use_database(self, database: str = 'default') -> 'ApexClient':
        """Switch to a named database. Creates it if it doesn't exist.

        'default' (or '') means the root directory — backward-compatible behaviour.
        Named databases (e.g. 'analytics') are stored in sub-directories of the
        root directory and each has its own isolated set of tables.

        Args:
            database: Database name. Use 'default' for the root-level tables.

        Returns:
            self (for method chaining)
        """
        self._check_connection()
        with self._lock:
            self._storage.use_database_(database)
            self._current_database = database if database else 'default'
            self._current_table = None
            self._query_cache.clear()
        return self

    def use(self, database: str = 'default', table: str = None) -> 'ApexClient':
        """Switch to a database and optionally select a table within it.

        Combines use_database() + use_table() / create_table() in one call.
        If the table does not exist in the target database it will be created.

        Args:
            database: Database name (default = root-level).
            table: Table name to switch to. If None only the database is switched.

        Returns:
            self (for method chaining)
        """
        self.use_database(database)
        if table is not None:
            with self._lock:
                try:
                    self.use_table(table)
                except Exception:
                    self.create_table(table)
        return self

    @property
    def current_database(self) -> str:
        """Return the currently active database name."""
        self._check_connection()
        return self._current_database

    def list_databases(self) -> list:
        """List all available databases.

        'default' is always included (represents root-level tables).
        Other entries are named sub-directories inside the root directory.

        Returns:
            Sorted list of database name strings.
        """
        self._check_connection()
        return self._storage.list_databases_()

    # ============ Table Management ============

    def use_table(self, table_name: str):
        self._check_connection()
        with self._lock:
            self._storage.use_table(table_name)
        self._current_table = table_name

    @property
    def current_table(self) -> str:
        self._check_connection()
        return self._current_table

    def create_table(self, table_name: str, schema: dict = None):
        """Create a new table.

        Args:
            table_name: Name of the table to create.
            schema: Optional dict mapping column names to type strings.
                    Pre-defining schema avoids type inference on the first insert,
                    providing a performance benefit for bulk loading.
                    Supported types: int8, int16, int32, int64, uint8, uint16,
                    uint32, uint64, float32, float64, bool, string, binary.
                    Example: {"name": "string", "age": "int64", "score": "float64"}
        """
        self._check_connection()
        with self._lock:
            try:
                self._storage.create_table(table_name, schema)
            except OSError as e:
                raise ValueError(str(e)) from e
        self._current_table = table_name

    def drop_table(self, table_name: str):
        self._check_connection()
        with self._lock:
            try:
                self._storage.drop_table(table_name)
            except (ValueError, RuntimeError):
                pass
        
        if table_name in self._fts_tables:
            self._fts_tables.pop(table_name, None)
            self._save_fts_config()

        # Best-effort: remove FTS index files for dropped table
        try:
            if table_name != self._current_table:
                original = self._current_table
                self.use_table(table_name)
                self._storage._fts_remove_engine(True)
                self.use_table(original)
            else:
                self._storage._fts_remove_engine(True)
        except Exception:
            pass

        # Best-effort Python-side cleanup in case the engine keeps files open
        try:
            fts_dir = self._dirpath / "fts_indexes"
            index_path = fts_dir / f"{table_name}.nfts"
            wal_path = fts_dir / f"{table_name}.nfts.wal"
            if index_path.exists():
                try:
                    index_path.unlink()
                except Exception:
                    pass
            if wal_path.exists():
                try:
                    wal_path.unlink()
                except Exception:
                    pass
        except Exception:
            pass
        
        if self._current_table == table_name:
            self._current_table = None

    def list_tables(self) -> List[str]:
        self._check_connection()
        with self._lock:
            return self._storage.list_tables()

    # ============ Compression ============

    def set_compression(self, compression: str) -> bool:
        """Set compression type for the current table.

        Only effective on empty tables (row_count == 0). Ignored if table
        already contains data. The setting persists across restarts.

        Args:
            compression: "none", "lz4", or "zstd".

        Returns:
            True if applied, False if the table is non-empty (no-op).

        Raises:
            ValueError: If *compression* is not a recognised algorithm name.
            RuntimeError: If no table is selected.
        """
        self._check_connection()
        self._ensure_table_selected()
        with self._lock:
            return self._storage.set_compression(compression)

    def get_compression(self) -> str:
        """Get the current compression type for the current table.

        Returns:
            "none", "lz4", or "zstd".
        """
        self._check_connection()
        self._ensure_table_selected()
        with self._lock:
            return self._storage.get_compression()

    # ============ FTS ============

    def init_fts(
        self,
        table_name: str = None,
        index_fields: Optional[List[str]] = None,
        lazy_load: bool = False,
        cache_size: int = 10000
    ) -> 'ApexClient':
        self._check_connection()
        
        table = table_name or self._current_table
        
        need_switch = table != self._current_table
        original_table = self._current_table if need_switch else None
        
        try:
            if need_switch:
                self.use_table(table)
            
            self._fts_tables[table] = {
                'enabled': True,
                'index_fields': index_fields,
                'config': {
                    'lazy_load': lazy_load,
                    'cache_size': cache_size,
                }
            }
            
            self._storage._init_fts(
                index_fields=index_fields,
                lazy_load=lazy_load,
                cache_size=cache_size
            )

            # Persist config so it auto-enables on reopen
            self._save_fts_config()
            
        finally:
            if need_switch and original_table is not None:
                self.use_table(original_table)
        
        return self

    def _fts_index_from_arrow(self, table: pa.Table, id_column: str = 'id', text_columns: List[str] = None) -> int:
        """Index FTS from an Arrow table using the Rust nanofts engine.
        
        Args:
            table: PyArrow Table with data
            id_column: Column to use as document ID (default 'id')
            text_columns: List of text columns to index (None = all string columns)
            
        Returns:
            Number of documents indexed
        """
        self._check_connection()
        table_name = self._current_table
        
        if not self._is_fts_enabled(table_name):
            raise ValueError(f"FTS not enabled for table '{table_name}'. Call init_fts() first.")
        
        fts_config = self._fts_tables.get(table_name, {})
        if text_columns is None:
            text_columns = fts_config.get('index_fields')
        
        if id_column not in table.column_names:
            id_column = table.column_names[0]
        
        # Determine text columns to index
        if text_columns:
            cols = [c for c in text_columns if c in table.column_names]
        else:
            import pyarrow as pa
            cols = [c for c in table.column_names if c != id_column
                    and pa.types.is_string(table.schema.field(c).type)]
        
        if not cols:
            return 0
        
        ids = table.column(id_column).to_pylist()
        columns = {c: [str(v) if v is not None else '' for v in table.column(c).to_pylist()] for c in cols}
        count = self._storage._fts_index_columns(ids, columns)
        self._storage._fts_flush()
        return count

    def _fts_index_from_pandas(self, df: pd.DataFrame, id_column: str = 'id', text_columns: List[str] = None) -> int:
        """Index FTS from a Pandas DataFrame using the Rust nanofts engine.
        
        Args:
            df: Pandas DataFrame with data
            id_column: Column to use as document ID (default 'id')
            text_columns: List of text columns to index (None = all string columns)
            
        Returns:
            Number of documents indexed
        """
        self._check_connection()
        table_name = self._current_table
        
        if not self._is_fts_enabled(table_name):
            raise ValueError(f"FTS not enabled for table '{table_name}'. Call init_fts() first.")
        
        fts_config = self._fts_tables.get(table_name, {})
        if text_columns is None:
            text_columns = fts_config.get('index_fields')
        
        if id_column not in df.columns:
            id_column = df.columns[0]
        
        # Determine text columns to index
        if text_columns:
            cols = [c for c in text_columns if c in df.columns]
        else:
            cols = [c for c in df.columns if c != id_column
                    and df[c].dtype == object]
        
        if not cols:
            return 0
        
        ids = df[id_column].tolist()
        columns = {c: df[c].fillna('').astype(str).tolist() for c in cols}
        count = self._storage._fts_index_columns(ids, columns)
        self._storage._fts_flush()
        return count

    def disable_fts(self, table_name: str = None) -> 'ApexClient':
        """Disable FTS for a table (keeps index files)."""
        self._check_connection()
        table = table_name or self._current_table

        cfg = self._fts_tables.get(table, {})
        if not isinstance(cfg, dict):
            cfg = {}

        cfg['enabled'] = False
        if 'config' not in cfg or not isinstance(cfg.get('config'), dict):
            cfg['config'] = {}
        self._fts_tables[table] = cfg
        self._save_fts_config()
        return self

    def drop_fts(self, table_name: str = None) -> 'ApexClient':
        """Drop FTS for a table: disable + delete index files."""
        self._check_connection()
        table = table_name or self._current_table

        # Keep config for initialization before deleting it
        prev_cfg = self._fts_tables.get(table)

        # Remove persisted config
        self._fts_tables.pop(table, None)
        self._save_fts_config()

        # Remove engine and index files in Rust layer
        try:
            # Ensure Rust manager exists; otherwise remove_engine is a no-op
            if not self._storage._is_fts_enabled():
                cfg = prev_cfg.get('config', {}) if isinstance(prev_cfg, dict) else {}
                index_fields = prev_cfg.get('index_fields') if isinstance(prev_cfg, dict) else None
                self._storage._init_fts(
                    index_fields=index_fields,
                    lazy_load=bool(cfg.get('lazy_load', False)),
                    cache_size=int(cfg.get('cache_size', 10000)),
                )

            if table_name and table_name != self._current_table:
                original = self._current_table
                self.use_table(table)
                self._storage._fts_remove_engine(True)
                self.use_table(original)
            else:
                self._storage._fts_remove_engine(True)
        except Exception:
            pass

        # Best-effort Python-side cleanup in case the engine keeps files open
        try:
            fts_dir = self._dirpath / "fts_indexes"
            index_path = fts_dir / f"{table}.nfts"
            wal_path = fts_dir / f"{table}.nfts.wal"
            if index_path.exists():
                try:
                    index_path.unlink()
                except Exception:
                    pass
            if wal_path.exists():
                try:
                    wal_path.unlink()
                except Exception:
                    pass
        except Exception:
            pass

        return self

    def _should_index_field(self, field_name: str, field_value, table_name: str = None) -> bool:
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
        table = table_name or self._current_table
        
        if not self._is_fts_enabled(table):
            return {}
        
        indexable = {}
        for key, value in data.items():
            if self._should_index_field(key, value, table):
                indexable[key] = str(value)
        return indexable

    # ============ Store Operations ============

    def store(self, data) -> None:
        self._check_connection()
        self._ensure_table_selected()
        self._query_cache.clear()  # Invalidate cache on write
        with self._lock:
            # 1. Columnar data Dict[str, list/ndarray]
            if isinstance(data, dict):
                first_value = next(iter(data.values()), None) if data else None
                if first_value is not None and (
                    isinstance(first_value, (list, tuple)) or 
                    hasattr(first_value, '__len__') and hasattr(first_value, 'dtype')
                ):
                    self._store_columnar(data)
                    return
        
            # 2. PyArrow Table - Convert to columnar dict for optimized storage
            if ARROW_AVAILABLE and hasattr(data, 'schema'):
                # Convert Arrow Table to columnar dict for zero-copy path
                columns = {}
                for name in data.column_names:
                    col = data[name]
                    # Convert to list for storage
                    if pa.types.is_string(col.type) or pa.types.is_large_string(col.type):
                        columns[name] = col.to_pylist()
                    elif pa.types.is_integer(col.type):
                        columns[name] = col.to_pylist()
                    elif pa.types.is_floating(col.type):
                        columns[name] = col.to_pylist()
                    elif pa.types.is_boolean(col.type):
                        columns[name] = col.to_pylist()
                    else:
                        columns[name] = col.to_pylist()
                self._store_columnar(columns)
                return
        
            # 3. Pandas DataFrame - Convert to columnar dict for optimized storage
            if ARROW_AVAILABLE and pd is not None and isinstance(data, pd.DataFrame):
                # Convert DataFrame to columnar dict
                columns = {}
                for name in data.columns:
                    col = data[name]
                    if col.dtype == 'object':
                        columns[name] = col.fillna('').tolist()
                    else:
                        columns[name] = col.tolist()
                self._store_columnar(columns)
                return
        
            # 4. Polars DataFrame - Convert to columnar dict for optimized storage
            if POLARS_AVAILABLE and pl is not None and hasattr(data, 'to_arrow'):
                # Convert to Arrow then to columnar dict
                arrow_table = data.to_arrow()
                columns = {}
                for name in arrow_table.column_names:
                    columns[name] = arrow_table[name].to_pylist()
                self._store_columnar(columns)
                return
        
            # 5. Single record dict
            if isinstance(data, dict):
                self._storage.store(data)
                return
            
            # 6. List[dict] - OPTIMIZED: Convert to columnar for better performance
            elif isinstance(data, list):
                if not data:
                    return
                # Auto-convert to columnar for batch processing (3x faster!)
                if len(data) > 1 and isinstance(data[0], dict):
                    self._store_batch_optimized(data)
                else:
                    self._store_batch(data)
                return
            else:
                raise ValueError("Data must be dict, list of dicts, Dict[str, list], pandas.DataFrame, polars.DataFrame, or pyarrow.Table")

    def _store_batch(self, records: List[dict]) -> None:
        if not records:
            return
        self._storage.store_batch(records)

    def _store_batch_optimized(self, records: List[dict]) -> None:
        """Store batch with automatic columnar conversion for 3x performance boost.
        
        This method automatically converts a list of dicts to columnar format,
        which is ~3x faster than row-by-row processing.
        
        Args:
            records: List of dict records to store
        """
        if not records:
            return
        
        # Convert to columnar format for optimal performance
        if records and isinstance(records[0], dict):
            keys = records[0].keys()
            columns = {key: [record.get(key) for record in records] for key in keys}
            self._store_columnar(columns)
        else:
            # Fallback to standard batch store
            self._storage.store_batch(records)

    def _store_columnar(self, columns: Dict[str, list]) -> None:
        if not columns:
            return
        
        # Convert numpy arrays to Python lists for Rust binding
        converted = {}
        for name, values in columns.items():
            if hasattr(values, 'tolist'):  # numpy array
                converted[name] = values.tolist()
            elif hasattr(values, 'to_list'):  # polars series
                converted[name] = values.to_list()
            else:
                converted[name] = list(values) if not isinstance(values, list) else values
        
        # Call native columnar storage - much faster than row-by-row
        self._storage.store_columnar(converted)

    # ============ Query Operations ============

    def execute(self, sql: str, show_internal_id: bool = None) -> 'ResultView':
        self._check_connection()
        # DDL (CREATE TABLE) is allowed without a table selected
        sql_upper = sql.strip().upper()

        # Detect multi-statement SQL: contains ';' with non-whitespace content after
        _trimmed = sql.strip().rstrip(';').strip()
        is_multi_stmt = ';' in _trimmed
        
        if not is_multi_stmt:
            # Allow execution without a selected table for DDL, CTE, or cross-database queries.
            # Cross-db queries use qualified db.table references (e.g. FROM default.users,
            # INSERT INTO analytics.events, UPDATE hr.employees, DELETE FROM default.logs).
            _qualified = re.search(r'\b\w+\.\w+\b', sql)
            _has_qualified_ref = bool(_qualified and '.' in _qualified.group(0))
            if not (sql_upper.startswith('CREATE ') or sql_upper.startswith('DROP TABLE')
                    or sql_upper.startswith('WITH ') or _has_qualified_ref):
                self._ensure_table_selected()
        else:
            # Multi-statement: only require table if we have one
            try:
                self._ensure_table_selected()
            except Exception:
                pass
        
        with self._lock:
            # Determine if _id should be shown based on SQL (like ApexClient)
            if show_internal_id is None:
                show_internal_id = self._should_show_internal_id(sql)
            
            # MULTI-STATEMENT PATH: route through Arrow IPC which handles
            # transactions (BEGIN/COMMIT/ROLLBACK) and cache invalidation in Rust
            if is_multi_stmt:
                ipc_bytes = self._storage._execute_arrow_ipc(sql)
                # Sync transaction state: check if BEGIN/COMMIT/ROLLBACK changed txn state
                su = sql_upper
                if 'BEGIN' in su or 'COMMIT' in su or 'ROLLBACK' in su:
                    # Determine final txn state from the SQL sequence
                    # Last txn command wins
                    for part in su.split(';'):
                        part = part.strip()
                        if part.startswith('BEGIN'):
                            self._in_txn = True
                        elif part in ('COMMIT', 'ROLLBACK') or part.startswith('COMMIT') or part == 'ROLLBACK':
                            self._in_txn = False
                if su.strip().rstrip(';').strip().startswith('CREATE TABLE'):
                    self._current_table = self._storage.current_table()
                reader = pa.ipc.open_stream(pa.BufferReader(ipc_bytes))
                table = reader.read_all()
                if table.num_rows == 0:
                    table = None
                rv = ResultView(arrow_table=table, data=None)
                rv._show_internal_id = show_internal_id
                return rv
            
            # FAST PATH: SELECT * LIMIT N (small N) — direct columnar transfer (no IPC)
            sql_up = sql.strip().upper()
            if (sql_up.startswith('SELECT *') and 'LIMIT' in sql_up
                    and 'WHERE' not in sql_up and 'ORDER' not in sql_up
                    and 'GROUP' not in sql_up and 'JOIN' not in sql_up):
                # Extract limit value — only use columnar path for small limits
                try:
                    limit_val = int(sql_up.rsplit('LIMIT', 1)[1].strip().rstrip(';'))
                except (ValueError, IndexError):
                    limit_val = 999999
                if limit_val <= 500:
                    try:
                        result = self._storage.execute(sql)
                        if result is not None:
                            columns_dict = result.get('columns_dict')
                            if columns_dict is None and 'columns' in result and 'rows' in result:
                                cols = result['columns']
                                rows = result['rows']
                                columns_dict = {c: [row[i] for row in rows] for i, c in enumerate(cols)}
                            if columns_dict is not None:
                                rv = ResultView(lazy_pydict=dict(columns_dict))
                                rv._show_internal_id = show_internal_id
                                if len(self._query_cache) > 200:
                                    self._query_cache.clear()
                                self._query_cache[sql] = rv
                                return rv
                    except Exception:
                        pass

            # FAST PATH: SELECT * WHERE col = 'val' — bypass pyarrow IPC for small results
            # Only safe when no writes have occurred (avoids stale data via different backend path)
            # Only for string equality (single-quoted value); skip numeric range / BETWEEN / IN
            if (not self._has_writes and sql_up.startswith('SELECT *') and 'WHERE' in sql_up
                    and 'LIMIT' not in sql_up and 'ORDER' not in sql_up
                    and 'GROUP' not in sql_up and 'JOIN' not in sql_up
                    and 'BETWEEN' not in sql_up and ' IN ' not in sql_up
                    and '>' not in sql_up and '<' not in sql_up
                    and "'" in sql):
                try:
                    result = self._storage.execute(sql)
                    if result is not None and 'columns' in result and 'rows' in result:
                        cols = result['columns']
                        rows = result['rows']
                        if not rows:
                            rv = ResultView(data=None)
                            rv._show_internal_id = show_internal_id
                            return rv
                        col_dict = {c: [row[i] for row in rows] for i, c in enumerate(cols)}
                        rv = ResultView(lazy_pydict=col_dict)
                        rv._show_internal_id = show_internal_id
                        if len(self._query_cache) > 200:
                            self._query_cache.clear()
                        self._query_cache[sql] = rv
                        return rv
                except Exception:
                    pass
            
            # Transaction commands: route through session-aware execute() binding
            is_txn_cmd = (sql_upper.startswith('BEGIN') or 
                         sql_upper in ('COMMIT', 'COMMIT;', 'ROLLBACK', 'ROLLBACK;') or
                         sql_upper.startswith('SAVEPOINT') or
                         sql_upper.startswith('RELEASE') or
                         sql_upper.startswith('ROLLBACK TO'))
            if is_txn_cmd:
                result = self._storage.execute(sql)
                # Track transaction state in Python client
                if sql_upper.startswith('BEGIN'):
                    self._in_txn = True
                elif sql_upper in ('COMMIT', 'COMMIT;', 'ROLLBACK', 'ROLLBACK;'):
                    self._in_txn = False
                rv = ResultView(data=None)
                rv._show_internal_id = show_internal_id
                return rv

            # Validate table name for non-fast-path queries (skip for DDL)
            if not (sql_upper.startswith('CREATE ') or sql_upper.startswith('DROP TABLE')):
                self._validate_table_in_sql(sql)
            
            # DML/SELECT within a transaction: route through session-aware execute() binding
            if getattr(self, '_in_txn', False) and sql_upper.startswith(('INSERT', 'DELETE', 'UPDATE', 'SELECT')):
                result = self._storage.execute(sql)
                if sql_upper.startswith('SELECT') and isinstance(result, dict) and 'columns' in result and 'rows' in result:
                    # Convert execute() dict result to columnar format for ResultView
                    cols = result['columns']
                    rows = result['rows']
                    if cols and rows:
                        col_dict = {c: [row[i] for row in rows] for i, c in enumerate(cols)}
                        rv = ResultView(lazy_pydict=col_dict)
                        rv._show_internal_id = show_internal_id
                        return rv
                rv = ResultView(data=None)
                rv._show_internal_id = show_internal_id
                return rv

            # Query result cache: return cached ResultView for identical read queries
            if sql_upper.startswith('SELECT'):
                cached = self._query_cache.get(sql)
                if cached is not None:
                    return cached
            elif sql_upper.startswith(('INSERT', 'DELETE', 'UPDATE', 'TRUNCATE', 'ALTER', 'DROP', 'CREATE')):
                self._query_cache.clear()
                self._has_writes = True
            
            # Standard path: Arrow IPC for efficient bulk transfer
            ipc_bytes = self._storage._execute_arrow_ipc(sql)
            
            # After CREATE TABLE, sync Python-side _current_table with Rust-side
            if sql_upper.startswith('CREATE TABLE'):
                self._current_table = self._storage.current_table()
            
            reader = pa.ipc.open_stream(pa.BufferReader(ipc_bytes))
            table = reader.read_all()
            
            if table.num_rows == 0:
                table = None
            
            rv = ResultView(arrow_table=table, data=None)
            rv._show_internal_id = show_internal_id
            
            # Cache the result for future identical read queries
            if sql_upper.startswith('SELECT'):
                if len(self._query_cache) > 200:
                    self._query_cache.clear()
                self._query_cache[sql] = rv
            
            return rv
    
    def _validate_table_in_sql(self, sql: str) -> None:
        """Validate that table names in SQL exist (skip for multi-statement SQL)"""
        # Skip validation for multi-statement SQL (contains CREATE TABLE/VIEW)
        if _RE_CREATE_TABLE.search(sql):
            return
        
        # Skip validation for CTE queries (WITH ... AS ...)
        if sql.strip().upper().startswith('WITH'):
            return
        
        # Extract table name from FROM clause
        m = _RE_FROM_TABLE.search(sql)
        if not m:
            return
        
        table_name = m.group(1).lower()

        # Skip validation for qualified db.table names (e.g. "default.users", "analytics.events")
        # The Rust executor resolves cross-database paths; we cannot validate them here.
        if '.' in table_name:
            return
        
        # Fast path: skip expensive list_tables/listdir for known tables
        if self._current_table and table_name == self._current_table.lower():
            return
        
        # Check .apex file exists directly (O(1) vs O(n) listdir)
        apex_path = os.path.join(self._dirpath, f"{table_name}.apex")
        if os.path.exists(apex_path):
            return
        
        raise ValueError(f"Table '{m.group(1)}' not found")
    
    def _should_show_internal_id(self, sql: str) -> bool:
        """Determine if _id should be visible based on SQL (mirrors ApexClient logic)"""
        # Fast path: if _id not mentioned at all, skip expensive regex
        if '_id' not in sql:
            return False
        
        # Check if _id is explicitly in SELECT clause
        m = re.search(r"\bselect\b(.*?)\bfrom\b", sql, flags=re.IGNORECASE | re.DOTALL)
        if not m:
            return False
        
        select_list = m.group(1)
        
        # Check for explicit _id reference (not in aggregate functions)
        def has_explicit_id(item: str) -> bool:
            s = item.strip()
            if re.search(r"\b(count|sum|avg|min|max)\s*\(", s, flags=re.IGNORECASE):
                return False
            return bool(re.search(r"(^|[^\w])(_id|\"_id\")([^\w]|$)|\._id([^\w]|$)", s, flags=re.IGNORECASE))
        
        # Split select items handling parentheses
        items = []
        buf = []
        depth = 0
        for ch in select_list:
            if ch == '(':
                depth += 1
            elif ch == ')':
                depth = max(0, depth - 1)
            elif ch == ',' and depth == 0:
                items.append(''.join(buf).strip())
                buf = []
                continue
            buf.append(ch)
        if buf:
            items.append(''.join(buf).strip())
        
        has_star = any(re.fullmatch(r"\*", it.strip()) for it in items)
        has_id = any(has_explicit_id(it) for it in items)
        
        # Show _id if explicitly referenced (and not just SELECT *)
        if has_id and not (len(items) == 1 and has_star):
            return True
        return False

    def query(self, sql: str = None, where_clause: str = None, limit: int = None) -> 'ResultView':
        """Query with SQL or WHERE clause (for ApexClient compatibility)"""
        self._ensure_table_selected()
        if sql is not None:
            # Check if it's a full SQL statement or a filter expression
            sql_upper = sql.strip().upper()
            if sql_upper.startswith("SELECT") or sql_upper.startswith("WITH"):
                # Full SQL statement
                return self.execute(sql)
            else:
                # Filter expression - convert to SELECT with WHERE
                full_sql = f"SELECT * FROM {self._current_table} WHERE {sql}"
                if limit:
                    full_sql += f" LIMIT {limit}"
                return self.execute(full_sql)
        elif where_clause is not None:
            full_sql = f"SELECT * FROM {self._current_table} WHERE {where_clause}"
            if limit:
                full_sql += f" LIMIT {limit}"
            return self.execute(full_sql)
        else:
            full_sql = f"SELECT * FROM {self._current_table}"
            if limit:
                full_sql += f" LIMIT {limit}"
            return self.execute(full_sql)

    def retrieve(self, id_: int) -> Optional[dict]:
        self._check_connection()
        self._ensure_table_selected()
        with self._lock:
            return self._storage.retrieve(id_)

    def retrieve_many(self, ids: List[int]) -> 'ResultView':
        self._check_connection()
        self._ensure_table_selected()
        with self._lock:
            if not ids:
                return _empty_result_view()
            
            results = self._storage.retrieve_many(ids)
            if not results:
                return _empty_result_view()
            
            # Reorder results to match the requested ID order
            id_to_record = {r.get('_id'): r for r in results}
            ordered_results = [id_to_record[id_] for id_ in ids if id_ in id_to_record]
            
            if not ordered_results:
                return _empty_result_view()
            
            # Convert list of dicts to Arrow table
            table = pa.Table.from_pylist(ordered_results)
            return ResultView(arrow_table=table)

    def retrieve_all(self) -> 'ResultView':
        self._check_connection()
        self._ensure_table_selected()
        with self._lock:
            results = self._storage.retrieve_all()
        if not results:
            return _empty_result_view()
        
        table = pa.Table.from_pylist(results)
        return ResultView(arrow_table=table)

    def list_fields(self) -> List[str]:
        self._check_connection()
        self._ensure_table_selected()
        with self._lock:
            return self._storage.list_fields()

    # ============ Delete/Replace ============

    def delete(
        self, 
        id: Optional[Union[int, List[int]]] = None, 
        where: Optional[str] = None
    ) -> Union[bool, int]:
        """Delete records by ID(s) or WHERE clause.
        
        Args:
            id: Single ID (int) or list of IDs to delete. Optional.
            where: SQL WHERE clause string for conditional deletion. Optional.
                   Example: "age > 30" or "status = 'inactive'"
        
        Returns:
            - If deleting by id: bool indicating success
            - If deleting by where: int count of deleted rows
        
        Raises:
            ValueError: If neither id nor where is provided (safety protection)
        
        Examples:
            client.delete(id=1)                    # Delete single record
            client.delete(id=[1, 2, 3])            # Delete multiple records
            client.delete(where="age > 30")        # Delete matching records
        """
        self._check_connection()
        self._ensure_table_selected()
        
        # Safety check: require at least one parameter to prevent accidental deletion of all data
        if id is None and where is None:
            raise ValueError(
                "delete() requires at least one argument: 'id' or 'where'. "
                "To delete all records, use delete(where='1=1') explicitly."
            )
        
        with self._lock:
            # Case 1: Delete by WHERE clause
            if where is not None:
                # Note: FTS cleanup for WHERE-based delete would require 
                # querying IDs first, which is expensive. Skip for now.
                return self._storage.delete_where(where)
            
            # Case 2: Delete by ID(s)
            if id is not None:
                # Remove from FTS index if enabled
                if self._fts_tables:
                    ids_to_remove = [id] if isinstance(id, int) else id
                    for doc_id in ids_to_remove:
                        self._storage._fts_remove(doc_id)
                
                if isinstance(id, int):
                    return self._storage.delete(id)
                elif isinstance(id, list):
                    return self._storage.delete_batch(id)
                else:
                    raise ValueError("id must be an int or a list of ints")

    def replace(self, id_: int, data: dict) -> bool:
        self._check_connection()
        self._ensure_table_selected()
        with self._lock:
            return self._storage.replace(id_, data)

    def batch_replace(self, data_dict: Dict[int, dict]) -> List[int]:
        self._check_connection()
        success_ids = []
        for id_, data in data_dict.items():
            if self.replace(id_, data):
                success_ids.append(id_)
        return success_ids

    # ============ DataFrame Import ============

    def from_pandas(self, df, table_name: str = None) -> 'ApexClient':
        if table_name is not None:
            self._select_or_create_table(table_name)
        self._ensure_table_selected()
        records = df.to_dict('records')
        self.store(records)
        return self

    def from_pyarrow(self, table, table_name: str = None) -> 'ApexClient':
        if table_name is not None:
            self._select_or_create_table(table_name)
        self._ensure_table_selected()
        records = table.to_pylist()
        self.store(records)
        return self

    def from_polars(self, df, table_name: str = None) -> 'ApexClient':
        if table_name is not None:
            self._select_or_create_table(table_name)
        self._ensure_table_selected()
        records = df.to_dicts()
        self.store(records)
        return self

    def _select_or_create_table(self, table_name: str):
        """Select an existing table or create a new one."""
        try:
            self.use_table(table_name)
        except (ValueError, RuntimeError):
            self.create_table(table_name)

    # ============ Utility ============

    def optimize(self):
        self._check_connection()
        # ApexStorage doesn't have optimize, just flush
        self.flush()

    def count_rows(self, table_name: str = None) -> int:
        self._check_connection()
        with self._lock:
            if table_name and table_name != self._current_table:
                original = self._current_table
                self.use_table(table_name)
                count = self._storage.row_count()
                if original is not None:
                    self.use_table(original)
                return count
            self._ensure_table_selected()
            return self._storage.row_count()

    def flush(self) -> None:
        self._check_connection()
        with self._lock:
            self._storage.flush()
    
    def flush_cache(self):
        self.flush()
    
    def set_auto_flush(self, rows: int = 0, bytes: int = 0) -> None:
        """Set auto-flush thresholds.
        
        When either threshold is exceeded during writes, data is automatically 
        written to file. Set to 0 to disable the respective threshold.
        
        Args:
            rows: Auto-flush when pending rows exceed this count (0 = disabled)
            bytes: Auto-flush when estimated memory exceeds this size (0 = disabled)
        """
        self._check_connection()
        with self._lock:
            self._storage.set_auto_flush(rows=rows, bytes=bytes)
    
    def get_auto_flush(self) -> tuple:
        """Get current auto-flush configuration.
        
        Returns:
            Tuple of (rows_threshold, bytes_threshold)
        """
        self._check_connection()
        with self._lock:
            return self._storage.get_auto_flush()
    
    def estimate_memory_bytes(self) -> int:
        """Get estimated memory usage in bytes."""
        self._check_connection()
        with self._lock:
            return self._storage.estimate_memory_bytes()

    # ============ Column Operations ============

    def drop_column(self, column_name: str):
        self._check_connection()
        if column_name == '_id':
            raise ValueError("Cannot drop _id column")
        self._storage.drop_column(column_name)

    def add_column(self, column_name: str, column_type: str):
        self._check_connection()
        self._storage.add_column(column_name, column_type)

    def rename_column(self, old_column_name: str, new_column_name: str):
        self._check_connection()
        if old_column_name == '_id':
            raise ValueError("Cannot rename _id column")
        self._storage.rename_column(old_column_name, new_column_name)

    def get_column_dtype(self, column_name: str) -> str:
        self._check_connection()
        return self._storage.get_column_dtype(column_name)

    # ============ FTS Search ==========

    def search_text(self, query: str, table_name: str = None) -> Optional[np.ndarray]:
        self._check_connection()
        table = table_name or self._current_table
        
        if not self._is_fts_enabled(table):
            raise ValueError(f"Full-text search is not enabled for table '{table}'. Call init_fts() first.")

        if not self._ensure_fts_initialized(table):
            return np.array([], dtype=np.int64)
        
        results = self._storage.search_text(query, limit=1000)
        if results is None:
            return np.array([], dtype=np.int64)
        if not results:
            return np.array([], dtype=np.int64)
        
        return np.array([r[0] for r in results], dtype=np.int64)

    def fuzzy_search_text(self, query: str, min_results: int = 1, table_name: str = None) -> Optional[np.ndarray]:
        self._check_connection()
        table = table_name or self._current_table
        
        if not self._is_fts_enabled(table):
            raise ValueError(f"Full-text search is not enabled for table '{table}'. Call init_fts() first.")

        if not self._ensure_fts_initialized(table):
            return np.array([], dtype=np.int64)
        
        results = self._storage.fuzzy_search_text(query, limit=1000)
        if not results:
            return np.array([], dtype=np.int64)
        
        return np.array([r[0] for r in results], dtype=np.int64)

    def search_and_retrieve(self, query: str, table_name: str = None, 
                           limit: Optional[int] = None, offset: int = 0) -> 'ResultView':
        self._check_connection()
        target_table = table_name or self._current_table
        
        if not self._is_fts_enabled(target_table):
            raise ValueError(f"Full-text search is not enabled for table '{target_table}'. Call init_fts() first.")

        if not self._ensure_fts_initialized(target_table):
            return _empty_result_view()
        
        # Switch to target table for search
        old_table = self._current_table
        if target_table != old_table:
            self.use_table(target_table)
        
        try:
            results = self._storage.search_and_retrieve(query, limit=limit)
            if not results:
                return _empty_result_view()
            
            table = pa.Table.from_pylist(results)
            return ResultView(arrow_table=table)
        finally:
            # Restore original table
            if target_table != old_table:
                self.use_table(old_table)

    def search_and_retrieve_top(self, query: str, n: int = 100, table_name: str = None) -> 'ResultView':
        self._check_connection()
        return self.search_and_retrieve(query, table_name=table_name, limit=n, offset=0)

    def set_fts_fuzzy_config(self, threshold: float = 0.7, max_distance: int = 2, 
                             max_candidates: int = 20, table_name: str = None):
        self._check_connection()
        pass  # ApexStorage doesn't expose this yet

    def get_fts_stats(self, table_name: str = None) -> Dict:
        self._check_connection()
        table = table_name or self._current_table
        
        if not self._is_fts_enabled(table):
            return {'fts_enabled': False, 'table': table}
        
        stats = self._storage.get_fts_stats()
        if stats:
            return {
                'fts_enabled': True,
                'engine_initialized': True,
                'doc_count': stats[0],
                'term_count': stats[1]
            }
        return {'fts_enabled': True, 'engine_initialized': False, 'table': table}

    def compact_fts_index(self, table_name: str = None):
        self._check_connection()
        pass  # ApexStorage doesn't expose this yet

    def warmup_fts_terms(self, terms: List[str], table_name: str = None) -> int:
        self._check_connection()
        return 0  # ApexStorage doesn't expose this yet

    # ============ Lifecycle ============

    def _force_close(self):
        try:
            if hasattr(self, '_storage') and self._storage is not None:
                self._storage.close()
                self._storage = None
        except Exception:
            pass
        self._is_closed = True

    def close(self):
        if self._is_closed:
            return
        
        try:
            if hasattr(self, '_storage') and self._storage is not None:
                # Best-effort: ensure FTS index is persisted across reopen
                try:
                    if any((isinstance(v, dict) and v.get('enabled', False)) for v in self._fts_tables.values()):
                        self._storage._fts_flush()
                except Exception:
                    pass
                self._storage.close()
                self._storage = None
        finally:
            self._is_closed = True
            if self._auto_manage:
                _registry.unregister(str(self._db_path))

    @classmethod
    def create_clean(cls, dirpath=None, **kwargs):
        kwargs['drop_if_exists'] = True
        return cls(dirpath=dirpath, **kwargs)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False

    def __del__(self):
        if hasattr(self, '_is_closed') and not self._is_closed:
            self._force_close()

    def __repr__(self):
        return f"ApexClient(path='{self._dirpath}', table='{self._current_table}')"
