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
        
        # Initialize V3 storage engine
        self._storage = ApexStorage(str(self._db_path), drop_if_exists=drop_if_exists)
        self._connected = True
        self._lock = threading.RLock()
        
        self._current_table = "default"  # Match ApexClient's default table name
        self._batch_size = batch_size
        self._enable_cache = enable_cache
        self._cache_size = cache_size
        
        # FTS configuration
        self._fts_tables: Dict[str, Dict] = {}
        self._fts_dirty: bool = False
        
        self._prefer_arrow_format = prefer_arrow_format and ARROW_AVAILABLE
        self._registry = _registry

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
        return True
    
    def _check_connection(self):
        if self._is_closed or self._storage is None:
            raise RuntimeError("ApexClient connection has been closed, cannot perform operations.")

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

    def create_table(self, table_name: str):
        self._check_connection()
        with self._lock:
            try:
                self._storage.create_table(table_name)
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
        
        if self._current_table == table_name:
            self._current_table = "default"

    def list_tables(self) -> List[str]:
        self._check_connection()
        with self._lock:
            return self._storage.list_tables()

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
            
        finally:
            if need_switch and original_table is not None:
                self.use_table(original_table)
        
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
        
            # 2. PyArrow Table
            if ARROW_AVAILABLE and hasattr(data, 'schema'):
                records = data.to_pylist()
                self._store_batch(records)
                return
        
            # 3. Pandas DataFrame
            if ARROW_AVAILABLE and pd is not None and isinstance(data, pd.DataFrame):
                records = data.to_dict('records')
                self._store_batch(records)
                return
        
            # 4. Polars DataFrame
            if POLARS_AVAILABLE and pl is not None and hasattr(data, 'to_arrow'):
                records = data.to_dicts()
                self._store_batch(records)
                return
        
            # 5. Single record dict
            if isinstance(data, dict):
                self._storage.store(data)
                return
            
            # 6. List[dict]
            elif isinstance(data, list):
                if not data:
                    return
                self._store_batch(data)
                return
            else:
                raise ValueError("Data must be dict, list of dicts, Dict[str, list], pandas.DataFrame, polars.DataFrame, or pyarrow.Table")

    def _store_batch(self, records: List[dict]) -> None:
        if not records:
            return
        self._storage.store_batch(records)

    def _store_columnar(self, columns: Dict[str, list]) -> None:
        if not columns:
            return
        
        # Validate all columns have the same length
        lengths = {name: len(col) if hasattr(col, '__len__') else 0 for name, col in columns.items()}
        unique_lengths = set(lengths.values())
        if len(unique_lengths) > 1:
            raise ValueError(f"All columns must have the same length: {lengths}")
        
        # Convert columnar to list of dicts
        first_col = next(iter(columns.values()))
        n_rows = len(first_col) if hasattr(first_col, '__len__') else 0
        
        records = []
        for i in range(n_rows):
            record = {}
            for name, values in columns.items():
                if hasattr(values, 'dtype'):  # numpy array
                    record[name] = values[i].item() if hasattr(values[i], 'item') else values[i]
                else:
                    record[name] = values[i]
            records.append(record)
        
        self._store_batch(records)

    # ============ Query Operations ============

    def execute(self, sql: str, show_internal_id: bool = None) -> 'ResultView':
        self._check_connection()
        with self._lock:
            # Validate table name if FROM clause exists
            self._validate_table_in_sql(sql)
            
            # Determine if _id should be shown based on SQL (like ApexClient)
            if show_internal_id is None:
                show_internal_id = self._should_show_internal_id(sql)
            
            # ApexStorage.execute returns a dict with columns and rows
            result = self._storage.execute(sql)
            
            if result is None:
                return _empty_result_view()
            
            # Convert dict result to Arrow table
            columns = result.get('columns', [])
            rows = result.get('rows', [])
            
            if not rows:
                return _empty_result_view()
            
            # Build dict of lists for Arrow table
            col_data = {col: [] for col in columns}
            for row in rows:
                for i, col in enumerate(columns):
                    col_data[col].append(row[i] if i < len(row) else None)
            
            table = pa.Table.from_pydict(col_data)
            rv = ResultView(arrow_table=table, data=None)
            rv._show_internal_id = show_internal_id
            return rv
    
    def _validate_table_in_sql(self, sql: str) -> None:
        """Validate that table names in SQL exist"""
        import re
        
        # Extract table name from FROM clause
        m = re.search(r"\bfrom\s+(\w+)", sql, flags=re.IGNORECASE)
        if not m:
            return
        
        table_name = m.group(1).lower()
        
        # Get available tables
        available_tables = set()
        available_tables.add('default')
        available_tables.add(self._current_table.lower())
        
        # Check table_paths for other tables
        if hasattr(self._storage, 'list_tables'):
            try:
                for t in self._storage.list_tables():
                    available_tables.add(t.lower())
            except Exception:
                pass
        
        # Also check for .apex files in directory
        import os
        if os.path.isdir(self._dirpath):
            for f in os.listdir(self._dirpath):
                if f.endswith('.apex'):
                    available_tables.add(f[:-5].lower())
        
        if table_name not in available_tables:
            raise ValueError(f"Table '{m.group(1)}' not found")
    
    def _should_show_internal_id(self, sql: str) -> bool:
        """Determine if _id should be visible based on SQL (mirrors ApexClient logic)"""
        import re
        
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
        with self._lock:
            return self._storage.retrieve(id_)

    def retrieve_many(self, ids: List[int]) -> 'ResultView':
        self._check_connection()
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
        with self._lock:
            results = self._storage.retrieve_all()
        if not results:
            return _empty_result_view()
        
        table = pa.Table.from_pylist(results)
        return ResultView(arrow_table=table)

    def list_fields(self) -> List[str]:
        self._check_connection()
        with self._lock:
            return self._storage.list_fields()

    # ============ Delete/Replace ============

    def delete(self, ids: Union[int, List[int]]) -> bool:
        self._check_connection()
        with self._lock:
            # Remove from FTS index if enabled
            if self._fts_tables:
                ids_to_remove = [ids] if isinstance(ids, int) else ids
                for doc_id in ids_to_remove:
                    self._storage._fts_remove(doc_id)
            
            if isinstance(ids, int):
                return self._storage.delete(ids)
            elif isinstance(ids, list):
                return self._storage.delete_batch(ids)
            else:
                raise ValueError("ids must be an int or a list of ints")

    def replace(self, id_: int, data: dict) -> bool:
        self._check_connection()
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

    def from_pandas(self, df) -> 'ApexClient':
        records = df.to_dict('records')
        self.store(records)
        return self

    def from_pyarrow(self, table) -> 'ApexClient':
        records = table.to_pylist()
        self.store(records)
        return self

    def from_polars(self, df) -> 'ApexClient':
        records = df.to_dicts()
        self.store(records)
        return self

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
                self.use_table(original)
                return count
            return self._storage.row_count()

    def flush(self) -> None:
        self._check_connection()
        with self._lock:
            self._storage.flush()
    
    def flush_cache(self):
        self.flush()

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

    # ============ FTS Search ============

    def search_text(self, query: str, table_name: str = None) -> Optional[np.ndarray]:
        table = table_name or self._current_table
        
        if not self._is_fts_enabled(table):
            raise ValueError(f"Full-text search is not enabled for table '{table}'. Call init_fts() first.")
        
        results = self._storage.search_text(query, limit=1000)
        if not results:
            return np.array([], dtype=np.int64)
        
        return np.array([r[0] for r in results], dtype=np.int64)

    def fuzzy_search_text(self, query: str, min_results: int = 1, table_name: str = None) -> Optional[np.ndarray]:
        table = table_name or self._current_table
        
        if not self._is_fts_enabled(table):
            raise ValueError(f"Full-text search is not enabled for table '{table}'. Call init_fts() first.")
        
        results = self._storage.fuzzy_search_text(query, limit=1000)
        if not results:
            return np.array([], dtype=np.int64)
        
        return np.array([r[0] for r in results], dtype=np.int64)

    def search_and_retrieve(self, query: str, table_name: str = None, 
                           limit: Optional[int] = None, offset: int = 0) -> 'ResultView':
        target_table = table_name or self._current_table
        
        if not self._is_fts_enabled(target_table):
            raise ValueError(f"Full-text search is not enabled for table '{target_table}'. Call init_fts() first.")
        
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
        return self.search_and_retrieve(query, table_name=table_name, limit=n, offset=0)

    def set_fts_fuzzy_config(self, threshold: float = 0.7, max_distance: int = 2, 
                             max_candidates: int = 20, table_name: str = None):
        pass  # ApexStorage doesn't expose this yet

    def get_fts_stats(self, table_name: str = None) -> Dict:
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
        pass  # ApexStorage doesn't expose this yet

    def warmup_fts_terms(self, terms: List[str], table_name: str = None) -> int:
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
