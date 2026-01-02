"""
ApexBase - 基于 Rust 核心的高性能嵌入式数据库

使用自定义单文件存储格式（.apex），提供高效的数据存储和查询功能。
"""

import shutil
import weakref
import atexit
from typing import List, Dict, Union, Optional, Literal
from pathlib import Path
import numpy as np

# 导入 Rust 核心
from apexbase._core import ApexStorage as RustStorage, __version__ as _core_version

# FTS 现在直接在 Rust 层实现，无需 Python nanofts 包
# 但保留兼容性标志
FTS_AVAILABLE = True  # 总是可用，因为已经集成到 Rust 核心

# 可选的数据框架支持
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
    """全局实例注册表"""
    
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
    """查询结果视图 - Arrow-first 高性能实现"""
    
    def __init__(self, arrow_table=None, data=None):
        """
        初始化 ResultView（Arrow-first 模式）
        
        Args:
            arrow_table: PyArrow Table（主要数据源，最快）
            data: List[dict] 数据（可选，用于回退）
        """
        self._arrow_table = arrow_table
        self._data = data  # 懒加载，从 Arrow 转换
        self._num_rows = arrow_table.num_rows if arrow_table is not None else (len(data) if data else 0)
    
    @classmethod
    def from_arrow_bytes(cls, arrow_bytes: bytes) -> 'ResultView':
        """从 Arrow IPC bytes 创建（最快路径）"""
        if not arrow_bytes or not ARROW_AVAILABLE:
            return cls(data=[])
        reader = pa.ipc.open_stream(arrow_bytes)
        table = reader.read_all()
        return cls(arrow_table=table)
    
    @classmethod
    def from_dicts(cls, data: List[dict]) -> 'ResultView':
        """从字典列表创建（回退路径）"""
        return cls(data=data)
    
    def _ensure_data(self):
        """确保 _data 可用（懒加载从 Arrow 转换）"""
        if self._data is None and self._arrow_table is not None:
            self._data = self._arrow_table.to_pylist()
        return self._data if self._data is not None else []
    
    def to_dict(self) -> List[dict]:
        """返回字典列表"""
        return self._ensure_data()
    
    def to_pandas(self, zero_copy: bool = True):
        """
        返回 Pandas DataFrame
        
        Parameters:
            zero_copy: bool, default True
                如果为 True，使用 ArrowDtype 实现零拷贝转换（pandas 2.0+）
                如果为 False，使用传统转换方式（复制数据到 NumPy）
        
        Returns:
            pandas.DataFrame
        
        Note:
            零拷贝模式下，DataFrame 列使用 Arrow 原生类型（如 string[pyarrow]）
            这在大多数场景下性能更好，但某些 NumPy 操作可能需要先转换类型
        """
        if not ARROW_AVAILABLE:
            raise ImportError("pandas not available. Install with: pip install pandas")
        
        if self._arrow_table is not None:
            if zero_copy:
                # 零拷贝模式：使用 ArrowDtype（pandas 2.0+）
                try:
                    df = self._arrow_table.to_pandas(types_mapper=pd.ArrowDtype)
                except (TypeError, AttributeError):
                    # 回退：pandas < 2.0 不支持 ArrowDtype
                    df = self._arrow_table.to_pandas()
            else:
                # 传统模式：复制数据到 NumPy 类型
                df = self._arrow_table.to_pandas()
            
            if '_id' in df.columns:
                df.set_index('_id', inplace=True)
                df.index.name = None
            return df
        
        # 回退
        df = pd.DataFrame(self._ensure_data())
        if '_id' in df.columns:
            df.set_index('_id', inplace=True)
            df.index.name = None
        return df
    
    def to_polars(self):
        """返回 Polars DataFrame（直接从 Arrow，最快）"""
        if not POLARS_AVAILABLE:
            raise ImportError("polars not available. Install with: pip install polars")
        
        if self._arrow_table is not None:
            return pl.from_arrow(self._arrow_table)
        return pl.DataFrame(self._ensure_data())
    
    def to_arrow(self):
        """返回 PyArrow Table（零拷贝，最快）"""
        if not ARROW_AVAILABLE:
            raise ImportError("pyarrow not available. Install with: pip install pyarrow")
        
        if self._arrow_table is not None:
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
        if self._arrow_table is not None and '_id' in self._arrow_table.column_names:
            return self._arrow_table.column('_id').to_pylist()
        return [r.get('_id') for r in self._ensure_data() if '_id' in r]
    
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
    ApexBase 客户端 - 基于 Rust 核心的高性能嵌入式数据库
    
    特点:
    - 自定义单文件存储格式 (.apex)
    - 高性能批量写入
    - 支持全文搜索 (NanoFTS)
    - 与 Pandas、Polars、PyArrow 集成
    - 可配置的持久化强度 (durability)
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
        初始化 ApexClient
        
        Parameters:
            dirpath: str
                数据存储目录路径，如果为 None，使用当前目录
            batch_size: int
                批量操作的大小
            drop_if_exists: bool
                如果为 True，删除已存在的数据库文件
            enable_cache: bool
                是否启用写入缓存
            cache_size: int
                缓存大小
            prefer_arrow_format: bool
                是否优先使用 Arrow 格式
            durability: Literal['fast', 'safe', 'max']
                持久化强度级别：
                - 'fast': 最高性能，数据先写入内存缓冲区，flush() 时才持久化
                          适合批量导入、可重建数据、对性能要求极高的场景
                - 'safe': 平衡模式，每次 flush() 确保数据完全落盘 (fsync)
                          适合大多数生产环境
                - 'max': 最强 ACID 保证，每次写入都立即 fsync
                         适合金融、订单等关键数据场景
        
        Note:
            FTS（全文搜索）功能需要在连接后通过 init_fts() 方法单独初始化。
            这样可以更灵活地配置每个表的 FTS 设置。
            
            ApexClient 支持上下文管理器，推荐使用 with 语句自动管理资源：
            
            >>> # 基本用法
            >>> with ApexClient("./my_db") as client:
            ...     client.store({"name": "Alice", "age": 25})
            ...     # 自动提交和关闭连接
            ... 
            >>> # 链式调用
            >>> with ApexClient("./my_db").init_fts(index_fields=['name']) as client:
            ...     client.store({"name": "Bob"})
            ...     # 自动关闭 FTS 索引和数据库连接
        """
        if dirpath is None:
            dirpath = "."
        
        self._dirpath = Path(dirpath)
        self._dirpath.mkdir(parents=True, exist_ok=True)
        
        # 使用 .apex 文件格式
        self._db_path = self._dirpath / "apexbase.apex"
        self._auto_manage = _auto_manage
        self._is_closed = False
        
        # 注册到全局注册表
        if self._auto_manage:
            _registry.register(self, str(self._db_path))
        
        # 处理 drop_if_exists
        if drop_if_exists and self._db_path.exists():
            self._db_path.unlink()
            # 同时清理 FTS 索引
            fts_dir = self._dirpath / "fts_indexes"
            if fts_dir.exists():
                shutil.rmtree(fts_dir)
        
        # 验证 durability 参数
        if durability not in ('fast', 'safe', 'max'):
            raise ValueError(f"durability must be 'fast', 'safe', or 'max', got '{durability}'")
        self._durability = durability
        
        # 初始化 Rust 存储引擎，传入 durability 配置
        self._storage = RustStorage(str(self._db_path), durability=durability)
        
        self._current_table = "default"
        self._batch_size = batch_size
        self._enable_cache = enable_cache
        self._cache_size = cache_size
        
        # FTS 配置 - 每个表独立管理
        # key: table_name, value: {'enabled': bool, 'index_fields': List[str], 'config': Dict}
        self._fts_tables: Dict[str, Dict] = {}
        
        self._prefer_arrow_format = prefer_arrow_format and ARROW_AVAILABLE

    def _is_fts_enabled(self, table_name: str = None) -> bool:
        """检查指定表是否启用了 FTS"""
        table = table_name or self._current_table
        return table in self._fts_tables and self._fts_tables[table].get('enabled', False)
    
    def _get_fts_config(self, table_name: str = None) -> Optional[Dict]:
        """获取指定表的 FTS 配置"""
        table = table_name or self._current_table
        return self._fts_tables.get(table)
    
    def _ensure_fts_initialized(self, table_name: str = None) -> bool:
        """确保指定表的 FTS 已初始化"""
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
        初始化全文搜索 (FTS) 功能
        
        此方法必须在 ApexClient 正确连接后调用。可以为不同的表配置不同的 FTS 设置。
        
        Parameters:
            table_name: str, optional
                要启用 FTS 的表名。如果为 None，使用当前表。
            index_fields: List[str], optional
                要索引的字段列表。如果为 None，索引所有字符串字段。
            lazy_load: bool, default False
                是否启用懒加载模式。懒加载模式下，索引在首次查询时才会完全加载到内存。
            cache_size: int, default 10000
                FTS 缓存大小。
        
        Returns:
            ApexClient: 返回 self，支持链式调用。
        
        Raises:
            RuntimeError: 如果 ApexClient 未正确连接。
        
        Example:
            >>> # 基本用法 - 为当前表启用 FTS
            >>> client = ApexClient("./my_db")
            >>> client.init_fts(index_fields=['title', 'content'])
            
            >>> # 为特定表启用 FTS
            >>> client.init_fts(table_name='articles', index_fields=['title', 'body'])
            
            >>> # 链式调用
            >>> client = ApexClient("./my_db").init_fts(index_fields=['name', 'description'])
            
            >>> # 高级配置
            >>> client.init_fts(
            ...     table_name='documents',
            ...     index_fields=['content'],
            ...     lazy_load=True,
            ...     cache_size=50000
            ... )
        """
        self._check_connection()
        
        table = table_name or self._current_table
        
        # 如果需要切换表
        need_switch = table != self._current_table
        original_table = self._current_table if need_switch else None
        
        try:
            if need_switch:
                self.use_table(table)
            
            # 保存 FTS 配置
            self._fts_tables[table] = {
                'enabled': True,
                'index_fields': index_fields,
                'config': {
                    'lazy_load': lazy_load,
                    'cache_size': cache_size,
                }
            }
            
            # 初始化 Rust 原生 FTS
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
        """判断字段是否应该被索引"""
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
        """提取可索引的内容"""
        table = table_name or self._current_table
        
        if not self._is_fts_enabled(table):
            return {}
        
        indexable = {}
        for key, value in data.items():
            if self._should_index_field(key, value, table):
                indexable[key] = str(value)
        return indexable

    def _check_connection(self):
        """检查连接状态"""
        if self._is_closed:
            raise RuntimeError("ApexClient 连接已关闭，无法执行操作。请创建新的实例。")

    # ============ 公共 API ============

    def use_table(self, table_name: str):
        """切换当前表"""
        self._check_connection()
        self._storage.use_table(table_name)
        self._current_table = table_name
        # FTS 引擎在 Rust 层按需创建，无需在 Python 层管理

    @property
    def current_table(self) -> str:
        """获取当前表名"""
        return self._current_table

    def create_table(self, table_name: str):
        """创建新表并切换到该表"""
        self._check_connection()
        self._storage.create_table(table_name)
        self._current_table = table_name
        # FTS 引擎在 Rust 层按需创建

    def drop_table(self, table_name: str):
        """删除表"""
        self._check_connection()
        self._storage.drop_table(table_name)
        
        # FTS 索引文件会在 Rust 层被清理（如果需要）
        # 也可以手动清理
        if self._is_fts_enabled(table_name):
            fts_index_file = self._dirpath / "fts_indexes" / f"{table_name}.nfts"
            fts_wal_file = self._dirpath / "fts_indexes" / f"{table_name}.nfts.wal"
            if fts_index_file.exists():
                fts_index_file.unlink()
            if fts_wal_file.exists():
                fts_wal_file.unlink()
            # 移除 FTS 配置
            self._fts_tables.pop(table_name, None)
        
        if self._current_table == table_name:
            self._current_table = "default"

    def list_tables(self) -> List[str]:
        """列出所有表"""
        self._check_connection()
        return self._storage.list_tables()

    def store(self, data) -> None:
        """
        存储数据 - 自动选择最优策略，极速写入
        
        支持多种输入格式：
        - dict: 单条记录
        - List[dict]: 多条记录 (自动转换为列式高速路径)
        - Dict[str, list]: 列式数据 (最快路径)
        - Dict[str, np.ndarray]: numpy 列式数据 (零拷贝，最快)
        - pandas.DataFrame: 批量存储
        - polars.DataFrame: 批量存储
        - pyarrow.Table: 批量存储
        
        Parameters:
            data: 要存储的数据
        
        Performance (10,000 rows):
            - Dict[str, np.ndarray] 纯数值: ~0.1ms (90M rows/s)
            - Dict[str, list] 混合类型: ~0.7ms (14M rows/s)
            - List[dict]: ~4.8ms (2M rows/s)
        
        Example:
            >>> # 最快: numpy 列式
            >>> client.store({
            ...     'id': np.arange(10000, dtype=np.int64),
            ...     'score': np.random.random(10000),
            ... })
            
            >>> # 快: list 列式
            >>> client.store({
            ...     'name': ['Alice', 'Bob', 'Charlie'],
            ...     'age': [25, 30, 35],
            ... })
            
            >>> # 单行
            >>> client.store({'name': 'Alice', 'age': 25})
        """
        self._check_connection()
        
        # 1. 检测列式数据 Dict[str, list/ndarray] - 最快路径
        if isinstance(data, dict):
            first_value = next(iter(data.values()), None) if data else None
            # 检测 list, tuple, 或 numpy array
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
        
        # 5. 单条记录 dict
        if isinstance(data, dict):
            # 获取插入前的行数作为新记录的 ID
            doc_id = self._storage.count_rows()
            self._storage._store_single_no_return(data)
            
            # 更新 FTS 索引（使用 Rust 原生实现）
            if self._is_fts_enabled() and self._ensure_fts_initialized():
                indexable = self._extract_indexable_content(data)
                if indexable:
                    self._storage._fts_add_document(doc_id, indexable)
                    self._storage._fts_flush()
            return
            
        # 6. List[dict] - 自动转换为列式存储
        elif isinstance(data, list):
            if not data:
                return
            self._store_list_fast(data)
            return
        else:
            raise ValueError("Data must be dict, list of dicts, Dict[str, list], pandas.DataFrame, polars.DataFrame, or pyarrow.Table")

    def _store_list_fast(self, data: List[dict]) -> None:
        """内部方法：高速 list 存储 - 自动转为列式，无返回值"""
        if not data:
            return
        
        # 获取插入前的行数，用于计算插入后的 ID 范围
        start_id = self._storage.count_rows()
        
        # 转换为列式格式
        int_cols = {}
        float_cols = {}
        str_cols = {}
        bool_cols = {}
        bin_cols = {}
        
        # 从第一行确定列类型
        first_row = data[0]
        col_types = {}  # name -> type
        
        for name, value in first_row.items():
            if name == '_id':
                continue
            if isinstance(value, bool):  # bool 必须在 int 之前检查
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
                col_types[name] = 'str'  # 默认转字符串
                str_cols[name] = []
        
        # 收集所有数据
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
        
        # 使用不返回 IDs 的高速 API
        # 如果启用 FTS，传入索引字段名让 Rust 直接构建 FTS 文档 (零边界跨越!)
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

    def query(self, where: str = None) -> ResultView:
        """
        使用 SQL 语法查询记录
        
        Parameters:
            where: SQL WHERE 子句，如 "age > 25 AND city = 'NYC'"
        
        Returns:
            ResultView: 查询结果视图，支持 to_dict(), to_pandas(), to_polars(), to_arrow()
        """
        self._check_connection()
        
        where_clause = where if where and where.strip() != "1=1" else "1=1"
        
        # Arrow-first: 直接使用 query_arrow（最快路径）
        if ARROW_AVAILABLE:
            try:
                arrow_bytes = self._storage._query_arrow(where_clause)
                return ResultView.from_arrow_bytes(arrow_bytes)
            except Exception:
                pass  # 回退到传统方式
        
        # 回退：使用传统 dict 方式
        results = self._storage.query(where_clause)
        return ResultView.from_dicts(results)

    def retrieve(self, id_: int) -> Optional[dict]:
        """获取单个记录"""
        self._check_connection()
        return self._storage.retrieve(id_)

    def retrieve_many(self, ids: List[int]) -> List[dict]:
        """
        获取多个记录
        
        Parameters:
            ids: 要检索的记录 ID 列表
        
        Returns:
            List[dict]: 记录列表
        """
        self._check_connection()
        if not ids:
            return []
        
        # 优先使用 Arrow 传输（如果可用且数据量较大）
        if ARROW_AVAILABLE and len(ids) > 50:
            try:
                arrow_bytes = self._storage._retrieve_many_arrow(ids)
                if arrow_bytes:
                    reader = pa.ipc.open_stream(arrow_bytes)
                    table = reader.read_all()
                    return table.to_pylist()
            except Exception:
                pass  # 回退到传统方式
        
        return self._storage.retrieve_many(ids)

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
            
            # 全部在 Rust 层完成：搜索 + 分页 + 检索 + Arrow 序列化
            arrow_bytes = self._storage._fts_search_and_retrieve(query, limit, offset)
            
            if arrow_bytes and len(arrow_bytes) > 0 and ARROW_AVAILABLE:
                return ResultView.from_arrow_bytes(arrow_bytes)
            else:
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

