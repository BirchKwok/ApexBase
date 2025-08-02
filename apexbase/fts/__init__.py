"""
基于 NanoFTS 的全文搜索模块
保持与原有 API 的兼容性
"""

from typing import List, Union, Dict
from nanofts import FullTextSearch as NanoFTS


class FullTextSearch:
    """
    全文搜索器的兼容性包装类，基于 NanoFTS 库实现
    
    保持与原有 API 完全兼容的同时提供更强的功能
    """
    
    def __init__(self, 
                 index_dir: str = None, 
                 max_chinese_length: int = 4, 
                 num_workers: int = 4,
                 shard_size: int = 100_000,
                 min_term_length: int = 2,
                 auto_save: bool = True,
                 batch_size: int = 1000,
                 drop_if_exists: bool = False,
                 buffer_size: int = 10000,
                 # 新增 NanoFTS 特有的参数
                 fuzzy_threshold: float = 0.6,
                 fuzzy_max_distance: int = 2):
        """
        初始化全文搜索器
        
        Args:
            index_dir: 索引文件存储目录，如果为None则使用内存索引
            max_chinese_length: 中文子串的最大长度，默认为4个字符
            num_workers: 并行构建索引的工作进程数，默认为4
            shard_size: 每个分片包含的文档数，默认10万
            min_term_length: 最小词长度，小于此长度的词不会被索引
            auto_save: 是否自动保存到磁盘，默认为True
            batch_size: 批量处理大小，达到此数量时才更新词组索引和保存，默认1000
            drop_if_exists: 如果索引文件存在，是否删除，默认为False
            buffer_size: 内存缓冲区大小，达到此大小时才写入磁盘，默认10000
            fuzzy_threshold: 模糊搜索相似度阈值 (0.0-1.0)，默认0.6
            fuzzy_max_distance: 模糊搜索最大编辑距离，默认2
        """
        # 保存参数用于兼容性
        self.index_dir = index_dir
        self.max_chinese_length = max_chinese_length
        self.num_workers = num_workers
        self.shard_size = shard_size
        self.min_term_length = min_term_length
        self.auto_save = auto_save
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        self.drop_if_exists = drop_if_exists
        
        # 创建 NanoFTS 实例
        self._fts = NanoFTS(
            index_dir=index_dir,
            max_chinese_length=max_chinese_length,
            num_workers=num_workers,
            shard_size=shard_size,
            min_term_length=min_term_length,
            auto_save=auto_save,
            batch_size=batch_size,
            buffer_size=buffer_size,
            drop_if_exists=drop_if_exists,
            fuzzy_threshold=fuzzy_threshold,
            fuzzy_max_distance=fuzzy_max_distance
        )
    
    def add_document(self, 
                     doc_id: Union[int, List[int]], 
                     fields: Union[Dict[str, Union[str, int, float]], List[Dict[str, Union[str, int, float]]]]):
        """
        添加文档到索引。支持单条文档和批量文档插入
        
        Args:
            doc_id: 文档ID，可以是单个整数或整数列表
            fields: 文档字段，可以是单个字典或字典列表。每个字典的值可以是字符串、整数或浮点数
        """
        return self._fts.add_document(doc_id, fields)
    
    def search(self, query: str) -> List[int]:
        """
        搜索文档
        
        Args:
            query: 搜索查询字符串
            
        Returns:
            包含匹配文档ID的列表
        """
        result = self._fts.search(query)
        # 确保返回格式为 List[int]，兼容原有API
        if hasattr(result, '__iter__') and not isinstance(result, (str, bytes)):
            return sorted(list(result))
        return []
    
    def remove_document(self, doc_id: Union[int, List[int]]):
        """
        从索引中删除文档
        
        Args:
            doc_id: 要删除的文档ID，可以是单个ID或ID列表
        """
        return self._fts.remove_document(doc_id)
    
    def flush(self):
        """
        强制将当前的更改保存到磁盘，并更新词组索引
        在批量添加完成后调用此方法以确保所有更改都已保存
        """
        return self._fts.flush()
    
    # 新增方法，利用 NanoFTS 的额外功能
    def update_document(self, 
                        doc_id: Union[int, List[int]], 
                        fields: Union[Dict[str, Union[str, int, float]], List[Dict[str, Union[str, int, float]]]]):
        """
        更新现有文档
        
        Args:
            doc_id: 文档ID，可以是单个整数或整数列表
            fields: 文档字段，可以是单个字典或字典列表
        """
        return self._fts.update_document(doc_id, fields)
    
    def fuzzy_search(self, query: str, min_results: int = 1) -> List[int]:
        """
        执行模糊搜索，可以处理拼写错误和相似词
        
        Args:
            query: 搜索查询字符串
            min_results: 最少结果数，如果精确搜索结果不足会启用模糊匹配
            
        Returns:
            包含匹配文档ID的列表
        """
        result = self._fts.fuzzy_search(query, min_results=min_results)
        # 确保返回格式为 List[int]，兼容原有API
        if hasattr(result, '__iter__') and not isinstance(result, (str, bytes)):
            return sorted(list(result))
        return []
    
    def get_fuzzy_config(self) -> Dict[str, Union[float, int]]:
        """
        获取当前模糊搜索配置
        
        Returns:
            包含 fuzzy_threshold 和 fuzzy_max_distance 的字典
        """
        return self._fts.get_fuzzy_config()
    
    def set_fuzzy_config(self, fuzzy_threshold: float = None, fuzzy_max_distance: int = None):
        """
        设置模糊搜索配置
        
        Args:
            fuzzy_threshold: 模糊搜索相似度阈值 (0.0-1.0)
            fuzzy_max_distance: 模糊搜索最大编辑距离
        """
        return self._fts.set_fuzzy_config(fuzzy_threshold, fuzzy_max_distance)
    
    def from_pandas(self, df, id_column: str = None, **kwargs):
        """
        从 Pandas DataFrame 导入数据
        
        Args:
            df: Pandas DataFrame
            id_column: 用作文档ID的列名
            **kwargs: 其他参数传递给 NanoFTS
        """
        return self._fts.from_pandas(df, id_column=id_column, **kwargs)
    
    def from_polars(self, df, id_column: str = None, **kwargs):
        """
        从 Polars DataFrame 导入数据
        
        Args:
            df: Polars DataFrame  
            id_column: 用作文档ID的列名
            **kwargs: 其他参数传递给 NanoFTS
        """
        return self._fts.from_polars(df, id_column=id_column, **kwargs)
    
    def from_arrow(self, table, id_column: str = None, **kwargs):
        """
        从 Apache Arrow Table 导入数据
        
        Args:
            table: Apache Arrow Table
            id_column: 用作文档ID的列名
            **kwargs: 其他参数传递给 NanoFTS
        """
        return self._fts.from_arrow(table, id_column=id_column, **kwargs)
    
    def from_parquet(self, file_path: str, id_column: str = None, **kwargs):
        """
        从 Parquet 文件导入数据
        
        Args:
            file_path: Parquet 文件路径
            id_column: 用作文档ID的列名
            **kwargs: 其他参数传递给 NanoFTS
        """
        return self._fts.from_parquet(file_path, id_column=id_column, **kwargs)
    
    def from_csv(self, file_path: str, id_column: str = None, **kwargs):
        """
        从 CSV 文件导入数据
        
        Args:
            file_path: CSV 文件路径
            id_column: 用作文档ID的列名
            **kwargs: 其他参数传递给 NanoFTS
        """
        return self._fts.from_csv(file_path, id_column=id_column, **kwargs)
    
    # 为了完全兼容，提供一些原实现中可能存在的方法
    def save(self):
        """保存索引到磁盘（别名方法，调用 flush）"""
        return self.flush()
    
    def close(self):
        """关闭搜索器（NanoFTS 可能不需要显式关闭，但保持兼容性）"""
        # NanoFTS 不需要显式关闭，但我们可以清理一些资源
        pass
    
    # 公开一些有用的属性用于调试和监控
    @property
    def config(self) -> Dict:
        """获取当前配置信息"""
        return {
            'index_dir': self.index_dir,
            'max_chinese_length': self.max_chinese_length,
            'num_workers': self.num_workers,
            'shard_size': self.shard_size,
            'min_term_length': self.min_term_length,
            'auto_save': self.auto_save,
            'batch_size': self.batch_size,
            'buffer_size': self.buffer_size,
            'drop_if_exists': self.drop_if_exists,
            **self.get_fuzzy_config()
        }


# 保持向后兼容，导出主要类
__all__ = ['FullTextSearch']