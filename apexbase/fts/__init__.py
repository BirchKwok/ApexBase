import re
import mmap
import msgpack
from collections import defaultdict
from typing import List, Union, Dict, Tuple
from pathlib import Path
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor

from pyroaring import BitMap

class LRUCache:
    """LRU缓存，用于管理内存中的活跃索引"""
    def __init__(self, maxsize: int = 10000):
        self.cache = {}
        self.maxsize = maxsize
        self.hits = defaultdict(int)
    
    def get(self, key: str) -> BitMap:
        if key in self.cache:
            self.hits[key] += 1
            return self.cache[key]
        return None
    
    def put(self, key: str, value: BitMap):
        if len(self.cache) >= self.maxsize:
            # 移除最少使用的项
            lru_key = min(self.hits.items(), key=lambda x: x[1])[0]
            del self.cache[lru_key]
            del self.hits[lru_key]
        self.cache[key] = value
        self.hits[key] = 1

class IndexFile:
    """索引文件管理器，使用内存映射优化读取性能"""
    def __init__(self, file_path: Path):
        self.file_path = file_path
        self.mmap = None
        self.index_meta = {}  # 存储索引项的位置和大小
        self._load()
    
    def _load(self):
        if self.file_path.exists():
            with open(self.file_path, 'rb') as f:
                # 读取元数据
                meta_size = int.from_bytes(f.read(4), 'big')
                meta_data = f.read(meta_size)
                self.index_meta = msgpack.unpackb(meta_data, raw=False)
                # 内存映射数据部分
                self.mmap = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
    
    def get_bitmap(self, key: str) -> BitMap:
        """获取指定键的 BitMap"""
        if key in self.index_meta:
            offset, size = self.index_meta[key]
            return BitMap.deserialize(self.mmap[offset:offset + size])
        return None
    
    def close(self):
        if self.mmap:
            self.mmap.close()

class FullTextSearch:
    def __init__(self, index_dir: str = None, 
                 max_chinese_length: int = 4, 
                 num_workers: int = 4,
                 shard_size: int = 100_000,  # 每个分片包含的文档数
                 min_term_length: int = 2):   # 最小词长度
        """
        初始化全文搜索器。

        Args:
            index_dir: 索引文件存储目录，如果为None则使用内存索引
            max_chinese_length: 中文子串的最大长度，默认为4个字符
            num_workers: 并行构建索引的工作进程数，默认为CPU核心数
            shard_size: 每个分片包含的文档数，默认10万
            min_term_length: 最小词长度，小于此长度的词不会被索引
        """
        self.chinese_pattern = re.compile(r'[\u4e00-\u9fff]+')
        self.index_dir = Path(index_dir) if index_dir else None
        self.max_chinese_length = max_chinese_length
        self.num_workers = num_workers or mp.cpu_count()
        self.shard_size = shard_size
        self.min_term_length = min_term_length
        
        # 使用 LRU 缓存管理内存中的索引
        self.cache = LRUCache(maxsize=10000)
        self.modified_keys = set()
        
        # 内存索引
        self.index = defaultdict(BitMap)
        self.word_index = defaultdict(BitMap)
        
        if self.index_dir:
            self.index_dir.mkdir(parents=True, exist_ok=True)
            self._load_index()

    def _get_shard_path(self, shard_id: int) -> Path:
        """获取分片文件路径"""
        return self.index_dir / 'shards' / f'shard_{shard_id}.apex'

    def _save_index(self, incremental: bool = True):
        """将索引保存到磁盘，使用分片存储"""
        if not incremental:
            # 完整保存时，先清理旧的分片
            shards_dir = self.index_dir / 'shards'
            if shards_dir.exists():
                for f in shards_dir.glob('*.apex'):
                    f.unlink()
            shards_dir.mkdir(exist_ok=True)
        
        # 按文档ID范围分片
        shards = defaultdict(dict)
        for term, bitmap in self.index.items():
            if len(term) < self.min_term_length:  # 跳过过短的词
                continue
            doc_ids = sorted(bitmap)
            for doc_id in doc_ids:
                shard_id = doc_id // self.shard_size
                if term not in shards[shard_id]:
                    shards[shard_id][term] = BitMap()
                shards[shard_id][term].add(doc_id)
        
        # 保存每个分片
        for shard_id, shard_data in shards.items():
            shard_path = self._get_shard_path(shard_id)
            self._save_shard(shard_data, shard_path, incremental)
        
        # 保存词组索引
        word_dir = self.index_dir / 'word'
        word_dir.mkdir(exist_ok=True)
        self._save_shard(self.word_index, word_dir / "index.apex", incremental)
        
        if incremental:
            self.modified_keys.clear()

    def _save_shard(self, shard_data: Dict[str, BitMap], shard_path: Path, incremental: bool):
        """保存单个分片"""
        if not shard_data:
            if shard_path.exists():
                shard_path.unlink()
            return
        
        # 如果是增量更新，先读取现有数据
        existing_meta = {}
        existing_data = {}
        if incremental and shard_path.exists():
            with open(shard_path, 'rb') as f:
                meta_size = int.from_bytes(f.read(4), 'big')
                meta_data = f.read(meta_size)
                existing_meta = msgpack.unpackb(meta_data, raw=False)
                
                for key, (offset, size) in existing_meta.items():
                    if key not in self.modified_keys:
                        f.seek(4 + meta_size + offset)
                        existing_data[key] = f.read(size)
        
        # 准备新数据
        data = {}
        meta = {}
        offset = 0
        
        # 处理现有数据
        for key, bitmap_data in existing_data.items():
            meta[key] = (offset, len(bitmap_data))
            data[key] = bitmap_data
            offset += len(bitmap_data)
        
        # 处理新数据
        for key, bitmap in shard_data.items():
            if not bitmap:  # 跳过空的 BitMap
                continue
            if not incremental or key in self.modified_keys:
                bitmap_data = self._bitmap_to_bytes(bitmap)
                data[key] = bitmap_data
                meta[key] = (offset, len(bitmap_data))
                offset += len(bitmap_data)
        
        # 如果没有数据要保存，删除文件
        if not data:
            if shard_path.exists():
                shard_path.unlink()
            return
        
        # 保存分片
        shard_path.parent.mkdir(exist_ok=True)
        with open(shard_path, 'wb') as f:
            meta_data = msgpack.packb(meta, use_bin_type=True)
            f.write(len(meta_data).to_bytes(4, 'big'))
            f.write(meta_data)
            for bitmap_data in data.values():
                f.write(bitmap_data)

    def _load_index(self) -> bool:
        """从磁盘加载索引"""
        try:
            # 加载所有分片
            shards_dir = self.index_dir / 'shards'
            if not shards_dir.exists():
                return False
            
            for shard_path in shards_dir.glob('*.apex'):
                self._load_shard(shard_path)
            
            # 加载词组索引
            word_dir = self.index_dir / 'word'
            if word_dir.exists():
                word_index_path = word_dir / "index.apex"
                if word_index_path.exists():
                    self._load_shard(word_index_path, is_word_index=True)
            
            return True
        except Exception as e:
            print(f"加载索引失败: {e}")
            return False

    def _load_shard(self, shard_path: Path, is_word_index: bool = False):
        """加载单个分片"""
        if not shard_path.exists():
            return
        
        with open(shard_path, 'rb') as f:
            meta_size = int.from_bytes(f.read(4), 'big')
            meta_data = f.read(meta_size)
            meta = msgpack.unpackb(meta_data, raw=False)
            
            for key, (offset, size) in meta.items():
                if len(key) >= self.min_term_length:  # 只加载符合最小长度要求的词
                    f.seek(4 + meta_size + offset)
                    bitmap_data = f.read(size)
                    if is_word_index:
                        self.word_index[key] |= self._bytes_to_bitmap(bitmap_data)
                    else:
                        self.index[key] |= self._bytes_to_bitmap(bitmap_data)

    @staticmethod
    def _process_chunk(chunk: Tuple[int, List[Dict[str, Union[str, int, float]]], int, int]) -> Dict[str, List[int]]:
        """
        并行处理数据块，返回部分索引结果
        
        Args:
            chunk: (起始ID, 数据块, 最大中文长度, 最小词长度)的元组
        """
        start_id, docs, max_chinese_length, min_term_length = chunk
        chinese_pattern = re.compile(r'[\u4e00-\u9fff]+')
        result = defaultdict(set)
        
        for i, doc in enumerate(docs):
            doc_id = start_id + i
            for field_value in doc.values():
                field_str = str(field_value).lower()
                
                # 添加完整字段到索引（如果长度足够）
                if len(field_str) >= min_term_length:
                    result[field_str].add(doc_id)
                
                # 处理中文部分
                if chinese_pattern.search(field_str):
                    segments = chinese_pattern.findall(field_str)
                    for seg in segments:
                        n = len(seg)
                        for length in range(min_term_length, min(n + 1, max_chinese_length + 1)):
                            for j in range(n - length + 1):
                                substr = seg[j:j + length]
                                result[substr].add(doc_id)
                
                # 处理词组
                if ' ' in field_str:
                    words = field_str.split()
                    # 只为单词建立索引
                    for word in words:
                        if not chinese_pattern.search(word) and len(word) >= min_term_length:
                            result[word].add(doc_id)
        
        return {k: list(v) for k, v in result.items()}

    def add_document(self, doc_id: Union[int, List[int]], fields: Union[Dict[str, Union[str, int, float]], List[Dict[str, Union[str, int, float]]]]):
        """
        添加文档到索引。支持单条文档和批量文档插入。

        Args:
            doc_id: 文档ID，可以是单个整数或整数列表
            fields: 文档字段，可以是单个字典或字典列表。每个字典的值可以是字符串、整数或浮点数
        """
        # 转换为列表格式以统一处理
        if isinstance(doc_id, int):
            doc_ids = [doc_id]
            docs = [fields] if isinstance(fields, dict) else fields
        else:
            doc_ids = doc_id
            docs = fields if isinstance(fields, list) else [fields]
        
        # 验证输入
        if len(doc_ids) != len(docs):
            raise ValueError("文档ID列表和文档列表长度必须相同")
        
        # 批量处理文档
        chunk_size = max(1, len(docs) // (self.num_workers * 2))
        chunks = []
        
        for i in range(0, len(docs), chunk_size):
            chunk_docs = docs[i:i + chunk_size]
            chunk_start_id = doc_ids[i]
            chunk = (chunk_start_id, chunk_docs, self.max_chinese_length, self.min_term_length)
            chunks.append(chunk)
        
        # 并行处理文档
        if len(chunks) > 1:
            with ProcessPoolExecutor(max_workers=self.num_workers) as executor:
                results = list(executor.map(self._process_chunk, chunks))
        else:
            results = [self._process_chunk(chunks[0])]
        
        # 合并结果
        for result in results:
            for key, doc_ids in result.items():
                if len(key) >= self.min_term_length:  # 只索引符合最小长度要求的词
                    self.index[key] |= BitMap(doc_ids)
                    self.modified_keys.add(key)
        
        # 更新词组索引
        self._build_word_index()
        
        # 如果使用磁盘存储，保存更新
        if self.index_dir:
            self._save_index(incremental=True)

    def remove_document(self, doc_id: int):
        """从索引中删除文档"""
        # 先从主索引中删除
        modified = False
        keys_to_remove = []
        for key, doc_ids in self.index.items():
            if doc_id in doc_ids:
                doc_ids.discard(doc_id)
                self.modified_keys.add(key)
                modified = True
                if not doc_ids:
                    keys_to_remove.append(key)
        
        # 删除空的键
        for key in keys_to_remove:
            del self.index[key]
        
        # 从词组索引中删除
        keys_to_remove = []
        for key, doc_ids in self.word_index.items():
            if doc_id in doc_ids:
                doc_ids.discard(doc_id)
                self.modified_keys.add(key)
                modified = True
                if not doc_ids:
                    keys_to_remove.append(key)
        
        # 删除空的键
        for key in keys_to_remove:
            del self.word_index[key]
        
        # 如果有修改，保存到磁盘
        if modified and self.index_dir:
            self._save_index(incremental=True)
        
        # 清除缓存
        self.cache = LRUCache(maxsize=self.cache.maxsize)

    def _bitmap_to_bytes(self, bitmap: BitMap) -> bytes:
        """将 BitMap 转换为字节串"""
        return bitmap.serialize()
    
    def _bytes_to_bitmap(self, data: bytes) -> BitMap:
        """将字节串转换回 BitMap"""
        return BitMap.deserialize(data)

    def _build_word_index(self):
        """构建词组反向索引"""
        self.word_index.clear()
        for field_str, doc_ids in self.index.items():
            if ' ' in field_str:
                words = field_str.lower().split()
                for word in words:
                    if not self.chinese_pattern.search(word) and len(word) >= self.min_term_length:
                        self.word_index[word] |= doc_ids

    def search(self, query: str) -> List[int]:
        """搜索实现"""
        query_key = query.lower()  # 统一转换为小写

        # 使用缓存获取结果
        cached_result = self.cache.get(query_key)
        if cached_result is not None:
            return sorted(cached_result)

        result = BitMap()
        
        # 检查是否为中文查询
        if self.chinese_pattern.search(query):
            # 中文查询
            if query_key in self.index:
                result |= self.index[query_key]
        else:
            # 非中文查询
            if query_key in self.index:
                result |= self.index[query_key]
            elif ' ' in query_key:
                # 词组查询
                words = query_key.split()
                # 检查每个单词是否存在于索引中
                word_results = []
                for word in words:
                    if len(word) >= self.min_term_length:  # 只处理符合最小长度要求的词
                        word_result = BitMap()
                        if word in self.word_index:
                            word_result |= self.word_index[word]
                        word_results.append(word_result)
                
                # 使用交集获取包含所有单词的文档
                if word_results:
                    result = word_results[0]
                    for other_result in word_results[1:]:
                        result &= other_result

        # 缓存结果
        self.cache.put(query_key, result)
        return sorted(result)


if __name__ == '__main__':
    # 示例数据：文档列表，每个文档是一个字典
    data = [
        {"title": "Hello World", "content": "Python 全文搜索器"},
        {"title": "GitHub Copilot", "content": "代码自动生成"},
        {"title": "全文搜索", "content": "支持多语言", "tags": "测试数据"},
        {"title": "hello", "content": "WORLD", "number": 123},
        {"title": "数据处理", "content": "搜索引擎"},
        {"title": "hello world", "content": "示例文本"},
        {"title": "混合文本", "content": "Mixed 全文内容测试"},
        {"title": "hello world 你好", "content": "示例文本"},
    ]
    
    # 使用磁盘索引
    index_dir = "fts_index"
    fts = FullTextSearch(index_dir=index_dir)
    
    # 批量添加文档
    doc_ids = list(range(len(data)))
    fts.add_document(doc_ids, data)
    
    # 基本搜索测试
    print("\n=== 基本搜索测试 ===")
    query1 = "Hello World"
    result1 = fts.search(query1)
    print(f"查询【{query1}】的行索引: {result1}")
    
    query2 = "hello world"
    result2 = fts.search(query2)
    print(f"查询【{query2}】的行索引: {result2}")
    
    query3 = "全文"
    result3 = fts.search(query3)
    print(f"查询【{query3}】的行索引: {result3}")
    
    query4 = "mixed"
    result4 = fts.search(query4)
    print(f"查询【{query4}】的行索引: {result4}")
    
    # 增量更新测试
    print("\n=== 增量更新测试 ===")
    # 添加新文档
    new_doc = {"title": "新增文档", "content": "测试全文搜索", "tags": "hello world test"}
    fts.add_document(len(data), new_doc)
    
    print("添加新文档后:")
    result5 = fts.search("新增")
    print(f"查询【新增】的行索引: {result5}")
    result6 = fts.search("hello world")
    print(f"查询【hello world】的行索引: {result6}")
    
    # 删除文档
    fts.remove_document(len(data))
    print("\n删除新文档后:")
    result7 = fts.search("新增")
    print(f"查询【新增】的行索引: {result7}")
    result8 = fts.search("hello world")
    print(f"查询【hello world】的行索引: {result8}")
    
    # 从磁盘加载索引
    print("\n=== 重新加载索引测试 ===")
    fts_reload = FullTextSearch(index_dir=index_dir)
    result9 = fts_reload.search("hello world")
    print(f"重新加载后查询【hello world】的行索引: {result9}")
    