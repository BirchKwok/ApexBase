import pytest
import os
import tempfile
import shutil
import pandas as pd
import pyarrow as pa
import polars as pl
from pathlib import Path
from apexbase import ApexClient
import random

@pytest.fixture
def temp_dir():
    """创建临时目录"""
    temp_path = tempfile.mkdtemp()
    yield temp_path
    shutil.rmtree(temp_path)

def test_initialization(temp_dir):
    """测试初始化参数"""
    # 测试默认参数
    client = ApexClient()
    assert Path("apexbase.db").exists()
    os.remove("apexbase.db")  # 清理
    
    # 测试指定目录
    client = ApexClient(temp_dir)
    db_path = Path(temp_dir) / "apexbase.db"
    assert db_path.exists()
    
    # 测试drop_if_exists
    client = ApexClient(temp_dir, drop_if_exists=True)  # 应该删除并重新创建数据库
    assert db_path.exists()
    
    # 测试目录自动创建
    nested_dir = os.path.join(temp_dir, "nested", "path")
    client = ApexClient(nested_dir)
    assert Path(nested_dir).exists()
    assert (Path(nested_dir) / "apexbase.db").exists()

def test_basic_operations(temp_dir):
    """测试基本操作：存储、查询、检索"""
    client = ApexClient(temp_dir)
    
    # 测试单条记录存储
    record = {"name": "John", "age": 30, "tags": ["python", "rust"]}
    id_ = client.store(record)
    assert id_ is not None
    assert isinstance(id_, int)
    
    # 测试记录检索
    retrieved = client.retrieve(id_)
    assert retrieved is not None
    assert retrieved["name"] == record["name"]
    assert retrieved["age"] == record["age"]
    assert retrieved["tags"] == record["tags"]
    
    # 测试查询
    results = client.query("age = 30")
    assert len(results) == 1
    assert results[0]["name"] == "John"

def test_batch_operations(temp_dir):
    """测试批量操作：批量存储、批量检索"""
    client = ApexClient(temp_dir)
    
    # 测试批量存储
    records = [
        {"name": "John", "age": 30},
        {"name": "Jane", "age": 25},
        {"name": "Bob", "age": 35}
    ]
    ids = client.store(records)
    assert ids is not None
    assert len(ids) == 3
    
    # 测试批量检索
    retrieved = client.retrieve_many(ids)
    assert len(retrieved) == 3
    assert all(r["name"] in ["John", "Jane", "Bob"] for r in retrieved)
    
    # 测试查询
    results = client.query("age > 25")
    assert len(results) == 2
    names = [r["name"] for r in results]
    assert "John" in names
    assert "Bob" in names

def test_count_rows(temp_dir):
    """测试行数统计功能"""
    client = ApexClient(temp_dir)
    
    # 测试空表
    assert client.count_rows() == 0
    
    # 测试单条记录
    client.store({"name": "John"})
    assert client.count_rows() == 1
    
    # 测试多条记录
    records = [
        {"name": "Jane"},
        {"name": "Bob"},
        {"name": "Alice"}
    ]
    client.store(records)
    assert client.count_rows() == 4
    
    # 测试多表
    client.create_table("test_table")
    client.use_table("test_table")
    assert client.count_rows() == 0  # 新表应该为空
    
    client.store({"name": "Test"})
    assert client.count_rows() == 1  # 新表应该有一条记录
    
    client.use_table("default")
    assert client.count_rows() == 4  # 默认表应该保持4条记录
    
    # 测试不存在的表
    with pytest.raises(ValueError):
        client.count_rows("nonexistent_table")

def test_delete_operations(temp_dir):
    """测试删除操作：单条删除、批量删除"""
    client = ApexClient(temp_dir)
    
    # 准备测试数据
    records = [
        {"name": "John", "age": 30},
        {"name": "Jane", "age": 25},
        {"name": "Bob", "age": 35}
    ]
    ids = client.store(records)
    assert ids is not None
    assert len(ids) == 3
    
    # 测试单条删除
    assert client.delete(ids[0]) is True
    assert client.retrieve(ids[0]) is None
    assert client.count_rows() == 2  # 验证记录数
    
    # 测试批量删除
    deleted_ids = client.batch_delete(ids[1:])
    assert len(deleted_ids) == 2
    assert all(client.retrieve(id_) is None for id_ in deleted_ids)
    assert client.count_rows() == 0  # 验证记录数

def test_replace_operations(temp_dir):
    """测试替换操作：单条替换、批量替换"""
    client = ApexClient(temp_dir)
    
    # 准备测试数据
    records = [
        {"name": "John", "age": 30},
        {"name": "Jane", "age": 25}
    ]
    ids = client.store(records)
    assert ids is not None
    assert len(ids) == 2
    
    # 测试单条替换
    new_data = {"name": "John Doe", "age": 31}
    assert client.replace(ids[0], new_data) is True
    updated = client.retrieve(ids[0])
    assert updated["name"] == "John Doe"
    assert updated["age"] == 31
    
    # 测试批量替换
    batch_data = {
        ids[0]: {"name": "John Smith", "age": 32},
        ids[1]: {"name": "Jane Smith", "age": 26}
    }
    success_ids = client.batch_replace(batch_data)
    assert len(success_ids) == 2
    assert all(client.retrieve(id_)["name"].endswith("Smith") for id_ in success_ids)

def test_data_import(temp_dir):
    """测试数据导入：Pandas、PyArrow、Polars"""
    client = ApexClient(temp_dir)
    
    # 准备测试数据
    data = {
        "name": ["John", "Jane", "Bob"],
        "age": [30, 25, 35],
        "city": ["New York", "London", "Paris"]
    }
    
    # 测试Pandas导入
    df_pandas = pd.DataFrame(data)
    client.from_pandas(df_pandas)
    results = client.query("age > 0")
    assert len(results.ids) == 3
    
    # 测试PyArrow导入
    table = pa.Table.from_pandas(df_pandas)
    client.from_pyarrow(table)
    results = client.query("age > 0")
    assert len(results.ids) == 3
    
    # 测试Polars导入
    df_polars = pl.DataFrame(data)
    client.from_polars(df_polars)
    results = client.query("age > 0")
    assert len(results.ids) == 3

def test_text_search(temp_dir):
    """测试全文搜索功能"""
    client = ApexClient(temp_dir)
    
    # 准备测试数据
    records = [
        {
            "title": "Python Programming",
            "content": "Python is a great programming language",
            "tags": ["python", "programming"]
        },
        {
            "title": "Rust Development",
            "content": "Rust is a systems programming language",
            "tags": ["rust", "programming"]
        },
        {
            "title": "Database Design",
            "content": "SQLite is a lightweight database",
            "tags": ["database", "sqlite"]
        }
    ]
    ids = client.store(records)
    assert ids is not None
    assert len(ids) == 3
    
    # 设置可搜索字段
    client.set_searchable("title", True)
    client.set_searchable("content", True)
    client.set_searchable("tags", True)
    
    # 测试全文搜索
    results = client.search_text("python")
    assert len(results) == 1
    
    results = client.search_text("programming")
    assert len(results) == 2
    
    results = client.search_text("database")
    assert len(results) == 1
    
    # 测试指定字段搜索
    results = client.search_text("python", fields=["title"])
    assert len(results) == 1
    
    # 测试重建索引
    client.rebuild_search_index()
    results = client.search_text("programming")
    assert len(results) == 2

def test_field_operations(temp_dir):
    """测试字段操作"""
    client = ApexClient(temp_dir)
    
    # 存储带有不同类型字段的记录
    record = {
        "text_field": "Hello",
        "int_field": 42,
        "float_field": 3.14,
        "bool_field": True,
        "list_field": [1, 2, 3],
        "dict_field": {"key": "value"}
    }
    id_ = client.store(record)
    assert id_ is not None
    
    # 测试字段列表
    fields = client.list_fields()
    assert "text_field" in fields
    assert "int_field" in fields
    assert "float_field" in fields
    assert "bool_field" in fields
    assert "list_field" in fields
    assert "dict_field" in fields
    
    # 测试字段搜索设置
    client.set_searchable("text_field", True)
    client.set_searchable("dict_field", False)
    
    # 验证搜索结果
    results = client.search_text("Hello")
    assert len(results) == 1

def test_concurrent_access(temp_dir):
    """测试并发访问"""
    import threading
    import random
    
    client = ApexClient(temp_dir)
    num_threads = 4
    records_per_thread = 100
    
    def worker():
        for _ in range(records_per_thread):
            record = {
                "value": random.randint(1, 1000),
                "thread_id": threading.get_ident()
            }
            id_ = client.store(record)
            assert id_ is not None
    
    threads = [threading.Thread(target=worker) for _ in range(num_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    
    # 验证所有记录都被正确存储
    results = client.query("1=1")
    assert len(results) == num_threads * records_per_thread

    # 验证每个线程的记录数
    for thread_id in set(r["thread_id"] for r in results):
        thread_results = client.query(f"thread_id = {thread_id}")
        assert len(thread_results) == records_per_thread

def test_error_handling(temp_dir):
    """测试错误处理"""
    client = ApexClient(temp_dir)
    
    # 测试无效数据存储
    with pytest.raises(ValueError):
        client.store("invalid data")
    
    # 测试不存在的记录检索
    assert client.retrieve(999) is None
    
    # 测试不存在的记录删除
    assert client.delete(999) is False
    
    # 测试不存在的记录替换
    assert client.replace(999, {"name": "test"}) is False
    
    # 测试无效的查询语句
    with pytest.raises(ValueError):
        client.query("INVALID SYNTAX !@#").ids
        client.query("non_existent_field = 'value'").ids

def test_large_batch_operations(temp_dir):
    """测试大批量操作性能"""
    client = ApexClient(temp_dir)
    
    # 创建大量测试数据
    num_records = 10000
    records = [
        {"id": i, "value": i * 2}
        for i in range(num_records)
    ]
    
    # 测试批量存储性能
    ids = client.store(records)
    assert ids is not None
    assert len(ids) == num_records
    
    # 测试查询性能
    results = client.query("value >= 0")
    assert len(results) == num_records
    
    # 测试范围查询性能
    results = client.query("value >= 1000 AND value <= 2000")
    assert len(results) == 501  # (2000 - 1000) / 2 + 1
    
    # 测试批量检索性能
    sample_size = 1000
    sample_ids = random.sample(ids, sample_size)
    retrieved = client.retrieve_many(sample_ids)
    assert len(retrieved) == sample_size

def test_complex_queries(temp_dir):
    """测试复杂查询"""
    client = ApexClient(temp_dir)
    
    # 准备测试数据
    records = [
        {"name": "John", "age": 30, "city": "New York", "score": 85.5},
        {"name": "Jane", "age": 25, "city": "London", "score": 92.0},
        {"name": "Bob", "age": 35, "city": "New York", "score": 78.5},
        {"name": "Alice", "age": 28, "city": "Paris", "score": 88.0}
    ]
    ids = client.store(records)
    assert ids is not None
    assert len(ids) == 4
    
    # 测试多条件查询
    results = client.query("age > 25 AND city = 'New York'")
    assert len(results) == 2
    assert all(r["city"] == "New York" and r["age"] > 25 for r in results)
    
    # 测试范围查询
    results = client.query("score >= 85.0 AND score <= 90.0")
    assert len(results) == 2
    assert all(85.0 <= r["score"] <= 90.0 for r in results)
    
    # 测试 LIKE 查询
    results = client.query("name LIKE 'J%'")
    assert len(results) == 2
    assert all(r["name"].startswith("J") for r in results)

def test_case_insensitive_search(temp_dir):
    """测试大小写不敏感搜索"""
    client = ApexClient(temp_dir)
    
    # 准备测试数据
    records = [
        {"name": "John Smith", "email": "JOHN@example.com"},
        {"name": "JANE DOE", "email": "jane@EXAMPLE.com"},
        {"name": "Bob Wilson", "email": "bob@Example.COM"}
    ]
    ids = client.store(records)
    assert ids is not None
    assert len(ids) == 3
    
    # 设置字段为可搜索
    client.set_searchable("name", True)
    client.set_searchable("email", True)
    
    # 1. 测试全文搜索大小写不敏感
    test_cases = [
        ("john", 1),      # 小写搜索大写内容
        ("JANE", 1),      # 大写搜索大写内容
        ("Bob", 1),       # 首字母大写搜索
        ("EXAMPLE", 3),   # 大写搜索混合大小写
        ("example", 3),   # 小写搜索混合大小写
        ("COM", 3),       # 大写搜索域名
        ("com", 3)        # 小写搜索域名
    ]
    
    for search_term, expected_count in test_cases:
        results = client.search_text(search_term)
        assert len(results) == expected_count, \
            f"搜索词 '{search_term}' 应该返回 {expected_count} 条结果，但返回了 {len(results)} 条"
    
    # 2. 测试SQL查询大小写不敏感
    sql_test_cases = [
        # LIKE 操作符大小写不敏感
        ("name LIKE '%JOHN%'", 1),
        ("name like '%john%'", 1),
        ("email LIKE '%COM%'", 3),
        ("email like '%com%'", 3),
        
        # 逻辑操作符大小写不敏感
        ("name LIKE '%John%' AND email LIKE '%example%'", 1),
        ("name like '%John%' and email like '%example%'", 1),
        ("name LIKE '%John%' OR name LIKE '%Jane%'", 2),
        ("name like '%John%' or name like '%Jane%'", 2)
    ]
    
    for query, expected_count in sql_test_cases:
        results = client.query(query)
        assert len(results) == expected_count, \
            f"查询 '{query}' 应该返回 {expected_count} 条结果，但返回了 {len(results)} 条" 