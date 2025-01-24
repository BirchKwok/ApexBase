import pytest
import os
import tempfile
import pandas as pd
import pyarrow as pa
import polars as pl
from apexbase import ApexClient

@pytest.fixture
def temp_db():
    """创建临时数据库文件"""
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
        temp_path = f.name
    yield temp_path
    os.unlink(temp_path)

def test_basic_operations(temp_db):
    """测试基本操作：存储、查询、检索"""
    client = ApexClient(temp_db)
    
    # 测试单条记录存储
    record = {"name": "John", "age": 30, "tags": ["python", "rust"]}
    id_ = client.store(record)
    assert isinstance(id_, int)
    
    # 测试记录检索
    retrieved = client.retrieve(id_)
    assert retrieved["name"] == "John"
    assert retrieved["age"] == 30
    assert retrieved["tags"] == ["python", "rust"]
    
    # 测试查询
    results = client.query("age = 30", return_ids_only=False)
    assert len(results) == 1
    assert results[0]["name"] == "John"

def test_batch_operations(temp_db):
    """测试批量操作：批量存储、批量检索"""
    client = ApexClient(temp_db)
    
    # 测试批量存储
    records = [
        {"name": "John", "age": 30},
        {"name": "Jane", "age": 25},
        {"name": "Bob", "age": 35}
    ]
    ids = client.store(records)
    assert len(ids) == 3
    
    # 测试批量检索
    retrieved = client.retrieve_many(ids)
    assert len(retrieved) == 3
    assert any(r["name"] == "John" for r in retrieved)
    assert any(r["name"] == "Jane" for r in retrieved)
    assert any(r["name"] == "Bob" for r in retrieved)

def test_delete_operations(temp_db):
    """测试删除操作：单条删除、批量删除"""
    client = ApexClient(temp_db)
    
    # 准备测试数据
    records = [
        {"name": "John", "age": 30},
        {"name": "Jane", "age": 25},
        {"name": "Bob", "age": 35}
    ]
    ids = client.store(records)
    
    # 测试单条删除
    assert client.delete(ids[0]) is True
    assert client.retrieve(ids[0]) is None
    
    # 测试批量删除
    deleted_ids = client.batch_delete(ids[1:])
    assert len(deleted_ids) == 2
    assert all(client.retrieve(id_) is None for id_ in deleted_ids)

def test_replace_operations(temp_db):
    """测试替换操作：单条替换、批量替换"""
    client = ApexClient(temp_db)
    
    # 准备测试数据
    records = [
        {"name": "John", "age": 30},
        {"name": "Jane", "age": 25}
    ]
    ids = client.store(records)
    
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

def test_data_import(temp_db):
    """测试数据导入：Pandas、PyArrow、Polars"""
    client = ApexClient(temp_db)
    
    # 准备测试数据
    data = {
        "name": ["John", "Jane", "Bob"],
        "age": [30, 25, 35],
        "city": ["New York", "London", "Paris"]
    }
    
    # 测试Pandas导入
    df_pandas = pd.DataFrame(data)
    client.from_pandas(df_pandas)
    assert len(client.query("age > 0")) == 3
    
    # 测试PyArrow导入
    table = pa.Table.from_pandas(df_pandas)
    client.from_pyarrow(table)
    assert len(client.query("age > 0")) == 6
    
    # 测试Polars导入
    df_polars = pl.DataFrame(data)
    client.from_polars(df_polars)
    assert len(client.query("age > 0")) == 9

def test_text_search(temp_db):
    """测试全文搜索功能"""
    client = ApexClient(temp_db)
    
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
    client.store(records)
    
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

def test_field_operations(temp_db):
    """测试字段操作"""
    client = ApexClient(temp_db)
    
    # 存储带有不同类型字段的记录
    record = {
        "text_field": "Hello",
        "int_field": 42,
        "float_field": 3.14,
        "bool_field": True,
        "list_field": [1, 2, 3],
        "dict_field": {"key": "value"}
    }
    client.store(record)
    
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

def test_concurrent_access(temp_db):
    """测试并发访问"""
    import threading
    import random
    
    client = ApexClient(temp_db)
    num_threads = 4
    records_per_thread = 100
    
    def worker():
        for _ in range(records_per_thread):
            record = {
                "value": random.randint(1, 1000),
                "thread_id": threading.get_ident()
            }
            client.store(record)
    
    threads = [threading.Thread(target=worker) for _ in range(num_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    
    # 验证所有记录都被正确存储
    results = client.query("1=1", return_ids_only=False)
    assert len(results) == num_threads * records_per_thread

def test_error_handling(temp_db):
    """测试错误处理"""
    client = ApexClient(temp_db)
    
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
    with pytest.raises(ValueError, match="Invalid query syntax"):
        client.query("INVALID SYNTAX !@#")

    # 测试不存在的字段
    results = client.query("non_existent_field = 'value'")
    assert len(results) == 0

def test_large_batch_operations(temp_db):
    """测试大批量操作性能"""
    client = ApexClient(temp_db)
    
    # 创建大量测试数据
    num_records = 10000
    records = [
        {"id": i, "value": i * 2}
        for i in range(num_records)
    ]
    
    # 测试批量存储性能
    ids = client.store(records)
    assert len(ids) == num_records
    
    # 测试查询性能
    results = client.query("value >= 0", return_ids_only=True)
    assert len(results) == num_records
    
    # 测试批量检索性能
    batch_size = 1000
    for i in range(0, num_records, batch_size):
        batch_ids = ids[i:i + batch_size]
        retrieved = client.retrieve_many(batch_ids)
        assert len(retrieved) == len(batch_ids)

def test_complex_queries(temp_db):
    """测试复杂查询"""
    client = ApexClient(temp_db)
    
    # 准备测试数据
    records = [
        {"name": "John", "age": 30, "city": "New York", "score": 85.5},
        {"name": "Jane", "age": 25, "city": "London", "score": 92.0},
        {"name": "Bob", "age": 35, "city": "New York", "score": 78.5},
        {"name": "Alice", "age": 28, "city": "Paris", "score": 88.0}
    ]
    client.store(records)
    
    # 测试多条件查询
    results = client.query("age > 25 AND city = 'New York'", return_ids_only=False)
    assert len(results) == 2  # John和Bob都符合条件
    assert all(r["city"] == "New York" and r["age"] > 25 for r in results)
    
    # 测试范围查询
    results = client.query("score >= 85 AND score <= 90", return_ids_only=False)
    assert len(results) == 2  # John和Alice符合条件
    assert all(85 <= r["score"] <= 90 for r in results)
    
    # 测试模糊查询
    results = client.query("name LIKE 'J%'", return_ids_only=False)
    assert len(results) == 2  # John和Jane符合条件
    assert all(r["name"].startswith("J") for r in results)
    
    # 测试排序
    results = client.query("1=1 ORDER BY age DESC", return_ids_only=False)
    assert len(results) == 4
    assert results[0]["age"] == 35  # Bob
    assert results[1]["age"] == 30  # John 