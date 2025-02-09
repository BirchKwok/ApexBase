import pytest
import pandas as pd
import pyarrow as pa
import polars as pl
from pathlib import Path
import shutil
from apexbase import ApexClient

@pytest.fixture(params=["sqlite", "duckdb"])
def client(request):
    # 设置测试目录
    test_dir = Path(f"test_data_{request.param}")
    if test_dir.exists():
        shutil.rmtree(test_dir)
    
    # 创建一个新的ApexClient实例
    client = ApexClient(dirpath=test_dir, backend=request.param, drop_if_exists=True)
    
    yield client
    
    # 清理测试数据
    client.close()
    if test_dir.exists():
        shutil.rmtree(test_dir)

def test_basic_operations(client):
    # 测试存储单条记录
    data = {"name": "John", "age": 30}
    id_ = client.store(data)
    assert isinstance(id_, int)
    
    # 测试检索记录
    retrieved = client.retrieve(id_)
    assert retrieved["name"] == "John"
    assert retrieved["age"] == 30
    
    # 测试批量存储
    batch_data = [
        {"name": "Alice", "age": 25},
        {"name": "Bob", "age": 35}
    ]
    ids = client.store(batch_data)
    assert len(ids) == 2
    
    # 测试批量检索
    results = client.retrieve_many(ids)
    assert len(results) == 2
    assert results[0]["name"] == "Alice"
    assert results[1]["name"] == "Bob"

def test_query_operations(client):
    # 插入测试数据
    test_data = [
        {"name": "John", "age": 30, "city": "New York"},
        {"name": "Alice", "age": 25, "city": "Boston"},
        {"name": "Bob", "age": 35, "city": "New York"}
    ]
    client.store(test_data)
    
    # 测试简单查询
    results = client.query("age > 28")
    assert results.shape == (2, 3)
    assert results.columns.tolist() == ["name", "age", "city"]
    
    # 测试复杂查询
    results = client.query("age > 28 AND city = 'New York'")
    assert results.shape == (2, 3)
    assert results.columns.tolist() == ["name", "age", "city"]
    
    # 测试检索所有记录
    all_results = client.retrieve_all()
    assert all_results.shape == (3, 3)
    assert all_results.columns.tolist() == ["name", "age", "city"]

def test_table_operations(client):
    # 测试创建新表
    client.create_table("users")
    
    # 测试切换表
    client.use_table("users")
    
    # 测试列出所有表
    tables = client.list_tables()
    assert "users" in tables
    assert "default" in tables
    
    # 测试在新表中存储数据
    client.store({"name": "John"})
    
    # 测试删除表
    client.drop_table("users")
    tables = client.list_tables()
    assert "users" not in tables

def test_update_operations(client):
    # 插入初始数据
    id_ = client.store({"name": "John", "age": 30})
    
    # 测试替换单条记录
    success = client.replace(id_, {"name": "John Doe", "age": 31})
    assert success
    
    updated = client.retrieve(id_)
    assert updated["name"] == "John Doe"
    assert updated["age"] == 31
    
    # 测试批量替换
    data_dict = {id_: {"name": "John Smith", "age": 32}}
    success_ids = client.batch_replace(data_dict)
    assert len(success_ids) == 1
    
    updated = client.retrieve(id_)
    assert updated["name"] == "John Smith"

def test_delete_operations(client):
    # 插入测试数据
    id1 = client.store({"name": "John"})
    id2 = client.store({"name": "Alice"})
    
    # 测试删除单条记录
    success = client.delete(id1)
    assert success
    assert client.retrieve(id1) is None
    
    # 测试批量删除
    success = client.delete([id2])
    assert success
    assert client.retrieve(id2) is None

def test_dataframe_imports(client):
    # 测试pandas导入
    pdf = pd.DataFrame({
        "name": ["John", "Alice"],
        "age": [30, 25]
    })
    client.from_pandas(pdf)
    
    # 测试pyarrow导入
    table = pa.Table.from_pandas(pdf)
    client.from_pyarrow(table)
    
    # 测试polars导入
    pldf = pl.DataFrame({
        "name": ["Bob", "Charlie"],
        "age": [35, 40]
    })
    client.from_polars(pldf)
    
    # 验证数据导入
    all_results = client.retrieve_all()
    assert all_results.shape == (6, 2)

def test_utility_operations(client):
    # 插入一些测试数据
    client.store({"name": "John", "age": 30})
    client.store({"name": "Alice", "age": 25})
    
    # 测试字段列表
    fields = client.list_fields()
    assert "name" in fields
    assert "age" in fields
    
    # 测试行数统计
    count = client.count_rows()
    assert count == 2
    
    # 测试优化
    client.optimize()  # 不会抛出异常

def test_backend_selection():
    # 测试SQLite后端
    sqlite_client = ApexClient(dirpath="test_sqlite", backend="sqlite")
    assert sqlite_client is not None
    sqlite_client.close()
    
    # 测试DuckDB后端
    duckdb_client = ApexClient(dirpath="test_duckdb", backend="duckdb")
    assert duckdb_client is not None
    duckdb_client.close()
    
    # 清理测试目录
    shutil.rmtree("test_sqlite", ignore_errors=True)
    shutil.rmtree("test_duckdb", ignore_errors=True) 