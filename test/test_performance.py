from pathlib import Path
import pytest
import time
import random
import string
import psutil
import os
from apexbase import ApexClient
import numpy as np
import tempfile
import json
import shutil

@pytest.fixture
def temp_dir():
    """创建临时目录"""
    temp_path = tempfile.mkdtemp()
    yield temp_path
    shutil.rmtree(temp_path)

def generate_random_string(length: int) -> str:
    """生成随机字符串"""
    return ''.join(random.choices(string.ascii_letters + string.digits, k=length))

def generate_test_records(count: int) -> list:
    """生成测试记录"""
    records = []
    for _ in range(count):
        record = {
            "name": generate_random_string(10),
            "age": random.randint(18, 80),
            "email": f"{generate_random_string(8)}@example.com",
            "tags": [generate_random_string(5) for _ in range(random.randint(1, 5))],
            "address": {
                "city": generate_random_string(8),
                "street": generate_random_string(15),
                "number": random.randint(1, 1000)
            }
        }
        records.append(record)
    return records

def measure_memory() -> float:
    """测量当前进程的内存使用（MB）"""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024

def test_large_scale_performance(temp_dir):
    """大规模性能测试"""
    print("\n=== 大规模性能测试开始 ===\n")
    
    client = ApexClient(temp_dir)
    
    # 1. 批量写入性能测试
    print("1. 批量写入性能测试")
    batch_sizes = [1000, 10000, 100000]
    
    for batch_size in batch_sizes:
        records = generate_test_records(batch_size)
        
        # 禁用自动FTS更新以提高写入性能
        client.set_auto_update_fts(False)
        
        start_time = time.time()
        ids = client.store(records)
        assert ids is not None
        assert len(ids) == batch_size
        end_time = time.time()
        
        print(f"写入 {batch_size} 条记录耗时: {end_time - start_time:.2f} 秒")
        print(f"平均每条记录写入耗时: {(end_time - start_time) * 1000 / batch_size:.2f} 毫秒")
        
        # 手动重建FTS索引
        print("重建FTS索引...")
        start_time = time.time()
        client.rebuild_search_index()
        end_time = time.time()
        print(f"重建索引耗时: {end_time - start_time:.2f} 秒")
        
        print()
    
    # 2. 查询性能测试
    print("\n2. 查询性能测试")
    
    # 2.1 简单查询
    print("2.1 简单查询性能")
    start_time = time.time()
    results = client.query("age > 30")
    end_time = time.time()
    print(f"简单查询耗时: {end_time - start_time:.2f} 秒")
    print(f"返回记录数: {len(results)}")
    
    # 2.2 复杂查询
    print("\n2.2 复杂查询性能")
    start_time = time.time()
    results = client.query("age > 30 AND age < 50")
    end_time = time.time()
    print(f"复杂查询耗时: {end_time - start_time:.2f} 秒")
    print(f"返回记录数: {len(results)}")
    
    # 2.3 全文搜索性能
    print("\n2.3 全文搜索性能")
    
    # 设置字段为可搜索
    client.set_searchable("email", True)
    client.set_searchable("name", True)
    
    # 测试不同类型的搜索
    search_terms = [
        "example",  # 简单词
        "john",     # 名字
        "com"       # 域名部分
    ]
    
    for term in search_terms:
        start_time = time.time()
        results = client.search_text(term)
        end_time = time.time()
        print(f"搜索词 '{term}' 耗时: {end_time - start_time:.2f} 秒")
        print(f"返回记录数: {len(results)}")
    
    # 3. 批量检索性能测试
    print("\n3. 批量检索性能测试")
    sample_size = min(1000, len(results))
    sample_ids = random.sample(results.ids, sample_size)
    
    start_time = time.time()
    retrieved = client.retrieve_many(sample_ids)
    end_time = time.time()
    print(f"批量检索 {sample_size} 条记录耗时: {end_time - start_time:.2f} 秒")
    print(f"平均每条记录检索耗时: {(end_time - start_time) * 1000 / sample_size:.2f} 毫秒")
    
    print("\n=== 大规模性能测试完成 ===")

def test_data_integrity(temp_dir):
    """数据完整性测试"""
    print("\n=== 数据完整性测试开始 ===\n")
    
    client = ApexClient(temp_dir)
    
    # 1. 生成测试数据
    print("1. 生成测试数据")
    total_records = 10000
    records = generate_test_records(total_records)
    
    # 记录原始数据的哈希值
    original_hashes = {}
    for record in records:
        record_str = json.dumps(record, sort_keys=True)
        original_hashes[hash(record_str)] = record
    
    # 2. 批量写入测试
    print("2. 批量写入测试")
    client.set_auto_update_fts(False)  # 禁用FTS以提高性能
    ids = client.store(records)
    assert ids is not None
    assert len(ids) == total_records
    
    # 3. 数据一致性验证
    print("3. 数据一致性验证")
    retrieved_records = client.retrieve_many(ids)
    
    # 验证记录数量
    assert len(retrieved_records) == total_records, \
        f"记录数量不匹配：期望 {total_records}，实际 {len(retrieved_records)}"
    
    # 验证数据内容
    for record in retrieved_records:
        # 移除_id字段再比较
        record_copy = record.copy()
        del record_copy['_id']
        record_str = json.dumps(record_copy, sort_keys=True)
        record_hash = hash(record_str)
        
        assert record_hash in original_hashes, \
            f"找不到记录的原始数据: {record}"
        
        original = original_hashes[record_hash]
        assert record_copy == original, \
            f"数据不匹配：\n原始：{original}\n实际：{record_copy}"
    
    print("基本数据一致性验证通过")
    
    # 4. 字段类型一致性测试
    print("\n4. 字段类型一致性测试")
    fields = client.list_fields()
    
    type_test_data = {
        "int_field": 42,
        "float_field": 3.14,
        "str_field": "test string",
        "bool_field": True,
        "list_field": [1, 2, 3],
        "dict_field": {"key": "value"},
        "null_field": None
    }
    
    # 存储测试数据
    record_id = client.store(type_test_data)
    assert record_id is not None
    retrieved = client.retrieve(record_id)
    
    # 验证字段类型
    for field, value in type_test_data.items():
        if value is None:
            # SQLite 会忽略 NULL 值字段，所以这些字段在检索时可能不存在
            if field in retrieved:
                assert retrieved[field] is None, \
                    f"NULL 值字段应该返回 None，实际为 {retrieved[field]}"
        else:
            assert field in retrieved, f"字段 {field} 丢失"
            if field == "bool_field":
                # SQLite 将布尔值存储为整数（1/0）
                assert isinstance(retrieved[field], int), \
                    f"布尔字段应该存储为整数类型，实际为 {type(retrieved[field])}"
                assert retrieved[field] in (0, 1), \
                    f"布尔字段的值应该是 0 或 1，实际为 {retrieved[field]}"
            else:
                assert type(retrieved[field]) == type(value), \
                    f"字段 {field} 类型不匹配：期望 {type(value)}，实际 {type(retrieved[field])}"
    
    print("字段类型一致性验证通过")
    
    # 5. 并发写入一致性测试
    print("\n5. 并发写入一致性测试")
    import threading
    
    concurrent_records = 1000
    threads_count = 4
    records_per_thread = concurrent_records // threads_count
    thread_results = {i: [] for i in range(threads_count)}
    
    def concurrent_write(thread_id):
        thread_records = generate_test_records(records_per_thread)
        thread_results[thread_id] = client.store(thread_records)
        assert thread_results[thread_id] is not None
        assert len(thread_results[thread_id]) == records_per_thread
    
    # 创建并启动线程
    threads = []
    for i in range(threads_count):
        t = threading.Thread(target=concurrent_write, args=(i,))
        threads.append(t)
        t.start()
    
    # 等待所有线程完成
    for t in threads:
        t.join()
    
    # 验证所有记录都被正确存储
    all_ids = []
    for ids in thread_results.values():
        all_ids.extend(ids)
    
    retrieved = client.retrieve_many(all_ids)
    assert len(retrieved) == concurrent_records, \
        f"并发写入记录数不匹配：期望 {concurrent_records}，实际 {len(retrieved)}"
    
    print("并发写入一致性验证通过")
    
    # 6. 事务一致性测试
    print("\n6. 事务一致性测试")
    
    # 准备更新数据
    update_data = {}
    for id_ in all_ids[:100]:
        update_data[id_] = {
            "name": "Updated " + generate_random_string(8),
            "age": random.randint(18, 80)
        }
    
    # 执行批量更新
    success_ids = client.batch_replace(update_data)
    assert len(success_ids) == len(update_data), \
        f"更新记录数不匹配：期望 {len(update_data)}，实际 {len(success_ids)}"
    
    # 验证更新结果
    updated_records = client.retrieve_many(success_ids)
    for record in updated_records:
        expected = update_data[record['_id']]
        for field, value in expected.items():
            assert record[field] == value, \
                f"更新后的值不匹配：期望 {value}，实际 {record[field]}"
    
    print("事务一致性验证通过")
    
    print("\n=== 数据完整性测试完成 ===")

if __name__ == "__main__":
    test_large_scale_performance(Path("."))
