from pathlib import Path
import pytest
import time
import random
import string
import psutil
import os
from apexbase import ApexClient
import numpy as np
from typing import List, Dict
import json


def generate_random_string(length: int) -> str:
    """生成随机字符串"""
    return ''.join(random.choices(string.ascii_letters + string.digits, k=length))


def generate_random_record() -> dict:
    """生成随机测试数据"""
    return {
        "name": generate_random_string(10),
        "age": random.randint(18, 80),
        "score": random.uniform(60, 100),
        "is_active": random.choice([True, False]),
        "tags": [generate_random_string(5) for _ in range(random.randint(1, 5))],
        "profile": {
            "city": random.choice(["北京", "上海", "广州", "深圳", "杭州"]),
            "skills": [generate_random_string(8) for _ in range(random.randint(2, 6))],
            "experience": random.randint(0, 20)
        },
        "data": np.random.rand(10).tolist()  # 随机数值数组
    }


def measure_memory() -> float:
    """测量当前进程的内存使用（MB）"""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024


def test_large_scale_performance(tmp_path):
    """测试大规模数据性能"""
    db_path = tmp_path / "perf_test.db"
    client = ApexClient(str(db_path))
    
    # 测试参数
    total_records = 1_000_000  # 总记录数
    batch_size = 10000  # 批量写入大小
    num_batches = total_records // batch_size
    
    print("\n=== 大规模性能测试开始 ===")
    
    # 1. 测试批量写入性能
    print("\n1. 批量写入性能测试")
    write_times = []
    memory_usages = []
    
    start_time = time.time()
    initial_memory = measure_memory()
    
    for i in range(num_batches):
        records = [generate_random_record() for _ in range(batch_size)]
        
        batch_start = time.time()
        # 使用store方法进行批量存储
        client.store(records)
        
        batch_time = time.time() - batch_start
        write_times.append(batch_time)
        memory_usages.append(measure_memory())
        
        if (i + 1) % 10 == 0:
            print(f"已写入 {(i + 1) * batch_size} 条记录")
            print(f"当前批次写入时间: {batch_time:.2f}秒")
            print(f"当前内存使用: {memory_usages[-1]:.2f}MB")
            print(f"每秒写入记录数: {batch_size/batch_time:.2f}条")
    
    total_time = time.time() - start_time
    memory_increase = max(memory_usages) - initial_memory
    
    print(f"\n写入完成:")
    print(f"总时间: {total_time:.2f}秒")
    print(f"平均每秒写入: {total_records/total_time:.2f}条")
    print(f"内存增长: {memory_increase:.2f}MB")
    
    # 2. 测试查询性能
    print("\n2. 查询性能测试")
    
    # 2.1 简单条件查询
    query_start = time.time()
    results = client.query("age > 30 AND is_active = 1", return_ids_only=True)
    query_time = time.time() - query_start
    print(f"简单条件查询耗时: {query_time:.2f}秒, 结果数量: {len(results)}")
    
    # 2.2 复杂条件查询
    query_start = time.time()
    results = client.query(
        "age > 25 AND score >= 80 AND is_active = 1 ORDER BY age DESC", 
        return_ids_only=True
    )
    query_time = time.time() - query_start
    print(f"复杂条件查询耗时: {query_time:.2f}秒, 结果数量: {len(results)}")
    
    # 2.3 JSON字段查询
    query_start = time.time()
    results = client.query(
        'json_extract(profile, "$.city") = "北京" AND json_extract(profile, "$.experience") = 1',
        return_ids_only=True
    )
    query_time = time.time() - query_start
    print(f"JSON字段查询耗时: {query_time:.2f}秒, 结果数量: {len(results)}")
    
    # 2.4 复杂JSON查询
    query_start = time.time()
    results = client.query(
        'json_extract(profile, "$.city") = "北京" AND ' +
        'json_extract(profile, "$.experience") = 6 AND ' +
        'json_extract(profile, "$.skills") IS NOT NULL',
        return_ids_only=True
    )
    query_time = time.time() - query_start
    print(f"复杂JSON查询耗时: {query_time:.2f}秒, 结果数量: {len(results)}")
    
    # 3. 测试全文搜索性能
    print("\n3. 全文搜索性能测试")
    
    # # 3.1 创建索引
    # index_start = time.time()
    # client.rebuild_search_index()
    # index_time = time.time() - index_start
    # print(f"重建全文索引耗时: {index_time:.2f}秒")
    
    # 3.2 执行搜索
    search_start = time.time()
    results = client.search_text("北京")
    search_time = time.time() - search_start
    print(f"全文搜索耗时: {search_time:.2f}秒, 结果数量: {len(results)}")
    
    # 4. 测试数据库大小
    db_size = os.path.getsize(str(db_path)) / (1024 * 1024)  # MB
    print(f"\n数据库文件大小: {db_size:.2f}MB")
    
    # 5. 测试优化性能
    print("\n4. 数据库优化测试")
    optimize_start = time.time()
    client.optimize()
    optimize_time = time.time() - optimize_start
    print(f"数据库优化耗时: {optimize_time:.2f}秒")
    
    # 优化后的数据库大小
    optimized_size = os.path.getsize(str(db_path)) / (1024 * 1024)  # MB
    print(f"优化后数据库大小: {optimized_size:.2f}MB")
    print(f"优化节省空间: {db_size - optimized_size:.2f}MB")
    
    print("\n=== 性能测试完成 ===")
    
    # 验证数据完整性
    total_count = len(client.query("1=1", return_ids_only=True))
    assert total_count == total_records, f"数据完整性检查失败：期望 {total_records} 条记录，实际 {total_count} 条记录" 


if __name__ == "__main__":
    test_large_scale_performance(Path("."))
