import pytest
import time
import random
import string
from pathlib import Path
import shutil
from apexbase import ApexClient
import psutil
import os

def generate_random_string(length=10):
    """生成随机字符串"""
    return ''.join(random.choices(string.ascii_letters + string.digits, k=length))

def generate_test_data(size=1000):
    """生成测试数据"""
    return [
        {
            "name": generate_random_string(),
            "age": random.randint(1, 100),
            "email": f"{generate_random_string()}@example.com",
            "score": random.uniform(0, 100),
            "is_active": random.choice([True, False]),
            "tags": [generate_random_string() for _ in range(3)],
            "metadata": {
                "created_at": generate_random_string(),
                "updated_at": generate_random_string(),
                "version": random.randint(1, 10)
            }
        }
        for _ in range(size)
    ]

def get_process_memory():
    """获取当前进程的内存使用情况（MB）"""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024

def measure_performance(func):
    """测量函数执行时间和内存使用的装饰器"""
    def wrapper(*args, **kwargs):
        start_time = time.time()
        start_memory = get_process_memory()
        
        result = func(*args, **kwargs)
        
        end_time = time.time()
        end_memory = get_process_memory()
        
        duration = end_time - start_time
        memory_used = end_memory - start_memory
        
        return result, duration, memory_used
    return wrapper

@pytest.fixture(params=["sqlite", "duckdb"])
def client(request):
    """创建测试客户端"""
    test_dir = Path(f"test_data_{request.param}")
    if test_dir.exists():
        shutil.rmtree(test_dir)
    
    client = ApexClient(dirpath=test_dir, backend=request.param, drop_if_exists=True)
    yield client
    
    client.close()
    if test_dir.exists():
        shutil.rmtree(test_dir)

def measure_time(func):
    """测量函数执行时间的装饰器"""
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        return result, end_time - start_time
    return wrapper

def test_single_store_performance(client):
    """测试单条记录存储性能"""
    data = generate_test_data(1)[0]
    
    @measure_time
    def store_single():
        return client.store(data)
    
    _, duration = store_single()
    print(f"\n{client.storage.__class__.__name__} 单条记录存储耗时: {duration:.4f}秒")
    assert duration < 1.0  # 确保单条存储在1秒内完成

def test_batch_store_performance(client):
    """测试批量记录存储性能"""
    batch_sizes = [100, 1000, 10000]
    
    for size in batch_sizes:
        data = generate_test_data(size)
        
        @measure_time
        def store_batch():
            return client.store(data)
        
        _, duration = store_batch()
        print(f"\n{client.storage.__class__.__name__} {size}条记录批量存储耗时: {duration:.4f}秒")
        assert duration < size * 0.001  # 每条记录平均不超过1毫秒

def test_single_query_performance(client):
    """测试单条记录查询性能"""
    # 准备数据
    data = generate_test_data(1)[0]
    id_ = client.store(data)
    
    @measure_time
    def query_single():
        return client.retrieve(id_)
    
    _, duration = query_single()
    print(f"\n{client.storage.__class__.__name__} 单条记录查询耗时: {duration:.4f}秒")
    assert duration < 0.1  # 确保单条查询在0.1秒内完成

def test_batch_query_performance(client):
    """测试批量记录查询性能"""
    batch_sizes = [100, 1000, 10000]
    
    for size in batch_sizes:
        # 准备数据
        data = generate_test_data(size)
        ids = client.store(data)
        
        @measure_time
        def query_batch():
            return client.retrieve_many(ids)
        
        _, duration = query_batch()
        print(f"\n{client.storage.__class__.__name__} {size}条记录批量查询耗时: {duration:.4f}秒")
        assert duration < size * 0.0005  # 每条记录平均不超过0.5毫秒

def test_single_update_performance(client):
    """测试单条记录更新性能"""
    # 准备数据
    data = generate_test_data(1)[0]
    id_ = client.store(data)
    update_data = generate_test_data(1)[0]
    
    @measure_time
    def update_single():
        return client.replace(id_, update_data)
    
    _, duration = update_single()
    print(f"\n{client.storage.__class__.__name__} 单条记录更新耗时: {duration:.4f}秒")
    assert duration < 0.5  # 确保单条更新在0.5秒内完成

def test_batch_update_performance(client):
    """测试批量记录更新性能"""
    batch_sizes = [100, 1000, 10000]
    
    for size in batch_sizes:
        # 准备数据
        data = generate_test_data(size)
        ids = client.store(data)
        update_data = {id_: record for id_, record in zip(ids, generate_test_data(size))}
        
        @measure_time
        def update_batch():
            return client.batch_replace(update_data)
        
        _, duration = update_batch()
        print(f"\n{client.storage.__class__.__name__} {size}条记录批量更新耗时: {duration:.4f}秒")
        assert duration < size * 0.001  # 每条记录平均不超过1毫秒

def test_single_delete_performance(client):
    """测试单条记录删除性能"""
    # 准备数据
    data = generate_test_data(1)[0]
    id_ = client.store(data)
    
    @measure_time
    def delete_single():
        return client.delete(id_)
    
    _, duration = delete_single()
    print(f"\n{client.storage.__class__.__name__} 单条记录删除耗时: {duration:.4f}秒")
    assert duration < 0.1  # 确保单条删除在0.1秒内完成

def test_batch_delete_performance(client):
    """测试批量记录删除性能"""
    batch_sizes = [100, 1000, 10000]
    
    for size in batch_sizes:
        # 准备数据
        data = generate_test_data(size)
        ids = client.store(data)
        
        @measure_time
        def delete_batch():
            return client.delete(ids)
        
        _, duration = delete_batch()
        print(f"\n{client.storage.__class__.__name__} {size}条记录批量删除耗时: {duration:.4f}秒")
        assert duration < size * 0.0005  # 每条记录平均不超过0.5毫秒

def test_complex_query_performance(client):
    """测试复杂查询性能"""
    # 准备数据
    size = 10000
    data = generate_test_data(size)
    client.store(data)
    
    # 测试不同类型的查询
    queries = [
        "age > 50",
        "score >= 80 AND is_active = true",
        "name LIKE 'A%' AND age BETWEEN 20 AND 30",
        "age > 30 OR (score < 60 AND is_active = false)"
    ]
    
    for query in queries:
        @measure_time
        def execute_query():
            return client.query(query)
        
        _, duration = execute_query()
        print(f"\n{client.storage.__class__.__name__} 复杂查询 '{query}' 耗时: {duration:.4f}秒")
        assert duration < 1.0  # 确保复杂查询在1秒内完成

def test_concurrent_operations_performance(client):
    """测试并发操作性能"""
    import threading
    
    # 准备数据
    size = 1000
    data = generate_test_data(size)
    ids = client.store(data)
    
    # 并发操作函数
    def concurrent_operation(op_type):
        if op_type == 'query':
            client.retrieve(random.choice(ids))
        elif op_type == 'update':
            id_ = random.choice(ids)
            client.replace(id_, generate_test_data(1)[0])
        elif op_type == 'delete':
            id_ = random.choice(ids)
            client.delete(id_)
    
    # 创建多个线程执行不同操作
    threads = []
    operations = ['query', 'update', 'delete'] * 10  # 每种操作10个线程
    
    @measure_time
    def run_concurrent_operations():
        for op in operations:
            thread = threading.Thread(target=concurrent_operation, args=(op,))
            threads.append(thread)
            thread.start()
        
        for thread in threads:
            thread.join()
    
    _, duration = run_concurrent_operations()
    print(f"\n{client.storage.__class__.__name__} 30个并发操作耗时: {duration:.4f}秒")
    assert duration < 3.0  # 确保30个并发操作在3秒内完成

def test_large_batch_store_performance(client):
    """测试大规模批量存储性能"""
    batch_sizes = [100000, 1000000]  # 10万和100万条记录
    chunk_size = 10000  # 每次处理1万条记录
    
    for total_size in batch_sizes:
        total_duration = 0
        total_memory = 0
        processed_count = 0
        
        print(f"\n{client.storage.__class__.__name__} 开始测试 {total_size} 条记录的批量存储")
        
        # 分批处理数据
        while processed_count < total_size:
            current_chunk_size = min(chunk_size, total_size - processed_count)
            data = generate_test_data(current_chunk_size)
            
            @measure_performance
            def store_batch():
                return client.store(data)
            
            _, duration, memory = store_batch()
            total_duration += duration
            total_memory += memory
            processed_count += current_chunk_size
            
            print(f"已处理 {processed_count}/{total_size} 条记录")
            print(f"当前批次耗时: {duration:.4f}秒, 内存使用: {memory:.2f}MB")
        
        print(f"\n{client.storage.__class__.__name__} {total_size}条记录批量存储总耗时: {total_duration:.4f}秒")
        print(f"平均每条记录耗时: {(total_duration/total_size)*1000:.4f}毫秒")
        print(f"总内存使用: {total_memory:.2f}MB")
        
        # 性能断言
        assert total_duration < total_size * 0.0005  # 每条记录平均不超过0.5毫秒
        assert total_memory < 1024  # 总内存使用不超过1GB

def test_large_batch_query_performance(client):
    """测试大规模批量查询性能"""
    total_size = 100000  # 10万条记录
    chunk_size = 10000  # 每次查询1万条
    
    # 准备数据
    print(f"\n{client.storage.__class__.__name__} 准备 {total_size} 条测试数据")
    all_ids = []
    for i in range(0, total_size, chunk_size):
        current_chunk_size = min(chunk_size, total_size - i)
        data = generate_test_data(current_chunk_size)
        chunk_ids = client.store(data)
        all_ids.extend(chunk_ids)
        print(f"已准备 {len(all_ids)}/{total_size} 条记录")
    
    # 测试批量查询
    total_duration = 0
    total_memory = 0
    processed_count = 0
    
    print(f"\n{client.storage.__class__.__name__} 开始测试 {total_size} 条记录的批量查询")
    
    # 分批查询数据
    for i in range(0, total_size, chunk_size):
        current_chunk = all_ids[i:i + chunk_size]
        
        @measure_performance
        def query_batch():
            return client.retrieve_many(current_chunk)
        
        _, duration, memory = query_batch()
        total_duration += duration
        total_memory += memory
        processed_count += len(current_chunk)
        
        print(f"已查询 {processed_count}/{total_size} 条记录")
        print(f"当前批次耗时: {duration:.4f}秒, 内存使用: {memory:.2f}MB")
    
    print(f"\n{client.storage.__class__.__name__} {total_size}条记录批量查询总耗时: {total_duration:.4f}秒")
    print(f"平均每条记录耗时: {(total_duration/total_size)*1000:.4f}毫秒")
    print(f"总内存使用: {total_memory:.2f}MB")
    
    # 性能断言
    assert total_duration < total_size * 0.0002  # 每条记录平均不超过0.2毫秒
    assert total_memory < 1024  # 总内存使用不超过1GB

def test_large_batch_update_performance(client):
    """测试大规模批量更新性能"""
    total_size = 100000  # 10万条记录
    chunk_size = 10000  # 每次更新1万条
    
    # 准备数据
    print(f"\n{client.storage.__class__.__name__} 准备 {total_size} 条测试数据")
    all_ids = []
    for i in range(0, total_size, chunk_size):
        current_chunk_size = min(chunk_size, total_size - i)
        data = generate_test_data(current_chunk_size)
        chunk_ids = client.store(data)
        all_ids.extend(chunk_ids)
        print(f"已准备 {len(all_ids)}/{total_size} 条记录")
    
    # 测试批量更新
    total_duration = 0
    total_memory = 0
    processed_count = 0
    
    print(f"\n{client.storage.__class__.__name__} 开始测试 {total_size} 条记录的批量更新")
    
    # 分批更新数据
    for i in range(0, total_size, chunk_size):
        current_chunk = all_ids[i:i + chunk_size]
        update_data = {id_: record for id_, record in zip(current_chunk, generate_test_data(len(current_chunk)))}
        
        @measure_performance
        def update_batch():
            return client.batch_replace(update_data)
        
        _, duration, memory = update_batch()
        total_duration += duration
        total_memory += memory
        processed_count += len(current_chunk)
        
        print(f"已更新 {processed_count}/{total_size} 条记录")
        print(f"当前批次耗时: {duration:.4f}秒, 内存使用: {memory:.2f}MB")
    
    print(f"\n{client.storage.__class__.__name__} {total_size}条记录批量更新总耗时: {total_duration:.4f}秒")
    print(f"平均每条记录耗时: {(total_duration/total_size)*1000:.4f}毫秒")
    print(f"总内存使用: {total_memory:.2f}MB")
    
    # 性能断言
    assert total_duration < total_size * 0.001  # 每条记录平均不超过1毫秒
    assert total_memory < 1024  # 总内存使用不超过1GB 