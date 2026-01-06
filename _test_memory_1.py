"""Test 1: Generate and store 10 million records with memory profiling"""
import time
import random
import tracemalloc
import psutil
import os

def get_memory_mb():
    """Get current process memory usage in MB"""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024

# Start memory tracking
tracemalloc.start()
initial_memory = get_memory_mb()
print(f"Initial memory: {initial_memory:.2f} MB")

from apexbase import ApexClient

after_import_memory = get_memory_mb()
print(f"After import: {after_import_memory:.2f} MB")

# 创建客户端
client = ApexClient(
    dirpath="./test_db_1m",
    drop_if_exists=True
)

after_client_memory = get_memory_mb()
print(f"After client creation: {after_client_memory:.2f} MB")

print("开始生成100万条测试数据...")
start_time = time.time()

# 生成100万条测试数据
test_data = []
for i in range(1000000):
    test_data.append({
        "title": f"Python编程指南第{i+1}部分",
        "content": f"学习Python的最佳实践 - 第{i+1}章节，包含详细的编程技巧和实例代码。",
        "number": random.randint(0, 10000)
    })

generation_time = time.time() - start_time
after_generation_memory = get_memory_mb()
print(f"数据生成完成，耗时: {generation_time:.2f}秒")
print(f"After data generation: {after_generation_memory:.2f} MB")

# 存储数据（自动更新FTS索引）
print("开始存储1000万条文档...")
store_start = time.time()
for i in range(10):
    client.store(test_data)
    client.flush()
    current_memory = get_memory_mb()
    print(f"  Round {i+1}/10 complete, memory: {current_memory:.2f} MB")

store_time = time.time() - store_start
after_store_memory = get_memory_mb()
print(f"存储完成，耗时: {store_time:.2f}秒")
print(f"After storage: {after_store_memory:.2f} MB")

# 清理test_data释放内存
del test_data
import gc
gc.collect()

after_cleanup_memory = get_memory_mb()
print(f"After cleanup test_data: {after_cleanup_memory:.2f} MB")

# 查询
loaded_start = time.time()
results = client.query("title like 'Python%'").to_pandas()
loaded_time = time.time() - loaded_start
after_query_memory = get_memory_mb()
print(f"全量读取完成，耗时: {loaded_time:.2f}秒")
print(f"Query result rows: {len(results)}")
print(f"After query: {after_query_memory:.2f} MB")

# Final summary
current, peak = tracemalloc.get_traced_memory()
tracemalloc.stop()

print("\n" + "="*50)
print("Memory Summary (Test 1 - Generate & Store):")
print("="*50)
print(f"Initial memory:        {initial_memory:.2f} MB")
print(f"After import:          {after_import_memory:.2f} MB")
print(f"After client creation: {after_client_memory:.2f} MB")
print(f"After data generation: {after_generation_memory:.2f} MB")
print(f"After storage:         {after_store_memory:.2f} MB")
print(f"After cleanup:         {after_cleanup_memory:.2f} MB")
print(f"After query:           {after_query_memory:.2f} MB")
print(f"Peak memory (process): {get_memory_mb():.2f} MB")
print(f"Peak memory (traced):  {peak / 1024 / 1024:.2f} MB")
