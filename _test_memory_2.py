"""Test 2: Load existing database with memory profiling"""
import time
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

# 加载已存在的数据库
print("Loading existing database with 10 million records...")
load_start = time.time()

client = ApexClient(
    dirpath="./test_db_1m",
    drop_if_exists=False
)

load_time = time.time() - load_start
after_load_memory = get_memory_mb()
print(f"Database loaded in: {load_time:.2f}秒")
print(f"After loading database: {after_load_memory:.2f} MB")

# 查询测试
print("\nRunning query test...")
query_start = time.time()
results = client.query("title like 'Python%'").to_pandas()
query_time = time.time() - query_start
after_query_memory = get_memory_mb()
print(f"Query completed in: {query_time:.2f}秒")
print(f"Query result rows: {len(results)}")
print(f"After query: {after_query_memory:.2f} MB")

# Final summary
current, peak = tracemalloc.get_traced_memory()
tracemalloc.stop()

print("\n" + "="*50)
print("Memory Summary (Test 2 - Load Existing DB):")
print("="*50)
print(f"Initial memory:         {initial_memory:.2f} MB")
print(f"After import:           {after_import_memory:.2f} MB")
print(f"After loading database: {after_load_memory:.2f} MB")
print(f"After query:            {after_query_memory:.2f} MB")
print(f"Peak memory (process):  {get_memory_mb():.2f} MB")
print(f"Peak memory (traced):   {peak / 1024 / 1024:.2f} MB")
