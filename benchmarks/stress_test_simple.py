#!/usr/bin/env python3
"""
简化版压力测试：模拟离线批量处理和在线服务场景

Usage:
    conda run -n dev python benchmarks/stress_test_simple.py
"""

import os, sys, gc, time, tempfile, json, threading, random
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import List, Dict, Tuple

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'apexbase', 'python'))

try:
    from apexbase import ApexClient
    HAS_APEX = True
except ImportError:
    HAS_APEX = False
    print("ERROR: ApexBase not found")
    sys.exit(1)

# ── 配置 ──────────────────────────────────────────────────────────────────
@dataclass
class TestConfig:
    # 离线测试配置
    offline_batch_size: int = 1_000_000     # 100万行批量写入
    offline_query_iters: int = 10           # 查询迭代次数
    
    # 在线测试配置
    online_concurrent_threads: int = 20     # 并发线程数
    online_ops_per_thread: int = 500        # 每线程操作数
    
    # 数据配置
    n_cities: int = 10
    n_categories: int = 20

config = TestConfig()

def timer(func):
    def wrapper(*args, **kwargs):
        t0 = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - t0
        return result, elapsed
    return wrapper

def generate_test_data(n_rows: int, seed: int = 42) -> List[Dict]:
    """生成测试数据"""
    rng = np.random.default_rng(seed)
    
    cities = [f"City_{i}" for i in range(config.n_cities)]
    categories = [f"Category_{i}" for i in range(config.n_categories)]
    
    data = []
    for i in range(n_rows):
        record = {
            '_id': i + 1,
            'user_id': rng.integers(1, 50_000),
            'category': rng.choice(categories),
            'city': rng.choice(cities),
            'price': rng.uniform(10.0, 1000.0),
            'quantity': rng.integers(1, 100),
            'timestamp': int(time.time() - rng.integers(0, 86400 * 30)),
            'is_active': rng.choice([True, False], p=[0.8, 0.2]),
            'score': rng.uniform(0.0, 1.0)
        }
        data.append(record)
    
    return data

# ── 离线批量写入测试 ───────────────────────────────────────────────────────
@timer
def test_offline_batch_write(client: ApexClient, data: List[Dict]) -> Dict:
    """离线批量写入测试"""
    print(f"  批量写入 {len(data):,} 条记录...")
    
    # 分批写入
    batch_size = 25_000
    total_time = 0
    
    for i in range(0, len(data), batch_size):
        batch = data[i:i+batch_size]
        start_time = time.perf_counter()
        
        columns = {}
        for key in batch[0].keys():
            columns[key] = [record[key] for record in batch]
        
        client.store(columns)
        total_time += time.perf_counter() - start_time
        
        if (i // batch_size + 1) % 8 == 0:
            print(f"    已写入 {i + batch_size:,} / {len(data):,} 条记录")
    
    rows_per_sec = len(data) / total_time
    return {
        'rows_written': len(data),
        'total_time': total_time,
        'rows_per_sec': rows_per_sec
    }

# ── 离线查询测试 ───────────────────────────────────────────────────────────
@timer
def test_offline_queries(client: ApexClient) -> Dict:
    """离线查询测试"""
    queries = [
        "SELECT city, COUNT(*) as cnt, AVG(price) as avg_price FROM sales_data GROUP BY city ORDER BY cnt DESC",
        "SELECT category, COUNT(*) as total_count, AVG(price) as avg_price FROM sales_data GROUP BY category ORDER BY total_count DESC",
        "SELECT COUNT(*) as total_orders, SUM(price) as total_revenue FROM sales_data WHERE is_active = true",
        "SELECT city, AVG(score) as avg_score FROM sales_data WHERE price > 100 GROUP BY city ORDER BY avg_score DESC",
        "SELECT * FROM sales_data WHERE price BETWEEN 50 AND 200 ORDER BY price DESC LIMIT 1000"
    ]
    
    query_times = []
    
    for i, query in enumerate(queries):
        print(f"    执行查询 {i+1}/{len(queries)}...")
        
        times = []
        for _ in range(config.offline_query_iters):
            gc.collect()
            start = time.perf_counter()
            result = client.execute(query)
            elapsed = time.perf_counter() - start
            times.append(elapsed)
        
        query_times.extend(times)
        avg_time = sum(times) / len(times)
        print(f"      平均: {avg_time*1000:.1f}ms")
    
    return {
        'total_queries': len(queries) * config.offline_query_iters,
        'avg_query_time': sum(query_times) / len(query_times),
        'queries_per_sec': 1.0 / (sum(query_times) / len(query_times))
    }

# ── 在线并发测试 ───────────────────────────────────────────────────────────
def worker_thread(thread_id: int, db_path: str, ops_count: int) -> Dict:
    """工作线程函数"""
    client = ApexClient(db_path)
    results = {
        'thread_id': thread_id,
        'operations': 0,
        'errors': 0,
        'read_times': [],
        'write_times': [],
        'total_time': 0
    }
    
    start_time = time.perf_counter()
    
    try:
        # 选择表
        client.use_table('sales_data')
        
        for i in range(ops_count):
            op_start = time.perf_counter()
            
            if random.random() < 0.8:  # 80%读操作
                # 点查询
                if random.random() < 0.6:
                    result = client.execute(f"SELECT * FROM sales_data WHERE _id = {random.randint(1, 100000)}")
                else:
                    # 简单聚合
                    result = client.execute("SELECT COUNT(*) FROM sales_data WHERE city = 'City_1'")
                
                results['read_times'].append(time.perf_counter() - op_start)
            else:  # 20%写操作
                new_record = {
                    'user_id': random.randint(1, 50000),
                    'category': f"Category_{random.randint(1, 20)}",
                    'city': f"City_{random.randint(1, 10)}",
                    'price': random.uniform(10.0, 1000.0),
                    'quantity': random.randint(1, 100),
                    'timestamp': int(time.time()),
                    'is_active': random.choice([True, False]),
                    'score': random.uniform(0.0, 1.0)
                }
                
                client.store(new_record)
                results['write_times'].append(time.perf_counter() - op_start)
            
            results['operations'] += 1
            
    except Exception as e:
        results['errors'] += 1
        print(f"Thread {thread_id} error: {e}")
    finally:
        try:
            client.close()
        except:
            pass
        results['total_time'] = time.perf_counter() - start_time
    
    return results

@timer
def test_online_concurrent(db_path: str) -> Dict:
    """在线高并发测试"""
    print(f"  启动 {config.online_concurrent_threads} 个并发线程...")
    
    with ThreadPoolExecutor(max_workers=config.online_concurrent_threads) as executor:
        futures = []
        for i in range(config.online_concurrent_threads):
            future = executor.submit(worker_thread, i, db_path, config.online_ops_per_thread)
            futures.append(future)
        
        all_results = []
        for future in as_completed(futures):
            result = future.result()
            all_results.append(result)
    
    total_ops = sum(r['operations'] for r in all_results)
    total_errors = sum(r['errors'] for r in all_results)
    all_read_times = []
    all_write_times = []
    
    for r in all_results:
        all_read_times.extend(r['read_times'])
        all_write_times.extend(r['write_times'])
    
    return {
        'threads': config.online_concurrent_threads,
        'total_operations': total_ops,
        'total_errors': total_errors,
        'error_rate': total_errors / total_ops if total_ops > 0 else 0,
        'ops_per_sec': total_ops / sum(r['total_time'] for r in all_results),
        'avg_read_time': sum(all_read_times) / len(all_read_times) if all_read_times else 0,
        'avg_write_time': sum(all_write_times) / len(all_write_times) if all_write_times else 0,
        'p95_read_time': np.percentile(all_read_times, 95) if all_read_times else 0,
        'p95_write_time': np.percentile(all_write_times, 95) if all_write_times else 0
    }

# ── 主测试流程 ─────────────────────────────────────────────────────────────
def main():
    print("\n" + "="*80)
    print("  ApexBase 简化压力测试")
    print("="*80 + "\n")
    
    tmpdir = tempfile.mkdtemp(prefix="apex_stress_simple_")
    db_path = os.path.join(tmpdir, "stress_test.apex")
    
    try:
        client = ApexClient(db_path)
        
        print(f"测试目录: {tmpdir}")
        print(f"数据库路径: {db_path}\n")
        
        # 创建表
        client.create_table('sales_data')
        
        # ── 1. 离线批量写入测试 ───────────────────────────────────────
        print("─" * 60)
        print("1. 离线批量写入测试")
        print("─" * 60)
        
        print("生成测试数据...")
        data = generate_test_data(config.offline_batch_size)
        print(f"  数据量: {len(data):,} 条记录")
        
        write_result, write_time = test_offline_batch_write(client, data)
        print(f"\n  写入性能:")
        print(f"    总时间: {write_time:.2f}s")
        print(f"    写入速度: {write_result['rows_per_sec']:,.0f} rows/sec")
        
        db_size = os.path.getsize(db_path) / 1e6
        print(f"    数据库大小: {db_size:.1f} MB")
        
        # ── 2. 离线查询测试 ───────────────────────────────────────────
        print("\n─" * 60)
        print("2. 离线查询测试")
        print("─" * 60)
        
        query_result, query_time = test_offline_queries(client)
        print(f"\n  查询性能:")
        print(f"    总查询数: {query_result['total_queries']}")
        print(f"    平均查询时间: {query_result['avg_query_time']*1000:.1f}ms")
        print(f"    查询吞吐量: {query_result['queries_per_sec']:.1f} queries/sec")
        
        # ── 3. 在线并发测试 ───────────────────────────────────────────
        print("\n─" * 60)
        print("3. 在线并发测试")
        print("─" * 60)
        
        concurrent_result, concurrent_time = test_online_concurrent(db_path)
        print(f"\n  并发性能:")
        print(f"    总操作数: {concurrent_result['total_operations']:,}")
        print(f"    错误数: {concurrent_result['total_errors']}")
        print(f"    错误率: {concurrent_result['error_rate']*100:.2f}%")
        print(f"    操作吞吐量: {concurrent_result['ops_per_sec']:.0f} ops/sec")
        print(f"    平均读延迟: {concurrent_result['avg_read_time']*1000:.1f}ms")
        print(f"    平均写延迟: {concurrent_result['avg_write_time']*1000:.1f}ms")
        print(f"    P95读延迟: {concurrent_result['p95_read_time']*1000:.1f}ms")
        print(f"    P95写延迟: {concurrent_result['p95_write_time']*1000:.1f}ms")
        
        # ── 总体评估 ─────────────────────────────────────────────────────
        print("\n─" * 60)
        print("总体评估")
        print("─" * 60)
        
        print(f"性能指标:")
        print(f"  批量写入: {write_result['rows_per_sec']:,.0f} rows/sec")
        print(f"  查询性能: {query_result['queries_per_sec']:.1f} queries/sec") 
        print(f"  并发吞吐: {concurrent_result['ops_per_sec']:.0f} ops/sec")
        print(f"  读延迟: {concurrent_result['avg_read_time']*1000:.1f}ms")
        print(f"  写延迟: {concurrent_result['avg_write_time']*1000:.1f}ms")
        
        # 性能评估
        grade = "A+"
        issues = []
        
        if write_result['rows_per_sec'] < 100_000:
            grade = "B"
            issues.append("写入性能偏低")
        
        if query_result['avg_query_time'] > 0.05:  # 50ms
            grade = "B"
            issues.append("查询延迟偏高")
        
        if concurrent_result['error_rate'] > 0.01:
            grade = "C"
            issues.append("并发错误率过高")
        
        if concurrent_result['avg_read_time'] > 0.01:  # 10ms
            grade = "B"
            issues.append("读延迟偏高")
        
        print(f"\n性能等级: {grade}")
        if issues:
            print(f"需要改进: {', '.join(issues)}")
        else:
            print("✅ 所有性能指标表现优秀")
        
        client.close()
        
    finally:
        import shutil
        shutil.rmtree(tmpdir, ignore_errors=True)

if __name__ == "__main__":
    main()
