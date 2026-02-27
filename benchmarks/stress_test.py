#!/usr/bin/env python3
"""
压力测试：模拟离线批量处理和在线服务场景

测试场景：
1. 离线批量写入 - 大数据量ETL
2. 离线复杂查询 - 分析型工作负载
3. 在线高并发读写 - 混合工作负载
4. 在线延迟敏感 - 实时查询

Usage:
    conda run -n dev python benchmarks/stress_test.py
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
    offline_batch_size: int = 5_000_000    # 500万行批量写入
    offline_query_iters: int = 20           # 复杂查询迭代次数
    
    # 在线测试配置
    online_concurrent_threads: int = 50     # 并发线程数
    online_ops_per_thread: int = 1000       # 每线程操作数
    online_duration_seconds: int = 60       # 持续时间测试
    
    # 数据配置
    n_cities: int = 20
    n_categories: int = 50
    n_products: int = 1000

config = TestConfig()

# ── 工具函数 ────────────────────────────────────────────────────────────────
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
    products = [f"Product_{i}" for i in range(config.n_products)]
    
    data = []
    for i in range(n_rows):
        record = {
            '_id': i + 1,
            'user_id': rng.integers(1, 100_000),
            'product_id': rng.choice(config.n_products),
            'category': rng.choice(categories),
            'city': rng.choice(cities),
            'price': rng.uniform(10.0, 1000.0),
            'quantity': rng.integers(1, 100),
            'timestamp': int(time.time() - rng.integers(0, 86400 * 30)), # 30天内
            'is_active': rng.choice([True, False], p=[0.8, 0.2]),
            'score': rng.uniform(0.0, 1.0)
        }
        data.append(record)
    
    return data

def format_bytes(bytes_val: float) -> str:
    """格式化字节数"""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if bytes_val < 1024.0:
            return f"{bytes_val:.1f}{unit}"
        bytes_val /= 1024.0
    return f"{bytes_val:.1f}TB"

# ── 离线批量写入测试 ───────────────────────────────────────────────────────
@timer
def test_offline_batch_write(client: ApexClient, data: List[Dict]) -> Dict:
    """离线批量写入测试"""
    print(f"  批量写入 {len(data):,} 条记录...")
    
    # 分批写入以避免内存压力
    batch_size = 50_000
    total_time = 0
    
    for i in range(0, len(data), batch_size):
        batch = data[i:i+batch_size]
        start_time = time.perf_counter()
        
        # 使用store_columnar批量写入
        columns = {}
        for key in batch[0].keys():
            columns[key] = [record[key] for record in batch]
        
        client.store(columns)
        total_time += time.perf_counter() - start_time
        
        if (i // batch_size + 1) % 10 == 0:
            print(f"    已写入 {i + batch_size:,} / {len(data):,} 条记录")
    
    rows_per_sec = len(data) / total_time
    return {
        'rows_written': len(data),
        'total_time': total_time,
        'rows_per_sec': rows_per_sec
    }

# ── 离线复杂查询测试 ───────────────────────────────────────────────────────
@timer
def test_offline_complex_queries(client: ApexClient) -> Dict:
    """离线复杂查询测试"""
    # 获取当前时间戳（毫秒）
    import time
    current_ts = int(time.time() * 1000)
    thirty_days_ago = current_ts - (30 * 24 * 60 * 60 * 1000)
    seven_days_ago = current_ts - (7 * 24 * 60 * 60 * 1000)
    fourteen_days_ago = current_ts - (14 * 24 * 60 * 60 * 1000)
    
    queries = [
        # 聚合查询
        "SELECT city, COUNT(*) as cnt, AVG(price) as avg_price FROM sales_data GROUP BY city ORDER BY cnt DESC",
        
        # 复杂聚合查询 - 使用直接时间戳比较替代strftime
        f"""
        SELECT category, 
               COUNT(*) as total_count,
               AVG(price) as avg_price
        FROM sales_data 
        WHERE timestamp > {seven_days_ago}
        GROUP BY category
        HAVING COUNT(*) > 100
        ORDER BY avg_price DESC
        """,
        
        # 时间窗口分析 - 简化查询，不使用乘法运算
        f"""
        SELECT 
            (timestamp / 86400000) as day_of_epoch,
            COUNT(*) as daily_orders,
            AVG(price) as avg_price,
            COUNT(DISTINCT user_id) as unique_users
        FROM sales_data
        WHERE timestamp >= {thirty_days_ago}
        GROUP BY day_of_epoch
        ORDER BY day_of_epoch
        """,
        
        # TOP-K分析
        """
        SELECT product_id, SUM(price) as total_price
        FROM sales_data
        WHERE is_active = true
        GROUP BY product_id
        ORDER BY total_price DESC
        LIMIT 20
        """,
        
        # 复杂过滤条件 - 使用直接时间戳比较
        f"""
        SELECT city, category, AVG(score) as avg_score, COUNT(*) as cnt
        FROM sales_data
        WHERE price BETWEEN 100 AND 500 
          AND quantity > 10 
          AND is_active = true
          AND timestamp > {fourteen_days_ago}
        GROUP BY city, category
        HAVING cnt > 50
        ORDER BY avg_score DESC, cnt DESC
        """
    ]
    
    query_times = []
    query_results = {}
    
    for i, query in enumerate(queries):
        print(f"    执行查询 {i+1}/{len(queries)}...")
        
        times = []
        for _ in range(config.offline_query_iters):
            gc.collect()
            start = time.perf_counter()
            result = client.execute(query)
            elapsed = time.perf_counter() - start
            times.append(elapsed)
            
            # 保存第一次结果用于验证
            if len(query_results) < len(queries):
                if hasattr(result, 'to_pandas'):
                    df = result.to_pandas()
                    query_results[f"query_{i+1}"] = len(df)
                else:
                    query_results[f"query_{i+1}"] = len(list(result))
        
        query_times.extend(times)
        avg_time = sum(times) / len(times)
        min_time = min(times)
        print(f"      平均: {avg_time*1000:.1f}ms, 最快: {min_time*1000:.1f}ms")
    
    return {
        'total_queries': len(queries) * config.offline_query_iters,
        'avg_query_time': sum(query_times) / len(query_times),
        'min_query_time': min(query_times),
        'max_query_time': max(query_times),
        'queries_per_sec': 1.0 / (sum(query_times) / len(query_times)),
        'query_results': query_results
    }

# ── 在线高并发测试 ─────────────────────────────────────────────────────────
def worker_thread(thread_id: int, db_path: str, ops_count: int) -> Dict:
    """工作线程函数"""
    results = {
        'thread_id': thread_id,
        'operations': 0,
        'errors': 0,
        'read_times': [],
        'write_times': [],
        'total_time': 0
    }
    
    start_time = time.perf_counter()
    
    # 每个线程创建独立的客户端连接
    client = None
    try:
        client = ApexClient(db_path)
        client.use_table('sales_data')
        
        for i in range(ops_count):
            op_start = time.perf_counter()
            
            # 随机选择操作类型
            if random.random() < 0.7:  # 70%读操作
                # 点查询
                if random.random() < 0.5:
                    result = client.execute(f"SELECT * FROM sales_data WHERE _id = {random.randint(1, 100000)}")
                else:
                    # 范围查询
                    result = client.execute(f"SELECT * FROM sales_data WHERE price > {random.uniform(100, 800)} LIMIT 100")
                
                results['read_times'].append(time.perf_counter() - op_start)
            else:  # 30%写操作
                # 插入新记录
                new_record = {
                    'user_id': random.randint(1, 100000),
                    'product_id': random.randint(1, 1000),
                    'category': f"Category_{random.randint(1, 50)}",
                    'city': f"City_{random.randint(1, 20)}",
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
        if client:
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
        
        # 收集结果
        all_results = []
        for future in as_completed(futures):
            result = future.result()
            all_results.append(result)
    
    # 汇总统计
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

# ── 在线延迟敏感测试 ───────────────────────────────────────────────────────
@timer
def test_online_latency_sensitive(client: ApexClient) -> Dict:
    """在线延迟敏感测试 - 模拟实时查询场景"""
    
    # 预热
    for _ in range(10):
        client.execute("SELECT COUNT(*) FROM sales_data")
    
    latency_tests = [
        ("点查询", lambda: client.execute(f"SELECT * FROM sales_data WHERE _id = {random.randint(1, 100000)}")),
        ("计数查询", lambda: client.execute("SELECT COUNT(*) FROM sales_data")),
        ("简单过滤", lambda: client.execute(f"SELECT * FROM sales_data WHERE city = 'City_1' LIMIT 10")),
        ("排序TOP-K", lambda: client.execute("SELECT * FROM sales_data ORDER BY price DESC LIMIT 10")),
        ("聚合查询", lambda: client.execute("SELECT city, COUNT(*) FROM sales_data GROUP BY city LIMIT 5"))
    ]
    
    results = {}
    
    for test_name, test_func in latency_tests:
        times = []
        for _ in range(100):  # 100次测试
            gc.collect()
            start = time.perf_counter()
            test_func()
            times.append(time.perf_counter() - start)
        
        results[test_name] = {
            'avg_ms': sum(times) / len(times) * 1000,
            'min_ms': min(times) * 1000,
            'max_ms': max(times) * 1000,
            'p50_ms': np.percentile(times, 50) * 1000,
            'p95_ms': np.percentile(times, 95) * 1000,
            'p99_ms': np.percentile(times, 99) * 1000
        }
    
    return results

# ── 主测试流程 ─────────────────────────────────────────────────────────────
def main():
    print("\n" + "="*80)
    print("  ApexBase 压力测试 - 离线 vs 在线工作负载")
    print("="*80 + "\n")
    
    tmpdir = tempfile.mkdtemp(prefix="apex_stress_")
    db_path = os.path.join(tmpdir, "stress_test.apex")
    
    try:
        # 初始化客户端
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
        data_size_mb = len(str(data).encode()) / 1e6  # 粗略估算
        print(f"  数据量: {len(data):,} 条记录, 约 {data_size_mb:.1f} MB")
        
        write_result, write_time = test_offline_batch_write(client, data)
        print(f"\n  写入性能:")
        print(f"    总时间: {write_time:.2f}s")
        print(f"    写入速度: {write_result['rows_per_sec']:,.0f} rows/sec")
        print(f"    吞吐量: {format_bytes(data_size_mb / write_time)}/sec")
        
        # 获取数据库文件大小
        db_size = os.path.getsize(db_path) / 1e6
        print(f"    数据库大小: {db_size:.1f} MB")
        
        # ── 2. 离线复杂查询测试 ───────────────────────────────────────
        print("\n─" * 60)
        print("2. 离线复杂查询测试")
        print("─" * 60)
        
        query_result, query_time = test_offline_complex_queries(client)
        print(f"\n  查询性能:")
        print(f"    总查询数: {query_result['total_queries']}")
        print(f"    平均查询时间: {query_result['avg_query_time']*1000:.1f}ms")
        print(f"    最快查询: {query_result['min_query_time']*1000:.1f}ms")
        print(f"    查询吞吐量: {query_result['queries_per_sec']:.1f} queries/sec")
        
        # ── 3. 在线高并发测试 ───────────────────────────────────────────
        print("\n─" * 60)
        print("3. 在线高并发测试")
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
        
        # ── 4. 在线延迟敏感测试 ───────────────────────────────────────
        print("\n─" * 60)
        print("4. 在线延迟敏感测试")
        print("─" * 60)
        
        latency_result, latency_time = test_online_latency_sensitive(client)
        print(f"\n  延迟分析:")
        for test_name, metrics in latency_result.items():
            print(f"    {test_name}:")
            print(f"      平均: {metrics['avg_ms']:.1f}ms")
            print(f"      P95: {metrics['p95_ms']:.1f}ms")
            print(f"      P99: {metrics['p99_ms']:.1f}ms")
        
        # ── 总体评估 ─────────────────────────────────────────────────────
        print("\n─" * 60)
        print("总体评估")
        print("─" * 60)
        
        print(f"测试环境:")
        print(f"  数据规模: {config.offline_batch_size:,} 行")
        print(f"  并发线程: {config.online_concurrent_threads}")
        print(f"  每线程操作: {config.online_ops_per_thread}")
        
        print(f"\n性能指标:")
        print(f"  批量写入: {write_result['rows_per_sec']:,.0f} rows/sec")
        print(f"  复杂查询: {query_result['queries_per_sec']:.1f} queries/sec") 
        print(f"  并发吞吐: {concurrent_result['ops_per_sec']:.0f} ops/sec")
        print(f"  点查询延迟: {latency_result['点查询']['avg_ms']:.1f}ms")
        
        # 性能等级评估
        grade = "A+"
        issues = []
        
        if write_result['rows_per_sec'] < 100_000:
            grade = "B"
            issues.append("写入性能偏低")
        
        if query_result['avg_query_time'] > 0.1:  # 100ms
            grade = "B"
            issues.append("复杂查询延迟偏高")
        
        if concurrent_result['error_rate'] > 0.01:  # 1%
            grade = "C"
            issues.append("并发错误率过高")
        
        if latency_result['点查询']['p95_ms'] > 10:  # 10ms
            grade = "B"
            issues.append("点查询延迟偏高")
        
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
