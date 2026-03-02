#!/usr/bin/env python3
"""
ApexBase 查询吞吐量 Benchmark: Q/s 测试

测量指标:
1. Single Query Q/s - 单线程顺序查询吞吐量
2. Concurrent Q/s - 多线程并发查询吞吐量

Usage:
    python benchmarks/bench_throughput.py [--rows N] [--iterations N] [--threads N]
"""

import argparse
import gc
import os
import random
import sys
import tempfile
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'apexbase', 'python'))

try:
    from apexbase import ApexClient
    HAS_APEX = True
except ImportError:
    HAS_APEX = False
    print("ERROR: ApexBase not found")

import pandas as pd

try:
    import duckdb
    HAS_DUCKDB = True
except ImportError:
    HAS_DUCKDB = False

try:
    import sqlite3
    HAS_SQLITE = True
except ImportError:
    HAS_SQLITE = False


# ── 配置 ─────────────────────────────────────────────────────────────────────
@dataclass
class BenchConfig:
    rows: int = 1_000_000          # 数据行数
    iterations: int = 10            # 查询迭代次数
    threads: int = 8                # 并发线程数
    warmup: int = 2                 # 预热次数
    batch_size: int = 25_000        # 写入批次大小


config = BenchConfig()

# 测试数据
CITIES = ["Beijing", "Shanghai", "Guangzhou", "Shenzhen", "Hangzhou"]
CATEGORIES = ["Electronics", "Clothing", "Food", "Sports", "Books"]


# ── 数据生成 ─────────────────────────────────────────────────────────────────
def generate_data(n: int) -> List[Dict]:
    """生成测试数据"""
    rng = random.Random(42)
    data = []
    for i in range(n):
        data.append({
            '_id': i + 1,
            'user_id': rng.randint(1, 50000),
            'category': rng.choice(CATEGORIES),
            'city': rng.choice(CITIES),
            'price': round(rng.uniform(10.0, 1000.0), 2),
            'quantity': rng.randint(1, 100),
            'is_active': rng.choice([True, False]),
            'score': rng.uniform(0.0, 100.0)
        })
    return data


def data_to_columns(data: List[Dict]) -> Dict:
    """将记录列表转换为列格式"""
    if not data:
        return {}
    columns = {}
    for key in data[0].keys():
        columns[key] = [record[key] for record in data]
    return columns


# ── 查询集 ─────────────────────────────────────────────────────────────────
# 不同复杂度的查询
QUERIES_SIMPLE = [
    "SELECT COUNT(*) FROM test_table",
    "SELECT SUM(price) FROM test_table",
    "SELECT AVG(score) FROM test_table",
]

QUERIES_MEDIUM = [
    "SELECT city, COUNT(*) FROM test_table GROUP BY city",
    "SELECT category, AVG(price) FROM test_table GROUP BY category",
    "SELECT city, category, SUM(quantity) FROM test_table GROUP BY city, category",
]

QUERIES_COMPLEX = [
    "SELECT city, COUNT(*) as cnt, AVG(price) as avg_price FROM test_table GROUP BY city ORDER BY cnt DESC",
    "SELECT category, COUNT(*) as total, AVG(score) as avg_score FROM test_table WHERE price > 100 GROUP BY category",
    "SELECT * FROM test_table WHERE price > 500 ORDER BY price DESC LIMIT 100",
]


# ── Benchmark 基类 ─────────────────────────────────────────────────────────
class BaseBenchmark:
    def __init__(self, tmpdir: str, data: List[Dict]):
        self.tmpdir = tmpdir
        self.data = data
        self.n = len(data)
    
    def setup(self):
        raise NotImplementedError
    
    def execute(self, query: str) -> float:
        """执行查询并返回耗时(秒)"""
        raise NotImplementedError
    
    def close(self):
        pass
    
    def name(self) -> str:
        return self.__class__.__name__


# ── ApexBase Benchmark ─────────────────────────────────────────────────────
class ApexBaseBench(BaseBenchmark):
    def __init__(self, tmpdir: str, data: List[Dict]):
        super().__init__(tmpdir, data)
        self.client = None
    
    def setup(self):
        self.client = ApexClient(self.tmpdir)
        self.client.create_table('test_table')
        
        # 批量写入数据
        batch_size = config.batch_size
        for i in range(0, len(self.data), batch_size):
            batch = self.data[i:i+batch_size]
            columns = data_to_columns(batch)
            self.client.store(columns)
        
        # 预热
        for _ in range(config.warmup):
            self.client.execute("SELECT COUNT(*) FROM test_table").to_pandas()
    
    def execute(self, query: str) -> float:
        t0 = time.perf_counter()
        result = self.client.execute(query)
        result.to_pandas()  # 确保结果被消费
        return time.perf_counter() - t0
    
    def close(self):
        if self.client:
            self.client.close()


# ── DuckDB Benchmark ─────────────────────────────────────────────────────
class DuckDBBench(BaseBenchmark):
    def __init__(self, tmpdir: str, data: List[Dict]):
        super().__init__(tmpdir, data)
        self.conn = None
    
    def setup(self):
        db_path = os.path.join(self.tmpdir, "duckdb.db")
        self.conn = duckdb.connect(db_path)
        
        # 创建表
        self.conn.execute("""
            CREATE TABLE test_table (
                _id INTEGER,
                user_id INTEGER,
                category VARCHAR,
                city VARCHAR,
                price DOUBLE,
                quantity INTEGER,
                is_active BOOLEAN,
                score DOUBLE
            )
        """)
        
        # 写入数据
        df = pd.DataFrame(self.data)
        self.conn.execute("INSERT INTO test_table SELECT * FROM df")
        
        # 预热
        for _ in range(config.warmup):
            self.conn.execute("SELECT COUNT(*) FROM test_table").fetchall()
    
    def execute(self, query: str) -> float:
        t0 = time.perf_counter()
        result = self.conn.execute(query).fetchall()
        return time.perf_counter() - t0
    
    def close(self):
        if self.conn:
            self.conn.close()


# ── SQLite Benchmark ─────────────────────────────────────────────────────
class SQLiteBench(BaseBenchmark):
    def __init__(self, tmpdir: str, data: List[Dict]):
        super().__init__(tmpdir, data)
        self.conn = None
    
    def setup(self):
        db_path = os.path.join(self.tmpdir, "sqlite.db")
        self.conn = sqlite3.connect(db_path)
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA synchronous=NORMAL")
        
        # 创建表
        self.conn.execute("""
            CREATE TABLE test_table (
                _id INTEGER PRIMARY KEY,
                user_id INTEGER,
                category TEXT,
                city TEXT,
                price REAL,
                quantity INTEGER,
                is_active INTEGER,
                score REAL
            )
        """)
        
        # 写入数据
        for record in self.data:
            self.conn.execute("""
                INSERT INTO test_table VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (record['_id'], record['user_id'], record['category'], 
                  record['city'], record['price'], record['quantity'],
                  1 if record['is_active'] else 0, record['score']))
        self.conn.commit()
        
        # 预热
        for _ in range(config.warmup):
            self.conn.execute("SELECT COUNT(*) FROM test_table").fetchall()
    
    def execute(self, query: str) -> float:
        # 转换 is_active 字段
        query = query.replace('is_active = true', 'is_active = 1')
        query = query.replace('is_active = false', 'is_active = 0')
        
        t0 = time.perf_counter()
        result = self.conn.execute(query).fetchall()
        return time.perf_counter() - t0
    
    def close(self):
        if self.conn:
            self.conn.close()


# ── 吞吐量测试 ─────────────────────────────────────────────────────────────
def run_single_query_throughput(bench: BaseBenchmark, queries: List[str]) -> Dict:
    """
    单线程顺序查询吞吐量测试
    
    测量方式: 顺序执行 N 次查询，统计每秒可执行多少次
    """
    print(f"\n  {bench.name()}: 单线程顺序查询吞吐量测试")
    
    # 预热
    for q in queries[:1]:
        bench.execute(q)
    
    # 执行测试
    total_queries = len(queries) * config.iterations
    
    # 计时
    gc.collect()
    t0 = time.perf_counter()
    
    for _ in range(config.iterations):
        for query in queries:
            bench.execute(query)
    
    elapsed = time.perf_counter() - t0
    
    qps = total_queries / elapsed
    
    print(f"    总查询数: {total_queries}, 耗时: {elapsed:.3f}s, Q/s: {qps:.1f}")
    
    return {
        'total_queries': total_queries,
        'elapsed_sec': elapsed,
        'qps': qps
    }


def run_concurrent_throughput(bench_class, tmpdir: str, data: List[Dict], 
                              queries: List[str], num_threads: int) -> Dict:
    """
    并发查询吞吐量测试
    
    测量方式: 多线程同时执行查询，统计系统每秒可处理的总查询数
    """
    print(f"\n  并发查询吞吐量测试 ({num_threads} 线程)")
    
    # 为每个线程创建独立的客户端/连接
    results_lock = threading.Lock()
    total_executed = [0]
    total_time = [0.0]
    errors = [0]
    
    def worker(thread_id: int):
        try:
            # 每个线程使用独立目录
            thread_dir = os.path.join(tmpdir, f'thread_{thread_id}')
            os.makedirs(thread_dir, exist_ok=True)
            
            # 创建独立的连接
            if bench_class == ApexBaseBench:
                worker_bench = ApexBaseBench(thread_dir, data)
            elif bench_class == DuckDBBench:
                worker_bench = DuckDBBench(thread_dir, data)
            elif bench_class == SQLiteBench:
                worker_bench = SQLiteBench(thread_dir, data)
            else:
                return
            
            worker_bench.setup()
            
            # 每个线程执行相同数量的查询
            queries_per_thread = config.iterations * len(queries)
            
            thread_start = time.perf_counter()
            for _ in range(config.iterations):
                for query in queries:
                    exec_time = worker_bench.execute(query)
                    with results_lock:
                        total_executed[0] += 1
                        total_time[0] += exec_time
            
            thread_elapsed = time.perf_counter() - thread_start
            worker_bench.close()
            
            return {
                'thread_id': thread_id,
                'queries': total_executed[0],
                'elapsed': thread_elapsed
            }
        except Exception as e:
            with results_lock:
                errors[0] += 1
            return {'thread_id': thread_id, 'error': str(e)}
    
    # 预热 - 先运行一次
    warmup_bench = bench_class(tmpdir, data)
    warmup_bench.setup()
    for q in queries[:1]:
        warmup_bench.execute(q)
    warmup_bench.close()
    
    # 并发测试
    gc.collect()
    
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = [executor.submit(worker, i) for i in range(num_threads)]
        worker_results = [f.result() for f in as_completed(futures)]
    
    # 计算 Q/s - 使用实际墙钟时间
    wall_clock_time = max([r.get('elapsed', 0) for r in worker_results if 'elapsed' in r])
    
    if wall_clock_time > 0:
        # 总查询数
        total_queries = num_threads * config.iterations * len(queries)
        concurrent_qps = total_queries / wall_clock_time
    else:
        concurrent_qps = 0
    
    print(f"    总查询数: {total_queries}, 墙钟时间: {wall_clock_time:.3f}s, 并发Q/s: {concurrent_qps:.1f}")
    
    return {
        'threads': num_threads,
        'total_queries': total_queries,
        'wall_clock_sec': wall_clock_time,
        'concurrent_qps': concurrent_qps,
        'errors': errors[0]
    }


def run_concurrent_sequential(bench_class, tmpdir: str, data: List[Dict],
                              queries: List[str], num_threads: int) -> Dict:
    """
    并发测试 - 顺序执行但多线程
    
    每个线程独立顺序执行所有查询，统计所有线程的总吞吐量
    """
    print(f"\n  并发顺序测试 ({num_threads} 线程, 每线程 {config.iterations} 轮)")
    
    results = []
    
    def worker(thread_id: int):
        try:
            # 每个线程使用独立目录
            thread_dir = os.path.join(tmpdir, f'thread_{thread_id}')
            os.makedirs(thread_dir, exist_ok=True)
            
            # 每个线程使用独立连接
            if bench_class == ApexBaseBench:
                bench = ApexBaseBench(thread_dir, data)
            elif bench_class == DuckDBBench:
                bench = DuckDBBench(thread_dir, data)
            elif bench_class == SQLiteBench:
                bench = SQLiteBench(thread_dir, data)
            else:
                return None
            
            bench.setup()
            
            # 记录每个查询的时间
            query_times = []
            
            for _ in range(config.iterations):
                for query in queries:
                    t0 = time.perf_counter()
                    bench.execute(query)
                    query_times.append(time.perf_counter() - t0)
            
            bench.close()
            
            return {
                'thread_id': thread_id,
                'queries': len(query_times) * config.iterations,
                'times': query_times
            }
        except Exception as e:
            return {'thread_id': thread_id, 'error': str(e)}
    
    # 预热 - 跳过，因为每个worker已经有自己的setup
    # (每个线程都会创建自己的独立目录和数据)
    
    # 并发执行
    gc.collect()
    t0 = time.perf_counter()
    
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = [executor.submit(worker, i) for i in range(num_threads)]
        worker_results = [f.result() for f in as_completed(futures)]
    
    wall_clock = time.perf_counter() - t0
    
    # 计算总查询数和平均Q/s
    total_queries = sum([r.get('queries', 0) for r in worker_results if 'queries' in r])
    
    # 所有查询的平均延迟
    all_times = []
    for r in worker_results:
        if 'times' in r:
            all_times.extend(r['times'])
    
    avg_latency = sum(all_times) / len(all_times) if all_times else 0
    
    # 真正的并发Q/s = 总查询数 / 墙钟时间
    concurrent_qps = total_queries / wall_clock if wall_clock > 0 else 0
    
    # 理想Q/s (如果完全并发) = 每线程Q/s * 线程数
    # 实际Q/s反映的是整体系统吞吐量
    
    print(f"    总查询数: {total_queries}")
    print(f"    墙钟时间: {wall_clock:.3f}s")
    print(f"    平均延迟: {avg_latency*1000:.2f}ms")
    print(f"    并发Q/s: {concurrent_qps:.1f}")
    
    return {
        'threads': num_threads,
        'total_queries': total_queries,
        'wall_clock_sec': wall_clock,
        'avg_latency_ms': avg_latency * 1000,
        'concurrent_qps': concurrent_qps
    }


# ── 主函数 ─────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description='ApexBase Q/s Benchmark')
    parser.add_argument('--rows', type=int, default=config.rows, 
                        help=f'数据行数 (default: {config.rows})')
    parser.add_argument('--iterations', type=int, default=config.iterations,
                        help=f'查询迭代次数 (default: {config.iterations})')
    parser.add_argument('--threads', type=int, default=config.threads,
                        help=f'并发线程数 (default: {config.threads})')
    parser.add_argument('--databases', nargs='+', default=['apex', 'duckdb', 'sqlite'],
                        help='要测试的数据库')
    
    args = parser.parse_args()
    
    config.rows = args.rows
    config.iterations = args.iterations
    config.threads = args.threads
    
    print("=" * 60)
    print("ApexBase 查询吞吐量 Benchmark")
    print("=" * 60)
    print(f"数据行数: {config.rows:,}")
    print(f"查询迭代: {config.iterations}")
    print(f"并发线程: {config.threads}")
    
    # 生成测试数据
    print("\n生成测试数据...")
    data = generate_data(config.rows)
    print(f"  生成 {len(data):,} 条记录")
    
    # 测试不同复杂度的查询
    all_queries = QUERIES_SIMPLE + QUERIES_MEDIUM + QUERIES_COMPLEX
    
    results = {}
    
    # 1. 单线程顺序 Q/s 测试
    print("\n" + "=" * 60)
    print("1. 单线程顺序查询吞吐量测试")
    print("=" * 60)
    
    for db_name in args.databases:
        tmpdir = tempfile.mkdtemp()
        
        try:
            if db_name == 'apex' and HAS_APEX:
                bench = ApexBaseBench(tmpdir, data)
            elif db_name == 'duckdb' and HAS_DUCKDB:
                bench = DuckDBBench(tmpdir, data)
            elif db_name == 'sqlite' and HAS_SQLITE:
                bench = SQLiteBench(tmpdir, data)
            else:
                continue
            
            print(f"\n>>> {db_name.upper()}")
            bench.setup()
            
            # 测试简单查询
            r_simple = run_single_query_throughput(bench, QUERIES_SIMPLE)
            
            # 测试中等查询
            r_medium = run_single_query_throughput(bench, QUERIES_MEDIUM)
            
            # 测试复杂查询
            r_complex = run_single_query_throughput(bench, QUERIES_COMPLEX)
            
            # 总体
            r_all = run_single_query_throughput(bench, all_queries)
            
            results[f'{db_name}_single'] = {
                'simple': r_simple,
                'medium': r_medium,
                'complex': r_complex,
                'all': r_all
            }
            
            bench.close()
            
        finally:
            import shutil
            shutil.rmtree(tmpdir, ignore_errors=True)
    
    # 2. 并发 Q/s 测试
    print("\n" + "=" * 60)
    print("2. 并发查询吞吐量测试")
    print("=" * 60)
    
    for db_name in args.databases:
        tmpdir = tempfile.mkdtemp()
        
        try:
            if db_name == 'apex' and HAS_APEX:
                bench_class = ApexBaseBench
            elif db_name == 'duckdb' and HAS_DUCKDB:
                bench_class = DuckDBBench
            elif db_name == 'sqlite' and HAS_SQLITE:
                bench_class = SQLiteBench
            else:
                continue
            
            print(f"\n>>> {db_name.upper()}")
            
            # 测试不同线程数
            thread_counts = [1, 2, 4, 8]
            
            for num_threads in thread_counts:
                if num_threads > config.threads:
                    break
                
                r = run_concurrent_sequential(bench_class, tmpdir, data, 
                                              QUERIES_SIMPLE, num_threads)
                
                if f'{db_name}_concurrent' not in results:
                    results[f'{db_name}_concurrent'] = {}
                
                results[f'{db_name}_concurrent'][num_threads] = r
        
        finally:
            import shutil
            shutil.rmtree(tmpdir, ignore_errors=True)
    
    # 3. 输出汇总
    print("\n" + "=" * 60)
    print("汇总结果")
    print("=" * 60)
    
    print("\n--- 单线程顺序 Q/s ---")
    for db in ['apex', 'duckdb', 'sqlite']:
        key = f'{db}_single'
        if key in results:
            r = results[key]['all']
            print(f"{db:10s}: {r['qps']:>8.1f} Q/s (总查询: {r['total_queries']}, 耗时: {r['elapsed_sec']:.3f}s)")
    
    print("\n--- 并发 Q/s (随线程数变化) ---")
    for db in ['apex', 'duckdb', 'sqlite']:
        key = f'{db}_concurrent'
        if key in results:
            print(f"\n{db.upper()}:")
            for threads, r in sorted(results[key].items()):
                print(f"  {threads} 线程: {r['concurrent_qps']:>8.1f} Q/s (延迟: {r['avg_latency_ms']:.2f}ms)")


if __name__ == '__main__':
    main()
