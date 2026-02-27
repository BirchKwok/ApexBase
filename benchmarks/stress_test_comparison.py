#!/usr/bin/env python3
"""
三数据库压力测试对比：ApexBase vs SQLite vs DuckDB

测试场景：
1. 离线批量写入 - 大数据量ETL
2. 离线复杂查询 - 分析型工作负载  
3. 在线高并发读写 - 混合工作负载
4. 在线延迟敏感 - 实时查询

Usage:
    conda run -n dev python benchmarks/stress_test_comparison.py
"""

import os, sys, gc, time, tempfile, json, threading, random
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import List, Dict, Tuple, Any
import sqlite3
import duckdb

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'apexbase', 'python'))

try:
    from apexbase import ApexClient
    HAS_APEX = True
except ImportError:
    HAS_APEX = False
    print("WARNING: ApexBase not found")

# ── 配置 ──────────────────────────────────────────────────────────────────
@dataclass
class TestConfig:
    # 离线测试配置
    offline_batch_size: int = 500_000      # 50万行批量写入
    offline_query_iters: int = 10           # 查询迭代次数
    
    # 在线测试配置
    online_concurrent_threads: int = 10     # 并发线程数
    online_ops_per_thread: int = 200        # 每线程操作数
    
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

# ── ApexBase 测试 ───────────────────────────────────────────────────────────
class ApexBaseTester:
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.client = None
    
    def setup(self, data: List[Dict] = None):
        self.client = ApexClient(self.db_path)
        # 尝试使用表，如果不存在则创建
        try:
            self.client.use_table('sales_data')
        except:
            self.client.create_table('sales_data')
            # 如果有数据，写入数据
            if data is not None:
                self._write_data(data)
    
    def _write_data(self, data: List[Dict]):
        """写入数据"""
        batch_size = 25_000
        for i in range(0, len(data), batch_size):
            batch = data[i:i+batch_size]
            columns = {}
            for key in batch[0].keys():
                columns[key] = [record[key] for record in batch]
            self.client.store(columns)
    
    def cleanup(self):
        if self.client:
            self.client.close()
    
    @timer
    def test_batch_write(self, data: List[Dict]) -> Dict:
        """批量写入测试"""
        print(f"    ApexBase 批量写入 {len(data):,} 条记录...")
        
        batch_size = 25_000
        total_time = 0
        
        for i in range(0, len(data), batch_size):
            batch = data[i:i+batch_size]
            start_time = time.perf_counter()
            
            columns = {}
            for key in batch[0].keys():
                columns[key] = [record[key] for record in batch]
            
            self.client.store(columns)
            total_time += time.perf_counter() - start_time
        
        return {
            'rows_written': len(data),
            'total_time': total_time,
            'rows_per_sec': len(data) / total_time
        }
    
    @timer
    def test_queries(self) -> Dict:
        """查询测试"""
        queries = [
            "SELECT city, COUNT(*) as cnt, AVG(price) as avg_price FROM sales_data GROUP BY city ORDER BY cnt DESC",
            "SELECT category, COUNT(*) as total_count, AVG(price) as avg_price FROM sales_data GROUP BY category ORDER BY total_count DESC",
            "SELECT COUNT(*) as total_orders, SUM(price) as total_revenue FROM sales_data WHERE is_active = true",
            "SELECT city, AVG(score) as avg_score FROM sales_data WHERE price > 100 GROUP BY city ORDER BY avg_score DESC",
            "SELECT _id, city, price FROM sales_data WHERE price BETWEEN 50 AND 200 ORDER BY price DESC LIMIT 1000"
        ]
        
        query_times = []
        
        for i, query in enumerate(queries):
            times = []
            for _ in range(config.offline_query_iters):
                gc.collect()
                start = time.perf_counter()
                result = self.client.execute(query)
                elapsed = time.perf_counter() - start
                times.append(elapsed)
            
            query_times.extend(times)
        
        return {
            'total_queries': len(queries) * config.offline_query_iters,
            'avg_query_time': sum(query_times) / len(query_times),
            'queries_per_sec': 1.0 / (sum(query_times) / len(query_times))
        }

# ── SQLite 测试 ───────────────────────────────────────────────────────────
class SQLiteTester:
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.conn = None
    
    def setup(self):
        self.conn = sqlite3.connect(self.db_path)
        # 如果表不存在则创建
        try:
            self.conn.execute('''
                CREATE TABLE sales_data (
                    _id INTEGER PRIMARY KEY,
                    user_id INTEGER,
                    category TEXT,
                    city TEXT,
                    price REAL,
                    quantity INTEGER,
                    timestamp INTEGER,
                    is_active INTEGER,
                    score REAL
                )
            ''')
            self.conn.commit()
        except sqlite3.OperationalError:
            pass  # 表已存在
    
    def cleanup(self):
        if self.conn:
            self.conn.close()
    
    @timer
    def test_batch_write(self, data: List[Dict]) -> Dict:
        """批量写入测试"""
        print(f"    SQLite 批量写入 {len(data):,} 条记录...")
        
        # SQLite uses _id directly (same as ApexBase)
        start_time = time.perf_counter()
        
        # 使用executemany批量插入
        self.conn.executemany('''
            INSERT INTO sales_data VALUES 
            (:_id, :user_id, :category, :city, :price, :quantity, :timestamp, :is_active, :score)
        ''', data)
        self.conn.commit()
        
        total_time = time.perf_counter() - start_time
        
        return {
            'rows_written': len(data),
            'total_time': total_time,
            'rows_per_sec': len(data) / total_time
        }
    
    @timer
    def test_queries(self) -> Dict:
        """查询测试"""
        queries = [
            "SELECT city, COUNT(*) as cnt, AVG(price) as avg_price FROM sales_data GROUP BY city ORDER BY cnt DESC",
            "SELECT category, COUNT(*) as total_count, AVG(price) as avg_price FROM sales_data GROUP BY category ORDER BY total_count DESC",
            "SELECT COUNT(*) as total_orders, SUM(price) as total_revenue FROM sales_data WHERE is_active = 1",
            "SELECT city, AVG(score) as avg_score FROM sales_data WHERE price > 100 GROUP BY city ORDER BY avg_score DESC",
            "SELECT * FROM sales_data WHERE price BETWEEN 50 AND 200 ORDER BY price DESC LIMIT 1000"
        ]
        
        query_times = []
        
        for i, query in enumerate(queries):
            times = []
            for _ in range(config.offline_query_iters):
                gc.collect()
                start = time.perf_counter()
                self.conn.execute(query)
                elapsed = time.perf_counter() - start
                times.append(elapsed)
            
            query_times.extend(times)
        
        return {
            'total_queries': len(queries) * config.offline_query_iters,
            'avg_query_time': sum(query_times) / len(query_times),
            'queries_per_sec': 1.0 / (sum(query_times) / len(query_times))
        }

# ── DuckDB 测试 ───────────────────────────────────────────────────────────
class DuckDBTester:
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.conn = None
    
    def setup(self):
        self.conn = duckdb.connect(self.db_path)
        # 如果表不存在则创建
        try:
            # Use _id to match ApexBase
            self.conn.execute('''
                CREATE TABLE sales_data (
                    _id INTEGER,
                    user_id INTEGER,
                    category VARCHAR,
                    city VARCHAR,
                    price DOUBLE,
                    quantity INTEGER,
                    timestamp BIGINT,
                    is_active BOOLEAN,
                    score DOUBLE
                )
            ''')
        except Exception:
            pass  # 表已存在
    
    def cleanup(self):
        if self.conn:
            self.conn.close()
    
    @timer
    def test_batch_write(self, data: List[Dict]) -> Dict:
        """批量写入测试"""
        print(f"    DuckDB 批量写入 {len(data):,} 条记录...")
        
        start_time = time.perf_counter()
        
        # 转换为DataFrame然后批量插入
        import pandas as pd
        df = pd.DataFrame(data)
        self.conn.execute('INSERT INTO sales_data SELECT * FROM df')
        
        total_time = time.perf_counter() - start_time
        
        return {
            'rows_written': len(data),
            'total_time': total_time,
            'rows_per_sec': len(data) / total_time
        }
    
    @timer
    def test_queries(self) -> Dict:
        """查询测试"""
        queries = [
            "SELECT city, COUNT(*) as cnt, AVG(price) as avg_price FROM sales_data GROUP BY city ORDER BY cnt DESC",
            "SELECT category, COUNT(*) as total_count, AVG(price) as avg_price FROM sales_data GROUP BY category ORDER BY total_count DESC",
            "SELECT COUNT(*) as total_orders, SUM(price) as total_revenue FROM sales_data WHERE is_active = true",
            "SELECT city, AVG(score) as avg_score FROM sales_data WHERE price > 100 GROUP BY city ORDER BY avg_score DESC",
            "SELECT * FROM sales_data WHERE price BETWEEN 50 AND 200 ORDER BY price DESC LIMIT 1000"
        ]
        
        query_times = []
        
        for i, query in enumerate(queries):
            times = []
            for _ in range(config.offline_query_iters):
                gc.collect()
                start = time.perf_counter()
                self.conn.execute(query).fetchall()
                elapsed = time.perf_counter() - start
                times.append(elapsed)
            
            query_times.extend(times)
        
        return {
            'total_queries': len(queries) * config.offline_query_iters,
            'avg_query_time': sum(query_times) / len(query_times),
            'queries_per_sec': 1.0 / (sum(query_times) / len(query_times))
        }

# ── 并发测试 ─────────────────────────────────────────────────────────────
def apex_worker(thread_id: int, db_path: str, ops_count: int) -> Dict:
    """ApexBase工作线程"""
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
        client.use_table('sales_data')
        
        for i in range(ops_count):
            op_start = time.perf_counter()
            
            if random.random() < 0.8:  # 80%读操作
                if random.random() < 0.6:
                    result = client.execute(f"SELECT * FROM sales_data WHERE _id = {random.randint(1, 100000)}")
                else:
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
    finally:
        try:
            client.close()
        except:
            pass
        results['total_time'] = time.perf_counter() - start_time
    
    return results

def sqlite_worker(thread_id: int, db_path: str, ops_count: int) -> Dict:
    """SQLite工作线程"""
    conn = sqlite3.connect(db_path, timeout=30.0)
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
        for i in range(ops_count):
            op_start = time.perf_counter()
            
            if random.random() < 0.8:  # 80%读操作
                if random.random() < 0.6:
                    result = conn.execute(f"SELECT * FROM sales_data WHERE _id = {random.randint(1, 100000)}").fetchall()
                else:
                    result = conn.execute("SELECT COUNT(*) FROM sales_data WHERE city = 'City_1'").fetchall()
                
                results['read_times'].append(time.perf_counter() - op_start)
            else:  # 20%写操作
                new_record = (
                    random.randint(1, 50000),
                    f"Category_{random.randint(1, 20)}",
                    f"City_{random.randint(1, 10)}",
                    random.uniform(10.0, 1000.0),
                    random.randint(1, 100),
                    int(time.time()),
                    random.choice([True, False]),
                    random.uniform(0.0, 1.0)
                )
                
                conn.execute('''
                    INSERT INTO sales_data (user_id, category, city, price, quantity, timestamp, is_active, score)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ''', new_record)
                conn.commit()
                results['write_times'].append(time.perf_counter() - op_start)
            
            results['operations'] += 1
            
    except Exception as e:
        results['errors'] += 1
    finally:
        conn.close()
        results['total_time'] = time.perf_counter() - start_time
    
    return results

def duckdb_worker(thread_id: int, db_path: str, ops_count: int) -> Dict:
    """DuckDB工作线程"""
    conn = duckdb.connect(db_path)
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
        for i in range(ops_count):
            op_start = time.perf_counter()
            
            if random.random() < 0.8:  # 80%读操作
                if random.random() < 0.6:
                    result = conn.execute(f"SELECT * FROM sales_data WHERE _id = {random.randint(1, 100000)}").fetchall()
                else:
                    result = conn.execute("SELECT COUNT(*) FROM sales_data WHERE city = 'City_1'").fetchall()
                
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
                
                conn.execute('''
                    INSERT INTO sales_data VALUES 
                    (NULL, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', [
                    new_record['user_id'], new_record['category'], new_record['city'],
                    new_record['price'], new_record['quantity'], new_record['timestamp'],
                    new_record['is_active'], new_record['score']
                ])
                results['write_times'].append(time.perf_counter() - op_start)
            
            results['operations'] += 1
            
    except Exception as e:
        results['errors'] += 1
    finally:
        conn.close()
        results['total_time'] = time.perf_counter() - start_time
    
    return results

@timer
def test_concurrent(db_path: str, worker_func, db_name: str) -> Dict:
    """并发测试"""
    print(f"    启动 {config.online_concurrent_threads} 个 {db_name} 并发线程...")
    
    with ThreadPoolExecutor(max_workers=config.online_concurrent_threads) as executor:
        futures = []
        for i in range(config.online_concurrent_threads):
            future = executor.submit(worker_func, i, db_path, config.online_ops_per_thread)
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
        'db_name': db_name,
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
    print("  三数据库压力测试对比: ApexBase vs SQLite vs DuckDB")
    print("="*80 + "\n")
    
    tmpdir = tempfile.mkdtemp(prefix="db_comparison_")
    
    db_paths = {
        'ApexBase': os.path.join(tmpdir, "apexbase.apex"),
        'SQLite': os.path.join(tmpdir, "sqlite.db"),
        'DuckDB': os.path.join(tmpdir, "duckdb.duckdb")
    }
    
    results = {}
    
    try:
        # 生成测试数据
        print("生成测试数据...")
        data = generate_test_data(config.offline_batch_size)
        print(f"  数据量: {len(data):,} 条记录\n")
        
        # ── 1. 离线批量写入测试 ───────────────────────────────────────
        print("─" * 60)
        print("1. 离线批量写入测试")
        print("─" * 60)
        
        # ApexBase
        if HAS_APEX:
            apex_tester = ApexBaseTester(db_paths['ApexBase'])
            apex_tester.setup()  # 创建表
            write_result, write_time = apex_tester.test_batch_write(data)
            results['ApexBase'] = {'write': write_result}
            # 不清理client，保留数据用于查询测试
            print(f"  ApexBase 写入: {write_result['rows_per_sec']:,.0f} rows/sec")
        
        # SQLite
        sqlite_tester = SQLiteTester(db_paths['SQLite'])
        sqlite_tester.setup()
        write_result, write_time = sqlite_tester.test_batch_write(data)
        results['SQLite'] = {'write': write_result}
        # 不关闭连接，保留数据用于查询测试
        print(f"  SQLite  写入: {write_result['rows_per_sec']:,.0f} rows/sec")
        
        # DuckDB
        duckdb_tester = DuckDBTester(db_paths['DuckDB'])
        duckdb_tester.setup()
        write_result, write_time = duckdb_tester.test_batch_write(data)
        results['DuckDB'] = {'write': write_result}
        # 不关闭连接，保留数据用于查询测试
        print(f"  DuckDB   写入: {write_result['rows_per_sec']:,.0f} rows/sec")
        
        # ── 2. 离线查询测试 ───────────────────────────────────────────
        print("\n─" * 60)
        print("2. 离线查询测试")
        print("─" * 60)
        
        # ApexBase - 使用已有的client
        if HAS_APEX:
            query_result, query_time = apex_tester.test_queries()
            results['ApexBase']['query'] = query_result
            apex_tester.cleanup()
            print(f"  ApexBase 查询: {query_result['queries_per_sec']:.1f} queries/sec")
        
        # SQLite - 使用已有的连接
        query_result, query_time = sqlite_tester.test_queries()
        results['SQLite']['query'] = query_result
        sqlite_tester.cleanup()
        print(f"  SQLite  查询: {query_result['queries_per_sec']:.1f} queries/sec")
        
        # DuckDB - 使用已有的连接
        query_result, query_time = duckdb_tester.test_queries()
        results['DuckDB']['query'] = query_result
        duckdb_tester.cleanup()
        print(f"  DuckDB   查询: {query_result['queries_per_sec']:.1f} queries/sec")
        
        # ── 3. 在线并发测试 ───────────────────────────────────────────
        print("\n─" * 60)
        print("3. 在线并发测试")
        print("─" * 60)
        
        # ApexBase - 现在支持多客户端连接
        if HAS_APEX:
            concurrent_result, concurrent_time = test_concurrent(
                db_paths['ApexBase'], apex_worker, 'ApexBase'
            )
            results['ApexBase']['concurrent'] = concurrent_result
            print(f"  ApexBase 并发: {concurrent_result['ops_per_sec']:.0f} ops/sec, 错误率: {concurrent_result['error_rate']*100:.1f}%")
        
        # SQLite
        concurrent_result, concurrent_time = test_concurrent(
            db_paths['SQLite'], sqlite_worker, 'SQLite'
        )
        results['SQLite']['concurrent'] = concurrent_result
        print(f"  SQLite  并发: {concurrent_result['ops_per_sec']:.0f} ops/sec, 错误率: {concurrent_result['error_rate']*100:.1f}%")
        
        # DuckDB
        concurrent_result, concurrent_time = test_concurrent(
            db_paths['DuckDB'], duckdb_worker, 'DuckDB'
        )
        results['DuckDB']['concurrent'] = concurrent_result
        print(f"  DuckDB   并发: {concurrent_result['ops_per_sec']:.0f} ops/sec, 错误率: {concurrent_result['error_rate']*100:.1f}%")
        
        # ── 4. 详细对比分析 ─────────────────────────────────────────────
        print("\n─" * 60)
        print("4. 详细对比分析")
        print("─" * 60)
        
        print(f"\n{'数据库':<10} {'写入(rows/sec)':<15} {'查询(q/sec)':<12} {'并发(ops/sec)':<15} {'错误率(%)':<10} {'读延迟(ms)':<12} {'写延迟(ms)':<12}")
        print("-" * 90)
        
        for db_name in ['ApexBase', 'SQLite', 'DuckDB']:
            if db_name not in results:
                continue
                
            r = results[db_name]
            write_speed = r['write']['rows_per_sec'] if 'write' in r else 0
            query_speed = r['query']['queries_per_sec'] if 'query' in r else 0
            concurrent_speed = r['concurrent']['ops_per_sec'] if 'concurrent' in r else 0
            error_rate = r['concurrent']['error_rate'] * 100 if 'concurrent' in r else 0
            read_latency = r['concurrent']['avg_read_time'] * 1000 if 'concurrent' in r else 0
            write_latency = r['concurrent']['avg_write_time'] * 1000 if 'concurrent' in r else 0
            
            print(f"{db_name:<10} {write_speed:<15,.0f} {query_speed:<12.1f} {concurrent_speed:<15,.0f} {error_rate:<10.1f} {read_latency:<12.1f} {write_latency:<12.1f}")
        
        # ── 5. 性能排名 ─────────────────────────────────────────────────
        print(f"\n─" * 60)
        print("5. 性能排名")
        print("─" * 60)
        
        rankings = {
            '写入性能': {},
            '查询性能': {},
            '并发性能': {},
            '稳定性': {}
        }
        
        for db_name in ['ApexBase', 'SQLite', 'DuckDB']:
            if db_name not in results:
                continue
                
            r = results[db_name]
            rankings['写入性能'][db_name] = r['write']['rows_per_sec'] if 'write' in r else 0
            rankings['查询性能'][db_name] = r['query']['queries_per_sec'] if 'query' in r else 0
            rankings['并发性能'][db_name] = r['concurrent']['ops_per_sec'] if 'concurrent' in r else 0
            rankings['稳定性'][db_name] = 100 - (r['concurrent']['error_rate'] * 100) if 'concurrent' in r else 0
        
        for metric, scores in rankings.items():
            sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
            print(f"\n{metric}:")
            for i, (db, score) in enumerate(sorted_scores, 1):
                if metric == '写入性能':
                    print(f"  {i}. {db}: {score:,.0f} rows/sec")
                elif metric == '查询性能':
                    print(f"  {i}. {db}: {score:.1f} queries/sec")
                elif metric == '并发性能':
                    print(f"  {i}. {db}: {score:,.0f} ops/sec")
                else:  # 稳定性
                    print(f"  {i}. {db}: {score:.1f}%")
        
        # ── 6. 总结建议 ─────────────────────────────────────────────────
        print(f"\n─" * 60)
        print("6. 总结建议")
        print("─" * 60)
        
        print("\n🎯 适用场景推荐:")
        
        # 写入性能最佳
        best_write = max(rankings['写入性能'].items(), key=lambda x: x[1])
        print(f"📝 批量写入场景: {best_write[0]} (优势: {best_write[1]:,.0f} rows/sec)")
        
        # 查询性能最佳
        best_query = max(rankings['查询性能'].items(), key=lambda x: x[1])
        print(f"🔍 分析查询场景: {best_query[0]} (优势: {best_query[1]:.1f} queries/sec)")
        
        # 并发性能最佳
        best_concurrent = max(rankings['并发性能'].items(), key=lambda x: x[1])
        print(f"⚡ 高并发场景: {best_concurrent[0]} (优势: {best_concurrent[1]:,.0f} ops/sec)")
        
        # 稳定性最佳
        best_stability = max(rankings['稳定性'].items(), key=lambda x: x[1])
        print(f"🛡️  稳定性要求: {best_stability[0]} (优势: {best_stability[1]:.1f}% 无错误)")
        
        print(f"\n💡 综合评估:")
        if HAS_APEX:
            apex_write = rankings['写入性能']['ApexBase']
            apex_query = rankings['查询性能']['ApexBase']
            apex_concurrent = rankings['并发性能']['ApexBase']
            
            if apex_write > rankings['写入性能']['SQLite'] * 0.8:
                print("✅ ApexBase 在写入场景表现优异")
            if apex_query > rankings['查询性能']['SQLite'] * 0.5:
                print("✅ ApexBase 查询性能可接受")
            if apex_concurrent > rankings['并发性能']['SQLite'] * 0.5:
                print("✅ ApexBase 并发处理能力良好")
        
        print(f"\n📊 测试配置:")
        print(f"  • 数据规模: {config.offline_batch_size:,} 行")
        print(f"  • 并发线程: {config.online_concurrent_threads}")
        print(f"  • 每线程操作: {config.online_ops_per_thread}")
        print(f"  • 查询迭代: {config.offline_query_iters}")
        
    finally:
        import shutil
        shutil.rmtree(tmpdir, ignore_errors=True)

if __name__ == "__main__":
    main()
