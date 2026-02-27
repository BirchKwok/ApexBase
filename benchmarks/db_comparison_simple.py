#!/usr/bin/env python3
"""
简化三数据库对比测试：ApexBase vs SQLite vs DuckDB

Usage:
    conda run -n dev python benchmarks/db_comparison_simple.py
"""

import os, sys, gc, time, tempfile, random
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import List, Dict
import sqlite3
import duckdb

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'apexbase', 'python'))

try:
    from apexbase import ApexClient
    HAS_APEX = True
except ImportError:
    HAS_APEX = False

@dataclass
class TestConfig:
    batch_size: int = 200_000
    query_iters: int = 5
    concurrent_threads: int = 8
    ops_per_thread: int = 100

config = TestConfig()

def timer(func):
    def wrapper(*args, **kwargs):
        t0 = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - t0
        return result, elapsed
    return wrapper

def generate_data(n_rows: int) -> List[Dict]:
    rng = np.random.default_rng(42)
    cities = [f"City_{i}" for i in range(10)]
    categories = [f"Category_{i}" for i in range(20)]
    
    data = []
    for i in range(n_rows):
        data.append({
            '_id': i + 1,
            'user_id': int(rng.integers(1, 10000)),
            'category': str(rng.choice(categories)),
            'city': str(rng.choice(cities)),
            'price': float(rng.uniform(10.0, 500.0)),
            'quantity': int(rng.integers(1, 50)),
            'is_active': bool(rng.choice([True, False], p=[0.8, 0.2]))
        })
    return data

# ── ApexBase 测试 ───────────────────────────────────────────────────────
def test_apexbase(data: List[Dict]) -> Dict:
    if not HAS_APEX:
        return {}
    
    # ApexClient 接受目录路径，会在其中创建 apexbase.apex
    tmpdir = tempfile.mkdtemp()
    
    try:
        client = ApexClient(tmpdir)
        client.create_table('test_table')
        
        # 写入测试
        start = time.perf_counter()
        batch_size = 20000
        for i in range(0, len(data), batch_size):
            batch = data[i:i+batch_size]
            columns = {}
            for key in batch[0].keys():
                columns[key] = [record[key] for record in batch]
            client.store(columns)
        write_time = time.perf_counter() - start
        
        # 查询测试
        queries = [
            "SELECT COUNT(*) FROM test_table",
            "SELECT city, COUNT(*) FROM test_table GROUP BY city",
            "SELECT category, AVG(price) FROM test_table GROUP BY category",
            "SELECT * FROM test_table WHERE price > 100 ORDER BY price DESC LIMIT 100"
        ]
        
        query_times = []
        for query in queries:
            for _ in range(config.query_iters):
                start = time.perf_counter()
                result = client.execute(query)
                # 确保结果被消费
                try:
                    result.to_pandas()
                except Exception:
                    pass
                query_times.append(time.perf_counter() - start)
        
        client.close()
        
        return {
            'write_speed': len(data) / write_time,
            'avg_query_time': sum(query_times) / len(query_times),
            'query_throughput': 1.0 / (sum(query_times) / len(query_times))
        }
    except Exception as e:
        print(f"  ApexBase 错误: {e}")
        return {}
    finally:
        import shutil
        shutil.rmtree(tmpdir, ignore_errors=True)

# ── SQLite 测试 ───────────────────────────────────────────────────────
def test_sqlite(data: List[Dict]) -> Dict:
    tmpdir = tempfile.mkdtemp()
    db_path = os.path.join(tmpdir, "sqlite.db")
    
    # Convert _id to id for SQLite
    sqlite_data = []
    for record in data:
        r = dict(record)
        r['id'] = r.pop('_id')
        sqlite_data.append(r)
    
    try:
        conn = sqlite3.connect(db_path)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute('''
            CREATE TABLE test_table (
                id INTEGER PRIMARY KEY,
                user_id INTEGER,
                category TEXT,
                city TEXT,
                price REAL,
                quantity INTEGER,
                is_active INTEGER
            )
        ''')
        
        # 写入测试
        start = time.perf_counter()
        conn.executemany('''
            INSERT INTO test_table VALUES 
            (:id, :user_id, :category, :city, :price, :quantity, :is_active)
        ''', sqlite_data)
        conn.commit()
        write_time = time.perf_counter() - start
        
        # 查询测试
        queries = [
            "SELECT COUNT(*) FROM test_table",
            "SELECT city, COUNT(*) FROM test_table GROUP BY city",
            "SELECT category, AVG(price) FROM test_table GROUP BY category",
            "SELECT * FROM test_table WHERE price > 100 ORDER BY price DESC LIMIT 100"
        ]
        
        query_times = []
        for query in queries:
            for _ in range(config.query_iters):
                start = time.perf_counter()
                conn.execute(query).fetchall()
                query_times.append(time.perf_counter() - start)
        
        conn.close()
        
        return {
            'write_speed': len(data) / write_time,
            'avg_query_time': sum(query_times) / len(query_times),
            'query_throughput': 1.0 / (sum(query_times) / len(query_times))
        }
    except Exception as e:
        print(f"  SQLite 错误: {e}")
        return {}
    finally:
        import shutil
        shutil.rmtree(tmpdir, ignore_errors=True)

# ── DuckDB 测试 ───────────────────────────────────────────────────────
def test_duckdb(data: List[Dict]) -> Dict:
    tmpdir = tempfile.mkdtemp()
    db_path = os.path.join(tmpdir, "duckdb.duckdb")
    
    # Use same column names as ApexBase (_id)
    try:
        conn = duckdb.connect(db_path)
        conn.execute('''
            CREATE TABLE test_table (
                _id INTEGER,
                user_id INTEGER,
                category VARCHAR,
                city VARCHAR,
                price DOUBLE,
                quantity INTEGER,
                is_active BOOLEAN
            )
        ''')
        
        # 写入测试
        start = time.perf_counter()
        import pandas as pd
        df = pd.DataFrame(data)
        conn.execute('INSERT INTO test_table SELECT * FROM df')
        write_time = time.perf_counter() - start
        
        # 查询测试
        queries = [
            "SELECT COUNT(*) FROM test_table",
            "SELECT city, COUNT(*) FROM test_table GROUP BY city",
            "SELECT category, AVG(price) FROM test_table GROUP BY category",
            "SELECT * FROM test_table WHERE price > 100 ORDER BY price DESC LIMIT 100"
        ]
        
        query_times = []
        for query in queries:
            for _ in range(config.query_iters):
                start = time.perf_counter()
                conn.execute(query).fetchall()
                query_times.append(time.perf_counter() - start)
        
        conn.close()
        
        return {
            'write_speed': len(data) / write_time,
            'avg_query_time': sum(query_times) / len(query_times),
            'query_throughput': 1.0 / (sum(query_times) / len(query_times))
        }
    except Exception as e:
        print(f"  DuckDB 错误: {e}")
        return {}
    finally:
        import shutil
        shutil.rmtree(tmpdir, ignore_errors=True)

# ── 主测试 ─────────────────────────────────────────────────────────────
def main():
    print("\n" + "="*70)
    print("  三数据库性能对比测试")
    print("="*70 + "\n")
    
    print(f"测试配置:")
    print(f"  • 数据规模: {config.batch_size:,} 行")
    print(f"  • 查询迭代: {config.query_iters} 次")
    print()
    
    # 生成测试数据
    print("生成测试数据...")
    data = generate_data(config.batch_size)
    print(f"  数据量: {len(data):,} 条记录\n")
    
    # 运行测试
    results = {}
    
    print("─" * 50)
    print("性能测试结果")
    print("─" * 50)
    
    # ApexBase
    if HAS_APEX:
        print("测试 ApexBase...")
        apex_result = test_apexbase(data)
        results['ApexBase'] = apex_result
        if apex_result:
            print(f"  写入: {apex_result['write_speed']:,.0f} rows/sec")
            print(f"  查询: {apex_result['avg_query_time']*1000:.1f}ms avg")
            print(f"  吞吐: {apex_result['query_throughput']:.1f} queries/sec")
    
    # SQLite
    print("\n测试 SQLite...")
    sqlite_result = test_sqlite(data)
    results['SQLite'] = sqlite_result
    if sqlite_result:
        print(f"  写入: {sqlite_result['write_speed']:,.0f} rows/sec")
        print(f"  查询: {sqlite_result['avg_query_time']*1000:.1f}ms avg")
        print(f"  吞吐: {sqlite_result['query_throughput']:.1f} queries/sec")
    
    # DuckDB
    print("\n测试 DuckDB...")
    duckdb_result = test_duckdb(data)
    results['DuckDB'] = duckdb_result
    if duckdb_result:
        print(f"  写入: {duckdb_result['write_speed']:,.0f} rows/sec")
        print(f"  查询: {duckdb_result['avg_query_time']*1000:.1f}ms avg")
        print(f"  吞吐: {duckdb_result['query_throughput']:.1f} queries/sec")
    
    # 对比分析
    valid_results = {k: v for k, v in results.items() if v}
    if not valid_results:
        print("\n没有有效的测试结果")
        return

    print("\n─" * 50)
    print("详细对比")
    print("─" * 50)
    
    print(f"\n{'数据库':<10} {'写入(rows/sec)':<15} {'查询延迟(ms)':<12} {'查询吞吐(q/s)':<15}")
    print("-" * 55)
    
    for db_name, result in valid_results.items():
        write_speed = result['write_speed']
        query_latency = result['avg_query_time'] * 1000
        query_throughput = result['query_throughput']
        print(f"{db_name:<10} {write_speed:<15,.0f} {query_latency:<12.1f} {query_throughput:<15.1f}")
    
    # 性能排名
    print(f"\n─" * 50)
    print("性能排名")
    print("─" * 50)
    
    # 写入性能排名
    write_ranking = sorted([(k, v['write_speed']) for k, v in valid_results.items()], 
                          key=lambda x: x[1], reverse=True)
    print(f"\n写入性能:")
    for i, (db, speed) in enumerate(write_ranking, 1):
        print(f"  {i}. {db}: {speed:,.0f} rows/sec")
    
    # 查询性能排名
    query_ranking = sorted([(k, v['query_throughput']) for k, v in valid_results.items()], 
                           key=lambda x: x[1], reverse=True)
    print(f"\n查询性能:")
    for i, (db, throughput) in enumerate(query_ranking, 1):
        print(f"  {i}. {db}: {throughput:.1f} queries/sec")
    
    # 综合评估
    print(f"\n─" * 50)
    print("综合评估")
    print("─" * 50)
    
    if HAS_APEX and 'ApexBase' in valid_results and 'SQLite' in valid_results and 'DuckDB' in valid_results:
        apex = valid_results['ApexBase']
        sqlite = valid_results['SQLite']
        duckdb_r = valid_results['DuckDB']
        
        print(f"\nApexBase 相对性能:")
        if apex['write_speed'] > sqlite['write_speed']:
            ratio = apex['write_speed'] / sqlite['write_speed']
            print(f"  ✅ 写入速度是 SQLite 的 {ratio:.1f}x")
        else:
            ratio = sqlite['write_speed'] / apex['write_speed']
            print(f"  ❌ 写入速度是 SQLite 的 {1/ratio:.2f}x ({ratio:.1f}x slower)")
        
        if apex['query_throughput'] > sqlite['query_throughput']:
            ratio = apex['query_throughput'] / sqlite['query_throughput']
            print(f"  ✅ 查询性能是 SQLite 的 {ratio:.1f}x")
        else:
            ratio = sqlite['query_throughput'] / apex['query_throughput']
            print(f"  ❌ 查询性能是 SQLite 的 {1/ratio:.2f}x ({ratio:.1f}x slower)")
        
        if apex['write_speed'] > duckdb_r['write_speed']:
            ratio = apex['write_speed'] / duckdb_r['write_speed']
            print(f"  ✅ 写入速度是 DuckDB 的 {ratio:.1f}x")
        else:
            ratio = duckdb_r['write_speed'] / apex['write_speed']
            print(f"  ❌ 写入速度是 DuckDB 的 {1/ratio:.2f}x ({ratio:.1f}x slower)")
        
        if apex['query_throughput'] > duckdb_r['query_throughput']:
            ratio = apex['query_throughput'] / duckdb_r['query_throughput']
            print(f"  ✅ 查询性能是 DuckDB 的 {ratio:.1f}x")
        else:
            ratio = duckdb_r['query_throughput'] / apex['query_throughput']
            print(f"  ❌ 查询性能是 DuckDB 的 {1/ratio:.2f}x ({ratio:.1f}x slower)")
    
    print(f"\n💡 推荐场景:")
    if 'ApexBase' in valid_results and valid_results['ApexBase']['write_speed'] > 500000:
        print("  📝 ApexBase: 高吞吐写入场景")
    if 'DuckDB' in valid_results and valid_results['DuckDB']['query_throughput'] > 100:
        print("  🔍 DuckDB: 复杂分析查询场景")
    if 'SQLite' in valid_results:
        print("  🛡️  SQLite: 轻量级嵌入式场景")

if __name__ == "__main__":
    main()
