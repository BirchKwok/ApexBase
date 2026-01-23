#!/usr/bin/env python3
"""
ApexBase vs DuckDB Performance Benchmark
Compare query performance with same data scale and operations.
"""

import time
import random
import tempfile
import shutil
import os

# Data generation
def generate_users(n=200_000):
    """Generate user data"""
    tiers = ['bronze', 'silver', 'gold', 'platinum']
    return [
        {
            'user_id': i,
            'name': f'User_{i}',
            'tier': tiers[i % 4],
            'score': random.randint(1, 100)
        }
        for i in range(n)
    ]

def generate_orders(n=1_000_000, user_count=200_000):
    """Generate order data"""
    return [
        {
            'order_id': i,
            'user_id': random.randint(0, user_count - 1),
            'amount': round(random.uniform(10, 500), 2),
            'qty': random.randint(1, 10)
        }
        for i in range(n)
    ]

def benchmark_apexbase(users, orders, tmpdir, runs=5):
    """Benchmark ApexBase"""
    from apexbase import ApexClient
    
    db_path = os.path.join(tmpdir, 'apex_bench')
    os.makedirs(db_path, exist_ok=True)
    
    db = ApexClient(db_path)
    
    # Store users table
    t0 = time.perf_counter()
    db.create_table('users')
    db.store(users)
    db.flush()
    store_users_time = time.perf_counter() - t0
    
    # Store orders table
    t0 = time.perf_counter()
    db.create_table('orders')
    db.store(orders)
    db.flush()
    store_orders_time = time.perf_counter() - t0
    
    # Queries to benchmark
    queries = {
        'join_agg': """
            SELECT u.tier, COUNT(*) as cnt, SUM(o.amount) as total
            FROM users u
            JOIN orders o ON u.user_id = o.user_id
            WHERE o.amount >= 50
            GROUP BY u.tier
            ORDER BY total DESC
            LIMIT 10
        """,
        'subquery': """
            SELECT tier, COUNT(*) as cnt
            FROM users
            WHERE user_id IN (
                SELECT user_id FROM orders WHERE amount > 100
            )
            GROUP BY tier
            ORDER BY cnt DESC
        """,
        'simple_agg': """
            SELECT user_id, SUM(amount) as total, AVG(amount) as avg_amt, COUNT(*) as cnt
            FROM orders
            GROUP BY user_id
            ORDER BY total DESC
            LIMIT 100
        """,
        'filter_only': """
            SELECT * FROM orders WHERE amount > 200 AND qty >= 5 LIMIT 1000
        """
    }
    
    results = {
        'store_users': store_users_time,
        'store_orders': store_orders_time,
        'queries': {}
    }
    
    for name, sql in queries.items():
        times = []
        for _ in range(runs):
            t0 = time.perf_counter()
            _ = db.execute(sql)
            times.append(time.perf_counter() - t0)
        results['queries'][name] = {
            'avg': sum(times) / len(times),
            'min': min(times),
            'max': max(times)
        }
    
    return results

def benchmark_duckdb(users, orders, tmpdir, runs=5):
    """Benchmark DuckDB"""
    import duckdb
    
    db_path = os.path.join(tmpdir, 'duck_bench.db')
    conn = duckdb.connect(db_path)
    
    # Store data
    import pandas as pd
    users_df = pd.DataFrame(users)
    orders_df = pd.DataFrame(orders)
    
    t0 = time.perf_counter()
    conn.execute("CREATE TABLE users AS SELECT * FROM users_df")
    store_users_time = time.perf_counter() - t0
    
    t0 = time.perf_counter()
    conn.execute("CREATE TABLE orders AS SELECT * FROM orders_df")
    store_orders_time = time.perf_counter() - t0
    
    # Same queries
    queries = {
        'join_agg': """
            SELECT u.tier, COUNT(*), SUM(o.amount)
            FROM users u
            JOIN orders o ON u.user_id = o.user_id
            WHERE o.amount >= 50
            GROUP BY u.tier
            ORDER BY SUM(o.amount) DESC
            LIMIT 10
        """,
        'subquery': """
            SELECT tier, COUNT(*) as cnt
            FROM users
            WHERE user_id IN (
                SELECT user_id FROM orders WHERE amount > 100
            )
            GROUP BY tier
            ORDER BY cnt DESC
        """,
        'simple_agg': """
            SELECT user_id, SUM(amount), AVG(amount), COUNT(*)
            FROM orders
            GROUP BY user_id
            ORDER BY SUM(amount) DESC
            LIMIT 100
        """,
        'filter_only': """
            SELECT * FROM orders WHERE amount > 200 AND qty >= 5 LIMIT 1000
        """
    }
    
    results = {
        'store_users': store_users_time,
        'store_orders': store_orders_time,
        'queries': {}
    }
    
    for name, sql in queries.items():
        times = []
        for _ in range(runs):
            t0 = time.perf_counter()
            _ = conn.execute(sql).fetchall()
            times.append(time.perf_counter() - t0)
        results['queries'][name] = {
            'avg': sum(times) / len(times),
            'min': min(times),
            'max': max(times)
        }
    
    conn.close()
    return results

def main():
    print("=" * 70)
    print("ApexBase vs DuckDB Performance Benchmark")
    print("=" * 70)
    print()
    
    # Generate data
    print("Generating test data...")
    t0 = time.perf_counter()
    users = generate_users(200_000)
    orders = generate_orders(1_000_000)
    print(f"  Users: {len(users):,} rows")
    print(f"  Orders: {len(orders):,} rows")
    print(f"  Generation time: {time.perf_counter() - t0:.2f}s")
    print()
    
    tmpdir = tempfile.mkdtemp()
    
    try:
        # Benchmark ApexBase
        print("Running ApexBase benchmark...")
        apex_results = benchmark_apexbase(users, orders, tmpdir)
        
        # Benchmark DuckDB
        print("Running DuckDB benchmark...")
        duck_results = benchmark_duckdb(users, orders, tmpdir)
        
        # Print results
        print()
        print("=" * 70)
        print("RESULTS")
        print("=" * 70)
        print()
        
        # Write performance
        print("### Write Performance ###")
        print(f"{'Operation':<20} {'ApexBase':>12} {'DuckDB':>12} {'Winner':>12}")
        print("-" * 56)
        
        for op in ['store_users', 'store_orders']:
            apex_t = apex_results[op]
            duck_t = duck_results[op]
            winner = 'ApexBase' if apex_t < duck_t else 'DuckDB'
            ratio = max(apex_t, duck_t) / min(apex_t, duck_t)
            print(f"{op:<20} {apex_t:>10.3f}s {duck_t:>10.3f}s {winner:>10} ({ratio:.1f}x)")
        print()
        
        # Query performance
        print("### Query Performance (avg of 5 runs) ###")
        print(f"{'Query':<20} {'ApexBase':>12} {'DuckDB':>12} {'Winner':>12}")
        print("-" * 56)
        
        for query_name in apex_results['queries']:
            apex_t = apex_results['queries'][query_name]['avg']
            duck_t = duck_results['queries'][query_name]['avg']
            winner = 'ApexBase' if apex_t < duck_t else 'DuckDB'
            ratio = max(apex_t, duck_t) / min(apex_t, duck_t)
            print(f"{query_name:<20} {apex_t:>10.3f}s {duck_t:>10.3f}s {winner:>10} ({ratio:.1f}x)")
        
        print()
        print("=" * 70)
        
        # Summary
        apex_total = sum(r['avg'] for r in apex_results['queries'].values())
        duck_total = sum(r['avg'] for r in duck_results['queries'].values())
        print(f"Total query time: ApexBase={apex_total:.3f}s, DuckDB={duck_total:.3f}s")
        
        if apex_total < duck_total:
            print(f"ApexBase is {duck_total/apex_total:.1f}x faster overall")
        else:
            print(f"DuckDB is {apex_total/duck_total:.1f}x faster overall")
        
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

if __name__ == '__main__':
    main()
