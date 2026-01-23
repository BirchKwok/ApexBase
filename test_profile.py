#!/usr/bin/env python3
"""Profile where time is spent in query execution."""

import time
import tempfile
import shutil
from apexbase import ApexClient

def main():
    temp_dir = tempfile.mkdtemp(prefix="profile_")
    
    try:
        client = ApexClient(dirpath=temp_dir, drop_if_exists=True)
        
        # Create smaller dataset for profiling
        print("Creating 100,000 rows...")
        docs = []
        for i in range(100_000):
            docs.append({
                "_id": f"doc_{i}",
                "category": f"cat_{i % 100}",
                "value": float(i % 1000),
                "status": "active" if i % 2 == 0 else "inactive",
            })
        client.store(docs)
        client.flush()
        print("Data created.\n")
        
        # Test 1: Simple COUNT - should be fast
        print("=== Profiling COUNT(*) ===")
        for _ in range(3):
            t0 = time.perf_counter()
            result = client.execute("SELECT COUNT(*) FROM default")
            t1 = time.perf_counter()
            # Access the data to force materialization
            _ = result.to_dict()
            t2 = time.perf_counter()
            print(f"  execute: {(t1-t0)*1000:.2f}ms, to_dict: {(t2-t1)*1000:.2f}ms")
        
        # Test 2: LIMIT query - small result
        print("\n=== Profiling SELECT * LIMIT 10 ===")
        for _ in range(3):
            t0 = time.perf_counter()
            result = client.execute("SELECT * FROM default LIMIT 10")
            t1 = time.perf_counter()
            _ = result.to_dict()
            t2 = time.perf_counter()
            print(f"  execute: {(t1-t0)*1000:.2f}ms, to_dict: {(t2-t1)*1000:.2f}ms")
        
        # Test 3: Filter + LIMIT - should early exit
        print("\n=== Profiling WHERE + LIMIT 100 ===")
        for _ in range(3):
            t0 = time.perf_counter()
            result = client.execute("SELECT * FROM default WHERE status = 'active' LIMIT 100")
            t1 = time.perf_counter()
            _ = result.to_dict()
            t2 = time.perf_counter()
            print(f"  execute: {(t1-t0)*1000:.2f}ms, to_dict: {(t2-t1)*1000:.2f}ms")
        
        # Test 4: GROUP BY - full scan required
        print("\n=== Profiling GROUP BY ===")
        for _ in range(3):
            t0 = time.perf_counter()
            result = client.execute("SELECT category, COUNT(*) FROM default GROUP BY category")
            t1 = time.perf_counter()
            _ = result.to_dict()
            t2 = time.perf_counter()
            print(f"  execute: {(t1-t0)*1000:.2f}ms, to_dict: {(t2-t1)*1000:.2f}ms")
        
        # Test 5: Full scan SELECT *
        print("\n=== Profiling SELECT * (full scan 100K rows) ===")
        for _ in range(2):
            t0 = time.perf_counter()
            result = client.execute("SELECT * FROM default")
            t1 = time.perf_counter()
            _ = result.to_dict()
            t2 = time.perf_counter()
            print(f"  execute: {(t1-t0)*1000:.2f}ms, to_dict: {(t2-t1)*1000:.2f}ms, rows: {len(result)}")
            
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

if __name__ == "__main__":
    main()
