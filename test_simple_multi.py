"""Simple test for multi-statement SQL"""
import tempfile
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'apexbase', 'python'))

from apexbase import ApexClient

with tempfile.TemporaryDirectory() as temp_dir:
    client = ApexClient(dirpath=temp_dir)
    
    # Test simple multi-statement
    print("Test 1: CREATE TABLE without IF NOT EXISTS")
    try:
        client.execute("CREATE TABLE t1")
        print("   OK")
    except Exception as e:
        print(f"   FAILED: {e}")
    
    print("\nTest 2: CREATE TABLE IF NOT EXISTS")
    try:
        client.execute("CREATE TABLE IF NOT EXISTS t2")
        print("   OK")
    except Exception as e:
        print(f"   FAILED: {e}")
    
    print("\nTest 3: Multi-statement CREATE TABLE")
    try:
        client.execute("CREATE TABLE t3; CREATE TABLE t4")
        print("   OK")
    except Exception as e:
        print(f"   FAILED: {e}")
    
    print("\nTest 4: Multi-statement with IF NOT EXISTS")
    try:
        client.execute("CREATE TABLE IF NOT EXISTS t5; CREATE TABLE IF NOT EXISTS t6")
        print("   OK")
    except Exception as e:
        print(f"   FAILED: {e}")
    
    tables = client.list_tables()
    print(f"\nTables: {tables}")
    
    client.close()
    print("\nDone!")
