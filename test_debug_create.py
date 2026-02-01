"""Debug test for CREATE TABLE parsing"""
import tempfile
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'apexbase', 'python'))

from apexbase import ApexClient

with tempfile.TemporaryDirectory() as temp_dir:
    client = ApexClient(dirpath=temp_dir)
    
    # Test single CREATE TABLE without IF NOT EXISTS
    print("Test 1: CREATE TABLE test1")
    try:
        client.execute("CREATE TABLE test1")
        print("   OK")
    except Exception as e:
        print(f"   FAILED: {e}")
    
    # Test CREATE TABLE with IF NOT EXISTS
    print("Test 2: CREATE TABLE IF NOT EXISTS test2")
    try:
        client.execute("CREATE TABLE IF NOT EXISTS test2")
        print("   OK")
    except Exception as e:
        print(f"   FAILED: {e}")
    
    # Test simple multi-statement
    print("Test 3: Multi-statement without IF NOT EXISTS")
    try:
        client.execute("CREATE TABLE test3; CREATE TABLE test4")
        print("   OK")
    except Exception as e:
        print(f"   FAILED: {e}")
    
    # Test multi-statement with IF NOT EXISTS
    print("Test 4: Multi-statement with IF NOT EXISTS")
    try:
        client.execute("CREATE TABLE IF NOT EXISTS test5; CREATE TABLE IF NOT EXISTS test6")
        print("   OK")
    except Exception as e:
        print(f"   FAILED: {e}")
    
    tables = client.list_tables()
    print(f"\nTables created: {tables}")
    
    client.close()
