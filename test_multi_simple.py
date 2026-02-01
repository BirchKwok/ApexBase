"""Test multi-statement SQL support"""
import tempfile
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'apexbase', 'python'))

from apexbase import ApexClient

def test_multi_statement_sql():
    """Test executing multiple SQL statements separated by semicolons"""
    with tempfile.TemporaryDirectory() as temp_dir:
        client = ApexClient(dirpath=temp_dir)
        
        print("Testing multi-statement SQL support...")
        
        # Test 1: Simple multi-statement
        print("\n1. Simple CREATE TABLE + INSERT...")
        client.execute("CREATE TABLE IF NOT EXISTS users")
        client.execute("ALTER TABLE users ADD COLUMN name STRING")
        client.execute("INSERT INTO users (name) VALUES ('Alice')")
        
        result = client.execute("SELECT * FROM users")
        assert len(result) == 1, f"Expected 1 row, got {len(result)}"
        print("   ✓ Basic SQL works")
        
        # Test 2: Multiple CREATE TABLE in single execute
        print("\n2. Multiple CREATE TABLE in single call...")
        try:
            client.execute("CREATE TABLE IF NOT EXISTS t1; CREATE TABLE IF NOT EXISTS t2")
            tables = client.list_tables()
            assert "t1" in tables, f"t1 not created. Tables: {tables}"
            assert "t2" in tables, f"t2 not created. Tables: {tables}"
            print("   ✓ Multiple CREATE TABLE works")
        except Exception as e:
            print(f"   ✗ Failed: {e}")
            import traceback
            traceback.print_exc()
        
        # Test 3: INSERT multiple rows
        print("\n3. Multiple INSERT...")
        try:
            client.execute("""
                INSERT INTO users (name) VALUES ('Bob');
                INSERT INTO users (name) VALUES ('Charlie')
            """)
            result = client.execute("SELECT * FROM users")
            print(f"   Got {len(result)} rows")
            print("   ✓ Multiple INSERT works")
        except Exception as e:
            print(f"   ✗ Failed: {e}")
        
        client.close()
        print("\n✓ Test completed!")
        return True

if __name__ == "__main__":
    try:
        test_multi_statement_sql()
    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
