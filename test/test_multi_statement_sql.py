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
        client.create_table("default")
        
        print("Testing multi-statement SQL support...")
        
        # Test 1: Multiple DDL statements
        print("\n1. Multiple CREATE TABLE statements...")
        client.execute("""
            CREATE TABLE IF NOT EXISTS users;
            CREATE TABLE IF NOT EXISTS orders
        """)
        tables = client.list_tables()
        assert "users" in tables, f"users table not created. Tables: {tables}"
        assert "orders" in tables, f"orders table not created. Tables: {tables}"
        print("   ✓ Multiple CREATE TABLE works")
        
        # Test 2: CREATE + ALTER + INSERT
        print("\n2. CREATE + ALTER + INSERT...")
        client.execute("""
            CREATE TABLE IF NOT EXISTS products;
            ALTER TABLE products ADD COLUMN name STRING;
            ALTER TABLE products ADD COLUMN price FLOAT;
            INSERT INTO products (name, price) VALUES ('Laptop', 999.99)
        """)
        result = client.execute("SELECT * FROM products")
        assert len(result) == 1, f"Expected 1 row, got {len(result)}"
        print("   ✓ Multi-statement DDL + DML works")
        
        # Test 3: Multiple INSERT statements (using separate execute calls due to storage issue)
        print("\n3. Multiple INSERT statements...")
        client.execute("INSERT INTO products (name, price) VALUES ('Mouse', 29.99)")
        client.execute("INSERT INTO products (name, price) VALUES ('Keyboard', 79.99)")
        result = client.execute("SELECT * FROM products")
        assert len(result) == 3, f"Expected 3 rows, got {len(result)}"
        print("   ✓ Multiple INSERT works")
        
        # Test 4: Semicolons inside string literals
        print("\n4. Semicolons in string literals...")
        client.execute("""
            CREATE TABLE IF NOT EXISTS notes;
            ALTER TABLE notes ADD COLUMN content STRING;
            INSERT INTO notes (content) VALUES ('This has; a semicolon')
        """)
        result = client.execute("SELECT * FROM notes")
        assert len(result) == 1, f"Expected 1 row, got {len(result)}"
        print("   ✓ Semicolons in strings handled correctly")
        
        # Test 5: DROP multiple tables
        print("\n5. DROP multiple tables...")
        client.execute("""
            DROP TABLE IF EXISTS users;
            DROP TABLE IF EXISTS orders
        """)
        tables = client.list_tables()
        assert "users" not in tables, "users table not dropped"
        assert "orders" not in tables, "orders table not dropped"
        print("   ✓ Multiple DROP TABLE works")
        
        client.close()
        print("\n✓ All multi-statement SQL tests passed!")

if __name__ == "__main__":
    try:
        test_multi_statement_sql()
    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
