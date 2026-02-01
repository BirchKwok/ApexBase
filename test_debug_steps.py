"""Debug multi-statement SQL issue"""
import tempfile
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'apexbase', 'python'))
from apexbase import ApexClient

with tempfile.TemporaryDirectory() as temp_dir:
    client = ApexClient(dirpath=temp_dir)
    
    # Test step by step
    print('Step 1: CREATE TABLE IF NOT EXISTS users')
    try:
        client.execute('CREATE TABLE IF NOT EXISTS users')
        print('   OK')
    except Exception as e:
        print(f'   FAILED: {e}')
    
    print('Step 2: CREATE TABLE IF NOT EXISTS orders')
    try:
        client.execute('CREATE TABLE IF NOT EXISTS orders')
        print('   OK')
    except Exception as e:
        print(f'   FAILED: {e}')
    
    print('Step 3: Multi-statement CREATE TABLE')
    try:
        client.execute('CREATE TABLE IF NOT EXISTS t1; CREATE TABLE IF NOT EXISTS t2')
        print('   OK')
    except Exception as e:
        print(f'   FAILED: {e}')
    
    print('Step 4: CREATE + ALTER + INSERT')
    try:
        client.execute('CREATE TABLE IF NOT EXISTS products')
        client.execute('ALTER TABLE products ADD COLUMN name STRING')
        client.execute('ALTER TABLE products ADD COLUMN price FLOAT')
        client.execute("INSERT INTO products (name, price) VALUES ('Laptop', 999.99)")
        print('   OK')
    except Exception as e:
        print(f'   FAILED: {e}')
    
    print('Step 5: Multi-statement DDL + DML')
    try:
        client.execute('''
            CREATE TABLE IF NOT EXISTS products2;
            ALTER TABLE products2 ADD COLUMN name STRING;
            ALTER TABLE products2 ADD COLUMN price FLOAT;
            INSERT INTO products2 (name, price) VALUES ('Laptop', 999.99)
        ''')
        print('   OK')
    except Exception as e:
        print(f'   FAILED: {e}')
        import traceback
        traceback.print_exc()
    
    tables = client.list_tables()
    print(f'\nTables: {tables}')
    
    client.close()
    print('\nDone!')
