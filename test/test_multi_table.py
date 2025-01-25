import pytest
import os
import tempfile
import shutil
from apexbase import ApexClient

@pytest.fixture
def temp_dir():
    """Create a temporary directory"""
    temp_path = tempfile.mkdtemp()
    yield temp_path
    shutil.rmtree(temp_path)

def test_table_management(temp_dir):
    """Test table management: create, list, switch, delete tables"""
    client = ApexClient(temp_dir)
    
    # Test default table
    assert "default" in client.list_tables()
    
    # Test create new tables
    client.create_table("users")
    client.create_table("orders")
    tables = client.list_tables()
    assert "users" in tables
    assert "orders" in tables
    assert len(tables) == 3  # default, users, orders
    
    # Test switch table
    client.use_table("users")
    assert client.current_table == "users"
    
    # Test delete table
    client.drop_table("orders")
    tables = client.list_tables()
    assert "orders" not in tables
    assert len(tables) == 2
    
    # Test cannot delete default table
    with pytest.raises(ValueError):
        client.drop_table("default")

def test_multi_table_operations(temp_dir):
    """Test multi-table operations: store and query data in different tables"""
    client = ApexClient(temp_dir)
    
    # Create test tables
    client.create_table("users")
    client.create_table("orders")
    
    # Store data in users table
    client.use_table("users")
    user_records = [
        {"name": "John", "age": 30, "email": "john@example.com"},
        {"name": "Jane", "age": 25, "email": "jane@example.com"}
    ]
    user_ids = client.store(user_records)
    assert user_ids is not None
    assert len(user_ids) == 2
    
    # Store data in orders table
    client.use_table("orders")
    order_records = [
        {"user_id": user_ids[0], "product": "Laptop", "price": 1000},
        {"user_id": user_ids[0], "product": "Mouse", "price": 50},
        {"user_id": user_ids[1], "product": "Keyboard", "price": 100}
    ]
    order_ids = client.store(order_records)
    assert order_ids is not None
    assert len(order_ids) == 3
    
    # Verify data in each table
    client.use_table("users")
    users = client.query("1=1")
    assert len(users) == 2
    assert all(u["name"] in ["John", "Jane"] for u in users)
    
    client.use_table("orders")
    orders = client.query("1=1")
    assert len(orders) == 3
    assert all(o["product"] in ["Laptop", "Mouse", "Keyboard"] for o in orders)
    
    # Test cross-table query (associated through code level)
    john_orders = client.query(f"user_id = {user_ids[0]}")
    assert len(john_orders) == 2
    assert all(o["user_id"] == user_ids[0] for o in john_orders)

def test_table_isolation(temp_dir):
    """Test table isolation: ensure data and structure in different tables are independent"""
    client = ApexClient(temp_dir)
    
    # Create two tables with different structures
    client.create_table("employees")
    client.create_table("departments")
    
    # Store data in employees table
    client.use_table("employees")
    employee = {
        "name": "John",
        "salary": 50000,
        "skills": ["python", "sql"]
    }
    emp_id = client.store(employee)
    assert emp_id is not None
    
    # Store different structure data in departments table
    client.use_table("departments")
    department = {
        "name": "IT",
        "location": "New York",
        "budget": 1000000
    }
    dept_id = client.store(department)
    assert dept_id is not None
    
    # Verify field isolation
    client.use_table("employees")
    employee_fields = client.list_fields()
    assert "salary" in employee_fields
    assert "skills" in employee_fields
    assert "budget" not in employee_fields
    
    client.use_table("departments")
    department_fields = client.list_fields()
    assert "budget" in department_fields
    assert "salary" not in department_fields
    assert "skills" not in department_fields

def test_multi_table_search(temp_dir):
    """Test multi-table search: test each table's search function is independent"""
    client = ApexClient(temp_dir)
    
    # Create and configure test tables
    client.create_table("articles")
    client.create_table("comments")
    
    # Store data in articles table
    client.use_table("articles")
    articles = [
        {"title": "Python Tutorial", "content": "Learn Python programming"},
        {"title": "SQL Basics", "content": "Introduction to SQL"}
    ]
    article_ids = client.store(articles)
    assert article_ids is not None
    assert len(article_ids) == 2
    client.set_searchable("title", True)
    client.set_searchable("content", True)
    
    # Store data in comments table
    client.use_table("comments")
    comments = [
        {"text": "Great Python tutorial!", "rating": 5},
        {"text": "Nice SQL introduction", "rating": 4}
    ]
    comment_ids = client.store(comments)
    assert comment_ids is not None
    assert len(comment_ids) == 2
    client.set_searchable("text", True)
    
    # Test articles table's search
    client.use_table("articles")
    python_articles = client.search_text("python")
    assert len(python_articles) == 1
    
    # Test comments table's search
    client.use_table("comments")
    python_comments = client.search_text("python")
    assert len(python_comments) == 1
    
    # Verify search result independence
    client.use_table("articles")
    sql_articles = client.search_text("sql")
    assert len(sql_articles) == 1
    
    client.use_table("comments")
    sql_comments = client.search_text("sql")
    assert len(sql_comments) == 1

def test_multi_table_concurrent_access(temp_dir):
    """Test multi-table concurrent access"""
    import threading
    import random
    
    client = ApexClient(temp_dir)
    client.create_table("table1")
    client.create_table("table2")
    
    num_threads = 4
    records_per_thread = 50
    
    def worker():
        for _ in range(records_per_thread):
            # Randomly select table
            table = random.choice(["table1", "table2"])
            client.use_table(table)
            record = {
                "value": random.randint(1, 1000),
                "thread_id": threading.get_ident(),
                "table": table
            }
            record_id = client.store(record)
            assert record_id is not None
    
    threads = [threading.Thread(target=worker) for _ in range(num_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    
    # Verify records in each table
    client.use_table("table1")
    table1_records = client.query("1=1")
    table1_count = len(table1_records)
    
    client.use_table("table2")
    table2_records = client.query("1=1")
    table2_count = len(table2_records)
    
    # Verify total record count
    assert table1_count + table2_count == num_threads * records_per_thread
    
    # Verify records in each table
    for table, records in [("table1", table1_records), ("table2", table2_records)]:
        client.use_table(table)
        for record in records:
            assert record["table"] == table
            assert 1 <= record["value"] <= 1000

def test_table_error_handling(temp_dir):
    """Test table operation error handling"""
    client = ApexClient(temp_dir)
    
    # Test create duplicate table
    client.create_table("test")
    client.create_table("test")  # Should not raise error, but return silently
    
    # Test use nonexistent table
    with pytest.raises(ValueError):
        client.use_table("nonexistent")
    
    # Test delete nonexistent table
    client.drop_table("nonexistent")  # Should not raise error, but return silently
    
    # Test delete default table
    with pytest.raises(ValueError):
        client.drop_table("default")
    
    # Test operation in nonexistent table
    client.use_table("test")
    client.drop_table("test")
    # After deleting current table, it should automatically switch to default table
    assert client.current_table == "default" 