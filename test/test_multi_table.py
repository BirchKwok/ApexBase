import pytest
import os
import tempfile
import shutil
from apexbase import ApexClient

@pytest.fixture
def temp_dir():
    """创建临时目录"""
    temp_path = tempfile.mkdtemp()
    yield temp_path
    shutil.rmtree(temp_path)

def test_table_management(temp_dir):
    """测试表管理功能：创建、列出、切换、删除表"""
    client = ApexClient(temp_dir)
    
    # 测试默认表
    assert "default" in client.list_tables()
    
    # 测试创建新表
    client.create_table("users")
    client.create_table("orders")
    tables = client.list_tables()
    assert "users" in tables
    assert "orders" in tables
    assert len(tables) == 3  # default, users, orders
    
    # 测试切换表
    client.use_table("users")
    assert client.current_table == "users"
    
    # 测试删除表
    client.drop_table("orders")
    tables = client.list_tables()
    assert "orders" not in tables
    assert len(tables) == 2
    
    # 测试不能删除默认表
    with pytest.raises(ValueError):
        client.drop_table("default")

def test_multi_table_operations(temp_dir):
    """测试多表操作：在不同表中存储和查询数据"""
    client = ApexClient(temp_dir)
    
    # 创建测试表
    client.create_table("users")
    client.create_table("orders")
    
    # 在users表中存储数据
    client.use_table("users")
    user_records = [
        {"name": "John", "age": 30, "email": "john@example.com"},
        {"name": "Jane", "age": 25, "email": "jane@example.com"}
    ]
    user_ids = client.store(user_records)
    assert user_ids is not None
    assert len(user_ids) == 2
    
    # 在orders表中存储数据
    client.use_table("orders")
    order_records = [
        {"user_id": user_ids[0], "product": "Laptop", "price": 1000},
        {"user_id": user_ids[0], "product": "Mouse", "price": 50},
        {"user_id": user_ids[1], "product": "Keyboard", "price": 100}
    ]
    order_ids = client.store(order_records)
    assert order_ids is not None
    assert len(order_ids) == 3
    
    # 验证每个表的数据
    client.use_table("users")
    users = client.query("1=1")
    assert len(users) == 2
    assert all(u["name"] in ["John", "Jane"] for u in users)
    
    client.use_table("orders")
    orders = client.query("1=1")
    assert len(orders) == 3
    assert all(o["product"] in ["Laptop", "Mouse", "Keyboard"] for o in orders)
    
    # 测试跨表查询（通过代码层面关联）
    john_orders = client.query(f"user_id = {user_ids[0]}")
    assert len(john_orders) == 2
    assert all(o["user_id"] == user_ids[0] for o in john_orders)

def test_table_isolation(temp_dir):
    """测试表隔离：确保不同表的数据和结构是独立的"""
    client = ApexClient(temp_dir)
    
    # 创建两个具有不同结构的表
    client.create_table("employees")
    client.create_table("departments")
    
    # 在employees表中存储数据
    client.use_table("employees")
    employee = {
        "name": "John",
        "salary": 50000,
        "skills": ["python", "sql"]
    }
    emp_id = client.store(employee)
    assert emp_id is not None
    
    # 在departments表中存储不同结构的数据
    client.use_table("departments")
    department = {
        "name": "IT",
        "location": "New York",
        "budget": 1000000
    }
    dept_id = client.store(department)
    assert dept_id is not None
    
    # 验证字段隔离
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
    """测试多表搜索：测试每个表的搜索功能是独立的"""
    client = ApexClient(temp_dir)
    
    # 创建并配置测试表
    client.create_table("articles")
    client.create_table("comments")
    
    # 在articles表中存储数据
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
    
    # 在comments表中存储数据
    client.use_table("comments")
    comments = [
        {"text": "Great Python tutorial!", "rating": 5},
        {"text": "Nice SQL introduction", "rating": 4}
    ]
    comment_ids = client.store(comments)
    assert comment_ids is not None
    assert len(comment_ids) == 2
    client.set_searchable("text", True)
    
    # 测试articles表的搜索
    client.use_table("articles")
    python_articles = client.search_text("python")
    assert len(python_articles) == 1
    
    # 测试comments表的搜索
    client.use_table("comments")
    python_comments = client.search_text("python")
    assert len(python_comments) == 1
    
    # 验证搜索结果的独立性
    client.use_table("articles")
    sql_articles = client.search_text("sql")
    assert len(sql_articles) == 1
    
    client.use_table("comments")
    sql_comments = client.search_text("sql")
    assert len(sql_comments) == 1

def test_multi_table_concurrent_access(temp_dir):
    """测试多表并发访问"""
    import threading
    import random
    
    client = ApexClient(temp_dir)
    client.create_table("table1")
    client.create_table("table2")
    
    num_threads = 4
    records_per_thread = 50
    
    def worker():
        for _ in range(records_per_thread):
            # 随机选择表
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
    
    # 验证每个表的记录
    client.use_table("table1")
    table1_records = client.query("1=1")
    table1_count = len(table1_records)
    
    client.use_table("table2")
    table2_records = client.query("1=1")
    table2_count = len(table2_records)
    
    # 验证总记录数
    assert table1_count + table2_count == num_threads * records_per_thread
    
    # 验证每个表的记录正确性
    for table, records in [("table1", table1_records), ("table2", table2_records)]:
        client.use_table(table)
        for record in records:
            assert record["table"] == table
            assert 1 <= record["value"] <= 1000

def test_table_error_handling(temp_dir):
    """测试表操作的错误处理"""
    client = ApexClient(temp_dir)
    
    # 测试创建重复表
    client.create_table("test")
    client.create_table("test")  # 不应该抛出错误，而是静默返回
    
    # 测试使用不存在的表
    with pytest.raises(ValueError):
        client.use_table("nonexistent")
    
    # 测试删除不存在的表
    client.drop_table("nonexistent")  # 不应该抛出错误，而是静默返回
    
    # 测试删除默认表
    with pytest.raises(ValueError):
        client.drop_table("default")
    
    # 测试在不存在的表中操作
    client.use_table("test")
    client.drop_table("test")
    # 删除当前表后，应该自动切换到默认表
    assert client.current_table == "default" 