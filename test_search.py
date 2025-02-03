from apexbase import ApexClient
import random
import string
import time

client = ApexClient("./test_db", drop_if_exists=True, backend="sqlite")

def generate_random_string(length: int) -> str:
    """生成随机字符串"""
    return ''.join(random.choices(string.ascii_letters + string.digits, k=length))

def generate_test_records(count: int) -> list:
    """生成测试记录"""
    records = []
    for _ in range(count):
        record = {
            "name": generate_random_string(10),
            "age": random.randint(18, 80),
            "email": f"{generate_random_string(8)}@example.com",
            "tags": [generate_random_string(5) for _ in range(random.randint(1, 5))],
            "address": {
                "city": generate_random_string(8),
                "street": generate_random_string(15),
                "number": random.randint(1, 1000)
            }
        }
        records.append(record)
    return records

# 生成测试数据
print("生成测试数据...")
records = generate_test_records(10000)  # 增加到1万条记录

# 添加一个包含"example"的记录
records.append({
    "name": "Test Example",
    "age": 30,
    "email": "test@example.com",
    "tags": ["example", "test"],
    "address": {
        "city": "Example City",
        "street": "Example Street",
        "number": 123
    }
})

# 存储数据
print("存储数据...")
start_time = time.time()
ids = client.store(records)
store_time = time.time() - start_time
print(f"存储 {len(records)} 条记录耗时: {store_time:.2f} 秒")

# 执行搜索
print("\n执行搜索...")
start_time = time.time()
results = client.search_text("example").to_pandas()
search_time = time.time() - start_time
print(f"搜索耗时: {search_time:.3f} 秒")
print(f"找到 {len(results)} 条记录包含 'example'")
print("\n搜索结果示例:")
print(results)
