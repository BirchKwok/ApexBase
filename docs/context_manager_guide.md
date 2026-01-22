# ApexClient 上下文管理器使用指南

ApexClient 现在支持 Python 上下文管理器，提供更安全、更简洁的资源管理。

## 基本用法

```python
from apexbase import ApexClient

# 基本用法 - 自动管理连接
with ApexClient("./my_db") as client:
    client.store({"name": "Alice", "age": 25})
    client.store({"name": "Bob", "age": 30})
    
    # 查询数据
    results = client.query("age > 20")
    print(f"找到 {len(results)} 条记录")
    
# 连接自动关闭，无需手动调用 client.close()
```

## 链式调用

```python
# 支持链式调用
with ApexClient("./my_db").init_fts(index_fields=['name']) as client:
    client.store({"name": "Charlie", "description": "Developer"})
    
    # 使用 FTS 搜索
    results = client.query("name LIKE 'Cha%'")
    print(results.to_dict())
# FTS 索引和连接都会自动关闭
```

## 异常处理

```python
try:
    with ApexClient("./my_db") as client:
        client.store({"name": "David"})
        
        # 模拟异常
        raise ValueError("发生错误")
        
except ValueError as e:
    print(f"捕获异常: {e}")
    
# 连接仍然会自动关闭
# 注意：异常可能导致事务回滚，David 可能不会被保存
```

## 手动管理 vs 自动管理

```python
# 自动管理（推荐）
with ApexClient("./my_db") as client:
    # 自动注册到全局注册表，自动清理
    client.store({"name": "Eve"})

# 手动管理
with ApexClient("./my_db", _auto_manage=False) as client:
    # 不注册到全局注册表，完全手动控制
    client.store({"name": "Frank"})
```

## 最佳实践

1. **优先使用上下文管理器**：确保资源正确释放
2. **异常处理**：在 `with` 块外处理异常
3. **链式调用**：结合 `init_fts()` 等方法使用
4. **资源清理**：依赖自动管理，避免手动 `close()`

## 兼容性

上下文管理器与现有的手动管理方式完全兼容：

```python
# 传统方式（仍然支持）
client = ApexClient("./my_db")
try:
    client.store({"name": "Grace"})
finally:
    client.close()

# 新方式（推荐）
with ApexClient("./my_db") as client:
    client.store({"name": "Henry"})
```
