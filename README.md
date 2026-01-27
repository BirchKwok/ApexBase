# ApexBase

**High-performance embedded database with Rust core and Python API**

ApexBase 是一个基于 Rust 核心的高性能嵌入式数据库，提供简洁的 Python API。

## ✨ 特性

- 🚀 **高性能** - Rust 核心，批量写入速度可达 97万+ ops/s
- 📦 **单文件存储** - 自定义 `.apex` 文件格式，无需外部依赖
- 🔍 **全文搜索** - 集成 NanoFTS，支持中文和模糊搜索
- 🐍 **Python 友好** - 简洁的 API，支持 Pandas/Polars/PyArrow
- 💾 **紧凑存储** - 相比传统方案节省约 45% 存储空间

## 📦 安装

```bash
# 从 PyPI 安装
pip install apexbase

# 从源码构建（推荐在 conda dev 环境中）
# conda activate dev
maturin develop --release
```

## 🚀 快速开始

```python
from apexbase import ApexClient

# 创建客户端
client = ApexClient("./data")

# 存储数据
client.store({"name": "Alice", "age": 30, "city": "Beijing"})
client.store([
    {"name": "Bob", "age": 25},
    {"name": "Charlie", "age": 35}
])

# SQL 查询（推荐）
results = client.execute("SELECT * FROM default WHERE age > 28")

# 也支持传入过滤表达式（兼容用法）
results2 = client.query("age > 28", limit=100)

# 按 _id 检索（_id 为内部自增 ID）
record = client.retrieve(0)
all_data = client.retrieve_all()

# 全文搜索
client.init_fts(index_fields=["name", "city"], lazy_load=True)
doc_ids = client.search_text("Alice")
records = client.search_and_retrieve("Beijing")

# 转换为 DataFrame
df = results.to_pandas()
pl_df = results.to_polars()

# 关闭连接
client.close()
```

## 📊 性能对比

| 操作 | ApexBase (Rust) | 传统方案 | 提升 |
|------|-----------------|----------|------|
| 批量写入 (10K) | 17ms | 57ms | **3.3x** |
| 单条检索 | 0.01ms | 0.4ms | **40x** |
| 批量检索 (100) | 0.08ms | 1.1ms | **14x** |
| 存储大小 | 2.1 MB | 3.9 MB | **1.8x 更小** |

## 📁 项目结构

```
ApexBase/
├── apexbase/                    # 主包目录
│   ├── src/                     # Rust 源代码
│   │   ├── storage/             # 存储引擎
│   │   ├── table/               # 表管理
│   │   ├── query/               # 查询执行器
│   │   ├── index/               # B-tree 索引
│   │   ├── cache/               # LRU 缓存
│   │   ├── data/                # 数据类型
│   │   └── python/              # PyO3 绑定
│   ├── python/                  # Python 包装层
│   │   └── apexbase/
│   │       └── __init__.py      # Python API
│   ├── Cargo.toml
│   └── pyproject.toml
├── Cargo.toml                   # 工作区配置
└── pyproject.toml               # 项目配置
```

## 🔧 API 参考

### ApexClient

```python
# 初始化
client = ApexClient(
    dirpath="./data",           # 数据目录
    drop_if_exists=False,       # 是否删除已存在的数据
    batch_size=1000,
    enable_cache=True,
    cache_size=10000,
    prefer_arrow_format=True,
    durability="fast",         # fast | safe | max
)

# 表操作
client.create_table("users")
client.use_table("users")
client.drop_table("users")
tables = client.list_tables()

# CRUD 操作
client.store({"key": "value"})
client.store([{...}, {...}])
record = client.retrieve(0)
records = client.retrieve_many([1, 2, 3])
client.replace(0, {"new": "data"})
client.delete(0)
client.delete([1, 2, 3])

# 查询
results = client.query("age > 30")
results = client.query("name LIKE 'A%'")
results = client.execute("SELECT name, age FROM default ORDER BY age DESC LIMIT 10")
count = client.count_rows()

# 全文搜索
client.init_fts(index_fields=["title", "content"], lazy_load=True)
ids = client.search_text("keyword")
ids = client.fuzzy_search_text("keywrd")  # 模糊搜索
records = client.search_and_retrieve("keyword")

# 数据框架集成
client.from_pandas(df)
client.from_polars(df)
results.to_pandas()
results.to_polars()
results.to_arrow()
```

## 🧪 开发与测试

```bash
# 运行测试（conda dev 环境推荐）
# conda activate dev
python run_tests.py

# 或直接 pytest
pytest -q
```

## 📦 发布流程（GitHub Actions）

当前仓库已提供基于 tag 的自动构建与发布流程：当推送 `v*` tag 时，会运行测试、构建 wheels/sdist 并使用 `twine` 发布到 PyPI。

- **Workflow**: `.github/workflows/build_release.yml`
- **Tag**: `v0.2.3` 这类格式
- **Secret**: `PYPI_API_TOKEN`

## 📚 文档

项目文档入口：`docs/README.md`

## 📄 License

Apache-2.0
