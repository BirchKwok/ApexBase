# 文档索引

## 快速入口

- 根 README：`../README.md`
- Python API 入口：`../apexbase/python/apexbase/__init__.py`
- 测试套件说明：`../test/README.md`
- CI 发布流程：`../.github/workflows/build_release.yml`

## 使用建议

- ApexBase 的主要对外接口是 `apexbase.ApexClient`。
- 数据持久化文件为单文件 `apexbase.apex`，默认保存在 `ApexClient(dirpath=...)` 指定目录下。
- 查询推荐使用 `execute(sql)`（完整 SQL），兼容场景可以使用 `query(where, limit=...)`（只传 WHERE 表达式）。

## 本地开发（conda dev 环境）

```bash
# conda activate dev

# 本地开发安装（Rust 扩展）
maturin develop --release

# 运行测试
python run_tests.py
```

## 发布准备清单

- 版本号一致性：`pyproject.toml` 与 `Cargo.toml` 的 `version` 保持一致
- 本地通过测试：`python run_tests.py`
- 打 tag 触发 CI：推送 `v*` tag（例如 `v0.2.3`）
- 配置 PyPI Token：GitHub Secrets 中设置 `PYPI_API_TOKEN`

## 已知限制/注意事项

- 当前项目以 Python API 为主要用户入口；Rust crate 主要用于 PyO3 扩展和内部引擎复用。
- 部分高级 SQL 能力（如复杂子查询、并发锁等）可能仍在演进中，建议以 `test/` 中覆盖的行为为准。
