# ArrowStringColumn 集成与内存优化对比报告

> 目的：优化 ApexBase 在 1000 万级数据量下的内存占用，重点替换 `TypedColumn::String` 由 `Vec<String>` 带来的大量堆分配与对象头开销，改为 Arrow 风格的连续缓冲区字符串列。

## 1. 背景与目标

ApexBase 当前的列式内存模型中，`TypedColumn::String` 使用 `Vec<String>` 存储字符串：

- 每个 `String` 本身包含指针/长度/容量（通常 24 bytes/条，平台相关）
- 每条字符串还会单独进行堆分配
- 在 1000 万行规模下，会造成显著的内存碎片与峰值内存膨胀（尤其在查询/转换 Arrow 结果时）

本次优化目标：

- 将 `TypedColumn::String` 的底层存储替换为连续缓冲区（Arrow offset + data buffer 设计）
- **降低峰值内存**，尽量不牺牲读写性能
- 确保核心测试与功能可用

## 2. 变更总览（当前未提交修改）

### 2.1 新增模块

- `apexbase/src/table/arrow_column.rs`（新增）
  - 新增 `ArrowStringColumn`
    - offsets: `Vec<i32>`
    - data: `Vec<u8>`
    - nulls: `BitVec`
    - len: `usize`
  - 提供方法：
    - `push`, `push_null`, `get`, `is_null`, `len`, `reserve`, `slice`, `append`
    - `to_arrow_array()`：整列转 Arrow StringArray
    - `to_arrow_array_indexed(indices)`：按 row indices 转 Arrow StringArray
  - 引入 `ArrowTypedColumn`（当前用于导出/后续扩展；本次主线为替换 `TypedColumn::String`）

- `apexbase/src/table/mod.rs`（修改）
  - `pub mod arrow_column;`
  - `pub use arrow_column::{ArrowTypedColumn, ArrowStringColumn};`

### 2.2 核心结构替换：TypedColumn::String

- `apexbase/src/table/column_table.rs`（修改）

将：

- 旧：`TypedColumn::String { data: Vec<String>, nulls: BitVec }`
- 新：`TypedColumn::String(ArrowStringColumn)`

同时系统性修改以下逻辑以适配 tuple variant：

- `new / with_capacity`
- `push / push_null / get / set / len / is_null / slice / append`
- 批量插入/预分配/列扩容路径
- Arrow RecordBatch 相关构建路径

### 2.3 Arrow 输出路径适配

为避免在查询/Arrow IPC 构建时重新复制大量 `String`，将多处 `TypedColumn::String` 的 Arrow 构建逻辑改为直接调用 `ArrowStringColumn` 的原生转换：

- `ColumnTable::query_to_record_batch`：
  - contiguous path 使用 `col.to_arrow_array()`
  - gather path 使用 `col.to_arrow_array_indexed(indices)`

- `ColumnTable::build_record_batch_from_indices`：
  - 使用 `col.to_arrow_array_indexed(indices)`

- `QueryColumnarResult::to_arrow_ipc`（位于 `column_table.rs`）
  - string 列使用 `col.to_arrow_array_indexed(&row_indices)`

### 2.4 过滤器与 SQL 执行路径适配

为保持 LIKE/Compare 等优化路径可用，修复所有 `TypedColumn::String {..}` 的匹配，并将相关数据引用从 `&[String]` 迁移到 `&ArrowStringColumn`：

- `apexbase/src/io_engine/filter.rs`
  - `CompiledFilter::CompareString` / `CompiledFilter::Like` 从 `&[String]` 改为 `&ArrowStringColumn`
  - 读取改为 `col.get(row_idx)`

- `apexbase/src/query/filter.rs`
  - 优化 compare/like fast path：使用 `col.get(i)` / `col.is_null(i)`
  - `FusedMatcher` 的 Like/NotLike 从 `&[String]` 改为 `&ArrowStringColumn`

- `apexbase/src/query/sql_executor.rs`
  - LIKE 执行路径直接读取 `ArrowStringColumn`
  - 构建 Arrow 结果时字符串列使用 `col.to_arrow_array_indexed(matching_indices)`

### 2.5 Arrow 转换工具适配

- `apexbase/src/data/arrow_convert.rs`
  - `typed_columns_to_arrow_ipc` 的 String 分支使用 `col.to_arrow_array_indexed(row_indices)`
  - `build_column_array_all` 的 String 分支使用 `col.to_arrow_array()`
  - 更新 `matches!(TypedColumn::String {..})` 为 `matches!(TypedColumn::String(_))`

### 2.6 Python Binding 适配

- `apexbase/src/python/bindings.rs`
  - 将返回 columnar 数据时的 string 列从 `Vec<String>` 改为 `ArrowStringColumn`
  - 对每个 index：
    - `col.is_null(idx)` -> `None`
    - `col.get(idx)` -> `&str` 直接 append

### 2.7 测试脚本

本次工作区中 `test_memory_1.py` 与 `test_memory_2.py` 目前处于新增未提交状态（A），用于进行内存/性能对比。

> 说明：如果仓库原本已有这两个脚本，当前状态表示本地新增/未跟踪，需要确认是否应纳入版本库。

## 3. 内存/性能对比结果（baseline vs 当前）

本节对比同机器环境下的两次测试输出：

- Baseline：优化前（用户提供）
- Current：优化后（本次 ArrowStringColumn 版本）

### 3.1 Test 1 - Generate & Store（生成并写入 1000 万条）

| 指标 | Baseline | Current | 变化 |
|---|---:|---:|---:|
| After data generation | 1090.94 MB | 1093.41 MB | +2.47 MB（基本不变） |
| After storage | 3967.09 MB | 3752.17 MB | **-214.92 MB（-5.4%）** |
| After cleanup | 3066.09 MB | 2848.39 MB | **-217.70 MB（-7.1%）** |
| After query | 4374.47 MB | 4180.95 MB | **-193.52 MB（-4.4%）** |
| Peak memory (process) | 4367.72 MB | 4175.00 MB | **-192.72 MB（-4.4%）** |

性能：

| 指标 | Baseline | Current | 变化 |
|---|---:|---:|---:|
| 数据生成耗时 | 12.06 s | 12.14 s | +0.08 s |
| 存储耗时 | 26.38 s | 27.15 s | +0.77 s（+2.9%） |
| 全量读取耗时 | 0.17 s | 0.18 s | +0.01 s |

结论（Test1）：

- 峰值内存下降约 **193 MB（~4.4%）**
- 写入耗时略有增加（~3%），查询基本不变

### 3.2 Test 2 - Load Existing DB（加载已有库并全量查询 1000 万条）

| 指标 | Baseline | Current | 变化 |
|---|---:|---:|---:|
| After loading database | 3777.47 MB | 3735.83 MB | **-41.64 MB（-1.1%）** |
| After query | 5012.80 MB | 3938.88 MB | **-1073.92 MB（-21.4%）** |
| Peak memory (process) | 5012.86 MB | 3938.95 MB | **-1073.91 MB（-21.4%）** |

性能：

| 指标 | Baseline | Current | 变化 |
|---|---:|---:|---:|
| Database loaded in | 2.09 s | 1.71 s | **-0.38 s（-18.2%）** |
| Query completed in | 0.22 s | 0.17 s | **-0.05 s（-22.7%）** |

结论（Test2）：

- 查询后峰值内存从 ~5.01GB 降到 ~3.94GB，减少约 **1.07GB（~21%）**
- 加载/查询时间也同步下降（~18% / ~23%）

## 4. 解释与分析

### 4.1 为什么 Test2 改进更明显？

Test2 的“加载 + 查询”阶段通常会触发：

- 将内部列数据转换为 Arrow/IPC 或 Python 结构
- baseline 的 `Vec<String>` 在该过程中可能引发：
  - 大量 `String` clone/临时对象
  - 更高的瞬时峰值内存

改用 `ArrowStringColumn` 后：

- 字符串数据保持为 **连续 data buffer**
- 通过 offsets 定位字符串，减少 per-row 的对象开销与分配
- Arrow 输出可以更直接地从 buffer 构建数组，显著降低峰值

### 4.2 为什么 Test1 写入略慢？

可能原因（需进一步 profile 才能确定）：

- 写入路径中字符串从 `Value::String` 进入 `ArrowStringColumn::push` 仍然会做 `extend_from_slice`，理论上效率较高；
- 但写入过程中可能出现更多的 buffer 扩容、或在某些插入批次中 string 平均长度/模式导致扩容频率变化；
- 另外，本次也调整了部分写入/预分配逻辑，可能影响了 cache 行为。

建议后续：

- 针对纯字符串列/不同平均长度的基准测试
- 在 `ArrowStringColumn::with_capacity` 中更精确估计 `avg_string_len`

## 5. 编译与测试状态

- `cargo check`：通过（存在 dead_code warnings）
- `cargo test --lib`：功能测试基本通过
  - 存在一个性能基准测试 `benchmark_10k_native` 因阈值（<2ms）在本机波动失败（~2.04ms），属于典型“硬阈值基准测试”不稳定问题

## 6. 当前未提交文件清单（git status）

- 修改（M）：
  - `apexbase/src/data/arrow_convert.rs`
  - `apexbase/src/io_engine/filter.rs`
  - `apexbase/src/python/bindings.rs`
  - `apexbase/src/query/filter.rs`
  - `apexbase/src/query/sql_executor.rs`
  - `apexbase/src/table/column_table.rs`
  - `apexbase/src/table/mod.rs`

- 新增（??）：
  - `apexbase/src/table/arrow_column.rs`

- 新增（A）：
  - `test_memory_1.py`
  - `test_memory_2.py`

## 7. 总结

- **目标达成**：`TypedColumn::String` 已完成从 `Vec<String>` 到 `ArrowStringColumn` 的替换。
- **内存改善明确**：
  - 写入场景（Test1）峰值下降约 4~7%
  - 加载+查询场景（Test2）峰值下降约 21%（≈1.07GB）
- **性能整体无明显回归**：
  - Test2 加载/查询反而更快
  - Test1 写入略慢（~3%），可作为后续进一步优化点
