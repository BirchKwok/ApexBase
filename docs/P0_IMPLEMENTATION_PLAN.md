# P0 Implementation Plan — ApexBase HTAP 基线修复

> 目标：修复 6 项致命差距，使 ApexBase 达到"可信赖的嵌入式数据库"基线。
> 原则：每完成一步 `cargo test` + `pytest`，确保不回退。

---

## 执行顺序与依赖关系

```
P0-1 save_v4() 原子写 ─────────────────────────────┐
P0-2 WAL per-record CRC32 ─────────────┐            │
P0-3 WAL 事务边界 ─────────────────────┤            │
P0-4 事务原子性 (依赖 P0-2, P0-3) ────┘            │
P0-5 约束系统 (PRIMARY KEY / NOT NULL / UNIQUE) ────┤ (独立)
P0-6 MVCC 读路径 (最大改动，最后做) ────────────────┘
```

**推荐顺序：P0-1 → P0-2 → P0-3 → P0-4 → P0-5 → P0-6**

---

## P0-1: save_v4() 原子写

**问题：** `save_v4()` 使用 `truncate(true)` 打开文件后全量重写。写到一半 crash = 数据全丢。

**方案：** Write-new + Rename (标准 atomic write pattern)

**改动文件：** `apexbase/src/storage/on_demand.rs`

**步骤：**
1. `save_v4()` 写入临时文件 `<path>.tmp` 而非直接写入 `<path>`
2. 写完并 flush/fsync 后，`std::fs::rename("<path>.tmp", "<path>")`
3. rename 在 POSIX 上是原子操作，Windows 上用 `ReplaceFile` 语义
4. 失败时删除 `.tmp` 文件，原文件不受影响

**关键代码改动：**
```rust
// Before:
let file = OpenOptions::new().write(true).create(true).truncate(true).open(&self.path)?;

// After:
let tmp_path = self.path.with_extension("apex.tmp");
let file = OpenOptions::new().write(true).create(true).truncate(true).open(&tmp_path)?;
// ... write all data ...
writer.flush()?;
if self.durability == DurabilityLevel::Max {
    writer.get_ref().sync_all()?;
}
drop(writer);
std::fs::rename(&tmp_path, &self.path)?;
```

**注意事项：**
- `append_row_group()` / `write_row_group_to_disk()` 是追加写，不需要改（中途 crash 只丢新 RG，旧数据不丢）
- 需要在 `open_with_durability()` 中检测并清理残留 `.tmp` 文件
- Windows 上 `rename` 可能需要先 close 所有 file handles

**验证：** 所有现有测试通过 + 新增测试模拟 crash 场景（写 .tmp 后不 rename，reopen 应读到旧数据）

---

## P0-2: WAL Per-Record CRC32

**问题：** WAL 记录无校验和，断电导致半写记录 → `from_bytes` 解析出垃圾数据。

**方案：** 每条 WAL 记录追加 4 字节 CRC32

**改动文件：** `apexbase/src/storage/incremental.rs`

**新记录格式：**
```
[record_type:u8][timestamp:i64][record_length:u32][record_data...][crc32:u32]
```

**步骤：**
1. `WalRecord::to_bytes()` — 末尾追加 `crc32fast::hash(&buf)` 4 字节
2. `WalRecord::from_bytes()` — 先读 record_type + timestamp + length + data，再读 4 字节 CRC
3. 计算实际 CRC 与读取值比较，不匹配 → 截断到此位置，视为 WAL 末尾
4. `WalReader::read_all()` — CRC 校验失败时 `break`（已有类似逻辑），返回已验证的记录
5. WAL 版本号 `WAL_VERSION` 从 1 → 2

**向后兼容：**
- WAL v1（无 CRC）仍可读取：`from_bytes` 检测版本，v1 跳过 CRC 校验
- 新写入总是 v2 格式

**验证：** 现有 WAL 测试通过 + 新增 CRC 损坏测试（手动截断/修改 WAL 字节）

---

## P0-3: WAL 事务边界

**问题：** WAL 只有 INSERT/DELETE 记录，无事务标记。Recovery 无法区分已提交和未完成的事务。

**方案：** 新增 WAL 记录类型标记事务边界

**改动文件：** `apexbase/src/storage/incremental.rs`

**新记录类型：**
```rust
const RECORD_TXN_BEGIN: u8 = 5;    // [txn_id: u64]
const RECORD_TXN_COMMIT: u8 = 6;   // [txn_id: u64]
const RECORD_TXN_ROLLBACK: u8 = 7; // [txn_id: u64]
```

**每条 DML 记录新增字段：**
```rust
// Insert record 新增 txn_id 字段（0 = 非事务 auto-commit）
WalRecord::Insert { id, data, txn_id: u64 }
WalRecord::Delete { id, txn_id: u64 }
```

**Recovery 逻辑变更 (`WalReader`)：**
1. 扫描所有记录，收集 `txn_id → Vec<WalRecord>` 和 `committed_txns: HashSet<u64>`
2. 只回放 `committed_txns` 中的事务记录 + `txn_id == 0`（auto-commit）记录
3. 忽略有 BEGIN 但无 COMMIT 的事务记录

**步骤：**
1. 扩展 `WalRecord` enum 新增 `TxnBegin/TxnCommit/TxnRollback`
2. 更新 `to_bytes()` / `from_bytes()` 序列化
3. 更新 `WalWriter::append()` 支持新记录类型
4. 在 `on_demand.rs` 的 WAL recovery 路径中加入事务过滤逻辑
5. 在 `executor.rs` 的 `execute_begin/commit/rollback` 中写入 WAL 事务标记

**验证：** 新增测试：BEGIN → INSERT → crash（无 COMMIT）→ reopen → 数据不应存在

---

## P0-4: 事务原子性

**问题：** COMMIT 是顺序执行 buffered DML，中途 crash = 部分写入。

**方案：** WAL-based atomic commit — 先写 WAL，再应用，最后写 COMMIT 标记

**改动文件：** `apexbase/src/query/executor.rs`, `apexbase/src/txn/manager.rs`

**新 COMMIT 流程：**
```
1. OCC 冲突检测（已有）
2. 将 TxnContext 中的 buffered writes 写入 WAL（带 txn_id）
3. 写入 RECORD_TXN_COMMIT
4. fsync WAL（durability 保证）
5. 应用 writes 到存储引擎（execute_insert / execute_delete / delta_update）
6. 如果 step 5 中途 crash → recovery 时 WAL 有完整 COMMIT → redo
```

**步骤：**
1. `execute_commit_txn()` 重构：先写 WAL 所有 DML + COMMIT 标记
2. WAL fsync 后再应用 writes
3. Recovery 路径：检测到 COMMIT 标记 → redo 该事务所有 DML
4. 应用完成后可选清理 WAL（checkpoint）

**依赖：** P0-2 (CRC) + P0-3 (事务边界)

**验证：** 事务隔离测试 + crash recovery 测试

---

## P0-5: 约束系统 (PRIMARY KEY / NOT NULL / UNIQUE)

**问题：** 无任何数据完整性约束，任何值都能插入。

**方案：** 在 schema 层扩展约束定义，在 DML 执行时检查

**改动文件：**
- `apexbase/src/storage/on_demand.rs` — schema 扩展
- `apexbase/src/query/sql_parser.rs` — DDL 语法
- `apexbase/src/query/executor.rs` — 约束检查
- `apexbase/src/python/bindings.rs` — Python store() 约束检查

### 5a: Schema 扩展

```rust
// 在 OnDemandSchema 或单独的 ConstraintStore 中
pub struct ColumnConstraint {
    pub not_null: bool,
    pub unique: bool,
    pub primary_key: bool,
    pub default_value: Option<Value>,
}

pub struct TableConstraints {
    pub primary_key: Option<Vec<String>>,  // PK 列名
    pub unique_constraints: Vec<Vec<String>>,  // UNIQUE 约束
    pub column_constraints: HashMap<String, ColumnConstraint>,
}
```

### 5b: DDL 语法

```sql
CREATE TABLE users (
    user_id INT NOT NULL PRIMARY KEY,
    name TEXT NOT NULL,
    email TEXT UNIQUE,
    age INT DEFAULT 0
);
```

**解析扩展：** `ColumnDef` 新增 `constraints: Vec<ColumnConstraintKind>` 字段

### 5c: DML 约束检查

- **INSERT**: 检查 NOT NULL（非空）、UNIQUE（唯一）、PK（主键唯一）
- **UPDATE**: 同上
- **DELETE**: 暂不支持 FK，无级联删除需求

**UNIQUE 检查实现：**
- 如果有 UNIQUE INDEX → 查索引
- 无索引时 → 全列扫描（小表可接受，大表应建议建索引）

### 5d: 约束持久化

约束存入 V4Footer schema 或独立 `.constraints` 文件。
推荐扩展 `OnDemandSchema` 序列化格式（schema v2）。

**注意：** Python API 不新增公共方法（遵守用户规则），约束通过 SQL DDL 定义，通过 `execute()` 调用。

**验证：** 新增约束测试（NOT NULL 拒绝、UNIQUE 重复拒绝、PK 重复拒绝）

---

## P0-6: MVCC 读路径接入

**问题：** VersionStore 完整实现但 SELECT 从不检查版本可见性。

**方案：** 分阶段接入

### Phase A: 事务内写可见（最小改动）

当前 `execute_in_txn()` 的 SELECT 不看 buffered writes。
改为：事务内 SELECT 先查 TxnContext buffered writes，overlay 到存储结果上。

### Phase B: 跨事务隔离（大改动）

1. INSERT/UPDATE/DELETE 写入 VersionStore（而非直接写存储）
2. SELECT 经过 VersionStore 可见性检查 (`Snapshot::is_visible`)
3. COMMIT 后将 VersionStore 条目合并到存储层
4. GC 清理旧版本

**改动文件：**
- `apexbase/src/query/executor.rs` — 读路径经过 VersionStore
- `apexbase/src/storage/mvcc/version_store.rs` — 接入存储
- `apexbase/src/txn/manager.rs` — COMMIT 时合并

**这是最大的改动项，建议独立分支进行。**

**验证：**
- 事务 A 写入 → 事务 B 看不到（隔离性）
- 事务 A COMMIT → 事务 B 新快照可见
- 并发读写不阻塞

---

## 测试策略

每个 P0 完成后运行：
```bash
# Rust tests
cargo test --lib 2>&1 | tail -5

# Python tests
conda activate dev
cd /Users/guobingming/RustroverProjects/ApexBase
maturin develop --release 2>&1 | tail -3
pytest test/ -x -q 2>&1 | tail -5
```

---

## 风险与注意事项

1. **P0-1 最安全** — 纯写路径改动，不影响读
2. **P0-2/3 需要 WAL 版本迁移** — 旧格式文件要能正常打开
3. **P0-4 改变 COMMIT 语义** — 必须确保 buffered write → WAL → apply 链路正确
4. **P0-5 涉及 schema 格式变更** — V4Footer 需要向后兼容
5. **P0-6 改动面最大** — 读路径核心逻辑变更，建议独立分支 + 充分测试
6. **不新增 Python 公共 API** — 约束和事务全部通过 `execute()` SQL 接口

---

## 进度追踪

| # | 任务 | 状态 | 完成日期 |
|---|------|:----:|:--------:|
| P0-1 | save_v4() 原子写 | ✅ | (previously done) |
| P0-2 | WAL per-record CRC32 | ✅ | (previously done) |
| P0-3 | WAL 事务边界 (txn_id in DML records + recovery filter) | ✅ | 2025-07 |
| P0-4 | 事务原子性 (WAL-first DML before apply) | ✅ | 2025-07 |
| P0-5 | 约束系统 (NOT NULL/UNIQUE/PK/DEFAULT + persistence) | ✅ | 2025-07 |
| P0-6 | MVCC 读路径 Phase A (read-your-writes in txn) | ✅ | 2025-07 |

### P0-5 详细完成项
- NOT NULL 约束检查 (INSERT/UPDATE)
- UNIQUE / PRIMARY KEY 唯一性检查 (INSERT/UPDATE，含批量去重)
- DEFAULT 值填充 (INSERT 缺省列)
- 约束序列化 (to_bytes/from_bytes 含 default value)
- save_v4() 约束保留修复
- 9 Rust + 18 Python 约束测试

### P0-3/4 详细完成项
- WAL Insert/Delete 记录新增 txn_id 字段 (新 record type 8/9，向后兼容)
- WAL 恢复过滤：仅回放 auto-commit (txn_id=0) 和已提交事务的 DML
- COMMIT 时先写 DML 到 WAL (带 txn_id)，再 apply 到 storage

### P0-6 Phase A 详细完成项
- execute_in_txn SELECT 支持 buffered write overlay (insert/delete/update)
- Python/Rust bindings 路由 SELECT within txn 到 transaction-aware 路径
- rows_to_apex_result 辅助函数：Value 行转 Arrow RecordBatch
