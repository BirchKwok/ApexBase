# ApexBase HTAP 全盘差距分析

> 基于 2026-02-08 代码 review (第三版)，对标 SQLite (嵌入式 OLTP)、DuckDB (嵌入式 OLAP)、TiDB/CockroachDB (HTAP) 的核心能力。
>
> **重要更新（第四版）：** P0 全部完成。P1 全部完成（含 Snapshot Isolation 初步接入）。P2 新增 7 项已实现：
> DECIMAL 类型、JSON 类型+函数、ANALYZE 统计收集、UPSERT、Savepoint、Parquet 导入/导出、Snapshot Isolation（VersionStore 已接入事务读路径）。
> **第四版新增完成：** LZ4 压缩（per-RG 自动 LZ4，透明解压）、CHECK 约束、FOREIGN KEY 约束（CASCADE/RESTRICT/SET NULL）、递归 CTE（WITH RECURSIVE + 迭代定点）。
> **第五版新增完成：** Zstd 压缩（默认，per-RG 自动 Zstd，LZ4 fallback，透明解压；save_deletion_vectors 压缩 RG 检测+回退 save_v4）。
> CBO 基础（TableStats 缓存 + 代价模型 + 选择率估算 + Join 顺序优化）、Statement-level 回滚（隐式 Savepoint）、GC 自动触发（接入 TxnManager commit）。
> **第六版新增完成：** AUTOINCREMENT 列（parser + 序列化 + INSERT 自动填充）、WAL auto-checkpoint（save 后自动截断 WAL）、
> Delta 文件原子写（write-tmp-then-rename 防崩溃）+ 残留 .tmp 清理、Per-RG zone map 剪枝（mmap 过滤读跳过不匹配 RG）。
> **第七版新增完成：** CBO 接入执行器（pre-lookup selectivity estimation + DML 后自动 invalidate stats）、
> RLE 编码（Int64 sorted/low-cardinality 列 Run-Length Encoding）、Bit-packing 编码（窄整型列位压缩）。
> 编码层透明集成 V4 格式（encoding_version=1 in RG header byte 29，backward-compatible）。
> **第八版新增完成：** Join 顺序优化接入 executor（INNER JOIN 链按右表行数升序重排，star join 模式）、
> 覆盖索引 / Index-Only Scan（等值谓词下 SELECT 列全在索引内时跳过回表）、
> Bool 列 RLE 编码（COL_ENCODING_RLE_BOOL=3，长 true/false 连续段压缩）、
> plan_with_stats() 完整接入 executor（CBO 路由决定是否跳过索引检查，聚合/全扫描查询不再触发 IndexManager）。
> P2 全部完成，无剩余差距。
> **第十版新增完成：** SAVEPOINT/ROLLBACK TO/RELEASE 事务内路由修复（Rust bindings 直接走 TxnManager）、
> 崩溃恢复测试（WAL recovery 跨 session + 事务持久化验证）、事务隔离测试（Read-your-writes + Savepoint 部分回滚 + 并发读写隔离）。
> 文档全面更新：SQL 兼容性/事务系统/索引系统/查询优化器均升级为 ✅；并发控制升级为 ✅ 基本成熟；测试覆盖薄弱项修复。206 Rust + 807 Python tests 全部通过。

---

## 一、总评

ApexBase 当前是一个 **功能全面的嵌入式列存 HTAP 数据库**，在分析查询性能上已经能与 DuckDB 竞争（13 项 benchmark 赢 10 项），
已具备事务支持（含 Snapshot Isolation）、约束系统、崩溃恢复、Parquet 互操作和丰富的 SQL 方言。
**P2 全部完成**，已达到生产级 HTAP 核心功能全覆盖。

### 成熟度打分（5分制）

| 维度 | 得分 | 说明 |
|------|:----:|------|
| 列存存储引擎 | ⭐⭐⭐⭐⭐ | V4 RG 格式、mmap、dict encoding、deletion vectors、原子写 — 生产级 |
| OLAP 查询性能 | ⭐⭐⭐⭐ | 1M 行 benchmark 多数指标领先 DuckDB/SQLite |
| SQL 兼容性 | ⭐⭐⭐⭐⭐ | CRUD + JOIN 5种 + UNION + Window + CTE + UPSERT + CTAS + EXPLAIN + Parquet + JSON + DECIMAL + Date/Timestamp |
| OLTP 事务 (ACID) | ⭐⭐⭐⭐ | OCC + WAL 事务边界 + CRC32 + 原子 COMMIT + Snapshot Isolation + Savepoint |
| 索引系统 | ⭐⭐⭐⭐½ | B-Tree + Hash 可用，等值/IN/范围加速 + 覆盖索引 + 多索引 AND 交集 + CBO 选择性 |
| **数据完整性** | ⭐⭐⭐⭐⭐ | NOT NULL / UNIQUE / PRIMARY KEY / DEFAULT / CHECK / FOREIGN KEY 约束全部实现 |
| 并发控制 | ⭐⭐⭐⭐ | mmap 快照读 + Lock-free 热路径 + fs2 文件锁 retry + lock-free row_count，10 个并发压力测试全部通过 |
| 崩溃恢复 | ⭐⭐⭐⭐ | WAL v2 (CRC32) + 事务边界 + 原子写 + recovery 过滤 + 幂等 replay |

---

## 二、逐模块差距详解

### 1. 存储引擎 — ✅ 基本成熟

**已有：**
- V4 Row Group 列存格式，append-only RG 追加写
- Per-RG deletion vectors（原地删除，无需全文件重写）
- Dict encoding（低基数字符串自动压缩）
- mmap zero-copy 读路径
- Delta writes (.apex.delta) 增量写

**缺失：**

| 缺失项 | 重要性 | 说明 |
|--------|:------:|------|
| ✅ **通用压缩** (Zstd 默认 + LZ4 fallback) | 🔴 高 | 已实现：save_v4/append_row_group 默认 Zstd 压缩（level 1），LZ4 fallback，读取时透明解压（zstd + lz4_flex）。save_deletion_vectors 自动检测压缩 RG 并回退 save_v4() |
| ✅ **RLE 编码** (Run-Length Encoding) | 🟡 中 | 已实现：Int64 sorted/low-cardinality 列自动 RLE 编码（需 ≥30% 空间节省），per-column encoding prefix（encoding_version=1） |
| ✅ **Bit-packing** (整型) | 🟡 中 | 已实现：窄整型列自动 Bit-packing（bit_width < 48，需 ≥30% 空间节省），与 RLE 竞争取小者 |
| ✅ **Zone maps / Min-Max per-RG** | 🔴 高 | 已实现：Per-RG per-column zone maps (Int64/Float64)，存储在 V4Footer。mmap 过滤读路径支持 per-RG zone map 剪枝（zone_map_prune_rgs + scan_columns_mmap_skip_rgs） |
| ✅ **Parquet 导入/导出** | 🟡 中 | 已实现：COPY table TO/FROM 'file.parquet'（基于 arrow-parquet crate） |
| ✅ **列级统计信息** (NDV, histogram) | 🟡 中 | 已实现：ANALYZE table 收集 NDV/min/max/null_count/row_count |

### 2. SQL 兼容性 — ✅ 全面

**已有：**
- SELECT / INSERT / UPDATE / DELETE / TRUNCATE
- WHERE (AND/OR/NOT/LIKE/IN/BETWEEN/IS NULL/REGEXP)
- GROUP BY / HAVING / ORDER BY / LIMIT / OFFSET / DISTINCT
- INNER JOIN / LEFT JOIN / RIGHT JOIN / FULL OUTER JOIN / CROSS JOIN（带 hash join 优化）
- UNION / UNION ALL
- Window Functions: ROW_NUMBER, RANK, DENSE_RANK, NTILE, LAG, LEAD 等 17 种
- Subquery: FROM 子查询、IN 子查询、EXISTS、Scalar 子查询（骨架）
- CREATE/DROP TABLE/INDEX, ALTER TABLE ADD/DROP/RENAME COLUMN
- CREATE/DROP VIEW（基础）
- CASE WHEN / CAST / 标量函数（LENGTH, UPPER, LOWER, SUBSTR, COALESCE 等）
- Multi-statement SQL（分号分隔）
- BEGIN/COMMIT/ROLLBACK

**缺失：**

| 缺失项 | 重要性 | 说明 |
|--------|:------:|------|
| ✅ **CTE (WITH ... AS)** | 🔴 高 | 已实现：多 CTE、嵌套 CTE、CTE + JOIN（非递归） |
| ✅ **递归 CTE (WITH RECURSIVE)** | 🟡 中 | 已实现：迭代定点算法，支持列别名、UNION ALL 锚定+递归，层次查询/图遍历/数列生成 |
| ✅ **RIGHT JOIN / FULL OUTER JOIN / CROSS JOIN** | 🟡 中 | 已实现：JoinType 支持 Inner/Left/Right/Full/Cross |
| ✅ **INSERT ... SELECT** | 🔴 高 | 已实现：支持 WHERE/ORDER/LIMIT/GROUP BY |
| ✅ **INSERT ... ON CONFLICT (UPSERT)** | 🟡 中 | 已实现：DO NOTHING / DO UPDATE SET |
| ✅ **多列 GROUP BY 表达式** | 🟡 中 | 已实现：`GROUP BY YEAR(date), city`，parser 支持表达式解析（parse_group_by_list），executor 自动物化虚拟列 |
| ✅ **完整子查询执行** | 🟡 中 | 已实现：IN 子查询/EXISTS/Scalar 子查询（含 correlated），执行器完整支持 |
| ✅ **EXPLAIN / EXPLAIN ANALYZE** | 🔴 高 | 已实现：输出查询计划 + 实际执行统计 |
| ✅ **CREATE TABLE AS SELECT (CTAS)** | 🟡 中 | 已实现：CREATE TABLE ... AS SELECT ... (含 IF NOT EXISTS) |
| ✅ **Date/Time 数据类型** | 🔴 高 | 已实现：原生 TIMESTAMP/DATE 类型，Arrow 输出 TimestampMicrosecondArray/Date32Array |
| ✅ **DECIMAL 精确数值类型** | 🟡 中 | 已实现：DataType::Decimal (i128, 16 bytes) |
| ✅ **JSON 类型和函数** | 🟡 中 | 已实现：DataType::Json + JSON_EXTRACT/JSON_VALUE/JSON_SET/JSON_ARRAY_LENGTH |
| **数组/嵌套类型** | 🟠 低 | DataType::Array 枚举已定义，但无 ColumnData 存储变体 |
| **MERGE / 多表 UPDATE/DELETE** | 🟠 低 | 复杂 DML |

### 3. 事务系统 (ACID) — ✅ 完整

**已有：**
- `TxnManager` 全局单例，OCC 冲突检测
- `TxnContext` 缓冲 INSERT/DELETE/UPDATE，COMMIT 时批量应用
- `ConflictDetector` 读写/写写冲突检测
- `VersionStore` 行版本链（`RowVersion` with begin_ts/end_ts）
- `SnapshotManager` 快照管理
- `GarbageCollector` 旧版本回收
- Python 绑定层 `BEGIN/COMMIT/ROLLBACK`
- ✅ WAL 事务边界（TxnBegin/TxnCommit/TxnRollback 记录类型 5/6/7）
- ✅ WAL DML 带 txn_id（INSERT_TXN/DELETE_TXN 记录类型 8/9）
- ✅ WAL-first atomic COMMIT（先写 WAL DML + COMMIT 标记，再 apply）
- ✅ Recovery 事务过滤（只回放 auto-commit 和已提交事务的 DML）
- ✅ Read-your-writes（事务内 SELECT 可见 buffered writes overlay）

**剩余差距：**

| 缺失项 | 重要性 | 说明 |
|--------|:------:|------|
| ✅ **跨事务隔离 (Snapshot Isolation)** | 🔴 高 | 已实现：VersionStore 已接入 execute_in_txn 读路径，snapshot_ts 可见性判断 |
| ✅ **Savepoint** | 🟡 中 | 已实现：SAVEPOINT name / ROLLBACK TO name / RELEASE name |
| ✅ **Statement-level 回滚** | 🟡 中 | 已实现：每条 DML 自动创建隐式 Savepoint，失败时只回滚该语句，事务保持活跃 |
| **死锁检测** | � 低 | OCC 理论上无死锁，但未来若加锁需要 |

### 4. 并发控制 — ✅ 成熟

**已有：**
- `parking_lot::RwLock` 保护所有内存结构
- 读写分离（多读单写）
- `rayon` 并行 GROUP BY / 聚合
- ✅ `fs2` 跨进程文件锁已接入（`bindings.rs` 中 shared/exclusive lock）
- ✅ **文件锁 retry + 指数退避**：100µs → 5ms 退避，最长 50ms 等待，避免并发下假性 WouldBlock 错误
- ✅ mmap 快照读：持久化数据读取不阻塞写操作（de-facto snapshot isolation for disk data）
- ✅ **Lock-free 读路径**：`cached_footer_offset` AtomicU64 替代 header RwLock，所有热读路径（to_arrow_batch, read_columns, is_v4_format 等 10+ 处）避免锁竞争
- ✅ **Lock-free row_count**：count_rows / row_count 不再获取文件锁，直接读 AtomicU64
- ✅ 多线程并发读：10 线程 × 50 查询压力测试通过
- ✅ 多线程并发写：5 writer + 5 reader 线程混合压力测试通过
- ✅ 多线程 SQL 混合查询：8 线程 × 24 并发 SQL（COUNT/GROUP BY/ORDER BY/BETWEEN/AVG）全部通过
- ✅ 数据完整性验证：6 线程 × 50 写入后逐线程验证行数正确
- ✅ 并发快照一致性：5 线程并发读验证 a+b=100 恒等式不变

**已解决：**

| 已解决项 | 说明 |
|----------|------|
| ✅ **mmap 快照读** | mmap 读路径提供 de-facto 快照读（读磁盘数据不阻塞写），写走 WAL 内存 buffer |
| ✅ **Reader-Writer 隔离** | mmap 路径读取已持久化数据时不阻塞写操作；RwLock 允许多并发读 |
| ✅ **文件锁 retry 防抖** | acquire_lock 指数退避重试（100µs→5ms，50ms 超时），消除并发下 spurious lock failure |
| ✅ **Lock-free V4 检测** | cached_footer_offset AtomicU64 替代 header RwLock 在所有热读路径 |
| ✅ **Lock-free row_count** | row_count 不获取文件锁，直接读 AtomicU64 active_count |

**剩余差距（P3）：**

| 缺失项 | 重要性 | 说明 |
|--------|:------:|------|
| **写操作期间内存 buffer 互斥** | 🟠 低 | RwLock write guard 期间读被阻塞，但临界区很短（append + WAL write），对嵌入式场景影响极小 |
| **连接池 / Session 管理** | 🟠 低 | 嵌入式场景通常不需要，但多线程应用需要 |

### 5. 崩溃恢复 — ✅ 基本可靠

**已有：**
- WAL 文件 (.apex.wal)：记录 Insert/Delete/BatchInsert/Checkpoint
- WAL recovery：`open_with_durability()` 重放 WAL 记录
- `DurabilityLevel`: Fast (无 fsync) / Safe (flush 时 fsync) / Max (每写 fsync)
- ✅ WAL v2 per-record CRC32 校验（`crc32fast::hash`，读取时验证）
- ✅ WAL 事务边界记录（TxnBegin/TxnCommit/TxnRollback）
- ✅ Recovery 事务过滤（只回放 auto-commit + 已提交事务）
- ✅ save_v4() 原子写（write `.apex.tmp` + `std::fs::rename`）
- ✅ open_with_durability() 清理残留 `.tmp` 文件

**剩余差距：**

| 缺失项 | 重要性 | 说明 |
|--------|:------:|------|
| ✅ **WAL replay 幂等** | 🟡 中 | 已修复：基于 base_next_id 过滤已持久化的 Insert/BatchInsert/Delete 记录 |
| ✅ **Delta 文件 recovery** | 🟡 中 | 已实现：DeltaStore save 使用 write-tmp-then-rename 原子写；open_with_durability() 清理残留 .deltastore.tmp 文件 |
| ✅ **WAL auto-checkpoint** | 🟡 中 | 已实现：save_v4() 成功后自动调用 checkpoint_wal()，截断 WAL 文件防止无限增长 |

### 6. 索引系统 — ✅ 成熟

**已有：**
- B-Tree 索引 + Hash 索引
- CREATE/DROP INDEX SQL
- WHERE col = X / WHERE col IN (...) 索引加速
- DML 操作自动维护索引

**缺失：**

| 缺失项 | 重要性 | 说明 |
|--------|:------:|------|
| ✅ **范围查询索引加速** | 🔴 高 | 已实现：B-Tree 索引加速 >, >=, <, <=, BETWEEN 查询 |
| ✅ **多索引 AND 交集** | 🟡 中 | 已实现：WHERE col1=X AND col2=Y 可利用多个单列索引交集加速 |
| ✅ **覆盖索引 (Covering Index)** | 🟡 中 | 已实现：Index-Only Scan — 等值谓词下 SELECT 列全在索引内时直接从索引构建结果，跳过回表 |
| **自动索引建议** | 🟠 低 | 基于查询历史推荐索引 |
| ✅ **Python store() API 更新索引** | 🔴 高 | 已修复：engine.rs write()/write_typed() 完成后调用 notify_indexes_after_write() |
| ✅ **索引选择性统计** | 🟡 中 | 已实现：CBO 代价模型接入 executor try_index_accelerated_read（pre-lookup selectivity estimation），DML 后自动 invalidate_table_stats |

### 7. 数据完整性 / 约束系统 — ✅ 基础已实现

**已有：**
- PRIMARY KEY 约束（DDL 语法 + INSERT/UPDATE 唯一性检查）
- UNIQUE 约束（DDL 语法 + INSERT/UPDATE 检查，含批量去重）
- NOT NULL 约束（DDL 语法 + INSERT/UPDATE 检查，含缺省列检测）
- DEFAULT 值（DDL 语法 + INSERT 时自动填充）
- 约束序列化/持久化到 V4Footer schema
- 9 Rust + 18 Python 约束测试

**缺失：**

| 缺失项 | 重要性 | 说明 |
|--------|:------:|------|
| ✅ **CHECK 约束** | 🟡 中 | 已实现：CHECK(expr) DDL + INSERT/UPDATE 时自动校验 |
| ✅ **FOREIGN KEY 约束** | 🟡 中 | 已实现：REFERENCES parent(col) + ON DELETE CASCADE/RESTRICT/SET NULL |
| ✅ **自增列 (AUTOINCREMENT)** | 🟡 中 | 已实现：AUTOINCREMENT / AUTO_INCREMENT 关键字解析、约束序列化（bit4=0x10）、INSERT 时自动查找 max+1 填充 |

### 8. 查询优化器 — ✅ CBO 全面接入

**已有：**
- `query/planner.rs`：基础查询特征分析
- 手写快速路径（COUNT(*) 直接返回、LIMIT push-down、string filter 等）
- 索引加速（等值查询）
- Late materialization (先过滤再读全列)

**缺失：**

| 缺失项 | 重要性 | 说明 |
|--------|:------:|------|
| ✅ **Cost-Based Optimizer (CBO) 基础** | 🔴 高 | 已实现：TableStats 缓存 + 代价模型 + 选择率估算 + plan_with_stats()，CBO 已接入 executor try_index_accelerated_read（pre-lookup selectivity），DML 后自动 invalidate stats |
| ✅ **统计信息收集 (ANALYZE)** | 🔴 高 | 已实现：ANALYZE table 收集 NDV/min/max/null_count/row_count |
| ✅ **Join 顺序优化** | 🟡 中 | 已实现：INNER JOIN 链按右表行数升序重排（star join 模式），plan_with_stats() 完整接入 executor |
| ✅ **谓词下推到存储层** | 🟡 中 | 已实现：系统性数值谓词下推（try_numeric_predicate_pushdown），支持 col >/>=/</<=/=/!= N，自动路由到 storage-level filtered read |
| ✅ **子查询去相关化** | 🟡 中 | 已实现：EXISTS/IN 相关子查询自动去相关化为 hash semi-join（try_decorrelate_exists/try_decorrelate_in），O(N+M) vs O(N*M) |
| **公共子表达式消除** | 🟠 低 | 重复表达式只计算一次 |
| ✅ **Projection push-down** | 🟡 中 | 已实现：required_columns() + get_col_refs() 系统性应用于所有 SELECT 读路径（含 predicate pushdown 路径） |
| ✅ **EXPLAIN 输出** | 🔴 高 | 已实现：EXPLAIN + EXPLAIN ANALYZE |

### 9. 死代码 / 未接入模块

以下模块有完整实现和单元测试，但**未完全接入实际数据路径**：

| 模块 | 位置 | 单测 | 状态 |
|------|------|:----:|------|
| **DeltaStore** (cell-level updates) | `storage/delta/` | 9 | ✅ 已接入 executor UPDATE 路径 + to_arrow_batch_mmap overlay |
| **VersionStore** (MVCC 行版本) | `storage/mvcc/version_store.rs` | ✅ | ✅ 已接入 execute_in_txn 读路径 (snapshot_ts 可见性) |
| **GarbageCollector** (旧版本回收) | `storage/mvcc/gc.rs` | ✅ | ✅ 已接入 TxnManager commit 路径，maybe_run() 自动触发 |
| **Horizontal Scaling** | `scaling/` | 18 | ❌ 完全 standalone，无任何接入点 |
| **CompactTypedColumn** (SSO + 字符串池) | `table/compact.rs` | — | ❌ 模块已移除，字符串压缩已由 Dict encoding + Zstd 覆盖 |
| **Query Planner** | `query/planner.rs` | 2 | ✅ CBO 全面接入 executor：plan_with_stats() 路由 + pre-lookup selectivity + Join 重排 + DML 后 invalidate_table_stats |

### 10. 测试覆盖

**当前状态：** 206 Rust + 817 Python tests（全部通过，0 跳过，0 xfail）

**已改善：**

| 领域 | 现状 |
|------|------|
| ✅ **并发测试** | 多线程并发读/写/混合测试全部通过（test_concurrent_reads, test_concurrent_writes, test_concurrent_reads_writes 等） |
| ✅ **事务测试** | 23 个事务测试覆盖 BEGIN/COMMIT/ROLLBACK/Multi-DML/READ ONLY/Savepoint/崩溃恢复/并发隔离 |
| ✅ **并发压力测试** | 10 个专项测试：多线程读写混合、SQL 并发、多表并发、事务并发、数据完整性、快照一致性、吞吐量测量 |
| ✅ **崩溃恢复测试** | WAL replay 幂等 + 跨 session 持久化 + 增量写入恢复 |
| ✅ **边界条件** | NULL/大数据/资源耗尽/特殊字符测试全面覆盖 |

**剩余薄弱区域（P3）：**

| 领域 | 现状 |
|------|------|
| **大数据量测试** | benchmark 有 1M 行，但无 10M+/100M+ 压力测试 |
| **Fuzz testing** | 无 SQL fuzzer |
| **多进程并发** | 跨进程文件锁已接入，但无多进程集成测试 |

---

## 三、优先级排序：走向真正 HTAP 的路线图

### P0 — 不修不能叫 HTAP（致命差距）— ✅ 全部完成

1. ✅ **save_v4() 原子写** — write-new (.apex.tmp) + rename
2. ✅ **WAL per-record CRC32** — WAL v2 格式，读取时验证
3. ✅ **WAL 事务边界** — TxnBegin/TxnCommit/TxnRollback + DML txn_id
4. ✅ **事务原子性** — WAL-first COMMIT (4-phase protocol)
5. ✅ **约束系统** — NOT NULL / UNIQUE / PRIMARY KEY / DEFAULT
6. ✅ **MVCC Phase A** — Read-your-writes in transaction
7. ✅ **跨进程文件锁** — fs2 shared/exclusive lock in bindings.rs
8. ✅ **Zone Maps (batch-level)** — Int64/Float64 min/max pruning in executor

### P1 — SQL 兼容性与 OLAP 增强 — ✅ 全部完成

1. ✅ **CTE (WITH ... AS)** — 已实现，支持多 CTE、嵌套 CTE、CTE + JOIN
2. ✅ **INSERT ... SELECT** — 已实现，支持 WHERE/ORDER/LIMIT/GROUP BY
3. ✅ **EXPLAIN / EXPLAIN ANALYZE** — 已实现，输出查询计划 + 实际执行统计
4. ✅ **RIGHT/FULL OUTER/CROSS JOIN** — 已实现，SQL 兼容性
5. ✅ **隔离级别 (Snapshot Isolation)** — 已实现，VersionStore 接入 execute_in_txn 读路径
6. ✅ **Date/Timestamp 类型** — 已实现，原生 TIMESTAMP/DATE 类型
7. ✅ **Zstd/LZ4 压缩** — 已实现：默认 Zstd，LZ4 fallback
8. ✅ **B-Tree 范围查询加速** — 已实现，索引加速 >/>=/</<=/BETWEEN
9. ✅ **Per-RG Zone Maps (存储层)** — 已实现，Int64/Float64 min/max per-RG per-column
10. ✅ **CREATE TABLE AS SELECT (CTAS)** — 已实现，含 IF NOT EXISTS

### P2 — 竞争力提升（当前优先级）

11. ✅ **Zstd/LZ4 压缩** — per-RG 默认 Zstd 压缩，LZ4 fallback，磁盘 I/O 优化
12. ✅ **CHECK 约束** — `CHECK(age > 0)` 数据完整性
13. ✅ **FOREIGN KEY 约束** — 表间引用完整性
14. ✅ **递归 CTE (WITH RECURSIVE)** — 层次查询、图遍历
15. ✅ **Cost-Based Optimizer** — CBO 接入 executor（pre-lookup selectivity + stats invalidation on DML）
16. ✅ **RLE / Bit-packing 编码** — Int64 列自动 RLE/Bit-pack 编码（encoding_version=1）
17. ✅ **覆盖索引** — Index-Only Scan（等值谓词跳过回表）
18. ✅ 崩溃恢复增强（Delta 原子写 + WAL auto-checkpoint）
19. ✅ **Join 顺序优化接入** — INNER JOIN 链按右表行数升序重排
20. ✅ **Bool 列 RLE 编码** — 长 true/false 连续段 Run-Length Encoding
21. ✅ **plan_with_stats() 完整接入** — CBO 路由决定执行策略

**P2 已完成项：**
- ✅ DECIMAL 精确类型 — DataType::Decimal (i128)
- ✅ JSON 类型+函数 — DataType::Json + JSON_EXTRACT/JSON_VALUE/JSON_SET
- ✅ 统计信息收集 (ANALYZE) — NDV/min/max/null_count/row_count
- ✅ INSERT ON CONFLICT (UPSERT) — DO NOTHING / DO UPDATE SET
- ✅ Savepoint — SAVEPOINT / ROLLBACK TO / RELEASE
- ✅ Parquet 导入/导出 — COPY table TO/FROM 'file.parquet'
- ✅ CHECK 约束 — CHECK(expr) DDL + INSERT/UPDATE 校验
- ✅ FOREIGN KEY 约束 — REFERENCES + CASCADE/RESTRICT/SET NULL
- ✅ 递归 CTE — WITH RECURSIVE 迭代定点算法
- ✅ AUTOINCREMENT 列 — parser + 序列化 + INSERT 自动填充
- ✅ WAL auto-checkpoint — save 后截断 WAL
- ✅ Delta 文件原子写 — write-tmp-then-rename + 残留清理
- ✅ Per-RG zone map 剪枝 — mmap 过滤读跳过不匹配 RG
- ✅ CBO 接入执行器 — pre-lookup selectivity estimation + DML 后 invalidate stats
- ✅ RLE 编码 — Int64 sorted/low-cardinality 列 Run-Length Encoding
- ✅ Bit-packing 编码 — 窄整型列位压缩（bit_width < 48）
- ✅ 覆盖索引 / Index-Only Scan — 等值谓词下 SELECT 列全在索引内时跳过回表
- ✅ Join 顺序优化接入 executor — INNER JOIN 链按右表行数升序重排（star join）
- ✅ Bool 列 RLE 编码 — COL_ENCODING_RLE_BOOL=3
- ✅ plan_with_stats() 完整接入 — CBO 路由决策（聚合/全扫描跳过索引检查）

### P3 — 锦上添花（低优先级）

27. 嵌套/数组类型
28. 水平扩展 (Scaling)
29. SQL Fuzzer
30. 自动索引建议
31. 公共子表达式消除
32. 死锁检测

---

## 四、与竞品对比

| 能力 | ApexBase | SQLite | DuckDB |
|------|:--------:|:------:|:------:|
| 列存引擎 | ✅ V4 RG | ❌ B-Tree 行存 | ✅ 向量化列存 |
| OLAP 性能 | ⭐⭐⭐⭐ | ⭐ | ⭐⭐⭐⭐⭐ |
| OLTP 性能 | ⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐ |
| 事务 ACID | ✅ OCC+MVCC+Savepoint | ✅ 完整 WAL | ✅ 完整 MVCC |
| SQL 兼容性 | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| 崩溃恢复 | ✅ WAL v2 + 原子写 | ✅ WAL + journal | ✅ WAL |
| 并发控制 | RwLock + fs2 文件锁 | 文件锁 + WAL | 进程内 MVCC |
| 约束系统 | ✅ PK/UNIQUE/NOT NULL | ✅ 完整 | ✅ 完整 |
| 压缩 | ✅ Zstd + LZ4 + Dict + RLE + BitPack | ❌ | LZ4 + Zstd + Delta + RLE |
| 索引 | B-Tree + Hash (等值+范围) | B-Tree (范围) | ART + 自适应 |
| CTE | ✅ 含递归 | ✅ 含递归 | ✅ 含递归 |
| Window Func | ✅ 17种 (Int64+Float64) | ✅ 完整 | ✅ 完整 |
| Date/Time 类型 | ✅ TIMESTAMP/DATE | ✅ | ✅ |
| JSON | ✅ JSON_EXTRACT/SET/VALUE | ✅ json1 | ✅ |
| EXPLAIN | ✅ + ANALYZE | ✅ | ✅ |
| Parquet | ✅ COPY TO/FROM | ❌ | ✅ 原生 |
| UPSERT | ✅ ON CONFLICT | ✅ | ✅ |
| Savepoint | ✅ | ✅ | ✅ |
| DECIMAL | ✅ i128 | ✅ | ✅ |
| ANALYZE | ✅ NDV/min/max | ✅ | ✅ |

---

## 五、核心结论

ApexBase 的 **存储引擎和 OLAP 查询性能是真正的强项**，在 1M 行 benchmark 上已超越 DuckDB 和 SQLite。
P0 + P1 全部达标。P2 已完成 22 项（DECIMAL、JSON、ANALYZE、UPSERT、Savepoint、Parquet、Snapshot Isolation、Zstd 压缩、CHECK 约束、FOREIGN KEY 约束、递归 CTE、AUTOINCREMENT、WAL checkpoint、Delta 原子写、Zone map 剪枝、CBO 接入执行器、RLE 编码、Bit-packing 编码、覆盖索引、Join 顺序优化、Bool RLE 编码、plan_with_stats 完整接入）。**P2 全部完成。**

✅ 已完成：

1. **Zstd + LZ4 压缩** — per-RG 默认 Zstd 压缩（level 1），LZ4 fallback，读取时透明解压（zstd + lz4_flex）。save_deletion_vectors 检测压缩 RG 自动回退 save_v4()
2. **CHECK 约束** — CHECK(expr) DDL + INSERT/UPDATE 时自动校验
3. **FOREIGN KEY 约束** — REFERENCES parent(col) + INSERT/UPDATE 引用完整性检查
4. **递归 CTE** — WITH RECURSIVE 迭代定点算法，支持列别名、层次查询、数列生成
5. **AUTOINCREMENT 列** — AUTOINCREMENT/AUTO_INCREMENT 解析 + bit4 序列化 + INSERT 时 max+1 自动填充
6. **WAL auto-checkpoint** — save_v4() 成功后截断 WAL（checkpoint_wal），防止无限增长
7. **Delta 文件原子写** — DeltaStore.save() 使用 write-tmp-then-rename；open_with_durability() 清理残留 .deltastore.tmp
8. **Per-RG zone map 剪枝** — mmap 过滤读路径（read_columns_filtered_mmap）基于 zone maps 跳过不匹配的 Row Groups
9. **CBO 接入执行器** — pre-lookup selectivity estimation（try_index_accelerated_read 中基于 ANALYZE stats 判断 index vs scan），DML 后自动 invalidate_table_stats
10. **RLE 编码** — Int64 sorted/low-cardinality 列 Run-Length Encoding（≥16 元素 + ≥30% 空间节省才启用），per-RG per-column 编码选择
11. **Bit-packing 编码** — 窄整型列位压缩（bit_width < 48 + ≥30% 空间节省），与 RLE 竞争取小者。V4 格式 backward-compatible（encoding_version byte 29）
12. **覆盖索引 (Index-Only Scan)** — 等值谓词下 SELECT 列全在索引内时跳过回表，直接从索引数据构建 Arrow RecordBatch
13. **Join 顺序优化接入 executor** — INNER JOIN 链按右表行数升序重排（star join 模式），maybe_reorder_joins()
14. **Bool 列 RLE 编码** — COL_ENCODING_RLE_BOOL=3，长 true/false 连续段 Run-Length Encoding（≥30% 空间节省才启用）
15. **plan_with_stats() 完整接入** — CBO 路由决策：聚合/全扫描查询跳过 IndexManager 检查，避免不必要的索引加载开销

**P2 全部完成，无剩余差距。**

**第九版新增完成：**
16. **多列 GROUP BY 表达式** — `GROUP BY YEAR(date), city`，parser 支持 parse_group_by_list，executor materialize_group_by_exprs 自动物化虚拟列
17. **谓词下推到存储层** — 系统性数值谓词下推（try_numeric_predicate_pushdown），支持 col >/>=/</<=/=/!= literal，自动路由到 storage-level filtered read
18. **子查询去相关化** — EXISTS/IN 相关子查询自动转换为 hash semi-join（try_decorrelate_exists/try_decorrelate_in），O(N+M) vs O(N*M)
19. **Projection push-down** — required_columns() + get_col_refs() 系统性应用于所有 SELECT 读路径

**第十版新增完成：**
20. **SAVEPOINT/ROLLBACK TO/RELEASE 事务内支持** — Rust bindings 直接路由到 TxnManager savepoint/rollback_to_savepoint/release_savepoint（不再经 SQL parser 落入 execute_parsed_multi 报错）
21. **崩溃恢复测试** — WAL recovery 跨 session、增量多 session 写入恢复、已提交事务持久化、已回滚事务不持久化
22. **事务隔离测试** — Read-your-writes、Savepoint 部分回滚、并发读写隔离（3 reader + 1 writer 线程无错误）、多表事务
23. **文档全面更新** — SQL 兼容性 ✅ 全面、事务系统 ✅ 完整、索引系统 ✅ 成熟、查询优化器 ✅ CBO 全面接入

**第十一版新增完成（并发控制完善）：**
24. **文件锁 retry + 指数退避** — acquire_lock 从 100µs→5ms 指数退避，最长 50ms 等待，消除并发下 spurious WouldBlock 错误
25. **Lock-free V4 检测** — `cached_footer_offset` AtomicU64 替代 header RwLock，10+ 处热读路径（to_arrow_batch, read_columns, is_v4_format 等）避免锁竞争
26. **Lock-free row_count** — row_count/count_rows 不再获取文件锁，直接读 AtomicU64 active_count
27. **并发压力测试** — 10 个专项测试：10 线程×50 查询、5 writer+5 reader 混合、8 线程×24 SQL 并发、多表并发写、事务并发、6 线程数据完整性、快照一致性、吞吐量测量
28. **并发控制升级 ⭐⭐⭐⭐** — 从 ⭐⭐⭐½ 提升到 ⭐⭐⭐⭐。206 Rust + 817 Python tests 全部通过

下一步重点：P3 锦上添花（嵌套类型、水平扩展、SQL Fuzzer 等）。
