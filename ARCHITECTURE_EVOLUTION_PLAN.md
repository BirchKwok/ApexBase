# ApexBase 渐进式架构演进计划

> 本文件是 ApexBase 架构演进工作的权威路线文件，用于防止执行过程偏离最初目标。
> 本文件不替代 `AGENTS.md`；执行任何工作时，必须同时遵守两份文件，其中更严格的约束优先。

## 1. 文件治理规则（强制）

1. 未经用户明确许可，任何 coding agent 都不得修改、删除、重命名、移动、覆盖、简化或重新生成本文件。
2. “继续”“按计划推进”“开始下一步”等指令只授权执行计划中的工作，不构成修改本文件的许可。
3. 只有用户明确表达“允许修改 `ARCHITECTURE_EVOLUTION_PLAN.md`”“更新计划文件”或同等明确含义时，agent 才能修改本文件。
4. 用户对本文件的一次修改许可仅适用于当次明确指定的内容，不得解释为永久授权或扩展到其他章节。
5. 阶段状态、完成比例、验收结果和路线文字都属于本文件内容；即使实际进度发生变化，未经用户明确许可也不得自行更新。
6. 如果建议调整路线，agent 必须先在对话中说明现状、偏差、原因、影响和拟议变更，等待用户明确许可后才能修改本文件。
7. 不得通过重新命名阶段、降低完成标准、把中间工作称为完整阶段，或增加未经说明的“新阶段”来规避本计划。
8. 本节规则自身同样不可由 agent 擅自修改或绕过。

## 2. 强制回顾机制

执行架构演进相关任务时，agent 必须在以下时点完整回顾本文件和 `AGENTS.md`：

1. 每个任务开始时。
2. 建立修改前性能基线之后、开始修改之前。
3. 每个原子修改步骤完成之后。
4. 准备进入下一步骤或下一阶段之前。
5. 发现新问题、性能波动、测试失败或计划外依赖时。
6. 宣布某一步或某一阶段完成之前。
7. 向用户提交最终结果之前。

每次回顾至少确认：

- 当前工作属于哪个原始阶段和哪一项任务。
- 是否改变了原定范围、顺序、依赖方向或完成标准。
- 是否引入额外抽象、复制、分配、IO、动态分派或热路径调用。
- 是否已经满足 `AGENTS.md` 规定的性能基线、release 构建和完整测试要求。
- 是否把前置工作、缺陷修复或局部里程碑错误地表述为阶段完成。
- 是否触碰了本文件或 `AGENTS.md` 的不可修改约束。

## 3. 偏差处理规则

1. 如果实际工作与本计划不一致，立即停止扩大修改范围。
2. 在对话中明确报告：原计划、实际情况、偏差原因、影响范围和建议方案。
3. 未经用户明确许可，不得改变阶段顺序、完成标准或依赖目标。
4. 计划执行中发现的 Bug 可以作为当前步骤的阻塞问题报告和修复，但不能因此把未完成的阶段标记为完成。
5. 如果修复 Bug 会实质改变当前阶段的范围或顺序，必须先取得用户许可。
6. 新增的中间护栏或前置整理只能标记为子步骤或前置工作，不得重新编号为原始第一、第二或第三阶段。
7. 阶段完成必须逐条满足该阶段的完成标准，不能以“测试通过”“性能未回退”替代尚未完成的架构目标。

## 4. 总体原则

- 渐进推进，每次只处理一个明确领域。
- 先建立可重复的性能与正确性护栏，再调整文件和依赖结构。
- 文件拆分阶段只移动代码，不同时修改算法。
- 不引入 trait object、额外复制、额外分配或新的热路径 FFI。
- 性能回归以 ApexBase 自身基线判断，而不是以是否仍领先 SQLite/DuckDB 判断。
- 每个实现步骤都必须执行 `AGENTS.md` 要求的 release 构建、完整 pytest、完整 cargo test 和前后 benchmark。
- `AGENTS.md` 永远不得修改；本文件也不得在没有用户明确许可时修改。

### 4.1 原始架构判断

ApexBase 的宏观架构方向合理，性能基础非常强；当前主要矛盾不是缺少能力，而是模块边界、缓存一致性和自动化性能门禁已经跟不上功能增长。因此应采用渐进调整，不进行大重构。

总体判断是：当前架构能够继续承载功能开发，但必须先补齐自身性能基线门禁并统一缓存/delta 语义，然后再拆分大文件。第一项真正值得实施的架构改动不是移动代码，而是建立“当前提交相对 ApexBase 自身基线无回退”的自动验收链路。

### 4.2 必须保护的合理设计

以下方向已经具有明确职责或性能价值，不应在演进过程中轻易推翻：

- Rust 核心负责存储、查询、事务和索引。
- PyO3、Embedded、PG Wire、Flight 作为不同接入层。
- mmap、列式布局、Arrow FFI 和专用 fast path 具有明确的性能目的。
- `query/executor` 与 `storage/on_demand` 已经开始按功能拆分，后续应沿职责继续渐进拆分。
- server/flight 通过 Cargo feature 隔离的方向合理。
- 单 crate 配合 fat LTO 当前有利于跨模块内联和运行时优化。

以下做法不是本计划的目标：

- 为追求形式上的整洁而引入动态分派、trait object 或通用算子框架。
- 为统一接口而增加中间对象、数据复制、序列化、分配、锁或 FFI 往返。
- 在缺少性能护栏时进行跨层大规模重写。
- 仅根据文件行数决定重构，不分析职责、依赖和热路径。

### 4.3 原始问题与阶段追踪矩阵

| 原始问题 | 主要证据 | 对应阶段 | 完成判断 |
|---|---|---|---|
| 文档边界与实际代码不一致 | `docs/STORAGE_ARCHITECTURE.md` 要求存储操作经过 `StorageEngine`，但 PyO3、DML、DDL 仍存在直接打开或访问 `TableStorageBackend` 的路径 | 第三阶段 | 不能仅靠规定“都调用 Engine”解决；必须先消除双向依赖并建立统一 façade |
| 查询层与存储层双向耦合 | `StorageEngine → ApexExecutor`，同时 `ApexExecutor / bindings → StorageEngine + TableStorageBackend` | 第三阶段 | 存储层不再反向调用查询层，上层 façade 统一编排后才算完成 |
| 缓存层次过多、失效协议分散 | StorageEngine read/schema/insert 缓存、Executor storage/index/FTS/SQL parse 缓存、PyO3 `cached_backends`、Python `_simple_sql_cache` | 第一、第三阶段 | 第一阶段建立契约；第三阶段由统一 table epoch 取代分散通知 |
| 读取语义不统一 | `StorageEngine::get_read_backend()` 遇到 delta 可能 compact，而 Executor 读取路径要求读取不触发 compact | 第一、第三阶段 | 先以契约固定现状，再在统一边界选择“合并读取”或“触发 compact” |
| 核心文件过大 | 首次评估时 `mmap_scan.rs`、`bindings.rs`、`aggregation.rs`、`dml.rs`、`sql_parser.rs`、`select.rs` 均超过约 6,000 行 | 第二阶段 | 按领域逐个拆分且不改变算法，每个领域独立完成性能和正确性验收 |
| warning 和不可达/未使用代码较多 | 首次 release 构建约产生 150 个 warning，包括 unreachable pattern、dead code 和 unused 项 | 第二阶段及日常治理 | 不能通过大范围清理混入文件拆分；只在明确归属的原子步骤中处理 |
| 缺少持续性能回归门禁 | 原有主要 workflow 偏向 tag/release；公开 benchmark 主要回答是否领先 SQLite/DuckDB | 第一阶段 | 固定本机必须以相同环境比较 ApexBase base/current 自身基线；GitHub CI 不承担性能验收 |
| pytest 时间门槛缺少可重复性 | 首次评估完整 pytest 为 11.11 秒，超过 9 秒；后续单次结果虽进入 9 秒内，但仍接近边界 | 第一阶段 | 完成重复测量、归因和稳定自动门禁后才算完成 |
| Python 构建迭代成本较重 | Python 构建默认同时启用 server 和 flight，release 构建耗时较长 | 第四阶段 | 前三阶段稳定后才评估拆 crate，不得提前以运行时性能换编译速度 |

### 4.4 首次评估历史快照

> 以下数据仅用于记录路线形成时的证据，是历史快照，不代表当前状态，也不能替代任何修改前后的新基线。测试数量、耗时、warning 和 benchmark 数值都会随代码与环境变化。

- release 构建成功，耗时 4 分 32 秒，并产生约 150 个 warning。
- `pytest`：1444/1444 通过，耗时 11.11 秒，超过项目规定的 9 秒上限。
- `cargo test`：391 个单元测试和 6 个文档测试全部通过。
- release 后 benchmark：
  - 表格型指标 72/72 领先。
  - 向量指标 6/6 领先。
  - 代表性结果：全行点查 1.75 微秒、批量导入 224.49 毫秒、TopK L2 7.67 毫秒。
- 首次评估未修改代码，当时已有的 4 个暂存文件保持不变。

该快照说明 ApexBase 已具有很强的正确性和性能基础，同时也说明：竞争对手比较结果不能证明自身无回退；pytest 单次耗时不能构成稳定门禁；编译时间和 warning 数量需要在不干扰热路径改造的前提下持续治理。

## 5. 第一阶段：建立护栏，不调整热路径

### 5.1 工作内容

1. benchmark 使用 `--output` 保存 JSON。
2. benchmark JSON 记录：
   - Git commit。
   - 机器与操作系统。
   - CPU 和核心数。
   - Python 版本。
   - ApexBase、SQLite、DuckDB、PyArrow、NumPy、Pandas、Polars、Maturin、Rust 等相关依赖或构建版本。
3. 日常开发在固定本机执行稳定 canary：base/current 使用同一台物理机器、同一依赖锁定和 release 构建，先按 B-C-C-B-B-C 交错采集每侧三个样本并比较中位数；若初判回退，再追加每侧两个样本并以五样本中位数复核。
4. 第一阶段最终验收在同一固定本机执行完整 78 项 base/current benchmark，保存全部 JSON 和比较结果；不在 GitHub 托管 runner 或回连本机的 workflow 中重复执行性能比较。
5. 比较 ApexBase 当前版本与 ApexBase 自身历史基线；“仍然领先 DuckDB/SQLite”不能作为无性能回退的依据。
6. 为 Python/Rust `QuerySignature` 建立共享 SQL 语料的一致性测试。
7. 为 cache invalidation 建立架构契约测试。
8. 为 delta-read、compact 和 reopen 行为建立架构契约测试。
9. 调查 pytest 为什么可能稳定超过或逼近 9 秒。
10. 在确认耗时来源和波动范围后，把 9 秒门槛变成可重复的自动检查。
11. 本地性能验收必须在接通电源、固定电源模式、热状态稳定且无明显后台负载的条件下进行；构建后至少等待 30 秒，报告目录保留 commit、系统、依赖、构建和全部样本信息。

### 5.2 完成标准

只有同时满足以下条件，第一阶段才能标记为完成：

- JSON 包含完整、可追溯的 commit、系统、依赖和构建元数据。
- 本地 canary 已在同一固定物理机器上成功比较 base/current；初始三样本若失败，必须完成五样本复核，不能选择性丢弃样本。
- 完整 78 项 base/current 已在同一固定本机以默认规模成功运行，并保留全部 JSON 和比较报告。
- 自身基线比较工具具有测试，缺失指标、配置不兼容和超阈值回退会失败。
- Python/Rust 分类器使用同一份可维护 SQL 语料进行一致性验证。
- cache invalidation 和 delta-read 均有独立契约测试。
- pytest 耗时已经完成重复测量与原因分析。
- 9 秒门槛具有可重复的自动检查方案，不依靠单次偶然结果。
- `maturin develop --release`、完整 `pytest`、完整 `cargo test` 和前后 benchmark 全部通过。

## 6. 第二阶段：零运行时成本的文件拆分

### 6.1 工作内容

严格按一次一个领域的方式拆分：

1. `aggregation.rs` 按 `scalar`、`grouped`、`distinct`、`having` 拆分。
2. `dml.rs` 按 `insert`、`update`、`delete`、`copy` 拆分。
3. `mmap_scan.rs` 按 `predicate`、`projection`、`groupby`、`vector`、`topk` 拆分。
4. `bindings.rs` 只保留 PyO3 wrapper，把内部实现提取到 `read`、`write`、`sql`、`blob`、`arrow` 模块。

### 6.2 强制约束

- 每次只拆一个文件中的一个领域。
- 拆分时不得同时修改算法或公开行为。
- 不引入 trait object 或新的动态分派。
- 不增加数据复制、内存分配、序列化、锁、IO 或 FFI 往返。
- 不借拆分之机进行无关格式化或大范围重命名。
- 每个拆分步骤都必须单独比较修改前后 benchmark。
- 每个步骤都必须通过 release 构建、完整 pytest 和完整 cargo test。

### 6.3 完成标准

只有四个目标文件都按计划完成职责拆分，并且每个领域都有独立的性能和正确性验收记录，第二阶段才能标记为完成。只抽取 Python 辅助函数、增加测试或修复 Bug，不等同于完成本阶段。

## 7. 第三阶段：修正真实依赖方向

### 7.1 目标依赖结构

```text
Python / Embedded / Server / Flight
              ↓
       Database / Session façade
         ↙              ↘
 Query Runtime       Storage Service
                          ↓
              TableStorageBackend
                          ↓
                OnDemandStorage
```

### 7.2 工作内容

1. 引入统一的 `Database` / `Session` façade，由上层同时编排查询和存储。
2. `StorageEngine` 只负责存储协调，不再反向导入或调用 `ApexExecutor`。
3. `OnDemandStorage` 等底层存储模块不得反向调用查询层进行缓存失效。
4. 引入统一的 table generation/epoch：
   - 每次逻辑写入只递增一次。
   - 各缓存记录其观察到的 epoch。
   - 缓存根据 epoch 判断是否失效，而不是由多个入口手动广播失效。
5. 明确定义 delta 读取的一致语义：
   - 要么所有入口执行合并读取。
   - 要么在统一边界触发 compact。
   - 不允许由 Python、Embedded、Server、Flight 或具体查询入口各自决定。
6. Python、Embedded、Server、Flight 应通过统一 façade 获得一致行为。

### 7.3 完成标准

- 存储层不存在对 `ApexExecutor` 或其他查询执行层的反向依赖。
- 上层入口通过统一 façade 编排查询与存储。
- table epoch 成为缓存一致性的统一依据，且一次逻辑写入只递增一次。
- delta 行为由统一策略决定，所有入口具有契约测试。
- 核心热路径没有新增动态分派、额外复制、额外分配或额外 FFI。
- 完整测试和前后 benchmark 证明没有性能回退。

## 8. 第四阶段：边界稳定后才考虑拆 crate

当前单 crate 配合 fat LTO 有利于运行时优化，因此本阶段不是当前最高优先级。

只有前三阶段完成、模块边界和依赖方向稳定后，才评估拆分为：

- `apex-core`
- `apex-python`
- `apex-server`
- `apex-flight`

评估重点：

- Python 构建默认同时启用 server 和 flight 所造成的迭代编译成本。
- crate 边界是否会妨碍 LTO、内联和核心路径优化。
- feature 组合、发布流程和 Python wheel 构建复杂度。
- 编译收益是否足以覆盖维护成本和潜在运行时风险。

未经前三阶段验收和用户明确许可，不得提前推进 crate 拆分。

## 9. 当前状态快照（2026-07-22）

> 本节已根据用户于 2026-07-22 的明确许可同步本地同机验收路线和当前工作进度。后续如需再次更新，仍须重新取得用户明确许可。

### 第一阶段：本地同机护栏已落地，完整 78 项本地验收待完成

已完成或已有验证记录：

- PR #3 `Add phase-one architecture guards` 已于 2026-07-19 squash 合并，合并提交为 `1ae6de0`。
- benchmark JSON 支持 `--output` 和微秒级精度，并记录 Git commit、branch、dirty 状态、系统、依赖和构建版本；完整元数据结构已有测试覆盖。
- ApexBase 自身 base/current 性能比较工具已有独立测试；缺失指标、配置不兼容、样本指标不一致和超阈值回退都会失败。
- 本地同机 runner 已实现：从临时 worktree 分别构建 base/current release wheel，锁定同一依赖，在同一临时 Python 环境交错采样，并保存可追溯报告。
- GitHub PR canary 和 nightly 性能 workflow 已移除；GitHub 托管 runner 的波动不能作为验收依据，而由 workflow 回连同一台开发机只会重复本地门禁。
- 2026-07-22 首次本地同机 canary 的三样本初判在 15 项中有 2 项超阈值，但 current 第三个样本已恢复到 base 水平，暴露出三样本对短时冷态/负载波动不够稳健；门禁据此增加“初判失败时追加至每侧五样本”的自动复核，禁止人工挑选样本。
- 修正后使用 `origin/main` 作为 base 在同一台机器复测，15/15 项通过；三样本中位数直接通过，无需追加复核，报告保存在 `local-perf-results/20260722-192651/`。其中相对变化最大的字符串等值过滤为 +14.23%，未超过 15% 相对阈值；其 0.022613 ms 绝对变化虽超过 0.005 ms，但门禁要求相对和绝对阈值同时超限才判定回退。
- 共享 SQL 路由语料已固化为独立、可维护的 `test/fixtures/query_signature_routes.jsonc`，19 类 SQL 同时经过 Python/Rust 分类器的一致性测试。
- 独立 cache invalidation 架构契约测试覆盖跨客户端直接写、事务提交和 schema rewrite；测试发现并修复了事务提交后字典型 GROUP BY 读取陈旧数据的问题。
- 独立 delta、compact、reopen 行为契约测试已建立。
- pytest 已完成重复串行测量和波动分析；release 重装后的前两次冷态单次结果为 10.81 和 9.25 秒，随后五次稳定样本为 8.66、8.61、8.73、8.65、8.58 秒，中位数 8.65 秒，波动范围 0.15 秒，说明单次结果不足以判断稳定回退。
- 可重复 9 秒门禁已实现并具有测试：五次串行完整 pytest 的中位数不得超过 9 秒、最多允许一个样本超过 9 秒、最大波动不得超过 1 秒；该门禁由固定本机的验收流程执行。
- 合并前完整验收已通过：`maturin develop --release` 成功；完整 pytest 1469/1469 通过；完整 cargo test 的 391 个单元测试和 6 个文档测试通过；修改后 72 项表格 benchmark 和 6 项向量 benchmark 均无实质性能回退。

尚未完成：

- 同一固定本机尚未完成一次默认规模的完整 78 项 base/current 比较和本地可重复 pytest 门禁，因此第一阶段仍不能标记为完成。

### 第二阶段：未开始正式拆分

- `aggregation.rs` 尚未按四个领域拆分。
- `dml.rs` 尚未按四个领域拆分。
- `mmap_scan.rs` 尚未按五个领域拆分。
- `bindings.rs` 尚未按 wrapper/read/write/sql/blob/arrow 拆分。
- Python SQL 分类函数抽取属于前置整理，不计为第二阶段完成项。

### 第三阶段：未开始实质依赖改造

- `StorageEngine` 仍反向调用 `ApexExecutor`。
- 底层存储写路径仍直接触发查询层缓存失效。
- 尚无统一 `Database` / `Session` façade。
- 尚无覆盖所有缓存的统一 table epoch。
- delta 读取/compact 策略尚未在统一边界定义。

### 第四阶段：按计划延期

- 当前不拆 crate。

## 10. 下一步唯一允许的默认方向

在用户没有另行明确授权调整路线时，下一步必须继续补齐第一阶段，顺序如下：

1. [x] 完善 benchmark JSON 的 commit、依赖和构建元数据，并增加测试。
2. [x] 将共享 SQL 语料固化为独立的可维护数据源。
3. [x] 增加独立 cache invalidation 架构契约测试。
4. [x] 重复测量并归因 pytest 耗时和波动。
5. [x] 设计可重复的 9 秒自动门禁。
6. [x] 实现无需 GitHub 的本地同机 base/current runner，并固定 release 构建、依赖锁定、交错采样、系统一致性检查和报告留存规则。
7. [x] 修正后的本地 canary 已在固定本机成功比较 15/15 项，并保留三样本报告；若后续初判失败，自动五样本复核仍为强制路径。
8. [ ] 在同一固定本机运行默认规模的完整 78 项 base/current benchmark，并保留全部 JSON 和最终比较报告。
9. [ ] 在同一固定本机运行五次串行完整 pytest 的可重复 9 秒门禁，并保留结果。
10. [ ] 完整执行 `AGENTS.md` 的其余验收流程，向用户逐项报告第一阶段最终结果；只有满足全部完成标准后，才能请求进入第二阶段。

如果任何一步需要改变以上顺序或范围，必须先报告并取得用户明确许可。
