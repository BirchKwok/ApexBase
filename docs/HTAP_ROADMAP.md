# ApexBase HTAP Roadmap — Honest Status Report

## Current State (Feb 2026)

ApexBase is an **OLAP-oriented embedded columnar database** with a delta-based incremental
write mechanism and newly wired secondary index support. Moving towards true HTAP.

### What Works (Production)

| Component | Location | Status |
|-----------|----------|--------|
| **V4 Row Group format** (`.apex`) | `on_demand.rs` | ✅ Production (default) |
| Delta writes (`.apex.delta`) | `on_demand.rs` | ✅ Production |
| **In-memory compaction** (load + merge + save V4) | `on_demand.rs` | ✅ Production |
| **Optimized save_v4** (pre-filter, no disk reload) | `on_demand.rs` | ✅ Production |
| **Dict encoding** (low-cardinality strings, disk-only) | `on_demand.rs` | ✅ Production |
| V3 backward compat reading | `on_demand.rs` | ✅ Production |
| mmap zero-copy reads (V3) / in-memory reads (V4) | `on_demand.rs` | ✅ Production |
| On-demand column projection | `on_demand.rs` | ✅ Production |
| Smart write routing (delta/full) | `engine.rs` | ✅ Production |
| LRU cache management | `engine.rs` | ✅ Production |
| JIT predicate compilation | `query/jit.rs` | ✅ Production |
| Bloom filter string filtering | `query/bloom.rs` | ✅ Production |
| SQL parser + executor | `query/` | ✅ Production |
| WAL + durability levels | `on_demand.rs` | ✅ Production |
| **Secondary indexes (B-Tree, Hash)** | `storage/index/` + `executor.rs` | ✅ Production |
| **CREATE/DROP INDEX SQL** | `sql_parser.rs` + `executor.rs` | ✅ Production |
| **Index-accelerated SELECT** (equality, IN) | `executor.rs` | ✅ Production |
| **Index maintenance on SQL DML** (INSERT/DELETE/UPDATE) | `executor.rs` | ✅ Production |

### What's Standalone / Not Wired (Dead Code)

These modules compile and have unit tests, but are **NOT connected** to any real data path.

| Module | Location | Unit Tests | Wired Into Data Path? |
|--------|----------|:----------:|:---------------------:|
| DeltaStore (new) | `storage/delta/` | 9 | ❌ |
| MVCC (VersionStore, Snapshot, GC) | `storage/mvcc/` | 12 | ⚠️ TxnManager wired |
| Transaction Manager (OCC) | `txn/` | 14 | ✅ Wired (Phase 4) |
| Query Planner (formal) | `query/planner.rs` | 2 | ⚠️ Partially (inline in executor) |
| Horizontal Scaling | `scaling/` | 18 | ❌ |

### Memory Efficiency

| Operation | Approach |
|-----------|----------|
| **save_v4()** (full file rewrite) | Pre-filter deleted rows into local vars, write V4, set in-memory state directly (no disk reload) |
| **Compaction** (delta → base merge) | Load all columns, merge delta, save as V4 |
| **Delta insert** | Metadata only (append to .delta file) |
| **Read** (SELECT) | V4: in-memory columns / V3: on-demand mmap |

### Known Limitations

| Issue | Impact | Fix Phase |
|-------|--------|-----------|
| ~~`save_v4()` does full file rewrite every time~~ | ~~O(n) write amplification~~ | ✅ Phase 2 |
| ~~DELETE soft-deletes in memory, but save physically removes rows~~ | ~~Must rewrite entire file~~ | ✅ Phase 3 |
| ~~Per-RG deletion vector format exists but always zeros~~ | ~~Deletion vectors never used~~ | ✅ Phase 3 |
| Index not updated via Python `store()` API (only SQL DML) | Stale index after non-SQL writes | Future |

## Actual Data Path

```
Python Client
    │
    ▼
bindings.rs (PyO3)
    │
    ▼
StorageEngine (engine.rs)         ← smart write routing, caching
    │
    ├── write() → insert_rows_to_delta()  ← append to .apex.delta (fast, metadata-only)
    ├── read()  → compact() + open()      ← in-memory merge + V4 read
    │
    ▼
OnDemandStorage (on_demand.rs)    ← V4 Row Group format, the REAL storage layer
    │
    ├── .apex file        (V4 Row Group columnar base)
    ├── .apex.delta file  (row-oriented appends)
    ├── .apex.wal file    (WAL for durability)
    └── .apex.idx/        (secondary indexes, managed by IndexManager)

ApexExecutor (executor.rs)        ← SQL execution
    │
    ├── CREATE/DROP INDEX → IndexManager (global cache)
    ├── SELECT WHERE col=X → try_index_accelerated_read() → IndexManager lookup
    ├── INSERT/DELETE/UPDATE → notify_index_insert/delete() → IndexManager maintenance
    ├── BEGIN/COMMIT/ROLLBACK → TxnManager (global singleton)
    └── DML in txn → buffer_insert/delete/update in TxnContext → apply on COMMIT
```

## Roadmap

### ✅ Phase 0: Memory-Efficient Compaction (DONE)
- Streaming compaction (V3): column-by-column via mmap — **removed, replaced by V4 path**

### ✅ Phase 1: Row Group File Format (V4) — COMPLETE
V4 is now the **default save format**. All read/write paths support V4.
V3 dead code removed (`compact_column_streaming`, `FORMAT_VERSION_V3`, `MAGIC_FOOTER_V3`).

**Key optimizations:**
- `save_v4()`: pre-filters data under read guards, drops guards early, writes V4, sets in-memory state directly (no `open_v4_data()` disk reload)
- `open_v4_data()`: decodes StringDict → String for in-memory correctness
- Dict encoding applied per-RG during save (disk-only optimization)
- Legacy V3 files auto-detected on read; first save converts to V4

**Critical bugs fixed:**
- StringDict silent data loss in `push_string()`/`extend_strings()` — caused 161GB runaway temp files
- `next_id` off-by-one for empty tables
- `open_for_schema_change` V3-only offsets for V4 files
- `add_column_with_padding` not loading V4 data first
- `compact()` using V3-only streaming path for V4 files

### ✅ Phase 1.5: Wire IndexManager + SQL Index Support — COMPLETE
IndexManager wired into executor with full SQL support and DML maintenance.

**What was done:**
- SQL parser: `CREATE [UNIQUE] INDEX [IF NOT EXISTS] name ON table (col) [USING HASH|BTREE]`
- SQL parser: `DROP INDEX [IF EXISTS] name ON table`
- Global `INDEX_CACHE` (per-table IndexManager, `parking_lot::Mutex`, LRU-aware)
- `execute_create_index()`: creates index, builds from existing data, persists to disk
- `execute_drop_index()`: drops index, invalidates cache
- `try_index_accelerated_read()`: intercepts `WHERE col = X` and `WHERE col IN (...)` on indexed columns
- Falls back to full scan if index doesn't exist or selectivity > 50%
- Supports ORDER BY, LIMIT, OFFSET, column projection on index results
- Index maintenance: `notify_index_insert()` / `notify_index_delete()` on SQL INSERT/DELETE/UPDATE
- Cache invalidation wired into `invalidate_storage_cache_dir()`

**Bugs fixed:**
- Pre-existing UPDATE bug: `row_idx` (enumeration index) used instead of `batch_row_idx` (actual row in batch), losing non-updated column values

**Tests:** 23 new Python tests (DDL, accelerated SELECT, DML maintenance, correctness vs scan)

### ✅ Phase 2: Append-Only Writes — COMPLETE
`engine.write_typed()` uses `append_row_group()` to append new Row Groups to existing V4 files
without rewriting. `save()` defaults to `save_v4()` for correctness in mixed-operation cases.

**What was done:**
- `write_row_group_to_disk()`: disk-only RG append (no in-memory state mutation)
- `append_row_group()`: disk write + in-memory update (for engine's direct-append path)
- `persisted_row_count` atomic: tracks actual on-disk row count independently of in-memory header
- `engine.write_typed()` fast path: schema-matching inserts append directly as new RG (O(new_data))

### ✅ Phase 3: Per-Row Group Deletion Vectors — COMPLETE
DELETE operations now update deletion vector bitmaps in-place on disk — no full file rewrite.
Compaction (full rewrite) triggers automatically when deletion ratio exceeds 50%.

**What was done:**
- `save_deletion_vectors()`: O(num_RGs) random writes to update deletion bitmaps + footer
- `open_v4_data()`: reads deletion vectors from each RG, populates in-memory `deleted` bitmap
- `save()` smart routing: delete-only on V4 → `save_deletion_vectors()`; else → `save_v4()`
- `header.row_count` updated to active count after deletion vector writes
- Deletion vectors persist across file close/reopen correctly

**Bugs found and fixed:**
- `replace()` mutated `header.row_count` before `save()`, causing wrong save-path selection
- `append_row_group()` double-added rows to memory when called from `save()` (refactored into disk-only helper)

### ✅ Phase 4: MVCC + TxnManager — COMPLETE
Transaction Manager wired into executor and Python bindings with full SQL support.

**What was done:**
- SQL parser: `BEGIN [TRANSACTION] [READ ONLY]`, `COMMIT`, `ROLLBACK` tokens + statement variants
- Global `TXN_MANAGER` singleton (`once_cell::sync::Lazy`) with `txn_manager()` accessor
- Executor dispatch: `execute_begin`, `execute_commit_txn`, `execute_rollback_txn`
- `execute_in_txn()`: buffers INSERT/DELETE/UPDATE in `TxnContext` write set
- `apply_txn_write()`: applies buffered writes on COMMIT via existing DML executors
- Python bindings: `current_txn_id: RwLock<Option<u64>>` per session, transaction-aware routing
- Python client: `_in_txn` state tracking, routes DML through session-aware bindings when in txn

**Bugs found and fixed (pre-existing):**
- `execute_update()` missing delete step in delete+insert pattern (UPDATE created duplicates)
- `coerce_numeric_for_comparison()` missing UInt64↔Int64 coercion (broke `_id` comparisons)
- `try_scalar_comparison()` missing UInt64Array handler (no fast-path for `_id` filters)
- `notify_index_insert()` used wrong `pre_insert_count` after UPDATE deletes (stale index)

**Tests:** 15 new Python tests (BEGIN/COMMIT/ROLLBACK, INSERT/DELETE/UPDATE in txn, multi-DML, READ ONLY, syntax variants)

### Phase 4.5: Wire Remaining HTAP Modules
1. ~~IndexManager~~ → ✅ DONE (Phase 1.5)
2. ~~QueryPlanner (inline)~~ → ⚠️ Partially done (index lookups inline in executor)
3. ~~MVCC + TxnManager~~ → ✅ DONE (Phase 4)
4. DeltaStore → per-RG update tracking
5. VersionStore snapshot reads → MVCC isolation levels

### Phase 5: Horizontal Scaling
Long-term. Requires Row Groups + Transactions to be stable.

## Test Status

```
Rust:   143+ passed (cargo check OK, PyO3 link requires conda env)
Pytest: 719/719 passed (681 existing + 23 index + 15 transaction tests)
Phase 4: zero regressions, all transaction lifecycle tests pass
```
