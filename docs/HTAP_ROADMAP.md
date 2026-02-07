# ApexBase HTAP Roadmap — Honest Status Report

## Current State (Feb 2026)

ApexBase is an **OLAP-oriented embedded columnar database** with a delta-based incremental
write mechanism. It is **NOT yet a true HTAP database**.

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

### What's Standalone / Not Wired (Dead Code)

These modules compile and have unit tests, but are **NOT connected** to any real data path.
`StorageEngine`, `ApexExecutor`, and `bindings.rs` do not reference them.

| Module | Location | Unit Tests | Wired Into Data Path? |
|--------|----------|:----------:|:---------------------:|
| IndexManager (B-Tree, Hash) | `storage/index/` | 15 | ❌ |
| DeltaStore (new) | `storage/delta/` | 9 | ❌ |
| MVCC (VersionStore, Snapshot, GC) | `storage/mvcc/` | 12 | ❌ |
| Transaction Manager (OCC) | `txn/` | 14 | ❌ |
| Query Planner | `query/planner.rs` | 2 | ❌ |
| Horizontal Scaling | `scaling/` | 18 | ❌ |

### Memory Efficiency

| Operation | Approach |
|-----------|----------|
| **save_v4()** (full file rewrite) | Pre-filter deleted rows into local vars, write V4, set in-memory state directly (no disk reload) |
| **Compaction** (delta → base merge) | Load all columns, merge delta, save as V4 |
| **Delta insert** | Metadata only (append to .delta file) |
| **Read** (SELECT) | V4: in-memory columns / V3: on-demand mmap |

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
    └── .apex.wal file    (WAL for durability)
```

**None of the HTAP modules participate in this path.**

## Roadmap

### ✅ Phase 0: Memory-Efficient Compaction (DONE)
- Streaming compaction (V3): column-by-column via mmap — **removed, replaced by V4 path**
- Tests: 143 Rust + 600 Pytest all passing

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

### Phase 2: Append-Only Writes
- New inserts buffer in memory → flush as new Row Group at file end
- No need to rewrite existing data
- `open_for_write()` no longer needs to load all columns

### Phase 3: Per-Row Group Deletion Vectors
- Each Row Group has a bitmap of deleted rows
- DELETE marks a bit — no file rewrite
- Compaction reclaims space when deletion ratio exceeds threshold

### Phase 4: Wire HTAP Modules
Prerequisites: Row Group format must be stable first.
1. IndexManager → auto-build `_id` hash index per Row Group
2. QueryPlanner → route point lookups to index, scans to columnar
3. MVCC + TxnManager → snapshot reads, OCC transactions
4. DeltaStore → per-RG update tracking

### Phase 5: Horizontal Scaling
Long-term. Requires Row Groups + Transactions to be stable.

## Test Status

```
Rust:   143/143 passed  (4 V4 Row Group tests)
Pytest: 663/663 passed
```
