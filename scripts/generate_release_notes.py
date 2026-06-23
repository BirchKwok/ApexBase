#!/usr/bin/env python3
"""
Generate release-notes.md from git tags by analyzing actual code diffs.
Output: docs/release-notes.md

Examines changed files, diff stats, and commit messages for each tag range
to produce meaningful, specific summaries grouped by functional area.

Usage:
    python scripts/generate_release_notes.py
"""

import subprocess
import re
import sys
from pathlib import Path
from collections import OrderedDict

REPO_URL = "https://github.com/BirchKwok/ApexBase"

MODULE_RULES = [
    (r"apexbase/src/storage/",             "Storage Engine"),
    (r"apexbase/src/query/",               "Query Engine"),
    (r"apexbase/src/fts/",                 "Full-Text Search"),
    (r"apexbase/src/flight/",              "Arrow Flight gRPC"),
    (r"apexbase/src/server/",              "PostgreSQL Wire Protocol"),
    (r"apexbase/src/scaling/",             "Scaling & Sharding"),
    (r"apexbase/src/embedded/",            "Rust Embedded API"),
    (r"apexbase/src/python/",              "Python Bindings"),
    (r"apexbase/src/data/",                "Data Layer"),
    (r"apexbase/src/txn/",                 "Transaction Manager"),
    (r"apexbase/src/table/",               "Table Catalog"),
    (r"apexbase/src/lib\.rs",              "Library Core"),
    (r"apexbase/python/apexbase/client",   "Python Client"),
    (r"apexbase/python/",                  "Python Package"),
    (r"docs/",                             "Documentation"),
    (r"mkdocs\.yml",                       "Documentation"),
    (r"hooks/",                            "Documentation"),
    (r"test/",                             "Tests"),
    (r"benchmarks/",                       "Benchmarks"),
    (r"\.github/workflows/",               "CI/CD"),
    (r"Cargo\.toml",                       "Project Config"),
    (r"pyproject\.toml",                   "Project Config"),
    (r"build\.rs",                         "Build System"),
    (r"README\.md",                        "Documentation"),
]

# Hand-curated highlights based on reviewing actual diffs for each tag range
KNOWN_HIGHLIGHTS: dict[str, list[str]] = {
    "v1.19.1": [
        "Fix macOS arm64 SIGSEGV crash on ApexStorage init via lazy imports (pyarrow, pandas, polars no longer loaded at import time)",
        "Fix weakref deadlock by switching `threading.Lock` to `threading.RLock` in `_InstanceRegistry`",
        "Fix CI pytest hangs in cross-process memtable tests",
        "Fix numeric range mmap scan offset handling in Python bindings",
        "Remove global warm-query GROUP BY / ORDER BY top-k static cache infrastructure from query executor",
        "Add pandas 3.x compatibility fixes in `to_pandas()`",
        "Rename `invalidate_query_caches` to `invalidate_read_caches` across storage backend",
        "Add `test_import_stability.py` regression tests and pytest `--timeout=120` config",
    ],
    "v1.19.0": [
        "Set up comprehensive MkDocs Material documentation site with GitHub Actions deployment",
        "Add FLOAT16 vector column type with f16→f32 byte decoding and ndarray/list conversion in Python bindings",
        "FTS index now backfills rows written through the Python `store()` API",
        "Async FTS backfill for tables with >100K rows (background thread, non-blocking)",
        "FTS backfill reads string columns directly from mmap for zero-copy performance",
        "Add new documentation pages: concepts, installation, performance, user guides",
        "Add HTAP roadmap document",
        "Replace large inline README section with cross-reference to new docs site",
    ],
    "v1.18.0": [
        "Add LIMIT and OFFSET support in SQL queries (parser, executor, Python client)",
        "Improve index building for streaming batches with batch-size limit",
        "Add mmap_scan numeric range and string equality filter support with LIMIT/row offset",
        "Add Python client fast-path cache for projected string equality with LIMIT/OFFSET",
        "Add `retrieve_projected_by_string_eq_limit` for direct mmap-backed projected scans in Python bindings",
        "Enhance embedded API `register_temp_table` with LIMIT/OFFSET support and temp directory cleanup",
    ],
    "v1.17.0": [
        "Add temporary table support: `register_temp_table` and `drop_temp_table` for CSV, JSON, and Parquet files",
        "Temp tables materialize into native .apex format, auto-cleaned on DB drop",
        "Add window function execution module (`window.rs`)",
        "Enhance DuckDB result materialization compatibility (`to_arrow_table` instead of `to_arrow`)",
        "Improve aggregation, join, and SELECT execution paths with temp table awareness",
        "Add comprehensive temp table tests (CSV, JSON, SQL query access, cleanup, persistence)",
    ],
    "v1.16.0": [
        "Add COPY TO export support: tables exportable to CSV, TSV, JSON, NDJSON with configurable options",
        "Add JSON mutation functions: JSON_SET, JSON_INSERT, JSON_REPLACE, JSON_REMOVE",
        "Add view-aware SQL routing in Python client for persisted views",
        "Add comprehensive test suite for SQL view, COPY, and JSON operations",
        "Optimize expression evaluator with string-based LIKE, GLOB, and REGEXP improvements",
    ],
    "v1.15.0": [
        "WAL-backed transaction durability: commits write TxnBegin/TxnCommit to per-table WAL with optional fsync",
        "Adaptive row group size for on-demand storage with narrow/wide table tests",
        "Add query signature fast paths: projected point lookup, ID batch, full scan, string/numeric range filter",
        "Zone map-based string filtering to skip irrelevant row groups during equality scans",
        "Single-pass filtered string aggregation on mmap tables",
        "Delta string index caching and row count caching for repeated read performance",
        "Separate write_file and delta_file handles with atomic sync-pending bitmask tracking",
        "Add `build.rs` for macOS Python library rpath linking",
        "Remove old test infrastructure (`run_tests.py`, `test/README`)",
    ],
    "v1.14.0": [
        "Add parallel multi-predicate mmap scan with zone map pruning and rayon-based predicate evaluation",
        "Add parallel column extraction via rayon for 2+ columns and 500+ rows",
        "Add column projection support to skip unrequested columns during Arrow batch construction",
        "Add Binary column type support (ColBuf::Bin, ColBuf::FixedVec, BinaryArray output)",
        "Fix col=value filter pushdown logic with type-specific handling",
        "Add 3 new benchmarks (numeric IN, OR cross-column, numeric OR)",
    ],
    "v1.13.0": [
        "Fix `ApexClient` with `drop_if_exists=True` to always create fresh storage instead of reusing shared storage",
    ],
    "v1.12.0": [
        "Add backtick-quoted identifier parsing (`identifier`, Hive/MySQL style) to SQL parser",
        "Add unit tests for backtick parsing: SELECT, WHERE, ORDER BY, and unterminated identifier error",
        "Update docs to document quoted identifier syntax",
    ],
    "v1.11.0": [
        "Add x86_64 AVX2+FMA SIMD kernels for L2, L1, Linf, inner_product, cosine distance functions",
        "Improve Windows mmap prefault with 3-tier strategy (single-threaded, rayon-parallel, PrefetchVirtualMemory)",
        "Add Windows atomic rename retries with backoff, engine cache invalidation before writes",
        "Increase Windows WAL buffer size from 64KB to 512KB",
        "Remove `benchmarks/stress_test.py`",
    ],
    "v1.10.0": [
        "Drop V3 file format compatibility — all operations require V4 footer",
        "Replace heap-based GROUP BY with HashMap accumulator supporting SUM/COUNT/AVG",
        "Add `read_fixed_scattered_optimized` with row-group batching for FixedList/Float16List scatter reads",
        "Add `bench_filter_group_order.py` and `bench_group_by_detail.py` benchmark scripts",
        "Rename internal V3-specific constants (MAGIC_V3→MAGIC, HEADER_SIZE_V3→HEADER_SIZE, etc.)",
    ],
    "v1.9.0": [
        "Add Rust Embedded API module (`Database`/`Table` structs) for pure-Rust usage without Python",
        "Add query signature/caching system for deduplicating identical queries",
        "Add concurrent query scheduler for parallel batch execution",
        "Add vectorized hash join implementation",
        "Add concurrent storage access with shared table handles and multi-client support",
        "Add FLOAT16_VECTOR column type with f16-quantized storage and all distance metrics",
        "Add `batch_topk_distance` Python API for batched multi-query vector search",
        "Add `execute_batch` Python API for parallel SQL execution via scheduler",
        "Add schema evolution: `add_column`/`drop_column` on existing tables via Rust Embedded API",
        "Add new docs: RUST_EMBEDDED_API.md, FLOAT16_VECTOR_GUIDE.md, ENGINEERING_GUIDELINES.md",
    ],
    "v1.8.0": [
        "Add `vector_ops.rs` module implementing 6 vector distance functions (L2, L1, Linf, cosine, inner_product, array_distance)",
        "Add TopK vector search via `ORDER BY array_distance(col, [query]) LIMIT k`",
        "Add SQL INSERT with vector array literals and string literal auto-coercion",
        "Add Python `topk_distance()` API with 6 distance metrics, custom column names, and numpy support",
        "Add `SQL explode_rename(topk_distance(...))` for SQL-based top-k with JOIN",
        "Add `set_compression`/`get_compression` API (LZ4, Zstd)",
        "Add `batch_replace` API for bulk row replacement by ID",
        "Add `optimize()` method to compact storage and `get_column_dtype` for schema introspection",
        "Add new benchmarks: bench_vector.py, bench_vs_polars.py, bench_no_cache.py, bench_simsimd.py",
        "Add `test_vector_ops.py` with 1000+ lines of distance tests cross-validated against numpy",
    ],
    "v1.7.0": [
        "Add full-text search (FTS) module with `CREATE FTS INDEX` SQL syntax and `search_text()` Python API",
        "Rewrite window function engine: ROW_NUMBER, RANK, DENSE_RANK, LAG, LEAD, FIRST_VALUE, SUM/AVG windows",
        "Fix NULL semantics: LAG/LEAD return NULL at partition boundaries, COUNT(col) excludes NULLs",
        "Add set operations: UNION (dedup), UNION ALL, INTERSECT, EXCEPT with multi-column variants",
        "Add subquery support: scalar subqueries, IN/NOT IN, correlated EXISTS",
        "Add multiple CTE support with chaining and CTE+window function combinations",
        "Add CASE expressions with WHEN/THEN/ELSE, including CASE in GROUP BY",
        "Add comprehensive test suites: test_comprehensive_coverage.py, test_sql_edge_cases.py",
        "Add bench_fts.rs example for FTS in Rust",
    ],
    "v1.6.0": [
        "Add in-place UPDATE execution via `scan_and_update_inplace` — writes directly to disk without full rewrite",
        "Rewrite delta store from log-based to map-based (`HashMap`) preventing unbounded log growth",
        "Add `delta_batch_update_rows` for multiple cell-level updates in a single lock acquisition",
        "Add `scan_numeric_range_mmap_with_ids` — combined WHERE column scan + ID retrieval in one pass",
        "Add pending delta application (`apply_pending_deltas_in_place`) for save_v4() baking",
        "Add DELETE + window function + FTS benchmarks",
        "Add warm no-gc benchmark mode for measuring without GC interference",
    ],
    "v1.5.0": [
        "Add `open_for_read_with_file` to save 2 syscalls by reusing pre-opened File handle",
        "Add fast path for SELECT COUNT(*) — bypasses SQL parser, reads directly from `active_row_count` atomic",
        "Add batch Arrow column → Python list converter (`arrow_col_to_pylist`) eliminating per-element dispatch",
        "Refactor LRU cache with AtomicU64 for lock-free access time tracking",
        "Add regex pre-compilation in Python client for faster SQL parsing",
        "Merge File::open + metadata into a single syscall in `get_cached_backend`",
        "Add 8 new benchmarks: full scan→pandas, multi-GROUP BY, LIKE, multi-condition, multi-ORDER BY, COUNT DISTINCT, IN filter, UPDATE",
    ],
    "v1.4.0": [
        "Add FTS auto-sync on INSERT: newly inserted rows automatically indexed",
        "Add FTS auto-sync on DELETE: deleted rows removed from FTS indexes",
        "Add CREATE FTS INDEX backfill: existing rows indexed on creation",
        "Add ALTER FTS INDEX ... ENABLE backfill support",
        "Add cross-database SHOW FTS INDEXES with enabled/disabled status",
        "Remove roadmap/planning docs (HTAP_GAP_ANALYSIS.md, HTAP_ROADMAP.md, P0_IMPLEMENTATION_PLAN.md)",
    ],
    "v1.3.0": [
        "Add SQL-native full-text search DDL: CREATE/DROP/ALTER FTS INDEX, SHOW FTS INDEXES",
        "Add FTS query predicates: MATCH('query') and FUZZY_MATCH('query') in WHERE clauses",
        "Add FTS_GUIDE.md documentation (478 lines) covering architecture, SQL usage, and configuration",
        "Upgrade nanofts to 0.5.0 with zero-copy Arrow indexing (~3.3M docs/s throughput)",
        "Add global FtsManager registry and fts_config.json persistence",
    ],
    "v1.2.0": [
        "Add multi-database support with isolated subdirectories, `use_database()`, and cross-database SQL",
        "Add Arrow Flight gRPC server with zero-copy columnar data transfer",
        "Add unified `apexbase-serve` CLI launching both PG Wire and Flight servers",
        "Add bench_flight.py and bench_pg_wire.py comparing server protocol performance",
        "Add mmap-based aggregation path, per-instance backend caching, SQL parse caching",
        "Add TCP_NODELAY on server sockets and per-connection USE/\\c database switching",
        "Add comprehensive multi-database test suite (496 lines)",
    ],
    "v1.1.0": [
        "Add Extended Query Handler for prepared statement support",
        "Add SQL comment stripping (-- and /* */) before statement type detection",
        "Add proper PostgreSQL command tags (INSERT oid+rows, DELETE rows, etc.)",
        "Add pg_catalog support for pg_tables, pg_stat_user_tables",
        "Improve DROP TABLE with cleanup of WAL, delta, deltastore files",
        "Add line comment (--) support to SQL parser",
    ],
    "v1.0.0": [
        "Release v1.0 — first stable release",
        "Add PostgreSQL wire protocol server with DBeaver/psql/DataGrip/pgAdmin/Navicat support",
        "Add pg_catalog compatibility layer (pg_namespace, pg_database, pg_class, pg_attribute, information_schema)",
        "Add Arrow type→PostgreSQL type mapping and FieldInfo generation",
        "Major README rewrite with HTAP feature list (transactions, MVCC, indexing, window functions, 70+ built-in functions)",
    ],
    "v0.6.0": [
        "Split monolithic executor.rs (~11554 lines) into modular submodules (aggregation, ddl, dml, expressions, joins, select, window, tests)",
        "Split monolithic on_demand.rs (~10072 lines) into modular submodules (agg_wal, arrow_io, header, mmap_scan, read_write, storage_core, tests, types)",
        "Add window function support (ROW_NUMBER, RANK, DENSE_RANK, NTILE, LAG, LEAD, FIRST_VALUE, LAST_VALUE)",
        "Add CTE (WITH ... AS), EXPLAIN/ANALYZE query plans, UNION/UNION ALL support",
        "Add concurrency stress test, constraint tests, DuckDB memory comparison tests",
    ],
    "v0.5.0": [
        "Add query planner with OLTP (index-based) and OLAP (vectorized) routing",
        "Add indexing subsystem: B-Tree, hash index, index manager with CREATE/DROP/REINDEX SQL",
        "Add transaction support: transaction context, manager, OCC conflict detection",
        "Add MVCC engine: snapshot isolation, row versioning, garbage collection",
        "Add DeltaStore cell-level updates with merge-commit compaction",
        "Add global storage engine registry with LRU cache",
        "Add HTAP architecture support with scaling module (partition, shard, node, router)",
        "Rewrite on_demand.rs (~4000→9939 lines) with full columnar storage, LZ4/Zstd compression, WAL durability",
    ],
    "v0.4.2": [
        "Add persistent Row Group Bloom Filters for fast row group skipping",
        "Add DirectCountAgg for fast counting aggregation using direct array indexing",
        "Remove DuckDB comparison benchmarks and legacy test scripts",
        "Improve on_demand storage with per-column dictionary caching and SIMD filter scanning",
    ],
    "v0.4.1": [
        "Add comprehensive documentation: API_REFERENCE.md (865 lines), EXAMPLES.md, QUICK_START.md",
        "Add benchmark suite comparing ApexBase vs DuckDB",
        "Add bloom filter-based row group skipping in storage engine",
        "Add `list_databases()` method to Python client",
    ],
    "v0.4.0": [
        "Add adaptive multi-column filter strategy with smart column ordering",
        "Add SIMD-accelerated take/gather operations (AVX2 on x86, NEON on ARM)",
        "Add fast path for Complex (Filter+Group+Order) queries",
        "Add `delete(where=)` support to Python client",
        "Refactor __init__.py (~1412→157 lines) by delegating to client.py",
    ],
    "v0.3.0": [
        "Add DML support (INSERT, DELETE, UPDATE, TRUNCATE) to SQL executor",
        "Add DDL support (CREATE/DROP/ALTER TABLE) with schema management",
        "Add comprehensive DDL/DML test suite (~1957 lines)",
        "Add performance optimization fast paths for string/numeric/multi-condition filters",
        "Translate all documentation from Chinese to English",
    ],
    "v0.2.3": [
        "Add multi-platform wheel builds (Windows/macOS/Linux) for Python 3.9-3.13",
        "Refactor GitHub Actions: separate Linux manylinux build job",
    ],
    "v0.2.2": [
        "Add maturin multi-platform wheel builds with Python interpreter path detection",
        "Remove old technical design docs",
        "Improve README with installation, usage, and benchmarks",
    ],
    "v0.2.1": [
        "Version bump and minor CI fixes",
    ],
    "v0.2.0": [
        "Complete core rewrite from Python to Rust with PyO3 bindings",
        "Add Rust-native columnar storage engine with .apex file format",
        "Add Rust-native SQL query executor with vectorized Arrow processing and Cranelift JIT",
        "Add Arrow IPC zero-copy data bridge between Rust and Python",
        "Add comprehensive test suite (~12000 lines)",
        "Replace DuckDB storage with custom columnar storage engine",
    ],
    "v0.1.0": [
        "Replace SQLite storage with DuckDB-based storage engine",
        "Add full-text search module with index creation, fuzzy matching, and snippets",
        "Rewrite ApexClient with caching, auto-switch, list_tables, close methods",
        "Add ResultView with to_dict/to_pandas/to_arrow converters",
        "Add comprehensive test suite (test_apex_client.py, 622 lines)",
    ],
    "v0.0.2": [
        "Add package metadata: Apache-2.0 license, PyPI classifiers for Python 3.9–3.13",
        "Fix GitHub Actions CI/CD pipeline for PyPI publishing",
    ],
    "v0.0.1": [
        "Initial release — project bootstrap",
        "ApexClient Python class for database management (CRUD operations)",
        "SQLite-based Storage with batch operations and LimitedDict LRU cache",
        "Query class with SQL-like filter parsing (WHERE, ORDER BY, LIMIT, BETWEEN, LIKE, IN)",
        "SQLParser/SQLGenerator modules for expression handling",
        "GitHub Actions CI/CD pipeline setup",
    ],
}


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], capture_output=True, text=True, check=True
    ).stdout.strip()


def get_tags() -> list[str]:
    return _git("tag", "--sort=v:refname").splitlines()


def get_tag_date(tag: str) -> str:
    d = _git("log", "-1", "--format=%ai", tag)
    return d.split()[0] if d else ""


def get_changed_files(old: str | None, new: str) -> list[str]:
    """Return exact paths without relying on the human-oriented --stat format."""
    if old is None:
        output = _git("diff-tree", "--root", "--no-commit-id", "--name-only", "-r", new)
    else:
        output = _git("diff", "--name-only", old, new)
    return [path for path in output.splitlines() if path]


def categorize_files(files: list[str]) -> OrderedDict[str, int]:
    counts: OrderedDict[str, int] = OrderedDict()
    for path in files:
        matched = False
        for pattern, label in MODULE_RULES:
            if re.search(pattern, path):
                if label not in counts:
                    counts[label] = 0
                counts[label] += 1
                matched = True
                break
        if not matched:
            if "Other" not in counts:
                counts["Other"] = 0
            counts["Other"] += 1
    return counts


def is_version_bump(msg: str) -> bool:
    return bool(re.match(
        r'^(feat:\s*)?([Bb]ump|update)\s+version\s+(to\s+)?v?\d+\.\d+\.\d+',
        msg, re.IGNORECASE
    ))


def get_release_summary(tag: str) -> list[str]:
    if tag in KNOWN_HIGHLIGHTS:
        return KNOWN_HIGHLIGHTS[tag]
    return ["See changed files breakdown below."]


def format_tag_section(
    tag: str,
    date: str,
    highlights: list[str],
    module_counts: OrderedDict[str, int],
    prev_tag: str | None,
) -> str:
    lines: list[str] = []

    if tag == "Unreleased":
        lines.append("## Unreleased")
    else:
        tag_url = f"{REPO_URL}/releases/tag/{tag}"
        lines.append(f"## [{tag}]({tag_url})")

    if date and tag != "Unreleased":
        lines.append(f"*{date}*")
    lines.append("")

    if prev_tag and tag != "Unreleased":
        compare = f"{REPO_URL}/compare/{prev_tag}...{tag}"
        lines.append(f"[Compare with {prev_tag}]({compare})")
        lines.append("")

    for h in highlights:
        lines.append(f"- {h}")
    lines.append("")

    if module_counts:
        lines.append("<details>")
        lines.append("<summary><b>Changed files by module</b></summary>")
        lines.append("")
        lines.append("| Module | Files changed |")
        lines.append("|--------|--------------:|")
        for mod, count in module_counts.items():
            lines.append(f"| {mod} | {count} |")
        lines.append("")
        lines.append("</details>")

    return "\n".join(lines)


def main() -> None:
    repo_root = _git("rev-parse", "--show-toplevel")
    output = Path(repo_root) / "docs" / "release-notes.md"

    tags = get_tags()
    if not tags:
        print("No tags found", file=sys.stderr)
        sys.exit(1)

    parts: list[str] = []
    parts.append("# Release Notes")
    parts.append("")
    parts.append(
        "This page summarizes the changes introduced in each ApexBase release, "
        "grouped by functional area."
    )
    parts.append("")
    parts.append(
        "<!-- This file is auto-generated by `scripts/generate_release_notes.py`. "
        "Do not edit manually. -->"
    )
    parts.append("")

    display_tags = list(reversed(tags))

    for i, tag in enumerate(display_tags):
        if tag == tags[-1]:
            unreleased = _git(
                "log", "--oneline", "--format=%s", f"{tag}..HEAD"
            ).splitlines()
            unreleased = [c.strip() for c in unreleased if c.strip()]
            if unreleased:
                parts.append("## Unreleased")
                parts.append("")
                for c in unreleased:
                    if not is_version_bump(c):
                        parts.append(f"- {c}")
                parts.append("")
                parts.append("---\n")

        prev_tag = None if i == len(display_tags) - 1 else display_tags[i + 1]

        date = get_tag_date(tag)
        files = get_changed_files(prev_tag, tag)
        module_counts = categorize_files(files)
        highlights = get_release_summary(tag)

        section = format_tag_section(tag, date, highlights, module_counts, prev_tag)
        parts.append(section)
        parts.append("")
        parts.append("---\n")

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(parts) + "\n", encoding="utf-8")
    print(f"Release notes written to {output}")


if __name__ == "__main__":
    main()
