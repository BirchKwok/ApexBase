# apexbase — Rust Core

High-performance HTAP storage engine for ApexBase.

## Overview

This is the Rust core library (`apexbase` crate) that powers ApexBase:

- **V4 Row Group columnar storage** — 64K-row groups with per-column null bitmaps and dictionary encoding
- **Delta write path** — append-only `.apex.delta` files for fast transactional inserts
- **SQL engine** — parser, executor, JIT-compiled predicates (Cranelift)
- **Arrow IPC bridge** — zero-copy data transfer to Python via PyArrow
- **Full-text search** — NanoFTS integration with fuzzy matching
- **WAL + durability** — configurable fast / safe / max levels
- **Cross-platform** — Linux, macOS, Windows (x86_64 & ARM64)

## Build

```bash
# Python extension (recommended)
maturin develop --release

# Rust library only
cargo build --release
```

## License

Apache-2.0

