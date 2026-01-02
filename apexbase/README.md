# apex-core

High-performance storage engine for ApexBase (Rust core).

## Overview

This is the Rust core library that provides:
- Custom single-file data storage format with high I/O efficiency
- B-tree indexing for fast queries
- LRU caching for frequently accessed data
- Python bindings via PyO3

## Features

- **Single-file Storage**: All data stored in one file with custom format
- **High Performance**: Optimized for both read and write operations
- **Memory Efficient**: Uses memory-mapped I/O and LRU cache
- **Python Integration**: Full Python API compatibility with the original ApexBase

## Installation

```bash
pip install apex-core
```

Or build from source:

```bash
cd apex-core
maturin develop --release
```

## Usage

```python
from apex_core import PyApexClient

# Create a client
client = PyApexClient("/path/to/data")

# Store data
client.store({"key": "value"})

# Query data
results = client.query("SELECT * FROM default_table")
```

## License

Apache-2.0

