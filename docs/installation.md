# Installation

This page covers the common ways to install ApexBase and work on the documentation site.

## Requirements

| Component | Requirement |
| --- | --- |
| Python | 3.9 or newer |
| Operating systems | Linux, macOS, Windows |
| Architectures | x86_64, ARM64 |
| Source builds | Rust toolchain, `maturin`, and `protobuf` tooling for server features |

## Install From PyPI

```bash
python -m pip install apexbase
```

Check that the package imports:

```bash
python - <<'PY'
from apexbase import ApexClient

with ApexClient("./apexbase_smoke") as client:
    client.create_table("smoke")
    client.store({"ok": True})
    print(client.execute("SELECT COUNT(*) AS rows FROM smoke").to_dict())
PY
```

## Build From Source

```bash
git clone https://github.com/BirchKwok/ApexBase.git
cd ApexBase
python -m pip install maturin
maturin develop --release
```

For a development environment with test tools:

```bash
python -m pip install -e ".[dev]"
maturin develop --release
python -m pytest test/ -q
```

## Upgrade

```bash
python -m pip install --upgrade apexbase
```

When developing from source, rebuild the extension after Rust or Python binding changes:

```bash
maturin develop --release
```
