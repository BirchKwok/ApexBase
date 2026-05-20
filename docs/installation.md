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

## Install Documentation Tools

The documentation site is built with MkDocs Material.

```bash
python -m pip install -r docs/requirements.txt
python -m mkdocs serve
```

Open the local URL printed by MkDocs. To verify the production build:

```bash
python -m mkdocs build --strict
```

## GitHub Pages Deployment

The repository includes `.github/workflows/docs.yml`. It uses `mike` so users can switch between historical documentation versions from the header.

- Builds the MkDocs Material site on pull requests.
- Deploys `pyproject.toml`'s current version on pushes to `main`.
- Deploys the tag version on `v*` tags.
- Updates the `latest` alias and keeps older version directories on the `gh-pages` branch.

In the GitHub repository settings, set **Pages -> Build and deployment -> Source** to **Deploy from a branch**, then choose **`gh-pages`** and **`/(root)`**. After the next successful deployment, the site is published at:

```text
https://birchkwok.github.io/ApexBase/
```

For local versioned preview, use `mike` after installing the documentation requirements:

```bash
mike deploy 1.18.0 latest
mike serve
```

## Upgrade

```bash
python -m pip install --upgrade apexbase
```

When developing from source, rebuild the extension after Rust or Python binding changes:

```bash
maturin develop --release
```
