# Documentation Index

## Quick Links

- Root README: `../README.md`
- Python API entry: `../apexbase/python/apexbase/__init__.py`
- Test suite notes: `../test/README.md`
- CI release workflow: `../.github/workflows/build_release.yml`

## Usage Notes

- The primary public entry point is `apexbase.ApexClient`.
- The persistence file is a single file `apexbase.apex`, stored by default under the directory specified by `ApexClient(dirpath=...)`.
- For queries, prefer `execute(sql)` (full SQL). For compatibility, you can use `query(where, limit=...)` (WHERE expression only).

## Local Development (conda `dev` environment)

```bash
# conda activate dev

# Local development install (Rust extension)
maturin develop --release

# Run tests
python run_tests.py
```

## Release Checklist

- Version consistency: keep `version` aligned between `pyproject.toml` and `Cargo.toml`
- Tests pass locally: `python run_tests.py`
- Tag to trigger CI: push a `v*` tag (e.g. `v0.4.0`)
- Configure PyPI token: set `PYPI_API_TOKEN` in GitHub Secrets

## Known Limitations / Notes

- This project is primarily consumed via the Python API; the Rust crate is mainly for the PyO3 extension and internal engine reuse.
- Some advanced SQL capabilities (e.g. complex subqueries, concurrency locks) are still evolving; treat the behavior covered in `test/` as the current source of truth.
