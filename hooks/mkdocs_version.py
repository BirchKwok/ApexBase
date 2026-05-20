"""Populate the documentation version from pyproject.toml."""

from __future__ import annotations

from pathlib import Path
import re


def _read_project_version(repo_root: Path) -> str:
    pyproject = repo_root / "pyproject.toml"
    text = pyproject.read_text(encoding="utf-8")

    try:
        import tomllib
    except ModuleNotFoundError:
        tomllib = None

    if tomllib is not None:
        data = tomllib.loads(text)
        version = data.get("project", {}).get("version")
        if version:
            return str(version)

    match = re.search(r'(?m)^version\s*=\s*"([^"]+)"', text)
    if not match:
        raise RuntimeError("Could not find [project].version in pyproject.toml")
    return match.group(1)


def on_config(config):
    repo_root = Path(__file__).resolve().parents[1]
    version = _read_project_version(repo_root)

    extra = config.setdefault("extra", {})
    version_config = extra.setdefault("version", {})
    version_config["current"] = version
    version_config.setdefault("label", f"v{version}")

    return config
