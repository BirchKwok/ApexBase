"""Regression tests for import-order / native-extension stability on macOS."""

from __future__ import annotations

import importlib
import os
import sys
import tempfile

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "apexbase", "python"))


def test_core_import_does_not_eagerly_load_pyarrow():
    """`from apexbase._core import ApexStorage` must not import pyarrow at init time."""
    for name in list(sys.modules):
        if name == "apexbase" or name.startswith("apexbase."):
            del sys.modules[name]
        if name in {"pyarrow", "pandas", "polars"} or name.startswith(
            ("pyarrow.", "pandas.", "polars.")
        ):
            del sys.modules[name]

    from apexbase._core import ApexStorage  # noqa: F401

    assert "pyarrow" not in sys.modules
    assert "pandas" not in sys.modules
    assert "polars" not in sys.modules


def test_apex_storage_constructs_on_empty_directory():
    from apexbase._core import ApexStorage

    db_dir = tempfile.mkdtemp(prefix="apexbase_import_stability_")
    storage = ApexStorage(os.path.join(db_dir, "apexbase.apex"))
    assert storage is not None


def test_lazy_apex_client_import():
    import apexbase

    assert "apexbase.client" not in sys.modules
    client_cls = apexbase.ApexClient
    assert client_cls.__name__ == "ApexClient"
    assert "apexbase.client" in sys.modules


@pytest.mark.parametrize("import_path", ["apexbase", "apexbase._core"])
def test_import_paths_do_not_segfault(import_path: str):
    importlib.import_module(import_path)
    from apexbase._core import ApexStorage

    db_dir = tempfile.mkdtemp(prefix="apexbase_import_path_")
    ApexStorage(os.path.join(db_dir, "apexbase.apex"))
