"""Regression tests for import-order / native-extension stability on macOS."""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "apexbase", "python"))


def _run_python_snippet(snippet: str) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        [p for p in [env.get("PYTHONPATH", ""), sys.path[0]] if p]
    )
    return subprocess.run(
        [sys.executable, "-c", snippet],
        check=False,
        capture_output=True,
        text=True,
        timeout=60,
        env=env,
    )


def test_core_import_does_not_eagerly_load_pyarrow():
    """`from apexbase._core import ApexStorage` must not import pyarrow at init time."""
    result = _run_python_snippet(
        """
import sys
from apexbase._core import ApexStorage  # noqa: F401
assert "pyarrow" not in sys.modules
assert "pandas" not in sys.modules
assert "polars" not in sys.modules
print("ok")
"""
    )
    assert result.returncode == 0, result.stderr or result.stdout


def test_apex_storage_constructs_on_empty_directory():
    from apexbase._core import ApexStorage

    db_dir = tempfile.mkdtemp(prefix="apexbase_import_stability_")
    storage = ApexStorage(os.path.join(db_dir, "apexbase.apex"))
    assert storage is not None


def test_lazy_apex_client_import():
    result = _run_python_snippet(
        """
import sys
import apexbase
assert "apexbase.client" not in sys.modules
client_cls = apexbase.ApexClient
assert client_cls.__name__ == "ApexClient"
assert "apexbase.client" in sys.modules
print("ok")
"""
    )
    assert result.returncode == 0, result.stderr or result.stdout


@pytest.mark.parametrize("import_path", ["apexbase", "apexbase._core"])
def test_import_paths_do_not_segfault(import_path: str):
    result = _run_python_snippet(
        f"""
import os
import tempfile
import importlib
importlib.import_module({import_path!r})
from apexbase._core import ApexStorage
db_dir = tempfile.mkdtemp(prefix="apexbase_import_path_")
ApexStorage(os.path.join(db_dir, "apexbase.apex"))
print("ok")
"""
    )
    assert result.returncode == 0, result.stderr or result.stdout
