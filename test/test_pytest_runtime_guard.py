import importlib.util
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def guard():
    path = ROOT / "benchmarks" / "check_pytest_runtime.py"
    spec = importlib.util.spec_from_file_location("check_pytest_runtime", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_extract_pytest_duration_uses_final_summary(guard):
    output = "collected 1461 items in 0.20s\n1461 passed in 8.91s\n"
    assert guard.extract_pytest_duration(output) == 8.91


def test_runtime_samples_pass_with_one_bounded_outlier(guard):
    errors = guard.evaluate_samples([8.82, 8.87, 8.91, 8.96, 9.12], 9.0, 1, 1.0)
    assert errors == []


def test_runtime_median_over_limit_fails(guard):
    errors = guard.evaluate_samples([8.90, 9.01, 9.02, 9.03, 9.04], 9.0, 4, 1.0)
    assert errors == ["median 9.02s exceeds 9.00s"]


def test_too_many_slow_samples_fail(guard):
    errors = guard.evaluate_samples([8.90, 8.95, 8.99, 9.01, 9.02], 9.0, 1, 1.0)
    assert errors == ["2 samples exceed 9.00s (allowed 1)"]


def test_unstable_sample_spread_fails(guard):
    errors = guard.evaluate_samples([8.50, 8.70, 8.80, 8.90, 9.60], 9.0, 1, 1.0)
    assert errors == ["spread 1.10s exceeds 1.00s"]
