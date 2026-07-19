import importlib.util
import json
from pathlib import Path
import re

import pytest


ROOT = Path(__file__).resolve().parents[1]


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _report(metrics, **config):
    return {
        "suite": "apexbase-canary",
        "config": {"rows": 1000, "warmup": 1, "iterations": 3, **config},
        "results": [
            {"category": "test", "query": name, "ApexBase": value}
            for name, value in metrics.items()
        ],
    }


@pytest.fixture(scope="module")
def guard():
    return _load(
        "compare_perf_baseline",
        ROOT / "benchmarks" / "compare_perf_baseline.py",
    )


@pytest.fixture(scope="module")
def benchmark():
    return _load(
        "bench_report_metadata",
        ROOT / "benchmarks" / "bench_vs_sqlite_duckdb.py",
    )


def test_comparison_passes_within_relative_threshold(guard):
    rows = guard.compare_reports(
        _report({"scan": 10.0}),
        _report({"scan": 10.9}),
        relative_threshold=0.10,
        absolute_threshold_ms=0.0,
    )

    assert rows == [{
        "query": "scan",
        "baseline_ms": 10.0,
        "current_ms": 10.9,
        "delta_ms": pytest.approx(0.9),
        "limit_ms": 11.0,
        "regressed": False,
    }]


def test_comparison_reports_regression(guard):
    rows = guard.compare_reports(
        _report({"scan": 10.0}),
        _report({"scan": 11.1}),
        relative_threshold=0.10,
        absolute_threshold_ms=0.0,
    )

    assert rows[0]["regressed"] is True


def test_absolute_tolerance_protects_microbenchmarks_from_noise(guard):
    rows = guard.compare_reports(
        _report({"point lookup": 0.002}),
        _report({"point lookup": 0.006}),
        relative_threshold=0.10,
        absolute_threshold_ms=0.005,
    )

    assert rows[0]["regressed"] is False


def test_missing_metric_is_an_error(guard):
    with pytest.raises(guard.ReportError, match="missing metrics"):
        guard.compare_reports(
            _report({"scan": 10.0, "insert": 5.0}),
            _report({"scan": 10.0}),
        )


def test_symmetric_samples_are_aggregated_by_median(guard):
    baseline = guard.aggregate_report_metrics([
        _report({"scan": 10.0}),
        _report({"scan": 12.0}),
        _report({"scan": 100.0}),
    ])
    current = guard.aggregate_report_metrics([
        _report({"scan": 11.5}),
        _report({"scan": 12.0}),
        _report({"scan": 50.0}),
    ])

    rows = guard.compare_metric_sets(
        baseline,
        current,
        relative_threshold=0.0,
        absolute_threshold_ms=0.0,
    )

    assert baseline == {"scan": 12.0}
    assert current == {"scan": 12.0}
    assert rows[0]["regressed"] is False


def test_sample_metric_mismatch_is_an_error(guard):
    with pytest.raises(guard.ReportError, match="sample 2 metric set differs: missing insert"):
        guard.aggregate_report_metrics([
            _report({"scan": 10.0, "insert": 5.0}),
            _report({"scan": 10.0}),
        ])


def test_incompatible_config_is_reported(guard):
    errors = guard.compatibility_errors(
        _report({"scan": 10.0}),
        _report({"scan": 10.0}, rows=2000),
    )

    assert errors == ["config.rows differs"]


def test_system_match_rejects_dependency_or_build_drift(guard):
    baseline = _report({"scan": 10.0})
    baseline.update({
        "system": {"machine": "arm64"},
        "dependencies": {"numpy": "2.1.3"},
        "build": {"rustc": "1.88.0"},
    })
    current = _report({"scan": 10.0})
    current.update({
        "system": {"machine": "arm64"},
        "dependencies": {"numpy": "2.4.3"},
        "build": {"rustc": "1.89.0"},
    })

    errors = guard.compatibility_errors(
        baseline,
        current,
        require_system_match=True,
    )

    assert errors == ["dependencies differs", "build differs"]


def test_vector_metrics_are_included(guard):
    report = _report({"scan": 10.0})
    report["vector_similarity"] = {
        "head_to_head": [{"query": "TopK L2", "ApexBase": 7.25}],
        "batch": [{"query": "Batch TopK L2", "ApexBase": 42.5}],
    }

    assert guard.extract_apex_metrics(report) == {
        "scan": 10.0,
        "TopK L2": 7.25,
        "Batch TopK L2": 42.5,
    }


def test_canary_manifest_references_real_benchmark_methods():
    canary = _load(
        "bench_perf_canary",
        ROOT / "benchmarks" / "bench_perf_canary.py",
    )

    methods = {method for _, method, _ in canary.CANARY_SPECS}
    available = set(dir(canary.full_bench.ApexBaseBench))
    assert methods <= available
    assert len(canary.CANARY_SPECS) == len({name for name, _, _ in canary.CANARY_SPECS})


def test_full_benchmark_json_keeps_microsecond_precision(capsys):
    canary = _load(
        "bench_perf_canary_precision",
        ROOT / "benchmarks" / "bench_perf_canary.py",
    )
    spec = [("micro", "unused", False, False, False, None)]

    rows = canary.full_bench.print_benchmark_section(
        "test",
        "test",
        spec,
        {"micro": {"ApexBase": 0.0004214}},
        ["ApexBase"],
        16,
    )

    capsys.readouterr()
    assert rows[0]["ApexBase"] == 0.000421


def test_benchmark_report_metadata_is_complete_and_serializable(benchmark):
    metadata = benchmark.get_report_metadata("test-suite")

    assert metadata["format_version"] == 1
    assert metadata["suite"] == "test-suite"
    assert re.fullmatch(r"[0-9a-f]{40}", metadata["git"]["commit"])
    assert set(metadata["git"]) == {"commit", "branch", "dirty"}
    assert {
        "platform", "machine", "processor", "cpu_count", "memory_gb", "python"
    } <= metadata["system"].keys()
    assert set(metadata["dependencies"]) == {
        "apexbase", "sqlite", "duckdb", "pyarrow", "numpy", "pandas", "polars"
    }
    assert set(metadata["build"]) == {"maturin", "rustc", "cargo"}
    assert all(isinstance(value, str) and value for value in metadata["dependencies"].values())
    assert all(isinstance(value, str) and value for value in metadata["build"].values())
    json.dumps(metadata)


def test_benchmark_git_metadata_honors_ci_source_override(benchmark, monkeypatch):
    commit = "a" * 40
    monkeypatch.setenv("APEXBASE_BENCHMARK_COMMIT", commit)
    monkeypatch.setenv("APEXBASE_BENCHMARK_BRANCH", "pull-request-base")

    git = benchmark.get_git_info()

    assert git["commit"] == commit
    assert git["branch"] == "pull-request-base"


def test_performance_workflow_isolates_base_and_current_cargo_targets():
    workflow = (ROOT / ".github" / "workflows" / "performance.yml").read_text()

    assert workflow.count('CARGO_TARGET_DIR="${PERF_BASE_TARGET_DIR}"') == 3
    assert workflow.count('CARGO_TARGET_DIR="${PERF_CURRENT_TARGET_DIR}"') == 3
    assert "CARGO_TARGET_DIR: ${{ github.workspace }}/target/performance\n" not in workflow


def test_nightly_full_comparison_uses_symmetric_median_samples():
    workflow = (ROOT / ".github" / "workflows" / "performance.yml").read_text()
    nightly = workflow.split("  nightly-full:\n", 1)[1]

    for side in ("base", "current"):
        for sample in range(1, 4):
            assert f'perf-{side}-full-{sample}.json' in nightly
    assert nightly.count("--baseline-sample") == 2
    assert nightly.count("--current-sample") == 2


def test_submillisecond_sql_metrics_use_calibrated_median_timing(benchmark):
    specs = {name: spec for name, *spec in benchmark.BENCHMARKS}

    for name, method in (
        ("COUNT WHERE category", "bench_count_where_category"),
        ("Point Lookup (SQL by ID)", "bench_point_lookup"),
    ):
        assert specs[name][1:4] == [False, False, True]
        assert specs[name][4] is None
        assert method in benchmark.MICRO_MEDIAN_BENCHMARK_METHODS
