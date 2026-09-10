"""R5.9: B-phase entry evidence tool (contention matrix + speedup curve).

Covers, against the installed release wheel:
1. Report integrity of both subcommands on a small generated dataset:
   required fields, window files, derived comparisons, exit status.
2. Argument validation failures (bad concurrency / parallel values,
   missing dataset without --rows).
"""

import json
import os
import subprocess
import sys
import tempfile

SCRIPT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "benchmarks",
    "bench_parallel_evidence.py",
)
ROWS = 4_000  # small: smoke scale, one row group (parallel fold falls back
# to serial; report integrity is the point of these tests)


def _run(args, expect_fail=False):
    result = subprocess.run(
        [sys.executable, SCRIPT] + args,
        capture_output=True,
        text=True,
        timeout=600,
    )
    if expect_fail:
        assert result.returncode != 0, (
            f"expected failure, got 0\nstdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}"
        )
        return result
    assert result.returncode == 0, (
        f"exit {result.returncode}\nstdout:\n{result.stdout}\n"
        f"stderr:\n{result.stderr}"
    )
    return result


def _load(path):
    return json.load(open(path))


def test_matrix_smoke_report_integrity():
    with tempfile.TemporaryDirectory() as db_dir, tempfile.TemporaryDirectory() as out:
        _run([
            "matrix", "--db-dir", db_dir, "--rows", str(ROWS),
            "--concurrency", "1,2", "--parallel", "off,2",
            "--windows", "2", "--queries", "20", "--warmup-queries", "4",
            "--output", out,
        ])
        summary = _load(os.path.join(out, "matrix-summary.json"))
        records = summary["records"]
        # 2 windows x 2 concurrencies x 2 parallel levels.
        assert len(records) == 8
        for record in records:
            assert record["total_queries"] == 20
            assert record["qps"] > 0
            assert record["p50_ms"] > 0 and record["p99_ms"] >= record["p50_ms"]
            assert record["rows"] == ROWS
            assert record["window"] in (1, 2)
            for field in ("loadavg", "nproc", "git"):
                assert field in record
        # One JSON per (config, window).
        for window in (1, 2):
            for concurrency in (1, 2):
                for parallel in ("off", "2"):
                    path = os.path.join(out, f"matrix-c{concurrency}-p{parallel}-w{window}.json")
                    assert os.path.exists(path)
        derived = summary["derived"]
        assert len(derived) == 2  # c=1 p=2 and c=2 p=2
        for d in derived:
            assert d["parallel"] == "2"
            assert len(d["per_window"]) == 2
            assert d["throughput_delta_pct"] is not None


def test_curve_smoke_report_integrity():
    with tempfile.TemporaryDirectory() as db_dir, tempfile.TemporaryDirectory() as out:
        _run([
            "curve", "--db-dir", db_dir, "--rows", str(ROWS),
            "--threads", "1,2", "--windows", "2", "--queries", "10",
            "--warmup-queries", "2", "--output", out,
        ])
        summary = _load(os.path.join(out, "curve-summary.json"))
        records = summary["records"]
        # 2 windows x 2 thread levels.
        assert len(records) == 4
        for record in records:
            assert record["total_queries"] == 10
            assert record["p50_ms"] > 0
        assert summary["speedup_vs_serial"]["1"] == 1.0
        for level in ("1", "2"):
            assert level in summary["per_thread_median_ms"]
            assert level in summary["effective_threads"]
            assert os.path.exists(os.path.join(out, f"curve-t{level}-w1.json"))


def test_argument_validation_fails():
    with tempfile.TemporaryDirectory() as db_dir:
        # Concurrency 0 is rejected before any dataset work.
        _run([
            "matrix", "--db-dir", db_dir, "--rows", str(ROWS),
            "--concurrency", "0", "--parallel", "off,2",
            "--windows", "1", "--queries", "4", "--warmup-queries", "1",
        ], expect_fail=True)
        # A parallel level of 1 is not a valid opt-in value (N>=2 or off).
        _run([
            "matrix", "--db-dir", db_dir, "--rows", str(ROWS),
            "--concurrency", "1", "--parallel", "1",
            "--windows", "1", "--queries", "4", "--warmup-queries", "1",
        ], expect_fail=True)
        # A curve without the serial baseline is rejected.
        _run([
            "curve", "--db-dir", db_dir, "--rows", str(ROWS),
            "--threads", "2,4", "--windows", "1", "--queries", "4",
            "--warmup-queries", "1",
        ], expect_fail=True)


def test_missing_dataset_requires_rows():
    with tempfile.TemporaryDirectory() as db_dir, tempfile.TemporaryDirectory() as out:
        _run([
            "matrix", "--db-dir", db_dir,
            "--concurrency", "1", "--parallel", "off",
            "--windows", "1", "--queries", "4", "--warmup-queries", "1",
            "--output", out,
        ], expect_fail=True)
