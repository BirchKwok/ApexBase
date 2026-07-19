"""Run the full pytest suite repeatedly and enforce a stable runtime limit."""

from __future__ import annotations

import argparse
import json
import re
import statistics
import subprocess
import sys
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
PYTEST_DURATION = re.compile(r"\bin ([0-9]+(?:\.[0-9]+)?)s(?:\s|$)")


def extract_pytest_duration(output):
    """Return pytest's reported suite duration from its final summary line."""
    matches = PYTEST_DURATION.findall(output)
    if not matches:
        raise ValueError("pytest output does not contain a suite duration")
    return float(matches[-1])


def evaluate_samples(samples, limit_seconds, allowed_over_limit, max_spread_seconds):
    """Return all reasons that a set of serial runtime samples is unstable."""
    if not samples:
        raise ValueError("at least one runtime sample is required")
    if limit_seconds <= 0 or allowed_over_limit < 0 or max_spread_seconds < 0:
        raise ValueError("runtime limits must be non-negative and the time limit positive")

    median = statistics.median(samples)
    spread = max(samples) - min(samples)
    over_limit = sum(sample > limit_seconds for sample in samples)
    errors = []
    if median > limit_seconds:
        errors.append(f"median {median:.2f}s exceeds {limit_seconds:.2f}s")
    if over_limit > allowed_over_limit:
        errors.append(
            f"{over_limit} samples exceed {limit_seconds:.2f}s "
            f"(allowed {allowed_over_limit})"
        )
    if spread > max_spread_seconds:
        errors.append(f"spread {spread:.2f}s exceeds {max_spread_seconds:.2f}s")
    return errors


def run_pytest_samples(runs, pytest_args):
    samples = []
    command = [sys.executable, "-m", "pytest", *pytest_args]
    for index in range(1, runs + 1):
        print(f"\npytest runtime sample {index}/{runs}", flush=True)
        result = subprocess.run(
            command,
            cwd=REPOSITORY_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        output = result.stdout + result.stderr
        print(output, end="" if output.endswith("\n") else "\n")
        if result.returncode != 0:
            raise RuntimeError(f"pytest sample {index} failed with exit code {result.returncode}")
        samples.append(extract_pytest_duration(output))
    return samples


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--limit-seconds", type=float, default=9.0)
    parser.add_argument("--allowed-over-limit", type=int, default=1)
    parser.add_argument("--max-spread-seconds", type=float, default=1.0)
    parser.add_argument("--output", type=Path)
    parser.add_argument("pytest_args", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    if args.runs <= 0:
        parser.error("--runs must be positive")

    try:
        samples = run_pytest_samples(args.runs, args.pytest_args)
        errors = evaluate_samples(
            samples,
            args.limit_seconds,
            args.allowed_over_limit,
            args.max_spread_seconds,
        )
    except (RuntimeError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    report = {
        "runs": args.runs,
        "limit_seconds": args.limit_seconds,
        "allowed_over_limit": args.allowed_over_limit,
        "max_spread_seconds": args.max_spread_seconds,
        "samples_seconds": samples,
        "median_seconds": statistics.median(samples),
        "spread_seconds": max(samples) - min(samples),
        "passed": not errors,
        "errors": errors,
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    print(
        "pytest runtime samples: "
        + ", ".join(f"{sample:.2f}s" for sample in samples)
        + f"; median={report['median_seconds']:.2f}s; "
        + f"spread={report['spread_seconds']:.2f}s"
    )
    if errors:
        for error in errors:
            print(f"ERROR: {error}", file=sys.stderr)
        return 1
    print("pytest runtime gate passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
