# ApexBase Performance Guard

The performance guard compares ApexBase with an earlier ApexBase commit. It is
separate from the public SQLite/DuckDB scoreboard: competitor wins do not prove
that a new ApexBase change avoided a regression.

## Fast canary

Build the revision in release mode, then write a canary report:

```bash
maturin develop --release
python benchmarks/bench_perf_canary.py \
  --rows 200000 \
  --warmup 2 \
  --iterations 7 \
  --output /tmp/apexbase-perf.json
```

The canary covers bulk insert, point reads, limited and full scans, filtering,
grouping, ordering, aggregation, small writes, updates, and physical deletion.
It uses the same benchmark implementations as the public scoreboard.

## Compare two reports

Run the base and current revisions on the same machine with identical arguments,
then compare their JSON reports:

```bash
python benchmarks/compare_perf_baseline.py \
  /tmp/apexbase-perf-base.json \
  /tmp/apexbase-perf-current.json \
  --relative-threshold 0.15 \
  --absolute-threshold-ms 0.005 \
  --require-system-match
```

A metric fails when its slowdown exceeds both the relative threshold and the
absolute tolerance. The absolute tolerance prevents ordinary timer noise from
failing microsecond-scale metrics. Missing metrics and incompatible benchmark
configurations are errors rather than silently skipped comparisons.

On variable shared runners, pass additional samples with `--baseline-sample`
and `--current-sample`. Each metric is compared using the median for its sample
set; every sample must contain the same metrics and compatible configuration.

## Automation

`.github/workflows/performance.yml` runs the canary for pull requests. It builds
the PR base commit and the proposed commit in release mode, benchmarks both on
the same runner, lets build load settle, then installs the prebuilt wheels in
balanced base/current/current/base/base/current order. It compares each commit's
three-sample median, records the actual source commit in every JSON report, and
uploads the reports. The scheduled full benchmark runs on a
dedicated Apple Silicon runner labelled `apexbase-performance`; a GitHub-hosted
runner is not treated as a stable nightly performance machine.

## Repeatable pytest runtime gate

The dedicated nightly runner also measures the complete, serial pytest suite
five times:

```bash
python benchmarks/check_pytest_runtime.py \
  --runs 5 \
  --limit-seconds 9.0 \
  --allowed-over-limit 1 \
  --max-spread-seconds 1.0 \
  --output /tmp/pytest-runtime.json
```

All five runs must pass functionally. The runtime gate requires a median no
greater than 9 seconds, permits at most one noisy sample above 9 seconds, and
rejects a sample spread above 1 second. This keeps 9 seconds as the central
limit without allowing one unusually fast run to hide a consistently slow
suite. Tests are never parallelized, skipped, or reduced for this check.

Local investigation on 2026-07-19 measured five adjacent full-suite runs at
9.55, 9.02, 9.12, 9.00, and 8.90 seconds (median 9.02, spread 0.65). JUnit
timing attributed about 8.18 seconds to the 1461 test cases themselves, with
the remaining time spread across collection, pytest overhead, process startup,
and filesystem cleanup. The slowest repeatable cases were the memory-pressure
case, two million-row SQL cases, and the cross-process visibility case; there
was no single disposable test responsible for the boundary.

After adding the guard's five unit tests, an end-to-end guard run measured
8.82, 8.84, 8.87, 8.79, and 8.68 seconds (median 8.82, spread 0.19), with all
1466 tests passing in every sample.

The repository's required local acceptance sequence in `AGENTS.md` remains the
source of truth. The canary supplements the full benchmark; it does not replace
it.
