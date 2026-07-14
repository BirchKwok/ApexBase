"""Keep every supported public Python API represented in the perf/memory suite."""

import importlib.util
from pathlib import Path
import re


BENCHMARK = Path(__file__).parents[1] / "benchmarks" / "bench_public_api_memory.py"
ROOT = Path(__file__).parents[1]
RUST_BENCHMARK = ROOT / "examples" / "bench_rust_public_api_memory.rs"
RUST_INTERNAL_BENCHMARK = ROOT / "examples" / "bench_rust_internal_memory.rs"


def _load_benchmark_module():
    spec = importlib.util.spec_from_file_location("bench_public_api_memory", BENCHMARK)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_every_public_python_api_has_a_perf_and_memory_case():
    benchmark = _load_benchmark_module()
    discovered = benchmark.discover_public_api_names()
    covered = set(benchmark.PUBLIC_API_CASES)
    assert discovered == covered, (
        f"missing cases: {sorted(discovered - covered)}; "
        f"stale cases: {sorted(covered - discovered)}"
    )


def test_public_api_case_names_are_unique():
    benchmark = _load_benchmark_module()
    assert len(benchmark.PUBLIC_API_CASES) == len(set(benchmark.PUBLIC_API_CASES))


def _rust_manifest():
    source = RUST_BENCHMARK.read_text(encoding="utf-8")
    body = re.search(
        r"RUST_PUBLIC_API_CASES:\s*&\[&str\]\s*=\s*&\[(.*?)\];",
        source,
        re.S,
    ).group(1)
    return re.findall(r'"([^"]+)"', body)


def _impl_methods(path, type_name):
    source = path.read_text(encoding="utf-8")
    marker = f"impl {type_name} {{"
    start = source.index(marker) + len(marker)
    depth = 1
    end = start
    while depth:
        if source[end] == "{":
            depth += 1
        elif source[end] == "}":
            depth -= 1
        end += 1
    body = source[start : end - 1]
    return set(re.findall(r"\bpub fn\s+([A-Za-z_][A-Za-z0-9_]*)", body))


def test_every_supported_public_rust_api_has_a_perf_and_memory_case():
    embedded = ROOT / "apexbase" / "src" / "embedded" / "mod.rs"
    sources = {
        "ApexDBBuilder": embedded,
        "ApexDB": embedded,
        "Table": embedded,
        "ResultSet": embedded,
        "Row": ROOT / "apexbase" / "src" / "data" / "row.rs",
        "Value": ROOT / "apexbase" / "src" / "data" / "value.rs",
        "DataType": ROOT / "apexbase" / "src" / "data" / "types.rs",
        "DurabilityLevel": ROOT / "apexbase" / "src" / "storage" / "mod.rs",
    }
    discovered = {
        f"{type_name}.{method}"
        for type_name, path in sources.items()
        for method in _impl_methods(path, type_name)
    }
    discovered.update(
        {"embedded.record_batch_to_rows", "embedded.arrow_value_at"}
    )
    covered = set(_rust_manifest())
    assert discovered == covered, (
        f"missing cases: {sorted(discovered - covered)}; "
        f"stale cases: {sorted(covered - discovered)}"
    )


def test_rust_public_api_case_names_are_unique():
    cases = _rust_manifest()
    assert len(cases) == len(set(cases))


def test_internal_rust_memory_hot_paths_have_isolated_cases():
    source = RUST_INTERNAL_BENCHMARK.read_text(encoding="utf-8")
    body = re.search(
        r"RUST_INTERNAL_MEMORY_CASES:\s*&\[&str\]\s*=\s*&\[(.*?)\];",
        source,
        re.S,
    ).group(1)
    cases = re.findall(r'"([^"]+)"', body)

    assert cases == ["row_insert", "delta_insert", "typed_append"]
    assert len(cases) == len(set(cases))
    for case in cases:
        assert f'"{case}" =>' in source
