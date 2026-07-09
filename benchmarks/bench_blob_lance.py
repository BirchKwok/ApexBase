#!/usr/bin/env python3
"""
Benchmark ApexBase descriptor-backed BLOB storage against Lance Blob.

The harness follows Lance's Blob API shape where available:
  - write with blob_field/blob_array
  - descriptor-only reads with dataset.to_table(columns=["payload"])
  - lazy blob reads with read_blobs/take_blobs

If the installed Lance build does not expose the Blob helpers, the Lance side is
skipped unless --allow-lance-large-binary-fallback is provided.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import platform
import random
import shutil
import statistics
import sys
import tempfile
import time
from dataclasses import dataclass
from typing import Any, Callable, Optional


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "apexbase", "python"))

TABLE_NAME = "blob_bench"
PAYLOAD_COLUMN = "payload"


try:
    from apexbase import ApexClient  # type: ignore

    HAS_APEX = True
    APEX_IMPORT_ERROR = None
except Exception as exc:  # pragma: no cover - benchmark diagnostics
    ApexClient = None  # type: ignore
    HAS_APEX = False
    APEX_IMPORT_ERROR = exc

try:
    import pyarrow as pa  # type: ignore

    HAS_ARROW = True
    ARROW_IMPORT_ERROR = None
except Exception as exc:  # pragma: no cover - benchmark diagnostics
    pa = None  # type: ignore
    HAS_ARROW = False
    ARROW_IMPORT_ERROR = exc

try:
    import lance  # type: ignore

    HAS_LANCE = True
    LANCE_IMPORT_ERROR = None
except Exception as exc:  # pragma: no cover - benchmark diagnostics
    lance = None  # type: ignore
    HAS_LANCE = False
    LANCE_IMPORT_ERROR = exc


@dataclass
class BlobDataset:
    row_ids: list[int]
    kinds: list[str]
    sizes: list[int]
    blobs: list[bytes]
    total_bytes: int


def make_blob(seed: int, row_index: int, size: int) -> bytes:
    out = bytearray()
    counter = 0
    while len(out) < size:
        out.extend(
            hashlib.blake2b(
                f"apexbase-blob-bench:{seed}:{row_index}:{counter}".encode(),
                digest_size=64,
            ).digest()
        )
        counter += 1
    return bytes(out[:size])


def build_dataset(args: argparse.Namespace) -> BlobDataset:
    rng = random.Random(args.seed)
    dedicated_count = int(args.rows * args.dedicated_fraction)
    packed_count = int(args.rows * args.packed_fraction)
    dedicated_count = max(0, min(args.rows, dedicated_count))
    packed_count = max(0, min(args.rows - dedicated_count, packed_count))
    small_count = args.rows - dedicated_count - packed_count

    kinds = (
        ["small"] * small_count
        + ["packed"] * packed_count
        + ["dedicated"] * dedicated_count
    )
    rng.shuffle(kinds)

    sizes = []
    blobs = []
    for i, kind in enumerate(kinds):
        if kind == "small":
            size = args.small_size
        elif kind == "packed":
            size = args.packed_size
        else:
            size = args.dedicated_size
        sizes.append(size)
        blobs.append(make_blob(args.seed, i, size))

    return BlobDataset(
        row_ids=list(range(args.rows)),
        kinds=kinds,
        sizes=sizes,
        blobs=blobs,
        total_bytes=sum(sizes),
    )


def mib(value: float) -> float:
    return value / (1024.0 * 1024.0)


def now() -> float:
    return time.perf_counter()


def summarize_times(times: list[float]) -> dict[str, float]:
    return {
        "min_s": min(times),
        "avg_s": statistics.fmean(times),
        "median_s": statistics.median(times),
        "max_s": max(times),
        "iterations": float(len(times)),
    }


def timed(
    fn: Callable[[], Any],
    *,
    warmup: int,
    iterations: int,
    consume: Optional[Callable[[Any], int]] = None,
) -> tuple[dict[str, float], int]:
    for _ in range(warmup):
        gc.collect()
        result = fn()
        if consume is not None:
            consume(result)

    times: list[float] = []
    consumed = 0
    for _ in range(iterations):
        gc.collect()
        start = now()
        result = fn()
        elapsed = now() - start
        consumed = consume(result) if consume is not None else 0
        times.append(elapsed)
    return summarize_times(times), consumed


def timed_write(
    writer: Callable[[str], Any],
    base_path: str,
    iterations: int,
) -> tuple[dict[str, float], str]:
    os.makedirs(base_path, exist_ok=True)
    times: list[float] = []
    last_path = ""
    for i in range(iterations):
        path = os.path.join(base_path, f"write_{i}")
        shutil.rmtree(path, ignore_errors=True)
        start = now()
        writer(path)
        times.append(now() - start)
        if last_path and last_path != path:
            shutil.rmtree(last_path, ignore_errors=True)
        last_path = path
    return summarize_times(times), last_path


def consume_arrow_table(table: Any) -> int:
    if table is None:
        return 0
    nbytes = int(getattr(table, "nbytes", 0) or 0)
    return int(getattr(table, "num_rows", 0) or 0) + nbytes


def consume_dicts(rows: Any) -> int:
    if rows is None:
        return 0
    total = 0
    for row in rows:
        if row is None:
            continue
        for value in row.values():
            if isinstance(value, (bytes, bytearray, memoryview)):
                total += len(value)
            else:
                total += 1
    return total


def choose_read_workload(args: argparse.Namespace, data: BlobDataset) -> tuple[list[int], list[int]]:
    rng = random.Random(args.seed + 1009)
    indices = [rng.randrange(len(data.blobs)) for _ in range(args.reads)]
    offsets: list[int] = []
    for idx in indices:
        size = data.sizes[idx]
        max_offset = max(0, size - 1)
        offsets.append(rng.randrange(max_offset + 1) if max_offset else 0)
    return indices, offsets


def blob_throughput_metric(metric: dict[str, float], total_bytes: int) -> dict[str, float]:
    out = dict(metric)
    min_s = max(out["min_s"], 1e-12)
    avg_s = max(out["avg_s"], 1e-12)
    out["throughput_mib_s_min_time"] = mib(total_bytes) / min_s
    out["throughput_mib_s_avg_time"] = mib(total_bytes) / avg_s
    return out


def run_apex(
    args: argparse.Namespace,
    data: BlobDataset,
    work_dir: str,
    read_indices: list[int],
    range_offsets: list[int],
) -> dict[str, Any]:
    if not HAS_APEX:
        return {"available": False, "error": repr(APEX_IMPORT_ERROR)}

    apex_root = os.path.join(work_dir, "apex")

    def write(path: str) -> None:
        client = ApexClient(path, drop_if_exists=True, durability=args.apex_durability)
        try:
            client.create_table(
                TABLE_NAME,
                {"row_id": "int64", "kind": "string", PAYLOAD_COLUMN: "blob"},
            )
            client.store(
                {
                    "row_id": data.row_ids,
                    "kind": data.kinds,
                    PAYLOAD_COLUMN: data.blobs,
                }
            )
            client.flush()
        finally:
            client.close()

    try:
        write_metric, db_path = timed_write(write, apex_root, args.write_iterations)
    except Exception as exc:
        return {
            "available": False,
            "error": (
                repr(exc)
                + " (rebuild/reinstall the ApexBase Python extension if it does not recognize blob schemas)"
            ),
        }
    write_metric = blob_throughput_metric(write_metric, data.total_bytes)

    client = ApexClient(db_path, durability=args.apex_durability)
    client.use_table(TABLE_NAME)

    def scan_metadata() -> Any:
        return client.execute(f"SELECT row_id, kind FROM {TABLE_NAME}").to_arrow()

    def descriptor_info() -> list[dict[str, Any]]:
        ids = [row_index + 1 for row_index in read_indices]
        return client.read_blob_infos(PAYLOAD_COLUMN, ids)

    def random_full_read() -> int:
        total = 0
        checksum = 0
        ids = [row_index + 1 for row_index in read_indices]
        for payload in client.read_blobs(PAYLOAD_COLUMN, ids):
            if payload:
                total += len(payload)
                checksum ^= payload[0]
        return total + checksum

    def random_range_read() -> int:
        total = 0
        checksum = 0
        ids = [row_index + 1 for row_index in read_indices]
        for payload in client.read_blob_ranges(
            PAYLOAD_COLUMN,
            ids,
            range_offsets,
            args.range_len,
        ):
            if payload:
                total += len(payload)
                checksum ^= payload[0]
        return total + checksum

    project_rows = min(args.project_rows, len(data.blobs))

    def project_payload() -> Any:
        return client.execute(
            f"SELECT {PAYLOAD_COLUMN} FROM {TABLE_NAME} LIMIT {project_rows}"
        ).to_arrow()

    try:
        scan_metric, scan_consumed = timed(
            scan_metadata,
            warmup=args.warmup,
            iterations=args.iterations,
            consume=consume_arrow_table,
        )
        info_metric, info_consumed = timed(
            descriptor_info,
            warmup=args.warmup,
            iterations=args.iterations,
            consume=consume_dicts,
        )
        full_metric, full_consumed = timed(
            random_full_read,
            warmup=args.warmup,
            iterations=args.iterations,
        )
        range_metric, range_consumed = timed(
            random_range_read,
            warmup=args.warmup,
            iterations=args.iterations,
        )
        project_metric, project_consumed = timed(
            project_payload,
            warmup=args.warmup,
            iterations=args.iterations,
            consume=consume_arrow_table,
        )
    finally:
        client.close()

    return {
        "available": True,
        "storage_path": db_path,
        "write": write_metric,
        "scan_without_blob_projection": {
            **scan_metric,
            "consumed": float(scan_consumed),
        },
        "descriptor_info_random": {
            **info_metric,
            "consumed": float(info_consumed),
        },
        "random_full_blob_read": {
            **full_metric,
            "consumed": float(full_consumed),
        },
        "random_range_blob_read": {
            **range_metric,
            "consumed": float(range_consumed),
        },
        "project_blob_column": {
            **project_metric,
            "consumed": float(project_consumed),
            "project_rows": float(project_rows),
        },
    }


def resolve_lance_blob_helpers() -> tuple[Optional[Callable[..., Any]], Optional[Callable[..., Any]], str]:
    if not HAS_LANCE:
        return None, None, "not_installed"

    blob_field = getattr(lance, "blob_field", None)
    blob_array = getattr(lance, "blob_array", None)
    if callable(blob_field) and callable(blob_array):
        return blob_field, blob_array, "lance_blob_top_level"

    try:
        import importlib

        lance_blob_v2 = importlib.import_module("lance.blob.v2")

        blob_field = getattr(lance_blob_v2, "blob_field", None)
        blob_array = getattr(lance_blob_v2, "blob_array", None)
        if callable(blob_field) and callable(blob_array):
            return blob_field, blob_array, "lance_blob_v2"
    except Exception:
        pass

    return None, None, "missing_blob_helpers"


def lance_schema_and_payload_array(
    args: argparse.Namespace,
    data: BlobDataset,
) -> tuple[Any, Any, str]:
    blob_field, blob_array, mode = resolve_lance_blob_helpers()
    if blob_field is not None and blob_array is not None:
        schema = pa.schema(
            [
                pa.field("row_id", pa.int64()),
                pa.field("kind", pa.string()),
                blob_field(PAYLOAD_COLUMN),
            ]
        )
        return schema, blob_array(data.blobs), mode

    if hasattr(lance, "BlobFile"):
        schema = pa.schema(
            [
                pa.field("row_id", pa.int64()),
                pa.field("kind", pa.string()),
                pa.field(
                    PAYLOAD_COLUMN,
                    pa.large_binary(),
                    metadata={b"lance-encoding:blob": b"true"},
                ),
            ]
        )
        return (
            schema,
            pa.array(data.blobs, type=pa.large_binary()),
            "lance_blob_field_metadata",
        )

    if not args.allow_lance_large_binary_fallback:
        raise RuntimeError(
            "Installed Lance does not expose blob_field/blob_array. "
            "Upgrade Lance or pass --allow-lance-large-binary-fallback."
        )

    schema = pa.schema(
        [
            pa.field("row_id", pa.int64()),
            pa.field("kind", pa.string()),
            pa.field(PAYLOAD_COLUMN, pa.large_binary()),
        ]
    )
    return schema, pa.array(data.blobs, type=pa.large_binary()), "large_binary_fallback"


def get_lance_dataset(path: str) -> Any:
    if hasattr(lance, "dataset"):
        return lance.dataset(path)
    return lance.LanceDataset(path)


def maybe_read_lance_blob_file(blob_file: Any, offset: int = 0, length: Optional[int] = None) -> bytes:
    handle = blob_file
    close_after = False
    if hasattr(blob_file, "__enter__"):
        handle = blob_file.__enter__()
        close_after = True
    try:
        needs_slice = offset
        if offset and hasattr(handle, "seek"):
            handle.seek(offset)
            needs_slice = 0
        if length is None:
            data = handle.read()
        else:
            if needs_slice:
                data = handle.read()
            else:
                try:
                    data = handle.read(length)
                except TypeError:
                    data = handle.read()
        if isinstance(data, memoryview):
            data = data.tobytes()
        else:
            data = bytes(data)
        if needs_slice:
            return data[offset : offset + length if length is not None else None]
        return data
    finally:
        if close_after:
            blob_file.__exit__(None, None, None)


def lance_take_blob_files(ds: Any, indices: list[int]) -> list[Any]:
    if hasattr(ds, "take_blobs"):
        return list(ds.take_blobs(PAYLOAD_COLUMN, indices))
    raise RuntimeError("Lance dataset does not expose take_blobs")


def lance_read_blobs(ds: Any, indices: list[int]) -> list[bytes]:
    if hasattr(ds, "read_blobs"):
        try:
            return [bytes(blob) for _, blob in ds.read_blobs(PAYLOAD_COLUMN, indices=indices)]
        except TypeError:
            return [bytes(blob) for _, blob in ds.read_blobs(PAYLOAD_COLUMN, indices)]

    blob_files = lance_take_blob_files(ds, indices)
    return [maybe_read_lance_blob_file(blob_file) for blob_file in blob_files]


def run_lance(
    args: argparse.Namespace,
    data: BlobDataset,
    work_dir: str,
    read_indices: list[int],
    range_offsets: list[int],
) -> dict[str, Any]:
    if not HAS_LANCE:
        return {"available": False, "error": repr(LANCE_IMPORT_ERROR)}
    if not HAS_ARROW:
        return {"available": False, "error": repr(ARROW_IMPORT_ERROR)}

    lance_root = os.path.join(work_dir, "lance")
    helper_mode_holder = {"mode": ""}

    def write(path: str) -> None:
        schema, payload_array, helper_mode = lance_schema_and_payload_array(args, data)
        helper_mode_holder["mode"] = helper_mode
        table = pa.Table.from_arrays(
            [
                pa.array(data.row_ids, type=pa.int64()),
                pa.array(data.kinds, type=pa.string()),
                payload_array,
            ],
            schema=schema,
        )
        kwargs: dict[str, Any] = {"mode": "overwrite"}
        if args.lance_data_storage_version and helper_mode != "lance_blob_field_metadata":
            kwargs["data_storage_version"] = args.lance_data_storage_version
        try:
            lance.write_dataset(table, path, **kwargs)
        except (TypeError, ValueError) as exc:
            if "data_storage_version" not in kwargs:
                raise
            kwargs.pop("data_storage_version", None)
            lance.write_dataset(table, path, **kwargs)

    try:
        write_metric, ds_path = timed_write(write, lance_root, args.write_iterations)
    except Exception as exc:
        return {
            "available": False,
            "error": repr(exc),
            "helper_mode": helper_mode_holder["mode"],
        }

    write_metric = blob_throughput_metric(write_metric, data.total_bytes)
    ds = get_lance_dataset(ds_path)

    def scan_metadata() -> Any:
        return ds.to_table(columns=["row_id", "kind"])

    def descriptor_info() -> Any:
        if hasattr(ds, "take_blobs"):
            return lance_take_blob_files(ds, read_indices)
        return ds.to_table(columns=[PAYLOAD_COLUMN])

    def random_full_read() -> int:
        total = 0
        checksum = 0
        for payload in lance_read_blobs(ds, read_indices):
            total += len(payload)
            if payload:
                checksum ^= payload[0]
        return total + checksum

    def random_range_read() -> int:
        total = 0
        checksum = 0
        try:
            blob_files = lance_take_blob_files(ds, read_indices)
            for blob_file, offset in zip(blob_files, range_offsets):
                payload = maybe_read_lance_blob_file(blob_file, offset, args.range_len)
                total += len(payload)
                if payload:
                    checksum ^= payload[0]
            return total + checksum
        except Exception:
            for payload, offset in zip(lance_read_blobs(ds, read_indices), range_offsets):
                chunk = payload[offset : offset + args.range_len]
                total += len(chunk)
                if chunk:
                    checksum ^= chunk[0]
            return total + checksum

    project_rows = min(args.project_rows, len(data.blobs))

    def project_payload() -> Any:
        if helper_mode_holder["mode"] != "large_binary_fallback":
            try:
                scanner = ds.scanner(
                    columns=[PAYLOAD_COLUMN],
                    blob_handling="all_binary",
                )
                if hasattr(scanner, "limit"):
                    scanner = scanner.limit(project_rows)
                    return scanner.to_table()
                table = scanner.to_table()
                return table.slice(0, project_rows)
            except TypeError:
                pass
        payloads = lance_read_blobs(ds, list(range(project_rows)))
        return pa.Table.from_pydict(
            {PAYLOAD_COLUMN: payloads},
            schema=pa.schema([pa.field(PAYLOAD_COLUMN, pa.large_binary())]),
        )

    scan_metric, scan_consumed = timed(
        scan_metadata,
        warmup=args.warmup,
        iterations=args.iterations,
        consume=consume_arrow_table,
    )
    descriptor_metric, descriptor_consumed = timed(
        descriptor_info,
        warmup=args.warmup,
        iterations=args.iterations,
        consume=lambda value: len(value)
        if isinstance(value, list)
        else consume_arrow_table(value),
    )
    full_metric, full_consumed = timed(
        random_full_read,
        warmup=args.warmup,
        iterations=args.iterations,
    )
    range_metric, range_consumed = timed(
        random_range_read,
        warmup=args.warmup,
        iterations=args.iterations,
    )
    project_metric, project_consumed = timed(
        project_payload,
        warmup=args.warmup,
        iterations=args.iterations,
        consume=consume_arrow_table,
    )

    return {
        "available": True,
        "storage_path": ds_path,
        "helper_mode": helper_mode_holder["mode"],
        "write": write_metric,
        "scan_without_blob_projection": {
            **scan_metric,
            "consumed": float(scan_consumed),
        },
        "descriptor_info_random": {
            **descriptor_metric,
            "consumed": float(descriptor_consumed),
        },
        "random_full_blob_read": {
            **full_metric,
            "consumed": float(full_consumed),
        },
        "random_range_blob_read": {
            **range_metric,
            "consumed": float(range_consumed),
        },
        "project_blob_column": {
            **project_metric,
            "consumed": float(project_consumed),
            "project_rows": float(project_rows),
        },
    }


def add_ratios(results: dict[str, Any]) -> None:
    apex = results.get("apexbase", {})
    lance_result = results.get("lance", {})
    if not apex.get("available") or not lance_result.get("available"):
        results["ratios"] = {}
        return

    ratios: dict[str, dict[str, Any]] = {}
    metric_names = [
        "write",
        "scan_without_blob_projection",
        "descriptor_info_random",
        "random_full_blob_read",
        "random_range_blob_read",
        "project_blob_column",
    ]
    for name in metric_names:
        apex_metric = apex.get(name, {})
        lance_metric = lance_result.get(name, {})
        apex_s = apex_metric.get("min_s")
        lance_s = lance_metric.get("min_s")
        if apex_s is None or lance_s is None or lance_s <= 0:
            continue
        ratio = apex_s / lance_s
        if ratio < 0.97:
            winner = "apexbase"
        elif ratio > 1.03:
            winner = "lance"
        else:
            winner = "tie_3_percent"
        ratios[name] = {
            "apex_min_s": apex_s,
            "lance_min_s": lance_s,
            "apex_over_lance": ratio,
            "winner": winner,
        }
    results["ratios"] = ratios


def print_summary(results: dict[str, Any]) -> None:
    print("\nApexBase vs Lance Blob Benchmark")
    print("=" * 72)
    dataset = results["dataset"]
    print(
        "Rows: {rows:,} | payload: {mib:.2f} MiB | reads: {reads:,} | range: {range_len} B".format(
            rows=int(dataset["rows"]),
            mib=dataset["total_mib"],
            reads=int(dataset["reads"]),
            range_len=int(dataset["range_len"]),
        )
    )
    print(
        "Mix: small={small:.1%}, packed={packed:.1%}, dedicated={dedicated:.1%}".format(
            small=dataset["small_fraction"],
            packed=dataset["packed_fraction"],
            dedicated=dataset["dedicated_fraction"],
        )
    )

    for engine_name in ("apexbase", "lance"):
        engine = results.get(engine_name, {})
        if engine.get("available"):
            extra = ""
            if engine_name == "lance":
                extra = f" ({engine.get('helper_mode', 'unknown')})"
            print(f"{engine_name}: available{extra}")
        else:
            print(f"{engine_name}: unavailable: {engine.get('error')}")

    ratios = results.get("ratios", {})
    if not ratios:
        print("\nNo ApexBase/Lance ratio available.")
        return

    print("\nMetric                          Apex min      Lance min     A/L    Winner")
    print("-" * 72)
    for name, metric in ratios.items():
        print(
            f"{name:<30} "
            f"{metric['apex_min_s'] * 1000:>9.3f} ms "
            f"{metric['lance_min_s'] * 1000:>10.3f} ms "
            f"{metric['apex_over_lance']:>6.2f} "
            f"{metric['winner']}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark ApexBase BLOB storage against Lance Blob."
    )
    parser.add_argument("--rows", type=int, default=200)
    parser.add_argument("--reads", type=int, default=200)
    parser.add_argument("--project-rows", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--write-iterations", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--small-size", type=int, default=4 * 1024)
    parser.add_argument("--packed-size", type=int, default=256 * 1024)
    parser.add_argument("--dedicated-size", type=int, default=5 * 1024 * 1024)
    parser.add_argument("--packed-fraction", type=float, default=0.28)
    parser.add_argument("--dedicated-fraction", type=float, default=0.02)
    parser.add_argument("--range-len", type=int, default=64 * 1024)
    parser.add_argument("--work-dir", default=None)
    parser.add_argument("--keep-data", action="store_true")
    parser.add_argument("--json-output", default=None)
    parser.add_argument("--skip-lance", action="store_true")
    parser.add_argument("--skip-apex", action="store_true")
    parser.add_argument("--apex-durability", choices=["fast", "safe", "max"], default="fast")
    parser.add_argument("--lance-data-storage-version", default="2.2")
    parser.add_argument(
        "--allow-lance-large-binary-fallback",
        action="store_true",
        help="Use regular large_binary on Lance if Blob helpers are unavailable.",
    )
    args = parser.parse_args()

    if args.rows <= 0:
        parser.error("--rows must be positive")
    if args.reads <= 0:
        parser.error("--reads must be positive")
    if args.iterations <= 0:
        parser.error("--iterations must be positive")
    if args.write_iterations <= 0:
        parser.error("--write-iterations must be positive")
    if args.packed_fraction < 0 or args.dedicated_fraction < 0:
        parser.error("fractions must be non-negative")
    if args.packed_fraction + args.dedicated_fraction > 1:
        parser.error("--packed-fraction + --dedicated-fraction must be <= 1")
    if args.small_size > 64 * 1024:
        parser.error("--small-size must be <= 64 KiB to exercise Apex inline BLOB")
    if not (64 * 1024 < args.packed_size <= 4 * 1024 * 1024):
        parser.error("--packed-size must be in (64 KiB, 4 MiB]")
    if args.dedicated_size <= 4 * 1024 * 1024:
        parser.error("--dedicated-size must be > 4 MiB")
    return args


def main() -> int:
    args = parse_args()
    work_dir = args.work_dir or tempfile.mkdtemp(prefix="apex_blob_lance_")
    os.makedirs(work_dir, exist_ok=True)

    data = build_dataset(args)
    read_indices, range_offsets = choose_read_workload(args, data)
    small_fraction = max(0.0, 1.0 - args.packed_fraction - args.dedicated_fraction)

    results: dict[str, Any] = {
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "work_dir": work_dir,
        },
        "dataset": {
            "rows": float(args.rows),
            "reads": float(args.reads),
            "range_len": float(args.range_len),
            "total_bytes": float(data.total_bytes),
            "total_mib": mib(float(data.total_bytes)),
            "small_size": float(args.small_size),
            "packed_size": float(args.packed_size),
            "dedicated_size": float(args.dedicated_size),
            "small_fraction": small_fraction,
            "packed_fraction": args.packed_fraction,
            "dedicated_fraction": args.dedicated_fraction,
        },
    }

    try:
        if args.skip_apex:
            results["apexbase"] = {"available": False, "error": "skipped"}
        else:
            results["apexbase"] = run_apex(
                args, data, work_dir, read_indices, range_offsets
            )

        if args.skip_lance:
            results["lance"] = {"available": False, "error": "skipped"}
        else:
            results["lance"] = run_lance(
                args, data, work_dir, read_indices, range_offsets
            )

        add_ratios(results)
        print_summary(results)

        if args.json_output:
            with open(args.json_output, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2, sort_keys=True)
            print(f"\nWrote JSON results: {args.json_output}")

        return 0
    finally:
        if not args.keep_data and args.work_dir is None:
            shutil.rmtree(work_dir, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
