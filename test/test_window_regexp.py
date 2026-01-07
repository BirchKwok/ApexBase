import pytest
import tempfile
import sys
import os
import time

# Add the apexbase python module to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'apexbase', 'python'))

try:
    from apexbase import ApexClient, ResultView
except ImportError as e:
    pytest.skip(f"ApexBase not available: {e}", allow_module_level=True)


class TestWindowRegexp:
    def test_regexp_and_row_number_over_partition_by(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            client = ApexClient(dirpath=temp_dir)

            rows = [
                {"col1": "A", "col2": "test_one", "col3": "a1", "col4": 2},
                {"col1": "A", "col2": "test_two", "col3": "a2", "col4": 1},
                {"col1": "A", "col2": "nope", "col3": "a3", "col4": 3},
                {"col1": "B", "col2": "test_x", "col3": "b1", "col4": 10},
                {"col1": "B", "col2": "test_y", "col3": "b2", "col4": 5},
                {"col1": "B", "col2": "zzz", "col3": "b3", "col4": 7},
            ]
            client.store(rows)

            sql = "SELECT col3, col1, col4, row_number() OVER (PARTITION BY col1 ORDER BY col4) AS rn FROM default WHERE col2 REGEXP \"test*\""

            t0 = time.perf_counter()
            result = client.execute(sql)
            t1 = time.perf_counter()

            assert isinstance(result, ResultView)

            out = result.to_dict()
            assert isinstance(out, list)

            # Should filter only rows where col2 matches "test*" prefix
            assert len(out) == 4

            # Validate per-partition ordering and row_number
            by_part = {}
            for r in out:
                by_part.setdefault(r["col1"], []).append(r)

            assert set(by_part.keys()) == {"A", "B"}

            for part, part_rows in by_part.items():
                # Must have rn and numeric
                assert all("rn" in rr for rr in part_rows)
                assert all(isinstance(rr["rn"], int) for rr in part_rows)

                # Sort expected by col4 ASC
                part_rows_sorted = sorted(part_rows, key=lambda x: x["col4"])
                rn_values = [r["rn"] for r in part_rows_sorted]
                assert rn_values == list(range(1, len(part_rows_sorted) + 1))

            # Basic timing output (not a strict perf assertion)
            print(f"window+regexp rows={len(out)} time_sec={(t1 - t0):.6f}")

            client.close()

    def test_row_number_over_implicit_alias(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            client = ApexClient(dirpath=temp_dir)

            rows = [
                {"title": "A", "number": 2},
                {"title": "A", "number": 1},
                {"title": "B", "number": 3},
            ]
            client.store(rows)
            client.flush()

            sql = "SELECT row_number() OVER (PARTITION BY title ORDER BY number ASC) rid FROM default"
            result = client.execute(sql)
            out = result.to_dict()
            assert len(out) == 3
            assert all("rid" in r for r in out)

            client.close()

    def test_select_star_with_row_number_alias(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            client = ApexClient(dirpath=temp_dir)

            rows = [
                {"title": "A", "number": 2},
                {"title": "A", "number": 1},
                {"title": "B", "number": 3},
            ]
            client.store(rows)
            client.flush()

            sql = "SELECT *, row_number() OVER (PARTITION BY title ORDER BY number ASC) rid FROM default"
            result = client.execute(sql)
            out = result.to_dict()
            assert len(out) == 3
            assert all("title" in r for r in out)
            assert all("number" in r for r in out)
            assert all("rid" in r for r in out)

            client.close()
