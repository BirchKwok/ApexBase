"""Tests for temporary table registration from data files (CSV/JSON/Parquet)."""

import os
import tempfile

import pytest

from apexbase import ApexClient


@pytest.fixture
def temp_dir():
    d = tempfile.mkdtemp()
    yield d
    import shutil
    shutil.rmtree(d, ignore_errors=True)


def test_register_temp_table_csv(temp_dir):
    client = ApexClient(temp_dir, drop_if_exists=True)
    client.create_table("_dummy")
    client.use_table("_dummy")

    csv_path = os.path.join(temp_dir, "test.csv")
    with open(csv_path, "w") as f:
        f.write("name,age,score\nAlice,25,85.5\nBob,30,90.0\nCharlie,35,78.2\n")

    client.register_temp_table("people", csv_path)

    result = client.execute("SELECT * FROM people").to_dict()
    assert len(result) == 3

    result = client.execute("SELECT * FROM people WHERE age > 28").to_dict()
    assert len(result) == 2

    client.close()


def test_register_temp_table_json_ndjson(temp_dir):
    client = ApexClient(temp_dir, drop_if_exists=True)
    client.create_table("_dummy")
    client.use_table("_dummy")

    json_path = os.path.join(temp_dir, "test.json")
    with open(json_path, "w") as f:
        f.write('{"x":1,"y":10}\n{"x":2,"y":20}\n{"x":3,"y":30}\n')

    client.register_temp_table("points", json_path)

    result = client.execute("SELECT * FROM points where x >= 2").to_dict()
    assert len(result) == 2
    assert result[0]["y"] == 20

    client.close()


def test_register_temp_table_aggregate(temp_dir):
    client = ApexClient(temp_dir, drop_if_exists=True)
    client.create_table("_dummy")
    client.use_table("_dummy")

    csv_path = os.path.join(temp_dir, "data.csv")
    with open(csv_path, "w") as f:
        f.write("category,val\nA,10\nA,20\nB,30\nB,40\nB,50\n")

    client.register_temp_table("sales", csv_path)

    result = client.execute(
        "SELECT category, SUM(val) as total FROM sales GROUP BY category ORDER BY category"
    ).to_dict()
    assert result[0]["category"] == "A"
    assert result[1]["category"] == "B"
    assert result[0]["total"] == 30
    assert result[1]["total"] == 120

    client.close()


def test_temp_table_drop(temp_dir):
    client = ApexClient(temp_dir, drop_if_exists=True)
    client.create_table("_dummy")
    client.use_table("_dummy")

    csv_path = os.path.join(temp_dir, "drop_test.csv")
    with open(csv_path, "w") as f:
        f.write("col\n1\n2\n3\n")

    client.register_temp_table("tmp_data", csv_path)
    result = client.execute("SELECT COUNT(*) FROM tmp_data").scalar()
    assert result == 3

    client.drop_temp_table("tmp_data")

    with pytest.raises(Exception):
        client.execute("SELECT * FROM tmp_data")

    client.close()


def test_temp_table_shadows_persistent(temp_dir):
    client = ApexClient(temp_dir, drop_if_exists=True)
    client.create_table("_dummy")
    client.use_table("_dummy")

    # Create persistent table
    client.create_table("shadow")
    client.use_table("shadow")
    client.store({"val": 999})
    client.flush()

    # Verify persistent data
    result = client.execute("SELECT COUNT(*) FROM shadow").scalar()
    assert result == 1

    # Register temp table with same name
    csv_path = os.path.join(temp_dir, "shadow.csv")
    with open(csv_path, "w") as f:
        f.write("val\n100\n200\n300\n")

    client.register_temp_table("shadow", csv_path)

    # Temp should shadow persistent
    result = client.execute("SELECT COUNT(*) FROM shadow").scalar()
    assert result == 3

    # Drop temp — persistent should be back
    client.drop_temp_table("shadow")
    result = client.execute("SELECT COUNT(*) FROM shadow").scalar()
    assert result == 1

    client.close()


def test_temp_table_multiple_queries_fast(temp_dir):
    """Verify temp table queries are fast (bypass file parsing)."""
    import time

    client = ApexClient(temp_dir, drop_if_exists=True)
    client.create_table("_dummy")
    client.use_table("_dummy")

    csv_path = os.path.join(temp_dir, "perf.csv")
    rows = 5000
    with open(csv_path, "w") as f:
        f.write("a,b,c,d\n")
        for i in range(rows):
            f.write(f"{i},{i*2},{i*3:.1f},str_{i}\n")

    client.register_temp_table("perf_test", csv_path)

    # Run multiple queries — should all be fast
    times = []
    for _ in range(10):
        t0 = time.perf_counter()
        result = client.execute("SELECT COUNT(*) FROM perf_test").scalar()
        elapsed = (time.perf_counter() - t0) * 1000
        times.append(elapsed)
        assert result == rows

    avg_time = sum(times) / len(times)
    assert avg_time < 50, f"Temp table queries too slow: {avg_time:.2f}ms avg"

    client.close()
