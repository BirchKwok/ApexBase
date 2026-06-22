from pathlib import Path

from apexbase import ApexClient


def _read_row_group_size(table_path: Path) -> int:
    header = table_path.read_bytes()[:32]
    assert len(header) >= 32
    return int.from_bytes(header[28:32], "little")


def test_adaptive_row_group_size_for_narrow_table(tmp_path):
    db_dir = tmp_path / "adaptive_rg"
    client = ApexClient(str(db_dir), drop_if_exists=True)
    client.create_table("narrow")
    client.use_table("narrow")

    row_count = 140_000
    client.store(
        {
            "a": list(range(row_count)),
            "b": list(range(row_count)),
        }
    )
    client.close()

    table_path = db_dir / "narrow.apex"
    assert table_path.exists()
    assert _read_row_group_size(table_path) == 131_072


def test_adaptive_row_group_size_for_wide_table(tmp_path):
    db_dir = tmp_path / "wide_rg"
    client = ApexClient(str(db_dir), drop_if_exists=True)
    client.create_table("wide")
    client.use_table("wide")

    row_count = 80_000
    wide_value = "x" * 256
    client.store(
        {
            "payload": [wide_value for _ in range(row_count)],
            "payload_copy": [wide_value for _ in range(row_count)],
        }
    )
    client.close()

    table_path = db_dir / "wide.apex"
    assert table_path.exists()
    assert _read_row_group_size(table_path) == 32_768


def test_string_encoding_stays_consistent_across_row_groups(tmp_path):
    db_dir = tmp_path / "mixed_string_encoding"
    client = ApexClient(str(db_dir), drop_if_exists=True)
    client.create_table("events")
    client.use_table("events")

    row_count = 70_000
    categories = ["common"] * 32_768 + [f"unique_{i}" for i in range(32_768, row_count)]
    client.store({
        "category": categories,
        "payload": ["x" * 256] * row_count,
    })
    client.close()

    reopened = ApexClient(str(db_dir))
    reopened.use_table("events")
    result = reopened.execute(
        "SELECT category FROM events WHERE category = 'unique_69999'"
    ).to_dict()
    assert result == [{"category": "unique_69999"}]
    reopened.close()
