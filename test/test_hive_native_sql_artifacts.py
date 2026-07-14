"""Lightweight contracts for generated Hive benchmark SQL artifacts."""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1] / "benchmarks" / "sql"
WORKLOADS = (
    "hive_user_360.sql",
    "hive_user_complex.sql",
    "hive_user_most_complex.sql",
    "hive_user_syntax_torture.sql",
)


def test_hive_native_sql_artifacts_exist_and_keep_dialect_contracts():
    for workload in WORKLOADS:
        source = (ROOT / workload).read_text(encoding="utf-8")
        duckdb = (ROOT / "duckdb" / workload).read_text(encoding="utf-8")
        sqlite = (ROOT / "sqlite" / workload).read_text(encoding="utf-8")

        assert "${biz_date}" in duckdb
        assert "CREATE OR REPLACE" in duckdb
        assert len(duckdb) > len(source) // 2

        assert "{biz_date}" in sqlite
        assert "ods_user__" in sqlite
        assert "ods_user." not in sqlite
        assert "WITH " in sqlite.lstrip().upper()
