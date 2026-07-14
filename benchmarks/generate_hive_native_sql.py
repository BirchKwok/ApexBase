"""Generate auditable DuckDB and SQLite SQL artifacts for the Hive benchmarks."""

from pathlib import Path

from bench_hive_complex_vs_duckdb import SQL_FILES, duckdb_sql
from bench_hive_sqlite_equivalent import PORTABLE_SQL, sqlite_sql


ROOT = Path(__file__).resolve().parent / "sql"
DUCKDB_ROOT = ROOT / "duckdb"
SQLITE_ROOT = ROOT / "sqlite"


def generated_sql() -> dict[Path, str]:
    files: dict[Path, str] = {}
    for source in SQL_FILES:
        hive = source.read_text(encoding="utf-8")
        files[DUCKDB_ROOT / source.name] = duckdb_sql(hive, "${biz_date}")
    for workload, sql in PORTABLE_SQL.items():
        files[SQLITE_ROOT / workload] = sqlite_sql(sql)
    return files


def main() -> None:
    files = generated_sql()
    for path, sql in files.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(sql, encoding="utf-8")
        print(f"generated {path.relative_to(ROOT.parent)} ({len(sql.splitlines()):,} lines)")


if __name__ == "__main__":
    main()
