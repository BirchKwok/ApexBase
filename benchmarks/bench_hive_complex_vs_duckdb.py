"""Reproducible benchmark for the 975-line Hive user-360 query.

The source query is kept verbatim in ``benchmarks/sql/hive_user_360.sql``.
Only ``${biz_date}`` is bound at runtime. DuckDB receives an automatically
rewritten, semantically equivalent dialect because it does not parse Hive's
LATERAL VIEW or INSERT OVERWRITE ... PARTITION syntax.

Usage:
    python benchmarks/bench_hive_complex_vs_duckdb.py
    python benchmarks/bench_hive_complex_vs_duckdb.py --rows 1000000 --iterations 2
"""

from __future__ import annotations

import argparse
import re
import statistics
import tempfile
import time
from pathlib import Path

import duckdb
from apexbase import ApexClient


ROOT = Path(__file__).resolve().parent
SQL_FILE = ROOT / "sql" / "hive_user_360.sql"
DATABASE_TABLES = {
    "ods_user": "user_profile_df",
    "dwd_behavior": "dwd_app_event_log_di",
    "dim": "dim_sku_df",
    "dwd_trade": "dwd_order_item_df",
    "ads_marketing": "ads_user_campaign_touch_di",
    "dwd_promotion": "dwd_user_coupon_df",
    "ods_service": "ods_customer_ticket_df",
}


def _split_args(text: str) -> list[str]:
    args, start, depth, quote = [], 0, 0, None
    for i, char in enumerate(text):
        if quote:
            if char == quote and (i == 0 or text[i - 1] != "\\"):
                quote = None
        elif char in "'\"":
            quote = char
        elif char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
        elif char == "," and depth == 0:
            args.append(text[start:i].strip())
            start = i + 1
    args.append(text[start:].strip())
    return args


def _rewrite_calls(sql: str, name: str, replacement) -> str:
    pattern = re.compile(rf"\b{re.escape(name)}\s*\(", re.IGNORECASE)
    while match := pattern.search(sql):
        depth, quote, end = 1, None, match.end()
        while end < len(sql) and depth:
            char = sql[end]
            if quote:
                if char == quote and sql[end - 1] != "\\":
                    quote = None
            elif char in "'\"":
                quote = char
            elif char == "(":
                depth += 1
            elif char == ")":
                depth -= 1
            end += 1
        args = _split_args(sql[match.end():end - 1])
        sql = sql[:match.start()] + replacement(args) + sql[end:]
    return sql


def _rewrite_stack_selects(sql: str) -> str:
    pattern = re.compile(
        r"SELECT\s+stack\s*\((.*?)\)\s+AS\s*\((.*?)\)",
        re.IGNORECASE | re.DOTALL,
    )
    while match := pattern.search(sql):
        args = _split_args(match.group(1))
        rows, values = int(args[0]), args[1:]
        width = len(values) // rows
        tuples = ["(" + ", ".join(values[i:i + width]) + ")"
                  for i in range(0, len(values), width)]
        aliases = match.group(2)
        replacement = f"SELECT * FROM (VALUES {', '.join(tuples)}) AS stack_values({aliases})"
        sql = sql[:match.start()] + replacement + sql[match.end():]
    return sql


def duckdb_sql(hive_sql: str, biz_date: str) -> str:
    sql = hive_sql.replace("${biz_date}", biz_date)
    sql = _rewrite_stack_selects(sql)
    sql = re.sub(
        r"LATERAL\s+VIEW\s+OUTER\s+explode\s*\(rb\.expose_sku_arr\)\s+e\s+AS\s+expose_sku_id",
        "LEFT JOIN LATERAL UNNEST(rb.expose_sku_arr) e(expose_sku_id) ON TRUE",
        sql,
        flags=re.IGNORECASE,
    )
    stack_match = re.search(
        r"LATERAL\s+VIEW\s+stack\s*\((.*?)\)\s+s\s+AS\s+metric_name\s*,\s*metric_value",
        sql,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if stack_match:
        args = _split_args(stack_match.group(1))
        values = args[1:]
        tuples = [f"({values[i]}, {values[i + 1]})" for i in range(0, len(values), 2)]
        lateral = "CROSS JOIN LATERAL (VALUES " + ", ".join(tuples) + ") s(metric_name, metric_value)"
        sql = sql[:stack_match.start()] + lateral + sql[stack_match.end():]

    sql = re.sub(
        r"DISTRIBUTE\s+BY\s+user_id\s*,\s*session_id\s+SORT\s+BY\s+user_id\s*,\s*session_id\s*,\s*event_time",
        "ORDER BY user_id, session_id, event_time",
        sql,
        flags=re.IGNORECASE,
    )
    sql = _rewrite_calls(sql, "concat_ws", lambda a: f"array_to_string({a[1]}, {a[0]})")
    sql = _rewrite_calls(sql, "slice", lambda a: f"list_slice({a[0]}, {a[1]}, ({a[1]}) + ({a[2]}) - 1)")
    sql = _rewrite_calls(sql, "percentile_approx", lambda a: f"quantile_cont({a[0]}, {a[1]})")
    # The map is an output-only wide-table field; DuckDB keeps the same serialized pairs.
    sql = _rewrite_calls(sql, "str_to_map", lambda a: a[0])
    sql = re.sub(r"\bdate_sub\s*\(", "hive_date_sub(", sql, flags=re.IGNORECASE)
    sql = re.sub(r"\bcurrent_timestamp\s*\(\s*\)", "current_timestamp", sql, flags=re.IGNORECASE)

    insert = re.search(
        r"INSERT\s+OVERWRITE\s+TABLE\s+ads_user\.ads_user_360_profile_wide_df\s+"
        r"PARTITION\s*\(\s*dt\s*=\s*'[^']+'\s*\)\s*",
        sql,
        flags=re.IGNORECASE,
    )
    if not insert:
        raise ValueError("final Hive INSERT OVERWRITE clause was not found")
    return (
        "CREATE OR REPLACE TABLE ads_user.ads_user_360_profile_wide_df AS\n"
        + sql[:insert.start()]
        + sql[insert.end():].replace("SELECT\n    *\nFROM final_user_360", f"SELECT *, '{biz_date}' AS dt\nFROM final_user_360", 1)
    )


def prepare_duckdb(con: duckdb.DuckDBPyConnection, rows: int, biz_date: str) -> None:
    users = max(1_000, min(10_000, rows // 100))
    small = users
    for schema in DATABASE_TABLES:
        con.execute(f"CREATE SCHEMA IF NOT EXISTS {schema}")
    con.execute("CREATE SCHEMA IF NOT EXISTS ads_user")
    con.execute("CREATE MACRO hive_date_sub(d, n) AS CAST(CAST(d AS DATE) - CAST(n AS INTEGER) AS VARCHAR)")
    con.execute("CREATE MACRO months_between(a, b) AS date_diff('month', CAST(b AS DATE), CAST(a AS DATE))")
    con.execute("CREATE MACRO datediff(a, b) AS date_diff('day', CAST(b AS DATE), CAST(a AS DATE))")
    con.execute("CREATE MACRO unix_timestamp(a) AS epoch(CAST(a AS TIMESTAMP))")
    con.execute("CREATE MACRO get_json_object(a, b) AS json_extract_string(a, b)")
    con.execute("CREATE MACRO collect_set(a) AS list_distinct(list(a))")
    con.execute("CREATE MACRO collect_list(a) AS list(a)")
    con.execute("CREATE MACRO sort_array(a) AS list_sort(a)")
    con.execute("CREATE MACRO size(a) AS len(a)")

    con.execute(f"""
        CREATE TABLE ods_user.user_profile_df AS
        SELECT 'u' || i AS user_id, '138' || lpad(i::VARCHAR, 8, '0') AS mobile,
               'user' || i || '@example.com' AS email,
               CASE WHEN i % 2 = 0 THEN 'M' ELSE 'F' END AS gender,
               1970 + i % 35 AS birth_year, '{biz_date} 08:00:00' AS register_time,
               'APP' AS register_channel, 'P' || i % 10 AS province,
               'C' || i % 100 AS city, 'd' || i % 3000 AS device_id,
               '9.0.0' AS app_version, 'GOLD' AS member_level,
               CASE WHEN i % 997 = 0 THEN 1 ELSE 0 END AS is_black_user,
               '{biz_date}' AS dt, 'CN' AS country_code
        FROM range({users}) t(i)
    """)
    con.execute(f"""
        CREATE TABLE dwd_behavior.dwd_app_event_log_di AS
        SELECT 'u' || i % {users} AS user_id, 's' || i % ({users} * 3) AS session_id,
               'e' || i AS event_id,
               ['page_view','search','sku_expose','sku_click','add_cart','favorite','submit_order','pay_success','coupon_receive','coupon_use'][1 + i % 10] AS event_type,
               '{biz_date} ' || lpad((i % 24)::VARCHAR, 2, '0') || ':00:00' AS event_time,
               'p' || i % 50 AS page_id, 'p' || i % 49 AS refer_page_id,
               ['APP','H5','MINI'][1 + i % 3] AS channel, 'd' || i % 3000 AS device_id,
               '10.0.' || i % 255 || '.' || i % 200 AS ip, 'ua' AS ua,
               '{{"sku_id":"sku' || i % 100 || '","spu_id":"spu' || i % 50 ||
               '","category_id":"cat' || i % 10 || '","search_keyword":"kw' || i % 20 ||
               '","coupon_id":"cp' || i % 100 || '","ab_test":{{"exp_id":"exp' || i % 5 ||
               '","group_id":"g' || i % 2 || '"}},"expose_sku_list":["sku' || i % 100 ||
               '","sku' || (i + 1) % 100 || '"]}}' AS ext_json,
               '{biz_date}' AS dt
        FROM range({rows}) t(i)
    """)
    con.execute(f"""
        CREATE TABLE dim.dim_sku_df AS
        SELECT 'sku' || i AS sku_id, 'cat' || i % 10 AS category_id,
               'Category ' || i % 10 AS category_name, 'brand' || i % 20 AS brand_id,
               'Brand ' || i % 20 AS brand_name, 'MID' AS price_band,
               (i % 2)::INTEGER AS is_self_operated, (i % 3 = 0)::INTEGER AS is_imported,
               '{biz_date}' AS dt FROM range(100) t(i)
    """)
    con.execute(f"""
        CREATE TABLE dwd_trade.dwd_order_item_df AS
        SELECT 'u' || i % {users} AS user_id, 'o' || i AS order_id, 'po' || i AS parent_order_id,
               'sku' || i % 100 AS sku_id, 'spu' || i % 50 AS spu_id, 'shop' || i % 25 AS shop_id,
               '{biz_date} 12:00:00' AS pay_time, '{biz_date} 11:00:00' AS order_time,
               'DONE' AS order_status, CASE WHEN i % 4 = 0 THEN 'UNPAID' ELSE 'PAID' END AS pay_status,
               CASE WHEN i % 20 = 0 THEN 'REFUNDED' ELSE 'NONE' END AS refund_status,
               1 + i % 3 AS quantity, 100.0 + i % 200 AS goods_amount, 10.0 AS discount_amount,
               5.0 AS coupon_amount, 0.0 AS freight_amount, 85.0 + i % 200 AS pay_amount,
               ['ALIPAY','WECHAT','CARD'][1 + i % 3] AS payment_method,
               'P' || i % 10 AS province, 'C' || i % 100 AS city,
               0 AS is_test_order, '{biz_date}' AS dt FROM range({small}) t(i)
    """)
    con.execute(f"""
        CREATE TABLE ads_marketing.ads_user_campaign_touch_di AS
        SELECT 'u' || i % {users} AS user_id, 'camp' || i % 20 AS campaign_id,
               'Campaign ' || i % 20 AS campaign_name, ['PUSH','SMS','EMAIL'][1 + i % 3] AS touch_channel,
               '{biz_date} 10:00:00' AS touch_time, 'mat' || i % 30 AS material_id,
               'st' || i % 5 AS strategy_id,
               ['NEW_USER_COUPON','RECALL','PRICE_DROP'][1 + i % 3] AS scene,
               '{biz_date}' AS dt FROM range({small}) t(i)
    """)
    con.execute(f"""
        CREATE TABLE dwd_promotion.dwd_user_coupon_df AS
        SELECT 'u' || i % {users} AS user_id, 'cp' || i AS coupon_id,
               ['USED','EXPIRED','RECEIVED'][1 + i % 3] AS coupon_status, 5.0 + i % 20 AS discount_amount,
               '{biz_date} 09:00:00' AS receive_time, '{biz_date} 13:00:00' AS use_time,
               '{biz_date}' AS dt FROM range({small}) t(i)
    """)
    con.execute(f"""
        CREATE TABLE ods_service.ods_customer_ticket_df AS
        SELECT 'u' || i % {users} AS user_id, 't' || i AS ticket_id,
               ['SOLVED','OPEN'][1 + i % 2] AS ticket_status,
               ['REFUND','COMPLAINT','CONSULT'][1 + i % 3] AS ticket_type,
               '{biz_date} 08:00:00' AS create_time, '{biz_date} 10:00:00' AS solve_time,
               '{biz_date}' AS dt FROM range({small}) t(i)
    """)


def load_apex_from_duckdb(con: duckdb.DuckDBPyConnection, root: Path, chunk_rows: int = 1_000_000) -> None:
    client = ApexClient(str(root))
    try:
        for database, table in DATABASE_TABLES.items():
            client.use(database=database, table=table)
            count = con.execute(f"SELECT count(*) FROM {database}.{table}").fetchone()[0]
            for offset in range(0, count, chunk_rows):
                arrow = con.execute(
                    f"SELECT * FROM {database}.{table} LIMIT {chunk_rows} OFFSET {offset}"
                ).fetch_arrow_table()
                client.store(arrow)
            client.flush()
    finally:
        client.close()


def create_apex_target(con: duckdb.DuckDBPyConnection, root: Path) -> None:
    description = con.execute("DESCRIBE ads_user.ads_user_360_profile_wide_df").fetchall()
    fields = []
    for name, duck_type, *_ in description:
        upper = duck_type.upper()
        if any(token in upper for token in ("INT", "HUGEINT", "UBIGINT")):
            apex_type = "BIGINT"
        elif any(token in upper for token in ("DOUBLE", "FLOAT", "DECIMAL")):
            apex_type = "DOUBLE"
        elif upper == "BOOLEAN":
            apex_type = "BOOLEAN"
        else:
            apex_type = "TEXT"
        fields.append(f'"{name}" {apex_type}')
    client = ApexClient(str(root))
    try:
        client.use(database="ads_user", table="ads_user_360_profile_wide_df")
        client.execute("DROP TABLE IF EXISTS ads_user_360_profile_wide_df")
        client.execute("CREATE TABLE ads_user_360_profile_wide_df (" + ",".join(fields) + ")")
    finally:
        client.close()


def timed(call, iterations: int) -> list[float]:
    values = []
    for _ in range(iterations):
        start = time.perf_counter()
        call()
        values.append(time.perf_counter() - start)
    return values


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=1_000_000)
    parser.add_argument("--iterations", type=int, default=2)
    parser.add_argument("--biz-date", default="2026-06-20")
    parser.add_argument("--keep-data", action="store_true")
    args = parser.parse_args()

    hive = SQL_FILE.read_text(encoding="utf-8")
    duck_sql = duckdb_sql(hive, args.biz_date)
    temp = tempfile.TemporaryDirectory(prefix="apex_hive_complex_bench_")
    root = Path(temp.name)
    con = duckdb.connect(str(root / "duckdb.db"))

    setup_start = time.perf_counter()
    prepare_duckdb(con, args.rows, args.biz_date)
    setup_seconds = time.perf_counter() - setup_start
    duck_times = timed(lambda: con.execute(duck_sql).fetchall(), args.iterations)
    load_start = time.perf_counter()
    load_apex_from_duckdb(con, root / "apex")
    create_apex_target(con, root / "apex")
    setup_seconds += time.perf_counter() - load_start

    client = ApexClient(str(root / "apex"))
    bound_hive = hive.replace("${biz_date}", args.biz_date)
    try:
        apex_times = timed(lambda: client.execute(bound_hive).to_dict(), args.iterations)
        apex_rows = client.execute(
            f"SELECT COUNT(*) AS n FROM ads_user.ads_user_360_profile_wide_df WHERE dt='{args.biz_date}'"
        ).to_dict()[0]["n"]
    finally:
        client.close()
    duck_rows = con.execute("SELECT count(*) FROM ads_user.ads_user_360_profile_wide_df").fetchone()[0]
    con.close()

    apex_mean, duck_mean = statistics.mean(apex_times), statistics.mean(duck_times)
    print(f"SQL: {SQL_FILE} ({len(hive.splitlines())} lines)")
    print(f"Rows: behavior={args.rows:,}; setup/load excluded={setup_seconds:.3f}s")
    print(f"ApexBase: {apex_times} mean={apex_mean:.3f}s rows={apex_rows}")
    print(f"DuckDB:   {duck_times} mean={duck_mean:.3f}s rows={duck_rows}")
    print(f"ApexBase/DuckDB: {apex_mean / duck_mean:.2f}x")
    if args.keep_data:
        print(f"Data kept at: {root}")
        temp.cleanup = lambda: None
    else:
        temp.cleanup()


if __name__ == "__main__":
    main()
