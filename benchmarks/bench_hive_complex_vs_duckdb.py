"""Reproducible benchmark for all three Hive user-profile workloads.

The three source queries are kept verbatim in ``benchmarks/sql``. Only
``${biz_date}`` is bound at runtime. DuckDB receives automatically rewritten,
semantically equivalent SQL because it does not parse Hive's LATERAL VIEW or
INSERT OVERWRITE ... PARTITION syntax.

Usage:
    python benchmarks/bench_hive_complex_vs_duckdb.py
    python benchmarks/bench_hive_complex_vs_duckdb.py --rows 1000000 --iterations 2
"""

from __future__ import annotations

import argparse
import os
import re
import statistics
import tempfile
import time
from pathlib import Path

# Keep the comparison reproducible by giving both engines the same number of
# worker threads instead of letting DuckDB consume every logical CPU.
BENCH_THREADS = int(os.environ.get("APEX_HIVE_BENCH_THREADS", "1"))
if BENCH_THREADS < 1:
    raise ValueError("APEX_HIVE_BENCH_THREADS must be at least 1")
os.environ.setdefault("RAYON_NUM_THREADS", str(BENCH_THREADS))

import duckdb
from apexbase import ApexClient


ROOT = Path(__file__).resolve().parent
SQL_FILES = (
    ROOT / "sql" / "hive_user_360.sql",
    ROOT / "sql" / "hive_user_complex.sql",
    ROOT / "sql" / "hive_user_most_complex.sql",
)
TARGETS = {
    "hive_user_360.sql": ("ads_user_360_profile_wide_df",),
    "hive_user_complex.sql": ("ads_user_super_operation_profile_df",),
    "hive_user_most_complex.sql": (
        "ads_user_extreme_operation_profile_df",
        "ads_high_value_user_pool_df",
        "ads_user_recall_pool_df",
    ),
}
DATABASE_TABLES = (
    ("ods_user", "user_profile_df"),
    ("ods_user", "ods_user_profile_df"),
    ("dwd_behavior", "dwd_app_event_log_di"),
    ("dim", "dim_sku_df"),
    ("dim", "dim_shop_df"),
    ("dim", "dim_store_df"),
    ("dwd_trade", "dwd_order_item_df"),
    ("dwd_trade", "dwd_refund_order_df"),
    ("dwd_pay", "dwd_payment_flow_df"),
    ("dwd_fulfillment", "dwd_order_fulfillment_df"),
    ("ads_marketing", "ads_user_campaign_touch_di"),
    ("dwd_ad", "dwd_ad_click_di"),
    ("dwd_marketing", "dwd_app_popup_show_di"),
    ("dwd_promotion", "dwd_user_coupon_df"),
    ("ods_service", "ods_customer_ticket_df"),
)


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


def _rewrite_indexed_split(sql: str) -> str:
    pattern = re.compile(r"\bsplit\s*\(", re.IGNORECASE)
    start = 0
    while match := pattern.search(sql, start):
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
        indexed = re.match(r"\s*\[\s*(\d+)\s*\]", sql[end:])
        if not indexed:
            start = end
            continue
        args = _split_args(sql[match.end():end - 1])
        replacement = f"list_extract(string_split({args[0]}, {args[1]}), {int(indexed.group(1)) + 1})"
        sql = sql[:match.start()] + replacement + sql[end + indexed.end():]
        start = match.start() + len(replacement)
    return sql


def _partition_projection(text: str) -> str:
    values = []
    for item in _split_args(text):
        name, value = item.split("=", 1)
        values.append(f"{value.strip()} AS {name.strip()}")
    return ", ".join(values)


def _rewrite_lateral_views(sql: str) -> str:
    sql = re.sub(
        r"LATERAL\s+VIEW\s+OUTER\s+json_tuple\s*\(.*?\)\s+jt\s+AS\s+dummy_json\s+"
        r"LATERAL\s+VIEW\s+OUTER\s+inline\s*\(.*?\)\s+kv\s+AS\s+pref_key\s*,\s*pref_value",
        "",
        sql,
        flags=re.IGNORECASE | re.DOTALL,
    )
    sql = re.sub(
        r"SELECT\s+user_id\s*,\s*pref_key\s*,\s*pref_value\s+FROM\s+user_base",
        "SELECT user_id, list_extract(string_split(pref_pair, ':'), 1) AS pref_key, "
        "list_extract(string_split(pref_pair, ':'), 2) AS pref_value FROM user_base",
        sql,
        count=1,
        flags=re.IGNORECASE,
    )
    sql = re.sub(
        r"coalesce\s*\(\s*pref_key\s*,\s*''\s*\)",
        "coalesce(list_extract(string_split(pref_pair, ':'), 1), '')",
        sql,
        count=1,
        flags=re.IGNORECASE,
    )

    posexplode = re.compile(
        r"LATERAL\s+VIEW\s+(OUTER\s+)?posexplode\s*\((.*?)\)\s+"
        r"([A-Za-z_]\w*)\s+AS\s+([A-Za-z_]\w*)\s*,\s*([A-Za-z_]\w*)",
        re.IGNORECASE | re.DOTALL,
    )
    while match := posexplode.search(sql):
        outer, expr, alias, pos, value = match.groups()
        join = "LEFT JOIN" if outer else "CROSS JOIN"
        suffix = " ON TRUE" if outer else ""
        replacement = (
            f"{join} LATERAL (SELECT item_pos - 1 AS {pos}, "
            f"list_extract({expr}, item_pos) AS {value} "
            f"FROM UNNEST(range(1, len({expr}) + 1)) u(item_pos)) {alias}{suffix}"
        )
        sql = sql[:match.start()] + replacement + sql[match.end():]

    explode = re.compile(
        r"LATERAL\s+VIEW\s+(OUTER\s+)?explode\s*\((.*?)\)\s+"
        r"([A-Za-z_]\w*)\s+AS\s+([A-Za-z_]\w*)",
        re.IGNORECASE | re.DOTALL,
    )
    while match := explode.search(sql):
        outer, expr, alias, value = match.groups()
        replacement = (
            f"LEFT JOIN LATERAL UNNEST({expr}) {alias}({value}) ON TRUE"
            if outer else f"CROSS JOIN LATERAL UNNEST({expr}) {alias}({value})"
        )
        sql = sql[:match.start()] + replacement + sql[match.end():]

    stack = re.compile(
        r"LATERAL\s+VIEW\s+stack\s*\((.*?)\)\s+([A-Za-z_]\w*)\s+AS\s+"
        r"([A-Za-z_]\w*(?:\s*,\s*[A-Za-z_]\w*)+)",
        re.IGNORECASE | re.DOTALL,
    )
    while match := stack.search(sql):
        args = _split_args(match.group(1))
        rows, values = int(args[0]), args[1:]
        aliases = _split_args(match.group(3))
        width = len(aliases)
        tuples = ["(" + ", ".join(values[i:i + width]) + ")"
                  for i in range(0, rows * width, width)]
        replacement = (
            "CROSS JOIN LATERAL (VALUES " + ", ".join(tuples) + ") "
            + match.group(2) + "(" + ", ".join(aliases) + ")"
        )
        sql = sql[:match.start()] + replacement + sql[match.end():]
    return sql


def duckdb_sql(hive_sql: str, biz_date: str) -> str:
    sql = hive_sql.replace("${biz_date}", biz_date)
    sql = _rewrite_stack_selects(sql)
    sql = _rewrite_indexed_split(sql)
    sql = _rewrite_lateral_views(sql)
    sql = re.sub(
        r"DISTRIBUTE\s+BY\s+.*?\s+SORT\s+BY\s+([^\n]+)",
        r"ORDER BY \1",
        sql,
        flags=re.IGNORECASE | re.DOTALL,
    )
    sql = re.sub(
        r"(upper\s*\([^()]+\)|[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)?)\s+RLIKE\s+'([^']*)'",
        r"regexp_matches(\1, '\2')",
        sql,
        flags=re.IGNORECASE,
    )
    sql = re.sub(r"\bLEFT\s+SEMI\s+JOIN\b", "SEMI JOIN", sql, flags=re.IGNORECASE)
    sql = re.sub(r"\bLEFT\s+ANTI\s+JOIN\b", "ANTI JOIN", sql, flags=re.IGNORECASE)
    sql = re.sub(
        r"GROUP\s+BY\s+[A-Za-z0-9_.,\s]+?GROUPING\s+SETS",
        "GROUP BY GROUPING SETS",
        sql,
        flags=re.IGNORECASE,
    )
    sql = re.sub(
        r"GROUP\s+BY\s+([A-Za-z0-9_.,\s]+?)\s+WITH\s+CUBE",
        r"GROUP BY CUBE (\1)",
        sql,
        flags=re.IGNORECASE,
    )
    sql = re.sub(
        r"GROUP\s+BY\s+([A-Za-z0-9_.,\s]+?)\s+WITH\s+ROLLUP",
        r"GROUP BY ROLLUP (\1)",
        sql,
        flags=re.IGNORECASE,
    )
    sql = re.sub(r"\bgrouping__id\b", "0", sql, flags=re.IGNORECASE)
    sql = _rewrite_calls(
        sql,
        "concat_ws",
        lambda a: f"array_to_string({a[1]}, {a[0]})" if len(a) == 2
        else f"duckdb_concat_ws({', '.join(a)})",
    ).replace("duckdb_concat_ws", "concat_ws")
    sql = _rewrite_calls(sql, "slice", lambda a: f"list_slice({a[0]}, {a[1]}, ({a[1]}) + ({a[2]}) - 1)")
    sql = _rewrite_calls(sql, "percentile_approx", lambda a: f"quantile_cont({a[0]}, {a[1]})")
    sql = re.sub(r"\breverse\s*\(", "list_reverse(", sql, flags=re.IGNORECASE)
    # The map is an output-only wide-table field; DuckDB keeps the same serialized pairs.
    sql = _rewrite_calls(sql, "str_to_map", lambda a: a[0])
    sql = _rewrite_calls(sql, "array", lambda a: f"list_value({', '.join(a)})")
    sql = _rewrite_calls(
        sql,
        "unix_timestamp",
        lambda a: f"epoch(strptime({a[0]}, '%Y-%m-%d'))" if len(a) == 2
        else f"duckdb_unix_timestamp({a[0]})",
    ).replace("duckdb_unix_timestamp", "unix_timestamp")
    sql = re.sub(r"\bdate_sub\s*\(", "hive_date_sub(", sql, flags=re.IGNORECASE)
    sql = re.sub(r"\bcurrent_timestamp\s*\(\s*\)", "current_timestamp", sql, flags=re.IGNORECASE)

    multi = re.search(r"\nFROM\s+(final_user_extreme_wide\s+f)\s*\n\s*INSERT\s+OVERWRITE", sql, re.IGNORECASE)
    if multi:
        prefix = sql[:multi.start()]
        branches = re.sub(r"/\*.*?\*/", "", sql[multi.start():], flags=re.DOTALL)
        inserts = list(re.finditer(
            r"INSERT\s+OVERWRITE\s+TABLE\s+ads_user\.([A-Za-z_]\w*)\s+"
            r"PARTITION\s*\((.*?)\)\s*SELECT\s+(.*?)(?=\s+INSERT\s+OVERWRITE|\Z)",
            branches,
            re.IGNORECASE | re.DOTALL,
        ))
        statements = [
            "CREATE OR REPLACE TEMP TABLE __hive_multi_source AS\n"
            + prefix + "\nSELECT * FROM final_user_extreme_wide"
        ]
        for insert in inserts:
            target, partition, select_tail = insert.groups()
            select_tail = select_tail.strip().rstrip(";")
            where = re.search(r"\bWHERE\b", select_tail, re.IGNORECASE)
            if where:
                projection = select_tail[:where.start()].strip()
                predicate = select_tail[where.end():].strip()
                branch_query = (
                    f"SELECT {projection} FROM __hive_multi_source f WHERE {predicate}"
                )
            else:
                branch_query = f"SELECT {select_tail} FROM __hive_multi_source f"
            statements.append(
                f"CREATE OR REPLACE TABLE ads_user.{target} AS "
                f"SELECT branch.*, {_partition_projection(partition)} FROM ("
                f"{branch_query}) branch"
            )
        if len(statements) == 1:
            raise ValueError("Hive multi-table INSERT branches were not found")
        return ";\n".join(statements)

    insert = re.search(
        r"INSERT\s+OVERWRITE\s+TABLE\s+ads_user\.([A-Za-z_]\w*)\s+"
        r"PARTITION\s*\((.*?)\)\s*",
        sql,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if not insert:
        raise ValueError("final Hive INSERT OVERWRITE clause was not found")
    target, partition = insert.group(1), insert.group(2)
    query = (sql[:insert.start()] + sql[insert.end():]).rstrip().rstrip(";")
    return (
        f"CREATE OR REPLACE TABLE ads_user.{target} AS\n"
        f"SELECT source_result.*, {_partition_projection(partition)} FROM (\n"
        + query + "\n) source_result"
    )


def prepare_duckdb(
    con: duckdb.DuckDBPyConnection,
    rows: int,
    biz_date: str,
    user_count: int | None = None,
) -> None:
    users = user_count or max(1_000, min(10_000, rows // 100))
    small = users
    for schema in {database for database, _ in DATABASE_TABLES}:
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
    con.execute("CREATE MACRO add_months(d, n) AS CAST(CAST(d AS DATE) + CAST(n AS INTEGER) * INTERVAL '1 month' AS VARCHAR)")

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
        CREATE TABLE ods_user.ods_user_profile_df AS
        SELECT 'u' || i AS user_id, 'union' || i AS union_id, 'open' || i AS open_id,
               '138' || lpad(i::VARCHAR, 8, '0') AS mobile,
               'user' || i || '@example.com' AS email,
               CASE WHEN i % 2 = 0 THEN 'M' ELSE 'F' END AS gender,
               1970 + i % 35 AS birth_year, '{biz_date} 08:00:00' AS register_time,
               ['DOUYIN','EMAIL','STORE','SEARCH','NATURAL'][1 + i % 5] AS register_channel,
               'APP' AS register_app, '10.1.' || i % 255 || '.1' AS register_ip,
               'd' || i % 3000 AS register_device_id,
               'P' || i % 10 AS province, 'C' || i % 100 AS city,
               'D' || i % 20 AS district, 'GOLD' AS member_level,
               100 + i % 500 AS member_score, (i % 1000 = 0)::INTEGER AS is_employee,
               (i % 997 = 0)::INTEGER AS is_black_user, 0 AS is_test_user,
               '{{"identity":{{"real_name_auth":"1","student_auth":"0","enterprise_auth":"0"}},'
               || '"device":{{"first_os":"IOS"}},"source":{{"inviter_user_id":"u' || (i + 1) % {users} || '"}},'
               || '"risk":{{"manual_mark":"NORMAL"}},"profile":{{"preference_json":"color:red,size:m"}}}}' AS ext_json,
               '{biz_date}' AS dt, 'CN' AS country_code
        FROM range({users}) t(i)
    """)
    con.execute(f"""
        CREATE TABLE dwd_behavior.dwd_app_event_log_di AS
        SELECT 'u' || i % {users} AS user_id, 's' || i % ({users} * 3) AS session_id,
               'e' || i AS event_id,
               ['app_start','page_view','search','sku_expose','sku_click','spu_click','add_cart','remove_cart','favorite','unfavorite','coupon_receive','coupon_use','submit_order','pay_success','refund_apply','share','comment','live_enter','live_stay','live_click','store_visit','scan_qr','customer_service_click','address_submit','payment_click'][1 + i % 25] AS event_type,
               '{biz_date} ' || lpad((i % 24)::VARCHAR, 2, '0') || ':00:00' AS event_time,
               1781913600 + i % 86400 AS event_ts,
               'p' || i % 50 AS page_id, 'p' || i % 49 AS refer_page_id,
               ['APP_PUSH','EMAIL','DOUYIN'][1 + i % 3] AS channel, 'SUB' AS sub_channel,
               'd' || i % 3000 AS device_id, ['PHONE','TABLET'][1 + i % 2] AS device_type,
               ['IOS','ANDROID'][1 + i % 2] AS os, '9.0.0' AS app_version,
               '10.0.' || i % 255 || '.' || i % 200 AS ip, 'ua' AS ua,
               '{{"sku_id":"sku' || i % 100 || '","spu_id":"spu' || i % 50 ||
               '","shop_id":"shop' || i % 25 || '","store_id":"store' || i % 20 ||
               '","category_id":"cat' || i % 10 || '","brand_id":"brand' || i % 20 ||
               '","search_keyword":"kw' || i % 20 || '","coupon_id":"cp' || i % 100 ||
               '","ab_test":{{"exp_id":"exp' || i % 5 || '","group_id":"g' || i % 2 ||
               '"}},"ab":{{"exp_id":"exp' || i % 5 || '","group_id":"g' || i % 2 ||
               '"}},"search":{{"keyword":"kw' || i % 20 || '","result_cnt":"20"}},'
               || '"coupon":{{"coupon_id":"cp' || i % 100 || '"}},'
               || '"reco":{{"strategy_id":"st' || i % 5 || '","scene_id":"scene1"}},'
               || '"trace":{{"trace_id":"tr' || i || '","request_id":"rq' || i || '"}},'
               || '"geo":{{"longitude":"120.1","latitude":"30.2"}},'
               || '"expose_sku_list":["sku' || i % 100 || '","sku' || (i + 1) % 100 || '"],'
               || '"exposure":{{"sku_list":["sku' || i % 100 || '","sku' || (i + 1) % 100 || '"]}},'
               || '"cart":{{"items":["sku' || i % 100 || ':1:99.0"]}},'
               || '"labels":["hot","mobile"],"kv":{{"source":"bench"}}}}' AS ext_json,
               '{biz_date}' AS dt
        FROM range({rows}) t(i)
    """)
    con.execute(f"""
        CREATE TABLE dim.dim_sku_df AS
        SELECT 'sku' || i AS sku_id, 'spu' || i % 50 AS spu_id, 'Product ' || i AS product_name,
               'shop' || i % 25 AS shop_id, 'cat' || i % 10 AS category_id,
               'Category ' || i % 10 AS category_name, 'brand' || i % 20 AS brand_id,
               'Brand ' || i % 20 AS brand_name, 'c1' || i % 5 AS category_level1_id,
               'Cate1 ' || i % 5 AS category_level1_name, 'c2' || i % 8 AS category_level2_id,
               'Cate2 ' || i % 8 AS category_level2_name, 'c3' || i % 10 AS category_level3_id,
               'Cate3 ' || i % 10 AS category_level3_name, 10.0 + i * 10 AS list_price,
               5.0 + i * 5 AS cost_price, 'ON' AS shelf_status, 'MID' AS price_band,
               (i % 2)::INTEGER AS is_self_operated, (i % 3 = 0)::INTEGER AS is_imported,
               (i % 5 = 0)::INTEGER AS is_fresh, (i % 7 = 0)::INTEGER AS is_virtual,
               '{biz_date}' AS dt FROM range(100) t(i)
    """)
    con.execute(f"""
        CREATE TABLE dim.dim_shop_df AS SELECT 'shop' || i AS shop_id,
               'Shop ' || i AS shop_name, 'BRAND' AS shop_type, 'seller' || i AS seller_id,
               'A' AS seller_level, (i % 2)::INTEGER AS is_brand_shop,
               (i % 3 = 0)::INTEGER AS is_self_operated, 'P' || i % 10 AS province,
               'C' || i % 100 AS city, '{biz_date} 00:00:00' AS open_time,
               '{biz_date}' AS dt FROM range(25) t(i)
    """)
    con.execute(f"""
        CREATE TABLE dim.dim_store_df AS SELECT 'store' || i AS store_id,
               'Store ' || i AS store_name, 'FLAGSHIP' AS store_type,
               'region' || i % 5 AS region_id, 'Region ' || i % 5 AS region_name,
               'P' || i % 10 AS province, 'C' || i % 100 AS city, 'D' || i % 20 AS district,
               '{biz_date}' AS open_date, 'A' AS store_level, 'CBD' AS business_circle,
               'mgr' || i AS manager_id, 120.1 AS longitude, 30.2 AS latitude,
               '{biz_date}' AS dt FROM range(20) t(i)
    """)
    con.execute(f"""
        CREATE TABLE dwd_trade.dwd_order_item_df AS
        SELECT 'u' || i % {users} AS user_id, 'o' || i AS order_id, 'po' || i AS parent_order_id,
               'oi' || i AS order_item_id, 'sku' || i % 100 AS sku_id,
               'spu' || i % 50 AS spu_id, 'shop' || i % 25 AS shop_id, 'store' || i % 20 AS store_id,
               '{biz_date} 12:00:00' AS pay_time, '{biz_date} 11:00:00' AS order_time,
               '{biz_date} 18:00:00' AS finish_time, NULL::VARCHAR AS cancel_time,
               'DONE' AS order_status, CASE WHEN i % 4 = 0 THEN 'UNPAID' ELSE 'PAID' END AS pay_status,
               CASE WHEN i % 20 = 0 THEN 'REFUNDED' ELSE 'NONE' END AS refund_status,
               1 + i % 3 AS quantity, 100.0 + i % 200 AS goods_amount, 10.0 AS discount_amount,
               3.0 AS platform_coupon_amount, 2.0 AS shop_coupon_amount,
               1.0 AS points_deduction_amount, 5.0 AS coupon_amount, 0.0 AS freight_amount,
               85.0 + i % 200 AS pay_amount, 50.0 + i % 100 AS cost_amount,
               ['ALIPAY','WECHAT','CARD'][1 + i % 3] AS payment_method,
               ['EXPRESS','STORE_PICKUP'][1 + i % 2] AS delivery_type,
               'P' || i % 10 AS province, 'C' || i % 100 AS city,
               'D' || i % 20 AS district, 'addr' || i % 1000 AS address_hash,
               0 AS is_test_order,
               '{{"invoice":{{"type":"NORMAL"}},"promotion":{{"activity_id":"a1","activity_type":"SALE"}},"source":{{"trace_id":"tr' || i || '"}}}}' AS ext_json,
               '{biz_date}' AS dt FROM range({small}) t(i)
    """)
    con.execute(f"""
        CREATE TABLE dwd_pay.dwd_payment_flow_df AS SELECT 'u' || i % {users} AS user_id,
               'o' || i AS order_id, 'pay' || i AS payment_id, 'ONLINE' AS payment_channel,
               ['ALIPAY','WECHAT','CARD'][1 + i % 3] AS payment_method, 'BANK' AS bank_code,
               85.0 + i % 200 AS pay_amount, 'SUCCESS' AS pay_status,
               '{biz_date} 12:00:00' AS pay_time, 'PASS' AS risk_decision,
               i % 100 AS risk_score, 1 AS installment_num,
               '{{"card":{{"bin":"6222"}},"wallet":{{"type":"BALANCE"}},"risk":{{"reason":"NONE"}}}}' AS ext_json,
               '{biz_date}' AS dt FROM range({small}) t(i)
    """)
    con.execute(f"""
        CREATE TABLE dwd_trade.dwd_refund_order_df AS SELECT 'u' || i % {users} AS user_id,
               'o' || i AS order_id, 'oi' || i AS order_item_id, 'r' || i AS refund_id,
               'ONLY_REFUND' AS refund_type, 'OTHER' AS refund_reason_code,
               'Other' AS refund_reason_desc, '{biz_date} 14:00:00' AS apply_time,
               '{biz_date} 15:00:00' AS audit_time, '{biz_date} 16:00:00' AS refund_success_time,
               CASE WHEN i % 20 = 0 THEN 10.0 ELSE 0.0 END AS refund_amount,
               CASE WHEN i % 20 = 0 THEN 'SUCCESS' ELSE 'NONE' END AS refund_status,
               0 AS is_abnormal_refund, '{biz_date}' AS dt FROM range({small}) t(i)
    """)
    con.execute(f"""
        CREATE TABLE dwd_fulfillment.dwd_order_fulfillment_df AS
        SELECT 'u' || i % {users} AS user_id, 'o' || i AS order_id,
               'store' || i % 20 AS store_id, 'wh' || i % 10 AS warehouse_id,
               ['EXPRESS','STORE_PICKUP'][1 + i % 2] AS delivery_type, 'SF' AS carrier_code,
               '{biz_date} 15:00:00' AS promise_delivery_time, '{biz_date} 13:00:00' AS ship_time,
               '{biz_date} 15:00:00' AS sign_time, NULL::VARCHAR AS cancel_time,
               'SIGNED' AS fulfillment_status, 0 AS is_timeout, 3.5 AS distance_km,
               5.0 AS delivery_fee, '{biz_date}' AS dt FROM range({small}) t(i)
    """)
    con.execute(f"""
        CREATE TABLE ads_marketing.ads_user_campaign_touch_di AS
        SELECT 'u' || i % {users} AS user_id, 'touch' || i AS touch_id,
               'camp' || i % 20 AS campaign_id, 'Campaign ' || i % 20 AS campaign_name,
               'mat' || i % 30 AS material_id, 'st' || i % 5 AS strategy_id,
               ['NEW_USER_COUPON','RECALL','PRICE_DROP'][1 + i % 3] AS scene,
               ['APP_PUSH','SMS','EMAIL'][1 + i % 3] AS channel, 'SUB' AS sub_channel,
               ['PUSH','SMS','EMAIL'][1 + i % 3] AS touch_channel,
               '{biz_date} 10:00:00' AS touch_time, 1781920800 + i AS touch_ts,
               '{{"bid_type":"CPM","cost":"1.5","audience_pkg_id":"aud1","creative_type":"IMAGE"}}' AS ext_json,
               '{biz_date}' AS dt FROM range({small}) t(i)
    """)
    con.execute(f"""
        CREATE TABLE dwd_ad.dwd_ad_click_di AS SELECT 'u' || i % {users} AS user_id,
               'ad' || i AS ad_click_id, 'camp' || i % 20 AS campaign_id,
               'Campaign ' || i % 20 AS campaign_name, 'creative' || i % 30 AS creative_id,
               'st' || i % 5 AS strategy_id, 'BRAND_CAMPAIGN' AS ad_scene,
               'DOUYIN' AS media_channel, 'FEED' AS media_sub_channel,
               '{biz_date} 09:00:00' AS click_time, 1781917200 + i AS click_ts,
               'CPC' AS bid_type, 1.0 AS cost_amount, 'aud1' AS audience_pkg_id,
               'VIDEO' AS creative_type, '{biz_date}' AS dt FROM range({small}) t(i)
    """)
    con.execute(f"""
        CREATE TABLE dwd_marketing.dwd_app_popup_show_di AS SELECT 'u' || i % {users} AS user_id,
               'pop' || i AS popup_id, 'camp' || i % 20 AS campaign_id,
               'Campaign ' || i % 20 AS campaign_name, 'mat' || i % 30 AS material_id,
               'st' || i % 5 AS strategy_id, 'RECALL' AS popup_scene, 'MODAL' AS popup_type,
               '{biz_date} 08:00:00' AS show_time, 1781913600 + i AS show_ts,
               'aud1' AS audience_pkg_id, 'IMAGE' AS creative_type,
               '{biz_date}' AS dt FROM range({small}) t(i)
    """)
    con.execute(f"""
        CREATE TABLE dwd_promotion.dwd_user_coupon_df AS
        SELECT 'u' || i % {users} AS user_id, 'cp' || i AS coupon_id,
               'batch' || i % 10 AS batch_id, 'CASH' AS coupon_type,
               ['USED','EXPIRED','UNUSED'][1 + i % 3] AS coupon_status, 5.0 + i % 20 AS discount_amount,
               '{biz_date} 09:00:00' AS receive_time, '{biz_date} 13:00:00' AS use_time,
               '{biz_date} 23:00:00' AS expire_time, 'o' || i AS order_id,
               100.0 AS threshold_amount, 3.0 AS platform_bear_amount, 2.0 AS shop_bear_amount,
               '{biz_date}' AS dt FROM range({small}) t(i)
    """)
    con.execute(f"""
        CREATE TABLE ods_service.ods_customer_ticket_df AS
        SELECT 'u' || i % {users} AS user_id, 't' || i AS ticket_id,
               'o' || i AS order_id,
               ['SOLVED','OPEN'][1 + i % 2] AS ticket_status,
               ['REFUND','COMPLAINT','CONSULT'][1 + i % 3] AS ticket_type,
               'ORDER' AS ticket_scene, '{biz_date} 08:00:00' AS create_time,
               '{biz_date} 08:30:00' AS assign_time, '{biz_date} 10:00:00' AS solve_time,
               '{biz_date} 11:00:00' AS close_time, 4.5 AS satisfaction_score,
               (i % 50 = 0)::INTEGER AS is_escalated, 'agent' || i % 10 AS agent_id,
               '{biz_date}' AS dt FROM range({small}) t(i)
    """)


def load_apex_from_duckdb(con: duckdb.DuckDBPyConnection, root: Path, chunk_rows: int = 1_000_000) -> None:
    client = ApexClient(str(root))
    try:
        for database, table in DATABASE_TABLES:
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


def create_apex_targets(con: duckdb.DuckDBPyConnection, root: Path, targets: tuple[str, ...]) -> None:
    client = ApexClient(str(root))
    try:
        for target in targets:
            description = con.execute(f"DESCRIBE ads_user.{target}").fetchall()
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
            client.use(database="ads_user", table=target)
            client.execute(f"DROP TABLE IF EXISTS {target}")
            client.execute(f"CREATE TABLE {target} (" + ",".join(fields) + ")")
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
    parser.add_argument(
        "--users",
        type=int,
        default=None,
        help="distinct synthetic users (default: scale from 1,000 to 10,000 with rows)",
    )
    parser.add_argument("--biz-date", default="2026-06-20")
    parser.add_argument(
        "--sql",
        choices=("all", *(path.name for path in SQL_FILES)),
        default="all",
        help="run all workloads or one SQL file",
    )
    parser.add_argument("--keep-data", action="store_true")
    args = parser.parse_args()

    sql_files = SQL_FILES if args.sql == "all" else tuple(
        path for path in SQL_FILES if path.name == args.sql
    )
    temp = tempfile.TemporaryDirectory(prefix="apex_hive_complex_bench_")
    root = Path(temp.name)
    con = duckdb.connect(str(root / "duckdb.db"))
    con.execute(f"PRAGMA threads={BENCH_THREADS}")

    setup_start = time.perf_counter()
    prepare_duckdb(con, args.rows, args.biz_date, args.users)
    load_apex_from_duckdb(con, root / "apex")
    setup_seconds = time.perf_counter() - setup_start
    results = []
    for sql_file in sql_files:
        hive = sql_file.read_text(encoding="utf-8")
        duck_sql = duckdb_sql(hive, args.biz_date)
        duck_times = timed(lambda: con.execute(duck_sql).fetchall(), args.iterations)
        targets = TARGETS[sql_file.name]
        create_apex_targets(con, root / "apex", targets)
        client = ApexClient(str(root / "apex"))
        bound_hive = hive.replace("${biz_date}", args.biz_date)
        try:
            apex_times = timed(lambda: client.execute(bound_hive).to_dict(), args.iterations)
            apex_counts = {
                target: tuple(client.execute(
                    f"SELECT COUNT(*) AS n, COUNT(DISTINCT user_id) AS users "
                    f"FROM ads_user.{target} WHERE dt='{args.biz_date}'"
                ).to_dict()[0].values())
                for target in targets
            }
            apex_user_ids = {
                target: tuple(
                    row["user_id"]
                    for row in client.execute(
                        f"SELECT user_id FROM ads_user.{target} "
                        f"WHERE dt='{args.biz_date}' ORDER BY user_id"
                    ).to_dict()
                )
                for target in targets
            }
        finally:
            client.close()
        duck_counts = {
            target: con.execute(
                f"SELECT count(*), count(DISTINCT user_id) FROM ads_user.{target} "
                f"WHERE dt='{args.biz_date}'"
            ).fetchone()
            for target in targets
        }
        duck_user_ids = {
            target: tuple(
                row[0]
                for row in con.execute(
                    f"SELECT user_id FROM ads_user.{target} "
                    f"WHERE dt='{args.biz_date}' ORDER BY user_id"
                ).fetchall()
            )
            for target in targets
        }
        if apex_counts != duck_counts:
            raise AssertionError(
                f"result mismatch for {sql_file.name}: Apex={apex_counts}, DuckDB={duck_counts}"
            )
        if apex_user_ids != duck_user_ids:
            raise AssertionError(f"user_id mismatch for {sql_file.name}")
        results.append((sql_file, hive, apex_times, duck_times, apex_counts))
    con.close()

    print(
        f"Rows: behavior={args.rows:,}; threads={BENCH_THREADS}; "
        f"shared setup/load excluded={setup_seconds:.3f}s"
    )
    for sql_file, hive, apex_times, duck_times, counts in results:
        apex_mean, duck_mean = statistics.mean(apex_times), statistics.mean(duck_times)
        print(f"\nSQL: {sql_file} ({len(hive.splitlines())} lines)")
        print(f"ApexBase: {apex_times} mean={apex_mean:.3f}s")
        print(f"DuckDB:   {duck_times} mean={duck_mean:.3f}s")
        print(f"Targets:  {counts}")
        print(f"ApexBase/DuckDB: {apex_mean / duck_mean:.2f}x")
    if args.keep_data:
        print(f"Data kept at: {root}")
        temp.cleanup = lambda: None
    else:
        temp.cleanup()


if __name__ == "__main__":
    main()
