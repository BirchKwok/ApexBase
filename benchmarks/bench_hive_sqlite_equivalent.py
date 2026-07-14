"""Portable SQLite-equivalent benchmark for the three Hive user workloads.

The original SQL files in benchmarks/sql are Hive dialect workloads. This
benchmark keeps the same synthetic tables as bench_hive_complex_vs_duckdb.py
and runs portable SQL equivalents that SQLite, DuckDB, and ApexBase can all
execute. The three queries intentionally scale in complexity with the source
files: user 360 aggregation, complex lifecycle/trade attribution, and the most
complex multi-CTE/window/tagging workload.

Usage:
    python benchmarks/bench_hive_sqlite_equivalent.py --rows 100000 500000 1000000
"""

from __future__ import annotations

import argparse
import os
import sqlite3
import statistics
import tempfile
import time
from pathlib import Path

BENCH_THREADS = int(os.environ.get("APEX_HIVE_BENCH_THREADS", "1"))
if BENCH_THREADS < 1:
    raise ValueError("APEX_HIVE_BENCH_THREADS must be at least 1")
os.environ.setdefault("RAYON_NUM_THREADS", str(BENCH_THREADS))

import duckdb
from apexbase import ApexClient

from bench_hive_complex_vs_duckdb import DATABASE_TABLES, load_apex_from_duckdb, prepare_duckdb


ROOT = Path(__file__).resolve().parent
SQLITE_SQL_ROOT = ROOT / "sql" / "sqlite"


SQLITE_TABLES = {
    f"{database}.{table}": f"{database}__{table}"
    for database, table in DATABASE_TABLES
}


PORTABLE_SQL: dict[str, str] = {
    "hive_user_360.sql": """
WITH behavior_30d AS (
    SELECT
        user_id,
        COUNT(*) AS event_cnt_30d,
        COUNT(DISTINCT session_id) AS session_cnt_30d,
        COUNT(DISTINCT device_id) AS device_cnt_30d,
        COUNT(DISTINCT ip) AS ip_cnt_30d,
        SUM(CASE WHEN event_type = 'page_view' THEN 1 ELSE 0 END) AS pv_cnt_30d,
        SUM(CASE WHEN event_type = 'search' THEN 1 ELSE 0 END) AS search_cnt_30d,
        SUM(CASE WHEN event_type = 'sku_click' THEN 1 ELSE 0 END) AS click_cnt_30d,
        SUM(CASE WHEN event_type = 'add_cart' THEN 1 ELSE 0 END) AS add_cart_cnt_30d,
        SUM(CASE WHEN event_type = 'pay_success' THEN 1 ELSE 0 END) AS pay_success_cnt_30d,
        MAX(event_time) AS last_event_time
    FROM dwd_behavior.dwd_app_event_log_di
    WHERE dt = '{biz_date}'
    GROUP BY user_id
),
order_30d AS (
    SELECT
        user_id,
        COUNT(DISTINCT order_id) AS order_cnt_30d,
        SUM(pay_amount) AS pay_amount_30d,
        AVG(pay_amount) AS avg_pay_amount_30d
    FROM dwd_trade.dwd_order_item_df
    WHERE dt = '{biz_date}' AND is_test_order = 0
    GROUP BY user_id
)
SELECT
    u.user_id,
    u.member_level,
    u.province,
    COALESCE(b.event_cnt_30d, 0) AS event_cnt_30d,
    COALESCE(b.session_cnt_30d, 0) AS session_cnt_30d,
    COALESCE(b.click_cnt_30d, 0) AS click_cnt_30d,
    COALESCE(o.order_cnt_30d, 0) AS order_cnt_30d,
    ROUND(COALESCE(o.pay_amount_30d, 0), 2) AS pay_amount_30d,
    ROUND(
        COALESCE(b.pay_success_cnt_30d, 0) * 1.0
        / (COALESCE(b.click_cnt_30d, 0) + CASE WHEN COALESCE(b.click_cnt_30d, 0) = 0 THEN 1 ELSE 0 END),
        6
    ) AS click_pay_cvr_30d,
    CASE
        WHEN COALESCE(o.pay_amount_30d, 0) >= 1000 THEN 'HIGH_VALUE'
        WHEN COALESCE(b.event_cnt_30d, 0) > 0 THEN 'ACTIVE'
        ELSE 'SILENT'
    END AS user_segment
FROM ods_user.user_profile_df u
LEFT JOIN behavior_30d b ON u.user_id = b.user_id
LEFT JOIN order_30d o ON u.user_id = o.user_id
WHERE u.dt = '{biz_date}' AND u.country_code = 'CN'
ORDER BY u.user_id
""",
    "hive_user_complex.sql": """
WITH user_base AS (
    SELECT
        user_id,
        member_level,
        province,
        city,
        register_channel,
        CASE
            WHEN register_channel IN ('DOUYIN', 'EMAIL') THEN 'ONLINE'
            WHEN register_channel = 'STORE' THEN 'OFFLINE'
            ELSE 'OTHER'
        END AS register_channel_group
    FROM ods_user.ods_user_profile_df
    WHERE dt = '{biz_date}' AND country_code = 'CN' AND COALESCE(is_test_user, 0) = 0
),
event_agg AS (
    SELECT
        user_id,
        COUNT(*) AS event_cnt,
        COUNT(DISTINCT session_id) AS session_cnt,
        COUNT(DISTINCT channel) AS channel_cnt,
        SUM(CASE WHEN event_type = 'search' THEN 1 ELSE 0 END) AS search_cnt,
        SUM(CASE WHEN event_type IN ('sku_click', 'spu_click') THEN 1 ELSE 0 END) AS product_click_cnt,
        SUM(CASE WHEN event_type = 'coupon_use' THEN 1 ELSE 0 END) AS coupon_use_cnt,
        SUM(CASE WHEN event_type = 'pay_success' THEN 1 ELSE 0 END) AS pay_success_cnt
    FROM dwd_behavior.dwd_app_event_log_di
    WHERE dt = '{biz_date}'
    GROUP BY user_id
),
trade_agg AS (
    SELECT
        user_id,
        COUNT(DISTINCT order_id) AS order_cnt,
        SUM(quantity) AS item_qty,
        SUM(pay_amount) AS pay_amount,
        SUM(pay_amount - cost_amount) AS gross_margin
    FROM dwd_trade.dwd_order_item_df
    WHERE dt = '{biz_date}' AND is_test_order = 0
    GROUP BY user_id
),
ranked AS (
    SELECT
        ub.user_id,
        ub.member_level,
        ub.province,
        ub.city,
        ub.register_channel_group,
        COALESCE(e.event_cnt, 0) AS event_cnt,
        COALESCE(e.session_cnt, 0) AS session_cnt,
        COALESCE(e.search_cnt, 0) AS search_cnt,
        COALESCE(e.product_click_cnt, 0) AS product_click_cnt,
        COALESCE(t.order_cnt, 0) AS order_cnt,
        ROUND(COALESCE(t.pay_amount, 0), 2) AS pay_amount,
        ROUND(COALESCE(t.gross_margin, 0), 2) AS gross_margin,
        ROW_NUMBER() OVER (
            PARTITION BY ub.province
            ORDER BY ub.user_id
        ) AS province_value_rank
    FROM user_base ub
    LEFT JOIN event_agg e ON ub.user_id = e.user_id
    LEFT JOIN trade_agg t ON ub.user_id = t.user_id
)
SELECT *
FROM ranked
WHERE province_value_rank <= 200
ORDER BY province, province_value_rank, user_id
""",
    "hive_user_most_complex.sql": """
WITH event_features AS (
    SELECT
        e.user_id,
        COUNT(*) AS event_cnt,
        COUNT(*) AS session_cnt,
        COUNT(*) AS device_cnt,
        COUNT(*) AS ip_cnt,
        SUM(CASE WHEN e.event_type = 'live_enter' THEN 1 ELSE 0 END) AS live_enter_cnt,
        SUM(CASE WHEN e.event_type = 'store_visit' THEN 1 ELSE 0 END) AS store_visit_cnt,
        SUM(CASE WHEN e.event_type = 'coupon_use' THEN 1 ELSE 0 END) AS coupon_use_cnt,
        SUM(CASE WHEN e.event_type = 'search' THEN 1 ELSE 0 END) AS search_cnt,
        SUM(CASE WHEN e.event_type = 'add_cart' THEN 1 ELSE 0 END) AS add_cart_cnt,
        SUM(CASE WHEN e.event_type = 'pay_success' THEN 1 ELSE 0 END) AS pay_success_cnt
    FROM dwd_behavior.dwd_app_event_log_di e
    WHERE e.dt = '{biz_date}'
    GROUP BY e.user_id
),
trade_features AS (
    SELECT
        o.user_id,
        COUNT(*) AS paid_order_cnt,
        SUM(o.pay_amount) AS pay_amount,
        SUM(o.discount_amount + o.coupon_amount) AS discount_amount,
        SUM(o.pay_amount - o.cost_amount) AS gross_margin
    FROM dwd_trade.dwd_order_item_df o
    WHERE o.dt = '{biz_date}' AND o.is_test_order = 0
    GROUP BY o.user_id
),
refund_features AS (
    SELECT
        user_id,
        COUNT(DISTINCT refund_id) AS refund_cnt,
        SUM(refund_amount) AS refund_amount
    FROM dwd_trade.dwd_refund_order_df
    WHERE dt = '{biz_date}'
    GROUP BY user_id
),
marketing_features AS (
    SELECT user_id, COUNT(DISTINCT campaign_id) AS campaign_touch_cnt
    FROM ads_marketing.ads_user_campaign_touch_di
    WHERE dt = '{biz_date}'
    GROUP BY user_id
),
wide AS (
    SELECT
        u.user_id,
        u.member_level,
        u.province,
        u.city,
        u.is_black_user,
        COALESCE(e.event_cnt, 0) AS event_cnt,
        COALESCE(e.session_cnt, 0) AS session_cnt,
        COALESCE(e.device_cnt, 0) AS device_cnt,
        COALESCE(e.ip_cnt, 0) AS ip_cnt,
        COALESCE(e.live_enter_cnt, 0) AS live_enter_cnt,
        COALESCE(e.store_visit_cnt, 0) AS store_visit_cnt,
        COALESCE(e.coupon_use_cnt, 0) AS coupon_use_cnt,
        COALESCE(e.search_cnt, 0) AS search_cnt,
        COALESCE(e.add_cart_cnt, 0) AS add_cart_cnt,
        COALESCE(e.pay_success_cnt, 0) AS pay_success_cnt,
        COALESCE(t.paid_order_cnt, 0) AS paid_order_cnt,
        ROUND(COALESCE(t.pay_amount, 0), 2) AS pay_amount,
        ROUND(COALESCE(t.gross_margin, 0), 2) AS gross_margin,
        ROUND(COALESCE(r.refund_amount, 0), 2) AS refund_amount,
        COALESCE(r.refund_cnt, 0) AS refund_cnt,
        COALESCE(m.campaign_touch_cnt, 0) AS campaign_touch_cnt,
        CASE
            WHEN COALESCE(t.pay_amount, 0) >= 5000 THEN 'SUPER_VALUE'
            WHEN COALESCE(t.pay_amount, 0) >= 1000 THEN 'HIGH_VALUE'
            WHEN COALESCE(e.event_cnt, 0) > 0 THEN 'ACTIVE'
            ELSE 'SILENT'
        END AS value_segment,
        CASE
            WHEN COALESCE(u.is_black_user, 0) = 1 OR COALESCE(r.refund_cnt, 0) >= 3 THEN 'RISK_HIGH'
            WHEN COALESCE(e.device_cnt, 0) >= 3 OR COALESCE(e.ip_cnt, 0) >= 5 THEN 'RISK_MEDIUM'
            ELSE 'RISK_LOW'
        END AS risk_segment
    FROM ods_user.ods_user_profile_df u
    LEFT JOIN event_features e ON u.user_id = e.user_id
    LEFT JOIN trade_features t ON u.user_id = t.user_id
    LEFT JOIN refund_features r ON u.user_id = r.user_id
    LEFT JOIN marketing_features m ON u.user_id = m.user_id
    WHERE u.dt = '{biz_date}' AND u.country_code = 'CN' AND COALESCE(u.is_test_user, 0) = 0
),
scored AS (
    SELECT
        *,
        ROUND(
            pay_amount * 0.6
            + event_cnt * 0.5
            + campaign_touch_cnt * 5
            + live_enter_cnt * 3
            - refund_amount * 0.3,
            4
        ) AS operation_score,
        ROW_NUMBER() OVER (
            PARTITION BY value_segment
            ORDER BY pay_amount DESC, event_cnt DESC, user_id
        ) AS segment_rank
    FROM wide
)
SELECT
    user_id,
    member_level,
    province,
    city,
    is_black_user,
    event_cnt,
    session_cnt,
    device_cnt,
    ip_cnt,
    live_enter_cnt,
    store_visit_cnt,
    coupon_use_cnt,
    search_cnt,
    add_cart_cnt,
    pay_success_cnt,
    paid_order_cnt,
    pay_amount,
    gross_margin,
    refund_amount,
    refund_cnt,
    campaign_touch_cnt,
    value_segment,
    risk_segment,
    operation_score
FROM scored
WHERE segment_rank <= 300
ORDER BY value_segment, segment_rank, user_id
""",
    "hive_user_syntax_torture.sql": """
WITH user_base AS (
    SELECT
        user_id,
        member_level,
        province,
        city,
        is_black_user,
        CASE
            WHEN register_channel IN ('DOUYIN', 'EMAIL') THEN 'ONLINE'
            WHEN register_channel = 'STORE' THEN 'OFFLINE'
            ELSE 'OTHER'
        END AS register_channel_group
    FROM ods_user.ods_user_profile_df
    WHERE dt = '{biz_date}' AND country_code = 'CN' AND COALESCE(is_test_user, 0) = 0
),
event_core AS (
    SELECT
        user_id,
        COUNT(*) AS event_cnt,
        COUNT(DISTINCT session_id) AS session_cnt,
        COUNT(DISTINCT device_id) AS device_cnt,
        COUNT(DISTINCT ip) AS ip_cnt,
        SUM(CASE WHEN event_type = 'search' THEN 1 ELSE 0 END) AS search_cnt,
        SUM(CASE WHEN event_type = 'sku_click' THEN 1 ELSE 0 END) AS click_cnt,
        SUM(CASE WHEN event_type = 'add_cart' THEN 1 ELSE 0 END) AS add_cart_cnt,
        SUM(CASE WHEN event_type = 'pay_success' THEN 1 ELSE 0 END) AS pay_success_cnt,
        MAX(event_ts) AS max_event_ts,
        MIN(event_ts) AS min_event_ts
    FROM dwd_behavior.dwd_app_event_log_di
    WHERE dt = '{biz_date}'
    GROUP BY user_id
),
event_session_depth AS (
    SELECT user_id, MAX(session_depth) AS max_session_depth
    FROM (
        SELECT user_id, session_id, COUNT(*) AS session_depth
        FROM dwd_behavior.dwd_app_event_log_di
        WHERE dt = '{biz_date}'
        GROUP BY user_id, session_id
    ) sessions
    GROUP BY user_id
),
event_agg AS (
    SELECT
        e.*,
        COALESCE(d.max_session_depth, 0) AS max_session_depth,
        COALESCE(e.max_event_ts - e.min_event_ts, 0) AS activity_span_sec,
        CASE WHEN e.event_cnt > 1 THEN 1.0 ELSE 0.0 END AS max_journey_percent_rank
    FROM event_core e
    LEFT JOIN event_session_depth d ON e.user_id = d.user_id
),
order_sequence AS (
    SELECT
        user_id,
        order_id,
        sku_id,
        quantity,
        pay_amount,
        cost_amount,
        pay_time,
        ROW_NUMBER() OVER (
            PARTITION BY user_id ORDER BY pay_time DESC
        ) AS latest_order_rank,
        DENSE_RANK() OVER (
            PARTITION BY user_id ORDER BY quantity DESC
        ) AS quantity_dense_rank,
        NTILE(4) OVER (
            PARTITION BY user_id ORDER BY quantity
        ) AS quantity_quartile,
        LAG(quantity, 1) OVER (
            PARTITION BY user_id ORDER BY pay_time
        ) AS previous_quantity
    FROM dwd_trade.dwd_order_item_df
    WHERE dt = '{biz_date}' AND is_test_order = 0
),
trade_agg AS (
    SELECT
        user_id,
        COUNT(DISTINCT order_id) AS paid_order_cnt,
        COUNT(DISTINCT sku_id) AS paid_sku_cnt,
        SUM(quantity) AS paid_item_qty,
        ROUND(SUM(pay_amount), 2) AS pay_amount,
        ROUND(SUM(pay_amount - cost_amount), 2) AS gross_margin,
        MAX(quantity_quartile) AS max_quantity_quartile,
        MAX(quantity_dense_rank) AS max_quantity_dense_rank,
        MAX(ABS(quantity - COALESCE(previous_quantity, quantity))) AS max_quantity_change,
        MAX(CASE WHEN latest_order_rank = 1 THEN order_id ELSE NULL END) AS latest_order_id
    FROM order_sequence
    GROUP BY user_id
),
refund_agg AS (
    SELECT
        user_id,
        COUNT(DISTINCT refund_id) AS refund_cnt,
        ROUND(SUM(refund_amount), 2) AS refund_amount,
        SUM(CASE WHEN COALESCE(is_abnormal_refund, 0) = 1 THEN 1 ELSE 0 END) AS abnormal_refund_cnt
    FROM dwd_trade.dwd_refund_order_df
    WHERE dt = '{biz_date}'
    GROUP BY user_id
),
marketing_agg AS (
    SELECT
        user_id,
        COUNT(DISTINCT touch_id) AS marketing_touch_cnt,
        COUNT(DISTINCT campaign_id) AS campaign_cnt,
        COUNT(DISTINCT touch_channel) AS marketing_channel_cnt
    FROM ads_marketing.ads_user_campaign_touch_di
    WHERE dt = '{biz_date}'
    GROUP BY user_id
),
service_agg AS (
    SELECT
        user_id,
        COUNT(DISTINCT ticket_id) AS ticket_cnt,
        SUM(CASE WHEN COALESCE(is_escalated, 0) = 1 THEN 1 ELSE 0 END) AS escalated_ticket_cnt,
        ROUND(AVG(satisfaction_score), 2) AS avg_satisfaction_score
    FROM ods_service.ods_customer_ticket_df
    WHERE dt = '{biz_date}'
    GROUP BY user_id
),
event_trade_full AS (
    SELECT
        COALESCE(e.user_id, t.user_id) AS user_id,
        COALESCE(e.event_cnt, 0) AS event_cnt,
        COALESCE(t.paid_order_cnt, 0) AS paid_order_cnt,
        COALESCE(t.pay_amount, 0) AS pay_amount
    FROM event_agg e
    FULL OUTER JOIN trade_agg t ON e.user_id = t.user_id
),
scored AS (
    SELECT
        u.user_id,
        u.member_level,
        u.province,
        u.city,
        u.register_channel_group,
        COALESCE(et.event_cnt, 0) AS event_cnt,
        COALESCE(e.session_cnt, 0) AS session_cnt,
        COALESCE(e.device_cnt, 0) AS device_cnt,
        COALESCE(e.ip_cnt, 0) AS ip_cnt,
        COALESCE(e.max_session_depth, 0) AS max_session_depth,
        COALESCE(e.activity_span_sec, 0) AS activity_span_sec,
        COALESCE(e.max_journey_percent_rank, 0) AS max_journey_percent_rank,
        COALESCE(et.paid_order_cnt, 0) AS paid_order_cnt,
        COALESCE(t.paid_item_qty, 0) AS paid_item_qty,
        ROUND(COALESCE(et.pay_amount, 0), 2) AS pay_amount,
        ROUND(COALESCE(t.gross_margin, 0), 2) AS gross_margin,
        ROUND(COALESCE(r.refund_amount, 0), 2) AS refund_amount,
        COALESCE(r.refund_cnt, 0) AS refund_cnt,
        COALESCE(r.abnormal_refund_cnt, 0) AS abnormal_refund_cnt,
        COALESCE(m.marketing_touch_cnt, 0) AS marketing_touch_cnt,
        COALESCE(m.campaign_cnt, 0) AS campaign_cnt,
        COALESCE(s.ticket_cnt, 0) AS ticket_cnt,
        COALESCE(s.escalated_ticket_cnt, 0) AS escalated_ticket_cnt,
        CASE WHEN e.user_id IS NOT NULL THEN 1 ELSE 0 END AS is_active_user,
        CASE WHEN t.user_id IS NULL THEN 1 ELSE 0 END AS is_no_order_user,
        ROUND(
            COALESCE(et.pay_amount, 0) * 0.05
            + COALESCE(et.event_cnt, 0) * 0.20
            + COALESCE(m.marketing_touch_cnt, 0) * 2.0
            - COALESCE(r.refund_amount, 0) * 0.30,
            4
        ) AS operation_score,
        COALESCE(u.is_black_user, 0) * 80
        + COALESCE(r.abnormal_refund_cnt, 0) * 10
        + CASE WHEN COALESCE(e.device_cnt, 0) >= 3 THEN 10 ELSE 0 END
        + CASE WHEN COALESCE(e.ip_cnt, 0) >= 5 THEN 10 ELSE 0 END
        + CASE WHEN COALESCE(s.escalated_ticket_cnt, 0) > 0 THEN 10 ELSE 0 END AS raw_risk_score
    FROM user_base u
    LEFT JOIN event_trade_full et ON u.user_id = et.user_id
    LEFT JOIN event_agg e ON u.user_id = e.user_id
    LEFT JOIN trade_agg t ON u.user_id = t.user_id
    LEFT JOIN refund_agg r ON u.user_id = r.user_id
    LEFT JOIN marketing_agg m ON u.user_id = m.user_id
    LEFT JOIN service_agg s ON u.user_id = s.user_id
),
scored_clamped AS (
    SELECT
        *,
        CASE
            WHEN raw_risk_score < 0 THEN 0
            WHEN raw_risk_score > 100 THEN 100
            ELSE raw_risk_score
        END AS risk_score
    FROM scored
),
ranked AS (
    SELECT
        *,
        ROW_NUMBER() OVER (
            PARTITION BY province ORDER BY operation_score DESC, user_id
        ) AS province_operation_rank,
        DENSE_RANK() OVER (
            PARTITION BY member_level ORDER BY pay_amount DESC
        ) AS member_value_dense_rank,
        NTILE(10) OVER (
            ORDER BY operation_score DESC, user_id
        ) AS global_score_decile,
        PERCENT_RANK() OVER (
            ORDER BY operation_score
        ) AS global_score_percent_rank,
        CUME_DIST() OVER (
            ORDER BY risk_score
        ) AS global_risk_cume_dist
    FROM scored_clamped
)
SELECT
    user_id,
    member_level,
    province,
    city,
    register_channel_group,
    event_cnt,
    session_cnt,
    device_cnt,
    ip_cnt,
    max_session_depth,
    activity_span_sec,
    max_journey_percent_rank,
    paid_order_cnt,
    paid_item_qty,
    pay_amount,
    gross_margin,
    refund_amount,
    refund_cnt,
    abnormal_refund_cnt,
    marketing_touch_cnt,
    campaign_cnt,
    ticket_cnt,
    escalated_ticket_cnt,
    is_active_user,
    is_no_order_user,
    operation_score,
    risk_score,
    province_operation_rank,
    member_value_dense_rank,
    global_score_decile,
    global_score_percent_rank,
    CASE
        WHEN risk_score >= 80 THEN 'BLOCK'
        WHEN risk_score >= 60 THEN 'MANUAL_REVIEW'
        WHEN pay_amount >= 5000 THEN 'SUPER_VALUE'
        WHEN pay_amount >= 1000 THEN 'HIGH_VALUE'
        WHEN is_active_user = 1 AND is_no_order_user = 1 THEN 'ACTIVE_NO_ORDER'
        WHEN event_cnt > 0 THEN 'ACTIVE'
        ELSE 'SILENT'
    END AS user_segment
FROM ranked
WHERE province_operation_rank <= 300
ORDER BY province, province_operation_rank, user_id
""",
}


def sqlite_sql(sql: str) -> str:
    for original, replacement in SQLITE_TABLES.items():
        sql = sql.replace(original, replacement)
    return sql


def checked_sqlite_sql(workload: str, template: str) -> str:
    """Load the checked-in SQLite-native workload and reject stale output."""
    native_path = SQLITE_SQL_ROOT / workload
    native = native_path.read_text(encoding="utf-8")
    expected = sqlite_sql(template)
    if native != expected:
        raise RuntimeError(
            f"stale SQLite SQL: {native_path}; run "
            "python benchmarks/generate_hive_native_sql.py"
        )
    return native


def apex_rows(result) -> list[tuple]:
    rows = result.to_dict()
    if not rows:
        return []
    keys = list(rows[0].keys())
    return [tuple(row.get(key) for key in keys) for row in rows]


def timed(call, iterations: int) -> tuple[list[float], list[tuple]]:
    times = []
    last_rows: list[tuple] = []
    for _ in range(iterations):
        start = time.perf_counter()
        last_rows = call()
        times.append(time.perf_counter() - start)
    return times, last_rows


def load_sqlite_from_duckdb(con: duckdb.DuckDBPyConnection, db_path: Path) -> sqlite3.Connection:
    sqlite = sqlite3.connect(str(db_path))
    sqlite.execute("PRAGMA journal_mode=OFF")
    sqlite.execute("PRAGMA synchronous=OFF")
    sqlite.execute("PRAGMA temp_store=MEMORY")
    sqlite.execute("PRAGMA cache_size=-200000")
    for database, table in DATABASE_TABLES:
        source = f"{database}.{table}"
        target = SQLITE_TABLES[source]
        df = con.execute(f"SELECT * FROM {source}").fetchdf()
        df.to_sql(target, sqlite, if_exists="replace", index=False)
    sqlite.commit()
    return sqlite


def normalize(rows: list[tuple]) -> list[tuple]:
    normalized = []
    for row in rows:
        values = []
        for value in row:
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                values.append(round(float(value), 4))
            elif hasattr(value, "__float__") and value.__class__.__module__ == "decimal":
                values.append(round(float(value), 4))
            else:
                values.append(value)
        normalized.append(tuple(values))
    return sorted(normalized, key=lambda row: tuple("" if value is None else str(value) for value in row))


def run_one_size(rows: int, args: argparse.Namespace) -> list[dict[str, object]]:
    temp = tempfile.TemporaryDirectory(prefix="apex_sqlite_equiv_bench_")
    root = Path(temp.name)
    duck = duckdb.connect(str(root / "duckdb.db"))
    duck.execute(f"PRAGMA threads={BENCH_THREADS}")

    setup_start = time.perf_counter()
    prepare_duckdb(duck, rows, args.biz_date, args.users)
    load_apex_from_duckdb(duck, root / "apex", chunk_rows=args.chunk_rows)
    sqlite = load_sqlite_from_duckdb(duck, root / "sqlite.db")
    setup_seconds = time.perf_counter() - setup_start

    client = ApexClient(str(root / "apex"))
    results: list[dict[str, object]] = []
    try:
        for workload, template in PORTABLE_SQL.items():
            sql = template.format(biz_date=args.biz_date)
            duck_times, duck_result = timed(lambda: duck.execute(sql).fetchall(), args.iterations)
            sqlite_query = checked_sqlite_sql(workload, template).format(biz_date=args.biz_date)
            sqlite_times, sqlite_result = timed(lambda: sqlite.execute(sqlite_query).fetchall(), args.iterations)
            apex_times, apex_result = timed(lambda: apex_rows(client.execute(sql)), args.iterations)
            if normalize(apex_result) != normalize(duck_result):
                raise AssertionError(f"ApexBase/DuckDB result mismatch: {workload}, rows={rows}")
            if normalize(sqlite_result) != normalize(duck_result):
                raise AssertionError(f"SQLite/DuckDB result mismatch: {workload}, rows={rows}")
            results.append({
                "rows": rows,
                "workload": workload,
                "apex": statistics.mean(apex_times),
                "duckdb": statistics.mean(duck_times),
                "sqlite": statistics.mean(sqlite_times),
                "result_rows": len(duck_result),
                "setup": setup_seconds,
            })
    finally:
        client.close()
        sqlite.close()
        duck.close()
        if args.keep_data:
            print(f"Data kept at: {root}")
            temp._finalizer.detach()
        else:
            temp.cleanup()
    return results


def print_table(results: list[dict[str, object]]) -> None:
    print(
        "| Rows | SQL workload | ApexBase | DuckDB | SQLite equivalent | "
        "Apex/DuckDB | Apex/SQLite | Result rows |"
    )
    print("|---:|---|---:|---:|---:|---:|---:|---:|")
    for row in results:
        apex = float(row["apex"])
        duck = float(row["duckdb"])
        sqlite = float(row["sqlite"])
        print(
            f"| {int(row['rows']):,} | `{row['workload']}` | {apex:.3f}s | "
            f"{duck:.3f}s | {sqlite:.3f}s | {apex / duck:.2f}x | "
            f"{apex / sqlite:.2f}x | {int(row['result_rows']):,} |"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", nargs="+", type=int, default=[100_000, 500_000, 1_000_000])
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--users", type=int, default=None)
    parser.add_argument("--biz-date", default="2026-06-20")
    parser.add_argument("--chunk-rows", type=int, default=1_000_000)
    parser.add_argument("--keep-data", action="store_true")
    args = parser.parse_args()

    all_results: list[dict[str, object]] = []
    for row_count in args.rows:
        all_results.extend(run_one_size(row_count, args))

    print(f"Threads: {BENCH_THREADS}; setup/load excluded from query timings")
    print_table(all_results)


if __name__ == "__main__":
    main()
