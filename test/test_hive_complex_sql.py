"""Hive compatibility coverage for the complex analytics execution path."""

import tempfile

from apexbase import ApexStorage


def _rows(result):
    columns = result.get("columns_dict", {})
    names = list(columns)
    if not names:
        return []
    return [tuple(columns[name][i] for name in names) for i in range(len(columns[names[0]]))]


def test_hive_lateral_conditional_agg_and_insert_overwrite_partition():
    with tempfile.TemporaryDirectory(prefix="apex_hive_complex_") as path:
        db = ApexStorage(path)
        db.execute(
            "CREATE TABLE IF NOT EXISTS hive_events (user_id TEXT, event_type TEXT, items TEXT, "
            "amount DOUBLE, dt TEXT)"
        )
        db.execute(
            "CREATE TABLE IF NOT EXISTS hive_profile_out (user_id TEXT, event_cnt BIGINT, "
            "paid_amount DOUBLE, risk_code TEXT, dt TEXT)"
        )
        db.execute("TRUNCATE TABLE hive_events")
        db.execute("TRUNCATE TABLE hive_profile_out")
        db.execute(
            "INSERT INTO hive_events VALUES "
            "('u1','pay','s1,s2',25.0,'2026-06-20'),"
            "('u1','view','s3',0.0,'2026-06-19'),"
            "('u2','pay','s4',80.0,'2026-06-20')"
        )
        db.execute(
            "INSERT INTO hive_profile_out VALUES "
            "('stale',99,99.0,'OLD','2026-06-20'),"
            "('keep',1,1.0,'OLD','2026-06-19')"
        )

        sql = """
        WITH params AS (
            SELECT '2026-06-20' AS biz_date, date_sub('2026-06-20', 7) AS d7
        ),
        risk_map AS (
            SELECT stack(3,
                'LOW', 0, 20,
                'MEDIUM', 21, 50,
                'HIGH', 51, 100
            ) AS (risk_code, score_min, score_max)
        ),
        expanded AS (
            SELECT e.user_id, e.event_type, e.amount, sku_id, e.dt
            FROM hive_events e
            JOIN params p ON e.dt BETWEEN p.d7 AND p.biz_date
            LATERAL VIEW OUTER explode(split(e.items, ',')) x AS sku_id
        ),
        agg AS (
            SELECT
                user_id,
                count(1) AS event_cnt,
                sum(if(event_type = 'pay', amount, 0)) AS paid_amount,
                count(DISTINCT if(event_type = 'pay', sku_id, NULL)) AS paid_sku_cnt,
                concat_ws(',', sort_array(collect_set(event_type))) AS event_types
            FROM expanded
            GROUP BY user_id
        ),
        scored AS (
            SELECT a.user_id, a.event_cnt, a.paid_amount,
                   greatest(a.paid_amount, 0) AS risk_score
            FROM agg a
        ),
        final_rows AS (
            SELECT s.user_id, s.event_cnt, s.paid_amount, r.risk_code
            FROM scored s
            LEFT JOIN risk_map r
              ON s.risk_score BETWEEN r.score_min AND r.score_max
        )
        INSERT OVERWRITE TABLE hive_profile_out PARTITION (dt='2026-06-20')
        SELECT user_id, event_cnt, paid_amount, risk_code FROM final_rows
        """
        db.execute(sql)

        current = db.execute(
            "SELECT user_id, event_cnt, paid_amount, risk_code "
            "FROM hive_profile_out WHERE dt='2026-06-20' ORDER BY user_id"
        )
        assert _rows(current) == [
            ("u1", 3, 50.0, "MEDIUM"),
            ("u2", 1, 80.0, "HIGH"),
        ]
        kept = db.execute("SELECT user_id FROM hive_profile_out WHERE dt='2026-06-19'")
        assert kept["columns_dict"]["user_id"] == ["keep"]
        db.close()


def test_hive_lateral_stack_and_simple_case():
    with tempfile.TemporaryDirectory(prefix="apex_hive_stack_") as path:
        db = ApexStorage(path)
        db.execute("CREATE TABLE IF NOT EXISTS hive_metrics (user_id TEXT, orders BIGINT, amount DOUBLE)")
        db.execute("TRUNCATE TABLE hive_metrics")
        db.execute("INSERT INTO hive_metrics VALUES ('u1', 2, 25.5)")
        result = db.execute(
            "SELECT user_id, metric_name, metric_value, "
            "CASE metric_name WHEN 'orders' THEN 'COUNT' ELSE 'MONEY' END AS kind "
            "FROM hive_metrics LATERAL VIEW stack(2, "
            "'orders', cast(orders AS double), 'amount', amount) s "
            "AS metric_name, metric_value ORDER BY metric_name"
        )
        assert _rows(result) == [
            ("u1", "amount", 25.5, "MONEY"),
            ("u1", "orders", 2.0, "COUNT"),
        ]
        db.close()


def test_hive_fuses_multiple_json_paths_without_changing_nested_semantics():
    with tempfile.TemporaryDirectory(prefix="apex_hive_json_fusion_") as path:
        db = ApexStorage(path)
        db.execute("CREATE TABLE IF NOT EXISTS hive_json (payload TEXT)")
        db.execute("TRUNCATE TABLE hive_json")
        db.execute(
            "INSERT INTO hive_json VALUES "
            "('{\"id\":\"top\",\"outer\":{\"id\":\"nested\"},"
            "\"items\":[\"a\",\"b\"]}')"
        )
        result = db.execute(
            "WITH parsed AS (SELECT "
            "get_json_object(payload, '$.id') AS top_id, "
            "get_json_object(payload, '$.outer.id') AS nested_id, "
            "split(regexp_replace(regexp_replace("
            "get_json_object(payload, '$.items'), '\\\\[|\\\\]|\"', ''), "
            "'\\\\s+', ''), ',') AS items FROM hive_json) "
            "SELECT top_id, nested_id, item FROM parsed "
            "LATERAL VIEW explode(items) e AS item ORDER BY item"
        )
        assert _rows(result) == [
            ("top", "nested", "a"),
            ("top", "nested", "b"),
        ]
        db.close()
