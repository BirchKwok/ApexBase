"""Hive compatibility coverage for the complex analytics execution path."""

import tempfile

from apexbase import ApexClient, ApexStorage


def _rows(result):
    if hasattr(result, "to_dict"):
        records = result.to_dict()
        if isinstance(records, list):
            return [tuple(record.values()) for record in records]
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
            "coalesce(get_json_object(payload, '$.items'), '[]'), "
            "'\\\\[|\\\\]|\"', ''), "
            "'\\\\s+', ''), ',') AS items FROM hive_json) "
            "SELECT top_id, nested_id, item FROM parsed "
            "LATERAL VIEW explode(items) e AS item ORDER BY item"
        )
        assert _rows(result) == [
            ("top", "nested", "a"),
            ("top", "nested", "b"),
        ]
        db.close()


def test_empty_join_cte_keeps_projected_marker_schema():
    with tempfile.TemporaryDirectory(prefix="apex_hive_empty_cte_") as path:
        db = ApexStorage(path)
        db.execute("CREATE TABLE IF NOT EXISTS hive_left (user_id TEXT)")
        db.execute("CREATE TABLE IF NOT EXISTS hive_right (user_id TEXT, score BIGINT)")
        db.execute("TRUNCATE TABLE hive_left")
        db.execute("TRUNCATE TABLE hive_right")
        db.execute("INSERT INTO hive_left VALUES ('u1')")
        db.execute("INSERT INTO hive_right VALUES ('u1', 1)")
        result = db.execute(
            "WITH empty_candidate AS ("
            "SELECT l.user_id, 1 AS candidate_marker FROM hive_left l "
            "JOIN hive_right r ON l.user_id=r.user_id WHERE r.score > 10"
            ") SELECT l.user_id, coalesce(c.candidate_marker, 0) AS marker "
            "FROM hive_left l LEFT JOIN empty_candidate c ON l.user_id=c.user_id"
        )
        assert _rows(result) == [("u1", 0)]
        db.close()


def test_fused_conditional_counts_and_shared_percentiles():
    with tempfile.TemporaryDirectory(prefix="apex_hive_fused_agg_") as path:
        db = ApexStorage(path)
        db.execute("CREATE TABLE IF NOT EXISTS hive_fused_agg (user_id TEXT, kind TEXT, value BIGINT)")
        db.execute("TRUNCATE TABLE hive_fused_agg")
        db.execute(
            "INSERT INTO hive_fused_agg VALUES "
            "('u1','a',1),('u1','b',2),('u1','a',3),('u1','c',4)"
        )
        result = db.execute(
            "SELECT user_id, "
            "sum(if(kind='a',1,0)) AS a_cnt, "
            "sum(if(kind='b',1,0)) AS b_cnt, "
            "round(sum(if(kind='a',1,0)) / greatest(sum(if(kind='b',1,0)),1), 2) AS ratio, "
            "percentile_approx(value,0.25) AS p25, "
            "percentile_approx(value,0.75) AS p75 "
            "FROM hive_fused_agg GROUP BY user_id"
        )
        assert _rows(result) == [("u1", 2.0, 1.0, 2.0, 2.0, 3.0)]
        db.close()


def test_hive_advanced_row_generators_joins_and_multi_insert():
    with tempfile.TemporaryDirectory(prefix="apex_hive_advanced_") as path:
        db = ApexClient(path)
        db.create_table("hive_users_adv", {"user_id": "string", "channel": "string", "prefs": "string"})
        db.create_table("events", {"user_id": "string", "items": "string", "score": "int"})
        db.create_table("active_out", {"user_id": "string", "pos": "int", "item": "string", "dt": "string"})
        db.create_table("inactive_out", {"user_id": "string", "dt": "string"})
        db.execute(
            "INSERT INTO hive_users_adv VALUES "
            "('u1','douyin','color:red,size:m'),"
            "('u2','email','color:blue'),"
            "('u3','store','size:l')"
        )
        db.execute(
            "INSERT INTO events VALUES ('u1','a,b',2),('u2','c',0)"
        )

        rows = db.execute(
            "SELECT u.user_id, pos, item, pref_key, pref_value "
            "FROM hive_users_adv u JOIN events e "
            "ON u.user_id=e.user_id AND e.score > 0 "
            "LATERAL VIEW OUTER posexplode(split(e.items, ',')) px AS pos, item "
            "LATERAL VIEW OUTER explode(split(u.prefs, ',')) p AS pref_pair "
            "LATERAL VIEW OUTER inline(array(named_struct("
            "'pref_key', split(pref_pair, ':')[0], "
            "'pref_value', split(pref_pair, ':')[1]))) kv AS pref_key, pref_value "
            "WHERE upper(u.channel) RLIKE 'DOUYIN|EMAIL' ORDER BY pos, pref_key"
        )
        assert _rows(rows) == [
            ("u1", 0, "a", "color", "red"),
            ("u1", 0, "a", "size", "m"),
            ("u1", 1, "b", "color", "red"),
            ("u1", 1, "b", "size", "m"),
        ]
        assert _rows(db.execute(
            "SELECT u.user_id FROM hive_users_adv u LEFT SEMI JOIN events e "
            "ON u.user_id=e.user_id AND e.score > 0"
        )) == [("u1",)]

        db.execute(
            "WITH active AS ("
            "SELECT e.user_id, pos, item FROM events e "
            "LATERAL VIEW posexplode(split(e.items, ',')) p AS pos, item "
            "WHERE e.score > 0"
            ") FROM active a "
            "INSERT OVERWRITE TABLE active_out PARTITION (dt='2026-06-20') "
            "SELECT * WHERE a.user_id IS NOT NULL "
            "INSERT OVERWRITE TABLE inactive_out PARTITION (dt='2026-06-20') "
            "SELECT user_id WHERE pos = 0"
        )
        assert _rows(db.execute("SELECT user_id, pos, item FROM active_out ORDER BY pos")) == [
            ("u1", 0, "a"),
            ("u1", 1, "b"),
        ]
        assert _rows(db.execute("SELECT user_id FROM inactive_out")) == [("u1",)]
        assert _rows(db.execute(
            "SELECT u.user_id FROM hive_users_adv u LEFT ANTI JOIN events e ON u.user_id=e.user_id"
        )) == [("u3",)]
        db.close()


def test_window_peer_semantics_and_multiple_order_columns():
    with tempfile.TemporaryDirectory(prefix="apex_window_peers_") as path:
        db = ApexStorage(path)
        db.execute("CREATE TABLE IF NOT EXISTS window_peers (label TEXT, score BIGINT)")
        db.execute("TRUNCATE TABLE window_peers")
        db.execute(
            "INSERT INTO window_peers VALUES "
            "('b',10),('a',10),('d',20),('c',20)"
        )
        result = db.execute(
            "SELECT label, "
            "ROW_NUMBER() OVER (ORDER BY score, label) AS row_num, "
            "NTILE(2) OVER (ORDER BY score, label) AS bucket, "
            "CUME_DIST() OVER (ORDER BY score) AS cumulative_distribution "
            "FROM window_peers ORDER BY label"
        )
        assert sorted(_rows(result)) == [
            ("a", 1, 1, 0.5),
            ("b", 2, 1, 0.5),
            ("c", 3, 2, 1.0),
            ("d", 4, 2, 1.0),
        ]
        db.close()


def test_wide_insert_overwrite_coerces_to_target_schema():
    with tempfile.TemporaryDirectory(prefix="apex_wide_overwrite_types_") as path:
        db = ApexStorage(path)
        columns = [f"c{i}" for i in range(65)]
        schema = ",".join(f"{column} BIGINT" for column in columns)
        projection = ",".join(f"{i}.0 AS {column}" for i, column in enumerate(columns))
        db.execute(f"CREATE TABLE IF NOT EXISTS wide_target ({schema})")
        db.execute("TRUNCATE TABLE wide_target")
        db.execute(f"INSERT OVERWRITE TABLE wide_target SELECT {projection}")
        result = db.execute(f"SELECT {','.join(columns)} FROM wide_target")
        assert _rows(result) == [tuple(range(65))]
        db.close()


def test_sql_three_valued_boolean_short_circuit_semantics():
    with tempfile.TemporaryDirectory(prefix="apex_sql_three_valued_logic_") as path:
        db = ApexStorage(path)
        db.execute("CREATE TABLE IF NOT EXISTS truth_values (id BIGINT, a BIGINT, b BIGINT)")
        db.execute("TRUNCATE TABLE truth_values")
        db.execute("INSERT INTO truth_values VALUES (1,1,NULL),(2,0,NULL)")
        assert _rows(db.execute(
            "SELECT id FROM truth_values WHERE a=1 OR (a=0 AND b=1) ORDER BY id"
        )) == [(1,)]
        assert _rows(db.execute(
            "SELECT id FROM truth_values WHERE NOT (a=1 AND b=1) ORDER BY id"
        )) == [(2,)]
        assert _rows(db.execute(
            "SELECT id, CASE WHEN a=1 THEN 0 WHEN a=2 THEN 100 ELSE 2.5 END AS score "
            "FROM truth_values ORDER BY id"
        )) == [(1, 0.0), (2, 2.5)]
        db.close()


def test_two_string_group_count_preserves_aliases_and_null_groups():
    with tempfile.TemporaryDirectory(prefix="apex_two_string_group_") as path:
        db = ApexStorage(path)
        db.execute("CREATE TABLE IF NOT EXISTS sessions (user_id TEXT, session_id TEXT)")
        db.execute("TRUNCATE TABLE sessions")
        db.execute(
            "INSERT INTO sessions VALUES "
            "('u1','s1'),('u1','s1'),('u1','s2'),"
            "('u2',NULL),('u2',NULL),(NULL,'orphan')"
        )
        rows = _rows(db.execute(
            "WITH source AS (SELECT user_id, session_id FROM sessions) "
            "SELECT user_id AS uid, session_id AS sid, COUNT(*) AS session_depth "
            "FROM source GROUP BY user_id, session_id"
        ))
        assert sorted(rows, key=lambda row: (str(row[0]), str(row[1]))) == [
            (None, "orphan", 1),
            ("u1", "s1", 2),
            ("u1", "s2", 1),
            ("u2", None, 2),
        ]
        db.close()
