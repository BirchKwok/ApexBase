-- ApexBase Hive compatibility torture workload.
-- Business scenario: omnichannel customer journey, value, risk, and recall profile.

WITH
runtime_params AS (
    SELECT
        '${biz_date}' AS biz_date,
        date_sub('${biz_date}', 1) AS d1,
        date_sub('${biz_date}', 7) AS d7,
        date_sub('${biz_date}', 30) AS d30,
        date_sub('${biz_date}', 90) AS d90,
        add_months('${biz_date}', -12) AS d365,
        'CN' AS country_code,
        1000 AS high_value_threshold,
        0.35 AS refund_risk_threshold
),

channel_rule AS (
    SELECT stack(
        7,
        'APP_PUSH',      0.18, 'PRIVATE',
        'EMAIL',         0.08, 'PRIVATE',
        'DOUYIN',        0.35, 'PAID_MEDIA',
        'WECHAT',        0.22, 'PRIVATE',
        'OFFLINE_STORE', 0.40, 'OFFLINE',
        'NATURAL',       0.05, 'ORGANIC',
        'UNKNOWN',       0.01, 'UNKNOWN'
    ) AS (channel, channel_weight, channel_group)
),

risk_rule AS (
    SELECT stack(
        5,
        'R0', 0,  19,  'LOW',
        'R1', 20, 39,  'MEDIUM_LOW',
        'R2', 40, 59,  'MEDIUM',
        'R3', 60, 79,  'HIGH',
        'R4', 80, 100, 'EXTREME'
    ) AS (risk_code, score_min, score_max, risk_desc)
),

user_base AS (
    SELECT
        u.user_id,
        u.mobile,
        u.email,
        u.gender,
        u.birth_year,
        u.register_time,
        u.register_channel,
        u.province,
        u.city,
        u.member_level,
        u.member_score,
        u.is_employee,
        u.is_black_user,
        floor(months_between(p.biz_date, concat(u.birth_year, '-01-01')) / 12) AS age,
        datediff(p.biz_date, substr(u.register_time, 1, 10)) AS register_days,
        regexp_replace(lower(coalesce(u.email, '')), '^.*@', '') AS email_domain,
        get_json_object(u.ext_json, '$.identity.real_name_auth') AS real_name_auth,
        get_json_object(u.ext_json, '$.device.first_os') AS first_os,
        get_json_object(u.ext_json, '$.source.inviter_user_id') AS inviter_user_id,
        CASE
            WHEN upper(u.register_channel) RLIKE 'DOUYIN|KUAISHOU|XIAOHONGSHU' THEN 'CONTENT_MEDIA'
            WHEN upper(u.register_channel) RLIKE 'EMAIL|SMS|PUSH|WECHAT' THEN 'PRIVATE_DOMAIN'
            WHEN upper(u.register_channel) RLIKE 'STORE|OFFLINE' THEN 'OFFLINE'
            ELSE 'OTHER'
        END AS register_channel_group,
        p.biz_date
    FROM ods_user.ods_user_profile_df u
    CROSS JOIN runtime_params p
    WHERE u.dt = p.biz_date
      AND u.country_code = p.country_code
      AND coalesce(u.is_test_user, 0) = 0
),

user_preference_long AS (
    SELECT
        ub.user_id,
        pref_pos,
        split(pref_pair, ':')[0] AS pref_key,
        split(pref_pair, ':')[1] AS pref_value
    FROM user_base ub
    LATERAL VIEW OUTER posexplode(
        split(coalesce(get_json_object(
            concat('{"profile":', coalesce('{}', '{}'), '}'),
            '$.profile.preference_json'
        ), 'color:red,size:m'), ',')
    ) p AS pref_pos, pref_pair
),

event_raw AS (
    SELECT
        e.user_id,
        e.session_id,
        e.event_id,
        e.event_type,
        e.event_time,
        e.event_ts,
        e.page_id,
        e.refer_page_id,
        e.channel,
        e.device_id,
        e.ip,
        get_json_object(e.ext_json, '$.sku_id') AS sku_id,
        get_json_object(e.ext_json, '$.category_id') AS category_id,
        get_json_object(e.ext_json, '$.search_keyword') AS search_keyword,
        get_json_object(e.ext_json, '$.coupon_id') AS coupon_id,
        get_json_object(e.ext_json, '$.ab_test.exp_id') AS exp_id,
        get_json_object(e.ext_json, '$.ab_test.group_id') AS group_id,
        split(
            regexp_replace(
                regexp_replace(
                    coalesce(get_json_object(e.ext_json, '$.expose_sku_list'), '[]'),
                    '\\[|\\]|"',
                    ''
                ),
                '\\s+',
                ''
            ),
            ','
        ) AS expose_sku_arr,
        e.dt
    FROM dwd_behavior.dwd_app_event_log_di e
    JOIN runtime_params p ON e.dt BETWEEN p.d90 AND p.biz_date
    WHERE e.user_id IS NOT NULL
),

event_exploded AS (
    SELECT
        er.*,
        expose_pos,
        expose_sku_id,
        if(er.event_type = 'pay_success', 1, 0) AS is_pay_event
    FROM event_raw er
    LATERAL VIEW OUTER posexplode(er.expose_sku_arr) ex AS expose_pos, expose_sku_id
),

event_windowed AS (
    SELECT
        ee.*,
        row_number() OVER (
            PARTITION BY ee.user_id, ee.session_id
            ORDER BY ee.event_ts
        ) AS event_seq,
        rank() OVER (
            PARTITION BY ee.user_id
            ORDER BY ee.event_ts
        ) AS event_time_rank,
        dense_rank() OVER (
            PARTITION BY ee.user_id
            ORDER BY ee.event_ts
        ) AS event_time_dense_rank,
        ntile(5) OVER (
            PARTITION BY ee.user_id
            ORDER BY ee.event_ts
        ) AS journey_quintile,
        percent_rank() OVER (
            PARTITION BY ee.user_id
            ORDER BY ee.event_ts
        ) AS journey_percent_rank,
        cume_dist() OVER (
            PARTITION BY ee.user_id
            ORDER BY ee.event_ts
        ) AS journey_cume_dist,
        lag(ee.event_ts, 1) OVER (
            PARTITION BY ee.user_id, ee.session_id
            ORDER BY ee.event_ts
        ) AS previous_event_ts,
        lead(ee.event_ts, 1) OVER (
            PARTITION BY ee.user_id, ee.session_id
            ORDER BY ee.event_ts
        ) AS next_event_ts,
        sum(ee.is_pay_event) OVER (
            PARTITION BY ee.user_id
            ORDER BY ee.event_ts
        ) AS running_pay_cnt
    FROM event_exploded ee
),

event_user_agg AS (
    SELECT
        user_id,
        count(1) AS event_row_cnt_90d,
        count(DISTINCT event_id) AS event_cnt_90d,
        count(DISTINCT session_id) AS session_cnt_90d,
        count(DISTINCT device_id) AS device_cnt_90d,
        count(DISTINCT ip) AS ip_cnt_90d,
        count(DISTINCT if(expose_sku_id IS NOT NULL AND expose_sku_id <> '', expose_sku_id, NULL)) AS exposed_sku_cnt_90d,
        sum(if(event_type = 'page_view', 1, 0)) AS page_view_cnt_90d,
        sum(if(event_type = 'search', 1, 0)) AS search_cnt_90d,
        sum(if(event_type = 'sku_click', 1, 0)) AS click_cnt_90d,
        sum(if(event_type = 'add_cart', 1, 0)) AS add_cart_cnt_90d,
        sum(if(event_type = 'pay_success', 1, 0)) AS pay_success_cnt_90d,
        max(event_ts - coalesce(previous_event_ts, event_ts)) AS max_event_gap_sec,
        max(coalesce(next_event_ts, event_ts) - event_ts) AS max_next_gap_sec,
        max(event_seq) AS max_session_depth,
        max(journey_quintile) AS max_journey_quintile,
        max(journey_percent_rank) AS max_journey_percent_rank,
        max(journey_cume_dist) AS max_journey_cume_dist,
        max(running_pay_cnt) AS running_pay_cnt_90d,
        concat_ws(',', sort_array(collect_set(channel))) AS channel_path_90d,
        concat_ws('>', collect_list(event_type)) AS raw_event_path_90d,
        percentile_approx(event_ts, 0.25) AS event_ts_p25,
        percentile_approx(event_ts, 0.50) AS event_ts_p50,
        percentile_approx(event_ts, 0.75) AS event_ts_p75,
        max(event_time) AS last_event_time,
        min(event_time) AS first_event_time
    FROM event_windowed
    GROUP BY user_id
),

order_raw AS (
    SELECT
        o.user_id,
        o.order_id,
        o.order_item_id,
        o.sku_id,
        o.shop_id,
        o.store_id,
        o.quantity,
        o.goods_amount,
        o.discount_amount,
        o.coupon_amount,
        o.pay_amount,
        o.cost_amount,
        o.payment_method,
        o.delivery_type,
        o.pay_time,
        o.order_status,
        o.refund_status,
        o.address_hash,
        o.dt
    FROM dwd_trade.dwd_order_item_df o
    JOIN runtime_params p ON o.dt BETWEEN p.d365 AND p.biz_date
    WHERE coalesce(o.is_test_order, 0) = 0
),

order_windowed AS (
    SELECT
        o.*,
        row_number() OVER (
            PARTITION BY o.user_id
            ORDER BY o.pay_time DESC
        ) AS latest_order_rank,
        dense_rank() OVER (
            PARTITION BY o.user_id
            ORDER BY o.quantity DESC
        ) AS quantity_dense_rank,
        ntile(4) OVER (
            PARTITION BY o.user_id
            ORDER BY o.quantity
        ) AS quantity_quartile,
        lag(o.quantity, 1) OVER (
            PARTITION BY o.user_id
            ORDER BY o.pay_time
        ) AS previous_quantity,
        lead(o.quantity, 1) OVER (
            PARTITION BY o.user_id
            ORDER BY o.pay_time
        ) AS next_quantity
    FROM order_raw o
),

trade_user_agg AS (
    SELECT
        user_id,
        count(DISTINCT order_id) AS paid_order_cnt_365d,
        count(DISTINCT sku_id) AS paid_sku_cnt_365d,
        count(DISTINCT shop_id) AS paid_shop_cnt_365d,
        count(DISTINCT address_hash) AS address_cnt_365d,
        sum(quantity) AS paid_item_qty_365d,
        round(sum(pay_amount), 2) AS pay_amount_365d,
        round(sum(discount_amount + coupon_amount), 2) AS discount_amount_365d,
        round(sum(pay_amount - cost_amount), 2) AS gross_margin_365d,
        max(quantity_quartile) AS max_quantity_quartile,
        max(quantity_dense_rank) AS max_quantity_dense_rank,
        max(abs(quantity - coalesce(previous_quantity, quantity))) AS max_quantity_change,
        max(if(latest_order_rank = 1, order_id, NULL)) AS latest_order_id,
        max(if(latest_order_rank = 1, sku_id, NULL)) AS latest_sku_id,
        concat_ws(',', sort_array(collect_set(payment_method))) AS payment_method_set_365d,
        concat_ws(',', sort_array(collect_set(delivery_type))) AS delivery_type_set_365d,
        percentile_approx(pay_amount, 0.50) AS median_item_pay_amount_365d,
        max(pay_time) AS last_pay_time
    FROM order_windowed
    GROUP BY user_id
),

refund_user_agg AS (
    SELECT
        r.user_id,
        count(DISTINCT r.refund_id) AS refund_cnt_365d,
        round(sum(r.refund_amount), 2) AS refund_amount_365d,
        sum(if(coalesce(r.is_abnormal_refund, 0) = 1, 1, 0)) AS abnormal_refund_cnt_365d,
        concat_ws(',', sort_array(collect_set(r.refund_reason_code))) AS refund_reason_set_365d
    FROM dwd_trade.dwd_refund_order_df r
    JOIN runtime_params p ON r.dt BETWEEN p.d365 AND p.biz_date
    GROUP BY r.user_id
),

marketing_weighted AS (
    SELECT
        m.user_id,
        m.touch_id,
        m.campaign_id,
        m.touch_channel,
        m.touch_ts,
        coalesce(cr.channel_weight, 0.01) AS channel_weight,
        coalesce(cr.channel_group, 'UNKNOWN') AS channel_group
    FROM ads_marketing.ads_user_campaign_touch_di m
    JOIN runtime_params p ON m.dt BETWEEN p.d90 AND p.biz_date
    LEFT JOIN channel_rule cr
      ON upper(coalesce(m.touch_channel, m.channel, 'UNKNOWN')) = cr.channel
),

marketing_user_agg AS (
    SELECT
        user_id,
        count(DISTINCT touch_id) AS marketing_touch_cnt_90d,
        count(DISTINCT campaign_id) AS campaign_cnt_90d,
        round(sum(channel_weight), 4) AS weighted_touch_score_90d,
        concat_ws(',', sort_array(collect_set(touch_channel))) AS marketing_channel_set_90d,
        concat_ws(',', sort_array(collect_set(channel_group))) AS marketing_group_set_90d,
        percentile_approx(touch_ts, 0.50) AS median_touch_ts_90d
    FROM marketing_weighted
    GROUP BY user_id
),

service_user_agg AS (
    SELECT
        s.user_id,
        count(DISTINCT s.ticket_id) AS ticket_cnt_365d,
        sum(if(s.ticket_status = 'SOLVED', 1, 0)) AS solved_ticket_cnt_365d,
        sum(if(coalesce(s.is_escalated, 0) = 1, 1, 0)) AS escalated_ticket_cnt_365d,
        round(avg(s.satisfaction_score), 2) AS avg_satisfaction_score_365d,
        concat_ws(',', sort_array(collect_set(s.ticket_type))) AS ticket_type_set_365d
    FROM ods_service.ods_customer_ticket_df s
    JOIN runtime_params p ON s.dt BETWEEN p.d365 AND p.biz_date
    GROUP BY s.user_id
),

active_user_marker AS (
    SELECT ub.user_id, 1 AS is_active_user
    FROM user_base ub
    LEFT SEMI JOIN event_user_agg eua
      ON ub.user_id = eua.user_id
     AND eua.event_cnt_90d > 0
),

no_order_user_marker AS (
    SELECT ub.user_id, 1 AS is_no_order_user
    FROM user_base ub
    LEFT ANTI JOIN trade_user_agg tua
      ON ub.user_id = tua.user_id
),

event_trade_full AS (
    SELECT
        coalesce(eua.user_id, tua.user_id) AS user_id,
        coalesce(eua.event_cnt_90d, 0) AS event_cnt_90d,
        coalesce(tua.paid_order_cnt_365d, 0) AS paid_order_cnt_365d,
        coalesce(tua.pay_amount_365d, 0) AS pay_amount_365d
    FROM event_user_agg eua
    FULL OUTER JOIN trade_user_agg tua
      ON eua.user_id = tua.user_id
),

metric_long AS (
    SELECT
        etf.user_id,
        metric_name,
        metric_value
    FROM event_trade_full etf
    LATERAL VIEW stack(
        3,
        'event_cnt_90d', cast(etf.event_cnt_90d AS double),
        'paid_order_cnt_365d', cast(etf.paid_order_cnt_365d AS double),
        'pay_amount_365d', cast(etf.pay_amount_365d AS double)
    ) m AS metric_name, metric_value
),

metric_map AS (
    SELECT
        user_id,
        str_to_map(
            concat_ws(',', sort_array(collect_set(
                concat(metric_name, ':', cast(round(metric_value, 2) AS string))
            ))),
            ',',
            ':'
        ) AS metric_map_365d
    FROM metric_long
    GROUP BY user_id
),

province_segment_summary AS (
    SELECT
        ub.province,
        ub.member_level,
        count(DISTINCT ub.user_id) AS user_cnt,
        round(sum(coalesce(tua.pay_amount_365d, 0)), 2) AS pay_amount
    FROM user_base ub
    LEFT JOIN trade_user_agg tua ON ub.user_id = tua.user_id
    GROUP BY ub.province, ub.member_level
    GROUPING SETS (
        (ub.province, ub.member_level),
        (ub.province),
        (ub.member_level),
        ()
    )
),

channel_member_cube AS (
    SELECT
        ub.register_channel_group,
        ub.member_level,
        count(DISTINCT ub.user_id) AS user_cnt
    FROM user_base ub
    GROUP BY ub.register_channel_group, ub.member_level
    WITH CUBE
),

province_city_rollup AS (
    SELECT
        ub.province,
        ub.city,
        count(DISTINCT ub.user_id) AS user_cnt
    FROM user_base ub
    GROUP BY ub.province, ub.city
    WITH ROLLUP
),

scored_user AS (
    SELECT
        ub.*,
        coalesce(eua.event_cnt_90d, 0) AS event_cnt_90d,
        coalesce(eua.session_cnt_90d, 0) AS session_cnt_90d,
        coalesce(eua.device_cnt_90d, 0) AS device_cnt_90d,
        coalesce(eua.ip_cnt_90d, 0) AS ip_cnt_90d,
        coalesce(eua.search_cnt_90d, 0) AS search_cnt_90d,
        coalesce(eua.click_cnt_90d, 0) AS click_cnt_90d,
        coalesce(eua.add_cart_cnt_90d, 0) AS add_cart_cnt_90d,
        coalesce(eua.pay_success_cnt_90d, 0) AS pay_success_cnt_90d,
        coalesce(eua.max_session_depth, 0) AS max_session_depth,
        coalesce(eua.max_event_gap_sec, 0) AS max_event_gap_sec,
        coalesce(eua.max_journey_percent_rank, 0) AS max_journey_percent_rank,
        coalesce(eua.channel_path_90d, '') AS channel_path_90d,
        coalesce(tua.paid_order_cnt_365d, 0) AS paid_order_cnt_365d,
        coalesce(tua.paid_item_qty_365d, 0) AS paid_item_qty_365d,
        coalesce(tua.pay_amount_365d, 0) AS pay_amount_365d,
        coalesce(tua.gross_margin_365d, 0) AS gross_margin_365d,
        coalesce(tua.latest_order_id, '') AS latest_order_id,
        coalesce(tua.latest_sku_id, '') AS latest_sku_id,
        coalesce(rua.refund_cnt_365d, 0) AS refund_cnt_365d,
        coalesce(rua.refund_amount_365d, 0) AS refund_amount_365d,
        coalesce(rua.abnormal_refund_cnt_365d, 0) AS abnormal_refund_cnt_365d,
        coalesce(mua.marketing_touch_cnt_90d, 0) AS marketing_touch_cnt_90d,
        coalesce(mua.weighted_touch_score_90d, 0) AS weighted_touch_score_90d,
        coalesce(sua.ticket_cnt_365d, 0) AS ticket_cnt_365d,
        coalesce(sua.escalated_ticket_cnt_365d, 0) AS escalated_ticket_cnt_365d,
        coalesce(aum.is_active_user, 0) AS is_active_user,
        coalesce(noum.is_no_order_user, 0) AS is_no_order_user,
        coalesce(mm.metric_map_365d, '') AS metric_map_365d,
        round(
            coalesce(tua.pay_amount_365d, 0) * 0.05
            + coalesce(eua.event_cnt_90d, 0) * 0.20
            + coalesce(mua.weighted_touch_score_90d, 0) * 2.0
            - coalesce(rua.refund_amount_365d, 0) * 0.30,
            4
        ) AS operation_score,
        least(
            100,
            greatest(
                0,
                coalesce(ub.is_black_user, 0) * 80
                + coalesce(rua.abnormal_refund_cnt_365d, 0) * 10
                + if(coalesce(eua.device_cnt_90d, 0) >= 3, 10, 0)
                + if(coalesce(eua.ip_cnt_90d, 0) >= 5, 10, 0)
                + if(coalesce(sua.escalated_ticket_cnt_365d, 0) > 0, 10, 0)
            )
        ) AS risk_score
    FROM user_base ub
    LEFT JOIN event_user_agg eua ON ub.user_id = eua.user_id
    LEFT JOIN trade_user_agg tua ON ub.user_id = tua.user_id
    LEFT JOIN refund_user_agg rua ON ub.user_id = rua.user_id
    LEFT JOIN marketing_user_agg mua ON ub.user_id = mua.user_id
    LEFT JOIN service_user_agg sua ON ub.user_id = sua.user_id
    LEFT JOIN active_user_marker aum ON ub.user_id = aum.user_id
    LEFT JOIN no_order_user_marker noum ON ub.user_id = noum.user_id
    LEFT JOIN metric_map mm ON ub.user_id = mm.user_id
),

ranked_user AS (
    SELECT
        su.*,
        row_number() OVER (
            PARTITION BY su.province
            ORDER BY su.operation_score DESC, su.user_id
        ) AS province_operation_rank,
        dense_rank() OVER (
            PARTITION BY su.member_level
            ORDER BY su.pay_amount_365d DESC
        ) AS member_value_dense_rank,
        ntile(10) OVER (
            ORDER BY su.operation_score DESC
        ) AS global_score_decile,
        percent_rank() OVER (
            ORDER BY su.operation_score
        ) AS global_score_percent_rank,
        cume_dist() OVER (
            ORDER BY su.risk_score
        ) AS global_risk_cume_dist
    FROM scored_user su
),

final_user_extreme_wide AS (
    SELECT
        ru.*,
        rr.risk_code,
        rr.risk_desc,
        CASE
            WHEN ru.risk_score >= 80 THEN 'BLOCK'
            WHEN ru.risk_score >= 60 THEN 'MANUAL_REVIEW'
            WHEN ru.pay_amount_365d >= 5000 THEN 'SUPER_VALUE'
            WHEN ru.pay_amount_365d >= 1000 THEN 'HIGH_VALUE'
            WHEN ru.is_active_user = 1 AND ru.is_no_order_user = 1 THEN 'ACTIVE_NO_ORDER'
            WHEN ru.event_cnt_90d > 0 THEN 'ACTIVE'
            ELSE 'SILENT'
        END AS user_segment,
        md5(concat_ws(
            '|',
            ru.user_id,
            cast(ru.event_cnt_90d AS string),
            cast(ru.pay_amount_365d AS string),
            cast(ru.risk_score AS string),
            coalesce(ru.metric_map_365d, '')
        )) AS feature_hash,
        current_timestamp() AS etl_time
    FROM ranked_user ru
    LEFT JOIN risk_rule rr
      ON ru.risk_score BETWEEN rr.score_min AND rr.score_max
)

FROM final_user_extreme_wide f

INSERT OVERWRITE TABLE ads_user.ads_user_syntax_torture_profile_df
PARTITION (dt = '${biz_date}', biz_line = 'OMNICHANNEL')
SELECT *
WHERE f.user_id IS NOT NULL
  AND coalesce(f.is_employee, 0) = 0

INSERT OVERWRITE TABLE ads_user.ads_user_syntax_torture_alert_df
PARTITION (dt = '${biz_date}')
SELECT
    f.user_id,
    f.mobile,
    f.email,
    f.province,
    f.city,
    f.member_level,
    f.event_cnt_90d,
    f.pay_amount_365d,
    f.refund_amount_365d,
    f.risk_score,
    f.risk_code,
    f.user_segment,
    f.operation_score,
    f.global_score_decile,
    f.feature_hash,
    current_timestamp() AS etl_time
WHERE coalesce(f.is_employee, 0) = 0
  AND (
      f.risk_score >= 60
      OR f.refund_amount_365d > f.pay_amount_365d * 0.35
      OR (f.is_active_user = 1 AND f.is_no_order_user = 1)
  )
;
