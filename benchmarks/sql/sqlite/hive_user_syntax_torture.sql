
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
    FROM ods_user__ods_user_profile_df
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
    FROM dwd_behavior__dwd_app_event_log_di
    WHERE dt = '{biz_date}'
    GROUP BY user_id
),
event_session_depth AS (
    SELECT user_id, MAX(session_depth) AS max_session_depth
    FROM (
        SELECT user_id, session_id, COUNT(*) AS session_depth
        FROM dwd_behavior__dwd_app_event_log_di
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
    FROM dwd_trade__dwd_order_item_df
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
    FROM dwd_trade__dwd_refund_order_df
    WHERE dt = '{biz_date}'
    GROUP BY user_id
),
marketing_agg AS (
    SELECT
        user_id,
        COUNT(DISTINCT touch_id) AS marketing_touch_cnt,
        COUNT(DISTINCT campaign_id) AS campaign_cnt,
        COUNT(DISTINCT touch_channel) AS marketing_channel_cnt
    FROM ads_marketing__ads_user_campaign_touch_di
    WHERE dt = '{biz_date}'
    GROUP BY user_id
),
service_agg AS (
    SELECT
        user_id,
        COUNT(DISTINCT ticket_id) AS ticket_cnt,
        SUM(CASE WHEN COALESCE(is_escalated, 0) = 1 THEN 1 ELSE 0 END) AS escalated_ticket_cnt,
        ROUND(AVG(satisfaction_score), 2) AS avg_satisfaction_score
    FROM ods_service__ods_customer_ticket_df
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
