
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
    FROM dwd_behavior__dwd_app_event_log_di e
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
    FROM dwd_trade__dwd_order_item_df o
    WHERE o.dt = '{biz_date}' AND o.is_test_order = 0
    GROUP BY o.user_id
),
refund_features AS (
    SELECT
        user_id,
        COUNT(DISTINCT refund_id) AS refund_cnt,
        SUM(refund_amount) AS refund_amount
    FROM dwd_trade__dwd_refund_order_df
    WHERE dt = '{biz_date}'
    GROUP BY user_id
),
marketing_features AS (
    SELECT user_id, COUNT(DISTINCT campaign_id) AS campaign_touch_cnt
    FROM ads_marketing__ads_user_campaign_touch_di
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
    FROM ods_user__ods_user_profile_df u
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
