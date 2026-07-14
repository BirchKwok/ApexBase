
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
    FROM dwd_behavior__dwd_app_event_log_di
    WHERE dt = '{biz_date}'
    GROUP BY user_id
),
order_30d AS (
    SELECT
        user_id,
        COUNT(DISTINCT order_id) AS order_cnt_30d,
        SUM(pay_amount) AS pay_amount_30d,
        AVG(pay_amount) AS avg_pay_amount_30d
    FROM dwd_trade__dwd_order_item_df
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
FROM ods_user__user_profile_df u
LEFT JOIN behavior_30d b ON u.user_id = b.user_id
LEFT JOIN order_30d o ON u.user_id = o.user_id
WHERE u.dt = '{biz_date}' AND u.country_code = 'CN'
ORDER BY u.user_id
