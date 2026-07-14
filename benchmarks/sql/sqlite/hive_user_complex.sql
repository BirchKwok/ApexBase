
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
    FROM ods_user__ods_user_profile_df
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
    FROM dwd_behavior__dwd_app_event_log_di
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
    FROM dwd_trade__dwd_order_item_df
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
