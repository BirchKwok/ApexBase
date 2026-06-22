WITH
/* ============================================================
   1. 参数虚拟表：模拟运行参数、业务线、时间窗口
   ============================================================ */
params AS (
    SELECT
        '${biz_date}' AS biz_date,
        date_sub('${biz_date}', 1) AS d1,
        date_sub('${biz_date}', 7) AS d7,
        date_sub('${biz_date}', 30) AS d30,
        date_sub('${biz_date}', 90) AS d90,
        'CN' AS country_code,
        30 AS active_window_days,
        1000 AS high_value_threshold
),

/* ============================================================
   2. 虚拟维表：业务自定义风险等级映射
   ============================================================ */
risk_level_map AS (
    SELECT stack(
        5,
        'LOW',       0,   20,  '低风险',
        'MEDIUM',   21,  50,  '中风险',
        'HIGH',     51,  80,  '高风险',
        'EXTREME',  81,  100, '极高风险',
        'UNKNOWN',  -1,  -1,  '未知'
    ) AS (risk_code, score_min, score_max, risk_desc)
),

/* ============================================================
   3. 用户基础信息：跨库 dwd_user
   ============================================================ */
base_user AS (
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
        u.device_id,
        u.app_version,
        u.member_level,
        u.is_black_user,
        p.biz_date,

        floor(months_between(p.biz_date, concat(u.birth_year, '-01-01')) / 12) AS age,

        CASE
            WHEN floor(months_between(p.biz_date, concat(u.birth_year, '-01-01')) / 12) < 18 THEN 'UNDER_18'
            WHEN floor(months_between(p.biz_date, concat(u.birth_year, '-01-01')) / 12) BETWEEN 18 AND 24 THEN '18_24'
            WHEN floor(months_between(p.biz_date, concat(u.birth_year, '-01-01')) / 12) BETWEEN 25 AND 34 THEN '25_34'
            WHEN floor(months_between(p.biz_date, concat(u.birth_year, '-01-01')) / 12) BETWEEN 35 AND 44 THEN '35_44'
            WHEN floor(months_between(p.biz_date, concat(u.birth_year, '-01-01')) / 12) >= 45 THEN '45_PLUS'
            ELSE 'UNKNOWN'
        END AS age_bucket,

        regexp_replace(lower(coalesce(u.email, '')), '^.*@', '') AS email_domain,

        if(
            datediff(p.biz_date, substr(u.register_time, 1, 10)) <= 30,
            1,
            0
        ) AS is_new_user

    FROM ods_user.user_profile_df u
    CROSS JOIN params p
    WHERE u.dt = p.biz_date
      AND u.country_code = p.country_code
),

/* ============================================================
   4. 行为日志：跨库 dwd_behavior，解析 JSON、explode 商品曝光列表
   ============================================================ */
raw_behavior AS (
    SELECT
        b.user_id,
        b.session_id,
        b.event_id,
        b.event_type,
        b.event_time,
        b.page_id,
        b.refer_page_id,
        b.channel,
        b.device_id,
        b.ip,
        b.ua,
        get_json_object(b.ext_json, '$.sku_id') AS sku_id,
        get_json_object(b.ext_json, '$.spu_id') AS spu_id,
        get_json_object(b.ext_json, '$.category_id') AS category_id,
        get_json_object(b.ext_json, '$.search_keyword') AS search_keyword,
        get_json_object(b.ext_json, '$.coupon_id') AS coupon_id,
        get_json_object(b.ext_json, '$.ab_test.exp_id') AS exp_id,
        get_json_object(b.ext_json, '$.ab_test.group_id') AS group_id,
        split(
            regexp_replace(
                regexp_replace(get_json_object(b.ext_json, '$.expose_sku_list'), '\\[|\\]|"', ''),
                '\\s+',
                ''
            ),
            ','
        ) AS expose_sku_arr,
        b.dt
    FROM dwd_behavior.dwd_app_event_log_di b
    JOIN params p
      ON b.dt BETWEEN p.d30 AND p.biz_date
    WHERE b.user_id IS NOT NULL
      AND b.event_type IN (
          'page_view',
          'search',
          'sku_expose',
          'sku_click',
          'add_cart',
          'favorite',
          'submit_order',
          'pay_success',
          'coupon_receive',
          'coupon_use'
      )
),

behavior_explode AS (
    SELECT
        rb.user_id,
        rb.session_id,
        rb.event_id,
        rb.event_type,
        rb.event_time,
        rb.page_id,
        rb.refer_page_id,
        rb.channel,
        rb.device_id,
        rb.ip,
        rb.ua,
        rb.sku_id,
        rb.spu_id,
        rb.category_id,
        rb.search_keyword,
        rb.coupon_id,
        rb.exp_id,
        rb.group_id,
        expose_sku_id,
        rb.dt
    FROM raw_behavior rb
    LATERAL VIEW OUTER explode(rb.expose_sku_arr) e AS expose_sku_id
),

/* ============================================================
   5. 行为聚合：多种 Hive 函数、窗口、条件聚合
   ============================================================ */
behavior_agg AS (
    SELECT
        be.user_id,

        count(1) AS event_cnt_30d,
        count(DISTINCT be.session_id) AS session_cnt_30d,
        count(DISTINCT be.device_id) AS device_cnt_30d,
        count(DISTINCT be.ip) AS ip_cnt_30d,

        sum(if(be.event_type = 'page_view', 1, 0)) AS pv_cnt_30d,
        sum(if(be.event_type = 'search', 1, 0)) AS search_cnt_30d,
        sum(if(be.event_type = 'sku_expose', 1, 0)) AS expose_cnt_30d,
        sum(if(be.event_type = 'sku_click', 1, 0)) AS click_cnt_30d,
        sum(if(be.event_type = 'add_cart', 1, 0)) AS add_cart_cnt_30d,
        sum(if(be.event_type = 'favorite', 1, 0)) AS favorite_cnt_30d,
        sum(if(be.event_type = 'submit_order', 1, 0)) AS submit_order_cnt_30d,
        sum(if(be.event_type = 'pay_success', 1, 0)) AS pay_success_cnt_30d,

        count(DISTINCT if(be.event_type = 'search', be.search_keyword, NULL)) AS search_keyword_cnt_30d,
        count(DISTINCT if(be.event_type IN ('sku_click', 'add_cart', 'favorite'), be.sku_id, NULL)) AS interacted_sku_cnt_30d,
        count(DISTINCT if(be.expose_sku_id IS NOT NULL AND be.expose_sku_id <> '', be.expose_sku_id, NULL)) AS exposed_sku_cnt_30d,

        collect_set(be.channel) AS channel_set_30d,
        concat_ws(',', sort_array(collect_set(be.channel))) AS channel_path_30d,

        max(be.event_time) AS last_event_time,
        min(be.event_time) AS first_event_time,

        datediff(
            max(be.dt),
            min(be.dt)
        ) + 1 AS active_span_days_30d,

        round(
            sum(if(be.event_type = 'sku_click', 1, 0))
            / greatest(sum(if(be.event_type = 'sku_expose', 1, 0)), 1),
            6
        ) AS ctr_30d,

        round(
            sum(if(be.event_type = 'pay_success', 1, 0))
            / greatest(sum(if(be.event_type = 'sku_click', 1, 0)), 1),
            6
        ) AS click_pay_cvr_30d,

        percentile_approx(
            unix_timestamp(be.event_time),
            0.5
        ) AS median_event_ts_30d,

        max(if(be.dt >= date_sub('${biz_date}', 7), 1, 0)) AS is_active_7d,
        max(if(be.dt >= date_sub('${biz_date}', 1), 1, 0)) AS is_active_1d

    FROM behavior_explode be
    GROUP BY be.user_id
),

/* ============================================================
   6. 行为路径排序：session 内事件链路
   ============================================================ */
session_path AS (
    SELECT
        user_id,
        session_id,
        concat_ws(
            '>',
            collect_list(event_type)
        ) AS raw_event_path,
        count(1) AS session_event_cnt,
        min(event_time) AS session_start_time,
        max(event_time) AS session_end_time,
        unix_timestamp(max(event_time)) - unix_timestamp(min(event_time)) AS session_duration_sec
    FROM (
        SELECT
            user_id,
            session_id,
            event_type,
            event_time
        FROM behavior_explode
        DISTRIBUTE BY user_id, session_id
        SORT BY user_id, session_id, event_time
    ) t
    GROUP BY user_id, session_id
),

session_agg AS (
    SELECT
        user_id,
        count(1) AS valid_session_cnt_30d,
        avg(session_event_cnt) AS avg_session_event_cnt_30d,
        percentile_approx(session_duration_sec, 0.5) AS median_session_duration_sec_30d,
        max(session_duration_sec) AS max_session_duration_sec_30d,
        concat_ws(
            '||',
            slice(
                sort_array(
                    collect_list(
                        concat(
                            lpad(cast(session_event_cnt AS string), 10, '0'),
                            ':',
                            raw_event_path
                        )
                    )
                ),
                1,
                5
            )
        ) AS top_session_paths
    FROM session_path
    GROUP BY user_id
),

/* ============================================================
   7. 订单明细：跨库 dwd_trade + 商品维度 dim_product
   ============================================================ */
order_detail AS (
    SELECT
        o.user_id,
        o.order_id,
        o.parent_order_id,
        o.sku_id,
        o.spu_id,
        o.shop_id,
        o.pay_time,
        o.order_time,
        o.order_status,
        o.pay_status,
        o.refund_status,
        o.quantity,
        o.goods_amount,
        o.discount_amount,
        o.coupon_amount,
        o.freight_amount,
        o.pay_amount,
        o.payment_method,
        o.province AS order_province,
        o.city AS order_city,
        p.category_id,
        p.category_name,
        p.brand_id,
        p.brand_name,
        p.price_band,
        p.is_self_operated,
        p.is_imported,
        o.dt
    FROM dwd_trade.dwd_order_item_df o
    JOIN dim.dim_sku_df p
      ON o.sku_id = p.sku_id
     AND p.dt = '${biz_date}'
    JOIN params pa
      ON o.dt BETWEEN pa.d90 AND pa.biz_date
    WHERE o.is_test_order = 0
      AND o.user_id IS NOT NULL
),

order_agg AS (
    SELECT
        user_id,

        count(DISTINCT order_id) AS order_cnt_90d,
        count(DISTINCT if(pay_status = 'PAID', order_id, NULL)) AS paid_order_cnt_90d,
        count(DISTINCT if(refund_status IN ('REFUNDING', 'REFUNDED'), order_id, NULL)) AS refund_order_cnt_90d,

        sum(quantity) AS total_quantity_90d,
        sum(goods_amount) AS total_goods_amount_90d,
        sum(discount_amount) AS total_discount_amount_90d,
        sum(coupon_amount) AS total_coupon_amount_90d,
        sum(freight_amount) AS total_freight_amount_90d,
        sum(if(pay_status = 'PAID', pay_amount, 0)) AS total_pay_amount_90d,

        round(
            sum(if(pay_status = 'PAID', pay_amount, 0))
            / greatest(count(DISTINCT if(pay_status = 'PAID', order_id, NULL)), 1),
            2
        ) AS avg_paid_order_amount_90d,

        count(DISTINCT category_id) AS bought_category_cnt_90d,
        count(DISTINCT brand_id) AS bought_brand_cnt_90d,
        count(DISTINCT shop_id) AS bought_shop_cnt_90d,

        max(pay_time) AS last_pay_time,
        min(pay_time) AS first_pay_time,

        datediff('${biz_date}', substr(max(pay_time), 1, 10)) AS days_since_last_pay,

        sum(if(is_self_operated = 1 AND pay_status = 'PAID', pay_amount, 0)) AS self_operated_pay_amount_90d,
        sum(if(is_imported = 1 AND pay_status = 'PAID', pay_amount, 0)) AS imported_pay_amount_90d,

        concat_ws(',', sort_array(collect_set(payment_method))) AS payment_method_set_90d,

        round(
            count(DISTINCT if(refund_status IN ('REFUNDING', 'REFUNDED'), order_id, NULL))
            / greatest(count(DISTINCT order_id), 1),
            6
        ) AS refund_order_rate_90d

    FROM order_detail
    GROUP BY user_id
),

/* ============================================================
   8. 用户品类偏好：窗口函数 + 排名 + 拼接
   ============================================================ */
category_preference_rank AS (
    SELECT
        user_id,
        category_id,
        category_name,
        sum(if(pay_status = 'PAID', pay_amount, 0)) AS category_pay_amount,
        count(DISTINCT if(pay_status = 'PAID', order_id, NULL)) AS category_order_cnt,
        row_number() OVER (
            PARTITION BY user_id
            ORDER BY sum(if(pay_status = 'PAID', pay_amount, 0)) DESC,
                     count(DISTINCT if(pay_status = 'PAID', order_id, NULL)) DESC
        ) AS rn
    FROM order_detail
    GROUP BY user_id, category_id, category_name
),

category_preference AS (
    SELECT
        user_id,
        max(if(rn = 1, category_id, NULL)) AS top1_category_id,
        max(if(rn = 1, category_name, NULL)) AS top1_category_name,
        max(if(rn = 2, category_id, NULL)) AS top2_category_id,
        max(if(rn = 2, category_name, NULL)) AS top2_category_name,
        max(if(rn = 3, category_id, NULL)) AS top3_category_id,
        max(if(rn = 3, category_name, NULL)) AS top3_category_name,
        concat_ws(
            ',',
            collect_list(
                concat(
                    cast(rn AS string),
                    ':',
                    cast(category_id AS string),
                    ':',
                    category_name,
                    ':',
                    cast(round(category_pay_amount, 2) AS string)
                )
            )
        ) AS category_preference_path
    FROM category_preference_rank
    WHERE rn <= 5
    GROUP BY user_id
),

/* ============================================================
   9. 营销触达：跨库 ads_marketing
   ============================================================ */
marketing_touch AS (
    SELECT
        m.user_id,
        m.campaign_id,
        m.campaign_name,
        m.touch_channel,
        m.touch_time,
        m.material_id,
        m.strategy_id,
        m.scene,
        m.dt,

        row_number() OVER (
            PARTITION BY m.user_id
            ORDER BY m.touch_time DESC
        ) AS last_touch_rn,

        row_number() OVER (
            PARTITION BY m.user_id, m.campaign_id
            ORDER BY m.touch_time ASC
        ) AS campaign_first_touch_rn

    FROM ads_marketing.ads_user_campaign_touch_di m
    JOIN params p
      ON m.dt BETWEEN p.d30 AND p.biz_date
    WHERE m.user_id IS NOT NULL
),

marketing_agg AS (
    SELECT
        user_id,

        count(1) AS marketing_touch_cnt_30d,
        count(DISTINCT campaign_id) AS campaign_cnt_30d,
        count(DISTINCT touch_channel) AS marketing_channel_cnt_30d,

        concat_ws(',', sort_array(collect_set(touch_channel))) AS marketing_channel_set_30d,

        max(if(last_touch_rn = 1, campaign_id, NULL)) AS last_campaign_id,
        max(if(last_touch_rn = 1, campaign_name, NULL)) AS last_campaign_name,
        max(if(last_touch_rn = 1, touch_channel, NULL)) AS last_touch_channel,
        max(if(last_touch_rn = 1, touch_time, NULL)) AS last_touch_time,

        sum(if(scene = 'NEW_USER_COUPON', 1, 0)) AS new_user_coupon_touch_cnt_30d,
        sum(if(scene = 'RECALL', 1, 0)) AS recall_touch_cnt_30d,
        sum(if(scene = 'PRICE_DROP', 1, 0)) AS price_drop_touch_cnt_30d

    FROM marketing_touch
    GROUP BY user_id
),

/* ============================================================
   10. 券数据：领取、核销、过期
   ============================================================ */
coupon_agg AS (
    SELECT
        c.user_id,

        count(DISTINCT c.coupon_id) AS coupon_receive_cnt_30d,
        count(DISTINCT if(c.coupon_status = 'USED', c.coupon_id, NULL)) AS coupon_used_cnt_30d,
        count(DISTINCT if(c.coupon_status = 'EXPIRED', c.coupon_id, NULL)) AS coupon_expired_cnt_30d,

        sum(if(c.coupon_status = 'USED', c.discount_amount, 0)) AS coupon_used_amount_30d,

        round(
            count(DISTINCT if(c.coupon_status = 'USED', c.coupon_id, NULL))
            / greatest(count(DISTINCT c.coupon_id), 1),
            6
        ) AS coupon_use_rate_30d,

        max(c.receive_time) AS last_coupon_receive_time,
        max(if(c.coupon_status = 'USED', c.use_time, NULL)) AS last_coupon_use_time

    FROM dwd_promotion.dwd_user_coupon_df c
    JOIN params p
      ON c.dt BETWEEN p.d30 AND p.biz_date
    WHERE c.user_id IS NOT NULL
    GROUP BY c.user_id
),

/* ============================================================
   11. 风控特征：设备、IP、黑名单、异常行为
   ============================================================ */
risk_feature AS (
    SELECT
        bu.user_id,

        coalesce(ba.device_cnt_30d, 0) AS device_cnt_30d,
        coalesce(ba.ip_cnt_30d, 0) AS ip_cnt_30d,
        bu.is_black_user,

        CASE
            WHEN bu.is_black_user = 1 THEN 100
            WHEN coalesce(ba.device_cnt_30d, 0) >= 10 THEN 85
            WHEN coalesce(ba.ip_cnt_30d, 0) >= 20 THEN 75
            WHEN coalesce(oa.refund_order_rate_90d, 0) >= 0.5 THEN 70
            WHEN coalesce(ba.click_pay_cvr_30d, 0) = 0
                 AND coalesce(ba.click_cnt_30d, 0) > 100 THEN 60
            WHEN bu.email_domain IN ('tempmail.com', 'mailinator.com', 'fake.com') THEN 55
            ELSE greatest(
                cast(coalesce(ba.device_cnt_30d, 0) * 5 AS int),
                cast(coalesce(ba.ip_cnt_30d, 0) * 3 AS int),
                cast(coalesce(oa.refund_order_rate_90d, 0) * 100 AS int)
            )
        END AS risk_score

    FROM base_user bu
    LEFT JOIN behavior_agg ba
      ON bu.user_id = ba.user_id
    LEFT JOIN order_agg oa
      ON bu.user_id = oa.user_id
),

risk_with_level AS (
    SELECT
        rf.user_id,
        rf.device_cnt_30d,
        rf.ip_cnt_30d,
        rf.is_black_user,
        rf.risk_score,
        rlm.risk_code,
        rlm.risk_desc
    FROM risk_feature rf
    LEFT JOIN risk_level_map rlm
      ON (
          rf.risk_score BETWEEN rlm.score_min AND rlm.score_max
          OR (
              rf.risk_score IS NULL
              AND rlm.risk_code = 'UNKNOWN'
          )
      )
),

/* ============================================================
   12. 行转列：把事件类型转成多列指标
   ============================================================ */
event_pivot AS (
    SELECT
        user_id,

        max(if(metric_name = 'page_view', metric_value, 0)) AS pivot_page_view_cnt_30d,
        max(if(metric_name = 'search', metric_value, 0)) AS pivot_search_cnt_30d,
        max(if(metric_name = 'sku_expose', metric_value, 0)) AS pivot_sku_expose_cnt_30d,
        max(if(metric_name = 'sku_click', metric_value, 0)) AS pivot_sku_click_cnt_30d,
        max(if(metric_name = 'add_cart', metric_value, 0)) AS pivot_add_cart_cnt_30d,
        max(if(metric_name = 'favorite', metric_value, 0)) AS pivot_favorite_cnt_30d,
        max(if(metric_name = 'submit_order', metric_value, 0)) AS pivot_submit_order_cnt_30d,
        max(if(metric_name = 'pay_success', metric_value, 0)) AS pivot_pay_success_cnt_30d

    FROM (
        SELECT
            user_id,
            event_type AS metric_name,
            count(1) AS metric_value
        FROM behavior_explode
        GROUP BY user_id, event_type
    ) t
    GROUP BY user_id
),

/* ============================================================
   13. 列转行：把订单核心指标转成 name-value 结构，再聚合成 map
   ============================================================ */
order_metric_long AS (
    SELECT
        user_id,
        metric_name,
        metric_value
    FROM order_agg
    LATERAL VIEW stack(
        8,
        'order_cnt_90d',              cast(order_cnt_90d AS double),
        'paid_order_cnt_90d',         cast(paid_order_cnt_90d AS double),
        'refund_order_cnt_90d',       cast(refund_order_cnt_90d AS double),
        'total_pay_amount_90d',       cast(total_pay_amount_90d AS double),
        'avg_paid_order_amount_90d',  cast(avg_paid_order_amount_90d AS double),
        'bought_category_cnt_90d',    cast(bought_category_cnt_90d AS double),
        'bought_brand_cnt_90d',       cast(bought_brand_cnt_90d AS double),
        'refund_order_rate_90d',      cast(refund_order_rate_90d AS double)
    ) s AS metric_name, metric_value
),

order_metric_map AS (
    SELECT
        user_id,
        str_to_map(
            concat_ws(
                ',',
                collect_list(
                    concat(metric_name, ':', cast(metric_value AS string))
                )
            ),
            ',',
            ':'
        ) AS order_metric_map
    FROM order_metric_long
    GROUP BY user_id
),

/* ============================================================
   14. 标签规则虚拟表：基于虚拟表打标签
   ============================================================ */
tag_rule AS (
    SELECT stack(
        8,
        'HIGH_VALUE_USER',      '高价值用户',     'total_pay_amount_90d >= 1000',
        'NEW_ACTIVE_USER',      '新晋活跃用户',   'is_new_user = 1 and is_active_7d = 1',
        'HIGH_RISK_USER',       '高风险用户',     'risk_score >= 70',
        'COUPON_SENSITIVE',     '优惠券敏感',     'coupon_use_rate_30d >= 0.5',
        'SEARCH_HEAVY_USER',    '搜索重度用户',   'search_cnt_30d >= 20',
        'CART_LOST_USER',       '加购未支付用户', 'add_cart_cnt_30d > pay_success_cnt_30d',
        'MULTI_DEVICE_USER',    '多设备用户',     'device_cnt_30d >= 3',
        'REFUND_HEAVY_USER',    '退款偏高用户',   'refund_order_rate_90d >= 0.3'
    ) AS (tag_code, tag_name, tag_expr)
),

/* ============================================================
   15. 标签计算：复杂 CASE 组合
   ============================================================ */
user_tag_long AS (
    SELECT
        x.user_id,
        tr.tag_code,
        tr.tag_name
    FROM (
        SELECT
            bu.user_id,
            bu.is_new_user,
            coalesce(ba.is_active_7d, 0) AS is_active_7d,
            coalesce(ba.search_cnt_30d, 0) AS search_cnt_30d,
            coalesce(ba.add_cart_cnt_30d, 0) AS add_cart_cnt_30d,
            coalesce(ba.pay_success_cnt_30d, 0) AS pay_success_cnt_30d,
            coalesce(oa.total_pay_amount_90d, 0) AS total_pay_amount_90d,
            coalesce(oa.refund_order_rate_90d, 0) AS refund_order_rate_90d,
            coalesce(ca.coupon_use_rate_30d, 0) AS coupon_use_rate_30d,
            coalesce(rwl.risk_score, 0) AS risk_score,
            coalesce(rwl.device_cnt_30d, 0) AS device_cnt_30d
        FROM base_user bu
        LEFT JOIN behavior_agg ba ON bu.user_id = ba.user_id
        LEFT JOIN order_agg oa ON bu.user_id = oa.user_id
        LEFT JOIN coupon_agg ca ON bu.user_id = ca.user_id
        LEFT JOIN risk_with_level rwl ON bu.user_id = rwl.user_id
    ) x
    CROSS JOIN tag_rule tr
    WHERE
        CASE tr.tag_code
            WHEN 'HIGH_VALUE_USER'
                THEN if(x.total_pay_amount_90d >= 1000, 1, 0)
            WHEN 'NEW_ACTIVE_USER'
                THEN if(x.is_new_user = 1 AND x.is_active_7d = 1, 1, 0)
            WHEN 'HIGH_RISK_USER'
                THEN if(x.risk_score >= 70, 1, 0)
            WHEN 'COUPON_SENSITIVE'
                THEN if(x.coupon_use_rate_30d >= 0.5, 1, 0)
            WHEN 'SEARCH_HEAVY_USER'
                THEN if(x.search_cnt_30d >= 20, 1, 0)
            WHEN 'CART_LOST_USER'
                THEN if(x.add_cart_cnt_30d > x.pay_success_cnt_30d, 1, 0)
            WHEN 'MULTI_DEVICE_USER'
                THEN if(x.device_cnt_30d >= 3, 1, 0)
            WHEN 'REFUND_HEAVY_USER'
                THEN if(x.refund_order_rate_90d >= 0.3, 1, 0)
            ELSE 0
        END = 1
),

user_tag_agg AS (
    SELECT
        user_id,
        concat_ws(',', sort_array(collect_set(tag_code))) AS tag_code_list,
        concat_ws(',', sort_array(collect_set(tag_name))) AS tag_name_list,
        size(collect_set(tag_code)) AS tag_cnt
    FROM user_tag_long
    GROUP BY user_id
),

/* ============================================================
   16. AB 实验维表融合
   ============================================================ */
ab_exp_agg AS (
    SELECT
        user_id,
        concat_ws(
            ',',
            sort_array(
                collect_set(
                    concat(
                        coalesce(exp_id, 'UNKNOWN_EXP'),
                        ':',
                        coalesce(group_id, 'UNKNOWN_GROUP')
                    )
                )
            )
        ) AS ab_exp_group_list_30d
    FROM behavior_explode
    WHERE exp_id IS NOT NULL
    GROUP BY user_id
),

/* ============================================================
   17. 最近一次支付订单：窗口函数
   ============================================================ */
last_paid_order AS (
    SELECT
        *
    FROM (
        SELECT
            od.user_id,
            od.order_id AS last_paid_order_id,
            od.pay_time AS last_paid_order_time,
            od.pay_amount AS last_paid_order_amount,
            od.category_id AS last_paid_category_id,
            od.category_name AS last_paid_category_name,
            od.brand_id AS last_paid_brand_id,
            od.brand_name AS last_paid_brand_name,
            row_number() OVER (
                PARTITION BY od.user_id
                ORDER BY od.pay_time DESC, od.order_id DESC
            ) AS rn
        FROM order_detail od
        WHERE od.pay_status = 'PAID'
    ) t
    WHERE rn = 1
),

/* ============================================================
   18. 跨库客户服务数据
   ============================================================ */
service_agg AS (
    SELECT
        s.user_id,

        count(1) AS service_ticket_cnt_90d,
        sum(if(s.ticket_status = 'SOLVED', 1, 0)) AS solved_ticket_cnt_90d,
        sum(if(s.ticket_type = 'REFUND', 1, 0)) AS refund_ticket_cnt_90d,
        sum(if(s.ticket_type = 'COMPLAINT', 1, 0)) AS complaint_ticket_cnt_90d,

        avg(
            unix_timestamp(s.solve_time) - unix_timestamp(s.create_time)
        ) AS avg_solve_duration_sec_90d,

        max(s.create_time) AS last_service_time,

        concat_ws(',', sort_array(collect_set(s.ticket_type))) AS service_type_set_90d

    FROM ods_service.ods_customer_ticket_df s
    JOIN params p
      ON s.dt BETWEEN p.d90 AND p.biz_date
    WHERE s.user_id IS NOT NULL
    GROUP BY s.user_id
),

/* ============================================================
   19. 最终大宽表
   ============================================================ */
final_user_360 AS (
    SELECT
        bu.biz_date,
        bu.user_id,

        /* 用户基础 */
        bu.mobile,
        bu.email,
        bu.email_domain,
        bu.gender,
        bu.age,
        bu.age_bucket,
        bu.register_time,
        bu.register_channel,
        bu.province,
        bu.city,
        bu.device_id AS register_device_id,
        bu.app_version,
        bu.member_level,
        bu.is_new_user,

        /* 行为 */
        coalesce(ba.event_cnt_30d, 0) AS event_cnt_30d,
        coalesce(ba.session_cnt_30d, 0) AS session_cnt_30d,
        coalesce(ba.pv_cnt_30d, 0) AS pv_cnt_30d,
        coalesce(ba.search_cnt_30d, 0) AS search_cnt_30d,
        coalesce(ba.expose_cnt_30d, 0) AS expose_cnt_30d,
        coalesce(ba.click_cnt_30d, 0) AS click_cnt_30d,
        coalesce(ba.add_cart_cnt_30d, 0) AS add_cart_cnt_30d,
        coalesce(ba.favorite_cnt_30d, 0) AS favorite_cnt_30d,
        coalesce(ba.submit_order_cnt_30d, 0) AS submit_order_cnt_30d,
        coalesce(ba.pay_success_cnt_30d, 0) AS pay_success_cnt_30d,
        coalesce(ba.ctr_30d, 0) AS ctr_30d,
        coalesce(ba.click_pay_cvr_30d, 0) AS click_pay_cvr_30d,
        coalesce(ba.search_keyword_cnt_30d, 0) AS search_keyword_cnt_30d,
        coalesce(ba.interacted_sku_cnt_30d, 0) AS interacted_sku_cnt_30d,
        coalesce(ba.exposed_sku_cnt_30d, 0) AS exposed_sku_cnt_30d,
        ba.channel_path_30d,
        ba.last_event_time,
        ba.first_event_time,
        coalesce(ba.is_active_7d, 0) AS is_active_7d,
        coalesce(ba.is_active_1d, 0) AS is_active_1d,

        /* Session */
        coalesce(sa.valid_session_cnt_30d, 0) AS valid_session_cnt_30d,
        coalesce(sa.avg_session_event_cnt_30d, 0) AS avg_session_event_cnt_30d,
        coalesce(sa.median_session_duration_sec_30d, 0) AS median_session_duration_sec_30d,
        coalesce(sa.max_session_duration_sec_30d, 0) AS max_session_duration_sec_30d,
        sa.top_session_paths,

        /* 订单 */
        coalesce(oa.order_cnt_90d, 0) AS order_cnt_90d,
        coalesce(oa.paid_order_cnt_90d, 0) AS paid_order_cnt_90d,
        coalesce(oa.refund_order_cnt_90d, 0) AS refund_order_cnt_90d,
        coalesce(oa.total_quantity_90d, 0) AS total_quantity_90d,
        coalesce(oa.total_goods_amount_90d, 0) AS total_goods_amount_90d,
        coalesce(oa.total_discount_amount_90d, 0) AS total_discount_amount_90d,
        coalesce(oa.total_coupon_amount_90d, 0) AS total_coupon_amount_90d,
        coalesce(oa.total_freight_amount_90d, 0) AS total_freight_amount_90d,
        coalesce(oa.total_pay_amount_90d, 0) AS total_pay_amount_90d,
        coalesce(oa.avg_paid_order_amount_90d, 0) AS avg_paid_order_amount_90d,
        coalesce(oa.bought_category_cnt_90d, 0) AS bought_category_cnt_90d,
        coalesce(oa.bought_brand_cnt_90d, 0) AS bought_brand_cnt_90d,
        coalesce(oa.bought_shop_cnt_90d, 0) AS bought_shop_cnt_90d,
        coalesce(oa.refund_order_rate_90d, 0) AS refund_order_rate_90d,
        oa.payment_method_set_90d,
        oa.last_pay_time,
        oa.first_pay_time,
        coalesce(oa.days_since_last_pay, 9999) AS days_since_last_pay,

        /* 品类偏好 */
        cp.top1_category_id,
        cp.top1_category_name,
        cp.top2_category_id,
        cp.top2_category_name,
        cp.top3_category_id,
        cp.top3_category_name,
        cp.category_preference_path,

        /* 最近支付 */
        lpo.last_paid_order_id,
        lpo.last_paid_order_time,
        lpo.last_paid_order_amount,
        lpo.last_paid_category_id,
        lpo.last_paid_category_name,
        lpo.last_paid_brand_id,
        lpo.last_paid_brand_name,

        /* 营销 */
        coalesce(ma.marketing_touch_cnt_30d, 0) AS marketing_touch_cnt_30d,
        coalesce(ma.campaign_cnt_30d, 0) AS campaign_cnt_30d,
        coalesce(ma.marketing_channel_cnt_30d, 0) AS marketing_channel_cnt_30d,
        ma.marketing_channel_set_30d,
        ma.last_campaign_id,
        ma.last_campaign_name,
        ma.last_touch_channel,
        ma.last_touch_time,
        coalesce(ma.new_user_coupon_touch_cnt_30d, 0) AS new_user_coupon_touch_cnt_30d,
        coalesce(ma.recall_touch_cnt_30d, 0) AS recall_touch_cnt_30d,
        coalesce(ma.price_drop_touch_cnt_30d, 0) AS price_drop_touch_cnt_30d,

        /* 优惠券 */
        coalesce(ca.coupon_receive_cnt_30d, 0) AS coupon_receive_cnt_30d,
        coalesce(ca.coupon_used_cnt_30d, 0) AS coupon_used_cnt_30d,
        coalesce(ca.coupon_expired_cnt_30d, 0) AS coupon_expired_cnt_30d,
        coalesce(ca.coupon_used_amount_30d, 0) AS coupon_used_amount_30d,
        coalesce(ca.coupon_use_rate_30d, 0) AS coupon_use_rate_30d,
        ca.last_coupon_receive_time,
        ca.last_coupon_use_time,

        /* 风控 */
        coalesce(rwl.risk_score, 0) AS risk_score,
        coalesce(rwl.risk_code, 'UNKNOWN') AS risk_code,
        coalesce(rwl.risk_desc, '未知') AS risk_desc,
        coalesce(rwl.device_cnt_30d, 0) AS risk_device_cnt_30d,
        coalesce(rwl.ip_cnt_30d, 0) AS risk_ip_cnt_30d,

        /* 行转列指标 */
        coalesce(ep.pivot_page_view_cnt_30d, 0) AS pivot_page_view_cnt_30d,
        coalesce(ep.pivot_search_cnt_30d, 0) AS pivot_search_cnt_30d,
        coalesce(ep.pivot_sku_expose_cnt_30d, 0) AS pivot_sku_expose_cnt_30d,
        coalesce(ep.pivot_sku_click_cnt_30d, 0) AS pivot_sku_click_cnt_30d,
        coalesce(ep.pivot_add_cart_cnt_30d, 0) AS pivot_add_cart_cnt_30d,
        coalesce(ep.pivot_favorite_cnt_30d, 0) AS pivot_favorite_cnt_30d,
        coalesce(ep.pivot_submit_order_cnt_30d, 0) AS pivot_submit_order_cnt_30d,
        coalesce(ep.pivot_pay_success_cnt_30d, 0) AS pivot_pay_success_cnt_30d,

        /* metric map */
        omm.order_metric_map,

        /* 标签 */
        coalesce(uta.tag_code_list, '') AS tag_code_list,
        coalesce(uta.tag_name_list, '') AS tag_name_list,
        coalesce(uta.tag_cnt, 0) AS tag_cnt,

        /* AB 实验 */
        coalesce(aea.ab_exp_group_list_30d, '') AS ab_exp_group_list_30d,

        /* 客服 */
        coalesce(sva.service_ticket_cnt_90d, 0) AS service_ticket_cnt_90d,
        coalesce(sva.solved_ticket_cnt_90d, 0) AS solved_ticket_cnt_90d,
        coalesce(sva.refund_ticket_cnt_90d, 0) AS refund_ticket_cnt_90d,
        coalesce(sva.complaint_ticket_cnt_90d, 0) AS complaint_ticket_cnt_90d,
        coalesce(sva.avg_solve_duration_sec_90d, 0) AS avg_solve_duration_sec_90d,
        sva.last_service_time,
        sva.service_type_set_90d,

        /* 综合分层 */
        CASE
            WHEN coalesce(rwl.risk_score, 0) >= 80 THEN 'BLOCK_OR_REVIEW'
            WHEN coalesce(oa.total_pay_amount_90d, 0) >= 10000
                 AND coalesce(ba.is_active_7d, 0) = 1 THEN 'S_PLUS_ACTIVE_BUYER'
            WHEN coalesce(oa.total_pay_amount_90d, 0) >= 1000 THEN 'HIGH_VALUE_BUYER'
            WHEN coalesce(ba.add_cart_cnt_30d, 0) > 0
                 AND coalesce(ba.pay_success_cnt_30d, 0) = 0 THEN 'CART_ABANDONED'
            WHEN bu.is_new_user = 1
                 AND coalesce(ba.is_active_7d, 0) = 1 THEN 'NEW_ACTIVE'
            WHEN coalesce(ba.is_active_7d, 0) = 0
                 AND coalesce(oa.paid_order_cnt_90d, 0) > 0 THEN 'SILENT_BUYER'
            ELSE 'NORMAL'
        END AS user_segment,

        current_timestamp() AS etl_time

    FROM base_user bu

    LEFT JOIN behavior_agg ba
      ON bu.user_id = ba.user_id

    LEFT JOIN session_agg sa
      ON bu.user_id = sa.user_id

    LEFT JOIN order_agg oa
      ON bu.user_id = oa.user_id

    LEFT JOIN category_preference cp
      ON bu.user_id = cp.user_id

    LEFT JOIN last_paid_order lpo
      ON bu.user_id = lpo.user_id

    LEFT JOIN marketing_agg ma
      ON bu.user_id = ma.user_id

    LEFT JOIN coupon_agg ca
      ON bu.user_id = ca.user_id

    LEFT JOIN risk_with_level rwl
      ON bu.user_id = rwl.user_id

    LEFT JOIN event_pivot ep
      ON bu.user_id = ep.user_id

    LEFT JOIN order_metric_map omm
      ON bu.user_id = omm.user_id

    LEFT JOIN user_tag_agg uta
      ON bu.user_id = uta.user_id

    LEFT JOIN ab_exp_agg aea
      ON bu.user_id = aea.user_id

    LEFT JOIN service_agg sva
      ON bu.user_id = sva.user_id
)

/* ============================================================
   20. 最终写入目标宽表
   ============================================================ */
INSERT OVERWRITE TABLE ads_user.ads_user_360_profile_wide_df
PARTITION (dt = '${biz_date}')
SELECT
    *
FROM final_user_360
WHERE user_id IS NOT NULL
  AND user_segment NOT IN ('BLOCK_OR_REVIEW')
;