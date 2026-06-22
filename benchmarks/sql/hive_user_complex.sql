-- SET hive.exec.dynamic.partition = true;
-- SET hive.exec.dynamic.partition.mode = nonstrict;
-- SET hive.vectorized.execution.enabled = true;
-- SET hive.cbo.enable = true;
-- SET hive.exec.parallel = true;
-- SET hive.map.aggr = true;
-- SET hive.groupby.skewindata = true;

WITH
/* ============================================================
   0. 参数虚拟表：运行日期、窗口、渠道、业务线、实验版本
   ============================================================ */
runtime_params AS (
    SELECT
        '${biz_date}'                                              AS biz_date,
        date_sub('${biz_date}', 1)                                  AS d1,
        date_sub('${biz_date}', 3)                                  AS d3,
        date_sub('${biz_date}', 7)                                  AS d7,
        date_sub('${biz_date}', 14)                                 AS d14,
        date_sub('${biz_date}', 30)                                 AS d30,
        date_sub('${biz_date}', 60)                                 AS d60,
        date_sub('${biz_date}', 90)                                 AS d90,
        date_sub('${biz_date}', 180)                                AS d180,
        add_months('${biz_date}', -12)                              AS d365,
        unix_timestamp('${biz_date}', 'yyyy-MM-dd')                 AS biz_ts,
        'CN'                                                        AS country_code,
        'NEW_RETAIL'                                                AS biz_line,
        1000                                                        AS high_value_threshold,
        5000                                                        AS super_value_threshold,
        0.35                                                        AS refund_high_rate_threshold,
        0.6                                                         AS coupon_sensitive_threshold,
        1800                                                        AS session_timeout_sec
),

/* ============================================================
   1. 虚拟维表：生命周期规则
   ============================================================ */
life_cycle_rule AS (
    SELECT stack(
        8,
        'NEW_USER',          0,    7,     '新用户',
        'GROWING_USER',      8,    30,    '成长期用户',
        'ACTIVE_USER',       31,   90,    '活跃用户',
        'MATURE_USER',       91,   180,   '成熟用户',
        'OLD_USER',          181,  9999,  '老用户',
        'SILENT_USER',       -1,   -1,    '沉默用户',
        'LOST_USER',         -2,   -2,    '流失用户',
        'UNKNOWN_USER',      -9,   -9,    '未知用户'
    ) AS (
        life_cycle_code,
        min_days,
        max_days,
        life_cycle_name
    )
),

/* ============================================================
   2. 虚拟维表：风险分段规则
   ============================================================ */
risk_score_rule AS (
    SELECT stack(
        6,
        'R0', 0,   10,  '极低风险',
        'R1', 11,  30,  '低风险',
        'R2', 31,  50,  '中风险',
        'R3', 51,  70,  '高风险',
        'R4', 71,  90,  '极高风险',
        'R5', 91,  100, '黑名单风险'
    ) AS (
        risk_level,
        min_score,
        max_score,
        risk_desc
    )
),

/* ============================================================
   3. 虚拟维表：渠道权重，用于多触点归因
   ============================================================ */
channel_weight_rule AS (
    SELECT stack(
        12,
        'APP_PUSH',        0.18, '私域触达',
        'SMS',             0.12, '私域触达',
        'EMAIL',           0.08, '私域触达',
        'WECHAT',          0.22, '私域触达',
        'DOUYIN',          0.35, '外部投放',
        'KUAISHOU',        0.30, '外部投放',
        'XIAOHONGSHU',     0.28, '内容种草',
        'SEARCH_ENGINE',   0.26, '搜索广告',
        'AFFILIATE',       0.20, '联盟分销',
        'OFFLINE_STORE',   0.40, '线下门店',
        'NATURAL',         0.05, '自然流量',
        'UNKNOWN',         0.01, '未知渠道'
    ) AS (
        channel,
        base_weight,
        channel_group
    )
),

/* ============================================================
   4. 虚拟维表：价格带规则
   ============================================================ */
price_band_rule AS (
    SELECT stack(
        7,
        'P0', 0,     19.99,    '超低价',
        'P1', 20,    49.99,    '低价',
        'P2', 50,    99.99,    '平价',
        'P3', 100,   299.99,   '中价',
        'P4', 300,   999.99,   '高价',
        'P5', 1000,  4999.99,  '高端',
        'P6', 5000,  99999999, '奢侈'
    ) AS (
        price_band_code,
        min_price,
        max_price,
        price_band_name
    )
),

/* ============================================================
   5. 用户基础层：跨库读取用户主档
   ============================================================ */
user_base_raw AS (
    SELECT
        u.user_id,
        u.union_id,
        u.open_id,
        u.mobile,
        u.email,
        u.gender,
        u.birth_year,
        u.register_time,
        u.register_channel,
        u.register_app,
        u.register_ip,
        u.register_device_id,
        u.province,
        u.city,
        u.district,
        u.member_level,
        u.member_score,
        u.is_employee,
        u.is_black_user,
        u.is_test_user,
        u.ext_json,
        rp.biz_date,
        rp.d7,
        rp.d30,
        rp.d90,
        rp.d180,
        rp.d365,
        regexp_replace(lower(coalesce(u.email, '')), '^.*@', '') AS email_domain,
        floor(months_between(rp.biz_date, concat(u.birth_year, '-01-01')) / 12) AS age,
        datediff(rp.biz_date, substr(u.register_time, 1, 10)) AS register_days,
        get_json_object(u.ext_json, '$.identity.real_name_auth') AS real_name_auth,
        get_json_object(u.ext_json, '$.identity.student_auth')   AS student_auth,
        get_json_object(u.ext_json, '$.identity.enterprise_auth') AS enterprise_auth,
        get_json_object(u.ext_json, '$.device.first_os')          AS first_os,
        get_json_object(u.ext_json, '$.source.inviter_user_id')   AS inviter_user_id
    FROM ods_user.ods_user_profile_df u
    CROSS JOIN runtime_params rp
    WHERE u.dt = rp.biz_date
      AND u.country_code = rp.country_code
      AND coalesce(u.is_test_user, 0) = 0
),

/* ============================================================
   6. 用户基础清洗：年龄段、注册来源、认证信息归一
   ============================================================ */
user_base AS (
    SELECT
        user_id,
        union_id,
        open_id,
        mobile,
        email,
        email_domain,
        gender,
        birth_year,
        age,
        CASE
            WHEN age IS NULL THEN 'UNKNOWN'
            WHEN age < 18 THEN 'UNDER_18'
            WHEN age BETWEEN 18 AND 24 THEN '18_24'
            WHEN age BETWEEN 25 AND 34 THEN '25_34'
            WHEN age BETWEEN 35 AND 44 THEN '35_44'
            WHEN age BETWEEN 45 AND 59 THEN '45_59'
            ELSE '60_PLUS'
        END AS age_bucket,
        register_time,
        register_days,
        register_channel,
        register_app,
        register_ip,
        register_device_id,
        province,
        city,
        district,
        member_level,
        member_score,
        is_employee,
        is_black_user,
        real_name_auth,
        student_auth,
        enterprise_auth,
        first_os,
        inviter_user_id,
        biz_date,
        d7,
        d30,
        d90,
        d180,
        d365,
        CASE
            WHEN register_channel RLIKE 'douyin|kuaishou|xiaohongshu' THEN 'CONTENT_MEDIA'
            WHEN register_channel RLIKE 'sms|push|wechat|email' THEN 'PRIVATE_DOMAIN'
            WHEN register_channel RLIKE 'store|offline' THEN 'OFFLINE'
            WHEN register_channel RLIKE 'search|sem|seo' THEN 'SEARCH'
            ELSE 'OTHER'
        END AS register_channel_group
    FROM user_base_raw
),

/* ============================================================
   7. 用户生命周期：基于注册天数 + 最近活跃后续补充
   ============================================================ */
user_life_cycle_pre AS (
    SELECT
        ub.*,
        lcr.life_cycle_code AS register_life_cycle_code,
        lcr.life_cycle_name AS register_life_cycle_name
    FROM user_base ub
    LEFT JOIN life_cycle_rule lcr
      ON ub.register_days BETWEEN lcr.min_days AND lcr.max_days
),

/* ============================================================
   8. 原始行为日志：解析复杂 JSON、map、数组
   ============================================================ */
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
        e.sub_channel,
        e.device_id,
        e.device_type,
        e.os,
        e.app_version,
        e.ip,
        e.ua,
        e.dt,

        get_json_object(e.ext_json, '$.sku_id')                         AS sku_id,
        get_json_object(e.ext_json, '$.spu_id')                         AS spu_id,
        get_json_object(e.ext_json, '$.shop_id')                        AS shop_id,
        get_json_object(e.ext_json, '$.store_id')                       AS store_id,
        get_json_object(e.ext_json, '$.category_id')                    AS category_id,
        get_json_object(e.ext_json, '$.brand_id')                       AS brand_id,
        get_json_object(e.ext_json, '$.search.keyword')                 AS search_keyword,
        get_json_object(e.ext_json, '$.search.result_cnt')              AS search_result_cnt,
        get_json_object(e.ext_json, '$.coupon.coupon_id')               AS coupon_id,
        get_json_object(e.ext_json, '$.ab.exp_id')                      AS exp_id,
        get_json_object(e.ext_json, '$.ab.group_id')                    AS group_id,
        get_json_object(e.ext_json, '$.geo.longitude')                  AS longitude,
        get_json_object(e.ext_json, '$.geo.latitude')                   AS latitude,
        get_json_object(e.ext_json, '$.reco.strategy_id')               AS reco_strategy_id,
        get_json_object(e.ext_json, '$.reco.scene_id')                  AS reco_scene_id,

        split(
            regexp_replace(
                regexp_replace(
                    coalesce(get_json_object(e.ext_json, '$.exposure.sku_list'), '[]'),
                    '\\[|\\]|"',
                    ''
                ),
                '\\s+',
                ''
            ),
            ','
        ) AS expose_sku_arr

    FROM dwd_behavior.dwd_app_event_log_di e
    JOIN runtime_params rp
      ON e.dt BETWEEN rp.d90 AND rp.biz_date
    WHERE e.user_id IS NOT NULL
      AND e.event_type IN (
          'app_start',
          'page_view',
          'search',
          'sku_expose',
          'sku_click',
          'spu_click',
          'add_cart',
          'remove_cart',
          'favorite',
          'unfavorite',
          'coupon_receive',
          'coupon_use',
          'submit_order',
          'pay_success',
          'refund_apply',
          'share',
          'comment',
          'live_enter',
          'live_stay',
          'store_visit',
          'scan_qr',
          'customer_service_click'
      )
),

/* ============================================================
   9. 曝光数组拆解：posexplode 保留位置
   ============================================================ */
event_expose_sku AS (
    SELECT
        er.user_id,
        er.session_id,
        er.event_id,
        er.event_time,
        er.event_ts,
        er.dt,
        er.channel,
        er.sub_channel,
        er.page_id,
        er.refer_page_id,
        er.device_id,
        er.ip,
        er.ua,
        er.reco_strategy_id,
        er.reco_scene_id,
        expose_pos,
        expose_sku_id
    FROM event_raw er
    LATERAL VIEW OUTER posexplode(er.expose_sku_arr) pe AS expose_pos, expose_sku_id
    WHERE coalesce(expose_sku_id, '') <> ''
),

/* ============================================================
   11. 行为基础聚合：90 天、30 天、7 天混合
   ============================================================ */
event_user_agg AS (
    SELECT
        er.user_id,

        count(1) AS event_cnt_90d,
        count(DISTINCT er.session_id) AS session_cnt_90d,
        count(DISTINCT er.device_id) AS device_cnt_90d,
        count(DISTINCT er.ip) AS ip_cnt_90d,
        count(DISTINCT er.ua) AS ua_cnt_90d,
        count(DISTINCT er.channel) AS channel_cnt_90d,
        count(DISTINCT er.sub_channel) AS sub_channel_cnt_90d,

        sum(if(er.dt >= date_sub('${biz_date}', 30), 1, 0)) AS event_cnt_30d,
        sum(if(er.dt >= date_sub('${biz_date}', 7), 1, 0)) AS event_cnt_7d,
        sum(if(er.dt >= date_sub('${biz_date}', 1), 1, 0)) AS event_cnt_1d,

        count(DISTINCT if(er.dt >= date_sub('${biz_date}', 30), er.session_id, NULL)) AS session_cnt_30d,
        count(DISTINCT if(er.dt >= date_sub('${biz_date}', 7), er.session_id, NULL)) AS session_cnt_7d,

        sum(if(er.event_type = 'app_start', 1, 0)) AS app_start_cnt_90d,
        sum(if(er.event_type = 'page_view', 1, 0)) AS page_view_cnt_90d,
        sum(if(er.event_type = 'search', 1, 0)) AS search_cnt_90d,
        sum(if(er.event_type = 'sku_expose', 1, 0)) AS sku_expose_cnt_90d,
        sum(if(er.event_type = 'sku_click', 1, 0)) AS sku_click_cnt_90d,
        sum(if(er.event_type = 'add_cart', 1, 0)) AS add_cart_cnt_90d,
        sum(if(er.event_type = 'favorite', 1, 0)) AS favorite_cnt_90d,
        sum(if(er.event_type = 'submit_order', 1, 0)) AS submit_order_cnt_90d,
        sum(if(er.event_type = 'pay_success', 1, 0)) AS pay_success_event_cnt_90d,
        sum(if(er.event_type = 'refund_apply', 1, 0)) AS refund_apply_event_cnt_90d,
        sum(if(er.event_type = 'live_enter', 1, 0)) AS live_enter_cnt_90d,
        sum(if(er.event_type = 'live_stay', 1, 0)) AS live_stay_cnt_90d,
        sum(if(er.event_type = 'store_visit', 1, 0)) AS store_visit_cnt_90d,

        count(DISTINCT if(er.event_type = 'search', lower(er.search_keyword), NULL)) AS search_keyword_cnt_90d,
        count(DISTINCT if(er.event_type IN ('sku_click', 'add_cart', 'favorite'), er.sku_id, NULL)) AS interacted_sku_cnt_90d,
        count(DISTINCT if(er.store_id IS NOT NULL, er.store_id, NULL)) AS interacted_store_cnt_90d,
        count(DISTINCT if(er.reco_strategy_id IS NOT NULL, er.reco_strategy_id, NULL)) AS reco_strategy_cnt_90d,

        min(er.event_time) AS first_event_time_90d,
        max(er.event_time) AS last_event_time_90d,

        datediff('${biz_date}', substr(max(er.event_time), 1, 10)) AS days_since_last_event,

        round(
            sum(if(er.event_type = 'sku_click', 1, 0))
            / greatest(sum(if(er.event_type = 'sku_expose', 1, 0)), 1),
            8
        ) AS expose_click_rate_90d,

        round(
            sum(if(er.event_type = 'add_cart', 1, 0))
            / greatest(sum(if(er.event_type = 'sku_click', 1, 0)), 1),
            8
        ) AS click_cart_rate_90d,

        round(
            sum(if(er.event_type = 'pay_success', 1, 0))
            / greatest(sum(if(er.event_type = 'submit_order', 1, 0)), 1),
            8
        ) AS submit_pay_rate_90d,

        concat_ws(',', sort_array(collect_set(er.channel))) AS channel_set_90d,
        concat_ws(',', sort_array(collect_set(er.device_type))) AS device_type_set_90d,
        concat_ws(',', sort_array(collect_set(er.os))) AS os_set_90d

    FROM event_raw er
    GROUP BY er.user_id
),

/* ============================================================
   12. Session 内事件排序，构造行为路径
   ============================================================ */
session_event_windowed AS (
    SELECT
        user_id,
        session_id,
        event_id,
        event_type,
        event_time,
        event_ts,
        page_id,
        refer_page_id,
        channel,
        sku_id,
        lag(event_ts, 1, event_ts) OVER (
            PARTITION BY user_id, session_id
            ORDER BY event_ts ASC, event_id ASC
        ) AS prev_event_ts
    FROM event_raw
),

session_event_ordered AS (
    SELECT
        *,
        event_ts - coalesce(prev_event_ts, event_ts) AS gap_from_prev_sec
    FROM session_event_windowed
),

session_path_agg AS (
    SELECT
        user_id,
        session_id,
        concat_ws('>', collect_list(event_type)) AS session_event_path,
        count(1) AS session_event_cnt,
        count(DISTINCT page_id) AS session_page_cnt,
        count(DISTINCT sku_id) AS session_sku_cnt,
        coalesce(max(event_ts), 0) - coalesce(min(event_ts), 0) AS session_duration_sec
    FROM (
        SELECT *
        FROM event_raw
        DISTRIBUTE BY user_id, session_id
        SORT BY user_id, session_id, event_ts, event_id
    ) x
    GROUP BY user_id, session_id
),

session_user_agg AS (
    SELECT
        user_id,
        count(1) AS valid_session_cnt_90d,
        avg(session_event_cnt) AS avg_session_event_cnt_90d,
        percentile_approx(session_event_cnt, 0.5) AS median_session_event_cnt_90d,
        avg(session_page_cnt) AS avg_session_page_cnt_90d,
        avg(session_sku_cnt) AS avg_session_sku_cnt_90d,
        avg(session_duration_sec) AS avg_session_duration_sec_90d,
        percentile_approx(session_duration_sec, 0.5) AS median_session_duration_sec_90d,
        max(session_duration_sec) AS max_session_duration_sec_90d,
        concat_ws(
            '||',
            slice(
                reverse(
                    sort_array(
                        collect_list(
                            concat(
                                lpad(cast(session_event_cnt AS string), 10, '0'),
                                '#',
                                lpad(cast(session_duration_sec AS string), 10, '0'),
                                '#',
                                session_event_path
                            )
                        )
                    )
                ),
                1,
                10
            )
        ) AS top10_complex_session_paths_90d
    FROM session_path_agg
    GROUP BY user_id
),

/* ============================================================
   13. 商品维度：跨库 dim_sku，补充价格带、品牌、类目树
   ============================================================ */
sku_dim AS (
    SELECT
        s.sku_id,
        s.spu_id,
        s.product_name,
        s.shop_id,
        s.brand_id,
        s.brand_name,
        s.category_id,
        s.category_name,
        s.category_level1_id,
        s.category_level1_name,
        s.category_level2_id,
        s.category_level2_name,
        s.category_level3_id,
        s.category_level3_name,
        s.list_price,
        s.cost_price,
        s.is_self_operated,
        s.is_imported,
        s.is_fresh,
        s.is_virtual,
        s.shelf_status,
        pbr.price_band_code,
        pbr.price_band_name
    FROM dim.dim_sku_df s
    LEFT JOIN price_band_rule pbr
      ON s.list_price BETWEEN pbr.min_price AND pbr.max_price
    WHERE s.dt = '${biz_date}'
),

/* ============================================================
   14. 曝光商品特征：结合商品维度
   ============================================================ */
exposure_sku_agg AS (
    SELECT
        ees.user_id,
        count(1) AS exposure_item_cnt_90d,
        count(DISTINCT ees.expose_sku_id) AS exposure_sku_cnt_90d,
        count(DISTINCT sd.category_level1_id) AS exposure_cate1_cnt_90d,
        count(DISTINCT sd.brand_id) AS exposure_brand_cnt_90d,
        sum(if(ees.expose_pos BETWEEN 0 AND 2, 1, 0)) AS exposure_top3_pos_cnt_90d,
        avg(ees.expose_pos) AS avg_exposure_pos_90d,
        percentile_approx(ees.expose_pos, 0.5) AS median_exposure_pos_90d,
        concat_ws(',', sort_array(collect_set(sd.price_band_code))) AS exposure_price_band_set_90d,
        concat_ws(',', sort_array(collect_set(sd.category_level1_name))) AS exposure_cate1_name_set_90d
    FROM event_expose_sku ees
    LEFT JOIN sku_dim sd
      ON ees.expose_sku_id = sd.sku_id
    GROUP BY ees.user_id
),

/* ============================================================
   15. 搜索词清洗：中文、英文、数字、特殊字符归一
   ============================================================ */
search_keyword_clean AS (
    SELECT
        user_id,
        lower(trim(search_keyword)) AS raw_keyword,
        regexp_replace(lower(trim(search_keyword)), '[^0-9a-zA-Z\\u4e00-\\u9fa5]+', '') AS clean_keyword,
        dt,
        event_time,
        cast(search_result_cnt AS int) AS search_result_cnt
    FROM event_raw
    WHERE event_type = 'search'
      AND search_keyword IS NOT NULL
      AND length(trim(search_keyword)) > 0
),

search_keyword_agg AS (
    SELECT
        user_id,
        count(1) AS search_times_90d,
        count(DISTINCT clean_keyword) AS distinct_search_keyword_cnt_90d,
        sum(if(search_result_cnt = 0, 1, 0)) AS zero_result_search_cnt_90d,
        round(
            sum(if(search_result_cnt = 0, 1, 0))
            / greatest(count(1), 1),
            8
        ) AS zero_result_search_rate_90d,
        concat_ws(
            ',',
            slice(
                reverse(
                    sort_array(
                        collect_list(
                            concat(
                                lpad(cast(keyword_cnt AS string), 10, '0'),
                                ':',
                                clean_keyword
                            )
                        )
                    )
                ),
                1,
                20
            )
        ) AS top20_search_keywords_90d
    FROM (
        SELECT
            user_id,
            clean_keyword,
            count(1) AS keyword_cnt,
            max(search_result_cnt) AS search_result_cnt
        FROM search_keyword_clean
        GROUP BY user_id, clean_keyword
    ) t
    GROUP BY user_id
),

/* ============================================================
   16. 订单明细：跨库交易、商品、门店、履约、支付
   ============================================================ */
order_item_raw AS (
    SELECT
        oi.user_id,
        oi.order_id,
        oi.parent_order_id,
        oi.order_item_id,
        oi.sku_id,
        oi.spu_id,
        oi.shop_id,
        oi.store_id,
        oi.quantity,
        oi.goods_amount,
        oi.discount_amount,
        oi.platform_coupon_amount,
        oi.shop_coupon_amount,
        oi.points_deduction_amount,
        oi.freight_amount,
        oi.pay_amount,
        oi.cost_amount,
        oi.order_status,
        oi.pay_status,
        oi.refund_status,
        oi.order_time,
        oi.pay_time,
        oi.finish_time,
        oi.cancel_time,
        oi.payment_method,
        oi.delivery_type,
        oi.province AS order_province,
        oi.city AS order_city,
        oi.address_hash,
        oi.dt,
        sd.product_name,
        sd.brand_id,
        sd.brand_name,
        sd.category_id,
        sd.category_name,
        sd.category_level1_id,
        sd.category_level1_name,
        sd.category_level2_id,
        sd.category_level2_name,
        sd.category_level3_id,
        sd.category_level3_name,
        sd.price_band_code,
        sd.price_band_name,
        sd.is_self_operated,
        sd.is_imported,
        sd.is_fresh,
        sd.is_virtual,
        sd.cost_price AS sku_cost_price
    FROM dwd_trade.dwd_order_item_df oi
    LEFT JOIN sku_dim sd
      ON oi.sku_id = sd.sku_id
    JOIN runtime_params rp
      ON oi.dt BETWEEN rp.d365 AND rp.biz_date
    WHERE oi.user_id IS NOT NULL
      AND coalesce(oi.is_test_order, 0) = 0
),

/* ============================================================
   17. 支付流水：跨库支付域
   ============================================================ */
payment_raw AS (
    SELECT
        p.user_id,
        p.order_id,
        p.payment_id,
        p.payment_channel,
        p.payment_method,
        p.bank_code,
        p.pay_amount,
        p.pay_status,
        p.pay_time,
        p.risk_decision,
        p.risk_score AS pay_risk_score,
        p.installment_num,
        p.dt
    FROM dwd_pay.dwd_payment_flow_df p
    JOIN runtime_params rp
      ON p.dt BETWEEN rp.d365 AND rp.biz_date
    WHERE p.user_id IS NOT NULL
),

/* ============================================================
   18. 订单退款：跨库售后域
   ============================================================ */
refund_raw AS (
    SELECT
        r.user_id,
        r.order_id,
        r.order_item_id,
        r.refund_id,
        r.refund_type,
        r.refund_reason_code,
        r.refund_reason_desc,
        r.apply_time,
        r.audit_time,
        r.refund_success_time,
        r.refund_amount,
        r.refund_status,
        r.is_abnormal_refund,
        r.dt
    FROM dwd_trade.dwd_refund_order_df r
    JOIN runtime_params rp
      ON r.dt BETWEEN rp.d365 AND rp.biz_date
    WHERE r.user_id IS NOT NULL
),

/* ============================================================
   19. 履约配送：跨库物流域
   ============================================================ */
fulfillment_raw AS (
    SELECT
        f.user_id,
        f.order_id,
        f.store_id,
        f.warehouse_id,
        f.delivery_type,
        f.carrier_code,
        f.promise_delivery_time,
        f.ship_time,
        f.sign_time,
        f.cancel_time,
        f.fulfillment_status,
        f.is_timeout,
        f.distance_km,
        f.delivery_fee,
        f.dt
    FROM dwd_fulfillment.dwd_order_fulfillment_df f
    JOIN runtime_params rp
      ON f.dt BETWEEN rp.d365 AND rp.biz_date
    WHERE f.user_id IS NOT NULL
),

/* ============================================================
   20. 订单宽明细：订单 + 支付 + 退款 + 履约
   ============================================================ */
order_item_enriched AS (
    SELECT
        oi.*,

        pr.payment_id,
        pr.payment_channel,
        pr.bank_code,
        pr.pay_risk_score,
        pr.installment_num,

        rr.refund_id,
        rr.refund_type,
        rr.refund_reason_code,
        rr.refund_reason_desc,
        rr.refund_amount,
        rr.is_abnormal_refund,
        rr.refund_success_time,

        fr.warehouse_id,
        fr.carrier_code,
        fr.promise_delivery_time,
        fr.ship_time,
        fr.sign_time,
        fr.fulfillment_status,
        fr.is_timeout AS fulfillment_is_timeout,
        fr.distance_km,
        fr.delivery_fee,

        unix_timestamp(oi.pay_time) - unix_timestamp(oi.order_time) AS order_to_pay_sec,
        unix_timestamp(fr.sign_time) - unix_timestamp(oi.pay_time) AS pay_to_sign_sec,

        oi.pay_amount - coalesce(oi.cost_amount, oi.quantity * oi.sku_cost_price, 0) AS gross_profit_amount,

        CASE
            WHEN oi.pay_status = 'PAID'
             AND coalesce(rr.refund_amount, 0) = 0 THEN oi.pay_amount
            WHEN oi.pay_status = 'PAID'
             AND coalesce(rr.refund_amount, 0) > 0 THEN greatest(oi.pay_amount - rr.refund_amount, 0)
            ELSE 0
        END AS net_pay_amount

    FROM order_item_raw oi
    LEFT JOIN payment_raw pr
      ON oi.user_id = pr.user_id
     AND oi.order_id = pr.order_id
    LEFT JOIN refund_raw rr
      ON oi.user_id = rr.user_id
     AND oi.order_item_id = rr.order_item_id
    LEFT JOIN fulfillment_raw fr
      ON oi.user_id = fr.user_id
     AND oi.order_id = fr.order_id
),

/* ============================================================
   21. 用户交易聚合：365/180/90/30/7 多窗口
   ============================================================ */
trade_user_agg AS (
    SELECT
        user_id,

        count(DISTINCT order_id) AS order_cnt_365d,
        count(DISTINCT if(dt >= date_sub('${biz_date}', 180), order_id, NULL)) AS order_cnt_180d,
        count(DISTINCT if(dt >= date_sub('${biz_date}', 90), order_id, NULL)) AS order_cnt_90d,
        count(DISTINCT if(dt >= date_sub('${biz_date}', 30), order_id, NULL)) AS order_cnt_30d,
        count(DISTINCT if(dt >= date_sub('${biz_date}', 7), order_id, NULL)) AS order_cnt_7d,

        count(DISTINCT if(pay_status = 'PAID', order_id, NULL)) AS paid_order_cnt_365d,
        count(DISTINCT if(pay_status = 'PAID' AND dt >= date_sub('${biz_date}', 90), order_id, NULL)) AS paid_order_cnt_90d,
        count(DISTINCT if(pay_status = 'PAID' AND dt >= date_sub('${biz_date}', 30), order_id, NULL)) AS paid_order_cnt_30d,

        count(DISTINCT if(refund_id IS NOT NULL, order_id, NULL)) AS refund_order_cnt_365d,
        count(DISTINCT if(refund_id IS NOT NULL AND dt >= date_sub('${biz_date}', 90), order_id, NULL)) AS refund_order_cnt_90d,

        sum(if(pay_status = 'PAID', pay_amount, 0)) AS gross_pay_amount_365d,
        sum(if(pay_status = 'PAID' AND dt >= date_sub('${biz_date}', 180), pay_amount, 0)) AS gross_pay_amount_180d,
        sum(if(pay_status = 'PAID' AND dt >= date_sub('${biz_date}', 90), pay_amount, 0)) AS gross_pay_amount_90d,
        sum(if(pay_status = 'PAID' AND dt >= date_sub('${biz_date}', 30), pay_amount, 0)) AS gross_pay_amount_30d,
        sum(if(pay_status = 'PAID' AND dt >= date_sub('${biz_date}', 7), pay_amount, 0)) AS gross_pay_amount_7d,

        sum(net_pay_amount) AS net_pay_amount_365d,
        sum(if(dt >= date_sub('${biz_date}', 90), net_pay_amount, 0)) AS net_pay_amount_90d,
        sum(if(dt >= date_sub('${biz_date}', 30), net_pay_amount, 0)) AS net_pay_amount_30d,

        sum(gross_profit_amount) AS gross_profit_amount_365d,
        sum(if(dt >= date_sub('${biz_date}', 90), gross_profit_amount, 0)) AS gross_profit_amount_90d,

        sum(quantity) AS item_quantity_365d,
        sum(if(dt >= date_sub('${biz_date}', 90), quantity, 0)) AS item_quantity_90d,

        sum(discount_amount) AS discount_amount_365d,
        sum(platform_coupon_amount) AS platform_coupon_amount_365d,
        sum(shop_coupon_amount) AS shop_coupon_amount_365d,
        sum(points_deduction_amount) AS points_deduction_amount_365d,

        round(
            sum(discount_amount)
            / greatest(sum(goods_amount), 1),
            8
        ) AS discount_rate_365d,

        round(
            count(DISTINCT if(refund_id IS NOT NULL, order_id, NULL))
            / greatest(count(DISTINCT if(pay_status = 'PAID', order_id, NULL)), 1),
            8
        ) AS refund_order_rate_365d,

        round(
            sum(coalesce(refund_amount, 0))
            / greatest(sum(if(pay_status = 'PAID', pay_amount, 0)), 1),
            8
        ) AS refund_amount_rate_365d,

        count(DISTINCT sku_id) AS bought_sku_cnt_365d,
        count(DISTINCT spu_id) AS bought_spu_cnt_365d,
        count(DISTINCT brand_id) AS bought_brand_cnt_365d,
        count(DISTINCT category_level1_id) AS bought_cate1_cnt_365d,
        count(DISTINCT shop_id) AS bought_shop_cnt_365d,
        count(DISTINCT store_id) AS bought_store_cnt_365d,
        count(DISTINCT order_province) AS order_province_cnt_365d,
        count(DISTINCT order_city) AS order_city_cnt_365d,
        count(DISTINCT address_hash) AS address_cnt_365d,

        max(pay_time) AS last_pay_time,
        min(pay_time) AS first_pay_time,
        datediff('${biz_date}', substr(max(pay_time), 1, 10)) AS days_since_last_pay,

        avg(order_to_pay_sec) AS avg_order_to_pay_sec_365d,
        percentile_approx(order_to_pay_sec, 0.5) AS median_order_to_pay_sec_365d,
        avg(pay_to_sign_sec) AS avg_pay_to_sign_sec_365d,
        percentile_approx(pay_to_sign_sec, 0.5) AS median_pay_to_sign_sec_365d,

        sum(if(fulfillment_is_timeout = 1, 1, 0)) AS timeout_fulfillment_cnt_365d,
        round(
            sum(if(fulfillment_is_timeout = 1, 1, 0))
            / greatest(count(DISTINCT order_id), 1),
            8
        ) AS timeout_fulfillment_rate_365d,

        concat_ws(',', sort_array(collect_set(payment_method))) AS payment_method_set_365d,
        concat_ws(',', sort_array(collect_set(payment_channel))) AS payment_channel_set_365d,
        concat_ws(',', sort_array(collect_set(delivery_type))) AS delivery_type_set_365d,
        concat_ws(',', sort_array(collect_set(price_band_code))) AS bought_price_band_set_365d

    FROM order_item_enriched
    GROUP BY user_id
),

/* ============================================================
   22. 商品偏好排名：多指标打分
   ============================================================ */
user_sku_preference_score AS (
    SELECT
        user_id,
        sku_id,
        max(product_name) AS product_name,
        max(category_level1_id) AS category_level1_id,
        max(category_level1_name) AS category_level1_name,
        max(category_level2_id) AS category_level2_id,
        max(category_level2_name) AS category_level2_name,
        max(category_level3_id) AS category_level3_id,
        max(category_level3_name) AS category_level3_name,
        max(brand_id) AS brand_id,
        max(brand_name) AS brand_name,
        max(price_band_code) AS price_band_code,
        count(DISTINCT order_id) AS sku_order_cnt_365d,
        sum(net_pay_amount) AS sku_net_pay_amount_365d,
        (
            log(1 + sum(net_pay_amount)) * 0.45
            + log(1 + count(DISTINCT order_id)) * 0.25
            + log(1 + sum(quantity)) * 0.15
            + 1 / greatest(datediff('${biz_date}', substr(max(pay_time), 1, 10)), 1) * 15
        ) AS sku_preference_score
    FROM order_item_enriched
    WHERE pay_status = 'PAID'
    GROUP BY user_id, sku_id
),

user_sku_preference_rank AS (
    SELECT
        *,
        row_number() OVER (
            PARTITION BY user_id
            ORDER BY sku_preference_score DESC,
                     sku_net_pay_amount_365d DESC,
                     sku_order_cnt_365d DESC
        ) AS sku_pref_rn,
        dense_rank() OVER (
            PARTITION BY user_id
            ORDER BY category_level1_id
        ) AS cate_dense_rank
    FROM user_sku_preference_score
),

user_sku_preference_agg AS (
    SELECT
        user_id,

        max(if(sku_pref_rn = 1, sku_id, NULL)) AS top1_sku_id,
        max(if(sku_pref_rn = 1, product_name, NULL)) AS top1_sku_name,
        max(if(sku_pref_rn = 1, sku_preference_score, NULL)) AS top1_sku_preference_score,

        max(if(sku_pref_rn = 2, sku_id, NULL)) AS top2_sku_id,
        max(if(sku_pref_rn = 2, product_name, NULL)) AS top2_sku_name,
        max(if(sku_pref_rn = 2, sku_preference_score, NULL)) AS top2_sku_preference_score,

        max(if(sku_pref_rn = 3, sku_id, NULL)) AS top3_sku_id,
        max(if(sku_pref_rn = 3, product_name, NULL)) AS top3_sku_name,
        max(if(sku_pref_rn = 3, sku_preference_score, NULL)) AS top3_sku_preference_score,

        concat_ws(
            '||',
            collect_list(
                concat(
                    cast(sku_pref_rn AS string),
                    ':',
                    sku_id,
                    ':',
                    regexp_replace(product_name, '[:|,]', '_'),
                    ':',
                    cast(round(sku_preference_score, 6) AS string)
                )
            )
        ) AS top20_sku_preference_path

    FROM user_sku_preference_rank
    WHERE sku_pref_rn <= 20
    GROUP BY user_id
),

/* ============================================================
   23. 类目偏好：行转列 + 排名
   ============================================================ */
user_category_score AS (
    SELECT
        user_id,
        category_level1_id,
        category_level1_name,
        sum(net_pay_amount) AS cate_net_pay_amount_365d,
        (
            log(1 + sum(net_pay_amount)) * 0.5
            + log(1 + count(DISTINCT order_id)) * 0.25
            + log(1 + count(DISTINCT sku_id)) * 0.15
            + log(1 + sum(quantity)) * 0.1
        ) AS cate_preference_score
    FROM order_item_enriched
    WHERE pay_status = 'PAID'
    GROUP BY user_id, category_level1_id, category_level1_name
),

user_category_rank AS (
    SELECT
        *,
        row_number() OVER (
            PARTITION BY user_id
            ORDER BY cate_preference_score DESC,
                     cate_net_pay_amount_365d DESC
        ) AS cate_rn
    FROM user_category_score
),

user_category_pivot AS (
    SELECT
        user_id,
        max(if(cate_rn = 1, category_level1_id, NULL)) AS top1_cate1_id,
        max(if(cate_rn = 1, category_level1_name, NULL)) AS top1_cate1_name,
        max(if(cate_rn = 1, cate_preference_score, NULL)) AS top1_cate1_score,

        max(if(cate_rn = 2, category_level1_id, NULL)) AS top2_cate1_id,
        max(if(cate_rn = 2, category_level1_name, NULL)) AS top2_cate1_name,
        max(if(cate_rn = 2, cate_preference_score, NULL)) AS top2_cate1_score,

        max(if(cate_rn = 3, category_level1_id, NULL)) AS top3_cate1_id,
        max(if(cate_rn = 3, category_level1_name, NULL)) AS top3_cate1_name,
        max(if(cate_rn = 3, cate_preference_score, NULL)) AS top3_cate1_score,

        concat_ws(
            ',',
            collect_list(
                concat(
                    cast(cate_rn AS string),
                    ':',
                    category_level1_id,
                    ':',
                    category_level1_name,
                    ':',
                    cast(round(cate_preference_score, 4) AS string)
                )
            )
        ) AS top10_cate1_preference_path
    FROM user_category_rank
    WHERE cate_rn <= 10
    GROUP BY user_id
),

/* ============================================================
   24. 营销触点：跨库营销触达、广告点击、站内弹窗
   ============================================================ */
marketing_touch_raw AS (
    SELECT
        user_id,
        touch_id,
        campaign_id,
        campaign_name,
        material_id,
        strategy_id,
        scene,
        channel,
        sub_channel,
        touch_time,
        touch_ts,
        dt,
        get_json_object(ext_json, '$.bid_type') AS bid_type,
        get_json_object(ext_json, '$.cost') AS touch_cost,
        get_json_object(ext_json, '$.audience_pkg_id') AS audience_pkg_id,
        get_json_object(ext_json, '$.creative_type') AS creative_type
    FROM ads_marketing.ads_user_campaign_touch_di
    WHERE dt BETWEEN date_sub('${biz_date}', 90) AND '${biz_date}'
      AND user_id IS NOT NULL

    UNION ALL

    SELECT
        user_id,
        concat('AD_', ad_click_id) AS touch_id,
        campaign_id,
        campaign_name,
        creative_id AS material_id,
        strategy_id,
        ad_scene AS scene,
        media_channel AS channel,
        media_sub_channel AS sub_channel,
        click_time AS touch_time,
        click_ts AS touch_ts,
        dt,
        bid_type,
        cast(cost_amount AS string) AS touch_cost,
        audience_pkg_id,
        creative_type
    FROM dwd_ad.dwd_ad_click_di
    WHERE dt BETWEEN date_sub('${biz_date}', 90) AND '${biz_date}'
      AND user_id IS NOT NULL

    UNION ALL

    SELECT
        user_id,
        concat('POP_', popup_id) AS touch_id,
        campaign_id,
        campaign_name,
        material_id,
        strategy_id,
        popup_scene AS scene,
        'APP_POPUP' AS channel,
        popup_type AS sub_channel,
        show_time AS touch_time,
        show_ts AS touch_ts,
        dt,
        'NONE' AS bid_type,
        '0' AS touch_cost,
        audience_pkg_id,
        creative_type
    FROM dwd_marketing.dwd_app_popup_show_di
    WHERE dt BETWEEN date_sub('${biz_date}', 90) AND '${biz_date}'
      AND user_id IS NOT NULL
),

/* ============================================================
   25. 营销触点加权：时间衰减 + 渠道权重
   ============================================================ */
marketing_touch_weighted AS (
    SELECT
        mtr.*,
        coalesce(cwr.channel_group, '未知渠道') AS channel_group,
        coalesce(cast(mtr.touch_cost AS double), 0) AS touch_cost_double,
        (
            coalesce(cwr.base_weight, 0.01)
            * pow(0.85, datediff('${biz_date}', substr(mtr.touch_time, 1, 10)))
            * CASE
                WHEN mtr.scene IN ('NEW_USER_COUPON', 'PRICE_DROP', 'RECALL') THEN 1.25
                WHEN mtr.scene IN ('BRAND_CAMPAIGN', 'BIG_PROMOTION') THEN 1.15
                ELSE 1.0
              END
        ) AS attribution_weight
    FROM marketing_touch_raw mtr
    LEFT JOIN channel_weight_rule cwr
      ON upper(mtr.channel) = cwr.channel
),

/* ============================================================
   26. 营销归因：触点与支付订单按时间窗口匹配
   ============================================================ */
marketing_order_match AS (
    SELECT
        mtw.user_id,
        mtw.touch_id,
        mtw.campaign_id,
        mtw.campaign_name,
        mtw.material_id,
        mtw.strategy_id,
        mtw.scene,
        mtw.channel,
        mtw.channel_group,
        mtw.touch_time,
        mtw.touch_ts,
        mtw.attribution_weight,
        mtw.touch_cost_double,

        oie.order_id,
        oie.order_item_id,
        oie.sku_id,
        oie.category_level1_id,
        oie.brand_id,
        oie.pay_time,
        oie.pay_amount,
        oie.net_pay_amount,
        unix_timestamp(oie.pay_time) - mtw.touch_ts AS touch_to_pay_sec,

        row_number() OVER (
            PARTITION BY oie.user_id, oie.order_item_id
            ORDER BY
                mtw.attribution_weight DESC,
                mtw.touch_ts DESC
        ) AS attribution_rn,

        sum(mtw.attribution_weight) OVER (
            PARTITION BY oie.user_id, oie.order_item_id
        ) AS total_candidate_weight

    FROM marketing_touch_weighted mtw
    JOIN order_item_enriched oie
      ON mtw.user_id = oie.user_id
     AND oie.pay_status = 'PAID'
     AND unix_timestamp(oie.pay_time) >= mtw.touch_ts
     AND unix_timestamp(oie.pay_time) - mtw.touch_ts BETWEEN 0 AND 7 * 24 * 3600
),

marketing_attribution_final AS (
    SELECT
        user_id,
        touch_id,
        campaign_id,
        campaign_name,
        material_id,
        strategy_id,
        scene,
        channel,
        channel_group,
        order_id,
        order_item_id,
        sku_id,
        category_level1_id,
        brand_id,
        pay_time,
        pay_amount,
        net_pay_amount,
        attribution_weight,
        total_candidate_weight,
        CASE
            WHEN total_candidate_weight > 0
                THEN net_pay_amount * attribution_weight / total_candidate_weight
            ELSE 0
        END AS attributed_net_pay_amount,
        touch_cost_double,
        touch_to_pay_sec,
        attribution_rn
    FROM marketing_order_match
),

marketing_user_agg AS (
    SELECT
        user_id,

        count(DISTINCT touch_id) AS marketing_touch_cnt_90d,
        count(DISTINCT campaign_id) AS marketing_campaign_cnt_90d,
        count(DISTINCT material_id) AS marketing_material_cnt_90d,
        count(DISTINCT strategy_id) AS marketing_strategy_cnt_90d,
        count(DISTINCT channel) AS marketing_channel_cnt_90d,
        concat_ws(',', sort_array(collect_set(channel))) AS marketing_channel_set_90d,
        concat_ws(',', sort_array(collect_set(channel_group))) AS marketing_channel_group_set_90d,

        sum(touch_cost_double) AS marketing_touch_cost_90d,
        sum(attributed_net_pay_amount) AS attributed_net_pay_amount_90d,

        round(
            sum(attributed_net_pay_amount)
            / greatest(sum(touch_cost_double), 1),
            8
        ) AS marketing_roi_90d,

        avg(touch_to_pay_sec) AS avg_touch_to_pay_sec_90d,
        percentile_approx(touch_to_pay_sec, 0.5) AS median_touch_to_pay_sec_90d,

        max(if(attribution_rn = 1, campaign_id, NULL)) AS last_effective_campaign_id,
        max(if(attribution_rn = 1, campaign_name, NULL)) AS last_effective_campaign_name

    FROM marketing_attribution_final
    GROUP BY user_id
),

/* ============================================================
   27. 优惠券：领取、使用、过期、补贴
   ============================================================ */
coupon_raw AS (
    SELECT
        c.user_id,
        c.coupon_id,
        c.batch_id,
        c.coupon_type,
        c.coupon_status,
        c.receive_time,
        c.use_time,
        c.expire_time,
        c.order_id,
        c.discount_amount,
        c.threshold_amount,
        c.platform_bear_amount,
        c.shop_bear_amount,
        c.dt
    FROM dwd_promotion.dwd_user_coupon_df c
    WHERE c.dt BETWEEN date_sub('${biz_date}', 180) AND '${biz_date}'
      AND c.user_id IS NOT NULL
),

coupon_user_agg AS (
    SELECT
        user_id,
        count(DISTINCT coupon_id) AS coupon_receive_cnt_180d,
        count(DISTINCT if(coupon_status = 'USED', coupon_id, NULL)) AS coupon_used_cnt_180d,
        count(DISTINCT if(coupon_status = 'EXPIRED', coupon_id, NULL)) AS coupon_expired_cnt_180d,
        count(DISTINCT if(coupon_status = 'UNUSED', coupon_id, NULL)) AS coupon_unused_cnt_180d,
        sum(if(coupon_status = 'USED', discount_amount, 0)) AS coupon_discount_amount_180d,
        sum(if(coupon_status = 'USED', platform_bear_amount, 0)) AS platform_coupon_subsidy_180d,
        sum(if(coupon_status = 'USED', shop_bear_amount, 0)) AS shop_coupon_subsidy_180d,
        avg(if(coupon_status = 'USED', threshold_amount, NULL)) AS avg_used_coupon_threshold_180d,
        round(
            count(DISTINCT if(coupon_status = 'USED', coupon_id, NULL))
            / greatest(count(DISTINCT coupon_id), 1),
            8
        ) AS coupon_use_rate_180d,
        round(
            count(DISTINCT if(coupon_status = 'EXPIRED', coupon_id, NULL))
            / greatest(count(DISTINCT coupon_id), 1),
            8
        ) AS coupon_expire_rate_180d,
        max(receive_time) AS last_coupon_receive_time,
        max(if(coupon_status = 'USED', use_time, NULL)) AS last_coupon_use_time
    FROM coupon_raw
    GROUP BY user_id
),

/* ============================================================
   28. 客服工单：投诉、退款、催单、满意度
   ============================================================ */
service_raw AS (
    SELECT
        s.user_id,
        s.ticket_id,
        s.order_id,
        s.ticket_type,
        s.ticket_scene,
        s.ticket_status,
        s.create_time,
        s.assign_time,
        s.solve_time,
        s.close_time,
        s.satisfaction_score,
        s.is_escalated,
        s.agent_id,
        s.dt
    FROM ods_service.ods_customer_ticket_df s
    WHERE s.dt BETWEEN date_sub('${biz_date}', 180) AND '${biz_date}'
      AND s.user_id IS NOT NULL
),

service_user_agg AS (
    SELECT
        user_id,
        count(1) AS service_ticket_cnt_180d,
        sum(if(ticket_type = 'REFUND', 1, 0)) AS refund_service_ticket_cnt_180d,
        sum(if(ticket_type = 'COMPLAINT', 1, 0)) AS complaint_ticket_cnt_180d,
        sum(if(ticket_type = 'DELIVERY_DELAY', 1, 0)) AS delivery_delay_ticket_cnt_180d,
        sum(if(ticket_status = 'SOLVED', 1, 0)) AS solved_ticket_cnt_180d,
        sum(if(is_escalated = 1, 1, 0)) AS escalated_ticket_cnt_180d,
        avg(satisfaction_score) AS avg_service_satisfaction_score_180d,
        percentile_approx(satisfaction_score, 0.5) AS median_service_satisfaction_score_180d,
        avg(unix_timestamp(solve_time) - unix_timestamp(create_time)) AS avg_ticket_solve_sec_180d,
        percentile_approx(unix_timestamp(solve_time) - unix_timestamp(create_time), 0.5) AS median_ticket_solve_sec_180d,
        max(create_time) AS last_service_create_time,
        concat_ws(',', sort_array(collect_set(ticket_type))) AS service_ticket_type_set_180d
    FROM service_raw
    GROUP BY user_id
),

/* ============================================================
   29. 门店维度：线下新零售场景
   ============================================================ */
store_dim AS (
    SELECT
        store_id,
        store_name,
        store_type,
        region_id,
        region_name,
        province,
        city,
        district,
        open_date,
        store_level,
        business_circle,
        manager_id,
        longitude,
        latitude
    FROM dim.dim_store_df
    WHERE dt = '${biz_date}'
),

/* ============================================================
   30. 门店访问 + 门店成交
   ============================================================ */
store_visit_agg AS (
    SELECT
        er.user_id,
        er.store_id,
        count(1) AS store_visit_cnt_90d,
        count(DISTINCT er.dt) AS store_visit_days_90d,
        max(er.event_time) AS last_store_visit_time
    FROM event_raw er
    WHERE er.event_type IN ('store_visit', 'scan_qr')
      AND er.store_id IS NOT NULL
    GROUP BY er.user_id, er.store_id
),

store_trade_agg AS (
    SELECT
        user_id,
        store_id,
        count(DISTINCT order_id) AS store_order_cnt_365d,
        sum(net_pay_amount) AS store_net_pay_amount_365d,
        max(pay_time) AS last_store_pay_time
    FROM order_item_enriched
    WHERE store_id IS NOT NULL
      AND pay_status = 'PAID'
    GROUP BY user_id, store_id
),

store_user_preference AS (
    SELECT
        *
    FROM (
        SELECT
            store_base.*,
            row_number() OVER (
                PARTITION BY user_id
                ORDER BY store_preference_score DESC
            ) AS store_rn
        FROM (
        SELECT
            coalesce(sva.user_id, sta.user_id) AS user_id,
            coalesce(sva.store_id, sta.store_id) AS store_id,
            sd.store_name,
            sd.store_type,
            sd.region_name,
            sd.city AS store_city,
            coalesce(sva.store_visit_cnt_90d, 0) AS store_visit_cnt_90d,
            coalesce(sva.store_visit_days_90d, 0) AS store_visit_days_90d,
            coalesce(sta.store_order_cnt_365d, 0) AS store_order_cnt_365d,
            coalesce(sta.store_net_pay_amount_365d, 0) AS store_net_pay_amount_365d,
            greatest(
                unix_timestamp(coalesce(sva.last_store_visit_time, '1970-01-01 00:00:00')),
                unix_timestamp(coalesce(sta.last_store_pay_time, '1970-01-01 00:00:00'))
            ) AS last_store_interaction_ts,
            (
                log(1 + coalesce(sva.store_visit_cnt_90d, 0)) * 0.25
                + log(1 + coalesce(sta.store_order_cnt_365d, 0)) * 0.30
                + log(1 + coalesce(sta.store_net_pay_amount_365d, 0)) * 0.45
            ) AS store_preference_score
        FROM store_visit_agg sva
        FULL OUTER JOIN store_trade_agg sta
          ON sva.user_id = sta.user_id
         AND sva.store_id = sta.store_id
        LEFT JOIN store_dim sd
          ON coalesce(sva.store_id, sta.store_id) = sd.store_id
        ) store_base
    ) t
    WHERE store_rn <= 5
),

store_user_agg AS (
    SELECT
        user_id,
        max(if(store_rn = 1, store_id, NULL)) AS top1_store_id,
        max(if(store_rn = 1, store_name, NULL)) AS top1_store_name,
        max(if(store_rn = 1, store_preference_score, NULL)) AS top1_store_preference_score,
        concat_ws(
            '||',
            collect_list(
                concat(
                    cast(store_rn AS string),
                    ':',
                    store_id,
                    ':',
                    coalesce(store_name, 'UNKNOWN_STORE'),
                    ':',
                    cast(round(store_preference_score, 4) AS string)
                )
            )
        ) AS top5_store_preference_path
    FROM store_user_preference
    GROUP BY user_id
),

/* ============================================================
   31. 风控设备图谱：设备共用、IP 共用、地址共用
   ============================================================ */
device_user_graph AS (
    SELECT
        device_id,
        count(DISTINCT user_id) AS device_bind_user_cnt_90d,
        concat_ws(',', slice(sort_array(collect_set(user_id)), 1, 20)) AS device_sample_user_list
    FROM event_raw
    WHERE device_id IS NOT NULL
    GROUP BY device_id
),

ip_user_graph AS (
    SELECT
        ip,
        count(DISTINCT user_id) AS ip_user_cnt_90d,
        concat_ws(',', slice(sort_array(collect_set(user_id)), 1, 20)) AS ip_sample_user_list
    FROM event_raw
    WHERE ip IS NOT NULL
    GROUP BY ip
),

address_user_graph AS (
    SELECT
        address_hash,
        count(DISTINCT user_id) AS address_user_cnt_365d,
        concat_ws(',', slice(sort_array(collect_set(user_id)), 1, 20)) AS address_sample_user_list
    FROM order_item_enriched
    WHERE address_hash IS NOT NULL
    GROUP BY address_hash
),

user_event_graph_risk AS (
    SELECT
        er.user_id,
        max(coalesce(dug.device_bind_user_cnt_90d, 0)) AS max_device_bind_user_cnt_90d,
        max(coalesce(iug.ip_user_cnt_90d, 0)) AS max_ip_user_cnt_90d,
        concat_ws(',', sort_array(collect_set(dug.device_sample_user_list))) AS risky_device_related_users,
        concat_ws(',', sort_array(collect_set(iug.ip_sample_user_list))) AS risky_ip_related_users
    FROM event_raw er
    LEFT JOIN device_user_graph dug
      ON er.device_id = dug.device_id
    LEFT JOIN ip_user_graph iug
      ON er.ip = iug.ip
    GROUP BY er.user_id
),

user_address_graph_risk AS (
    SELECT
        oie.user_id,
        max(coalesce(aug.address_user_cnt_365d, 0)) AS max_address_user_cnt_365d,
        concat_ws(',', sort_array(collect_set(aug.address_sample_user_list))) AS risky_address_related_users
    FROM order_item_enriched oie
    LEFT JOIN address_user_graph aug
      ON oie.address_hash = aug.address_hash
    GROUP BY oie.user_id
),

user_graph_risk AS (
    SELECT
        ub.user_id,

        coalesce(uegr.max_device_bind_user_cnt_90d, 0) AS max_device_bind_user_cnt_90d,
        coalesce(uegr.max_ip_user_cnt_90d, 0) AS max_ip_user_cnt_90d,
        coalesce(uagr.max_address_user_cnt_365d, 0) AS max_address_user_cnt_365d,

        uegr.risky_device_related_users,
        uegr.risky_ip_related_users,
        uagr.risky_address_related_users

    FROM user_base ub
    LEFT JOIN user_event_graph_risk uegr
      ON ub.user_id = uegr.user_id
    LEFT JOIN user_address_graph_risk uagr
      ON ub.user_id = uagr.user_id
),

/* ============================================================
   32. 综合风险评分：行为 + 交易 + 图谱 + 支付风控 + 售后
   ============================================================ */
risk_score_calc AS (
    SELECT
        ub.user_id,

        least(
            100,
            greatest(
                0,
                cast(
                    (
                        if(coalesce(ub.is_black_user, 0) = 1, 100, 0)

                        + if(coalesce(eua.device_cnt_90d, 0) >= 10, 20, 0)
                        + if(coalesce(eua.ip_cnt_90d, 0) >= 20, 15, 0)
                        + if(coalesce(ugr.max_device_bind_user_cnt_90d, 0) >= 5, 20, 0)
                        + if(coalesce(ugr.max_ip_user_cnt_90d, 0) >= 10, 15, 0)
                        + if(coalesce(ugr.max_address_user_cnt_365d, 0) >= 5, 15, 0)

                        + if(coalesce(tua.refund_order_rate_365d, 0) >= 0.35, 15, 0)
                        + if(coalesce(tua.refund_amount_rate_365d, 0) >= 0.50, 15, 0)
                        + if(coalesce(sua.complaint_ticket_cnt_180d, 0) >= 3, 10, 0)
                        + if(coalesce(sua.escalated_ticket_cnt_180d, 0) >= 2, 10, 0)

                        + if(ub.email_domain IN ('tempmail.com', 'mailinator.com', 'fake.com', 'trashmail.com'), 15, 0)
                        + if(coalesce(tua.avg_order_to_pay_sec_365d, 999999) < 3, 10, 0)
                    ) AS int
                )
            )
        ) AS risk_score,

        concat_ws(
            ',',
            array(
                if(coalesce(ub.is_black_user, 0) = 1, 'BLACK_USER', NULL),
                if(coalesce(eua.device_cnt_90d, 0) >= 10, 'MANY_DEVICES', NULL),
                if(coalesce(eua.ip_cnt_90d, 0) >= 20, 'MANY_IPS', NULL),
                if(coalesce(ugr.max_device_bind_user_cnt_90d, 0) >= 5, 'DEVICE_GRAPH_RISK', NULL),
                if(coalesce(ugr.max_ip_user_cnt_90d, 0) >= 10, 'IP_GRAPH_RISK', NULL),
                if(coalesce(ugr.max_address_user_cnt_365d, 0) >= 5, 'ADDRESS_GRAPH_RISK', NULL),
                if(coalesce(tua.refund_order_rate_365d, 0) >= 0.35, 'HIGH_REFUND_RATE', NULL),
                if(coalesce(tua.refund_amount_rate_365d, 0) >= 0.50, 'HIGH_REFUND_AMOUNT_RATE', NULL),
                if(ub.email_domain IN ('tempmail.com', 'mailinator.com', 'fake.com', 'trashmail.com'), 'TEMP_EMAIL', NULL)
            )
        ) AS risk_reason_raw

    FROM user_base ub
    LEFT JOIN event_user_agg eua
      ON ub.user_id = eua.user_id
    LEFT JOIN trade_user_agg tua
      ON ub.user_id = tua.user_id
    LEFT JOIN service_user_agg sua
      ON ub.user_id = sua.user_id
    LEFT JOIN user_graph_risk ugr
      ON ub.user_id = ugr.user_id
),

risk_user_agg AS (
    SELECT
        rsc.user_id,
        rsc.risk_score,
        rsr.risk_level,
        rsr.risk_desc,
        regexp_replace(
            regexp_replace(rsc.risk_reason_raw, ',+', ','),
            '^,|,$',
            ''
        ) AS risk_reason_list
    FROM risk_score_calc rsc
    LEFT JOIN risk_score_rule rsr
      ON rsc.risk_score BETWEEN rsr.min_score AND rsr.max_score
),

/* ============================================================
   33. 列转行：把用户核心指标转成长表
   ============================================================ */
user_metric_long AS (
    SELECT
        user_id,
        metric_group,
        metric_name,
        metric_value
    FROM (
        SELECT
            ub.user_id,

            coalesce(eua.event_cnt_90d, 0) AS event_cnt_90d,
            coalesce(eua.session_cnt_90d, 0) AS session_cnt_90d,
            coalesce(eua.expose_click_rate_90d, 0) AS expose_click_rate_90d,
            coalesce(eua.click_cart_rate_90d, 0) AS click_cart_rate_90d,
            coalesce(eua.submit_pay_rate_90d, 0) AS submit_pay_rate_90d,

            coalesce(tua.order_cnt_365d, 0) AS order_cnt_365d,
            coalesce(tua.paid_order_cnt_365d, 0) AS paid_order_cnt_365d,
            coalesce(tua.gross_pay_amount_365d, 0) AS gross_pay_amount_365d,
            coalesce(tua.net_pay_amount_365d, 0) AS net_pay_amount_365d,
            coalesce(tua.refund_order_rate_365d, 0) AS refund_order_rate_365d,

            coalesce(cua.coupon_use_rate_180d, 0) AS coupon_use_rate_180d,
            coalesce(mua.marketing_roi_90d, 0) AS marketing_roi_90d,
            coalesce(rua.risk_score, 0) AS risk_score

        FROM user_base ub
        LEFT JOIN event_user_agg eua ON ub.user_id = eua.user_id
        LEFT JOIN trade_user_agg tua ON ub.user_id = tua.user_id
        LEFT JOIN coupon_user_agg cua ON ub.user_id = cua.user_id
        LEFT JOIN marketing_user_agg mua ON ub.user_id = mua.user_id
        LEFT JOIN risk_user_agg rua ON ub.user_id = rua.user_id
    ) x
    LATERAL VIEW stack(
        13,
        'behavior',  'event_cnt_90d',             cast(event_cnt_90d AS double),
        'behavior',  'session_cnt_90d',           cast(session_cnt_90d AS double),
        'behavior',  'expose_click_rate_90d',     cast(expose_click_rate_90d AS double),
        'behavior',  'click_cart_rate_90d',       cast(click_cart_rate_90d AS double),
        'behavior',  'submit_pay_rate_90d',       cast(submit_pay_rate_90d AS double),

        'trade',     'order_cnt_365d',            cast(order_cnt_365d AS double),
        'trade',     'paid_order_cnt_365d',       cast(paid_order_cnt_365d AS double),
        'trade',     'gross_pay_amount_365d',     cast(gross_pay_amount_365d AS double),
        'trade',     'net_pay_amount_365d',       cast(net_pay_amount_365d AS double),
        'trade',     'refund_order_rate_365d',    cast(refund_order_rate_365d AS double),

        'coupon',    'coupon_use_rate_180d',      cast(coupon_use_rate_180d AS double),
        'marketing', 'marketing_roi_90d',         cast(marketing_roi_90d AS double),
        'risk',      'risk_score',                cast(risk_score AS double)
    ) s AS metric_group, metric_name, metric_value
),

/* ============================================================
   34. 指标长表转 map：复杂 map 聚合
   ============================================================ */
user_metric_map AS (
    SELECT
        user_id,
        str_to_map(
            concat_ws(
                ',',
                collect_list(
                    concat(metric_name, ':', cast(round(metric_value, 8) AS string))
                )
            ),
            ',',
            ':'
        ) AS metric_value_map,
        str_to_map(
            concat_ws(
                ',',
                collect_list(
                    concat(metric_name, ':', metric_group)
                )
            ),
            ',',
            ':'
        ) AS metric_group_map
    FROM user_metric_long
    GROUP BY user_id
),

/* ============================================================
   35. 行转列：event_type pivot
   ============================================================ */
event_type_pivot AS (
    SELECT
        user_id,
        max(if(event_type = 'app_start', metric_cnt, 0)) AS pivot_app_start_cnt_90d,
        max(if(event_type = 'page_view', metric_cnt, 0)) AS pivot_page_view_cnt_90d,
        max(if(event_type = 'search', metric_cnt, 0)) AS pivot_search_cnt_90d,
        max(if(event_type = 'sku_expose', metric_cnt, 0)) AS pivot_sku_expose_cnt_90d,
        max(if(event_type = 'sku_click', metric_cnt, 0)) AS pivot_sku_click_cnt_90d,
        max(if(event_type = 'add_cart', metric_cnt, 0)) AS pivot_add_cart_cnt_90d,
        max(if(event_type = 'remove_cart', metric_cnt, 0)) AS pivot_remove_cart_cnt_90d,
        max(if(event_type = 'favorite', metric_cnt, 0)) AS pivot_favorite_cnt_90d,
        max(if(event_type = 'unfavorite', metric_cnt, 0)) AS pivot_unfavorite_cnt_90d,
        max(if(event_type = 'coupon_receive', metric_cnt, 0)) AS pivot_coupon_receive_cnt_90d,
        max(if(event_type = 'coupon_use', metric_cnt, 0)) AS pivot_coupon_use_cnt_90d,
        max(if(event_type = 'submit_order', metric_cnt, 0)) AS pivot_submit_order_cnt_90d,
        max(if(event_type = 'pay_success', metric_cnt, 0)) AS pivot_pay_success_cnt_90d,
        max(if(event_type = 'refund_apply', metric_cnt, 0)) AS pivot_refund_apply_cnt_90d,
        max(if(event_type = 'share', metric_cnt, 0)) AS pivot_share_cnt_90d,
        max(if(event_type = 'comment', metric_cnt, 0)) AS pivot_comment_cnt_90d,
        max(if(event_type = 'live_enter', metric_cnt, 0)) AS pivot_live_enter_cnt_90d,
        max(if(event_type = 'store_visit', metric_cnt, 0)) AS pivot_store_visit_cnt_90d
    FROM (
        SELECT
            user_id,
            event_type,
            count(1) AS metric_cnt
        FROM event_raw
        GROUP BY user_id, event_type
    ) t
    GROUP BY user_id
),

/* ============================================================
   36. AB 实验：用户实验组组合
   ============================================================ */
ab_exp_user_agg AS (
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
        ) AS ab_exp_group_list_90d,
        count(DISTINCT exp_id) AS ab_exp_cnt_90d
    FROM event_raw
    WHERE exp_id IS NOT NULL
    GROUP BY user_id
),

/* ============================================================
   37. 标签规则虚拟表
   ============================================================ */
tag_rule AS (
    SELECT stack(
        16,
        'SUPER_VALUE_USER',       '超级高价值用户',       'net_pay_amount_365d >= 5000',
        'HIGH_VALUE_USER',        '高价值用户',           'net_pay_amount_365d >= 1000',
        'NEW_ACTIVE_USER',        '新晋活跃用户',         'register_days <= 30 and event_cnt_7d > 0',
        'SILENT_USER',            '沉默用户',             'days_since_last_event >= 30',
        'LOST_USER',              '流失用户',             'days_since_last_event >= 90',
        'HIGH_RISK_USER',         '高风险用户',           'risk_score >= 70',
        'REFUND_HEAVY_USER',      '退款偏高用户',         'refund_order_rate_365d >= 0.35',
        'COUPON_SENSITIVE_USER',  '优惠券敏感用户',       'coupon_use_rate_180d >= 0.6',
        'CONTENT_MEDIA_USER',     '内容渠道用户',         'register_channel_group = CONTENT_MEDIA',
        'OFFLINE_STORE_USER',     '线下门店用户',         'store_visit_cnt_90d > 0',
        'LIVE_USER',              '直播用户',             'live_enter_cnt_90d > 0',
        'SEARCH_HEAVY_USER',      '搜索重度用户',         'search_cnt_90d >= 20',
        'CART_LOST_USER',         '加购流失用户',         'add_cart_cnt_90d > pay_success_event_cnt_90d',
        'MULTI_DEVICE_USER',      '多设备用户',           'device_cnt_90d >= 3',
        'HIGH_COMPLAINT_USER',    '高投诉用户',           'complaint_ticket_cnt_180d >= 3',
        'MARKETING_POSITIVE_USER','营销正反馈用户',       'marketing_roi_90d >= 1'
    ) AS (
        tag_code,
        tag_name,
        tag_expr
    )
),

/* ============================================================
   38. 用户标签计算：规则 CASE 化
   ============================================================ */
user_tag_long AS (
    SELECT
        x.user_id,
        tr.tag_code,
        tr.tag_name
    FROM (
        SELECT
            ub.user_id,
            ub.register_days,
            ub.register_channel_group,
            coalesce(eua.event_cnt_7d, 0) AS event_cnt_7d,
            coalesce(eua.days_since_last_event, 9999) AS days_since_last_event,
            coalesce(eua.live_enter_cnt_90d, 0) AS live_enter_cnt_90d,
            coalesce(eua.search_cnt_90d, 0) AS search_cnt_90d,
            coalesce(eua.add_cart_cnt_90d, 0) AS add_cart_cnt_90d,
            coalesce(eua.pay_success_event_cnt_90d, 0) AS pay_success_event_cnt_90d,
            coalesce(eua.device_cnt_90d, 0) AS device_cnt_90d,
            coalesce(tua.net_pay_amount_365d, 0) AS net_pay_amount_365d,
            coalesce(tua.refund_order_rate_365d, 0) AS refund_order_rate_365d,
            coalesce(cua.coupon_use_rate_180d, 0) AS coupon_use_rate_180d,
            coalesce(rua.risk_score, 0) AS risk_score,
            coalesce(sua.complaint_ticket_cnt_180d, 0) AS complaint_ticket_cnt_180d,
            coalesce(mua.marketing_roi_90d, 0) AS marketing_roi_90d,
            if(sup.top1_store_id IS NOT NULL, 1, 0) AS has_store_visit_or_trade
        FROM user_base ub
        LEFT JOIN event_user_agg eua ON ub.user_id = eua.user_id
        LEFT JOIN trade_user_agg tua ON ub.user_id = tua.user_id
        LEFT JOIN coupon_user_agg cua ON ub.user_id = cua.user_id
        LEFT JOIN risk_user_agg rua ON ub.user_id = rua.user_id
        LEFT JOIN service_user_agg sua ON ub.user_id = sua.user_id
        LEFT JOIN marketing_user_agg mua ON ub.user_id = mua.user_id
        LEFT JOIN store_user_agg sup ON ub.user_id = sup.user_id
    ) x
    CROSS JOIN tag_rule tr
    WHERE
        CASE tr.tag_code
            WHEN 'SUPER_VALUE_USER'
                THEN if(x.net_pay_amount_365d >= 5000, 1, 0)
            WHEN 'HIGH_VALUE_USER'
                THEN if(x.net_pay_amount_365d >= 1000, 1, 0)
            WHEN 'NEW_ACTIVE_USER'
                THEN if(x.register_days <= 30 AND x.event_cnt_7d > 0, 1, 0)
            WHEN 'SILENT_USER'
                THEN if(x.days_since_last_event >= 30 AND x.days_since_last_event < 90, 1, 0)
            WHEN 'LOST_USER'
                THEN if(x.days_since_last_event >= 90, 1, 0)
            WHEN 'HIGH_RISK_USER'
                THEN if(x.risk_score >= 70, 1, 0)
            WHEN 'REFUND_HEAVY_USER'
                THEN if(x.refund_order_rate_365d >= 0.35, 1, 0)
            WHEN 'COUPON_SENSITIVE_USER'
                THEN if(x.coupon_use_rate_180d >= 0.6, 1, 0)
            WHEN 'CONTENT_MEDIA_USER'
                THEN if(x.register_channel_group = 'CONTENT_MEDIA', 1, 0)
            WHEN 'OFFLINE_STORE_USER'
                THEN if(x.has_store_visit_or_trade = 1, 1, 0)
            WHEN 'LIVE_USER'
                THEN if(x.live_enter_cnt_90d > 0, 1, 0)
            WHEN 'SEARCH_HEAVY_USER'
                THEN if(x.search_cnt_90d >= 20, 1, 0)
            WHEN 'CART_LOST_USER'
                THEN if(x.add_cart_cnt_90d > x.pay_success_event_cnt_90d, 1, 0)
            WHEN 'MULTI_DEVICE_USER'
                THEN if(x.device_cnt_90d >= 3, 1, 0)
            WHEN 'HIGH_COMPLAINT_USER'
                THEN if(x.complaint_ticket_cnt_180d >= 3, 1, 0)
            WHEN 'MARKETING_POSITIVE_USER'
                THEN if(x.marketing_roi_90d >= 1, 1, 0)
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
   39. 生命周期最终修正：结合活跃与交易
   ============================================================ */
user_life_cycle_final AS (
    SELECT
        ub.user_id,
        CASE
            WHEN coalesce(eua.days_since_last_event, 9999) >= 90
             AND coalesce(tua.days_since_last_pay, 9999) >= 90 THEN 'LOST_USER'
            WHEN coalesce(eua.days_since_last_event, 9999) >= 30
             AND coalesce(eua.days_since_last_event, 9999) < 90 THEN 'SILENT_USER'
            WHEN ub.register_days <= 7 THEN 'NEW_USER'
            WHEN ub.register_days <= 30 THEN 'GROWING_USER'
            WHEN coalesce(eua.event_cnt_30d, 0) >= 10
              OR coalesce(tua.paid_order_cnt_30d, 0) >= 1 THEN 'ACTIVE_USER'
            WHEN ub.register_days >= 180 THEN 'MATURE_USER'
            ELSE coalesce(ulcp.register_life_cycle_code, 'UNKNOWN_USER')
        END AS final_life_cycle_code,

        CASE
            WHEN coalesce(eua.days_since_last_event, 9999) >= 90
             AND coalesce(tua.days_since_last_pay, 9999) >= 90 THEN '流失用户'
            WHEN coalesce(eua.days_since_last_event, 9999) >= 30
             AND coalesce(eua.days_since_last_event, 9999) < 90 THEN '沉默用户'
            WHEN ub.register_days <= 7 THEN '新用户'
            WHEN ub.register_days <= 30 THEN '成长期用户'
            WHEN coalesce(eua.event_cnt_30d, 0) >= 10
              OR coalesce(tua.paid_order_cnt_30d, 0) >= 1 THEN '活跃用户'
            WHEN ub.register_days >= 180 THEN '成熟用户'
            ELSE coalesce(ulcp.register_life_cycle_name, '未知用户')
        END AS final_life_cycle_name

    FROM user_base ub
    LEFT JOIN user_life_cycle_pre ulcp
      ON ub.user_id = ulcp.user_id
    LEFT JOIN event_user_agg eua
      ON ub.user_id = eua.user_id
    LEFT JOIN trade_user_agg tua
      ON ub.user_id = tua.user_id
),

/* ============================================================
   40. 经营汇总：GROUPING SETS 多粒度汇总
   ============================================================ */
biz_summary_grouping_sets AS (
    SELECT
        coalesce(ub.province, 'ALL') AS province,
        coalesce(ub.city, 'ALL') AS city,
        coalesce(ub.register_channel_group, 'ALL') AS register_channel_group,
        coalesce(ulcf.final_life_cycle_code, 'ALL') AS life_cycle_code,
        coalesce(rua.risk_level, 'ALL') AS risk_level,

        grouping__id AS grouping_id,

        count(DISTINCT ub.user_id) AS user_cnt,
        sum(coalesce(tua.net_pay_amount_365d, 0)) AS total_net_pay_amount_365d,
        sum(coalesce(tua.gross_profit_amount_365d, 0)) AS total_gross_profit_amount_365d,
        avg(coalesce(tua.net_pay_amount_365d, 0)) AS avg_user_net_pay_amount_365d,
        sum(coalesce(eua.event_cnt_90d, 0)) AS total_event_cnt_90d,
        sum(coalesce(cua.platform_coupon_subsidy_180d, 0)) AS total_platform_coupon_subsidy_180d,
        sum(coalesce(mua.marketing_touch_cost_90d, 0)) AS total_marketing_cost_90d,
        sum(coalesce(mua.attributed_net_pay_amount_90d, 0)) AS total_marketing_attributed_pay_90d

    FROM user_base ub
    LEFT JOIN trade_user_agg tua
      ON ub.user_id = tua.user_id
    LEFT JOIN event_user_agg eua
      ON ub.user_id = eua.user_id
    LEFT JOIN coupon_user_agg cua
      ON ub.user_id = cua.user_id
    LEFT JOIN marketing_user_agg mua
      ON ub.user_id = mua.user_id
    LEFT JOIN user_life_cycle_final ulcf
      ON ub.user_id = ulcf.user_id
    LEFT JOIN risk_user_agg rua
      ON ub.user_id = rua.user_id
    GROUP BY
        ub.province,
        ub.city,
        ub.register_channel_group,
        ulcf.final_life_cycle_code,
        rua.risk_level
    GROUPING SETS (
        (ub.province, ub.city, ub.register_channel_group, ulcf.final_life_cycle_code, rua.risk_level),
        (ub.province, ub.city, ub.register_channel_group, ulcf.final_life_cycle_code),
        (ub.province, ub.city, ub.register_channel_group),
        (ub.province, ub.city),
        (ub.province),
        ()
    )
),

/* ============================================================
   41. CUBE：商品类目、价格带、渠道综合立方体
   ============================================================ */
category_channel_cube AS (
    SELECT
        coalesce(oie.category_level1_name, 'ALL') AS category_level1_name,
        coalesce(oie.price_band_name, 'ALL') AS price_band_name,
        coalesce(oie.payment_channel, 'ALL') AS payment_channel,
        coalesce(oie.delivery_type, 'ALL') AS delivery_type,
        grouping__id AS grouping_id,

        count(DISTINCT oie.user_id) AS buyer_cnt,
        count(DISTINCT oie.order_id) AS order_cnt,
        count(DISTINCT oie.sku_id) AS sku_cnt,
        sum(oie.net_pay_amount) AS net_pay_amount,
        sum(oie.gross_profit_amount) AS gross_profit_amount,
        sum(oie.quantity) AS quantity,
        avg(oie.pay_to_sign_sec) AS avg_pay_to_sign_sec

    FROM order_item_enriched oie
    WHERE oie.pay_status = 'PAID'
    GROUP BY
        oie.category_level1_name,
        oie.price_band_name,
        oie.payment_channel,
        oie.delivery_type
    WITH CUBE
),

/* ============================================================
   42. 最终用户级超级宽表
   ============================================================ */
final_user_super_wide AS (
    SELECT
        ub.biz_date,
        ub.user_id,

        /* ---------------- 基础身份 ---------------- */
        ub.union_id,
        ub.open_id,
        ub.mobile,
        ub.email,
        ub.email_domain,
        ub.gender,
        ub.birth_year,
        ub.age,
        ub.age_bucket,
        ub.register_time,
        ub.register_days,
        ub.register_channel,
        ub.register_channel_group,
        ub.register_app,
        ub.register_ip,
        ub.register_device_id,
        ub.province,
        ub.city,
        ub.district,
        ub.member_level,
        ub.member_score,
        ub.is_employee,
        ub.is_black_user,
        ub.real_name_auth,
        ub.student_auth,
        ub.enterprise_auth,
        ub.first_os,
        ub.inviter_user_id,

        /* ---------------- 生命周期 ---------------- */
        ulcf.final_life_cycle_code,
        ulcf.final_life_cycle_name,

        /* ---------------- 行为指标 ---------------- */
        coalesce(eua.event_cnt_90d, 0) AS event_cnt_90d,
        coalesce(eua.event_cnt_30d, 0) AS event_cnt_30d,
        coalesce(eua.event_cnt_7d, 0) AS event_cnt_7d,
        coalesce(eua.event_cnt_1d, 0) AS event_cnt_1d,
        coalesce(eua.session_cnt_90d, 0) AS session_cnt_90d,
        coalesce(eua.session_cnt_30d, 0) AS session_cnt_30d,
        coalesce(eua.session_cnt_7d, 0) AS session_cnt_7d,
        coalesce(eua.device_cnt_90d, 0) AS device_cnt_90d,
        coalesce(eua.ip_cnt_90d, 0) AS ip_cnt_90d,
        coalesce(eua.ua_cnt_90d, 0) AS ua_cnt_90d,
        coalesce(eua.channel_cnt_90d, 0) AS channel_cnt_90d,
        coalesce(eua.sub_channel_cnt_90d, 0) AS sub_channel_cnt_90d,

        coalesce(eua.app_start_cnt_90d, 0) AS app_start_cnt_90d,
        coalesce(eua.page_view_cnt_90d, 0) AS page_view_cnt_90d,
        coalesce(eua.search_cnt_90d, 0) AS search_cnt_90d,
        coalesce(eua.sku_expose_cnt_90d, 0) AS sku_expose_cnt_90d,
        coalesce(eua.sku_click_cnt_90d, 0) AS sku_click_cnt_90d,
        coalesce(eua.add_cart_cnt_90d, 0) AS add_cart_cnt_90d,
        coalesce(eua.favorite_cnt_90d, 0) AS favorite_cnt_90d,
        coalesce(eua.submit_order_cnt_90d, 0) AS submit_order_cnt_90d,
        coalesce(eua.pay_success_event_cnt_90d, 0) AS pay_success_event_cnt_90d,
        coalesce(eua.refund_apply_event_cnt_90d, 0) AS refund_apply_event_cnt_90d,
        coalesce(eua.live_enter_cnt_90d, 0) AS live_enter_cnt_90d,
        coalesce(eua.live_stay_cnt_90d, 0) AS live_stay_cnt_90d,
        coalesce(eua.store_visit_cnt_90d, 0) AS store_visit_cnt_90d,

        coalesce(eua.search_keyword_cnt_90d, 0) AS search_keyword_cnt_90d,
        coalesce(eua.interacted_sku_cnt_90d, 0) AS interacted_sku_cnt_90d,
        coalesce(eua.interacted_store_cnt_90d, 0) AS interacted_store_cnt_90d,
        coalesce(eua.reco_strategy_cnt_90d, 0) AS reco_strategy_cnt_90d,
        eua.first_event_time_90d,
        eua.last_event_time_90d,
        coalesce(eua.days_since_last_event, 9999) AS days_since_last_event,
        coalesce(eua.expose_click_rate_90d, 0) AS expose_click_rate_90d,
        coalesce(eua.click_cart_rate_90d, 0) AS click_cart_rate_90d,
        coalesce(eua.submit_pay_rate_90d, 0) AS submit_pay_rate_90d,
        eua.channel_set_90d,
        eua.device_type_set_90d,
        eua.os_set_90d,

        /* ---------------- Session ---------------- */
        coalesce(sua2.valid_session_cnt_90d, 0) AS valid_session_cnt_90d,
        coalesce(sua2.avg_session_event_cnt_90d, 0) AS avg_session_event_cnt_90d,
        coalesce(sua2.median_session_event_cnt_90d, 0) AS median_session_event_cnt_90d,
        coalesce(sua2.avg_session_page_cnt_90d, 0) AS avg_session_page_cnt_90d,
        coalesce(sua2.avg_session_sku_cnt_90d, 0) AS avg_session_sku_cnt_90d,
        coalesce(sua2.avg_session_duration_sec_90d, 0) AS avg_session_duration_sec_90d,
        coalesce(sua2.median_session_duration_sec_90d, 0) AS median_session_duration_sec_90d,
        coalesce(sua2.max_session_duration_sec_90d, 0) AS max_session_duration_sec_90d,
        sua2.top10_complex_session_paths_90d,

        /* ---------------- 曝光 ---------------- */
        coalesce(esa.exposure_item_cnt_90d, 0) AS exposure_item_cnt_90d,
        coalesce(esa.exposure_sku_cnt_90d, 0) AS exposure_sku_cnt_90d,
        coalesce(esa.exposure_cate1_cnt_90d, 0) AS exposure_cate1_cnt_90d,
        coalesce(esa.exposure_brand_cnt_90d, 0) AS exposure_brand_cnt_90d,
        coalesce(esa.exposure_top3_pos_cnt_90d, 0) AS exposure_top3_pos_cnt_90d,
        coalesce(esa.avg_exposure_pos_90d, 0) AS avg_exposure_pos_90d,
        coalesce(esa.median_exposure_pos_90d, 0) AS median_exposure_pos_90d,
        esa.exposure_price_band_set_90d,
        esa.exposure_cate1_name_set_90d,

        /* ---------------- 搜索 ---------------- */
        coalesce(ska.search_times_90d, 0) AS search_times_90d,
        coalesce(ska.distinct_search_keyword_cnt_90d, 0) AS distinct_search_keyword_cnt_90d,
        coalesce(ska.zero_result_search_cnt_90d, 0) AS zero_result_search_cnt_90d,
        coalesce(ska.zero_result_search_rate_90d, 0) AS zero_result_search_rate_90d,
        ska.top20_search_keywords_90d,

        /* ---------------- 交易 ---------------- */
        coalesce(tua.order_cnt_365d, 0) AS order_cnt_365d,
        coalesce(tua.order_cnt_180d, 0) AS order_cnt_180d,
        coalesce(tua.order_cnt_90d, 0) AS order_cnt_90d,
        coalesce(tua.order_cnt_30d, 0) AS order_cnt_30d,
        coalesce(tua.order_cnt_7d, 0) AS order_cnt_7d,

        coalesce(tua.paid_order_cnt_365d, 0) AS paid_order_cnt_365d,
        coalesce(tua.paid_order_cnt_90d, 0) AS paid_order_cnt_90d,
        coalesce(tua.paid_order_cnt_30d, 0) AS paid_order_cnt_30d,

        coalesce(tua.refund_order_cnt_365d, 0) AS refund_order_cnt_365d,
        coalesce(tua.refund_order_cnt_90d, 0) AS refund_order_cnt_90d,

        coalesce(tua.gross_pay_amount_365d, 0) AS gross_pay_amount_365d,
        coalesce(tua.gross_pay_amount_180d, 0) AS gross_pay_amount_180d,
        coalesce(tua.gross_pay_amount_90d, 0) AS gross_pay_amount_90d,
        coalesce(tua.gross_pay_amount_30d, 0) AS gross_pay_amount_30d,
        coalesce(tua.gross_pay_amount_7d, 0) AS gross_pay_amount_7d,

        coalesce(tua.net_pay_amount_365d, 0) AS net_pay_amount_365d,
        coalesce(tua.net_pay_amount_90d, 0) AS net_pay_amount_90d,
        coalesce(tua.net_pay_amount_30d, 0) AS net_pay_amount_30d,

        coalesce(tua.gross_profit_amount_365d, 0) AS gross_profit_amount_365d,
        coalesce(tua.gross_profit_amount_90d, 0) AS gross_profit_amount_90d,

        coalesce(tua.item_quantity_365d, 0) AS item_quantity_365d,
        coalesce(tua.item_quantity_90d, 0) AS item_quantity_90d,

        coalesce(tua.discount_amount_365d, 0) AS discount_amount_365d,
        coalesce(tua.platform_coupon_amount_365d, 0) AS platform_coupon_amount_365d,
        coalesce(tua.shop_coupon_amount_365d, 0) AS shop_coupon_amount_365d,
        coalesce(tua.points_deduction_amount_365d, 0) AS points_deduction_amount_365d,
        coalesce(tua.discount_rate_365d, 0) AS discount_rate_365d,
        coalesce(tua.refund_order_rate_365d, 0) AS refund_order_rate_365d,
        coalesce(tua.refund_amount_rate_365d, 0) AS refund_amount_rate_365d,

        coalesce(tua.bought_sku_cnt_365d, 0) AS bought_sku_cnt_365d,
        coalesce(tua.bought_spu_cnt_365d, 0) AS bought_spu_cnt_365d,
        coalesce(tua.bought_brand_cnt_365d, 0) AS bought_brand_cnt_365d,
        coalesce(tua.bought_cate1_cnt_365d, 0) AS bought_cate1_cnt_365d,
        coalesce(tua.bought_shop_cnt_365d, 0) AS bought_shop_cnt_365d,
        coalesce(tua.bought_store_cnt_365d, 0) AS bought_store_cnt_365d,
        coalesce(tua.order_province_cnt_365d, 0) AS order_province_cnt_365d,
        coalesce(tua.order_city_cnt_365d, 0) AS order_city_cnt_365d,
        coalesce(tua.address_cnt_365d, 0) AS address_cnt_365d,

        tua.last_pay_time,
        tua.first_pay_time,
        coalesce(tua.days_since_last_pay, 9999) AS days_since_last_pay,

        coalesce(tua.avg_order_to_pay_sec_365d, 0) AS avg_order_to_pay_sec_365d,
        coalesce(tua.median_order_to_pay_sec_365d, 0) AS median_order_to_pay_sec_365d,
        coalesce(tua.avg_pay_to_sign_sec_365d, 0) AS avg_pay_to_sign_sec_365d,
        coalesce(tua.median_pay_to_sign_sec_365d, 0) AS median_pay_to_sign_sec_365d,
        coalesce(tua.timeout_fulfillment_cnt_365d, 0) AS timeout_fulfillment_cnt_365d,
        coalesce(tua.timeout_fulfillment_rate_365d, 0) AS timeout_fulfillment_rate_365d,

        tua.payment_method_set_365d,
        tua.payment_channel_set_365d,
        tua.delivery_type_set_365d,
        tua.bought_price_band_set_365d,

        /* ---------------- 商品偏好 ---------------- */
        uspa.top1_sku_id,
        uspa.top1_sku_name,
        uspa.top1_sku_preference_score,
        uspa.top2_sku_id,
        uspa.top2_sku_name,
        uspa.top2_sku_preference_score,
        uspa.top3_sku_id,
        uspa.top3_sku_name,
        uspa.top3_sku_preference_score,
        uspa.top20_sku_preference_path,

        /* ---------------- 类目偏好 ---------------- */
        ucp.top1_cate1_id,
        ucp.top1_cate1_name,
        ucp.top1_cate1_score,
        ucp.top2_cate1_id,
        ucp.top2_cate1_name,
        ucp.top2_cate1_score,
        ucp.top3_cate1_id,
        ucp.top3_cate1_name,
        ucp.top3_cate1_score,
        ucp.top10_cate1_preference_path,

        /* ---------------- 营销归因 ---------------- */
        coalesce(mua.marketing_touch_cnt_90d, 0) AS marketing_touch_cnt_90d,
        coalesce(mua.marketing_campaign_cnt_90d, 0) AS marketing_campaign_cnt_90d,
        coalesce(mua.marketing_material_cnt_90d, 0) AS marketing_material_cnt_90d,
        coalesce(mua.marketing_strategy_cnt_90d, 0) AS marketing_strategy_cnt_90d,
        coalesce(mua.marketing_channel_cnt_90d, 0) AS marketing_channel_cnt_90d,
        mua.marketing_channel_set_90d,
        mua.marketing_channel_group_set_90d,
        coalesce(mua.marketing_touch_cost_90d, 0) AS marketing_touch_cost_90d,
        coalesce(mua.attributed_net_pay_amount_90d, 0) AS attributed_net_pay_amount_90d,
        coalesce(mua.marketing_roi_90d, 0) AS marketing_roi_90d,
        coalesce(mua.avg_touch_to_pay_sec_90d, 0) AS avg_touch_to_pay_sec_90d,
        coalesce(mua.median_touch_to_pay_sec_90d, 0) AS median_touch_to_pay_sec_90d,
        mua.last_effective_campaign_id,
        mua.last_effective_campaign_name,

        /* ---------------- 优惠券 ---------------- */
        coalesce(cua.coupon_receive_cnt_180d, 0) AS coupon_receive_cnt_180d,
        coalesce(cua.coupon_used_cnt_180d, 0) AS coupon_used_cnt_180d,
        coalesce(cua.coupon_expired_cnt_180d, 0) AS coupon_expired_cnt_180d,
        coalesce(cua.coupon_unused_cnt_180d, 0) AS coupon_unused_cnt_180d,
        coalesce(cua.coupon_discount_amount_180d, 0) AS coupon_discount_amount_180d,
        coalesce(cua.platform_coupon_subsidy_180d, 0) AS platform_coupon_subsidy_180d,
        coalesce(cua.shop_coupon_subsidy_180d, 0) AS shop_coupon_subsidy_180d,
        coalesce(cua.avg_used_coupon_threshold_180d, 0) AS avg_used_coupon_threshold_180d,
        coalesce(cua.coupon_use_rate_180d, 0) AS coupon_use_rate_180d,
        coalesce(cua.coupon_expire_rate_180d, 0) AS coupon_expire_rate_180d,
        cua.last_coupon_receive_time,
        cua.last_coupon_use_time,

        /* ---------------- 客服 ---------------- */
        coalesce(svc.service_ticket_cnt_180d, 0) AS service_ticket_cnt_180d,
        coalesce(svc.refund_service_ticket_cnt_180d, 0) AS refund_service_ticket_cnt_180d,
        coalesce(svc.complaint_ticket_cnt_180d, 0) AS complaint_ticket_cnt_180d,
        coalesce(svc.delivery_delay_ticket_cnt_180d, 0) AS delivery_delay_ticket_cnt_180d,
        coalesce(svc.solved_ticket_cnt_180d, 0) AS solved_ticket_cnt_180d,
        coalesce(svc.escalated_ticket_cnt_180d, 0) AS escalated_ticket_cnt_180d,
        coalesce(svc.avg_service_satisfaction_score_180d, 0) AS avg_service_satisfaction_score_180d,
        coalesce(svc.median_service_satisfaction_score_180d, 0) AS median_service_satisfaction_score_180d,
        coalesce(svc.avg_ticket_solve_sec_180d, 0) AS avg_ticket_solve_sec_180d,
        coalesce(svc.median_ticket_solve_sec_180d, 0) AS median_ticket_solve_sec_180d,
        svc.last_service_create_time,
        svc.service_ticket_type_set_180d,

        /* ---------------- 门店 ---------------- */
        sup.top1_store_id,
        sup.top1_store_name,
        sup.top1_store_preference_score,
        sup.top5_store_preference_path,

        /* ---------------- 风控 ---------------- */
        coalesce(rua.risk_score, 0) AS risk_score,
        coalesce(rua.risk_level, 'R0') AS risk_level,
        coalesce(rua.risk_desc, '极低风险') AS risk_desc,
        rua.risk_reason_list,

        coalesce(ugr.max_device_bind_user_cnt_90d, 0) AS max_device_bind_user_cnt_90d,
        coalesce(ugr.max_ip_user_cnt_90d, 0) AS max_ip_user_cnt_90d,
        coalesce(ugr.max_address_user_cnt_365d, 0) AS max_address_user_cnt_365d,
        ugr.risky_device_related_users,
        ugr.risky_ip_related_users,
        ugr.risky_address_related_users,

        /* ---------------- 行转列指标 ---------------- */
        coalesce(etp.pivot_app_start_cnt_90d, 0) AS pivot_app_start_cnt_90d,
        coalesce(etp.pivot_page_view_cnt_90d, 0) AS pivot_page_view_cnt_90d,
        coalesce(etp.pivot_search_cnt_90d, 0) AS pivot_search_cnt_90d,
        coalesce(etp.pivot_sku_expose_cnt_90d, 0) AS pivot_sku_expose_cnt_90d,
        coalesce(etp.pivot_sku_click_cnt_90d, 0) AS pivot_sku_click_cnt_90d,
        coalesce(etp.pivot_add_cart_cnt_90d, 0) AS pivot_add_cart_cnt_90d,
        coalesce(etp.pivot_remove_cart_cnt_90d, 0) AS pivot_remove_cart_cnt_90d,
        coalesce(etp.pivot_favorite_cnt_90d, 0) AS pivot_favorite_cnt_90d,
        coalesce(etp.pivot_unfavorite_cnt_90d, 0) AS pivot_unfavorite_cnt_90d,
        coalesce(etp.pivot_coupon_receive_cnt_90d, 0) AS pivot_coupon_receive_cnt_90d,
        coalesce(etp.pivot_coupon_use_cnt_90d, 0) AS pivot_coupon_use_cnt_90d,
        coalesce(etp.pivot_submit_order_cnt_90d, 0) AS pivot_submit_order_cnt_90d,
        coalesce(etp.pivot_pay_success_cnt_90d, 0) AS pivot_pay_success_cnt_90d,
        coalesce(etp.pivot_refund_apply_cnt_90d, 0) AS pivot_refund_apply_cnt_90d,
        coalesce(etp.pivot_share_cnt_90d, 0) AS pivot_share_cnt_90d,
        coalesce(etp.pivot_comment_cnt_90d, 0) AS pivot_comment_cnt_90d,
        coalesce(etp.pivot_live_enter_cnt_90d, 0) AS pivot_live_enter_cnt_90d,
        coalesce(etp.pivot_store_visit_cnt_90d, 0) AS pivot_store_visit_cnt_90d,

        /* ---------------- map 指标 ---------------- */
        umm.metric_value_map,
        umm.metric_group_map,

        /* ---------------- 标签 ---------------- */
        coalesce(uta.tag_code_list, '') AS tag_code_list,
        coalesce(uta.tag_name_list, '') AS tag_name_list,
        coalesce(uta.tag_cnt, 0) AS tag_cnt,

        /* ---------------- AB 实验 ---------------- */
        coalesce(aeu.ab_exp_group_list_90d, '') AS ab_exp_group_list_90d,
        coalesce(aeu.ab_exp_cnt_90d, 0) AS ab_exp_cnt_90d,

        /* ---------------- 终极业务分层 ---------------- */
        CASE
            WHEN coalesce(rua.risk_score, 0) >= 90 THEN 'S0_BLOCK'
            WHEN coalesce(rua.risk_score, 0) >= 70 THEN 'S1_MANUAL_REVIEW'
            WHEN coalesce(tua.net_pay_amount_365d, 0) >= 5000
             AND coalesce(eua.event_cnt_30d, 0) >= 10
             AND coalesce(tua.refund_order_rate_365d, 0) < 0.2 THEN 'S2_SUPER_VALUE_ACTIVE'
            WHEN coalesce(tua.net_pay_amount_365d, 0) >= 1000
             AND coalesce(eua.event_cnt_30d, 0) >= 3 THEN 'S3_HIGH_VALUE_ACTIVE'
            WHEN coalesce(tua.net_pay_amount_365d, 0) >= 1000
             AND coalesce(eua.days_since_last_event, 9999) >= 30 THEN 'S4_HIGH_VALUE_SILENT'
            WHEN ub.register_days <= 30
             AND coalesce(eua.event_cnt_7d, 0) >= 3 THEN 'S5_NEW_GROWING'
            WHEN coalesce(eua.add_cart_cnt_90d, 0) > coalesce(eua.pay_success_event_cnt_90d, 0)
             AND coalesce(tua.paid_order_cnt_30d, 0) = 0 THEN 'S6_CART_LOST'
            WHEN coalesce(cua.coupon_use_rate_180d, 0) >= 0.6
             AND coalesce(tua.net_pay_amount_90d, 0) > 0 THEN 'S7_COUPON_SENSITIVE_BUYER'
            WHEN coalesce(mua.marketing_roi_90d, 0) >= 1.0 THEN 'S8_MARKETING_POSITIVE'
            WHEN coalesce(eua.days_since_last_event, 9999) >= 90 THEN 'S9_LOST'
            ELSE 'S10_NORMAL'
        END AS final_user_segment,

        md5(
            concat_ws(
                '|',
                ub.user_id,
                coalesce(ulcf.final_life_cycle_code, ''),
                coalesce(cast(tua.net_pay_amount_365d AS string), '0'),
                coalesce(cast(eua.event_cnt_90d AS string), '0'),
                coalesce(cast(rua.risk_score AS string), '0'),
                coalesce(uta.tag_code_list, '')
            )
        ) AS user_feature_hash,

        current_timestamp() AS etl_time

    FROM user_base ub
    LEFT JOIN user_life_cycle_final ulcf
      ON ub.user_id = ulcf.user_id
    LEFT JOIN event_user_agg eua
      ON ub.user_id = eua.user_id
    LEFT JOIN session_user_agg sua2
      ON ub.user_id = sua2.user_id
    LEFT JOIN exposure_sku_agg esa
      ON ub.user_id = esa.user_id
    LEFT JOIN search_keyword_agg ska
      ON ub.user_id = ska.user_id
    LEFT JOIN trade_user_agg tua
      ON ub.user_id = tua.user_id
    LEFT JOIN user_sku_preference_agg uspa
      ON ub.user_id = uspa.user_id
    LEFT JOIN user_category_pivot ucp
      ON ub.user_id = ucp.user_id
    LEFT JOIN marketing_user_agg mua
      ON ub.user_id = mua.user_id
    LEFT JOIN coupon_user_agg cua
      ON ub.user_id = cua.user_id
    LEFT JOIN service_user_agg svc
      ON ub.user_id = svc.user_id
    LEFT JOIN store_user_agg sup
      ON ub.user_id = sup.user_id
    LEFT JOIN risk_user_agg rua
      ON ub.user_id = rua.user_id
    LEFT JOIN user_graph_risk ugr
      ON ub.user_id = ugr.user_id
    LEFT JOIN event_type_pivot etp
      ON ub.user_id = etp.user_id
    LEFT JOIN user_metric_map umm
      ON ub.user_id = umm.user_id
    LEFT JOIN user_tag_agg uta
      ON ub.user_id = uta.user_id
    LEFT JOIN ab_exp_user_agg aeu
      ON ub.user_id = aeu.user_id
)

/* ============================================================
   43. 最终写入 ADS 用户级超级宽表
   ============================================================ */
INSERT OVERWRITE TABLE ads_user.ads_user_super_operation_profile_df
PARTITION (dt = '${biz_date}', biz_line = 'NEW_RETAIL')
SELECT
    *
FROM final_user_super_wide
WHERE user_id IS NOT NULL
  AND coalesce(is_employee, 0) = 0
  AND final_user_segment NOT IN ('S0_BLOCK')
;
