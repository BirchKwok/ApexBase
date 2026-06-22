-- SET hive.exec.dynamic.partition = true;
-- SET hive.exec.dynamic.partition.mode = nonstrict;
-- SET hive.exec.parallel = true;
-- SET hive.cbo.enable = true;
-- SET hive.vectorized.execution.enabled = true;
-- SET hive.groupby.skewindata = true;
-- SET hive.map.aggr = true;
-- SET hive.auto.convert.join = true;
-- SET hive.exec.max.dynamic.partitions = 10000;
-- SET hive.exec.max.dynamic.partitions.pernode = 2000;

WITH

/* ============================================================
   0. 运行参数虚拟表
   ============================================================ */
p AS (
    SELECT
        '${biz_date}' AS biz_date,
        date_sub('${biz_date}', 1) AS d1,
        date_sub('${biz_date}', 3) AS d3,
        date_sub('${biz_date}', 7) AS d7,
        date_sub('${biz_date}', 14) AS d14,
        date_sub('${biz_date}', 30) AS d30,
        date_sub('${biz_date}', 60) AS d60,
        date_sub('${biz_date}', 90) AS d90,
        date_sub('${biz_date}', 180) AS d180,
        date_sub('${biz_date}', 365) AS d365,
        unix_timestamp('${biz_date}', 'yyyy-MM-dd') AS biz_ts,
        'CN' AS country_code,
        'NEW_RETAIL_ECOM' AS biz_line,
        1000.0 AS high_value_threshold,
        5000.0 AS super_value_threshold,
        0.35 AS high_refund_rate_threshold,
        0.6 AS coupon_sensitive_threshold,
        0.8 AS high_marketing_roi_threshold,
        1800 AS session_gap_sec
),

/* ============================================================
   1. 虚拟规则维表：生命周期
   ============================================================ */
life_cycle_rule AS (
    SELECT stack(
        9,
        'NEW_1D',       0,    1,     '当日新客',
        'NEW_7D',       2,    7,     '7日新客',
        'GROWING_30D',  8,    30,    '成长期用户',
        'ACTIVE_90D',   31,   90,    '活跃用户',
        'MATURE_180D',  91,   180,   '成熟用户',
        'OLD_365D',     181,  365,   '老用户',
        'SUPER_OLD',    366,  99999, '超老用户',
        'SILENT',       -1,   -1,    '沉默用户',
        'LOST',         -2,   -2,    '流失用户'
    ) AS (
        life_cycle_code,
        min_register_days,
        max_register_days,
        life_cycle_name
    )
),

/* ============================================================
   2. 虚拟规则维表：风险等级
   ============================================================ */
risk_rule AS (
    SELECT stack(
        7,
        'R0', 0,   9,   '无风险',
        'R1', 10,  29,  '低风险',
        'R2', 30,  49,  '中风险',
        'R3', 50,  69,  '高风险',
        'R4', 70,  89,  '极高风险',
        'R5', 90,  100, '黑名单风险',
        'RX', -1,  -1,  '未知风险'
    ) AS (
        risk_level,
        min_score,
        max_score,
        risk_name
    )
),

/* ============================================================
   3. 虚拟规则维表：渠道权重
   ============================================================ */
channel_rule AS (
    SELECT stack(
        16,
        'APP_PUSH',        'PRIVATE', 0.18, 1.00,
        'SMS',             'PRIVATE', 0.12, 0.90,
        'EMAIL',           'PRIVATE', 0.08, 0.80,
        'WECHAT',          'PRIVATE', 0.22, 1.10,
        'DOUYIN',          'CONTENT', 0.35, 1.30,
        'KUAISHOU',        'CONTENT', 0.30, 1.20,
        'XIAOHONGSHU',     'CONTENT', 0.28, 1.15,
        'BILIBILI',        'CONTENT', 0.20, 1.05,
        'SEARCH_ENGINE',   'SEARCH',  0.26, 1.00,
        'SEO',             'SEARCH',  0.10, 0.70,
        'SEM',             'SEARCH',  0.32, 1.20,
        'AFFILIATE',       'DIST',    0.20, 0.95,
        'OFFLINE_STORE',   'OFFLINE', 0.40, 1.50,
        'LIVE',            'LIVE',    0.38, 1.40,
        'NATURAL',         'NATURAL', 0.05, 0.50,
        'UNKNOWN',         'UNKNOWN', 0.01, 0.10
    ) AS (
        channel_code,
        channel_group,
        base_weight,
        quality_weight
    )
),

/* ============================================================
   4. 虚拟规则维表：价格带
   ============================================================ */
price_band_rule AS (
    SELECT stack(
        10,
        'P00', 0.00,     9.99,      '白菜价',
        'P01', 10.00,    19.99,     '超低价',
        'P02', 20.00,    49.99,     '低价',
        'P03', 50.00,    99.99,     '平价',
        'P04', 100.00,   199.99,    '中低价',
        'P05', 200.00,   499.99,    '中价',
        'P06', 500.00,   999.99,    '中高价',
        'P07', 1000.00,  2999.99,   '高价',
        'P08', 3000.00,  9999.99,   '高端',
        'P09', 10000.00, 999999999, '奢侈'
    ) AS (
        price_band_code,
        min_price,
        max_price,
        price_band_name
    )
),

/* ============================================================
   5. 虚拟规则维表：标签规则
   ============================================================ */
tag_rule AS (
    SELECT stack(
        24,
        'T_SUPER_VALUE',       '超级高价值',       'net_pay_amount_365d >= 5000',
        'T_HIGH_VALUE',        '高价值',           'net_pay_amount_365d >= 1000',
        'T_LOW_VALUE',         '低价值',           'net_pay_amount_365d < 100',
        'T_NEW_ACTIVE',        '新客活跃',         'register_days <= 30 and event_cnt_7d > 0',
        'T_NEW_PAY',           '新客成交',         'register_days <= 30 and paid_order_cnt_30d > 0',
        'T_SILENT',            '沉默',             'days_since_last_event >= 30',
        'T_LOST',              '流失',             'days_since_last_event >= 90',
        'T_RISK_HIGH',         '高风险',           'risk_score >= 70',
        'T_BLACK',             '黑名单',           'is_black_user = 1',
        'T_REFUND_HEAVY',      '退款偏高',         'refund_rate_365d >= 0.35',
        'T_COUPON_SENSITIVE',  '券敏感',           'coupon_use_rate_180d >= 0.6',
        'T_MARKETING_POSITIVE','营销正反馈',       'marketing_roi_90d >= 0.8',
        'T_LIVE_USER',         '直播用户',         'live_enter_cnt_90d > 0',
        'T_SEARCH_HEAVY',      '搜索重度',         'search_cnt_90d >= 30',
        'T_CART_LOST',         '加购流失',         'add_cart_cnt_90d > pay_success_cnt_90d',
        'T_MULTI_DEVICE',      '多设备',           'device_cnt_90d >= 3',
        'T_MULTI_IP',          '多IP',             'ip_cnt_90d >= 5',
        'T_OFFLINE_USER',      '线下用户',         'store_visit_cnt_90d > 0',
        'T_COMPLAINT_HIGH',    '高投诉',           'complaint_ticket_cnt_180d >= 3',
        'T_FAST_PAY',          '极速支付',         'avg_order_to_pay_sec_365d <= 5',
        'T_DELAY_SENSITIVE',   '履约敏感',         'timeout_fulfillment_rate_365d >= 0.2',
        'T_HIGH_MARGIN',       '高毛利',           'gross_margin_rate_365d >= 0.35',
        'T_PRICE_SENSITIVE',   '价格敏感',         'discount_rate_365d >= 0.3',
        'T_EXPERIMENT_USER',   '实验用户',         'ab_exp_cnt_90d > 0'
    ) AS (
        tag_code,
        tag_name,
        tag_expr
    )
),

/* ============================================================
   6. 用户基础表
   ============================================================ */
user_raw AS (
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
        p.biz_date,
        p.d7,
        p.d30,
        p.d90,
        p.d180,
        p.d365,
        regexp_replace(lower(coalesce(u.email, '')), '^.*@', '') AS email_domain,
        floor(months_between(p.biz_date, concat(u.birth_year, '-01-01')) / 12) AS age,
        datediff(p.biz_date, substr(u.register_time, 1, 10)) AS register_days,
        get_json_object(u.ext_json, '$.identity.real_name_auth') AS real_name_auth,
        get_json_object(u.ext_json, '$.identity.student_auth') AS student_auth,
        get_json_object(u.ext_json, '$.identity.enterprise_auth') AS enterprise_auth,
        get_json_object(u.ext_json, '$.device.first_os') AS first_os,
        get_json_object(u.ext_json, '$.source.inviter_user_id') AS inviter_user_id,
        get_json_object(u.ext_json, '$.profile.preference_json') AS preference_json
    FROM ods_user.ods_user_profile_df u
    CROSS JOIN p
    WHERE u.dt = p.biz_date
      AND u.country_code = p.country_code
      AND coalesce(u.is_test_user, 0) = 0
),

user_base AS (
    SELECT
        *,
        CASE
            WHEN age IS NULL THEN 'UNKNOWN'
            WHEN age < 18 THEN 'UNDER_18'
            WHEN age BETWEEN 18 AND 24 THEN '18_24'
            WHEN age BETWEEN 25 AND 34 THEN '25_34'
            WHEN age BETWEEN 35 AND 44 THEN '35_44'
            WHEN age BETWEEN 45 AND 59 THEN '45_59'
            ELSE '60_PLUS'
        END AS age_bucket,

        CASE
            WHEN upper(register_channel) RLIKE 'DOUYIN|KUAISHOU|XIAOHONGSHU|BILIBILI|LIVE' THEN 'CONTENT'
            WHEN upper(register_channel) RLIKE 'SMS|PUSH|EMAIL|WECHAT' THEN 'PRIVATE'
            WHEN upper(register_channel) RLIKE 'STORE|OFFLINE' THEN 'OFFLINE'
            WHEN upper(register_channel) RLIKE 'SEO|SEM|SEARCH' THEN 'SEARCH'
            WHEN upper(register_channel) RLIKE 'NATURAL|ORGANIC' THEN 'NATURAL'
            ELSE 'OTHER'
        END AS register_channel_group
    FROM user_raw
),

/* ============================================================
   7. 用户偏好 JSON 拆解
   ============================================================ */
user_preference_kv AS (
    SELECT
        user_id,
        pref_key,
        pref_value
    FROM user_base
    LATERAL VIEW OUTER explode(
        split(
            regexp_replace(
                regexp_replace(
                    regexp_replace(coalesce(preference_json, '{}'), '^\\{|\\}$', ''),
                    '"',
                    ''
                ),
                '\\s+',
                ''
            ),
            ','
        )
    ) pref AS pref_pair
    LATERAL VIEW OUTER json_tuple(
        concat('{', regexp_replace(pref_pair, '^([^:]+):(.*)$', '"$1":"$2"'), '}'),
        'dummy'
    ) jt AS dummy_json
    LATERAL VIEW OUTER inline(
        array(
            named_struct(
                'pref_key',
                split(pref_pair, ':')[0],
                'pref_value',
                split(pref_pair, ':')[1]
            )
        )
    ) kv AS pref_key, pref_value
    WHERE coalesce(pref_key, '') <> ''
),

user_preference_map AS (
    SELECT
        user_id,
        str_to_map(
            concat_ws(',', collect_list(concat(pref_key, ':', pref_value))),
            ',',
            ':'
        ) AS preference_map,
        concat_ws(',', sort_array(collect_set(pref_key))) AS preference_key_list
    FROM user_preference_kv
    GROUP BY user_id
),

/* ============================================================
   8. 商品维度
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
        s.shelf_status,
        s.is_self_operated,
        s.is_imported,
        s.is_fresh,
        s.is_virtual,
        pbr.price_band_code,
        pbr.price_band_name
    FROM dim.dim_sku_df s
    LEFT JOIN price_band_rule pbr
      ON s.list_price BETWEEN pbr.min_price AND pbr.max_price
    WHERE s.dt = '${biz_date}'
),

/* ============================================================
   9. 店铺维度
   ============================================================ */
shop_dim AS (
    SELECT
        shop_id,
        shop_name,
        shop_type,
        seller_id,
        seller_level,
        is_brand_shop,
        is_self_operated AS shop_is_self_operated
    FROM dim.dim_shop_df
    WHERE dt = '${biz_date}'
),

/* ============================================================
   10. 门店维度
   ============================================================ */
store_dim AS (
    SELECT
        store_id,
        store_name,
        store_type,
        region_id,
        region_name,
        city AS store_city,
        business_circle,
        store_level,
        open_date,
        manager_id,
        longitude,
        latitude
    FROM dim.dim_store_df
    WHERE dt = '${biz_date}'
),

/* ============================================================
   11. 行为日志原始层：复杂 JSON + 数组 + trace
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

        get_json_object(e.ext_json, '$.sku_id') AS sku_id,
        get_json_object(e.ext_json, '$.spu_id') AS spu_id,
        get_json_object(e.ext_json, '$.shop_id') AS shop_id,
        get_json_object(e.ext_json, '$.store_id') AS store_id,
        get_json_object(e.ext_json, '$.category_id') AS category_id,
        get_json_object(e.ext_json, '$.brand_id') AS brand_id,
        get_json_object(e.ext_json, '$.search.keyword') AS search_keyword,
        get_json_object(e.ext_json, '$.search.result_cnt') AS search_result_cnt,
        get_json_object(e.ext_json, '$.coupon.coupon_id') AS coupon_id,
        get_json_object(e.ext_json, '$.ab.exp_id') AS exp_id,
        get_json_object(e.ext_json, '$.ab.group_id') AS group_id,
        get_json_object(e.ext_json, '$.reco.strategy_id') AS reco_strategy_id,
        get_json_object(e.ext_json, '$.reco.scene_id') AS reco_scene_id,
        get_json_object(e.ext_json, '$.trace.trace_id') AS trace_id,
        get_json_object(e.ext_json, '$.geo.longitude') AS longitude,
        get_json_object(e.ext_json, '$.geo.latitude') AS latitude,

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
    JOIN p
      ON e.dt BETWEEN p.d90 AND p.biz_date
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
          'live_click',
          'store_visit',
          'scan_qr',
          'customer_service_click',
          'address_submit',
          'payment_click'
      )
),

/* ============================================================
   12. 曝光商品数组拆解：posexplode
   ============================================================ */
event_exposure_long AS (
    SELECT
        er.user_id,
        er.session_id,
        er.event_id,
        er.event_time,
        er.event_ts,
        er.dt,
        er.channel,
        er.sub_channel,
        er.device_id,
        er.ip,
        er.reco_strategy_id,
        er.reco_scene_id,
        expose_pos,
        expose_sku_id
    FROM event_raw er
    LATERAL VIEW OUTER posexplode(er.expose_sku_arr) px AS expose_pos, expose_sku_id
    WHERE coalesce(expose_sku_id, '') <> ''
),

/* ============================================================
   15. Session 排序 + gap + 前后事件
   ============================================================ */
session_event_lagged AS (
    SELECT
        er.*,
        lag(event_ts, 1, event_ts) OVER (
            PARTITION BY user_id, session_id
            ORDER BY event_ts ASC, event_id ASC
        ) AS prev_event_ts

    FROM event_raw er
),

session_event_marked AS (
    SELECT
        *,
        event_ts - coalesce(prev_event_ts, event_ts) AS gap_from_prev_sec,
        if(event_ts - coalesce(prev_event_ts, event_ts) > 1800, 1, 0) AS virtual_session_boundary
    FROM session_event_lagged
),

session_event_ordered AS (
    SELECT
        *,
        sum(virtual_session_boundary) OVER (
            PARTITION BY user_id, session_id
            ORDER BY event_ts ASC, event_id ASC
        ) AS virtual_session_seq
    FROM session_event_marked
),

/* ============================================================
   16. 虚拟 session 聚合
   ============================================================ */
virtual_session_agg AS (
    SELECT
        user_id,
        session_id,
        virtual_session_seq,
        coalesce(max(event_ts), 0) - coalesce(min(event_ts), 0) AS virtual_session_duration_sec,
        count(1) AS virtual_session_event_cnt,
        count(DISTINCT page_id) AS virtual_session_page_cnt,
        count(DISTINCT sku_id) AS virtual_session_sku_cnt,
        concat_ws(
            '>',
            collect_list(event_type)
        ) AS virtual_session_event_path
    FROM (
        SELECT *
        FROM session_event_ordered
        DISTRIBUTE BY user_id, session_id, virtual_session_seq
        SORT BY user_id, session_id, virtual_session_seq, event_ts, event_id
    ) t
    GROUP BY user_id, session_id, virtual_session_seq
),

/* ============================================================
   17. 用户行为聚合
   ============================================================ */
event_user_agg AS (
    SELECT
        er.user_id,

        count(1) AS event_cnt_90d,
        sum(if(er.dt >= date_sub('${biz_date}', 30), 1, 0)) AS event_cnt_30d,
        sum(if(er.dt >= date_sub('${biz_date}', 7), 1, 0)) AS event_cnt_7d,
        sum(if(er.dt >= date_sub('${biz_date}', 1), 1, 0)) AS event_cnt_1d,

        count(DISTINCT er.session_id) AS session_cnt_90d,
        count(DISTINCT if(er.dt >= date_sub('${biz_date}', 30), er.session_id, NULL)) AS session_cnt_30d,
        count(DISTINCT if(er.dt >= date_sub('${biz_date}', 7), er.session_id, NULL)) AS session_cnt_7d,

        count(DISTINCT er.device_id) AS device_cnt_90d,
        count(DISTINCT er.ip) AS ip_cnt_90d,
        count(DISTINCT er.ua) AS ua_cnt_90d,
        count(DISTINCT er.channel) AS channel_cnt_90d,
        count(DISTINCT er.sub_channel) AS sub_channel_cnt_90d,

        sum(if(er.event_type = 'app_start', 1, 0)) AS app_start_cnt_90d,
        sum(if(er.event_type = 'page_view', 1, 0)) AS page_view_cnt_90d,
        sum(if(er.event_type = 'search', 1, 0)) AS search_cnt_90d,
        sum(if(er.event_type = 'sku_expose', 1, 0)) AS sku_expose_cnt_90d,
        sum(if(er.event_type = 'sku_click', 1, 0)) AS sku_click_cnt_90d,
        sum(if(er.event_type = 'add_cart', 1, 0)) AS add_cart_cnt_90d,
        sum(if(er.event_type = 'remove_cart', 1, 0)) AS remove_cart_cnt_90d,
        sum(if(er.event_type = 'favorite', 1, 0)) AS favorite_cnt_90d,
        sum(if(er.event_type = 'submit_order', 1, 0)) AS submit_order_cnt_90d,
        sum(if(er.event_type = 'pay_success', 1, 0)) AS pay_success_cnt_90d,
        sum(if(er.event_type = 'refund_apply', 1, 0)) AS refund_apply_cnt_90d,
        sum(if(er.event_type = 'live_enter', 1, 0)) AS live_enter_cnt_90d,
        sum(if(er.event_type = 'live_stay', 1, 0)) AS live_stay_cnt_90d,
        sum(if(er.event_type = 'store_visit', 1, 0)) AS store_visit_cnt_90d,

        count(DISTINCT if(er.event_type = 'search', lower(trim(er.search_keyword)), NULL)) AS search_keyword_cnt_90d,
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
        concat_ws(',', sort_array(collect_set(er.sub_channel))) AS sub_channel_set_90d,
        concat_ws(',', sort_array(collect_set(er.device_type))) AS device_type_set_90d,
        concat_ws(',', sort_array(collect_set(er.os))) AS os_set_90d

    FROM event_raw er
    GROUP BY er.user_id
),

/* ============================================================
   18. 虚拟 Session 用户聚合
   ============================================================ */
virtual_session_user_agg AS (
    SELECT
        user_id,
        count(1) AS virtual_session_cnt_90d,
        avg(virtual_session_event_cnt) AS avg_virtual_session_event_cnt_90d,
        percentile_approx(virtual_session_event_cnt, 0.5) AS median_virtual_session_event_cnt_90d,
        avg(virtual_session_page_cnt) AS avg_virtual_session_page_cnt_90d,
        avg(virtual_session_sku_cnt) AS avg_virtual_session_sku_cnt_90d,
        avg(virtual_session_duration_sec) AS avg_virtual_session_duration_sec_90d,
        percentile_approx(virtual_session_duration_sec, 0.5) AS median_virtual_session_duration_sec_90d,
        max(virtual_session_duration_sec) AS max_virtual_session_duration_sec_90d,
        concat_ws(
            '||',
            slice(
                reverse(
                    sort_array(
                        collect_list(
                            concat(
                                lpad(cast(virtual_session_event_cnt AS string), 10, '0'),
                                '#',
                                lpad(cast(virtual_session_duration_sec AS string), 10, '0'),
                                '#',
                                virtual_session_event_path
                            )
                        )
                    )
                ),
                1,
                10
            )
        ) AS top10_virtual_session_path_90d
    FROM virtual_session_agg
    GROUP BY user_id
),

/* ============================================================
   19. 曝光商品聚合
   ============================================================ */
exposure_user_agg AS (
    SELECT
        eel.user_id,
        count(1) AS exposure_item_cnt_90d,
        count(DISTINCT eel.expose_sku_id) AS exposure_sku_cnt_90d,
        count(DISTINCT sd.category_level1_id) AS exposure_cate1_cnt_90d,
        count(DISTINCT sd.brand_id) AS exposure_brand_cnt_90d,
        sum(if(eel.expose_pos BETWEEN 0 AND 2, 1, 0)) AS exposure_top3_cnt_90d,
        sum(if(eel.expose_pos BETWEEN 0 AND 9, 1, 0)) AS exposure_top10_cnt_90d,
        avg(eel.expose_pos) AS avg_exposure_pos_90d,
        percentile_approx(eel.expose_pos, 0.5) AS median_exposure_pos_90d,
        concat_ws(',', sort_array(collect_set(sd.category_level1_name))) AS exposure_cate1_name_set_90d,
        concat_ws(',', sort_array(collect_set(sd.price_band_code))) AS exposure_price_band_set_90d
    FROM event_exposure_long eel
    LEFT JOIN sku_dim sd
      ON eel.expose_sku_id = sd.sku_id
    GROUP BY eel.user_id
),

/* ============================================================
   20. 搜索词聚合
   ============================================================ */
search_keyword_clean AS (
    SELECT
        user_id,
        regexp_replace(lower(trim(search_keyword)), '[^0-9a-zA-Z\\u4e00-\\u9fa5]+', '') AS clean_keyword,
        cast(search_result_cnt AS int) AS search_result_cnt,
        event_time,
        dt
    FROM event_raw
    WHERE event_type = 'search'
      AND search_keyword IS NOT NULL
      AND length(trim(search_keyword)) > 0
),

search_keyword_rank AS (
    SELECT
        user_id,
        clean_keyword,
        count(1) AS keyword_search_cnt,
        sum(if(search_result_cnt = 0, 1, 0)) AS zero_result_cnt,
        row_number() OVER (
            PARTITION BY user_id
            ORDER BY count(1) DESC, max(event_time) DESC
        ) AS keyword_rn
    FROM search_keyword_clean
    GROUP BY user_id, clean_keyword
),

search_user_agg AS (
    SELECT
        user_id,
        sum(keyword_search_cnt) AS search_times_90d,
        count(DISTINCT clean_keyword) AS distinct_search_keyword_cnt_90d,
        sum(zero_result_cnt) AS zero_result_search_cnt_90d,
        round(
            sum(zero_result_cnt) / greatest(sum(keyword_search_cnt), 1),
            8
        ) AS zero_result_search_rate_90d,
        max(if(keyword_rn = 1, clean_keyword, NULL)) AS top1_search_keyword,
        max(if(keyword_rn = 2, clean_keyword, NULL)) AS top2_search_keyword,
        max(if(keyword_rn = 3, clean_keyword, NULL)) AS top3_search_keyword,
        concat_ws(
            ',',
            collect_list(
                concat(
                    cast(keyword_rn AS string),
                    ':',
                    clean_keyword,
                    ':',
                    cast(keyword_search_cnt AS string)
                )
            )
        ) AS top20_search_keyword_path
    FROM search_keyword_rank
    WHERE keyword_rn <= 20
    GROUP BY user_id
),

/* ============================================================
   21. 订单明细
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
        oi.address_hash,
        oi.is_test_order,
        oi.ext_json,
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
        sd.cost_price AS sku_cost_price,

        sh.shop_name,
        sh.shop_type,
        sh.seller_id,
        sh.seller_level,
        sh.is_brand_shop,
        sh.shop_is_self_operated,

        st.store_name,
        st.store_type,
        st.region_id,
        st.region_name,
        st.store_level,
        st.business_circle,

        get_json_object(oi.ext_json, '$.promotion.activity_id') AS activity_id,
        get_json_object(oi.ext_json, '$.promotion.activity_type') AS activity_type

    FROM dwd_trade.dwd_order_item_df oi
    LEFT JOIN sku_dim sd
      ON oi.sku_id = sd.sku_id
    LEFT JOIN shop_dim sh
      ON oi.shop_id = sh.shop_id
    LEFT JOIN store_dim st
      ON oi.store_id = st.store_id
    JOIN p
      ON oi.dt BETWEEN p.d365 AND p.biz_date
    WHERE oi.user_id IS NOT NULL
      AND coalesce(oi.is_test_order, 0) = 0
),

/* ============================================================
   22. 支付流水
   ============================================================ */
payment_raw AS (
    SELECT
        pay.user_id,
        pay.order_id,
        pay.payment_id,
        pay.payment_channel,
        pay.payment_method,
        pay.bank_code,
        pay.pay_status AS payment_status,
        pay.risk_decision AS payment_risk_decision,
        pay.risk_score AS payment_risk_score,
        pay.installment_num,
        pay.ext_json,
        pay.dt,
        get_json_object(pay.ext_json, '$.card.bin') AS card_bin,
        get_json_object(pay.ext_json, '$.wallet.type') AS wallet_type,
        get_json_object(pay.ext_json, '$.risk.reason') AS payment_risk_reason
    FROM dwd_pay.dwd_payment_flow_df pay
    JOIN p
      ON pay.dt BETWEEN p.d365 AND p.biz_date
    WHERE pay.user_id IS NOT NULL
),

/* ============================================================
   23. 退款售后
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
    JOIN p
      ON r.dt BETWEEN p.d365 AND p.biz_date
    WHERE r.user_id IS NOT NULL
),

/* ============================================================
   24. 履约物流
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
    JOIN p
      ON f.dt BETWEEN p.d365 AND p.biz_date
    WHERE f.user_id IS NOT NULL
),

/* ============================================================
   25. 订单宽明细：订单 + 支付 + 退款 + 履约
   ============================================================ */
order_item_enriched AS (
    SELECT
        oi.*,

        pay.payment_id,
        pay.payment_channel,
        pay.bank_code,
        pay.payment_status,
        pay.payment_risk_decision,
        pay.payment_risk_score,
        pay.installment_num,
        pay.card_bin,
        pay.wallet_type,
        pay.payment_risk_reason,

        rf.refund_id,
        rf.refund_type,
        rf.refund_reason_code,
        rf.refund_reason_desc,
        rf.refund_success_time,
        rf.refund_amount,
        rf.is_abnormal_refund,

        ful.warehouse_id,
        ful.carrier_code,
        ful.promise_delivery_time,
        ful.ship_time,
        ful.sign_time,
        ful.fulfillment_status,
        ful.is_timeout AS fulfillment_is_timeout,
        ful.distance_km,
        ful.delivery_fee,

        unix_timestamp(oi.pay_time) - unix_timestamp(oi.order_time) AS order_to_pay_sec,
        unix_timestamp(ful.sign_time) - unix_timestamp(oi.pay_time) AS pay_to_sign_sec,

        oi.pay_amount - coalesce(oi.cost_amount, oi.quantity * oi.sku_cost_price, 0) AS gross_profit_amount,

        CASE
            WHEN oi.pay_status = 'PAID'
             AND coalesce(rf.refund_amount, 0) = 0
                THEN oi.pay_amount
            WHEN oi.pay_status = 'PAID'
             AND coalesce(rf.refund_amount, 0) > 0
                THEN greatest(oi.pay_amount - rf.refund_amount, 0)
            ELSE 0
        END AS net_pay_amount

    FROM order_item_raw oi
    LEFT JOIN payment_raw pay
      ON oi.user_id = pay.user_id
     AND oi.order_id = pay.order_id
    LEFT JOIN refund_raw rf
      ON oi.user_id = rf.user_id
     AND oi.order_item_id = rf.order_item_id
    LEFT JOIN fulfillment_raw ful
      ON oi.user_id = ful.user_id
     AND oi.order_id = ful.order_id
),

/* ============================================================
   26. 交易用户聚合：多窗口、多指标
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
        count(DISTINCT if(pay_status = 'PAID' AND dt >= date_sub('${biz_date}', 180), order_id, NULL)) AS paid_order_cnt_180d,
        count(DISTINCT if(pay_status = 'PAID' AND dt >= date_sub('${biz_date}', 90), order_id, NULL)) AS paid_order_cnt_90d,
        count(DISTINCT if(pay_status = 'PAID' AND dt >= date_sub('${biz_date}', 30), order_id, NULL)) AS paid_order_cnt_30d,

        count(DISTINCT if(refund_id IS NOT NULL, order_id, NULL)) AS refund_order_cnt_365d,
        count(DISTINCT if(refund_id IS NOT NULL AND dt >= date_sub('${biz_date}', 90), order_id, NULL)) AS refund_order_cnt_90d,

        sum(if(pay_status = 'PAID', pay_amount, 0)) AS gross_pay_amount_365d,
        sum(if(pay_status = 'PAID' AND dt >= date_sub('${biz_date}', 180), pay_amount, 0)) AS gross_pay_amount_180d,
        sum(if(pay_status = 'PAID' AND dt >= date_sub('${biz_date}', 90), pay_amount, 0)) AS gross_pay_amount_90d,
        sum(if(pay_status = 'PAID' AND dt >= date_sub('${biz_date}', 30), pay_amount, 0)) AS gross_pay_amount_30d,

        sum(net_pay_amount) AS net_pay_amount_365d,
        sum(if(dt >= date_sub('${biz_date}', 180), net_pay_amount, 0)) AS net_pay_amount_180d,
        sum(if(dt >= date_sub('${biz_date}', 90), net_pay_amount, 0)) AS net_pay_amount_90d,
        sum(if(dt >= date_sub('${biz_date}', 30), net_pay_amount, 0)) AS net_pay_amount_30d,

        sum(gross_profit_amount) AS gross_profit_amount_365d,
        sum(if(dt >= date_sub('${biz_date}', 90), gross_profit_amount, 0)) AS gross_profit_amount_90d,

        round(
            sum(gross_profit_amount) / greatest(sum(net_pay_amount), 1),
            8
        ) AS gross_margin_rate_365d,

        sum(quantity) AS item_quantity_365d,
        sum(discount_amount) AS discount_amount_365d,
        sum(platform_coupon_amount) AS platform_coupon_amount_365d,
        sum(shop_coupon_amount) AS shop_coupon_amount_365d,
        sum(points_deduction_amount) AS points_deduction_amount_365d,

        round(
            sum(discount_amount) / greatest(sum(goods_amount), 1),
            8
        ) AS discount_rate_365d,

        round(
            count(DISTINCT if(refund_id IS NOT NULL, order_id, NULL))
            / greatest(count(DISTINCT if(pay_status = 'PAID', order_id, NULL)), 1),
            8
        ) AS refund_rate_365d,

        round(
            sum(coalesce(refund_amount, 0))
            / greatest(sum(if(pay_status = 'PAID', pay_amount, 0)), 1),
            8
        ) AS refund_amount_rate_365d,

        count(DISTINCT sku_id) AS bought_sku_cnt_365d,
        count(DISTINCT spu_id) AS bought_spu_cnt_365d,
        count(DISTINCT brand_id) AS bought_brand_cnt_365d,
        count(DISTINCT category_level1_id) AS bought_cate1_cnt_365d,
        count(DISTINCT category_level2_id) AS bought_cate2_cnt_365d,
        count(DISTINCT shop_id) AS bought_shop_cnt_365d,
        count(DISTINCT store_id) AS bought_store_cnt_365d,
        count(DISTINCT address_hash) AS address_cnt_365d,
        count(DISTINCT payment_channel) AS payment_channel_cnt_365d,
        count(DISTINCT payment_method) AS payment_method_cnt_365d,

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

        concat_ws(',', sort_array(collect_set(payment_channel))) AS payment_channel_set_365d,
        concat_ws(',', sort_array(collect_set(payment_method))) AS payment_method_set_365d,
        concat_ws(',', sort_array(collect_set(delivery_type))) AS delivery_type_set_365d,
        concat_ws(',', sort_array(collect_set(price_band_code))) AS bought_price_band_set_365d

    FROM order_item_enriched
    GROUP BY user_id
),

/* ============================================================
   27. SKU 偏好打分
   ============================================================ */
sku_preference_score AS (
    SELECT
        user_id,
        sku_id,
        max(product_name) AS product_name,
        max(category_level1_id) AS category_level1_id,
        max(category_level1_name) AS category_level1_name,
        max(category_level2_id) AS category_level2_id,
        max(category_level2_name) AS category_level2_name,
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

sku_preference_rank AS (
    SELECT
        *,
        row_number() OVER (
            PARTITION BY user_id
            ORDER BY sku_preference_score DESC,
                     sku_net_pay_amount_365d DESC,
                     sku_order_cnt_365d DESC
        ) AS sku_rn
    FROM sku_preference_score
),

sku_preference_pivot AS (
    SELECT
        user_id,
        max(if(sku_rn = 1, sku_id, NULL)) AS top1_sku_id,
        max(if(sku_rn = 1, product_name, NULL)) AS top1_sku_name,
        max(if(sku_rn = 1, sku_preference_score, NULL)) AS top1_sku_score,

        max(if(sku_rn = 2, sku_id, NULL)) AS top2_sku_id,
        max(if(sku_rn = 2, product_name, NULL)) AS top2_sku_name,
        max(if(sku_rn = 2, sku_preference_score, NULL)) AS top2_sku_score,

        max(if(sku_rn = 3, sku_id, NULL)) AS top3_sku_id,
        max(if(sku_rn = 3, product_name, NULL)) AS top3_sku_name,
        max(if(sku_rn = 3, sku_preference_score, NULL)) AS top3_sku_score,

        concat_ws(
            '||',
            collect_list(
                concat(
                    cast(sku_rn AS string),
                    ':',
                    sku_id,
                    ':',
                    regexp_replace(product_name, '[:|,]', '_'),
                    ':',
                    cast(round(sku_preference_score, 6) AS string)
                )
            )
        ) AS top20_sku_preference_path
    FROM sku_preference_rank
    WHERE sku_rn <= 20
    GROUP BY user_id
),

/* ============================================================
   28. 类目偏好
   ============================================================ */
category_preference_score AS (
    SELECT
        user_id,
        category_level1_id,
        category_level1_name,
        sum(net_pay_amount) AS cate_net_pay_amount_365d,

        (
            log(1 + sum(net_pay_amount)) * 0.50
            + log(1 + count(DISTINCT order_id)) * 0.25
            + log(1 + count(DISTINCT sku_id)) * 0.15
            + log(1 + sum(quantity)) * 0.10
        ) AS cate_preference_score

    FROM order_item_enriched
    WHERE pay_status = 'PAID'
    GROUP BY user_id, category_level1_id, category_level1_name
),

category_preference_rank AS (
    SELECT
        *,
        row_number() OVER (
            PARTITION BY user_id
            ORDER BY cate_preference_score DESC,
                     cate_net_pay_amount_365d DESC
        ) AS cate_rn
    FROM category_preference_score
),

category_preference_pivot AS (
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
            '||',
            collect_list(
                concat(
                    cast(cate_rn AS string),
                    ':',
                    category_level1_id,
                    ':',
                    category_level1_name,
                    ':',
                    cast(round(cate_preference_score, 6) AS string)
                )
            )
        ) AS top10_cate1_preference_path
    FROM category_preference_rank
    WHERE cate_rn <= 10
    GROUP BY user_id
),

/* ============================================================
   29. 营销触点：三路 UNION ALL
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
        upper(channel) AS channel,
        upper(sub_channel) AS sub_channel,
        touch_time,
        touch_ts,
        dt,
        get_json_object(ext_json, '$.bid_type') AS bid_type,
        cast(get_json_object(ext_json, '$.cost') AS double) AS touch_cost,
        get_json_object(ext_json, '$.audience_pkg_id') AS audience_pkg_id,
        get_json_object(ext_json, '$.creative_type') AS creative_type,
        'CAMPAIGN_TOUCH' AS touch_source
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
        upper(media_channel) AS channel,
        upper(media_sub_channel) AS sub_channel,
        click_time AS touch_time,
        click_ts AS touch_ts,
        dt,
        bid_type,
        cost_amount AS touch_cost,
        audience_pkg_id,
        creative_type,
        'AD_CLICK' AS touch_source
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
        upper(popup_type) AS sub_channel,
        show_time AS touch_time,
        show_ts AS touch_ts,
        dt,
        'NONE' AS bid_type,
        0.0 AS touch_cost,
        audience_pkg_id,
        creative_type,
        'APP_POPUP_SHOW' AS touch_source
    FROM dwd_marketing.dwd_app_popup_show_di
    WHERE dt BETWEEN date_sub('${biz_date}', 90) AND '${biz_date}'
      AND user_id IS NOT NULL
),

marketing_touch_weighted AS (
    SELECT
        m.*,
        coalesce(cr.channel_group, 'UNKNOWN') AS channel_group,
        coalesce(cr.base_weight, 0.01) AS base_weight,
        coalesce(cr.quality_weight, 0.1) AS quality_weight,

        (
            coalesce(cr.base_weight, 0.01)
            * coalesce(cr.quality_weight, 0.1)
            * pow(0.85, datediff('${biz_date}', substr(m.touch_time, 1, 10)))
            * CASE
                WHEN m.scene IN ('BIG_PROMOTION', 'NEW_USER_COUPON', 'RECALL', 'PRICE_DROP') THEN 1.35
                WHEN m.scene IN ('BRAND_CAMPAIGN', 'LIVE_PROMOTION') THEN 1.20
                ELSE 1.00
              END
        ) AS attribution_weight
    FROM marketing_touch_raw m
    LEFT JOIN channel_rule cr
      ON m.channel = cr.channel_code
),

/* ============================================================
   30. 营销归因：触点匹配订单
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
        mtw.touch_source,
        mtw.touch_time,
        mtw.touch_ts,
        mtw.touch_cost,
        mtw.attribution_weight,

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
            ORDER BY mtw.attribution_weight DESC,
                     mtw.touch_ts DESC,
                     mtw.touch_id DESC
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
        *,
        CASE
            WHEN total_candidate_weight > 0
                THEN net_pay_amount * attribution_weight / total_candidate_weight
            ELSE 0
        END AS attributed_net_pay_amount
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

        sum(touch_cost) AS marketing_touch_cost_90d,
        sum(attributed_net_pay_amount) AS attributed_net_pay_amount_90d,

        round(
            sum(attributed_net_pay_amount) / greatest(sum(touch_cost), 1),
            8
        ) AS marketing_roi_90d,

        avg(touch_to_pay_sec) AS avg_touch_to_pay_sec_90d,
        percentile_approx(touch_to_pay_sec, 0.5) AS median_touch_to_pay_sec_90d,

        max(if(attribution_rn = 1, campaign_id, NULL)) AS last_effective_campaign_id,
        max(if(attribution_rn = 1, campaign_name, NULL)) AS last_effective_campaign_name,
        max(if(attribution_rn = 1, channel, NULL)) AS last_effective_channel

    FROM marketing_attribution_final
    GROUP BY user_id
),

/* ============================================================
   31. 优惠券聚合
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
   32. 客服聚合
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
   33. 门店偏好：FULL OUTER JOIN
   ============================================================ */
store_visit_agg AS (
    SELECT
        user_id,
        store_id,
        count(1) AS store_visit_cnt_90d,
        count(DISTINCT dt) AS store_visit_days_90d
    FROM event_raw
    WHERE event_type IN ('store_visit', 'scan_qr')
      AND store_id IS NOT NULL
    GROUP BY user_id, store_id
),

store_trade_agg AS (
    SELECT
        user_id,
        store_id,
        count(DISTINCT order_id) AS store_order_cnt_365d,
        sum(net_pay_amount) AS store_net_pay_amount_365d
    FROM order_item_enriched
    WHERE store_id IS NOT NULL
      AND pay_status = 'PAID'
    GROUP BY user_id, store_id
),

store_preference_rank AS (
    SELECT
        *,
        row_number() OVER (
            PARTITION BY user_id
            ORDER BY store_preference_score DESC,
                     store_net_pay_amount_365d DESC,
                     store_visit_cnt_90d DESC
        ) AS store_rn
    FROM (
        SELECT
            coalesce(sv.user_id, st.user_id) AS user_id,
            coalesce(sv.store_id, st.store_id) AS store_id,
            sd.store_name,
            sd.store_type,
            sd.region_name,
            sd.store_city,

            coalesce(sv.store_visit_cnt_90d, 0) AS store_visit_cnt_90d,
            coalesce(sv.store_visit_days_90d, 0) AS store_visit_days_90d,
            coalesce(st.store_order_cnt_365d, 0) AS store_order_cnt_365d,
            coalesce(st.store_net_pay_amount_365d, 0) AS store_net_pay_amount_365d,

            (
                log(1 + coalesce(sv.store_visit_cnt_90d, 0)) * 0.25
                + log(1 + coalesce(st.store_order_cnt_365d, 0)) * 0.30
                + log(1 + coalesce(st.store_net_pay_amount_365d, 0)) * 0.45
            ) AS store_preference_score

        FROM store_visit_agg sv
        FULL OUTER JOIN store_trade_agg st
          ON sv.user_id = st.user_id
         AND sv.store_id = st.store_id
        LEFT JOIN store_dim sd
          ON coalesce(sv.store_id, st.store_id) = sd.store_id
    ) x
),

store_preference_pivot AS (
    SELECT
        user_id,
        max(if(store_rn = 1, store_id, NULL)) AS top1_store_id,
        max(if(store_rn = 1, store_name, NULL)) AS top1_store_name,
        max(if(store_rn = 1, store_preference_score, NULL)) AS top1_store_score,
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
                    cast(round(store_preference_score, 6) AS string)
                )
            )
        ) AS top5_store_preference_path
    FROM store_preference_rank
    WHERE store_rn <= 5
    GROUP BY user_id
),

/* ============================================================
   34. 设备/IP/地址图谱
   ============================================================ */
device_graph AS (
    SELECT
        device_id,
        count(DISTINCT user_id) AS device_bind_user_cnt_90d,
        concat_ws(',', slice(sort_array(collect_set(user_id)), 1, 30)) AS device_user_sample
    FROM event_raw
    WHERE device_id IS NOT NULL
    GROUP BY device_id
),

ip_graph AS (
    SELECT
        ip,
        count(DISTINCT user_id) AS ip_bind_user_cnt_90d,
        concat_ws(',', slice(sort_array(collect_set(user_id)), 1, 30)) AS ip_user_sample
    FROM event_raw
    WHERE ip IS NOT NULL
    GROUP BY ip
),

address_graph AS (
    SELECT
        address_hash,
        count(DISTINCT user_id) AS address_bind_user_cnt_365d,
        concat_ws(',', slice(sort_array(collect_set(user_id)), 1, 30)) AS address_user_sample
    FROM order_item_enriched
    WHERE address_hash IS NOT NULL
    GROUP BY address_hash
),

user_event_graph_risk AS (
    SELECT
        er.user_id,
        max(coalesce(dg.device_bind_user_cnt_90d, 0)) AS max_device_bind_user_cnt_90d,
        max(coalesce(ig.ip_bind_user_cnt_90d, 0)) AS max_ip_bind_user_cnt_90d,
        concat_ws(',', sort_array(collect_set(dg.device_user_sample))) AS risky_device_user_sample,
        concat_ws(',', sort_array(collect_set(ig.ip_user_sample))) AS risky_ip_user_sample
    FROM event_raw er
    LEFT JOIN device_graph dg
      ON er.device_id = dg.device_id
    LEFT JOIN ip_graph ig
      ON er.ip = ig.ip
    GROUP BY er.user_id
),

user_address_graph_risk AS (
    SELECT
        oi.user_id,
        max(coalesce(ag.address_bind_user_cnt_365d, 0)) AS max_address_bind_user_cnt_365d,
        concat_ws(',', sort_array(collect_set(ag.address_user_sample))) AS risky_address_user_sample
    FROM order_item_enriched oi
    LEFT JOIN address_graph ag
      ON oi.address_hash = ag.address_hash
    GROUP BY oi.user_id
),

user_graph_risk AS (
    SELECT
        ub.user_id,
        coalesce(uegr.max_device_bind_user_cnt_90d, 0) AS max_device_bind_user_cnt_90d,
        coalesce(uegr.max_ip_bind_user_cnt_90d, 0) AS max_ip_bind_user_cnt_90d,
        coalesce(uagr.max_address_bind_user_cnt_365d, 0) AS max_address_bind_user_cnt_365d,
        uegr.risky_device_user_sample,
        uegr.risky_ip_user_sample,
        uagr.risky_address_user_sample
    FROM user_base ub
    LEFT JOIN user_event_graph_risk uegr
      ON ub.user_id = uegr.user_id
    LEFT JOIN user_address_graph_risk uagr
      ON ub.user_id = uagr.user_id
),

/* ============================================================
   35. 风险评分
   ============================================================ */
risk_score_calc AS (
    SELECT
        ub.user_id,

        least(
            100,
            greatest(
                0,
                cast(
                    if(coalesce(ub.is_black_user, 0) = 1, 100, 0)
                    + if(coalesce(eua.device_cnt_90d, 0) >= 10, 15, 0)
                    + if(coalesce(eua.ip_cnt_90d, 0) >= 20, 15, 0)
                    + if(coalesce(ugr.max_device_bind_user_cnt_90d, 0) >= 5, 20, 0)
                    + if(coalesce(ugr.max_ip_bind_user_cnt_90d, 0) >= 10, 15, 0)
                    + if(coalesce(ugr.max_address_bind_user_cnt_365d, 0) >= 5, 15, 0)
                    + if(coalesce(tua.refund_rate_365d, 0) >= 0.35, 15, 0)
                    + if(coalesce(tua.refund_amount_rate_365d, 0) >= 0.50, 15, 0)
                    + if(coalesce(sua.complaint_ticket_cnt_180d, 0) >= 3, 10, 0)
                    + if(coalesce(sua.escalated_ticket_cnt_180d, 0) >= 2, 10, 0)
                    + if(ub.email_domain IN ('tempmail.com', 'mailinator.com', 'fake.com', 'trashmail.com'), 15, 0)
                    + if(coalesce(tua.avg_order_to_pay_sec_365d, 999999) <= 3, 10, 0)
                    AS int
                )
            )
        ) AS risk_score,

        regexp_replace(
            regexp_replace(
                concat_ws(
                    ',',
                    array(
                        if(coalesce(ub.is_black_user, 0) = 1, 'BLACK_USER', NULL),
                        if(coalesce(eua.device_cnt_90d, 0) >= 10, 'MANY_DEVICE', NULL),
                        if(coalesce(eua.ip_cnt_90d, 0) >= 20, 'MANY_IP', NULL),
                        if(coalesce(ugr.max_device_bind_user_cnt_90d, 0) >= 5, 'DEVICE_GRAPH_RISK', NULL),
                        if(coalesce(ugr.max_ip_bind_user_cnt_90d, 0) >= 10, 'IP_GRAPH_RISK', NULL),
                        if(coalesce(ugr.max_address_bind_user_cnt_365d, 0) >= 5, 'ADDRESS_GRAPH_RISK', NULL),
                        if(coalesce(tua.refund_rate_365d, 0) >= 0.35, 'HIGH_REFUND_RATE', NULL),
                        if(coalesce(tua.refund_amount_rate_365d, 0) >= 0.50, 'HIGH_REFUND_AMOUNT_RATE', NULL),
                        if(ub.email_domain IN ('tempmail.com', 'mailinator.com', 'fake.com', 'trashmail.com'), 'TEMP_EMAIL', NULL)
                    )
                ),
                ',+',
                ','
            ),
            '^,|,$',
            ''
        ) AS risk_reason_list

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
        coalesce(rr.risk_level, 'RX') AS risk_level,
        coalesce(rr.risk_name, '未知风险') AS risk_name,
        rsc.risk_reason_list
    FROM risk_score_calc rsc
    LEFT JOIN risk_rule rr
      ON rsc.risk_score BETWEEN rr.min_score AND rr.max_score
),

/* ============================================================
   36. AB 实验
   ============================================================ */
ab_user_agg AS (
    SELECT
        user_id,
        count(DISTINCT exp_id) AS ab_exp_cnt_90d,
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
        ) AS ab_exp_group_list_90d
    FROM event_raw
    WHERE exp_id IS NOT NULL
    GROUP BY user_id
),

/* ============================================================
   37. 生命周期最终修正
   ============================================================ */
life_cycle_final AS (
    SELECT
        ub.user_id,
        CASE
            WHEN coalesce(eua.days_since_last_event, 9999) >= 90
             AND coalesce(tua.days_since_last_pay, 9999) >= 90 THEN 'LOST'
            WHEN coalesce(eua.days_since_last_event, 9999) >= 30 THEN 'SILENT'
            WHEN ub.register_days <= 1 THEN 'NEW_1D'
            WHEN ub.register_days <= 7 THEN 'NEW_7D'
            WHEN ub.register_days <= 30 THEN 'GROWING_30D'
            WHEN coalesce(eua.event_cnt_30d, 0) >= 10
              OR coalesce(tua.paid_order_cnt_30d, 0) >= 1 THEN 'ACTIVE_90D'
            WHEN ub.register_days >= 365 THEN 'SUPER_OLD'
            ELSE coalesce(lcr.life_cycle_code, 'MATURE_180D')
        END AS final_life_cycle_code,

        CASE
            WHEN coalesce(eua.days_since_last_event, 9999) >= 90
             AND coalesce(tua.days_since_last_pay, 9999) >= 90 THEN '流失用户'
            WHEN coalesce(eua.days_since_last_event, 9999) >= 30 THEN '沉默用户'
            WHEN ub.register_days <= 1 THEN '当日新客'
            WHEN ub.register_days <= 7 THEN '7日新客'
            WHEN ub.register_days <= 30 THEN '成长期用户'
            WHEN coalesce(eua.event_cnt_30d, 0) >= 10
              OR coalesce(tua.paid_order_cnt_30d, 0) >= 1 THEN '活跃用户'
            WHEN ub.register_days >= 365 THEN '超老用户'
            ELSE coalesce(lcr.life_cycle_name, '成熟用户')
        END AS final_life_cycle_name

    FROM user_base ub
    LEFT JOIN life_cycle_rule lcr
      ON ub.register_days BETWEEN lcr.min_register_days AND lcr.max_register_days
    LEFT JOIN event_user_agg eua
      ON ub.user_id = eua.user_id
    LEFT JOIN trade_user_agg tua
      ON ub.user_id = tua.user_id
),

/* ============================================================
   38. 指标列转行
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
            coalesce(tua.net_pay_amount_365d, 0) AS net_pay_amount_365d,
            coalesce(tua.gross_margin_rate_365d, 0) AS gross_margin_rate_365d,
            coalesce(tua.refund_rate_365d, 0) AS refund_rate_365d,

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
        'trade',     'net_pay_amount_365d',       cast(net_pay_amount_365d AS double),
        'trade',     'gross_margin_rate_365d',    cast(gross_margin_rate_365d AS double),
        'trade',     'refund_rate_365d',          cast(refund_rate_365d AS double),

        'coupon',    'coupon_use_rate_180d',      cast(coupon_use_rate_180d AS double),
        'marketing', 'marketing_roi_90d',         cast(marketing_roi_90d AS double),
        'risk',      'risk_score',                cast(risk_score AS double)
    ) s AS metric_group, metric_name, metric_value
),

/* ============================================================
   39. 长指标转 map
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
   40. 事件行转列
   ============================================================ */
event_type_pivot AS (
    SELECT
        user_id,
        max(if(event_type = 'app_start', cnt, 0)) AS pivot_app_start_cnt_90d,
        max(if(event_type = 'page_view', cnt, 0)) AS pivot_page_view_cnt_90d,
        max(if(event_type = 'search', cnt, 0)) AS pivot_search_cnt_90d,
        max(if(event_type = 'sku_expose', cnt, 0)) AS pivot_sku_expose_cnt_90d,
        max(if(event_type = 'sku_click', cnt, 0)) AS pivot_sku_click_cnt_90d,
        max(if(event_type = 'add_cart', cnt, 0)) AS pivot_add_cart_cnt_90d,
        max(if(event_type = 'remove_cart', cnt, 0)) AS pivot_remove_cart_cnt_90d,
        max(if(event_type = 'favorite', cnt, 0)) AS pivot_favorite_cnt_90d,
        max(if(event_type = 'submit_order', cnt, 0)) AS pivot_submit_order_cnt_90d,
        max(if(event_type = 'pay_success', cnt, 0)) AS pivot_pay_success_cnt_90d,
        max(if(event_type = 'refund_apply', cnt, 0)) AS pivot_refund_apply_cnt_90d,
        max(if(event_type = 'live_enter', cnt, 0)) AS pivot_live_enter_cnt_90d,
        max(if(event_type = 'live_stay', cnt, 0)) AS pivot_live_stay_cnt_90d,
        max(if(event_type = 'store_visit', cnt, 0)) AS pivot_store_visit_cnt_90d,
        max(if(event_type = 'scan_qr', cnt, 0)) AS pivot_scan_qr_cnt_90d,
        max(if(event_type = 'customer_service_click', cnt, 0)) AS pivot_customer_service_click_cnt_90d
    FROM (
        SELECT
            user_id,
            event_type,
            count(1) AS cnt
        FROM event_raw
        GROUP BY user_id, event_type
    ) t
    GROUP BY user_id
),

/* ============================================================
   41. 用户标签计算
   ============================================================ */
user_tag_long AS (
    SELECT
        x.user_id,
        tr.tag_code,
        tr.tag_name
    FROM (
        SELECT
            ub.user_id,
            coalesce(ub.is_black_user, 0) AS is_black_user,
            ub.register_days,
            ub.register_channel_group,

            coalesce(eua.event_cnt_7d, 0) AS event_cnt_7d,
            coalesce(eua.days_since_last_event, 9999) AS days_since_last_event,
            coalesce(eua.live_enter_cnt_90d, 0) AS live_enter_cnt_90d,
            coalesce(eua.search_cnt_90d, 0) AS search_cnt_90d,
            coalesce(eua.add_cart_cnt_90d, 0) AS add_cart_cnt_90d,
            coalesce(eua.pay_success_cnt_90d, 0) AS pay_success_cnt_90d,
            coalesce(eua.device_cnt_90d, 0) AS device_cnt_90d,
            coalesce(eua.ip_cnt_90d, 0) AS ip_cnt_90d,
            coalesce(eua.store_visit_cnt_90d, 0) AS store_visit_cnt_90d,

            coalesce(tua.net_pay_amount_365d, 0) AS net_pay_amount_365d,
            coalesce(tua.paid_order_cnt_30d, 0) AS paid_order_cnt_30d,
            coalesce(tua.refund_rate_365d, 0) AS refund_rate_365d,
            coalesce(tua.avg_order_to_pay_sec_365d, 999999) AS avg_order_to_pay_sec_365d,
            coalesce(tua.timeout_fulfillment_rate_365d, 0) AS timeout_fulfillment_rate_365d,
            coalesce(tua.gross_margin_rate_365d, 0) AS gross_margin_rate_365d,
            coalesce(tua.discount_rate_365d, 0) AS discount_rate_365d,

            coalesce(cua.coupon_use_rate_180d, 0) AS coupon_use_rate_180d,
            coalesce(mua.marketing_roi_90d, 0) AS marketing_roi_90d,
            coalesce(rua.risk_score, 0) AS risk_score,
            coalesce(sua.complaint_ticket_cnt_180d, 0) AS complaint_ticket_cnt_180d,
            coalesce(abu.ab_exp_cnt_90d, 0) AS ab_exp_cnt_90d

        FROM user_base ub
        LEFT JOIN event_user_agg eua ON ub.user_id = eua.user_id
        LEFT JOIN trade_user_agg tua ON ub.user_id = tua.user_id
        LEFT JOIN coupon_user_agg cua ON ub.user_id = cua.user_id
        LEFT JOIN marketing_user_agg mua ON ub.user_id = mua.user_id
        LEFT JOIN risk_user_agg rua ON ub.user_id = rua.user_id
        LEFT JOIN service_user_agg sua ON ub.user_id = sua.user_id
        LEFT JOIN ab_user_agg abu ON ub.user_id = abu.user_id
    ) x
    CROSS JOIN tag_rule tr
    WHERE
        CASE tr.tag_code
            WHEN 'T_SUPER_VALUE'
                THEN if(x.net_pay_amount_365d >= 5000, 1, 0)
            WHEN 'T_HIGH_VALUE'
                THEN if(x.net_pay_amount_365d >= 1000, 1, 0)
            WHEN 'T_LOW_VALUE'
                THEN if(x.net_pay_amount_365d < 100, 1, 0)
            WHEN 'T_NEW_ACTIVE'
                THEN if(x.register_days <= 30 AND x.event_cnt_7d > 0, 1, 0)
            WHEN 'T_NEW_PAY'
                THEN if(x.register_days <= 30 AND x.paid_order_cnt_30d > 0, 1, 0)
            WHEN 'T_SILENT'
                THEN if(x.days_since_last_event >= 30 AND x.days_since_last_event < 90, 1, 0)
            WHEN 'T_LOST'
                THEN if(x.days_since_last_event >= 90, 1, 0)
            WHEN 'T_RISK_HIGH'
                THEN if(x.risk_score >= 70, 1, 0)
            WHEN 'T_BLACK'
                THEN if(x.is_black_user = 1, 1, 0)
            WHEN 'T_REFUND_HEAVY'
                THEN if(x.refund_rate_365d >= 0.35, 1, 0)
            WHEN 'T_COUPON_SENSITIVE'
                THEN if(x.coupon_use_rate_180d >= 0.6, 1, 0)
            WHEN 'T_MARKETING_POSITIVE'
                THEN if(x.marketing_roi_90d >= 0.8, 1, 0)
            WHEN 'T_LIVE_USER'
                THEN if(x.live_enter_cnt_90d > 0, 1, 0)
            WHEN 'T_SEARCH_HEAVY'
                THEN if(x.search_cnt_90d >= 30, 1, 0)
            WHEN 'T_CART_LOST'
                THEN if(x.add_cart_cnt_90d > x.pay_success_cnt_90d, 1, 0)
            WHEN 'T_MULTI_DEVICE'
                THEN if(x.device_cnt_90d >= 3, 1, 0)
            WHEN 'T_MULTI_IP'
                THEN if(x.ip_cnt_90d >= 5, 1, 0)
            WHEN 'T_OFFLINE_USER'
                THEN if(x.store_visit_cnt_90d > 0, 1, 0)
            WHEN 'T_COMPLAINT_HIGH'
                THEN if(x.complaint_ticket_cnt_180d >= 3, 1, 0)
            WHEN 'T_FAST_PAY'
                THEN if(x.avg_order_to_pay_sec_365d <= 5, 1, 0)
            WHEN 'T_DELAY_SENSITIVE'
                THEN if(x.timeout_fulfillment_rate_365d >= 0.2, 1, 0)
            WHEN 'T_HIGH_MARGIN'
                THEN if(x.gross_margin_rate_365d >= 0.35, 1, 0)
            WHEN 'T_PRICE_SENSITIVE'
                THEN if(x.discount_rate_365d >= 0.3, 1, 0)
            WHEN 'T_EXPERIMENT_USER'
                THEN if(x.ab_exp_cnt_90d > 0, 1, 0)
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
   42. 半连接：近 30 天有行为且非黑产候选
   ============================================================ */
active_non_risk_candidate AS (
    SELECT
        ub.user_id,
        1 AS active_non_risk_marker
    FROM user_base ub
    LEFT SEMI JOIN event_user_agg eua
      ON ub.user_id = eua.user_id
     AND eua.event_cnt_30d > 0
    LEFT JOIN risk_user_agg rua
      ON ub.user_id = rua.user_id
    WHERE coalesce(rua.risk_score, 0) < 70
),

/* ============================================================
   43. 反连接：有行为但 90 天未支付
   ============================================================ */
active_no_pay_user AS (
    SELECT
        eua.user_id,
        1 AS active_no_pay_marker
    FROM event_user_agg eua
    LEFT ANTI JOIN (
        SELECT DISTINCT user_id
        FROM order_item_enriched
        WHERE pay_status = 'PAID'
          AND dt >= date_sub('${biz_date}', 90)
    ) paid
      ON eua.user_id = paid.user_id
    WHERE eua.event_cnt_30d > 0
),

/* ============================================================
   44. 经营汇总 GROUPING SETS
   ============================================================ */
biz_grouping_sets_summary AS (
    SELECT
        coalesce(ub.province, 'ALL') AS province,
        coalesce(ub.city, 'ALL') AS city,
        coalesce(ub.register_channel_group, 'ALL') AS register_channel_group,
        coalesce(lcf.final_life_cycle_code, 'ALL') AS life_cycle_code,
        coalesce(rua.risk_level, 'ALL') AS risk_level,
        grouping__id AS grouping_id,

        count(DISTINCT ub.user_id) AS user_cnt,
        sum(coalesce(tua.net_pay_amount_365d, 0)) AS total_net_pay_amount_365d,
        sum(coalesce(tua.gross_profit_amount_365d, 0)) AS total_gross_profit_amount_365d,
        sum(coalesce(cua.platform_coupon_subsidy_180d, 0)) AS total_platform_coupon_subsidy_180d,
        sum(coalesce(mua.marketing_touch_cost_90d, 0)) AS total_marketing_cost_90d,
        sum(coalesce(mua.attributed_net_pay_amount_90d, 0)) AS total_marketing_attributed_pay_90d,
        avg(coalesce(rua.risk_score, 0)) AS avg_risk_score,
        avg(coalesce(eua.event_cnt_90d, 0)) AS avg_event_cnt_90d

    FROM user_base ub
    LEFT JOIN event_user_agg eua ON ub.user_id = eua.user_id
    LEFT JOIN trade_user_agg tua ON ub.user_id = tua.user_id
    LEFT JOIN coupon_user_agg cua ON ub.user_id = cua.user_id
    LEFT JOIN marketing_user_agg mua ON ub.user_id = mua.user_id
    LEFT JOIN risk_user_agg rua ON ub.user_id = rua.user_id
    LEFT JOIN life_cycle_final lcf ON ub.user_id = lcf.user_id
    GROUP BY
        ub.province,
        ub.city,
        ub.register_channel_group,
        lcf.final_life_cycle_code,
        rua.risk_level
    GROUPING SETS (
        (ub.province, ub.city, ub.register_channel_group, lcf.final_life_cycle_code, rua.risk_level),
        (ub.province, ub.city, ub.register_channel_group, lcf.final_life_cycle_code),
        (ub.province, ub.city, ub.register_channel_group),
        (ub.province, ub.city),
        (ub.province),
        ()
    )
),

/* ============================================================
   45. 商品渠道 CUBE
   ============================================================ */
category_channel_cube AS (
    SELECT
        coalesce(category_level1_name, 'ALL') AS category_level1_name,
        coalesce(price_band_name, 'ALL') AS price_band_name,
        coalesce(payment_channel, 'ALL') AS payment_channel,
        coalesce(delivery_type, 'ALL') AS delivery_type,
        grouping__id AS grouping_id,

        count(DISTINCT user_id) AS buyer_cnt,
        count(DISTINCT order_id) AS order_cnt,
        count(DISTINCT sku_id) AS sku_cnt,
        sum(net_pay_amount) AS net_pay_amount,
        sum(gross_profit_amount) AS gross_profit_amount,
        sum(quantity) AS quantity,
        avg(pay_to_sign_sec) AS avg_pay_to_sign_sec

    FROM order_item_enriched
    WHERE pay_status = 'PAID'
    GROUP BY
        category_level1_name,
        price_band_name,
        payment_channel,
        delivery_type
    WITH CUBE
),

/* ============================================================
   46. 门店 ROLLUP
   ============================================================ */
store_rollup_summary AS (
    SELECT
        coalesce(region_name, 'ALL') AS region_name,
        coalesce(store_type, 'ALL') AS store_type,
        coalesce(store_level, 'ALL') AS store_level,
        grouping__id AS grouping_id,

        count(DISTINCT store_id) AS store_cnt,
        count(DISTINCT user_id) AS buyer_cnt,
        count(DISTINCT order_id) AS order_cnt,
        sum(net_pay_amount) AS net_pay_amount,
        sum(gross_profit_amount) AS gross_profit_amount,
        avg(distance_km) AS avg_distance_km,
        avg(pay_to_sign_sec) AS avg_pay_to_sign_sec

    FROM order_item_enriched
    WHERE pay_status = 'PAID'
      AND store_id IS NOT NULL
    GROUP BY
        region_name,
        store_type,
        store_level
    WITH ROLLUP
),

/* ============================================================
   47. 最终超级宽表
   ============================================================ */
final_user_extreme_wide AS (
    SELECT
        ub.biz_date,
        ub.user_id,

        /* 基础身份 */
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

        /* preference map */
        upm.preference_map,
        upm.preference_key_list,

        /* 生命周期 */
        lcf.final_life_cycle_code,
        lcf.final_life_cycle_name,

        /* 行为 */
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
        coalesce(eua.remove_cart_cnt_90d, 0) AS remove_cart_cnt_90d,
        coalesce(eua.favorite_cnt_90d, 0) AS favorite_cnt_90d,
        coalesce(eua.submit_order_cnt_90d, 0) AS submit_order_cnt_90d,
        coalesce(eua.pay_success_cnt_90d, 0) AS pay_success_cnt_90d,
        coalesce(eua.refund_apply_cnt_90d, 0) AS refund_apply_cnt_90d,
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
        eua.sub_channel_set_90d,
        eua.device_type_set_90d,
        eua.os_set_90d,

        /* 虚拟 session */
        coalesce(vsua.virtual_session_cnt_90d, 0) AS virtual_session_cnt_90d,
        coalesce(vsua.avg_virtual_session_event_cnt_90d, 0) AS avg_virtual_session_event_cnt_90d,
        coalesce(vsua.median_virtual_session_event_cnt_90d, 0) AS median_virtual_session_event_cnt_90d,
        coalesce(vsua.avg_virtual_session_page_cnt_90d, 0) AS avg_virtual_session_page_cnt_90d,
        coalesce(vsua.avg_virtual_session_sku_cnt_90d, 0) AS avg_virtual_session_sku_cnt_90d,
        coalesce(vsua.avg_virtual_session_duration_sec_90d, 0) AS avg_virtual_session_duration_sec_90d,
        coalesce(vsua.median_virtual_session_duration_sec_90d, 0) AS median_virtual_session_duration_sec_90d,
        coalesce(vsua.max_virtual_session_duration_sec_90d, 0) AS max_virtual_session_duration_sec_90d,
        vsua.top10_virtual_session_path_90d,

        /* 曝光 */
        coalesce(exua.exposure_item_cnt_90d, 0) AS exposure_item_cnt_90d,
        coalesce(exua.exposure_sku_cnt_90d, 0) AS exposure_sku_cnt_90d,
        coalesce(exua.exposure_cate1_cnt_90d, 0) AS exposure_cate1_cnt_90d,
        coalesce(exua.exposure_brand_cnt_90d, 0) AS exposure_brand_cnt_90d,
        coalesce(exua.exposure_top3_cnt_90d, 0) AS exposure_top3_cnt_90d,
        coalesce(exua.exposure_top10_cnt_90d, 0) AS exposure_top10_cnt_90d,
        coalesce(exua.avg_exposure_pos_90d, 0) AS avg_exposure_pos_90d,
        coalesce(exua.median_exposure_pos_90d, 0) AS median_exposure_pos_90d,
        exua.exposure_cate1_name_set_90d,
        exua.exposure_price_band_set_90d,

        /* 搜索 */
        coalesce(sua.search_times_90d, 0) AS search_times_90d,
        coalesce(sua.distinct_search_keyword_cnt_90d, 0) AS distinct_search_keyword_cnt_90d,
        coalesce(sua.zero_result_search_cnt_90d, 0) AS zero_result_search_cnt_90d,
        coalesce(sua.zero_result_search_rate_90d, 0) AS zero_result_search_rate_90d,
        sua.top1_search_keyword,
        sua.top2_search_keyword,
        sua.top3_search_keyword,
        sua.top20_search_keyword_path,

        /* 交易 */
        coalesce(tua.order_cnt_365d, 0) AS order_cnt_365d,
        coalesce(tua.order_cnt_180d, 0) AS order_cnt_180d,
        coalesce(tua.order_cnt_90d, 0) AS order_cnt_90d,
        coalesce(tua.order_cnt_30d, 0) AS order_cnt_30d,
        coalesce(tua.order_cnt_7d, 0) AS order_cnt_7d,
        coalesce(tua.paid_order_cnt_365d, 0) AS paid_order_cnt_365d,
        coalesce(tua.paid_order_cnt_180d, 0) AS paid_order_cnt_180d,
        coalesce(tua.paid_order_cnt_90d, 0) AS paid_order_cnt_90d,
        coalesce(tua.paid_order_cnt_30d, 0) AS paid_order_cnt_30d,
        coalesce(tua.refund_order_cnt_365d, 0) AS refund_order_cnt_365d,
        coalesce(tua.refund_order_cnt_90d, 0) AS refund_order_cnt_90d,

        coalesce(tua.gross_pay_amount_365d, 0) AS gross_pay_amount_365d,
        coalesce(tua.gross_pay_amount_180d, 0) AS gross_pay_amount_180d,
        coalesce(tua.gross_pay_amount_90d, 0) AS gross_pay_amount_90d,
        coalesce(tua.gross_pay_amount_30d, 0) AS gross_pay_amount_30d,
        coalesce(tua.net_pay_amount_365d, 0) AS net_pay_amount_365d,
        coalesce(tua.net_pay_amount_180d, 0) AS net_pay_amount_180d,
        coalesce(tua.net_pay_amount_90d, 0) AS net_pay_amount_90d,
        coalesce(tua.net_pay_amount_30d, 0) AS net_pay_amount_30d,
        coalesce(tua.gross_profit_amount_365d, 0) AS gross_profit_amount_365d,
        coalesce(tua.gross_profit_amount_90d, 0) AS gross_profit_amount_90d,
        coalesce(tua.gross_margin_rate_365d, 0) AS gross_margin_rate_365d,

        coalesce(tua.item_quantity_365d, 0) AS item_quantity_365d,
        coalesce(tua.discount_amount_365d, 0) AS discount_amount_365d,
        coalesce(tua.platform_coupon_amount_365d, 0) AS platform_coupon_amount_365d,
        coalesce(tua.shop_coupon_amount_365d, 0) AS shop_coupon_amount_365d,
        coalesce(tua.points_deduction_amount_365d, 0) AS points_deduction_amount_365d,
        coalesce(tua.discount_rate_365d, 0) AS discount_rate_365d,
        coalesce(tua.refund_rate_365d, 0) AS refund_rate_365d,
        coalesce(tua.refund_amount_rate_365d, 0) AS refund_amount_rate_365d,

        coalesce(tua.bought_sku_cnt_365d, 0) AS bought_sku_cnt_365d,
        coalesce(tua.bought_spu_cnt_365d, 0) AS bought_spu_cnt_365d,
        coalesce(tua.bought_brand_cnt_365d, 0) AS bought_brand_cnt_365d,
        coalesce(tua.bought_cate1_cnt_365d, 0) AS bought_cate1_cnt_365d,
        coalesce(tua.bought_cate2_cnt_365d, 0) AS bought_cate2_cnt_365d,
        coalesce(tua.bought_shop_cnt_365d, 0) AS bought_shop_cnt_365d,
        coalesce(tua.bought_store_cnt_365d, 0) AS bought_store_cnt_365d,
        coalesce(tua.address_cnt_365d, 0) AS address_cnt_365d,
        coalesce(tua.payment_channel_cnt_365d, 0) AS payment_channel_cnt_365d,
        coalesce(tua.payment_method_cnt_365d, 0) AS payment_method_cnt_365d,

        tua.last_pay_time,
        tua.first_pay_time,
        coalesce(tua.days_since_last_pay, 9999) AS days_since_last_pay,
        coalesce(tua.avg_order_to_pay_sec_365d, 0) AS avg_order_to_pay_sec_365d,
        coalesce(tua.median_order_to_pay_sec_365d, 0) AS median_order_to_pay_sec_365d,
        coalesce(tua.avg_pay_to_sign_sec_365d, 0) AS avg_pay_to_sign_sec_365d,
        coalesce(tua.median_pay_to_sign_sec_365d, 0) AS median_pay_to_sign_sec_365d,
        coalesce(tua.timeout_fulfillment_cnt_365d, 0) AS timeout_fulfillment_cnt_365d,
        coalesce(tua.timeout_fulfillment_rate_365d, 0) AS timeout_fulfillment_rate_365d,
        tua.payment_channel_set_365d,
        tua.payment_method_set_365d,
        tua.delivery_type_set_365d,
        tua.bought_price_band_set_365d,

        /* SKU 偏好 */
        spp.top1_sku_id,
        spp.top1_sku_name,
        spp.top1_sku_score,
        spp.top2_sku_id,
        spp.top2_sku_name,
        spp.top2_sku_score,
        spp.top3_sku_id,
        spp.top3_sku_name,
        spp.top3_sku_score,
        spp.top20_sku_preference_path,

        /* 类目偏好 */
        cpp.top1_cate1_id,
        cpp.top1_cate1_name,
        cpp.top1_cate1_score,
        cpp.top2_cate1_id,
        cpp.top2_cate1_name,
        cpp.top2_cate1_score,
        cpp.top3_cate1_id,
        cpp.top3_cate1_name,
        cpp.top3_cate1_score,
        cpp.top10_cate1_preference_path,

        /* 营销 */
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
        mua.last_effective_channel,

        /* 优惠券 */
        coalesce(cua.coupon_receive_cnt_180d, 0) AS coupon_receive_cnt_180d,
        coalesce(cua.coupon_used_cnt_180d, 0) AS coupon_used_cnt_180d,
        coalesce(cua.coupon_expired_cnt_180d, 0) AS coupon_expired_cnt_180d,
        coalesce(cua.coupon_unused_cnt_180d, 0) AS coupon_unused_cnt_180d,
        coalesce(cua.coupon_discount_amount_180d, 0) AS coupon_discount_amount_180d,
        coalesce(cua.platform_coupon_subsidy_180d, 0) AS platform_coupon_subsidy_180d,
        coalesce(cua.shop_coupon_subsidy_180d, 0) AS shop_coupon_subsidy_180d,
        coalesce(cua.coupon_use_rate_180d, 0) AS coupon_use_rate_180d,
        coalesce(cua.coupon_expire_rate_180d, 0) AS coupon_expire_rate_180d,
        cua.last_coupon_receive_time,
        cua.last_coupon_use_time,

        /* 客服 */
        coalesce(serv.service_ticket_cnt_180d, 0) AS service_ticket_cnt_180d,
        coalesce(serv.refund_service_ticket_cnt_180d, 0) AS refund_service_ticket_cnt_180d,
        coalesce(serv.complaint_ticket_cnt_180d, 0) AS complaint_ticket_cnt_180d,
        coalesce(serv.delivery_delay_ticket_cnt_180d, 0) AS delivery_delay_ticket_cnt_180d,
        coalesce(serv.solved_ticket_cnt_180d, 0) AS solved_ticket_cnt_180d,
        coalesce(serv.escalated_ticket_cnt_180d, 0) AS escalated_ticket_cnt_180d,
        coalesce(serv.avg_service_satisfaction_score_180d, 0) AS avg_service_satisfaction_score_180d,
        coalesce(serv.median_service_satisfaction_score_180d, 0) AS median_service_satisfaction_score_180d,
        coalesce(serv.avg_ticket_solve_sec_180d, 0) AS avg_ticket_solve_sec_180d,
        coalesce(serv.median_ticket_solve_sec_180d, 0) AS median_ticket_solve_sec_180d,
        serv.last_service_create_time,
        serv.service_ticket_type_set_180d,

        /* 门店偏好 */
        stp.top1_store_id,
        stp.top1_store_name,
        stp.top1_store_score,
        stp.top5_store_preference_path,

        /* 风控 */
        coalesce(rua.risk_score, 0) AS risk_score,
        coalesce(rua.risk_level, 'RX') AS risk_level,
        coalesce(rua.risk_name, '未知风险') AS risk_name,
        rua.risk_reason_list,

        coalesce(ugr.max_device_bind_user_cnt_90d, 0) AS max_device_bind_user_cnt_90d,
        coalesce(ugr.max_ip_bind_user_cnt_90d, 0) AS max_ip_bind_user_cnt_90d,
        coalesce(ugr.max_address_bind_user_cnt_365d, 0) AS max_address_bind_user_cnt_365d,
        ugr.risky_device_user_sample,
        ugr.risky_ip_user_sample,
        ugr.risky_address_user_sample,

        /* 行转列 */
        coalesce(etp.pivot_app_start_cnt_90d, 0) AS pivot_app_start_cnt_90d,
        coalesce(etp.pivot_page_view_cnt_90d, 0) AS pivot_page_view_cnt_90d,
        coalesce(etp.pivot_search_cnt_90d, 0) AS pivot_search_cnt_90d,
        coalesce(etp.pivot_sku_expose_cnt_90d, 0) AS pivot_sku_expose_cnt_90d,
        coalesce(etp.pivot_sku_click_cnt_90d, 0) AS pivot_sku_click_cnt_90d,
        coalesce(etp.pivot_add_cart_cnt_90d, 0) AS pivot_add_cart_cnt_90d,
        coalesce(etp.pivot_remove_cart_cnt_90d, 0) AS pivot_remove_cart_cnt_90d,
        coalesce(etp.pivot_favorite_cnt_90d, 0) AS pivot_favorite_cnt_90d,
        coalesce(etp.pivot_submit_order_cnt_90d, 0) AS pivot_submit_order_cnt_90d,
        coalesce(etp.pivot_pay_success_cnt_90d, 0) AS pivot_pay_success_cnt_90d,
        coalesce(etp.pivot_refund_apply_cnt_90d, 0) AS pivot_refund_apply_cnt_90d,
        coalesce(etp.pivot_live_enter_cnt_90d, 0) AS pivot_live_enter_cnt_90d,
        coalesce(etp.pivot_live_stay_cnt_90d, 0) AS pivot_live_stay_cnt_90d,
        coalesce(etp.pivot_store_visit_cnt_90d, 0) AS pivot_store_visit_cnt_90d,
        coalesce(etp.pivot_scan_qr_cnt_90d, 0) AS pivot_scan_qr_cnt_90d,
        coalesce(etp.pivot_customer_service_click_cnt_90d, 0) AS pivot_customer_service_click_cnt_90d,

        /* metric map */
        umm.metric_value_map,
        umm.metric_group_map,

        /* 标签 */
        coalesce(uta.tag_code_list, '') AS tag_code_list,
        coalesce(uta.tag_name_list, '') AS tag_name_list,
        coalesce(uta.tag_cnt, 0) AS tag_cnt,

        /* AB */
        coalesce(abu.ab_exp_cnt_90d, 0) AS ab_exp_cnt_90d,
        coalesce(abu.ab_exp_group_list_90d, '') AS ab_exp_group_list_90d,

        /* 半连接、反连接结果 */
        coalesce(anc.active_non_risk_marker, 0) AS is_active_non_risk_candidate,
        coalesce(anp.active_no_pay_marker, 0) AS is_active_no_pay_user,

        /* 综合分层 */
        CASE
            WHEN coalesce(rua.risk_score, 0) >= 90 THEN 'S00_BLOCK'
            WHEN coalesce(rua.risk_score, 0) >= 70 THEN 'S01_MANUAL_REVIEW'
            WHEN coalesce(tua.net_pay_amount_365d, 0) >= 5000
             AND coalesce(eua.event_cnt_30d, 0) >= 10
             AND coalesce(tua.refund_rate_365d, 0) < 0.2 THEN 'S02_SUPER_VALUE_ACTIVE'
            WHEN coalesce(tua.net_pay_amount_365d, 0) >= 1000
             AND coalesce(eua.event_cnt_30d, 0) >= 3 THEN 'S03_HIGH_VALUE_ACTIVE'
            WHEN coalesce(tua.net_pay_amount_365d, 0) >= 1000
             AND coalesce(eua.days_since_last_event, 9999) >= 30 THEN 'S04_HIGH_VALUE_SILENT'
            WHEN ub.register_days <= 30
             AND coalesce(eua.event_cnt_7d, 0) >= 3 THEN 'S05_NEW_GROWING'
            WHEN coalesce(eua.add_cart_cnt_90d, 0) > coalesce(eua.pay_success_cnt_90d, 0)
             AND coalesce(tua.paid_order_cnt_30d, 0) = 0 THEN 'S06_CART_LOST'
            WHEN coalesce(cua.coupon_use_rate_180d, 0) >= 0.6
             AND coalesce(tua.net_pay_amount_90d, 0) > 0 THEN 'S07_COUPON_SENSITIVE_BUYER'
            WHEN coalesce(mua.marketing_roi_90d, 0) >= 0.8 THEN 'S08_MARKETING_POSITIVE'
            WHEN coalesce(eua.days_since_last_event, 9999) >= 90 THEN 'S09_LOST'
            ELSE 'S10_NORMAL'
        END AS final_user_segment,

        md5(
            concat_ws(
                '|',
                ub.user_id,
                coalesce(lcf.final_life_cycle_code, ''),
                coalesce(cast(eua.event_cnt_90d AS string), '0'),
                coalesce(cast(tua.net_pay_amount_365d AS string), '0'),
                coalesce(cast(rua.risk_score AS string), '0'),
                coalesce(uta.tag_code_list, '')
            )
        ) AS user_feature_hash,

        current_timestamp() AS etl_time

    FROM user_base ub
    LEFT JOIN user_preference_map upm ON ub.user_id = upm.user_id
    LEFT JOIN life_cycle_final lcf ON ub.user_id = lcf.user_id
    LEFT JOIN event_user_agg eua ON ub.user_id = eua.user_id
    LEFT JOIN virtual_session_user_agg vsua ON ub.user_id = vsua.user_id
    LEFT JOIN exposure_user_agg exua ON ub.user_id = exua.user_id
    LEFT JOIN search_user_agg sua ON ub.user_id = sua.user_id
    LEFT JOIN trade_user_agg tua ON ub.user_id = tua.user_id
    LEFT JOIN sku_preference_pivot spp ON ub.user_id = spp.user_id
    LEFT JOIN category_preference_pivot cpp ON ub.user_id = cpp.user_id
    LEFT JOIN marketing_user_agg mua ON ub.user_id = mua.user_id
    LEFT JOIN coupon_user_agg cua ON ub.user_id = cua.user_id
    LEFT JOIN service_user_agg serv ON ub.user_id = serv.user_id
    LEFT JOIN store_preference_pivot stp ON ub.user_id = stp.user_id
    LEFT JOIN risk_user_agg rua ON ub.user_id = rua.user_id
    LEFT JOIN user_graph_risk ugr ON ub.user_id = ugr.user_id
    LEFT JOIN event_type_pivot etp ON ub.user_id = etp.user_id
    LEFT JOIN user_metric_map umm ON ub.user_id = umm.user_id
    LEFT JOIN user_tag_agg uta ON ub.user_id = uta.user_id
    LEFT JOIN ab_user_agg abu ON ub.user_id = abu.user_id
    LEFT JOIN active_non_risk_candidate anc ON ub.user_id = anc.user_id
    LEFT JOIN active_no_pay_user anp ON ub.user_id = anp.user_id
)

/* ============================================================
   48. 多目标写入：用户超级宽表
   ============================================================ */
FROM final_user_extreme_wide f

INSERT OVERWRITE TABLE ads_user.ads_user_extreme_operation_profile_df
PARTITION (dt = '${biz_date}', biz_line = 'NEW_RETAIL_ECOM')
SELECT
    *
WHERE f.user_id IS NOT NULL
  AND coalesce(f.is_employee, 0) = 0

/* ============================================================
   49. 同一条 SQL 派生写入：高价值用户池
   ============================================================ */
INSERT OVERWRITE TABLE ads_user.ads_high_value_user_pool_df
PARTITION (dt = '${biz_date}')
SELECT
    f.user_id,
    f.mobile,
    f.email,
    f.province,
    f.city,
    f.member_level,
    f.final_life_cycle_code,
    f.final_user_segment,
    f.net_pay_amount_365d,
    f.gross_margin_rate_365d,
    f.marketing_roi_90d,
    f.risk_score,
    f.tag_code_list,
    f.user_feature_hash,
    current_timestamp() AS etl_time
WHERE f.net_pay_amount_365d >= 1000
  AND f.risk_score < 70

/* ============================================================
   50. 同一条 SQL 派生写入：召回用户池
   ============================================================ */
INSERT OVERWRITE TABLE ads_user.ads_user_recall_pool_df
PARTITION (dt = '${biz_date}')
SELECT
    f.user_id,
    f.mobile,
    f.email,
    f.province,
    f.city,
    f.final_life_cycle_code,
    f.days_since_last_event,
    f.days_since_last_pay,
    f.net_pay_amount_365d,
    f.top1_sku_id,
    f.top1_sku_name,
    f.top1_cate1_id,
    f.top1_cate1_name,
    f.coupon_use_rate_180d,
    f.marketing_channel_set_90d,
    CASE
        WHEN f.net_pay_amount_365d >= 5000 THEN 'RECALL_SUPER_VALUE'
        WHEN f.net_pay_amount_365d >= 1000 THEN 'RECALL_HIGH_VALUE'
        WHEN f.add_cart_cnt_90d > f.pay_success_cnt_90d THEN 'RECALL_CART_LOST'
        ELSE 'RECALL_NORMAL'
    END AS recall_strategy_code,
    current_timestamp() AS etl_time
WHERE f.risk_score < 70
  AND (
      f.days_since_last_event >= 30
      OR f.days_since_last_pay >= 30
      OR f.is_active_no_pay_user = 1
  )
;
