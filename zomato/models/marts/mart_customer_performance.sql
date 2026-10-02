{{ config(
    materialized='table'
) }}

WITH customer_orders AS (

    SELECT
        customer_id,

        COUNT(*) AS total_orders,

        COUNTIF(
            LOWER(order_status) = 'delivered'
        ) AS delivered_orders,

        COUNTIF(
            LOWER(order_status) = 'cancelled'
        ) AS cancelled_orders,

        SUM(
            CASE
                WHEN LOWER(order_status) = 'delivered'
                THEN sales_amount
                ELSE 0
            END
        ) AS total_spend,

        AVG(
            CASE
                WHEN LOWER(order_status) = 'delivered'
                THEN sales_amount
            END
        ) AS average_order_value,

        AVG(
            CASE
                WHEN LOWER(order_status) = 'delivered'
                THEN customer_rating
            END
        ) AS average_rating_given,

        MIN(order_date) AS first_order_date,

        MAX(order_date) AS last_order_date

    FROM {{ ref('fct_orders') }}

    GROUP BY customer_id
),

customer_city AS (

    SELECT
        customer_id,
        ARRAY_AGG(
            city
            IGNORE NULLS
            ORDER BY order_date DESC
            LIMIT 1
        )[SAFE_OFFSET(0)] AS latest_city

    FROM {{ ref('fct_orders') }}

    GROUP BY customer_id
),

final AS (

    SELECT
        c.customer_id,
        c.customer_name,
        c.email,
        c.age,
        c.age_segment,
        c.gender,
        c.marital_status,
        c.occupation,
        c.income_band,
        c.education,
        c.family_size,

        cc.latest_city AS city,

        COALESCE(o.total_orders, 0) AS total_orders,

        COALESCE(o.delivered_orders, 0) AS delivered_orders,

        COALESCE(o.cancelled_orders, 0) AS cancelled_orders,

        COALESCE(o.total_spend, 0) AS total_spend,

        ROUND(
            COALESCE(o.average_order_value, 0),
            2
        ) AS average_order_value,

        ROUND(
            COALESCE(o.average_rating_given, 0),
            2
        ) AS average_rating_given,

        ROUND(
            SAFE_DIVIDE(
                COALESCE(o.cancelled_orders, 0),
                NULLIF(o.total_orders, 0)
            ) * 100,
            2
        ) AS cancel_rate,

        o.first_order_date,

        o.last_order_date

    FROM {{ ref('dim_customer') }} c

    LEFT JOIN customer_orders o
        ON c.customer_id = o.customer_id

    LEFT JOIN customer_city cc
        ON c.customer_id = cc.customer_id
)

SELECT *
FROM final