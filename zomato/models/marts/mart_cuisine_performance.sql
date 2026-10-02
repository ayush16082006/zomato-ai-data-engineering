    {{ config(
    materialized='table'
) }}

WITH cuisine_performance AS (

    SELECT
        cuisine,

        COUNT(*) AS total_orders,

        COUNTIF(
            LOWER(order_status) = 'delivered'
        ) AS delivered_orders,

        COUNTIF(
            LOWER(order_status) = 'cancelled'
        ) AS cancelled_orders,

        COUNTIF(
            is_delivered = TRUE
        ) AS successful_deliveries,

        COUNT(DISTINCT restaurant_id) AS restaurants_count,

        COUNT(DISTINCT city) AS cities_count,

        SUM(
            CASE
                WHEN is_delivered = TRUE
                THEN sales_amount
                ELSE 0
            END
        ) AS total_revenue,

        SUM(
            CASE
                WHEN is_delivered = TRUE
                THEN sales_qty
                ELSE 0
            END
        ) AS total_items_sold,

        AVG(
            CASE
                WHEN is_delivered = TRUE
                THEN sales_amount
            END
        ) AS average_order_value,

        AVG(
            CASE
                WHEN customer_rating IS NOT NULL
                THEN customer_rating
            END
        ) AS average_customer_rating,

        AVG(
            CASE
                WHEN is_delivered = TRUE
                THEN delivery_time_min
            END
        ) AS average_delivery_time_min,

        MIN(order_date) AS first_order_date,

        MAX(order_date) AS last_order_date

    FROM {{ ref('fct_orders') }}

    WHERE cuisine IS NOT NULL
      AND TRIM(cuisine) != ''

    GROUP BY cuisine
)

SELECT
    cuisine,

    total_orders,

    delivered_orders,

    cancelled_orders,

    successful_deliveries,

    restaurants_count,

    cities_count,

    total_items_sold,

    ROUND(
        total_revenue,
        2
    ) AS total_revenue,

    ROUND(
        average_order_value,
        2
    ) AS average_order_value,

    ROUND(
        average_customer_rating,
        2
    ) AS average_customer_rating,

    ROUND(
        average_delivery_time_min,
        2
    ) AS average_delivery_time_min,

    ROUND(
        SAFE_DIVIDE(
            cancelled_orders,
            total_orders
        ) * 100,
        2
    ) AS cancel_rate,

    ROUND(
        SAFE_DIVIDE(
            delivered_orders,
            total_orders
        ) * 100,
        2
    ) AS delivery_success_rate,

    first_order_date,

    last_order_date

FROM cuisine_performance