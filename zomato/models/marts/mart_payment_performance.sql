{{ config(
    materialized='table'
) }}

WITH payment_performance AS (

    SELECT
        payment_method,

        COUNT(*) AS total_orders,

        COUNT(DISTINCT customer_id) AS unique_customers,

        COUNT(DISTINCT restaurant_id) AS unique_restaurants,

        COUNT(DISTINCT city) AS cities_count,

        COUNTIF(
            LOWER(order_status) = 'delivered'
        ) AS delivered_orders,

        COUNTIF(
            LOWER(order_status) = 'cancelled'
        ) AS cancelled_orders,

        COUNTIF(
            is_delivered = TRUE
        ) AS successful_deliveries,

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
        ) AS average_delivery_time_min

    FROM {{ ref('fct_orders') }}

    WHERE payment_method IS NOT NULL
      AND TRIM(payment_method) != ''

    GROUP BY payment_method
)

SELECT
    payment_method,

    total_orders,

    unique_customers,

    unique_restaurants,

    cities_count,

    delivered_orders,

    cancelled_orders,

    successful_deliveries,

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
    ) AS delivery_success_rate

FROM payment_performance