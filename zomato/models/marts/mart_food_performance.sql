{{ config(
    materialized='table'
) }}

WITH food_performance AS (

    SELECT
        f_id,

        COUNT(DISTINCT order_id) AS total_orders,

        COUNT(*) AS total_order_items,

        SUM(quantity) AS total_quantity_sold,

        SUM(line_amount) AS total_revenue,

        AVG(price) AS average_item_price,

        AVG(line_amount) AS average_line_amount,

        COUNT(DISTINCT restaurant_id) AS restaurants_selling,

        COUNT(DISTINCT city) AS cities_selling,

        MIN(order_date) AS first_order_date,

        MAX(order_date) AS last_order_date

    FROM {{ ref('fact_order_items') }}

    GROUP BY f_id
)

SELECT
    f.f_id,
    f.food_name,
    f.veg_or_non_veg,

    COALESCE(p.total_orders, 0) AS total_orders,

    COALESCE(p.total_order_items, 0) AS total_order_items,

    COALESCE(p.total_quantity_sold, 0) AS total_quantity_sold,

    ROUND(
        COALESCE(p.total_revenue, 0),
        2
    ) AS total_revenue,

    ROUND(
        COALESCE(p.average_item_price, 0),
        2
    ) AS average_item_price,

    ROUND(
        COALESCE(p.average_line_amount, 0),
        2
    ) AS average_line_amount,

    COALESCE(p.restaurants_selling, 0) AS restaurants_selling,

    COALESCE(p.cities_selling, 0) AS cities_selling,

    p.first_order_date,

    p.last_order_date

FROM {{ ref('dim_food') }} f

LEFT JOIN food_performance p
    ON f.f_id = p.f_id