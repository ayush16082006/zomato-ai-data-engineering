{{ config(
    materialized='view'
) }}

select
    oi.order_item_id,
    oi.order_id,
    oi.restaurant_id,
    oi.f_id,
    o.order_timestamp as order_ts,
    DATE(o.order_date) as order_date,
    o.city,
    oi.price,
    oi.quantity,
    oi.line_amount

from {{ ref('stg_order_items') }} oi

inner join {{ ref('stg_orders') }} o
    using (order_id)