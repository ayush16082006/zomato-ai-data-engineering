select
    city,
    extract(hour from order_timestamp) as order_hour,
    countif(is_delivered) as delivered_orders,

    round(
        approx_quantiles(delivery_time_min, 100)[offset(50)],
        1
    ) as p50,

    round(
        approx_quantiles(delivery_time_min, 100)[offset(90)],
        1
    ) as p90

from {{ ref('fct_orders') }}

where is_delivered

group by 1, 2