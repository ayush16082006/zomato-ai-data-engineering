select
    order_date,
    city,

    count(*) as orders,

    countif(is_delivered) as delivered_orders,

    round(
        safe_divide(
            countif(order_status = 'Cancelled'),
            count(*)
        ),
        4
    ) as cancel_rate,

    sum(
        if(is_delivered, sales_amount, 0)
    ) as gmv,

    round(
        safe_divide(
            sum(if(is_delivered, sales_amount, 0)),
            countif(is_delivered)
        ),
        2
    ) as aov

from {{ ref('fct_orders') }}

group by 1, 2