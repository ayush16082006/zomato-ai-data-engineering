select
    order_id,
    order_timestamp,
    order_date,
    user_id as customer_id,
    safe_cast(r_id as int64) as restaurant_id,

    trim(
        coalesce(
            regexp_extract(restaurant_city, r'[^,]+$'),
            restaurant_city
        )
    ) as city,

    cuisine,
    items_count,
    sales_qty,
    subtotal,
    discount,
    delivery_fee,
    gst,
    sales_amount,
    currency,
    payment_method,
    order_status,

    order_status = 'Delivered' as is_delivered,

    customer_rating,
    delivery_time_min

from {{ source('zomato_raw', 'orders') }}