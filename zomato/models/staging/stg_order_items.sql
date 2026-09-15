select
    order_item_id,
    order_id,
    safe_cast(r_id as int64) as restaurant_id,
    f_id,
    safe_cast(price as numeric) as price,
    safe_cast(quantity as int64) as quantity,
    safe_cast(line_amount as numeric) as line_amount
from {{ source('zomato_raw', 'order_items') }}