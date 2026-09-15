select
    menu_id,
    safe_cast(r_id as int64) as restaurant_id,
    f_id,
    cuisine,
    safe_cast(price as numeric) as price
from {{ source('zomato_raw', 'menu') }}
where safe_cast(r_id as int64) is not null
  and safe_cast(price as numeric) > 0