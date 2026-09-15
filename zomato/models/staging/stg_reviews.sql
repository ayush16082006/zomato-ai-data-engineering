select
    r.review_id,
    r.order_id,
    safe_cast(r.user_id as int64) as customer_id,
    safe_cast(r.restaurant_id as string) as restaurant_id,
    safe_cast(r.rating as int64) as rating,
    safe_cast(r.comment as string) as comment,
    safe_cast(r.review_date as date) as review_date,
    res.city as city

from {{ source('zomato_raw', 'reviews') }} r

left join {{ ref('stg_restaurant') }} res
    on safe_cast(r.restaurant_id as string) = safe_cast(res.restaurant_id as string)

where r.comment is not null