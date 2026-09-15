select
    safe_cast(user_id as int64) as customer_id,
    name as customer_name,
    lower(email) as email,

    safe_cast(`Age` as int64) as age,

    `Gender` as gender,

    `Marital Status` as marital_status,

    `Occupation` as occupation,

    `Monthly Income` as income_band,

    `Educational Qualifications` as education,

    safe_cast(`Family size` as int64) as family_size

from {{ source('zomato_raw', 'users') }}

where safe_cast(user_id as int64) is not null