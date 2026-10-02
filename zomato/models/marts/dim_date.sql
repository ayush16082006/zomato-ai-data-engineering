with date_spine as (

    select
        date_day
    from unnest(
        generate_date_array(
            date '2024-01-01',
            date '2026-12-31',
            interval 1 day
        )
    ) as date_day

)

select
    date_day,
    extract(year from date_day) as year,
    extract(month from date_day) as month,

    format_date('%B', date_day) as month_name,

    format_date('%A', date_day) as day_name,

    extract(dayofweek from date_day) in (1, 7) as is_weekend

from date_spine