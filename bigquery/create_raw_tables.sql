-- Zomato Data Engineering Project
-- RAW / BRONZE Layer
-- Create RAW tables from the ZOMATO landing/source dataset

-- Create RAW tables using the existing source tables.
-- SELECT * preserves the original columns and data types.

CREATE OR REPLACE TABLE
`zomato-ai-data-engineering.raw.restaurant` AS
SELECT *
FROM `zomato-ai-data-engineering.zomato.restaurant`;


CREATE OR REPLACE TABLE
`zomato-ai-data-engineering.raw.users` AS
SELECT *
FROM `zomato-ai-data-engineering.zomato.users`;


CREATE OR REPLACE TABLE
`zomato-ai-data-engineering.raw.food` AS
SELECT *
FROM `zomato-ai-data-engineering.zomato.food`;


CREATE OR REPLACE TABLE
`zomato-ai-data-engineering.raw.menu` AS
SELECT *
FROM `zomato-ai-data-engineering.zomato.menu`;


CREATE OR REPLACE TABLE
`zomato-ai-data-engineering.raw.orders` AS
SELECT *
FROM `zomato-ai-data-engineering.zomato.orders`;


CREATE OR REPLACE TABLE
`zomato-ai-data-engineering.raw.order_items` AS
SELECT *
FROM `zomato-ai-data-engineering.zomato.order_items`;


CREATE OR REPLACE TABLE
`zomato-ai-data-engineering.raw.reviews` AS
SELECT *
FROM `zomato-ai-data-engineering.zomato.reviews`;

-- Verification


SELECT 'restaurant' AS table_name, COUNT(*) AS row_count
FROM `zomato-ai-data-engineering.raw.restaurant`

UNION ALL

SELECT 'users', COUNT(*)
FROM `zomato-ai-data-engineering.raw.users`

UNION ALL

SELECT 'food', COUNT(*)
FROM `zomato-ai-data-engineering.raw.food`

UNION ALL

SELECT 'menu', COUNT(*)
FROM `zomato-ai-data-engineering.raw.menu`

UNION ALL

SELECT 'orders', COUNT(*)
FROM `zomato-ai-data-engineering.raw.orders`

UNION ALL

SELECT 'order_items', COUNT(*)
FROM `zomato-ai-data-engineering.raw.order_items`

UNION ALL

SELECT 'reviews', COUNT(*)
FROM `zomato-ai-data-engineering.raw.reviews`

ORDER BY table_name;