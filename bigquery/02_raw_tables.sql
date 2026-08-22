
-- Zomato AI Data Engineering Project
-- BigQuery RAW Tables
-- RAW = Bronze layer
-- These tables represent the source Zomato data.
-- Dimension/source CSVs may contain a leading unnamed index
-- column. We keep it as _idx in RAW and remove it later
-- during the dbt staging/Silver transformation.



-- RESTAURANTS

CREATE OR REPLACE TABLE
`zomato-ai-data-engineering.raw.restaurants`
(
  _idx STRING,
  id STRING,
  name STRING,
  city STRING,
  rating STRING,
  rating_count STRING,
  cost STRING,
  cuisine STRING,
  lic_no STRING,
  link STRING,
  address STRING,
  menu STRING
);



-- USERS


CREATE OR REPLACE TABLE
`zomato-ai-data-engineering.raw.users`
(
  _idx STRING,
  user_id STRING,
  name STRING,
  email STRING,
  password STRING,
  age STRING,
  gender STRING,
  marital_status STRING,
  occupation STRING,
  monthly_income STRING,
  education STRING,
  family_size STRING
);



-- FOOD


CREATE OR REPLACE TABLE
`zomato-ai-data-engineering.raw.food`
(
  _idx STRING,
  f_id STRING,
  item STRING,
  veg_or_non_veg STRING
);



-- MENU


CREATE OR REPLACE TABLE
`zomato-ai-data-engineering.raw.menu`
(
  _idx STRING,
  menu_id STRING,
  r_id STRING,
  f_id STRING,
  cuisine STRING,
  price STRING
);



-- ORDERS

CREATE OR REPLACE TABLE
`zomato-ai-data-engineering.raw.orders`
(
  order_id INT64,
  order_timestamp TIMESTAMP,
  order_date DATE,
  user_id INT64,
  r_id INT64,
  restaurant_city STRING,
  cuisine STRING,
  items_count INT64,
  sales_qty INT64,
  subtotal NUMERIC,
  discount NUMERIC,
  delivery_fee NUMERIC,
  gst NUMERIC,
  sales_amount NUMERIC,
  currency STRING,
  payment_method STRING,
  order_status STRING,
  customer_rating NUMERIC,
  delivery_time_min INT64
);



-- ORDER ITEMS

CREATE OR REPLACE TABLE
`zomato-ai-data-engineering.raw.order_items`
(
  order_item_id INT64,
  order_id INT64,
  r_id INT64,
  f_id STRING,
  price NUMERIC,
  quantity INT64,
  line_amount NUMERIC
);



-- REVIEWS

CREATE OR REPLACE TABLE
`zomato-ai-data-engineering.raw.reviews`
(
  review_id INT64,
  order_id INT64,
  user_id INT64,
  restaurant_id INT64,
  rating INT64,
  comment STRING,
  review_date DATE
);