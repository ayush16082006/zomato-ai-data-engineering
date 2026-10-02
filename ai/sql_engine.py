import os
import re
import json

import pandas as pd

from google.cloud import bigquery
from openai import OpenAI
from dotenv import load_dotenv


# ============================================================
# LOAD ENVIRONMENT
# ============================================================

load_dotenv()


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ID = "zomato-ai-data-engineering"
BQ_LOCATION = "asia-south1"

# Ollama
OLLAMA_BASE_URL = os.getenv(
    "OLLAMA_BASE_URL",
    "http://localhost:11434/v1"
)

OLLAMA_API_KEY = os.getenv(
    "OLLAMA_API_KEY",
    "ollama"
)

# Your Ollama Cloud model
MODEL = "gpt-oss:120b-cloud"

MAX_ROWS = 100


# ============================================================
# EXAMPLE QUESTIONS
# ============================================================

EXAMPLE_QUESTIONS = [
    "Top 10 cities by GMV",
    "Which cuisine has the most orders?",
    "What are the top 10 restaurants by revenue?",
    "What is the average delivery time by city?",
    "Which cities have the highest cancellation rate?",
    "What are the most common negative review topics?",
    "What are the most common customer complaints?",
    "What is the average sentiment score?",
    "How many negative reviews are there?",
    "Which restaurants have the highest number of orders?",
]


# ============================================================
# BIGQUERY CLIENT
# ============================================================

bq_client = bigquery.Client(project=PROJECT_ID)


# OLLAMA CLIENT
# ============================================================

ollama_client = OpenAI(
    base_url=OLLAMA_BASE_URL,
    api_key=OLLAMA_API_KEY
)


# ============================================================
# VERIFIED ZOMATO PROJECT SCHEMA
# ============================================================

SCHEMA = """
You are working with the Zomato AI Data Engineering project.
Google Cloud project: zomato-ai-data-engineering
BigQuery location: asia-south1

Use ONLY these verified tables and columns. Never invent tables or columns.

RAW REVIEWS
`zomato-ai-data-engineering.raw.reviews`
- review_id INTEGER
- order_id INTEGER
- user_id INTEGER
- restaurant_id INTEGER
- rating INTEGER
- comment STRING
- review_date DATE

AI REVIEW ENRICHMENT
`zomato-ai-data-engineering.ai.review_enriched`
- review_id STRING
- sentiment_label STRING
- sentiment_score FLOAT
- topic STRING
- key_issue STRING
- processed_at TIMESTAMP
- model_used STRING
Use for sentiment, topics, complaints, key issues and enrichment information.
Do NOT use marts.mart_review_insights; it does not exist.

FACT ORDERS
`zomato-ai-data-engineering.marts.fct_orders`
- order_id INTEGER
- order_timestamp TIMESTAMP
- order_date STRING
- customer_id INTEGER
- restaurant_id INTEGER
- city STRING
- cuisine STRING
- payment_method STRING
- order_status STRING
- is_delivered BOOLEAN
- items_count INTEGER
- sales_qty INTEGER
- subtotal INTEGER
- discount INTEGER
- delivery_fee INTEGER
- gst INTEGER
- sales_amount INTEGER
- customer_rating FLOAT
- delivery_time_min FLOAT

FACT ORDER ITEMS
`zomato-ai-data-engineering.marts.fact_order_items`
- order_item_id INTEGER
- order_id INTEGER
- restaurant_id INTEGER
- f_id STRING
- order_ts TIMESTAMP
- order_date DATE
- city STRING
- price NUMERIC
- quantity INTEGER
- line_amount NUMERIC

DIM RESTAURANT
`zomato-ai-data-engineering.marts.dim_restaurant`
- restaurant_id INTEGER
- restaurant_name STRING
- city STRING
- cuisine STRING
- rating NUMERIC
- rating_count INTEGER
- cost_for_two INTEGER

DIM CUSTOMER
`zomato-ai-data-engineering.marts.dim_customer`
- customer_id INTEGER
- customer_name STRING
- email STRING
- age INTEGER
- age_segment STRING
- gender STRING
- marital_status STRING
- occupation STRING
- income_band STRING
- education STRING
- family_size INTEGER
IMPORTANT: dim_customer does NOT contain city or order-performance metrics.

DIM FOOD
`zomato-ai-data-engineering.marts.dim_food`
- f_id STRING
- food_name STRING
- veg_or_non_veg STRING

DIM DATE
`zomato-ai-data-engineering.marts.dim_date`
- date_day DATE
- year INTEGER
- month INTEGER
- month_name STRING
- day_name STRING
- is_weekend BOOLEAN
For fct_orders date analysis, fct_orders.order_date is STRING; use SAFE_CAST(order_date AS DATE) when needed.

MART DAILY CITY REVENUE
`zomato-ai-data-engineering.marts.mart_daily_city_revenue`
- order_date STRING
- city STRING
- orders INTEGER
- delivered_orders INTEGER
- cancel_rate FLOAT
- gmv INTEGER
- aov FLOAT
Use for city/daily revenue, GMV, AOV and cancellation analysis.

MART RESTAURANT PERFORMANCE
`zomato-ai-data-engineering.marts.mart_restaurant_performance`
- restaurant_id INTEGER
- restaurant_name STRING
- city STRING
- cuisine STRING
- orders INTEGER
- revenue INTEGER
- avg_customer_rating FLOAT
- avg_delivery_min FLOAT
Use for restaurant performance.

MART DELIVERY SLA
`zomato-ai-data-engineering.marts.mart_delivery_sla`
- city STRING
- order_hour INTEGER
- delivered_orders INTEGER
- p50 FLOAT
- p90 FLOAT
Use for delivery performance and SLA analysis.

MART CUSTOMER PERFORMANCE
`zomato-ai-data-engineering.marts.mart_customer_performance`
- customer_id INTEGER
- customer_name STRING
- email STRING
- age INTEGER
- age_segment STRING
- gender STRING
- marital_status STRING
- occupation STRING
- income_band STRING
- education STRING
- family_size INTEGER
- city STRING
- total_orders INTEGER
- delivered_orders INTEGER
- cancelled_orders INTEGER
- total_spend INTEGER
- average_order_value FLOAT
- average_rating_given FLOAT
- cancel_rate FLOAT
- first_order_date STRING
- last_order_date STRING
Use for customer performance, spending, orders, cancellation and city analysis.

MART FOOD PERFORMANCE
`zomato-ai-data-engineering.marts.mart_food_performance`
- f_id STRING
- food_name STRING
- veg_or_non_veg STRING
- total_orders INTEGER
- total_order_items INTEGER
- total_quantity_sold INTEGER
- total_revenue NUMERIC
- average_item_price NUMERIC
- average_line_amount NUMERIC
- restaurants_selling INTEGER
- cities_selling INTEGER
- first_order_date DATE
- last_order_date DATE
Use for food performance and item sales.

MART CUISINE PERFORMANCE
`zomato-ai-data-engineering.marts.mart_cuisine_performance`
- cuisine STRING
- total_orders INTEGER
- delivered_orders INTEGER
- cancelled_orders INTEGER
- successful_deliveries INTEGER
- restaurants_count INTEGER
- cities_count INTEGER
- total_items_sold INTEGER
- total_revenue FLOAT
- average_order_value FLOAT
- average_customer_rating FLOAT
- average_delivery_time_min FLOAT
- cancel_rate FLOAT
- delivery_success_rate FLOAT
- first_order_date STRING
- last_order_date STRING
Use for cuisine performance.

MART PAYMENT PERFORMANCE
`zomato-ai-data-engineering.marts.mart_payment_performance`
- payment_method STRING
- total_orders INTEGER
- unique_customers INTEGER
- unique_restaurants INTEGER
- cities_count INTEGER
- delivered_orders INTEGER
- cancelled_orders INTEGER
- successful_deliveries INTEGER
- total_items_sold INTEGER
- total_revenue FLOAT
- average_order_value FLOAT
- average_customer_rating FLOAT
- average_delivery_time_min FLOAT
- cancel_rate FLOAT
- delivery_success_rate FLOAT
Use for payment-method performance.

REVIEW JOIN
Join raw.reviews to ai.review_enriched using:
CAST(raw.reviews.review_id AS STRING) = ai.review_enriched.review_id
when both original review data and enrichment are needed.

BUSINESS DEFINITIONS
- GMV = delivered revenue.
- For fct_orders: SUM(sales_amount) WHERE is_delivered = TRUE.
- AOV = GMV / delivered orders.
- Cancellation rate = cancelled orders / total orders.
- Delivered order = is_delivered = TRUE.
- Restaurant revenue = SUM(sales_amount) for delivered orders.

TABLE SELECTION
Prefer the smallest appropriate verified MART.
- City revenue/GMV -> mart_daily_city_revenue
- Restaurant -> mart_restaurant_performance
- Delivery -> mart_delivery_sla
- Customer -> mart_customer_performance
- Food -> mart_food_performance
- Cuisine -> mart_cuisine_performance
- Payment -> mart_payment_performance
- Order-level -> fct_orders
- Order-item -> fact_order_items
- Customer demographics -> dim_customer
- Restaurant info -> dim_restaurant
- Food info -> dim_food
- Date analysis -> dim_date
- Review sentiment/topics -> ai.review_enriched
- Original comments/ratings -> raw.reviews

Never use marts.mart_review_insights.
Never confuse raw.reviews with ai.review_enriched.
Never assume dim_customer contains city.
Never assume fct_orders.order_date is DATE; it is STRING.
Never assume ai.review_enriched.review_id is INTEGER; it is STRING.
"""


# SYSTEM PROMPT
# ============================================================

SYSTEM_PROMPT = f"""
You are an expert BigQuery SQL generator for a Zomato
data engineering project.

Your task is to convert the user's natural-language question
into ONE safe BigQuery SQL query.

{SCHEMA}


============================================================
SQL GENERATION RULES
============================================================

1. Generate exactly ONE SQL query.

2. Only SELECT or WITH queries are allowed.

3. Never modify data.

4. Never generate:

DROP
DELETE
TRUNCATE
ALTER
UPDATE
INSERT
CREATE
REPLACE
MERGE
GRANT
REVOKE
CALL
EXPORT DATA


5. Always use fully-qualified BigQuery table names.

Example:

`zomato-ai-data-engineering.marts.fct_orders`


6. Prefer the appropriate MART table when it directly answers
the question.


7. For sentiment/topic/complaint questions use:

`zomato-ai-data-engineering.ai.review_enriched`


8. For original review comments and ratings use:

`zomato-ai-data-engineering.raw.reviews`


9. If both original review information and AI enrichment are
needed, join the two tables using review_id.


10. Use LOWER() when comparing text labels if capitalization
could vary.

Example:

WHERE LOWER(sentiment_label) = 'negative'


11. For ranking questions use ORDER BY and LIMIT.


12. Use LIMIT 100 or less for normal result queries.


13. A single aggregate query does not require LIMIT.


14. Use SAFE_DIVIDE for division.


15. Do not invent tables or columns.


16. If the question cannot be answered using the available
schema, return:

{{"sql": null, "reason": "brief explanation"}}


17. Return ONLY valid JSON.

Valid response:

{{"sql": "SELECT ..."}}

or:

{{"sql": null, "reason": "..."}}
"""


# ============================================================
# CLEAN SQL
# ============================================================

def clean_sql(sql):

    if not sql:
        return ""

    sql = sql.strip()

    # Remove ```sql
    sql = re.sub(
        r"^```(?:sql)?\s*",
        "",
        sql,
        flags=re.IGNORECASE
    )

    # Remove ```
    sql = re.sub(
        r"\s*```$",
        "",
        sql
    )

    return sql.strip().rstrip(";").strip()


# ============================================================
# GENERATE SQL
# ============================================================

def generate_sql(question):

    response = ollama_client.chat.completions.create(
        model=MODEL,
        temperature=0,
        messages=[
            {
                "role": "system",
                "content": SYSTEM_PROMPT
            },
            {
                "role": "user",
                "content": question
            }
        ]
    )

    content = response.choices[0].message.content

    if not content:
        raise ValueError(
            "Ollama returned an empty response."
        )

    content = content.strip()

    # Remove markdown JSON fences
    content = re.sub(
        r"^```(?:json)?\s*",
        "",
        content,
        flags=re.IGNORECASE
    )

    content = re.sub(
        r"\s*```$",
        "",
        content
    )

    try:

        result = json.loads(content)

    except json.JSONDecodeError as e:

        raise ValueError(
            "Ollama did not return valid JSON.\n\n"
            f"Model response:\n{content}"
        ) from e

    sql = result.get("sql")

    if sql is None:

        return (
            None,
            result.get(
                "reason",
                "The question cannot be answered using "
                "the available Zomato data."
            )
        )

    return clean_sql(sql), None


# ============================================================
# SQL SAFETY
# ============================================================

FORBIDDEN_PATTERNS = [
    r"\bDROP\b",
    r"\bDELETE\b",
    r"\bTRUNCATE\b",
    r"\bALTER\b",
    r"\bUPDATE\b",
    r"\bINSERT\b",
    r"\bCREATE\b",
    r"\bREPLACE\b",
    r"\bMERGE\b",
    r"\bGRANT\b",
    r"\bREVOKE\b",
    r"\bCALL\b",
    r"\bEXPORT\s+DATA\b",
]


def is_safe_sql(sql):

    if not sql:
        return False, "SQL query is empty."

    sql = sql.strip()

    # Must begin with SELECT or WITH
    if not re.match(
        r"^(SELECT|WITH)\b",
        sql,
        re.IGNORECASE
    ):
        return (
            False,
            "Only SELECT/WITH queries are allowed."
        )

    # Prevent multiple statements
    if ";" in sql:
        return (
            False,
            "Multiple SQL statements are not allowed."
        )

    # Check dangerous operations
    for pattern in FORBIDDEN_PATTERNS:

        if re.search(
            pattern,
            sql,
            flags=re.IGNORECASE
        ):
            return (
                False,
                f"Forbidden SQL operation detected: {pattern}"
            )

    return True, None


# ============================================================
# RUN BIGQUERY
# ============================================================

def run_query(sql):

    query_job = bq_client.query(
        sql,
        location=BQ_LOCATION
    )

    result = query_job.result()

    return result.to_dataframe()


# ============================================================
# SIMPLE SQL ENGINE API
# ============================================================

def ask_sql(question):
    """Generate safe SQL, execute it, and return a structured result."""
    category = classify_question(question)
    sql, reason = generate_sql(question)

    if sql is None:
        return {
            "success": False,
            "question": question,
            "category": category,
            "sql": None,
            "data": pd.DataFrame(),
            "reason": reason,
            "error": None,
        }

    safe, safety_error = is_safe_sql(sql)
    if not safe:
        return {
            "success": False,
            "question": question,
            "category": category,
            "sql": sql,
            "data": pd.DataFrame(),
            "reason": None,
            "error": safety_error,
        }

    try:
        df = run_query(sql)
        return {
            "success": True,
            "question": question,
            "category": category,
            "sql": sql,
            "data": df,
            "reason": None,
            "error": None,
        }
    except Exception as e:
        return {
            "success": False,
            "question": question,
            "category": category,
            "sql": sql,
            "data": pd.DataFrame(),
            "reason": None,
            "error": str(e),
        }


# ============================================================
# QUESTION CATEGORY
# ============================================================

def classify_question(question):

    q = question.lower()

    if any(
        word in q
        for word in [
            "sentiment",
            "positive review",
            "negative review",
            "neutral review",
            "complaint",
            "complaints",
            "review topic",
            "review topics",
            "key issue",
            "issues",
            "feedback"
        ]
    ):
        return "Review analytics"

    if any(
        word in q
        for word in [
            "delivery",
            "late",
            "p50",
            "p90",
            "sla"
        ]
    ):
        return "Delivery analytics"

    if any(
        word in q
        for word in [
            "restaurant",
            "restaurants",
            "cuisine"
        ]
    ):
        return "Restaurant analytics"

    if any(
        word in q
        for word in [
            "customer",
            "customers",
            "user",
            "users",
            "age",
            "gender"
        ]
    ):
        return "Customer analytics"

    if any(
        word in q
        for word in [
            "gmv",
            "revenue",
            "sales",
            "order",
            "orders"
        ]
    ):
        return "Order/revenue analytics"

    return "General analytics"


# ============================================================

# ============================================================
# OPTIONAL DIRECT TEST
# ============================================================

if __name__ == "__main__":
    print("=" * 70)
    print("ZOMATO SQL ENGINE")
    print("=" * 70)
    print(f"Project  : {PROJECT_ID}")
    print(f"Location : {BQ_LOCATION}")
    print(f"Model    : {MODEL}")
    print("=" * 70)

    question = input("Enter a test question (or press Enter to exit): ").strip()
    if question:
        result = ask_sql(question)
        print("\nCategory:", result["category"])
        if result["sql"]:
            print("\nGenerated SQL:\n", result["sql"])
        if result["success"]:
            print(f"\nRows returned: {len(result['data']):,}")
            print(result["data"].head(10).to_string(index=False))
        elif result["reason"]:
            print("\nReason:", result["reason"])
        elif result["error"]:
            print("\nError:", result["error"])
