import os
import re
import json

import pandas as pd
import streamlit as st

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

try:

    bq_client = bigquery.Client(
        project=PROJECT_ID
    )

except Exception as e:

    st.error(
        f"Could not connect to BigQuery: {e}"
    )

    st.stop()


# ============================================================
# OLLAMA CLIENT
# ============================================================

ollama_client = OpenAI(
    base_url=OLLAMA_BASE_URL,
    api_key=OLLAMA_API_KEY
)


# ============================================================
# ACTUAL ZOMATO PROJECT SCHEMA
# ============================================================

SCHEMA = """
You are working with the Zomato AI Data Engineering project.

Google Cloud project:
zomato-ai-data-engineering

BigQuery location:
asia-south1


============================================================
IMPORTANT TABLES
============================================================


------------------------------------------------------------
RAW REVIEWS
------------------------------------------------------------

Table:

`zomato-ai-data-engineering.raw.reviews`

Columns:

- review_id INTEGER
- order_id INTEGER
- user_id INTEGER
- restaurant_id INTEGER
- rating INTEGER
- comment STRING
- review_date DATE


------------------------------------------------------------
AI REVIEW ENRICHMENT
------------------------------------------------------------

Table:

`zomato-ai-data-engineering.ai.review_enriched`

Columns:

- review_id STRING
- sentiment_label STRING
- sentiment_score FLOAT
- topic STRING
- key_issue STRING
- processed_at TIMESTAMP
- model_used STRING

IMPORTANT:

This table contains the AI-enriched review information.

Use this table for questions about:

- sentiment
- positive reviews
- negative reviews
- neutral reviews
- sentiment score
- topics
- complaints
- issues
- key issues
- review enrichment
- models used for enrichment

Do NOT use a nonexistent table called:

marts.mart_review_insights


============================================================
STAGING TABLES
============================================================


`zomato-ai-data-engineering.staging.stg_food`

Columns:

- f_id
- food_name
- veg_or_non_veg


`zomato-ai-data-engineering.staging.stg_menu`

Columns:

- menu_id
- restaurant_id
- f_id
- cuisine
- price


`zomato-ai-data-engineering.staging.stg_order_items`

Columns:

- order_item_id
- order_id
- restaurant_id
- f_id
- price
- quantity
- line_amount


`zomato-ai-data-engineering.staging.stg_orders`

Columns:

- order_id
- order_timestamp
- order_date
- customer_id
- restaurant_id
- city
- cuisine
- items_count
- sales_qty
- subtotal
- discount
- delivery_fee
- gst
- sales_amount
- currency
- payment_method
- order_status
- is_delivered
- customer_rating
- delivery_time_min


`zomato-ai-data-engineering.staging.stg_restaurant`

Columns:

- restaurant_id
- restaurant_name
- city
- rating
- rating_count
- cost_for_two
- cuisine
- license_no


`zomato-ai-data-engineering.staging.stg_reviews`

Columns:

- review_id
- order_id
- customer_id
- restaurant_id
- rating
- comment
- review_date
- city


`zomato-ai-data-engineering.staging.stg_users`

Columns:

- customer_id
- customer_name
- email
- age
- gender
- marital_status
- occupation
- income_band
- education
- family_size


============================================================
MART TABLES
============================================================


------------------------------------------------------------
FACT ORDERS
------------------------------------------------------------

Table:

`zomato-ai-data-engineering.marts.fct_orders`

Columns:

- order_id
- order_timestamp
- order_date
- customer_id
- restaurant_id
- city
- cuisine
- payment_method
- order_status
- is_delivered
- items_count
- sales_qty
- subtotal
- discount
- delivery_fee
- gst
- sales_amount
- customer_rating
- delivery_time_min


Use this table for:

- orders
- revenue
- sales
- GMV
- payment methods
- order status
- customer rating
- delivery time
- order analysis


------------------------------------------------------------
FACT ORDER ITEMS
------------------------------------------------------------

Table:

`zomato-ai-data-engineering.marts.fact_order_items`

Columns:

- order_item_id
- order_id
- restaurant_id
- f_id
- order_ts
- order_date
- city
- price
- quantity
- line_amount


Use this for:

- food item sales
- item quantities
- item revenue
- order item analysis


------------------------------------------------------------
RESTAURANT
------------------------------------------------------------

Table:

`zomato-ai-data-engineering.marts.dim_restaurant`

Columns:

- restaurant_id
- restaurant_name
- city
- cuisine
- rating
- rating_count
- cost_for_two


Use this for:

- restaurant information
- cuisine
- restaurant rating
- restaurant city
- cost for two


------------------------------------------------------------
CUSTOMER
------------------------------------------------------------

Table:

`zomato-ai-data-engineering.marts.dim_customer`

Columns:

- customer_id
- customer_name
- email
- age
- age_segment
- gender
- marital_status
- occupation
- income_band
- education
- family_size


Use this for:

- customer demographics
- age
- gender
- occupation
- education
- income
- family size


------------------------------------------------------------
DAILY CITY REVENUE
------------------------------------------------------------

Table:

`zomato-ai-data-engineering.marts.mart_daily_city_revenue`

Columns:

- order_date
- city
- orders
- delivered_orders
- cancel_rate
- gmv
- aov


Use this table for:

- city GMV
- city revenue
- AOV
- daily city performance
- cancellation rate


------------------------------------------------------------
RESTAURANT PERFORMANCE
------------------------------------------------------------

Table:

`zomato-ai-data-engineering.marts.mart_restaurant_performance`

Columns:

- restaurant_id
- restaurant_name
- city
- cuisine
- orders
- revenue
- avg_customer_rating
- avg_delivery_min


Use this for:

- restaurant revenue
- restaurant orders
- restaurant performance
- restaurant ratings
- restaurant delivery performance
- cuisine performance


------------------------------------------------------------
DELIVERY SLA
------------------------------------------------------------

Table:

`zomato-ai-data-engineering.marts.mart_delivery_sla`

Columns:

- city
- order_hour
- delivered_orders
- p50
- p90


Use this for:

- delivery performance
- median delivery time
- p50 delivery time
- p90 delivery time
- delivery performance by city
- delivery performance by hour


============================================================
REVIEW TABLE RELATIONSHIP
============================================================

For original review information use:

`zomato-ai-data-engineering.raw.reviews`

For AI-enriched review information use:

`zomato-ai-data-engineering.ai.review_enriched`

They can be joined using:

CAST(raw.reviews.review_id AS STRING)
=
ai.review_enriched.review_id


============================================================
BUSINESS DEFINITIONS
============================================================

GMV:

Delivered revenue.

Use:

SUM(sales_amount)

with:

is_delivered = TRUE


Average Order Value:

GMV / delivered orders.


Cancellation rate:

cancelled orders / total orders.


Delivered order:

is_delivered = TRUE.


Restaurant revenue:

SUM(sales_amount)

for delivered orders.


============================================================
TABLE SELECTION RULES
============================================================

Use the smallest appropriate table.

For city revenue/GMV:

marts.mart_daily_city_revenue


For restaurant performance:

marts.mart_restaurant_performance


For delivery:

marts.mart_delivery_sla


For order-level analysis:

marts.fct_orders


For customer analysis:

marts.dim_customer


For restaurant information:

marts.dim_restaurant


For review sentiment/topics/issues:

ai.review_enriched


For original review comments/ratings:

raw.reviews


For combined review + sentiment information:

JOIN raw.reviews with ai.review_enriched


============================================================
IMPORTANT
============================================================

Never invent a table.

Never invent a column.

Never use:

marts.mart_review_insights

because it does not exist.

Never assume raw.restaurant and raw.restaurants are the same table.

Use fully-qualified BigQuery table names.

Example:

`zomato-ai-data-engineering.ai.review_enriched`
"""


# ============================================================
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
# STREAMLIT CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Zomato Text-to-SQL",
    page_icon="🍴",
    layout="wide"
)


# ============================================================
# TITLE
# ============================================================

st.title("🍴 Chat with your Zomato Data")

st.caption(
    "Natural Language → Ollama Cloud → BigQuery SQL → Results"
)

st.divider()


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("Example Questions")

    for q in EXAMPLE_QUESTIONS:

        if st.button(
            q,
            use_container_width=True
        ):

            st.session_state[
                "selected_question"
            ] = q

    st.divider()

    st.subheader("Current Configuration")

    st.write(
        f"**LLM:** `{MODEL}`"
    )

    st.write(
        f"**Embeddings:** `mxbai-embed-large:latest`"
    )

    st.write(
        f"**BigQuery:** `{PROJECT_ID}`"
    )

    st.write(
        f"**Location:** `{BQ_LOCATION}`"
    )


# ============================================================
# QUESTION INPUT
# ============================================================

question = st.text_input(
    "Ask a question about your Zomato data:",
    value=st.session_state.get(
        "selected_question",
        ""
    ),
    placeholder=(
        "Example: What are the most common "
        "negative review topics?"
    )
)


# ============================================================
# PROCESS QUESTION
# ============================================================

if question:

    st.info(
        f"Question type: **{classify_question(question)}**"
    )

    # --------------------------------------------------------
    # GENERATE SQL
    # --------------------------------------------------------

    with st.spinner(
        "gpt-oss:120b-cloud is generating SQL..."
    ):

        try:

            sql, reason = generate_sql(
                question
            )

        except Exception as e:

            st.error(
                "❌ Ollama request failed."
            )

            st.code(
                str(e)
            )

            st.info(
                "Check that Ollama is running and "
                "gpt-oss:120b-cloud is available."
            )

            st.stop()


    # --------------------------------------------------------
    # QUESTION NOT SUPPORTED
    # --------------------------------------------------------

    if sql is None:

        st.warning(
            "The question cannot be answered using "
            "the available Zomato data."
        )

        st.info(
            reason
        )

        st.stop()


    # --------------------------------------------------------
    # SHOW SQL
    # --------------------------------------------------------

    st.subheader("Generated SQL")

    st.code(
        sql,
        language="sql"
    )


    # --------------------------------------------------------
    # SAFETY CHECK
    # --------------------------------------------------------

    safe, safety_error = is_safe_sql(
        sql
    )

    if not safe:

        st.error(
            "❌ SQL blocked for safety."
        )

        st.error(
            safety_error
        )

        st.stop()

    st.success(
        "✅ SQL passed the read-only safety check."
    )


    # --------------------------------------------------------
    # EXECUTE QUERY
    # --------------------------------------------------------

    with st.spinner(
        "Running query in BigQuery..."
    ):

        try:

            df = run_query(
                sql
            )

        except Exception as e:

            st.error(
                "❌ BigQuery could not execute "
                "the generated SQL."
            )

            st.code(
                str(e)
            )

            st.info(
                "The model generated SQL, but BigQuery "
                "rejected it. Check the generated table "
                "or column names."
            )

            st.stop()


    # --------------------------------------------------------
    # DISPLAY RESULTS
    # --------------------------------------------------------

    st.subheader("Results")

    if df.empty:

        st.warning(
            "Query executed successfully, "
            "but returned no rows."
        )

    else:

        st.success(
            f"✅ {len(df):,} rows returned."
        )

        st.dataframe(
            df,
            hide_index=True,
            use_container_width=True
        )


    # --------------------------------------------------------
    # SIMPLE VISUALIZATION
    # --------------------------------------------------------

    if (
        len(df) > 1
        and len(df.columns) == 2
        and pd.api.types.is_numeric_dtype(
            df.iloc[:, 1]
        )
    ):

        st.subheader("Visualization")

        try:

            st.bar_chart(
                df,
                x=df.columns[0],
                y=df.columns[1]
            )

        except Exception:
            pass


# ============================================================
# ABOUT
# ============================================================

with st.expander(
    "ℹ️ About this Text-to-SQL pipeline"
):

    st.write(
        """
        This application converts natural-language questions
        into read-only BigQuery SQL.

        Pipeline:

        1. User asks a question.
        2. gpt-oss:120b-cloud generates BigQuery SQL.
        3. The SQL passes a safety check.
        4. BigQuery executes the query.
        5. Results are displayed.
        6. Simple two-column numeric results can be visualized.

        Project models:

        LLM:
        gpt-oss:120b-cloud

        Embedding model:
        mxbai-embed-large:latest

        The embedding model belongs to the RAG pipeline.
        It is not required for Text-to-SQL.
        """
    )