from datetime import datetime

from airflow import DAG
from airflow.providers.standard.operators.bash import BashOperator


# ============================================================
# CONFIGURATION
# ============================================================

DBT = "/opt/airflow/dbt_venv/bin/dbt"

DBT_PROJECT = "/opt/airflow/dbt/zomato"

AI_ENRICH_SCRIPT = "/opt/airflow/ai/enrich_reviews.py"


# ============================================================
# DAG
# ============================================================

with DAG(
    dag_id="zomato_batch",

    start_date=datetime(2024, 1, 1),

    schedule="@daily",

    catchup=False,

    tags=[
        "zomato",
        "bigquery",
        "dbt",
        "ai",
    ],

    doc_md="""
    # Zomato Batch Pipeline

    Pipeline:

    1. Build core dbt models
    2. Enrich customer reviews using OmniRoute
    3. Build AI dbt models

    Architecture:

    BigQuery Raw
        ↓
    dbt Staging
        ↓
    dbt Marts
        ↓
    AI Review Enrichment
        ↓
    dbt AI Models
    """,

) as dag:

    # ========================================================
    # TASK 1
    # DBT CORE BUILD
    # ========================================================

    dbt_build_core = BashOperator(

        task_id="dbt_build_core",

        bash_command=(
            f"{DBT} build "
            f"--project-dir {DBT_PROJECT} "
            f"--profiles-dir {DBT_PROJECT} "
            f"--exclude tag:ai"
        ),

    )


    # ========================================================
    # TASK 2
    # REVIEW ENRICHMENT
    # ========================================================

    enrich_reviews = BashOperator(

        task_id="enrich_reviews",

        bash_command=(
            f"python {AI_ENRICH_SCRIPT} "
            f"--worker 1 "
            f"--workers 1"
        ),

    )


    # ========================================================
    # TASK 3
    # DBT AI BUILD
    # ========================================================

    dbt_build_ai = BashOperator(

        task_id="dbt_build_ai",

        bash_command=(
            f"{DBT} build "
            f"--project-dir {DBT_PROJECT} "
            f"--profiles-dir {DBT_PROJECT} "
            f"--select tag:ai"
        ),

    )


    # ========================================================
    # TASK DEPENDENCIES
    # ========================================================

    dbt_build_core >> enrich_reviews >> dbt_build_ai