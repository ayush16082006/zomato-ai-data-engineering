# ============================================================
# Apache Airflow 3.x
# BigQuery + dbt + AI / OmniRoute
# ============================================================

FROM apache/airflow:3.0.3


# ============================================================
# AIRFLOW PROVIDERS + PYTHON DEPENDENCIES
# ============================================================

RUN pip install --no-cache-dir \
    apache-airflow-providers-google \
    apache-airflow-providers-fab \
    google-cloud-bigquery \
    requests


# ============================================================
# DBT VIRTUAL ENVIRONMENT
# ============================================================
#
# Keep dbt separate from Airflow's Python environment.
# This avoids dependency conflicts between Airflow and dbt.
#

RUN python -m venv /opt/airflow/dbt_venv && \
    /opt/airflow/dbt_venv/bin/pip install --no-cache-dir \
    "dbt-bigquery==1.12.*"


# ============================================================
# CREATE PROJECT DIRECTORIES
# ============================================================

RUN mkdir -p /opt/airflow/dbt \
             /opt/airflow/ai \
             /opt/airflow/dags


# ============================================================
# ENVIRONMENT
# ============================================================

ENV PYTHONUNBUFFERED=1

ENV PATH="/opt/airflow/dbt_venv/bin:${PATH}"