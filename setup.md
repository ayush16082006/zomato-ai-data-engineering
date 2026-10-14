# Setup Guide

## 1. Prerequisites

- Python and Git
- Docker Desktop (for local Airflow)
- Google Cloud project with BigQuery access
- A configured Hugging Face / Qdrant Cloud setup
- Ollama, if using the local Ollama-based components

## 2. Python Environment

Open a terminal in the project root:

```bash
python -m venv zomato_env

Activate it on Windows:

zomato_env\Scripts\activate

Install dependencies:

pip install -r requirements.txt


## 3. Environment Configuration

Configure the required environment variables for the components you intend to run, including:

- BigQuery credentials and project configuration
- Qdrant Cloud URL and API key
- LLM provider URL, API key, and model configuration

Use local environment files or system environment variables. Never commit credentials, API keys, service-account JSON files, or passwords to GitHub.

## 4. BigQuery and dbt

The project uses BigQuery project `zomato-ai-data-engineering` in the `asia-south1` region.

Configure dbt using the project's profile template and your own Google Cloud credentials.

From the project root, validate the dbt configuration:

```bash
dbt debug --project-dir zomato --profiles-dir zomato

Run dbt models only when the required BigQuery storage capacity is available. The current free-storage quota limitation affects the `fct_orders` model.

## 5. Run the Streamlit Application

From the project root, activate the Python environment and run:

```bash
streamlit run ai/app.py
```

Open the local URL printed in the terminal.

The deployed application is available at:

https://zomato-ai-data-engineering.onrender.com

## 6. Local Airflow

Start Docker Desktop and wait for the Docker engine to become ready.

From the project root:

```bat
cd airflow
docker compose up -d
docker compose ps

Open the Airflow UI at:

http://localhost:8081

To stop the local services:

```bat
docker compose down

Do not add `-v` to the shutdown command unless you intentionally want to remove Docker volumes.

The `zomato_batch` DAG is currently paused. Do not trigger it until the BigQuery storage limitation and the enrichment task's execution behavior have been reviewed.

## 7. Current Limitations

- The BigQuery free-storage quota has been exceeded, blocking the current `fct_orders` table build.
- Further review enrichment is deferred because of external API quota limitations.
- Do not run full dbt builds or review-enrichment jobs merely to test the setup.

For system design, data flow, and component relationships, see `architecture.md`.