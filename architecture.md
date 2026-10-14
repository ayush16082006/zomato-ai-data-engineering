# Zomato AI — System Architecture

## 1. Overview

Zomato AI is a data engineering and AI analytics assistant that combines structured business analytics with semantic customer-review analysis.

The system uses:

- **Google BigQuery** for analytical storage and SQL execution.
- **dbt** for data transformation and analytical modeling.
- **Apache Airflow** for scheduled pipeline orchestration.
- **Python and Streamlit** for the AI application.
- **Text-to-SQL** for structured business questions.
- **Hugging Face embeddings and Qdrant Cloud** for semantic review retrieval.
- **Retrieval-Augmented Generation (RAG)** for evidence-based review answers.
- **LLM-based routing and orchestration** to coordinate the available query paths.

The system is designed to answer business questions from structured tables, customer-opinion questions from retrieved reviews, and combined questions using both sources.

## 2. High-Level Architecture

flowchart TD
    U[User Question] --> UI[Streamlit Application]
    UI --> DG[Domain Guard]
    DG --> R[Semantic Query Router]

    R -->|Structured question| SQL[Text-to-SQL]
    R -->|Review question| RAG[RAG Retrieval]
    R -->|Combined question| H[Hybrid Planner]

    SQL --> SAFE[SQL Safety Validation]
    SAFE --> BQ[BigQuery Marts]

    H --> SP[SQL Planning and Execution]
    SP --> SAFE
    SP --> HR[Focused Review Query]
    HR --> RAG

    RAG --> Q[Qdrant Cloud]
    Q --> REV[Retrieved Review Evidence]

    REV --> LLM[LLM Answer Generation]
    BQ --> HA[Hybrid Answer Assembly]
    REV --> HA
    HA --> LLM

    LLM --> UI
    UI --> U


The diagram represents the logical query architecture. Exact execution paths depend on the question type and the application's current routing implementation.

## 3. Data Engineering Architecture

The data platform follows a layered warehouse design.


flowchart TD
    CSV[Source CSV Files] --> LOAD[Python Upload Scripts]
    LOAD --> LAND[BigQuery Landing Dataset]
    LAND --> RAW[BigQuery Raw Dataset]
    RAW --> DBT[dbt Staging Models]
    DBT --> STG[Staging / Silver]
    STG --> MART[dbt Marts / Gold]
    MART --> BI[Structured Analytics and Text-to-SQL]
    RAW --> AI[AI Review Data]
    AI --> ENR[Review Enrichment]
    AI --> EMB[Review Embedding Workflow]
    ENR --> AIT[Enriched Review Records]
    EMB --> VDB[Qdrant Cloud Vector Collection]
    AIT --> VDB

The diagram shows the intended logical flow. Review enrichment, vector ingestion, and warehouse transformations are separate processes and should not be assumed to have processed every source record.

### 3.1 Source ingestion

Source datasets are supplied as CSV files. Python scripts in `scripts/` handle uploads for selected datasets, including food, menu, orders, and order items.

Relevant files:

scripts/
├── upload_food.py
├── upload_menu.py
├── upload_orders.py
└── upload_order_items.py

BigQuery is the warehouse and landing environment. AWS S3 is not part of this architecture.

### 3.2 BigQuery datasets

The project uses the BigQuery project:

`zomato-ai-data-engineering`

The configured warehouse location is:

`asia-south1` — Mumbai.

| Dataset | Responsibility |
|---|---|
| `zomato` | Landing and test uploads |
| `raw` | Source tables loaded into BigQuery |
| `staging` | Cleaned and standardized dbt models |
| `marts` | Analytical models for business questions |
| `ai` | AI-related review data |
| `snapshots` | Reserved for snapshot-related modeling |

These dataset roles describe the project design; the exact deployed objects should be checked in BigQuery.

Raw source tables include `food`, `menu`, `order_items`, `orders`, `restaurant`, `restaurants`, `reviews`, and `users`, depending on the available source data. The dbt restaurant source uses **`raw.restaurant`**, singular.

## 4. dbt Transformation Layer

The dbt project is located in:


zomato/

dbt converts raw source data into cleaned staging models and analytical marts.

### 4.1 Staging / Silver

The staging layer standardizes source fields and prepares data for downstream transformations.

Key models include:

stg_food
stg_menu
stg_orders
stg_order_items
stg_restaurant
stg_reviews
stg_users


Source definitions are maintained in YAML files under `models/staging/`.

The staging layer is responsible for source normalization and providing consistent inputs to downstream models.

### 4.2 Marts / Gold

The marts layer provides business-oriented models for analytics.

Current model areas include:

- Customer, date, food, and restaurant dimensions.
- Order and order-item facts.
- Cuisine performance.
- Customer performance.
- Daily city revenue.
- Delivery service-level metrics.
- Food and restaurant performance.
- Payment performance.
- Review insights.

The exact metrics, joins, and column definitions are determined by the SQL in `zomato/models/marts/`.

### 4.3 Structured query serving

The Text-to-SQL component uses the analytical marts to answer questions involving revenue, GMV, orders, rankings, customers, restaurants, cities, payment methods, and dates.

Structured numeric answers should come from executed SQL results, not be estimated by the LLM.

## 5. Review Enrichment

Review enrichment is implemented in:


ai/enrich_reviews.py

The enrichment workflow uses an LLM service to add review-level metadata such as:

- Sentiment label.
- Sentiment score.
- Topic.
- Key issue.
- Processing and model metadata.

The configured destination is:

ai.review_enriched

Review enrichment and embedding are separate operations. Enrichment adds semantic labels and attributes; embedding represents review text as vectors for similarity search.

**Current status:** Approximately 13,000 reviews have been enriched in the current project scope. Enrichment of the full review population is incomplete and has been deferred because of external model/API quota limitations. The system must not assume every review has enrichment fields.

## 6. Embedding and Vector Search Architecture

### 6.1 Deployed embedding path

The deployed semantic-search integration uses:

| Component | Configuration |
|---|---|
| Embedding model | `sentence-transformers/all-MiniLM-L6-v2` |
| Vector dimension | 384 |
| Vector database | Qdrant Cloud |
| Collection | `zomato_reviews` |
| Retrieval purpose | Semantic search over customer reviews |

The embedding model represents review text and queries in a vector space. Qdrant retrieves vectors that are similar to the user's query.

The query embedding configuration must be compatible with the model used for the stored vectors.

### 6.2 Logical retrieval flow

flowchart TD
    Q[Review Question] --> E[Compatible Query Embedding]
    E --> SEARCH[Qdrant Similarity Search]
    SEARCH --> PAYLOAD[Review Payload and Metadata]
    PAYLOAD --> FILTER[Retrieval and Evidence Selection]
    FILTER --> CONTEXT[Selected Review Evidence]
    CONTEXT --> LLM[LLM Answer Generation]
    LLM --> ANSWER[Grounded Review Answer]

The collection's actual record count, payload schema, and retrieval filters should be obtained from the current Qdrant configuration and diagnostics rather than inferred from local files.

### 6.3 Historical local embedding artifacts

The repository contains artifacts from an earlier local embedding workflow:

ai/
├── rag_index/
│   ├── embeddings.npy
│   ├── reviews.parquet
│   └── metadata.json
└── review_embeddings/

The earlier workflow used `mxbai-embed-large:latest` through Ollama and produced 1024-dimensional vectors.

These files document the earlier development path. They are **not the active vector store used by the deployed Qdrant Cloud path** and should not be treated as synchronized with the Qdrant collection.

## 7. RAG Answer Generation

The RAG system combines retrieved review evidence with an LLM to produce a natural-language answer.

Logical workflow:

1. Receive a customer-review question.
2. Generate a query embedding using the compatible embedding model.
3. Search Qdrant Cloud for semantically relevant reviews.
4. Select useful evidence and avoid unnecessary duplication.
5. Pass the selected reviews and the original question to the answer-generation model.
6. Return an answer grounded in the retrieved evidence.

The RAG answer-generation prompt is intended to prevent invented claims, discourage generalizations from a small review sample, and acknowledge when the retrieved evidence is insufficient.

The deployed retrieval implementation is the source of truth for the exact candidate count, filtering strategy, ranking algorithm, and number of reviews sent to the LLM.

## 8. Text-to-SQL Architecture

Important files:

ai/text_to_sql.py
ai/sql_engine.py

The Text-to-SQL path converts a structured business question into SQL for BigQuery.

flowchart TD
    Q[Business Question] --> GEN[SQL Generation]
    GEN --> VALID[SQL Safety Validation]
    VALID -->|Approved| EXEC[BigQuery Execution]
    VALID -->|Rejected| ERR[Safe Rejection]
    EXEC --> RES[Structured Result]
    RES --> FORMAT[Natural-Language Response]

The SQL safety layer is designed to reject data-modifying or destructive operations, including `DROP`, `DELETE`, `UPDATE`, `INSERT`, `ALTER`, `TRUNCATE`, and similar statements.

Only validated, read-only queries should reach BigQuery through the application. The safety implementation and its tests remain the source of truth for the exact restrictions.

## 9. Domain Guard, Routing, and Orchestration

Important files:

ai/domain_guard.py
ai/router.py
ai/orchestrator.py

### Domain guard

The domain guard checks whether a question belongs to the project's supported domain before it is routed to the analytics workflow.

### Semantic router

The router determines whether the question is primarily:

- **SQL:** Structured business metrics.
- **RAG:** Customer-review evidence.
- **HYBRID:** A combination of structured analytics and review evidence.

### Application orchestrator

The orchestrator coordinates the selected route, invokes the relevant components, and prepares the final response.

This separation keeps structured SQL analytics distinct from semantic review retrieval while allowing both to contribute to combined questions.

## 10. Hybrid SQL + RAG

Hybrid reasoning is intended for questions that require both a structured result and customer-review context.

Example:

> Which restaurant generated the most revenue, and what do customers say about it?

Logical workflow:

1. Identify the structured revenue question and the review-analysis question.
2. Generate and validate SQL.
3. Execute SQL against BigQuery.
4. Use the structured result to focus the review query.
5. Retrieve relevant customer reviews from Qdrant Cloud.
6. Combine the numeric result and retrieved evidence into a final response.

The SQL result supplies the business metric; retrieved reviews supply the customer-opinion evidence. The LLM should not invent either.

## 11. Streamlit Application and Deployment

The user-facing application is implemented primarily in:

ai/app.py
ai/rag_chat.py
ai/streamlit_worker.py

Streamlit provides the user interface for submitting natural-language questions and displaying answers.

The deployed application is available at:

https://zomato-ai-data-engineering.onrender.com

The deployed environment is separate from the local Windows development environment. Required credentials and provider configuration must be supplied through the appropriate environment configuration rather than hard-coded into source files.

## 12. Airflow Orchestration Architecture

Airflow is implemented under:

airflow/
├── Dockerfile
├── docker-compose.yaml
└── dags/
    └── zomato_batch.py

Docker Compose runs the Airflow services, including the API server, scheduler, DAG processor, and PostgreSQL metadata database.

The DAG is named:

`zomato_batch`

Its configured schedule is `@daily`, with `catchup=False`.

### 12.1 Task dependency

flowchart TD
    S[Airflow Scheduler] --> D[zomato_batch DAG]
    D --> CORE[dbt_build_core]
    CORE --> BQ[BigQuery Core Models]
    BQ --> ENR[enrich_reviews]
    ENR --> AI[dbt_build_ai]

The configured task sequence is:

1. `dbt_build_core` — executes the core dbt build while excluding models tagged `ai`.
2. `enrich_reviews` — calls the review enrichment script.
3. `dbt_build_ai` — builds dbt models tagged `ai`.

The dependency chain ensures downstream tasks wait for upstream success under the default task trigger behavior.

### 12.2 Verified behavior and limitation

The Airflow prototype has been verified for:

- Docker and Docker Compose startup.
- Airflow container operation.
- DAG discovery and task registration.
- Task dependency configuration.
- Airflow-to-dbt execution.
- dbt-to-BigQuery connectivity.
- Failure propagation.

The core dbt task reached BigQuery, but `fct_orders` failed because the project's BigQuery free-storage quota was exceeded. Airflow correctly marked downstream tasks as `upstream_failed`.

The DAG was manually paused after verification to prevent scheduled executions from repeatedly encountering the known limitation.

**A successful full end-to-end execution of all three tasks has not been established.**

### 12.3 Enrichment safety

The DAG's enrichment task can consume external API quota. Further review enrichment is currently deferred. The task must remain protected from unintended execution before the DAG is resumed or triggered.

## 13. Security and Configuration

The project relies on external credentials for cloud services and model providers.

Sensitive configuration includes:

- BigQuery service-account credentials.
- dbt profile credentials.
- Qdrant Cloud API keys.
- LLM provider API keys.
- Airflow environment configuration and passwords.

Credentials must be provided through local environment files, environment variables, or the appropriate deployment secret configuration. They must not be committed to source control.

Read-only SQL restrictions, domain checks, and validation layers are important safeguards for the AI analytics application, but they do not replace appropriate cloud IAM permissions and secret management.

## 14. Reliability, Constraints, and Failure Behavior

### BigQuery storage

The free-storage quota is currently exceeded. This blocks the `fct_orders` table build and prevents us from claiming a successful full dbt/Airflow execution.

Existing tables should not be deleted merely to make the pipeline appear successful. Any storage or billing change should be evaluated separately.

### Review enrichment

Further enrichment is deferred because of external model/API quota constraints. The project should report only verified enrichment counts and should not imply that the entire review population has been processed.

### Vector retrieval

Qdrant Cloud is the deployed vector-search path. Its current record count and retrieval quality should be assessed using the collection and application diagnostics.

### Failure handling

Airflow's dependency behavior was verified: when the core build fails, downstream tasks do not proceed normally and are marked `upstream_failed`.

## 15. Repository Map

zomato_project/
├── ai/
│   ├── app.py
│   ├── domain_guard.py
│   ├── router.py
│   ├── orchestrator.py
│   ├── text_to_sql.py
│   ├── sql_engine.py
│   ├── rag_chat.py
│   ├── rag_engine.py
│   ├── enrich_reviews.py
│   ├── embed_all_reviews.py
│   ├── embed_reviews_qdrant.py
│   └── tests and development artifacts
│
├── airflow/
│   ├── Dockerfile
│   ├── docker-compose.yaml
│   └── dags/
│       └── zomato_batch.py
│
├── bigquery/
│   └── SQL setup and table definitions
│
├── scripts/
│   └── source-data upload scripts
│
├── zomato/
│   ├── dbt_project.yml
│   ├── profiles.example.yml
│   ├── models/staging/
│   ├── models/marts/
│   └── macros/
│
├── README.md
├── setup.md
├── architecture.md
├── requirements.txt
└── .gitignore

Generated files, local embedding artifacts, logs, caches, and dbt target outputs are development artifacts unless explicitly required for a particular workflow.

## 16. Architecture Summary

The project separates responsibilities across the data and AI layers:

- **BigQuery** stores structured source data and analytical models.
- **dbt** transforms raw data into staging models and business marts.
- **Airflow** coordinates the scheduled dbt and enrichment tasks.
- **Text-to-SQL** answers structured business questions using validated SQL.
- **Hugging Face embeddings and Qdrant Cloud** support semantic review retrieval.
- **RAG** uses retrieved reviews as evidence for customer-opinion answers.
- **Hybrid orchestration** combines structured metrics with relevant review evidence.
- **Streamlit** exposes the functionality through a natural-language interface.

The architecture is implemented as a working prototype with verified AI query workflows and Airflow task wiring. The main documented constraints are the BigQuery free-storage limitation affecting `fct_orders` and deferred review enrichment due to external model/API quota limits.