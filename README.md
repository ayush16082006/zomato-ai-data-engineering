# Zomato AI — Data Engineering & AI Analytics Assistant

An end-to-end prototype combining **Google BigQuery, dbt, Apache Airflow, Ollama, Text-to-SQL, semantic routing, vector search, RAG, and Streamlit** to answer natural-language questions about a Zomato dataset.

## 1. Project Goal

The project combines structured analytics and unstructured customer-review analysis.

Examples of supported questions:

- Which restaurant generated the most revenue?
- Which city has the highest sales?
- What do customers say about food quality?
- What do customers say about delivery?
- Which restaurant generated the most revenue and what do customers say about it?

Structured questions are handled through SQL/BigQuery. Review questions use semantic retrieval and RAG. Mixed questions can use a hybrid SQL + RAG workflow.

## 2. Architecture

                         USER QUESTION
                              |
                              v
                       Streamlit UI
                              |
                              v
                        Domain Guard
                              |
                              v
                         Query Router
                              |
             +----------------+----------------+
             |                |                |
             v                v                v
            SQL              RAG             HYBRID
             |                |                |
             v                v                v
        Text-to-SQL       Embedding        SQL Planner
             |           Vector Search          |
             v                |                 v
        Safety Check          v             SQL Engine
             |          Diverse Reviews          |
             v                |                 v
         BigQuery             |            SQL Result
                              |                 |
                              |          RAG Query Planner
                              |                 |
                              +---------> RAG Engine
                                                |
                                                v
                                         Final LLM Answer
```

### Data engineering flow

CSV Data
   |
   v
BigQuery Landing
   |
   v
BigQuery Raw
   |
   v
dbt Staging / Silver
   |
   v
dbt Marts / Gold
   |
   +--------------------+
   |                    |
   v                    v
Structured Analytics   AI Review Data
                           |
                           v
                    Review Enrichment
                           |
                           v
                    Review Embeddings
                           |
                           v
                        RAG Index

## 3. Technology Stack

| Component | Technology |
|---|---|
| Data warehouse | Google BigQuery |
| BigQuery location | `asia-south1` |
| Transformation | dbt |
| Orchestration | Apache Airflow |
| Containers | Docker / Docker Compose |
| LLM runtime | Ollama |
| Chat model | `gpt-oss:120b-cloud` |
| Embedding model | `mxbai-embed-large:latest` |
| Embedding size | 1024 dimensions |
| AI application | Python |
| UI | Streamlit |
| Data processing | Pandas / NumPy |
| Version control | Git / GitHub |

## 4. BigQuery Architecture

Project:

zomato-ai-data-engineering

Conceptual datasets:

zomato       -> landing/test upload
raw          -> source data
staging      -> dbt Silver layer
marts        -> dbt Gold layer
snapshots    -> reserved snapshot layer
ai           -> AI-enriched data
```

Raw tables currently include:

food
menu
order_items
orders
restaurant
restaurants
reviews
users

Important: the dbt restaurant source uses **`raw.restaurant` (singular)**.

## 5. dbt

The dbt project is located in:

zomato/

### Staging models
models/staging/
├── sources.yml
├── _ai_sources.yml
├── stg_food.sql
├── stg_menu.sql
├── stg_orders.sql
├── stg_order_items.sql
├── stg_restaurant.sql
├── stg_reviews.sql
└── stg_users.sql

### Mart models

The current marts include models for:

- customers
- dates
- food
- restaurants
- order items
- orders
- cuisine performance
- customer performance
- daily city revenue
- delivery SLA
- food performance
- payment performance
- restaurant performance
- review insights

The individual SQL files remain the source of truth for exact transformations and column definitions.

## 6. AI Review Enrichment

Implemented in:
ai/enrich_reviews.py

The AI enrichment produces information such as:

- sentiment label
- sentiment score
- topic
- key issue
- processing metadata
- model information

Target table:
zomato-ai-data-engineering.ai.review_enriched


**Important limitation:** LLM enrichment is not complete for the entire review population because of limited external model/OmniRoute quota. The project must not claim that all ~300,000 reviews have been LLM-enriched.

## 7. Review Embeddings

Implemented in:

ai/embed_all_reviews.py

Model:

mxbai-embed-large:latest

Persistent RAG index:
ai/rag_index/
├── embeddings.npy
├── reviews.parquet
└── metadata.json

Verified persistent index:

~300,000 reviews
1024-dimensional embeddings
float32 vectors

The repository also contains batched embedding files under:

ai/review_embeddings/


## 8. RAG

Current RAG engine:


ai/rag_engine.py

Flow:

User question
      |
      v
Question embedding
      |
      v
Persistent embedding index
      |
      v
Candidate retrieval
      |
      v
MMR diversity selection
      |
      v
Retrieved reviews
      |
      v
Evidence-aware LLM
      |
      v
Answer
```

Current configuration:

Candidate K = 40
Final K     = 5
MMR lambda  = 0.70
Duplicate similarity threshold = 0.95

The RAG prompt instructs the LLM to use only retrieved evidence, avoid invention, avoid overgeneralizing from a small sample, mention relevant review IDs, and acknowledge insufficient evidence.

## 9. Text-to-SQL

Implemented in:

ai/text_to_sql.py
ai/sql_engine.py

The system converts natural-language analytical questions into BigQuery SQL.

It is intended for questions involving:

- revenue
- GMV
- orders
- rankings
- restaurants
- cities
- customers
- payments
- dates
- analytical aggregates

Generated SQL is restricted to read-only operations. Destructive/modifying operations such as `DROP`, `DELETE`, `TRUNCATE`, `ALTER`, `UPDATE`, `INSERT`, `CREATE`, `MERGE`, `GRANT`, `REVOKE`, and similar operations are rejected by the safety layer.

## 10. Routing and Orchestration

Important AI files:

ai/domain_guard.py
ai/router.py
ai/orchestrator.py


Main routes:

SQL
RAG
HYBRID
DOMAIN GUARD


The domain guard prevents unrelated questions from being treated as Zomato questions.

The orchestrator coordinates the existing router, SQL engine, RAG engine, hybrid planners, and final answer generation.

## 11. Hybrid SQL + RAG

Example:

> Which restaurant generated the most revenue and what do customers say about it?

The workflow is:

Original question
       |
       v
Hybrid routing
       |
       +--------------------+
       |                    |
       v                    v
SQL task planner      Structured intent
       |                    |
       v                    |
SQL engine                 |
       |                    |
       v                    |
BigQuery result             |
       |                    |
       +----------+---------+
                  |
                  v
           Focused RAG query
                  |
                  v
              RAG engine
                  |
                  v
          Retrieved reviews
                  |
                  v
            Final answer


## 12. Streamlit

The AI directory contains the user-facing Streamlit components:

ai/app.py
ai/rag_chat.py
ai/streamlit_worker.py

The interface provides natural-language interaction with the analytics system and supports the AI query workflows.

## 13. Airflow

Airflow is implemented under:

airflow/
├── Dockerfile
├── docker-compose.yaml
└── dags/
    └── zomato_batch.py

Main DAG:

zomato_batch


Verified task dependency:

dbt_build_core
       |
       v
enrich_reviews
       |
       v
dbt_build_ai

Airflow runtime, DAG discovery, task registration, dependencies, Airflow → dbt execution, dbt → BigQuery connectivity, and failure propagation were verified for the prototype.

The DAG was intentionally paused after verification.

## 14. Known Limitation

The `fct_orders` dbt model currently fails because the BigQuery project exceeds the available free storage quota.

The verified behavior is:


dbt_build_core
      |
      v
fct_orders -> FAILED
      |
      v
downstream tasks -> upstream_failed


This is a **BigQuery storage/quota limitation**, not an Airflow DAG-wiring problem.

The project does not delete existing tables merely to hide this limitation.

## 15. Security

Sensitive files must not be committed:

airflow/credentials/service-account.json
airflow/.env
zomato/profiles.yml


The repository contains:


zomato/profiles.example.yml


for showing the expected configuration structure without committing real credentials.

Never commit API keys, passwords, service-account JSON files, or other secrets.

## 16. Repository Structure
zomato_project/
├── ai/
│   ├── app.py
│   ├── benchmark_models.py
│   ├── build_rag_index.py
│   ├── domain_guard.py
│   ├── embed_all_reviews.py
│   ├── enrich_reviews.py
│   ├── orchestrator.py
│   ├── rag_chat.py
│   ├── rag_engine.py
│   ├── router.py
│   ├── sql_engine.py
│   ├── streamlit_worker.py
│   ├── text_to_sql.py
│   ├── rag_index/
│   └── review_embeddings/
│
├── airflow/
│   ├── Dockerfile
│   ├── docker-compose.yaml
│   └── dags/
│       └── zomato_batch.py
│
├── bigquery/
│   ├── 01_setup.sql
│   ├── 02_raw_tables.sql
│   └── create_raw_tables.sql
│
├── scripts/
│   ├── upload_food.py
│   ├── upload_menu.py
│   ├── upload_orders.py
│   └── upload_order_items.py
│
├── zomato/
│   ├── dbt_project.yml
│   ├── profiles.example.yml
│   ├── models/
│   │   ├── staging/
│   │   └── marts/
│   └── macros/
│
├── requirements.txt
└── .gitignore


The repository also contains generated logs, dbt `target` files, Python cache files, tests, benchmark outputs, and embedding artifacts. These should be treated as generated/development artifacts rather than core source code.

## 17. Current Status

### Completed / verified

- BigQuery data architecture
- dbt staging and mart project structure
- Text-to-SQL
- SQL safety validation
- domain guarding
- semantic routing
- RAG retrieval
- persistent review embedding index
- hybrid SQL + RAG workflow
- Streamlit AI interface
- Airflow prototype wiring
- Airflow → dbt → BigQuery integration

### Partial / limited

- LLM review enrichment is only partially complete because of external model quota.
- `fct_orders` is blocked by BigQuery free-storage quota.
- Airflow is paused after verification because of the known storage limitation.

## 18. Future Scope

- Complete remaining review enrichment.
- Add stronger retrieval and answer-quality evaluation.
- Add confidence/faithfulness evaluation.
- Improve hybrid planning.
- Add retrieval caching.
- Optimize large fact tables with incremental dbt models.
- Add more data-quality tests.
- Improve secret management and CI/CD.
- Add monitoring and alerting.
- Move the prototype to an appropriate production-scale BigQuery configuration.

## 19. Demo Flow

A clear project demonstration can follow this sequence:

1. Show CSV → BigQuery → dbt architecture.
2. Explain Raw → Staging → Marts.
3. Demonstrate a Text-to-SQL question such as:
   `Which city generated the highest revenue?`
4. Demonstrate a RAG question such as:
   `What do customers say about food quality?`
5. Demonstrate a hybrid question:
   `Which restaurant generated the most revenue and what do customers say about it?`
6. Show the Airflow DAG and explain the current `fct_orders` BigQuery quota limitation honestly.

## 20. Project Principles

### SQL for structured facts

Revenue, orders, rankings, counts, and aggregates should come from BigQuery rather than being invented by an LLM.

### RAG for review evidence

Customer opinions should be retrieved from the review knowledge base and supplied to the LLM as evidence.

### Explicit uncertainty

When evidence is insufficient, the system should say so rather than fabricate an answer.

---

## Summary

This project combines modern data engineering with generative AI:


BigQuery
   +
dbt
   +
Airflow
   +
Domain Guard
   +
Semantic Routing
   +
Text-to-SQL
   +
Vector Search
   +
RAG
   +
Hybrid Reasoning
   +
Streamlit


The prototype has verified AI, RAG, Text-to-SQL, Streamlit, and Airflow workflows. Its two major documented limitations are **partial LLM review enrichment due to model quota constraints** and the **BigQuery free-storage limitation affecting `fct_orders`**.
