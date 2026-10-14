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


Review enrichment is partially complete. Approximately 13,000 reviews have been enriched in the current project scope; the entire review population has not been enriched because of external model/API quota limitations.

Further enrichment is deferred. Do not assume that all approximately 300,000 reviews have sentiment labels, sentiment scores, topics, or key issues.

## 7. Review Embeddings

### Purpose

Review embeddings enable semantic search over customer reviews. Instead of matching only exact keywords, the system represents review text as numerical vectors so that semantically related reviews can be retrieved.

### Embedding Model

The project uses the Hugging Face Sentence Transformers model:

`sentence-transformers/all-MiniLM-L6-v2`

The model generates **384-dimensional embeddings** for semantic similarity search.

### Vector Database

The deployed AI application uses **Qdrant Cloud** as its vector database.

Qdrant stores review vectors and associated payload data, enabling semantic retrieval of relevant customer reviews. The application uses Qdrant Cloud's inference capabilities for embedding and vector-search operations.

**Current vector-search configuration:**

- **Embedding model:** `sentence-transformers/all-MiniLM-L6-v2`
- **Embedding dimension:** 384
- **Vector database:** Qdrant Cloud
- **Collection:** `zomato_reviews`
- **Search method:** Semantic vector search
- **Primary use case:** Retrieving relevant customer reviews for AI-powered review analysis

### Review Embedding Pipeline

The embedding workflow is designed to make customer reviews searchable by meaning rather than relying exclusively on keyword matching.

1. Obtain the customer review text.
2. Generate or request its embedding using the configured Hugging Face model.
3. Store the vector and corresponding review payload in Qdrant Cloud.
4. Embed or process the user's search query using the compatible embedding configuration.
5. Search Qdrant for semantically similar reviews.
6. Pass retrieved evidence to the RAG answer-generation component.

The embedding model used for query processing must remain compatible with the model used to generate the stored review vectors.

### Batch Processing and Progress Tracking

The repository contains scripts for review embedding and Qdrant ingestion, including:

- `ai/embed_reviews_qdrant.py`

The repository also contains ingestion progress metadata, including `ai/qdrant_ingestion_progress.json`.

These files support the embedding and ingestion workflow. The actual progress and number of indexed reviews should be determined from the existing progress metadata or Qdrant collection rather than inferred from the number of files.

### Current Status and Limitations

- Qdrant Cloud is the vector database used by the deployed application.
- The local index files under `ai/rag_index/` are not the deployed application's active vector store.
- The repository retains local RAG index artifacts from earlier development work.
- Embedding coverage and retrieval quality should be validated against the current Qdrant collection.
- Review enrichment and review embedding are separate processes: a review can be embedded and searchable even if it does not have all enrichment fields.

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

The Airflow runtime, DAG discovery, task registration, task dependencies, Airflow → dbt execution, dbt → BigQuery connectivity, and failure propagation were verified for the prototype.

The DAG was manually paused after verification to prevent scheduled runs from repeatedly encountering the known BigQuery storage limitation. The orchestration wiring is verified, but a successful full end-to-end run of all three tasks has not been established.

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

This section summarizes the implementation status of the Zomato AI Data Engineering project based on the current repository, previous tests, and deployment observations.

### Completed / Implemented

- **Data engineering foundation:** BigQuery is used as the analytical warehouse, with Raw, Staging, and Marts layers.
- **dbt transformation project:** Staging models and analytical marts have been implemented.
- **AI review enrichment:** Review enrichment has been implemented, with approximately 13,000 reviews enriched in the current project scope.
- **Vector-search integration:** Qdrant Cloud is used by the deployed AI application for review retrieval, using the Hugging Face model `sentence-transformers/all-MiniLM-L6-v2` with 384-dimensional embeddings.
- **AI application:** The Streamlit application and AI orchestration components have been deployed.
- **Text-to-SQL:** Natural-language analytical questions can be converted into SQL and executed against the configured BigQuery data.
- **RAG integration:** Retrieved customer reviews can be used as evidence for review-analysis responses.
- **Hybrid analytics:** The application includes routing and orchestration components for analytical SQL and review-based questions.
- **Airflow orchestration:** The `zomato_batch` DAG and its task dependencies have been implemented and tested.
- **Security configuration:** Repository credentials and secrets have been checked to ensure they are not committed to GitHub.

### Partial / Limited

- **Review enrichment:** Approximately 13,000 reviews have been enriched. Additional enrichment is deferred because of external model/API quota limitations.
- **BigQuery storage:** The project's free storage quota has been exceeded. The `fct_orders` model failed during an Airflow-triggered dbt build because of this limitation.
- **Airflow end-to-end execution:** Task dependencies and failure propagation have been verified. A successful full end-to-end execution of all three tasks has not been established.
- **Airflow scheduling:** The `zomato_batch` DAG is manually paused to prevent additional scheduled executions while the storage limitation remains unresolved.
- **Vector retrieval:** The deployed application uses Qdrant Cloud, but the number of successfully indexed reviews and retrieval-quality metrics should be reported only after checking the current collection and available diagnostics.
- **AI answer quality:** Further evaluation of retrieval relevance, answer grounding, and confidence is a potential improvement.

### Known Limitations

- BigQuery storage constraints prevent a complete successful run of the affected dbt model until the storage issue is resolved.
- Further review enrichment is deferred; the entire review population should not be assumed to have enrichment labels.
- A successful individual AI query or a passing structural test does not prove that every component has passed a full end-to-end test.
- The repository contains local RAG index artifacts from earlier development. These should not be confused with the Qdrant Cloud vector store used by the deployed application.

### Deferred Work

- Reassess BigQuery storage capacity before attempting another full dbt or Airflow run.
- Resume review enrichment only when additional API quota or an approved alternative processing method is available.
- Validate Qdrant collection coverage and retrieval quality.
- Improve data-quality testing, monitoring, and operational diagnostics.
- Evaluate answer grounding, confidence, and retrieval performance.
- Improve CI/CD and incremental transformation strategies where appropriate.

### Overall Assessment

The project has implemented its core data engineering, AI application, semantic retrieval, and orchestration components. The deployed application supports analytical and review-based queries. However, the BigQuery storage limitation prevents us from claiming a successful full Airflow pipeline execution, and additional enrichment remains deferred.

## 18. Future Scope

- Resume review enrichment only if additional API quota or an alternative approved processing method becomes available.
- Add stronger retrieval and answer-quality evaluation.
- Add confidence and faithfulness evaluation.
- Improve hybrid SQL + RAG planning.
- Add retrieval caching.
- Optimize large fact tables with incremental dbt models where appropriate.
- Add more data-quality tests.
- Improve secret management and CI/CD.
- Add monitoring and alerting.
- Revisit BigQuery storage capacity before attempting another full dbt/Airflow run.

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



## Documentation

- [Setup Guide](setup.md) — Prerequisites, environment configuration, and instructions for running the project.
- [Architecture Guide](architecture.md) — System architecture, data flow, AI components, and Airflow orchestration.
