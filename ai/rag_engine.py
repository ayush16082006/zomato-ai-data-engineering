"""
ZOMATO AI - QDRANT + ADAPTIVE PROGRESSIVE RAG ENGINE
=====================================================

Production RAG engine for the Zomato AI project.

Architecture
------------

                    USER QUESTION
                         |
                         v
              QDRANT CLOUD INFERENCE
                         |
                         v
                   VECTOR SEARCH
                         |
                         v
              PROGRESSIVE RETRIEVAL
            100 -> 500 -> 1000 -> 2500 -> 5000
                         |
                         v
               ADAPTIVE RELEVANCE
                         |
                         v
              QUERY-AWARE EVIDENCE
                   FILTERING
                         |
                         v
                DUPLICATE REMOVAL
                         |
                         v
              DIVERSITY SELECTION
                         |
                         v
                  TOP REVIEWS
                         |
                         v
                    BIGQUERY
                         |
                         v
             REVIEW + METADATA
                         |
                         v
               EVIDENCE-AWARE LLM
                         |
                         v
                    RAG ANSWER


IMPORTANT
---------

This file intentionally preserves the public API used by:

    orchestrator.py
    Streamlit
    future API layer

Main function:

    answer_review_question(question)

returns:

    answer, top_reviews


Old architecture is NOT used:

    - rag_index/embeddings.npy
    - local mxbai embeddings
    - local NumPy vector search
    - local review parquet as retrieval source
    - MMR

Current architecture:

    - Qdrant Cloud
    - Qdrant Cloud Inference
    - sentence-transformers/all-MiniLM-L6-v2
    - 384 dimensions
    - BigQuery as review source of truth
    - Adaptive relevance
    - Query-aware evidence filtering
    - Progressive Retrieval
    - Duplicate removal
    - Diversity selection
    - Ollama Cloud gpt-oss:120b-cloud
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import ollama

from dotenv import load_dotenv
from google.cloud import bigquery
from qdrant_client import QdrantClient, models


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent


# ============================================================
# QDRANT
# ============================================================

QDRANT_COLLECTION = "zomato_reviews"

EMBEDDING_MODEL = (
    "sentence-transformers/all-MiniLM-L6-v2"
)

EMBEDDING_DIMENSION = 384


# ============================================================
# OLLAMA
# ============================================================

LLM_MODEL = "gpt-oss:120b-cloud"


# ============================================================
# BIGQUERY
# ============================================================

DEFAULT_PROJECT_ID = (
    "zomato-ai-data-engineering"
)

BQ_LOCATION = "asia-south1"

REVIEWS_DATASET = "raw"

REVIEWS_TABLE_NAME = "reviews"

ENRICHED_DATASET = "ai"

ENRICHED_TABLE_NAME = "review_enriched"


# ============================================================
# PROGRESSIVE RETRIEVAL
# ============================================================

CANDIDATE_LEVELS = [
    100,
    500,
    1000,
    2500,
    5000,
]

FINAL_K = 5


# ============================================================
# ADAPTIVE RELEVANCE
# ============================================================

"""
The old engine used:

    score >= 0.50

for every query.

Our diagnostic showed that this is too strict.

Examples from the actual dataset:

Low ratings:
    best score = 0.4331
    actual comment = relevant

Value for money:
    best score = 0.4165
    actual comment = relevant

Customer problems:
    best score = 0.4051
    actual comment = relevant

Negative experience:
    best score = 0.3598
    result was NOT genuinely negative

Therefore we use:

    MIN_RELEVANCE_SCORE = 0.36

and an adaptive threshold:

    max(
        MIN_RELEVANCE_SCORE,
        top_score - SCORE_WINDOW
    )

This allows legitimate lower-scoring evidence while
still rejecting extremely weak matches.
"""

MIN_RELEVANCE_SCORE = 0.36

SCORE_WINDOW = 0.08


# ============================================================
# TEXT DUPLICATE CONTROL
# ============================================================

TEXT_DUPLICATE_THRESHOLD = 0.75


# ============================================================
# SEMANTIC DIVERSITY
# ============================================================

SELECTED_SIMILARITY_MAX = 0.70


# ============================================================
# PATTERN DIVERSITY
# ============================================================

DOMINANT_PATTERN_RATIO = 0.60

MAX_SAME_PATTERN = 2


# ============================================================
# LLM
# ============================================================

MAX_REVIEW_TEXT_LENGTH = 1000


# ============================================================
# CLIENT CACHE
# ============================================================

_QDRANT_CLIENT: QdrantClient | None = None

_BQ_CLIENT: bigquery.Client | None = None


# ============================================================
# STANDARD RESULT COLUMNS
# ============================================================

REVIEW_COLUMNS = [
    "review_id",
    "rating",
    "comment",
    "review_date",
    "score",
    "sentiment_label",
    "sentiment_score",
    "topic",
    "key_issue",
]


# ============================================================
# UTILITY
# ============================================================

def _print_separator(
    char: str = "=",
    width: int = 70,
) -> None:

    print(char * width)


def _load_environment() -> None:
    """
    Load environment variables.

    Qdrant credentials are expected in:

        airflow/.env
    """

    airflow_env = (
        PROJECT_DIR
        / "airflow"
        / ".env"
    )

    if airflow_env.exists():

        load_dotenv(
            airflow_env,
            override=False,
        )

    load_dotenv(
        override=False
    )


def _get_project_id() -> str:
    """
    Get the active Google Cloud project.
    """

    return os.getenv(
        "GOOGLE_CLOUD_PROJECT",
        DEFAULT_PROJECT_ID,
    )


def _get_reviews_table() -> str:

    return (
        f"{_get_project_id()}."
        f"{REVIEWS_DATASET}."
        f"{REVIEWS_TABLE_NAME}"
    )


def _get_enriched_reviews_table() -> str:

    return (
        f"{_get_project_id()}."
        f"{ENRICHED_DATASET}."
        f"{ENRICHED_TABLE_NAME}"
    )


def _safe_text(
    value: Any,
) -> str:

    if value is None:
        return ""

    try:

        if pd.isna(value):
            return ""

    except (
        TypeError,
        ValueError,
    ):

        pass

    return str(value)


# ============================================================
# EMPTY REVIEW DATAFRAME
# ============================================================

def _empty_review_dataframe() -> pd.DataFrame:
    """
    Always return the same columns even when no evidence exists.

    This fixes the old test_rag.py warning where an empty
    DataFrame had no columns.
    """

    return pd.DataFrame(
        columns=REVIEW_COLUMNS
    )


# ============================================================
# TEXT NORMALIZATION
# ============================================================

def normalize_text(
    text: Any,
) -> str:
    """
    Normalize text for duplicate detection.
    """

    if text is None:
        return ""

    text = str(
        text
    ).lower().strip()

    text = re.sub(
        r"\s+",
        " ",
        text,
    )

    text = re.sub(
        r"[^\w\s]",
        "",
        text,
        flags=re.UNICODE,
    )

    return text.strip()


# ============================================================
# TOKEN SIMILARITY
# ============================================================

def token_similarity(
    text_a: str,
    text_b: str,
) -> float:
    """
    Jaccard token similarity.

    Used only for textual duplicate detection.
    """

    a = set(
        normalize_text(
            text_a
        ).split()
    )

    b = set(
        normalize_text(
            text_b
        ).split()
    )

    if not a and not b:
        return 1.0

    if not a or not b:
        return 0.0

    intersection = len(
        a.intersection(b)
    )

    union = len(
        a.union(b)
    )

    if union == 0:
        return 0.0

    return (
        intersection / union
    )


# ============================================================
# QUERY INTENT
# ============================================================

def _question_has_any(
    question: str,
    words: List[str],
) -> bool:

    question = normalize_text(
        question
    )

    return any(
        word in question
        for word in words
    )


def detect_query_intents(
    question: str,
) -> Dict[str, bool]:
    """
    Detect lightweight query intent.

    This is NOT an LLM classifier.

    It is a deterministic precision layer used after
    semantic retrieval.

    The purpose is to prevent cases such as:

        "restaurant service"

    returning only:

        "food was delicious"

    or:

        "high ratings"

    returning:

        rating = 3

    """

    q = normalize_text(
        question
    )

    intents = {
        "rating_low": False,
        "rating_high": False,

        "food": False,
        "delivery": False,
        "service": False,
        "price": False,
        "packaging": False,

        "positive": False,
        "negative": False,
        "problem": False,

        "general": False,
    }

    # --------------------------------------------------------
    # Rating intent
    # --------------------------------------------------------

    if (
        "low rating" in q
        or "low ratings" in q
        or "poor rating" in q
        or "poor ratings" in q
        or "bad rating" in q
        or "bad ratings" in q
        or "low rated" in q
        or "low-rated" in q
    ):

        intents["rating_low"] = True

    if (
        "high rating" in q
        or "high ratings" in q
        or "high rated" in q
        or "high-rated" in q
        or "good rating" in q
        or "good ratings" in q
        or "top rated" in q
        or "top-rated" in q
        or "5 star" in q
        or "5-star" in q
    ):

        intents["rating_high"] = True

    # --------------------------------------------------------
    # Food
    # --------------------------------------------------------

    if _question_has_any(
        q,
        [
            "food",
            "taste",
            "tasty",
            "flavor",
            "flavour",
            "quality",
            "fresh",
            "freshness",
            "meal",
            "dish",
            "cuisine",
        ],
    ):

        intents["food"] = True

    # --------------------------------------------------------
    # Delivery
    # --------------------------------------------------------

    if _question_has_any(
        q,
        [
            "delivery",
            "delivered",
            "delivery partner",
            "arrived",
            "late",
            "delayed",
            "delivery time",
        ],
    ):

        intents["delivery"] = True

    # --------------------------------------------------------
    # Service / staff
    # --------------------------------------------------------

    if _question_has_any(
        q,
        [
            "service",
            "staff",
            "waiter",
            "hospitality",
            "restaurant service",
            "customer service",
        ],
    ):

        intents["service"] = True

    # --------------------------------------------------------
    # Price / value
    # --------------------------------------------------------

    if _question_has_any(
        q,
        [
            "price",
            "pricing",
            "value",
            "value for money",
            "cost",
            "cheap",
            "expensive",
            "affordable",
            "worth",
            "money",
        ],
    ):

        intents["price"] = True

    # --------------------------------------------------------
    # Packaging
    # --------------------------------------------------------

    if _question_has_any(
        q,
        [
            "packaging",
            "package",
            "packed",
            "box",
            "container",
        ],
    ):

        intents["packaging"] = True

    # --------------------------------------------------------
    # Positive
    # --------------------------------------------------------

    if _question_has_any(
        q,
        [
            "positive",
            "good",
            "best",
            "like",
            "liked",
            "praise",
            "satisfied",
            "happy",
        ],
    ):

        intents["positive"] = True

    # --------------------------------------------------------
    # Negative
    # --------------------------------------------------------

    if _question_has_any(
        q,
        [
            "negative",
            "bad",
            "worst",
            "dislike",
            "disliked",
            "complaint",
            "complaints",
            "complain",
            "rude",
            "poor",
        ],
    ):

        intents["negative"] = True

    # --------------------------------------------------------
    # Problems
    # --------------------------------------------------------

    if _question_has_any(
        q,
        [
            "problem",
            "problems",
            "issue",
            "issues",
            "complaint",
            "complaints",
            "complain",
            "wrong",
            "missing",
            "never arrived",
        ],
    ):

        intents["problem"] = True

    # --------------------------------------------------------
    # General questions
    # --------------------------------------------------------

    if not any(
        [
            intents["rating_low"],
            intents["rating_high"],
            intents["food"],
            intents["delivery"],
            intents["service"],
            intents["price"],
            intents["packaging"],
            intents["positive"],
            intents["negative"],
            intents["problem"],
        ]
    ):

        intents["general"] = True

    return intents


# ============================================================
# EVIDENCE KEYWORDS
# ============================================================

FOOD_TERMS = {
    "food",
    "taste",
    "tasty",
    "delicious",
    "flavor",
    "flavour",
    "fresh",
    "freshness",
    "meal",
    "dish",
    "cuisine",
    "cooked",
    "cooking",
    "bland",
    "stale",
    "cold",
    "hot",
    "spicy",
    "authentic",
    "yummy",
}

DELIVERY_TERMS = {
    "delivery",
    "delivered",
    "arrived",
    "arrival",
    "late",
    "delayed",
    "delay",
    "partner",
    "courier",
    "quick",
    "fast",
    "slow",
    "time",
}

SERVICE_TERMS = {
    "service",
    "staff",
    "waiter",
    "hospitality",
    "restaurant",
    "manager",
    "helpful",
    "rude",
    "friendly",
    "unhelpful",
}

PRICE_TERMS = {
    "price",
    "pricing",
    "value",
    "cost",
    "cheap",
    "expensive",
    "affordable",
    "worth",
    "money",
    "pricey",
}

PACKAGING_TERMS = {
    "packaging",
    "package",
    "packed",
    "box",
    "container",
    "pack",
    "eco",
}

POSITIVE_TERMS = {
    "good",
    "great",
    "excellent",
    "amazing",
    "delicious",
    "tasty",
    "fresh",
    "helpful",
    "polite",
    "friendly",
    "perfect",
    "perfectly",
    "best",
    "love",
    "loved",
    "fast",
    "quick",
    "value",
    "worth",
    "satisfied",
}

NEGATIVE_TERMS = {
    "bad",
    "terrible",
    "worst",
    "poor",
    "rude",
    "late",
    "delayed",
    "delay",
    "problem",
    "problems",
    "issue",
    "issues",
    "complaint",
    "complaints",
    "wrong",
    "missing",
    "cold",
    "stale",
    "bland",
    "tasteless",
    "dropped",
    "never",
    "unhelpful",
    "slow",
}


def _comment_matches_terms(
    comment: str,
    terms: set[str],
) -> bool:

    text = normalize_text(
        comment
    )

    tokens = set(
        text.split()
    )

    return bool(
        tokens.intersection(
            terms
        )
    )


# ============================================================
# QUERY-AWARE EVIDENCE FILTER
# ============================================================

def _apply_query_constraints(
    candidates: pd.DataFrame,
    question: str,
) -> pd.DataFrame:
    """
    Apply lightweight query-aware precision filtering.

    Important:

    We DO NOT require every query term to appear in the
    comment.

    For multi-topic questions, a review can match at least
    one requested aspect.

    Examples:

        "food and service"

    can retrieve a food-related review even if it does not
    mention service. The LLM is then instructed to state
    that service evidence is limited or absent.

    This is intentionally conservative.
    """

    if candidates.empty:

        return candidates.copy()

    working = candidates.copy()

    intents = detect_query_intents(
        question
    )

    # --------------------------------------------------------
    # Explicit rating constraints
    # --------------------------------------------------------

    if intents["rating_low"]:

        working = working[
            pd.to_numeric(
                working["rating"],
                errors="coerce",
            )
            <= 2
        ]

    elif intents["rating_high"]:

        working = working[
            pd.to_numeric(
                working["rating"],
                errors="coerce",
            )
            >= 4
        ]

    # --------------------------------------------------------
    # If explicit rating filtering removed everything,
    # return empty rather than inventing evidence.
    # --------------------------------------------------------

    if working.empty:

        return working.reset_index(
            drop=True
        )

    # --------------------------------------------------------
    # Determine semantic aspect terms.
    #
    # We only apply this when the question has a specific
    # aspect that can be reliably recognized.
    # --------------------------------------------------------

    aspect_sets = []

    if intents["food"]:

        aspect_sets.append(
            FOOD_TERMS
        )

    if intents["delivery"]:

        aspect_sets.append(
            DELIVERY_TERMS
        )

    if intents["service"]:

        aspect_sets.append(
            SERVICE_TERMS
        )

    if intents["price"]:

        aspect_sets.append(
            PRICE_TERMS
        )

    if intents["packaging"]:

        aspect_sets.append(
            PACKAGING_TERMS
        )

    # --------------------------------------------------------
    # Positive / negative intent
    # --------------------------------------------------------

    sentiment_sets = []

    if (
        intents["positive"]
        and not intents["negative"]
    ):

        sentiment_sets.append(
            POSITIVE_TERMS
        )

    if (
        intents["negative"]
        or intents["problem"]
    ):

        sentiment_sets.append(
            NEGATIVE_TERMS
        )

    # --------------------------------------------------------
    # Combine aspect + sentiment terms.
    #
    # A candidate is accepted if it matches ANY relevant
    # aspect/sentiment family.
    # --------------------------------------------------------

    all_term_sets = (
        aspect_sets
        + sentiment_sets
    )

    if not all_term_sets:

        return working.reset_index(
            drop=True
        )

    keep_mask = []

    for _, row in (
        working.iterrows()
    ):

        comment = _safe_text(
            row.get(
                "comment",
                "",
            )
        )

        matched = any(
            _comment_matches_terms(
                comment,
                terms,
            )
            for terms in all_term_sets
        )

        keep_mask.append(
            matched
        )

    filtered = working[
        keep_mask
    ].copy()

    filtered.reset_index(
        drop=True,
        inplace=True,
    )

    return filtered


# ============================================================
# QDRANT CLIENT
# ============================================================

def _get_qdrant_client() -> QdrantClient:
    """
    Create/cache Qdrant Cloud client.
    """

    global _QDRANT_CLIENT

    if _QDRANT_CLIENT is not None:

        return _QDRANT_CLIENT

    _load_environment()

    qdrant_url = os.getenv(
        "QDRANT_URL"
    )

    qdrant_api_key = os.getenv(
        "QDRANT_API_KEY"
    )

    if not qdrant_url:

        raise RuntimeError(
            "QDRANT_URL is not set. "
            "Expected it in airflow/.env."
        )

    if not qdrant_api_key:

        raise RuntimeError(
            "QDRANT_API_KEY is not set. "
            "Expected it in airflow/.env."
        )

    _QDRANT_CLIENT = QdrantClient(
        url=qdrant_url,
        api_key=qdrant_api_key,
        cloud_inference=True,
    )

    return _QDRANT_CLIENT


# ============================================================
# BIGQUERY CLIENT
# ============================================================

def _get_bigquery_client():
    """
    Create/cache BigQuery client.

    Supports:

        Local Windows
        Airflow Docker
        Render
    """

    global _BQ_CLIENT

    if _BQ_CLIENT is not None:

        return _BQ_CLIENT

    _load_environment()

    project_id = (
        _get_project_id()
    )

    project_root = (
        Path(__file__).resolve().parent.parent
    )

    local_credentials = (
        project_root
        / "airflow"
        / "credentials"
        / "service-account.json"
    )

    airflow_credentials = Path(
        "/opt/airflow/credentials/service-account.json"
    )

    render_credentials = Path(
        "/etc/secrets/zomato-service-account.json"
    )

    credential_candidates = [
        local_credentials,
        airflow_credentials,
        render_credentials,
    ]

    credential_file = None

    for candidate in (
        credential_candidates
    ):

        if candidate.exists():

            credential_file = candidate
            break

    if credential_file is not None:

        print()
        print(
            "Using BigQuery credentials:"
        )

        print(
            f"  {credential_file}"
        )

        _BQ_CLIENT = (
            bigquery.Client
            .from_service_account_json(
                str(
                    credential_file
                ),
                project=project_id,
            )
        )

    else:

        print()
        print(
            "No explicit BigQuery "
            "service-account file found."
        )

        print(
            "Using Application Default Credentials."
        )

        _BQ_CLIENT = (
            bigquery.Client(
                project=project_id
            )
        )

    return _BQ_CLIENT


# ============================================================
# QDRANT COLLECTION VALIDATION
# ============================================================

def _validate_qdrant_collection() -> None:
    """
    Verify production Qdrant collection.
    """

    client = (
        _get_qdrant_client()
    )

    try:

        info = client.get_collection(
            collection_name=QDRANT_COLLECTION
        )

    except Exception as exc:

        raise RuntimeError(
            f"Could not access Qdrant collection "
            f"'{QDRANT_COLLECTION}'."
        ) from exc

    vectors = (
        info.config.params.vectors
    )

    size = getattr(
        vectors,
        "size",
        None,
    )

    if size != EMBEDDING_DIMENSION:

        raise RuntimeError(
            "Unexpected Qdrant vector dimension. "
            f"Expected {EMBEDDING_DIMENSION}, "
            f"found {size}."
        )


# ============================================================
# QUESTION DOCUMENT
# ============================================================

def generate_question_embedding(
    question: str,
) -> models.Document:
    """
    Prepare a question for Qdrant Cloud Inference.

    No local embedding is generated.
    """

    if (
        not question
        or not question.strip()
    ):

        raise ValueError(
            "Question cannot be empty."
        )

    return models.Document(
        text=question.strip(),
        model=EMBEDDING_MODEL,
    )


# ============================================================
# QDRANT SEARCH
# ============================================================

def semantic_search(
    question_embedding: models.Document,
    candidate_k: int,
) -> pd.DataFrame:
    """
    Retrieve candidate review IDs and vectors from Qdrant.
    """

    client = (
        _get_qdrant_client()
    )

    candidate_k = max(
        1,
        int(candidate_k),
    )

    response = client.query_points(
        collection_name=QDRANT_COLLECTION,

        query=question_embedding,

        limit=candidate_k,

        with_payload=False,

        with_vectors=True,
    )

    rows = []

    for point in response.points:

        vector = getattr(
            point,
            "vector",
            None,
        )

        if isinstance(
            vector,
            dict,
        ):

            vector = next(
                iter(
                    vector.values()
                )
            )

        rows.append(
            {
                "review_id": str(
                    point.id
                ),

                "score": float(
                    point.score
                ),

                "_qdrant_vector": (
                    np.asarray(
                        vector,
                        dtype=np.float32,
                    )
                    if vector is not None
                    else None
                ),
            }
        )

    if not rows:

        return pd.DataFrame(
            columns=[
                "review_id",
                "score",
                "_qdrant_vector",
            ]
        )

    result = pd.DataFrame(
        rows
    )

    result.sort_values(
        "score",
        ascending=False,
        inplace=True,
    )

    result.reset_index(
        drop=True,
        inplace=True,
    )

    return result


# ============================================================
# ADAPTIVE SCORE FILTER
# ============================================================

def apply_adaptive_relevance_filter(
    candidates: pd.DataFrame,
) -> Tuple[
    pd.DataFrame,
    float,
]:
    """
    Apply adaptive semantic relevance.

    Old:

        score >= 0.50

    New:

        threshold =
            max(
                0.36,
                best_score - 0.08
            )

    This means:

    If best score = 0.4331:

        threshold = 0.36

    If best score = 0.5105:

        threshold = 0.4305

    If best score = 0.3598:

        threshold = 0.36

        -> no candidates

    This is intentionally conservative.
    """

    if candidates.empty:

        return (
            candidates.copy(),
            MIN_RELEVANCE_SCORE,
        )

    top_score = float(
        candidates[
            "score"
        ].max()
    )

    adaptive_threshold = max(
        MIN_RELEVANCE_SCORE,
        top_score - SCORE_WINDOW,
    )

    result = candidates[
        candidates["score"]
        >= adaptive_threshold
    ].copy()

    result.reset_index(
        drop=True,
        inplace=True,
    )

    return (
        result,
        adaptive_threshold,
    )


# ============================================================
# BIGQUERY REVIEW RETRIEVAL
# ============================================================

def fetch_reviews_from_bigquery(
    review_ids: List[str],
) -> pd.DataFrame:
    """
    Fetch review text and optional enrichment from BigQuery.
    """

    if not review_ids:

        return _empty_review_dataframe()

    ordered_ids = list(
        dict.fromkeys(
            str(review_id)
            for review_id in review_ids
        )
    )

    client = (
        _get_bigquery_client()
    )

    reviews_table = (
        _get_reviews_table()
    )

    enriched_table = (
        _get_enriched_reviews_table()
    )

    query = f"""
        WITH latest_enrichment AS (

            SELECT

                CAST(
                    review_id
                    AS STRING
                ) AS review_id,

                sentiment_label,

                sentiment_score,

                topic,

                key_issue

            FROM `{enriched_table}`

            QUALIFY ROW_NUMBER() OVER (

                PARTITION BY
                    CAST(
                        review_id
                        AS STRING
                    )

                ORDER BY
                    processed_at DESC

            ) = 1
        )

        SELECT

            CAST(
                r.review_id
                AS STRING
            ) AS review_id,

            SAFE_CAST(
                r.rating
                AS INT64
            ) AS rating,

            CAST(
                r.comment
                AS STRING
            ) AS comment,

            r.review_date,

            e.sentiment_label,

            e.sentiment_score,

            e.topic,

            e.key_issue

        FROM `{reviews_table}` AS r

        LEFT JOIN latest_enrichment AS e

            ON e.review_id =
               CAST(
                   r.review_id
                   AS STRING
               )

        WHERE CAST(
            r.review_id
            AS STRING
        ) IN UNNEST(
            @review_ids
        )
    """

    job_config = (
        bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ArrayQueryParameter(
                    "review_ids",
                    "STRING",
                    ordered_ids,
                )
            ]
        )
    )

    rows = (
        client.query(
            query,
            job_config=job_config,
            location=BQ_LOCATION,
        )
        .result()
    )

    result = (
        rows.to_dataframe()
    )

    if result.empty:

        return _empty_review_dataframe()

    order_map = {
        review_id: position
        for position, review_id
        in enumerate(
            ordered_ids
        )
    }

    result[
        "_retrieval_order"
    ] = (
        result[
            "review_id"
        ]
        .astype(str)
        .map(order_map)
    )

    result.sort_values(
        "_retrieval_order",
        inplace=True,
    )

    result.drop(
        columns=[
            "_retrieval_order"
        ],
        inplace=True,
    )

    result.reset_index(
        drop=True,
        inplace=True,
    )

    return result


# ============================================================
# ATTACH BIGQUERY
# ============================================================

def attach_bigquery_evidence(
    candidates: pd.DataFrame,
) -> pd.DataFrame:
    """
    Map Qdrant review IDs to BigQuery evidence.
    """

    if candidates.empty:

        return candidates.copy()

    review_ids = (
        candidates[
            "review_id"
        ]
        .astype(str)
        .tolist()
    )

    reviews = (
        fetch_reviews_from_bigquery(
            review_ids
        )
    )

    if reviews.empty:

        return pd.DataFrame(
            columns=list(
                candidates.columns
            )
            + [
                column
                for column in REVIEW_COLUMNS
                if column
                not in candidates.columns
            ]
        )

    result = candidates.merge(
        reviews,
        on="review_id",
        how="inner",
        sort=False,
    )

    result.sort_values(
        "score",
        ascending=False,
        inplace=True,
    )

    result.reset_index(
        drop=True,
        inplace=True,
    )

    return result


# ============================================================
# EXACT DUPLICATES
# ============================================================

def remove_exact_duplicates(
    candidates: pd.DataFrame,
) -> Tuple[
    pd.DataFrame,
    int,
]:
    """
    Remove exact duplicate review text.

    Important:

    Multiple review IDs with the exact same text are NOT
    treated as independent evidence.
    """

    if candidates.empty:

        return (
            candidates.copy(),
            0,
        )

    seen = set()

    keep_rows = []

    removed = 0

    for idx, row in (
        candidates.iterrows()
    ):

        normalized = normalize_text(
            row.get(
                "comment",
                "",
            )
        )

        if not normalized:

            removed += 1

            continue

        if normalized in seen:

            removed += 1

            continue

        seen.add(
            normalized
        )

        keep_rows.append(
            idx
        )

    result = candidates.loc[
        keep_rows
    ].copy()

    result.reset_index(
        drop=True,
        inplace=True,
    )

    return (
        result,
        removed,
    )


# ============================================================
# NEAR DUPLICATES
# ============================================================

def remove_near_duplicates(
    candidates: pd.DataFrame,
) -> Tuple[
    pd.DataFrame,
    int,
]:
    """
    Remove highly similar review wording.
    """

    if candidates.empty:

        return (
            candidates.copy(),
            0,
        )

    kept_indices = []

    removed = 0

    for idx, row in (
        candidates.iterrows()
    ):

        current_text = _safe_text(
            row.get(
                "comment",
                "",
            )
        )

        is_duplicate = False

        for kept_idx in (
            kept_indices
        ):

            kept_text = _safe_text(
                candidates.loc[
                    kept_idx,
                    "comment",
                ]
            )

            similarity = (
                token_similarity(
                    current_text,
                    kept_text,
                )
            )

            if (
                similarity
                >= TEXT_DUPLICATE_THRESHOLD
            ):

                is_duplicate = True

                removed += 1

                break

        if not is_duplicate:

            kept_indices.append(
                idx
            )

    result = candidates.loc[
        kept_indices
    ].copy()

    result.reset_index(
        drop=True,
        inplace=True,
    )

    return (
        result,
        removed,
    )


# ============================================================
# REVIEW PATTERN
# ============================================================

def detect_review_pattern(
    comment: str,
) -> str:
    """
    Lightweight pattern detection.

    Used only for diversity.
    """

    text = normalize_text(
        comment
    )

    patterns = []

    if _comment_matches_terms(
        text,
        FOOD_TERMS,
    ):

        patterns.append(
            "food"
        )

    if _comment_matches_terms(
        text,
        DELIVERY_TERMS,
    ):

        patterns.append(
            "delivery"
        )

    if _comment_matches_terms(
        text,
        SERVICE_TERMS,
    ):

        patterns.append(
            "service"
        )

    if _comment_matches_terms(
        text,
        PRICE_TERMS,
    ):

        patterns.append(
            "price"
        )

    if _comment_matches_terms(
        text,
        PACKAGING_TERMS,
    ):

        patterns.append(
            "packaging"
        )

    if _comment_matches_terms(
        text,
        NEGATIVE_TERMS,
    ):

        patterns.append(
            "negative"
        )

    if _comment_matches_terms(
        text,
        POSITIVE_TERMS,
    ):

        patterns.append(
            "positive"
        )

    if not patterns:

        patterns.append(
            "other"
        )

    return "|".join(
        sorted(
            set(patterns)
        )
    )


# ============================================================
# DIVERSITY SELECTION
# ============================================================

def select_diverse_reviews(
    candidates: pd.DataFrame,
    final_k: int = FINAL_K,
) -> Tuple[
    pd.DataFrame,
    Dict[str, Any],
]:
    """
    Select final evidence using:

        1. relevance
        2. exact-text diversity
        3. semantic vector diversity
        4. pattern diversity

    It is acceptable to return fewer than five reviews.

    We never insert irrelevant reviews simply to reach five.
    """

    if candidates.empty:

        return (
            _empty_review_dataframe(),
            {
                "selected_count": 0,
                "dominant_patterns": [],
                "distribution": {},
            },
        )

    working = candidates.copy()

    working[
        "pattern"
    ] = working[
        "comment"
    ].apply(
        detect_review_pattern
    )

    pattern_counts = (
        working[
            "pattern"
        ]
        .value_counts()
        .to_dict()
    )

    total = len(
        working
    )

    dominant_patterns = []

    for pattern, count in (
        pattern_counts.items()
    ):

        if (
            total > 0
            and (
                count / total
            )
            >= DOMINANT_PATTERN_RATIO
        ):

            dominant_patterns.append(
                pattern
            )

    selected_indices = []

    pattern_selected = {}

    selected_texts = []

    selected_vectors = []

    # --------------------------------------------------------
    # FIRST PASS
    # --------------------------------------------------------

    for idx, row in (
        working.iterrows()
    ):

        if (
            len(
                selected_indices
            )
            >= final_k
        ):

            break

        pattern = row[
            "pattern"
        ]

        current_pattern_count = (
            pattern_selected.get(
                pattern,
                0,
            )
        )

        if (
            pattern
            in dominant_patterns
            and current_pattern_count
            >= MAX_SAME_PATTERN
        ):

            continue

        # ----------------------------------------------------
        # Text diversity
        # ----------------------------------------------------

        duplicate_text = False

        for previous_text in (
            selected_texts
        ):

            similarity = (
                token_similarity(
                    row["comment"],
                    previous_text,
                )
            )

            if (
                similarity
                >= TEXT_DUPLICATE_THRESHOLD
            ):

                duplicate_text = True

                break

        if duplicate_text:

            continue

        # ----------------------------------------------------
        # Semantic diversity
        # ----------------------------------------------------

        current_vector = row.get(
            "_qdrant_vector"
        )

        if current_vector is not None:

            current_vector = np.asarray(
                current_vector,
                dtype=np.float32,
            )

            norm = np.linalg.norm(
                current_vector
            )

            if norm > 0:

                current_vector = (
                    current_vector
                    / norm
                )

        semantic_duplicate = False

        if current_vector is not None:

            for previous_vector in (
                selected_vectors
            ):

                similarity = float(
                    np.dot(
                        current_vector,
                        previous_vector,
                    )
                )

                if (
                    similarity
                    >= SELECTED_SIMILARITY_MAX
                ):

                    semantic_duplicate = True

                    break

        if semantic_duplicate:

            continue

        # ----------------------------------------------------
        # Accept
        # ----------------------------------------------------

        selected_indices.append(
            idx
        )

        selected_texts.append(
            _safe_text(
                row["comment"]
            )
        )

        if current_vector is not None:

            selected_vectors.append(
                current_vector
            )

        pattern_selected[
            pattern
        ] = (
            current_pattern_count
            + 1
        )

    # --------------------------------------------------------
    # SECOND PASS
    #
    # Relax pattern limit, but keep duplicate protection.
    # --------------------------------------------------------

    if (
        len(
            selected_indices
        )
        < final_k
    ):

        for idx, row in (
            working.iterrows()
        ):

            if (
                len(
                    selected_indices
                )
                >= final_k
            ):

                break

            if idx in selected_indices:

                continue

            # ------------------------------------------------
            # Text diversity
            # ------------------------------------------------

            duplicate_text = False

            for previous_text in (
                selected_texts
            ):

                similarity = (
                    token_similarity(
                        row["comment"],
                        previous_text,
                    )
                )

                if (
                    similarity
                    >= TEXT_DUPLICATE_THRESHOLD
                ):

                    duplicate_text = True

                    break

            if duplicate_text:

                continue

            # ------------------------------------------------
            # Vector diversity
            # ------------------------------------------------

            current_vector = row.get(
                "_qdrant_vector"
            )

            if current_vector is not None:

                current_vector = np.asarray(
                    current_vector,
                    dtype=np.float32,
                )

                norm = np.linalg.norm(
                    current_vector
                )

                if norm > 0:

                    current_vector = (
                        current_vector
                        / norm
                    )

            semantic_duplicate = False

            if current_vector is not None:

                for previous_vector in (
                    selected_vectors
                ):

                    similarity = float(
                        np.dot(
                            current_vector,
                            previous_vector,
                        )
                    )

                    if (
                        similarity
                        >= SELECTED_SIMILARITY_MAX
                    ):

                        semantic_duplicate = True

                        break

            if semantic_duplicate:

                continue

            # ------------------------------------------------
            # Accept
            # ------------------------------------------------

            selected_indices.append(
                idx
            )

            selected_texts.append(
                _safe_text(
                    row["comment"]
                )
            )

            if current_vector is not None:

                selected_vectors.append(
                    current_vector
                )

            pattern = row[
                "pattern"
            ]

            pattern_selected[
                pattern
            ] = (
                pattern_selected.get(
                    pattern,
                    0,
                )
                + 1
            )

    result = working.loc[
        selected_indices
    ].copy()

    result.reset_index(
        drop=True,
        inplace=True,
    )

    # --------------------------------------------------------
    # Remove internal columns from returned result
    # --------------------------------------------------------

    diagnostics = {
        "selected_count": len(
            result
        ),
        "dominant_patterns": (
            dominant_patterns
        ),
        "distribution": (
            pattern_selected
        ),
    }

    return (
        result,
        diagnostics,
    )


# ============================================================
# PROGRESSIVE RETRIEVAL
# ============================================================

def progressive_retrieve(
    question: str,
    question_embedding: models.Document,
) -> Tuple[
    pd.DataFrame,
    Dict[str, Any],
]:
    """
    Progressive retrieval with adaptive relevance.

    Stages:

        100
        500
        1000
        2500
        5000

    The search expands until five sufficiently different
    reviews are found or the maximum stage is reached.
    """

    print()

    _print_separator()

    print(
        "ADAPTIVE PROGRESSIVE QDRANT RETRIEVAL"
    )

    _print_separator()

    intents = detect_query_intents(
        question
    )

    print()

    print(
        "Detected query intent:"
    )

    print(
        f"  {intents}"
    )

    final_reviews = (
        _empty_review_dataframe()
    )

    final_diagnostics = {}

    stopped_early = False

    previous_selected_ids = set()

    for stage_index, candidate_level in (
        enumerate(
            CANDIDATE_LEVELS
        )
    ):

        print()

        print(
            f"[Stage {stage_index + 1}] "
            f"Searching top "
            f"{candidate_level:,} "
            f"reviews in Qdrant..."
        )

        # ----------------------------------------------------
        # Qdrant
        # ----------------------------------------------------

        qdrant_candidates = (
            semantic_search(
                question_embedding,
                candidate_level,
            )
        )

        print(
            f"Qdrant candidates: "
            f"{len(qdrant_candidates):,}"
        )

        if qdrant_candidates.empty:

            final_reviews = (
                _empty_review_dataframe()
            )

            continue

        # ----------------------------------------------------
        # Adaptive relevance
        # ----------------------------------------------------

        (
            qdrant_candidates,
            adaptive_threshold,
        ) = (
            apply_adaptive_relevance_filter(
                qdrant_candidates
            )
        )

        print(
            f"Adaptive relevance threshold: "
            f"{adaptive_threshold:.4f}"
        )

        print(
            f"After adaptive relevance: "
            f"{len(qdrant_candidates):,}"
        )

        if qdrant_candidates.empty:

            candidates = (
                pd.DataFrame()
            )

            exact_removed = 0
            near_removed = 0
            query_filtered = 0

        else:

            # ------------------------------------------------
            # BigQuery
            # ------------------------------------------------

            candidates = (
                attach_bigquery_evidence(
                    qdrant_candidates
                )
            )

            print(
                f"BigQuery evidence rows: "
                f"{len(candidates):,}"
            )

            before_query_filter = len(
                candidates
            )

            # ------------------------------------------------
            # Query-aware filtering
            # ------------------------------------------------

            candidates = (
                _apply_query_constraints(
                    candidates,
                    question,
                )
            )

            query_filtered = (
                before_query_filter
                - len(candidates)
            )

            print(
                f"Query-aware evidence filtering "
                f"removed: "
                f"{query_filtered:,}"
            )

            # ------------------------------------------------
            # Exact duplicates
            # ------------------------------------------------

            (
                candidates,
                exact_removed,
            ) = (
                remove_exact_duplicates(
                    candidates
                )
            )

            print(
                f"Exact duplicates removed: "
                f"{exact_removed:,}"
            )

            # ------------------------------------------------
            # Near duplicates
            # ------------------------------------------------

            (
                candidates,
                near_removed,
            ) = (
                remove_near_duplicates(
                    candidates
                )
            )

            print(
                f"Near-duplicate reviews removed: "
                f"{near_removed:,}"
            )

            print(
                f"Different review pool: "
                f"{len(candidates):,}"
            )

        # ----------------------------------------------------
        # Diversity
        # ----------------------------------------------------

        (
            selected,
            diversity_diagnostics,
        ) = (
            select_diverse_reviews(
                candidates,
                FINAL_K,
            )
        )

        print(
            f"Sufficiently different reviews: "
            f"{len(selected):,}"
        )

        # ----------------------------------------------------
        # Keep the best available stage result.
        #
        # If a later stage accidentally produces less useful
        # evidence, do not erase a previously valid result.
        # ----------------------------------------------------

        selected_ids = set()

        if not selected.empty:

            selected_ids = set(
                selected[
                    "review_id"
                ]
                .astype(str)
            )

        if (
            len(selected)
            > len(final_reviews)
        ):

            final_reviews = selected

        elif (
            final_reviews.empty
            and not selected.empty
        ):

            final_reviews = selected

        # ----------------------------------------------------
        # Diagnostics
        # ----------------------------------------------------

        final_diagnostics = {
            "candidate_level": (
                candidate_level
            ),

            "semantic_candidates": (
                len(qdrant_candidates)
            ),

            "adaptive_threshold": (
                adaptive_threshold
                if not qdrant_candidates.empty
                else MIN_RELEVANCE_SCORE
            ),

            "bigquery_evidence_rows": (
                len(candidates)
            ),

            "query_filtered": (
                query_filtered
            ),

            "exact_duplicates_removed": (
                exact_removed
            ),

            "near_duplicates_removed": (
                near_removed
            ),

            "selected_count": (
                len(selected)
            ),

            "dominant_patterns": (
                diversity_diagnostics[
                    "dominant_patterns"
                ]
            ),

            "distribution": (
                diversity_diagnostics[
                    "distribution"
                ]
            ),
        }

        # ----------------------------------------------------
        # Five reviews found
        # ----------------------------------------------------

        if (
            len(selected)
            >= FINAL_K
        ):

            stopped_early = True

            print()

            print(
                f"✓ Found {FINAL_K} "
                f"sufficiently different reviews."
            )

            print(
                f"✓ Stopping at top "
                f"{candidate_level:,}."
            )

            break

        # ----------------------------------------------------
        # Expand
        # ----------------------------------------------------

        if (
            stage_index
            < len(
                CANDIDATE_LEVELS
            ) - 1
        ):

            next_level = (
                CANDIDATE_LEVELS[
                    stage_index + 1
                ]
            )

            print()

            print(
                f"Only {len(selected)} "
                f"sufficiently different "
                f"reviews found."
            )

            print(
                f"→ Expanding retrieval: "
                f"{candidate_level:,} → "
                f"{next_level:,}"
            )

        else:

            print()

            print(
                "→ Maximum candidate pool reached."
            )

    final_diagnostics[
        "stopped_early"
    ] = stopped_early

    final_diagnostics[
        "final_reviews"
    ] = len(
        final_reviews
    )

    return (
        final_reviews,
        final_diagnostics,
    )


# ============================================================
# EVIDENCE FORMATTER
# ============================================================

def format_evidence(
    reviews: pd.DataFrame,
) -> str:
    """
    Convert reviews into LLM evidence.
    """

    if reviews.empty:

        return (
            "No sufficiently relevant "
            "customer reviews were retrieved."
        )

    evidence_parts = []

    for _, row in (
        reviews.iterrows()
    ):

        review_id = _safe_text(
            row.get(
                "review_id",
                "unknown",
            )
        )

        rating = _safe_text(
            row.get(
                "rating",
                "unknown",
            )
        )

        comment = _safe_text(
            row.get(
                "comment",
                "",
            )
        ).strip()

        if (
            len(comment)
            > MAX_REVIEW_TEXT_LENGTH
        ):

            comment = (
                comment[
                    :MAX_REVIEW_TEXT_LENGTH
                ]
                + "..."
            )

        block = [
            f"Review ID: {review_id}",
            f"Rating: {rating}",
            f"Comment: {comment}",
        ]

        sentiment = _safe_text(
            row.get(
                "sentiment_label",
                "",
            )
        )

        sentiment_score = row.get(
            "sentiment_score",
            None,
        )

        topic = _safe_text(
            row.get(
                "topic",
                "",
            )
        )

        key_issue = _safe_text(
            row.get(
                "key_issue",
                "",
            )
        )

        if sentiment:

            block.append(
                f"Sentiment: {sentiment}"
            )

        if (
            sentiment_score
            is not None
        ):

            try:

                if not pd.isna(
                    sentiment_score
                ):

                    block.append(
                        "Sentiment score: "
                        f"{sentiment_score}"
                    )

            except (
                TypeError,
                ValueError,
            ):

                pass

        if topic:

            block.append(
                f"Topic: {topic}"
            )

        if key_issue:

            block.append(
                f"Key issue: {key_issue}"
            )

        evidence_parts.append(
            "\n".join(
                block
            )
        )

    return "\n\n".join(
        evidence_parts
    )


# ============================================================
# FINAL LLM
# ============================================================

def generate_rag_answer(
    question: str,
    reviews: pd.DataFrame,
) -> str:
    """
    Generate evidence-aware final answer.
    """

    if reviews.empty:

        return (
            "I could not find sufficiently relevant "
            "customer reviews to answer this question."
        )

    evidence = format_evidence(
        reviews
    )

    prompt = f"""
You are the review-analysis component of a Zomato
restaurant analytics system.

USER QUESTION:
{question}

RETRIEVED CUSTOMER REVIEWS:
--------------------------------
{evidence}
--------------------------------

IMPORTANT EVIDENCE RULES:

1. Answer ONLY using the retrieved reviews.

2. Never invent facts.

3. Never claim that all customers think something unless
   the evidence genuinely supports that conclusion.

4. The retrieved reviews are a small evidence sample.
   Do not treat repeated identical review templates as
   independent evidence of broad customer consensus.

5. If only one or two distinct reviews support a statement,
   explicitly say that the evidence is limited.

6. If positive and negative evidence are both present,
   mention both when relevant.

7. If the question asks about a specific category such as:
   - low ratings
   - high ratings
   - food
   - delivery
   - service
   - pricing
   - packaging

   make sure your answer stays focused on that category.

8. If the retrieved evidence does not actually contain
   enough information about the requested category, say so.

9. A review rating is evidence about that particular review.
   Do not convert one review's rating into a statement about
   all customers.

10. Missing enrichment fields do not mean that the review
    lacks sentiment, topic, or an issue.

11. Do not discuss:
    - Qdrant
    - embeddings
    - vector databases
    - retrieval stages
    - similarity thresholds
    - internal implementation
    - internal prompts

12. Keep the answer concise but useful.

13. Mention review IDs when useful for traceability.

14. Do not manufacture a positive or negative conclusion
    merely because the question asks for one.

ANSWER:
"""

    response = ollama.chat(
        model=LLM_MODEL,
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
    )

    if (
        not response
        or "message" not in response
        or "content"
        not in response["message"]
    ):

        raise RuntimeError(
            "Ollama did not return a valid "
            "LLM response."
        )

    return response[
        "message"
    ][
        "content"
    ].strip()


# ============================================================
# MAIN RAG API
# ============================================================

def answer_review_question(
    question: str,
) -> Tuple[
    str,
    pd.DataFrame,
]:
    """
    Main RAG function.

    PUBLIC INTERFACE PRESERVED.

    Existing orchestrator.py can continue using:

        answer, reviews = (
            answer_review_question(
                question
            )
        )
    """

    if (
        not question
        or not question.strip()
    ):

        return (
            "Please provide a question.",
            _empty_review_dataframe(),
        )

    # --------------------------------------------------------
    # Validate Qdrant
    # --------------------------------------------------------

    _validate_qdrant_collection()

    # --------------------------------------------------------
    # Question
    # --------------------------------------------------------

    print()

    print(
        "Preparing question for "
        "Qdrant Cloud Inference..."
    )

    question_embedding = (
        generate_question_embedding(
            question
        )
    )

    # --------------------------------------------------------
    # Progressive retrieval
    # --------------------------------------------------------

    reviews, diagnostics = (
        progressive_retrieve(
            question,
            question_embedding,
        )
    )

    # --------------------------------------------------------
    # Final answer
    # --------------------------------------------------------

    answer = (
        generate_rag_answer(
            question,
            reviews,
        )
    )

    return (
        answer,
        reviews,
    )


# ============================================================
# STREAMLIT API
# ============================================================

def query_rag(
    question: str,
) -> Dict[str, Any]:
    """
    Streamlit-friendly wrapper.
    """

    if (
        not question
        or not question.strip()
    ):

        return {
            "success": False,
            "answer": (
                "Please enter a question."
            ),
            "reviews": (
                _empty_review_dataframe()
            ),
            "review_count": 0,
            "diagnostics": {},
        }

    try:

        _validate_qdrant_collection()

        question_embedding = (
            generate_question_embedding(
                question
            )
        )

        reviews, diagnostics = (
            progressive_retrieve(
                question,
                question_embedding,
            )
        )

        answer = (
            generate_rag_answer(
                question,
                reviews,
            )
        )

        display_columns = [
            column
            for column in REVIEW_COLUMNS
            if column in reviews.columns
        ]

        if reviews.empty:

            display_reviews = (
                _empty_review_dataframe()
            )

        else:

            display_reviews = (
                reviews[
                    display_columns
                ].copy()
            )

        return {
            "success": True,
            "question": question,
            "answer": answer,
            "reviews": display_reviews,
            "review_count": int(
                len(display_reviews)
            ),
            "diagnostics": diagnostics,
        }

    except Exception as exc:

        return {
            "success": False,
            "question": question,
            "answer": (
                "The review analysis could "
                "not be completed."
            ),
            "reviews": (
                _empty_review_dataframe()
            ),
            "review_count": 0,
            "diagnostics": {},
            "error": str(exc),
        }


# ============================================================
# RAG SYSTEM INFORMATION
# ============================================================

def get_rag_index_info() -> Dict[str, Any]:
    """
    Return information about the Qdrant-based RAG system.

    Function name preserved for compatibility.
    """

    _load_environment()

    _validate_qdrant_collection()

    client = (
        _get_qdrant_client()
    )

    info = client.get_collection(
        collection_name=QDRANT_COLLECTION
    )

    vectors = (
        info.config.params.vectors
    )

    return {
        "vector_store": (
            "Qdrant Cloud"
        ),

        "collection": (
            QDRANT_COLLECTION
        ),

        "embedding_model": (
            EMBEDDING_MODEL
        ),

        "embedding_dimension": (
            EMBEDDING_DIMENSION
        ),

        "distance": (
            "COSINE"
        ),

        "points_count": (
            int(
                info.points_count
            )
            if info.points_count
            is not None
            else None
        ),

        "indexed_vectors_count": (
            int(
                info.indexed_vectors_count
            )
            if info.indexed_vectors_count
            is not None
            else None
        ),

        "payload": (
            "none"
        ),

        "memory_mapped": (
            False
        ),

        "cached": (
            _QDRANT_CLIENT
            is not None
        ),

        "qdrant_vector_config_size": (
            int(
                vectors.size
            )
        ),

        "bigquery_reviews_table": (
            _get_reviews_table()
        ),

        "bigquery_enrichment_table": (
            _get_enriched_reviews_table()
        ),

        "candidate_levels": (
            CANDIDATE_LEVELS
        ),

        "final_k": (
            FINAL_K
        ),

        "minimum_relevance_score": (
            MIN_RELEVANCE_SCORE
        ),

        "score_window": (
            SCORE_WINDOW
        ),
    }


# ============================================================
# PRELOAD COMPATIBILITY
# ============================================================

def preload_rag_index() -> Dict[str, Any]:
    """
    Compatibility wrapper.

    There is no local embedding matrix to preload.
    """

    return (
        get_rag_index_info()
    )


# ============================================================
# COMMAND-LINE TEST
# ============================================================

def run_test() -> None:
    """
    Run a complete local RAG test.

    Command:

        python rag_engine.py
    """

    _print_separator()

    print(
        "ZOMATO AI - "
        "ADAPTIVE QDRANT + PROGRESSIVE + "
        "DIVERSE + EVIDENCE-AWARE "
        "RAG ENGINE TEST"
    )

    _print_separator()

    print()

    print(
        "Vector store:"
    )

    print(
        f"  {QDRANT_COLLECTION}"
    )

    print()

    print(
        "Embedding model:"
    )

    print(
        f"  {EMBEDDING_MODEL}"
    )

    print()

    print(
        "Embedding dimension:"
    )

    print(
        f"  {EMBEDDING_DIMENSION}"
    )

    print()

    print(
        "LLM model:"
    )

    print(
        f"  {LLM_MODEL}"
    )

    print()

    print(
        "BigQuery reviews:"
    )

    print(
        f"  {_get_reviews_table()}"
    )

    print()

    print(
        "BigQuery enrichment:"
    )

    print(
        f"  {_get_enriched_reviews_table()}"
    )

    print()

    print(
        "Retrieval configuration:"
    )

    print(
        f"  Candidate levels          : "
        f"{CANDIDATE_LEVELS}"
    )

    print(
        f"  Final K                   : "
        f"{FINAL_K}"
    )

    print(
        f"  Minimum relevance score   : "
        f"{MIN_RELEVANCE_SCORE}"
    )

    print(
        f"  Score window              : "
        f"{SCORE_WINDOW}"
    )

    print(
        f"  Text duplicate threshold  : "
        f"{TEXT_DUPLICATE_THRESHOLD}"
    )

    print(
        f"  Selected similarity max   : "
        f"{SELECTED_SIMILARITY_MAX}"
    )

    print(
        f"  Dominant pattern ratio    : "
        f"{DOMINANT_PATTERN_RATIO}"
    )

    print(
        f"  Max same pattern          : "
        f"{MAX_SAME_PATTERN}"
    )

    # --------------------------------------------------------
    # Qdrant validation
    # --------------------------------------------------------

    print()

    print(
        "Validating Qdrant collection..."
    )

    info = (
        get_rag_index_info()
    )

    print()

    _print_separator()

    print(
        "QDRANT RAG INFORMATION"
    )

    _print_separator()

    print(
        f"Collection          : "
        f"{info['collection']}"
    )

    print(
        f"Points count        : "
        f"{info['points_count']}"
    )

    print(
        f"Vector dimension    : "
        f"{info['embedding_dimension']}"
    )

    print(
        f"Distance            : "
        f"{info['distance']}"
    )

    print(
        f"Payload             : "
        f"{info['payload']}"
    )

    # --------------------------------------------------------
    # Test question
    # --------------------------------------------------------

    print()

    _print_separator()

    print(
        "TEST QUESTION"
    )

    _print_separator()

    question = (
        "What do customers say about food quality?"
    )

    print()

    print(
        f"Question: {question}"
    )

    # --------------------------------------------------------
    # Run RAG
    # --------------------------------------------------------

    answer, reviews = (
        answer_review_question(
            question
        )
    )

    # --------------------------------------------------------
    # Answer
    # --------------------------------------------------------

    print()

    _print_separator()

    print(
        "RAG ANSWER"
    )

    _print_separator()

    print(
        answer
    )

    # --------------------------------------------------------
    # Evidence
    # --------------------------------------------------------

    print()

    _print_separator()

    print(
        "FINAL RETRIEVED REVIEWS"
    )

    _print_separator()

    if reviews.empty:

        print(
            "No reviews retrieved."
        )

    else:

        display_columns = [
            column
            for column in REVIEW_COLUMNS
            if column in reviews.columns
        ]

        print(
            reviews[
                display_columns
            ].to_string(
                index=False
            )
        )

    print()

    _print_separator()

    print(
        "RAG ENGINE TEST COMPLETED"
    )

    _print_separator()


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    try:

        run_test()

    except KeyboardInterrupt:

        print()

        print(
            "RAG engine test interrupted."
        )

    except Exception as exc:

        print()

        print(
            "=" * 70
        )

        print(
            "RAG ENGINE TEST FAILED"
        )

        print(
            "=" * 70
        )

        print()

        print(
            f"{type(exc).__name__}: {exc}"
        )

        raise