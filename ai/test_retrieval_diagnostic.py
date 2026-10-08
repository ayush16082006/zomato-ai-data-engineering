"""
ZOMATO AI - RETRIEVAL QUALITY DIAGNOSTIC
========================================

Purpose
-------
This is a READ-ONLY diagnostic script.

It checks whether zero-evidence RAG questions are caused by:

1. insufficient topic coverage in the 300k reviews,
2. similarity scores being below the current 0.50 threshold, or
3. Qdrant returning semantically unrelated reviews.

IMPORTANT
---------
This script DOES NOT:

- modify Qdrant
- modify BigQuery
- modify rag_engine.py
- modify the 300k vectors
- change the relevance threshold
- upload or delete anything

It intentionally DOES NOT apply the 0.50 relevance threshold.

It retrieves the top 20 raw Qdrant results and then fetches
their actual review text from BigQuery.

Run from:
    C:\\Users\\shand\\OneDrive\\Desktop\\zomato_project\\ai

Command:
    python test_retrieval_diagnostic.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
from dotenv import load_dotenv
from google.cloud import bigquery
from qdrant_client import QdrantClient, models


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent

ENV_FILE = PROJECT_ROOT / "airflow" / ".env"

SERVICE_ACCOUNT_FILE = (
    PROJECT_ROOT
    / "airflow"
    / "credentials"
    / "service-account.json"
)

COLLECTION_NAME = "zomato_reviews"

EMBEDDING_MODEL = (
    "sentence-transformers/all-MiniLM-L6-v2"
)

# Number of raw Qdrant results to inspect.
TOP_K = 20

BIGQUERY_PROJECT = os.getenv(
    "GOOGLE_CLOUD_PROJECT",
    "zomato-ai-data-engineering",
)

BIGQUERY_LOCATION = "asia-south1"


# ============================================================
# TEST QUESTIONS
# ============================================================

TEST_QUESTIONS = [
    (
        "Low Ratings",
        "What do customers with low ratings complain about?",
    ),
    (
        "Customer Problems",
        "What are the most common problems mentioned in customer reviews?",
    ),
    (
        "Restaurant Service",
        "What do customers say about restaurant service?",
    ),
    (
        "Value for Money",
        "What do customers say about value for money?",
    ),
    (
        "Negative Experience",
        "What do customers say about negative experiences?",
    ),
]


# ============================================================
# PRINT HELPERS
# ============================================================

def print_separator(
    char: str = "=",
    width: int = 90,
) -> None:
    print(char * width)


def fail(message: str) -> None:

    print()
    print_separator()

    print("DIAGNOSTIC FAILED")

    print_separator()

    print(message)

    print()

    sys.exit(1)


# ============================================================
# LOAD ENVIRONMENT
# ============================================================

def load_environment() -> tuple[str, str]:

    print()
    print_separator()

    print("LOADING CONFIGURATION")

    print_separator()

    if not ENV_FILE.exists():

        fail(
            f"airflow/.env was not found:\n{ENV_FILE}"
        )

    load_dotenv(
        ENV_FILE,
        override=False,
    )

    qdrant_url = os.getenv(
        "QDRANT_URL"
    )

    qdrant_api_key = os.getenv(
        "QDRANT_API_KEY"
    )

    if not qdrant_url:

        fail(
            "QDRANT_URL is missing from airflow/.env"
        )

    if not qdrant_api_key:

        fail(
            "QDRANT_API_KEY is missing from airflow/.env"
        )

    global BIGQUERY_PROJECT

    BIGQUERY_PROJECT = os.getenv(
        "GOOGLE_CLOUD_PROJECT",
        BIGQUERY_PROJECT,
    )

    print(
        f"Environment file : {ENV_FILE}"
    )

    print(
        f"Qdrant collection: {COLLECTION_NAME}"
    )

    print(
        f"Embedding model  : {EMBEDDING_MODEL}"
    )

    print(
        f"Top results      : {TOP_K}"
    )

    print(
        f"BigQuery project : {BIGQUERY_PROJECT}"
    )

    return (
        qdrant_url,
        qdrant_api_key,
    )


# ============================================================
# CREATE CLIENTS
# ============================================================

def create_clients(
    qdrant_url: str,
    qdrant_api_key: str,
) -> tuple[
    QdrantClient,
    bigquery.Client,
]:

    print()
    print_separator()

    print("CONNECTING TO QDRANT + BIGQUERY")

    print_separator()

    # --------------------------------------------------------
    # Qdrant
    # --------------------------------------------------------

    print(
        "Connecting to Qdrant Cloud..."
    )

    qdrant = QdrantClient(
        url=qdrant_url,
        api_key=qdrant_api_key,
        cloud_inference=True,
    )

    print(
        "✓ Qdrant connection created."
    )

    # --------------------------------------------------------
    # BigQuery
    # --------------------------------------------------------

    print()

    print(
        "Connecting to BigQuery..."
    )

    if SERVICE_ACCOUNT_FILE.exists():

        print(
            f"Using service account: "
            f"{SERVICE_ACCOUNT_FILE}"
        )

        bq = (
            bigquery.Client
            .from_service_account_json(
                str(
                    SERVICE_ACCOUNT_FILE
                ),
                project=BIGQUERY_PROJECT,
            )
        )

    else:

        print(
            "Service-account file not found."
        )

        print(
            "Using Application Default Credentials."
        )

        bq = bigquery.Client(
            project=BIGQUERY_PROJECT,
        )

    print(
        "✓ BigQuery connection created."
    )

    return (
        qdrant,
        bq,
    )


# ============================================================
# VALIDATE QDRANT COLLECTION
# ============================================================

def validate_collection(
    qdrant: QdrantClient,
) -> None:

    print()
    print_separator()

    print(
        "VALIDATING QDRANT COLLECTION"
    )

    print_separator()

    try:

        info = qdrant.get_collection(
            COLLECTION_NAME
        )

    except Exception as exc:

        fail(
            "Could not access Qdrant collection "
            f"'{COLLECTION_NAME}'.\n\n"
            f"{type(exc).__name__}: {exc}"
        )

    vectors = (
        info.config.params.vectors
    )

    if isinstance(
        vectors,
        dict,
    ):

        vector_config = next(
            iter(
                vectors.values()
            )
        )

    else:

        vector_config = vectors

    dimension = getattr(
        vector_config,
        "size",
        None,
    )

    distance = getattr(
        vector_config,
        "distance",
        None,
    )

    points_count = getattr(
        info,
        "points_count",
        None,
    )

    print(
        f"Collection     : "
        f"{COLLECTION_NAME}"
    )

    print(
        f"Points count   : "
        f"{points_count}"
    )

    print(
        f"Vector size    : "
        f"{dimension}"
    )

    print(
        f"Distance       : "
        f"{distance}"
    )

    if dimension != 384:

        print(
            "⚠ Warning: expected "
            f"384-dimensional vectors, "
            f"found {dimension}."
        )

    print(
        "✓ Collection accessible."
    )


# ============================================================
# QDRANT SEARCH
# ============================================================

def search_qdrant(
    qdrant: QdrantClient,
    question: str,
) -> List[Dict[str, Any]]:

    response = qdrant.query_points(
        collection_name=COLLECTION_NAME,

        query=models.Document(
            text=question,
            model=EMBEDDING_MODEL,
        ),

        limit=TOP_K,

        # Final architecture has no payload.
        with_payload=False,

        # We only need IDs and scores.
        with_vectors=False,
    )

    points = response.points

    results = []

    for point in points:

        results.append(
            {
                "review_id": str(
                    point.id
                ),
                "score": float(
                    point.score
                ),
            }
        )

    return results


# ============================================================
# BIGQUERY LOOKUP
# ============================================================

def fetch_bigquery_reviews(
    bq: bigquery.Client,
    review_ids: List[str],
) -> pd.DataFrame:

    if not review_ids:

        return pd.DataFrame(
            columns=[
                "review_id",
                "rating",
                "comment",
                "review_date",
            ]
        )

    query = f"""
    SELECT
        CAST(review_id AS STRING)
            AS review_id,

        rating,

        comment,

        review_date

    FROM `{BIGQUERY_PROJECT}.raw.reviews`

    WHERE CAST(review_id AS STRING)
          IN UNNEST(@review_ids)
    """

    job_config = (
        bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ArrayQueryParameter(
                    "review_ids",
                    "STRING",
                    review_ids,
                )
            ]
        )
    )

    result = bq.query(
        query,
        job_config=job_config,
        location=BIGQUERY_LOCATION,
    ).result()

    rows = [
        dict(row)
        for row in result
    ]

    if not rows:

        return pd.DataFrame(
            columns=[
                "review_id",
                "rating",
                "comment",
                "review_date",
            ]
        )

    return pd.DataFrame(
        rows
    )


# ============================================================
# COMBINE QDRANT + BIGQUERY
# ============================================================

def combine_results(
    qdrant_results: List[
        Dict[str, Any]
    ],
    bq_reviews: pd.DataFrame,
) -> pd.DataFrame:

    qdrant_df = pd.DataFrame(
        qdrant_results
    )

    # --------------------------------------------------------
    # No Qdrant results
    # --------------------------------------------------------

    if qdrant_df.empty:

        return pd.DataFrame(
            columns=[
                "rank",
                "review_id",
                "score",
                "rating",
                "comment",
                "review_date",
            ]
        )

    # --------------------------------------------------------
    # No BigQuery rows
    # --------------------------------------------------------

    if bq_reviews.empty:

        qdrant_df.insert(
            0,
            "rank",
            range(
                1,
                len(qdrant_df) + 1,
            ),
        )

        qdrant_df[
            "rating"
        ] = None

        qdrant_df[
            "comment"
        ] = None

        qdrant_df[
            "review_date"
        ] = None

        return qdrant_df[
            [
                "rank",
                "review_id",
                "score",
                "rating",
                "comment",
                "review_date",
            ]
        ]

    # --------------------------------------------------------
    # Map Qdrant IDs → BigQuery
    # --------------------------------------------------------

    merged = qdrant_df.merge(
        bq_reviews,
        on="review_id",
        how="left",
    )

    merged.insert(
        0,
        "rank",
        range(
            1,
            len(merged) + 1,
        ),
    )

    return merged[
        [
            "rank",
            "review_id",
            "score",
            "rating",
            "comment",
            "review_date",
        ]
    ]


# ============================================================
# ANALYZE RESULTS
# ============================================================

def analyze_results(
    results: pd.DataFrame,
) -> Dict[str, Any]:

    if results.empty:

        return {
            "count": 0,
            "max_score": None,
            "min_score": None,
            "avg_score": None,
            "mapped_count": 0,
            "unique_comments": 0,
        }

    scores = pd.to_numeric(
        results["score"],
        errors="coerce",
    )

    comments = (
        results["comment"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    mapped_count = int(
        comments.ne("").sum()
    )

    unique_comments = int(
        comments[
            comments.ne("")
        ].nunique()
    )

    return {
        "count": len(results),

        "max_score": float(
            scores.max()
        ),

        "min_score": float(
            scores.min()
        ),

        "avg_score": float(
            scores.mean()
        ),

        "mapped_count": mapped_count,

        "unique_comments": unique_comments,
    }


# ============================================================
# PRINT RESULTS
# ============================================================

def print_results(
    category: str,
    question: str,
    results: pd.DataFrame,
) -> None:

    print()
    print_separator()

    print(
        "RETRIEVAL DIAGNOSTIC"
    )

    print_separator()

    print(
        f"Category : {category}"
    )

    print(
        f"Question : {question}"
    )

    print()

    print(
        "IMPORTANT:"
    )

    print(
        "0.50 relevance threshold is NOT applied."
    )

    if results.empty:

        print()

        print(
            "No Qdrant results were returned."
        )

        return

    stats = analyze_results(
        results
    )

    print()

    print(
        f"Top results         : "
        f"{stats['count']}"
    )

    print(
        f"Highest score       : "
        f"{stats['max_score']:.4f}"
    )

    print(
        f"Lowest score        : "
        f"{stats['min_score']:.4f}"
    )

    print(
        f"Average score       : "
        f"{stats['avg_score']:.4f}"
    )

    print(
        f"BigQuery IDs mapped : "
        f"{stats['mapped_count']}/"
        f"{stats['count']}"
    )

    print(
        f"Unique comments     : "
        f"{stats['unique_comments']}"
    )

    print()

    print(
        "-" * 90
    )

    for _, row in results.iterrows():

        if pd.notna(
            row["comment"]
        ):

            comment = str(
                row["comment"]
            )

        else:

            comment = (
                "[NO BIGQUERY COMMENT]"
            )

        if pd.notna(
            row["rating"]
        ):

            rating = row["rating"]

        else:

            rating = "N/A"

        if pd.notna(
            row["review_date"]
        ):

            review_date = (
                row["review_date"]
            )

        else:

            review_date = "N/A"

        print(
            f"#{int(row['rank']):02d} "
            f"ID={row['review_id']} "
            f"SCORE={float(row['score']):.4f} "
            f"RATING={rating} "
            f"DATE={review_date}"
        )

        print(
            f"    {comment}"
        )

        print(
            "-" * 90
        )


# ============================================================
# INITIAL INTERPRETATION
# ============================================================

def print_interpretation(
    results: pd.DataFrame,
) -> None:

    if results.empty:

        print()

        print(
            "Interpretation:"
        )

        print(
            "Qdrant returned no results. "
            "This would be unusual and should "
            "be investigated."
        )

        return

    stats = analyze_results(
        results
    )

    max_score = stats[
        "max_score"
    ]

    unique_comments = stats[
        "unique_comments"
    ]

    print()

    print_separator(
        "-"
    )

    print(
        "INITIAL INTERPRETATION"
    )

    print_separator(
        "-"
    )

    # --------------------------------------------------------
    # Score interpretation
    # --------------------------------------------------------

    if max_score < 0.40:

        print(
            "→ Best semantic score is below 0.40."
        )

        print(
            "→ The query has weak semantic similarity "
            "to the indexed reviews."
        )

        print(
            "→ This may indicate insufficient topic "
            "coverage OR a query/embedding mismatch."
        )

    elif max_score < 0.50:

        print(
            "→ Relevant-looking candidates may exist "
            "below the current 0.50 threshold."
        )

        print(
            "→ The threshold may be too strict for "
            "this particular query."
        )

        print(
            "→ We should inspect the actual comments "
            "before changing the threshold."
        )

    else:

        print(
            "→ At least one candidate has a score "
            ">= 0.50."
        )

        print(
            "→ If those candidates are unrelated to "
            "the question, the issue is retrieval "
            "semantics rather than simply data size."
        )

    # --------------------------------------------------------
    # Duplicate interpretation
    # --------------------------------------------------------

    if unique_comments == 1:

        print(
            "→ All mapped results use the same "
            "review template."
        )

        print(
            "→ This confirms the duplicate-heavy "
            "dataset behavior for this query."
        )

    elif unique_comments > 1:

        print(
            f"→ Top results contain "
            f"{unique_comments} different comments."
        )

        print(
            "→ There is at least some diversity "
            "in the retrieved evidence."
        )


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    print()

    print_separator()

    print(
        "ZOMATO AI - RETRIEVAL QUALITY DIAGNOSTIC"
    )

    print_separator()

    print()

    print(
        "READ-ONLY DIAGNOSTIC"
    )

    print(
        "No Qdrant, BigQuery, or RAG data will be modified."
    )

    print(
        "The 0.50 relevance threshold is intentionally ignored."
    )

    # --------------------------------------------------------
    # Configuration
    # --------------------------------------------------------

    (
        qdrant_url,
        qdrant_api_key,
    ) = load_environment()

    # --------------------------------------------------------
    # Clients
    # --------------------------------------------------------

    (
        qdrant,
        bq,
    ) = create_clients(
        qdrant_url,
        qdrant_api_key,
    )

    # --------------------------------------------------------
    # Collection validation
    # --------------------------------------------------------

    validate_collection(
        qdrant
    )

    # --------------------------------------------------------
    # Questions
    # --------------------------------------------------------

    for number, (
        category,
        question,
    ) in enumerate(
        TEST_QUESTIONS,
        start=1,
    ):

        print()
        print()

        print_separator(
            "#"
        )

        print(
            f"DIAGNOSTIC QUESTION "
            f"{number}/"
            f"{len(TEST_QUESTIONS)}"
        )

        print_separator(
            "#"
        )

        print()

        print(
            f"Category: {category}"
        )

        print(
            f"Question: {question}"
        )

        try:

            # ------------------------------------------------
            # Raw Qdrant retrieval
            # ------------------------------------------------

            print()

            print(
                f"Searching Qdrant top "
                f"{TOP_K} WITHOUT relevance filtering..."
            )

            qdrant_results = (
                search_qdrant(
                    qdrant,
                    question,
                )
            )

            print(
                f"✓ Qdrant returned "
                f"{len(qdrant_results)} results."
            )

            # ------------------------------------------------
            # Extract IDs
            # ------------------------------------------------

            review_ids = [
                item["review_id"]
                for item in qdrant_results
            ]

            # ------------------------------------------------
            # BigQuery lookup
            # ------------------------------------------------

            print()

            print(
                "Fetching those review IDs "
                "from BigQuery..."
            )

            bq_reviews = (
                fetch_bigquery_reviews(
                    bq,
                    review_ids,
                )
            )

            print(
                f"✓ BigQuery returned "
                f"{len(bq_reviews)} rows."
            )

            # ------------------------------------------------
            # Combine
            # ------------------------------------------------

            results = combine_results(
                qdrant_results,
                bq_reviews,
            )

            # ------------------------------------------------
            # Print
            # ------------------------------------------------

            print_results(
                category,
                question,
                results,
            )

            # ------------------------------------------------
            # Interpret
            # ------------------------------------------------

            print_interpretation(
                results
            )

        except Exception as exc:

            print()

            print(
                "❌ Diagnostic failed "
                "for this question."
            )

            print(
                f"{type(exc).__name__}: "
                f"{exc}"
            )

    # --------------------------------------------------------
    # Complete
    # --------------------------------------------------------

    print()

    print_separator()

    print(
        "DIAGNOSTIC COMPLETED"
    )

    print_separator()

    print()

    print(
        "Next step:"
    )

    print(
        "Inspect the similarity scores and actual "
        "review comments."
    )

    print()

    print(
        "DO NOT change RELEVANCE_THRESHOLD yet."
    )

    print(
        "DO NOT rebuild the Qdrant collection yet."
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()