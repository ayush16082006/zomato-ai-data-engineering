import os
from pathlib import Path

from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client import models
from google.cloud import bigquery


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

ENV_FILE = PROJECT_ROOT / "airflow" / ".env"

load_dotenv(ENV_FILE)


QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

PROJECT_ID = os.getenv(
    "GOOGLE_CLOUD_PROJECT",
    "zomato-ai-data-engineering",
)

BQ_LOCATION = "asia-south1"

COLLECTION_NAME = "zomato_reviews"

EMBEDDING_MODEL = (
    "sentence-transformers/all-MiniLM-L6-v2"
)


# ============================================================
# VALIDATE ENV
# ============================================================

if not QDRANT_URL:
    raise RuntimeError(
        "QDRANT_URL is not set."
    )

if not QDRANT_API_KEY:
    raise RuntimeError(
        "QDRANT_API_KEY is not set."
    )


# ============================================================
# CLIENTS
# ============================================================

print()
print("=" * 70)
print("QDRANT → BIGQUERY MAPPING TEST")
print("=" * 70)

print()
print("Connecting to Qdrant...")

qdrant = QdrantClient(
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY,
    cloud_inference=True,
)

print("✓ Qdrant connected.")


print()
print("Connecting to BigQuery...")

credential_file = (
    PROJECT_ROOT
    / "airflow"
    / "credentials"
    / "service-account.json"
)

if credential_file.exists():

    bq = bigquery.Client.from_service_account_json(
        str(credential_file),
        project=PROJECT_ID,
    )

else:

    bq = bigquery.Client(
        project=PROJECT_ID,
    )

print("✓ BigQuery connected.")


# ============================================================
# TEST QUESTION
# ============================================================

QUESTION = (
    "What do customers say about food quality?"
)


# ============================================================
# QDRANT SEARCH
# ============================================================

print()
print("=" * 70)
print("QDRANT SEARCH")
print("=" * 70)

print()
print(f"Question: {QUESTION}")

response = qdrant.query_points(
    collection_name=COLLECTION_NAME,
    query=models.Document(
        text=QUESTION,
        model=EMBEDDING_MODEL,
    ),
    limit=20,
    with_payload=False,
)

points = response.points

print()
print(f"Qdrant results: {len(points)}")


# ============================================================
# DISPLAY QDRANT IDs
# ============================================================

review_ids = []

print()
print("Qdrant review IDs:")

for rank, point in enumerate(
    points,
    start=1,
):

    review_id = str(point.id)

    review_ids.append(
        review_id
    )

    print(
        f"{rank:02d}. "
        f"ID={review_id} "
        f"score={point.score:.6f}"
    )


# ============================================================
# BIGQUERY LOOKUP
# ============================================================

print()
print("=" * 70)
print("BIGQUERY LOOKUP")
print("=" * 70)


query = f"""
SELECT
    CAST(review_id AS STRING) AS review_id,
    rating,
    comment,
    review_date
FROM `{PROJECT_ID}.raw.reviews`
WHERE CAST(review_id AS STRING)
      IN UNNEST(@review_ids)
ORDER BY review_id
"""


job_config = bigquery.QueryJobConfig(
    query_parameters=[
        bigquery.ArrayQueryParameter(
            "review_ids",
            "STRING",
            review_ids,
        )
    ]
)


rows = bq.query(
    query,
    job_config=job_config,
    location=BQ_LOCATION,
).result()


results = []

for row in rows:

    results.append(
        {
            "review_id": str(
                row.review_id
            ),
            "rating": row.rating,
            "comment": row.comment,
            "review_date": row.review_date,
        }
    )


# ============================================================
# RESULTS
# ============================================================

print()
print(
    f"BigQuery rows returned: "
    f"{len(results)}"
)

print()
print("=" * 70)
print("QDRANT ID → BIGQUERY REVIEW")
print("=" * 70)


result_by_id = {
    item["review_id"]: item
    for item in results
}


for rank, review_id in enumerate(
    review_ids,
    start=1,
):

    item = result_by_id.get(
        review_id
    )

    print()

    print(
        f"Rank {rank}"
    )

    print(
        f"Review ID : {review_id}"
    )

    if item is None:

        print(
            "❌ NOT FOUND IN BIGQUERY"
        )

        continue

    print(
        f"Rating    : {item['rating']}"
    )

    print(
        f"Date      : {item['review_date']}"
    )

    print(
        f"Comment   : {item['comment']}"
    )


# ============================================================
# DUPLICATE ANALYSIS
# ============================================================

print()
print("=" * 70)
print("DUPLICATE ANALYSIS")
print("=" * 70)


comments = [
    str(item["comment"]).strip()
    for item in results
    if item["comment"] is not None
]


unique_comments = set(
    comments
)


print()
print(
    f"Total BigQuery reviews : "
    f"{len(results)}"
)

print(
    f"Unique comments        : "
    f"{len(unique_comments)}"
)

print(
    f"Duplicate comments     : "
    f"{len(results) - len(unique_comments)}"
)


# ============================================================
# COMPLETION
# ============================================================

print()
print("=" * 70)
print("MAPPING TEST COMPLETED")
print("=" * 70)