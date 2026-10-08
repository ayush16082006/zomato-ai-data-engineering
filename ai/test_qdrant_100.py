"""
ZOMATO AI
STEP 4 - 100 REVIEW QDRANT CLOUD INFERENCE TEST

Tests:

    Parquet
       ↓
    100 reviews
       ↓
    Qdrant Cloud Inference
       ↓
    all-MiniLM-L6-v2
       ↓
    384-dimensional embeddings
       ↓
    Qdrant collection
       ↓
    Vector search verification
"""

from pathlib import Path
import os

import time

start_time = time.perf_counter()

import pyarrow.parquet as pq
from dotenv import load_dotenv

from qdrant_client import QdrantClient, models


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

ENV_FILE = BASE_DIR / "airflow" / ".env"

PARQUET_FILES = [
    BASE_DIR
    / "ai"
    / "review_embeddings"
    / "batch_000001.parquet",

    BASE_DIR
    / "ai"
    / "review_embeddings"
    / "batch_000002.parquet",
]

COLLECTION_NAME = "zomato_reviews_10000_test"

MODEL_NAME = (
    "sentence-transformers/all-MiniLM-L6-v2"
)

EXPECTED_DIMENSION = 384

TEST_ROWS = 10000


# ============================================================
# LOAD ENVIRONMENT
# ============================================================

print()
print("=" * 70)
print("ZOMATO AI - QDRANT 100 REVIEW TEST")
print("=" * 70)

print()
print("Loading environment:")
print(ENV_FILE)

if not ENV_FILE.exists():
    raise FileNotFoundError(
        f"Environment file not found:\n{ENV_FILE}"
    )

load_dotenv(ENV_FILE)


QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")


if not QDRANT_URL:
    raise RuntimeError(
        "QDRANT_URL is missing from airflow/.env"
    )

if not QDRANT_API_KEY:
    raise RuntimeError(
        "QDRANT_API_KEY is missing from airflow/.env"
    )


print("QDRANT_URL found      : YES")
print("QDRANT_API_KEY found  : YES")


# ============================================================
# CONNECT TO QDRANT
# ============================================================

print()
print("=" * 70)
print("CONNECTING TO QDRANT CLOUD")
print("=" * 70)

client = QdrantClient(
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY,
    cloud_inference=True,
)

print("Qdrant connection      : OK")
print("Cloud inference        : ENABLED")
print(f"Embedding model        : {MODEL_NAME}")


# ============================================================
# LOAD 100 REVIEWS
# ============================================================

print()
print("=" * 70)
print("LOADING 100 REVIEWS")
print("=" * 70)

print()
print("Source files:")

for parquet_file in PARQUET_FILES:
    print(parquet_file)

    if not parquet_file.exists():
        raise FileNotFoundError(
            f"Parquet file not found:\n{parquet_file}"
        )


rows = []

for parquet_file in PARQUET_FILES:

    table = pq.read_table(
        parquet_file,
        columns=[
            "review_id",
            "rating",
            "comment",
        ],
    )

    batch_rows = table.to_pylist()

    rows.extend(batch_rows)

    print(
        f"Loaded {len(batch_rows):,} rows "
        f"from {parquet_file.name}"
    )


# Keep exactly the requested number of reviews.
rows = rows[:TEST_ROWS]


if len(rows) != TEST_ROWS:
    raise RuntimeError(
        f"Expected {TEST_ROWS} reviews, "
        f"but loaded {len(rows)}."
    )


print()
print(f"Rows selected       : {len(rows):,}")
print(f"Expected rows       : {TEST_ROWS:,}")
print(f"Columns             : {table.column_names}")


# ============================================================
# VALIDATE REVIEW DATA
# ============================================================

for row in rows:

    if row["review_id"] is None:
        raise ValueError(
            "A review has no review_id."
        )

    if row["comment"] is None:
        raise ValueError(
            f"Review {row['review_id']} has no comment."
        )


if len(rows) != TEST_ROWS:
    raise RuntimeError(
        f"Expected {TEST_ROWS:,} reviews, "
        f"but loaded {len(rows):,}."
    )

print("Review validation   : PASSED")


# ============================================================
# CREATE TEST COLLECTION
# ============================================================

print()
print("=" * 70)
print("CREATING TEST COLLECTION")
print("=" * 70)

print()
print(f"Collection : {COLLECTION_NAME}")
print(f"Vector dim : {EXPECTED_DIMENSION}")
print("Distance   : COSINE")


# Remove an old test collection if it exists.
#
# This is safe because this is ONLY our temporary
# 100-review test collection.

existing_collections = client.get_collections()

existing_names = {
    collection.name
    for collection in existing_collections.collections
}


if COLLECTION_NAME in existing_names:

    print()
    print(
        "Existing test collection found."
    )

    print(
        "Deleting old test collection..."
    )

    client.delete_collection(
        collection_name=COLLECTION_NAME
    )

    print("Old test collection deleted.")


client.create_collection(
    collection_name=COLLECTION_NAME,
    vectors_config=models.VectorParams(
        size=EXPECTED_DIMENSION,
        distance=models.Distance.COSINE,
    ),
)

print()
print("Collection created: YES")


# ============================================================
# PREPARE QDRANT POINTS
# ============================================================

print()
print("=" * 70)
print("PREPARING 100 REVIEWS")
print("=" * 70)

points = []


for row in rows:

    review_id = int(
        row["review_id"]
    )

    comment = str(
        row["comment"]
    ).strip()

    rating = (
        int(row["rating"])
        if row["rating"] is not None
        else None
    )

    points.append(
        models.PointStruct(
            id=review_id,

            # IMPORTANT:
            #
            # Qdrant Cloud Inference generates
            # the embedding from the text.
            #
            # We DO NOT use the old 1024-dimensional
            # embedding stored in the parquet file.
            vector=models.Document(
                text=comment,
                model=MODEL_NAME,
            ),

            payload={
                "review_id": review_id,
                "rating": rating,
            },
        )
    )


print()
print(f"Points prepared: {len(points)}")


# ============================================================
# UPSERT
# ============================================================

print()
print("=" * 70)
print("SENDING REVIEWS TO QDRANT CLOUD INFERENCE")
print("=" * 70)

print()
print("Generating embeddings...")
print("Please wait...")


client.upsert(
    collection_name=COLLECTION_NAME,
    points=points,
    wait=True,
)


print()
print("Embedding + upload: SUCCESS")


# ============================================================
# VERIFY COLLECTION
# ============================================================

print()
print("=" * 70)
print("VERIFYING QDRANT COLLECTION")
print("=" * 70)

collection_info = client.get_collection(
    collection_name=COLLECTION_NAME
)


print()
print(
    f"Points stored: "
    f"{collection_info.points_count}"
)


if collection_info.points_count != TEST_ROWS:

    raise RuntimeError(
        "Qdrant point count does not match "
        f"expected {TEST_ROWS}."
    )


print("Point count verification: PASSED")


# ============================================================
# VERIFY VECTOR DIMENSION
# ============================================================

print()
print("=" * 70)
print("VERIFYING VECTOR DIMENSION")
print("=" * 70)


# Retrieve one point with its vector.

sample_points = client.retrieve(
    collection_name=COLLECTION_NAME,
    ids=[
        int(rows[0]["review_id"])
    ],
    with_vectors=True,
)


if not sample_points:

    raise RuntimeError(
        "Could not retrieve the inserted test point."
    )


sample_vector = sample_points[0].vector


if sample_vector is None:

    raise RuntimeError(
        "Retrieved point does not contain a vector."
    )


# Depending on the Qdrant client response format,
# a named vector can be returned as a dictionary.
if isinstance(sample_vector, dict):

    vector_values = next(
        iter(
            sample_vector.values()
        )
    )

else:

    vector_values = sample_vector


actual_dimension = len(
    vector_values
)


print()
print(
    f"Expected dimension : "
    f"{EXPECTED_DIMENSION}"
)

print(
    f"Actual dimension   : "
    f"{actual_dimension}"
)


if actual_dimension != EXPECTED_DIMENSION:

    raise RuntimeError(
        "Unexpected embedding dimension."
    )


print()
print(
    "Vector dimension verification: PASSED"
)


# ============================================================
# VECTOR SEARCH TEST
# ============================================================

print()
print("=" * 70)
print("TESTING VECTOR SEARCH")
print("=" * 70)


search_text = (
    "food quality and taste"
)


print()
print(f"Search query: {search_text}")


search_result = client.query_points(
    collection_name=COLLECTION_NAME,

    query=models.Document(
        text=search_text,
        model=MODEL_NAME,
    ),

    limit=5,

    with_payload=True,
)


print()
print(
    f"Search results returned: "
    f"{len(search_result.points)}"
)


if not search_result.points:

    raise RuntimeError(
        "Vector search returned no results."
    )


print()
print("-" * 70)
print("TOP SEARCH RESULTS")
print("-" * 70)


for rank, point in enumerate(
    search_result.points,
    start=1,
):

    payload = point.payload or {}

    print()
    print(
        f"{rank}. "
        f"Review ID: "
        f"{payload.get('review_id')}"
    )

    print(
        f"   Rating: "
        f"{payload.get('rating')}"
    )

    print(
        f"   Score: "
        f"{point.score}"
    )


# ============================================================
# FINAL RESULT
# ============================================================

print()
print("=" * 70)
print("STEP 4 - 100 REVIEW TEST COMPLETED")
print("=" * 70)

print()
print("✓ Qdrant Cloud connection        : PASS")
print("✓ Qdrant Cloud Inference         : PASS")
print("✓ 100 reviews loaded             : PASS")
print("✓ Old 1024d embeddings ignored   : PASS")
print("✓ all-MiniLM-L6-v2 used          : PASS")
print("✓ 384-dimensional vectors        : PASS")
print("✓ 100 vectors stored             : PASS")
print("✓ Vector search                  : PASS")

print()
print(
    "Step 4.1 — 100-review embedding test "
    "is COMPLETE."
)

print()

end_time = time.perf_counter()

elapsed_time = end_time - start_time

print()
print("=" * 70)
print(f"TOTAL TIME TAKEN: {elapsed_time:.2f} seconds")
print(f"TOTAL TIME TAKEN: {elapsed_time / 60:.2f} minutes")
print("=" * 70)