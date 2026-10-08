"""
ZOMATO AI - QDRANT 300K REVIEW INGESTION
========================================

Purpose
-------
Resumable production ingestion of approximately 300,000 Zomato
reviews into Qdrant Cloud.

Architecture
------------
Local Parquet batches
        |
        v
Read review_id + comment
        |
        | old embedding column ignored
        v
Qdrant Cloud Inference
        |
        v
sentence-transformers/all-MiniLM-L6-v2
        |
        v
384-dimensional vector
        |
        v
Qdrant collection

Qdrant point:
    ID      = review_id
    Vector  = 384d
    Payload = none

Important
---------
- Does NOT use the old 1024-dimensional embeddings.
- Does NOT load ai/rag_index/embeddings.npy.
- Does NOT use pandas because the Parquet files contain
  metadata that previously caused pandas/dbdate issues.
- Uses PyArrow for Parquet reading.
- Uses Qdrant Cloud Inference.
- Supports resumable batch ingestion.
- Safe to rerun because Qdrant upsert is idempotent by point ID.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List

import pyarrow.parquet as pq
from dotenv import load_dotenv
from qdrant_client import QdrantClient, models


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent

# ------------------------------------------------------------
# Source data
# ------------------------------------------------------------

EMBEDDING_BATCH_DIR = (
    BASE_DIR / "review_embeddings"
)

# ------------------------------------------------------------
# Progress checkpoint
# ------------------------------------------------------------

PROGRESS_FILE = (
    BASE_DIR / "qdrant_ingestion_progress.json"
)

# ------------------------------------------------------------
# Qdrant
# ------------------------------------------------------------

COLLECTION_NAME = "zomato_reviews"

# ------------------------------------------------------------
# Embedding model
# ------------------------------------------------------------

EMBEDDING_MODEL = (
    "sentence-transformers/all-MiniLM-L6-v2"
)

EMBEDDING_DIMENSION = 384

# ------------------------------------------------------------
# Upload configuration
# ------------------------------------------------------------

UPLOAD_BATCH_SIZE = 256

# ------------------------------------------------------------
# Expected source batches
# ------------------------------------------------------------

EXPECTED_BATCH_COUNT = 60


# ============================================================
# ENVIRONMENT
# ============================================================

ENV_FILE = (
    PROJECT_DIR
    / "airflow"
    / ".env"
)

load_dotenv(
    ENV_FILE
)

QDRANT_URL = os.getenv(
    "QDRANT_URL"
)

QDRANT_API_KEY = os.getenv(
    "QDRANT_API_KEY"
)


# ============================================================
# VALIDATION
# ============================================================

def validate_environment() -> None:
    """
    Validate required configuration before ingestion.
    """

    print()
    print("=" * 70)
    print("VALIDATING ENVIRONMENT")
    print("=" * 70)

    if not QDRANT_URL:
        raise RuntimeError(
            "QDRANT_URL is missing from airflow/.env"
        )

    if not QDRANT_API_KEY:
        raise RuntimeError(
            "QDRANT_API_KEY is missing from airflow/.env"
        )

    if not EMBEDDING_BATCH_DIR.exists():
        raise FileNotFoundError(
            "Embedding batch directory does not exist:\n"
            f"{EMBEDDING_BATCH_DIR}"
        )

    print(
        "✓ QDRANT_URL found"
    )

    print(
        "✓ QDRANT_API_KEY found"
    )

    print(
        f"✓ Source directory found: "
        f"{EMBEDDING_BATCH_DIR}"
    )


# ============================================================
# QDRANT CONNECTION
# ============================================================

def create_qdrant_client() -> QdrantClient:
    """
    Create Qdrant Cloud client with Cloud Inference enabled.
    """

    print()
    print(
        "Connecting to Qdrant Cloud..."
    )

    client = QdrantClient(
        url=QDRANT_URL,
        api_key=QDRANT_API_KEY,
        cloud_inference=True,
    )

    # --------------------------------------------------------
    # Connection test
    # --------------------------------------------------------

    client.get_collections()

    print(
        "✓ Qdrant Cloud connection successful."
    )

    return client


# ============================================================
# COLLECTION
# ============================================================

def ensure_collection(
    client: QdrantClient,
) -> None:
    """
    Create the production collection if it does not exist.

    If it already exists, verify its vector configuration.
    """

    print()
    print("=" * 70)
    print("CHECKING QDRANT COLLECTION")
    print("=" * 70)

    collections = client.get_collections()

    existing_names = {
        collection.name
        for collection in collections.collections
    }

    # --------------------------------------------------------
    # Create collection
    # --------------------------------------------------------

    if COLLECTION_NAME not in existing_names:

        print(
            f"Collection '{COLLECTION_NAME}' "
            f"does not exist."
        )

        print(
            "Creating production collection..."
        )

        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=models.VectorParams(
                size=EMBEDDING_DIMENSION,
                distance=models.Distance.COSINE,
            ),
        )

        print(
            f"✓ Created collection: "
            f"{COLLECTION_NAME}"
        )

        return

    # --------------------------------------------------------
    # Existing collection
    # --------------------------------------------------------

    print(
        f"Collection already exists: "
        f"{COLLECTION_NAME}"
    )

    collection_info = client.get_collection(
        COLLECTION_NAME
    )

    vectors_config = (
        collection_info.config.params.vectors
    )

    # --------------------------------------------------------
    # Verify vector size
    # --------------------------------------------------------

    if isinstance(
        vectors_config,
        models.VectorParams,
    ):

        actual_dimension = (
            vectors_config.size
        )

        if (
            actual_dimension
            != EMBEDDING_DIMENSION
        ):
            raise RuntimeError(
                "Existing Qdrant collection has "
                "an unexpected vector dimension.\n"
                f"Expected: {EMBEDDING_DIMENSION}\n"
                f"Found:    {actual_dimension}"
            )

    print(
        f"✓ Vector dimension verified: "
        f"{EMBEDDING_DIMENSION}"
    )

    print(
        "✓ Existing production collection is usable."
    )


# ============================================================
# SOURCE FILE DISCOVERY
# ============================================================

def discover_batch_files() -> List[Path]:
    """
    Discover all review embedding batch files.

    The files are sorted numerically by filename.
    """

    files = sorted(
        EMBEDDING_BATCH_DIR.glob(
            "batch_*.parquet"
        )
    )

    if not files:
        raise FileNotFoundError(
            "No batch_*.parquet files found in:\n"
            f"{EMBEDDING_BATCH_DIR}"
        )

    print()
    print("=" * 70)
    print("SOURCE BATCH DISCOVERY")
    print("=" * 70)

    print(
        f"Found {len(files)} Parquet batch files."
    )

    if len(files) != EXPECTED_BATCH_COUNT:

        print(
            "WARNING:"
        )

        print(
            f"Expected approximately "
            f"{EXPECTED_BATCH_COUNT} batches, "
            f"but found {len(files)}."
        )

        print(
            "The script will use the files that "
            "actually exist."
        )

    for index, file_path in enumerate(
        files,
        start=1,
    ):

        print(
            f"[{index:02d}] "
            f"{file_path.name}"
        )

    return files


# ============================================================
# PROGRESS CHECKPOINT
# ============================================================

def load_progress() -> Dict[str, Any]:
    """
    Load resumable ingestion progress.

    If the progress file does not exist, start from zero.
    """

    if not PROGRESS_FILE.exists():

        return {
            "collection": COLLECTION_NAME,
            "embedding_model": EMBEDDING_MODEL,
            "embedding_dimension": (
                EMBEDDING_DIMENSION
            ),
            "completed_batches": [],
        }

    with open(
        PROGRESS_FILE,
        "r",
        encoding="utf-8",
    ) as file:

        progress = json.load(file)

    # --------------------------------------------------------
    # Basic validation
    # --------------------------------------------------------

    if (
        progress.get("collection")
        != COLLECTION_NAME
    ):
        raise RuntimeError(
            "Progress file belongs to another "
            "Qdrant collection."
        )

    if (
        progress.get("embedding_model")
        != EMBEDDING_MODEL
    ):
        raise RuntimeError(
            "Progress file belongs to another "
            "embedding model."
        )

    if (
        progress.get("embedding_dimension")
        != EMBEDDING_DIMENSION
    ):
        raise RuntimeError(
            "Progress file has an unexpected "
            "embedding dimension."
        )

    progress.setdefault(
        "completed_batches",
        [],
    )

    return progress


def save_progress(
    progress: Dict[str, Any],
) -> None:
    """
    Atomically save ingestion progress.

    A temporary file is written first so a partially written
    checkpoint is less likely if the process stops unexpectedly.
    """

    temporary_file = (
        PROGRESS_FILE.with_suffix(
            ".tmp"
        )
    )

    with open(
        temporary_file,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            progress,
            file,
            indent=2,
        )

    temporary_file.replace(
        PROGRESS_FILE
    )


# ============================================================
# PARQUET READING
# ============================================================

def read_review_batch(
    parquet_file: Path,
) -> List[Dict[str, Any]]:
    """
    Read review_id and comment from a Parquet batch.

    IMPORTANT:
    The old 'embedding' column is intentionally ignored.
    """

    print()
    print(
        f"Reading: {parquet_file.name}"
    )

    table = pq.read_table(
        parquet_file,
        columns=[
            "review_id",
            "comment",
        ],
    )

    # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------

    required_columns = {
        "review_id",
        "comment",
    }

    available_columns = set(
        table.column_names
    )

    missing = (
        required_columns
        - available_columns
    )

    if missing:

        raise RuntimeError(
            f"{parquet_file.name} is missing "
            f"required columns: {sorted(missing)}"
        )

    review_ids = (
        table["review_id"]
        .to_pylist()
    )

    comments = (
        table["comment"]
        .to_pylist()
    )

    if len(review_ids) != len(comments):

        raise RuntimeError(
            f"Column length mismatch in "
            f"{parquet_file.name}."
        )

    reviews = []

    seen_ids = set()

    for review_id, comment in zip(
        review_ids,
        comments,
    ):

        # ----------------------------------------------------
        # Review ID validation
        # ----------------------------------------------------

        if review_id is None:

            raise RuntimeError(
                f"Found NULL review_id in "
                f"{parquet_file.name}"
            )

        review_id = int(
            review_id
        )

        # ----------------------------------------------------
        # Duplicate ID protection
        # ----------------------------------------------------

        if review_id in seen_ids:

            raise RuntimeError(
                f"Duplicate review_id "
                f"{review_id} found inside "
                f"{parquet_file.name}"
            )

        seen_ids.add(
            review_id
        )

        # ----------------------------------------------------
        # Comment validation
        # ----------------------------------------------------

        if comment is None:

            raise RuntimeError(
                f"Review {review_id} has "
                f"a NULL comment in "
                f"{parquet_file.name}"
            )

        comment = str(
            comment
        ).strip()

        if not comment:

            raise RuntimeError(
                f"Review {review_id} has "
                f"an empty comment in "
                f"{parquet_file.name}"
            )

        reviews.append(
            {
                "review_id": review_id,
                "comment": comment,
            }
        )

    print(
        f"✓ Loaded {len(reviews):,} reviews."
    )

    print(
        "✓ Old embedding column was ignored."
    )

    return reviews


# ============================================================
# POINT CREATION
# ============================================================

def build_points(
    reviews: List[Dict[str, Any]],
) -> List[models.PointStruct]:
    """
    Build Qdrant points.

    Production point structure:

        ID      = review_id
        Vector  = Cloud Inference document
        Payload = none
    """

    points = []

    for review in reviews:

        point = models.PointStruct(
            id=review["review_id"],

            vector=models.Document(
                text=review["comment"],
                model=EMBEDDING_MODEL,
            ),

            # ------------------------------------------------
            # IMPORTANT:
            # No payload.
            # ------------------------------------------------
        )

        points.append(
            point
        )

    return points


# ============================================================
# BATCH INGESTION
# ============================================================

def ingest_batch(
    client: QdrantClient,
    parquet_file: Path,
) -> Dict[str, Any]:
    """
    Ingest one complete Parquet batch.

    Qdrant upload_points performs batched upserts.
    """

    start_time = time.perf_counter()

    reviews = read_review_batch(
        parquet_file
    )

    points = build_points(
        reviews
    )

    print(
        f"Uploading {len(points):,} "
        f"vectors to Qdrant..."
    )

    client.upload_points(
        collection_name=COLLECTION_NAME,
        points=points,
        batch_size=UPLOAD_BATCH_SIZE,
    )

    elapsed = (
        time.perf_counter()
        - start_time
    )

    rate = (
        len(points) / elapsed
        if elapsed > 0
        else 0
    )

    print(
        f"✓ Batch uploaded successfully."
    )

    print(
        f"  Reviews : {len(points):,}"
    )

    print(
        f"  Time    : {elapsed:.2f} sec"
    )

    print(
        f"  Rate    : {rate:.2f} reviews/sec"
    )

    return {
        "file": parquet_file.name,
        "reviews": len(points),
        "elapsed_seconds": round(
            elapsed,
            2,
        ),
        "reviews_per_second": round(
            rate,
            2,
        ),
    }


# ============================================================
# MAIN INGESTION
# ============================================================

def run_ingestion(
    dry_run: bool = False,
) -> None:
    """
    Run resumable ingestion across all Parquet batches.
    """

    overall_start = time.perf_counter()

    validate_environment()

    batch_files = discover_batch_files()

    progress = load_progress()

    completed_batches = set(
        progress.get(
            "completed_batches",
            [],
        )
    )

    print()
    print("=" * 70)
    print("INGESTION STATUS")
    print("=" * 70)

    print(
        f"Completed batches : "
        f"{len(completed_batches):,}"
    )

    print(
        f"Remaining batches : "
        f"{len(batch_files) - len(completed_batches):,}"
    )

    if dry_run:

        print()
        print(
            "DRY RUN MODE"
        )

        print(
            "No Qdrant writes will be performed."
        )

        for file_path in batch_files:

            status = (
                "COMPLETED"
                if file_path.name
                in completed_batches
                else "PENDING"
            )

            print(
                f"{status:10s} "
                f"{file_path.name}"
            )

        return

    # --------------------------------------------------------
    # Connect to Qdrant
    # --------------------------------------------------------

    client = create_qdrant_client()

    ensure_collection(
        client
    )

    # --------------------------------------------------------
    # Ingestion loop
    # --------------------------------------------------------

    successful_batches = 0
    skipped_batches = 0
    total_reviews_this_run = 0

    for batch_number, parquet_file in enumerate(
        batch_files,
        start=1,
    ):

        filename = parquet_file.name

        print()
        print("=" * 70)
        print(
            f"BATCH {batch_number}/{len(batch_files)}"
        )
        print("=" * 70)

        # ----------------------------------------------------
        # Skip completed batch
        # ----------------------------------------------------

        if filename in completed_batches:

            print(
                f"✓ Already completed: "
                f"{filename}"
            )

            skipped_batches += 1

            continue

        # ----------------------------------------------------
        # Ingest
        # ----------------------------------------------------

        try:

            result = ingest_batch(
                client=client,
                parquet_file=parquet_file,
            )

            # ------------------------------------------------
            # IMPORTANT:
            #
            # Save checkpoint ONLY after the entire batch
            # has successfully uploaded.
            # ------------------------------------------------

            progress[
                "completed_batches"
            ].append(
                filename
            )

            progress[
                "completed_batches"
            ] = sorted(
                set(
                    progress[
                        "completed_batches"
                    ]
                )
            )

            progress[
                "last_completed_batch"
            ] = filename

            progress[
                "last_batch_reviews"
            ] = result[
                "reviews"
            ]

            progress[
                "last_batch_time_seconds"
            ] = result[
                "elapsed_seconds"
            ]

            save_progress(
                progress
            )

            successful_batches += 1

            total_reviews_this_run += (
                result["reviews"]
            )

            print(
                f"✓ Checkpoint saved."
            )

        except Exception as exc:

            print()
            print("=" * 70)
            print(
                "❌ BATCH FAILED"
            )
            print("=" * 70)

            print(
                f"Batch : {filename}"
            )

            print(
                f"Error : "
                f"{type(exc).__name__}: {exc}"
            )

            print()
            print(
                "This batch was NOT marked "
                "as completed."
            )

            print(
                "You can safely rerun the script "
                "after fixing the problem."
            )

            raise

    # --------------------------------------------------------
    # Final status
    # --------------------------------------------------------

    elapsed = (
        time.perf_counter()
        - overall_start
    )

    print()
    print("=" * 70)
    print("QDRANT INGESTION COMPLETED")
    print("=" * 70)

    print(
        f"Successful batches : "
        f"{successful_batches:,}"
    )

    print(
        f"Skipped batches    : "
        f"{skipped_batches:,}"
    )

    print(
        f"Reviews uploaded this run: "
        f"{total_reviews_this_run:,}"
    )

    print(
        f"Total runtime      : "
        f"{elapsed:.2f} seconds"
    )

    print()
    print(
        f"Collection: {COLLECTION_NAME}"
    )

    print(
        "Embedding model:"
    )

    print(
        f"  {EMBEDDING_MODEL}"
    )

    print(
        f"Dimension: {EMBEDDING_DIMENSION}"
    )

    print(
        "Payload: none"
    )


# ============================================================
# COMMAND-LINE
# ============================================================

def main() -> None:

    parser = argparse.ArgumentParser(
        description=(
            "Resumable Zomato review ingestion "
            "into Qdrant Cloud."
        )
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Show batch status without "
            "writing anything to Qdrant."
        ),
    )

    args = parser.parse_args()

    run_ingestion(
        dry_run=args.dry_run
    )


if __name__ == "__main__":

    try:

        main()

    except KeyboardInterrupt:

        print()
        print(
            "Ingestion interrupted by user."
        )

    except Exception as exc:

        print()
        print("=" * 70)
        print("❌ INGESTION FAILED")
        print("=" * 70)

        print(
            f"{type(exc).__name__}: {exc}"
        )

        raise