"""
ZOMATO AI - PERSISTENT RAG INDEX BUILDER

Builds a persistent RAG index from all review embedding parquet files.

Source:
    review_embeddings/
        batch_000001.parquet
        ...
        batch_000060.parquet

Output:
    rag_index/
        embeddings.npy
        reviews.parquet
        metadata.json

The builder intentionally reloads and rebuilds the complete index.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

SOURCE_DIR = BASE_DIR / "review_embeddings"
OUTPUT_DIR = BASE_DIR / "rag_index"

EMBEDDINGS_FILE = OUTPUT_DIR / "embeddings.npy"
REVIEWS_FILE = OUTPUT_DIR / "reviews.parquet"
METADATA_FILE = OUTPUT_DIR / "metadata.json"

EXPECTED_EMBEDDING_DIMENSION = 1024


# ============================================================
# HELPERS
# ============================================================

def print_header(title: str) -> None:
    print()
    print("=" * 70)
    print(title)
    print("=" * 70)


def fail(message: str) -> None:
    print()
    print("=" * 70)
    print("❌ INDEX BUILD FAILED")
    print("=" * 70)
    print()
    print(message)
    print()
    sys.exit(1)


# ============================================================
# FIND SOURCE FILES
# ============================================================

def find_embedding_files() -> list[Path]:

    if not SOURCE_DIR.exists():
        fail(
            f"Source directory does not exist:\n"
            f"{SOURCE_DIR}"
        )

    files = sorted(
        SOURCE_DIR.glob("batch_*.parquet")
    )

    if not files:
        fail(
            f"No parquet embedding files found in:\n"
            f"{SOURCE_DIR}"
        )

    return files


# ============================================================
# LOAD ONE PARQUET FILE
# ============================================================

def load_parquet_file(
    file_path: Path,
) -> tuple[pd.DataFrame, np.ndarray]:
    """
    Load one parquet file using PyArrow directly.

    This avoids pandas/pyarrow dtype conversion problems with
    the review_date date32[day] column.
    """

    try:
        table = pq.read_table(
            file_path
        )
    except Exception as exc:
        raise RuntimeError(
            f"Could not read parquet file:\n"
            f"{file_path}\n\n"
            f"Error: {exc}"
        ) from exc

    column_names = table.column_names

    # --------------------------------------------------------
    # Find embedding column
    # --------------------------------------------------------

    embedding_column = None

    for column in [
        "embedding",
        "embeddings",
        "vector",
    ]:
        if column in column_names:
            embedding_column = column
            break

    if embedding_column is None:
        raise RuntimeError(
            f"No embedding column found in:\n"
            f"{file_path}\n\n"
            f"Available columns:\n"
            f"{column_names}"
        )

    # --------------------------------------------------------
    # Extract embedding column using Arrow
    # --------------------------------------------------------

    embedding_array = table[
        embedding_column
    ]

    try:
        embeddings = np.asarray(
            embedding_array.to_pylist(),
            dtype=np.float32,
        )
    except Exception as exc:
        raise RuntimeError(
            f"Could not convert embeddings in:\n"
            f"{file_path}\n\n"
            f"Error: {exc}"
        ) from exc

    if embeddings.ndim != 2:
        raise RuntimeError(
            f"Invalid embedding shape in:\n"
            f"{file_path}\n\n"
            f"Shape: {embeddings.shape}"
        )

    if embeddings.shape[1] != EXPECTED_EMBEDDING_DIMENSION:
        raise RuntimeError(
            f"Unexpected embedding dimension in:\n"
            f"{file_path}\n\n"
            f"Expected: "
            f"{EXPECTED_EMBEDDING_DIMENSION}\n"
            f"Found: {embeddings.shape[1]}"
        )

    # --------------------------------------------------------
    # Convert metadata separately
    # --------------------------------------------------------

    metadata_columns = [
        column
        for column in column_names
        if column != embedding_column
    ]

    metadata_table = table.select(
        metadata_columns
    )

    # Convert Arrow metadata to Python objects first.
    #
    # This prevents pandas from trying to interpret the
    # date32[day] column through the problematic dtype path.
    metadata_rows = metadata_table.to_pylist()

    metadata = pd.DataFrame(
        metadata_rows
    )

    # --------------------------------------------------------
    # Explicitly normalize review_date
    # --------------------------------------------------------

    if "review_date" in metadata.columns:

        metadata["review_date"] = pd.to_datetime(
            metadata["review_date"],
            errors="coerce",
        ).dt.date

    # --------------------------------------------------------
    # Validate alignment
    # --------------------------------------------------------

    metadata.reset_index(
        drop=True,
        inplace=True,
    )

    if len(metadata) != len(embeddings):
        raise RuntimeError(
            f"Metadata/embedding mismatch in:\n"
            f"{file_path}\n\n"
            f"Metadata rows: {len(metadata)}\n"
            f"Embedding rows: {len(embeddings)}"
        )

    return metadata, embeddings


# ============================================================
# LOAD ALL PARQUET FILES
# ============================================================

def load_all_batches(
    files: list[Path],
) -> tuple[pd.DataFrame, np.ndarray]:

    print_header(
        "LOADING REVIEW EMBEDDINGS"
    )

    print(
        f"Source directory : {SOURCE_DIR}"
    )

    print(
        f"Output directory : {OUTPUT_DIR}"
    )

    print()
    print(
        f"Embedding files found: {len(files)}"
    )

    metadata_frames = []
    embedding_arrays = []

    total_loaded = 0

    for index, file_path in enumerate(
        files,
        start=1,
    ):

        print(
            f"[{index}/{len(files)}] "
            f"Loading: {file_path.name}..."
        )

        try:

            metadata, embeddings = (
                load_parquet_file(
                    file_path
                )
            )

        except Exception as exc:

            fail(str(exc))

        metadata_frames.append(
            metadata
        )

        embedding_arrays.append(
            embeddings
        )

        total_loaded += len(metadata)

        print(
            f"Loaded {len(metadata):,} reviews"
        )

    if not metadata_frames:
        fail(
            "No review data was loaded."
        )

    print()
    print(
        "Combining review metadata..."
    )

    reviews = pd.concat(
        metadata_frames,
        ignore_index=True,
    )

    print(
        "Building embedding matrix..."
    )

    embeddings = np.vstack(
        embedding_arrays
    ).astype(
        np.float32,
        copy=False,
    )

    print(
        f"Raw matrix shape: "
        f"{embeddings.shape}"
    )

    return reviews, embeddings


# ============================================================
# REVIEW ID COLUMN
# ============================================================

def find_review_id_column(
    reviews: pd.DataFrame,
) -> str:

    for column in [
        "review_id",
        "id",
    ]:

        if column in reviews.columns:
            return column

    fail(
        "Could not find review ID column.\n\n"
        f"Available columns:\n"
        f"{list(reviews.columns)}"
    )

    return ""


# ============================================================
# REMOVE DUPLICATES
# ============================================================

def remove_duplicate_reviews(
    reviews: pd.DataFrame,
    embeddings: np.ndarray,
) -> tuple[pd.DataFrame, np.ndarray, int]:

    print_header(
        "CHECKING DUPLICATE REVIEW IDs"
    )

    review_id_column = (
        find_review_id_column(
            reviews
        )
    )

    duplicate_mask = reviews[
        review_id_column
    ].duplicated(
        keep="first"
    )

    duplicate_count = int(
        duplicate_mask.sum()
    )

    print(
        f"Duplicate IDs found: "
        f"{duplicate_count:,}"
    )

    if duplicate_count == 0:
        return (
            reviews,
            embeddings,
            0,
        )

    keep_mask = (
        ~duplicate_mask.to_numpy()
    )

    reviews = reviews.loc[
        keep_mask
    ].reset_index(
        drop=True
    )

    embeddings = embeddings[
        keep_mask
    ]

    return (
        reviews,
        embeddings,
        duplicate_count,
    )


# ============================================================
# VALIDATE ALIGNMENT
# ============================================================

def validate_alignment(
    reviews: pd.DataFrame,
    embeddings: np.ndarray,
) -> None:

    print_header(
        "VALIDATING REVIEW/EMBEDDING ALIGNMENT"
    )

    if len(reviews) != len(embeddings):

        fail(
            "Review/embedding alignment failed.\n\n"
            f"Reviews: {len(reviews):,}\n"
            f"Embeddings: {len(embeddings):,}"
        )

    if embeddings.ndim != 2:

        fail(
            f"Embedding matrix must be 2-dimensional.\n"
            f"Shape: {embeddings.shape}"
        )

    if embeddings.shape[1] != (
        EXPECTED_EMBEDDING_DIMENSION
    ):

        fail(
            "Unexpected embedding dimension.\n"
            f"Expected: "
            f"{EXPECTED_EMBEDDING_DIMENSION}\n"
            f"Found: {embeddings.shape[1]}"
        )

    if not np.isfinite(
        embeddings
    ).all():

        fail(
            "Embedding matrix contains "
            "NaN or infinite values."
        )

    print(
        "Alignment check: PASSED"
    )

    print(
        f"Reviews     : "
        f"{len(reviews):,}"
    )

    print(
        f"Embeddings  : "
        f"{len(embeddings):,}"
    )

    print(
        f"Dimension   : "
        f"{embeddings.shape[1]}"
    )


# ============================================================
# NORMALIZE EMBEDDINGS
# ============================================================

def normalize_embeddings(
    embeddings: np.ndarray,
) -> np.ndarray:

    print_header(
        "NORMALIZING EMBEDDING MATRIX"
    )

    embeddings = np.asarray(
        embeddings,
        dtype=np.float32,
    )

    norms = np.linalg.norm(
        embeddings,
        axis=1,
        keepdims=True,
    )

    zero_norm_mask = (
        norms.squeeze() == 0
    )

    zero_count = int(
        zero_norm_mask.sum()
    )

    if zero_count > 0:

        fail(
            f"Found {zero_count:,} "
            "zero-vector embeddings."
        )

    embeddings /= norms

    print(
        "Normalization: PASSED"
    )

    return embeddings


# ============================================================
# PREPARE OUTPUT DIRECTORY
# ============================================================

def prepare_output_directory() -> None:

    print_header(
        "PREPARING RAG INDEX DIRECTORY"
    )

    try:

        OUTPUT_DIR.mkdir(
            parents=True,
            exist_ok=True,
        )

    except Exception as exc:

        fail(
            f"Could not create output directory:\n"
            f"{OUTPUT_DIR}\n\n"
            f"Error: {exc}"
        )

    print(
        f"Index directory ready:\n"
        f"{OUTPUT_DIR}"
    )


# ============================================================
# SAVE EMBEDDINGS
# ============================================================

def save_embeddings(
    embeddings: np.ndarray,
) -> None:

    print()
    print(
        "Saving embeddings.npy..."
    )

    try:

        np.save(
            EMBEDDINGS_FILE,
            embeddings,
        )

    except Exception as exc:

        fail(
            f"Could not save:\n"
            f"{EMBEDDINGS_FILE}\n\n"
            f"Error: {exc}"
        )

    if not EMBEDDINGS_FILE.exists():

        fail(
            "embeddings.npy was not created."
        )

    size_gb = (
        EMBEDDINGS_FILE.stat().st_size
        / (1024 ** 3)
    )

    print(
        f"Saved embeddings.npy "
        f"({size_gb:.2f} GB)"
    )


# ============================================================
# SAVE REVIEWS
# ============================================================

def save_reviews(
    reviews: pd.DataFrame,
) -> None:

    print()
    print(
        "Saving reviews.parquet..."
    )

    try:

        reviews.to_parquet(
            REVIEWS_FILE,
            index=False,
        )

    except Exception as exc:

        fail(
            f"Could not save:\n"
            f"{REVIEWS_FILE}\n\n"
            f"Error: {exc}"
        )

    if not REVIEWS_FILE.exists():

        fail(
            "reviews.parquet was not created."
        )

    size_mb = (
        REVIEWS_FILE.stat().st_size
        / (1024 ** 2)
    )

    print(
        f"Saved reviews.parquet "
        f"({size_mb:.2f} MB)"
    )


# ============================================================
# SAVE METADATA
# ============================================================

def save_metadata(
    reviews: pd.DataFrame,
    embeddings: np.ndarray,
    source_file_count: int,
    source_review_count: int,
    duplicates_removed: int,
) -> None:

    print()
    print(
        "Saving metadata.json..."
    )

    metadata = {

        "index_version": "1.0",

        "created_at":
            datetime.now().isoformat(),

        "source": {

            "directory":
                str(SOURCE_DIR),

            "file_count":
                source_file_count,
        },

        "reviews": {

            "source_reviews_loaded":
                source_review_count,

            "duplicates_removed":
                duplicates_removed,

            "final_reviews":
                len(reviews),
        },

        "embeddings": {

            "dimension":
                int(embeddings.shape[1]),

            "count":
                int(embeddings.shape[0]),

            "dtype":
                str(embeddings.dtype),

            "normalized":
                True,

            "metric":
                "cosine",
        },

        "files": {

            "embeddings":
                "embeddings.npy",

            "reviews":
                "reviews.parquet",

            "metadata":
                "metadata.json",
        },
    }

    try:

        with open(
            METADATA_FILE,
            "w",
            encoding="utf-8",
        ) as file:

            json.dump(
                metadata,
                file,
                indent=2,
            )

    except Exception as exc:

        fail(
            f"Could not save:\n"
            f"{METADATA_FILE}\n\n"
            f"Error: {exc}"
        )

    if not METADATA_FILE.exists():

        fail(
            "metadata.json was not created."
        )

    print(
        "Saved metadata.json"
    )


# ============================================================
# VALIDATE GENERATED INDEX
# ============================================================

def validate_generated_index(
    expected_count: int,
    expected_dimension: int,
) -> None:

    print_header(
        "VALIDATING GENERATED INDEX"
    )

    required_files = [

        EMBEDDINGS_FILE,

        REVIEWS_FILE,

        METADATA_FILE,
    ]

    for file_path in required_files:

        if not file_path.exists():

            fail(
                f"Required index file is missing:\n"
                f"{file_path}"
            )

        print(
            f"Found: {file_path.name}"
        )

    # --------------------------------------------------------
    # Validate embeddings
    # --------------------------------------------------------

    try:

        saved_embeddings = np.load(
            EMBEDDINGS_FILE,
            mmap_mode="r",
        )

    except Exception as exc:

        fail(
            "Could not reopen embeddings.npy.\n\n"
            f"Error: {exc}"
        )

    expected_shape = (
        expected_count,
        expected_dimension,
    )

    if saved_embeddings.shape != (
        expected_shape
    ):

        fail(
            "Saved embedding matrix has "
            "an unexpected shape.\n\n"
            f"Expected: {expected_shape}\n"
            f"Found: {saved_embeddings.shape}"
        )

    # --------------------------------------------------------
    # Validate reviews
    # --------------------------------------------------------

    try:

        saved_reviews = pd.read_parquet(
            REVIEWS_FILE
        )

    except Exception as exc:

        fail(
            "Could not reopen reviews.parquet.\n\n"
            f"Error: {exc}"
        )

    if len(saved_reviews) != (
        expected_count
    ):

        fail(
            "Saved review count does not "
            "match embedding count.\n\n"
            f"Reviews: "
            f"{len(saved_reviews):,}\n"
            f"Embeddings: "
            f"{expected_count:,}"
        )

    # --------------------------------------------------------
    # Validate metadata
    # --------------------------------------------------------

    try:

        with open(
            METADATA_FILE,
            "r",
            encoding="utf-8",
        ) as file:

            metadata = json.load(
                file
            )

    except Exception as exc:

        fail(
            "Could not reopen metadata.json.\n\n"
            f"Error: {exc}"
        )

    if metadata[
        "embeddings"
    ][
        "count"
    ] != expected_count:

        fail(
            "metadata.json embedding count "
            "does not match the generated index."
        )

    print()
    print(
        "Index validation: PASSED"
    )


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    print("=" * 70)
    print(
        "ZOMATO AI - RAG INDEX BUILDER"
    )
    print("=" * 70)

    print()
    print(
        "This process will rebuild the persistent "
        "RAG index from all parquet files."
    )

    print()
    print(
        f"Source directory : {SOURCE_DIR}"
    )

    print(
        f"Output directory : {OUTPUT_DIR}"
    )

    # --------------------------------------------------------
    # Find files
    # --------------------------------------------------------

    files = find_embedding_files()

    # --------------------------------------------------------
    # Load ALL 60 parquet files
    # --------------------------------------------------------

    (
        reviews,
        embeddings,
    ) = load_all_batches(
        files
    )

    source_review_count = len(
        reviews
    )

    # --------------------------------------------------------
    # Remove duplicates
    # --------------------------------------------------------

    (
        reviews,
        embeddings,
        duplicates_removed,
    ) = remove_duplicate_reviews(
        reviews,
        embeddings,
    )

    # --------------------------------------------------------
    # Validate
    # --------------------------------------------------------

    validate_alignment(
        reviews,
        embeddings,
    )

    # --------------------------------------------------------
    # Normalize
    # --------------------------------------------------------

    embeddings = normalize_embeddings(
        embeddings
    )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print_header(
        "INDEX BUILD SUMMARY"
    )

    print(
        f"Source files          : "
        f"{len(files):,}"
    )

    print(
        f"Reviews loaded        : "
        f"{source_review_count:,}"
    )

    print(
        f"Duplicates removed    : "
        f"{duplicates_removed:,}"
    )

    print(
        f"Final reviews         : "
        f"{len(reviews):,}"
    )

    print(
        f"Embedding dimension   : "
        f"{embeddings.shape[1]}"
    )

    print(
        f"Embedding dtype       : "
        f"{embeddings.dtype}"
    )

    memory_mb = (
        embeddings.nbytes
        / (1024 ** 2)
    )

    print(
        f"Embedding memory      : "
        f"{memory_mb:,.2f} MB"
    )

    print(
        "Normalized            : True"
    )

    # --------------------------------------------------------
    # Prepare output
    # --------------------------------------------------------

    prepare_output_directory()

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    print_header(
        "SAVING PERSISTENT RAG INDEX"
    )

    save_embeddings(
        embeddings
    )

    save_reviews(
        reviews
    )

    save_metadata(
        reviews=reviews,
        embeddings=embeddings,
        source_file_count=len(files),
        source_review_count=source_review_count,
        duplicates_removed=duplicates_removed,
    )

    # --------------------------------------------------------
    # Validate
    # --------------------------------------------------------

    validate_generated_index(
        expected_count=len(reviews),
        expected_dimension=(
            embeddings.shape[1]
        ),
    )

    # --------------------------------------------------------
    # Success
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print(
        "✅ RAG INDEX BUILD COMPLETED SUCCESSFULLY"
    )
    print("=" * 70)

    print()
    print(
        f"Index location : {OUTPUT_DIR}"
    )

    print()
    print("Files:")

    print(
        f"  ✓ embeddings.npy "
        f"({EMBEDDINGS_FILE.stat().st_size / (1024 ** 3):.2f} GB)"
    )

    print(
        f"  ✓ reviews.parquet "
        f"({REVIEWS_FILE.stat().st_size / (1024 ** 2):.2f} MB)"
    )

    print(
        "  ✓ metadata.json"
    )

    print()
    print(
        f"Reviews indexed : "
        f"{len(reviews):,}"
    )

    print(
        f"Embedding size  : "
        f"{embeddings.shape[1]}"
    )

    print()
    print("=" * 70)


if __name__ == "__main__":
    main()
