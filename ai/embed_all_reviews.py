import os
import json
import time
from pathlib import Path

import pandas as pd
import requests
from google.cloud import bigquery
from dotenv import load_dotenv


# ============================================================
# CONFIGURATION
# ============================================================

load_dotenv()

PROJECT_ID = "zomato-ai-data-engineering"

REVIEWS_TABLE = f"{PROJECT_ID}.raw.reviews"


# ============================================================
# OLLAMA CONFIGURATION
# ============================================================

OLLAMA_URL = "http://localhost:11434/api/embed"

EMBEDDING_MODEL = "mxbai-embed-large:latest"

# Number of reviews sent to Ollama in one request
EMBED_BATCH_SIZE = 128


# ============================================================
# BIGQUERY CONFIGURATION
# ============================================================

BQ_LOCATION = "asia-south1"

# Number of reviews fetched from BigQuery at one time
BQ_BATCH_SIZE = 5000


# ============================================================
# OUTPUT CONFIGURATION
# ============================================================

OUTPUT_DIR = (
    Path(__file__).parent / "review_embeddings"
)

PROGRESS_FILE = (
    OUTPUT_DIR / "progress.json"
)


# ============================================================
# RETRY CONFIGURATION
# ============================================================

OLLAMA_TIMEOUT = 300

RETRY_DELAY = 5

MAX_RETRIES = 3


# ============================================================
# CREATE OUTPUT DIRECTORY
# ============================================================

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# BIGQUERY CLIENT
# ============================================================

bq_client = bigquery.Client(
    project=PROJECT_ID
)


# ============================================================
# PROGRESS MANAGEMENT
# ============================================================

def default_progress():

    return {
        "last_review_id": None,
        "total_processed": 0,
        "batches_completed": 0
    }


def load_progress():

    if not PROGRESS_FILE.exists():

        return default_progress()

    try:

        with open(
            PROGRESS_FILE,
            "r",
            encoding="utf-8"
        ) as f:

            progress = json.load(f)

        # ----------------------------------------------------
        # Make sure all expected keys exist
        # ----------------------------------------------------

        progress.setdefault(
            "last_review_id",
            None
        )

        progress.setdefault(
            "total_processed",
            0
        )

        progress.setdefault(
            "batches_completed",
            0
        )

        # ----------------------------------------------------
        # IMPORTANT:
        # BigQuery review_id is INT64.
        #
        # Older progress.json files may contain it as STRING.
        # Convert it here so future queries use INT64.
        # ----------------------------------------------------

        if progress["last_review_id"] is not None:

            progress["last_review_id"] = int(
                progress["last_review_id"]
            )

        progress["total_processed"] = int(
            progress["total_processed"]
        )

        progress["batches_completed"] = int(
            progress["batches_completed"]
        )

        return progress

    except Exception as e:

        print(
            "WARNING: Could not read progress.json."
        )

        print(
            f"Error: {e}"
        )

        print(
            "Starting with fresh progress."
        )

        return default_progress()


def save_progress(
    last_review_id,
    total_processed,
    batches_completed
):

    # Make absolutely sure review_id is stored as integer
    if last_review_id is not None:

        last_review_id = int(
            last_review_id
        )

    progress = {

        "last_review_id":
            last_review_id,

        "total_processed":
            int(total_processed),

        "batches_completed":
            int(batches_completed)
    }

    temp_file = (
        OUTPUT_DIR / "progress.tmp"
    )

    with open(
        temp_file,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            progress,
            f,
            indent=2
        )

    # Atomic replacement
    os.replace(
        temp_file,
        PROGRESS_FILE
    )


# ============================================================
# CHECK OLLAMA
# ============================================================

def check_ollama():

    print()
    print("Checking Ollama...")

    try:

        response = requests.get(
            "http://localhost:11434/api/tags",
            timeout=10
        )

        response.raise_for_status()

        models = response.json().get(
            "models",
            []
        )

        model_names = [
            model.get("name")
            for model in models
        ]

        if EMBEDDING_MODEL not in model_names:

            print()
            print(
                f"ERROR: {EMBEDDING_MODEL} "
                f"was not found in Ollama."
            )

            print()
            print("Available models:")

            for name in model_names:

                print(
                    f"  - {name}"
                )

            print()

            print(
                f"Run:"
            )

            print(
                f"ollama pull {EMBEDDING_MODEL}"
            )

            raise SystemExit(1)

        print(
            f"OK: {EMBEDDING_MODEL} is available."
        )

    except requests.RequestException as e:

        print()
        print(
            "ERROR: Could not connect to Ollama."
        )

        print(
            "Make sure Ollama is running."
        )

        print(
            "URL: http://localhost:11434"
        )

        print(
            f"Error: {e}"
        )

        raise SystemExit(1)


# ============================================================
# GET TOTAL REVIEW COUNT
# ============================================================

def get_total_reviews():

    query = f"""
        SELECT COUNT(*) AS total
        FROM `{REVIEWS_TABLE}`
        WHERE comment IS NOT NULL
          AND TRIM(comment) != ''
    """

    result = (
        bq_client
        .query(
            query,
            location=BQ_LOCATION
        )
        .result()
    )

    row = next(
        iter(result)
    )

    return int(
        row.total
    )


# ============================================================
# FETCH REVIEWS FROM BIGQUERY
# ============================================================

def fetch_reviews(last_review_id):

    # ========================================================
    # FIRST BATCH
    # ========================================================

    if last_review_id is None:

        query = f"""
            SELECT
                review_id,
                rating,
                comment,
                review_date

            FROM `{REVIEWS_TABLE}`

            WHERE comment IS NOT NULL
              AND TRIM(comment) != ''

            ORDER BY review_id

            LIMIT {BQ_BATCH_SIZE}
        """

        result = (
            bq_client
            .query(
                query,
                location=BQ_LOCATION
            )
            .result()
        )

        return result.to_dataframe()


    # ========================================================
    # RESUME FROM LAST REVIEW ID
    # ========================================================

    query = f"""
        SELECT
            review_id,
            rating,
            comment,
            review_date

        FROM `{REVIEWS_TABLE}`

        WHERE comment IS NOT NULL
          AND TRIM(comment) != ''

          AND review_id > @last_review_id

        ORDER BY review_id

        LIMIT {BQ_BATCH_SIZE}
    """


    # ========================================================
    # IMPORTANT FIX
    #
    # review_id in BigQuery = INT64
    #
    # Therefore parameter MUST be INT64.
    # ========================================================

    job_config = bigquery.QueryJobConfig(

        query_parameters=[

            bigquery.ScalarQueryParameter(
                "last_review_id",
                "INT64",
                int(last_review_id)
            )

        ]
    )


    result = (
        bq_client
        .query(
            query,
            job_config=job_config,
            location=BQ_LOCATION
        )
        .result()
    )


    return result.to_dataframe()


# ============================================================
# EMBED ONE OLLAMA BATCH
# ============================================================

def embed_batch(texts):

    if not texts:

        return []


    for attempt in range(
        1,
        MAX_RETRIES + 1
    ):

        try:

            response = requests.post(

                OLLAMA_URL,

                json={

                    "model":
                        EMBEDDING_MODEL,

                    "input":
                        texts
                },

                timeout=OLLAMA_TIMEOUT
            )


            response.raise_for_status()


            data = response.json()


            embeddings = data.get(
                "embeddings"
            )


            if embeddings is None:

                raise RuntimeError(
                    "Ollama response did not "
                    "contain 'embeddings'."
                )


            if len(embeddings) != len(texts):

                raise RuntimeError(

                    f"Ollama returned "
                    f"{len(embeddings)} embeddings "
                    f"for {len(texts)} texts."
                )


            return embeddings


        except Exception as e:

            print()

            print(
                f"Embedding request failed "
                f"(attempt {attempt}/{MAX_RETRIES})"
            )

            print(
                f"Error: {e}"
            )


            if attempt < MAX_RETRIES:

                print(
                    f"Waiting {RETRY_DELAY} seconds..."
                )

                time.sleep(
                    RETRY_DELAY
                )

            else:

                print(
                    "Maximum retries reached."
                )

                raise


# ============================================================
# SAVE EMBEDDING BATCH
# ============================================================

def save_batch(
    df,
    embeddings,
    batch_number
):

    output_file = (

        OUTPUT_DIR
        / f"batch_{batch_number:06d}.parquet"

    )


    result = df.copy()


    result["embedding"] = embeddings


    result.to_parquet(

        output_file,

        index=False
    )


    return output_file


# ============================================================
# MAIN EMBEDDING PIPELINE
# ============================================================

def main():

    print(
        "=" * 70
    )

    print(
        "ZOMATO REVIEW EMBEDDING PIPELINE"
    )

    print(
        "=" * 70
    )


    print(
        f"Project       : {PROJECT_ID}"
    )

    print(
        f"Source table  : {REVIEWS_TABLE}"
    )

    print(
        f"Embedding     : {EMBEDDING_MODEL}"
    )

    print(
        f"Ollama URL    : {OLLAMA_URL}"
    )

    print(
        f"BigQuery batch: {BQ_BATCH_SIZE}"
    )

    print(
        f"Ollama batch  : {EMBED_BATCH_SIZE}"
    )

    print(
        f"Output        : {OUTPUT_DIR}"
    )

    print(
        "=" * 70
    )


    # ========================================================
    # CHECK OLLAMA
    # ========================================================

    check_ollama()


    # ========================================================
    # COUNT REVIEWS
    # ========================================================

    print()
    print(
        "Counting reviews in BigQuery..."
    )


    total_reviews = get_total_reviews()


    print(
        f"Reviews available: "
        f"{total_reviews:,}"
    )


    # ========================================================
    # LOAD PROGRESS
    # ========================================================

    progress = load_progress()


    last_review_id = progress.get(
        "last_review_id"
    )


    # Extra protection
    if last_review_id is not None:

        last_review_id = int(
            last_review_id
        )


    total_processed = int(
        progress.get(
            "total_processed",
            0
        )
    )


    batches_completed = int(
        progress.get(
            "batches_completed",
            0
        )
    )


    print()

    print(
        f"Previously processed: "
        f"{total_processed:,}"
    )

    print(
        f"Last review ID: "
        f"{last_review_id}"
    )

    print(
        f"Completed batches: "
        f"{batches_completed}"
    )


    # ========================================================
    # CHECK IF ALREADY COMPLETE
    # ========================================================

    if total_processed >= total_reviews:

        print()

        print(
            "All available reviews "
            "appear to be processed."
        )

        return


    # ========================================================
    # MAIN LOOP
    # ========================================================

    while True:

        df = fetch_reviews(
            last_review_id
        )


        # ----------------------------------------------------
        # No more reviews
        # ----------------------------------------------------

        if df.empty:

            print()

            print(
                "No more reviews found."
            )

            break


        print()

        print(
            "-" * 70
        )

        print(
            f"Fetched {len(df):,} reviews"
        )


        print(
            f"Review range: "
            f"{df.iloc[0]['review_id']} "
            f"→ "
            f"{df.iloc[-1]['review_id']}"
        )


        print(
            "-" * 70
        )


        # ====================================================
        # PROCESS OLLAMA BATCHES
        # ====================================================

        all_embeddings = []


        for start in range(

            0,

            len(df),

            EMBED_BATCH_SIZE

        ):


            end = min(

                start + EMBED_BATCH_SIZE,

                len(df)

            )


            batch_df = df.iloc[
                start:end
            ]


            texts = (

                batch_df["comment"]

                .astype(str)

                .tolist()

            )


            print(

                f"Embedding "

                f"{start + 1:,}-"

                f"{end:,} / "

                f"{len(df):,}..."

            )


            embeddings = embed_batch(
                texts
            )


            all_embeddings.extend(
                embeddings
            )


        # ====================================================
        # SAVE BATCH
        # ====================================================

        batches_completed += 1


        output_file = save_batch(

            df,

            all_embeddings,

            batches_completed

        )


        # ====================================================
        # UPDATE PROGRESS
        #
        # IMPORTANT:
        # Progress is updated ONLY after parquet
        # has been successfully saved.
        # ====================================================

        last_review_id = int(

            df.iloc[-1]["review_id"]

        )


        total_processed += len(df)


        save_progress(

            last_review_id,

            total_processed,

            batches_completed

        )


        # ====================================================
        # SHOW PROGRESS
        # ====================================================

        print()

        print(
            f"Saved: "
            f"{output_file.name}"
        )


        print(
            f"Processed: "
            f"{total_processed:,} / "
            f"{total_reviews:,}"
        )


        percentage = (

            total_processed

            / total_reviews

            * 100

        )


        print(
            f"Progress: "
            f"{percentage:.2f}%"
        )


        # ====================================================
        # CHECK LAST BIGQUERY BATCH
        # ====================================================

        if len(df) < BQ_BATCH_SIZE:

            print()

            print(
                "Last BigQuery batch reached."
            )

            break


    # ========================================================
    # FINISHED
    # ========================================================

    print()

    print(
        "=" * 70
    )

    print(
        "EMBEDDING PIPELINE COMPLETED"
    )

    print(
        "=" * 70
    )


    print(
        f"Total processed : "
        f"{total_processed:,}"
    )


    print(
        f"Embedding model : "
        f"{EMBEDDING_MODEL}"
    )


    print(
        f"Output directory: "
        f"{OUTPUT_DIR}"
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()