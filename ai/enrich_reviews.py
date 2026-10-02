import os
import json
import time
import argparse
from datetime import datetime, timezone

import requests
from google.cloud import bigquery


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ID = "zomato-ai-data-engineering"

SOURCE_TABLE = f"{PROJECT_ID}.raw.reviews"
TARGET_TABLE = f"{PROJECT_ID}.ai.review_enriched"

# ------------------------------------------------------------
# OMNIROUTE
# ------------------------------------------------------------

OMNIROUTE_URL = os.getenv(
    "OMNIROUTE_URL",
    "http://localhost:20128/v1/chat/completions",
)

OMNIROUTE_API_KEY = os.getenv("OMNIROUTE_API_KEY")


# ============================================================
# MODEL FALLBACK CHAIN
# ============================================================
#
# IMPORTANT:
# Each batch gets ONE attempt on each model.
#
# If Model 1 fails:
#     -> Model 2
#
# If Model 2 fails:
#     -> Model 3
#
# If Model 3 fails:
#     -> Model 4
#
# No model is retried.
#
# Change these IDs to exactly match the models available
# in your OmniRoute /v1/models catalog.
# ============================================================

MODELS = [
    "nvidia/nvidia/nemotron-3-super-120b-a12b",

    "antigravity/gemini-3.7-flash-high",

    "antigravity/gemini-3.7-flash-medium",

    "antigravity/gemini-3.1-flash-lite",
]


# ============================================================
# PROCESSING SETTINGS
# ============================================================

# Keep this small initially.
# After successful testing you can increase it.
BATCH_SIZE = 15

REQUEST_TIMEOUT = 600

# Small delay between switching models.
MODEL_SWITCH_DELAY = 1


# ============================================================
# COMMAND LINE ARGUMENTS
# ============================================================

parser = argparse.ArgumentParser(
    description="Zomato Review Enrichment Pipeline"
)

parser.add_argument(
    "--worker",
    type=int,
    default=1,
    help="Worker number. Example: 1"
)

parser.add_argument(
    "--workers",
    type=int,
    default=4,
    help="Total number of workers. Example: 4"
)

args = parser.parse_args()

WORKER_ID = args.worker
TOTAL_WORKERS = args.workers


# ============================================================
# VALIDATION
# ============================================================

if TOTAL_WORKERS < 1:
    raise ValueError("--workers must be at least 1.")

if WORKER_ID < 1 or WORKER_ID > TOTAL_WORKERS:
    raise ValueError(
        f"--worker must be between 1 and {TOTAL_WORKERS}."
    )

if not OMNIROUTE_API_KEY:
    raise RuntimeError(
        "OMNIROUTE_API_KEY environment variable is not set."
    )


# ============================================================
# BIGQUERY CLIENT
# ============================================================

bq_client = bigquery.Client(
    project=PROJECT_ID
)


# ============================================================
# HEADER
# ============================================================

print()
print("=" * 75)
print("ZOMATO REVIEW ENRICHMENT PIPELINE")
print("=" * 75)

print(f"Worker        : {WORKER_ID} / {TOTAL_WORKERS}")
print(f"Source        : {SOURCE_TABLE}")
print(f"Target        : {TARGET_TABLE}")
print(f"Batch size    : {BATCH_SIZE}")
print(f"Models        : {len(MODELS)}")
print()

print("MODEL FALLBACK ORDER")
print("-" * 75)

for index, model in enumerate(MODELS, start=1):
    print(f"{index}. {model}")

print("=" * 75)


# ============================================================
# CREATE TARGET TABLE
# ============================================================

def ensure_target_table():

    print()
    print("Checking target table...")

    schema = [

        bigquery.SchemaField(
            "review_id",
            "STRING",
            mode="REQUIRED"
        ),

        bigquery.SchemaField(
            "sentiment_label",
            "STRING",
            mode="NULLABLE"
        ),

        bigquery.SchemaField(
            "sentiment_score",
            "FLOAT64",
            mode="NULLABLE"
        ),

        bigquery.SchemaField(
            "topic",
            "STRING",
            mode="NULLABLE"
        ),

        bigquery.SchemaField(
            "key_issue",
            "STRING",
            mode="NULLABLE"
        ),

        bigquery.SchemaField(
            "processed_at",
            "TIMESTAMP",
            mode="NULLABLE"
        ),

        bigquery.SchemaField(
            "model_used",
            "STRING",
            mode="NULLABLE"
        ),
    ]

    table = bigquery.Table(
        TARGET_TABLE,
        schema=schema
    )

    try:

        existing = bq_client.get_table(
            TARGET_TABLE
        )

        print(
            f"Target table already exists: "
            f"{existing.project}."
            f"{existing.dataset_id}."
            f"{existing.table_id}"
        )

    except Exception:

        print("Target table does not exist.")
        print("Creating target table...")

        created = bq_client.create_table(
            table
        )

        print(
            f"Target table created: "
            f"{created.project}."
            f"{created.dataset_id}."
            f"{created.table_id}"
        )


# ============================================================
# GET UNPROCESSED REVIEWS
# ============================================================

def get_unprocessed_reviews():

    print()
    print("Finding unprocessed reviews...")

    print(
        f"Assigning partition "
        f"{WORKER_ID} of {TOTAL_WORKERS}..."
    )

    partition_number = WORKER_ID - 1

    query = f"""

        SELECT

            CAST(r.review_id AS STRING) AS review_id,

            SAFE_CAST(r.rating AS INT64) AS rating,

            CAST(r.comment AS STRING) AS comment

        FROM `{SOURCE_TABLE}` AS r

        WHERE r.comment IS NOT NULL

          AND TRIM(
              CAST(r.comment AS STRING)
          ) != ''

          AND NOT EXISTS (

              SELECT 1

              FROM `{TARGET_TABLE}` AS e

              WHERE e.review_id =
                    CAST(r.review_id AS STRING)
          )

          AND MOD(

              SAFE_CAST(
                  r.review_id AS INT64
              ),

              {TOTAL_WORKERS}

          ) = {partition_number}

        ORDER BY

            SAFE_CAST(
                r.review_id AS INT64
            ),

            CAST(
                r.review_id AS STRING
            )

    """

    query_job = bq_client.query(
        query
    )

    return query_job.result()


# ============================================================
# BUILD PROMPT
# ============================================================

def build_prompt(reviews):

    reviews_json = json.dumps(
        reviews,
        ensure_ascii=False,
        indent=2,
        default=str
    )

    prompt = f"""
You are analyzing customer reviews for a food delivery
and restaurant analytics system.

For every review determine:

1. sentiment_label

Allowed values:

positive
negative
neutral

2. sentiment_score

Must be between -1.0 and 1.0.

Negative sentiment should be below 0.
Neutral sentiment should be around 0.
Positive sentiment should be above 0.

3. topic

Choose the most appropriate topic.

Allowed examples:

food_quality
taste
delivery
service
pricing
quantity
packaging
restaurant
staff
hygiene
ambience
order_issue
app
other

4. key_issue

If there is a specific problem mentioned,
summarize it briefly.

If there is no specific issue,
use null.

IMPORTANT:

- Return exactly one result for every input review.
- Preserve the exact review_id.
- Do not invent reviews.
- Do not omit reviews.
- Return ONLY valid JSON.
- Do not use Markdown.
- Do not use ```json.
- Return a JSON array.

Expected format:

[
  {{
    "review_id": "123",
    "sentiment_label": "positive",
    "sentiment_score": 0.85,
    "topic": "food_quality",
    "key_issue": null
  }}
]

Reviews:

{reviews_json}
"""

    return prompt


# ============================================================
# EXTRACT JSON
# ============================================================

def extract_json(text):

    if not text:
        raise ValueError(
            "Empty model response."
        )

    text = text.strip()

    # Remove Markdown fences
    if text.startswith("```"):

        lines = text.splitlines()

        if lines:
            lines = lines[1:]

        if (
            lines
            and lines[-1].strip().startswith("```")
        ):
            lines = lines[:-1]

        text = "\n".join(lines).strip()

    # Find JSON array
    start = text.find("[")
    end = text.rfind("]")

    if start == -1 or end == -1 or end <= start:

        raise ValueError(
            "Could not find JSON array in model response."
        )

    text = text[
        start:end + 1
    ]

    return json.loads(text)


# ============================================================
# VALIDATE MODEL OUTPUT
# ============================================================

def validate_results(
    results,
    input_reviews
):

    if not isinstance(results, list):

        raise ValueError(
            "Model output is not a JSON array."
        )

    input_ids = {
        str(review["review_id"])
        for review in input_reviews
    }

    output_ids = []

    for item in results:

        if not isinstance(item, dict):

            raise ValueError(
                "Model result is not an object."
            )

        required_fields = [
            "review_id",
            "sentiment_label",
            "sentiment_score",
            "topic",
            "key_issue",
        ]

        for field in required_fields:

            if field not in item:

                raise ValueError(
                    f"Missing field '{field}'."
                )

        review_id = str(
            item["review_id"]
        )

        if review_id not in input_ids:

            raise ValueError(
                f"Unexpected review_id: {review_id}"
            )

        output_ids.append(
            review_id
        )

        sentiment = item[
            "sentiment_label"
        ]

        if sentiment not in {
            "positive",
            "negative",
            "neutral",
        }:

            raise ValueError(
                f"Invalid sentiment_label: {sentiment}"
            )

        try:

            score = float(
                item["sentiment_score"]
            )

        except Exception:

            raise ValueError(
                f"Invalid sentiment_score "
                f"for review {review_id}"
            )

        if score < -1.0 or score > 1.0:

            raise ValueError(
                f"sentiment_score outside "
                f"[-1,1] for review {review_id}"
            )

    # Number check
    if len(results) != len(input_reviews):

        raise ValueError(
            f"Expected {len(input_reviews)} "
            f"results but received {len(results)}."
        )

    # Duplicate check
    if len(set(output_ids)) != len(output_ids):

        raise ValueError(
            "Model returned duplicate review IDs."
        )

    # Missing / extra IDs
    if set(output_ids) != input_ids:

        missing = (
            input_ids - set(output_ids)
        )

        extra = (
            set(output_ids) - input_ids
        )

        raise ValueError(
            f"Review ID mismatch. "
            f"Missing={missing}, Extra={extra}"
        )

    return True


# ============================================================
# CALL ONE MODEL
# ============================================================

def call_one_model(
    model_name,
    reviews
):

    prompt = build_prompt(
        reviews
    )

    headers = {
        "Authorization":
            f"Bearer {OMNIROUTE_API_KEY}",

        "Content-Type":
            "application/json",
    }

    payload = {

        "model": model_name,

        "messages": [

            {
                "role": "system",

                "content":
                    "You are a structured "
                    "data extraction model. "
                    "Return only valid JSON."
            },

            {
                "role": "user",

                "content": prompt,
            },
        ],

        "temperature": 0,
    }

    start_time = time.time()

    response = requests.post(
        OMNIROUTE_URL,
        headers=headers,
        json=payload,
        timeout=REQUEST_TIMEOUT
    )

    elapsed = (
        time.time()
        - start_time
    )

    print(
        f"    HTTP status : "
        f"{response.status_code}"
    )

    print(
        f"    Response time : "
        f"{elapsed:.2f} seconds"
    )

    response.raise_for_status()

    data = response.json()

    choices = data.get(
        "choices"
    )

    if not choices:

        raise ValueError(
            "Response does not contain choices."
        )

    message = choices[0].get(
        "message",
        {}
    )

    content = message.get(
        "content"
    )

    if not content:

        raise ValueError(
            "Model returned empty content."
        )

    results = extract_json(
        content
    )

    validate_results(
        results,
        reviews
    )

    actual_model = data.get(
        "model",
        model_name
    )

    return results, actual_model


# ============================================================
# MODEL FALLBACK CHAIN
# ============================================================

def call_with_fallback(
    reviews
):

    last_error = None

    print()
    print(
        "    Starting model fallback chain..."
    )

    # IMPORTANT:
    # Each model is attempted ONLY ONCE.
    for index, model_name in enumerate(
        MODELS,
        start=1
    ):

        print()
        print(
            f"    MODEL {index}/{len(MODELS)}"
        )

        print(
            f"    Trying: {model_name}"
        )

        try:

            results, actual_model = (
                call_one_model(
                    model_name,
                    reviews
                )
            )

            print()
            print(
                f"    ✅ MODEL {index} SUCCESS"
            )

            print(
                f"    Model used: "
                f"{actual_model}"
            )

            return results, actual_model

        except Exception as exc:

            last_error = exc

            print()
            print(
                f"    ❌ MODEL {index} FAILED"
            )

            print(
                f"    Error: "
                f"{type(exc).__name__}: {exc}"
            )

            # Do NOT retry this model.

            if index < len(MODELS):

                print(
                    f"    Switching immediately "
                    f"to MODEL {index + 1}..."
                )

                time.sleep(
                    MODEL_SWITCH_DELAY
                )

    # Every model failed
    raise RuntimeError(
        "ALL MODELS FAILED. "
        f"Last error: {last_error}"
    )


# ============================================================
# REMOVE ALREADY WRITTEN
# ============================================================

def remove_already_written(
    results
):

    if not results:
        return []

    review_ids = [
        str(item["review_id"])
        for item in results
    ]

    query = f"""

        SELECT review_id

        FROM `{TARGET_TABLE}`

        WHERE review_id
        IN UNNEST(@review_ids)

    """

    job_config = (
        bigquery.QueryJobConfig(

            query_parameters=[

                bigquery.ArrayQueryParameter(
                    "review_ids",
                    "STRING",
                    review_ids
                )
            ]
        )
    )

    rows = bq_client.query(
        query,
        job_config=job_config
    ).result()

    existing_ids = {
        str(row.review_id)
        for row in rows
    }

    if existing_ids:

        print(
            f"    Duplicate protection removed "
            f"{len(existing_ids)} rows."
        )

    return [
        item

        for item in results

        if str(item["review_id"])
        not in existing_ids
    ]


# ============================================================
# WRITE RESULTS
# ============================================================

def write_results(
    results,
    model_used
):

    if not results:

        print(
            "    Nothing to write."
        )

        return 0

    processed_at = datetime.now(
        timezone.utc
    ).isoformat()

    rows_to_write = []

    for item in results:

        key_issue = item.get(
            "key_issue"
        )

        if key_issue is not None:

            key_issue = str(
                key_issue
            ).strip()

            if key_issue == "":
                key_issue = None

        rows_to_write.append(

            {

                "review_id":
                    str(
                        item["review_id"]
                    ),

                "sentiment_label":
                    str(
                        item[
                            "sentiment_label"
                        ]
                    ).strip(),

                "sentiment_score":
                    float(
                        item[
                            "sentiment_score"
                        ]
                    ),

                "topic":
                    str(
                        item["topic"]
                    ).strip(),

                "key_issue":
                    key_issue,

                "processed_at":
                    processed_at,

                "model_used":
                    str(
                        model_used
                    ),
            }
        )

    print(
        f"    Writing "
        f"{len(rows_to_write)} rows..."
    )

    job_config = (
        bigquery.LoadJobConfig(

            write_disposition=
                bigquery.WriteDisposition
                .WRITE_APPEND
        )
    )

    load_job = (
        bq_client
        .load_table_from_json(
            rows_to_write,
            TARGET_TABLE,
            job_config=job_config
        )
    )

    load_job.result()

    print(
        f"    ✅ Successfully wrote "
        f"{len(rows_to_write)} rows."
    )

    return len(rows_to_write)


# ============================================================
# PROCESS ONE BATCH
# ============================================================

def process_batch(
    current_batch,
    batch_number,
    total_selected
):

    print()
    print("=" * 75)

    print(
        f"WORKER {WORKER_ID}/{TOTAL_WORKERS}"
        f" → BATCH {batch_number}"
    )

    print(
        f"Reviews: "
        f"{current_batch[0]['review_id']} -> "
        f"{current_batch[-1]['review_id']}"
    )

    print(
        f"Count: "
        f"{len(current_batch)}"
    )

    print("=" * 75)

    batch_start = time.time()

    try:

        # ----------------------------------------------------
        # THIS IS THE IMPORTANT PART
        #
        # If model 1 fails:
        #     model 2
        #
        # If model 2 fails:
        #     model 3
        #
        # If model 3 fails:
        #     model 4
        #
        # NO SAME-MODEL RETRY
        # ----------------------------------------------------

        results, model_used = (
            call_with_fallback(
                current_batch
            )
        )

        results = (
            remove_already_written(
                results
            )
        )

        written = write_results(
            results,
            model_used
        )

        batch_time = (
            time.time()
            - batch_start
        )

        print(
            f"    Batch completed in "
            f"{batch_time:.2f} seconds."
        )

        print(
            f"    Successful model: "
            f"{model_used}"
        )

        return written, True

    except Exception as exc:

        print()
        print(
            "    !!! BATCH FAILED !!!"
        )

        print(
            f"    Error: "
            f"{type(exc).__name__}: {exc}"
        )

        print(
            "    All configured models "
            "failed for this batch."
        )

        print(
            "    This batch will NOT be written."
        )

        print(
            "    Worker will continue "
            "with next batch."
        )

        return 0, False


# ============================================================
# MAIN PIPELINE
# ============================================================

def main():

    pipeline_start = time.time()

    ensure_target_table()

    rows = get_unprocessed_reviews()

    total_selected = 0
    total_written = 0
    total_failed = 0
    batches_completed = 0

    current_batch = []

    print()
    print(
        f"Worker {WORKER_ID}/{TOTAL_WORKERS} "
        f"is processing its assigned reviews..."
    )

    # --------------------------------------------------------
    # STREAM REVIEWS
    # --------------------------------------------------------

    for row in rows:

        current_batch.append(

            {

                "review_id":
                    str(
                        row.review_id
                    ),

                "rating":
                    row.rating,

                "comment":
                    str(
                        row.comment
                    ),
            }
        )

        if len(current_batch) < BATCH_SIZE:
            continue

        total_selected += len(
            current_batch
        )

        batch_number = (
            (total_selected - 1)
            // BATCH_SIZE
        ) + 1

        written, success = (
            process_batch(
                current_batch,
                batch_number,
                total_selected
            )
        )

        total_written += written

        if success:
            batches_completed += 1
        else:
            total_failed += 1

        current_batch = []

    # --------------------------------------------------------
    # FINAL PARTIAL BATCH
    # --------------------------------------------------------

    if current_batch:

        total_selected += len(
            current_batch
        )

        batch_number = (
            (total_selected - 1)
            // BATCH_SIZE
        ) + 1

        written, success = (
            process_batch(
                current_batch,
                batch_number,
                total_selected
            )
        )

        total_written += written

        if success:
            batches_completed += 1
        else:
            total_failed += 1

    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    total_time = (
        time.time()
        - pipeline_start
    )

    print()
    print("=" * 75)

    print(
        f"WORKER {WORKER_ID}/{TOTAL_WORKERS}"
        " SUMMARY"
    )

    print("=" * 75)

    print(
        f"Reviews selected : "
        f"{total_selected}"
    )

    print(
        f"Batch size       : "
        f"{BATCH_SIZE}"
    )

    print(
        f"Models available : "
        f"{len(MODELS)}"
    )

    print(
        f"Batches completed: "
        f"{batches_completed}"
    )

    print(
        f"Rows written     : "
        f"{total_written}"
    )

    print(
        f"Batches failed   : "
        f"{total_failed}"
    )

    print(
        f"Total time       : "
        f"{total_time:.2f} seconds"
    )

    print("=" * 75)

    if total_failed == 0:

        print(
            "STATUS: SUCCESS"
        )

    else:

        print(
            "STATUS: COMPLETED WITH "
            "FAILED BATCHES"
        )

        print(
            "Failed batches can be "
            "processed later."
        )

    print("=" * 75)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()