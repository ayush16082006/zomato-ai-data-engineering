"""
Zomato Review Model Benchmark

Compares:
    1. NVIDIA Nemotron 3 Super 120B
    2. Claude Sonnet 4.5

Benchmark design:
    - 100 reviews total
    - 25 reviews per batch
    - 4 batches per model
    - Same reviews are used for both models
    - Automatic retries for failed API requests
    - Per-batch timing
    - No writes to BigQuery

BigQuery:
    Project : zomato-ai-data-engineering
    Source  : raw.reviews

OmniRoute:
    http://localhost:20128/v1/chat/completions
"""

import os
import json
import time
from datetime import datetime
from pathlib import Path

import requests
from google.cloud import bigquery
from google.oauth2 import service_account


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ID = "zomato-ai-data-engineering"

SOURCE_TABLE = (
    f"{PROJECT_ID}.raw.reviews"
)

NUM_REVIEWS = 100

BATCH_SIZE = 25

MAX_RETRIES = 3

RETRY_DELAY_SECONDS = 5

OMNIROUTE_URL = (
    "http://localhost:20128/v1/chat/completions"
)


# ============================================================
# API KEY
# ============================================================

OMNIROUTE_API_KEY = os.getenv(
    "OMNIROUTE_API_KEY"
)

if not OMNIROUTE_API_KEY:
    raise RuntimeError(
        "\nOMNIROUTE_API_KEY environment variable "
        "is not set.\n\n"
        "Check it with:\n"
        "    echo %OMNIROUTE_API_KEY%\n"
    )


# ============================================================
# MODELS
# ============================================================

MODELS = {
    "nemotron": "nvidia/nemotron-3-super-120b-a12b",
    "claude": "claude-sonnet-4.5",
}


# ============================================================
# OUTPUT DIRECTORY
# ============================================================

OUTPUT_DIR = Path(
    "benchmark_results"
)

OUTPUT_DIR.mkdir(
    exist_ok=True
)


# ============================================================
# SYSTEM PROMPT
# ============================================================

SYSTEM_PROMPT = """
You are a review-enrichment system for a food delivery platform.

Analyze each customer review and return ONLY valid JSON.

For every review return:

{
  "review_id": "string",
  "sentiment_label": "positive|negative|neutral",
  "sentiment_score": number,
  "topic": "string",
  "key_issue": "string or null"
}

Rules:

1. sentiment_label must be exactly:
   positive
   negative
   neutral

2. sentiment_score must be between 0 and 1.

3. topic should describe the main subject of the review.

Examples:
   food_quality
   delivery
   service
   packaging
   price
   restaurant
   experience
   other

4. key_issue should contain the main problem mentioned
   in the review.

5. If there is no clear issue, use null.

6. Do not invent information.

7. Return ONLY a JSON array.

8. Return exactly one result for every input review.
"""


# ============================================================
# BIGQUERY CLIENT
# ============================================================

def get_bigquery_client():
    """
    Create a BigQuery client.

    If GOOGLE_APPLICATION_CREDENTIALS is set,
    use that service-account JSON file.

    Otherwise use Application Default Credentials.
    """

    credentials_path = os.getenv(
        "GOOGLE_APPLICATION_CREDENTIALS"
    )

    if credentials_path:

        credentials = (
            service_account
            .Credentials
            .from_service_account_file(
                credentials_path
            )
        )

        return bigquery.Client(
            project=PROJECT_ID,
            credentials=credentials,
        )

    return bigquery.Client(
        project=PROJECT_ID
    )


# ============================================================
# GET REVIEWS
# ============================================================

def get_reviews(
    client,
    limit=100,
):
    """
    Read a fixed set of reviews from BigQuery.

    These reviews are selected once and then reused
    for BOTH models.
    """

    query = f"""
        SELECT
            CAST(review_id AS STRING) AS review_id,
            CAST(rating AS INT64) AS rating,
            CAST(comment AS STRING) AS comment
        FROM `{SOURCE_TABLE}`
        WHERE comment IS NOT NULL
          AND TRIM(comment) != ''
        ORDER BY review_id
        LIMIT @limit
    """

    job_config = (
        bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ScalarQueryParameter(
                    "limit",
                    "INT64",
                    limit,
                )
            ]
        )
    )

    print("=" * 70)
    print("READING REVIEWS FROM BIGQUERY")
    print("=" * 70)

    rows = (
        client.query(
            query,
            job_config=job_config,
        )
        .result()
    )

    reviews = []

    for row in rows:

        reviews.append(
            {
                "review_id": str(
                    row.review_id
                ),
                "rating": row.rating,
                "comment": row.comment,
            }
        )

    print(
        f"Reviews loaded: {len(reviews)}"
    )

    return reviews


# ============================================================
# SPLIT INTO BATCHES
# ============================================================

def create_batches(
    reviews,
    batch_size,
):
    """
    Split reviews into batches.
    """

    batches = []

    for i in range(
        0,
        len(reviews),
        batch_size,
    ):

        batch = reviews[
            i:i + batch_size
        ]

        batches.append(batch)

    return batches


# ============================================================
# BUILD MODEL PROMPT
# ============================================================

def build_prompt(
    reviews,
):
    """
    Build the user prompt for a batch.
    """

    review_data = []

    for review in reviews:

        review_data.append(
            {
                "review_id": review[
                    "review_id"
                ],
                "rating": review[
                    "rating"
                ],
                "comment": review[
                    "comment"
                ],
            }
        )

    return f"""
Analyze the following Zomato customer reviews.

Return exactly one JSON object for each review.

Return ONLY a JSON array.

Reviews:

{json.dumps(
    review_data,
    ensure_ascii=False,
    indent=2
)}
"""


# ============================================================
# CALL OMNIROUTE
# ============================================================

def call_model(
    model_name,
    reviews,
):
    """
    Call one model for one batch.

    Returns:
        content
        elapsed_seconds
        raw_api_response
    """

    prompt = build_prompt(
        reviews
    )

    payload = {
        "model": model_name,

        "messages": [
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],

        "temperature": 0,
    }

    headers = {
        "Authorization": (
            f"Bearer {OMNIROUTE_API_KEY}"
        ),
        "Content-Type": "application/json",
    }

    start_time = time.perf_counter()

    response = requests.post(
        OMNIROUTE_URL,
        headers=headers,
        json=payload,
        timeout=600,
    )

    elapsed = (
        time.perf_counter()
        - start_time
    )

    response.raise_for_status()

    data = response.json()

    try:

        content = (
            data[
                "choices"
            ][0][
                "message"
            ][
                "content"
            ]
        )

    except (
        KeyError,
        IndexError,
        TypeError,
    ):

        raise RuntimeError(
            "Unexpected OmniRoute response:\n"
            + json.dumps(
                data,
                indent=2
            )
        )

    return (
        content,
        elapsed,
        data,
    )


# ============================================================
# CLEAN JSON RESPONSE
# ============================================================

def clean_json_response(
    text,
):
    """
    Remove Markdown code fences if a model
    returns them.
    """

    text = text.strip()

    if text.startswith(
        "```json"
    ):

        text = text[
            len("```json"):
        ]

    elif text.startswith(
        "```"
    ):

        text = text[
            len("```"):
        ]

    if text.endswith(
        "```"
    ):

        text = text[
            :-len("```")
        ]

    return text.strip()


# ============================================================
# VALIDATE MODEL RESULT
# ============================================================

def validate_results(
    raw_text,
    original_reviews,
):
    """
    Validate one model batch.

    Checks:
        - valid JSON
        - array
        - correct number of results
        - required fields
        - sentiment labels
        - sentiment score
        - review IDs
    """

    validation = {
        "json_valid": False,
        "is_array": False,

        "expected_count": len(
            original_reviews
        ),

        "actual_count": 0,

        "count_match": False,

        "required_fields_valid": False,

        "sentiment_values_valid": False,

        "sentiment_scores_valid": False,

        "review_ids_valid": False,

        "overall_valid": False,

        "errors": [],
    }

    # --------------------------------------------------------
    # Parse JSON
    # --------------------------------------------------------

    try:

        cleaned = (
            clean_json_response(
                raw_text
            )
        )

        parsed = json.loads(
            cleaned
        )

        validation[
            "json_valid"
        ] = True

    except Exception as e:

        validation[
            "errors"
        ].append(
            f"Invalid JSON: {str(e)}"
        )

        return (
            None,
            validation,
        )

    # --------------------------------------------------------
    # Check array
    # --------------------------------------------------------

    if not isinstance(
        parsed,
        list,
    ):

        validation[
            "errors"
        ].append(
            "Model output is not a JSON array."
        )

        return (
            None,
            validation,
        )

    validation[
        "is_array"
    ] = True

    validation[
        "actual_count"
    ] = len(parsed)

    # --------------------------------------------------------
    # Check result count
    # --------------------------------------------------------

    if (
        len(parsed)
        == len(original_reviews)
    ):

        validation[
            "count_match"
        ] = True

    else:

        validation[
            "errors"
        ].append(
            f"Expected "
            f"{len(original_reviews)} "
            f"results but received "
            f"{len(parsed)}."
        )

    # --------------------------------------------------------
    # Required fields
    # --------------------------------------------------------

    required_fields = {
        "review_id",
        "sentiment_label",
        "sentiment_score",
        "topic",
        "key_issue",
    }

    required_valid = True

    sentiment_valid = True

    score_valid = True

    expected_ids = {
        str(
            review["review_id"]
        )
        for review in original_reviews
    }

    returned_ids = set()

    # --------------------------------------------------------
    # Validate every result
    # --------------------------------------------------------

    for item in parsed:

        if not isinstance(
            item,
            dict,
        ):

            required_valid = False

            validation[
                "errors"
            ].append(
                "A result is not a JSON object."
            )

            continue

        missing = (
            required_fields
            - set(item.keys())
        )

        if missing:

            required_valid = False

            validation[
                "errors"
            ].append(
                f"Missing fields: "
                f"{missing}"
            )

        # ----------------------------------------------------
        # Sentiment
        # ----------------------------------------------------

        sentiment = item.get(
            "sentiment_label"
        )

        if sentiment not in {
            "positive",
            "negative",
            "neutral",
        }:

            sentiment_valid = False

        # ----------------------------------------------------
        # Score
        # ----------------------------------------------------

        score = item.get(
            "sentiment_score"
        )

        if not isinstance(
            score,
            (int, float),
        ):

            score_valid = False

        elif not (
            0 <= score <= 1
        ):

            score_valid = False

        # ----------------------------------------------------
        # Review ID
        # ----------------------------------------------------

        if (
            "review_id"
            in item
        ):

            returned_ids.add(
                str(
                    item[
                        "review_id"
                    ]
                )
            )

    # --------------------------------------------------------
    # Store validation results
    # --------------------------------------------------------

    validation[
        "required_fields_valid"
    ] = required_valid

    validation[
        "sentiment_values_valid"
    ] = sentiment_valid

    validation[
        "sentiment_scores_valid"
    ] = score_valid

    if (
        expected_ids
        == returned_ids
    ):

        validation[
            "review_ids_valid"
        ] = True

    else:

        validation[
            "errors"
        ].append(
            "Returned review IDs do not "
            "exactly match input review IDs."
        )

    validation[
        "overall_valid"
    ] = all(
        [
            validation[
                "json_valid"
            ],

            validation[
                "is_array"
            ],

            validation[
                "count_match"
            ],

            validation[
                "required_fields_valid"
            ],

            validation[
                "sentiment_values_valid"
            ],

            validation[
                "sentiment_scores_valid"
            ],

            validation[
                "review_ids_valid"
            ],
        ]
    )

    return (
        parsed,
        validation,
    )


# ============================================================
# CALL WITH RETRIES
# ============================================================

def call_model_with_retry(
    model_name,
    reviews,
    model_key,
    batch_number,
    total_batches,
):
    """
    Call a model with automatic retries.

    Maximum attempts:
        MAX_RETRIES + 1

    Example:
        MAX_RETRIES = 3

        Attempt 1
        Attempt 2
        Attempt 3
        Attempt 4
    """

    attempts = []

    for attempt in range(
        1,
        MAX_RETRIES + 2,
    ):

        print(
            f"\n    Attempt "
            f"{attempt}/"
            f"{MAX_RETRIES + 1}"
        )

        try:

            (
                raw_text,
                elapsed,
                api_response,
            ) = call_model(
                model_name,
                reviews,
            )

            (
                parsed_results,
                validation,
            ) = validate_results(
                raw_text,
                reviews,
            )

            attempts.append(
                {
                    "attempt": attempt,
                    "status": "success",
                    "seconds": elapsed,
                    "validation": validation,
                }
            )

            if validation[
                "overall_valid"
            ]:

                return {
                    "success": True,
                    "model": model_name,
                    "model_key": model_key,
                    "batch_number": batch_number,
                    "total_batches": total_batches,
                    "attempts": attempts,
                    "attempts_used": attempt,
                    "elapsed_seconds": elapsed,
                    "results": parsed_results,
                    "raw_response": raw_text,
                    "api_response": api_response,
                    "validation": validation,
                }

            # ------------------------------------------------
            # JSON/API response received but invalid
            # ------------------------------------------------

            print(
                "    Response received "
                "but validation failed."
            )

            if validation[
                "errors"
            ]:

                for error in validation[
                    "errors"
                ]:

                    print(
                        f"      - {error}"
                    )

            attempts[-1][
                "status"
            ] = "invalid"

        except Exception as e:

            error_text = str(e)

            print(
                f"    Request failed: "
                f"{error_text}"
            )

            attempts.append(
                {
                    "attempt": attempt,
                    "status": "failed",
                    "error": error_text,
                }
            )

        # ----------------------------------------------------
        # Retry
        # ----------------------------------------------------

        if attempt <= MAX_RETRIES:

            print(
                f"    Waiting "
                f"{RETRY_DELAY_SECONDS} "
                f"seconds before retry..."
            )

            time.sleep(
                RETRY_DELAY_SECONDS
            )

    # --------------------------------------------------------
    # All attempts failed
    # --------------------------------------------------------

    return {
        "success": False,
        "model": model_name,
        "model_key": model_key,
        "batch_number": batch_number,
        "total_batches": total_batches,
        "attempts": attempts,
        "attempts_used": len(attempts),
        "elapsed_seconds": None,
        "results": [],
        "raw_response": None,
        "api_response": None,
        "validation": {
            "overall_valid": False,
            "errors": [
                "All retry attempts failed."
            ],
        },
    }


# ============================================================
# BENCHMARK ONE MODEL
# ============================================================

def benchmark_model(
    model_name,
    model_key,
    batches,
):
    """
    Run all batches for one model.
    """

    print("\n\n")
    print("=" * 70)
    print(
        f"TESTING MODEL: {model_name}"
    )
    print("=" * 70)

    model_start = time.perf_counter()

    batch_results = []

    all_results = []

    successful_batches = 0

    failed_batches = 0

    # --------------------------------------------------------
    # Process every batch
    # --------------------------------------------------------

    for index, batch in enumerate(
        batches,
        start=1,
    ):

        print("\n" + "-" * 70)

        print(
            f"Batch {index}/"
            f"{len(batches)}"
        )

        print(
            f"Reviews: "
            f"{len(batch)}"
        )

        print(
            f"Review IDs: "
            f"{batch[0]['review_id']}"
            f" -> "
            f"{batch[-1]['review_id']}"
        )

        batch_start = time.perf_counter()

        result = call_model_with_retry(
            model_name=model_name,
            reviews=batch,
            model_key=model_key,
            batch_number=index,
            total_batches=len(batches),
        )

        batch_wall_time = (
            time.perf_counter()
            - batch_start
        )

        result[
            "batch_wall_time_seconds"
        ] = batch_wall_time

        batch_results.append(
            result
        )

        # ----------------------------------------------------
        # Successful batch
        # ----------------------------------------------------

        if result["success"]:

            successful_batches += 1

            all_results.extend(
                result["results"]
            )

            print(
                f"\n    SUCCESS"
            )

            print(
                f"    API time: "
                f"{result['elapsed_seconds']:.2f} sec"
            )

            print(
                f"    Wall time: "
                f"{batch_wall_time:.2f} sec"
            )

        # ----------------------------------------------------
        # Failed batch
        # ----------------------------------------------------

        else:

            failed_batches += 1

            print(
                f"\n    FAILED AFTER "
                f"{result['attempts_used']} "
                f"ATTEMPTS"
            )

    # --------------------------------------------------------
    # Total model time
    # --------------------------------------------------------

    total_wall_time = (
        time.perf_counter()
        - model_start
    )

    successful_reviews = len(
        all_results
    )

    average_per_successful_review = (
        total_wall_time
        / successful_reviews
        if successful_reviews
        else None
    )

    print("\n" + "=" * 70)

    print(
        f"MODEL COMPLETE: "
        f"{model_name}"
    )

    print("=" * 70)

    print(
        f"Total wall time     : "
        f"{total_wall_time:.2f} sec"
    )

    print(
        f"Batches successful  : "
        f"{successful_batches}/"
        f"{len(batches)}"
    )

    print(
        f"Batches failed      : "
        f"{failed_batches}/"
        f"{len(batches)}"
    )

    print(
        f"Reviews successful  : "
        f"{successful_reviews}/"
        f"{sum(len(b) for b in batches)}"
    )

    if average_per_successful_review:

        print(
            f"Avg wall time/review: "
            f"{average_per_successful_review:.3f} sec"
        )

    return {
        "model": model_name,
        "model_key": model_key,

        "reviews_requested": sum(
            len(b)
            for b in batches
        ),

        "reviews_successful": successful_reviews,

        "total_wall_time_seconds": (
            total_wall_time
        ),

        "average_wall_time_per_successful_review": (
            average_per_successful_review
        ),

        "successful_batches": (
            successful_batches
        ),

        "failed_batches": (
            failed_batches
        ),

        "batch_results": batch_results,

        "results": all_results,
    }


# ============================================================
# COMPARE OUTPUTS
# ============================================================

def compare_outputs(
    nemotron_results,
    claude_results,
):
    """
    Compare successful outputs for the same review IDs.
    """

    nemotron_by_id = {
        str(
            item["review_id"]
        ): item
        for item in nemotron_results
        if isinstance(
            item,
            dict,
        )
    }

    claude_by_id = {
        str(
            item["review_id"]
        ): item
        for item in claude_results
        if isinstance(
            item,
            dict,
        )
    }

    common_ids = sorted(
        set(
            nemotron_by_id
        )
        &
        set(
            claude_by_id
        )
    )

    comparison = []

    for review_id in common_ids:

        nemotron = (
            nemotron_by_id[
                review_id
            ]
        )

        claude = (
            claude_by_id[
                review_id
            ]
        )

        sentiment_match = (
            nemotron.get(
                "sentiment_label"
            )
            ==
            claude.get(
                "sentiment_label"
            )
        )

        topic_match = (
            nemotron.get(
                "topic"
            )
            ==
            claude.get(
                "topic"
            )
        )

        score_difference = None

        try:

            score_difference = abs(
                float(
                    nemotron[
                        "sentiment_score"
                    ]
                )
                -
                float(
                    claude[
                        "sentiment_score"
                    ]
                )
            )

        except Exception:
            pass

        comparison.append(
            {
                "review_id": review_id,

                "nemotron_sentiment": (
                    nemotron.get(
                        "sentiment_label"
                    )
                ),

                "claude_sentiment": (
                    claude.get(
                        "sentiment_label"
                    )
                ),

                "sentiment_match": (
                    sentiment_match
                ),

                "nemotron_topic": (
                    nemotron.get(
                        "topic"
                    )
                ),

                "claude_topic": (
                    claude.get(
                        "topic"
                    )
                ),

                "topic_match": (
                    topic_match
                ),

                "nemotron_score": (
                    nemotron.get(
                        "sentiment_score"
                    )
                ),

                "claude_score": (
                    claude.get(
                        "sentiment_score"
                    )
                ),

                "score_difference": (
                    score_difference
                ),

                "nemotron_key_issue": (
                    nemotron.get(
                        "key_issue"
                    )
                ),

                "claude_key_issue": (
                    claude.get(
                        "key_issue"
                    )
                ),
            }
        )

    return comparison


# ============================================================
# CALCULATE AGREEMENT
# ============================================================

def calculate_agreement(
    comparison,
):
    """
    Calculate sentiment and topic agreement.
    """

    compared = len(
        comparison
    )

    sentiment_matches = sum(
        1
        for row in comparison
        if row[
            "sentiment_match"
        ]
    )

    topic_matches = sum(
        1
        for row in comparison
        if row[
            "topic_match"
        ]
    )

    sentiment_agreement = (
        sentiment_matches
        / compared
        * 100
        if compared
        else 0
    )

    topic_agreement = (
        topic_matches
        / compared
        * 100
        if compared
        else 0
    )

    return {
        "reviews_compared": compared,

        "sentiment_matches": (
            sentiment_matches
        ),

        "sentiment_agreement_percent": (
            sentiment_agreement
        ),

        "topic_matches": (
            topic_matches
        ),

        "topic_agreement_percent": (
            topic_agreement
        ),
    }


# ============================================================
# PRINT BATCH TIMINGS
# ============================================================

def print_batch_timings(
    model_result,
):
    """
    Print a clean per-batch timing table.
    """

    print("\n")
    print("=" * 70)

    print(
        f"BATCH TIMINGS: "
        f"{model_result['model']}"
    )

    print("=" * 70)

    print(
        f"{'Batch':<10}"
        f"{'Status':<12}"
        f"{'Attempts':<10}"
        f"{'Time (sec)':<15}"
    )

    print("-" * 70)

    for batch in model_result[
        "batch_results"
    ]:

        status = (
            "SUCCESS"
            if batch["success"]
            else "FAILED"
        )

        time_value = (
            f"{batch['batch_wall_time_seconds']:.2f}"
        )

        print(
            f"{batch['batch_number']:<10}"
            f"{status:<12}"
            f"{batch['attempts_used']:<10}"
            f"{time_value:<15}"
        )


# ============================================================
# MAIN
# ============================================================

def main():

    print("\n")

    print("=" * 70)
    print(
        "ZOMATO REVIEW MODEL BENCHMARK"
    )
    print("=" * 70)

    print(
        f"Project       : "
        f"{PROJECT_ID}"
    )

    print(
        f"Source        : "
        f"{SOURCE_TABLE}"
    )

    print(
        f"Reviews       : "
        f"{NUM_REVIEWS}"
    )

    print(
        f"Batch size    : "
        f"{BATCH_SIZE}"
    )

    print(
        f"Batches/model : "
        f"{(NUM_REVIEWS + BATCH_SIZE - 1) // BATCH_SIZE}"
    )

    print(
        f"Max retries   : "
        f"{MAX_RETRIES}"
    )

    print(
        "Models        : "
        "Nemotron + Claude Sonnet 4.5"
    )

    print(
        "BigQuery write: DISABLED"
    )

    print("=" * 70)

    # ========================================================
    # BIGQUERY
    # ========================================================

    client = (
        get_bigquery_client()
    )

    # ========================================================
    # GET EXACT TEST SET
    # ========================================================

    reviews = get_reviews(
        client,
        NUM_REVIEWS,
    )

    if len(reviews) != NUM_REVIEWS:

        raise RuntimeError(
            f"Expected "
            f"{NUM_REVIEWS} reviews "
            f"but received "
            f"{len(reviews)}."
        )

    # ========================================================
    # CREATE BATCHES
    # ========================================================

    batches = create_batches(
        reviews,
        BATCH_SIZE,
    )

    print(
        f"\nCreated "
        f"{len(batches)} batches."
    )

    for i, batch in enumerate(
        batches,
        start=1,
    ):

        print(
            f"  Batch {i}: "
            f"{len(batch)} reviews"
        )

    # ========================================================
    # TIMESTAMP
    # ========================================================

    timestamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )

    # ========================================================
    # SAVE TEST SET
    # ========================================================

    test_set_file = (
        OUTPUT_DIR
        / f"benchmark_test_set_{timestamp}.json"
    )

    with open(
        test_set_file,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            reviews,
            f,
            ensure_ascii=False,
            indent=2,
        )

    print(
        f"\nTest set saved: "
        f"{test_set_file}"
    )

    # ========================================================
    # TEST NEMOTRON
    # ========================================================

    nemotron_result = benchmark_model(
        model_name=MODELS[
            "nemotron"
        ],

        model_key="nemotron",

        batches=batches,
    )

    print_batch_timings(
        nemotron_result
    )

    # ========================================================
    # TEST CLAUDE
    # ========================================================

    claude_result = benchmark_model(
        model_name=MODELS[
            "claude"
        ],

        model_key="claude",

        batches=batches,
    )

    print_batch_timings(
        claude_result
    )

    # ========================================================
    # COMPARE
    # ========================================================

    comparison = compare_outputs(
        nemotron_result[
            "results"
        ],

        claude_result[
            "results"
        ],
    )

    agreement = calculate_agreement(
        comparison
    )

    # ========================================================
    # SAVE FULL RESULT
    # ========================================================

    final_result = {
        "benchmark_timestamp": (
            datetime.now().isoformat()
        ),

        "configuration": {
            "project_id": PROJECT_ID,

            "source_table": (
                SOURCE_TABLE
            ),

            "reviews_tested": (
                NUM_REVIEWS
            ),

            "batch_size": (
                BATCH_SIZE
            ),

            "max_retries": (
                MAX_RETRIES
            ),

            "retry_delay_seconds": (
                RETRY_DELAY_SECONDS
            ),

            "omniroute_url": (
                OMNIROUTE_URL
            ),
        },

        "test_set": reviews,

        "nemotron": nemotron_result,

        "claude": claude_result,

        "agreement": agreement,

        "comparison": comparison,
    }

    result_file = (
        OUTPUT_DIR
        / f"benchmark_result_{timestamp}.json"
    )

    with open(
        result_file,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            final_result,
            f,
            ensure_ascii=False,
            indent=2,
        )

    # ========================================================
    # FINAL REPORT
    # ========================================================

    print("\n\n")

    print("=" * 70)
    print(
        "FINAL BENCHMARK REPORT"
    )
    print("=" * 70)

    # --------------------------------------------------------
    # Nemotron
    # --------------------------------------------------------

    print("\nNEMOTRON")
    print("-" * 70)

    print(
        f"Reviews successful : "
        f"{nemotron_result['reviews_successful']}/"
        f"{nemotron_result['reviews_requested']}"
    )

    print(
        f"Batches successful : "
        f"{nemotron_result['successful_batches']}/"
        f"{len(batches)}"
    )

    print(
        f"Total wall time    : "
        f"{nemotron_result['total_wall_time_seconds']:.2f} sec"
    )

    if (
        nemotron_result[
            "average_wall_time_per_successful_review"
        ]
        is not None
    ):

        print(
            f"Avg/review         : "
            f"{nemotron_result['average_wall_time_per_successful_review']:.3f} sec"
        )

    # --------------------------------------------------------
    # Claude
    # --------------------------------------------------------

    print("\nCLAUDE SONNET 4.5")
    print("-" * 70)

    print(
        f"Reviews successful : "
        f"{claude_result['reviews_successful']}/"
        f"{claude_result['reviews_requested']}"
    )

    print(
        f"Batches successful : "
        f"{claude_result['successful_batches']}/"
        f"{len(batches)}"
    )

    print(
        f"Total wall time    : "
        f"{claude_result['total_wall_time_seconds']:.2f} sec"
    )

    if (
        claude_result[
            "average_wall_time_per_successful_review"
        ]
        is not None
    ):

        print(
            f"Avg/review         : "
            f"{claude_result['average_wall_time_per_successful_review']:.3f} sec"
        )

    # --------------------------------------------------------
    # Agreement
    # --------------------------------------------------------

    print("\nMODEL AGREEMENT")
    print("-" * 70)

    print(
        f"Reviews compared   : "
        f"{agreement['reviews_compared']}"
    )

    print(
        f"Sentiment agreement: "
        f"{agreement['sentiment_agreement_percent']:.2f}%"
    )

    print(
        f"Topic agreement    : "
        f"{agreement['topic_agreement_percent']:.2f}%"
    )

    # --------------------------------------------------------
    # Files
    # --------------------------------------------------------

    print("\nFILES")
    print("-" * 70)

    print(
        f"Test set           : "
        f"{test_set_file}"
    )

    print(
        f"Benchmark result   : "
        f"{result_file}"
    )

    print("\n")

    print("=" * 70)
    print(
        "BENCHMARK COMPLETE"
    )
    print("=" * 70)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()