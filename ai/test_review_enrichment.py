import os
import json
import time
import re
from datetime import datetime

import requests
from google.cloud import bigquery


# =============================================================================
# CONFIGURATION
# =============================================================================

PROJECT_ID = "zomato-ai-data-engineering"

SOURCE_TABLE = "zomato-ai-data-engineering.raw.reviews"

OMNIROUTE_URL = "http://localhost:20128/v1"

# Your new production combo
COMBO_NAME = "zomato_production"

# Number of normal reviews fetched from BigQuery
NORMAL_TEST_REVIEWS = 6

# Total requests sent to the combo
# We add a few deliberately difficult test cases below.
TOTAL_TEST_CASES = 10

OUTPUT_FILE = "test_review_enrichment_output.json"

TEMPERATURE = 0.0

REQUEST_TIMEOUT = 120


# =============================================================================
# EXPECTED JSON SCHEMA
# =============================================================================

REQUIRED_FIELDS = {
    "review_id",
    "sentiment_label",
    "sentiment_score",
    "topic",
    "key_issue",
}


# =============================================================================
# BIGQUERY
# =============================================================================

def get_bigquery_client():
    """
    Creates a BigQuery client using the current Google credentials.
    """

    return bigquery.Client(project=PROJECT_ID)


def fetch_test_reviews():
    """
    Fetch a small set of real reviews from BigQuery.

    Important:
    The raw.reviews table uses review_id, NOT id.
    """

    print("Fetching test reviews from BigQuery...")
    print(f"Table: {SOURCE_TABLE}")
    print()

    client = get_bigquery_client()

    query = f"""
        SELECT
            review_id,
            order_id,
            user_id,
            restaurant_id,
            rating,
            comment,
            review_date
        FROM `{SOURCE_TABLE}`
        WHERE comment IS NOT NULL
          AND TRIM(comment) != ''
        ORDER BY review_id
        LIMIT {NORMAL_TEST_REVIEWS}
    """

    rows = client.query(query).result()

    reviews = []

    for row in rows:
        reviews.append(
            {
                "review_id": str(row.review_id),
                "order_id": str(row.order_id),
                "user_id": str(row.user_id),
                "restaurant_id": str(row.restaurant_id),
                "rating": row.rating,
                "comment": row.comment,
                "review_date": (
                    str(row.review_date)
                    if row.review_date is not None
                    else None
                ),
            }
        )

    print(f"Reviews fetched: {len(reviews)}")
    print()

    return reviews


# =============================================================================
# DIFFICULT / EDGE CASE TEST REVIEWS
# =============================================================================

def create_edge_case_reviews():
    """
    These are deliberately difficult inputs.

    They are NOT written to BigQuery.

    Their purpose is to test whether the combo/models can handle:
      - very short reviews
      - mixed complaints
      - punctuation
      - neutral text
      - positive text
      - ambiguous wording
    """

    return [
        {
            "review_id": "TEST-EDGE-001",
            "order_id": None,
            "user_id": None,
            "restaurant_id": None,
            "rating": 1,
            "comment": "Late.",
            "review_date": None,
        },
        {
            "review_id": "TEST-EDGE-002",
            "order_id": None,
            "user_id": None,
            "restaurant_id": None,
            "rating": 5,
            "comment": "Amazing food!!! Fast delivery and excellent packaging.",
            "review_date": None,
        },
        {
            "review_id": "TEST-EDGE-003",
            "order_id": None,
            "user_id": None,
            "restaurant_id": None,
            "rating": 3,
            "comment": "Food was okay. Nothing special, but nothing terrible either.",
            "review_date": None,
        },
        {
            "review_id": "TEST-EDGE-004",
            "order_id": None,
            "user_id": None,
            "restaurant_id": None,
            "rating": 1,
            "comment": "Cold food, missing item, late delivery, terrible packaging, and expensive.",
            "review_date": None,
        },
    ]


# =============================================================================
# PROMPT
# =============================================================================

def build_prompt(review):
    """
    Builds the structured enrichment prompt.
    """

    comment = review.get("comment")

    prompt = f"""
You are a food-delivery review enrichment system.

Analyze ONE customer review.

Return ONLY valid JSON.
Do not use markdown.
Do not use ```json.
Do not add explanations before or after the JSON.

Required JSON structure:

{{
  "review_id": "{review['review_id']}",
  "sentiment_label": "positive|negative|neutral",
  "sentiment_score": -1.0,
  "topic": "delivery|food_quality|price|service|packaging|other",
  "key_issue": "short description or null"
}}

Rules:

1. sentiment_label must be exactly:
   positive
   negative
   neutral

2. sentiment_score must be a number between -1.0 and 1.0.

3. topic must be one of:
   delivery
   food_quality
   price
   service
   packaging
   other

4. key_issue must be short and directly supported by the review.

5. Do not invent information.

6. If the review does not contain a specific issue, key_issue may be null.

Review ID:
{review['review_id']}

Rating:
{review.get('rating')}

Customer comment:
{comment}
"""

    return prompt.strip()


# =============================================================================
# JSON EXTRACTION
# =============================================================================

def extract_json(text):
    """
    Attempts to extract JSON even if a model accidentally adds extra text.
    """

    if not text:
        raise ValueError("Empty model response")

    text = text.strip()

    # Direct JSON
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Remove markdown fences
    cleaned = re.sub(
        r"```(?:json)?",
        "",
        text,
        flags=re.IGNORECASE,
    )

    cleaned = cleaned.replace("```", "").strip()

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    # Find first JSON object
    start = cleaned.find("{")
    end = cleaned.rfind("}")

    if start != -1 and end != -1 and end > start:
        candidate = cleaned[start:end + 1]

        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass

    raise ValueError(
        f"Could not parse JSON from model response: {text[:500]}"
    )


# =============================================================================
# VALIDATION
# =============================================================================

def validate_result(result, expected_review_id):
    """
    Validates the enrichment result.
    """

    if not isinstance(result, dict):
        raise ValueError("Model response is not a JSON object")

    missing = REQUIRED_FIELDS - set(result.keys())

    if missing:
        raise ValueError(
            f"Missing required fields: {sorted(missing)}"
        )

    if str(result["review_id"]) != str(expected_review_id):
        raise ValueError(
            f"Wrong review_id. Expected {expected_review_id}, "
            f"got {result['review_id']}"
        )

    sentiment = result["sentiment_label"]

    if sentiment not in {"positive", "negative", "neutral"}:
        raise ValueError(
            f"Invalid sentiment_label: {sentiment}"
        )

    score = result["sentiment_score"]

    if not isinstance(score, (int, float)):
        raise ValueError(
            f"sentiment_score is not numeric: {score}"
        )

    if score < -1.0 or score > 1.0:
        raise ValueError(
            f"sentiment_score outside range: {score}"
        )

    allowed_topics = {
        "delivery",
        "food_quality",
        "price",
        "service",
        "packaging",
        "other",
    }

    if result["topic"] not in allowed_topics:
        raise ValueError(
            f"Invalid topic: {result['topic']}"
        )

    return True


# =============================================================================
# OMNIROUTE REQUEST
# =============================================================================

def send_to_omniroute(review):
    """
    Sends ONE review to the OmniRoute combo.

    Important:
    We intentionally do NOT implement repeated retries here.

    If the combo is configured correctly, OmniRoute should handle
    routing/failover between its configured models/providers.
    """

    api_key = os.getenv("OMNIROUTE_API_KEY")

    if not api_key:
        raise RuntimeError(
            "OMNIROUTE_API_KEY environment variable is not set."
        )

    url = f"{OMNIROUTE_URL}/chat/completions"

    prompt = build_prompt(review)

    payload = {
        "model": COMBO_NAME,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a strict JSON food-review enrichment "
                    "service. Return only valid JSON."
                ),
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
        "temperature": TEMPERATURE,
    }

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    started = time.perf_counter()

    try:

        response = requests.post(
            url,
            headers=headers,
            json=payload,
            timeout=REQUEST_TIMEOUT,
        )

        elapsed = time.perf_counter() - started

        print(
            f"    HTTP status: {response.status_code}"
        )

        print(
            f"    Response time: {elapsed:.2f} seconds"
        )

        if not response.ok:
            print(
                f"    ERROR RESPONSE:"
            )
            print(
                response.text[:2000]
            )

            response.raise_for_status()

        data = response.json()

        # OpenAI-compatible response
        choices = data.get("choices")

        if not choices:
            raise ValueError(
                "OmniRoute response does not contain choices"
            )

        message = choices[0].get("message", {})

        content = message.get("content")

        if not content:
            raise ValueError(
                "Model returned empty content"
            )

        return {
            "raw_content": content,
            "response_time": elapsed,
            "http_status": response.status_code,
            "omniroute_response": data,
        }

    except Exception as exc:

        elapsed = time.perf_counter() - started

        print(
            f"    REQUEST ERROR after {elapsed:.2f} seconds:"
        )

        print(
            f"    {type(exc).__name__}: {exc}"
        )

        raise


# =============================================================================
# SINGLE TEST
# =============================================================================

def process_one_review(review, index, total):
    """
    Processes one review and returns a structured test result.
    """

    print()
    print("-" * 80)
    print(
        f"TEST {index}/{total} "
        f"→ Review {review['review_id']}"
    )
    print("-" * 80)

    print(
        f"Rating : {review.get('rating')}"
    )

    print(
        f"Comment: {review.get('comment')}"
    )

    print()

    try:

        print("Sending to OmniRoute combo...")
        print(f"Combo: {COMBO_NAME}")

        response = send_to_omniroute(review)

        raw_content = response["raw_content"]

        print()
        print("RAW MODEL RESPONSE")
        print("=" * 80)
        print(raw_content)
        print("=" * 80)

        try:

            parsed = extract_json(raw_content)

            validate_result(
                parsed,
                review["review_id"],
            )

            print()
            print("✅ JSON VALID")

            print(
                f"Sentiment : {parsed['sentiment_label']}"
            )

            print(
                f"Score     : {parsed['sentiment_score']}"
            )

            print(
                f"Topic     : {parsed['topic']}"
            )

            print(
                f"Key issue : {parsed['key_issue']}"
            )

            return {
                "test_number": index,
                "review": review,
                "success": True,
                "response_time": response["response_time"],
                "result": parsed,
                "raw_response": raw_content,
                "error": None,
            }

        except Exception as exc:

            print()
            print("❌ JSON / VALIDATION ERROR")
            print(
                f"{type(exc).__name__}: {exc}"
            )

            return {
                "test_number": index,
                "review": review,
                "success": False,
                "response_time": response["response_time"],
                "result": None,
                "raw_response": raw_content,
                "error": (
                    f"{type(exc).__name__}: {exc}"
                ),
            }

    except Exception as exc:

        print()
        print("❌ OMNIROUTE REQUEST FAILED")
        print(
            f"{type(exc).__name__}: {exc}"
        )

        return {
            "test_number": index,
            "review": review,
            "success": False,
            "response_time": None,
            "result": None,
            "raw_response": None,
            "error": (
                f"{type(exc).__name__}: {exc}"
            ),
        }


# =============================================================================
# SAVE RESULTS
# =============================================================================

def save_results(results):
    """
    Saves the complete test report locally.
    """

    successful = sum(
        1 for x in results
        if x["success"]
    )

    failed = len(results) - successful

    response_times = [
        x["response_time"]
        for x in results
        if x["response_time"] is not None
    ]

    average_time = (
        sum(response_times) / len(response_times)
        if response_times
        else None
    )

    report = {
        "test_timestamp": datetime.now().isoformat(),
        "project": PROJECT_ID,
        "source_table": SOURCE_TABLE,
        "omniroute_url": OMNIROUTE_URL,
        "combo": COMBO_NAME,
        "total_tests": len(results),
        "successful_tests": successful,
        "failed_tests": failed,
        "average_response_time_seconds": average_time,
        "results": results,
    }

    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            report,
            f,
            indent=2,
            ensure_ascii=False,
            default=str,
        )

    print()
    print(
        f"Test report saved to: {OUTPUT_FILE}"
    )


# =============================================================================
# SUMMARY
# =============================================================================

def print_summary(results):
    """
    Prints final test summary.
    """

    total = len(results)

    successful = sum(
        1 for x in results
        if x["success"]
    )

    failed = total - successful

    response_times = [
        x["response_time"]
        for x in results
        if x["response_time"] is not None
    ]

    print()
    print()
    print("=" * 80)
    print("COMBO TEST SUMMARY")
    print("=" * 80)

    print(
        f"Combo              : {COMBO_NAME}"
    )

    print(
        f"Total tests        : {total}"
    )

    print(
        f"Successful         : {successful}"
    )

    print(
        f"Failed             : {failed}"
    )

    if response_times:

        print(
            f"Average response  : "
            f"{sum(response_times) / len(response_times):.2f} sec"
        )

        print(
            f"Fastest response  : "
            f"{min(response_times):.2f} sec"
        )

        print(
            f"Slowest response  : "
            f"{max(response_times):.2f} sec"
        )

    print()
    print("Individual results:")
    print("-" * 80)

    for result in results:

        review_id = result["review"]["review_id"]

        if result["success"]:

            enrichment = result["result"]

            print(
                f"✅ {review_id} | "
                f"{enrichment['sentiment_label']} | "
                f"{enrichment['sentiment_score']} | "
                f"{enrichment['topic']}"
            )

        else:

            print(
                f"❌ {review_id} | "
                f"{result['error']}"
            )

    print()
    print("=" * 80)

    if failed == 0:

        print(
            "🎉 ALL TESTS PASSED"
        )

    else:

        print(
            f"⚠️ {failed} TEST(S) FAILED"
        )

    print("=" * 80)

    print()
    print(
        "IMPORTANT:"
    )

    print(
        "This test script does NOT write anything to "
        "BigQuery ai.review_enriched."
    )

    print(
        "It only tests the OmniRoute combo."
    )


# =============================================================================
# MAIN
# =============================================================================

def main():

    print("=" * 80)
    print("ZOMATO REVIEW ENRICHMENT - COMBO TEST")
    print("=" * 80)

    print()
    print(
        f"Project       : {PROJECT_ID}"
    )

    print(
        f"Source        : {SOURCE_TABLE}"
    )

    print(
        f"OmniRoute     : {OMNIROUTE_URL}"
    )

    print(
        f"Combo         : {COMBO_NAME}"
    )

    print(
        f"Normal reviews: {NORMAL_TEST_REVIEWS}"
    )

    print(
        f"Edge cases    : 4"
    )

    print(
        f"Total tests   : {TOTAL_TEST_CASES}"
    )

    print()

    # -------------------------------------------------------------------------
    # Fetch real reviews
    # -------------------------------------------------------------------------

    reviews = fetch_test_reviews()

    if not reviews:

        print(
            "❌ No reviews were fetched from BigQuery."
        )

        return

    # -------------------------------------------------------------------------
    # Add difficult test cases
    # -------------------------------------------------------------------------

    edge_cases = create_edge_case_reviews()

    test_reviews = reviews + edge_cases

    # Ensure total does not exceed requested number
    test_reviews = test_reviews[:TOTAL_TEST_CASES]

    print()
    print(
        f"Total test cases prepared: {len(test_reviews)}"
    )

    # -------------------------------------------------------------------------
    # Process every review
    # -------------------------------------------------------------------------

    results = []

    for index, review in enumerate(
        test_reviews,
        start=1,
    ):

        result = process_one_review(
            review,
            index,
            len(test_reviews),
        )

        results.append(result)

    # -------------------------------------------------------------------------
    # Save report
    # -------------------------------------------------------------------------

    save_results(results)

    # -------------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------------

    print_summary(results)


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    main()