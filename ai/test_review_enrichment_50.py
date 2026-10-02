import os
import json
import re
import time
from datetime import datetime

from google.cloud import bigquery
from openai import OpenAI


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ID = "zomato-ai-data-engineering"

# Your existing BigQuery reviews table
SOURCE_TABLE = f"{PROJECT_ID}.raw.reviews"

# OmniRoute running locally
OMNIROUTE_BASE_URL = "http://localhost:20128/v1"

# IMPORTANT:
# Put the API key in the OMNIROUTE_API_KEY environment variable.
OMNIROUTE_API_KEY = os.getenv("OMNIROUTE_API_KEY")

# Number of reviews for this first test
TEST_REVIEW_COUNT = 50

# The Combo you created in OmniRoute
# If your Combo has a different model/ID exposed by OmniRoute,
# change this after checking /v1/models.
COMBO_NAME = "zomato-review-enrichment"


# ============================================================
# CHECK API KEY
# ============================================================

if not OMNIROUTE_API_KEY:
    raise RuntimeError(
        "OMNIROUTE_API_KEY is not set.\n"
        "In CMD run:\n"
        "set OMNIROUTE_API_KEY=YOUR_OMNIROUTE_API_KEY"
    )


# ============================================================
# CLIENTS
# ============================================================

bq_client = bigquery.Client(project=PROJECT_ID)

omniroute_client = OpenAI(
    api_key=OMNIROUTE_API_KEY,
    base_url=OMNIROUTE_BASE_URL,
)


# ============================================================
# STEP 1: GET REVIEWS FROM BIGQUERY
# ============================================================

def get_test_reviews():

    query = f"""
        WITH positive_reviews AS (
            SELECT
                CAST(review_id AS STRING) AS review_id,
                CAST(order_id AS STRING) AS order_id,
                CAST(user_id AS STRING) AS user_id,
                CAST(restaurant_id AS STRING) AS restaurant_id,
                CAST(rating AS INT64) AS rating,
                CAST(comment AS STRING) AS comment,
                CAST(review_date AS DATE) AS review_date
            FROM `{SOURCE_TABLE}`
            WHERE comment IS NOT NULL
              AND TRIM(comment) != ''
              AND SAFE_CAST(rating AS INT64) >= 4
            ORDER BY RAND()
            LIMIT 10
        ),

        negative_reviews AS (
            SELECT
                CAST(review_id AS STRING) AS review_id,
                CAST(order_id AS STRING) AS order_id,
                CAST(user_id AS STRING) AS user_id,
                CAST(restaurant_id AS STRING) AS restaurant_id,
                CAST(rating AS INT64) AS rating,
                CAST(comment AS STRING) AS comment,
                CAST(review_date AS DATE) AS review_date
            FROM `{SOURCE_TABLE}`
            WHERE comment IS NOT NULL
              AND TRIM(comment) != ''
              AND SAFE_CAST(rating AS INT64) <= 2
            ORDER BY RAND()
            LIMIT 10
        ),

        medium_reviews AS (
            SELECT
                CAST(review_id AS STRING) AS review_id,
                CAST(order_id AS STRING) AS order_id,
                CAST(user_id AS STRING) AS user_id,
                CAST(restaurant_id AS STRING) AS restaurant_id,
                CAST(rating AS INT64) AS rating,
                CAST(comment AS STRING) AS comment,
                CAST(review_date AS DATE) AS review_date
            FROM `{SOURCE_TABLE}`
            WHERE comment IS NOT NULL
              AND TRIM(comment) != ''
              AND SAFE_CAST(rating AS INT64) = 3
            ORDER BY RAND()
            LIMIT 10
        ),

        random_reviews AS (
            SELECT
                CAST(review_id AS STRING) AS review_id,
                CAST(order_id AS STRING) AS order_id,
                CAST(user_id AS STRING) AS user_id,
                CAST(restaurant_id AS STRING) AS restaurant_id,
                CAST(rating AS INT64) AS rating,
                CAST(comment AS STRING) AS comment,
                CAST(review_date AS DATE) AS review_date
            FROM `{SOURCE_TABLE}`
            WHERE comment IS NOT NULL
              AND TRIM(comment) != ''
            ORDER BY RAND()
            LIMIT 20
        )

        SELECT * FROM positive_reviews
        UNION ALL
        SELECT * FROM negative_reviews
        UNION ALL
        SELECT * FROM medium_reviews
        UNION ALL
        SELECT * FROM random_reviews
        LIMIT 50
    """

    print("\nFetching mixed reviews from BigQuery...")
    print(f"Table: {SOURCE_TABLE}")

    query_job = bq_client.query(query)
    rows = list(query_job.result())

    reviews = []

    for row in rows:
        reviews.append({
            "review_id": row.review_id,
            "order_id": row.order_id,
            "user_id": row.user_id,
            "restaurant_id": row.restaurant_id,
            "rating": row.rating,
            "comment": row.comment,
            "review_date": (
                str(row.review_date)
                if row.review_date is not None
                else None
            ),
        })

    print(f"Reviews fetched: {len(reviews)}")

    return reviews


# ============================================================
# STEP 2: BUILD PROMPT
# ============================================================

def build_prompt(reviews):

    review_text = []

    for review in reviews:
        review_text.append(
            f"""
Review ID: {review["review_id"]}
Star Rating: {review["rating"]}
Review: {review["comment"]}
""".strip()
        )

    joined_reviews = "\n\n---\n\n".join(review_text)

    prompt = f"""
You are a review-analysis system for a food-delivery platform.

Analyze each review independently.

For every review, identify:

1. sentiment_label
   - positive
   - negative
   - neutral

2. sentiment_score
   - number between -1.0 and 1.0
   - -1.0 = extremely negative
   - 0.0 = neutral
   - 1.0 = extremely positive

3. topic
   Choose the main topic of the review.
   Examples:
   - food_quality
   - taste
   - delivery
   - packaging
   - price
   - service
   - restaurant
   - hygiene
   - quantity
   - app
   - other

4. key_issue
   - Short description of the main problem or important point.
   - If there is no specific issue, use null.

IMPORTANT:
Return ONLY valid JSON.
Do not use Markdown.
Do not put the JSON inside ```json fences.

Return exactly this structure:

{{
  "results": [
    {{
      "review_id": "string",
      "sentiment_label": "positive|negative|neutral",
      "sentiment_score": 0.0,
      "topic": "string",
      "key_issue": "string or null"
    }}
  ]
}}

Here are the reviews:

{joined_reviews}
"""

    return prompt


# ============================================================
# STEP 3: CLEAN POSSIBLE MARKDOWN JSON
# ============================================================

def clean_json_response(text):

    text = text.strip()

    # Remove ```json ... ``` if a model ignores our instruction
    text = re.sub(r"^```json\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"^```\s*", "", text)
    text = re.sub(r"\s*```$", "", text)

    return text.strip()


# ============================================================
# STEP 4: CALL OMNIROUTE
# ============================================================

def call_omniroute(prompt):

    print("\nSending reviews to OmniRoute...")
    print(f"Combo: {COMBO_NAME}")

    start_time = time.time()

    response = omniroute_client.chat.completions.create(
        model=COMBO_NAME,
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0
    )

    elapsed = time.time() - start_time

    print(f"OmniRoute response time: {elapsed:.2f} seconds")

    content = response.choices[0].message.content

    if not content:
        raise ValueError("OmniRoute returned an empty response.")

    return content, elapsed


# ============================================================
# STEP 5: VALIDATE JSON
# ============================================================

def validate_result(raw_response, original_reviews):

    cleaned = clean_json_response(raw_response)

    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError as e:
        print("\n❌ INVALID JSON")
        print("JSON error:", e)
        return None

    if not isinstance(data, dict):
        print("\n❌ Response is not a JSON object.")
        return None

    if "results" not in data:
        print("\n❌ Missing 'results' field.")
        return None

    results = data["results"]

    if not isinstance(results, list):
        print("\n❌ 'results' must be a list.")
        return None

    expected_ids = {
        str(review["review_id"])
        for review in original_reviews
    }

    returned_ids = set()

    for item in results:

        required_fields = [
            "review_id",
            "sentiment_label",
            "sentiment_score",
            "topic",
            "key_issue",
        ]

        for field in required_fields:
            if field not in item:
                print(
                    f"\n❌ Missing field '{field}' "
                    f"in review {item.get('review_id')}"
                )
                return None

        returned_ids.add(str(item["review_id"]))

        if item["sentiment_label"] not in [
            "positive",
            "negative",
            "neutral"
        ]:
            print(
                f"\n❌ Invalid sentiment label: "
                f"{item['sentiment_label']}"
            )
            return None

        try:
            score = float(item["sentiment_score"])

            if score < -1 or score > 1:
                print(
                    f"\n❌ Invalid sentiment score: {score}"
                )
                return None

        except (ValueError, TypeError):
            print(
                f"\n❌ Invalid sentiment score: "
                f"{item['sentiment_score']}"
            )
            return None

    missing_ids = expected_ids - returned_ids

    if missing_ids:
        print("\n⚠️ Some reviews were missing from the response:")
        for review_id in missing_ids:
            print("   ", review_id)

    extra_ids = returned_ids - expected_ids

    if extra_ids:
        print("\n⚠️ Unexpected review IDs returned:")
        for review_id in extra_ids:
            print("   ", review_id)

    print("\n✅ JSON structure is valid.")

    return data


# ============================================================
# STEP 6: PRINT RESULTS
# ============================================================

def print_results(data):

    print("\n")
    print("=" * 80)
    print("ENRICHMENT RESULTS")
    print("=" * 80)

    for item in data["results"]:

        print("\nReview ID:", item["review_id"])
        print("Sentiment:", item["sentiment_label"])
        print("Score:", item["sentiment_score"])
        print("Topic:", item["topic"])
        print("Key Issue:", item["key_issue"])

    print("\n" + "=" * 80)


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 80)
    print("ZOMATO REVIEW ENRICHMENT - OMNIROUTE TEST")
    print("=" * 80)

    print("\nProject:", PROJECT_ID)
    print("Source:", SOURCE_TABLE)
    print("OmniRoute:", OMNIROUTE_BASE_URL)
    print("Combo:", COMBO_NAME)
    print("Test reviews:", TEST_REVIEW_COUNT)

    # 1. Get reviews
    reviews = get_test_reviews()

    if not reviews:
        raise RuntimeError(
            "No reviews were found in BigQuery."
        )

    # Show what we are sending
    print("\nSample reviews:")
    print("-" * 80)

    for review in reviews:
        print(
            f'\nID: {review["review_id"]}'
            f'\nRating: {review["rating"]}'
            f'\nComment: {review["comment"]}'
        )

    # 2. Build prompt
    prompt = build_prompt(reviews)

    # 3. Call OmniRoute
    raw_response, elapsed = call_omniroute(prompt)

    # 4. Print raw response
    print("\n")
    print("=" * 80)
    print("RAW MODEL RESPONSE")
    print("=" * 80)
    print(raw_response)

    # 5. Validate JSON
    data = validate_result(
        raw_response,
        reviews
    )

    if data is None:
        print("\n❌ TEST FAILED")
        print("The model response needs to be fixed before scaling.")
        return

    # 6. Print clean results
    print_results(data)

    # 7. Save test output locally
    output_file = "test_review_enrichment_output.json"

    with open(
        output_file,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            {
                "timestamp": datetime.now().isoformat(),
                "combo": COMBO_NAME,
                "review_count": len(reviews),
                "response_time_seconds": round(elapsed, 2),
                "results": data["results"],
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    print(
        f"\n✅ Test output saved to: {output_file}"
    )

    print("\n🎉 SMALL TEST COMPLETED SUCCESSFULLY")
    print(
        "\nIMPORTANT: Nothing was written to "
        "BigQuery ai.review_enriched yet."
    )


if __name__ == "__main__":
    main()