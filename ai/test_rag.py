"""
ZOMATO AI - COMPLETE RAG TEST SUITE
===================================

Purpose
-------
Production-oriented test suite for the NEW Qdrant-based RAG engine.

This test file is designed for the current architecture:

    User Question
          |
          v
    Qdrant Cloud Inference
          |
          v
    Qdrant Vector Search
          |
          v
    Progressive Retrieval
          |
          v
    Duplicate / Diversity Filtering
          |
          v
    BigQuery Evidence Retrieval
          |
          v
    Evidence-Aware LLM
          |
          v
    Final Answer


IMPORTANT
---------
This test file does NOT:

- load the old 300k embedding matrix
- load ai/rag_index/embeddings.npy
- load ai/rag_index/reviews.parquet
- call the old local embedding architecture
- call MMR
- call initialize_rag_engine()
- call check_ollama()

It uses only the current public RAG API:

    rag_engine.answer_review_question(question)

The purpose is to test the complete RAG pipeline end-to-end.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List

import pandas as pd

import rag_engine


# ============================================================================
# CONFIGURATION
# ============================================================================

# This is the target number of reviews the RAG engine attempts to provide.
# It is NOT treated as a hard requirement because duplicate-heavy evidence
# may legitimately result in fewer distinct reviews.
EXPECTED_TOP_K = 5


# ============================================================================
# TEST QUESTIONS
# ============================================================================

"""
The questions are intentionally varied.

They test:

1. Food quality
2. Delivery
3. Service
4. Pricing
5. Packaging
6. Positive experiences
7. Negative experiences
8. Complaints
9. High ratings
10. Low ratings
11. Overall experience
12. Cross-topic questions
13. Quality + delivery
14. Service + food
15. Price + quality
16. Delivery problems
17. Customer satisfaction
18. General review patterns
19. Evidence limitation
20. Broad customer opinion
21. Food taste
22. Delivery partner
23. Value for money
24. Common issues
25. Complete experience
"""


TEST_QUESTIONS: List[Dict[str, Any]] = [

    # ========================================================================
    # 1. FOOD QUALITY
    # ========================================================================

    {
        "number": 1,
        "category": "Food Quality",
        "question": (
            "What do customers say about food quality?"
        ),
    },

    {
        "number": 2,
        "category": "Food Taste",
        "question": (
            "What do customers say about the taste of the food?"
        ),
    },

    {
        "number": 3,
        "category": "Food Freshness",
        "question": (
            "Do customers mention anything about food freshness?"
        ),
    },

    # ========================================================================
    # 2. DELIVERY
    # ========================================================================

    {
        "number": 4,
        "category": "Delivery Experience",
        "question": (
            "What do customers say about the delivery experience?"
        ),
    },

    {
        "number": 5,
        "category": "Late Delivery",
        "question": (
            "What complaints do customers have about late delivery?"
        ),
    },

    {
        "number": 6,
        "category": "Delivery Partner",
        "question": (
            "What do customers say about delivery partners?"
        ),
    },

    # ========================================================================
    # 3. SERVICE
    # ========================================================================

    {
        "number": 7,
        "category": "Restaurant Service",
        "question": (
            "What do customers say about restaurant service?"
        ),
    },

    {
        "number": 8,
        "category": "Staff Experience",
        "question": (
            "What do customers say about staff behavior and service?"
        ),
    },

    # ========================================================================
    # 4. PRICE / VALUE
    # ========================================================================

    {
        "number": 9,
        "category": "Pricing",
        "question": (
            "What do customers say about food prices and value for money?"
        ),
    },

    {
        "number": 10,
        "category": "Value for Money",
        "question": (
            "Do customers feel they receive good value for the money?"
        ),
    },

    # ========================================================================
    # 5. PACKAGING
    # ========================================================================

    {
        "number": 11,
        "category": "Packaging",
        "question": (
            "What do customers say about food packaging?"
        ),
    },

    # ========================================================================
    # 6. POSITIVE EXPERIENCE
    # ========================================================================

    {
        "number": 12,
        "category": "Positive Experience",
        "question": (
            "What do customers like most about their Zomato orders?"
        ),
    },

    {
        "number": 13,
        "category": "Positive Feedback",
        "question": (
            "What are the main positive points mentioned by customers?"
        ),
    },

    # ========================================================================
    # 7. NEGATIVE EXPERIENCE
    # ========================================================================

    {
        "number": 14,
        "category": "Negative Experience",
        "question": (
            "What are the main complaints from customers?"
        ),
    },

    {
        "number": 15,
        "category": "Negative Feedback",
        "question": (
            "What are the main negative experiences mentioned in reviews?"
        ),
    },

    {
        "number": 16,
        "category": "Customer Problems",
        "question": (
            "What problems do customers commonly mention in their reviews?"
        ),
    },

    # ========================================================================
    # 8. RATINGS
    # ========================================================================

    {
        "number": 17,
        "category": "High Ratings",
        "question": (
            "What do customers with high ratings say about their experience?"
        ),
    },

    {
        "number": 18,
        "category": "Low Ratings",
        "question": (
            "What do customers with low ratings complain about?"
        ),
    },

    # ========================================================================
    # 9. COMBINED QUESTIONS
    # ========================================================================

    {
        "number": 19,
        "category": "Food + Delivery",
        "question": (
            "How do customers describe both food quality and delivery experience?"
        ),
    },

    {
        "number": 20,
        "category": "Food + Service",
        "question": (
            "How do customers describe food quality and restaurant service?"
        ),
    },

    {
        "number": 21,
        "category": "Price + Quality",
        "question": (
            "How do customers describe the relationship between food quality and price?"
        ),
    },

    {
        "number": 22,
        "category": "Delivery + Service",
        "question": (
            "What do customers say about delivery and service together?"
        ),
    },

    # ========================================================================
    # 10. BROADER QUESTIONS
    # ========================================================================

    {
        "number": 23,
        "category": "Overall Experience",
        "question": (
            "What are the main things customers like and dislike about their Zomato orders?"
        ),
    },

    {
        "number": 24,
        "category": "Overall Problems",
        "question": (
            "What are the most common problems mentioned in customer reviews?"
        ),
    },

    {
        "number": 25,
        "category": "Complete Experience",
        "question": (
            "What patterns can be seen across food quality, delivery, service, pricing, and packaging?"
        ),
    },
]


# ============================================================================
# PRINT HELPERS
# ============================================================================

def print_separator(
    character: str = "=",
    width: int = 90,
) -> None:
    """Print a consistent terminal separator."""

    print(character * width)


def print_header(
    title: str,
) -> None:
    """Print a large section header."""

    print()
    print_separator()
    print(title)
    print_separator()


# ============================================================================
# PRINT RETRIEVED REVIEWS
# ============================================================================

def print_retrieved_reviews(
    top_reviews: pd.DataFrame,
) -> None:
    """
    Print reviews returned by the current RAG engine.

    The number of retrieved reviews is informational.

    Fewer than EXPECTED_TOP_K reviews is NOT automatically treated
    as a failure because the dataset contains many duplicate review
    templates.
    """

    print()
    print("-" * 90)
    print("RETRIEVED EVIDENCE")
    print("-" * 90)

    if top_reviews is None:

        print(
            "No review dataframe was returned."
        )

        return

    if top_reviews.empty:

        print(
            "No sufficiently relevant reviews found."
        )

        return

    print(
        f"Retrieved distinct reviews: "
        f"{len(top_reviews)}"
    )

    for rank, (_, row) in enumerate(
        top_reviews.iterrows(),
        start=1,
    ):

        review_id = row.get(
            "review_id",
            "unknown",
        )

        rating = row.get(
            "rating",
            "unknown",
        )

        score = row.get(
            "score",
            None,
        )

        comment = row.get(
            "comment",
            "",
        )

        review_date = row.get(
            "review_date",
            "unknown",
        )

        print()
        print(
            f"[{rank}] Review ID : {review_id}"
        )

        print(
            f"    Rating    : {rating}"
        )

        if score is not None:

            try:

                print(
                    f"    Score     : "
                    f"{float(score):.4f}"
                )

            except (
                TypeError,
                ValueError,
            ):

                print(
                    f"    Score     : {score}"
                )

        print(
            f"    Date      : {review_date}"
        )

        print(
            f"    Comment   : {comment}"
        )

        # ------------------------------------------------------------
        # Optional enriched fields.
        #
        # These may be NaN because only part of the dataset has been
        # enriched by the current enrichment pipeline.
        # ------------------------------------------------------------

        sentiment = row.get(
            "sentiment_label",
            None,
        )

        sentiment_score = row.get(
            "sentiment_score",
            None,
        )

        topic = row.get(
            "topic",
            None,
        )

        key_issue = row.get(
            "key_issue",
            None,
        )

        if (
            sentiment is not None
            and not pd.isna(sentiment)
        ):

            print(
                f"    Sentiment  : {sentiment}"
            )

        if (
            sentiment_score is not None
            and not pd.isna(sentiment_score)
        ):

            print(
                f"    Sent. Score: {sentiment_score}"
            )

        if (
            topic is not None
            and not pd.isna(topic)
        ):

            print(
                f"    Topic      : {topic}"
            )

        if (
            key_issue is not None
            and not pd.isna(key_issue)
        ):

            print(
                f"    Key Issue  : {key_issue}"
            )

        print("-" * 90)


# ============================================================================
# CHECK RETRIEVED EVIDENCE
# ============================================================================

def validate_retrieved_reviews(
    top_reviews: pd.DataFrame,
) -> Dict[str, Any]:
    """
    Validate the structure of the returned evidence.

    This does NOT judge whether the answer is semantically correct.
    It only checks whether the RAG engine returned usable evidence.
    """

    diagnostics = {
        "valid": True,
        "review_count": 0,
        "missing_required_columns": [],
        "duplicate_review_ids": 0,
        "empty_comments": 0,
    }

    if top_reviews is None:

        diagnostics["valid"] = False

        return diagnostics

    if not isinstance(
        top_reviews,
        pd.DataFrame,
    ):

        diagnostics["valid"] = False

        return diagnostics

    diagnostics["review_count"] = len(
        top_reviews
    )

    required_columns = {
        "review_id",
        "rating",
        "comment",
        "score",
    }

    missing_columns = (
        required_columns
        - set(top_reviews.columns)
    )

    diagnostics[
        "missing_required_columns"
    ] = sorted(
        missing_columns
    )

    if missing_columns:

        diagnostics["valid"] = False

    if not top_reviews.empty:

        duplicate_ids = (
            top_reviews["review_id"]
            .duplicated()
            .sum()
        )

        diagnostics[
            "duplicate_review_ids"
        ] = int(
            duplicate_ids
        )

        if duplicate_ids > 0:

            diagnostics["valid"] = False

        empty_comments = (
            top_reviews["comment"]
            .fillna("")
            .astype(str)
            .str.strip()
            .eq("")
            .sum()
        )

        diagnostics[
            "empty_comments"
        ] = int(
            empty_comments
        )

    return diagnostics


# ============================================================================
# RUN ONE QUESTION
# ============================================================================

def run_question(
    item: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Execute one complete RAG test.

    Current API:

        rag_engine.answer_review_question(question)

    Returns structured test information.
    """

    number = item["number"]
    category = item["category"]
    question = item["question"]

    print()
    print()
    print_separator()

    print(
        f"RAG TEST QUESTION {number}"
    )

    print_separator()

    print()
    print(
        f"Category : {category}"
    )

    print()
    print(
        f"Question : {question}"
    )

    print()

    start_time = time.perf_counter()

    try:

        print(
            "Running Qdrant → BigQuery → "
            "Progressive Retrieval → LLM..."
        )

        # ------------------------------------------------------------
        # IMPORTANT:
        #
        # This is the ONLY public RAG call required.
        #
        # No local embeddings are loaded here.
        # ------------------------------------------------------------

        answer, top_reviews = (
            rag_engine.answer_review_question(
                question
            )
        )

        elapsed = (
            time.perf_counter()
            - start_time
        )

        print()
        print(
            "✓ RAG pipeline completed."
        )

    except Exception as exc:

        elapsed = (
            time.perf_counter()
            - start_time
        )

        print()
        print(
            "✗ RAG pipeline failed."
        )

        print(
            f"Error: "
            f"{type(exc).__name__}: {exc}"
        )

        return {
            "number": number,
            "category": category,
            "question": question,
            "success": False,
            "answer": "",
            "review_count": 0,
            "elapsed_seconds": elapsed,
            "error": (
                f"{type(exc).__name__}: {exc}"
            ),
        }

    # ------------------------------------------------------------------------
    # Evidence validation
    # ------------------------------------------------------------------------

    evidence_diagnostics = (
        validate_retrieved_reviews(
            top_reviews
        )
    )

    print()
    print(
        f"Retrieved evidence: "
        f"{len(top_reviews):,} distinct review(s)"
    )

    if len(top_reviews) < EXPECTED_TOP_K:

        print(
            "ℹ Fewer than 5 distinct reviews were "
            "returned. This is not automatically a "
            "failure because duplicate review templates "
            "are removed by the RAG engine."
        )

    if not evidence_diagnostics["valid"]:

        print(
            "⚠ Evidence structure validation "
            "found an issue."
        )

        if evidence_diagnostics[
            "missing_required_columns"
        ]:

            print(
                "  Missing columns: "
                + ", ".join(
                    evidence_diagnostics[
                        "missing_required_columns"
                    ]
                )
            )

        if (
            evidence_diagnostics[
                "duplicate_review_ids"
            ]
            > 0
        ):

            print(
                "  Duplicate review IDs detected: "
                f"{evidence_diagnostics['duplicate_review_ids']}"
            )

        if (
            evidence_diagnostics[
                "empty_comments"
            ]
            > 0
        ):

            print(
                "  Empty comments detected: "
                f"{evidence_diagnostics['empty_comments']}"
            )

    # ------------------------------------------------------------------------
    # Print retrieved evidence
    # ------------------------------------------------------------------------

    print_retrieved_reviews(
        top_reviews
    )

    # ------------------------------------------------------------------------
    # Print answer
    # ------------------------------------------------------------------------

    print()
    print_separator()

    print(
        "RAG ANSWER"
    )

    print_separator()

    print()

    print(answer)

    print()

    print(
        f"Execution time: "
        f"{elapsed:.2f} seconds"
    )

    # ------------------------------------------------------------------------
    # Success
    # ------------------------------------------------------------------------

    print()

    print(
        f"✓ Question {number} completed."
    )

    return {
        "number": number,
        "category": category,
        "question": question,
        "success": True,
        "answer": answer,
        "review_count": len(
            top_reviews
        ),
        "elapsed_seconds": elapsed,
        "evidence_valid": (
            evidence_diagnostics["valid"]
        ),
        "error": "",
    }


# ============================================================================
# PRINT TEST SUMMARY
# ============================================================================

def print_summary(
    results: List[Dict[str, Any]],
) -> None:
    """
    Print final RAG test summary.
    """

    total = len(results)

    successful = sum(
        1
        for result in results
        if result["success"]
    )

    failed = total - successful

    evidence_valid = sum(
        1
        for result in results
        if (
            result["success"]
            and result.get(
                "evidence_valid",
                False,
            )
        )
    )

    zero_evidence = sum(
        1
        for result in results
        if (
            result["success"]
            and result["review_count"] == 0
        )
    )

    total_time = sum(
        result.get(
            "elapsed_seconds",
            0.0,
        )
        for result in results
    )

    average_time = (
        total_time / total
        if total
        else 0.0
    )

    print()
    print()
    print_separator()

    print(
        "RAG TEST SUMMARY"
    )

    print_separator()

    print()

    print(
        f"Total questions        : {total}"
    )

    print(
        f"Successful questions   : {successful}"
    )

    print(
        f"Failed questions       : {failed}"
    )

    print(
        f"Valid evidence results : {evidence_valid}"
    )

    print(
        f"Zero-evidence results  : {zero_evidence}"
    )

    print(
        f"Total execution time   : "
        f"{total_time:.2f} seconds"
    )

    print(
        f"Average question time  : "
        f"{average_time:.2f} seconds"
    )

    # ------------------------------------------------------------------------
    # Per-question status
    # ------------------------------------------------------------------------

    print()
    print("-" * 90)
    print("QUESTION STATUS")
    print("-" * 90)

    for result in results:

        number = result["number"]
        category = result["category"]
        count = result["review_count"]
        elapsed = result[
            "elapsed_seconds"
        ]

        if result["success"]:

            status = "PASS"

        else:

            status = "FAIL"

        print(
            f"{number:02d}. "
            f"{status:<5} | "
            f"{category:<22} | "
            f"reviews={count:<2} | "
            f"time={elapsed:.2f}s"
        )

    # ------------------------------------------------------------------------
    # Failed questions
    # ------------------------------------------------------------------------

    failed_results = [
        result
        for result in results
        if not result["success"]
    ]

    if failed_results:

        print()
        print("-" * 90)
        print("FAILED QUESTIONS")
        print("-" * 90)

        for result in failed_results:

            print()
            print(
                f"Question {result['number']}: "
                f"{result['category']}"
            )

            print(
                f"Question: "
                f"{result['question']}"
            )

            print(
                f"Error: "
                f"{result['error']}"
            )

    # ------------------------------------------------------------------------
    # Final status
    # ------------------------------------------------------------------------

    print()

    if failed == 0:

        print_separator()

        print(
            "ALL RAG TEST QUESTIONS COMPLETED"
        )

        print_separator()

        print()

        print(
            "✓ Qdrant retrieval executed"
        )

        print(
            "✓ BigQuery evidence retrieval executed"
        )

        print(
            "✓ Progressive retrieval executed"
        )

        print(
            "✓ Duplicate filtering executed"
        )

        print(
            "✓ Diversity selection executed"
        )

        print(
            "✓ Evidence-aware LLM generation executed"
        )

        print(
            "✓ Multiple question categories tested"
        )

    else:

        print_separator()

        print(
            "RAG TEST COMPLETED WITH FAILURES"
        )

        print_separator()

        print()

        print(
            f"Successful: {successful}"
        )

        print(
            f"Failed    : {failed}"
        )


# ============================================================================
# MAIN
# ============================================================================

def main() -> None:
    """
    Run the complete RAG test suite.
    """

    print_separator()

    print(
        "ZOMATO AI - COMPLETE QDRANT RAG TEST SUITE"
    )

    print_separator()

    print()

    print(
        "Current RAG architecture:"
    )

    print(
        "  Question"
    )

    print(
        "      ↓"
    )

    print(
        "  Qdrant Cloud Inference"
    )

    print(
        "      ↓"
    )

    print(
        "  Progressive Retrieval"
    )

    print(
        "      ↓"
    )

    print(
        "  Duplicate / Diversity Filtering"
    )

    print(
        "      ↓"
    )

    print(
        "  BigQuery Evidence Retrieval"
    )

    print(
        "      ↓"
    )

    print(
        "  Ollama Cloud LLM"
    )

    print(
        "      ↓"
    )

    print(
        "  Final RAG Answer"
    )

    print()

    print(
        f"Number of test questions: "
        f"{len(TEST_QUESTIONS)}"
    )

    print(
        f"Expected final evidence target: "
        f"{EXPECTED_TOP_K}"
    )

    print()

    print(
        "IMPORTANT:"
    )

    print(
        "Fewer than 5 retrieved reviews is NOT "
        "automatically considered a failure."
    )

    print(
        "The RAG engine intentionally removes "
        "duplicate review templates."
    )

    print()

    # ========================================================================
    # RUN TESTS
    # ========================================================================

    results: List[
        Dict[str, Any]
    ] = []

    for item in TEST_QUESTIONS:

        result = run_question(
            item
        )

        results.append(
            result
        )

    # ========================================================================
    # FINAL SUMMARY
    # ========================================================================

    print_summary(
        results
    )


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":

    main()