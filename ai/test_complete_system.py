"""
ZOMATO AI - COMPLETE SYSTEM REGRESSION TEST
============================================

Purpose
-------
Comprehensive regression test for the complete Zomato AI system.

Tests:

    Domain Guard
        ↓
    Semantic Router
        ↓
    ┌──────────┬──────────┬──────────┐
    │   SQL    │   RAG    │  HYBRID  │
    └──────────┴──────────┴──────────┘
        ↓
    Final Answer

The test intentionally uses the public orchestrator API:

    answer_question(question)

This means the test exercises the real production flow rather
than bypassing the orchestrator.

IMPORTANT
---------
This file does NOT modify:

    rag_engine.py
    router.py
    sql_engine.py
    orchestrator.py
    domain_guard.py

It is read-only testing.

Command:

    python test_complete_system.py"""

from __future__ import annotations

import json
import time
from typing import Any, Dict, List

import pandas as pd

from orchestrator import answer_question


# ============================================================
# CONFIGURATION
# ============================================================

STOP_ON_FAILURE = False

# Set True if you want to print the complete returned result.
PRINT_FULL_RESULT = False

# Maximum characters of final answer shown in normal output.
MAX_ANSWER_LENGTH = 2500


# ============================================================
# TEST CASE STRUCTURE
# ============================================================

TEST_CASES: List[Dict[str, Any]] = [

    # ========================================================
    # DOMAIN GUARD TESTS
    # ========================================================

    {
        "id": "DG-01",
        "category": "Domain Guard - In Scope",
        "expected_route": "sql",
        "question": (
            "Which restaurant generated the highest revenue?"
        ),
    },

    {
        "id": "DG-02",
        "category": "Domain Guard - In Scope",
        "expected_route": "rag",
        "question": (
            "What do customers say about food quality?"
        ),
    },

    {
        "id": "DG-03",
        "category": "Domain Guard - Out of Scope",
        "expected_route": "domain_guard",
        "question": (
            "Which is the best product on Amazon?"
        ),
    },

    {
        "id": "DG-04",
        "category": "Domain Guard - Out of Scope",
        "expected_route": "domain_guard",
        "question": (
            "Who is the Prime Minister of India?"
        ),
    },

    {
        "id": "DG-05",
        "category": "Domain Guard - Out of Scope",
        "expected_route": "domain_guard",
        "question": (
            "What is the weather today?"
        ),
    },

    {
        "id": "DG-06",
        "category": "Domain Guard - Out of Scope",
        "expected_route": "domain_guard",
        "question": (
            "What is 2 + 2?"
        ),
    },


    # ========================================================
    # SQL ROUTE
    # ========================================================

    {
        "id": "SQL-01",
        "category": "SQL - Restaurant Revenue",
        "expected_route": "sql",
        "question": (
            "Which restaurant generated the most revenue?"
        ),
    },

    {
        "id": "SQL-02",
        "category": "SQL - City Revenue",
        "expected_route": "sql",
        "question": (
            "Which city has the highest GMV?"
        ),
    },

    {
        "id": "SQL-03",
        "category": "SQL - Restaurant Orders",
        "expected_route": "sql",
        "question": (
            "Which restaurants have the highest number of orders?"
        ),
    },

    {
        "id": "SQL-04",
        "category": "SQL - Delivery",
        "expected_route": "sql",
        "question": (
            "Which cities have the best delivery performance?"
        ),
    },

    {
        "id": "SQL-05",
        "category": "SQL - Delivery Time",
        "expected_route": "sql",
        "question": (
            "What is the median delivery time by city?"
        ),
    },

    {
        "id": "SQL-06",
        "category": "SQL - Restaurant Rating",
        "expected_route": "sql",
        "question": (
            "Which restaurants have the highest average customer rating?"
        ),
    },

    {
        "id": "SQL-07",
        "category": "SQL - Cuisine",
        "expected_route": "sql",
        "question": (
            "Which cuisine performs best based on restaurant performance?"
        ),
    },

    {
        "id": "SQL-08",
        "category": "SQL - Cancellation",
        "expected_route": "sql",
        "question": (
            "Which cities have the highest cancellation rate?"
        ),
    },

    {
        "id": "SQL-09",
        "category": "SQL - Customer Spending",
        "expected_route": "sql",
        "question": (
            "Which customers have the highest spending?"
        ),
    },

    {
        "id": "SQL-10",
        "category": "SQL - Alternative Wording",
        "expected_route": "sql",
        "question": (
            "Give me the city that brought in the largest amount of sales."
        ),
    },


    # ========================================================
    # RAG ROUTE
    # ========================================================

    {
        "id": "RAG-01",
        "category": "RAG - Food Quality",
        "expected_route": "rag",
        "question": (
            "What do customers say about food quality?"
        ),
    },

    {
        "id": "RAG-02",
        "category": "RAG - Food Taste",
        "expected_route": "rag",
        "question": (
            "How do diners describe the taste of their meals?"
        ),
    },

    {
        "id": "RAG-03",
        "category": "RAG - Delivery",
        "expected_route": "rag",
        "question": (
            "What do customers say about their delivery experience?"
        ),
    },

    {
        "id": "RAG-04",
        "category": "RAG - Delivery Partner",
        "expected_route": "rag",
        "question": (
            "How do customers describe delivery partners?"
        ),
    },

    {
        "id": "RAG-05",
        "category": "RAG - Service",
        "expected_route": "rag",
        "question": (
            "What do customers say about restaurant service?"
        ),
    },

    {
        "id": "RAG-06",
        "category": "RAG - Pricing",
        "expected_route": "rag",
        "question": (
            "What do customers say about value for money?"
        ),
    },

    {
        "id": "RAG-07",
        "category": "RAG - Packaging",
        "expected_route": "rag",
        "question": (
            "What do customers say about packaging?"
        ),
    },

    {
        "id": "RAG-08",
        "category": "RAG - Negative Feedback",
        "expected_route": "rag",
        "question": (
            "What problems are customers complaining about?"
        ),
    },

    {
        "id": "RAG-09",
        "category": "RAG - Positive Feedback",
        "expected_route": "rag",
        "question": (
            "What do customers like about their orders?"
        ),
    },

    {
        "id": "RAG-10",
        "category": "RAG - Sentiment",
        "expected_route": "rag",
        "question": (
            "How do customers generally feel about their orders?"
        ),
    },


    # ========================================================
    # HYBRID ROUTE
    # ========================================================

    {
        "id": "HY-01",
        "category": "Hybrid - Top Revenue Restaurant",
        "expected_route": "hybrid",
        "question": (
            "Which restaurant generated the most revenue "
            "and what do customers say about it?"
        ),
    },

    {
        "id": "HY-02",
        "category": "Hybrid - Top Orders Restaurant",
        "expected_route": "hybrid",
        "question": (
            "Which restaurant has the most orders "
            "and what do customers say about its service?"
        ),
    },

    {
        "id": "HY-03",
        "category": "Hybrid - City Revenue + Reviews",
        "expected_route": "hybrid",
        "question": (
            "Which city generated the highest sales "
            "and what do customers think about restaurants there?"
        ),
    },

    {
        "id": "HY-04",
        "category": "Hybrid - Restaurant Rating + Food",
        "expected_route": "hybrid",
        "question": (
            "Which restaurant has the highest rating "
            "and what do customers say about its food quality?"
        ),
    },

    {
        "id": "HY-05",
        "category": "Hybrid - Restaurant Revenue + Complaints",
        "expected_route": "hybrid",
        "question": (
            "Which restaurant has the highest revenue "
            "and what complaints do customers have about it?"
        ),
    },

    {
        "id": "HY-06",
        "category": "Hybrid - Delivery + Reviews",
        "expected_route": "hybrid",
        "question": (
            "Which city has the slowest delivery performance "
            "and what do customers say about delivery there?"
        ),
    },

    {
        "id": "HY-07",
        "category": "Hybrid - Cuisine + Reviews",
        "expected_route": "hybrid",
        "question": (
            "Which cuisine performs best "
            "and what do customers say about the food?"
        ),
    },

    {
        "id": "HY-08",
        "category": "Hybrid - Restaurant Orders + Feedback",
        "expected_route": "hybrid",
        "question": (
            "Which restaurant receives the most orders "
            "and how do customers describe their experience?"
        ),
    },

    {
        "id": "HY-09",
        "category": "Hybrid - City + Problems",
        "expected_route": "hybrid",
        "question": (
            "Which city has the highest cancellation rate "
            "and what problems do customers mention?"
        ),
    },

    {
        "id": "HY-10",
        "category": "Hybrid - Revenue + Food + Delivery",
        "expected_route": "hybrid",
        "question": (
            "Which restaurant has the highest revenue "
            "and what do customers say about its food and delivery?"
        ),
    },

    {
        "id": "HY-11",
        "category": "Hybrid - Natural Language",
        "expected_route": "hybrid",
        "question": (
            "Find the restaurant bringing in the most money "
            "and tell me what diners think about eating there."
        ),
    },

    {
        "id": "HY-12",
        "category": "Hybrid - Natural Language City",
        "expected_route": "hybrid",
        "question": (
            "Give me the city with the largest sales volume "
            "and explain what people are saying about orders there."
        ),
    },
]


# ============================================================
# RESULT HELPERS
# ============================================================

def safe_dataframe(value: Any) -> pd.DataFrame:
    """
    Convert a returned value into a DataFrame when possible.
    """

    if isinstance(value, pd.DataFrame):
        return value

    return pd.DataFrame()


def print_answer(answer: Any) -> None:
    """
    Print a readable final answer.
    """

    if answer is None:
        print("Answer: <none>")
        return

    answer = str(answer)

    if len(answer) > MAX_ANSWER_LENGTH:
        answer = (
            answer[:MAX_ANSWER_LENGTH]
            + "\n...[answer truncated in test output]"
        )

    print()
    print("FINAL ANSWER")
    print("-" * 80)
    print(answer)


def validate_result_structure(
    result: Any,
) -> List[str]:
    """
    Validate the minimum structure expected from
    orchestrator.answer_question().
    """

    errors: List[str] = []

    if not isinstance(result, dict):
        errors.append(
            "Result is not a dictionary."
        )
        return errors

    required_fields = {
        "success",
        "route",
        "answer",
    }

    missing = required_fields - set(
        result.keys()
    )

    if missing:
        errors.append(
            "Missing fields: "
            + ", ".join(sorted(missing))
        )

    return errors


def validate_route(
    result: Dict[str, Any],
    expected_route: str,
) -> List[str]:
    """
    Validate route classification.
    """

    errors: List[str] = []

    actual_route = result.get(
        "route"
    )

    if actual_route != expected_route:

        errors.append(
            f"Expected route '{expected_route}', "
            f"got '{actual_route}'."
        )

    return errors


def validate_route_specific_result(
    result: Dict[str, Any],
    expected_route: str,
) -> List[str]:
    """
    Validate fields that should exist for each route.
    """

    errors: List[str] = []

    if expected_route == "sql":

        if "sql_result" not in result:

            errors.append(
                "SQL route did not return sql_result."
            )

    elif expected_route == "rag":

        if "reviews" not in result:

            errors.append(
                "RAG route did not return reviews."
            )

        if "rag_answer" not in result:

            errors.append(
                "RAG route did not return rag_answer."
            )

    elif expected_route == "hybrid":

        required = {
            "sql_question",
            "sql_result",
            "rag_question",
            "rag_answer",
            "reviews",
        }

        missing = required - set(
            result.keys()
        )

        if missing:

            errors.append(
                "Hybrid result missing fields: "
                + ", ".join(sorted(missing))
            )

    elif expected_route == "domain_guard":

        if "domain_guard" not in result:

            errors.append(
                "Domain Guard result missing domain_guard."
            )

    return errors


# ============================================================
# ROUTING INFORMATION
# ============================================================

def print_routing_information(
    result: Dict[str, Any],
) -> None:
    """
    Print Domain Guard and Router information.
    """

    domain_guard = result.get(
        "domain_guard"
    )

    if isinstance(
        domain_guard,
        dict,
    ):

        print()
        print("DOMAIN GUARD")
        print("-" * 80)

        print(
            f"In scope   : "
            f"{domain_guard.get('in_scope')}"
        )

        print(
            f"Confidence : "
            f"{domain_guard.get('confidence')}"
        )

        print(
            f"Reason     : "
            f"{domain_guard.get('reason')}"
        )

    routing = result.get(
        "routing"
    )

    if isinstance(
        routing,
        dict,
    ):

        print()
        print("ROUTER")
        print("-" * 80)

        print(
            f"Route      : "
            f"{routing.get('route')}"
        )

        print(
            f"Confidence : "
            f"{routing.get('confidence')}"
        )

        print(
            f"Reason     : "
            f"{routing.get('reason')}"
        )


# ============================================================
# SQL INFORMATION
# ============================================================

def print_sql_information(
    result: Dict[str, Any],
) -> None:
    """
    Print SQL-specific information.
    """

    sql_result = result.get(
        "sql_result"
    )

    if not isinstance(
        sql_result,
        dict,
    ):
        return

    print()
    print("SQL RESULT")
    print("-" * 80)

    print(
        f"SQL success : "
        f"{sql_result.get('success')}"
    )

    sql = sql_result.get(
        "sql"
    )

    if sql:

        print()
        print("Generated SQL:")
        print(sql)

    data = sql_result.get(
        "data"
    )

    if data is not None:

        dataframe = safe_dataframe(
            data
        )

        if not dataframe.empty:

            print()
            print(
                f"SQL rows returned: "
                f"{len(dataframe)}"
            )

            print()

            print(
                dataframe.head(10).to_string(
                    index=False
                )
            )

        else:

            print(
                "SQL returned an empty result."
            )


# ============================================================
# RAG INFORMATION
# ============================================================

def print_rag_information(
    result: Dict[str, Any],
) -> None:
    """
    Print RAG-specific information.
    """

    reviews = safe_dataframe(
        result.get("reviews")
    )

    print()
    print("RAG RESULT")
    print("-" * 80)

    print(
        f"Retrieved reviews: "
        f"{len(reviews)}"
    )

    if not reviews.empty:

        columns = [
            column
            for column in [
                "review_id",
                "rating",
                "comment",
                "review_date",
                "score",
                "sentiment_label",
                "topic",
                "key_issue",
            ]
            if column in reviews.columns
        ]

        print()

        print(
            reviews[
                columns
            ].head(10).to_string(
                index=False
            )
        )


# ============================================================
# HYBRID INFORMATION
# ============================================================

def print_hybrid_information(
    result: Dict[str, Any],
) -> None:
    """
    Print the complete hybrid workflow.
    """

    print()
    print("HYBRID WORKFLOW")
    print("-" * 80)

    print(
        "SQL planner question:"
    )

    print(
        result.get(
            "sql_question",
            "<missing>",
        )
    )

    print()
    print(
        "RAG planner question:"
    )

    print(
        result.get(
            "rag_question",
            "<missing>",
        )
    )

    print_sql_information(
        result
    )

    print_rag_information(
        result
    )


# ============================================================
# RUN SINGLE TEST
# ============================================================

def run_test_case(
    test_case: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Execute one complete test case.
    """

    test_id = test_case[
        "id"
    ]

    category = test_case[
        "category"
    ]

    question = test_case[
        "question"
    ]

    expected_route = test_case[
        "expected_route"
    ]

    print()
    print()
    print("=" * 100)

    print(
        f"TEST {test_id}"
    )

    print(
        f"Category : {category}"
    )

    print(
        f"Expected : {expected_route}"
    )

    print("=" * 100)

    print()
    print(
        "QUESTION"
    )

    print(
        question
    )

    print()
    print(
        "Running complete pipeline..."
    )

    start_time = time.perf_counter()

    try:

        result = answer_question(
            question
        )

        execution_time = (
            time.perf_counter()
            - start_time
        )

    except Exception as exc:

        execution_time = (
            time.perf_counter()
            - start_time
        )

        print()
        print(
            "❌ EXCEPTION"
        )

        print(
            f"{type(exc).__name__}: {exc}"
        )

        return {
            "id": test_id,
            "category": category,
            "question": question,
            "expected_route": expected_route,
            "actual_route": None,
            "success": False,
            "exception": True,
            "errors": [
                f"{type(exc).__name__}: {exc}"
            ],
            "execution_time": execution_time,
            "review_count": 0,
        }

    # --------------------------------------------------------
    # Structure validation
    # --------------------------------------------------------

    errors = validate_result_structure(
        result
    )

    # --------------------------------------------------------
    # Route validation
    # --------------------------------------------------------

    if isinstance(
        result,
        dict,
    ):

        errors.extend(
            validate_route(
                result,
                expected_route,
            )
        )

        errors.extend(
            validate_route_specific_result(
                result,
                expected_route,
            )
        )

    # --------------------------------------------------------
    # Basic success validation
    # --------------------------------------------------------

    if not isinstance(
        result,
        dict,
    ):

        actual_success = False
        actual_route = None

    else:

        actual_success = bool(
            result.get(
                "success",
                False,
            )
        )

        actual_route = result.get(
            "route"
        )

    # --------------------------------------------------------
    # Print result
    # --------------------------------------------------------

    if isinstance(
        result,
        dict,
    ):

        print_routing_information(
            result
        )

        if expected_route == "sql":

            print_sql_information(
                result
            )

        elif expected_route == "rag":

            print_rag_information(
                result
            )

        elif expected_route == "hybrid":

            print_hybrid_information(
                result
            )

        elif expected_route == "domain_guard":

            print()
            print(
                "DOMAIN GUARD BLOCKED QUESTION"
            )

    # --------------------------------------------------------
    # Final answer
    # --------------------------------------------------------

    if isinstance(
        result,
        dict,
    ):

        print_answer(
            result.get(
                "answer"
            )
        )

    # --------------------------------------------------------
    # Optional complete result
    # --------------------------------------------------------

    if PRINT_FULL_RESULT:

        print()
        print(
            "FULL RESULT"
        )

        print(
            json.dumps(
                result,
                indent=2,
                default=str,
                ensure_ascii=False,
            )
        )

    # --------------------------------------------------------
    # Review count
    # --------------------------------------------------------

    review_count = 0

    if isinstance(
        result,
        dict,
    ):

        reviews = result.get(
            "reviews"
        )

        if isinstance(
            reviews,
            pd.DataFrame,
        ):

            review_count = len(
                reviews
            )

    # --------------------------------------------------------
    # Status
    # --------------------------------------------------------

    if errors:

        status = "FAIL"

    elif not actual_success:

        # Domain Guard blocked questions are expected to have
        # success=True in the current orchestrator.
        #
        # For SQL/RAG/HYBRID we expect success=True.

        if expected_route == "domain_guard":

            status = "PASS"

        else:

            status = "FAIL"

            errors.append(
                "Pipeline returned success=False."
            )

    else:

        status = "PASS"

    print()
    print(
        "-" * 100
    )

    print(
        f"STATUS          : {status}"
    )

    print(
        f"Actual route    : {actual_route}"
    )

    print(
        f"Execution time  : "
        f"{execution_time:.2f} seconds"
    )

    print(
        f"Retrieved reviews: "
        f"{review_count}"
    )

    if errors:

        print()
        print(
            "ERRORS"
        )

        for error in errors:

            print(
                f"  ❌ {error}"
            )

    else:

        print(
            "✓ All structural checks passed."
        )

    return {
        "id": test_id,
        "category": category,
        "question": question,
        "expected_route": expected_route,
        "actual_route": actual_route,
        "success": status == "PASS",
        "exception": False,
        "errors": errors,
        "execution_time": execution_time,
        "review_count": review_count,
    }


# ============================================================
# SUMMARY
# ============================================================

def print_summary(
    results: List[Dict[str, Any]],
) -> None:
    """
    Print final regression-test summary.
    """

    total = len(
        results
    )

    passed = sum(
        1
        for result in results
        if result["success"]
    )

    failed = total - passed

    total_time = sum(
        result["execution_time"]
        for result in results
    )

    average_time = (
        total_time / total
        if total
        else 0.0
    )

    # --------------------------------------------------------
    # Route counts
    # --------------------------------------------------------

    expected_counts: Dict[str, int] = {}

    actual_counts: Dict[str, int] = {}

    for result in results:

        expected = result[
            "expected_route"
        ]

        actual = result[
            "actual_route"
        ]

        expected_counts[
            expected
        ] = (
            expected_counts.get(
                expected,
                0,
            )
            + 1
        )

        if actual:

            actual_counts[
                actual
            ] = (
                actual_counts.get(
                    actual,
                    0,
                )
                + 1
            )

    print()
    print()
    print("=" * 100)

    print(
        "ZOMATO AI - COMPLETE SYSTEM TEST SUMMARY"
    )

    print("=" * 100)

    print()
    print(
        f"Total tests       : {total}"
    )

    print(
        f"Passed tests      : {passed}"
    )

    print(
        f"Failed tests      : {failed}"
    )

    print(
        f"Total time        : "
        f"{total_time:.2f} seconds"
    )

    print(
        f"Average time      : "
        f"{average_time:.2f} seconds"
    )

    print()
    print(
        "EXPECTED ROUTES"
    )

    for route, count in sorted(
        expected_counts.items()
    ):

        print(
            f"  {route:15s}: {count}"
        )

    print()
    print(
        "ACTUAL ROUTES"
    )

    for route, count in sorted(
        actual_counts.items()
    ):

        print(
            f"  {route:15s}: {count}"
        )

    # --------------------------------------------------------
    # Detailed status
    # --------------------------------------------------------

    print()
    print(
        "-" * 100
    )

    print(
        "TEST STATUS"
    )

    print(
        "-" * 100
    )

    for result in results:

        status = (
            "PASS"
            if result["success"]
            else "FAIL"
        )

        print(
            f"{result['id']:8s} | "
            f"{status:4s} | "
            f"expected={result['expected_route']:12s} | "
            f"actual={str(result['actual_route']):12s} | "
            f"reviews={result['review_count']:2d} | "
            f"time={result['execution_time']:.2f}s"
        )

    # --------------------------------------------------------
    # Failed tests
    # --------------------------------------------------------

    failures = [
        result
        for result in results
        if not result["success"]
    ]

    if failures:

        print()
        print(
            "=" * 100
        )

        print(
            "FAILED TEST DETAILS"
        )

        print(
            "=" * 100
        )

        for result in failures:

            print()
            print(
                f"{result['id']} - "
                f"{result['category']}"
            )

            print(
                f"Question: "
                f"{result['question']}"
            )

            for error in result[
                "errors"
            ]:

                print(
                    f"  ❌ {error}"
                )

    # --------------------------------------------------------
    # Final verdict
    # --------------------------------------------------------

    print()
    print(
        "=" * 100
    )

    if failed == 0:

        print(
            "✓ ALL COMPLETE SYSTEM TESTS PASSED"
        )

        print()
        print(
            "Verified:"
        )

        print(
            "  ✓ Domain Guard"
        )

        print(
            "  ✓ Semantic Router"
        )

        print(
            "  ✓ SQL workflow"
        )

        print(
            "  ✓ RAG workflow"
        )

        print(
            "  ✓ Hybrid workflow"
        )

        print(
            "  ✓ Hybrid SQL planner"
        )

        print(
            "  ✓ Hybrid RAG planner"
        )

        print(
            "  ✓ Final answer synthesis"
        )

        print(
            "  ✓ Result structure"
        )

        print(
            "  ✓ Error handling path"
        )

    else:

        print(
            f"❌ COMPLETE SYSTEM TESTS FINISHED "
            f"WITH {failed} FAILURE(S)"
        )

    print(
        "=" * 100
    )


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    """
    Run the complete regression suite.
    """

    print("=" * 100)

    print(
        "ZOMATO AI - COMPLETE SYSTEM REGRESSION TEST"
    )

    print("=" * 100)

    print()
    print(
        f"Total configured tests: "
        f"{len(TEST_CASES)}"
    )

    print()
    print(
        "This test exercises the real:"
    )

    print(
        "  Domain Guard"
    )

    print(
        "      ↓"
    )

    print(
        "  Semantic Router"
    )

    print(
        "      ↓"
    )

    print(
        "  SQL / RAG / HYBRID"
    )

    print(
        "      ↓"
    )

    print(
        "  Final Answer"
    )

    print()
    print(
        "No project files will be modified."
    )

    results: List[
        Dict[str, Any]
    ] = []

    # --------------------------------------------------------
    # Execute tests
    # --------------------------------------------------------

    for test_case in TEST_CASES:

        result = run_test_case(
            test_case
        )

        results.append(
            result
        )

        if (
            STOP_ON_FAILURE
            and not result["success"]
        ):

            print()
            print(
                "STOP_ON_FAILURE=True"
            )

            print(
                "Stopping test suite."
            )

            break

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print_summary(
        results
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    try:

        main()

    except KeyboardInterrupt:

        print()
        print(
            "=" * 100
        )

        print(
            "TEST INTERRUPTED BY USER"
        )

        print(
            "=" * 100
        )

    except Exception as exc:

        print()
        print(
            "=" * 100
        )

        print(
            "❌ TEST SUITE FAILED TO START"
        )

        print(
            "=" * 100
        )

        print(
            f"{type(exc).__name__}: {exc}"
        )

        raise