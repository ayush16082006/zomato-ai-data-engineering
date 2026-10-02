from orchestrator import answer_question


TEST_CASES = [
    # ============================================================
    # SQL TESTS
    # ============================================================

    (
        "SQL-01",
        "What is the average order value?",
        "sql",
    ),

    (
        "SQL-02",
        "Which restaurant generated the most revenue?",
        "sql",
    ),

    (
        "SQL-03",
        "Which city generated the highest revenue?",
        "sql",
    ),

    (
        "SQL-04",
        "Which payment method has the most orders?",
        "sql",
    ),

    (
        "SQL-05",
        "How many orders were delivered?",
        "sql",
    ),

    # ============================================================
    # RAG TESTS
    # ============================================================

    (
        "RAG-01",
        "What do customers say about food quality?",
        "rag",
    ),

    (
        "RAG-02",
        "What are the main complaints from customers?",
        "rag",
    ),

    (
        "RAG-03",
        "What do customers say about delivery experience?",
        "rag",
    ),

    (
        "RAG-04",
        "What do customers appreciate most about their food orders?",
        "rag",
    ),

    (
        "RAG-05",
        "How satisfied are customers with the service?",
        "rag",
    ),

    # ============================================================
    # HYBRID - ENTITY DISCOVERY
    # ============================================================

    (
        "HYB-01",
        "Which restaurant generated the most revenue and what do customers say about it?",
        "hybrid",
    ),

    (
        "HYB-02",
        "Which restaurant has the most orders and what are customers saying about its service?",
        "hybrid",
    ),

    (
        "HYB-03",
        "Find the highest revenue restaurant and summarize customer complaints about it.",
        "hybrid",
    ),

    # ============================================================
    # HYBRID - CITY
    # ============================================================

    (
        "HYB-04",
        "Which city has the highest sales and what do customers think about restaurants there?",
        "hybrid",
    ),

    # ============================================================
    # HYBRID - MULTIPLE RAG REQUIREMENTS
    # ============================================================

    (
        "HYB-05",
        "Which restaurant generated the most revenue and what do customers like and dislike about its food?",
        "hybrid",
    ),

    (
        "HYB-06",
        "Which restaurant has the most orders and what problems do customers report about its delivery?",
        "hybrid",
    ),

    # ============================================================
    # NATURAL LANGUAGE VARIATIONS
    # ============================================================

    (
        "SQL-06",
        "Tell me the restaurant that earned the largest amount from orders.",
        "sql",
    ),

    (
        "SQL-07",
        "What is the typical value of an order?",
        "sql",
    ),

    (
        "RAG-06",
        "How are people feeling about the food they receive?",
        "rag",
    ),

    (
        "RAG-07",
        "What issues are diners experiencing with their orders?",
        "rag",
    ),

    # ============================================================
    # COMPLEX HYBRID
    # ============================================================

    (
        "HYB-07",
        "Find the restaurant with the highest revenue, tell me how much it earned, and summarize what customers say about its food and delivery.",
        "hybrid",
    ),

    (
        "HYB-08",
        "Which city brings in the most money and what do customers say about food and delivery in that city?",
        "hybrid",
    ),
]


def run_test(test_id, question, expected_route):
    print("\n" + "=" * 80)
    print(test_id)
    print("=" * 80)

    print("Question:")
    print(question)

    print("\nExpected route:")
    print(expected_route.upper())

    try:
        result = answer_question(question)

        if not isinstance(result, dict):
            print("\n❌ INVALID RESULT")
            print("Expected dictionary but received:", type(result))
            return False

        actual_route = str(
            result.get("route", "")
        ).lower()

        success = result.get("success", False)

        print("\nActual route:")
        print(actual_route.upper())

        print("\nSuccess:")
        print(success)

        if actual_route != expected_route:
            print("\n❌ ROUTE MISMATCH")
            return False

        if not success:
            print("\n❌ EXECUTION FAILED")
            print("Result:")
            print(result)
            return False

        if not result.get("answer"):
            print("\n❌ EMPTY FINAL ANSWER")
            return False

        # --------------------------------------------------------
        # Additional validation for HYBRID
        # --------------------------------------------------------

        if expected_route == "hybrid":

            if not result.get("sql_result"):
                print("\n❌ Missing SQL result")
                return False

            if not result.get("rag_question"):
                print("\n❌ Missing RAG question")
                return False

            if not result.get("rag_answer"):
                print("\n❌ Missing RAG answer")
                return False

            print("\nHybrid components:")
            print("✓ SQL result")
            print("✓ RAG question")
            print("✓ RAG answer")
            print("✓ Final answer")

        # --------------------------------------------------------
        # Additional validation for SQL
        # --------------------------------------------------------

        if expected_route == "sql":

            if not result.get("sql_result"):
                print("\n❌ Missing SQL result")
                return False

            if not result["sql_result"].get("success"):
                print("\n❌ SQL execution failed")
                return False

            print("\nSQL components:")
            print("✓ SQL result")
            print("✓ Final answer")

        # --------------------------------------------------------
        # Additional validation for RAG
        # --------------------------------------------------------

        if expected_route == "rag":

            if not result.get("rag_answer"):
                print("\n❌ Missing RAG answer")
                return False

            print("\nRAG components:")
            print("✓ RAG answer")
            print("✓ Final answer")

        print("\nFINAL ANSWER:")
        print(result["answer"])

        print("\n✅ TEST PASSED")

        return True

    except Exception as e:

        print("\n❌ EXCEPTION")
        print(type(e).__name__)
        print(str(e))

        return False


def main():

    print("=" * 80)
    print("ZOMATO AI - STRONG ORCHESTRATOR INTEGRATION TEST")
    print("=" * 80)

    total = len(TEST_CASES)
    passed = 0
    failed = 0

    results = []

    for test_id, question, expected_route in TEST_CASES:

        success = run_test(
            test_id,
            question,
            expected_route,
        )

        if success:
            passed += 1
            results.append((test_id, "PASS"))
        else:
            failed += 1
            results.append((test_id, "FAIL"))

    # ============================================================
    # FINAL SUMMARY
    # ============================================================

    print("\n\n")
    print("=" * 80)
    print("FINAL TEST SUMMARY")
    print("=" * 80)

    print(f"Total tests : {total}")
    print(f"Passed      : {passed}")
    print(f"Failed      : {failed}")

    print("\nTest results:")
    print("-" * 80)

    for test_id, status in results:
        print(f"{test_id:<12} {status}")

    print("-" * 80)

    if failed == 0:

        print("\n🎉 ALL TESTS PASSED")
        print("The orchestrator passed the complete integration test.")

    else:

        print("\n⚠ SOME TESTS FAILED")
        print("Do not modify the architecture yet.")
        print("Inspect the failed test cases first.")


if __name__ == "__main__":
    main()