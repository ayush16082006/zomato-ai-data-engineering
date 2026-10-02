
# ============================================================
# ZOMATO AI - STRONG SEMANTIC ROUTER TEST
# ============================================================

from router import route_question


# ============================================================
# TEST CASES
# ============================================================

TEST_CASES = [

    # --------------------------------------------------------
    # 1. BASIC SQL
    # --------------------------------------------------------

    {
        "expected": "sql",
        "question": "How much revenue did each city generate?",
    },

    {
        "expected": "sql",
        "question": "Which restaurant has the highest number of orders?",
    },

    {
        "expected": "sql",
        "question": "What is the average amount spent per order?",
    },


    # --------------------------------------------------------
    # 2. SEMANTICALLY DIFFERENT SQL
    # --------------------------------------------------------

    {
        "expected": "sql",
        "question": "Which location brought in the most money?",
    },

    {
        "expected": "sql",
        "question": "Tell me the restaurant with the largest sales volume.",
    },

    {
        "expected": "sql",
        "question": "Which payment option do customers use most often?",
    },

    {
        "expected": "sql",
        "question": "How many completed transactions do we have?",
    },


    # --------------------------------------------------------
    # 3. BASIC RAG
    # --------------------------------------------------------

    {
        "expected": "rag",
        "question": "What do customers say about the taste of the food?",
    },

    {
        "expected": "rag",
        "question": "What are people complaining about?",
    },

    {
        "expected": "rag",
        "question": "How do diners feel about the delivery experience?",
    },


    # --------------------------------------------------------
    # 4. SEMANTICALLY DIFFERENT RAG
    # --------------------------------------------------------

    {
        "expected": "rag",
        "question": "Are diners generally happy with their meals?",
    },

    {
        "expected": "rag",
        "question": "What aspects of ordering food do people appreciate?",
    },

    {
        "expected": "rag",
        "question": "What frustrates people when ordering from restaurants?",
    },

    {
        "expected": "rag",
        "question": "How satisfied are customers with the service they receive?",
    },

    {
        "expected": "rag",
        "question": "What do people praise or criticize about their orders?",
    },


    # --------------------------------------------------------
    # 5. HYBRID
    # --------------------------------------------------------

    {
        "expected": "hybrid",
        "question": (
            "Which restaurant generated the most revenue "
            "and what do customers say about it?"
        ),
    },

    {
        "expected": "hybrid",
        "question": (
            "Which restaurant has the highest number of orders "
            "and why do customers like it?"
        ),
    },

    {
        "expected": "hybrid",
        "question": (
            "Which city has the highest sales and what do "
            "customers think about restaurants there?"
        ),
    },

    {
        "expected": "hybrid",
        "question": (
            "Find the most successful restaurant and summarize "
            "the complaints customers have about it."
        ),
    },


    # --------------------------------------------------------
    # 6. SEMANTIC HYBRID
    # --------------------------------------------------------

    {
        "expected": "hybrid",
        "question": (
            "Which restaurant is the most popular in Kolkata "
            "and what do diners say about their experience there?"
        ),
    },

    {
        "expected": "hybrid",
        "question": (
            "Identify the top earning restaurant and tell me "
            "whether customers are happy with it."
        ),
    },

    {
        "expected": "hybrid",
        "question": (
            "Which place gets the most business and what are "
            "people saying about its food?"
        ),
    },

    {
        "expected": "hybrid",
        "question": (
            "Find the restaurant with the highest rating count "
            "and summarize its customer feedback."
        ),
    },


    # --------------------------------------------------------
    # 7. QUESTIONS THAT CONTAIN BOTH TYPES OF WORDS
    # --------------------------------------------------------

    {
        "expected": "rag",
        "question": (
            "What do customers say about restaurants with "
            "high ratings?"
        ),
    },

    {
        "expected": "rag",
        "question": (
            "How do customers feel about expensive restaurants?"
        ),
    },

    {
        "expected": "sql",
        "question": (
            "Which restaurants have the highest customer ratings?"
        ),
    },


    # --------------------------------------------------------
    # 8. NATURAL / CONVERSATIONAL QUESTIONS
    # --------------------------------------------------------

    {
        "expected": "rag",
        "question": "Are people enjoying the food?",
    },

    {
        "expected": "rag",
        "question": "Why are diners unhappy with their orders?",
    },

    {
        "expected": "sql",
        "question": "Where are we making the most money?",
    },

    {
        "expected": "sql",
        "question": "Which restaurant is doing the most business?",
    },


    # --------------------------------------------------------
    # 9. HARDER HYBRID QUESTIONS
    # --------------------------------------------------------

    {
        "expected": "hybrid",
        "question": (
            "Which restaurant in Kolkata receives the most orders "
            "and what do customers complain about?"
        ),
    },

    {
        "expected": "hybrid",
        "question": (
            "Which city performs best financially and how do "
            "customers describe their experiences there?"
        ),
    },

    {
        "expected": "hybrid",
        "question": (
            "Find the restaurant with the largest sales and "
            "explain what customers like and dislike about it."
        ),
    },
]


# ============================================================
# RUN TEST
# ============================================================

print("=" * 90)
print("ZOMATO AI - STRONG SEMANTIC ROUTER TEST")
print("=" * 90)

total = len(TEST_CASES)
passed = 0
failed = 0


for index, test in enumerate(TEST_CASES, start=1):

    question = test["question"]
    expected = test["expected"]

    print()
    print("-" * 90)

    print(f"Test       : {index}/{total}")
    print(f"Question   : {question}")
    print(f"Expected   : {expected.upper()}")

    try:

        result = route_question(question)

        actual = result["route"]
        confidence = result["confidence"]
        reason = result["reason"]

        print(f"Actual     : {actual.upper()}")
        print(f"Confidence : {confidence:.2f}")
        print(f"Reason     : {reason}")

        if actual == expected:

            print("Result     : PASS")
            passed += 1

        else:

            print("Result     : FAIL")
            failed += 1

    except Exception as exc:

        print("Result     : ERROR")
        print(f"Error      : {exc}")

        failed += 1


# ============================================================
# SUMMARY
# ============================================================

print()
print("=" * 90)
print("TEST SUMMARY")
print("=" * 90)

print(f"Total tests : {total}")
print(f"Passed      : {passed}")
print(f"Failed      : {failed}")

accuracy = (
    (passed / total) * 100
    if total > 0
    else 0
)

print(f"Accuracy    : {accuracy:.2f}%")

print("=" * 90)

if failed == 0:

    print("ALL TESTS PASSED")

else:

    print("SOME TESTS FAILED - REVIEW THE FAILED CASES")
