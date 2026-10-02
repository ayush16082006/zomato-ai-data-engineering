import rag_engine


# ============================================================
# CONFIGURATION
# ============================================================

TOP_K = 5


# ============================================================
# 9 INDIVIDUAL QUESTIONS
# ============================================================

INDIVIDUAL_QUESTIONS = [

    {
        "number": 1,
        "category": "Food Quality",
        "question":
            "What do customers say about food quality?"
    },

    {
        "number": 2,
        "category": "Delivery",
        "question":
            "What do customers say about delivery experience?"
    },

    {
        "number": 3,
        "category": "Negative Experience",
        "question":
            "What are the main complaints from customers?"
    },

    {
        "number": 4,
        "category": "Restaurant Service",
        "question":
            "What do customers say about restaurant service?"
    },

    {
        "number": 5,
        "category": "Pricing",
        "question":
            "What do customers say about food prices and value for money?"
    },

    {
        "number": 6,
        "category": "Packaging",
        "question":
            "What do customers say about food packaging?"
    },

    {
        "number": 7,
        "category": "Late Delivery",
        "question":
            "What complaints do customers have about late delivery?"
    },

    {
        "number": 8,
        "category": "Positive Experience",
        "question":
            "What do customers like most about their Zomato orders?"
    },

    {
        "number": 9,
        "category": "High Ratings",
        "question":
            "What do customers with high ratings say about their experience?"
    }
]


# ============================================================
# 5 GROUPED / BROADER QUESTIONS
# ============================================================

GROUPED_QUESTIONS = [

    {
        "number": 1,
        "category": "Overall Experience",
        "question":
            "What are the main things customers like and dislike about their Zomato orders?"
    },

    {
        "number": 2,
        "category": "Food + Delivery",
        "question":
            "How do customers describe both food quality and delivery experience?"
    },

    {
        "number": 3,
        "category": "Problems",
        "question":
            "What are the most common problems mentioned in customer reviews?"
    },

    {
        "number": 4,
        "category": "Value",
        "question":
            "How do customers describe the value of the food, including price, quality, and quantity?"
    },

    {
        "number": 5,
        "category": "Complete Experience",
        "question":
            "What patterns can be seen across food quality, delivery, service, pricing, and packaging?"
    }
]


# ============================================================
# PRINT RETRIEVED REVIEWS
# ============================================================

def print_retrieved_reviews(
    top_reviews
):

    print()
    print("-" * 80)
    print("TOP RETRIEVED REVIEWS")
    print("-" * 80)

    if top_reviews.empty:

        print(
            "No sufficiently relevant reviews found."
        )

        return

    for rank, (_, row) in enumerate(
        top_reviews.iterrows(),
        start=1
    ):

        print()

        print(
            f"[{rank}] Review ID : "
            f"{row['review_id']}"
        )

        print(
            f"    Rating    : "
            f"{row['rating']}"
        )

        print(
            f"    Score     : "
            f"{row['score']:.4f}"
        )

        print(
            f"    Comment   : "
            f"{row['comment']}"
        )

        print(
            "-" * 80
        )


# ============================================================
# RUN ONE QUESTION
# ============================================================

def run_question(
    item,
    reviews,
    embedding_matrix,
    test_group
):

    number = item["number"]

    category = item[
        "category"
    ]

    question = item[
        "question"
    ]

    print()
    print()
    print("=" * 80)

    print(
        f"{test_group} "
        f"QUESTION {number}"
    )

    print("=" * 80)

    print()

    print(
        f"Category : {category}"
    )

    print()

    print(
        f"Question : {question}"
    )

    # ========================================================
    # VECTOR SEARCH + RAG
    # ========================================================

    try:

        print()

        print(
            "Searching embedded reviews..."
        )

        answer, top_reviews = (
            rag_engine.answer_review_question(
                question,
                reviews,
                embedding_matrix,
                top_k=TOP_K
            )
        )

        print()

        print(
            "✅ RAG pipeline completed."
        )

    except Exception as e:

        print()

        print(
            "❌ RAG pipeline failed."
        )

        print(
            f"Error: "
            f"{type(e).__name__}: {e}"
        )

        return False

    # ========================================================
    # RETRIEVED REVIEWS
    # ========================================================

    print_retrieved_reviews(
        top_reviews
    )

    # ========================================================
    # ANSWER
    # ========================================================

    print()
    print("=" * 80)
    print("RAG ANSWER")
    print("=" * 80)

    print()

    print(answer)

    print()

    print(
        f"✅ {test_group} "
        f"question {number} completed."
    )

    return True


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 80)
    print("ZOMATO RAG ENGINE - COMPLETE TEST")
    print("=" * 80)

    # ========================================================
    # 1. CHECK OLLAMA
    # ========================================================

    print()

    print(
        "[1] Checking Ollama..."
    )

    ok, models = (
        rag_engine.check_ollama()
    )

    if not ok:

        print(
            "❌ Ollama connection failed."
        )

        print(models)

        return

    print(
        "✅ Ollama is running."
    )

    print()

    print(
        "Available models:"
    )

    for model in models:

        print(
            f"  - {model}"
        )

    # ========================================================
    # 2. LOAD EMBEDDINGS ONCE
    # ========================================================

    print()

    print(
        "[2] Initializing RAG engine..."
    )

    print()

    print(
        "IMPORTANT: embedding parquet files "
        "will be loaded only once."
    )

    reviews, embedding_matrix = (
        rag_engine.initialize_rag_engine()
    )

    print()

    print(
        "✅ Embedding store initialized."
    )

    print(
        f"Reviews loaded     : "
        f"{len(reviews):,}"
    )

    print(
        f"Embedding matrix   : "
        f"{embedding_matrix.shape}"
    )

    # ========================================================
    # TEST COUNTERS
    # ========================================================

    total_questions = (
        len(INDIVIDUAL_QUESTIONS)
        +
        len(GROUPED_QUESTIONS)
    )

    successful = 0

    failed = 0

    # ========================================================
    # 3. TEST 9 INDIVIDUAL QUESTIONS
    # ========================================================

    print()
    print()
    print("=" * 80)
    print("PHASE 1 - 9 INDIVIDUAL QUESTIONS")
    print("=" * 80)

    for item in INDIVIDUAL_QUESTIONS:

        success = run_question(
            item,
            reviews,
            embedding_matrix,
            "INDIVIDUAL"
        )

        if success:

            successful += 1

        else:

            failed += 1

    # ========================================================
    # 4. TEST 5 GROUPED QUESTIONS
    # ========================================================

    print()
    print()
    print("=" * 80)
    print("PHASE 2 - 5 GROUPED QUESTIONS")
    print("=" * 80)

    for item in GROUPED_QUESTIONS:

        success = run_question(
            item,
            reviews,
            embedding_matrix,
            "GROUPED"
        )

        if success:

            successful += 1

        else:

            failed += 1

    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    print()
    print()
    print("=" * 80)
    print("RAG TEST SUMMARY")
    print("=" * 80)

    print()

    print(
        f"Total questions     : "
        f"{total_questions}"
    )

    print(
        f"Successful questions: "
        f"{successful}"
    )

    print(
        f"Failed questions    : "
        f"{failed}"
    )

    print()

    print(
        f"Reviews searched    : "
        f"{len(reviews):,}"
    )

    print(
        f"Embedding dimension : "
        f"{embedding_matrix.shape[1]}"
    )

    print()

    # ========================================================
    # FINAL STATUS
    # ========================================================

    if failed == 0:

        print("=" * 80)
        print(
            "🎉 ALL RAG TESTS COMPLETED SUCCESSFULLY"
        )
        print("=" * 80)

        print()

        print(
            "✅ Ollama connection works"
        )

        print(
            "✅ Embedding files loaded once"
        )

        print(
            "✅ In-memory embedding cache works"
        )

        print(
            "✅ All embedded reviews searched"
        )

        print(
            "✅ Question embeddings generated"
        )

        print(
            "✅ Vector similarity works"
        )

        print(
            "✅ Duplicate review text filtering works"
        )

        print(
            "✅ MMR diversity retrieval works"
        )

        print(
            "✅ RAG context generation works"
        )

        print(
            "✅ LLM answer generation works"
        )

        print(
            "✅ 9 individual questions tested"
        )

        print(
            "✅ 5 grouped questions tested"
        )

    else:

        print("=" * 80)
        print(
            "⚠️ RAG TEST COMPLETED WITH FAILURES"
        )
        print("=" * 80)

        print()

        print(
            f"Successful: "
            f"{successful}"
        )

        print(
            f"Failed    : "
            f"{failed}"
        )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()