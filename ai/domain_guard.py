"""
Zomato AI - Domain Guard

Purpose:
    Prevent SQL/RAG from answering questions that are outside
    the scope of the Zomato dataset.

The guard sits before the main orchestrator/router.

Flow:

    User Question
          |
          v
    Domain Guard
       /       \
   IN SCOPE   OUT OF SCOPE
      |            |
      v            v
    Router      Safe response
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict

import ollama


# ============================================================
# CONFIGURATION
# ============================================================

LLM_MODEL = "gpt-oss:120b-cloud"


# ============================================================
# DATASET SCOPE
# ============================================================

DOMAIN_DESCRIPTION = """
This application is a Zomato restaurant analytics assistant.

The available dataset contains information related to:

- restaurants
- restaurant names
- restaurant revenue
- restaurant performance
- restaurant cities
- cuisines
- food
- orders
- order items
- customers
- customer spending
- ratings
- restaurant reviews
- customer feedback
- delivery feedback
- service feedback

The system can answer questions that can reasonably be answered
from this Zomato restaurant/order/customer/review dataset.

The system CANNOT answer questions about unrelated external
products, companies, websites, shopping platforms, television,
movies, politics, sports, weather, stocks, or general knowledge
unless the question can clearly be answered from the Zomato
dataset.
"""


# ============================================================
# OBVIOUS OUT-OF-DOMAIN TERMS
# ============================================================

OBVIOUS_EXTERNAL_TERMS = [
    "amazon",
    "myntra",
    "flipkart",
    "netflix",
    "youtube",
    "instagram",
    "facebook",
    "prime video",
    "hotstar",
    "spotify",
    "lg tv",
    "samsung tv",
    "iphone",
    "ipad",
    "playstation",
    "xbox",
    "weather",
    "temperature",
    "stock market",
    "share price",
    "football",
    "cricket",
    "tennis",
    "politics",
    "president",
    "prime minister",
    "election",
]


# ============================================================
# SAFE RESPONSE
# ============================================================

OUT_OF_SCOPE_MESSAGE = (
    "I can only answer questions related to the Zomato data "
    "available in this system. You can ask about restaurants, "
    "revenue, orders, customers, cuisines, cities, ratings, "
    "or customer reviews."
)


# ============================================================
# NORMALIZE QUESTION
# ============================================================

def _normalize_question(question: str) -> str:
    """
    Normalize the question before checking it.
    """

    question = question.strip().lower()

    question = re.sub(
        r"\s+",
        " ",
        question,
    )

    return question


# ============================================================
# FAST EXTERNAL-DOMAIN CHECK
# ============================================================

def _check_obvious_external_domain(
    question: str,
) -> Dict[str, Any]:
    """
    Quickly detect clearly unrelated external topics.

    This is only a first safety layer.
    """

    normalized = _normalize_question(question)

    matched_terms = []

    for term in OBVIOUS_EXTERNAL_TERMS:

        if term in normalized:
            matched_terms.append(term)

    if matched_terms:

        return {
            "checked": True,
            "external": True,
            "matched_terms": matched_terms,
            "reason": (
                "The question contains terms associated with "
                "an external domain outside the Zomato dataset."
            ),
        }

    return {
        "checked": True,
        "external": False,
        "matched_terms": [],
        "reason": None,
    }


# ============================================================
# LLM DOMAIN CLASSIFIER
# ============================================================

def _classify_with_llm(
    question: str,
) -> Dict[str, Any]:
    """
    Ask the LLM whether the question is answerable
    from the Zomato dataset.
    """

    prompt = f"""
You are a strict domain classifier for a Zomato analytics system.

{DOMAIN_DESCRIPTION}

User question:
{question}

Your task is ONLY to determine whether the question can
reasonably be answered using the available Zomato dataset.

Important rules:

1. If the question concerns restaurants, restaurant revenue,
   orders, customers, spending, cuisines, cities, ratings,
   food, or restaurant/customer reviews, it can be IN SCOPE.

2. If the question concerns an unrelated external company,
   product, website, shopping platform, television, movie,
   politics, sports, weather, stocks, or general knowledge,
   it is OUT OF SCOPE.

3. Do NOT answer the user's question.

4. Return ONLY valid JSON.

Return exactly this structure:

{{
    "in_scope": true,
    "confidence": 0.95,
    "reason": "Short explanation"
}}

The value of in_scope must be either true or false.
The value of confidence must be between 0 and 1.
"""

    try:

        response = ollama.chat(
            model=LLM_MODEL,
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            options={
                "temperature": 0,
            },
        )

        content = response["message"]["content"].strip()

        # ----------------------------------------------------
        # Remove accidental markdown fences.
        # ----------------------------------------------------

        content = re.sub(
            r"^```json\s*",
            "",
            content,
            flags=re.IGNORECASE,
        )

        content = re.sub(
            r"\s*```$",
            "",
            content,
            flags=re.IGNORECASE,
        )

        result = json.loads(content)

        return {
            "in_scope": bool(
                result.get("in_scope", False)
            ),
            "confidence": float(
                result.get("confidence", 0.0)
            ),
            "reason": str(
                result.get(
                    "reason",
                    "",
                )
            ),
        }

    except Exception as exc:

        # ----------------------------------------------------
        # Fail closed.
        #
        # If the classifier itself fails, we do NOT want
        # unrelated questions reaching RAG.
        # ----------------------------------------------------

        return {
            "in_scope": False,
            "confidence": 0.0,
            "reason": (
                "Domain classification failed safely: "
                f"{type(exc).__name__}"
            ),
            "error": str(exc),
        }


# ============================================================
# MAIN DOMAIN GUARD
# ============================================================

def check_domain(
    question: str,
) -> Dict[str, Any]:
    """
    Validate whether a user question belongs to the
    Zomato dataset domain.
    """

    # --------------------------------------------------------
    # Basic validation
    # --------------------------------------------------------

    if not isinstance(question, str):

        return {
            "in_scope": False,
            "confidence": 1.0,
            "reason": "Question must be a string.",
            "answer": OUT_OF_SCOPE_MESSAGE,
        }

    question = question.strip()

    if not question:

        return {
            "in_scope": False,
            "confidence": 1.0,
            "reason": "Empty question.",
            "answer": OUT_OF_SCOPE_MESSAGE,
        }

    # --------------------------------------------------------
    # Layer 1
    # --------------------------------------------------------

    fast_check = _check_obvious_external_domain(
        question
    )

    if fast_check["external"]:

        return {
            "in_scope": False,
            "confidence": 0.99,
            "reason": fast_check["reason"],
            "matched_terms": fast_check["matched_terms"],
            "answer": OUT_OF_SCOPE_MESSAGE,
        }

    # --------------------------------------------------------
    # Layer 2
    # --------------------------------------------------------

    classification = _classify_with_llm(
        question
    )

    if classification["in_scope"]:

        return {
            "in_scope": True,
            "confidence": classification["confidence"],
            "reason": classification["reason"],
            "answer": None,
        }

    return {
        "in_scope": False,
        "confidence": classification["confidence"],
        "reason": classification["reason"],
        "answer": OUT_OF_SCOPE_MESSAGE,
    }


# ============================================================
# SIMPLE HELPER
# ============================================================

def is_in_domain(
    question: str,
) -> bool:
    """
    Convenience function.

    Returns only True/False.
    """

    result = check_domain(question)

    return bool(
        result.get("in_scope", False)
    )


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    print("=" * 70)
    print("ZOMATO AI - DOMAIN GUARD TEST")
    print("=" * 70)

    test_questions = [
        "Which restaurant generated the most revenue?",
        "Which restaurant generated the most revenue in Mumbai?",
        "What do customers say about food quality?",
        "What do customers say about delivery?",
        "Which customer spent the most?",
        "Which is the best product on Amazon?",
        "What is customer saying about the LG 46 inch Android TV?",
        "What is customer saying about Myntra shopping app?",
        "Who is the Prime Minister of India?",
        "What is 2 + 2?",
    ]

    for question in test_questions:

        print()
        print("-" * 70)
        print("QUESTION:")
        print(question)

        result = check_domain(
            question
        )

        print()
        print("RESULT:")
        print(json.dumps(
            result,
            indent=2,
            ensure_ascii=False,
        ))