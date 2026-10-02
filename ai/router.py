# ============================================================
# ZOMATO AI - SEMANTIC QUERY ROUTER
# ============================================================
#
# Purpose:
# Decide whether a user's question should be handled by:
#
#   1. SQL Engine
#   2. RAG Engine
#   3. HYBRID (SQL + RAG)
#
# The actual SQL and RAG processing remains inside:
#
#   sql_engine.py
#   rag_engine.py
#
# This file ONLY handles semantic routing.
#
# IMPORTANT:
# - No hard-coded keyword routing
# - No manual test questions
# - No SQL generation
# - No RAG retrieval
# - No final answer generation
# ============================================================


import json
import re
from typing import Any, Dict

import requests


# ============================================================
# ROUTING TYPES
# ============================================================

SQL = "sql"
RAG = "rag"
HYBRID = "hybrid"

VALID_ROUTES = {
    SQL,
    RAG,
    HYBRID,
}


# ============================================================
# OLLAMA CONFIGURATION
# ============================================================

OLLAMA_URL = "http://localhost:11434"

# Existing Ollama model used by the project.
ROUTER_MODEL = "gpt-oss:120b-cloud"

# Maximum time to wait for the router model.
ROUTER_TIMEOUT = 120


# ============================================================
# ROUTER SYSTEM PROMPT
# ============================================================

ROUTER_SYSTEM_PROMPT = """
You are the semantic query router for a Zomato analytics assistant.

Your ONLY job is to classify the user's original question.

You must NOT:
- answer the question
- generate SQL
- generate Python code
- retrieve reviews
- perform calculations
- provide recommendations
- modify the user's question

You must classify the question into exactly ONE of these routes:

------------------------------------------------------------
1. SQL
------------------------------------------------------------

Use "sql" when the question requires structured information
or business analytics from the database.

Typical SQL requirements include:

- revenue
- sales
- GMV
- order counts
- average order value
- rankings
- highest or lowest values
- cities
- restaurants
- payment methods
- dates
- counts
- sums
- averages
- structured business metrics
- numerical comparisons

Examples:

"Which city generated the highest revenue?"
-> sql

"Give me the location that brought in the most money."
-> sql

"Which restaurant has the largest number of orders?"
-> sql

"Which payment method is used most frequently?"
-> sql


------------------------------------------------------------
2. RAG
------------------------------------------------------------

Use "rag" when the question requires information from
customer review text or customer opinions.

Typical RAG requirements include:

- customer opinions
- customer experiences
- complaints
- praise
- sentiment
- satisfaction
- dissatisfaction
- food quality opinions
- delivery experiences
- packaging feedback
- service feedback
- problems mentioned by customers
- things customers like or dislike

Examples:

"What do customers say about food quality?"
-> rag

"How well are diners enjoying their meals?"
-> rag

"Are people satisfied with how their orders arrive?"
-> rag

"What problems are customers experiencing?"
-> rag


------------------------------------------------------------
3. HYBRID
------------------------------------------------------------

Use "hybrid" when the question requires BOTH:

A. Structured information from the database
AND
B. Information from customer reviews.

Hybrid questions often require the SQL result to provide
an entity or condition that is then used by the RAG system.

Example:

"Which restaurant generated the most revenue and what do
customers say about it?"

This requires:

SQL
-> identify the restaurant

RAG
-> analyze reviews about that restaurant

Therefore:
-> hybrid


Another example:

"Which restaurant in Kolkata has the most orders and what
do customers complain about?"

This requires:

SQL
-> find the restaurant with the most orders in Kolkata

RAG
-> analyze customer complaints about that restaurant

Therefore:
-> hybrid


------------------------------------------------------------
SEMANTIC UNDERSTANDING
------------------------------------------------------------

Classify based on the MEANING and INTENT of the question,
not individual keywords.

The user may express the same intent using completely
different words.

For example:

"Where are we making the most money?"
-> sql

"Which location brought in the most money?"
-> sql

"How happy are diners with their meals?"
-> rag

"What do people appreciate about the food?"
-> rag

"Find the highest earning restaurant and tell me what
customers think about it."
-> hybrid


------------------------------------------------------------
IMPORTANT DISTINCTION
------------------------------------------------------------

A question mentioning customers, restaurants, ratings,
orders, or other overlapping concepts is NOT automatically
hybrid.

Determine what information the user actually wants.

Example:

"What do customers think about highly rated restaurants?"
-> rag

The question is asking for customer opinions.

Example:

"Which restaurants have the highest ratings?"
-> sql

The question is asking for a structured ranking.

Example:

"Which restaurants have the highest ratings and what do
customers think about them?"
-> hybrid

The question requires both structured ranking and review
analysis.


------------------------------------------------------------
OUTPUT FORMAT
------------------------------------------------------------

Return ONLY valid JSON.

Do not return:
- Markdown
- code fences
- explanations outside the JSON
- SQL
- the answer to the user's question

The JSON MUST contain exactly these fields:

{
  "route": "sql" | "rag" | "hybrid",
  "confidence": 0.0,
  "reason": "short explanation"
}


------------------------------------------------------------
CONFIDENCE
------------------------------------------------------------

"confidence" must:

- be a number
- be between 0.0 and 1.0
- NOT be expressed as a percentage

The reason should be short and explain why the selected
route matches the user's intent.


------------------------------------------------------------
FINAL RULE
------------------------------------------------------------

Classify ONLY the original user question.

Do not solve it.
"""


# ============================================================
# NORMALIZE QUESTION
# ============================================================

def normalize_question(question: Any) -> str:
    """
    Normalize whitespace without changing the meaning
    of the user's question.
    """

    if not isinstance(question, str):
        return ""

    question = question.strip()

    question = re.sub(
        r"\s+",
        " ",
        question,
    )

    return question


# ============================================================
# FALLBACK
# ============================================================

def fallback_route(question: str) -> Dict[str, Any]:
    """
    Safe fallback used when:

    - Ollama is unavailable
    - the model returns malformed JSON
    - the response cannot be validated

    The fallback does NOT pretend that semantic classification
    succeeded.

    SQL is used as the conservative application fallback so
    that the application can continue operating.
    """

    return {
        "route": SQL,
        "confidence": 0.0,
        "reason": (
            "Semantic router unavailable; using SQL as the "
            "application fallback."
        ),
        "question": question,
        "fallback": True,
    }


# ============================================================
# EXTRACT JSON
# ============================================================

def extract_json(text: str) -> Dict[str, Any]:
    """
    Extract a JSON object from the model response.

    Handles:

    1. Normal JSON
    2. JSON accidentally wrapped in markdown
    3. JSON surrounded by additional text
    """

    if not isinstance(text, str):
        raise ValueError(
            "Model response must be a string."
        )

    text = text.strip()

    if not text:
        raise ValueError(
            "Model returned an empty response."
        )

    # --------------------------------------------------------
    # Attempt 1:
    # Parse the complete response directly.
    # --------------------------------------------------------

    try:

        result = json.loads(text)

        if isinstance(result, dict):
            return result

    except json.JSONDecodeError:
        pass

    # --------------------------------------------------------
    # Attempt 2:
    # Remove accidental markdown code fences.
    # --------------------------------------------------------

    cleaned = re.sub(
        r"^```(?:json)?\s*",
        "",
        text,
        flags=re.IGNORECASE,
    )

    cleaned = re.sub(
        r"\s*```$",
        "",
        cleaned,
    )

    cleaned = cleaned.strip()

    try:

        result = json.loads(cleaned)

        if isinstance(result, dict):
            return result

    except json.JSONDecodeError:
        pass

    # --------------------------------------------------------
    # Attempt 3:
    # Extract the first JSON object from the response.
    # --------------------------------------------------------

    start = cleaned.find("{")
    end = cleaned.rfind("}")

    if start != -1 and end != -1 and end > start:

        json_text = cleaned[start:end + 1]

        try:

            result = json.loads(json_text)

            if isinstance(result, dict):
                return result

        except json.JSONDecodeError:
            pass

    raise ValueError(
        "Could not extract valid JSON from model response."
    )


# ============================================================
# VALIDATE ROUTER RESULT
# ============================================================

def validate_router_result(
    result: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Validate and normalize the JSON returned by the LLM.
    """

    if not isinstance(result, dict):
        raise ValueError(
            "Router result must be a JSON object."
        )

    # --------------------------------------------------------
    # Validate route
    # --------------------------------------------------------

    route = result.get("route")

    if not isinstance(route, str):
        raise ValueError(
            "Router response is missing a valid route."
        )

    route = route.strip().lower()

    if route not in VALID_ROUTES:
        raise ValueError(
            f"Invalid route returned by model: {route}"
        )

    # --------------------------------------------------------
    # Validate confidence
    # --------------------------------------------------------

    confidence = result.get("confidence")

    try:

        confidence = float(confidence)

    except (TypeError, ValueError):

        raise ValueError(
            "Router response contains invalid confidence."
        )

    if not 0.0 <= confidence <= 1.0:

        raise ValueError(
            "Router confidence must be between 0.0 and 1.0."
        )

    # --------------------------------------------------------
    # Validate reason
    # --------------------------------------------------------

    reason = result.get("reason")

    if not isinstance(reason, str):

        raise ValueError(
            "Router response is missing a valid reason."
        )

    reason = reason.strip()

    if not reason:

        raise ValueError(
            "Router reason cannot be empty."
        )

    # --------------------------------------------------------
    # Return clean validated result
    # --------------------------------------------------------

    return {
        "route": route,
        "confidence": confidence,
        "reason": reason,
    }


# ============================================================
# CALL OLLAMA
# ============================================================

def call_ollama_router(
    question: str,
) -> Dict[str, Any]:
    """
    Send the user's question to Ollama and receive
    the semantic routing decision.
    """

    payload = {
        "model": ROUTER_MODEL,

        "messages": [
            {
                "role": "system",
                "content": ROUTER_SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": question,
            },
        ],

        "stream": False,

        # Ask Ollama for JSON output.
        "format": "json",

        "options": {
            # Deterministic classification.
            "temperature": 0,
        },
    }

    response = requests.post(
        f"{OLLAMA_URL}/api/chat",
        json=payload,
        timeout=ROUTER_TIMEOUT,
    )

    response.raise_for_status()

    data = response.json()

    # --------------------------------------------------------
    # Extract assistant message
    # --------------------------------------------------------

    message = data.get("message")

    if not isinstance(message, dict):

        raise ValueError(
            "Ollama response does not contain a valid message."
        )

    content = message.get("content")

    if not isinstance(content, str):

        raise ValueError(
            "Ollama response does not contain valid content."
        )

    # --------------------------------------------------------
    # Parse and validate JSON
    # --------------------------------------------------------

    result = extract_json(content)

    return validate_router_result(result)


# ============================================================
# CLASSIFY QUESTION
# ============================================================

def classify_question(
    question: str,
) -> Dict[str, Any]:
    """
    Classify a question using the semantic LLM router.

    Returns:

    {
        "route": "sql" | "rag" | "hybrid",
        "confidence": float,
        "reason": str
    }
    """

    question = normalize_question(question)

    if not question:

        raise ValueError(
            "Question cannot be empty."
        )

    return call_ollama_router(question)


# ============================================================
# GET ROUTING DETAILS
# ============================================================

def get_routing_details(
    question: str,
) -> Dict[str, Any]:
    """
    Return detailed routing information.

    Useful for debugging and application-level logging.

    This function does NOT execute SQL or RAG.
    """

    if not isinstance(question, str):

        raise ValueError(
            "Question must be a string."
        )

    original_question = question

    normalized_question = normalize_question(
        question
    )

    if not normalized_question:

        raise ValueError(
            "Question cannot be empty."
        )

    try:

        result = classify_question(
            normalized_question
        )

        return {
            "question": original_question,
            "route": result["route"],
            "confidence": result["confidence"],
            "reason": result["reason"],
            "fallback": False,
        }

    except Exception as exc:

        fallback = fallback_route(
            original_question
        )

        fallback["error"] = str(exc)

        return fallback


# ============================================================
# MAIN PUBLIC FUNCTION
# ============================================================

def route_question(
    question: str,
) -> Dict[str, Any]:
    """
    Main public routing function.

    Accepts the original user question.

    Returns:

    {
        "route": "sql" | "rag" | "hybrid",
        "confidence": float,
        "reason": str,
        "question": original_question,
        "fallback": bool
    }

    This function ONLY determines the route.

    It does NOT:

    - generate SQL
    - execute SQL
    - retrieve reviews
    - generate embeddings
    - generate the final answer
    """

    if not isinstance(question, str):

        raise ValueError(
            "Question must be a string."
        )

    original_question = question

    normalized_question = normalize_question(
        question
    )

    if not normalized_question:

        raise ValueError(
            "Question cannot be empty."
        )

    try:

        result = classify_question(
            normalized_question
        )

        return {
            "route": result["route"],
            "confidence": result["confidence"],
            "reason": result["reason"],
            "question": original_question,
            "fallback": False,
        }

    except Exception as exc:

        fallback = fallback_route(
            original_question
        )

        fallback["error"] = str(exc)

        return fallback
