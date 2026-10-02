# ============================================================
# ZOMATO AI - QUERY ORCHESTRATOR
# ============================================================
#
# Purpose:
# Coordinate the complete Zomato AI query pipeline.
#
# Architecture:
#
#                         USER QUESTION
#                              |
#                              v
#                        DOMAIN GUARD
#                              |
#                    +---------+---------+
#                    |                   |
#                 OUT OF SCOPE        IN SCOPE
#                    |                   |
#                    v                   v
#               SAFE MESSAGE          ROUTER
#                                        |
#                         +--------------+--------------+
#                         |              |              |
#                         v              v              v
#                        SQL            RAG           HYBRID
#                         |              |              |
#                         v              v              v
#                    SQL ENGINE     RAG ENGINE    HYBRID WORKFLOW
#                                                       |
#                                                       v
#                                                   FINAL LLM
#                                                       |
#                                                       v
#                                                 FINAL ANSWER
#
#
# IMPORTANT:
#
# This file does NOT:
# - generate SQL itself
# - execute BigQuery itself
# - perform vector search itself
# - generate embeddings itself
# - implement RAG retrieval itself
# - implement domain classification itself
#
# It coordinates the existing modules.
# ============================================================


import json
import re
from typing import Any, Dict, Optional

import requests

from router import route_question
from sql_engine import ask_sql
from rag_engine import answer_review_question

# Domain Guard
from domain_guard import check_domain


# ============================================================
# OLLAMA CONFIGURATION
# ============================================================

OLLAMA_URL = "http://localhost:11434"

# Model used for:
#
# 1. Hybrid SQL task planning
# 2. Hybrid RAG query planning
# 3. Final answer synthesis
#
# SQL engine, router, and domain guard keep using their
# own configured logic.
ORCHESTRATOR_MODEL = "gpt-oss:120b-cloud"

OLLAMA_TIMEOUT = 120


# ============================================================
# HYBRID SQL TASK PLANNER PROMPT
# ============================================================

HYBRID_SQL_PLANNER_PROMPT = """
You are the SQL-task planner for a Zomato AI assistant.

The user's question has already been classified as HYBRID.

Your job is to extract ONLY the structured-data part of the
user's question that should be answered by the SQL engine.

The SQL engine is responsible for:

- counts
- sums
- averages
- rankings
- revenue
- sales
- orders
- restaurant metrics
- city metrics
- finding an entity such as the top restaurant
- customer spending
- cuisine metrics

The SQL engine is NOT responsible for:

- customer opinions
- reviews
- complaints
- sentiment
- what people say
- review summaries
- customer experiences

------------------------------------------------------------
RULES
------------------------------------------------------------

1. Return exactly ONE natural-language SQL question.

2. Preserve the structured-data intent.

3. Remove the review/opinion/customer-feedback part of the
   question.

4. Do NOT generate SQL.

5. Do NOT answer the question.

6. Do NOT include explanations.

7. Do NOT invent information.

8. The output must be directly suitable as input to the
   existing SQL engine.

------------------------------------------------------------
EXAMPLES
------------------------------------------------------------

User:
Which restaurant generated the most revenue and what do
customers say about it?

Output:
Which restaurant generated the most revenue?

------------------------------------------------------------

User:
Which restaurant has the most orders and what are customers
saying about its service?

Output:
Which restaurant has the most orders?

------------------------------------------------------------

User:
Which city has the highest sales and what do customers think
about restaurants there?

Output:
Which city has the highest sales?

------------------------------------------------------------

User:
Find the highest revenue restaurant and summarize customer
complaints about it.

Output:
Which restaurant generated the most revenue?

------------------------------------------------------------

User:
Which restaurant has the highest revenue and how do people
feel about its food quality and delivery?

Output:
Which restaurant has the highest revenue?

------------------------------------------------------------

Return ONLY the SQL question.
"""


# ============================================================
# HYBRID RAG QUERY PLANNER PROMPT
# ============================================================

HYBRID_QUERY_PLANNER_PROMPT = """
You are the query planner for a Zomato AI assistant.

The user's question has already been classified as HYBRID.

The structured part of the question has already been
processed by the SQL engine.

Your job is to create ONE focused natural-language question
for the review RAG system.

You will receive:

1. The original user question.
2. The SQL result.

Your task is to use the SQL result to identify the entity
or context that the RAG system should investigate.

------------------------------------------------------------
RULES
------------------------------------------------------------

1. Return exactly ONE natural-language RAG question.

2. Preserve the review-related intent of the original
   user question.

3. Use entities discovered by the SQL result when relevant.

4. If SQL identifies a restaurant, include that restaurant
   in the RAG question when appropriate.

5. If SQL identifies a city, include that city when
   appropriate.

6. If SQL identifies another relevant entity, use it.

7. Do NOT ask for structured metrics.

8. Do NOT generate SQL.

9. Do NOT answer the original question.

10. Do NOT invent entities.

11. Do NOT include explanations.

12. The output must be suitable for semantic review
    retrieval.

13. If the SQL result is empty, do not invent a restaurant,
    city, or other entity.

------------------------------------------------------------
EXAMPLES
------------------------------------------------------------

Original:
Which restaurant generated the most revenue and what do
customers say about it?

SQL result:
Restaurant = ABC Restaurant
Revenue = 2500000

Output:
What do customers say about ABC Restaurant?

------------------------------------------------------------

Original:
Which restaurant has the most orders and what are customers
saying about its service?

SQL result:
Restaurant = XYZ Restaurant
Orders = 18420

Output:
What do customers say about the service at XYZ Restaurant?

------------------------------------------------------------

Original:
Which city has the highest sales and what do customers think
about restaurants there?

SQL result:
City = Kolkata
Revenue = 50000000

Output:
What do customers think about restaurants in Kolkata?

------------------------------------------------------------

Return ONLY the RAG question.
"""


# ============================================================
# FINAL ANSWER PROMPT
# ============================================================

FINAL_ANSWER_PROMPT = """
You are the final answer generator for a Zomato AI assistant.

Answer the user's ORIGINAL question using the provided
information.

The information may contain:

- SQL results
- RAG results
- both SQL and RAG results

------------------------------------------------------------
RULES
------------------------------------------------------------

1. Answer the original user question directly.

2. Use only information supported by the provided results.

3. Do not invent facts.

4. Do not invent numbers.

5. Clearly distinguish structured metrics from customer
   opinions.

6. Treat review information as customer feedback, not as
   a universal fact about every customer.

7. If the available information is insufficient for part
   of the question, say so rather than guessing.

8. Do not mention internal components such as:
   - router
   - orchestrator
   - SQL engine
   - RAG engine
   - query planner
   - prompts
   - domain guard

9. Do not expose internal reasoning.

10. Keep the answer clear and reasonably concise.

11. If SQL identifies a specific entity and RAG provides
    review information about that entity, connect the two
    naturally in the final answer.

12. If the SQL result is empty or insufficient, do not claim
    that the RAG reviews belong to an entity that was not
    actually identified by SQL.

13. Do not use outside knowledge.

14. Do not answer questions outside the provided Zomato
    information.

Return ONLY the final user-facing answer.
"""


# ============================================================
# DOMAIN GUARD FALLBACK MESSAGE
# ============================================================

DOMAIN_GUARD_FALLBACK_MESSAGE = (
    "I can only answer questions related to the Zomato data "
    "available in this system. You can ask about restaurants, "
    "revenue, orders, customers, cuisines, cities, ratings, "
    "or customer reviews."
)


# ============================================================
# TEXT NORMALIZATION
# ============================================================

def normalize_text(value: Any) -> str:
    """
    Normalize whitespace in a text value.
    """

    if not isinstance(value, str):
        return ""

    return re.sub(
        r"\s+",
        " ",
        value.strip(),
    )


# ============================================================
# DOMAIN GUARD RESULT NORMALIZATION
# ============================================================

def normalize_domain_result(
    result: Any,
) -> Dict[str, Any]:
    """
    Normalize the result returned by domain_guard.py.

    Expected structure:

    {
        "in_scope": True/False,
        "confidence": 0.98,
        "reason": "...",
        "answer": "..."
    }

    This function also protects the orchestrator from a malformed
    domain guard response.
    """

    if not isinstance(result, dict):
        raise ValueError(
            "Domain guard returned an invalid result."
        )

    if "in_scope" not in result:
        raise ValueError(
            "Domain guard result does not contain 'in_scope'."
        )

    normalized = dict(result)

    normalized["in_scope"] = bool(
        normalized.get("in_scope")
    )

    confidence = normalized.get(
        "confidence"
    )

    try:
        if confidence is not None:
            normalized["confidence"] = float(
                confidence
            )
    except (TypeError, ValueError):
        normalized["confidence"] = None

    if not normalized.get("reason"):
        normalized["reason"] = (
            "No domain classification reason was provided."
        )

    return normalized


# ============================================================
# CHECK DOMAIN
# ============================================================

def validate_domain(
    question: str,
) -> Dict[str, Any]:
    """
    Run the Domain Guard before routing.

    This is the most important protection against questions
    such as:

        Which is the best product on Amazon?
        Who is the Prime Minister of India?
        What is 2 + 2?
        What do customers say about an LG TV?

    Those questions should NEVER reach the SQL or RAG engines.
    """

    domain_result = check_domain(
        question
    )

    return normalize_domain_result(
        domain_result
    )


# ============================================================
# SQL RESULT SERIALIZATION
# ============================================================

def serialize_sql_result(
    sql_result: Dict[str, Any],
) -> str:
    """
    Convert the SQL engine result into JSON-safe text
    that can be supplied to an LLM.

    The existing SQL engine returns the query result
    as a pandas DataFrame under the 'data' field.
    """

    if not isinstance(sql_result, dict):
        return str(sql_result)

    if not sql_result.get("success"):

        result = {
            "success": False,
            "reason": sql_result.get("reason"),
            "error": sql_result.get("error"),
        }

        return json.dumps(
            result,
            default=str,
            ensure_ascii=False,
        )

    data = sql_result.get(
        "data"
    )

    # --------------------------------------------------------
    # Convert pandas DataFrame to records.
    # --------------------------------------------------------

    if data is not None and hasattr(
        data,
        "to_dict",
    ):

        try:

            data = data.to_dict(
                orient="records"
            )

        except Exception:

            data = str(data)

    result = {
        "success": True,
        "sql": sql_result.get("sql"),
        "data": data,
        "reason": sql_result.get("reason"),
    }

    return json.dumps(
        result,
        default=str,
        ensure_ascii=False,
    )


# ============================================================
# OLLAMA CHAT HELPER
# ============================================================

def call_ollama(
    system_prompt: str,
    user_prompt: str,
    temperature: float = 0.0,
) -> str:
    """
    Generic Ollama chat helper.

    Used by:

    - Hybrid SQL planner
    - Hybrid RAG planner
    - Final answer generator
    """

    payload = {
        "model": ORCHESTRATOR_MODEL,

        "messages": [
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": user_prompt,
            },
        ],

        "stream": False,

        "options": {
            "temperature": temperature,
        },
    }

    response = requests.post(
        f"{OLLAMA_URL}/api/chat",
        json=payload,
        timeout=OLLAMA_TIMEOUT,
    )

    response.raise_for_status()

    data = response.json()

    message = data.get(
        "message"
    )

    if not isinstance(
        message,
        dict,
    ):

        raise ValueError(
            "Ollama response does not contain "
            "a valid message."
        )

    content = message.get(
        "content"
    )

    if not isinstance(
        content,
        str,
    ):

        raise ValueError(
            "Ollama response does not contain "
            "valid content."
        )

    content = normalize_text(
        content
    )

    if not content:

        raise ValueError(
            "Ollama returned an empty response."
        )

    return content


# ============================================================
# GENERATE SQL SUB-QUESTION FOR HYBRID QUERY
# ============================================================

def generate_sql_question(
    original_question: str,
) -> str:
    """
    Extract ONLY the structured-data portion of a HYBRID
    question.
    """

    user_prompt = f"""
Original user question:

{original_question}


Extract ONLY the SQL/structured-data question.
"""

    sql_question = call_ollama(
        system_prompt=HYBRID_SQL_PLANNER_PROMPT,
        user_prompt=user_prompt,
        temperature=0.0,
    )

    # Remove accidental markdown fences.

    sql_question = re.sub(
        r"^```(?:text)?\s*",
        "",
        sql_question,
        flags=re.IGNORECASE,
    )

    sql_question = re.sub(
        r"\s*```$",
        "",
        sql_question,
    )

    sql_question = normalize_text(
        sql_question
    )

    if not sql_question:

        raise ValueError(
            "Hybrid SQL planner generated "
            "an empty SQL question."
        )

    return sql_question


# ============================================================
# GENERATE FOCUSED RAG QUESTION
# ============================================================

def generate_rag_question(
    original_question: str,
    sql_result: Dict[str, Any],
) -> str:
    """
    Generate the focused RAG question for a HYBRID query.
    """

    sql_context = serialize_sql_result(
        sql_result
    )

    user_prompt = f"""
Original user question:

{original_question}


SQL result:

{sql_context}


Generate the single focused RAG question now.
"""

    rag_question = call_ollama(
        system_prompt=HYBRID_QUERY_PLANNER_PROMPT,
        user_prompt=user_prompt,
        temperature=0.0,
    )

    # Remove accidental markdown fences.

    rag_question = re.sub(
        r"^```(?:text)?\s*",
        "",
        rag_question,
        flags=re.IGNORECASE,
    )

    rag_question = re.sub(
        r"\s*```$",
        "",
        rag_question,
    )

    rag_question = normalize_text(
        rag_question
    )

    if not rag_question:

        raise ValueError(
            "Hybrid query planner generated "
            "an empty RAG question."
        )

    return rag_question


# ============================================================
# GENERATE FINAL ANSWER
# ============================================================

def generate_final_answer(
    original_question: str,
    sql_result: Optional[Dict[str, Any]] = None,
    rag_result: Optional[str] = None,
) -> str:
    """
    Generate the final answer from SQL and/or RAG information.
    """

    if sql_result is not None:

        sql_context = serialize_sql_result(
            sql_result
        )

    else:

        sql_context = (
            "No SQL result was used."
        )

    if rag_result:

        rag_context = rag_result

    else:

        rag_context = (
            "No RAG result was used."
        )

    user_prompt = f"""
Original user question:

{original_question}


SQL information:

{sql_context}


RAG information:

{rag_context}


Generate the final answer for the user.
"""

    return call_ollama(
        system_prompt=FINAL_ANSWER_PROMPT,
        user_prompt=user_prompt,
        temperature=0.2,
    )


# ============================================================
# SQL WORKFLOW
# ============================================================

def handle_sql(
    question: str,
) -> Dict[str, Any]:
    """
    Execute the SQL route.

    Existing SQL engine handles:
    - SQL generation
    - SQL validation
    - BigQuery execution
    """

    sql_result = ask_sql(
        question
    )

    if not isinstance(
        sql_result,
        dict,
    ):

        raise ValueError(
            "SQL engine returned an invalid result."
        )

    if not sql_result.get(
        "success"
    ):

        return {
            "success": False,
            "route": "sql",
            "answer": (
                sql_result.get("reason")
                or sql_result.get("error")
                or "The SQL query could not be completed."
            ),
            "sql_result": sql_result,
        }

    final_answer = generate_final_answer(
        original_question=question,
        sql_result=sql_result,
    )

    return {
        "success": True,
        "route": "sql",
        "answer": final_answer,
        "sql_result": sql_result,
    }


# ============================================================
# RAG WORKFLOW
# ============================================================

def handle_rag(
    question: str,
) -> Dict[str, Any]:
    """
    Execute the RAG route.

    Existing RAG engine handles:
    - question embedding
    - vector search
    - progressive candidate expansion
    - duplicate filtering
    - diversity filtering
    - review context
    - RAG answer generation
    """

    rag_answer, top_reviews = (
        answer_review_question(
            question
        )
    )

    return {
        "success": True,
        "route": "rag",
        "answer": rag_answer,
        "rag_answer": rag_answer,
        "reviews": top_reviews,
    }


# ============================================================
# HYBRID WORKFLOW
# ============================================================

def handle_hybrid(
    question: str,
) -> Dict[str, Any]:
    """
    Execute the complete HYBRID workflow.

    Step 1:
        Original hybrid question
        ->
        Hybrid SQL planner

    Step 2:
        SQL-only question
        ->
        SQL engine

    Step 3:
        SQL result
        ->
        RAG query planner

    Step 4:
        Focused RAG question
        ->
        RAG engine

    Step 5:
        SQL result + RAG result
        ->
        Final LLM

    Step 6:
        Final LLM
        ->
        User answer
    """

    # ========================================================
    # STEP 1 - EXTRACT SQL TASK
    # ========================================================

    sql_question = generate_sql_question(
        original_question=question,
    )

    # ========================================================
    # STEP 2 - SQL
    # ========================================================

    sql_result = ask_sql(
        sql_question
    )

    if not isinstance(
        sql_result,
        dict,
    ):

        raise ValueError(
            "SQL engine returned an invalid result."
        )

    if not sql_result.get(
        "success"
    ):

        return {
            "success": False,
            "route": "hybrid",
            "answer": (
                "I could not complete the structured "
                "part of the question."
            ),
            "sql_question": sql_question,
            "sql_result": sql_result,
        }

    # ========================================================
    # STEP 3 - QUERY PLANNER
    # ========================================================

    rag_question = generate_rag_question(
        original_question=question,
        sql_result=sql_result,
    )

    # ========================================================
    # STEP 4 - RAG
    # ========================================================

    rag_answer, top_reviews = (
        answer_review_question(
            rag_question
        )
    )

    # ========================================================
    # STEP 5 - FINAL LLM
    # ========================================================

    final_answer = generate_final_answer(
        original_question=question,
        sql_result=sql_result,
        rag_result=rag_answer,
    )

    # ========================================================
    # STEP 6 - RETURN EVERYTHING IMPORTANT
    # ========================================================

    return {
        "success": True,
        "route": "hybrid",

        # Final user-facing response.
        "answer": final_answer,

        # SQL planner output.
        "sql_question": sql_question,

        # SQL engine output.
        "sql_result": sql_result,

        # RAG planner output.
        "rag_question": rag_question,

        # RAG engine output.
        "rag_answer": rag_answer,

        # Retrieved reviews.
        "reviews": top_reviews,
    }


# ============================================================
# MAIN PUBLIC FUNCTION
# ============================================================

def answer_question(
    question: str,
) -> Dict[str, Any]:
    """
    Main entry point for the complete Zomato AI system.

    Pipeline:

        Question
           |
           v
      Domain Guard
           |
       IN SCOPE?
        /     \
      NO       YES
      |         |
      v         v
    Return    Router
    safely      |
          +-----+-----+
          |     |     |
         SQL   RAG  HYBRID

    The Streamlit UI should call ONLY this function.
    """

    # ========================================================
    # VALIDATE INPUT
    # ========================================================

    if not isinstance(
        question,
        str,
    ):

        raise ValueError(
            "Question must be a string."
        )

    question = normalize_text(
        question
    )

    if not question:

        raise ValueError(
            "Question cannot be empty."
        )

    # ========================================================
    # STEP 1 - DOMAIN GUARD
    # ========================================================
    #
    # THIS MUST HAPPEN BEFORE THE ROUTER.
    #
    # This prevents unrelated questions from reaching:
    #
    #     SQL
    #     RAG
    #     HYBRID
    #
    # Examples:
    #
    # "Which is the best product on Amazon?"
    # "Who is the Prime Minister of India?"
    # "What is 2 + 2?"
    # "What do customers say about LG TV?"
    #
    # These should stop here.
    # ========================================================

    try:

        domain_result = validate_domain(
            question
        )

    except Exception as exc:

        # If the Domain Guard itself fails, DO NOT silently
        # allow the question to reach SQL/RAG.
        #
        # This is important for safety of the domain boundary.

        return {
            "success": False,
            "route": "domain_guard",
            "answer": (
                "I could not verify whether this question "
                "is related to the Zomato data."
            ),
            "error": str(exc),
            "domain_guard": {
                "in_scope": False,
                "confidence": 0.0,
                "reason": (
                    "Domain guard failed, so the request "
                    "was blocked."
                ),
            },
        }

    # ========================================================
    # BLOCK OUT-OF-SCOPE QUESTIONS
    # ========================================================

    if not domain_result.get(
        "in_scope",
        False,
    ):

        guard_answer = normalize_text(
            domain_result.get(
                "answer"
            )
            or DOMAIN_GUARD_FALLBACK_MESSAGE
        )

        return {
            "success": True,

            # Special route used by UI.
            "route": "domain_guard",

            # User-facing answer.
            "answer": guard_answer,

            # Preserve the complete domain decision
            # for debugging/UI.
            "domain_guard": domain_result,

            # Router was intentionally NOT called.
            "routing": {
                "route": "domain_guard",
                "confidence": domain_result.get(
                    "confidence"
                ),
                "reason": domain_result.get(
                    "reason"
                ),
                "question": question,
                "fallback": False,
            },
        }

    # ========================================================
    # STEP 2 - ROUTER
    # ========================================================

    routing = route_question(
        question
    )

    if not isinstance(
        routing,
        dict,
    ):

        raise ValueError(
            "Router returned an invalid result."
        )

    route = routing.get(
        "route"
    )

    # ========================================================
    # STEP 3 - EXECUTE ROUTE
    # ========================================================

    try:

        if route == "sql":

            result = handle_sql(
                question
            )

        elif route == "rag":

            result = handle_rag(
                question
            )

        elif route == "hybrid":

            result = handle_hybrid(
                question
            )

        else:

            raise ValueError(
                f"Unknown route returned by router: {route}"
            )

        # ----------------------------------------------------
        # Preserve both Domain Guard and Router information.
        # ----------------------------------------------------

        result["domain_guard"] = (
            domain_result
        )

        result["routing"] = routing

        return result

    except Exception as exc:

        # ====================================================
        # CENTRALIZED ORCHESTRATION ERROR HANDLING
        # ====================================================

        return {
            "success": False,

            "route": route,

            "answer": (
                "I could not complete the request because "
                "an internal processing step failed."
            ),

            "error": str(exc),

            "domain_guard": domain_result,

            "routing": routing,
        }


# ============================================================
# OPTIONAL COMMAND-LINE TEST
# ============================================================

if __name__ == "__main__":

    print("=" * 70)
    print("ZOMATO AI - ORCHESTRATOR TEST")
    print("=" * 70)

    test_questions = [
        "Which restaurant generated the most revenue?",
        "What do customers say about food quality?",
        "Which restaurant generated the most revenue and what do customers say about it?",
        "Which is the best product on Amazon?",
        "Who is the Prime Minister of India?",
        "What is 2 + 2?",
    ]

    for question in test_questions:

        print()
        print("-" * 70)
        print(f"QUESTION: {question}")
        print("-" * 70)

        try:

            result = answer_question(
                question
            )

            print(
                json.dumps(
                    result,
                    indent=2,
                    default=str,
                    ensure_ascii=False,
                )
            )

        except Exception as exc:

            print(
                json.dumps(
                    {
                        "success": False,
                        "error": str(exc),
                    },
                    indent=2,
                    ensure_ascii=False,
                )
            )