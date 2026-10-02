"""
ZOMATO AI - PROGRESSIVE + DIVERSE + EVIDENCE-AWARE RAG ENGINE

Purpose
-------
Production-ready RAG engine for the Zomato AI project.

Architecture
------------
                    User Question
                         |
                         v
                 Question Embedding
                         |
                         v
             +-------------------------+
             | Progressive Retrieval   |
             |                         |
             | Top 100                 |
             |    |                    |
             |    v                    |
             | Top 500                 |
             |    |                    |
             |    v                    |
             | Top 1000                |
             +-------------------------+
                         |
                         v
              Relevance Filtering
                         |
                         v
             Exact Duplicate Removal
                         |
                         v
           Near-Duplicate Text Removal
                         |
                         v
             Diversity / Pattern Filter
                         |
                         v
                  Final Reviews
                         |
                         v
               Evidence-Aware LLM
                         |
                         v
                    RAG Answer


Important
---------
This file is designed to be imported by:

    orchestrator.py
    Streamlit UI
    future API layer

The persistent index is built separately using:

    build_rag_index.py

This file NEVER reloads the original 60 parquet files.
It only loads:

    rag_index/embeddings.npy
    rag_index/reviews.parquet
    rag_index/metadata.json
"""

from __future__ import annotations

import json
import os
import re
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import ollama


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

INDEX_DIR = BASE_DIR / "rag_index"

EMBEDDINGS_FILE = INDEX_DIR / "embeddings.npy"
REVIEWS_FILE = INDEX_DIR / "reviews.parquet"
METADATA_FILE = INDEX_DIR / "metadata.json"


# Ollama models
EMBEDDING_MODEL = "mxbai-embed-large:latest"
LLM_MODEL = "gpt-oss:120b-cloud"


# ------------------------------------------------------------
# Progressive retrieval
# ------------------------------------------------------------

CANDIDATE_LEVELS = [100, 500, 1000]

FINAL_K = 5


# ------------------------------------------------------------
# Relevance
# ------------------------------------------------------------

RELEVANCE_THRESHOLD = 0.50


# ------------------------------------------------------------
# Text diversity
# ------------------------------------------------------------

TEXT_DUPLICATE_THRESHOLD = 0.75


# ------------------------------------------------------------
# Semantic diversity
#
# A selected review should not be too similar to a previously
# selected review.
# ------------------------------------------------------------

SELECTED_SIMILARITY_MAX = 0.70


# ------------------------------------------------------------
# Dominant pattern protection
#
# Prevents 5 almost identical reviews from dominating the final
# evidence.
# ------------------------------------------------------------

DOMINANT_PATTERN_RATIO = 0.60

MAX_SAME_PATTERN = 2


# ------------------------------------------------------------
# Answer configuration
# ------------------------------------------------------------

MAX_REVIEW_TEXT_LENGTH = 1000


# ============================================================
# GLOBAL CACHE
# ============================================================

_INDEX_CACHE: Dict[str, Any] | None = None


# ============================================================
# UTILITY
# ============================================================

def _print_separator(char: str = "=", width: int = 70) -> None:
    print(char * width)


# ============================================================
# TEXT NORMALIZATION
# ============================================================

def normalize_text(text: Any) -> str:
    """
    Normalize review text for duplicate detection.

    Examples:
        "The FOOD was absolutely delicious!"
        "the food was absolutely delicious"

    become approximately the same normalized representation.
    """

    if text is None:
        return ""

    text = str(text).lower().strip()

    text = re.sub(r"\s+", " ", text)

    text = re.sub(
        r"[^\w\s]",
        "",
        text,
        flags=re.UNICODE,
    )

    return text.strip()


# ============================================================
# TOKEN SET SIMILARITY
# ============================================================

def token_similarity(text_a: str, text_b: str) -> float:
    """
    Calculate Jaccard similarity between two review texts.

    This avoids sklearn dependency.

    Returns:
        value between 0 and 1
    """

    a = set(normalize_text(text_a).split())
    b = set(normalize_text(text_b).split())

    if not a and not b:
        return 1.0

    if not a or not b:
        return 0.0

    intersection = len(a.intersection(b))
    union = len(a.union(b))

    if union == 0:
        return 0.0

    return intersection / union


# ============================================================
# LOAD PERSISTENT INDEX
# ============================================================

def _load_index() -> Dict[str, Any]:
    """
    Load the persistent RAG index.

    Uses memory mapping for embeddings so the 1.14 GB matrix
    does not need to be copied unnecessarily into RAM.
    """

    global _INDEX_CACHE

    if _INDEX_CACHE is not None:
        return _INDEX_CACHE

    print()
    _print_separator()
    print("LOADING PERSISTENT RAG INDEX")
    _print_separator()

    print(f"Index directory : {INDEX_DIR}")
    print(f"Embeddings file : {EMBEDDINGS_FILE}")
    print(f"Reviews file    : {REVIEWS_FILE}")
    print(f"Metadata file   : {METADATA_FILE}")

    if not INDEX_DIR.exists():
        raise FileNotFoundError(
            f"RAG index directory does not exist:\n{INDEX_DIR}\n\n"
            "Run build_rag_index.py first."
        )

    required_files = [
        EMBEDDINGS_FILE,
        REVIEWS_FILE,
        METADATA_FILE,
    ]

    for file_path in required_files:
        if not file_path.exists():
            raise FileNotFoundError(
                f"Required RAG index file is missing:\n{file_path}"
            )

    # --------------------------------------------------------
    # Embeddings
    # --------------------------------------------------------

    print()
    print("Loading embeddings.npy with memory mapping...")

    embeddings = np.load(
        EMBEDDINGS_FILE,
        mmap_mode="r",
    )

    # --------------------------------------------------------
    # Reviews
    # --------------------------------------------------------

    print("Loading reviews.parquet...")

    reviews = pd.read_parquet(
        REVIEWS_FILE,
        engine="pyarrow",
    )

    # --------------------------------------------------------
    # Metadata
    # --------------------------------------------------------

    print("Loading metadata.json...")

    with open(
        METADATA_FILE,
        "r",
        encoding="utf-8",
    ) as f:
        metadata = json.load(f)

    # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------

    if len(embeddings) != len(reviews):
        raise ValueError(
            "RAG index is inconsistent.\n"
            f"Embeddings: {len(embeddings)}\n"
            f"Reviews:    {len(reviews)}"
        )

    if embeddings.ndim != 2:
        raise ValueError(
            f"Expected 2D embedding matrix, got shape "
            f"{embeddings.shape}"
        )

    if "review_id" not in reviews.columns:
        raise ValueError(
            "reviews.parquet does not contain review_id."
        )

    if "comment" not in reviews.columns:
        raise ValueError(
            "reviews.parquet does not contain comment."
        )

    _INDEX_CACHE = {
        "embeddings": embeddings,
        "reviews": reviews,
        "metadata": metadata,
    }

    print()
    _print_separator()
    print("RAG INDEX READY")
    _print_separator()

    print(f"Reviews indexed    : {len(reviews):,}")
    print(
        f"Embedding dimension: {embeddings.shape[1]:,}"
    )
    print(
        f"Embedding dtype    : {embeddings.dtype}"
    )
    print(
        f"Review rows        : {len(reviews):,}"
    )
    print("Index cache        : ENABLED")
    print("Memory mapping     : ENABLED")

    return _INDEX_CACHE


# ============================================================
# PUBLIC INDEX INFORMATION
# ============================================================

def get_rag_index_info() -> Dict[str, Any]:
    """
    Return index information for Streamlit/UI.

    Example:

        info = get_rag_index_info()

    """

    index = _load_index()

    embeddings = index["embeddings"]
    reviews = index["reviews"]
    metadata = index["metadata"]

    return {
        "index_directory": str(INDEX_DIR),
        "reviews": int(len(reviews)),
        "embedding_dimension": int(
            embeddings.shape[1]
        ),
        "embedding_dtype": str(
            embeddings.dtype
        ),
        "embedding_shape": list(
            embeddings.shape
        ),
        "memory_mapped": True,
        "cached": _INDEX_CACHE is not None,
        "metadata": metadata,
    }


# ============================================================
# QUESTION EMBEDDING
# ============================================================

def generate_question_embedding(
    question: str,
) -> np.ndarray:
    """
    Generate a normalized embedding for the user question.
    """

    if not question or not question.strip():
        raise ValueError(
            "Question cannot be empty."
        )

    response = ollama.embeddings(
        model=EMBEDDING_MODEL,
        prompt=question.strip(),
    )

    if "embedding" not in response:
        raise RuntimeError(
            "Ollama did not return an embedding."
        )

    vector = np.asarray(
        response["embedding"],
        dtype=np.float32,
    )

    norm = np.linalg.norm(vector)

    if norm == 0:
        raise ValueError(
            "Question embedding has zero magnitude."
        )

    vector = vector / norm

    return vector


# ============================================================
# SEMANTIC SEARCH
# ============================================================

def semantic_search(
    question_embedding: np.ndarray,
    candidate_k: int,
) -> pd.DataFrame:
    """
    Search the persistent embedding matrix.

    Because embeddings are normalized, dot product is cosine
    similarity.
    """

    index = _load_index()

    embeddings = index["embeddings"]
    reviews = index["reviews"]

    candidate_k = min(
        candidate_k,
        len(reviews),
    )

    # --------------------------------------------------------
    # Calculate cosine similarity.
    #
    # Chunking prevents unnecessary giant temporary arrays.
    # --------------------------------------------------------

    chunk_size = 50_000

    scores = np.empty(
        len(embeddings),
        dtype=np.float32,
    )

    for start in range(
        0,
        len(embeddings),
        chunk_size,
    ):
        end = min(
            start + chunk_size,
            len(embeddings),
        )

        chunk = embeddings[start:end]

        scores[start:end] = np.asarray(
            np.dot(
                chunk,
                question_embedding,
            ),
            dtype=np.float32,
        )

    # --------------------------------------------------------
    # Top candidate indices
    # --------------------------------------------------------

    if candidate_k < len(scores):

        candidate_indices = np.argpartition(
            scores,
            -candidate_k,
        )[-candidate_k:]

        candidate_indices = candidate_indices[
            np.argsort(
                scores[candidate_indices]
            )[::-1]
        ]

    else:

        candidate_indices = np.argsort(
            scores
        )[::-1]

    result = reviews.iloc[
        candidate_indices
    ].copy()

    result["score"] = scores[
        candidate_indices
    ]

    result.reset_index(
        drop=True,
        inplace=True,
    )

    return result


# ============================================================
# RELEVANCE FILTER
# ============================================================

def apply_relevance_filter(
    candidates: pd.DataFrame,
) -> pd.DataFrame:
    """
    Remove reviews below semantic relevance threshold.
    """

    if candidates.empty:
        return candidates.copy()

    result = candidates[
        candidates["score"]
        >= RELEVANCE_THRESHOLD
    ].copy()

    result.reset_index(
        drop=True,
        inplace=True,
    )

    return result


# ============================================================
# EXACT DUPLICATE REMOVAL
# ============================================================

def remove_exact_duplicates(
    candidates: pd.DataFrame,
) -> Tuple[pd.DataFrame, int]:
    """
    Remove reviews whose normalized text is exactly identical.
    """

    if candidates.empty:
        return candidates.copy(), 0

    seen = set()
    keep_rows = []

    removed = 0

    for idx, row in candidates.iterrows():

        normalized = normalize_text(
            row["comment"]
        )

        if normalized in seen:
            removed += 1
            continue

        seen.add(normalized)
        keep_rows.append(idx)

    result = candidates.loc[
        keep_rows
    ].copy()

    result.reset_index(
        drop=True,
        inplace=True,
    )

    return result, removed


# ============================================================
# NEAR DUPLICATE REMOVAL
# ============================================================

def remove_near_duplicates(
    candidates: pd.DataFrame,
) -> Tuple[pd.DataFrame, int]:
    """
    Remove reviews whose wording is too similar.

    Important:
    We only compare against reviews already retained.

    This keeps the highest semantic-score review among
    near-identical reviews.
    """

    if candidates.empty:
        return candidates.copy(), 0

    kept_indices = []

    removed = 0

    for idx, row in candidates.iterrows():

        current_text = row["comment"]

        is_duplicate = False

        for kept_idx in kept_indices:

            kept_text = candidates.loc[
                kept_idx,
                "comment",
            ]

            similarity = token_similarity(
                current_text,
                kept_text,
            )

            if similarity >= TEXT_DUPLICATE_THRESHOLD:

                is_duplicate = True
                removed += 1
                break

        if not is_duplicate:
            kept_indices.append(idx)

    result = candidates.loc[
        kept_indices
    ].copy()

    result.reset_index(
        drop=True,
        inplace=True,
    )

    return result, removed


# ============================================================
# REVIEW PATTERN DETECTION
# ============================================================

def detect_review_pattern(
    comment: str,
) -> str:
    """
    Create a lightweight semantic pattern label.

    This is NOT sentiment analysis.

    It is only used to prevent one repeated review template
    from occupying the entire final evidence set.
    """

    text = normalize_text(comment)

    patterns = []

    # --------------------------------------------------------
    # Food
    # --------------------------------------------------------

    food_positive = [
        "delicious",
        "tasty",
        "amazing food",
        "great food",
        "excellent food",
        "best meal",
        "fresh",
        "flavorful",
        "yummy",
    ]

    food_negative = [
        "bad food",
        "terrible food",
        "cold food",
        "stale",
        "tasteless",
        "bland",
        "not tasty",
        "food was bad",
    ]

    # --------------------------------------------------------
    # Delivery
    # --------------------------------------------------------

    delivery_positive = [
        "delivery was quick",
        "delivered quickly",
        "fast delivery",
        "delivery partner was polite",
        "delivery partner was helpful",
        "delivery was good",
    ]

    delivery_negative = [
        "late delivery",
        "delivery was late",
        "delayed delivery",
        "delivery partner was rude",
        "delivery was bad",
        "never arrived",
    ]

    # --------------------------------------------------------
    # Service
    # --------------------------------------------------------

    service_positive = [
        "good service",
        "great service",
        "excellent service",
        "friendly staff",
        "helpful staff",
    ]

    service_negative = [
        "bad service",
        "poor service",
        "rude staff",
        "unhelpful staff",
    ]

    # --------------------------------------------------------
    # Match
    # --------------------------------------------------------

    if any(
        phrase in text
        for phrase in food_positive
    ):
        patterns.append("food_positive")

    if any(
        phrase in text
        for phrase in food_negative
    ):
        patterns.append("food_negative")

    if any(
        phrase in text
        for phrase in delivery_positive
    ):
        patterns.append("delivery_positive")

    if any(
        phrase in text
        for phrase in delivery_negative
    ):
        patterns.append("delivery_negative")

    if any(
        phrase in text
        for phrase in service_positive
    ):
        patterns.append("service_positive")

    if any(
        phrase in text
        for phrase in service_negative
    ):
        patterns.append("service_negative")

    if not patterns:
        patterns.append("other")

    return "|".join(sorted(patterns))


# ============================================================
# DIVERSITY SELECTION
# ============================================================

def select_diverse_reviews(
    candidates: pd.DataFrame,
    final_k: int = FINAL_K,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Select final reviews while balancing:

    1. Semantic relevance
    2. Different wording
    3. Different semantic content
    4. Review-pattern distribution

    Important design decision:

    We DO NOT blindly require five completely different
    reviews.

    If the actual evidence contains only two distinct review
    patterns, we return two rather than inserting irrelevant
    reviews merely to reach five.
    """

    if candidates.empty:

        return (
            candidates.copy(),
            {
                "selected_count": 0,
                "dominant_patterns": [],
                "distribution": {},
            },
        )

    # --------------------------------------------------------
    # Pattern assignment
    # --------------------------------------------------------

    working = candidates.copy()

    working["pattern"] = working[
        "comment"
    ].apply(
        detect_review_pattern
    )

    # --------------------------------------------------------
    # Determine dominant patterns
    # --------------------------------------------------------

    pattern_counts = (
        working["pattern"]
        .value_counts()
        .to_dict()
    )

    total = len(working)

    dominant_patterns = []

    for pattern, count in pattern_counts.items():

        if (
            total > 0
            and count / total
            >= DOMINANT_PATTERN_RATIO
        ):
            dominant_patterns.append(
                pattern
            )

    # --------------------------------------------------------
    # Selection
    # --------------------------------------------------------

    selected_indices: List[int] = []

    pattern_selected: Dict[str, int] = {}

    selected_texts: List[str] = []

    selected_embedding_indices = []

    index = _load_index()

    all_embeddings = index["embeddings"]

    # --------------------------------------------------------
    # Map review IDs back to embedding rows.
    #
    # We use review_id because the reviews dataframe has been
    # reordered by semantic search.
    # --------------------------------------------------------

    review_id_to_index = {
        str(review_id): idx
        for idx, review_id in zip(
            index["reviews"]["review_id"],
            range(len(index["reviews"])),
        )
    }

    # --------------------------------------------------------
    # First pass:
    #
    # Prefer different patterns.
    # --------------------------------------------------------

    for idx, row in working.iterrows():

        if len(selected_indices) >= final_k:
            break

        pattern = row["pattern"]

        count_for_pattern = pattern_selected.get(
            pattern,
            0,
        )

        # ----------------------------------------------------
        # Prevent a dominant pattern from consuming all slots.
        # ----------------------------------------------------

        if (
            pattern in dominant_patterns
            and count_for_pattern >= MAX_SAME_PATTERN
        ):
            continue

        # ----------------------------------------------------
        # Text-level diversity
        # ----------------------------------------------------

        text_is_too_similar = False

        for previous_text in selected_texts:

            similarity = token_similarity(
                row["comment"],
                previous_text,
            )

            if similarity >= TEXT_DUPLICATE_THRESHOLD:

                text_is_too_similar = True
                break

        if text_is_too_similar:
            continue

        # ----------------------------------------------------
        # Semantic diversity
        # ----------------------------------------------------

        review_id = str(
            row["review_id"]
        )

        embedding_idx = review_id_to_index.get(
            review_id
        )

        if embedding_idx is None:
            continue

        current_embedding = np.asarray(
            all_embeddings[
                embedding_idx
            ],
            dtype=np.float32,
        )

        semantic_duplicate = False

        for previous_embedding in selected_embedding_indices:

            similarity = float(
                np.dot(
                    current_embedding,
                    previous_embedding,
                )
            )

            if (
                similarity
                >= SELECTED_SIMILARITY_MAX
            ):
                semantic_duplicate = True
                break

        if semantic_duplicate:
            continue

        # ----------------------------------------------------
        # Accept
        # ----------------------------------------------------

        selected_indices.append(idx)

        selected_texts.append(
            row["comment"]
        )

        selected_embedding_indices.append(
            current_embedding
        )

        pattern_selected[pattern] = (
            count_for_pattern + 1
        )

    # --------------------------------------------------------
    # Second pass
    #
    # If we still don't have enough reviews, relax pattern
    # restriction but KEEP duplicate protection.
    #
    # This prevents the system from returning only one review
    # when genuinely different reviews exist.
    # --------------------------------------------------------

    if len(selected_indices) < final_k:

        for idx, row in working.iterrows():

            if len(selected_indices) >= final_k:
                break

            if idx in selected_indices:
                continue

            text_is_too_similar = False

            for previous_text in selected_texts:

                similarity = token_similarity(
                    row["comment"],
                    previous_text,
                )

                if similarity >= TEXT_DUPLICATE_THRESHOLD:

                    text_is_too_similar = True
                    break

            if text_is_too_similar:
                continue

            review_id = str(
                row["review_id"]
            )

            embedding_idx = review_id_to_index.get(
                review_id
            )

            if embedding_idx is None:
                continue

            current_embedding = np.asarray(
                all_embeddings[
                    embedding_idx
                ],
                dtype=np.float32,
            )

            semantic_duplicate = False

            for previous_embedding in selected_embedding_indices:

                similarity = float(
                    np.dot(
                        current_embedding,
                        previous_embedding,
                    )
                )

                if (
                    similarity
                    >= SELECTED_SIMILARITY_MAX
                ):
                    semantic_duplicate = True
                    break

            if semantic_duplicate:
                continue

            selected_indices.append(idx)

            selected_texts.append(
                row["comment"]
            )

            selected_embedding_indices.append(
                current_embedding
            )

            pattern = row["pattern"]

            pattern_selected[pattern] = (
                pattern_selected.get(
                    pattern,
                    0,
                )
                + 1
            )

    # --------------------------------------------------------
    # Final result
    # --------------------------------------------------------

    result = working.loc[
        selected_indices
    ].copy()

    result.reset_index(
        drop=True,
        inplace=True,
    )

    diagnostics = {
        "selected_count": len(result),
        "dominant_patterns": dominant_patterns,
        "distribution": pattern_selected,
    }

    return result, diagnostics


# ============================================================
# PROGRESSIVE RETRIEVAL
# ============================================================

def progressive_retrieve(
    question: str,
    question_embedding: np.ndarray,
) -> Tuple[
    pd.DataFrame,
    Dict[str, Any],
]:
    """
    Progressive retrieval architecture:

        Top 100
           |
           | not enough diversity
           v
        Top 500
           |
           | not enough diversity
           v
        Top 1000
           |
           v
        Final evidence

    This prevents unnecessary retrieval of 1000 reviews
    when 100 already provide sufficient evidence.
    """

    print()
    _print_separator()
    print("PROGRESSIVE DIVERSE RETRIEVAL")
    _print_separator()

    final_candidates = pd.DataFrame()

    final_diagnostics = {}

    stopped_early = False

    for stage_index, candidate_level in enumerate(
        CANDIDATE_LEVELS
    ):

        print()
        print(
            f"[Stage {stage_index + 1}] "
            f"Searching top {candidate_level:,} candidates..."
        )

        candidates = semantic_search(
            question_embedding,
            candidate_level,
        )

        print(
            f"Semantic candidates: "
            f"{len(candidates):,}"
        )

        # ----------------------------------------------------
        # Relevance
        # ----------------------------------------------------

        candidates = apply_relevance_filter(
            candidates
        )

        print(
            f"After relevance threshold "
            f"({RELEVANCE_THRESHOLD:.2f}): "
            f"{len(candidates):,}"
        )

        # ----------------------------------------------------
        # Exact duplicates
        # ----------------------------------------------------

        candidates, exact_removed = (
            remove_exact_duplicates(
                candidates
            )
        )

        print(
            f"Exact duplicates removed: "
            f"{exact_removed:,}"
        )

        # ----------------------------------------------------
        # Near duplicates
        # ----------------------------------------------------

        candidates, near_removed = (
            remove_near_duplicates(
                candidates
            )
        )

        print(
            f"Near-duplicate reviews removed: "
            f"{near_removed:,}"
        )

        print(
            f"Different review pool: "
            f"{len(candidates):,}"
        )

        # ----------------------------------------------------
        # Diversity
        # ----------------------------------------------------

        selected, diagnostics = (
            select_diverse_reviews(
                candidates,
                FINAL_K,
            )
        )

        print(
            f"Sufficiently different reviews: "
            f"{len(selected):,}"
        )

        final_candidates = selected

        final_diagnostics = {
            "candidate_level": candidate_level,
            "semantic_candidates": len(
                candidates
            ),
            "relevant_candidates": len(
                candidates
            ),
            "exact_duplicates_removed": (
                exact_removed
            ),
            "near_duplicates_removed": (
                near_removed
            ),
            "selected_count": len(
                selected
            ),
            "dominant_patterns": diagnostics[
                "dominant_patterns"
            ],
            "distribution": diagnostics[
                "distribution"
            ],
        }

        # ----------------------------------------------------
        # Enough evidence?
        # ----------------------------------------------------

        if len(selected) >= FINAL_K:

            stopped_early = True

            print()
            print(
                f"✓ Found {FINAL_K} sufficiently "
                f"different reviews."
            )

            print(
                f"✓ Stopping at top "
                f"{candidate_level:,}."
            )

            break

        # ----------------------------------------------------
        # Expand
        # ----------------------------------------------------

        if stage_index < len(
            CANDIDATE_LEVELS
        ) - 1:

            next_level = CANDIDATE_LEVELS[
                stage_index + 1
            ]

            print()
            print(
                f"Only {len(selected)} "
                f"sufficiently different reviews found."
            )

            print(
                f"→ Expanding retrieval: "
                f"{candidate_level:,} → "
                f"{next_level:,}"
            )

        else:

            print()
            print(
                "→ Maximum candidate pool reached."
            )

    final_diagnostics[
        "stopped_early"
    ] = stopped_early

    final_diagnostics[
        "final_reviews"
    ] = len(final_candidates)

    return (
        final_candidates,
        final_diagnostics,
    )


# ============================================================
# EVIDENCE FORMATTER
# ============================================================

def format_evidence(
    reviews: pd.DataFrame,
) -> str:
    """
    Convert retrieved reviews into compact evidence for the LLM.
    """

    if reviews.empty:
        return (
            "No relevant customer reviews were "
            "retrieved."
        )

    evidence_parts = []

    for _, row in reviews.iterrows():

        review_id = row.get(
            "review_id",
            "unknown",
        )

        rating = row.get(
            "rating",
            "unknown",
        )

        comment = str(
            row.get(
                "comment",
                "",
            )
        )

        comment = comment.strip()

        if len(comment) > MAX_REVIEW_TEXT_LENGTH:

            comment = (
                comment[
                    :MAX_REVIEW_TEXT_LENGTH
                ]
                + "..."
            )

        evidence_parts.append(
            f"Review ID: {review_id}\n"
            f"Rating: {rating}\n"
            f"Comment: {comment}"
        )

    return "\n\n".join(
        evidence_parts
    )


# ============================================================
# FINAL LLM
# ============================================================

def generate_rag_answer(
    question: str,
    reviews: pd.DataFrame,
) -> str:
    """
    Generate an evidence-aware answer.

    The LLM is explicitly instructed NOT to infer the entire
    customer population from a small retrieved subset.
    """

    if reviews.empty:

        return (
            "I could not find sufficiently relevant "
            "customer reviews to answer this question."
        )

    evidence = format_evidence(
        reviews
    )

    prompt = f"""
You are the review-analysis component of a restaurant
analytics system.

User question:
{question}

Retrieved customer reviews:
--------------------------------
{evidence}
--------------------------------

Instructions:

1. Answer ONLY using the retrieved reviews.
2. Do not invent information.
3. Do not claim that all customers think something unless
   the evidence actually supports that statement.
4. Distinguish between:
   - what the retrieved customers said
   - what can safely be concluded
5. If the retrieved evidence is limited, explicitly say so.
6. If reviews contain both positive and negative opinions,
   mention both.
7. Do not ignore negative evidence.
8. Do not ignore positive evidence.
9. Do not treat repeated identical review templates as
   independent evidence of a broad customer consensus.
10. Give a concise, useful answer.
11. Mention review IDs when useful for traceability.

Answer:
"""

    response = ollama.chat(
        model=LLM_MODEL,
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
    )

    if (
        not response
        or "message" not in response
        or "content" not in response["message"]
    ):
        raise RuntimeError(
            "Ollama did not return a valid LLM response."
        )

    return response["message"][
        "content"
    ].strip()


# ============================================================
# MAIN RAG API
# ============================================================

def answer_review_question(
    question: str,
) -> Tuple[str, pd.DataFrame]:
    """
    Main RAG function.

    This is the function that orchestrator.py should call.

    Returns:

        (
            rag_answer,
            top_reviews
        )

    Example:

        answer, reviews = answer_review_question(
            "What do customers say about food quality?"
        )
    """

    if not question or not question.strip():

        return (
            "Please provide a question.",
            pd.DataFrame(),
        )

    # --------------------------------------------------------
    # Ensure index is loaded.
    # --------------------------------------------------------

    _load_index()

    # --------------------------------------------------------
    # Question embedding
    # --------------------------------------------------------

    print()
    print(
        "Generating question embedding..."
    )

    question_embedding = (
        generate_question_embedding(
            question
        )
    )

    # --------------------------------------------------------
    # Progressive retrieval
    # --------------------------------------------------------

    reviews, diagnostics = (
        progressive_retrieve(
            question,
            question_embedding,
        )
    )

    # --------------------------------------------------------
    # Final answer
    # --------------------------------------------------------

    answer = generate_rag_answer(
        question,
        reviews,
    )

    return (
        answer,
        reviews,
    )


# ============================================================
# STREAMLIT-FRIENDLY HIGH LEVEL API
# ============================================================

def query_rag(
    question: str,
) -> Dict[str, Any]:
    """
    Streamlit-friendly wrapper.

    Instead of returning only:

        answer, reviews

    this returns a complete structured dictionary.

    Example:

        result = query_rag(question)

        st.write(result["answer"])
        st.dataframe(result["reviews"])
    """

    if not question or not question.strip():

        return {
            "success": False,
            "answer": "Please enter a question.",
            "reviews": pd.DataFrame(),
            "diagnostics": {},
        }

    try:

        _load_index()

        question_embedding = (
            generate_question_embedding(
                question
            )
        )

        reviews, diagnostics = (
            progressive_retrieve(
                question,
                question_embedding,
            )
        )

        answer = generate_rag_answer(
            question,
            reviews,
        )

        # ----------------------------------------------------
        # Convert dataframe to a UI-friendly representation
        # while retaining the dataframe itself.
        # ----------------------------------------------------

        display_columns = [
            column
            for column in [
                "review_id",
                "rating",
                "comment",
                "review_date",
                "score",
            ]
            if column in reviews.columns
        ]

        display_reviews = reviews[
            display_columns
        ].copy()

        return {
            "success": True,
            "question": question,
            "answer": answer,
            "reviews": display_reviews,
            "review_count": int(
                len(display_reviews)
            ),
            "diagnostics": diagnostics,
        }

    except Exception as exc:

        return {
            "success": False,
            "question": question,
            "answer": (
                "The review analysis could not "
                "be completed."
            ),
            "reviews": pd.DataFrame(),
            "review_count": 0,
            "diagnostics": {},
            "error": str(exc),
        }


# ============================================================
# INDEX PRELOAD
# ============================================================

def preload_rag_index() -> Dict[str, Any]:
    """
    Explicitly preload the persistent index.

    Useful for Streamlit startup.

    Example:

        preload_rag_index()
    """

    return get_rag_index_info()


# ============================================================
# COMMAND-LINE TEST
# ============================================================

def run_test() -> None:
    """
    Local test for the RAG engine.

    This section runs ONLY when:

        python rag_engine.py

    It does NOT run when imported by Streamlit/orchestrator.
    """

    _print_separator()

    print(
        "ZOMATO AI - "
        "PROGRESSIVE + DIVERSE + "
        "EVIDENCE-AWARE RAG ENGINE TEST"
    )

    _print_separator()

    print()
    print("Persistent index:")
    print(f"  {INDEX_DIR}")

    print()
    print("Embedding model:")
    print(f"  {EMBEDDING_MODEL}")

    print()
    print("LLM model:")
    print(f"  {LLM_MODEL}")

    print()
    print("Retrieval configuration:")
    print(
        f"  Candidate levels          : "
        f"{CANDIDATE_LEVELS}"
    )
    print(
        f"  Final K                   : "
        f"{FINAL_K}"
    )
    print(
        f"  Relevance threshold       : "
        f"{RELEVANCE_THRESHOLD}"
    )
    print(
        f"  Text duplicate threshold  : "
        f"{TEXT_DUPLICATE_THRESHOLD}"
    )
    print(
        f"  Selected similarity max   : "
        f"{SELECTED_SIMILARITY_MAX}"
    )
    print(
        f"  Dominant pattern ratio    : "
        f"{DOMINANT_PATTERN_RATIO}"
    )
    print(
        f"  Max same pattern          : "
        f"{MAX_SAME_PATTERN}"
    )

    print()
    print("Loading index...")

    info = get_rag_index_info()

    print()
    _print_separator()
    print("INDEX INFORMATION")
    _print_separator()

    print(
        f"Reviews             : "
        f"{info['reviews']:,}"
    )

    print(
        f"Embedding dimension : "
        f"{info['embedding_dimension']}"
    )

    print(
        f"Embedding dtype     : "
        f"{info['embedding_dtype']}"
    )

    print(
        f"Embedding shape     : "
        f"{info['embedding_shape']}"
    )

    print(
        f"Index cached        : "
        f"{info['cached']}"
    )

    print(
        f"Memory mapped       : "
        f"{info['memory_mapped']}"
    )

    print()
    _print_separator()
    print("TEST QUESTION")
    _print_separator()

    question = (
        "What do customers say about food quality?"
    )

    print()
    print(f"Question: {question}")

    answer, reviews = (
        answer_review_question(
            question
        )
    )

    print()
    _print_separator()
    print("RAG ANSWER")
    _print_separator()

    print(answer)

    print()
    _print_separator()
    print("FINAL RETRIEVED REVIEWS")
    _print_separator()

    if reviews.empty:

        print(
            "No reviews retrieved."
        )

    else:

        display_columns = [
            column
            for column in [
                "review_id",
                "rating",
                "comment",
                "review_date",
                "score",
            ]
            if column in reviews.columns
        ]

        print(
            reviews[
                display_columns
            ].to_string(
                index=False
            )
        )

    print()
    _print_separator()
    print("RAG ENGINE TEST COMPLETED")
    _print_separator()


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    run_test()
