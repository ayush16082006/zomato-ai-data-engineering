import os
import requests
import numpy as np
import pandas as pd
import streamlit as st

from google.cloud import bigquery
from dotenv import load_dotenv


# ============================================================
# LOAD ENVIRONMENT VARIABLES
# ============================================================

load_dotenv()


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ID = "zomato-ai-data-engineering"

REVIEWS_TABLE = f"{PROJECT_ID}.raw.reviews"


# ============================================================
# OLLAMA CONFIGURATION
# ============================================================

OLLAMA_URL = "http://localhost:11434"

# Embedding model
EMBEDDING_MODEL = "mxbai-embed-large:latest"

# Final answer model
CHAT_MODEL = "gpt-oss:120b-cloud"


# ============================================================
# RAG CONFIGURATION
# ============================================================

# Start with 500 reviews for testing.
# Later we can increase this.
NEW_REVIEWS = 500

# Number of reviews retrieved for each question
TOP_K = 5

# Ollama embedding batch size
EMBED_BATCH_SIZE = 32


# ============================================================
# CACHE
# ============================================================

# IMPORTANT:
# This is a NEW cache file.
# Do not reuse the old Gemini embedding cache.
CACHE_FILE = "review_embeddings_mxbai.parquet"


# ============================================================
# BIGQUERY CLIENT
# ============================================================

bq_client = bigquery.Client(
    project=PROJECT_ID
)


# ============================================================
# OLLAMA HEALTH CHECK
# ============================================================

def check_ollama():

    try:

        response = requests.get(
            f"{OLLAMA_URL}/api/tags",
            timeout=10
        )

        response.raise_for_status()

        models = response.json().get("models", [])

        model_names = [
            model.get("name")
            for model in models
        ]

        print("Ollama is running.")
        print("Available Ollama models:")

        for name in model_names:
            print(f"  - {name}")

        return True

    except Exception as e:

        st.error(
            "Could not connect to Ollama.\n\n"
            f"Error: {e}\n\n"
            "Make sure Ollama is running."
        )

        return False


# ============================================================
# LOAD REVIEWS FROM BIGQUERY
# ============================================================

def read_reviews_from_bigquery():

    query = f"""
        SELECT
            review_id,
            rating,
            comment,
            review_date
        FROM `{REVIEWS_TABLE}`
        WHERE comment IS NOT NULL
          AND TRIM(comment) != ''
        ORDER BY review_id
        LIMIT {NEW_REVIEWS}
    """

    print("Fetching reviews from BigQuery...")
    print(f"Table: {REVIEWS_TABLE}")

    df = (
        bq_client
        .query(query)
        .result()
        .to_dataframe()
    )

    df.columns = [
        col.lower()
        for col in df.columns
    ]

    print(
        f"Reviews fetched: {len(df)}"
    )

    return df


# ============================================================
# CREATE OLLAMA EMBEDDINGS
# ============================================================

def embed(texts):

    if not texts:
        return []

    all_embeddings = []

    total = len(texts)

    for start in range(
        0,
        total,
        EMBED_BATCH_SIZE
    ):

        batch = texts[
            start:start + EMBED_BATCH_SIZE
        ]

        print(
            f"Creating Ollama embeddings: "
            f"{start + 1}-{start + len(batch)} / {total}"
        )

        payload = {
            "model": EMBEDDING_MODEL,
            "input": batch
        }

        response = requests.post(
            f"{OLLAMA_URL}/api/embed",
            json=payload,
            timeout=300
        )

        response.raise_for_status()

        data = response.json()

        embeddings = data.get(
            "embeddings"
        )

        if not embeddings:

            raise RuntimeError(
                "Ollama did not return embeddings."
            )

        all_embeddings.extend(
            embeddings
        )

    return all_embeddings


# ============================================================
# LOAD REVIEWS + EMBEDDINGS
# ============================================================

@st.cache_data
def load_reviews():

    # --------------------------------------------------------
    # USE EXISTING OLLAMA CACHE
    # --------------------------------------------------------

    if os.path.exists(CACHE_FILE):

        st.info(
            "Loading reviews from local "
            "Ollama embedding cache..."
        )

        df = pd.read_parquet(
            CACHE_FILE
        )

        if (
            "embedding" in df.columns
            and len(df) > 0
        ):

            print(
                f"Loaded {len(df)} reviews "
                "from embedding cache."
            )

            return df

        st.warning(
            "Existing cache is invalid. "
            "Creating a new cache..."
        )

    # --------------------------------------------------------
    # LOAD REVIEWS FROM BIGQUERY
    # --------------------------------------------------------

    df = read_reviews_from_bigquery()

    # --------------------------------------------------------
    # CREATE OLLAMA EMBEDDINGS
    # --------------------------------------------------------

    st.info(
        f"Creating {EMBEDDING_MODEL} "
        f"embeddings for {len(df):,} reviews..."
    )

    embeddings = embed(
        df["comment"].tolist()
    )

    if len(embeddings) != len(df):

        raise RuntimeError(
            "Number of embeddings does not match "
            "number of reviews."
        )

    df["embedding"] = embeddings

    # --------------------------------------------------------
    # SAVE CACHE
    # --------------------------------------------------------

    df.to_parquet(
        CACHE_FILE,
        index=False
    )

    st.success(
        f"Created and cached embeddings for "
        f"{len(df):,} reviews."
    )

    return df


# ============================================================
# COSINE SIMILARITY
# ============================================================

def cosine_similarity(
    vec_a,
    vec_b
):

    vec_a = np.asarray(
        vec_a,
        dtype=np.float32
    )

    vec_b = np.asarray(
        vec_b,
        dtype=np.float32
    )

    denominator = (
        np.linalg.norm(vec_a)
        *
        np.linalg.norm(vec_b)
    )

    if denominator == 0:

        return 0.0

    return float(
        np.dot(
            vec_a,
            vec_b
        )
        /
        denominator
    )


# ============================================================
# FIND SIMILAR REVIEWS
# ============================================================

def find_similar_reviews(
    question,
    df
):

    # --------------------------------------------------------
    # EMBED USER QUESTION
    # --------------------------------------------------------

    question_embedding = embed(
        [question]
    )[0]

    # --------------------------------------------------------
    # CALCULATE SIMILARITY
    # --------------------------------------------------------

    scores = []

    for review_embedding in df[
        "embedding"
    ]:

        score = cosine_similarity(
            question_embedding,
            review_embedding
        )

        scores.append(score)

    # --------------------------------------------------------
    # ADD SCORES
    # --------------------------------------------------------

    result = df.copy()

    result["score"] = scores

    # --------------------------------------------------------
    # TOP K
    # --------------------------------------------------------

    return result.nlargest(
        TOP_K,
        "score"
    )


# ============================================================
# BUILD RAG CONTEXT
# ============================================================

def build_context(top_reviews):

    context = ""

    for _, row in top_reviews.iterrows():

        context += (
            f"Review ID: {row['review_id']}\n"
            f"Rating: {row['rating']} stars\n"
            f"Review: {row['comment']}\n"
            f"Review date: {row['review_date']}\n"
            f"Similarity score: "
            f"{row['score']:.4f}\n"
            f"---\n"
        )

    return context


# ============================================================
# ASK OLLAMA
# ============================================================

def ask_llm(
    question,
    top_reviews
):

    context = build_context(
        top_reviews
    )

    system_prompt = """
You are a RAG assistant for a Zomato review
analytics system.

Answer the user's question ONLY using the
customer reviews provided in the context.

Rules:

1. Do not invent information.

2. Do not use outside knowledge.

3. If the retrieved reviews do not contain
   enough information, clearly say that the
   available reviews do not provide enough
   information.

4. Keep the answer concise and easy to understand.

5. When useful, mention relevant review IDs.

6. Do not claim that the retrieved reviews
   represent all Zomato reviews.

7. Base your answer only on the retrieved
   customer reviews.
"""

    user_prompt = f"""
Question:
{question}

Retrieved customer reviews:

{context}

Answer the question using ONLY the
retrieved reviews.
"""

    payload = {

        "model": CHAT_MODEL,

        "messages": [

            {
                "role": "system",
                "content": system_prompt
            },

            {
                "role": "user",
                "content": user_prompt
            }

        ],

        "stream": False,

        "options": {
            "temperature": 0.2
        }
    }

    response = requests.post(
        f"{OLLAMA_URL}/api/chat",
        json=payload,
        timeout=300
    )

    response.raise_for_status()

    data = response.json()

    message = data.get(
        "message",
        {}
    )

    answer = message.get(
        "content"
    )

    if not answer:

        raise RuntimeError(
            "Ollama returned no answer."
        )

    return answer


# ============================================================
# STREAMLIT PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Zomato Review RAG",
    page_icon="🍴",
    layout="wide"
)


# ============================================================
# TITLE
# ============================================================

st.title(
    "🍴 Chat with your Zomato Reviews"
)

st.caption(
    f"BigQuery → {EMBEDDING_MODEL} → "
    f"Similarity Search → {CHAT_MODEL}"
)

st.divider()


# ============================================================
# CHECK OLLAMA
# ============================================================

if not check_ollama():

    st.stop()


# ============================================================
# LOAD RAG KNOWLEDGE BASE
# ============================================================

with st.spinner(
    "Loading Zomato review knowledge base..."
):

    try:

        review_df = load_reviews()

    except Exception as e:

        st.error(
            f"Failed to load review knowledge base:\n\n{e}"
        )

        st.stop()


st.success(
    f"RAG knowledge base loaded: "
    f"{len(review_df):,} reviews"
)


# ============================================================
# QUESTION INPUT
# ============================================================

question = st.text_input(

    "Ask a question about your Zomato reviews:",

    placeholder=(
        "Example: What are the most common "
        "complaints about delivery?"
    )
)


# ============================================================
# RAG PIPELINE
# ============================================================

if question:

    # --------------------------------------------------------
    # STEP 1 — RETRIEVAL
    # --------------------------------------------------------

    with st.spinner(
        "Searching relevant reviews..."
    ):

        try:

            top_reviews = find_similar_reviews(
                question,
                review_df
            )

        except Exception as e:

            st.error(
                f"Embedding/retrieval failed: {e}"
            )

            st.stop()


    # --------------------------------------------------------
    # SHOW RETRIEVED REVIEWS
    # --------------------------------------------------------

    st.subheader(
        "🔎 Retrieved Reviews"
    )

    display_df = top_reviews[
        [
            "review_id",
            "rating",
            "comment",
            "review_date",
            "score"
        ]
    ].copy()

    display_df["score"] = (
        display_df["score"]
        .round(4)
    )

    st.dataframe(
        display_df,
        hide_index=True,
        width="stretch",
    )

    st.divider()


    # --------------------------------------------------------
    # STEP 2 — GENERATION
    # --------------------------------------------------------

    with st.spinner(
        f"{CHAT_MODEL} is generating the answer..."
    ):

        try:

            answer = ask_llm(
                question,
                top_reviews
            )

            st.subheader(
                "🤖 Answer"
            )

            st.write(
                answer
            )

        except Exception as e:

            st.error(
                f"Ollama generation failed: {e}"
            )

            st.info(
                "Check that Ollama is running and "
                f"that '{CHAT_MODEL}' is available."
            )


# ============================================================
# INFORMATION
# ============================================================

with st.expander(
    "ℹ️ About this RAG pipeline"
):

    st.write(
        f"""
This application follows a Retrieval-Augmented
Generation (RAG) pipeline:

1. Reviews are loaded from BigQuery.

2. Review comments are converted into
   embeddings using Ollama's
   {EMBEDDING_MODEL} model.

3. The user's question is converted into
   an embedding using the same model.

4. Cosine similarity finds the most relevant
   reviews.

5. The top {TOP_K} reviews are provided to
   the LLM as context.

6. Ollama's {CHAT_MODEL} generates the
   final answer.

7. The LLM is instructed to answer only
   from the retrieved reviews and not
   invent information.

The review embeddings are cached locally
in:

{CACHE_FILE}
"""
    )

