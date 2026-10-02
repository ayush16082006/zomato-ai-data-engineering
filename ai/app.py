import streamlit as st

import multiprocessing as mp
import queue
import re


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Zomato AI",
    page_icon="🍽️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# BACKGROUND WORKER
# ============================================================
#
# IMPORTANT:
#
# The worker is imported from streamlit_worker.py.
#
# DO NOT define the multiprocessing worker directly inside
# app.py.
#
# Windows uses multiprocessing "spawn", and keeping the worker
# in a separate importable module prevents:
#
# PicklingError:
# Can't pickle <function _answer_worker ...>
#
# ============================================================

from streamlit_worker import answer_worker


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    /* ========================================================
       GLOBAL
    ======================================================== */

    .stApp {
        background-color: #0f1117;
        color: #f5f5f5;
    }

    [data-testid="stSidebar"] {
        background-color: #202124;
        border-right: 1px solid #303238;
    }

    [data-testid="stSidebar"] > div:first-child {
        padding-top: 1.5rem;
    }


    /* ========================================================
       SIDEBAR
    ======================================================== */

    .sidebar-brand {
        font-size: 28px;
        font-weight: 700;
        color: #ffffff;
        margin-bottom: 8px;
    }

    .sidebar-description {
        color: #aeb4c0;
        font-size: 15px;
        line-height: 1.6;
        margin-bottom: 25px;
    }

    .sidebar-section {
        color: #ffffff;
        font-size: 20px;
        font-weight: 700;
        margin-top: 25px;
        margin-bottom: 15px;
    }


    /* ========================================================
       MAIN HEADER
    ======================================================== */

    .main-header {
        text-align: center;
        padding-top: 15px;
        padding-bottom: 35px;
    }

    .main-header-subtitle {
        color: #9da3b4;
        font-size: 17px;
        margin-top: -10px;
    }

    .main-header-small {
        color: #737b91;
        font-size: 14px;
        margin-top: 15px;
    }


    /* ========================================================
       CHAT AREA
    ======================================================== */

    .user-message {
        background-color: #1b1e27;
        border-radius: 12px;
        padding: 15px 18px;
        margin-top: 15px;
        margin-bottom: 10px;
        border: 1px solid #292d38;
    }

    .assistant-message {
        background-color: transparent;
        padding: 5px 18px 20px 18px;
        margin-bottom: 10px;
    }

    .user-label {
        color: #ff4b4b;
        font-weight: 700;
        margin-bottom: 7px;
    }

    .assistant-label {
        color: #ff9f0a;
        font-weight: 700;
        margin-bottom: 7px;
    }

    .route-badge {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 12px;
        background-color: #252936;
        color: #9da3b4;
        font-size: 12px;
        margin-top: 8px;
    }


    /* ========================================================
       INFO BOXES
    ======================================================== */

    .info-box {
        background-color: #181b23;
        border: 1px solid #2b2f3a;
        border-radius: 10px;
        padding: 12px 15px;
        margin-top: 10px;
        margin-bottom: 10px;
    }


    /* ========================================================
       EXAMPLE BUTTONS
    ======================================================== */

    [data-testid="stSidebar"] .stButton > button {
        width: 100%;
        min-height: 55px;
        background-color: #292b34;
        color: #f5f5f5;
        border: 1px solid #454854;
        border-radius: 10px;
        font-size: 14px;
        text-align: center;
        transition: all 0.2s ease;
        margin-bottom: 8px;
    }

    [data-testid="stSidebar"] .stButton > button:hover {
        border-color: #777b88;
        background-color: #323540;
    }


    /* ========================================================
       CHAT INPUT
    ======================================================== */

    [data-testid="stChatInput"] {
        padding-bottom: 15px;
    }

    [data-testid="stChatInput"] textarea {
        background-color: #292b34 !important;
        color: #ffffff !important;
        border: 1px solid #454854 !important;
    }


    /* ========================================================
       EXPANDERS
    ======================================================== */

    .streamlit-expanderHeader {
        background-color: #181b23;
        border-radius: 8px;
    }


    /* ========================================================
       DATAFRAME
    ======================================================== */

    [data-testid="stDataFrame"] {
        border-radius: 10px;
        overflow: hidden;
    }


    /* ========================================================
       MARKDOWN TABLES
    ======================================================== */

    .stMarkdown table {
        width: 100%;
        border-collapse: collapse;
        margin-top: 12px;
        margin-bottom: 18px;
        font-size: 14px;
    }

    .stMarkdown table thead tr {
        background-color: #252936;
    }

    .stMarkdown table th {
        color: #ffffff;
        font-weight: 700;
        padding: 11px 12px;
        text-align: left;
        border: 1px solid #3a3e4b;
    }

    .stMarkdown table td {
        color: #e6e8ed;
        padding: 10px 12px;
        border: 1px solid #303440;
        vertical-align: top;
    }

    .stMarkdown table tbody tr:nth-child(even) {
        background-color: #181b23;
    }

    .stMarkdown table tbody tr:nth-child(odd) {
        background-color: #141720;
    }


    /* ========================================================
       GENERATION STATUS
    ======================================================== */

    .generation-status {
        background-color: #181b23;
        border: 1px solid #303440;
        border-radius: 10px;
        padding: 12px 16px;
        margin: 10px 0;
        color: #aeb4c0;
        text-align: center;
        font-size: 14px;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# SESSION STATE
# ============================================================

if "messages" not in st.session_state:
    st.session_state.messages = []

if "pending_question" not in st.session_state:
    st.session_state.pending_question = None

if "generating" not in st.session_state:
    st.session_state.generating = False

if "generation_process" not in st.session_state:
    st.session_state.generation_process = None

if "generation_queue" not in st.session_state:
    st.session_state.generation_queue = None

if "generation_question" not in st.session_state:
    st.session_state.generation_question = None


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def first_not_none(*values):
    """
    Return the first value that is not None.
    """

    for value in values:
        if value is not None:
            return value

    return None


def normalize_route(result):
    """
    Normalize route names.

    SQL
    sql
    Sql

    all become:

    SQL
    """

    if not isinstance(result, dict):
        return "UNKNOWN"

    return str(
        result.get("route", "UNKNOWN")
    ).upper()


# ============================================================
# MARKDOWN TABLE REPAIR
# ============================================================

def normalize_answer_for_ui(answer):
    """
    Clean common formatting problems in LLM-generated answers.

    In particular, converts a one-line Markdown table:

        | Rank | Restaurant | Revenue | |------|------------|---------| | 1 | ABC | ₹1000 |

    into:

        | Rank | Restaurant | Revenue |
        |------|------------|---------|
        | 1 | ABC | ₹1000 |

    Streamlit can then render the table correctly.
    """

    if answer is None:
        return ""

    answer = str(answer)

    answer = answer.replace("\r\n", "\n")
    answer = answer.replace("\r", "\n")

    # --------------------------------------------------------
    # Remove accidental "canvascanvas" text.
    #
    # Some LLM outputs may produce this when attempting
    # structured/table output.
    # --------------------------------------------------------

    answer = re.sub(
        r"\bcanvascanvas\b",
        "",
        answer,
        flags=re.IGNORECASE,
    )

    # --------------------------------------------------------
    # Normalize excessive spaces.
    # --------------------------------------------------------

    answer = re.sub(
        r"[ \t]+\n",
        "\n",
        answer,
    )

    # --------------------------------------------------------
    # Detect Markdown table separator.
    #
    # Example:
    #
    # | Rank | Restaurant | Revenue | |------|------------|---------|
    #
    # --------------------------------------------------------

    separator_pattern = re.compile(
        r"\|\s*:?-{3,}:?\s*(?:\||$)"
    )

    # --------------------------------------------------------
    # If a table exists on one or more lines, repair it.
    # --------------------------------------------------------

    lines = answer.split("\n")

    repaired_lines = []

    for line in lines:

        stripped = line.strip()

        if not stripped:
            repaired_lines.append(line)
            continue

        # ----------------------------------------------------
        # If the line contains multiple table rows glued
        # together, split before separator rows.
        # ----------------------------------------------------

        if "|" in stripped:

            # Split before markdown separator patterns.
            stripped = re.sub(
                r"\s+(?=\|\s*:?-{3,})",
                "\n",
                stripped,
            )

            # Split before numeric table rows.
            stripped = re.sub(
                r"\s+(?=\|\s*\d+\s*\|)",
                "\n",
                stripped,
            )

            # Split before rows beginning with common rank.
            stripped = re.sub(
                r"\s+(?=\|\s*Rank\s*\|)",
                "\n",
                stripped,
                flags=re.IGNORECASE,
            )

            repaired_lines.extend(
                stripped.split("\n")
            )

        else:
            repaired_lines.append(line)

    answer = "\n".join(repaired_lines)

    # --------------------------------------------------------
    # Second pass:
    #
    # Sometimes LLM output is:
    #
    # | Rank | Restaurant |
    # |------|------------| | 1 | ABC |
    #
    # --------------------------------------------------------

    answer = re.sub(
        r"(\|(?:\s*:?-{3,}:?\s*\|)+)\s+(?=\|)",
        r"\1\n",
        answer,
    )

    # --------------------------------------------------------
    # Ensure table rows are separated.
    # --------------------------------------------------------

    answer = re.sub(
        r"(\|)\s+(?=\|\s*\d+\s*\|)",
        r"\1\n",
        answer,
    )

    # --------------------------------------------------------
    # Remove excessive blank lines.
    # --------------------------------------------------------

    answer = re.sub(
        r"\n{3,}",
        "\n\n",
        answer,
    )

    return answer.strip()


# ============================================================
# SQL EXTRACTION
# ============================================================

def extract_sql_query(result):
    """
    Extract SQL from either the current result structure
    or nested sql_result structure.
    """

    sql_result = result.get("sql_result")

    if isinstance(sql_result, dict):

        nested_sql = first_not_none(
            sql_result.get("sql"),
            sql_result.get("query"),
            sql_result.get("generated_sql"),
        )

    else:
        nested_sql = None

    return first_not_none(
        result.get("sql"),
        result.get("query"),
        result.get("generated_sql"),
        nested_sql,
    )


def extract_sql_data(result):
    """
    Extract SQL data from either top-level or nested result.
    """

    sql_result = result.get("sql_result")

    if isinstance(sql_result, dict):
        nested_data = sql_result.get("data")
    else:
        nested_data = None

    return first_not_none(
        result.get("data"),
        result.get("result_data"),
        result.get("rows"),
        nested_data,
    )


# ============================================================
# RAG EXTRACTION
# ============================================================

def extract_rag_reviews(result):
    """
    Extract retrieved review data.
    """

    return first_not_none(
        result.get("reviews"),
        result.get("retrieved_reviews"),
        result.get("documents"),
    )


def extract_rag_analysis(result):
    """
    Extract optional RAG analysis.
    """

    return first_not_none(
        result.get("rag_analysis"),
        result.get("analysis"),
    )


# ============================================================
# MESSAGE MANAGEMENT
# ============================================================

def add_assistant_message(question, result):
    """
    Store assistant result in conversation history.
    """

    st.session_state.messages.append(
        {
            "role": "assistant",
            "question": question,
            "result": result,
        }
    )


# ============================================================
# GENERATION CONTROL
# ============================================================

def start_generation(question):
    """
    Start a new Zomato AI request in a separate process.

    Windows uses "spawn", so the worker comes from the
    importable streamlit_worker module.
    """

    context = mp.get_context("spawn")

    result_queue = context.Queue()

    process = context.Process(
        target=answer_worker,
        args=(question, result_queue),
    )

    process.daemon = True

    process.start()

    st.session_state.generation_process = process
    st.session_state.generation_queue = result_queue
    st.session_state.generation_question = question
    st.session_state.generating = True


# ============================================================
# FINISH GENERATION
# ============================================================

def finish_generation(result):
    """
    Finish a completed generation.
    """

    process = st.session_state.get(
        "generation_process"
    )

    if process is not None:

        try:
            process.join(timeout=1)
        except Exception:
            pass

    question = st.session_state.get(
        "generation_question"
    )

    if not isinstance(result, dict):

        result = {
            "success": False,
            "route": "ERROR",
            "answer": str(result),
        }

    add_assistant_message(
        question,
        result,
    )

    st.session_state.generating = False

    st.session_state.generation_process = None
    st.session_state.generation_queue = None
    st.session_state.generation_question = None


# ============================================================
# STOP GENERATION
# ============================================================

def stop_generation():
    """
    Actually terminate the current background process.
    """

    process = st.session_state.get(
        "generation_process"
    )

    question = st.session_state.get(
        "generation_question"
    )

    if process is not None:

        try:

            if process.is_alive():

                process.terminate()

                process.join(
                    timeout=3
                )

                if process.is_alive():

                    try:
                        process.kill()
                    except Exception:
                        pass

        except Exception:
            pass

    # --------------------------------------------------------
    # Add stopped message.
    # --------------------------------------------------------

    add_assistant_message(
        question or "",
        {
            "success": True,
            "route": "STOPPED",
            "answer": (
                "Generation was stopped. "
                "You can ask a new question now."
            ),
        },
    )

    st.session_state.generating = False

    st.session_state.generation_process = None
    st.session_state.generation_queue = None
    st.session_state.generation_question = None


# ============================================================
# GENERATION MONITOR
# ============================================================

@st.fragment(run_every=0.5)
def generation_controller():

    if not st.session_state.get(
        "generating",
        False,
    ):
        return

    st.markdown(
        """
        <div class="generation-status">
            ⏳ Zomato AI is generating the answer...
        </div>
        """,
        unsafe_allow_html=True,
    )

    # --------------------------------------------------------
    # STOP BUTTON
    # --------------------------------------------------------

    if st.button(
        "⏹️ Stop generation",
        key="stop_generation_button",
        use_container_width=True,
    ):

        stop_generation()

        st.rerun()

    # --------------------------------------------------------
    # CHECK BACKGROUND PROCESS
    # --------------------------------------------------------

    result_queue = st.session_state.get(
        "generation_queue"
    )

    process = st.session_state.get(
        "generation_process"
    )

    if result_queue is None or process is None:
        return

    try:

        message = result_queue.get_nowait()

    except queue.Empty:

        return

    # --------------------------------------------------------
    # RESULT RECEIVED
    # --------------------------------------------------------

    if not isinstance(message, dict):

        result = {
            "success": False,
            "route": "ERROR",
            "answer": (
                "The AI worker returned an invalid response."
            ),
        }

    else:

        result = message.get(
            "result"
        )

    finish_generation(
        result
    )

    st.rerun()


# ============================================================
# EXAMPLE QUESTIONS
# ============================================================

EXAMPLE_QUESTIONS = [

    "Which restaurant generated the most revenue?",

    "Which restaurant generated the most revenue in Mumbai?",

    "Which customer spent the most?",

    "What do customers say about food quality?",

    "What do customers say about delivery?",

    "Which restaurant generated the most revenue and what do customers say about it?",

]


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown(
        '<div class="sidebar-brand">🍽️ Zomato AI</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="sidebar-description">'
        "Ask questions about your Zomato dataset "
        "using natural language."
        "</div>",
        unsafe_allow_html=True,
    )

    st.divider()

    st.markdown(
        '<div class="sidebar-section">💡 Example questions</div>',
        unsafe_allow_html=True,
    )

    generating = st.session_state.get(
        "generating",
        False,
    )

    for question in EXAMPLE_QUESTIONS:

        if st.button(
            question,
            key=f"example_{question}",
            use_container_width=True,
            disabled=generating,
        ):

            st.session_state.pending_question = question

    st.divider()

    if st.button(
        "🗑️ Clear conversation",
        use_container_width=True,
        disabled=generating,
    ):

        st.session_state.messages = []

        st.session_state.pending_question = None

        st.rerun()


# ============================================================
# MAIN HEADER
# ============================================================

st.markdown(
    '<div class="main-header">',
    unsafe_allow_html=True,
)

st.markdown(
    "### 🍽️ Zomato AI"
)

st.markdown(
    '<div class="main-header-subtitle">'
    "Intelligent restaurant analytics using SQL, RAG and Hybrid AI"
    "</div>",
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="main-header-small">'
    "Zomato AI · SQL + RAG + Hybrid Intelligence"
    "</div>",
    unsafe_allow_html=True,
)

st.markdown(
    "</div>",
    unsafe_allow_html=True,
)


# ============================================================
# DISPLAY PREVIOUS CONVERSATION
# ============================================================

for message in st.session_state.messages:

    role = message.get(
        "role"
    )

    question = message.get(
        "question",
        "",
    )

    result = message.get(
        "result",
        {},
    )

    # ========================================================
    # USER MESSAGE
    # ========================================================

    if role == "user":

        st.markdown(
            '<div class="user-message">'
            '<div class="user-label">🔴 You</div>'
            f"{question}"
            "</div>",
            unsafe_allow_html=True,
        )

    # ========================================================
    # ASSISTANT MESSAGE
    # ========================================================

    elif role == "assistant":

        answer = result.get(
            "answer",
            "No answer was returned.",
        )

        route = normalize_route(
            result
        )

        # ----------------------------------------------------
        # FORMAT ANSWER
        # ----------------------------------------------------

        answer = normalize_answer_for_ui(
            answer
        )

        st.markdown(
            '<div class="assistant-message">'
            '<div class="assistant-label">🤖 Zomato AI</div>',
            unsafe_allow_html=True,
        )

        # ----------------------------------------------------
        # IMPORTANT:
        #
        # Render the answer as Markdown so properly formatted
        # Markdown tables become actual Streamlit tables.
        # ----------------------------------------------------

        st.markdown(
            answer
        )

        st.markdown(
            f'<div class="route-badge">Route: {route}</div>',
            unsafe_allow_html=True,
        )

        st.markdown(
            "</div>",
            unsafe_allow_html=True,
        )

        # ====================================================
        # SQL DETAILS
        # ====================================================

        if route == "SQL":

            sql_query = extract_sql_query(
                result
            )

            sql_data = extract_sql_data(
                result
            )

            if sql_query:

                with st.expander(
                    "🔎 SQL details"
                ):

                    st.code(
                        sql_query,
                        language="sql",
                    )

            if sql_data is not None:

                with st.expander(
                    "📊 Query result"
                ):

                    try:

                        st.dataframe(
                            sql_data,
                            use_container_width=True,
                            hide_index=True,
                        )

                    except Exception:

                        st.write(
                            sql_data
                        )

        # ====================================================
        # RAG DETAILS
        # ====================================================

        elif route == "RAG":

            rag_reviews = extract_rag_reviews(
                result
            )

            rag_analysis = extract_rag_analysis(
                result
            )

            if rag_analysis:

                with st.expander(
                    "🔍 RAG analysis"
                ):

                    st.markdown(
                        normalize_answer_for_ui(
                            rag_analysis
                        )
                    )

            if rag_reviews is not None:

                with st.expander(
                    "📝 Retrieved customer reviews"
                ):

                    try:

                        st.dataframe(
                            rag_reviews,
                            use_container_width=True,
                            hide_index=True,
                        )

                    except Exception:

                        st.write(
                            rag_reviews
                        )

        # ====================================================
        # HYBRID DETAILS
        # ====================================================

        elif route == "HYBRID":

            sql_query = extract_sql_query(
                result
            )

            sql_data = extract_sql_data(
                result
            )

            rag_reviews = extract_rag_reviews(
                result
            )

            rag_analysis = extract_rag_analysis(
                result
            )

            if sql_query:

                with st.expander(
                    "🔎 SQL analysis"
                ):

                    st.code(
                        sql_query,
                        language="sql",
                    )

            if sql_data is not None:

                with st.expander(
                    "📊 SQL result"
                ):

                    try:

                        st.dataframe(
                            sql_data,
                            use_container_width=True,
                            hide_index=True,
                        )

                    except Exception:

                        st.write(
                            sql_data
                        )

            if rag_analysis:

                with st.expander(
                    "🔍 RAG analysis"
                ):

                    st.markdown(
                        normalize_answer_for_ui(
                            rag_analysis
                        )
                    )

            if rag_reviews is not None:

                with st.expander(
                    "📝 Customer review analysis"
                ):

                    try:

                        st.dataframe(
                            rag_reviews,
                            use_container_width=True,
                            hide_index=True,
                        )

                    except Exception:

                        st.write(
                            rag_reviews
                        )

        # ====================================================
        # DOMAIN GUARD
        # ====================================================

        elif route in [
            "DOMAIN_GUARD",
            "OUT_OF_SCOPE",
            "BLOCKED",
        ]:

            domain_guard = result.get(
                "domain_guard"
            )

            reason = first_not_none(

                result.get("reason"),

                (
                    domain_guard.get("reason")
                    if isinstance(
                        domain_guard,
                        dict,
                    )
                    else None
                ),

            )

            if reason:

                with st.expander(
                    "🛡️ Routing information"
                ):

                    st.write(
                        reason
                    )

        # ====================================================
        # STOPPED
        # ====================================================

        elif route == "STOPPED":

            pass

        # ====================================================
        # ERROR
        # ====================================================

        elif route == "ERROR":

            error = result.get(
                "error"
            )

            if error:

                with st.expander(
                    "⚠️ Error details"
                ):

                    st.code(
                        str(error)
                    )


# ============================================================
# HANDLE EXAMPLE QUESTION
# ============================================================

pending_question = st.session_state.get(
    "pending_question"
)

if pending_question is not None:

    st.session_state.pending_question = None

else:

    pending_question = None


# ============================================================
# GENERATION CONTROLLER
# ============================================================

generation_controller()


# ============================================================
# CHAT INPUT
# ============================================================

generating = st.session_state.get(
    "generating",
    False,
)

question = st.chat_input(
    "Ask about restaurants, revenue, customers, orders, cuisines or reviews...",
    disabled=generating,
)


# ============================================================
# EXAMPLE QUESTION HAS PRIORITY
# ============================================================

if pending_question and not generating:

    question = pending_question


# ============================================================
# PROCESS NEW QUESTION
# ============================================================

if question and not generating:

    question = question.strip()

    if not question:

        st.stop()

    # --------------------------------------------------------
    # ADD USER MESSAGE
    # --------------------------------------------------------

    st.session_state.messages.append(
        {
            "role": "user",
            "question": question,
        }
    )

    # --------------------------------------------------------
    # START BACKGROUND GENERATION
    # --------------------------------------------------------

    start_generation(
        question
    )

    # --------------------------------------------------------
    # RERUN IMMEDIATELY
    #
    # This makes the input disabled while generation runs.
    # --------------------------------------------------------

    st.rerun()


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div style="
        text-align:center;
        color:#555b6e;
        font-size:12px;
        padding:25px 0 10px 0;
    ">
        Zomato AI · SQL + RAG + Hybrid Intelligence
    </div>
    """,
    unsafe_allow_html=True,
)