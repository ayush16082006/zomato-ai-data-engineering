"""
Zomato AI - Streamlit Background Worker

This module contains the multiprocessing worker used by app.py.

IMPORTANT:
The worker is kept in a separate importable module because Windows
uses the "spawn" multiprocessing method. Keeping the worker outside
app.py avoids Streamlit/__main__ pickling problems.
"""


def answer_worker(question, result_queue):
    """
    Run the Zomato AI orchestrator in a separate process.

    Parameters
    ----------
    question : str
        User's natural-language question.

    result_queue : multiprocessing.Queue
        Queue used to send the result back to Streamlit.
    """

    try:
        # Import inside the worker so the child process does not
        # initialize the Streamlit application.
        from orchestrator import answer_question

        result = answer_question(question)

        result_queue.put(
            {
                "status": "success",
                "result": result,
            }
        )

    except Exception as exc:

        result_queue.put(
            {
                "status": "error",
                "result": {
                    "success": False,
                    "route": "ERROR",
                    "answer": (
                        "I encountered an error while processing "
                        "your question."
                    ),
                    "error": str(exc),
                },
            }
        )