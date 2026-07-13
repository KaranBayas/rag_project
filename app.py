import math
from typing import Any

import pandas as pd
import streamlit as st

from cosine_similar_vectors import cosine_similar_vectors
from embedding import create_embedding
from llm_response import generate_response
from prompt import create_prompt


PAGE_TITLE = "AI Lecture Assistant"
TOP_K = 10


def configure_page() -> None:
    """Set Streamlit page configuration."""
    st.set_page_config(
        page_title=PAGE_TITLE,
        page_icon="🎓",
        layout="wide",
        initial_sidebar_state="expanded",
    )


def inject_styles() -> None:
    """Apply lightweight custom styling for a modern chat UI."""
    st.markdown(
        """
        <style>
        .stApp {
            background:
                radial-gradient(circle at top left, rgba(59, 130, 246, 0.12), transparent 28%),
                radial-gradient(circle at top right, rgba(16, 185, 129, 0.10), transparent 24%),
                linear-gradient(180deg, #f8fafc 0%, #eef2ff 100%);
        }

        .block-container {
            max-width: 1100px;
            padding-top: 2rem;
            padding-bottom: 2rem;
        }

        .hero {
            padding: 1.5rem 1.75rem;
            border: 1px solid rgba(15, 23, 42, 0.08);
            border-radius: 24px;
            background: rgba(255, 255, 255, 0.82);
            backdrop-filter: blur(8px);
            box-shadow: 0 20px 50px rgba(15, 23, 42, 0.08);
            margin-bottom: 1.25rem;
        }

        .hero h1 {
            margin: 0 0 0.35rem 0;
            font-size: 2.2rem;
            color: #0f172a;
        }

        .hero p {
            margin: 0;
            color: #334155;
            font-size: 1rem;
            line-height: 1.6;
        }

        .meta-card {
            background: rgba(255, 255, 255, 0.78);
            border: 1px solid rgba(15, 23, 42, 0.08);
            border-radius: 18px;
            padding: 1rem 1.1rem;
            margin-top: 0.75rem;
            box-shadow: 0 10px 30px rgba(15, 23, 42, 0.05);
        }

        .meta-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
            gap: 0.75rem;
            margin-top: 0.75rem;
        }

        .meta-item {
            padding: 0.8rem 0.9rem;
            border-radius: 14px;
            background: #f8fafc;
            border: 1px solid rgba(148, 163, 184, 0.2);
        }

        .meta-label {
            font-size: 0.78rem;
            text-transform: uppercase;
            letter-spacing: 0.04em;
            color: #64748b;
            margin-bottom: 0.2rem;
        }

        .meta-value {
            font-size: 0.98rem;
            color: #0f172a;
            font-weight: 600;
        }

        .sidebar-note {
            color: #475569;
            font-size: 0.95rem;
            line-height: 1.55;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def init_session_state() -> None:
    """Initialize chat history."""
    if "messages" not in st.session_state:
        st.session_state.messages = []


def render_sidebar() -> None:
    """Render sidebar content."""
    with st.sidebar:
        st.title("AI Lecture Assistant")
        st.markdown(
            '<p class="sidebar-note">Ask natural-language questions about AI and ML lectures and get grounded answers with direct video references.</p>',
            unsafe_allow_html=True,
        )

        st.subheader("Project Features")
        st.markdown(
            "\n".join(
                [
                    "- Whisper Speech-to-Text",
                    "- Semantic Search",
                    "- FAISS Vector Search",
                    "- Gemini LLM",
                    "- Retrieval-Augmented Generation",
                ]
            )
        )

        st.subheader("System Information")
        st.markdown("**Embedding Model:** `nomic-embed-text`")
        st.markdown("**Vector Search:** `FAISS`")
        st.markdown("**LLM:** `Gemini`")


def render_header() -> None:
    """Render page title and summary."""
    st.markdown(
        """
        <div class="hero">
            <h1>AI Lecture Assistant</h1>
            <p>
                A production-style RAG interface for exploring AI/ML lecture videos with semantic search,
                transcript-grounded answers, and direct pointers to the most relevant lecture segment.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def format_timestamp(seconds: Any) -> str:
    """Convert seconds to HH:MM:SS format."""
    try:
        total_seconds = max(0, int(float(seconds)))
    except (TypeError, ValueError):
        return "Unknown"

    hours, remainder = divmod(total_seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def safe_value(value: Any, fallback: str = "Unavailable") -> str:
    """Return a readable string for UI display."""
    if value is None:
        return fallback
    if isinstance(value, float) and math.isnan(value):
        return fallback
    text = str(value).strip()
    return text if text else fallback


def build_response_metadata(retrieved_df: pd.DataFrame) -> dict[str, Any]:
    """Extract UI metadata from the top retrieved chunk."""
    if retrieved_df.empty:
        return {
            "video_name": "Unavailable",
            "timestamp": "Unavailable",
            "chunk_count": 0,
        }

    top_row = retrieved_df.iloc[0]
    start_time = format_timestamp(top_row.get("start"))
    end_time = format_timestamp(top_row.get("end"))

    return {
        "video_name": safe_value(top_row.get("title")),
        "timestamp": f"{start_time} - {end_time}",
        "chunk_count": len(retrieved_df),
    }


def run_rag_pipeline(question: str) -> dict[str, Any]:
    """Call the existing backend functions in sequence."""
    try:
        embeddings = create_embedding([question])
        if not embeddings:
            raise ValueError("Embedding generation returned no vectors.")

        question_embedding = embeddings[0]
        retrieved_df = cosine_similar_vectors(question_embedding, top_k=TOP_K)

        if not isinstance(retrieved_df, pd.DataFrame):
            raise TypeError("Retrieval did not return a pandas DataFrame.")
        if retrieved_df.empty:
            raise ValueError("No lecture chunks were retrieved.")

        prompt = create_prompt(question, retrieved_df, TOP_K)
        answer = generate_response(prompt)

        return {
            "answer": safe_value(answer, fallback="No response generated."),
            "retrieved_df": retrieved_df,
            "metadata": build_response_metadata(retrieved_df),
        }
    except Exception as exc:
        raise RuntimeError(f"Unable to process the lecture query: {exc}") from exc


def render_retrieved_context(retrieved_df: pd.DataFrame) -> None:
    """Render retrieved chunks inside an expandable section."""
    with st.expander("Retrieved Context"):
        for idx, row in retrieved_df.reset_index(drop=True).iterrows():
            st.markdown(f"**Chunk {idx + 1}**")
            st.markdown(f"**Video:** {safe_value(row.get('title'))}")
            st.markdown(
                f"**Timestamp:** {format_timestamp(row.get('start'))} - {format_timestamp(row.get('end'))}"
            )
            st.markdown(safe_value(row.get("text")))
            if idx < len(retrieved_df) - 1:
                st.divider()


def render_assistant_metadata(metadata: dict[str, Any], retrieved_df: pd.DataFrame) -> None:
    """Render structured metadata below each assistant response."""
    st.markdown(
        f"""
        <div class="meta-card">
            <div class="meta-grid">
                <div class="meta-item">
                    <div class="meta-label">Video Name</div>
                    <div class="meta-value">{metadata["video_name"]}</div>
                </div>
                <div class="meta-item">
                    <div class="meta-label">Timestamp</div>
                    <div class="meta-value">{metadata["timestamp"]}</div>
                </div>
                <div class="meta-item">
                    <div class="meta-label">Retrieved Chunks</div>
                    <div class="meta-value">{metadata["chunk_count"]}</div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    render_retrieved_context(retrieved_df)


def render_chat_history() -> None:
    """Render all previous chat messages."""
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message["role"] == "assistant" and "metadata" in message and "retrieved_df" in message:
                render_assistant_metadata(message["metadata"], message["retrieved_df"])


def handle_user_query(question: str) -> None:
    """Run the pipeline for a user query and update chat history."""
    st.session_state.messages.append({"role": "user", "content": question})

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        try:
            with st.spinner("Searching lectures..."):
                result = run_rag_pipeline(question)

            st.markdown(result["answer"])
            render_assistant_metadata(result["metadata"], result["retrieved_df"])
            st.success("Response generated successfully.")

            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": result["answer"],
                    "metadata": result["metadata"],
                    "retrieved_df": result["retrieved_df"],
                }
            )
        except Exception as exc:
            error_message = str(exc)
            st.error(error_message)
            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": f"Error: {error_message}",
                }
            )


def main() -> None:
    """Streamlit application entry point."""
    configure_page()
    inject_styles()
    init_session_state()
    render_sidebar()
    render_header()
    render_chat_history()

    question = st.chat_input("Ask a question about your lecture videos...")
    if question:
        handle_user_query(question.strip())


if __name__ == "__main__":
    main()
